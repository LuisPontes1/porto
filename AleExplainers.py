import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import re

import random

from AleTransforms import *
from AleDatasetAnalysis import *

class CamClassificacao:

    def __init__(self, clf, df, df_exp, cols_features = None, num_loop = 5, random_state = None):
        self.__tam = df_exp.shape[0]
        if(cols_features is None):
            self.__features = np.array([col for col in df.columns])
        else:
            self.__features = np.array(cols_features)
        self.__num_loop = num_loop
        
        if(random_state is not None):
            random.seed(random_state)
        
        self.__calcula_variacoes_predict(df, clf, df_exp)
        
    def __calcula_variacoes_predict(self, df, clf, df_exp):
        df_temp = df_exp[self.__features].copy()
        self.__y_prob = clf.predict_proba(df_temp)[:, 1]
        
        vetor_diff_prob = []
        vetor_incerteza = []
        for feature in self.__features:
            valores_feature = df[feature].tolist()
            diff_probs_steps = []
            for i in range(0, self.__num_loop):
                df_temp[feature] = random.sample(valores_feature, self.__tam)
                y_prob_temp = clf.predict_proba(df_temp)[:, 1]
                diff_probs_steps.append(y_prob_temp - self.__y_prob)
            df_temp[feature] = df_exp[feature]
            diff_probs_steps = np.array(diff_probs_steps)
            vetor_diff_prob.append(np.mean(diff_probs_steps, axis = 0))
            vetor_incerteza.append(np.std(diff_probs_steps, axis = 0))
        vetor_diff_prob = -1*np.array(vetor_diff_prob) #inverte o sinal pq se a mudança tende a diminuir a prob, é pq esse valor está impactando positivamente
        vetor_incerteza = np.array(vetor_incerteza)
        
        self.df_exp_varprob = pd.DataFrame(np.transpose(vetor_diff_prob), columns = self.__features, index = df_exp.index)
        self.df_exp_incerteza = pd.DataFrame(np.transpose(vetor_incerteza), columns = self.__features, index = df_exp.index)
    
    def styler_mapa_calor(self, df_exp, ordenar_linhas = False, ordenar_colunas = False, opacity_text = False, pos_colunas_restantes = 'left'):
        #Normaliza os valores para plotar em cor
        def normaliza_media(media):
            minimo = np.min(media)
            maximo = np.max(media)
            if(maximo > 0 and minimo < 0):
                if(maximo > abs(minimo)):
                    norm = media/maximo
                else:
                    norm = media/np.abs(minimo)
            elif(maximo > 0 and minimo >= 0):
                norm = media/maximo
            elif(maximo <= 0 and minimo < 0):
                norm = media/np.abs(minimo)
            return norm
        
        paleta_cores = sns.color_palette("colorblind") #Paleta de cores para daltonico
        N = 256
        cor_neg = paleta_cores[0]
        cor_pos = paleta_cores[1]
        vals = np.ones((N, 4))
        regua = np.linspace(-1, 1, N)
        vals[:, 0] = np.array([cor_pos[0] if v > 0 else cor_neg[0] for v in regua])
        vals[:, 1] = np.array([cor_pos[1] if v > 0 else cor_neg[1] for v in regua])
        vals[:, 2] = np.array([cor_pos[2] if v > 0 else cor_neg[2] for v in regua]) 
        vals[:, 3] = np.array([(v**2)**(1/2) for v in regua]) #Aqui podemos alterar a velocidade com que o alpha muda
        cmap = mpl.colors.ListedColormap(vals)
        cores = cmap(np.arange(cmap.N))
        
        df_calor = pd.DataFrame(np.nan, columns = df_exp.columns, index = df_exp.index)
        for i in df_exp.index:
            norm = normaliza_media(self.df_exp_varprob.loc[i, self.__features].values)
            cores_col = cores[np.floor((norm + 1)*(N-1)/2).astype(int)]
            cores_col = np.rint(255*cores_col).astype(int)
            if(opacity_text):
                cores_col_hex = ['background-color: ' + '#{:02x}{:02x}{:02x}'.format(*cores_col[i,:3]) + '; opacity: ' + str(cores_col[i,3]/255) + '; color: black' for i in range(0, cores_col.shape[0])]
            else:
                #cores_col_hex = ['background-color: ' + '#{:02x}{:02x}{:02x}{:02x}'.format(*cores_col[i,:]) + '; opacity: 1.0' + '; color: black' for i in range(0, cores_col.shape[0])] #Guarda a cor em hexadecimal (com o alpha)
                cores_col_hex = ['background-color: ' + 'rgba({},{},{},'.format(*cores_col[i,:3]) + str(cores_col[i,3]/255) + ')' + '; color: black' for i in range(0, cores_col.shape[0])]
            df_calor.loc[i, self.__features] = cores_col_hex
        df_calor = df_calor.fillna('background-color: white' + '; color: black')
        
        if(ordenar_linhas == 'ascending'):
            inds = np.array(pd.Series(np.arange(self.df_exp_varprob.shape[0]), index = self.df_exp_varprob.index)[df_exp.index])
            y_prob = self.__y_prob[inds]
            inds_ordenados = np.argsort(y_prob)
            df_calor = df_calor.iloc[inds_ordenados, :]
            df_exp_plot = df_exp.iloc[inds_ordenados, :]
        elif(ordenar_linhas == 'descending'):
            inds = np.array(pd.Series(np.arange(self.df_exp_varprob.shape[0]), index = self.df_exp_varprob.index)[df_exp.index])
            y_prob = self.__y_prob[inds]
            inds_ordenados = np.argsort(y_prob)[::-1]
            df_calor = df_calor.iloc[inds_ordenados, :]
            df_exp_plot = df_exp.iloc[inds_ordenados, :]
        else:
            df_exp_plot = df_exp
            
        if(ordenar_colunas == 'ascending'):
            cols_ordenadas = np.argsort(np.sum(np.abs(self.df_exp_varprob.loc[df_exp.index, :].values), axis = 0))
            if(pos_colunas_restantes == 'left'):
                cols = np.append([c for c in df_exp.columns if c not in self.__features], self.__features[cols_ordenadas])
            else:
                cols = np.append(self.__features[cols_ordenadas], [c for c in df_exp.columns if c not in self.__features])
            df_calor = df_calor.loc[:, cols]
            df_exp_plot = df_exp_plot.loc[:, cols]
        elif(ordenar_colunas == 'descending'):
            cols_ordenadas = np.argsort(np.sum(np.abs(self.df_exp_varprob.loc[df_exp.index, :].values), axis = 0))[::-1]
            if(pos_colunas_restantes == 'left'):
                cols = np.append([c for c in df_exp.columns if c not in self.__features], self.__features[cols_ordenadas])
            else:
                cols = np.append(self.__features[cols_ordenadas], [c for c in df_exp.columns if c not in self.__features])
            df_calor = df_calor.loc[:, cols]
            df_exp_plot = df_exp_plot.loc[:, cols]
        else:
            df_exp_plot = df_exp_plot
        
        def apply_color(x):
            return df_calor
        styles = [dict(selector="th", props=[("background-color", "white"), ("color", "black")]),
                  dict(selector="tr", props=[("background-color", "white"), ("color", "black")])]
        styler = df_exp_plot.style.set_table_styles(styles).apply(apply_color, axis = None)
        return styler

##############################

##############################

class CamRegressao:

    def __init__(self, rgs, df, df_exp, cols_features = None, num_loop = 5, random_state = None):
        self.__tam = df_exp.shape[0]
        if(cols_features is None):
            self.__features = np.array([col for col in df.columns])
        else:
            self.__features = np.array(cols_features)
        self.__num_loop = num_loop
        
        if(random_state is not None):
            random.seed(random_state)
        
        self.__calcula_variacoes_predict(df, rgs, df_exp)
        
    def __calcula_variacoes_predict(self, df, rgs, df_exp):
        df_temp = df_exp[self.__features].copy()
        self.__y_pred = rgs.predict(df_temp)
        
        vetor_diff_pred = []
        vetor_incerteza = []
        for feature in self.__features:
            valores_feature = df[feature].tolist()
            diff_preds_steps = []
            for i in range(0, self.__num_loop):
                df_temp[feature] = random.sample(valores_feature, self.__tam)
                y_prob_temp = rgs.predict(df_temp)
                diff_preds_steps.append(y_prob_temp - self.__y_pred)
            df_temp[feature] = df_exp[feature]
            diff_preds_steps = np.array(diff_preds_steps)
            vetor_diff_pred.append(np.mean(diff_preds_steps, axis = 0))
            vetor_incerteza.append(np.std(diff_preds_steps, axis = 0))
        vetor_diff_pred = -1*np.array(vetor_diff_pred) #inverte o sinal pq se a mudança tende a diminuir a prob, é pq esse valor está impactando positivamente
        vetor_incerteza = np.array(vetor_incerteza)
        
        self.df_exp_varpred = pd.DataFrame(np.transpose(vetor_diff_pred), columns = self.__features, index = df_exp.index)
        self.df_exp_incerteza = pd.DataFrame(np.transpose(vetor_incerteza), columns = self.__features, index = df_exp.index)
    
    def styler_mapa_calor(self, df_exp, ordenar_linhas = False, ordenar_colunas = False, opacity_text = False, pos_colunas_restantes = 'left'):
        #Normaliza os valores para plotar em cor
        def normaliza_media(media):
            minimo = np.min(media)
            maximo = np.max(media)
            if(maximo > 0 and minimo < 0):
                if(maximo > abs(minimo)):
                    norm = media/maximo
                else:
                    norm = media/np.abs(minimo)
            elif(maximo > 0 and minimo >= 0):
                norm = media/maximo
            elif(maximo <= 0 and minimo < 0):
                norm = media/np.abs(minimo)
            return norm
        
        paleta_cores = sns.color_palette("colorblind") #Paleta de cores para daltonico
        N = 256
        cor_neg = paleta_cores[0]
        cor_pos = paleta_cores[1]
        vals = np.ones((N, 4))
        regua = np.linspace(-1, 1, N)
        vals[:, 0] = np.array([cor_pos[0] if v > 0 else cor_neg[0] for v in regua])
        vals[:, 1] = np.array([cor_pos[1] if v > 0 else cor_neg[1] for v in regua])
        vals[:, 2] = np.array([cor_pos[2] if v > 0 else cor_neg[2] for v in regua]) 
        vals[:, 3] = np.array([(v**2)**(1/2) for v in regua]) #Aqui podemos alterar a velocidade com que o alpha muda
        cmap = mpl.colors.ListedColormap(vals)
        cores = cmap(np.arange(cmap.N))
        
        df_calor = pd.DataFrame(np.nan, columns = df_exp.columns, index = df_exp.index)
        for i in df_exp.index:
            norm = normaliza_media(self.df_exp_varpred.loc[i, self.__features].values)
            cores_col = cores[np.floor((norm + 1)*(N-1)/2).astype(int)]
            cores_col = np.rint(255*cores_col).astype(int)
            if(opacity_text):
                cores_col_hex = ['background-color: ' + '#{:02x}{:02x}{:02x}'.format(*cores_col[i,:3]) + '; opacity: ' + str(cores_col[i,3]/255) + '; color: black' for i in range(0, cores_col.shape[0])]
            else:
                #cores_col_hex = ['background-color: ' + '#{:02x}{:02x}{:02x}{:02x}'.format(*cores_col[i,:]) + '; opacity: 1.0' + '; color: black' for i in range(0, cores_col.shape[0])] #Guarda a cor em hexadecimal (com o alpha)
                cores_col_hex = ['background-color: ' + 'rgba({},{},{},'.format(*cores_col[i,:3]) + str(cores_col[i,3]/255) + ')' + '; color: black' for i in range(0, cores_col.shape[0])]
            df_calor.loc[i, self.__features] = cores_col_hex
        df_calor = df_calor.fillna('background-color: white' + '; color: black')
        
        if(ordenar_linhas == 'ascending'):
            inds = np.array(pd.Series(np.arange(self.df_exp_varpred.shape[0]), index = self.df_exp_varpred.index)[df_exp.index])
            y_pred = self.__y_pred[inds]
            inds_ordenados = np.argsort(y_pred)
            df_calor = df_calor.iloc[inds_ordenados, :]
            df_exp_plot = df_exp.iloc[inds_ordenados, :]
        elif(ordenar_linhas == 'descending'):
            inds = np.array(pd.Series(np.arange(self.df_exp_varpred.shape[0]), index = self.df_exp_varpred.index)[df_exp.index])
            y_pred = self.__y_pred[inds]
            inds_ordenados = np.argsort(y_pred)[::-1]
            df_calor = df_calor.iloc[inds_ordenados, :]
            df_exp_plot = df_exp.iloc[inds_ordenados, :]
        else:
            df_exp_plot = df_exp
            
        if(ordenar_colunas == 'ascending'):
            cols_ordenadas = np.argsort(np.sum(np.abs(self.df_exp_varpred.loc[df_exp.index, :].values), axis = 0))
            if(pos_colunas_restantes == 'left'):
                cols = np.append([c for c in df_exp.columns if c not in self.__features], self.__features[cols_ordenadas])
            else:
                cols = np.append(self.__features[cols_ordenadas], [c for c in df_exp.columns if c not in self.__features])
            df_calor = df_calor.loc[:, cols]
            df_exp_plot = df_exp_plot.loc[:, cols]
        elif(ordenar_colunas == 'descending'):
            cols_ordenadas = np.argsort(np.sum(np.abs(self.df_exp_varpred.loc[df_exp.index, :].values), axis = 0))[::-1]
            if(pos_colunas_restantes == 'left'):
                cols = np.append([c for c in df_exp.columns if c not in self.__features], self.__features[cols_ordenadas])
            else:
                cols = np.append(self.__features[cols_ordenadas], [c for c in df_exp.columns if c not in self.__features])
            df_calor = df_calor.loc[:, cols]
            df_exp_plot = df_exp_plot.loc[:, cols]
        else:
            df_exp_plot = df_exp_plot
        
        def apply_color(x):
            return df_calor
        styles = [dict(selector="th", props=[("background-color", "white"), ("color", "black")]),
                  dict(selector="tr", props=[("background-color", "white"), ("color", "black")])]
        styler = df_exp_plot.style.set_table_styles(styles).apply(apply_color, axis = None)
        return styler

##############################

##############################

class ImportanciaVariaveisClassificacao:

    def __init__(self, clf, df, col_alvo, cols_features = None, num_loop = 5, random_state = None):
        self.__y = df[col_alvo].values
        self.__tam = self.__y.size
        if(cols_features is None):
            self.__features = np.array([col for col in list(df.columns) if col != col_alvo])
        else:
            self.__features = np.array(cols_features)
        self.__col_alvo = col_alvo
        self.__num_loop = num_loop
        
        self.__importancias = None
        self.__incerteza = None
        self.__dict_imp = None
        
        if(random_state is not None):
            rng = np.random.default_rng(seed=42)
        else:
            rng = np.random.default_rng()
            
        self.__calcula_importancias(df, clf, rng)
        
    def __calcula_importancias(self, df, clf, rng):
        df_temp = df[self.__features].copy()
        y_prob = clf.predict_proba(df_temp)[:, 1]
        logloss_ini = logloss(self.__y, y_prob)
        
        vetor_logloss = np.array([])
        vetor_incerteza = np.array([])
        for feature in self.__features:
            logloss_steps = np.array([])
            valores_feature = df_temp[feature].values
            for i in range(0, self.__num_loop):
                df_temp[feature] = rng.choice(valores_feature, self.__tam, replace = False)
                y_prob = clf.predict_proba(df_temp)[:, 1]
                logloss_steps = np.append(logloss_steps, logloss(self.__y, y_prob))
            df_temp[feature] = df[feature]
            vetor_logloss = np.append(vetor_logloss, np.mean(logloss_steps))
            vetor_incerteza = np.append(vetor_incerteza, np.std(logloss_steps))
        
        self.__piora_logloss = vetor_logloss/logloss_ini - 1
        self.__incerteza_piora = divisao_vetores(vetor_incerteza, logloss_ini)
        
        self.__importancias = np.abs(self.__piora_logloss)
        soma = soma_vetor(self.__importancias)
        self.__importancias = divisao_vetores(self.__importancias, soma)
        
        self.__incerteza = divisao_vetores(vetor_incerteza, logloss_ini*soma)
        
        self.__dict_piora_logloss = {self.__features[i]:self.__piora_logloss[i] for i in range(0, self.__features.size)}
        #self.__dict_piora_logloss = dict(reversed(sorted(self.__dict_piora_logloss.items(), key = lambda x: x[1])))
        
        self.__dict_imp = {self.__features[i]:self.__importancias[i] for i in range(0, self.__features.size)}
        #self.__dict_imp = dict(reversed(sorted(self.__dict_imp.items(), key = lambda x: x[1])))
    
    def retorna_piora_logloss(self):
        return self.__dict_piora_logloss
        
    def retorna_incertezas_piora(self):
        return self.__incerteza_piora
    
    def retorna_importancias(self):
        return self.__dict_imp
        
    def retorna_incertezas(self):
        return self.__incerteza
        
##############################

##############################

class ImportanciaVariaveisRegressao:

    def __init__(self, rgs, df, col_alvo, cols_features = None, num_loop = 5, random_state = None):
        self.__y = df[col_alvo].values
        self.__tam = self.__y.size
        if(cols_features is None):
            self.__features = np.array([col for col in list(df.columns) if col != col_alvo])
        else:
            self.__features = np.array(cols_features)
        self.__col_alvo = col_alvo
        self.__num_loop = num_loop
        
        self.__importancias = None
        self.__incerteza = None
        self.__dict_imp = None
        
        if(random_state is not None):
            rng = np.random.default_rng(seed=42)
        else:
            rng = np.random.default_rng()
            
        self.__calcula_importancias(df, rgs, rng)
        
    def __calcula_importancias(self, df, rgs, rng):
        df_temp = df[self.__features].copy()
        y_pred = rgs.predict(df_temp)
        mse_ini = calcula_mse(diferenca_vetores(self.__y, y_pred))
        
        vetor_mse = np.array([])
        vetor_incerteza = np.array([])
        for feature in self.__features:
            mse_steps = np.array([])
            valores_feature = df_temp[feature].values
            for i in range(0, self.__num_loop):
                df_temp[feature] = rng.choice(valores_feature, self.__tam, replace = False)
                y_pred = rgs.predict(df_temp)
                mse_steps = np.append(mse_steps, calcula_mse(diferenca_vetores(self.__y, y_pred)))
            df_temp[feature] = df[feature]
            vetor_mse = np.append(vetor_mse, np.mean(mse_steps))
            vetor_incerteza = np.append(vetor_incerteza, np.std(mse_steps))
        
        self.__piora_mse = vetor_mse/mse_ini - 1
        self.__incerteza_piora = divisao_vetores(vetor_incerteza, mse_ini)
        
        self.__importancias = np.abs(self.__piora_mse)
        soma = soma_vetor(self.__importancias)
        self.__importancias = divisao_vetores(self.__importancias, soma)
        
        self.__incerteza = divisao_vetores(vetor_incerteza, mse_ini*soma)
        
        self.__dict_piora_mse = {self.__features[i]:self.__piora_mse[i] for i in range(0, self.__features.size)}
        #self.__dict_piora_mse = dict(reversed(sorted(self.__dict_piora_mse.items(), key = lambda x: x[1])))
        
        self.__dict_imp = {self.__features[i]:self.__importancias[i] for i in range(0, self.__features.size)}
        #self.__dict_imp = dict(reversed(sorted(self.__dict_imp.items(), key = lambda x: x[1])))
    
    def retorna_piora_mse(self):
        return self.__dict_piora_mse
        
    def retorna_incertezas_piora(self):
        return self.__incerteza_piora
    
    def retorna_importancias(self):
        return self.__dict_imp
        
    def retorna_incertezas(self):
        return self.__incerteza