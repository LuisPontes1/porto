import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import re

from AleTransforms import *

class DistribuicoesDataset:

    def __init__(self, df, num_div = None, num_cat = None, unit = None, autorun = True):
    
        self.__df = df.copy()
        self.__shape = df.shape
        self.__colunas = df.columns.values
        
        self.__dict_flag_na = df.isna().to_dict(orient = 'list')
        self.__dict_qtds_na = dict(df.isna().sum())
        
        self.__dict_tipos = dict(df.dtypes)
        self.__tipo_numerico = np.array([])
        self.__tipo_categorico = np.array([])
        self.__tipo_temporal = np.array([])
        for col in self.__colunas:
            if(self.__dict_tipos[col] in [np.number, 'int64', 'int32', 'float64', 'float32']):
                self.__tipo_numerico = np.append(self.__tipo_numerico, col)
            elif(self.__dict_tipos[col] in ['<M8[ns]', 'datetime64[ns]']):
                self.__tipo_temporal = np.append(self.__tipo_temporal, col)
            else:
                self.__tipo_categorico = np.append(self.__tipo_categorico, col)
        
        self.__valores_unicos = {}
        self.__qtds_unicos = {}
        self.__colunas_tratadas = np.array([])
        
        self.__tratadf = TrataDataset(df, num_div = num_div, num_cat = num_cat, 
                                      features_numericas = list(self.__tipo_numerico), 
                                      features_categoricas = list(self.__tipo_categorico),
                                      features_temporais = list(self.__tipo_temporal),
                                      unit = unit, autorun = autorun)        
        self.__dict_intervs, self.__dict_filtroscat = self.__tratadf.retorna_instancias_tratamento()
        
        if(autorun):
            for col in self.__tipo_numerico:
                self.__valores_unicos[col], self.__qtds_unicos[col], _ = unicos_qtds(df[col].dropna().values)
            for col in self.__tipo_temporal:
                self.__valores_unicos[col], self.__qtds_unicos[col], _ = unicos_qtds(df[col].dropna().values.view('i8'))
                self.__valores_unicos[col] = self.__valores_unicos[col].astype('<M8[ns]')
            self.__colunas_tratadas = self.__colunas

    def trata_coluna(self, df, feature, parametros_padrao = True, num_div = 20, num_cat = 5, unit = None):
        self.__tratadf.trata_coluna(df, feature, parametros_padrao, num_div, num_cat, unit)
        self.__dict_intervs, self.__dict_filtroscat = self.__tratadf.retorna_instancias_tratamento()
        if(self.__tipo_numerico.size > 0 and (feature in self.__tipo_numerico)):
            self.__valores_unicos[feature], self.__qtds_unicos[feature], _ = unicos_qtds(df[feature].dropna().values)
        elif(self.__tipo_temporal.size > 0 and (feature in self.__tipo_temporal)):
            self.__valores_unicos[feature], self.__qtds_unicos[feature], _ = unicos_qtds(df[feature].dropna().values.view('i8'))
            self.__valores_unicos[feature] = self.__valores_unicos[feature].astype('<M8[ns]')
        self.__colunas_tratadas = np.append(self.__colunas_tratadas, feature)
        
    def coluna_foi_tratada(self, feature):
        if(feature in self.__colunas_tratadas):
            return True
        else:
            return False
            
    def coluna_tem_densidade(self, feature):
        if(feature in self.__dict_intervs.keys()):
            return True
        else:
            return False
    
    def retorna_colunas_dataset(self):
        return self.__colunas
        
    def retorna_shape_dataset(self):
        return self.__shape
        
    def retorna_trata_dataset(self):
        return self.__tratadf
        
    def retorna_flag_na(self, col_ref = None):
        if(col_ref is None):
            return self.__dict_flag_na
        else:
            return self.__dict_flag_na[col_ref]
        
    def retorna_qtds_na(self, col_ref = None):
        if(col_ref is None):
            return self.__dict_qtds_na
        else:
            return self.__dict_qtds_na[col_ref]
    
    def retorna_valores(self, df, col_ref):
        if(self.coluna_foi_tratada(col_ref)):
            if(col_ref in self.__dict_intervs.keys()):
                valores = self.__dict_intervs[col_ref].aplica_discretizacao(df[col_ref].values, usar_ponto_medio = False)
                #tipo = 'intervalo'
            elif(col_ref in self.__dict_filtroscat.keys()):
                valores = self.__dict_filtroscat[col_ref].aplica_filtro_categorias(df[col_ref].values, considera_resto = True, usar_str = False)
                #tipo = 'categoria'
            else:
                valores = df[col_ref].values
                if(self.__tipo_temporal.size > 0 and (col_ref in self.__tipo_temporal)):
                    valores = valores.view('i8')
                #tipo = 'discreto'
            return valores
    
    def curva_densidade(self, col_ref):
        if(self.coluna_foi_tratada(col_ref)):
            if(col_ref in self.__dict_intervs.keys()):
                valores, fracL = self.__dict_intervs[col_ref].curva_densidade()
                #tipo = 'intervalo'
                return valores, fracL
    
    def curva_distribuicao(self, col_ref, bins = None, explicita_resto = False):
        if(self.coluna_foi_tratada(col_ref)):
            if(col_ref in self.__dict_intervs.keys()):
                valores, frac, largura = self.__dict_intervs[col_ref].curva_distribuicao(bins = bins)
                #tipo = 'intervalo'
            elif(col_ref in self.__dict_filtroscat.keys()):
                valores, frac = self.__dict_filtroscat[col_ref].curva_distribuicao(explicita_resto = explicita_resto)
                largura = 1
                #tipo = 'categoria'
            else:
                valores = self.__valores_unicos[col_ref]
                frac = self.__qtds_unicos[col_ref]/self.__shape[0]
                diff_valores = np.diff(valores)
                if(diff_valores.size > 0):
                    largura = np.min(np.diff(valores))
                else:
                    largura = 1
                #tipo = 'discreto'
            return valores, frac, largura

    def info_distribuicao(self, col_ref):
        if(self.coluna_foi_tratada(col_ref)):
            if(col_ref in self.__dict_intervs.keys()):
                df_info = self.__dict_intervs[col_ref].info_discretizacao()
                #tipo = 'intervalo'
            elif(col_ref in self.__dict_filtroscat.keys()):
                df_info = self.__dict_filtroscat[col_ref].info_categorias()
                #tipo = 'categoria'
            else:
                df_info = pd.DataFrame(zip(self.__qtds_unicos[col_ref], self.__valores_unicos[col_ref]), columns = ['QTD', 'Valor'])
                #tipo = 'discreto'
            return df_info

    def grafico_densidade(self, col_ref = [], alpha = 0.5, rot = None, figsize = [6, 4]):
        #Transforma uma string única em uma lista
        if(isinstance(col_ref, str)):
            col_ref = [col_ref]
        #Se não passar nada plota o gráfico de todas as colunas
        if(len(col_ref) == 0):
            colunas = self.__colunas
        else:
            colunas = col_ref
        for col_ref in colunas:
            if(self.coluna_tem_densidade(col_ref)):
                paleta_cores = sns.color_palette("colorblind") #Paleta de cores para daltonico
                #Plota informações da densidade da variável de referência nos dados
                with sns.axes_style("whitegrid"):
                    fig, ax = plt.subplots(1, 1, figsize = figsize)
                    valores, fracL = self.curva_densidade(col_ref)
                    frac_na = self.__dict_qtds_na[col_ref]/self.__shape[0]
                    ax.fill_between(valores, fracL, color = paleta_cores[0], alpha = alpha)
                    ax.plot(valores, fracL, color = paleta_cores[0])
                    ax.set_ylabel('Fração/L')
                    plt.gcf().text(1, 0.8, 'Fração de NA = ' + '%.2g' % frac_na, bbox = dict(facecolor = 'white', edgecolor = 'k', boxstyle = 'round'))
                    ax.set_xlabel(col_ref)
                    ax.set_ylim(bottom = 0.0)
                    if(rot is not None):
                        plt.xticks(rotation = rot)
                    plt.show()

    def grafico_distribuicao(self, col_ref = [], alpha = 0.5, bins = None, explicita_resto = False, rot = None, conv_str = False, ticks_chars = None, figsize = [6, 4]):
        #Transforma uma string única em uma lista
        if(isinstance(col_ref, str)):
            col_ref = [col_ref]
        #Se não passar nada plota o gráfico de todas as colunas
        if(len(col_ref) == 0):
            colunas = self.__colunas
        else:
            colunas = col_ref
        for col_ref in colunas:
            if(self.coluna_foi_tratada(col_ref)):
                paleta_cores = sns.color_palette("colorblind") #Paleta de cores para daltonico
                #Plota informações da distribuição da variável de referência nos dados
                with sns.axes_style("whitegrid"):
                    fig, ax = plt.subplots(1, 1, figsize = figsize)
                    valores, frac, largura = self.curva_distribuicao(col_ref, bins = bins, explicita_resto = explicita_resto)
                    frac_na = self.__dict_qtds_na[col_ref]/self.__shape[0]
                    if(conv_str):
                        valores = valores.astype(str)
                        largura = 1
                    if(ticks_chars is not None):
                        valores = [re.sub("(.{" + str(ticks_chars) + "})", "\\1\n", v, 0, re.DOTALL) for v in valores]
                    ax.bar(valores, frac, color = paleta_cores[0], alpha = alpha, width = largura, linewidth = 2, edgecolor = paleta_cores[0])
                    ax.set_ylabel('Fração')
                    plt.gcf().text(1, 0.8, 'Fração de NA = ' + '%.2g' % frac_na, bbox = dict(facecolor = 'white', edgecolor = 'k', boxstyle = 'round'))
                    ax.set_xlabel(col_ref)
                    ax.set_ylim(bottom = 0.0)
                    if(rot is not None):
                        plt.xticks(rotation = rot)
                    plt.show()
                    
##############################

##############################

class AvaliaDatasetsDistribuicoes:

    def __init__(self, dict_dfs, num_div = None, num_cat = None, unit = None, autorun = False):
        self.__num_dfs = len(dict_dfs)
        self.__chaves = list(dict_dfs.keys())
        
        self.__dict_distribuicoes = {}
        for chave in self.__chaves:
            self.__dict_distribuicoes[chave] = DistribuicoesDataset(dict_dfs[chave], num_div = num_div, num_cat = num_cat, unit = unit, autorun = autorun)
        
        self.__colunas = self.__dict_distribuicoes[self.__chaves[0]].retorna_colunas_dataset()
        
    def trata_coluna(self, dict_dfs, col_ref = [], parametros_padrao = True, num_div = 20, num_cat = 5, unit = None):
        colunas = self.__colunas
        #Transforma uma string única em uma lista
        if(isinstance(col_ref, str)):
            col_ref = [col_ref]
        if(len(col_ref) != 0):
            colunas = col_ref
        for col_ref in colunas:
            for chave in self.__chaves:
                self.__dict_distribuicoes[chave].trata_coluna(dict_dfs[chave], col_ref, parametros_padrao, num_div, num_cat, unit)
                    
    def info_distribuicao(self, col_ref):
        if(self.__dict_distribuicoes[self.__chaves[0]].coluna_foi_tratada(col_ref)):
            d = {}
            for chave in self.__chaves:
                d[chave] = self.__dict_distribuicoes[chave].info_distribuicao(col_ref)
            return d
    
    def grafico_densidade(self, col_ref = [], alpha = 0.5, rot = None, figsize = [6, 4]):
        colunas = self.__colunas
        #Transforma uma string única em uma lista
        if(isinstance(col_ref, str)):
            col_ref = [col_ref]
        if(len(col_ref) != 0):
            colunas = col_ref
            
        for col_ref in colunas:
            if(self.__dict_distribuicoes[self.__chaves[0]].coluna_tem_densidade(col_ref)):
                paleta_cores = sns.color_palette("colorblind") #Paleta de cores para daltonico
                #Plota informações da distribuição da variável de referência nos dados
                with sns.axes_style("whitegrid"):
                    fig, ax = plt.subplots(1, 1, figsize = figsize)
                    i = 0
                    for chave in self.__chaves:
                        valores, fracL = self.__dict_distribuicoes[chave].curva_densidade(col_ref)
                        frac_na = self.__dict_distribuicoes[chave].retorna_qtds_na(col_ref)/self.__dict_distribuicoes[chave].retorna_shape_dataset()[0]
                        label_na = ' (Fração de NA: ' + '%.2g' % frac_na + ')'
                        ax.fill_between(valores, fracL, color = paleta_cores[i], alpha = alpha)
                        ax.plot(valores, fracL, color = paleta_cores[i], label = chave + label_na)
                        i = i + 1
                    ax.set_ylabel('Fração/L')
                    ax.set_xlabel(col_ref)
                    ax.set_ylim(bottom = 0.0)
                    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                    if(rot is not None):
                        plt.xticks(rotation = rot)
                    plt.show()
                
    def grafico_distribuicao(self, col_ref = [], alpha = 0.5, bins = None, explicita_resto = False, rot = None, conv_str = False, ticks_chars = None, figsize = [6, 4]):
        colunas = self.__colunas
        #Transforma uma string única em uma lista
        if(isinstance(col_ref, str)):
            col_ref = [col_ref]
        if(len(col_ref) != 0):
            colunas = col_ref
            
        for col_ref in colunas:
            if(self.__dict_distribuicoes[self.__chaves[0]].coluna_foi_tratada(col_ref)):
                paleta_cores = sns.color_palette("colorblind") #Paleta de cores para daltonico
                #Plota informações da distribuição da variável de referência nos dados
                with sns.axes_style("whitegrid"):
                    fig, ax = plt.subplots(1, 1, figsize = figsize)
                    i = 0
                    for chave in self.__chaves:
                        valores, frac, largura = self.__dict_distribuicoes[chave].curva_distribuicao(col_ref, bins = bins, explicita_resto = explicita_resto)
                        frac_na = self.__dict_distribuicoes[chave].retorna_qtds_na(col_ref)/self.__dict_distribuicoes[chave].retorna_shape_dataset()[0]
                        if(conv_str):
                            valores = valores.astype(str)
                            largura = 1
                        if(ticks_chars is not None):
                            valores = [re.sub("(.{" + str(ticks_chars) + "})", "\\1\n", v, 0, re.DOTALL) for v in valores]
                        label_na = ' (Fração de NA: ' + '%.2g' % frac_na + ')'
                        ax.bar(valores, frac, color = paleta_cores[i], alpha = alpha, width = largura, linewidth = 2, edgecolor = paleta_cores[i], label = chave + label_na)
                        i = i + 1
                    ax.set_ylabel('Fração')
                    ax.set_xlabel(col_ref)
                    ax.set_ylim(bottom = 0.0)
                    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                    if(rot is not None):
                        plt.xticks(rotation = rot)
                    plt.show()