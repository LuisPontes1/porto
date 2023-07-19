import numpy as np
import pandas as pd

from AleCFunctions import *

import matplotlib.pyplot as plt
import seaborn as sns
import re

from sklearn.preprocessing import OneHotEncoder

from itertools import combinations_with_replacement, combinations

class CortaIntervalosQuasiUniforme:

    def __init__(self, vetor, num_div, eh_temporal = False, unit = None):
        
        self.__eh_temporal = eh_temporal
        if(eh_temporal == True):
            valores, qtds, qtd_unicos = unicos_qtds(vetor.view('i8'))
        else:
            valores, qtds, qtd_unicos = unicos_qtds(vetor)
        
        pts_corte, qtds_corte = pontos_corte(qtds, qtd_unicos, num_div)
        pts_corte, self.__qtds_corte = minimiza_desvio_padrao(pts_corte, qtds_corte, qtds, qtd_unicos)
        self.__valores_min, self.__valores_max = calcula_valores_corte(valores, pts_corte)
        self.__valores_medios = calcula_pontos_medios(self.__valores_min, self.__valores_max)
        
        if(eh_temporal):
            self.__valores_min = self.__valores_min.astype('<M8[ns]')
            self.__valores_max = self.__valores_max.astype('<M8[ns]')
            self.__valores_medios = self.__valores_medios.astype('<M8[ns]')
            self.__strings_intervalos = self.__calcula_intervalos_datetime(self.__valores_min, self.__valores_max, unit)
            self.__valores_digitize = self.__valores_max[:-1].view('i8')
        else:
            self.__strings_intervalos = self.__calcula_intervalos_algasignif(self.__valores_min, self.__valores_max)
            self.__valores_digitize = self.__valores_max[:-1]
        
        self.__pares_minimo_maximo = calcula_pares_minmax(self.__valores_min, self.__valores_max)
        
    def __calcula_intervalos_algasignif(self, valores_min, valores_max):
        valores_corte = np.append(valores_min, valores_max[-1])
        alga_signif = 1
        str_conv = ''.join(['%.', np.str(alga_signif), 'g'])
        cortes_interv = np.array([float(str_conv % valores_corte[i]) for i in range(valores_corte.size)])
        flag = calcula_min_diff(cortes_interv) #não pode ter pontos de corte iguais
        while flag == 0:
            alga_signif += 1
            str_conv = ''.join(['%.', np.str(alga_signif), 'g'])
            cortes_interv = np.array([float(str_conv % valores_corte[i]) for i in range(valores_corte.size)])
            flag = calcula_min_diff(cortes_interv)
        strings = [str(v) for v in cortes_interv]
        strings = [v if v[-2:] != '.0' else v[:-2] for v in strings]
        return np.array(['['+strings[i]+', '+strings[i+1]+')' for i in range(valores_corte.size-1)])

    def __calcula_intervalos_datetime(self, valores_min, valores_max, unit):
        valores_corte = np.append(valores_min, valores_max[-1])
        strings = [np.datetime_as_string(v, unit = unit) for v in valores_corte]
        return np.array(['['+strings[i]+', '+strings[i+1]+')' for i in range(valores_corte.size-1)])
    
    def pares_minimo_maximo_discretizacao(self):
        return self.__pares_minimo_maximo

    def valores_medios_discretizacao(self):
        return self.__valores_medios

    def strings_intervalos_discretizacao(self):
        return self.__strings_intervalos
    
    def info_discretizacao(self):
        #Retorna um dataframe com as informações relevantes de cada intervalo da discretização
        df = pd.DataFrame(zip(self.__qtds_corte, self.__valores_min, self.__valores_max, self.__valores_medios, self.__strings_intervalos), 
                          columns = ['QTD', 'Min', 'Max', 'Meio', 'Str'])
        return df
    
    def curva_densidade(self):
        valores = np.array([x for y in self.__pares_minimo_maximo for x in y])
        if(self.__eh_temporal):
            fracL = np.repeat(self.__qtds_corte/(self.__valores_max - self.__valores_min).astype(float), 2)
        else:
            fracL = np.repeat(self.__qtds_corte/(self.__valores_max - self.__valores_min), 2)
        return valores, fracL
    
    def curva_distribuicao(self, bins = None):
        if bins is None:
            bins = self.__valores_medios.size
        val_min = self.__valores_min[0]
        val_max = self.__valores_max[-1]
        L = (val_max - val_min)/bins
        valores_corte_bins = [val_min + L*i for i in range(0, bins+1)]
        qtds_bins = [conta_qtds_bin(valores_corte_bins[i], valores_corte_bins[i+1], self.__valores_min, self.__valores_max, self.__qtds_corte) for i in range(0, bins)]
        
        valores_medios_bins = np.array([val_min + L*(i + 1/2) for i in range(1, bins+1)])
        frac_bins = np.array(qtds_bins)/np.sum(self.__qtds_corte)
        return valores_medios_bins, frac_bins, L
   
    def grafico_densidade(self, alpha = 0.5, rot = None, figsize = [6, 4]):
        paleta_cores = sns.color_palette("colorblind")
        with sns.axes_style("whitegrid"):
            fig, ax = plt.subplots(1, 1, figsize = figsize)
            valores, fracL = self.curva_densidade()
            ax.fill_between(valores, fracL, color = paleta_cores[0], alpha = alpha)
            ax.plot(valores, fracL, color = paleta_cores[0])
            ax.set_xlabel('Valores')
            ax.set_ylabel('Fração/L')
            ax.set_ylim(bottom = 0.0)
            if(rot is not None):
                plt.xticks(rotation = rot)
            plt.show()

    def grafico_distribuicao(self, alpha = 0.5, bins = None, rot = None, figsize = [6, 4]):
        paleta_cores = sns.color_palette("colorblind")
        with sns.axes_style("whitegrid"):
            fig, ax = plt.subplots(1, 1, figsize = figsize)
            valores, frac, largura = self.curva_distribuicao(bins = bins)
            ax.bar(valores, frac, color = paleta_cores[0], alpha = alpha, width = largura, linewidth = 2, edgecolor = paleta_cores[0])
            ax.set_xlabel('Valores')
            ax.set_ylabel('Fração')
            ax.set_ylim(bottom = 0.0)
            if(rot is not None):
                plt.xticks(rotation = rot)
            plt.show()
            
    def aplica_discretizacao(self, vetor, usar_ponto_medio = False):
        if(self.__eh_temporal):
            flag_na = np.isnan(vetor)
            disc = discretiza_vetor(vetor.view('i8'), self.__valores_digitize)
            if(usar_ponto_medio):
                disc = disc.astype(int)
                disc = np.array([self.__valores_medios[disc[i]] if ~flag_na[i] else np.datetime64("NaT") for i in range(disc.size)])
                return disc
            else:
                disc[flag_na] = np.nan
                return disc
        else:
            if(usar_ponto_medio):
                return discretiza_vetor_media(vetor, self.__valores_digitize, self.__valores_medios)
            else:
                return discretiza_vetor(vetor, self.__valores_digitize)

########################
 
########################

class CortaIntervalosGanhoInformacao:

    def __init__(self, vetor, alvo, num_div, qtd_min = 0, eh_temporal = False, unit = None, balancear = False):
        
        self.__eh_temporal = eh_temporal
        if(eh_temporal == True):
            valores, qtds, qtds_alvo, qtd_unicos = unicos_qtds_alvos(vetor.view('i8'), alvo)
        else:
            valores, qtds, qtds_alvo, qtd_unicos = unicos_qtds_alvos(vetor, alvo)
        
        pts_corte, qtds_corte, qtds_alvo_corte = pontos_corte_alvo(qtds, qtd_unicos, qtds_alvo, num_div)
        if(balancear):
            pts_corte, self.__qtds_corte, self.__qtds_alvo_corte = minimiza_entropia_balanceada(pts_corte, qtds_corte, qtds, qtds_alvo_corte, qtds_alvo, qtd_unicos, qtd_min)
        else:
            pts_corte, self.__qtds_corte, self.__qtds_alvo_corte = minimiza_entropia(pts_corte, qtds_corte, qtds, qtds_alvo_corte, qtds_alvo, qtd_unicos, qtd_min)
        self.__valores_min, self.__valores_max = calcula_valores_corte(valores, pts_corte)
        self.__valores_medios = calcula_pontos_medios(self.__valores_min, self.__valores_max)
        
        if(eh_temporal):
            self.__valores_min = self.__valores_min.astype('<M8[ns]')
            self.__valores_max = self.__valores_max.astype('<M8[ns]')
            self.__valores_medios = self.__valores_medios.astype('<M8[ns]')
            self.__strings_intervalos = self.__calcula_intervalos_datetime(self.__valores_min, self.__valores_max, unit)
            self.__valores_digitize = self.__valores_max[:-1].view('i8')
        else:
            self.__strings_intervalos = self.__calcula_intervalos_algasignif(self.__valores_min, self.__valores_max)
            self.__valores_digitize = self.__valores_max[:-1]
        
        self.__pares_minimo_maximo = calcula_pares_minmax(self.__valores_min, self.__valores_max)
        
    def __calcula_intervalos_algasignif(self, valores_min, valores_max):
        valores_corte = np.append(valores_min, valores_max[-1])
        alga_signif = 1
        str_conv = ''.join(['%.', np.str(alga_signif), 'g'])
        cortes_interv = np.array([float(str_conv % valores_corte[i]) for i in range(valores_corte.size)])
        flag = calcula_min_diff(cortes_interv) #não pode ter pontos de corte iguais
        while flag == 0:
            alga_signif += 1
            str_conv = ''.join(['%.', np.str(alga_signif), 'g'])
            cortes_interv = np.array([float(str_conv % valores_corte[i]) for i in range(valores_corte.size)])
            flag = calcula_min_diff(cortes_interv)
        strings = [str(v) for v in cortes_interv]
        strings = [v if v[-2:] != '.0' else v[:-2] for v in strings]
        return np.array(['['+strings[i]+', '+strings[i+1]+')' for i in range(valores_corte.size-1)])

    def __calcula_intervalos_datetime(self, valores_min, valores_max, unit):
        valores_corte = np.append(valores_min, valores_max[-1])
        strings = [np.datetime_as_string(v, unit = unit) for v in valores_corte]
        return np.array(['['+strings[i]+', '+strings[i+1]+')' for i in range(valores_corte.size-1)])
    
    def pares_minimo_maximo_discretizacao(self):
        return self.__pares_minimo_maximo

    def valores_medios_discretizacao(self):
        return self.__valores_medios

    def strings_intervalos_discretizacao(self):
        return self.__strings_intervalos
    
    def info_discretizacao(self):
        #Retorna um dataframe com as informações relevantes de cada intervalo da discretização
        df = pd.DataFrame(zip(self.__qtds_corte, self.__qtds_alvo_corte, self.__valores_min, self.__valores_max, self.__valores_medios, self.__strings_intervalos), 
                          columns = ['QTD', 'QTD_Alvos', 'Min', 'Max', 'Meio', 'Str'])
        return df
    
    def curva_densidade(self):
        valores = np.array([x for y in self.__pares_minimo_maximo for x in y])
        if(self.__eh_temporal):
            fracL = np.repeat(self.__qtds_corte/(self.__valores_max - self.__valores_min).astype(float), 2)
        else:
            fracL = np.repeat(self.__qtds_corte/(self.__valores_max - self.__valores_min), 2)
        return valores, fracL
    
    def curva_distribuicao(self, bins = None):
        if bins is None:
            bins = self.__valores_medios.size
        val_min = self.__valores_min[0]
        val_max = self.__valores_max[-1]
        L = (val_max - val_min)/bins
        valores_corte_bins = [val_min + L*i for i in range(0, bins+1)]
        qtds_bins = [conta_qtds_bin(valores_corte_bins[i], valores_corte_bins[i+1], self.__valores_min, self.__valores_max, self.__qtds_corte) for i in range(0, bins)]
        
        valores_medios_bins = np.array([val_min + L*(i + 1/2) for i in range(1, bins+1)])
        frac_bins = np.array(qtds_bins)/np.sum(self.__qtds_corte)
        return valores_medios_bins, frac_bins, L
        
    def curva_entropia(self):
        valores = np.array([x for y in self.__pares_minimo_maximo for x in y])
        entropias = np.repeat(entropia_shannon(self.__qtds_alvo_corte/self.__qtds_corte), 2)
        return valores, entropias
   
    def grafico_densidade(self, alpha = 0.5, rot = None, figsize = [6, 4]):
        paleta_cores = sns.color_palette("colorblind")
        with sns.axes_style("whitegrid"):
            fig, ax = plt.subplots(1, 1, figsize = figsize)
            valores, fracL = self.curva_densidade()
            ax.fill_between(valores, fracL, color = paleta_cores[0], alpha = alpha)
            ax.plot(valores, fracL, color = paleta_cores[0])
            ax.set_xlabel('Valores')
            ax.set_ylabel('Fração/L')
            ax.set_ylim(bottom = 0.0)
            if(rot is not None):
                plt.xticks(rotation = rot)
            plt.show()

    def grafico_distribuicao(self, alpha = 0.5, bins = None, rot = None, figsize = [6, 4]):
        paleta_cores = sns.color_palette("colorblind")
        with sns.axes_style("whitegrid"):
            fig, ax = plt.subplots(1, 1, figsize = figsize)
            valores, frac, largura = self.curva_distribuicao(bins = bins)
            ax.bar(valores, frac, color = paleta_cores[0], alpha = alpha, width = largura, linewidth = 2, edgecolor = paleta_cores[0])
            ax.set_xlabel('Valores')
            ax.set_ylabel('Fração')
            ax.set_ylim(bottom = 0.0)
            if(rot is not None):
                plt.xticks(rotation = rot)
            plt.show()
            
    def grafico_entropia(self, alpha = 0.5, rot = None, figsize = [6, 4]):
        paleta_cores = sns.color_palette("colorblind")
        with sns.axes_style("whitegrid"):
            fig, ax = plt.subplots(1, 1, figsize = figsize)
            valores, entropias = self.curva_entropia()
            ax.fill_between(valores, entropias, color = paleta_cores[0], alpha = alpha)
            ax.plot(valores, entropias, color = paleta_cores[0])
            ax.set_xlabel('Valores')
            ax.set_ylabel('Entropia')
            ax.set_ylim(bottom = 0.0)
            #ax.set_ylim(top = 1.01)
            if(rot is not None):
                plt.xticks(rotation = rot)
            plt.show()
            
    def ganho_informacao(self):
        p1 = np.sum(self.__qtds_alvo_corte)/np.sum(self.__qtds_corte)
        entropia_ini = - p1*np.log2(p1) - (1 - p1)*np.log2(1 - p1)
        entropia = np.sum(self.__qtds_corte*entropia_shannon(self.__qtds_alvo_corte/self.__qtds_corte))/np.sum(self.__qtds_corte)
        return 1 - entropia/entropia_ini
        
    def ganho_informacao_balanceado(self):
        entropia = np.sum(entropia_shannon(self.__qtds_alvo_corte/self.__qtds_corte))/self.__qtds_corte.size
        return 1 - entropia
    
    def calcula_ig(self, vetor, alvo):
        p1 = np.sum(alvo)/alvo.size
        entropia_ini = - p1*np.log2(p1) - (1 - p1)*np.log2(1 - p1)
        inds_ordenado, primeira_ocorrencia, qtds_corte, _ = indices_qtds(self.aplica_discretizacao(vetor))
        alvo_agrup = np.split(alvo[inds_ordenado], primeira_ocorrencia[1:])
        qtds_alvo_corte = np.array([soma_vetor(v) for v in alvo_agrup])
        entropia = np.sum(qtds_corte*entropia_shannon(qtds_alvo_corte/qtds_corte))/np.sum(qtds_corte)
        return 1 - entropia/entropia_ini
        
    def calcula_ig_bal(self, vetor, alvo):
        inds_ordenado, primeira_ocorrencia, qtds_corte, _ = indices_qtds(self.aplica_discretizacao(vetor))
        alvo_agrup = np.split(alvo[inds_ordenado], primeira_ocorrencia[1:])
        qtds_alvo_corte = np.array([soma_vetor(v) for v in alvo_agrup])
        entropia = np.sum(entropia_shannon(qtds_alvo_corte/qtds_corte))/qtds_corte.size
        return 1 - entropia
            
    def aplica_discretizacao(self, vetor, usar_ponto_medio = False):
        if(self.__eh_temporal):
            flag_na = np.isnan(vetor)
            disc = discretiza_vetor(vetor.view('i8'), self.__valores_digitize)
            if(usar_ponto_medio):
                disc = disc.astype(int)
                disc = np.array([self.__valores_medios[disc[i]] if ~flag_na[i] else np.datetime64("NaT") for i in range(disc.size)])
                return disc
            else:
                disc[flag_na] = np.nan
                return disc
        else:
            if(usar_ponto_medio):
                return discretiza_vetor_media(vetor, self.__valores_digitize, self.__valores_medios)
            else:
                return discretiza_vetor(vetor, self.__valores_digitize)

########################
 
########################

class CortaIntervalosR2:

    def __init__(self, vetor, alvo, num_div, qtd_min = 0, eh_temporal = False, unit = None, balancear = False):
        
        self.__eh_temporal = eh_temporal
        if(eh_temporal == True):
            if(balancear):
                valores, qtds, qtds_alvo, qtds_alvo2, qtd_unicos = unicos_qtds_alvos2(vetor.view('i8'), alvo)
            else:
                valores, qtds, qtds_alvo, qtd_unicos = unicos_qtds_alvos(vetor.view('i8'), alvo)
        else:
            if(balancear):
                valores, qtds, qtds_alvo, qtds_alvo2, qtd_unicos = unicos_qtds_alvos2(vetor, alvo)
            else:
                valores, qtds, qtds_alvo, qtd_unicos = unicos_qtds_alvos(vetor, alvo)
        
        self.__media_alvo2 = calcula_mse(alvo) #np.mean(np.power(alvo, 2)) 
        
        if(balancear):
            pts_corte, qtds_corte, qtds_alvo_corte, qtds_alvo2_corte = pontos_corte_alvo2(qtds, qtd_unicos, qtds_alvo, qtds_alvo2, num_div)
            pts_corte, self.__qtds_corte, self.__qtds_alvo_corte = minimiza_mse_balanceado(pts_corte, qtds_corte, qtds, qtds_alvo_corte, qtds_alvo, qtds_alvo2_corte, qtds_alvo2, qtd_unicos, qtd_min)
        else:
            pts_corte, qtds_corte, qtds_alvo_corte = pontos_corte_alvo(qtds, qtd_unicos, qtds_alvo, num_div)
            pts_corte, self.__qtds_corte, self.__qtds_alvo_corte = minimiza_mse(pts_corte, qtds_corte, qtds, qtds_alvo_corte, qtds_alvo, qtd_unicos, qtd_min)
        self.__valores_min, self.__valores_max = calcula_valores_corte(valores, pts_corte)
        self.__valores_medios = calcula_pontos_medios(self.__valores_min, self.__valores_max)
        
        if(eh_temporal):
            self.__valores_min = self.__valores_min.astype('<M8[ns]')
            self.__valores_max = self.__valores_max.astype('<M8[ns]')
            self.__valores_medios = self.__valores_medios.astype('<M8[ns]')
            self.__strings_intervalos = self.__calcula_intervalos_datetime(self.__valores_min, self.__valores_max, unit)
            self.__valores_digitize = self.__valores_max[:-1].view('i8')
        else:
            self.__strings_intervalos = self.__calcula_intervalos_algasignif(self.__valores_min, self.__valores_max)
            self.__valores_digitize = self.__valores_max[:-1]
        
        self.__pares_minimo_maximo = calcula_pares_minmax(self.__valores_min, self.__valores_max)
        
        inds_ordenado, primeira_ocorrencia, _, _ = indices_qtds(self.aplica_discretizacao(vetor))
        alvo_agrup = np.split(alvo[inds_ordenado], primeira_ocorrencia[1:])
        self.__media_alvo2_agrup = np.array([calcula_mse(v) for v in alvo_agrup]) #np.mean(np.power(v, 2)) 
        
    def __calcula_intervalos_algasignif(self, valores_min, valores_max):
        valores_corte = np.append(valores_min, valores_max[-1])
        alga_signif = 1
        str_conv = ''.join(['%.', np.str(alga_signif), 'g'])
        cortes_interv = np.array([float(str_conv % valores_corte[i]) for i in range(valores_corte.size)])
        flag = calcula_min_diff(cortes_interv) #não pode ter pontos de corte iguais
        while flag == 0:
            alga_signif += 1
            str_conv = ''.join(['%.', np.str(alga_signif), 'g'])
            cortes_interv = np.array([float(str_conv % valores_corte[i]) for i in range(valores_corte.size)])
            flag = calcula_min_diff(cortes_interv)
        strings = [str(v) for v in cortes_interv]
        strings = [v if v[-2:] != '.0' else v[:-2] for v in strings]
        return np.array(['['+strings[i]+', '+strings[i+1]+')' for i in range(valores_corte.size-1)])

    def __calcula_intervalos_datetime(self, valores_min, valores_max, unit):
        valores_corte = np.append(valores_min, valores_max[-1])
        strings = [np.datetime_as_string(v, unit = unit) for v in valores_corte]
        return np.array(['['+strings[i]+', '+strings[i+1]+')' for i in range(valores_corte.size-1)])
    
    def pares_minimo_maximo_discretizacao(self):
        return self.__pares_minimo_maximo

    def valores_medios_discretizacao(self):
        return self.__valores_medios

    def strings_intervalos_discretizacao(self):
        return self.__strings_intervalos
    
    def info_discretizacao(self):
        #Retorna um dataframe com as informações relevantes de cada intervalo da discretização
        df = pd.DataFrame(zip(self.__qtds_corte, self.__qtds_alvo_corte, self.__valores_min, self.__valores_max, self.__valores_medios, self.__strings_intervalos), 
                          columns = ['QTD', 'QTD_Alvos', 'Min', 'Max', 'Meio', 'Str'])
        return df
    
    def curva_densidade(self):
        valores = np.array([x for y in self.__pares_minimo_maximo for x in y])
        if(self.__eh_temporal):
            fracL = np.repeat(self.__qtds_corte/(self.__valores_max - self.__valores_min).astype(float), 2)
        else:
            fracL = np.repeat(self.__qtds_corte/(self.__valores_max - self.__valores_min), 2)
        return valores, fracL
    
    def curva_distribuicao(self, bins = None):
        if bins is None:
            bins = self.__valores_medios.size
        val_min = self.__valores_min[0]
        val_max = self.__valores_max[-1]
        L = (val_max - val_min)/bins
        valores_corte_bins = [val_min + L*i for i in range(0, bins+1)]
        qtds_bins = [conta_qtds_bin(valores_corte_bins[i], valores_corte_bins[i+1], self.__valores_min, self.__valores_max, self.__qtds_corte) for i in range(0, bins)]
        
        valores_medios_bins = np.array([val_min + L*(i + 1/2) for i in range(1, bins+1)])
        frac_bins = np.array(qtds_bins)/np.sum(self.__qtds_corte)
        return valores_medios_bins, frac_bins, L
        
    def curva_mse_normalizado(self):
        valores = np.array([x for y in self.__pares_minimo_maximo for x in y])
        media_geral = np.sum(self.__qtds_alvo_corte)/np.sum(self.__qtds_corte)
        mse_ini_agroup = self.__media_alvo2_agrup + media_geral*(media_geral - 2*self.__qtds_alvo_corte/self.__qtds_corte)
        mse_agrup = self.__media_alvo2_agrup - np.power(self.__qtds_alvo_corte/self.__qtds_corte, 2)
        mse_norm = np.repeat(mse_agrup/mse_ini_agroup, 2)
        return valores, mse_norm
   
    def grafico_densidade(self, alpha = 0.5, rot = None, figsize = [6, 4]):
        paleta_cores = sns.color_palette("colorblind")
        with sns.axes_style("whitegrid"):
            fig, ax = plt.subplots(1, 1, figsize = figsize)
            valores, fracL = self.curva_densidade()
            ax.fill_between(valores, fracL, color = paleta_cores[0], alpha = alpha)
            ax.plot(valores, fracL, color = paleta_cores[0])
            ax.set_xlabel('Valores')
            ax.set_ylabel('Fração/L')
            ax.set_ylim(bottom = 0.0)
            if(rot is not None):
                plt.xticks(rotation = rot)
            plt.show()

    def grafico_distribuicao(self, alpha = 0.5, bins = None, rot = None, figsize = [6, 4]):
        paleta_cores = sns.color_palette("colorblind")
        with sns.axes_style("whitegrid"):
            fig, ax = plt.subplots(1, 1, figsize = figsize)
            valores, frac, largura = self.curva_distribuicao(bins = bins)
            ax.bar(valores, frac, color = paleta_cores[0], alpha = alpha, width = largura, linewidth = 2, edgecolor = paleta_cores[0])
            ax.set_xlabel('Valores')
            ax.set_ylabel('Fração')
            ax.set_ylim(bottom = 0.0)
            if(rot is not None):
                plt.xticks(rotation = rot)
            plt.show()
            
    def grafico_mse_normalizado(self, alpha = 0.5, rot = None, figsize = [6, 4]):
        paleta_cores = sns.color_palette("colorblind")
        with sns.axes_style("whitegrid"):
            fig, ax = plt.subplots(1, 1, figsize = figsize)
            valores, mse_norm = self.curva_mse_normalizado()
            ax.fill_between(valores, mse_norm, color = paleta_cores[0], alpha = alpha)
            ax.plot(valores, mse_norm, color = paleta_cores[0])
            ax.set_xlabel('Valores')
            ax.set_ylabel('MSE Normalizado')
            ax.set_ylim(bottom = 0.0)
            #ax.set_ylim(top = 1.01)
            if(rot is not None):
                plt.xticks(rotation = rot)
            plt.show()
            
    def r2(self):
        mse = self.__media_alvo2 - np.sum(np.power(self.__qtds_alvo_corte, 2)/self.__qtds_corte)/np.sum(self.__qtds_corte)
        mse_ini = self.__media_alvo2 - (np.sum(self.__qtds_alvo_corte)/np.sum(self.__qtds_corte))**2
        return 1 - mse/mse_ini
        
    def r2_balanceado(self):
        media_geral = np.sum(self.__qtds_alvo_corte)/np.sum(self.__qtds_corte)
        #mse_ini_agroup = self.__media_alvo2_agrup + media_geral*(media_geral - 2*self.__qtds_alvo_corte/self.__qtds_corte)
        mse_agrup = self.__media_alvo2_agrup - np.power(self.__qtds_alvo_corte/self.__qtds_corte, 2)
        #mse_ini = np.mean(mse_ini_agroup)
        mse_ini = self.__media_alvo2 - (np.sum(self.__qtds_alvo_corte)/np.sum(self.__qtds_corte))**2
        mse = np.mean(mse_agrup)
        return 1 - mse/mse_ini
        
    def calcula_r2(self, vetor, alvo):
        media_alvo2 = calcula_mse(alvo) #np.mean(np.power(alvo, 2))
        inds_ordenado, primeira_ocorrencia, qtds_corte, _ = indices_qtds(self.aplica_discretizacao(vetor))
        alvo_agrup = np.split(alvo[inds_ordenado], primeira_ocorrencia[1:])
        qtds_alvo_corte = np.array([soma_vetor(v) for v in alvo_agrup])
        mse = media_alvo2 - np.sum(np.power(qtds_alvo_corte, 2)/qtds_corte)/np.sum(qtds_corte)
        mse_ini = media_alvo2 - (np.sum(qtds_alvo_corte)/np.sum(qtds_corte))**2
        return 1 - mse/mse_ini
        
    def calcula_r2_bal(self, vetor, alvo):
        media_alvo2 = calcula_mse(alvo) #np.mean(np.power(alvo, 2))
        inds_ordenado, primeira_ocorrencia, qtds_corte, _ = indices_qtds(self.aplica_discretizacao(vetor))
        alvo_agrup = np.split(alvo[inds_ordenado], primeira_ocorrencia[1:])
        qtds_alvo_corte = np.array([soma_vetor(v) for v in alvo_agrup])
        media_alvo2_agrup = np.array([calcula_mse(v) for v in alvo_agrup]) #np.mean(np.power(v, 2))
        media_geral = np.sum(qtds_alvo_corte)/np.sum(qtds_corte)
        #mse_ini_agroup = media_alvo2_agrup + media_geral*(media_geral - 2*qtds_alvo_corte/qtds_corte)
        mse_agrup = media_alvo2_agrup - np.power(qtds_alvo_corte/qtds_corte, 2)
        #mse_ini = np.mean(mse_ini_agroup)
        mse_ini = media_alvo2 - (np.sum(qtds_alvo_corte)/np.sum(qtds_corte))**2
        mse = np.mean(mse_agrup)
        return 1 - mse/mse_ini
            
    def aplica_discretizacao(self, vetor, usar_ponto_medio = False):
        if(self.__eh_temporal):
            flag_na = np.isnan(vetor)
            disc = discretiza_vetor(vetor.view('i8'), self.__valores_digitize)
            if(usar_ponto_medio):
                disc = disc.astype(int)
                disc = np.array([self.__valores_medios[disc[i]] if ~flag_na[i] else np.datetime64("NaT") for i in range(disc.size)])
                return disc
            else:
                disc[flag_na] = np.nan
                return disc
        else:
            if(usar_ponto_medio):
                return discretiza_vetor_media(vetor, self.__valores_digitize, self.__valores_medios)
            else:
                return discretiza_vetor(vetor, self.__valores_digitize)

########################
 
########################
 
class FiltraCategoriasRelevantes:
 
    def __init__(self, vetor, num_cat):
        self.__cats, self.__qtds = np.unique(vetor, return_counts = True)
        
        qtd_unicos = self.__qtds.size
        if(num_cat is not None and num_cat < qtd_unicos):
            self.__i_resto = qtd_unicos - num_cat
        else:
            self.__i_resto = 0
        
        self.__inds_sorted, self.__qtds_sorted = indices_e_ordenacao(self.__qtds)
        self.__cats_sorted = self.__cats[self.__inds_sorted]
        
        self.__dict_cats_apl = {self.__cats_sorted[i]:0 for i in range(self.__i_resto)}
        shift = 1 - self.__i_resto
        self.__dict_cats_apl.update({self.__cats_sorted[i]:i + shift for i in range(self.__i_resto, qtd_unicos)})
    
    def __pega_valor_dicionario(self, v):
        try:
            return self.__dict_cats_apl[v]
        except:
            return np.nan
    
    def __pega_valor(self, v, considera_resto):
        try:
            if(self.__dict_cats_apl[v] > 0):
                return v
            else:
                if(considera_resto):
                    return 'resto'
                else:
                    return np.nan
        except:
            return np.nan
    
    def strings_categorias(self):
        return np.array(list(self.__dict_cats_apl.keys()))
        
    def info_categorias(self):
        codigo = np.unique(list(self.__dict_cats_apl.values()))
        if(self.__i_resto > 0):
            qtds = np.append(np.sum(self.__qtds_sorted[:self.__i_resto]), self.__qtds_sorted[self.__i_resto:])
            categorias = np.append(', '.join((self.__cats_sorted[:self.__i_resto])), self.__cats_sorted[self.__i_resto:])
        else:
            qtds = self.__qtds_sorted
            categorias = self.__cats_sorted
        df = pd.DataFrame(zip(qtds, codigo, categorias), columns = ['QTD', 'Código', 'Categoria'])
        return df
        
    def curva_distribuicao(self, explicita_resto = False):
        if(self.__i_resto > 0):
            qtds = np.append(np.sum(self.__qtds_sorted[:self.__i_resto]), self.__qtds_sorted[self.__i_resto:])
            if(self.__i_resto > 1):
                if(explicita_resto):
                    categorias = np.append(', '.join((self.__cats_sorted[:self.__i_resto])), self.__cats_sorted[self.__i_resto:])
                else:
                    categorias = np.append('resto', self.__cats_sorted[self.__i_resto:])
            else:
                categorias = self.__cats_sorted
        else:
            qtds = self.__qtds_sorted
            categorias = self.__cats_sorted
        frac = qtds/np.sum(qtds)
        return categorias, frac
    
    def grafico_distribuicao(self, alpha = 0.5, explicita_resto = False, rot = None, ticks_chars = None, figsize = [6, 4]):
        paleta_cores = sns.color_palette("colorblind")
        with sns.axes_style("whitegrid"):
            fig, ax = plt.subplots(1, 1, figsize = figsize)
            valores, frac = self.curva_distribuicao(explicita_resto = explicita_resto)
            if(ticks_chars is not None):
                valores = [re.sub("(.{" + str(ticks_chars) + "})", "\\1\n", v, 0, re.DOTALL) for v in valores]
            ax.bar(valores, frac, color = paleta_cores[0], alpha = alpha, width = 1, linewidth = 2, edgecolor = paleta_cores[0])
            ax.set_xlabel('Valores')
            ax.set_ylabel('Fração')
            ax.set_ylim(bottom = 0.0)
            if(rot is not None):
                plt.xticks(rotation = rot)
            plt.show()
    
    def aplica_filtro_categorias(self, vetor, considera_resto = True, usar_str = False):
        flag_na = pd.isna(vetor)
        qtd_na = soma_vetor(flag_na)
        if(qtd_na > 0):
            cats_apl, i_rev = np.unique(vetor[~flag_na], return_inverse = True)
        else:
            cats_apl, i_rev = np.unique(vetor, return_inverse = True)
            
        if(usar_str):
            cats_fil = np.array([self.__pega_valor(c, considera_resto) for c in cats_apl], dtype = object)
        else:
            cats_fil = np.array([self.__pega_valor_dicionario(c) for c in cats_apl], dtype = float)
            if(considera_resto == False):
                cats_fil[cats_fil == 0] = np.nan
                
        if(qtd_na > 0):
            pos_na = flag_na.nonzero()[0]
            return np.insert(cats_fil[i_rev], pos_na - np.arange(qtd_na), np.nan)
        else:
            return cats_fil[i_rev]
  
########################
 
########################

class TrataDataset:
 
    def __init__(self, df, num_div = 20, num_cat = 5, unit = None, features_numericas = None, features_categoricas = None, features_temporais = None, autorun = True):
        
        self.__num_div = num_div
        self.__num_cat = num_cat
        self.__unit = unit
        
        self.__features_numericas = features_numericas
        self.__features_categoricas = features_categoricas
        self.__features_temporais = features_temporais
        
        self.__features_numericas_tratadas = np.array([])
        self.__features_categoricas_tratadas = np.array([])
        self.__features_temporais_tratadas = np.array([])
        
        self.__dict_intervs = {}
        self.__dict_filtroscat = {}
        
        if(autorun):
            if(isinstance(self.__features_numericas, list) and num_div is not None):
                for feature in self.__features_numericas:
                    if(conta_qtd_unicos(df[feature].dropna().values) > num_div):
                        try:
                            self.__dict_intervs[feature] = CortaIntervalosQuasiUniforme(df[feature].dropna().values, num_div = num_div)
                            self.__features_numericas_tratadas = np.append(self.__features_numericas_tratadas, feature)
                        except:
                            print('Erro no tratamento da coluna: ' + feature)
            
            if(isinstance(self.__features_categoricas, list)):
                for feature in self.__features_categoricas:
                    try:
                        self.__dict_filtroscat[feature] = FiltraCategoriasRelevantes(df[feature].dropna().values, num_cat = num_cat)
                        self.__features_categoricas_tratadas = np.append(self.__features_categoricas_tratadas, feature)
                    except:
                        print('Erro no tratamento da coluna: ' + feature)
            
            if(isinstance(self.__features_temporais, list) and num_div is not None):
                for feature in self.__features_temporais:
                    if(conta_qtd_unicos(df[feature].dropna().values) > num_div):
                        try:
                            self.__dict_intervs[feature] = CortaIntervalosQuasiUniforme(df[feature].dropna().values, num_div = num_div, eh_temporal = True, unit = unit)
                            self.__features_temporais_tratadas = np.append(self.__features_temporais_tratadas, feature)
                        except:
                            print('Erro no tratamento da coluna: ' + feature)
        
        self.__encoder = None
        self.__considera_resto_ohe = None
        self.__usar_str_ohe = None
    
    def trata_coluna(self, df, feature, parametros_padrao = True, num_div = 20, num_cat = 5, unit = None):
        if(parametros_padrao):
            num_div = self.__num_div
            num_cat = self.__num_cat
            unit = self.__unit
    
        if(isinstance(self.__features_numericas, list) and feature in self.__features_numericas):
            if(num_div is None):
                try:
                    del self.__dict_intervs[feature]
                    self.__features_numericas_tratadas = np.array([v for v in self.__features_numericas_tratadas if v != feature])
                except:
                    pass
            else:
                if(conta_qtd_unicos(df[feature].dropna().values) > num_div):
                    self.__dict_intervs[feature] = CortaIntervalosQuasiUniforme(df[feature].dropna().values, num_div = num_div)
                    if(self.__features_numericas_tratadas.size == 0 or np.sum(np.where(self.__features_numericas_tratadas == feature)) == 0):
                        self.__features_numericas_tratadas = np.append(self.__features_numericas_tratadas, feature)
                else:
                    try:
                        del self.__dict_intervs[feature]
                        self.__features_numericas_tratadas = np.array([v for v in self.__features_numericas_tratadas if v != feature])
                    except:
                        pass
             
        if(isinstance(self.__features_categoricas, list) and feature in self.__features_categoricas):
            self.__dict_filtroscat[feature] = FiltraCategoriasRelevantes(df[feature].dropna().values, num_cat = num_cat)
            if(self.__features_categoricas_tratadas.size == 0 or np.sum(np.where(self.__features_categoricas_tratadas == feature)) == 0):
                self.__features_categoricas_tratadas = np.append(self.__features_categoricas_tratadas, feature)
             
        if(isinstance(self.__features_temporais, list) and feature in self.__features_temporais):
            if(num_div is None):
                try:
                    del self.__dict_intervs[feature]
                    self.__features_temporais_tratadas = np.array([v for v in self.__features_temporais_tratadas if v != feature])
                except:
                    pass
            else:
                if(conta_qtd_unicos(df[feature].dropna().values) > num_div):
                    self.__dict_intervs[feature] = CortaIntervalosQuasiUniforme(df[feature].dropna().values, num_div = num_div, eh_temporal = True, unit = unit)
                    if(self.__features_temporais_tratadas.size == 0 or np.sum(np.where(self.__features_temporais_tratadas == feature)) == 0):
                        self.__features_temporais_tratadas = np.append(self.__features_temporais_tratadas, feature)
                else:
                    try:
                        del self.__dict_intervs[feature]
                        self.__features_temporais_tratadas = np.array([v for v in self.__features_temporais_tratadas if v != feature])
                    except:
                        pass
    
    def retorna_instancias_tratamento(self):
        return self.__dict_intervs, self.__dict_filtroscat
        
    def cria_one_hot_encoder(self, df, considera_resto = True, usar_str = False):
        if(isinstance(self.__features_categoricas, list)):
            lista_cats = []
            for feature in self.__features_categoricas:
                try:
                    transf = self.__dict_filtroscat[feature].aplica_filtro_categorias(df[feature].values, considera_resto = considera_resto, usar_str = usar_str)
                    lista_cats.append(transf)
                except:
                    lista_cats.append(df[feature].values)
            df_cat = pd.DataFrame(zip(*lista_cats), columns = self.__features_categoricas, index = df.index)
            valores = df_cat.values
            if(usar_str):
                flag_na = pd.isna(df_cat).values
                valores[flag_na] = '-1'
            else:
                flag_na = np.isnan(valores, where = True)
                valores[flag_na] = -1
                valores = valores.astype(int)
            self.__encoder = OneHotEncoder(handle_unknown = 'ignore', sparse = False).fit(valores)
            self.__considera_resto_ohe = considera_resto
            self.__usar_str_ohe = usar_str
    
    def aplica_transformacao(self, df_inp, usar_ponto_medio = False, considera_resto = True, usar_str = False):
        df_aplic = df_inp.copy()
        if(self.__features_numericas_tratadas.size > 0):
            for feature in self.__features_numericas_tratadas:
                df_aplic[feature] = self.__dict_intervs[feature].aplica_discretizacao(df_aplic[feature].values, usar_ponto_medio = usar_ponto_medio)
        if(self.__features_categoricas_tratadas.size > 0):
            for feature in self.__features_categoricas_tratadas:
                df_aplic[feature] = self.__dict_filtroscat[feature].aplica_filtro_categorias(df_aplic[feature].values, considera_resto = considera_resto, usar_str = usar_str)
        if(self.__features_temporais_tratadas.size > 0):
            for feature in self.__features_temporais_tratadas:
                df_aplic[feature] = self.__dict_intervs[feature].aplica_discretizacao(df_aplic[feature].values, usar_ponto_medio = usar_ponto_medio)  
        return df_aplic
        
    def aplica_transformacao_ohe(self, df_inp, usar_ponto_medio = False):
        df_aplic = df_inp.copy()
        df_aplic = self.aplica_transformacao(df_inp, usar_ponto_medio = usar_ponto_medio, considera_resto = self.__considera_resto_ohe, usar_str = self.__usar_str_ohe)                                
        if(isinstance(self.__features_categoricas, list)):
            valores = df_aplic[self.__features_categoricas].values
            if(self.__usar_str_ohe):
                flag_na = pd.isna(df_aplic[self.__features_categoricas]).values
                valores[flag_na] = '-1'
            else:
                flag_na = np.isnan(valores, where = True)
                valores[flag_na] = -1
                valores = valores.astype(int)
            matriz_cat = self.__encoder.transform(valores)
            nome_colunas = self.__encoder.get_feature_names(self.__features_categoricas)
            df_cat = pd.DataFrame(matriz_cat, columns = nome_colunas, index = df_aplic.index)
            df_cat = df_cat.drop([col for col in nome_colunas if col[-3:] == '_-1'], axis = 1).astype(int)
            df_aplic = df_aplic.drop(self.__features_categoricas, axis = 1)
            df_aplic = pd.concat([df_aplic, df_cat], axis = 1)
        return df_aplic
        
 ########################
 
 ########################

#Faz as contas com o Dataset em Pandas e retorna o Dataset em Pandas com tudo calculado
#Ou seja, ocupa memória
class TaylorLaurentSeries:

    def __init__(self, laurent = False, ordem = 2, apenas_interacoes = False, features_numericas = None):
        self.__laurent = laurent
        self.__apenas_interacoes = apenas_interacoes
        self.__features_numericas = features_numericas
        if(self.__apenas_interacoes):
            self.__ordem = min(ordem, len(self.__features_numericas))
        else:
            self.__ordem = ordem
        
        if(laurent == False):
            self.__lista_features = [self.__features_numericas]
        else:
            #Para evitar de fazer a conta (1/x)*x = 1 ou (1/x)(1/x)*x = x
            combs = [[]]
            for feature in self.__features_numericas:
                combs_temp = combs.copy()
                for var in combs_temp:
                    v_new1 = var.copy()
                    v_new2 = var.copy()
                    v_new1.append(feature)
                    v_new2.append('1/' + feature)
                    combs.pop(0)
                    combs.append(v_new1)
                    combs.append(v_new2)
            self.__lista_features = combs
        
        #Pega todas as colunas que vamos ter que calcular, removendo as repetições
        self.__lista_combs = []
        for i in range(2, self.__ordem + 1):
            conjunto_combs = set() #Remove as combinações repetidas no update por sem um conjunto
            for features in self.__lista_features:
                if(self.__apenas_interacoes):
                    #Não precisa de potencias das features
                    comb = set(combinations(features, r = i))
                else:
                    comb = set(combinations_with_replacement(features, r = i))
                conjunto_combs.update(comb)
            conjunto_combs = list(conjunto_combs)
            if(self.__apenas_interacoes):
                #remove as combinações que são inversos multiplicativos
                def inverso_multiplicativo(tupla):
                    return tuple(v[2:] if v[:2] == '1/' else '1/' + v for v in tupla)
                conjunto_filtrado = []
                while(len(conjunto_combs) > 1):
                    tupla = conjunto_combs[0]
                    conjunto_filtrado.append(tupla)
                    conjunto_combs.remove(tupla)
                    try:
                        conjunto_combs.remove(inverso_multiplicativo(tupla))
                    except:
                        pass
                if(len(conjunto_combs) == 1):
                    conjunto_filtrado.append(conjunto_combs[0])
                conjunto_combs = conjunto_filtrado
                #Inverte quando todos as features da tupla estão invertidas
                def eh_tudo_inverso(tupla):
                    return np.sum(np.array([False if v[:2] == '1/' else True for v in tupla])) == 0
                conjunto_combs = [inverso_multiplicativo(v) if eh_tudo_inverso(v) else v for v in conjunto_combs]
            self.__lista_combs.append(conjunto_combs) #Popula a lista de combinações por ordem de número de features (posição 0 => pares 2 a 2)
            
        self.__variaveis_criadas = []
        if(self.__laurent and self.__apenas_interacoes == False):
            self.__variaveis_criadas.extend(['1/' + v for v in self.__features_numericas])
        for combs in self.__lista_combs:
            self.__variaveis_criadas.extend([str(col).replace("'","") for col in combs])
            
    def variaveis_criadas(self):
        return self.__variaveis_criadas
                
    def aplica_transformacao(self, df):
        def inverso_multiplicativo(tupla):
            tupla_inverso = []
            for v in tupla:
                if(v[:2] == '1/'):
                    tupla_inverso.append(v[2:])
                else:
                    tupla_inverso.append('1/' + v)
            return tuple(tupla_inverso)
        
        X = df[self.__features_numericas].copy()
        if(self.__laurent):
            X_inv = (1/X).add_prefix('1/')
            
        if(self.__apenas_interacoes or self.__laurent == False):
            X_res = pd.DataFrame(index = df.index)
        else:
            X_res = X_inv.copy()
        
        if(self.__laurent):
            for i in range(0, len(self.__lista_combs)):
                X_temp = pd.DataFrame(index = df.index)
                if(i == 0):
                    for comb in self.__lista_combs[i]:
                        if(comb[0][:2] == '1/' and comb[1][:2] == '1/'):
                            X_temp[comb] =  multiplica_vetores(X_inv[comb[0]].values, X_inv[comb[1]].values)
                        elif(comb[0][:2] == '1/'):
                            X_temp[comb] =  multiplica_vetores(X_inv[comb[0]].values, X[comb[1]].values)
                        elif(comb[1][:2] == '1/'):
                            X_temp[comb] =  multiplica_vetores(X[comb[0]].values, X_inv[comb[1]].values)
                        else:
                            X_temp[comb] =  multiplica_vetores(X[comb[0]].values, X[comb[1]].values)
                else:
                    for comb in self.__lista_combs[i]:
                        if(comb[0][:2] == '1/'):
                            try:
                                X_temp[comb] = multiplica_vetores(X_inv[comb[0]].values, X_ant[comb[1:]].values)
                            except:
                                X_temp[comb] = multiplica_vetores(X_inv[comb[0]].values, X_ant[inverso_multiplicativo(comb[1:])].values)
                        else:
                            try:
                                X_temp[comb] = multiplica_vetores(X[comb[0]].values, X_ant[comb[1:]].values)
                            except:
                                X_temp[comb] = multiplica_vetores(X[comb[0]].values, X_ant[inverso_multiplicativo(comb[1:])].values)
                X_res = pd.concat([X_res, X_temp], axis = 1, sort = False)
                X_ant = X_temp
        else:
            for i in range(0, len(self.__lista_combs)):
                X_temp = pd.DataFrame(index = df.index)
                if(i == 0):
                    for comb in self.__lista_combs[i]:
                        X_temp[comb] = multiplica_vetores(X[comb[0]].values, X[comb[1]].values)
                else:
                    for comb in self.__lista_combs[i]:
                        X_temp[comb] = multiplica_vetores(X[comb[0]].values, X_ant[comb[1:]].values)
                X_res = pd.concat([X_res, X_temp], axis = 1, sort = False)
                X_ant = X_temp
        
        X_res.columns = self.__variaveis_criadas
        X_res = pd.concat([df, X_res], axis = 1, sort = False)
        return X_res