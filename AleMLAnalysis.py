import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import re

from AleTransforms import *
from AleDatasetAnalysis import *
from AleMetrics import *

class AvaliaClassificacao:

    def __init__(self, df, col_alvo, col_prob = None, num_div_prob = None, p_corte = None, p01_corte = [0, 0], p_ref = None, num_div = None, num_cat = None, unit = None):
        self.distribuicoes = DistribuicoesDataset(df, num_div = num_div, num_cat = num_cat, unit = unit, autorun = False)
        
        self.__col_alvo = col_alvo
        self.__col_prob = col_prob
        
        if(col_prob is not None):
            self.__num_div_prob = num_div_prob
            
            #Calculas as métricas gerais do dataset
            self.metricas_gerais = AletricasClassificacao(df[col_alvo].values, df[col_prob].values, num_div = num_div_prob, p_corte = p_corte, p01_corte = p01_corte, 
                                                          p_ref = p_ref)
            
            #Probabilidades de Corte para Avaliação de Tomada de Decisão (Usa do Ganho de Informação se não for passado)
            probs_ig = self.metricas_gerais.valor_prob_ig()
            if(p_corte is None):
                self.__p_corte = probs_ig['Prob_Corte']
            else:
                self.__p_corte = p_corte
            
            if(np.sum(np.array(p01_corte)) == 0):
                self.__p01_corte = [probs_ig['Prob0_Corte'], probs_ig['Prob1_Corte']]
            else:
                self.__p01_corte = p01_corte
                
            if(p_ref is None):
                self.__p_ref = self.metricas_gerais.retorna_p_ref()
            else:
                self.__p_ref = p_ref
            
            #OBS: Note que, dessa forma, se for o dataset de treino, podemos não passar as probs e usar como corte o decidido pelo IG no próprio dataset
            #Porém, se for um dataset de Validação ou Teste, podemos passar as probs de corte que foram obtidas na avaliação do dataset de Treino
            
        else:
            self.__y_prob = None
            self.__num_div_prob = None
            self.__p_corte = None
            self.__p01_corte = [0, 0]
        
        self.__dict_qtds1 = {}
        self.__dict_prob1 = {}
        self.__dict_ig = {} #Ganho de Informação
        self.__dict_rg = {} #Razão de Ganho
        
        self.__dict_soma_prob = {}
        self.__dict_media_prob = {}
        self.__dict_metricas = {}
    
    def colunas_metricas_condicionais_calculadas(self):
        return self.__dict_qtds1.keys()
    
    def calcula_metricas_condicionais(self, df, col_ref = [], parametros_padrao = True, num_div_prob = None, num_div = 20, num_cat = 5, unit = None):
        #Transforma uma string única em uma lista
        colunas = self.distribuicoes.retorna_colunas_dataset()
        if(isinstance(col_ref, str)):
            col_ref = [col_ref]
        if(len(col_ref) != 0):
            colunas = col_ref
          
        num_linhas = self.distribuicoes.retorna_shape_dataset()[0]
        
        if(parametros_padrao):
            num_div_prob = self.__num_div_prob
          
        for col_ref in colunas:
            self.distribuicoes.trata_coluna(df, col_ref, parametros_padrao, num_div, num_cat, unit)
            
            valores = self.distribuicoes.retorna_valores(df, col_ref)
            flag_na = np.array(self.distribuicoes.retorna_flag_na(col_ref))
            qtd_nao_nulo = num_linhas - self.distribuicoes.retorna_qtds_na(col_ref)
            
            valores = valores[~flag_na]
            y = df.loc[~flag_na, self.__col_alvo].values
            
            inds_ordenado, primeira_ocorrencia, qtds, qtd_unicos = indices_qtds(valores)
            y_agrup = np.split(y[inds_ordenado], primeira_ocorrencia[1:])
            
            qtds1 = np.array([soma_vetor(v) for v in y_agrup])
            probs1 = qtds1/qtds
            
            self.__dict_qtds1[col_ref] = qtds1
            self.__dict_prob1[col_ref] = probs1
            self.__dict_ig[col_ref], self.__dict_rg[col_ref] = calcula_ig_rg_condicional(qtds1, qtds, probs1, qtd_nao_nulo)
            
            if(self.__col_prob is not None):
                y_prob = df.loc[~flag_na, self.__col_prob].values
                y_prob_agrup = np.split(y_prob[inds_ordenado], primeira_ocorrencia[1:])
                soma_prob = np.array([soma_vetor(v) for v in y_prob_agrup])
                self.__dict_soma_prob[col_ref] = soma_prob
                self.__dict_media_prob[col_ref] = soma_prob/qtds
                self.__dict_metricas[col_ref] = np.array([AletricasClassificacao(y_agrup[i], y_prob_agrup[i], num_div = num_div_prob,
                                                          p_corte = self.__p_corte, p01_corte = self.__p01_corte, p_ref = self.__p_ref) for i in range(qtd_unicos)])

        #Ordena os Ganhos de Informação
        self.__dict_ig = dict(reversed(sorted(self.__dict_ig.items(), key = lambda x: x[1])))
        self.__dict_rg = dict(reversed(sorted(self.__dict_rg.items(), key = lambda x: x[1])))
    
    def ganho_info(self):
        return pd.Series(self.__dict_ig)

    def razao_ganho_info(self):
        return pd.Series(self.__dict_rg)
    
    def valor_metricas_condicionais(self, col_ref, retorna_valores = False, explicita_resto = False):
        df = pd.DataFrame()
        if(col_ref in self.__dict_qtds1.keys()):
            
            df = self.distribuicoes.info_distribuicao(col_ref)
            
            df['QTD_0'] = df['QTD'] - self.__dict_qtds1[col_ref]
            df['QTD_1'] = self.__dict_qtds1[col_ref]
            df['Frac_0'] = 1 - self.__dict_prob1[col_ref]
            df['Frac_1'] = self.__dict_prob1[col_ref]
            
            if(self.__col_prob is not None):
                df['Soma_Prob'] = self.__dict_soma_prob[col_ref]
                df['Media_Prob'] = self.__dict_media_prob[col_ref]
                df['Metricas'] = self.__dict_metricas[col_ref]
                df = pd.concat([df, df['Metricas'].apply(lambda x: x.valor_metricas(estatisticas_globais = False))], axis = 1)
                df = df.drop('Metricas', axis = 1)
                
        if(retorna_valores):
            df_colunas = list(df.columns)
            if('Str' in df_colunas):
                valores = df['Str'].values
            elif('Categoria' in df_colunas):
                df = df.sort_values('Frac_1')
                if(explicita_resto):
                    valores = df['Categoria'].values
                else:
                    valores = np.where(df['Código'].values == 0, 'resto', df['Categoria'].values)
            elif('Valor' in df_colunas):
                valores = df['Valor'].values
            return df, valores
        else:
            return df
    
    def grafico_probabilidade_condicional(self, col_ref, ymax = 0, explicita_resto = False, rot = None, 
                                          alga_signif = None, unit = None, conv_str = True, ticks_chars = None, figsize = [6, 4]):
        if(col_ref in self.__dict_qtds1.keys()):
            paleta_cores = sns.color_palette("colorblind") #Paleta de cores para daltonico
            #Plot a curva de probabilidade dada pelo Alvo e pela Prob do Classificador
            with sns.axes_style("whitegrid"):
                fig, axs = plt.subplots(1, 1, figsize = figsize)
                
                df, valores = self.valor_metricas_condicionais(col_ref, retorna_valores = True, explicita_resto = explicita_resto)
                    
                prob1 = df['Frac_1'].values
                if(self.__col_prob is not None):
                    media_prob = df['Media_Prob'].values
                else:
                    media_prob = None
                ig = self.__dict_ig[col_ref]
                rg = self.__dict_rg[col_ref]
                
                if(alga_signif is not None):
                    str_conv = ''.join(['%.', np.str(alga_signif), 'g'])
                    valores = np.array([float(str_conv % v) for v in valores])
                if(unit is not None):
                    valores = np.array([np.datetime_as_string(v, unit = unit) for v in valores])
                if(conv_str):
                    valores = valores.astype(str)
                if(ticks_chars is not None):
                    valores = [re.sub("(.{" + str(ticks_chars) + "})", "\\1\n", v, 0, re.DOTALL) for v in valores]
                
                axs.bar(valores, prob1, color = paleta_cores[0], label = 'Real')
                if(media_prob is not None):
                    axs.plot(valores, media_prob, '-o', color = 'black', linewidth = 2, label = 'Classificador')
                
                plt.gcf().text(1, 0.5, 'IG = ' + '%.2g' % ig + '\n' + 'RG = ' + '%.2g' % rg, bbox = dict(facecolor = 'white', edgecolor = 'k', boxstyle = 'round'))
                
                if(ymax > 0 and ymax <= 1):
                    axs.set_ylim([0, ymax])
                axs.set_xticks(valores)
                axs.set_xlabel(col_ref)
                axs.set_ylabel('Probabilidade de 1')
                axs.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                if(rot is not None):
                    plt.xticks(rotation = rot)
                plt.show()

    def grafico_metricas_condicionais(self, col_ref, metricas, ylim = [0, 0], explicita_resto = False, rot = None, 
                                      alga_signif = None, unit = None, conv_str = True, ticks_chars = None, figsize = [6, 4]):
        if(col_ref in self.__dict_qtds1.keys()):
            paleta_cores = sns.color_palette("colorblind") #Paleta de cores para daltonico 
            #Plot a curva de métrica em função da coluna de referência
            with sns.axes_style("whitegrid"):
                fig, axs = plt.subplots(1, 1, figsize = figsize)
                
                df, valores = self.valor_metricas_condicionais(col_ref, retorna_valores = True, explicita_resto = explicita_resto)
                
                if(alga_signif is not None):
                    str_conv = ''.join(['%.', np.str(alga_signif), 'g'])
                    valores = np.array([float(str_conv % v) for v in valores])
                if(unit is not None):
                    valores = np.array([np.datetime_as_string(v, unit = unit) for v in valores])                
                if(conv_str):
                    valores = valores.astype(str)
                if(ticks_chars is not None):
                    valores = [re.sub("(.{" + str(ticks_chars) + "})", "\\1\n", v, 0, re.DOTALL) for v in valores]
                
                for i in range(len(metricas)):
                    axs.plot(valores, df[metricas[i]].values, '-o', color = paleta_cores[i], label = metricas[i])

                if(ylim[1] > ylim[0]):
                    axs.set_ylim(ylim)
                axs.set_xticks(valores)
                axs.set_xlabel(col_ref)
                axs.set_ylabel('Metricas') 
                axs.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                if(rot is not None):
                    plt.xticks(rotation = rot)
                plt.show()
                
    def barra_calor(self, col_ref, alpha_max = 1, sinal_cor = False, explicita_resto = False, rot = None, 
                    alga_signif = None, unit = None, conv_str = True, ticks_chars = None, figsize = [6, 4]):
        #Normaliza os valores para plotar em cor
        def normaliza_media(sinal_cor, media):
            minimo = np.min(media)
            maximo = np.max(media)
            if(sinal_cor):
                if(maximo > 0 and minimo < 0):
                    norm = np.array([v/maximo if v > 0 else v/np.abs(minimo) for v in media])
                elif(maximo > 0 and minimo >= 0):
                    norm = media/maximo
                elif(maximo <= 0 and minimo < 0):
                    norm = media/np.abs(minimo)
            else:
                if(maximo == minimo):
                    norm = np.array([0 for v in media]).astype(float)
                else:
                    norm = (2*media - maximo - minimo)/(maximo - minimo)
            return norm
    
        if(col_ref in self.__dict_qtds1.keys()):
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
            
            with sns.axes_style("whitegrid"):
                fig, ax = plt.subplots(1, 1, figsize = figsize, constrained_layout = True)
            
                df, valores = self.valor_metricas_condicionais(col_ref, retorna_valores = True, explicita_resto = explicita_resto)
                                
                if(self.__col_prob is not None):
                    media_prob = df['Media_Prob'].values
                else:
                    media_prob = df['Frac_1'].values
                
                if(alga_signif is not None):
                    str_conv = ''.join(['%.', np.str(alga_signif), 'g'])
                    valores = np.array([float(str_conv % v) for v in valores])
                if(unit is not None):
                    valores = np.array([np.datetime_as_string(v, unit = unit) for v in valores])  
                if(conv_str):
                    valores = valores.astype(str)
                if(ticks_chars is not None):
                    valores = [re.sub("(.{" + str(ticks_chars) + "})", "\\1\n", v, 0, re.DOTALL) for v in valores]
                
                media_norm = alpha_max*normaliza_media(sinal_cor, media_prob)            
                cores_plot = cores[np.floor((media_norm + 1)*(N-1)/2).astype(int)]
                ax.imshow([cores_plot], aspect = 0.5*(valores.size/10), interpolation = 'spline16')
                ax.set_yticks([])
                ax.grid(False)
                
                ax.set_xticks(range(0, valores.size))
                ax.set_xticklabels(valores.astype(str))
                ax.set_title(col_ref + ':', loc = 'left')
                if(rot is not None):
                    plt.xticks(rotation = rot)              
                plt.show()

##############################

##############################

class AvaliaDatasetsClassificacao:

    def __init__(self, dict_dfs, col_alvo, col_prob = None, num_div_prob = None, num_div = None, num_cat = None, unit = None, chave_treino = 'Treino'):
        self.__chaves = dict_dfs.keys()
        self.__num_dfs = len(dict_dfs)
        self.__chave_treino = chave_treino
        self.__col_alvo = col_alvo
        self.__col_prob = col_prob
        
        if(col_prob is not None):
            self.__num_div_prob = num_div_prob
        
        self.__dict_avaliaclf = {}
        if(chave_treino in self.__chaves):
            avaliaclf_treino = AvaliaClassificacao(dict_dfs[chave_treino], col_alvo, col_prob, num_div_prob = num_div_prob, 
                                                   num_div = num_div, num_cat = num_cat, unit = unit)
            #Probabilidades de Corte para Avaliação de Tomada de Decisão
            probs_ig = avaliaclf_treino.metricas_gerais.valor_prob_ig()
            self.__p_corte = probs_ig['Prob_Corte']
            self.__p01_corte = [probs_ig['Prob0_Corte'], probs_ig['Prob1_Corte']]
            self.__p_ref = avaliaclf_treino.metricas_gerais.retorna_p_ref()
            self.__dict_avaliaclf[chave_treino] = avaliaclf_treino
        for chave in self.__chaves:
            if(chave != chave_treino):
                self.__dict_avaliaclf[chave] = AvaliaClassificacao(dict_dfs[chave], col_alvo, col_prob, num_div_prob, self.__p_corte, self.__p01_corte, self.__p_ref, 
                                                                   num_div, num_cat, unit)
                                                                   
        self.__distribuicoes_geral = DistribuicoesDataset(pd.concat(list(dict_dfs.values())), num_div = num_div, num_cat = num_cat, unit = unit, autorun = False)
        self.__dict_dfs_metricas = {}
        
    def avaliadores_individuais(self):
        return self.__dict_avaliaclf
        
    def calcula_metricas_condicionais(self, dict_dfs, col_ref = [], parametros_padrao = True, num_div_prob = None, num_div = 20, num_cat = 5, unit = None):
        for chave in self.__chaves:
            self.__dict_avaliaclf[chave].calcula_metricas_condicionais(dict_dfs[chave], col_ref, parametros_padrao, num_div_prob, num_div, num_cat, unit)
            
        colunas = self.__dict_avaliaclf[self.__chave_treino].distribuicoes.retorna_colunas_dataset()
        #Transforma uma string única em uma lista
        if(isinstance(col_ref, str)):
            col_ref = [col_ref]
        if(len(col_ref) != 0):
            colunas = col_ref
            
        if(parametros_padrao):
            num_div_prob = self.__num_div_prob
            
        for col_ref in colunas:
            self.__distribuicoes_geral.trata_coluna(pd.concat(list(dict_dfs.values())), col_ref, parametros_padrao, num_div, num_cat, unit)
            
            dfs_metricas = {}
            for chave in self.__chaves:
                valores = self.__distribuicoes_geral.retorna_valores(dict_dfs[chave], col_ref)
                flag_na = np.array(self.__dict_avaliaclf[chave].distribuicoes.retorna_flag_na(col_ref))
                num_linhas = self.__dict_avaliaclf[chave].distribuicoes.retorna_shape_dataset()[0]
                qtd_nao_nulo = num_linhas - self.__dict_avaliaclf[chave].distribuicoes.retorna_qtds_na(col_ref)
                
                valores = valores[~flag_na]
                y = dict_dfs[chave].loc[~flag_na, self.__col_alvo].values
                
                inds_ordenado, primeira_ocorrencia, qtds, qtd_unicos = indices_qtds(valores)
                valores_unicos = valores[inds_ordenado][primeira_ocorrencia]
                y_agrup = np.split(y[inds_ordenado], primeira_ocorrencia[1:])
                
                qtds1 = np.array([soma_vetor(v) for v in y_agrup])
                probs1 = qtds1/qtds
                
                if(self.__col_prob is not None):
                    y_prob = dict_dfs[chave].loc[~flag_na, self.__col_prob].values
                    y_prob_agrup = np.split(y_prob[inds_ordenado], primeira_ocorrencia[1:])
                    soma_prob = np.array([soma_vetor(v) for v in y_prob_agrup])
                    aletricas_vetor = np.array([AletricasClassificacao(y_agrup[i], y_prob_agrup[i], num_div = num_div_prob,
                                                p_corte = self.__p_corte, p01_corte = self.__p01_corte, p_ref = self.__p_ref) for i in range(qtd_unicos)])
                                                              
                df = self.__distribuicoes_geral.info_distribuicao(col_ref)
                if('Valor' in list(df.columns)):
                    if(df['Valor'].dtype in ['<M8[ns]', 'datetime64[ns]']):
                        indices = df.index[df['Valor'].view('i8').isin(valores_unicos)]
                    else:
                        indices = df.index[df['Valor'].isin(valores_unicos)]
                else:
                    indices = valores_unicos
                df['QTD_0'] = pd.Series(qtds - qtds1, index = indices)
                df['QTD_1'] = pd.Series(qtds1, index = indices)
                df['Frac_0'] = pd.Series(1 - probs1, index = indices)
                df['Frac_1'] = pd.Series(probs1, index = indices)
                if(self.__col_prob is not None):
                    df['Soma_Prob'] = pd.Series(soma_prob, index = indices)
                    df['Media_Prob'] = pd.Series(soma_prob/qtds, index = indices)
                    df['Metricas'] = pd.Series(aletricas_vetor, index = indices)
                    df = pd.concat([df, df['Metricas'].dropna().apply(lambda x: x.valor_metricas(estatisticas_globais = False))], axis = 1)
                    df = df.drop('Metricas', axis = 1)
                
                dfs_metricas[chave] = df
            self.__dict_dfs_metricas[col_ref] = dfs_metricas
    
    def valor_metricas(self, estatisticas_globais = True, probs_corte = True, probs_condicionais = True, lifts = True):
        df = pd.DataFrame(self.__dict_avaliaclf[self.__chave_treino].metricas_gerais.valor_metricas(estatisticas_globais, probs_corte, probs_condicionais, lifts), columns = [self.__chave_treino])
        for chave in self.__chaves:
            if(chave != self.__chave_treino):
                df[chave] = self.__dict_avaliaclf[chave].metricas_gerais.valor_metricas(estatisticas_globais, probs_corte, probs_condicionais, lifts)
        return df
    
    def grafico_roc(self, roc_usual = True, figsize = [6, 4]):
        paleta_cores = sns.color_palette("colorblind")
        with sns.axes_style("whitegrid"):
            fig, axs = plt.subplots(1, 1, figsize = figsize)
            i = 0
            for chave in self.__chaves:
                curva_tnp, curva_tvp, auc = self.__dict_avaliaclf[chave].metricas_gerais.curva_roc()
                if(roc_usual):
                    axs.plot(1-curva_tnp, curva_tvp, color = paleta_cores[i], label = chave)
                else:
                    axs.plot(curva_tnp, curva_tvp, color = paleta_cores[i], label = chave)
                i = i + 1
            if(roc_usual):
                axs.plot([0, 1], [0, 1], color = 'k', linestyle = '--', label = 'Linha de Ref.')
                axs.set_xlabel('Taxa de Falso Positivo')
            else:
                axs.plot([0, 1], [1, 0], color = 'k', linestyle = '--', label = 'Linha de Ref.')
                axs.set_xlabel('Taxa de Verdadeiro Negativo')
            axs.set_ylabel('Taxa de Verdadeiro Positivo')
            axs.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.show()

    def grafico_revocacao(self, figsize = [6, 4]): 
        paleta_cores = sns.color_palette("colorblind")
        with sns.axes_style("whitegrid"):
            fig, axs = plt.subplots(1, 1, figsize = figsize)
            i = 0
            for chave in self.__chaves:
                y_prob_plot, curva_revoc0_plot, curva_revoc1_plot, pos_max, ks = self.__dict_avaliaclf[chave].metricas_gerais.curva_revocacao()
                axs.plot(y_prob_plot, curva_revoc0_plot, color = paleta_cores[i], alpha = 0.6)
                axs.plot(y_prob_plot, curva_revoc1_plot, color = paleta_cores[i], alpha = 0.4)
                axs.vlines(pos_max, 0, 1, color = paleta_cores[i], linestyle = '--', label = chave)
                i = i + 1
            axs.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            axs.set_xlabel('Probabilidade de Corte')
            axs.set_ylabel('Revocação')
            plt.show()
            
    def grafico_ks(self, figsize = [6, 4]): 
        paleta_cores = sns.color_palette("colorblind")
        with sns.axes_style("whitegrid"):
            fig, axs = plt.subplots(1, 1, figsize = figsize)
            i = 0
            for chave in self.__chaves:
                y_prob_plot, curva_revoc0_plot, curva_revoc1_plot, pos_max, ks = self.__dict_avaliaclf[chave].metricas_gerais.curva_revocacao()
                axs.plot(y_prob_plot, curva_revoc0_plot - curva_revoc1_plot, color = paleta_cores[i])
                axs.vlines(pos_max, 0, ks, color = paleta_cores[i], linestyle = '--', label = chave)
                i = i + 1
            axs.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            axs.set_xlabel('Probabilidade de Corte')
            axs.set_ylabel('Distância entre Revocações')
            plt.show()
            
    def grafico_informacao(self, mostrar_ig_2d = False, figsize = [6, 4]):
        paleta_cores = sns.color_palette("colorblind")
        with sns.axes_style("whitegrid"):
            fig, axs = plt.subplots(1, 1, figsize = figsize)
            i = 0
            for chave in self.__chaves:
                y_prob_plot, curva_ig_plot, pos_max, ig, p0_corte, p1_corte, ig_2d = self.__dict_avaliaclf[chave].metricas_gerais.curva_informacao()
                axs.plot(y_prob_plot, curva_ig_plot, color = paleta_cores[i], label = chave)
                axs.vlines(pos_max, 0, ig, color = paleta_cores[i], linestyle = '--')
                if(mostrar_ig_2d and ig_2d != np.nan):
                    axs.vlines(p0_corte, 0, ig_2d, color = paleta_cores[i], alpha = 0.5, linestyle = '--')
                    axs.vlines(p1_corte, 0, ig_2d, color = paleta_cores[i], alpha = 0.5, linestyle = '--')
                i = i + 1
            axs.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            axs.set_xlabel('Probabilidade de Corte')
            axs.set_ylabel('Ganho de Informação')
            plt.show()
            
    def grafico_informacao_2d(self, plot_3d = True, figsize = [7, 6]):
        paleta_cores = sns.color_palette("colorblind")
        if(plot_3d):
            with sns.axes_style("whitegrid"):
                fig = plt.figure(figsize = figsize)
                axs = fig.add_subplot(111, projection='3d')
                i = 0
                hlds = []
                for chave in self.__chaves:
                    x, y, z, p0_corte, p1_corte, ig_2d = self.__dict_avaliaclf[chave].metricas_gerais.curva_informacao_2d()
                    N = 256
                    vals = np.ones((N, 4)) #A última componente (quarta) é o alpha que é o índice de transparência
                    cor = paleta_cores[i]
                    #Define as Cores RGB pelas componentes (no caso é o azul -> 0,0,255)
                    vals[:, 0] = np.linspace(cor[0], 1, N)
                    vals[:, 1] = np.linspace(cor[1], 1, N)
                    vals[:, 2] = np.linspace(cor[2], 1, N)
                    cmap = mpl.colors.ListedColormap(vals[::-1])
                    axs.scatter(x, y, z, c = z, marker = 'o', cmap = cmap)
                    hlds.append(mpl.patches.Patch(color = cor, label = chave))
                    i = i + 1
                axs.set_xlabel('Probabilidade de Corte 0')
                axs.set_ylabel('Probabilidade de Corte 1')
                axs.set_zlabel('Ganho de Informação')
                axs.legend(handles = hlds, bbox_to_anchor = (1.05, 1), loc = 'upper left')
                plt.show()
        else:
            with sns.axes_style("whitegrid"):
                fig, axs = plt.subplots(1, 1, figsize = figsize)
                i = 0
                hlds = []
                mini = 1
                maxi = 0
                for chave in self.__chaves:
                    x, y, z, p0_corte, p1_corte, ig_2d = self.__dict_avaliaclf[chave].metricas_gerais.curva_informacao_2d()
                    mini = min(mini, min(x[0], y[0]))
                    maxi = max(maxi, max(x[-1], y[-1]))
                    N = 256
                    cor = paleta_cores[i]
                    vals = np.ones((N, 4))
                    vals[:, 0] = cor[0]
                    vals[:, 1] = cor[1]
                    vals[:, 2] = cor[2]
                    cmap_linhas = mpl.colors.ListedColormap(vals[::-1])
                    vals[:, 3] = np.linspace(0, 1, N)[::-1]
                    cmap = mpl.colors.ListedColormap(vals[::-1])
                    axs.tricontour(x, y, z, levels = 14, linewidths = 1.0, cmap = cmap_linhas)
                    #cntr = axs.tricontourf(x, y, z, levels = 14, cmap = cmap)
                    axs.scatter(p0_corte, p1_corte, color = cor)
                    axs.vlines(p0_corte, 0, p1_corte, color = cor, alpha = 0.5, linestyle = '--')
                    axs.hlines(p1_corte, 0, p0_corte, color = cor, alpha = 0.5, linestyle = '--')
                    hlds.append(mpl.patches.Patch(color = cor, label = chave))
                    i = i + 1
                axs.set_xlabel('Probabilidade de Corte 0')
                axs.set_ylabel('Probabilidade de Corte 1')
                axs.set_xlim([mini, maxi])
                axs.set_ylim([mini, maxi])
                axs.legend(handles = hlds, bbox_to_anchor = (1.05, 1), loc = 'upper left')
                plt.show()
    
    def valor_metricas_condicionais(self, col_ref):
        if(col_ref in self.__dict_avaliaclf[self.__chave_treino].colunas_metricas_condicionais_calculadas()):
            d = {}
            for chave in self.__chaves:
                d[chave] = self.__dict_avaliaclf[chave].valor_metricas_condicionais(col_ref)
            return d
            
    def valor_metricas_condicionais_geral(self, col_ref):
        if(col_ref in self.__dict_avaliaclf[self.__chave_treino].colunas_metricas_condicionais_calculadas()):
            return self.__dict_dfs_metricas[col_ref]
    
    def grafico_distribuicao(self, col_ref = [], alpha = 0.5, bins = None, explicita_resto = False, rot = None, conv_str = False, ticks_chars = None, figsize = [6, 4]):
        colunas = self.__dict_avaliaclf[self.__chave_treino].distribuicoes.retorna_colunas_dataset()
        #Transforma uma string única em uma lista
        if(isinstance(col_ref, str)):
            col_ref = [col_ref]
        if(len(col_ref) != 0):
            colunas = col_ref
            
        for col_ref in colunas:
            if(col_ref in self.__dict_avaliaclf[self.__chave_treino].colunas_metricas_condicionais_calculadas()):
                paleta_cores = sns.color_palette("colorblind") #Paleta de cores para daltonico
                #Plota informações da distribuição da variável de referência nos dados
                with sns.axes_style("whitegrid"):
                    fig, ax = plt.subplots(1, 1, figsize = figsize)
                    i = 0
                    for chave in self.__chaves:
                        valores, frac, largura = self.__dict_avaliaclf[chave].distribuicoes.curva_distribuicao(col_ref, bins, explicita_resto)
                        frac_na = self.__dict_avaliaclf[chave].distribuicoes.retorna_qtds_na(col_ref)/self.__dict_avaliaclf[chave].distribuicoes.retorna_shape_dataset()[0]
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
                
    def grafico_probabilidade_condicional(self, col_ref, ymax = 0, explicita_resto = False, rot = None, 
                                          alga_signif = None, unit = None, conv_str = True, ticks_chars = None, figsize_base = [6, 4]):
        if(col_ref in self.__dict_avaliaclf[self.__chave_treino].colunas_metricas_condicionais_calculadas()):
            paleta_cores = sns.color_palette("colorblind") #Paleta de cores para daltonico
            #Plot a curva de probabilidade dada pelo Alvo e pela Prob do Classificador
            with sns.axes_style("whitegrid"):
                if(ymax > 0 and ymax <= 1):
                    fig, axs = plt.subplots(1, self.__num_dfs, figsize = [figsize_base[0]*self.__num_dfs, figsize_base[1]], 
                                            sharex = False, sharey = True)
                    plt.subplots_adjust(wspace = 0.01)
                else:
                    fig, axs = plt.subplots(1, self.__num_dfs, figsize = [figsize_base[0]*self.__num_dfs, figsize_base[1]])
                i = 0
                hlds = []             
                
                for chave in self.__chaves:
                    if(self.__num_dfs > 1):
                        ax = axs[i]
                    else:
                        ax = axs
                    df, valores = self.__dict_avaliaclf[chave].valor_metricas_condicionais(col_ref, retorna_valores = True, explicita_resto = explicita_resto)
                    
                    prob1 = df['Frac_1'].values
                    if(self.__col_prob is not None):
                        media_prob = df['Media_Prob'].values
                    else:
                        media_prob = None
                    
                    if(alga_signif is not None):
                        str_conv = ''.join(['%.', np.str(alga_signif), 'g'])
                        valores = np.array([float(str_conv % v) for v in valores])
                    if(unit is not None):
                        valores = np.array([np.datetime_as_string(v, unit = unit) for v in valores])
                    if(conv_str):
                        valores = valores.astype(str)
                    if(ticks_chars is not None):
                        valores = [re.sub("(.{" + str(ticks_chars) + "})", "\\1\n", v, 0, re.DOTALL) for v in valores]
                    
                    ax.bar(valores, prob1, color = paleta_cores[i])
                    if(media_prob is not None):
                        ax.plot(valores, media_prob, '-o', color = 'black', linewidth = 2)
                    
                    ax.set_xticks(valores)
                    if(rot is not None):
                        ax.set_xticklabels(valores, rotation = rot)
                        
                    if(ymax > 0 and ymax <= 1):
                        ax.set_ylim([0, ymax])
                        if(i == 0):
                            ax.set_xlabel(col_ref)
                            ax.set_ylabel('Probabilidade de 1')
                    else:
                        ax.set_xlabel(col_ref)
                        ax.set_ylabel('Probabilidade de 1')
                    
                    ax.set_title(chave)
                    hlds.append(mpl.patches.Patch(color = paleta_cores[i], label = chave))
                    i = i + 1   
                    
                hlds.append(mpl.patches.Patch(color = 'black', label = 'Classificador'))
                plt.legend(handles = hlds, bbox_to_anchor = (1.05, 1), loc = 'upper left')
                plt.show()
                
    def grafico_metricas_condicionais(self, col_ref, metricas, ylim = [0, 0], explicita_resto = False, rot = None, 
                                      alga_signif = None, unit = None, conv_str = True, ticks_chars = None, figsize_base = [6, 4]):
        if(col_ref in self.__dict_avaliaclf[self.__chave_treino].colunas_metricas_condicionais_calculadas()):
            paleta_cores = sns.color_palette("colorblind") #Paleta de cores para daltonico
            with sns.axes_style("whitegrid"):
                if(ylim[1] > ylim[0]):
                    fig, axs = plt.subplots(1, self.__num_dfs, figsize = [figsize_base[0]*self.__num_dfs, figsize_base[1]], 
                                            sharex = False, sharey = True)
                    plt.subplots_adjust(wspace = 0.01)
                else:
                    fig, axs = plt.subplots(1, self.__num_dfs, figsize = [figsize_base[0]*self.__num_dfs, figsize_base[1]])
                j = 0
                hlds = []
                for chave in self.__chaves:
                    if(self.__num_dfs > 1):
                        ax = axs[j]
                    else:
                        ax = axs

                    df, valores = self.__dict_avaliaclf[chave].valor_metricas_condicionais(col_ref, retorna_valores = True, explicita_resto = explicita_resto)
                    
                    if(alga_signif is not None):
                        str_conv = ''.join(['%.', np.str(alga_signif), 'g'])
                        valores = np.array([float(str_conv % v) for v in valores])
                    if(unit is not None):
                        valores = np.array([np.datetime_as_string(v, unit = unit) for v in valores])
                    if(conv_str):
                        valores = valores.astype(str)
                    if(ticks_chars is not None):
                        valores = [re.sub("(.{" + str(ticks_chars) + "})", "\\1\n", v, 0, re.DOTALL) for v in valores]
                    
                    for i in range(len(metricas)):
                        ax.plot(valores, df[metricas[i]].values, '-o', color = paleta_cores[i])     
                        if(chave == self.__chave_treino):
                            hlds.append(mpl.patches.Patch(color = paleta_cores[i], label = metricas[i]))
                    
                    ax.set_xticks(valores)
                    if(rot is not None):
                        ax.set_xticklabels(valores, rotation = rot)
                    
                    if(ylim[1] > ylim[0]):
                        ax.set_ylim(ylim)
                        if(j == 0):
                            ax.set_xlabel(col_ref)
                            ax.set_ylabel('Metricas')
                    else:
                        ax.set_xlabel(col_ref)
                        ax.set_ylabel('Metricas')
                        
                    ax.set_title(chave)
                    j = j + 1
                plt.legend(handles = hlds, bbox_to_anchor = (1.05, 1), loc = 'upper left')
                plt.show()
                
    def grafico_metrica_condicional_geral(self, col_ref, metrica, ylim = [0, 0], explicita_resto = False, rot = None, 
                                          alga_signif = None, unit = None, conv_str = True, ticks_chars = None, figsize = [6, 4]):
        if(col_ref in self.__dict_avaliaclf[self.__chave_treino].colunas_metricas_condicionais_calculadas()):
            paleta_cores = sns.color_palette("colorblind") #Paleta de cores para daltonico
            with sns.axes_style("whitegrid"):
                fig, ax = plt.subplots(1, 1, figsize = figsize)
                i = 0
                for chave in self.__chaves:

                    df = self.__dict_dfs_metricas[col_ref][chave]
                    df_colunas = list(df.columns)
                    if('Str' in df_colunas):
                        valores = df['Str'].values
                    elif('Categoria' in df_colunas):
                        df = df.sort_values('Frac_1')
                        if(explicita_resto):
                            valores = df['Categoria'].values
                        else:
                            valores = np.where(df['Código'].values == 0, 'resto', df['Categoria'].values)
                    elif('Valor' in df_colunas):
                        valores = df['Valor'].values
                    
                    if(alga_signif is not None):
                        str_conv = ''.join(['%.', np.str(alga_signif), 'g'])
                        valores = np.array([float(str_conv % v) for v in valores])
                    if(unit is not None):
                        valores = np.array([np.datetime_as_string(v, unit = unit) for v in valores])
                    if(conv_str):
                        valores = valores.astype(str)
                    if(ticks_chars is not None):
                        valores = [re.sub("(.{" + str(ticks_chars) + "})", "\\1\n", v, 0, re.DOTALL) for v in valores]
                    
                    ax.plot(valores, df[metrica].values, '-o', color = paleta_cores[i], label = chave)
                    i = i + 1
                    
                ax.set_xticks(valores)
                if(rot is not None):
                    ax.set_xticklabels(valores, rotation = rot)
                    
                if(ylim[1] > ylim[0]):
                    ax.set_ylim(ylim)
                ax.set_xlabel(col_ref)
                ax.set_ylabel(metrica)

                ax.legend(bbox_to_anchor = (1.05, 1), loc = 'upper left')
                plt.show()

###########################################

###########################################

class AvaliaRegressao:

    def __init__(self, df, col_alvo, col_pred = None, y_ref = None, y2_ref = None, num_kendalltau = 10000, num_div = None, num_cat = None, unit = None):
        self.distribuicoes = DistribuicoesDataset(df, num_div = num_div, num_cat = num_cat, unit = unit, autorun = False)
        
        self.__col_alvo = col_alvo
        self.__col_pred = col_pred
        self.__num_kendalltau = num_kendalltau
        
        if(col_pred is not None):
            #Calculas as métricas gerais do dataset
            self.metricas_gerais = AletricasRegressao(df[col_alvo].values, df[col_pred].values, y_ref = y_ref, y2_ref = y2_ref, num_kendalltau = num_kendalltau)
            
            #Valor de referência para as métricas
            if(y_ref is None or y2_ref is None):
                self.__y_ref, self.__y2_ref = self.metricas_gerais.valor_medias_alvo()
            else:
                self.__y_ref = y_ref
                self.__y2_ref = y2_ref
        else:
            self.__y_pred = None
            self.__y_ref = None
            self.__y2_ref = None
        
        self.__dict_soma_alvo = {}
        self.__dict_media_alvo = {}
        self.__dict_r2 = {}
        self.__dict_ratio_r2 = {}
        
        self.__dict_soma_pred = {}
        self.__dict_media_pred = {}
        self.__dict_metricas = {}
    
    def colunas_metricas_condicionais_calculadas(self):
        return self.__dict_soma_alvo.keys()
    
    def calcula_metricas_condicionais(self, df, col_ref = [], parametros_padrao = True, num_kendalltau = 10000, num_div = 20, num_cat = 5, unit = None):
        #Transforma uma string única em uma lista
        colunas = self.distribuicoes.retorna_colunas_dataset()
        if(isinstance(col_ref, str)):
            col_ref = [col_ref]
        if(len(col_ref) != 0):
            colunas = col_ref
        
        num_linhas = self.distribuicoes.retorna_shape_dataset()[0]
        
        if(parametros_padrao):
            num_kendalltau = self.__num_kendalltau
        
        for col_ref in colunas:
            self.distribuicoes.trata_coluna(df, col_ref, parametros_padrao, num_div, num_cat, unit)
            
            valores = self.distribuicoes.retorna_valores(df, col_ref)
            flag_na = np.array(self.distribuicoes.retorna_flag_na(col_ref))
            qtd_nao_nulo = num_linhas - self.distribuicoes.retorna_qtds_na(col_ref)
        
            valores = valores[~flag_na]
            y = df.loc[~flag_na, self.__col_alvo].values
            
            inds_ordenado, primeira_ocorrencia, qtds, qtd_unicos = indices_qtds(valores)
            y_agrup = np.split(y[inds_ordenado], primeira_ocorrencia[1:])
            
            y_agrup = np.split(y[inds_ordenado], primeira_ocorrencia[1:])
            soma = np.array([soma_vetor(v) for v in y_agrup])
            self.__dict_soma_alvo[col_ref] = soma
            media = soma/qtds
            self.__dict_media_alvo[col_ref] = media
            
            #Precisa ser Otimizado!!!
            vars_cond = np.array([np.var(v) for v in y_agrup])
            try:
                feature = df.loc[~flag_na, col_ref].values
                vars_cond_feature = np.array([np.var(v) for v in np.split(feature[inds_ordenado], primeira_ocorrencia[1:])])
            except:
                #No caso de strings (categóricas), por definição, o split deixa a feature totalmente explicada
                feature = valores
                vars_cond_feature = np.zeros(qtd_unicos)
            self.__dict_r2[col_ref], self.__dict_ratio_r2[col_ref] = calcula_r2_ratio_r2_condicional(y, vars_cond, qtds, qtd_nao_nulo, feature, vars_cond_feature)
            
            if(self.__col_pred is not None):
                y_pred = df.loc[~flag_na, self.__col_pred].values
                y_pred_agrup = np.split(y_pred[inds_ordenado], primeira_ocorrencia[1:])
                soma_pred = np.array([np.sum(v) for v in y_pred_agrup])
                self.__dict_soma_pred[col_ref] = soma_pred
                self.__dict_media_pred[col_ref] = soma_pred/qtds
                self.__dict_metricas[col_ref] = np.array([AletricasRegressao(y_agrup[i], y_pred_agrup[i], 
                                                          y_ref = self.__y_ref, y2_ref = self.__y2_ref, num_kendalltau = num_kendalltau) for i in range(qtd_unicos)])
                                                          
        #Ordena
        self.__dict_r2 = dict(reversed(sorted(self.__dict_r2.items(), key = lambda x: x[1])))
        self.__dict_ratio_r2 = dict(reversed(sorted(self.__dict_ratio_r2.items(), key = lambda x: x[1])))
    
    def r2(self):
        return pd.Series(self.__dict_r2)

    def razao_r2(self):
        return pd.Series(self.__dict_ratio_r2)
    
    def valor_metricas_condicionais(self, col_ref, retorna_valores = False, explicita_resto = False):
        df = pd.DataFrame()
        if(col_ref in self.__dict_soma_alvo.keys()):
        
            df = self.distribuicoes.info_distribuicao(col_ref)
            
            df['Soma_Alvo'] = self.__dict_soma_alvo[col_ref]
            df['Media_Alvo'] = self.__dict_media_alvo[col_ref]
            
            if(self.__col_pred is not None):
                df['Soma_Pred'] = self.__dict_soma_pred[col_ref]
                df['Media_Pred'] = self.__dict_media_pred[col_ref]
                df['Metricas'] = self.__dict_metricas[col_ref]
                df = pd.concat([df, df['Metricas'].apply(lambda x: x.valor_metricas(estatisticas_globais = False))], axis = 1)
                df = df.drop('Metricas', axis = 1)
                
        if(retorna_valores):
            df_colunas = list(df.columns)
            if('Str' in df_colunas):
                valores = df['Str'].values
            elif('Categoria' in df_colunas):
                df = df.sort_values('Media_Alvo')
                if(explicita_resto):
                    valores = df['Categoria'].values
                else:
                    valores = np.where(df['Código'].values == 0, 'resto', df['Categoria'].values)
            elif('Valor' in df_colunas):
                valores = df['Valor'].values
            return df, valores
        else:
            return df
    
    def grafico_media_condicional(self, col_ref, ylim = [0, 0], explicita_resto = False, rot = None, 
                                  alga_signif = None, unit = None, conv_str = True, ticks_chars = None, figsize = [6, 4]):
        if(col_ref in self.__dict_soma_alvo.keys()):
            paleta_cores = sns.color_palette("colorblind") #Paleta de cores para daltonico
            
            #Plot a curva de probabilidade dada pelo Alvo e pela Prob do Classificador
            with sns.axes_style("whitegrid"):
                fig, axs = plt.subplots(1, 1, figsize = figsize)
                
                df, valores = self.valor_metricas_condicionais(col_ref, retorna_valores = True, explicita_resto = explicita_resto)
                
                media_alvo = df['Media_Alvo'].values
                if(self.__col_pred is not None):
                    media_pred = df['Media_Pred'].values
                else:
                    media_pred = None
                r2 = self.__dict_r2[col_ref]
                razao_r2 = self.__dict_ratio_r2[col_ref]
                
                if(alga_signif is not None):
                    str_conv = ''.join(['%.', np.str(alga_signif), 'g'])
                    valores = np.array([float(str_conv % v) for v in valores])
                if(unit is not None):
                    valores = np.array([np.datetime_as_string(v, unit = unit) for v in valores])
                if(conv_str):
                    valores = valores.astype(str)
                if(ticks_chars is not None):
                    valores = [re.sub("(.{" + str(ticks_chars) + "})", "\\1\n", v, 0, re.DOTALL) for v in valores]
                
                axs.plot(valores, media_alvo, 'o-', color = paleta_cores[0], label = 'Real')
                if(media_pred is not None):
                    axs.plot(valores, media_pred, '-o', color = 'black', linewidth = 2, label = 'Regressor')
                
                plt.gcf().text(1, 0.5, 'R2 = ' + '%.2g' % r2 + '\n' + 'Razão R2 = ' + '%.2g' % razao_r2, bbox = dict(facecolor = 'white', edgecolor = 'k', boxstyle = 'round'))
                
                if(ylim[1] > ylim[0]):
                    axs.set_ylim(ylim)
                axs.set_xticks(valores)
                axs.set_xlabel(col_ref)
                axs.set_ylabel('Média dos Valores')
                axs.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                if(rot is not None):
                    plt.xticks(rotation = rot)
                plt.show()

    def grafico_metricas_condicionais(self, col_ref, metricas, ylim = [0, 0], explicita_resto = False, rot = None, 
                                      alga_signif = None, unit = None, conv_str = True, ticks_chars = None, figsize = [6, 4]):
        if(col_ref in self.__dict_soma_alvo.keys()):
            paleta_cores = sns.color_palette("colorblind") #Paleta de cores para daltonico 
            #Plot a curva de métrica em função da coluna de referência
            with sns.axes_style("whitegrid"):
                fig, axs = plt.subplots(1, 1, figsize = figsize)
                
                df, valores = self.valor_metricas_condicionais(col_ref, retorna_valores = True, explicita_resto = explicita_resto)
                
                if(alga_signif is not None):
                    str_conv = ''.join(['%.', np.str(alga_signif), 'g'])
                    valores = np.array([float(str_conv % v) for v in valores])
                if(unit is not None):
                    valores = np.array([np.datetime_as_string(v, unit = unit) for v in valores])                
                if(conv_str):
                    valores = valores.astype(str)
                if(ticks_chars is not None):
                    valores = [re.sub("(.{" + str(ticks_chars) + "})", "\\1\n", v, 0, re.DOTALL) for v in valores]
                
                for i in range(len(metricas)):
                    axs.plot(valores, df[metricas[i]].values, '-o', color = paleta_cores[i], label = metricas[i])

                if(ylim[1] > ylim[0]):
                    axs.set_ylim(ylim)
                axs.set_xticks(valores)
                axs.set_xlabel(col_ref)
                axs.set_ylabel('Metricas') 
                axs.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                if(rot is not None):
                    plt.xticks(rotation = rot)
                plt.show()
                
    def barra_calor(self, col_ref, alpha_max = 1, sinal_cor = False, explicita_resto = False, rot = None, 
                    alga_signif = None, unit = None, conv_str = True, ticks_chars = None, figsize = [6, 4]):
        #Normaliza os valores para plotar em cor
        def normaliza_media(sinal_cor, media):
            minimo = np.min(media)
            maximo = np.max(media)
            if(sinal_cor):
                if(maximo > 0 and minimo < 0):
                    norm = np.array([v/maximo if v > 0 else v/np.abs(minimo) for v in media])
                elif(maximo > 0 and minimo >= 0):
                    norm = media/maximo
                elif(maximo <= 0 and minimo < 0):
                    norm = media/np.abs(minimo)
            else:
                if(maximo == minimo):
                    norm = np.array([0 for v in media]).astype(float)
                else:
                    norm = (2*media - maximo - minimo)/(maximo - minimo)
            return norm
    
        if(col_ref in self.__dict_soma_alvo.keys()):
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
            
            with sns.axes_style("whitegrid"):
                fig, ax = plt.subplots(1, 1, figsize = figsize, constrained_layout = True)
                
                df, valores = self.valor_metricas_condicionais(col_ref, retorna_valores = True, explicita_resto = explicita_resto)
                
                if(self.__col_pred is not None):
                    media_pred = df['Media_Pred'].values
                else:
                    media_pred = df['Media_Alvo'].values
                
                if(alga_signif is not None):
                    str_conv = ''.join(['%.', np.str(alga_signif), 'g'])
                    valores = np.array([float(str_conv % v) for v in valores])
                if(unit is not None):
                    valores = np.array([np.datetime_as_string(v, unit = unit) for v in valores])
                if(conv_str):
                    valores = valores.astype(str)
                if(ticks_chars is not None):
                    valores = [re.sub("(.{" + str(ticks_chars) + "})", "\\1\n", v, 0, re.DOTALL) for v in valores]
                
                media_norm = alpha_max*normaliza_media(sinal_cor, media_pred)            
                cores_plot = cores[np.floor((media_norm + 1)*(N-1)/2).astype(int)]
                ax.imshow([cores_plot], aspect = 0.5*(valores.size/10), interpolation = 'spline16')
                ax.set_yticks([])
                ax.grid(False)
                
                ax.set_xticks(range(0, valores.size))
                ax.set_xticklabels(valores.astype(str))
                ax.set_title(col_ref + ':', loc = 'left')
                if(rot is not None):
                    plt.xticks(rotation = rot)              
                plt.show()
                
##############################

##############################

class AvaliaDatasetsRegressao:

    def __init__(self, dict_dfs, col_alvo, col_pred = None, num_kendalltau = 10000, num_div = None, num_cat = None, unit = None, chave_treino = 'Treino'):
        self.__chaves = dict_dfs.keys()
        self.__num_dfs = len(dict_dfs)
        self.__chave_treino = chave_treino
        self.__col_alvo = col_alvo
        self.__col_pred = col_pred
        
        self.__num_kendalltau = num_kendalltau
        
        self.__dict_avaliargs = {}
        if(self.__chave_treino in self.__chaves):
            avaliargs_treino = AvaliaRegressao(dict_dfs[chave_treino], col_alvo, col_pred, num_kendalltau = num_kendalltau, 
                                               num_div = num_div, num_cat = num_cat, unit = unit)
            #Pega o y_ref do treino
            self.__y_ref, self.__y2_ref = avaliargs_treino.metricas_gerais.valor_medias_alvo()
            self.__dict_avaliargs[self.__chave_treino] = avaliargs_treino
        for chave in self.__chaves:
            if(chave != self.__chave_treino):
                self.__dict_avaliargs[chave] = AvaliaRegressao(dict_dfs[chave], col_alvo, col_pred, self.__y_ref, self.__y2_ref, num_kendalltau,
                                                               num_div, num_cat, unit)
                                                               
        self.__distribuicoes_geral = DistribuicoesDataset(pd.concat(list(dict_dfs.values())), num_div = num_div, num_cat = num_cat, unit = unit, autorun = False)
        self.__dict_dfs_metricas = {}
        
    def avaliadores_individuais(self):
        return self.__dict_avaliargs
        
    def calcula_metricas_condicionais(self, dict_dfs, col_ref = [], parametros_padrao = True, num_kendalltau = 10000, num_div = 20, num_cat = 5, unit = None):
        for chave in self.__chaves:
            self.__dict_avaliargs[chave].calcula_metricas_condicionais(dict_dfs[chave], col_ref, parametros_padrao, num_kendalltau, num_div, num_cat, unit)
            
        colunas = self.__dict_avaliargs[self.__chave_treino].distribuicoes.retorna_colunas_dataset()
        #Transforma uma string única em uma lista
        if(isinstance(col_ref, str)):
            col_ref = [col_ref]
        if(len(col_ref) != 0):
            colunas = col_ref
            
        if(parametros_padrao):
            num_kendalltau = self.__num_kendalltau
            
        for col_ref in colunas:
            self.__distribuicoes_geral.trata_coluna(pd.concat(list(dict_dfs.values())), col_ref, parametros_padrao, num_div, num_cat, unit)
            
            dfs_metricas = {}
            for chave in self.__chaves:
                valores = self.__distribuicoes_geral.retorna_valores(dict_dfs[chave], col_ref)
                flag_na = np.array(self.__dict_avaliargs[chave].distribuicoes.retorna_flag_na(col_ref))
                num_linhas = self.__dict_avaliargs[chave].distribuicoes.retorna_shape_dataset()[0]
                qtd_nao_nulo = num_linhas - self.__dict_avaliargs[chave].distribuicoes.retorna_qtds_na(col_ref)
                
                valores = valores[~flag_na]
                y = dict_dfs[chave].loc[~flag_na, self.__col_alvo].values
                
                inds_ordenado, primeira_ocorrencia, qtds, qtd_unicos = indices_qtds(valores)
                valores_unicos = valores[inds_ordenado][primeira_ocorrencia]
                y_agrup = np.split(y[inds_ordenado], primeira_ocorrencia[1:])
            
                y_agrup = np.split(y[inds_ordenado], primeira_ocorrencia[1:])
                soma = np.array([soma_vetor(v) for v in y_agrup])
            
                if(self.__col_pred is not None):
                    y_pred = dict_dfs[chave].loc[~flag_na, self.__col_pred].values
                    y_pred_agrup = np.split(y_pred[inds_ordenado], primeira_ocorrencia[1:])
                    soma_pred = np.array([np.sum(v) for v in y_pred_agrup])
                    metricas_vetor = np.array([AletricasRegressao(y_agrup[i], y_pred_agrup[i], 
                                               y_ref = self.__y_ref, y2_ref = self.__y2_ref, num_kendalltau = num_kendalltau) for i in range(qtd_unicos)])
                                                              
                df = self.__distribuicoes_geral.info_distribuicao(col_ref)
                if('Valor' in list(df.columns)):
                    if(df['Valor'].dtype in ['<M8[ns]', 'datetime64[ns]']):
                        indices = df.index[df['Valor'].view('i8').isin(valores_unicos)]
                    else:
                        indices = df.index[df['Valor'].isin(valores_unicos)]
                else:
                    indices = valores_unicos
                df['Soma_Alvo'] = pd.Series(soma, index = indices)
                df['Media_Alvo'] = pd.Series(soma/qtds, index = indices)
                if(self.__col_pred is not None):
                    df['Soma_Pred'] = pd.Series(soma_pred, index = indices)
                    df['Media_Pred'] = pd.Series(soma_pred/qtds, index = indices)
                    df['Metricas'] = pd.Series(metricas_vetor, index = indices)
                    df = pd.concat([df, df['Metricas'].dropna().apply(lambda x: x.valor_metricas(estatisticas_globais = False))], axis = 1)
                    df = df.drop('Metricas', axis = 1)
                dfs_metricas[chave] = df
            self.__dict_dfs_metricas[col_ref] = dfs_metricas
    
    def valor_metricas(self, estatisticas_globais = True, metricas_ref = True, alga_signif = 0, conv_str = False):
        df = pd.DataFrame(self.__dict_avaliargs[self.__chave_treino].metricas_gerais.valor_metricas(estatisticas_globais, metricas_ref, alga_signif, conv_str), columns = [self.__chave_treino])
        for chave in self.__chaves:
            if(chave != self.__chave_treino):
                df[chave] = self.__dict_avaliargs[chave].metricas_gerais.valor_metricas(estatisticas_globais, metricas_ref, alga_signif, conv_str)
        return df
        
    def valor_metricas_condicionais(self, col_ref):
        if(col_ref in self.__dict_avaliargs[self.__chave_treino].colunas_metricas_condicionais_calculadas()):
            d = {}
            for chave in self.__chaves:
                d[chave] = self.__dict_avaliargs[chave].valor_metricas_condicionais(col_ref)
            return d
            
    def valor_metricas_condicionais_geral(self, col_ref):
        if(col_ref in self.__dict_avaliargs[self.__chave_treino].colunas_metricas_condicionais_calculadas()):
            return self.__dict_dfs_metricas[col_ref]
            
    def grafico_distribuicao_acumulada(self, figsize_base = [6, 4], mesmo_y = False):
        paleta_cores = sns.color_palette("colorblind") #Paleta de cores para daltonico
        with sns.axes_style("whitegrid"):
            if(mesmo_y):
                fig, axs = plt.subplots(1, self.__num_dfs, figsize = [figsize_base[0]*self.__num_dfs, figsize_base[1]], 
                                        sharex = False, sharey = True)
                plt.subplots_adjust(wspace = 0.01)
            else:
                fig, axs = plt.subplots(1, self.__num_dfs, figsize = [figsize_base[0]*self.__num_dfs, figsize_base[1]])
            i = 0
            hlds = []             
            
            for chave in self.__chaves:
                if(self.__num_dfs > 1):
                    ax = axs[i]
                else:
                    ax = axs
                
                y_unicos, y_acum, y_pred_acum, ks = self.__dict_avaliargs[chave].metricas_gerais.curva_distribuicao_acumulada()
                
                ax.plot(y_unicos, y_acum, color = 'black', alpha = 1.0)
                ax.plot(y_unicos, y_pred_acum, color = paleta_cores[i], alpha = 1.0)

                ax.set_xlabel('Valores')
                
                if(mesmo_y):
                    ax.set_ylim([0, 1.01])
                    if(i == 0):
                        ax.set_ylabel('Probabilidade Acumulada')
                else:
                    ax.set_ylabel('Probabilidade Acumulada')
                
                ax.set_title(chave)
                hlds.append(mpl.patches.Patch(color = paleta_cores[i], label = chave))
                i = i + 1   
                
            hlds.append(mpl.patches.Patch(color = 'black', label = 'Real'))
            plt.legend(handles = hlds, bbox_to_anchor = (1.05, 1), loc = 'upper left')
            plt.show()
                
    def grafico_distribuicao(self, col_ref = [], alpha = 0.5, bins = None, explicita_resto = False, rot = None, conv_str = False, ticks_chars = None, figsize = [6, 4]):
        colunas = self.__dict_avaliargs[self.__chave_treino].distribuicoes.retorna_colunas_dataset()
        #Transforma uma string única em uma lista
        if(isinstance(col_ref, str)):
            col_ref = [col_ref]
        if(len(col_ref) != 0):
            colunas = col_ref
            
        for col_ref in colunas:
            if(col_ref in self.__dict_avaliargs[self.__chave_treino].colunas_metricas_condicionais_calculadas()):
                paleta_cores = sns.color_palette("colorblind") #Paleta de cores para daltonico
                #Plota informações da distribuição da variável de referência nos dados
                with sns.axes_style("whitegrid"):
                    fig, ax = plt.subplots(1, 1, figsize = figsize)
                    i = 0
                    for chave in self.__chaves:
                        valores, frac, largura = self.__dict_avaliargs[chave].distribuicoes.curva_distribuicao(col_ref, bins, explicita_resto)
                        frac_na = self.__dict_avaliargs[chave].distribuicoes.retorna_qtds_na(col_ref)/self.__dict_avaliargs[chave].distribuicoes.retorna_shape_dataset()[0]
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
                
    def grafico_media_condicional(self, col_ref, ylim = [0, 0], explicita_resto = False, rot = None, 
                                  alga_signif = None, unit = None, conv_str = True, ticks_chars = None, figsize_base = [6, 4]):
        if(col_ref in self.__dict_avaliargs[self.__chave_treino].colunas_metricas_condicionais_calculadas()):
            paleta_cores = sns.color_palette("colorblind") #Paleta de cores para daltonico
            #Plot a curva de probabilidade dada pelo Alvo e pela Prob do Classificador
            with sns.axes_style("whitegrid"):
                if(ylim[1] > ylim[0]):
                    fig, axs = plt.subplots(1, self.__num_dfs, figsize = [figsize_base[0]*self.__num_dfs, figsize_base[1]], 
                                            sharex = False, sharey = True)
                    plt.subplots_adjust(wspace = 0.01)
                else:
                    fig, axs = plt.subplots(1, self.__num_dfs, figsize = [figsize_base[0]*self.__num_dfs, figsize_base[1]])
                i = 0
                hlds = []
                for chave in self.__chaves:
                    if(self.__num_dfs > 1):
                        ax = axs[i]
                    else:
                        ax = axs
                    
                    df, valores = self.__dict_avaliargs[chave].valor_metricas_condicionais(col_ref, retorna_valores = True, explicita_resto = explicita_resto)
                
                    media_alvo = df['Media_Alvo'].values
                    if(self.__col_pred is not None):
                        media_pred = df['Media_Pred'].values
                    else:
                        media_pred = None
                    
                    if(alga_signif is not None):
                        str_conv = ''.join(['%.', np.str(alga_signif), 'g'])
                        valores = np.array([float(str_conv % v) for v in valores])
                    if(unit is not None):
                        valores = np.array([np.datetime_as_string(v, unit = unit) for v in valores])
                    if(conv_str):
                        valores = valores.astype(str)
                    if(ticks_chars is not None):
                        valores = [re.sub("(.{" + str(ticks_chars) + "})", "\\1\n", v, 0, re.DOTALL) for v in valores]
                    
                    ax.plot(valores, media_alvo, 'o-', color = paleta_cores[i])
                    if(media_pred is not None):
                        ax.plot(valores, media_pred, '-o', color = 'black', linewidth = 2)
                        
                    ax.set_xticks(valores)
                    if(rot is not None):
                        ax.set_xticklabels(valores, rotation = rot)
                        
                    if(ylim[1] > ylim[0]):
                        ax.set_ylim(ylim)
                        if(i == 0):
                            ax.set_xlabel(col_ref)
                            ax.set_ylabel('Média dos Valores')
                    else:
                        ax.set_xlabel(col_ref)
                        ax.set_ylabel('Média dos Valores')
                        
                    ax.set_title(chave)
                    hlds.append(mpl.patches.Patch(color = paleta_cores[i], label = chave))
                    i = i + 1
                    
                hlds.append(mpl.patches.Patch(color = 'black', label = 'Regressor'))
                plt.legend(handles = hlds, bbox_to_anchor = (1.05, 1), loc = 'upper left')
                plt.show()
                
    def grafico_metricas_condicionais(self, col_ref, metricas, ylim = [0, 0], explicita_resto = False, rot = None, 
                                      alga_signif = None, unit = None, conv_str = True, ticks_chars = None, figsize_base = [6, 4]):
        if(col_ref in self.__dict_avaliargs[self.__chave_treino].colunas_metricas_condicionais_calculadas()):
            paleta_cores = sns.color_palette("colorblind") #Paleta de cores para daltonico
            with sns.axes_style("whitegrid"):
                if(ylim[1] > ylim[0]):
                    fig, axs = plt.subplots(1, self.__num_dfs, figsize = [figsize_base[0]*self.__num_dfs, figsize_base[1]], 
                                            sharex = False, sharey = True)
                    plt.subplots_adjust(wspace = 0.01)
                else:
                    fig, axs = plt.subplots(1, self.__num_dfs, figsize = [figsize_base[0]*self.__num_dfs, figsize_base[1]])
                j = 0
                hlds = []
                for chave in self.__chaves:
                    if(self.__num_dfs > 1):
                        ax = axs[j]
                    else:
                        ax = axs

                    df, valores = self.__dict_avaliargs[chave].valor_metricas_condicionais(col_ref, retorna_valores = True, explicita_resto = explicita_resto)
                    
                    if(alga_signif is not None):
                        str_conv = ''.join(['%.', np.str(alga_signif), 'g'])
                        valores = np.array([float(str_conv % v) for v in valores])
                    if(unit is not None):
                        valores = np.array([np.datetime_as_string(v, unit = unit) for v in valores])
                    if(conv_str):
                        valores = valores.astype(str)
                    if(ticks_chars is not None):
                        valores = [re.sub("(.{" + str(ticks_chars) + "})", "\\1\n", v, 0, re.DOTALL) for v in valores]
                    
                    for i in range(len(metricas)):
                        ax.plot(valores, df[metricas[i]].values, '-o', color = paleta_cores[i])     
                        if(chave == self.__chave_treino):
                            hlds.append(mpl.patches.Patch(color = paleta_cores[i], label = metricas[i]))
                    
                    ax.set_xticks(valores)
                    if(rot is not None):
                        ax.set_xticklabels(valores, rotation = rot)
                    
                    if(ylim[1] > ylim[0]):
                        ax.set_ylim(ylim)
                        if(j == 0):
                            ax.set_xlabel(col_ref)
                            ax.set_ylabel('Metricas')
                    else:
                        ax.set_xlabel(col_ref)
                        ax.set_ylabel('Metricas')
                        
                    ax.set_title(chave)
                    j = j + 1
                plt.legend(handles = hlds, bbox_to_anchor = (1.05, 1), loc = 'upper left')
                plt.show()
                
    def grafico_metrica_condicional_geral(self, col_ref, metrica, ylim = [0, 0], explicita_resto = False, rot = None, 
                                          alga_signif = None, unit = None, conv_str = True, ticks_chars = None, figsize = [6, 4]):
        if(col_ref in self.__dict_avaliargs[self.__chave_treino].colunas_metricas_condicionais_calculadas()):
            paleta_cores = sns.color_palette("colorblind") #Paleta de cores para daltonico
            with sns.axes_style("whitegrid"):
                fig, ax = plt.subplots(1, 1, figsize = figsize)
                i = 0
                for chave in self.__chaves:

                    df = self.__dict_dfs_metricas[col_ref][chave]
                    df_colunas = list(df.columns)
                    if('Str' in df_colunas):
                        valores = df['Str'].values
                    elif('Categoria' in df_colunas):
                        df = df.sort_values('Media_Alvo')
                        if(explicita_resto):
                            valores = df['Categoria'].values
                        else:
                            valores = np.where(df['Código'].values == 0, 'resto', df['Categoria'].values)
                    elif('Valor' in df_colunas):
                        valores = df['Valor'].values
                    
                    if(alga_signif is not None):
                        str_conv = ''.join(['%.', np.str(alga_signif), 'g'])
                        valores = np.array([float(str_conv % v) for v in valores])
                    if(unit is not None):
                        valores = np.array([np.datetime_as_string(v, unit = unit) for v in valores])
                    if(conv_str):
                        valores = valores.astype(str)
                    if(ticks_chars is not None):
                        valores = [re.sub("(.{" + str(ticks_chars) + "})", "\\1\n", v, 0, re.DOTALL) for v in valores]
                    
                    ax.plot(valores, df[metrica].values, '-o', color = paleta_cores[i], label = chave)
                    i = i + 1
                    
                ax.set_xticks(valores)
                if(rot is not None):
                    ax.set_xticklabels(valores, rotation = rot)
                    
                if(ylim[1] > ylim[0]):
                    ax.set_ylim(ylim)
                ax.set_xlabel(col_ref)
                ax.set_ylabel(metrica)

                ax.legend(bbox_to_anchor = (1.05, 1), loc = 'upper left')
                plt.show()
                
###########################################

###########################################

class AvaliaDistribuicaoProbabilidade:

    def __init__(self, df, col_alvo, cols_prob = None, discretizador = None, y_ref = None, y2_ref = None, num_div = None, num_cat = None, unit = None):
        self.distribuicoes = DistribuicoesDataset(df, num_div = num_div, num_cat = num_cat, unit = unit, autorun = False)
        
        self.__col_alvo = col_alvo
        self.__cols_prob = cols_prob
        self.__discretizador = discretizador
        
        if(cols_prob is not None and discretizador is not None):
            #Calculas as métricas gerais do dataset
            self.metricas_gerais = AletricasDistribuicaoProbabilidade(df[col_alvo].values, df[cols_prob].values, discretizador, y_ref = y_ref, y2_ref = y2_ref)
            
            #Valor de referência para as métricas
            if(y_ref is None or y2_ref is None):
                self.__y_ref, self.__y2_ref = self.metricas_gerais.valor_medias_alvo()
            else:
                self.__y_ref = y_ref
                self.__y2_ref = y2_ref
        else:
            self.__y_pred = None
            self.__y_ref = None
            self.__y2_ref = None
        
        self.__dict_soma_alvo = {}
        self.__dict_media_alvo = {}
        
        self.__dict_desvio_alvo = {}
        
        self.__dict_ks = {}
        self.__dict_coef_ks = {}
        
        self.__dict_soma_pred = {}
        self.__dict_media_pred = {}
        
        self.__dict_desvio_pred = {}
        
        self.__dict_metricas = {}
    
    def colunas_metricas_condicionais_calculadas(self):
        return self.__dict_soma_alvo.keys()
    
    def calcula_metricas_condicionais(self, df, col_ref = [], parametros_padrao = True, num_div = 20, num_cat = 5, unit = None):
        #Transforma uma string única em uma lista
        colunas = self.distribuicoes.retorna_colunas_dataset()
        if(isinstance(col_ref, str)):
            col_ref = [col_ref]
        if(len(col_ref) != 0):
            colunas = col_ref
        
        num_linhas = self.distribuicoes.retorna_shape_dataset()[0]
        
        y = df.loc[:, self.__col_alvo].values
        media_y = calcula_media(y)
        desvio_y = math.sqrt(calcula_mse(y) - math.pow(media_y, 2))
        
        for col_ref in colunas:
            self.distribuicoes.trata_coluna(df, col_ref, parametros_padrao, num_div, num_cat, unit)
            
            valores = self.distribuicoes.retorna_valores(df, col_ref)
            flag_na = np.array(self.distribuicoes.retorna_flag_na(col_ref))
            qtd_nao_nulo = num_linhas - self.distribuicoes.retorna_qtds_na(col_ref)
        
            valores = valores[~flag_na]
            y = df.loc[~flag_na, self.__col_alvo].values
            
            inds_ordenado, primeira_ocorrencia, qtds, qtd_unicos = indices_qtds(valores)
            y_agrup = np.split(y[inds_ordenado], primeira_ocorrencia[1:])
            
            y_agrup = np.split(y[inds_ordenado], primeira_ocorrencia[1:])
            soma = np.array([soma_vetor(v) for v in y_agrup])
            self.__dict_soma_alvo[col_ref] = soma
            media = soma/qtds
            self.__dict_media_alvo[col_ref] = media
            desvio = np.array([calcula_desvio(v) for v in y_agrup])
            self.__dict_desvio_alvo[col_ref] = desvio

            y_unicos = np.unique(y)
            y_acum = calcula_distribuicao_acumulada_pontos(y, y_unicos)
            y_acum_ref = calcula_distribuicao_acumulada_normal(media_y, desvio_y, y_unicos)
            ks_ref = calcula_ks(y_acum, y_acum_ref)
            y_acum_cond = calcula_distribuicao_acumulada_normal_condicional(media, desvio, qtds, qtd_unicos, y_unicos)
            ks_cond = calcula_ks(y_acum, y_acum_cond)
            self.__dict_ks[col_ref] = ks_cond
            self.__dict_coef_ks[col_ref] = 1 - ks_cond/ks_ref
            
            if(self.__cols_prob is not None):
                y_probs = df.loc[~flag_na, self.__cols_prob].values
                y_probs_agrup = np.split(y_probs[inds_ordenado], primeira_ocorrencia[1:], axis = 0)
                y_prob = np.array([np.sum(v, axis = 0)/v.shape[0] for v in y_probs_agrup])
                media_pred = np.array([np.sum(v*self.__discretizador.media) for v in y_prob])
                media2_pred = np.array([np.sum(v*self.__discretizador.media2) for v in y_prob])
                self.__dict_media_pred[col_ref] = media_pred
                self.__dict_soma_pred[col_ref] = media_pred*qtds
                self.__dict_desvio_pred[col_ref] = np.sqrt(media2_pred - np.power((media_pred), 2))
                self.__dict_metricas[col_ref] = np.array([AletricasDistribuicaoProbabilidade(y_agrup[i], y_probs_agrup[i], self.__discretizador, 
                                                          y_ref = self.__y_ref, y2_ref = self.__y2_ref) for i in range(qtd_unicos)])
                                                          
        #Ordena
        self.__dict_ks = dict(sorted(self.__dict_ks.items(), key = lambda x: x[1]))
        self.__dict_coef_ks = dict(reversed(sorted(self.__dict_coef_ks.items(), key = lambda x: x[1])))
    
    def ks(self):
        return pd.Series(self.__dict_ks)

    def coef_ks(self):
        return pd.Series(self.__dict_coef_ks)
    
    def valor_metricas_condicionais(self, col_ref, retorna_valores = False, explicita_resto = False):
        df = pd.DataFrame()
        if(col_ref in self.__dict_soma_alvo.keys()):
        
            df = self.distribuicoes.info_distribuicao(col_ref)
            
            df['Soma_Alvo'] = self.__dict_soma_alvo[col_ref]
            df['Media_Alvo'] = self.__dict_media_alvo[col_ref]
            df['Desvio_Alvo'] = self.__dict_desvio_alvo[col_ref]
            
            if(self.__cols_prob is not None):
                df['Soma_Pred'] = self.__dict_soma_pred[col_ref]
                df['Media_Pred'] = self.__dict_media_pred[col_ref]
                df['Desvio_Pred'] = self.__dict_desvio_pred[col_ref]
                df['Metricas'] = self.__dict_metricas[col_ref]
                df = pd.concat([df, df['Metricas'].apply(lambda x: x.valor_metricas(estatisticas_globais = False))], axis = 1)
                df = df.drop('Metricas', axis = 1)
                
        if(retorna_valores):
            df_colunas = list(df.columns)
            if('Str' in df_colunas):
                valores = df['Str'].values
            elif('Categoria' in df_colunas):
                df = df.sort_values('Media_Alvo')
                if(explicita_resto):
                    valores = df['Categoria'].values
                else:
                    valores = np.where(df['Código'].values == 0, 'resto', df['Categoria'].values)
            elif('Valor' in df_colunas):
                valores = df['Valor'].values
            return df, valores
        else:
            return df
    
    def grafico_distribuicao_condicional(self, col_ref, ylim = [0, 0], explicita_resto = False, rot = None, 
                                  alga_signif = None, unit = None, conv_str = True, ticks_chars = None, figsize = [6, 4]):
        if(col_ref in self.__dict_soma_alvo.keys()):
            paleta_cores = sns.color_palette("colorblind") #Paleta de cores para daltonico
            
            #Plot a curva de probabilidade dada pelo Alvo e pela Prob do Classificador
            with sns.axes_style("whitegrid"):
                fig, axs = plt.subplots(1, 1, figsize = figsize)
                
                df, valores = self.valor_metricas_condicionais(col_ref, retorna_valores = True, explicita_resto = explicita_resto)
                
                media_alvo = df['Media_Alvo'].values
                desvio_alvo = df['Desvio_Alvo'].values
                if(self.__cols_prob is not None):
                    media_pred = df['Media_Pred'].values
                    desvio_pred = df['Desvio_Pred'].values
                else:
                    media_pred = None
                    desvio_pred = None
                ks = self.__dict_ks[col_ref]
                coef_ks = self.__dict_coef_ks[col_ref]
                
                if(alga_signif is not None):
                    str_conv = ''.join(['%.', np.str(alga_signif), 'g'])
                    valores = np.array([float(str_conv % v) for v in valores])
                if(unit is not None):
                    valores = np.array([np.datetime_as_string(v, unit = unit) for v in valores])
                if(conv_str):
                    valores = valores.astype(str)
                if(ticks_chars is not None):
                    valores = [re.sub("(.{" + str(ticks_chars) + "})", "\\1\n", v, 0, re.DOTALL) for v in valores]
                
                axs.plot(valores, media_alvo, 'o-', color = paleta_cores[0], label = 'Real')
                axs.fill_between(valores, media_alvo - desvio_alvo, media_alvo + desvio_alvo, color = paleta_cores[0], alpha = 0.2)
                if(media_pred is not None):
                    axs.errorbar(valores, media_pred, fmt = 'o-', color = 'black', linewidth = 2, yerr = desvio_pred, label = 'Regressor')
                
                plt.gcf().text(1, 0.5, 'KS = ' + '%.2g' % ks + '\n' + 'Coef. KS = ' + '%.2g' % coef_ks, bbox = dict(facecolor = 'white', edgecolor = 'k', boxstyle = 'round'))
                
                if(ylim[1] > ylim[0]):
                    axs.set_ylim(ylim)
                axs.set_xticks(valores)
                axs.set_xlabel(col_ref)
                axs.set_ylabel('Média dos Valores')
                axs.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                if(rot is not None):
                    plt.xticks(rotation = rot)
                plt.show()

    def grafico_metricas_condicionais(self, col_ref, metricas, ylim = [0, 0], explicita_resto = False, rot = None, 
                                      alga_signif = None, unit = None, conv_str = True, ticks_chars = None, figsize = [6, 4]):
        if(col_ref in self.__dict_soma_alvo.keys()):
            paleta_cores = sns.color_palette("colorblind") #Paleta de cores para daltonico 
            #Plot a curva de métrica em função da coluna de referência
            with sns.axes_style("whitegrid"):
                fig, axs = plt.subplots(1, 1, figsize = figsize)
                
                df, valores = self.valor_metricas_condicionais(col_ref, retorna_valores = True, explicita_resto = explicita_resto)
                
                if(alga_signif is not None):
                    str_conv = ''.join(['%.', np.str(alga_signif), 'g'])
                    valores = np.array([float(str_conv % v) for v in valores])
                if(unit is not None):
                    valores = np.array([np.datetime_as_string(v, unit = unit) for v in valores])                
                if(conv_str):
                    valores = valores.astype(str)
                if(ticks_chars is not None):
                    valores = [re.sub("(.{" + str(ticks_chars) + "})", "\\1\n", v, 0, re.DOTALL) for v in valores]
                
                for i in range(len(metricas)):
                    axs.plot(valores, df[metricas[i]].values, '-o', color = paleta_cores[i], label = metricas[i])

                if(ylim[1] > ylim[0]):
                    axs.set_ylim(ylim)
                axs.set_xticks(valores)
                axs.set_xlabel(col_ref)
                axs.set_ylabel('Metricas') 
                axs.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                if(rot is not None):
                    plt.xticks(rotation = rot)
                plt.show()
                
    def barra_calor(self, col_ref, alpha_max = 1, sinal_cor = False, explicita_resto = False, rot = None, 
                    alga_signif = None, unit = None, conv_str = True, ticks_chars = None, figsize = [6, 4]):
        #Normaliza os valores para plotar em cor
        def normaliza_media(sinal_cor, media):
            minimo = np.min(media)
            maximo = np.max(media)
            if(sinal_cor):
                if(maximo > 0 and minimo < 0):
                    norm = np.array([v/maximo if v > 0 else v/np.abs(minimo) for v in media])
                elif(maximo > 0 and minimo >= 0):
                    norm = media/maximo
                elif(maximo <= 0 and minimo < 0):
                    norm = media/np.abs(minimo)
            else:
                if(maximo == minimo):
                    norm = np.array([0 for v in media]).astype(float)
                else:
                    norm = (2*media - maximo - minimo)/(maximo - minimo)
            return norm
    
        if(col_ref in self.__dict_soma_alvo.keys()):
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
            
            with sns.axes_style("whitegrid"):
                fig, ax = plt.subplots(1, 1, figsize = figsize, constrained_layout = True)
                
                df, valores = self.valor_metricas_condicionais(col_ref, retorna_valores = True, explicita_resto = explicita_resto)
                
                if(self.__cols_prob is not None):
                    media_pred = df['Media_Pred'].values
                else:
                    media_pred = df['Media_Alvo'].values
                
                if(alga_signif is not None):
                    str_conv = ''.join(['%.', np.str(alga_signif), 'g'])
                    valores = np.array([float(str_conv % v) for v in valores])
                if(unit is not None):
                    valores = np.array([np.datetime_as_string(v, unit = unit) for v in valores])
                if(conv_str):
                    valores = valores.astype(str)
                if(ticks_chars is not None):
                    valores = [re.sub("(.{" + str(ticks_chars) + "})", "\\1\n", v, 0, re.DOTALL) for v in valores]
                
                media_norm = alpha_max*normaliza_media(sinal_cor, media_pred)            
                cores_plot = cores[np.floor((media_norm + 1)*(N-1)/2).astype(int)]
                ax.imshow([cores_plot], aspect = 0.5*(valores.size/10), interpolation = 'spline16')
                ax.set_yticks([])
                ax.grid(False)
                
                ax.set_xticks(range(0, valores.size))
                ax.set_xticklabels(valores.astype(str))
                ax.set_title(col_ref + ':', loc = 'left')
                if(rot is not None):
                    plt.xticks(rotation = rot)              
                plt.show()
                
##############################

##############################

class AvaliaDatasetsDistribuicaoProbabilidade:

    def __init__(self, dict_dfs, col_alvo, cols_prob = None, discretizador = None, num_div = None, num_cat = None, unit = None, chave_treino = 'Treino'):
        self.__chaves = dict_dfs.keys()
        self.__num_dfs = len(dict_dfs)
        self.__chave_treino = chave_treino
        self.__col_alvo = col_alvo
        self.__cols_prob = cols_prob
        self.__discretizador = discretizador
        
        self.__dict_avaliargs = {}
        if(self.__chave_treino in self.__chaves):
            avaliargs_treino = AvaliaDistribuicaoProbabilidade(dict_dfs[chave_treino], col_alvo, cols_prob, discretizador, 
                                                               num_div = num_div, num_cat = num_cat, unit = unit)
            #Pega o y_ref do treino
            self.__y_ref, self.__y2_ref = avaliargs_treino.metricas_gerais.valor_medias_alvo()
            self.__dict_avaliargs[self.__chave_treino] = avaliargs_treino
        for chave in self.__chaves:
            if(chave != self.__chave_treino):
                self.__dict_avaliargs[chave] = AvaliaDistribuicaoProbabilidade(dict_dfs[chave], col_alvo, cols_prob, discretizador, self.__y_ref, self.__y2_ref, 
                                                                               num_div, num_cat, unit)
                                                               
        self.__distribuicoes_geral = DistribuicoesDataset(pd.concat(list(dict_dfs.values())), num_div = num_div, num_cat = num_cat, unit = unit, autorun = False)
        self.__dict_dfs_metricas = {}
        
    def avaliadores_individuais(self):
        return self.__dict_avaliargs
        
    def calcula_metricas_condicionais(self, dict_dfs, col_ref = [], parametros_padrao = True, num_div = 20, num_cat = 5, unit = None):
        for chave in self.__chaves:
            self.__dict_avaliargs[chave].calcula_metricas_condicionais(dict_dfs[chave], col_ref, parametros_padrao, num_div, num_cat, unit)
            
        colunas = self.__dict_avaliargs[self.__chave_treino].distribuicoes.retorna_colunas_dataset()
        #Transforma uma string única em uma lista
        if(isinstance(col_ref, str)):
            col_ref = [col_ref]
        if(len(col_ref) != 0):
            colunas = col_ref
            
        for col_ref in colunas:
            self.__distribuicoes_geral.trata_coluna(pd.concat(list(dict_dfs.values())), col_ref, parametros_padrao, num_div, num_cat, unit)
            
            dfs_metricas = {}
            for chave in self.__chaves:
                valores = self.__distribuicoes_geral.retorna_valores(dict_dfs[chave], col_ref)
                flag_na = np.array(self.__dict_avaliargs[chave].distribuicoes.retorna_flag_na(col_ref))
                num_linhas = self.__dict_avaliargs[chave].distribuicoes.retorna_shape_dataset()[0]
                qtd_nao_nulo = num_linhas - self.__dict_avaliargs[chave].distribuicoes.retorna_qtds_na(col_ref)
                
                valores = valores[~flag_na]
                y = dict_dfs[chave].loc[~flag_na, self.__col_alvo].values
                
                inds_ordenado, primeira_ocorrencia, qtds, qtd_unicos = indices_qtds(valores)
                valores_unicos = valores[inds_ordenado][primeira_ocorrencia]
                y_agrup = np.split(y[inds_ordenado], primeira_ocorrencia[1:])
            
                y_agrup = np.split(y[inds_ordenado], primeira_ocorrencia[1:])
                soma = np.array([soma_vetor(v) for v in y_agrup])
                desvio = np.array([calcula_desvio(v) for v in y_agrup])
            
                if(self.__cols_prob is not None):
                    y_probs = dict_dfs[chave].loc[~flag_na, self.__cols_prob].values
                    y_probs_agrup = np.split(y_probs[inds_ordenado], primeira_ocorrencia[1:])
                    y_prob = np.array([np.sum(v, axis = 0)/v.shape[0] for v in y_probs_agrup])
                    media_pred = np.array([np.sum(v*self.__discretizador.media) for v in y_prob])
                    media2_pred = np.array([np.sum(v*self.__discretizador.media2) for v in y_prob])
                    metricas_vetor = np.array([AletricasDistribuicaoProbabilidade(y_agrup[i], y_probs_agrup[i], self.__discretizador, 
                                               y_ref = self.__y_ref, y2_ref = self.__y2_ref) for i in range(qtd_unicos)])
                                                              
                df = self.__distribuicoes_geral.info_distribuicao(col_ref)
                if('Valor' in list(df.columns)):
                    if(df['Valor'].dtype in ['<M8[ns]', 'datetime64[ns]']):
                        indices = df.index[df['Valor'].view('i8').isin(valores_unicos)]
                    else:
                        indices = df.index[df['Valor'].isin(valores_unicos)]
                else:
                    indices = valores_unicos
                df['Soma_Alvo'] = pd.Series(soma, index = indices)
                df['Media_Alvo'] = pd.Series(soma/qtds, index = indices)
                df['Desvio_Alvo'] = pd.Series(desvio, index = indices)
                if(self.__cols_prob is not None):
                    df['Soma_Pred'] = pd.Series(media_pred*qtds, index = indices)
                    df['Media_Pred'] = pd.Series(media_pred, index = indices)
                    df['Desvio_Pred'] = pd.Series(np.sqrt(media2_pred - np.power((media_pred), 2)), index = indices)
                    df['Metricas'] = pd.Series(metricas_vetor, index = indices)
                    df = pd.concat([df, df['Metricas'].dropna().apply(lambda x: x.valor_metricas(estatisticas_globais = False))], axis = 1)
                    df = df.drop('Metricas', axis = 1)
                dfs_metricas[chave] = df
            self.__dict_dfs_metricas[col_ref] = dfs_metricas
    
    def valor_metricas(self, estatisticas_globais = True, metricas_ref = True, alga_signif = 0, conv_str = False):
        df = pd.DataFrame(self.__dict_avaliargs[self.__chave_treino].metricas_gerais.valor_metricas(estatisticas_globais, metricas_ref, alga_signif, conv_str), columns = [self.__chave_treino])
        for chave in self.__chaves:
            if(chave != self.__chave_treino):
                df[chave] = self.__dict_avaliargs[chave].metricas_gerais.valor_metricas(estatisticas_globais, metricas_ref, alga_signif, conv_str)
        return df
        
    def valor_metricas_condicionais(self, col_ref):
        if(col_ref in self.__dict_avaliargs[self.__chave_treino].colunas_metricas_condicionais_calculadas()):
            d = {}
            for chave in self.__chaves:
                d[chave] = self.__dict_avaliargs[chave].valor_metricas_condicionais(col_ref)
            return d
            
    def valor_metricas_condicionais_geral(self, col_ref):
        if(col_ref in self.__dict_avaliargs[self.__chave_treino].colunas_metricas_condicionais_calculadas()):
            return self.__dict_dfs_metricas[col_ref]
    
    def grafico_distribuicao_acumulada(self, figsize_base = [6, 4], mesmo_y = False):
        paleta_cores = sns.color_palette("colorblind") #Paleta de cores para daltonico
        with sns.axes_style("whitegrid"):
            if(mesmo_y):
                fig, axs = plt.subplots(1, self.__num_dfs, figsize = [figsize_base[0]*self.__num_dfs, figsize_base[1]], 
                                        sharex = False, sharey = True)
                plt.subplots_adjust(wspace = 0.01)
            else:
                fig, axs = plt.subplots(1, self.__num_dfs, figsize = [figsize_base[0]*self.__num_dfs, figsize_base[1]])
            i = 0
            hlds = []             
            
            for chave in self.__chaves:
                if(self.__num_dfs > 1):
                    ax = axs[i]
                else:
                    ax = axs
                
                y_unicos, y_acum, y_pred_acum, ks = self.__dict_avaliargs[chave].metricas_gerais.curva_distribuicao_acumulada()
                
                ax.plot(y_unicos, y_acum, color = 'black', alpha = 1.0)
                ax.plot(y_unicos, y_pred_acum, color = paleta_cores[i], alpha = 1.0)

                ax.set_xlabel('Valores')
                
                if(mesmo_y):
                    ax.set_ylim([0, 1.01])
                    if(i == 0):
                        ax.set_ylabel('Probabilidade Acumulada')
                else:
                    ax.set_ylabel('Probabilidade Acumulada')
                
                ax.set_title(chave)
                hlds.append(mpl.patches.Patch(color = paleta_cores[i], label = chave))
                i = i + 1   
                
            hlds.append(mpl.patches.Patch(color = 'black', label = 'Real'))
            plt.legend(handles = hlds, bbox_to_anchor = (1.05, 1), loc = 'upper left')
            plt.show()
    
    #Essa distribuição é da feature, não tem relação com o alvo
    def grafico_distribuicao(self, col_ref = [], alpha = 0.5, bins = None, explicita_resto = False, rot = None, conv_str = False, ticks_chars = None, figsize = [6, 4]):
        colunas = self.__dict_avaliargs[self.__chave_treino].distribuicoes.retorna_colunas_dataset()
        #Transforma uma string única em uma lista
        if(isinstance(col_ref, str)):
            col_ref = [col_ref]
        if(len(col_ref) != 0):
            colunas = col_ref
            
        for col_ref in colunas:
            if(col_ref in self.__dict_avaliargs[self.__chave_treino].colunas_metricas_condicionais_calculadas()):
                paleta_cores = sns.color_palette("colorblind") #Paleta de cores para daltonico
                #Plota informações da distribuição da variável de referência nos dados
                with sns.axes_style("whitegrid"):
                    fig, ax = plt.subplots(1, 1, figsize = figsize)
                    i = 0
                    for chave in self.__chaves:
                        valores, frac, largura = self.__dict_avaliargs[chave].distribuicoes.curva_distribuicao(col_ref, bins, explicita_resto)
                        frac_na = self.__dict_avaliargs[chave].distribuicoes.retorna_qtds_na(col_ref)/self.__dict_avaliargs[chave].distribuicoes.retorna_shape_dataset()[0]
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
                
    def grafico_distribuicao_condicional(self, col_ref, ylim = [0, 0], explicita_resto = False, rot = None, 
                                  alga_signif = None, unit = None, conv_str = True, ticks_chars = None, figsize_base = [6, 4]):
        if(col_ref in self.__dict_avaliargs[self.__chave_treino].colunas_metricas_condicionais_calculadas()):
            paleta_cores = sns.color_palette("colorblind") #Paleta de cores para daltonico
            #Plot a curva de probabilidade dada pelo Alvo e pela Prob do Classificador
            with sns.axes_style("whitegrid"):
                if(ylim[1] > ylim[0]):
                    fig, axs = plt.subplots(1, self.__num_dfs, figsize = [figsize_base[0]*self.__num_dfs, figsize_base[1]], 
                                            sharex = False, sharey = True)
                    plt.subplots_adjust(wspace = 0.01)
                else:
                    fig, axs = plt.subplots(1, self.__num_dfs, figsize = [figsize_base[0]*self.__num_dfs, figsize_base[1]])
                i = 0
                hlds = []
                for chave in self.__chaves:
                    if(self.__num_dfs > 1):
                        ax = axs[i]
                    else:
                        ax = axs
                    
                    df, valores = self.__dict_avaliargs[chave].valor_metricas_condicionais(col_ref, retorna_valores = True, explicita_resto = explicita_resto)
                
                    media_alvo = df['Media_Alvo'].values
                    desvio_alvo = df['Desvio_Alvo'].values
                    if(self.__cols_prob is not None):
                        media_pred = df['Media_Pred'].values
                        desvio_pred = df['Desvio_Pred'].values
                    else:
                        media_pred = None
                        desvio_pred = None
                    
                    if(alga_signif is not None):
                        str_conv = ''.join(['%.', np.str(alga_signif), 'g'])
                        valores = np.array([float(str_conv % v) for v in valores])
                    if(unit is not None):
                        valores = np.array([np.datetime_as_string(v, unit = unit) for v in valores])
                    if(conv_str):
                        valores = valores.astype(str)
                    if(ticks_chars is not None):
                        valores = [re.sub("(.{" + str(ticks_chars) + "})", "\\1\n", v, 0, re.DOTALL) for v in valores]
                    
                    ax.plot(valores, media_alvo, 'o-', color = paleta_cores[i])
                    ax.fill_between(valores, media_alvo - desvio_alvo, media_alvo + desvio_alvo, color = paleta_cores[i], alpha = 0.25)
                    if(media_pred is not None):
                        ax.errorbar(valores, media_pred, fmt = 'o-', color = 'black', linewidth = 2, yerr = desvio_pred)
                        
                    ax.set_xticks(valores)
                    if(rot is not None):
                        ax.set_xticklabels(valores, rotation = rot)
                        
                    if(ylim[1] > ylim[0]):
                        ax.set_ylim(ylim)
                        if(i == 0):
                            ax.set_xlabel(col_ref)
                            ax.set_ylabel('Média dos Valores')
                    else:
                        ax.set_xlabel(col_ref)
                        ax.set_ylabel('Média dos Valores')
                        
                    ax.set_title(chave)
                    hlds.append(mpl.patches.Patch(color = paleta_cores[i], label = chave))
                    i = i + 1
                    
                hlds.append(mpl.patches.Patch(color = 'black', label = 'Regressor'))
                plt.legend(handles = hlds, bbox_to_anchor = (1.05, 1), loc = 'upper left')
                plt.show()
                
    def grafico_metricas_condicionais(self, col_ref, metricas, ylim = [0, 0], explicita_resto = False, rot = None, 
                                      alga_signif = None, unit = None, conv_str = True, ticks_chars = None, figsize_base = [6, 4]):
        if(col_ref in self.__dict_avaliargs[self.__chave_treino].colunas_metricas_condicionais_calculadas()):
            paleta_cores = sns.color_palette("colorblind") #Paleta de cores para daltonico
            with sns.axes_style("whitegrid"):
                if(ylim[1] > ylim[0]):
                    fig, axs = plt.subplots(1, self.__num_dfs, figsize = [figsize_base[0]*self.__num_dfs, figsize_base[1]], 
                                            sharex = False, sharey = True)
                    plt.subplots_adjust(wspace = 0.01)
                else:
                    fig, axs = plt.subplots(1, self.__num_dfs, figsize = [figsize_base[0]*self.__num_dfs, figsize_base[1]])
                j = 0
                hlds = []
                for chave in self.__chaves:
                    if(self.__num_dfs > 1):
                        ax = axs[j]
                    else:
                        ax = axs

                    df, valores = self.__dict_avaliargs[chave].valor_metricas_condicionais(col_ref, retorna_valores = True, explicita_resto = explicita_resto)
                    
                    if(alga_signif is not None):
                        str_conv = ''.join(['%.', np.str(alga_signif), 'g'])
                        valores = np.array([float(str_conv % v) for v in valores])
                    if(unit is not None):
                        valores = np.array([np.datetime_as_string(v, unit = unit) for v in valores])
                    if(conv_str):
                        valores = valores.astype(str)
                    if(ticks_chars is not None):
                        valores = [re.sub("(.{" + str(ticks_chars) + "})", "\\1\n", v, 0, re.DOTALL) for v in valores]
                    
                    for i in range(len(metricas)):
                        ax.plot(valores, df[metricas[i]].values, '-o', color = paleta_cores[i])     
                        if(chave == self.__chave_treino):
                            hlds.append(mpl.patches.Patch(color = paleta_cores[i], label = metricas[i]))
                    
                    ax.set_xticks(valores)
                    if(rot is not None):
                        ax.set_xticklabels(valores, rotation = rot)
                    
                    if(ylim[1] > ylim[0]):
                        ax.set_ylim(ylim)
                        if(j == 0):
                            ax.set_xlabel(col_ref)
                            ax.set_ylabel('Metricas')
                    else:
                        ax.set_xlabel(col_ref)
                        ax.set_ylabel('Metricas')
                        
                    ax.set_title(chave)
                    j = j + 1
                plt.legend(handles = hlds, bbox_to_anchor = (1.05, 1), loc = 'upper left')
                plt.show()
                
    def grafico_metrica_condicional_geral(self, col_ref, metrica, ylim = [0, 0], explicita_resto = False, rot = None, 
                                          alga_signif = None, unit = None, conv_str = True, ticks_chars = None, figsize = [6, 4]):
        if(col_ref in self.__dict_avaliargs[self.__chave_treino].colunas_metricas_condicionais_calculadas()):
            paleta_cores = sns.color_palette("colorblind") #Paleta de cores para daltonico
            with sns.axes_style("whitegrid"):
                fig, ax = plt.subplots(1, 1, figsize = figsize)
                i = 0
                for chave in self.__chaves:

                    df = self.__dict_dfs_metricas[col_ref][chave]
                    df_colunas = list(df.columns)
                    if('Str' in df_colunas):
                        valores = df['Str'].values
                    elif('Categoria' in df_colunas):
                        df = df.sort_values('Media_Alvo')
                        if(explicita_resto):
                            valores = df['Categoria'].values
                        else:
                            valores = np.where(df['Código'].values == 0, 'resto', df['Categoria'].values)
                    elif('Valor' in df_colunas):
                        valores = df['Valor'].values
                    
                    if(alga_signif is not None):
                        str_conv = ''.join(['%.', np.str(alga_signif), 'g'])
                        valores = np.array([float(str_conv % v) for v in valores])
                    if(unit is not None):
                        valores = np.array([np.datetime_as_string(v, unit = unit) for v in valores])
                    if(conv_str):
                        valores = valores.astype(str)
                    if(ticks_chars is not None):
                        valores = [re.sub("(.{" + str(ticks_chars) + "})", "\\1\n", v, 0, re.DOTALL) for v in valores]
                    
                    ax.plot(valores, df[metrica].values, '-o', color = paleta_cores[i], label = chave)
                    i = i + 1
                    
                ax.set_xticks(valores)
                if(rot is not None):
                    ax.set_xticklabels(valores, rotation = rot)
                    
                if(ylim[1] > ylim[0]):
                    ax.set_ylim(ylim)
                ax.set_xlabel(col_ref)
                ax.set_ylabel(metrica)

                ax.legend(bbox_to_anchor = (1.05, 1), loc = 'upper left')
                plt.show()