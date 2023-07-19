from numba.types.misc import DeferredType
import numpy as np
import pandas as pd

from itertools import combinations_with_replacement, combinations
from itertools import groupby
from operator import itemgetter

from functools import reduce
import math

from AleCFunctionsLearners import *
from AleTransforms import *

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from IPython import display
    
#Faz as contas com matriz numpy e retorna o calcula do termo desejado quando é solicitado
#Ou seja, não ocupa memória mas custa processamento
class TaylorLaurentExpansion:

    def __init__(self, laurent = False, ordem = 2, apenas_interacoes = False, num_features = None):
        self.__laurent = laurent
        self.__apenas_interacoes = apenas_interacoes
        self.__num_features = num_features
        if(self.__apenas_interacoes):
            self.__ordem = min(ordem, self.__num_features)
        else:
            self.__ordem = ordem
        
        def cria_tupla(tupla):
            unique, count = np.unique(tupla, return_counts = True)
            return tuple((unique[i], count[i]) for i in range(0, unique.size))
        
        self.__lista_termos = []
        features = np.arange(0, self.__num_features)
        for i in range(1, self.__ordem + 1):
            if(self.__apenas_interacoes):
                comb = list(combinations(features, r = i)) #Não precisa de potencias das features
            else:
                comb = list(combinations_with_replacement(features, r = i))
            comb = [cria_tupla(v) for v in comb]
            self.__lista_termos.extend(comb)
        
        if(self.__laurent):
            if(self.__apenas_interacoes):
                def expande_laurent(tupla):
                    sinais = [[]]
                    tam = len(tupla)
                    for i in range(0, tam):
                        sinais_temp = sinais.copy()
                        for var in sinais_temp:
                            v_new1 = var.copy()
                            v_new2 = var.copy()
                            v_new1.append(1)
                            v_new2.append(-1)
                            sinais.pop(0)
                            sinais.append(v_new1)
                            sinais.append(v_new2)
                    #Remove os sinais que darão inversos multiplicativos
                    sinais_filtrados = []
                    while(len(sinais) > 1):
                        sinal = sinais[0]
                        sinais_filtrados.append(sinal)
                        sinais.remove(sinal)
                        try:
                            sinais.remove(list(-1*np.array(sinal)))
                        except:
                            pass
                    if(len(sinais) == 1):
                        sinais_filtrados.append(sinais[0])
                    sinais = sinais_filtrados
                    return [tuple((tupla[j][0], s[j]*tupla[j][1]) for j in range(0, tam)) for s in sinais]
            else:
                def expande_laurent(tupla):
                    sinais = [[]]
                    tam = len(tupla)
                    for i in range(0, tam):
                        sinais_temp = sinais.copy()
                        for var in sinais_temp:
                            v_new1 = var.copy()
                            v_new2 = var.copy()
                            v_new1.append(1)
                            v_new2.append(-1)
                            sinais.pop(0)
                            sinais.append(v_new1)
                            sinais.append(v_new2)
                    return [tuple((tupla[j][0], s[j]*tupla[j][1]) for j in range(0, tam)) for s in sinais]
            lista_aux = []
            for tupla in self.__lista_termos:
                lista_aux.extend(expande_laurent(tupla))
            self.__lista_termos = lista_aux
        
        self.__num_termos = len(self.__lista_termos)
    
    def filtro_binario(self, lista_features_binarias):
        def eh_termo_indevido(termo, feature):
            for v in termo:
                if(v[0] == feature and v[1] != 1):
                    return True
            return False
        for feature in lista_features_binarias:
            self.__lista_termos = [termo for termo in self.__lista_termos if eh_termo_indevido(termo, feature) == False]
        self.__num_termos = len(self.__lista_termos)
    
    def numero_termos_expansao(self):
        return self.__num_termos
        
    def lista_termos(self):
        return self.__lista_termos
                
    def calcula_termo(self, X, pos_termo):
        return calcula_termo_serie(X, self.__lista_termos[pos_termo])

########################
 
########################

class DiscretizaAlvoRegressivo:

    def __init__(self, y, num_div = 10):
        self.intervalos = CortaIntervalosQuasiUniforme(y, num_div = num_div)

        y_disc = self.intervalos.aplica_discretizacao(y)
        inds_ordenado, primeira_ocorrencia, qtds, qtd_unicos = indices_qtds(y_disc)
        y_agrup = np.split(y[inds_ordenado], primeira_ocorrencia[1:])

        self.media = np.array([calcula_media(v) for v in y_agrup])
        self.media2 = np.array([calcula_mse(v) for v in y_agrup])

    def retorna_momentos(self):
        return self.media, self.media2

    def calcula_probabilidades(self, y):
        y_disc = self.intervalos.aplica_discretizacao(y)
        valores, qtds = np.unique(y_disc, return_counts = True)
        vetor_probs = np.zeros(self.intervalos.valores_medios_discretizacao().size)
        vetor_probs[valores.astype(int)] = qtds/y.size
        return vetor_probs

    def curva_probabilidade(self, probs):
        if(len(probs.shape) == 2):
            vetor_probs = np.sum(probs, axis = 0)/probs.shape[0]
        else:
            vetor_probs = probs
        faixas = self.intervalos.strings_intervalos_discretizacao()
        return faixas, vetor_probs

    def curva_distribuicao_probabilidade(self, probs, bins = None):
        if(len(probs.shape) == 2):
            vetor_probs = np.sum(probs, axis = 0)/probs.shape[0]
        else:
            vetor_probs = probs
        valores_min = np.array([v[0] for v in self.intervalos.pares_minimo_maximo_discretizacao()])
        valores_max = np.array([v[1] for v in self.intervalos.pares_minimo_maximo_discretizacao()])
        if(bins is None):
            valores = np.array([x for y in self.intervalos.pares_minimo_maximo_discretizacao() for x in y])
            fracL = np.repeat(vetor_probs/(valores_max - valores_min), 2)
            return valores, fracL
        else:
            val_min = valores_min[0]
            val_max = valores_max[-1]
            L = (val_max - val_min)/bins
            valores_corte_bins = [val_min + L*i for i in range(0, bins+1)]
            valores_medios_bins = np.array([val_min + L*(i + 1/2) for i in range(1, bins+1)])
            probs_bins = [conta_prob_bin(valores_corte_bins[i], valores_corte_bins[i+1], valores_min, valores_max, vetor_probs) for i in range(0, bins)]
            return valores_medios_bins, probs_bins, L

    def grafico_probabilidade(self, probs, rot = None, figsize = [6, 4]):
        paleta_cores = sns.color_palette("colorblind")
        with sns.axes_style("whitegrid"):
            fig, ax = plt.subplots(1, 1, figsize = figsize)
            valores, probs = self.curva_probabilidade(probs)
            ax.bar(valores, probs, color = paleta_cores[0])
            ax.set_ylabel('Probabilidade')
            ax.set_xlabel('Valores')
            ax.set_ylim(bottom = 0.0)
            if(rot is not None):
                plt.xticks(rotation = rot)
            plt.show()

    def grafico_distribuicao_probabilidade(self, probs, bins = None, alpha = 0.5, rot = None, figsize = [6, 4]):
        paleta_cores = sns.color_palette("colorblind")
        with sns.axes_style("whitegrid"):
            fig, ax = plt.subplots(1, 1, figsize = figsize)
            if(bins is None):
                valores, fracL = self.curva_distribuicao_probabilidade(probs, bins = None)
                ax.fill_between(valores, fracL, color = paleta_cores[0], alpha = alpha)
                ax.plot(valores, fracL, color = paleta_cores[0])
                ax.set_ylabel('Prob./L')
            else:
                valores, frac, largura = self.curva_distribuicao_probabilidade(probs, bins = bins)
                ax.bar(valores, frac, color = paleta_cores[0], alpha = alpha, width = largura, linewidth = 2, edgecolor = paleta_cores[0])
                ax.set_ylabel('Probabilidade')
            ax.set_xlabel('Valores')
            ax.set_ylim(bottom = 0.0)
            if(rot is not None):
                plt.xticks(rotation = rot)
            plt.show()

 ########################
 
 ########################

class ChecaMultiDatasets:

    def __init__(self, dict_Xy):
        self.nome_vars = []
        self.dict_Xy = {}
        self.num_linhas = []
        self.num_cols = []
        self.num_Xy = 0
        self.flag_features_binarias = None
        
        self.erro = False
        self.erro_predict = False
        
        for key, value in dict_Xy.items():
            self.__checa_Xy(value[0], value[1], key)
        
        if(len(self.nome_vars) > 0 and len(self.num_cols) != len(self.nome_vars)):
            self.erro = True
        else:
            if(len(set(self.num_cols)) == 1):
                self.num_cols = self.num_cols[0]
                if(len(self.nome_vars) > 0):
                    self.nome_vars = self.nome_vars[0]
            else:
                self.erro = True
        
        self.num_Xy = len(self.dict_Xy)
        self.chaves = list(self.dict_Xy.keys())
        
        if(len(self.nome_vars) == 0):
            self.nome_vars = np.array(['x' + str(i) for i in range(0, self.num_cols)])
        
        for key, value in self.dict_Xy.items():
            self.__checa_feature_binaria(value[0])
        self.features_binarias = list(self.flag_features_binarias.nonzero()[0])
    
    def __checa_feature_binaria(self, X):
        flag_feature_binaria = ~np.array([((X[:, feature] != 0) & (X[:, feature] != 1)).any() for feature in range(X.shape[1])])
        if self.flag_features_binarias is None:
            self.flag_features_binarias = flag_feature_binaria
        else:
            self.flag_features_binarias = flag_feature_binaria & flag_feature_binaria
    
    def __checa_Xy(self, X, y, key):
        if isinstance(X, pd.DataFrame):
            Xt = X.values
            self.nome_vars.append(X.columns.values)
        else:
            try:
                if len(X.shape) == 2:
                    Xt = X
                else:
                    print("Valores de entrada de treino não adequados")
                    self.erro = True
            except:
                print("Valores de entrada de treino não adequados")
                self.erro = True
        self.num_linhas.append(Xt.shape[0])
        self.num_cols.append(Xt.shape[1])
        
        if (isinstance(y, pd.DataFrame) and len(y.columns) == 1) or (isinstance(y, pd.Series)):
            yt = y.values
        else:
            try:
                if len(y.shape) == 1:
                    yt = y
                else:
                    print("Valores de alvo de treino não adequados")
                    self.erro = True
            except:
                print("Valores de alvo de treino não adequados")
                self.erro = True
        if(yt.size != Xt.shape[0]):
            print("Quantidade de exemplos não coindicem em X e y")
            self.erro = True
        self.dict_Xy[key] = (Xt, yt)
            
    def checa_X_predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        else:
            try:
                if len(X.shape) != 2:
                    print("Valores de entrada de predição não adequados")
                    self.erro_predict = True
            except:
                print("Valores de entrada de predição não adequados")
                self.erro_predict = True
        return X 
            
 ########################
 
 ########################
            
class SeriesMultiGradientRegressor:
                
    def __guarda_melhor_modelo(self):
        flag_alterar_valor = self.__lista_r2_melhor < self.__r2_temp
        self.__lista_r2_melhor[flag_alterar_valor] = self.__r2_temp[flag_alterar_valor]
        self.__lista_num_passos_melhor[flag_alterar_valor] = cria_repeticao(self.__num_passos, calcula_somatorio(flag_alterar_valor))
        for i in range(0, self.__num_Xy):
            if(flag_alterar_valor[i]):
                self.__lista_thetas_melhor[i] = self.__thetas
    
    def __derivadas_parciais_custo(self, diff, X, num_cols, flag_filtro):
        valores = cria_zeros(num_cols + 1)
        if(flag_filtro[0] == True):
            valores[0] = calcula_media(diff)
        for i in range(0, num_cols):
            if(flag_filtro[i+1] == True):
                valores[i+1] = derivada_parcial_custo_normalizado(diff, calcula_termo_serie(X, self.__lista_termos[i]), self.__means[i], self.__stds[i])
        return valores
    
    def __calcula_funclinear(self, X, num_cols, thetas):
        valores = cria_repeticao(thetas[0], X.shape[0])
        for i in range(0, num_cols):
            valores = valores + feature_normalizada_dot_theta(calcula_termo_serie(X, self.__lista_termos[i]), self.__means[i], self.__stds[i], thetas[i+1])
        return valores
        
    def __calcula_funclinear_final(self, X):
        valores = cria_repeticao(self.__thetas_finais[0], X.shape[0])
        for i in range(0, self.__num_cols_finais):
            valores = valores + feature_normalizada_dot_theta(calcula_termo_serie(X, self.__lista_termos_finais[i]), 
                                                              self.__means_finais[i], self.__stds_finais[i], self.__thetas_finais[i+1])
        return valores
    
    def setup_datasets(self, dict_Xy, laurent = False, ordem = 1, apenas_interacoes = False):
        #Equivalente a construção da rede (constrói os parâmetros que serão fitados)
        
        #Configurações de quantidade de parâmetros da função (equivalente aos neurônios e camadas)
        self.__laurent = laurent
        self.__ordem = ordem
        self.__apenas_interacoes = apenas_interacoes
        self.__series = None
        
        print('Checando Datasets')
        
        #Checa se os datasets estão adequados e extrai informações deles
        self.__checa_dados = ChecaMultiDatasets(dict_Xy)
        self.__nome_vars = self.__checa_dados.nome_vars
        self.__dict_Xy = self.__checa_dados.dict_Xy
        self.__num_linhas = self.__checa_dados.num_linhas
        self.__num_cols = self.__checa_dados.num_cols
        self.__num_Xy = self.__checa_dados.num_Xy
        self.__chaves = self.__checa_dados.chaves
        self.__features_binarias = self.__checa_dados.features_binarias           
        
        print('Filtrando Features sem Variância')
        
        #Filtro inicial de features sem variância (em algum dataset)
        lista_flag_vars_validas = []
        for key, value in self.__dict_Xy.items():
            lista_flag_vars_validas.append(calcula_flag_com_variancia(value[0], self.__num_cols))
        self.__flag_vars_validas = reduce(lambda x, y: calcula_produto(x, y), lista_flag_vars_validas)
        
        self.__num_cols = calcula_somatorio(self.__flag_vars_validas) 
        self.__nome_vars = self.__nome_vars[self.__flag_vars_validas]
        for key, value in self.__dict_Xy.items():
            self.__dict_Xy[key] = (value[0][:, self.__flag_vars_validas], value[1])
        
        print('Criando a Expansão em Série')
        
        #Expansão em série
        self.__series = TaylorLaurentExpansion(self.__laurent, self.__ordem, self.__apenas_interacoes, self.__num_cols)
        if(len(self.__features_binarias) > 0):
            self.__series.filtro_binario(self.__features_binarias)
        self.__num_cols = self.__series.numero_termos_expansao()
        if(self.__ordem == 1):
            termos_temp = self.__series.lista_termos()
            self.__lista_termos = np.empty(self.__num_cols, dtype = object)
            for i in range(self.__num_cols):
                self.__lista_termos[i] = termos_temp[i]
        else:
            self.__lista_termos = np.array(self.__series.lista_termos(), dtype = object)
        
        print("Número de Parâmetros da Expansão: " + str(self.__num_cols + 1))
        
        print('Calculando as Normalizações das Features')
        
        #Faz a normalização (e filtra colunas com desvio padrão zero)
        #Facilita calculo de Inv. de Matriz e também facilita a Descida do Gradiente
        lista_medias = []
        lista_variancias = []
        for key, value in self.__dict_Xy.items():
            medias = []
            variancias = []
            for j in range(self.__num_cols):
                vetor = calcula_termo_serie(value[0], self.__lista_termos[j])
                media, variancia = calcula_media_variancia(vetor)
                medias.append(media)
                variancias.append(variancia)
            lista_medias.append(np.array(medias))
            lista_variancias.append(np.array(variancias))
        self.__means = calcula_divisao(reduce(lambda x, y: calcula_soma(x, y), lista_medias), self.__num_Xy)
        self.__stds = calcula_divisao_e_raiz(reduce(lambda x, y: calcula_soma(x, y), lista_variancias), self.__num_Xy)
        
        #Termos efetivos (só os que tem desvio padrão não nulo)
        flag_termos_validos = reduce(lambda x, y: ambos_positivos(x, y), lista_variancias)
        self.__lista_termos = self.__lista_termos[flag_termos_validos]
        self.__means = self.__means[flag_termos_validos]
        self.__stds = self.__stds[flag_termos_validos]
        self.__num_cols = self.__lista_termos.size
        
        print("Número de Parâmetros: " + str(self.__num_cols + 1))
        
        print('Normalizando o Alvo')
        
        #Normaliza o y também: ajuda na Descida do Gradiente (Aqui não pode ter nulos mesmo!!)
        lista_medias_y = []
        lista_variancias_y = []
        for key, value in self.__dict_Xy.items():
            media, variancia = calcula_media_variancia(value[1])
            lista_medias_y.append(media)
            lista_variancias_y.append(variancia)
        self.__media_y = sum(lista_medias_y)/self.__num_Xy
        self.__desvio_y = math.sqrt(sum(lista_medias_y)/self.__num_Xy)
        
        for key, value in self.__dict_Xy.items():
            self.__dict_Xy[key] = (value[0], normaliza_vetor(value[1], self.__media_y, self.__desvio_y)) 
        
    def setup_optimizer(self, alpha = 0.01, epsilon = 1e-6, early_stop = 10, filtro_atualizacao_thetas = True):
        
        #Parâmetros da busca pelos valores dos parâmetros (equivalente ao optimizer)
        self.__alpha_max = alpha #-> Quanto queremos que a função de custo (R2 Ajust) mude a cada passo
        self.__alpha = alpha #-> Quanto queremos que a função de custo (R2 Ajust) mude a cada passo (esse pode ser atualizado no meio da otimização)
        self.__epsilon = epsilon #-> Quanto queremos de melhora mínima da função de custo (R2 Ajust) para continuar os passos
        self.__early_stop = early_stop #-> Quantos passos aceitamos com queda na função de custo (R2 Ajust) sem parar o treinamento
        self.__filtro_atualizacao_thetas = filtro_atualizacao_thetas
        
        self.__max_passos = 0 #Quantos passos já demos na busca dos parâmetros (equivalente as épocas)
        
        print('Criando o Baseline Constante')
        
        #Faz o modelo baseline: só o termo theta_0 -> cte!!! (prob média)
        self.__num_passos = 0 #Número de passos da busca de parametros
        self.__thetas = np.array([0]) #Valor dos thetas do regressor
        
        self.__lista_diff = [-value[1] for key, value in self.__dict_Xy.items()]
        
        #Salva o mse do modelo baseline (só constante = 0)
        self.__lista_mse_baseline = [calcula_mse_baseline(diff) for diff in self.__lista_diff]
        self.__mse_referencia = sum(self.__lista_mse_baseline)/self.__num_Xy
        
        #####
        ##### Algoritmo de busca no espaço de soluções usando a Descida do Gradiente #####
        #####
        
        print('Inicializando os Parâmetros para Fitting')
        
        #Inicializamos as variaveis para criar a curva viés-variância
        self.__lista_curvas_r2 = [[0] for i in range(0, self.__num_Xy)]
        
        #Começa todos os thetas zerados (com excessão do theta_0 que é o baseline)
        self.__thetas = np.append(self.__thetas, cria_zeros(self.__num_cols))
        
        self.__lista_r2_melhor = cria_zeros(self.__num_Xy)
        self.__lista_num_passos_melhor = cria_zeros(self.__num_Xy)
        self.__lista_thetas_melhor = [self.__thetas for i in range(0, self.__num_Xy)]
        self.__r2_temp = cria_zeros(self.__num_Xy) #Vetor de valores temporários para atualizar as listas acima
        
        #Inicia parametros de parada de treinamento
        self.__diff_ultimo_ganho = 0
        self.__diff_r2 = self.__epsilon
        self.__norma_quadrada_gradiente = 1
        self.__flag_filtro = cria_repeticao(True, self.__num_cols + 1)
        self.__num_params_atuali = self.__num_cols + 1
    
    def fit(self, num_passos = 0, verbose = 2, plot = True):
        if(num_passos >= 0):
            self.__max_passos = self.__max_passos + num_passos
        
        if(plot):
            display.clear_output(wait = True)
            #self.grafico_vies_variancia()
            paleta_cores = sns.color_palette("colorblind")
            with sns.axes_style("whitegrid"):
                fig, axs = plt.subplots(1, 1, figsize = (8, 6))
                axs.set_xlabel('Número de Passos')
                axs.set_ylabel('R2 Ajustado')
                for i in range(self.__num_Xy):
                    axs.plot(self.__lista_curvas_r2[i], color = paleta_cores[i % len(paleta_cores)], label = self.__chaves[i])
                    axs.scatter(self.__lista_num_passos_melhor[i], self.__lista_r2_melhor[i], color = paleta_cores[i % len(paleta_cores)])
                axs.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.show()
        if(verbose >= 1):
            print("Número de Parâmetros sendo Atualizados: " + str(self.__num_params_atuali) + " (" + str(self.__num_passos) + ")")
           
        #Loop do processo incremental de passos
        while(self.__num_passos < self.__max_passos and self.__norma_quadrada_gradiente > 0 and self.__diff_r2 >= self.__epsilon and (self.__diff_ultimo_ganho < self.__early_stop or self.__early_stop == 0) and self.__num_params_atuali > 0):
            
            #Calcula todos valores das derividas parciais dos termos
            lista_derivadas_parciais = [self.__derivadas_parciais_custo(self.__lista_diff[i], self.__dict_Xy[self.__chaves[i]][0], self.__num_cols, self.__flag_filtro) for i in range(0, self.__num_Xy)]
            
            #Faz a média entre os gradientes de cada conjunto e remove as direções que são divergentes
            lista_sinais = [calcula_sinal(lista_derivadas_parciais[i]) for i in range(0, self.__num_Xy)]
            derivadas_parciais, sinais_iguais = filtra_derivadas_parciais(reduce(lambda x, y: calcula_soma(x, y), lista_derivadas_parciais), 
                                                                          reduce(lambda x, y: calcula_soma(x, y), lista_sinais), self.__num_Xy)
            
            #Atualiza os thetas com o passo alpha definido (de acordo com o otimizador SGD modificado - Anotações)            
            self.__norma_quadrada_gradiente, self.__thetas, self.__num_passos = calcula_passo_thetas(derivadas_parciais, self.__mse_referencia, 
                                                                                                     self.__alpha, self.__thetas, self.__num_passos)
            
            self.__lista_preds = [self.__calcula_funclinear(value[0], self.__num_cols, self.__thetas) for key, value in self.__dict_Xy.items()]
            self.__lista_diff = [calcula_diferenca(self.__lista_preds[i], self.__dict_Xy[self.__chaves[i]][1]) for i in range(0, self.__num_Xy)]
            
            #Calcula os MSE e adiciona na curva de Viés-Variância
            for i in range(0, self.__num_Xy):
                r2 = calcula_r2(self.__lista_diff[i], self.__lista_mse_baseline[i])
                self.__r2_temp[i] = r2
                self.__lista_curvas_r2[i].append(r2)
            self.__guarda_melhor_modelo()
            
            #Atualiza o alpha (learning rate) para ter valores menores quando estivermos perto da "saturação" -> ponto de convergência
            self.__alpha = min(self.__alpha_max, sum([abs(self.__lista_curvas_r2[i][-1] - self.__lista_curvas_r2[i][-2]) for i in range(0, self.__num_Xy)])/self.__num_Xy)
            
            ### Calculo de parâmetros para decidir alterações ou paradas na otimização ###
                
            self.__diff_ultimo_ganho = self.__num_passos - max(self.__lista_num_passos_melhor) 
            if(self.__diff_ultimo_ganho == 0):
                self.__diff_r2 = max([self.__lista_curvas_r2[i][-1] - self.__lista_curvas_r2[i][-2] for i in range(0, self.__num_Xy)])
            
            if(self.__filtro_atualizacao_thetas):
                self.__num_params_atuali, self.__flag_filtro, self.__norma_quadrada_gradiente, self.__diff_r2 = calcula_filtros_atualizacao_thetas(sinais_iguais,  
                                                                                                                                                   self.__norma_quadrada_gradiente, 
                                                                                                                                                   self.__diff_r2, 
                                                                                                                                                   self.__num_cols, 
                                                                                                                                                   self.__epsilon)
            
            ### Plots e Prints de report do processo de treinamento ###
            
            if(plot):
                display.clear_output(wait = True)
                #self.grafico_vies_variancia()
                with sns.axes_style("whitegrid"):
                    fig, axs = plt.subplots(1, 1, figsize = (8, 6))
                    axs.set_xlabel('Número de Passos')
                    axs.set_ylabel('R2 Ajustado')
                    for i in range(self.__num_Xy):
                        axs.plot(self.__lista_curvas_r2[i], color = paleta_cores[i % len(paleta_cores)], label = self.__chaves[i])
                        axs.scatter(self.__lista_num_passos_melhor[i], self.__lista_r2_melhor[i], color = paleta_cores[i % len(paleta_cores)])
                    axs.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.show()
            if(verbose >= 2):
                print(' / '.join(self.__r2_temp.astype(str)) + " (" + str(self.__num_passos) + ")")
            if(verbose >= 1):
                print("Número de Parâmetros sendo Atualizados: " + str(self.__num_params_atuali) + " (" + str(self.__num_passos) + ")")
        
        #Criação do Modelo médio com os melhores parâmetros para cada amostra
        self.__thetas_finais = calcula_divisao(reduce(lambda x, y: calcula_soma(x, y), self.__lista_thetas_melhor), self.__num_Xy)
        
        #Filtra termos da expansão com coeficiente nulo no modelo final (otimiza o predict do modelo)
        flag_coef_nao_nulo = self.__thetas_finais[1:] != 0
        self.__thetas_finais = self.__thetas_finais[np.append(True, flag_coef_nao_nulo)]
        self.__lista_termos_finais = self.__lista_termos[flag_coef_nao_nulo]
        self.__num_cols_finais = self.__lista_termos_finais.size
        self.__means_finais = self.__means[flag_coef_nao_nulo]
        self.__stds_finais = self.__stds[flag_coef_nao_nulo]
        
        self.__calcula_importancias()
    
    def predict(self, X):
        X = self.__checa_dados.checa_X_predict(X)
        X = X[:, self.__flag_vars_validas]
        y_pred = self.__calcula_funclinear_final(X)
        y_pred = self.__media_y + self.__desvio_y*y_pred
        return y_pred
    
    def grafico_vies_variancia(self, pos_ini = None, pos_fim = None, ymin = None, figsize = [8, 6]):        
        #Prepara os valores e vetores de plot
        if(pos_ini == None):
            pos_ini = 0
        if(pos_fim == None):
            pos_fim = self.__num_passos + 1
        curva_num_passos = np.arange(pos_ini, pos_fim)
        #Plota as curvas e o ponto de parada do treinamento pela validação
        paleta_cores = sns.color_palette("colorblind")
        with sns.axes_style("whitegrid"):
            fig, axs = plt.subplots(1, 1, figsize = figsize)
            for i in range(0, self.__num_Xy):
                curva_r2 = self.__lista_curvas_r2[i][pos_ini:pos_fim]
                axs.plot(curva_num_passos, curva_r2, color = paleta_cores[i % len(paleta_cores)], label = self.__chaves[i])
                axs.scatter(self.__lista_num_passos_melhor[i], self.__lista_r2_melhor[i], color = paleta_cores[i % len(paleta_cores)])
            if(ymin is not None):
                axs.set_ylim(bottom = ymin)
            axs.set_xlabel('Número de Passos')
            axs.set_ylabel('R2 Ajustado')
            axs.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.show()
            
    def __calcula_importancias(self):
        coef_abs = np.abs(self.__thetas_finais[1:])
        termos = self.__lista_termos_finais
        def traduz_termo(termo, nome_vars):
            return tuple((nome_vars[v[0]], v[1]) for v in termo)
        self.feature_names_terms_ = np.array([str(traduz_termo(v, self.__nome_vars)).replace("'","") for v in termos])
        self.feature_importances_terms_ = coef_abs/np.sum(coef_abs)
        def pares_feature_coef_abs(termo, coef):
            return [(v[0], coef) for v in termo]
        lista_pesos = []
        for i in range(0, coef_abs.size):
            lista_pesos.extend(pares_feature_coef_abs(termos[i], coef_abs[i]))
        lista_agrupada = [(key, sum(map(itemgetter(1), ele))) for key, ele in groupby(sorted(lista_pesos, key = itemgetter(0)), key = itemgetter(0))]
        self.feature_names_ = np.array([self.__nome_vars[x[0]] for x in lista_agrupada])
        self.feature_importances_ = np.array([x[1] for x in lista_agrupada])
        self.feature_importances_ = self.feature_importances_/np.sum(self.feature_importances_)
        
    def grafico_importancias(self, num_vars = None, figsize = [8, 6]):        
        if(num_vars == None):
            num_vars = self.feature_names_.size
        vars_nomes = self.feature_names_
        vars_imp = self.feature_importances_
        inds_ordenado = np.argsort(vars_imp)[::-1]
        vars_nomes = vars_nomes[inds_ordenado[:num_vars]]
        vars_imp = vars_imp[inds_ordenado[:num_vars]]
        paleta_cores = sns.color_palette("colorblind")
        with sns.axes_style("whitegrid"):
            fig, axs = plt.subplots(1, 1, figsize = figsize)
            axs.barh(vars_nomes[::-1], vars_imp[::-1], color = paleta_cores[0])
            axs.set_xlabel('Importância')
            axs.set_ylabel('Variável')
            plt.show()
            
    def grafico_importancias_termos(self, num_vars = None, figsize = [8, 6]):        
        if(num_vars == None):
            num_vars = self.feature_names_terms_.size
        vars_nomes = self.feature_names_terms_
        vars_imp = self.feature_importances_terms_
        inds_ordenado = np.argsort(vars_imp)[::-1]
        vars_nomes = vars_nomes[inds_ordenado[:num_vars]]
        vars_imp = vars_imp[inds_ordenado[:num_vars]]
        paleta_cores = sns.color_palette("colorblind")
        with sns.axes_style("whitegrid"):
            fig, axs = plt.subplots(1, 1, figsize = figsize)
            axs.barh(vars_nomes[::-1], vars_imp[::-1], color = paleta_cores[0])
            axs.set_xlabel('Importância')
            axs.set_ylabel('Variável')
            plt.show()
    
##############################

##############################
            
class SeriesMultiGradientClassifier:
      
    def __guarda_melhor_modelo(self):
        flag_alterar_valor = self.__lista_coef_logloss_melhor < self.__coef_logloss_temp
        self.__lista_coef_logloss_melhor[flag_alterar_valor] = self.__coef_logloss_temp[flag_alterar_valor]
        self.__lista_num_passos_melhor[flag_alterar_valor] = cria_repeticao(self.__num_passos, calcula_somatorio(flag_alterar_valor))
        for i in range(0, self.__num_Xy):
            if(flag_alterar_valor[i]):
                self.__lista_thetas_melhor[i] = self.__thetas
    
    def __derivadas_parciais_custo(self, diff, X, num_cols, flag_filtro):
        valores = cria_zeros(num_cols + 1)
        if(flag_filtro[0] == True):
            valores[0] = calcula_media(diff)
        for i in range(0, num_cols):
            if(flag_filtro[i+1] == True):
                valores[i+1] = derivada_parcial_custo_normalizado(diff, calcula_termo_serie(X, self.__lista_termos[i]), self.__means[i], self.__stds[i])
        return valores
    
    def __calcula_logistica(self, X, num_cols, thetas):
        valores = cria_repeticao(thetas[0], X.shape[0])
        for i in range(0, num_cols):
            valores = valores + feature_normalizada_dot_theta(calcula_termo_serie(X, self.__lista_termos[i]), self.__means[i], self.__stds[i], thetas[i+1])
        return func_logistica(valores)
        
    def __calcula_logistica_final(self, X):
        valores = cria_repeticao(self.__thetas_finais[0], X.shape[0])
        for i in range(0, self.__num_cols_finais):
            valores = valores + feature_normalizada_dot_theta(calcula_termo_serie(X, self.__lista_termos_finais[i]), 
                                                              self.__means_finais[i], self.__stds_finais[i], self.__thetas_finais[i+1])
        return func_logistica(valores)
    
    def setup_datasets(self, dict_Xy, laurent = False, ordem = 1, apenas_interacoes = False):
        #Equivalente a construção da rede (constrói os parâmetros que serão fitados)
        
        #Configurações de quantidade de parâmetros da função (equivalente aos neurônios e camadas)
        self.__laurent = laurent
        self.__ordem = ordem
        self.__apenas_interacoes = apenas_interacoes
        self.__series = None
        
        print('Checando Datasets')
        
        #Checa se os datasets estão adequados e extrai informações deles
        self.__checa_dados = ChecaMultiDatasets(dict_Xy)
        self.__nome_vars = self.__checa_dados.nome_vars
        self.__dict_Xy = self.__checa_dados.dict_Xy
        self.__num_linhas = self.__checa_dados.num_linhas
        self.__num_cols = self.__checa_dados.num_cols
        self.__num_Xy = self.__checa_dados.num_Xy
        self.__chaves = self.__checa_dados.chaves
        self.__features_binarias = self.__checa_dados.features_binarias           
        
        print('Filtrando Features sem Variância')
        
        #Filtro inicial de features sem variância (em algum dataset)
        lista_flag_vars_validas = []
        for key, value in self.__dict_Xy.items():
            lista_flag_vars_validas.append(calcula_flag_com_variancia(value[0], self.__num_cols))
        self.__flag_vars_validas = reduce(lambda x, y: calcula_produto(x, y), lista_flag_vars_validas)
        
        self.__num_cols = calcula_somatorio(self.__flag_vars_validas) 
        self.__nome_vars = self.__nome_vars[self.__flag_vars_validas]
        for key, value in self.__dict_Xy.items():
            self.__dict_Xy[key] = (value[0][:, self.__flag_vars_validas], value[1])
        
        print('Criando a Expansão em Série')
        
        #Expansão em série
        self.__series = TaylorLaurentExpansion(self.__laurent, self.__ordem, self.__apenas_interacoes, self.__num_cols)
        if(len(self.__features_binarias) > 0):
            self.__series.filtro_binario(self.__features_binarias)
        self.__num_cols = self.__series.numero_termos_expansao()
        if(self.__ordem == 1):
            termos_temp = self.__series.lista_termos()
            self.__lista_termos = np.empty(self.__num_cols, dtype = object)
            for i in range(self.__num_cols):
                self.__lista_termos[i] = termos_temp[i]
        else:
            self.__lista_termos = np.array(self.__series.lista_termos(), dtype = object)
        
        print("Número de Parâmetros da Expansão: " + str(self.__num_cols + 1))
        
        print('Calculando as Normalizações das Features')
        
        #Faz a normalização (e filtra colunas com desvio padrão zero)
        #Facilita calculo de Inv. de Matriz e também facilita a Descida do Gradiente
        lista_medias = []
        lista_variancias = []
        for key, value in self.__dict_Xy.items():
            medias = []
            variancias = []
            for j in range(self.__num_cols):
                vetor = calcula_termo_serie(value[0], self.__lista_termos[j])
                media, variancia = calcula_media_variancia(vetor)
                medias.append(media)
                variancias.append(variancia)
            lista_medias.append(np.array(medias))
            lista_variancias.append(np.array(variancias))
        self.__means = calcula_divisao(reduce(lambda x, y: calcula_soma(x, y), lista_medias), self.__num_Xy)
        self.__stds = calcula_divisao_e_raiz(reduce(lambda x, y: calcula_soma(x, y), lista_variancias), self.__num_Xy)
        
        #Termos efetivos (só os que tem desvio padrão não nulo)
        flag_termos_validos = reduce(lambda x, y: ambos_positivos(x, y), lista_variancias)
        self.__lista_termos = self.__lista_termos[flag_termos_validos]
        self.__means = self.__means[flag_termos_validos]
        self.__stds = self.__stds[flag_termos_validos]
        self.__num_cols = self.__lista_termos.size
        
        print("Número de Parâmetros: " + str(self.__num_cols + 1))
        
    def setup_optimizer(self, alpha = 0.01, epsilon = 1e-6, early_stop = 10, filtro_atualizacao_thetas = True):
    
        #Parâmetros da busca pelos valores dos parâmetros (equivalente ao optimizer)
        self.__alpha_max = alpha #-> Quanto queremos que a função de custo (R2 Ajust) mude a cada passo
        self.__alpha = alpha #-> Quanto queremos que a função de custo (R2 Ajust) mude a cada passo (esse pode ser atualizado no meio da otimização)
        self.__epsilon = epsilon #-> Quanto queremos de melhora mínima da função de custo (R2 Ajust) para continuar os passos
        self.__early_stop = early_stop #-> Quantos passos aceitamos com queda na função de custo (R2 Ajust) sem parar o treinamento
        self.__filtro_atualizacao_thetas = filtro_atualizacao_thetas
        
        self.__max_passos = 0 #Quantos passos já demos na busca dos parâmetros (equivalente as épocas)
        
        print('Criando o Baseline Constante')
        
        #(Aqui não pode ter nulos mesmo!!)
        lista_medias_y = []
        for key, value in self.__dict_Xy.items():
            lista_medias_y.append(calcula_media(value[1]))
        self.__media_y = sum(lista_medias_y)/self.__num_Xy
        
        #Faz o modelo baseline: só o termo theta_0 -> cte!!! (prob média)
        self.__num_passos = 0 #Número de passos na busca de parâmetros
        theta_0 = math.log(self.__media_y/(1 - self.__media_y)) #Dedução da função logística
        self.__thetas = np.array([theta_0]) #Valor dos thetas do regressor
        
        self.__lista_diff = [calcula_diferenca(self.__media_y, value[1])  for key, value in self.__dict_Xy.items()]
        
        #Salva o loss do modelo baseline (só constante)
        self.__lista_logloss_baseline = [calcula_logloss_baseline(self.__dict_Xy[self.__chaves[i]][1], self.__media_y) for i in range(0, self.__num_Xy)]
        self.__logloss_referencia = sum(self.__lista_logloss_baseline)/self.__num_Xy
            
        #####
        ##### Algoritmo de busca no espaço de soluções usando a Descida do Gradiente #####
        #####
        
        print('Inicializando os Parâmetros para Fitting')
        
        #Inicializamos as variaveis para criar a curva viés-variância
        self.__lista_curvas_coef_logloss = [[0] for i in range(0, self.__num_Xy)]
        
        #Começa todos os thetas zerados (com excessão do theta_0 que é o baseline)
        self.__thetas = np.append(self.__thetas, cria_zeros(self.__num_cols)) 
        
        self.__lista_coef_logloss_melhor = cria_zeros(self.__num_Xy)
        self.__lista_num_passos_melhor = cria_zeros(self.__num_Xy)
        self.__lista_thetas_melhor = [self.__thetas for i in range(0, self.__num_Xy)]
        self.__coef_logloss_temp = cria_zeros(self.__num_Xy) #Vetor de valores temporários para atualizar as listas acima
        
        #Inicia parametros de parada de treinamento
        self.__diff_ultimo_ganho = 0
        self.__diff_coef_logloss = self.__epsilon
        self.__norma_quadrada_gradiente = 1
        self.__flag_filtro = cria_repeticao(True, self.__num_cols + 1)
        self.__num_params_atuali = self.__num_cols + 1
    
    def fit(self, num_passos, verbose = 2, plot = True):
        if(num_passos >= 0):
            self.__max_passos = self.__max_passos + num_passos
        
        if(plot):
            display.clear_output(wait = True)
            #self.grafico_vies_variancia()
            paleta_cores = sns.color_palette("colorblind")
            with sns.axes_style("whitegrid"):
                fig, axs = plt.subplots(1, 1, figsize = (8, 6))
                axs.set_xlabel('Número de Passos')
                axs.set_ylabel('Coeficiente LogLoss')
                for i in range(self.__num_Xy):
                    axs.plot(self.__lista_curvas_coef_logloss[i], color = paleta_cores[i % len(paleta_cores)], label = self.__chaves[i])
                    axs.scatter(self.__lista_num_passos_melhor[i], self.__lista_coef_logloss_melhor[i], color = paleta_cores[i % len(paleta_cores)])
                axs.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.show()
        if(verbose >= 1):
            print("Número de Parâmetros sendo Atualizados: " + str(self.__num_params_atuali) + " (" + str(self.__num_passos) + ")")
        
        #Loop do processo incremental de passos
        while(self.__num_passos < self.__max_passos and self.__norma_quadrada_gradiente > 0 and self.__diff_coef_logloss >= self.__epsilon and (self.__diff_ultimo_ganho < self.__early_stop or self.__early_stop == 0) and self.__num_params_atuali > 0):
            
            #Calcula todos valores das derividas parciais dos termos
            lista_derivadas_parciais = [self.__derivadas_parciais_custo(self.__lista_diff[i], self.__dict_Xy[self.__chaves[i]][0], self.__num_cols, self.__flag_filtro) for i in range(0, self.__num_Xy)]
            
            #Faz a média entre os gradientes de cada conjunto e remove as direções que são divergentes
            lista_sinais = [calcula_sinal(lista_derivadas_parciais[i]) for i in range(0, self.__num_Xy)]
            derivadas_parciais, sinais_iguais = filtra_derivadas_parciais(reduce(lambda x, y: calcula_soma(x, y), lista_derivadas_parciais), 
                                                                          reduce(lambda x, y: calcula_soma(x, y), lista_sinais), self.__num_Xy)
            
            #Atualiza os thetas com o passo alpha definido (de acordo com o otimizador SGD modificado - Anotações)            
            self.__norma_quadrada_gradiente, self.__thetas, self.__num_passos = calcula_passo_thetas(derivadas_parciais, self.__logloss_referencia, 
                                                                                                     self.__alpha, self.__thetas, self.__num_passos)
            
            self.__lista_probs = [self.__calcula_logistica(value[0], self.__num_cols, self.__thetas) for key, value in self.__dict_Xy.items()]
            self.__lista_diff = [calcula_diferenca(self.__lista_probs[i], self.__dict_Xy[self.__chaves[i]][1]) for i in range(0, self.__num_Xy)]
            
            #Calcula os Coef Logloss e adiciona na curva de Viés-Variância
            for i in range(0, self.__num_Xy):
                coef_logloss = calcula_coef_logloss(self.__dict_Xy[self.__chaves[i]][1], self.__lista_probs[i], self.__lista_logloss_baseline[i])
                self.__coef_logloss_temp[i] = coef_logloss
                self.__lista_curvas_coef_logloss[i].append(coef_logloss)
            self.__guarda_melhor_modelo()
            
            #Atualiza o alpha (learning rate) para ter valores menores quando estivermos perto da "saturação" -> ponto de convergência
            self.__alpha = min(self.__alpha_max, sum([abs(self.__lista_curvas_coef_logloss[i][-1] - self.__lista_curvas_coef_logloss[i][-2]) for i in range(0, self.__num_Xy)])/self.__num_Xy)
            
            ### Calculo de parâmetros para decidir alterações ou paradas na otimização ###
            
            self.__diff_ultimo_ganho = self.__num_passos - max(self.__lista_num_passos_melhor) 
            if(self.__diff_ultimo_ganho == 0):
                self.__diff_coef_logloss = max([self.__lista_curvas_coef_logloss[i][-1] - self.__lista_curvas_coef_logloss[i][-2] for i in range(0, self.__num_Xy)])
            
            if(self.__filtro_atualizacao_thetas):
                self.__num_params_atuali, self.__flag_filtro, self.__norma_quadrada_gradiente, self.__diff_coef_logloss = calcula_filtros_atualizacao_thetas(sinais_iguais,  
                                                                                                                                                   self.__norma_quadrada_gradiente, 
                                                                                                                                                   self.__diff_coef_logloss, 
                                                                                                                                                   self.__num_cols, 
                                                                                                                                                   self.__epsilon)
            
            ### Plots e Prints de report do processo de treinamento ###
            
            if(plot):
                display.clear_output(wait = True)
                #self.grafico_vies_variancia()
                with sns.axes_style("whitegrid"):
                    fig, axs = plt.subplots(1, 1, figsize = (8, 6))
                    axs.set_xlabel('Número de Passos')
                    axs.set_ylabel('Coeficiente LogLoss')
                    for i in range(self.__num_Xy):
                        axs.plot(self.__lista_curvas_coef_logloss[i], color = paleta_cores[i % len(paleta_cores)], label = self.__chaves[i])
                        axs.scatter(self.__lista_num_passos_melhor[i], self.__lista_coef_logloss_melhor[i], color = paleta_cores[i % len(paleta_cores)])
                    axs.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.show()
            if(verbose >= 2):
                print(' / '.join(self.__coef_logloss_temp.astype(str)) + " (" + str(self.__num_passos) + ")")
            if(verbose >= 1):
                print("Número de Parâmetros sendo Atualizados: " + str(self.__num_params_atuali) + " (" + str(self.__num_passos) + ")")
        
        #Criação do Modelo médio com os melhores parâmetros para cada amostra
        self.__thetas_finais = calcula_divisao(reduce(lambda x, y: calcula_soma(x, y), self.__lista_thetas_melhor), self.__num_Xy)
        
        #Filtra termos da expansão com coeficiente nulo no modelo final (otimiza o predict do modelo)
        flag_coef_nao_nulo = self.__thetas_finais[1:] != 0
        self.__thetas_finais = self.__thetas_finais[np.append(True, flag_coef_nao_nulo)]
        self.__lista_termos_finais = self.__lista_termos[flag_coef_nao_nulo]
        self.__num_cols_finais = self.__lista_termos_finais.size
        self.__means_finais = self.__means[flag_coef_nao_nulo]
        self.__stds_finais = self.__stds[flag_coef_nao_nulo]
        
        self.__calcula_importancias()
        
    def predict_proba(self, X):
        X = self.__checa_dados.checa_X_predict(X)
        X = X[:, self.__flag_vars_validas]
        y_pred = self.__calcula_logistica_final(X)
        y_prob = np.dstack((1 - y_pred,y_pred))[0]
        return y_prob
        
    def predict(self, X):
        y_prob = self.predict_proba(X)
        y_pred = (y_prob[:, 1] >= self.__media_y).astype(int)
        return y_pred
    
    def grafico_vies_variancia(self, pos_ini = None, pos_fim = None, ymin = None, figsize = [8, 6]):        
        #Prepara os valores e vetores de plot
        if(pos_ini == None):
            pos_ini = 0
        if(pos_fim == None):
            pos_fim = self.__num_passos + 1
        curva_num_passos = np.arange(pos_ini, pos_fim)
        #Plota as curvas e o ponto de parada do treinamento pela validação
        paleta_cores = sns.color_palette("colorblind")
        with sns.axes_style("whitegrid"):
            fig, axs = plt.subplots(1, 1, figsize = figsize)
            for i in range(0, self.__num_Xy):
                curva_coef_logloss = self.__lista_curvas_coef_logloss[i][pos_ini:pos_fim]
                axs.plot(curva_num_passos, curva_coef_logloss, color = paleta_cores[i % len(paleta_cores)], label = self.__chaves[i])
                axs.scatter(self.__lista_num_passos_melhor[i], self.__lista_coef_logloss_melhor[i], color = paleta_cores[i % len(paleta_cores)])
            if(ymin is not None):
                axs.set_ylim(bottom = ymin)
            axs.set_xlabel('Número de Passos')
            axs.set_ylabel('Coeficiente LogLoss')
            axs.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.show()
            
    def __calcula_importancias(self):
        coef_abs = np.abs(self.__thetas_finais[1:])
        termos = self.__lista_termos_finais
        def traduz_termo(termo, nome_vars):
            return tuple((nome_vars[v[0]], v[1]) for v in termo)
        self.feature_names_terms_ = np.array([str(traduz_termo(v, self.__nome_vars)).replace("'","") for v in termos])
        self.feature_importances_terms_ = coef_abs/np.sum(coef_abs)
        def pares_feature_coef_abs(termo, coef):
            return [(v[0], coef) for v in termo]
        lista_pesos = []
        for i in range(0, coef_abs.size):
            lista_pesos.extend(pares_feature_coef_abs(termos[i], coef_abs[i]))
        lista_agrupada = [(key, sum(map(itemgetter(1), ele))) for key, ele in groupby(sorted(lista_pesos, key = itemgetter(0)), key = itemgetter(0))]
        self.feature_names_ = np.array([self.__nome_vars[x[0]] for x in lista_agrupada])
        self.feature_importances_ = np.array([x[1] for x in lista_agrupada])
        self.feature_importances_ = self.feature_importances_/np.sum(self.feature_importances_)
        
    def grafico_importancias(self, num_vars = None, figsize = [8, 6]):        
        if(num_vars == None):
            num_vars = self.feature_names_.size
        vars_nomes = self.feature_names_
        vars_imp = self.feature_importances_
        inds_ordenado = np.argsort(vars_imp)[::-1]
        vars_nomes = vars_nomes[inds_ordenado[:num_vars]]
        vars_imp = vars_imp[inds_ordenado[:num_vars]]
        paleta_cores = sns.color_palette("colorblind")
        with sns.axes_style("whitegrid"):
            fig, axs = plt.subplots(1, 1, figsize = figsize)
            axs.barh(vars_nomes[::-1], vars_imp[::-1], color = paleta_cores[0])
            axs.set_xlabel('Importância')
            axs.set_ylabel('Variável')
            plt.show()
            
    def grafico_importancias_termos(self, num_vars = None, figsize = [8, 6]):        
        if(num_vars == None):
            num_vars = self.feature_names_terms_.size
        vars_nomes = self.feature_names_terms_
        vars_imp = self.feature_importances_terms_
        inds_ordenado = np.argsort(vars_imp)[::-1]
        vars_nomes = vars_nomes[inds_ordenado[:num_vars]]
        vars_imp = vars_imp[inds_ordenado[:num_vars]]
        paleta_cores = sns.color_palette("colorblind")
        with sns.axes_style("whitegrid"):
            fig, axs = plt.subplots(1, 1, figsize = figsize)
            axs.barh(vars_nomes[::-1], vars_imp[::-1], color = paleta_cores[0])
            axs.set_xlabel('Importância')
            axs.set_ylabel('Variável')
            plt.show()