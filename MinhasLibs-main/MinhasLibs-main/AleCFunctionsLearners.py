from numba import jit_module, jit, prange
import numpy as np
from functools import reduce

###############################
#Para o Cálculo da Expansão em Série
###############################

def calcula_termo_serie(X, termo):
    #valores = np.power(X[:, termo[0][0]], termo[0][1])
    #for i in prange(1, len(termo)):
    #    valores = valores * np.power(X[:, termo[i][0]], termo[i][1])
    #return valores
    return reduce(lambda x, y: x * y, [np.power(X[:, termo[i][0]], termo[i][1]) for i in prange(0, len(termo))])

###############################
#Para operações gerais do algoritmo
###############################

def calcula_soma(a, b):
    return a + b
   
def calcula_diferenca(a, b):
    return a - b

def calcula_produto(a, b):
    return a * b

def calcula_divisao(a, b):
    return a/b
    
def ambos_positivos(a, b):
    return (a > 0) * (b > 0)
    
def calcula_divisao_e_raiz(vetor, divisor):
    return np.sqrt(vetor/divisor)
    
def calcula_media(vetor):
    return np.mean(vetor)
    
def calcula_somatorio(vetor):
    return np.sum(vetor)

def calcula_sinal(vetor):
    return np.sign(vetor)

def cria_zeros(num_repeticao):
    return np.zeros(num_repeticao)

def cria_repeticao(valor, num_repeticao):
    return np.repeat(valor, num_repeticao)

def calcula_flag_com_variancia(X, num_cols):
    flags = []
    for c in prange(num_cols):
        vetor = X[:, c]
        vetor_nn = vetor[~np.isnan(vetor)]
        media = np.mean(vetor_nn)
        vec_aux = vetor_nn - media
        variancia = np.dot(vec_aux, vec_aux)/vetor_nn.size
        flags.append(variancia > 0)
    return np.array(flags)
    
def calcula_media_variancia(vetor):
    vetor_nn = vetor[(~np.isnan(vetor))&(vetor < np.inf)&(vetor > -np.inf)]
    if(vetor_nn.size > 0):
        media = np.mean(vetor_nn)
        vec_aux = vetor_nn - media
        variancia = np.dot(vec_aux, vec_aux)/vetor_nn.size  
        return media, variancia
    else:
        return 0, 0

def normaliza_vetor(vetor, mean, std):
    return (vetor - mean)/std

def derivada_parcial_custo_normalizado(diff, vetor, mean, std):
    vetor_res = diff*(vetor - mean)/std
    vetor_res = vetor_res[(~np.isnan(vetor_res))&(vetor_res < np.inf)&(vetor_res > -np.inf)]
    return np.sum(vetor_res)/vetor.size

def filtra_derivadas_parciais(soma_derivadas_parciais, vetor_soma_sinais, num_Xy):
    sinais_iguais = np.abs(vetor_soma_sinais) == num_Xy
    soma_derivadas_parciais[~sinais_iguais] = 0
    return soma_derivadas_parciais/num_Xy, sinais_iguais
    
def calcula_passo_thetas(derivadas_parciais, metrica_ref, alpha, thetas, num_passos):
    norma_quadrada_gradiente = np.sum(np.power(derivadas_parciais, 2))
    if(norma_quadrada_gradiente > 0):
        cte = metrica_ref/norma_quadrada_gradiente
        thetas = thetas - alpha*cte*derivadas_parciais
    num_passos = num_passos + 1
    return norma_quadrada_gradiente, thetas, num_passos

def feature_normalizada_dot_theta(vetor, mean, std, theta):
    vetor_res = (vetor - mean)/std
    vetor_res[np.isnan(vetor_res)|(vetor_res == np.inf)|(vetor_res == -np.inf)] = 0
    return vetor_res*theta

def calcula_filtros_atualizacao_thetas(sinais_iguais, norma_quadrada_gradiente, diff_metrica, num_cols, epsilon):
    num_params_atuali = np.sum(sinais_iguais)
    if(num_params_atuali != num_cols + 1):
        if(num_params_atuali == 0):
            num_params_atuali = num_cols + 1
            sinais_iguais = np.repeat(True, num_cols + 1)
        if(norma_quadrada_gradiente == 0):
            norma_quadrada_gradiente = 1
        if(diff_metrica < epsilon):
            diff_metrica = epsilon
    return num_params_atuali, sinais_iguais, norma_quadrada_gradiente, diff_metrica

###############################
#Específicas para o Classificador
###############################

def func_logistica(valores):
    return 1/(1 + np.exp(-valores))

def calcula_logloss_baseline(y, y_h):
    return -1*np.mean(np.where(y == 1, np.log(y_h), np.log(1 - y_h)))

#def calcula_logloss(y, y_h):
#    return -1*np.mean(np.where(y == 1, np.log(y_h), np.log(1 - y_h)))

def calcula_coef_logloss(y, y_h, logloss_baseline):
    logloss = -1*np.mean(np.where(y == 1, np.log(y_h), np.log(1 - y_h)))
    return 1 - logloss/logloss_baseline
    
###############################
#Específicas para o Regressor
###############################

def calcula_mse_baseline(diff):
    return np.mean(np.power(diff, 2))

#def calcula_mse(diff):
#    return np.mean(np.power(diff, 2))

def calcula_r2(diff, mse_baseline):
    mse = np.mean(np.power(diff, 2))
    return 1 - mse/mse_baseline
    
jit_module(nopython = True, 
           #cache = True
           )