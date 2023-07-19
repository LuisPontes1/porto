from numba import jit_module, jit, prange
import numpy as np
from scipy.stats import norm

######################################
##Operações Básicas
######################################
   
def soma_vetor(vetor):
    return np.sum(vetor)
    
def soma_acumulada_vetor(vetor):
    return np.cumsum(vetor)
    
def diferenca_vetores(vetor1, vetor2):
    return vetor1 - vetor2
    
def divisao_vetores(vetor1, vetor2):
    return vetor1/vetor2

######################################
##Operações de Métricas de Classificação
######################################

#Estima a área abaixo da curva por Soma de Riemann
def area(x,y):
    dx = np.diff(x)
    h = (y[:-1] + y[1:])/2
    A = np.sum(h*dx)
    return A  
    
def argmax_vetor(vetor):
    return np.argmax(vetor)
    
def entropia_shannon(vetor_p1):
    entropia = []
    for p1 in vetor_p1:
        p0 = 1 - p1
        if p0 == 0 or p1 == 0:
            entropia.append(0)
        else:
            entropia.append(-p0*np.log2(p0) - p1*np.log2(p1))
    return np.array(entropia)

def calcula_curva_ig(entropia_parcial, qtds_acum, entropia_parcial_c, qtds_acum_c, qtd_tot, entropia_ini):
    entropia = (entropia_parcial*qtds_acum + entropia_parcial_c*qtds_acum_c)/qtd_tot
    #Coloca o valor [-1] que removemos no começo do calcula da entropia
    entropia = np.append(entropia, entropia_ini)
    return (entropia_ini - entropia)/entropia_ini

def calcula_vetor_entropia_parcial(vetor_entropia, entropia_aux, entropia_parcial_c, qtd_acum_c, entropia_parcial_r, qtd_resto, qtd_tot):
    entropia = entropia_aux + (entropia_parcial_c*qtd_acum_c + entropia_parcial_r*qtd_resto)/qtd_tot
    return np.append(vetor_entropia, entropia)
    
def normaliza_vetor_entropia(vetor_entropia, entropia_ini):
    return (entropia_ini - vetor_entropia)/entropia_ini
    
def logloss(y, y_h):
    return -1*np.mean(np.where(y == 1, np.log(y_h), np.log(1 - y_h)))
    
def calcula_media(y):
    return np.mean(y)

######################################    
##Operações de Métricas de Regressão
######################################

def calcula_mae(diff):
    return np.mean(np.abs(diff))
    
def calcula_mse(diff):
    return np.mean(np.power(diff, 2))
    
def checa_ordenacao(pares_indices, pares_valores, i):
    ind_a = pares_indices[i]
    ind_b = pares_indices[i+1]
    a = pares_valores[ind_a]
    b = pares_valores[ind_b]
    return (a[0] > a[1] and b[0] > b[1]) or (a[0] < a[1] and b[0] < b[1]) or (a[0] == a[1] and b[0] == b[1])

######################################    
##Operações de Métricas de Distribuições de Probabilidade
######################################

def calcula_distribuicao_acumulada_pontos(y_pontos, y):
    return np.array([np.sum(y_pontos <= v) for v in y])/y_pontos.size

def calcula_ks(y_acum1, y_acum2):
    return np.max(np.abs(y_acum1 - y_acum2))

def calcula_desvio(vetor):
    return np.std(vetor)

def conta_prob_bin(inf_bin, sup_bin, valores_min, valores_max, probs):
    prob_bin = 0
    for i in prange(probs.size):
        if(inf_bin <= valores_min[i] and sup_bin >= valores_max[i]):
            prob_bin = prob_bin + probs[i]
        elif(inf_bin <= valores_min[i] and sup_bin < valores_max[i] and sup_bin > valores_min[i]):
            prob_bin = prob_bin + probs[i]*(sup_bin - valores_min[i])/(valores_max[i] - valores_min[i])
        elif(inf_bin > valores_min[i] and sup_bin >= valores_max[i] and inf_bin < valores_max[i]):
            prob_bin = prob_bin + probs[i]*(valores_max[i] - inf_bin)/(valores_max[i] - valores_min[i])
        elif(inf_bin > valores_min[i] and sup_bin < valores_max[i]):
            prob_bin = prob_bin + probs[i]*(sup_bin - inf_bin)/(valores_max[i] - valores_min[i])
    return prob_bin

######################################
##Operações de Agrupamento e Ordenação
######################################

def qtd_unicos(vetor):
    qtd_unicos = 0
    v = np.nan
    for u in np.sort(vetor):
        if u != v:
            qtd_unicos = qtd_unicos + 1
            v = u
    return qtd_unicos

def unicos_qtds(vetor):
    valores = []
    qtds = []
    qtd_unicos = 0
    v = np.nan
    qtd = 0
    for u in np.sort(vetor):
        if u == v:
            qtd = qtd + 1
        else:
            valores.append(v)
            qtds.append(qtd)
            qtd_unicos = qtd_unicos + 1
            v = u
            qtd = 1
    valores.append(v)
    qtds.append(qtd)
    return np.array(valores[1:]), np.array(qtds[1:]), qtd_unicos

def indices_qtds(vetor):
    inds_sorted = np.argsort(vetor)
    vetor_sorted = vetor[inds_sorted]
    qtds = []
    qtd_unicos = 0
    v = np.nan
    qtd = 0
    for u in vetor_sorted:
        if u == v:
            qtd = qtd + 1
        else:
            qtds.append(qtd)
            qtd_unicos = qtd_unicos + 1
            v = u
            qtd = 1
    qtds.append(qtd)
    return inds_sorted, np.cumsum(np.array(qtds[:-1])), np.array(qtds[1:]), qtd_unicos
    
def indices_unicos_qtds(vetor):
    inds_sorted = np.argsort(vetor)
    vetor_sorted = vetor[inds_sorted]
    valores = []
    qtds = []
    qtd_unicos = 0
    v = np.nan
    qtd = 0
    for u in vetor_sorted:
        if u == v:
            qtd = qtd + 1
        else:
            valores.append(v)
            qtds.append(qtd)
            qtd_unicos = qtd_unicos + 1
            v = u
            qtd = 1
    valores.append(v)
    qtds.append(qtd)
    return inds_sorted, np.array(valores[1:]), np.cumsum(np.array(qtds[:-1])), np.array(qtds[1:]), qtd_unicos

###############################
#Para Calculo de Metricas Condicionais (Classificação e Regressão)
###############################

#Calcula o Ganho de Informação e a Razão de Ganho (Normalizados pela Entropia Inicial)
def calcula_ig_rg_condicional(qtds1, qtds, probs1, qtd_nao_nulo):
    p_ini = np.sum(qtds1)/qtd_nao_nulo
    pc_ini = 1 - p_ini
    if(p_ini == 0 or pc_ini == 0):
        entropia_ini = 0
    else:
        entropia_ini = -p_ini*np.log2(p_ini) - pc_ini*np.log2(pc_ini)
    
    entropias_parciais = []
    for x in probs1:
        xc = 1 - x
        if(x == 0 or xc == 0):
            entropias_parciais.append(0)
        else:
            entropias_parciais.append(-x*np.log2(x) - xc*np.log2(xc))
    entropias_parciais = np.array(entropias_parciais)
    entropia = (np.sum(entropias_parciais*qtds)/qtd_nao_nulo)
    
    ig = (entropia_ini - entropia)/entropia_ini
    
    if(qtds.size > 1):
        fracs = qtds/qtd_nao_nulo
        entropia_divisao = -np.sum(fracs*np.log2(fracs))
        rg = ig/entropia_divisao
    else:
        rg = 0
    
    return ig, rg
    
#Calcula o Ganho de Informação e a Razão de Ganho (Normalizados pela Entropia Inicial)
def calcula_r2_ratio_r2_condicional(y, vars_cond, qtds, qtd_nao_nulo, valores, vars_cond_feature):
    var_ini = np.var(y)
    r2 = 1 - np.sum(qtds*vars_cond)/(qtd_nao_nulo*var_ini)
    
    var_ini_feature = np.var(valores)
    r2_feature = 1 - np.sum(qtds*vars_cond_feature)/(qtd_nao_nulo*var_ini_feature)
    
    if(r2_feature > 0):
        ratio_r2 = r2/r2_feature
    else:
        ratio_r2 = 0
    
    return r2, ratio_r2

###############################
#Para o CortaIntervalosQuasiUniforme
###############################

def pontos_corte(qtds, qtd_unicos, num_div):
    passo = int(qtd_unicos/num_div)
    pts_corte = [i*passo for i in prange(num_div+1)]
    pts_corte[-1] = qtd_unicos
    qtds_corte = [np.sum(qtds[pts_corte[i]:pts_corte[i+1]]) for i in prange(num_div)]
    pts_corte = [p - 1 for p in pts_corte[1:]]
    return np.array(pts_corte), np.array(qtds_corte)

def minimiza_desvio_padrao(pts_corte, qtds_corte, qtds, qtd_unicos):
    fim = pts_corte.size - 1
    prefim = fim - 1
    permutado = True
    while(permutado):
        permutado = False
        for i in prange(fim):
            a = qtds_corte[prefim-i]
            b = qtds_corte[fim-i]
            p = pts_corte[prefim-i]
            qe = qtds[p]
            qd = qtds[p+1]
            if(a - b > qe and a > qe):
                qtds_corte[fim-i] = b + qe
                qtds_corte[prefim-i] = a - qe
                pts_corte[prefim-i] = p - 1
                permutado = True
            elif(b - a > qd and b > qd):
                qtds_corte[fim-i] = b - qd
                qtds_corte[prefim-i] = a + qd
                pts_corte[prefim-i] = p + 1
                permutado = True
    #pts_corte = pts_corte[qtds_corte != 0]
    #qtds_corte = qtds_corte[qtds_corte != 0]
    return pts_corte, qtds_corte

def calcula_valores_corte(valores, pts_corte):
    valores_intermed = (valores[pts_corte[:-1]] + valores[pts_corte[:-1]+1])/2
    valores_min = np.append(valores[0], valores_intermed)
    valores_max =  np.append(valores_intermed, valores[pts_corte[-1]])
    return valores_min, valores_max

def calcula_min_diff(vetor):
    return np.min(np.diff(vetor))

def calcula_pontos_medios(valores_min, valores_max):
    return (valores_min + valores_max)/2

def calcula_pares_minmax(valores_min, valores_max):
    return np.array(list(zip(valores_min, valores_max)))

def conta_qtds_bin(inf_bin, sup_bin, valores_min, valores_max, qtds):
    qtd_bin = 0
    for i in prange(qtds.size):
        if(inf_bin <= valores_min[i] and sup_bin >= valores_max[i]):
            qtd_bin = qtd_bin + qtds[i]
        elif(inf_bin <= valores_min[i] and sup_bin < valores_max[i] and sup_bin > valores_min[i]):
            qtd_bin = qtd_bin + qtds[i]*(sup_bin - valores_min[i])/(valores_max[i] - valores_min[i])
        elif(inf_bin > valores_min[i] and sup_bin >= valores_max[i] and inf_bin < valores_max[i]):
            qtd_bin = qtd_bin + qtds[i]*(valores_max[i] - inf_bin)/(valores_max[i] - valores_min[i])
        elif(inf_bin > valores_min[i] and sup_bin < valores_max[i]):
            qtd_bin = qtd_bin + qtds[i]*(sup_bin - inf_bin)/(valores_max[i] - valores_min[i])
    return qtd_bin

def discretiza_vetor(vetor, valores_digitize):
    flag_nan = np.isnan(vetor)
    disc = np.digitize(vetor, valores_digitize)
    return np.where(flag_nan, np.nan, disc)

def discretiza_vetor_media(vetor, valores_digitize, valores_medios):
    flag_nan = np.isnan(vetor)
    disc = np.digitize(vetor, valores_digitize)
    return np.where(flag_nan, np.nan, valores_medios[disc])

###############################
#Para o CortaIntervalosGanhoInformacao e CortaIntervalosR2
###############################

def unicos_qtds_alvos(vetor, alvo):
    inds_sorted = np.argsort(vetor)
    vetor_sorted = vetor[inds_sorted]
    alvo_sorted = alvo[inds_sorted]
    valores = []
    qtds = []
    qtds_alvo = []
    qtd_unicos = 0
    v = np.nan
    qtd = 0
    qtd_alvo = 0
    for i in prange(vetor_sorted.size):
        if vetor_sorted[i] == v:
            qtd = qtd + 1
            qtd_alvo = qtd_alvo + alvo_sorted[i]
        else:
            valores.append(v)
            qtds.append(qtd)
            qtds_alvo.append(qtd_alvo)
            qtd_unicos = qtd_unicos + 1
            v = vetor_sorted[i]
            qtd = 1
            qtd_alvo = alvo_sorted[i]
    valores.append(v)
    qtds.append(qtd)
    qtds_alvo.append(qtd_alvo)
    return np.array(valores[1:]), np.array(qtds[1:]), np.array(qtds_alvo[1:]), qtd_unicos
    
def unicos_qtds_alvos2(vetor, alvo):
    inds_sorted = np.argsort(vetor)
    vetor_sorted = vetor[inds_sorted]
    alvo_sorted = alvo[inds_sorted]
    valores = []
    qtds = []
    qtds_alvo = []
    qtds_alvo2 = []
    qtd_unicos = 0
    v = np.nan
    qtd = 0
    qtd_alvo = 0
    qtd_alvo2 = 0
    for i in prange(vetor_sorted.size):
        if vetor_sorted[i] == v:
            qtd = qtd + 1
            qtd_alvo = qtd_alvo + alvo_sorted[i]
            qtd_alvo2 = qtd_alvo2 + alvo_sorted[i]**2
        else:
            valores.append(v)
            qtds.append(qtd)
            qtds_alvo.append(qtd_alvo)
            qtds_alvo2.append(qtd_alvo2)
            qtd_unicos = qtd_unicos + 1
            v = vetor_sorted[i]
            qtd = 1
            qtd_alvo = alvo_sorted[i]
            qtd_alvo2 = alvo_sorted[i]**2
    valores.append(v)
    qtds.append(qtd)
    qtds_alvo.append(qtd_alvo)
    return np.array(valores[1:]), np.array(qtds[1:]), np.array(qtds_alvo[1:]), np.array(qtds_alvo2[1:]), qtd_unicos

def pontos_corte_alvo(qtds, qtd_unicos, qtds_alvo, num_div):
    passo = int(qtd_unicos/num_div)
    pts_corte = [i*passo for i in prange(num_div+1)]
    pts_corte[-1] = qtd_unicos
    qtds_corte = [np.sum(qtds[pts_corte[i]:pts_corte[i+1]]) for i in prange(num_div)]
    qtds_alvo_corte = [np.sum(qtds_alvo[pts_corte[i]:pts_corte[i+1]]) for i in prange(num_div)]
    pts_corte = [p - 1 for p in pts_corte[1:]]
    return np.array(pts_corte), np.array(qtds_corte), np.array(qtds_alvo_corte)

def pontos_corte_alvo2(qtds, qtd_unicos, qtds_alvo, qtds_alvo2, num_div):
    passo = int(qtd_unicos/num_div)
    pts_corte = [i*passo for i in prange(num_div+1)]
    pts_corte[-1] = qtd_unicos
    qtds_corte = [np.sum(qtds[pts_corte[i]:pts_corte[i+1]]) for i in prange(num_div)]
    qtds_alvo_corte = [np.sum(qtds_alvo[pts_corte[i]:pts_corte[i+1]]) for i in prange(num_div)]
    qtds_alvo2_corte = [np.sum(qtds_alvo2[pts_corte[i]:pts_corte[i+1]]) for i in prange(num_div)]
    pts_corte = [p - 1 for p in pts_corte[1:]]
    return np.array(pts_corte), np.array(qtds_corte), np.array(qtds_alvo_corte), np.array(qtds_alvo2_corte)

def funcao_comparacao_entropia(Qe, Ae, Qd, Ad):
    retorno = Qe*np.log2(Qe) + Qd*np.log2(Qd)
    if(Ae > 0):
        retorno = retorno - Ae*np.log2(Ae)
    if(Ae < Qe):
        retorno = retorno - (Qe-Ae)*np.log2(Qe-Ae)
    if(Ad > 0):
        retorno = retorno - Ad*np.log2(Ad)
    if(Ad < Qd):
        retorno = retorno - (Qd-Ad)*np.log2(Qd-Ad)
    return retorno
    
def funcao_comparacao_entropia_balanceada(Qe, Ae, Qd, Ad):
    retorno = np.log2(Qe) + np.log2(Qd)
    if(Ae > 0):
        retorno = retorno - Ae*np.log2(Ae)/Qe
    if(Ae < Qe):
        retorno = retorno - (Qe-Ae)*np.log2(Qe-Ae)/Qe
    if(Ad > 0):
        retorno = retorno - Ad*np.log2(Ad)/Qd
    if(Ad < Qd):
        retorno = retorno - (Qd-Ad)*np.log2(Qd-Ad)/Qd
    return retorno

def minimiza_entropia(pts_corte, qtds_corte, qtds, qtds_alvo_corte, qtds_alvo, qtd_unicos, qtd_min):
    fim = pts_corte.size - 1
    prefim = fim - 1
    permutado = True
    while(permutado):
        permutado = False
        for i in prange(fim):
            Qe = qtds_corte[prefim-i]
            Qd = qtds_corte[fim-i]
            Ae = qtds_alvo_corte[prefim-i]
            Ad = qtds_alvo_corte[fim-i]
            p = pts_corte[prefim-i]
            
            p_inf = 0 if i == prefim else pts_corte[prefim-i-1]
            e = p
            while(permutado == False and e > p_inf):
                qe = np.sum(qtds[e:p+1])
                ae = np.sum(qtds_alvo[e:p+1])
                if(Qe > qe + qtd_min and funcao_comparacao_entropia(Qe-qe, Ae-ae, Qd+qe, Ad+ae) < funcao_comparacao_entropia(Qe, Ae, Qd, Ad)):
                    qtds_corte[fim-i] = Qd + qe
                    qtds_corte[prefim-i] = Qe - qe
                    qtds_alvo_corte[fim-i] = Ad + ae
                    qtds_alvo_corte[prefim-i] = Ae - ae
                    pts_corte[prefim-i] = e - 1
                    permutado = True
                else:
                    e = e - 1
            
            p_sup = pts_corte[fim-i]
            d = p+1
            while(permutado == False and d < p_sup):
                qd = np.sum(qtds[p+1:d+1])
                ad = np.sum(qtds_alvo[p+1:d+1])
                if(Qd > qd + qtd_min and funcao_comparacao_entropia(Qe+qd, Ae+ad, Qd-qd, Ad-ad) < funcao_comparacao_entropia(Qe, Ae, Qd, Ad)):
                    qtds_corte[fim-i] = Qd - qd
                    qtds_corte[prefim-i] = Qe + qd
                    qtds_alvo_corte[fim-i] = Ad - ad
                    qtds_alvo_corte[prefim-i] = Ae + ad
                    pts_corte[prefim-i] = d
                    permutado = True
                else:
                    d = d + 1
                
    return pts_corte, qtds_corte, qtds_alvo_corte
    
def minimiza_entropia_balanceada(pts_corte, qtds_corte, qtds, qtds_alvo_corte, qtds_alvo, qtd_unicos, qtd_min):
    fim = pts_corte.size - 1
    prefim = fim - 1
    permutado = True
    while(permutado):
        permutado = False
        for i in prange(fim):
            Qe = qtds_corte[prefim-i]
            Qd = qtds_corte[fim-i]
            Ae = qtds_alvo_corte[prefim-i]
            Ad = qtds_alvo_corte[fim-i]
            p = pts_corte[prefim-i]
            
            p_inf = 0 if i == prefim else pts_corte[prefim-i-1]
            e = p
            while(permutado == False and e > p_inf):
                qe = np.sum(qtds[e:p+1])
                ae = np.sum(qtds_alvo[e:p+1])
                if(Qe > qe + qtd_min and funcao_comparacao_entropia_balanceada(Qe-qe, Ae-ae, Qd+qe, Ad+ae) < funcao_comparacao_entropia_balanceada(Qe, Ae, Qd, Ad)):
                    qtds_corte[fim-i] = Qd + qe
                    qtds_corte[prefim-i] = Qe - qe
                    qtds_alvo_corte[fim-i] = Ad + ae
                    qtds_alvo_corte[prefim-i] = Ae - ae
                    pts_corte[prefim-i] = e - 1
                    permutado = True
                else:
                    e = e - 1
            
            p_sup = pts_corte[fim-i]
            d = p+1
            while(permutado == False and d < p_sup):
                qd = np.sum(qtds[p+1:d+1])
                ad = np.sum(qtds_alvo[p+1:d+1])
                if(Qd > qd + qtd_min and funcao_comparacao_entropia_balanceada(Qe+qd, Ae+ad, Qd-qd, Ad-ad) < funcao_comparacao_entropia_balanceada(Qe, Ae, Qd, Ad)):
                    qtds_corte[fim-i] = Qd - qd
                    qtds_corte[prefim-i] = Qe + qd
                    qtds_alvo_corte[fim-i] = Ad - ad
                    qtds_alvo_corte[prefim-i] = Ae + ad
                    pts_corte[prefim-i] = d
                    permutado = True
                else:
                    d = d + 1
                
    return pts_corte, qtds_corte, qtds_alvo_corte

def funcao_comparacao_mse(Qe, Ae, Qd, Ad):
    return -(Ae**2/Qe + Ad**2/Qd)
    
def funcao_comparacao_mse_balanceado(Qe, Ae, Ae2, Qd, Ad, Ad2):
    return Ae2/Qe + Ad2/Qd - (Ae/Qe)**2 - (Ad/Qd)**2

def minimiza_mse(pts_corte, qtds_corte, qtds, qtds_alvo_corte, qtds_alvo, qtd_unicos, qtd_min):
    fim = pts_corte.size - 1
    prefim = fim - 1
    permutado = True
    while(permutado):
        permutado = False
        for i in prange(fim):
            Qe = qtds_corte[prefim-i]
            Qd = qtds_corte[fim-i]
            Ae = qtds_alvo_corte[prefim-i]
            Ad = qtds_alvo_corte[fim-i]
            p = pts_corte[prefim-i]
            
            p_inf = 0 if i == prefim else pts_corte[prefim-i-1]
            e = p
            while(permutado == False and e > p_inf):
                qe = np.sum(qtds[e:p+1])
                ae = np.sum(qtds_alvo[e:p+1])
                if(Qe > qe + qtd_min and funcao_comparacao_mse(Qe-qe, Ae-ae, Qd+qe, Ad+ae) < funcao_comparacao_mse(Qe, Ae, Qd, Ad)):
                    qtds_corte[fim-i] = Qd + qe
                    qtds_corte[prefim-i] = Qe - qe
                    qtds_alvo_corte[fim-i] = Ad + ae
                    qtds_alvo_corte[prefim-i] = Ae - ae
                    pts_corte[prefim-i] = e - 1
                    permutado = True
                else:
                    e = e - 1
            
            p_sup = pts_corte[fim-i]
            d = p+1
            while(permutado == False and d < p_sup):
                qd = np.sum(qtds[p+1:d+1])
                ad = np.sum(qtds_alvo[p+1:d+1])
                if(Qd > qd + qtd_min and funcao_comparacao_mse(Qe+qd, Ae+ad, Qd-qd, Ad-ad) < funcao_comparacao_mse(Qe, Ae, Qd, Ad)):
                    qtds_corte[fim-i] = Qd - qd
                    qtds_corte[prefim-i] = Qe + qd
                    qtds_alvo_corte[fim-i] = Ad - ad
                    qtds_alvo_corte[prefim-i] = Ae + ad
                    pts_corte[prefim-i] = d
                    permutado = True
                else:
                    d = d + 1
                
    return pts_corte, qtds_corte, qtds_alvo_corte

def minimiza_mse_balanceado(pts_corte, qtds_corte, qtds, qtds_alvo_corte, qtds_alvo, qtds_alvo2_corte, qtds_alvo2, qtd_unicos, qtd_min):
    fim = pts_corte.size - 1
    prefim = fim - 1
    permutado = True
    while(permutado):
        permutado = False
        for i in prange(fim):
            Qe = qtds_corte[prefim-i]
            Qd = qtds_corte[fim-i]
            Ae = qtds_alvo_corte[prefim-i]
            Ad = qtds_alvo_corte[fim-i]
            Ae2 = qtds_alvo2_corte[prefim-i]
            Ad2 = qtds_alvo2_corte[fim-i]
            p = pts_corte[prefim-i]
            
            p_inf = 0 if i == prefim else pts_corte[prefim-i-1]
            e = p
            while(permutado == False and e > p_inf):
                qe = np.sum(qtds[e:p+1])
                ae = np.sum(qtds_alvo[e:p+1])
                ae2 = np.sum(qtds_alvo2[e:p+1])
                if(Qe > qe + qtd_min and funcao_comparacao_mse_balanceado(Qe-qe, Ae-ae, Ae2-ae2, Qd+qe, Ad+ae, Ad2+ae2) < funcao_comparacao_mse_balanceado(Qe, Ae, Ae2, Qd, Ad, Ad2)):
                    qtds_corte[fim-i] = Qd + qe
                    qtds_corte[prefim-i] = Qe - qe
                    qtds_alvo_corte[fim-i] = Ad + ae
                    qtds_alvo_corte[prefim-i] = Ae - ae
                    qtds_alvo2_corte[fim-i] = Ad2 + ae2
                    qtds_alvo2_corte[prefim-i] = Ae2 - ae2
                    pts_corte[prefim-i] = e - 1
                    permutado = True
                else:
                    e = e - 1
            
            p_sup = pts_corte[fim-i]
            d = p+1
            while(permutado == False and d < p_sup):
                qd = np.sum(qtds[p+1:d+1])
                ad = np.sum(qtds_alvo[p+1:d+1])
                ad2 = np.sum(qtds_alvo2[p+1:d+1])
                if(Qd > qd + qtd_min and funcao_comparacao_mse_balanceado(Qe+qd, Ae+ad, Ae2+ad2, Qd-qd, Ad-ad, Ad2-ad2) < funcao_comparacao_mse_balanceado(Qe, Ae, Ae2, Qd, Ad, Ad2)):
                    qtds_corte[fim-i] = Qd - qd
                    qtds_corte[prefim-i] = Qe + qd
                    qtds_alvo_corte[fim-i] = Ad - ad
                    qtds_alvo_corte[prefim-i] = Ae + ad
                    qtds_alvo2_corte[fim-i] = Ad2 - ad2
                    qtds_alvo2_corte[prefim-i] = Ae2 + ad2
                    pts_corte[prefim-i] = d
                    permutado = True
                else:
                    d = d + 1
                
    return pts_corte, qtds_corte, qtds_alvo_corte

###############################
#Para FiltraCategoriasRelevantes
###############################

def indices_e_ordenacao(qtds):
    inds_sorted = np.argsort(qtds)
    qtds_sorted = qtds[inds_sorted]
    return inds_sorted, qtds_sorted
    
###############################
#Para TrataDataset
###############################

def conta_qtd_unicos(vetor):
    return np.unique(vetor).size
    
###############################
#Para TaylorLaurentSeries
###############################

def multiplica_vetores(vetor1, vetor2):
    return vetor1 * vetor2
    
jit_module(nopython = True, 
           #cache = True
           )

def calcula_distribuicao_acumulada(y_prob, discretizador, y):
    vetor_y_minmax = discretizador.intervalos.pares_minimo_maximo_discretizacao()
    vetor_y_min = np.array([v[0] for v in vetor_y_minmax])
    vetor_y_max = np.array([v[1] for v in vetor_y_minmax])
    y_disc = discretizador.intervalos.aplica_discretizacao(y).astype(int)
    L = vetor_y_max - vetor_y_min
    y_prob_acum = np.array([np.sum(y_prob[:y_disc[i]]) + y_prob[y_disc[i]]*(y[i] - vetor_y_min[y_disc[i]])/L[y_disc[i]] for i in range(y.size)])
    return y_prob_acum

def calcula_distribuicao_acumulada_normal(media, desvio, y):
    return np.array([norm.cdf(v, loc = media, scale = desvio) for v in y])

def calcula_distribuicao_acumulada_normal_condicional(media, desvio, qtds, qtd_unicos, y):
    return np.array([np.sum([qtds[i]*norm.cdf(v, loc = media[i], scale = desvio[i]) for i in range(qtd_unicos)]) for v in y])/np.sum(qtds)