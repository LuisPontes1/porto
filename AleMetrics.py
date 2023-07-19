import numpy as np
import pandas as pd

import math

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

from AleTransforms import *

#Para funcionar direito, não pode haver nulos em y ou y_prob
class AletricasClassificacao:
    
    def __init__(self, y, y_prob, num_div = None, p_corte = None, p01_corte = [0, 0], p_ref = None):
        self.__y = y
        self.__y_prob = y_prob
        self.__y_prob_inicial = y_prob
        self.__qtd_unicos = qtd_unicos(y_prob)
        
        #Probabilidades de Corte para Avaliação de Tomada de Decisão
        #OBS: Quando nenhuma prob de corte é passada, 
        #usamos a prob de ganho de informação para a tomada de decisão
        self.__p_corte = p_corte
        self.__p01_corte = np.array(p01_corte)
        self.__p_ref = p_ref
        
        #Variaveis caso queira fazer as contas por intervalos de prob
        self.__num_div = num_div
        self.__interv = None
        
        self.__qtd_tot = None
        self.__soma_probs = None
        self.__qtd1_tot = None
        self.__qtd0_tot = None
        
        self.__y_prob_unico = None
        self.__qtds = None
        self.__qtds1 = None
        
        self.__qtds_acum = None
        self.__qtds1_acum = None
        self.__qtds0_acum = None
        
        #_c indica o conjunto complementar (o que ainda não foi somado)
        self.__qtds_acum_c = None
        self.__qtds1_acum_c = None 
        self.__qtds0_acum_c = None 
        
        #vp: verdadeiro positivo, p_tot: total de positivos
        #vn: verdadeiro negativo, n_tot: total de negativos
        self.__curva_tvp = None #Armazena a curva de taxa de verdadeiro positivo (vp / p_tot)
        self.__curva_tvn = None #Armazena a curva de taxa verdadeiro negativo (vn / n_tot)
        self.__auc = None
        
        self.__curva_revoc1 = None #Armazena a curva de revocacao de 1
        self.__curva_revoc0 = None #Armazena a curva de revocacao de 0
        self.__pos_max_dif = None
        self.__ks = None
        
        self.__liftF_10 = None
        self.__alavF_10 = None
        self.__liftF_20 = None
        self.__alavF_20 = None
        self.__liftV_10 = None
        self.__alavV_10 = None
        self.__liftV_20 = None
        self.__alavV_20 = None
        
        self.__curva_ig = None #Armazena a curva de ganho de informação
        self.__pos_max_ig = None
        self.__ig = None
        
        self.__vetor_p0_ig_2d = None
        self.__vetor_p1_ig_2d = None
        self.__vetor_ig_2d = None
        self.__pos_max_ig_2d = None
        self.__ig_2d = None
        
        self.matriz_confusao = None
        self.matriz_confusao_2d = None
        self.__frac_incerto_2d = None
        self.__p00 = None #Prob de ser 0 dado que o modelo disse que é 0
        self.__p11 = None #Prob de ser 1 dado que o modelo disse que é 1
        self.__p00_2d = None #Prob de ser 0 dado que o modelo disse que é 0 (corte com p0 e p1)
        self.__p11_2d = None #Prob de ser 1 dado que o modelo disse que é 1 (corte com p0 e p1)
        self.__acuracia = None
        self.__acuracia_balanceada = None
        self.__acuracia_2d = None 
        self.__acuracia_balanceada_2d = None
        
        self.__tvn = None
        self.__tvp = None
        self.__tvn_2d = None
        self.__tvp_2d = None
        
        self.__logloss_baseline = None
        self.__logloss = None
        self.__coef_logloss = None
        self.__coef_logloss_ref = None
        
        self.__calcula_metricas()
        
    def __ordena_probs(self):
        self.__qtd_tot = self.__y.size
        self.__soma_probs = np.sum(self.__y_prob)
        
        if(self.__num_div is not None and self.__qtd_unicos >= self.__num_div):
            self.__interv = CortaIntervalosQuasiUniforme(self.__y_prob, num_div = self.__num_div)
            self.__y_prob = self.__interv.aplica_discretizacao(self.__y_prob).astype(int)

        inds_ordenado, self.__y_prob_unico, primeira_ocorrencia, self.__qtds, _ = indices_unicos_qtds(self.__y_prob)
        y_agrup = np.split(self.__y[inds_ordenado], primeira_ocorrencia[1:])
        self.__qtds1 = np.array([soma_vetor(v) for v in y_agrup])
        
        self.__qtds_acum = soma_acumulada_vetor(self.__qtds)
        self.__qtds1_acum = soma_acumulada_vetor(self.__qtds1)
        self.__qtds0_acum = diferenca_vetores(self.__qtds_acum, self.__qtds1_acum)
        
        self.__qtd1_tot = self.__qtds1_acum[-1]
        self.__qtd0_tot = self.__qtd_tot - self.__qtd1_tot
        
        self.__qtds_acum_c = diferenca_vetores(self.__qtd_tot, self.__qtds_acum)
        self.__qtds1_acum_c = diferenca_vetores(self.__qtd1_tot, self.__qtds1_acum)
        self.__qtds0_acum_c = diferenca_vetores(self.__qtd0_tot, self.__qtds0_acum)
        
        #Especial para calcular o Lift direito
        self.__qtds_acum_c_lift = np.append(self.__qtd_tot, self.__qtds_acum_c[:-1])
        self.__qtds1_acum_c_lift = np.append(self.__qtd1_tot, self.__qtds1_acum_c[:-1])
        self.__qtds0_acum_c_lift = np.append(self.__qtd0_tot, self.__qtds0_acum_c[:-1])
    
    def __calcula_roc(self):
        self.__curva_tnp = divisao_vetores(self.__qtds0_acum, self.__qtd0_tot)
        self.__curva_tvp = divisao_vetores(self.__qtds1_acum_c, self.__qtd1_tot)
        
        #Coloca na mão o valor inicial da curva ROC
        self.__curva_tnp = np.insert(self.__curva_tnp, 0, 0)
        self.__curva_tvp = np.insert(self.__curva_tvp, 0, 1)
            
        self.__auc = area(self.__curva_tnp, self.__curva_tvp)
    
    def __calcula_ks(self):
        self.__curva_revoc0 = divisao_vetores(self.__qtds0_acum, self.__qtd0_tot)
        self.__curva_revoc1 = divisao_vetores(self.__qtds1_acum, self.__qtd1_tot)
        
        curva_dif = diferenca_vetores(self.__curva_revoc0, self.__curva_revoc1)
        self.__pos_max_dif = argmax_vetor(curva_dif) #Pega as posições em que encontrou o máximo
        
        #Pega o valor máximo (tenta ver se pos_max é um vetor ou um número)
        try:
            self.__ks = curva_dif[self.__pos_max_dif[0]]
        except:
            self.__ks = curva_dif[self.__pos_max_dif]
    
    def __calcula_lift_alavancagem(self, decrescente = False, frac = 0.5):
        qtd_ref = frac*self.__qtd_tot
        if(decrescente == False):
            pos_ini = np.sum(self.__qtds_acum <= qtd_ref) - 1
            if(self.__qtds_acum[pos_ini] == qtd_ref or pos_ini == self.__qtds_acum.size - 1):
                lift = self.__qtds0_acum[pos_ini]/self.__qtd0_tot
                alav = lift/frac
            else:
                qtd0_medio = (self.__qtds0_acum[pos_ini] + self.__qtds0_acum[pos_ini+1])/2
                qtd_ref_medio = (self.__qtds_acum[pos_ini] + self.__qtds_acum[pos_ini+1])/2
                lift = qtd0_medio/self.__qtd0_tot
                alav = lift*self.__qtd_tot/qtd_ref_medio
        else:
            pos_ini = self.__qtds_acum_c_lift.size - np.sum(self.__qtds_acum_c_lift <= qtd_ref)
            if(pos_ini == self.__qtds_acum_c_lift.size or self.__qtds_acum_c_lift[pos_ini] == qtd_ref):
                lift = self.__qtds1_acum_c_lift[min(pos_ini, self.__qtds_acum_c_lift.size-1)]/self.__qtd1_tot
                alav = lift/frac
            else:
                qtd1_medio = (self.__qtds1_acum_c_lift[pos_ini] + self.__qtds1_acum_c_lift[pos_ini-1])/2
                qtd_ref_medio = (self.__qtds_acum_c_lift[pos_ini] + self.__qtds_acum_c_lift[pos_ini-1])/2
                lift = qtd1_medio/self.__qtd1_tot
                alav = lift*self.__qtd_tot/qtd_ref_medio
        return lift, alav
                
    def __calcula_ig(self):
        #Calcula a Entropia de Shannon
        def entropia_shannon_unica(p1):
            p0 = 1 - p1
            if p0 == 0 or p1 == 0:
                return 0
            else:
                return -p0*math.log2(p0) - p1*math.log2(p1)
        
        p1 = self.__qtd1_tot/self.__qtd_tot
        entropia_ini = entropia_shannon_unica(p1)

        #O último corte por definição não dá informação nenhuma, então nem faz a conta (por isso o [:-1])
        qtds_acum = self.__qtds_acum[:-1]
        qtds1_acum = self.__qtds1_acum[:-1]
        p1_acum = divisao_vetores(qtds1_acum, qtds_acum)
        entropia_parcial = entropia_shannon(p1_acum)

        qtds_acum_c = self.__qtds_acum_c[:-1]
        qtds1_acum_c = self.__qtds1_acum_c[:-1]
        p1c_acum = divisao_vetores(qtds1_acum_c, qtds_acum_c)
        entropia_parcial_c = entropia_shannon(p1c_acum)

        self.__curva_ig = calcula_curva_ig(entropia_parcial, qtds_acum, entropia_parcial_c, qtds_acum_c, self.__qtd_tot, entropia_ini)
        
        self.__pos_max_ig = argmax_vetor(self.__curva_ig) #Pega as posições em que encontrou o máximo
        #Pega o valor máximo (tenta ver se pos_max é um vetor ou um número)
        try:
            self.__ig = self.__curva_ig[self.__pos_max_ig[0]] 
        except:
            self.__ig = self.__curva_ig[self.__pos_max_ig]
        
    def __calcula_ig_2d(self):
        #Calcula a Entropia de Shannon
        def entropia_shannon_unica(p1):
            p0 = 1 - p1
            if p0 == 0 or p1 == 0:
                return 0
            else:
                return -p0*math.log2(p0) - p1*math.log2(p1)
        
        p1_ini = self.__qtd1_tot/self.__qtd_tot
        entropia_ini = entropia_shannon_unica(p1_ini) 
        
        vetor_p0 = np.array([])
        vetor_p1 = np.array([])
        vetor_entropia = np.array([])
        vetor_ig = np.array([])
        #Temos a subtração -1 pois como já discutido, o último corte por definição não trás ganho de informação
        num_loop = self.__y_prob_unico.size - 1
        #Subtrai mais um aqui pq queremos garantir que todo o loop tem um intervalo de resto
        for i in range(num_loop-1):
            start_loop2 = i + 1 #O segundo loop começa sempre 1 a frente pq queremos que sobre um intervalo de resto
            vetor_p0 = np.append(vetor_p0, np.repeat(self.__y_prob_unico[i], num_loop - start_loop2))
            qtd_acum = self.__qtds_acum[i]
            qtd1_acum = self.__qtds1_acum[i]
            p1 = qtd1_acum/qtd_acum
            entropia_parcial = entropia_shannon_unica(p1)
            
            entropia_aux = entropia_parcial*qtd_acum/self.__qtd_tot
            
            #Segundo loop implicito nos vetores
            vetor_p1 = np.append(vetor_p1, self.__y_prob_unico[start_loop2:num_loop])
            qtd_acum_c = self.__qtds_acum_c[start_loop2:num_loop]
            qtd1_acum_c = self.__qtds1_acum_c[start_loop2:num_loop]
            p1c = divisao_vetores(qtd1_acum_c, qtd_acum_c)
            entropia_parcial_c = entropia_shannon(p1c)
            
            qtd_resto = self.__qtd_tot - qtd_acum - qtd_acum_c
            qtd1_acum_resto = self.__qtd1_tot - qtd1_acum - qtd1_acum_c
            p1r = divisao_vetores(qtd1_acum_resto, qtd_resto)
            entropia_parcial_r = entropia_shannon(p1r)
            
            vetor_entropia = calcula_vetor_entropia_parcial(vetor_entropia, entropia_aux, entropia_parcial_c, qtd_acum_c, entropia_parcial_r, qtd_resto, self.__qtd_tot)
                
        self.__vetor_ig_2d = normaliza_vetor_entropia(vetor_entropia, entropia_ini)
        self.__vetor_p0_ig_2d = vetor_p0
        self.__vetor_p1_ig_2d = vetor_p1
        
        self.__pos_max_ig_2d = np.argmax(self.__vetor_ig_2d) #Pega as posições em que encontrou o máximo
        try:
            self.__ig_2d = self.__vetor_ig_2d[self.__pos_max_ig_2d[0]] 
        except:
            self.__ig_2d = self.__vetor_ig_2d[self.__pos_max_ig_2d]
    
    def calcula_matriz_confusao(self, p0, p1, normalizado = False):
        if(p0 == p1):
            y_pred = np.array([0 if p <= p0 else 1 for p in self.__y_prob_inicial])
            y = self.__y
            frac_incerto = 0
        else:
            y_pred = np.array([0 if p <= p0 else 1 if p > p1 else np.nan for p in self.__y_prob_inicial])
            flag_na = np.isnan(y_pred, where = True)
            y_pred = y_pred[~flag_na]
            y = self.__y[~flag_na]
            frac_incerto = np.sum(flag_na)/self.__y_prob_inicial.size
        flag_vp = (y == y_pred)&(y_pred == 1)
        flag_vn = (y == y_pred)&(y_pred == 0)
        flag_fp = (y != y_pred)&(y_pred == 1)
        flag_fn = (y != y_pred)&(y_pred == 0)
        #Linhas: Preditos, Colunas: Labels
        #Normalização: Tem como objetivo obter as probabilidades condicionais
        #Isto é, dado que o modelo prediz um certo label, qual a prob de ser esse label mesmo e qual a prob de ser o outro label
        if(normalizado):
            vn = np.sum(flag_vn)
            fn = np.sum(flag_fn)
            norm_n = vn + fn
            fp = np.sum(flag_fp)
            vp = np.sum(flag_vp)
            norm_p = vp + fp
            matriz = np.matrix([[vn/norm_n, fn/norm_n], [fp/norm_p, vp/norm_p]])
        else:
            matriz = np.matrix([[np.sum(flag_vn), np.sum(flag_fn)], [np.sum(flag_fp), np.sum(flag_vp)]])
        return matriz, frac_incerto
    
    def __calcula_probabilidades_condicionais(self, matriz):
        soma = np.sum(matriz[0, :])
        if soma > 0:
            p00 = matriz[0, 0]/soma
        else:
            p00 = np.nan
        soma = np.sum(matriz[1, :])
        if soma > 0:
            p11 = matriz[1, 1]/soma
        else:
            p11 = np.nan
        return p00, p11
        
    def __calcula_acuracia(self, matriz):
        soma = np.sum(matriz)
        if soma > 0:
            acuracia = (matriz[0, 0] + matriz[1, 1])/soma
        else:
            acuracia = np.nan
        return acuracia
        
    def __calcula_acuracia_balanceada(self, matriz):
        soma_0 = matriz[0, 0] + matriz[1, 0]
        soma_1 = matriz[0, 1] + matriz[1, 1]
        if soma_0 > 0:
            acuracia_0 = matriz[0, 0]/soma_0
        else:
            acuracia_0 = None
        if soma_1 > 0:
            acuracia_1 = matriz[1, 1]/soma_1
        else:
            acuracia_1 = None
        if soma_0 > 0 and soma_1 > 0:
            acuracia_bal = (acuracia_0 + acuracia_1)*0.5
        else:
            acuracia_bal = None
        return acuracia_bal, acuracia_0, acuracia_1
        
    def __calcula_logloss(self):
        self.__logloss = logloss(self.__y, self.__y_prob_inicial)
        if(self.__p_ref is None):
            self.__p_ref = calcula_media(self.__y)
            self.__logloss_baseline = logloss(self.__y, np.repeat(self.__p_ref, self.__y.size))
            self.__logloss_ref = self.__logloss_baseline
        else:
            self.__logloss_baseline = logloss(self.__y, np.repeat(calcula_media(self.__y), self.__y.size))
            self.__logloss_ref = logloss(self.__y, np.repeat(self.__p_ref, self.__y.size))
        self.__coef_logloss = 1 - self.__logloss/self.__logloss_baseline
        self.__coef_logloss_ref = 1 - self.__logloss/self.__logloss_ref
        
    def retorna_p_ref(self):
        return self.__p_ref
    
    def __calcula_metricas(self):
        self.__ordena_probs()
        if(self.__qtd0_tot*self.__qtd1_tot > 0):
            self.__calcula_logloss()
            if(self.__y_prob_unico.size > 2):
                self.__calcula_roc()
                self.__calcula_ks()
                self.__liftF_10, self.__alavF_10 = self.__calcula_lift_alavancagem(decrescente = False, frac = 0.1)
                self.__liftF_20, self.__alavF_20 = self.__calcula_lift_alavancagem(decrescente = False, frac = 0.2)
                self.__liftV_10, self.__alavV_10 = self.__calcula_lift_alavancagem(decrescente = True, frac = 0.1)
                self.__liftV_20, self.__alavV_20 = self.__calcula_lift_alavancagem(decrescente = True, frac = 0.2)
                self.__calcula_ig()
                self.__calcula_ig_2d()
                probs_ig = self.valor_prob_ig()
                if(self.__p_corte == None):
                    self.__p_corte = probs_ig['Prob_Corte']
                if(np.sum(self.__p01_corte) == 0):
                    self.__p01_corte = np.array([probs_ig['Prob0_Corte'], probs_ig['Prob1_Corte']])
                self.matriz_confusao, _ = self.calcula_matriz_confusao(p0 = self.__p_corte, p1 = self.__p_corte)
                self.matriz_confusao_2d, self.__frac_incerto_2d = self.calcula_matriz_confusao(p0 = self.__p01_corte[0], p1 = self.__p01_corte[1])
                self.__p00, self.__p11 = self.__calcula_probabilidades_condicionais(self.matriz_confusao)
                self.__p00_2d, self.__p11_2d = self.__calcula_probabilidades_condicionais(self.matriz_confusao_2d)
                self.__acuracia = self.__calcula_acuracia(self.matriz_confusao)
                self.__acuracia_balanceada, self.__tvn, self.__tvp = self.__calcula_acuracia_balanceada(self.matriz_confusao)
                self.__acuracia_2d = self.__calcula_acuracia(self.matriz_confusao_2d)
                self.__acuracia_balanceada_2d, self.__tvn_2d, self.__tvp_2d = self.__calcula_acuracia_balanceada(self.matriz_confusao_2d)
            elif(self.__y_prob_unico.size > 1):
                self.__calcula_roc()
                self.__calcula_ks()
                self.__liftF_10, self.__alavF_10 = self.__calcula_lift_alavancagem(decrescente = False, frac = 0.1)
                self.__liftF_20, self.__alavF_20 = self.__calcula_lift_alavancagem(decrescente = False, frac = 0.2)
                self.__liftV_10, self.__alavV_10 = self.__calcula_lift_alavancagem(decrescente = True, frac = 0.1)
                self.__liftV_20, self.__alavV_20 = self.__calcula_lift_alavancagem(decrescente = True, frac = 0.2)
                self.__calcula_ig()
                probs_ig = self.valor_prob_ig()
                if(self.__p_corte == None):
                    self.__p_corte = probs_ig['Prob_Corte']
                self.matriz_confusao, _ = self.calcula_matriz_confusao(p0 = self.__p_corte, p1 = self.__p_corte)
                self.__p00, self.__p11 = self.__calcula_probabilidades_condicionais(self.matriz_confusao)
                self.__acuracia = self.__calcula_acuracia(self.matriz_confusao)
                self.__acuracia_balanceada, self.__tvn, self.__tvp = self.__calcula_acuracia_balanceada(self.matriz_confusao)
    
    def valor_prob_ig(self):
        #Retorna um pd.Series com as probs de corte encontradas no ganho de informação
        d = {}
        prob_corte = None
        p0_corte = None
        p1_corte = None
        if(self.__num_div is not None and self.__qtd_unicos >= self.__num_div):
            if(self.__pos_max_ig != None):
                prob_corte = self.__interv.valores_medios_discretizacao()[self.__pos_max_ig]
            if(self.__pos_max_ig_2d != None):
                pos_p0_aux = int(self.__vetor_p0_ig_2d[self.__pos_max_ig_2d])
                pos_p1_aux = int(self.__vetor_p1_ig_2d[self.__pos_max_ig_2d])
                p0_corte = self.__interv.valores_medios_discretizacao()[pos_p0_aux]
                p1_corte = self.__interv.valores_medios_discretizacao()[pos_p1_aux]
        else:
            if(self.__pos_max_ig != None):
                prob_corte = (self.__y_prob_unico[self.__pos_max_ig] + self.__y_prob_unico[self.__pos_max_ig+1])/2 #Simetriza a prob decorte
            if(self.__pos_max_ig_2d != None):
                p0_corte = self.__vetor_p0_ig_2d[self.__pos_max_ig_2d] 
                p0_corte = (p0_corte + self.__y_prob_unico[np.searchsorted(self.__y_prob_unico, p0_corte)+1])/2 #Simetriza a prob decorte
                p1_corte = self.__vetor_p1_ig_2d[self.__pos_max_ig_2d]
                p1_corte = (p1_corte + self.__y_prob_unico[np.searchsorted(self.__y_prob_unico, p1_corte)+1])/2 #Simetriza a prob decorte
        d['Prob_Corte'] = prob_corte
        d['Prob0_Corte'] = p0_corte
        d['Prob1_Corte'] = p1_corte
        return d
    
    def valor_metricas(self, estatisticas_globais = True, probs_corte = True, probs_condicionais = True, lifts = True):
        #Retorna um pd.Series com as metricas calculadas
        #Esse formato é bom para criar dataframes
        d = {}
        if(estatisticas_globais):
            d['QTD'] = self.__qtd_tot
            d['QTD_0'] = self.__qtd_tot - self.__qtd1_tot
            d['QTD_1'] = self.__qtd1_tot
            d['Frac_0'] = (self.__qtd_tot - self.__qtd1_tot)/self.__qtd_tot
            d['Frac_1'] = self.__qtd1_tot/self.__qtd_tot
            d['Soma_Prob'] = self.__soma_probs
            d['Media_Prob'] = self.__soma_probs/self.__qtd_tot 
        d['LogLoss'] = self.__logloss
        d['CoefLogLoss'] = self.__coef_logloss
        d['CoefLogLoss_ref'] = self.__coef_logloss_ref
        d['AUC'] = self.__auc
        d['KS'] = self.__ks
        if(lifts):
            d['LiftF_10'] = self.__liftF_10
            d['LiftV_10'] = self.__liftV_10
            d['LiftF_20'] = self.__liftF_20
            d['LiftV_20'] = self.__liftV_20
            d['AlavF_10'] = self.__alavF_10
            d['AlavV_10'] = self.__alavV_10
            d['AlavF_20'] = self.__alavF_20
            d['AlavV_20'] = self.__alavV_20
        d['IG'] = self.__ig
        d['IG_2D'] = self.__ig_2d
        d['Frac_Incerto_2D'] = self.__frac_incerto_2d
        if(probs_corte):
            d.update(self.valor_prob_ig())
        d['Acurácia'] = self.__acuracia
        d['Acurácia_Balanceada'] = self.__acuracia_balanceada
        d['TVN'] = self.__tvn
        d['TVP'] = self.__tvp
        d['Acurácia_2D'] = self.__acuracia_2d
        d['Acurácia_Balanceada_2D'] = self.__acuracia_balanceada_2d
        d['TVN_2D'] = self.__tvn_2d
        d['TVP_2D'] = self.__tvp_2d
        if(self.__p00 != None and self.__p11 != None):
            d['Acurácia_Balanceada_Cond'] = (self.__p00 + self.__p11)*0.5
        else:
            d['Acurácia_Balanceada_Cond'] = None
        if(self.__p00_2d != None and self.__p11_2d != None):
            d['Acurácia_Balanceada_Cond_2D'] = (self.__p00_2d + self.__p11_2d)*0.5
        else:
            d['Acurácia_Balanceada_Cond_2D'] = None
        if(probs_condicionais):
            d['P(0|0)'] = self.__p00
            d['P(1|1)'] = self.__p11
            d['P_2D(0|0)'] = self.__p00_2d
            d['P_2D(1|1)'] = self.__p11_2d
        return pd.Series(d, index = d.keys())
    
    def curva_roc(self):
        if(self.__y_prob_unico.size > 1):
            curva_tnp = self.__curva_tnp
            curva_tvp = self.__curva_tvp
            auc = self.__auc
        else:
            curva_tnp = np.array([])
            curva_tvp = np.array([])
            auc = np.nan
        return curva_tnp, curva_tvp, auc
        
    def curva_revocacao(self):
        if(self.__y_prob_unico.size > 1):
            if(self.__num_div is not None and self.__qtd_unicos >= self.__num_div):
                y_prob_plot = [x for y in self.__interv.pares_minimo_maximo_discretizacao()[self.__y_prob_unico.astype(int)] for x in y] 
                curva_revoc0_plot = np.repeat(self.__curva_revoc0, 2)
                curva_revoc1_plot = np.repeat(self.__curva_revoc1, 2)
                pos_max = self.__interv.valores_medios_discretizacao()[self.__pos_max_dif]
            else:
                y_prob_plot = self.__y_prob_unico
                curva_revoc0_plot = self.__curva_revoc0
                curva_revoc1_plot = self.__curva_revoc1
                pos_max = self.__y_prob_unico[self.__pos_max_dif]
            ks = self.__ks
        else:
            y_prob_plot = np.array([])
            curva_revoc0_plot = np.array([])
            curva_revoc1_plot = np.array([])
            pos_max = np.nan
            ks = np.nan
        return y_prob_plot, curva_revoc0_plot, curva_revoc1_plot, pos_max, ks
        
    def curva_informacao(self):
        if(self.__y_prob_unico.size > 1):
            if(self.__num_div is not None and self.__qtd_unicos >= self.__num_div):
                y_prob_plot = [x for y in self.__interv.pares_minimo_maximo_discretizacao()[self.__y_prob_unico.astype(int)] for x in y]
                curva_ig_plot = np.repeat(self.__curva_ig, 2)
                pos_max = self.__interv.valores_medios_discretizacao()[self.__pos_max_ig]
                ig = self.__ig
                #Se eu quiser plotar o intervalo de prob "confiável" calculado com a informação 2D
                if(self.__y_prob_unico.size > 2):
                    pos_p0_aux = int(self.__vetor_p0_ig_2d[self.__pos_max_ig_2d])
                    pos_p1_aux = int(self.__vetor_p1_ig_2d[self.__pos_max_ig_2d])
                    p0_corte = self.__interv.valores_medios_discretizacao()[pos_p0_aux]
                    p1_corte = self.__interv.valores_medios_discretizacao()[pos_p1_aux]
                    ig_2d = self.__ig_2d
                else:
                    p0_corte = np.nan
                    p1_corte = np.nan
                    ig_2d = np.nan   
            else:
                y_prob_plot = self.__y_prob_unico
                curva_ig_plot = self.__curva_ig
                pos_max = self.__y_prob_unico[self.__pos_max_ig]
                ig = self.__ig
                if(self.__y_prob_unico.size > 2):
                    p0_corte = self.__vetor_p0_ig_2d[self.__pos_max_ig_2d]
                    p1_corte = self.__vetor_p1_ig_2d[self.__pos_max_ig_2d]
                    ig_2d = self.__ig_2d
                else:
                    p0_corte = np.nan
                    p1_corte = np.nan
                    ig_2d = np.nan
        else:
            y_prob_plot = np.array([])
            curva_ig_plot = np.array([])
            pos_max = np.nan
            ig = np.nan
            p0_corte = np.nan
            p1_corte = np.nan
            ig_2d = np.nan
        return y_prob_plot, curva_ig_plot, pos_max, ig, p0_corte, p1_corte, ig_2d
        
    def curva_informacao_2d(self):
        if(self.__y_prob_unico.size > 2):
            if(self.__num_div is not None and self.__qtd_unicos >= self.__num_div):
                x = [i for j in self.__interv.pares_minimo_maximo_discretizacao()[self.__vetor_p0_ig_2d.astype(int)] for i in j]
                y = [i for j in self.__interv.pares_minimo_maximo_discretizacao()[self.__vetor_p1_ig_2d.astype(int)] for i in j]
                x.extend(list(self.__interv.valores_medios_discretizacao()[self.__vetor_p0_ig_2d.astype(int)]))
                y.extend(list(self.__interv.valores_medios_discretizacao()[self.__vetor_p1_ig_2d.astype(int)]))
                z = np.repeat(self.__vetor_ig_2d, 2)
                z = np.append(z, self.__vetor_ig_2d)
                pos_p0_aux = int(self.__vetor_p0_ig_2d[self.__pos_max_ig_2d])
                pos_p1_aux = int(self.__vetor_p1_ig_2d[self.__pos_max_ig_2d])
                p0_corte = self.__interv.valores_medios_discretizacao()[pos_p0_aux]
                p1_corte = self.__interv.valores_medios_discretizacao()[pos_p1_aux]
                ig_2d = self.__ig_2d
            else:
                x = self.__vetor_p0_ig_2d
                y = self.__vetor_p1_ig_2d
                z = self.__vetor_ig_2d
                p0_corte = self.__vetor_p0_ig_2d[self.__pos_max_ig_2d]
                p1_corte = self.__vetor_p1_ig_2d[self.__pos_max_ig_2d]
                ig_2d = self.__ig_2d
            
        else:
            x = np.array([])
            y = np.array([])
            z = np.array([])
            p0_corte = np.nan
            p1_corte = np.nan
            ig_2d = np.nan
        return x, y, z, p0_corte, p1_corte, ig_2d
    
    def grafico_roc(self, roc_usual = True, figsize = [6, 4]):
        paleta_cores = sns.color_palette("colorblind")
        with sns.axes_style("whitegrid"):
            fig, axs = plt.subplots(1, 1, figsize = figsize)
            curva_tnp, curva_tvp, auc = self.curva_roc()
            if(roc_usual):
                axs.plot(1-curva_tnp, curva_tvp, color = paleta_cores[0], label = 'Curva ROC')
                axs.plot([0, 1], [0, 1], color = 'k', linestyle = '--', label = 'Linha de Ref.')
                axs.set_xlabel('Taxa de Falso Positivo')
            else:
                axs.plot(curva_tnp, curva_tvp, color = paleta_cores[0], label = 'Curva ROC')
                axs.plot([0, 1], [1, 0], color = 'k', linestyle = '--', label = 'Linha de Ref.')
                axs.set_xlabel('Taxa de Verdadeiro Negativo')
            plt.gcf().text(1, 0.5, 'AUC = ' + '%.2g' % auc, bbox = dict(facecolor = 'white', edgecolor = 'k', boxstyle = 'round'))
            axs.set_ylabel('Taxa de Verdadeiro Positivo')
            axs.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.show()
        
    def grafico_revocacao(self, figsize = [6, 4]): 
        paleta_cores = sns.color_palette("colorblind")
        with sns.axes_style("whitegrid"):
            fig, axs = plt.subplots(1, 1, figsize = figsize)
            y_prob_plot, curva_revoc0_plot, curva_revoc1_plot, pos_max, ks = self.curva_revocacao()
            axs.plot(y_prob_plot, curva_revoc0_plot, color = paleta_cores[0], alpha = 1.0, label = 'Revocação 0')
            axs.plot(y_prob_plot, curva_revoc1_plot, color = paleta_cores[1], alpha = 1.0, label = 'Revocação 1')
            axs.vlines(pos_max, 0, 1, color = 'k', linestyle = '--', label = 'Ponto KS')
            plt.gcf().text(1, 0.5, 'KS = ' + '%.2g' % ks, bbox = dict(facecolor = 'white', edgecolor = 'k', boxstyle = 'round'))
            axs.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            axs.set_xlabel('Probabilidade de Corte')
            axs.set_ylabel('Revocação')
            plt.show()
            
    def grafico_ks(self, figsize = [6, 4]): 
        paleta_cores = sns.color_palette("colorblind")
        with sns.axes_style("whitegrid"):
            fig, axs = plt.subplots(1, 1, figsize = figsize)
            y_prob_plot, curva_revoc0_plot, curva_revoc1_plot, pos_max, ks = self.curva_revocacao()
            axs.plot(y_prob_plot, curva_revoc0_plot - curva_revoc1_plot, color = paleta_cores[0], alpha = 1.0, label = 'Curva KS')
            axs.vlines(pos_max, 0, ks, color = 'k', linestyle = '--', label = 'Ponto KS')
            plt.gcf().text(1, 0.5, 'KS = ' + '%.2g' % ks, bbox = dict(facecolor = 'white', edgecolor = 'k', boxstyle = 'round'))
            axs.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            axs.set_xlabel('Probabilidade de Corte')
            axs.set_ylabel('Distância entre Revocações')
            plt.show()
    
    def grafico_informacao(self, mostrar_ig_2d = False, figsize = [6, 4]):
        paleta_cores = sns.color_palette("colorblind")
        with sns.axes_style("whitegrid"):
            fig, axs = plt.subplots(1, 1, figsize = figsize)
            y_prob_plot, curva_ig_plot, pos_max, ig, p0_corte, p1_corte, ig_2d = self.curva_informacao()
            axs.plot(y_prob_plot, curva_ig_plot, color = paleta_cores[0], label = 'Curva IG')
            axs.vlines(pos_max, 0, ig, color = 'k', linestyle = '--', label = 'Ganho Máx.')
            if(mostrar_ig_2d and ig_2d != np.nan):
                axs.vlines(p0_corte, 0, ig_2d, color = 'k', alpha = 0.5, linestyle = '--', label = 'Ganho Máx. 2D')
                axs.vlines(p1_corte, 0, ig_2d, color = 'k', alpha = 0.5, linestyle = '--')
                plt.gcf().text(1, 0.5, 'IG = ' + '%.2g' % ig + '\n' + 'IG 2D = ' + '%.2g' % ig_2d, 
                               bbox = dict(facecolor = 'white', edgecolor = 'k', boxstyle = 'round'))
                plt.gcf().text(1, 0.25, 'Prob Corte = ' + '%.2g' % pos_max + '\n\n' + 'Prob0 Corte = ' + '%.2g' % p0_corte + '\n' + 'Prob1 Corte = ' + '%.2g' % p1_corte, 
                               bbox = dict(facecolor = 'white', edgecolor = 'k', boxstyle = 'round'))
            else:
                plt.gcf().text(1, 0.5, 'IG = ' + '%.2g' % ig, bbox = dict(facecolor = 'white', edgecolor = 'k', boxstyle = 'round'))
                plt.gcf().text(1, 0.3, 'Prob Corte = ' + '%.2g' % pos_max, bbox = dict(facecolor = 'white', edgecolor = 'k', boxstyle = 'round'))
            axs.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            axs.set_xlabel('Probabilidade de Corte')
            axs.set_ylabel('Ganho de Informação')
            plt.show()
        
    def grafico_informacao_2d(self, plot_3d = True, figsize = [7, 6]):
        paleta_cores = sns.color_palette("colorblind")
        x, y, z, p0_corte, p1_corte, ig_2d = self.curva_informacao_2d()
        if(plot_3d):
            with sns.axes_style("whitegrid"):
                fig = plt.figure(figsize = figsize)
                axs = fig.add_subplot(111, projection='3d')
                #Constrói gradiente de uma cor até o branco (1,1,1) -> Lembrar que em RGB a mistura de todas as cores é que é o branco
                N = 256
                vals = np.ones((N, 4)) #A última componente (quarta) é o alpha que é o índice de transparência
                cor = paleta_cores[0]
                #Define as Cores RGB pelas componentes (no caso é o azul -> 0,0,255)
                vals[:, 0] = np.linspace(cor[0], 1, N)
                vals[:, 1] = np.linspace(cor[1], 1, N)
                vals[:, 2] = np.linspace(cor[2], 1, N)
                cmap = mpl.colors.ListedColormap(vals[::-1])
                axs.scatter(x, y, z, c = z, marker = 'o', cmap = cmap)
                axs.set_xlabel('Probabilidade de Corte 0')
                axs.set_ylabel('Probabilidade de Corte 1')
                axs.set_zlabel('Ganho de Informação')
                plt.show()
        else:
            with sns.axes_style("whitegrid"):
                fig, axs = plt.subplots(1, 1, figsize = figsize)
                #Faz uma mapa de cores com base em uma cor e mudando a transparência
                N = 256
                cor = paleta_cores[0]
                vals = np.ones((N, 4))
                vals[:, 0] = cor[0]
                vals[:, 1] = cor[1]
                vals[:, 2] = cor[2]
                cmap_linhas = mpl.colors.ListedColormap(vals[::-1])
                vals[:, 3] = np.linspace(0, 1, N)[::-1]
                cmap = mpl.colors.ListedColormap(vals[::-1])
                axs.tricontour(x, y, z, levels = 14, linewidths = 0.5, cmap = cmap_linhas)
                cntr = axs.tricontourf(x, y, z, levels = 14, cmap = cmap)
                cbar = plt.colorbar(cntr, ax = axs)
                cbar.ax.set_title('Ganho Info.')
                axs.scatter(p0_corte, p1_corte, color = 'k')
                axs.vlines(p0_corte, 0, p1_corte, color = 'k', alpha = 0.5, linestyle = '--')
                axs.hlines(p1_corte, 0, p0_corte, color = 'k', alpha = 0.5, linestyle = '--')
                plt.gcf().text(1, 0.8, 'IG 2D = ' + '%.2g' % ig_2d, bbox = dict(facecolor = 'white', edgecolor = 'k', boxstyle = 'round'))
                plt.gcf().text(1, 0.7, 'Prob0 Corte = ' + '%.2g' % p0_corte + '\n' + 'Prob1 Corte = ' + '%.2g' % p1_corte, 
                               bbox = dict(facecolor = 'white', edgecolor = 'k', boxstyle = 'round'))
                axs.set_xlabel('Probabilidade de Corte 0')
                axs.set_ylabel('Probabilidade de Corte 1')
                axs.set_xlim([min(x[0], y[0]), max(x[-1], y[-1])])
                axs.set_ylim([min(x[0], y[0]), max(x[-1], y[-1])])
                plt.show()

##############################

##############################

#Para funcionar direito, não pode haver nulos em y ou y_pred
class AletricasRegressao:
    
    def __init__(self, y, y_pred, y_ref = None, y2_ref = None, num_kendalltau = 10000):
        self.__y = y
        self.__y_pred = y_pred
        self.__y_ref = y_ref
        self.__num_kendalltau = num_kendalltau
        
        self.__qtd_tot = y.size
        
        self.__soma_preds = soma_vetor(y_pred)
        self.__soma_y = soma_vetor(y)
       
        self.__media_preds = self.__soma_preds/self.__qtd_tot
        self.__media_y = self.__soma_y/self.__qtd_tot
        
        self.__media_y2 = calcula_mse(y)
        
        self.__diff_y = diferenca_vetores(self.__y, self.__y_pred)
        self.__diff_ym = diferenca_vetores(self.__y, self.__media_y)
        if(y_ref is None):
            self.__diff_yr = self.__diff_ym
        else:
            self.__diff_yr = diferenca_vetores(self.__y, self.__y_ref)
        
        self.__mae = None
        self.__mse = None
        self.__rmse = None
        
        self.__rae = None
        self.__rse = None
        self.__rrse = None
     
        self.__rae_ref = None
        self.__rse_ref = None
        self.__rrse_ref = None
        
        self.__r1 = None
        self.__r2 = None
        self.__rr2 = None
        
        self.__r1_ref = None
        self.__r2_ref = None
        self.__rr2_ref = None
        
        self.__kendalltau_conc = None
        self.__kendalltau_disc = None
        self.__kendalltau = None
        
        self.__calcula_metricas()
        
        ### Métricas de Distribuição ###

        self.__y_unicos = np.unique(y)
        self.__y_acum = calcula_distribuicao_acumulada_pontos(y, self.__y_unicos)
        y_acum_ref = calcula_distribuicao_acumulada_normal(self.__media_y, np.std(y), self.__y_unicos)
        self.__y_pred_acum = calcula_distribuicao_acumulada_pontos(y_pred, self.__y_unicos)
        
        self.__ks = calcula_ks(self.__y_acum, self.__y_pred_acum)
        ks_ref = calcula_ks(self.__y_acum, y_acum_ref)
        self.__coef_ks = 1 - self.__ks/ks_ref
        
        if(y_ref is None or y2_ref is None):
            self.__coef_ks_ref = self.__coef_ks
        else:
            y_acum_ref = calcula_distribuicao_acumulada_normal(y_ref, math.sqrt(y2_ref - math.pow(y_ref, 2)), self.__y_unicos)
            ks_ref = calcula_ks(self.__y_acum, y_acum_ref)
            self.__coef_ks_ref = 1 - self.__ks/ks_ref
    
    def __calcula_mae(self):
        self.__mae = calcula_mae(self.__diff_y)
        
    def __calcula_mse(self):
        self.__mse = calcula_mse(self.__diff_y)
        
    def __calcula_rmse(self):
        self.__rmse = math.sqrt(self.__mse)
        
    def __calcula_rae(self):
        div = calcula_mae(self.__diff_ym)
        if(div > 0):
            self.__rae = self.__mae/div
        else:
            self.__rae = np.nan

        div = calcula_mae(self.__diff_yr)
        if(div > 0):
            self.__rae_ref = self.__mae/div
        else:
            self.__rae_ref = np.nan
            
    def __calcula_rse(self):
        div = calcula_mse(self.__diff_ym)
        if(div > 0):
            self.__rse = self.__mse/div
        else:
            self.__rse = np.nan

        div = calcula_mse(self.__diff_yr)
        if(div > 0):
            self.__rse_ref = self.__mse/div
        else:
            self.__rse_ref = np.nan
          
    def __calcula_rrse(self):
        if(np.isnan(self.__rse)):
            self.__rrse = np.nan
        else:
            self.__rrse = math.sqrt(self.__rse)

        if(np.isnan(self.__rse_ref)):
            self.__rrse_ref = np.nan
        else:
            self.__rrse_ref = math.sqrt(self.__rse_ref)
        
    def __calcula_r1(self):
        self.__r1 = 1 - self.__rae
        self.__r1_ref = 1 - self.__rae_ref
            
    def __calcula_r2(self):
        self.__r2 = 1 - self.__rse
        self.__r2_ref = 1 - self.__rse_ref
            
    def __calcula_rr2(self):
        self.__rr2 = 1 - self.__rrse
        self.__rr2_ref = 1 - self.__rrse_ref
    
    def __calcula_kendalltau(self):
    
        #Remove os valores absolutos e pega só a ordem de crescimento (transforma em ordinal)
        def transforma_ordinal(vetor):
            v_unicos, inds_inverso = np.unique(vetor, return_inverse = True)
            v_contador = np.arange(0, v_unicos.size)
            return v_contador[inds_inverso]
        y_ordinal = transforma_ordinal(self.__y)
        y_pred_ordinal = transforma_ordinal(self.__y_pred)

        #Agora queremos ver quanto a ordenação pela predição consegue ordenar o y_ordinal (coeficiente Kandall Tau modificado)
        pares_valores = np.array(list(zip(y_ordinal, y_pred_ordinal)))
        pares_indices = np.random.choice(y_ordinal.size, self.__num_kendalltau + 1, replace = True)
        vetor_bool = np.array([checa_ordenacao(pares_indices, pares_valores, i) for i in range(0, pares_indices.size-1)])
        self.__kendalltau_conc = soma_vetor(vetor_bool)/vetor_bool.size
        self.__kendalltau_disc = 1 - self.__kendalltau_conc
        self.__kendalltau = self.__kendalltau_conc - self.__kendalltau_disc
    
    def __calcula_metricas(self):
        if(self.__qtd_tot > 0):
            self.__calcula_mae()
            self.__calcula_mse()
            self.__calcula_rmse()
            
            self.__calcula_rae()
            self.__calcula_rse()
            self.__calcula_rrse()
            
            self.__calcula_r1()
            self.__calcula_r2()
            self.__calcula_rr2()
            
            if(self.__qtd_tot > 1 and self.__num_kendalltau > 0):
                self.__calcula_kendalltau()
    
    def valor_medias_alvo(self):
        return self.__media_y, self.__media_y2
    
    def curva_distribuicao_acumulada(self):
        return self.__y_unicos, self.__y_acum, self.__y_pred_acum, self.__ks
    
    def grafico_distribuicao_acumulada(self, figsize = [6, 4]): 
        paleta_cores = sns.color_palette("colorblind")
        with sns.axes_style("whitegrid"):
            fig, axs = plt.subplots(1, 1, figsize = figsize)
            y_unicos, y_acum, y_pred_acum, ks = self.curva_distribuicao_acumulada()
            axs.plot(y_unicos, y_acum, color = paleta_cores[0], alpha = 1.0, label = 'Real')
            axs.plot(y_unicos, y_pred_acum, color = paleta_cores[1], alpha = 1.0, label = 'Predito')
            plt.gcf().text(1, 0.5, 'KS = ' + '%.2g' % ks, bbox = dict(facecolor = 'white', edgecolor = 'k', boxstyle = 'round'))
            axs.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            axs.set_xlabel('Valores')
            axs.set_ylabel('Probabilidade Acumulada')
            plt.show()
    
    def valor_metricas(self, estatisticas_globais = True, metricas_ref = True, alga_signif = 0, conv_str = False):
        #Retorna um pd.Series com as metricas calculadas
        #Esse formato é bom para criar dataframes
        d = {}
        if(estatisticas_globais):
            d['QTD'] = self.__qtd_tot
            d['Soma_Alvo'] = self.__soma_y
            d['Soma_Pred'] = self.__soma_preds
            d['Media_Alvo'] = self.__media_y
            d['Media_Pred'] = self.__media_preds
        d['MAE'] = self.__mae
        d['MSE'] = self.__mse
        d['RMSE'] = self.__rmse
        d['RAE'] = self.__rae
        d['RSE'] = self.__rse
        d['RRSE'] = self.__rrse
        if(metricas_ref):
            d['RAE_ref'] = self.__rae_ref
            d['RSE_ref'] = self.__rse_ref
            d['RRSE_ref'] = self.__rrse_ref
        d['R1'] = self.__r1
        d['R2'] = self.__r2
        d['RR2'] = self.__rr2
        if(metricas_ref):
            d['R1_ref'] = self.__r1_ref
            d['R2_ref'] = self.__r2_ref
            d['RR2_ref'] = self.__rr2_ref
        d['KendallTau_Conc'] = self.__kendalltau_conc
        d['KendallTau_Disc'] = self.__kendalltau_disc
        d['KendallTau'] = self.__kendalltau
        d['KS'] = self.__ks
        d['Coef_KS'] = self.__coef_ks
        d['Coef_KS_ref'] = self.__coef_ks_ref
        if(alga_signif > 0):
            str_conv = '%.' + str(alga_signif) + 'g'
            for key in d.keys():
                try:
                    d[key] = float(str_conv % d[key])
                except:
                    d[key] = np.nan
        if(conv_str):
            for key in d.keys():
                try:
                    d[key] = str(d[key])
                except:
                    d[key] = np.nan
        return pd.Series(d, index = d.keys())
        
##############################

##############################

#Para funcionar direito, não pode haver nulos em y ou y_pred
class AletricasDistribuicaoProbabilidade:
    
    def __init__(self, y, matriz_y_prob, discretizador, y_ref = None, y2_ref = None):
        self.__y = y
        self.__discretizador = discretizador
        self.__y_ref = y_ref
        self.__y2_ref = y2_ref
        self.__qtd_tot = y.size
        self.__y_prob = np.sum(matriz_y_prob, axis = 0)/self.__qtd_tot

        self.__soma_y = soma_vetor(y)
        self.__media_y = self.__soma_y/self.__qtd_tot
        self.__media_y2 = calcula_mse(y)
        
        self.__media_preds = np.sum(self.__y_prob*self.__discretizador.media)
        self.__soma_preds = self.__media_preds*self.__qtd_tot
        
        ### Métricas de Distribuição ###

        self.__y_unicos = np.unique(y)
        self.__y_acum = calcula_distribuicao_acumulada_pontos(y, self.__y_unicos)
        y_acum_ref = calcula_distribuicao_acumulada_normal(self.__media_y, np.std(y), self.__y_unicos)
        self.__y_pred_acum = calcula_distribuicao_acumulada(self.__y_prob, self.__discretizador, self.__y_unicos)
        
        self.__ks = calcula_ks(self.__y_acum, self.__y_pred_acum)
        ks_ref = calcula_ks(self.__y_acum, y_acum_ref)
        self.__coef_ks = 1 - self.__ks/ks_ref
        
        if(y_ref is None or y2_ref is None):
            self.__coef_ks_ref = self.__coef_ks
        else:
            y_acum_ref = calcula_distribuicao_acumulada_normal(y_ref, math.sqrt(y2_ref - math.pow(y_ref, 2)), self.__y_unicos)
            ks_ref = calcula_ks(self.__y_acum, y_acum_ref)
            self.__coef_ks_ref = 1 - self.__ks/ks_ref
    
    def valor_medias_alvo(self):
        return self.__media_y, self.__media_y2
    
    def curva_distribuicao_acumulada(self):
        return self.__y_unicos, self.__y_acum, self.__y_pred_acum, self.__ks
    
    def grafico_distribuicao_acumulada(self, figsize = [6, 4]): 
        paleta_cores = sns.color_palette("colorblind")
        with sns.axes_style("whitegrid"):
            fig, axs = plt.subplots(1, 1, figsize = figsize)
            y_unicos, y_acum, y_pred_acum, ks = self.curva_distribuicao_acumulada()
            axs.plot(y_unicos, y_acum, color = paleta_cores[0], alpha = 1.0, label = 'Real')
            axs.plot(y_unicos, y_pred_acum, color = paleta_cores[1], alpha = 1.0, label = 'Predito')
            plt.gcf().text(1, 0.5, 'KS = ' + '%.2g' % ks, bbox = dict(facecolor = 'white', edgecolor = 'k', boxstyle = 'round'))
            axs.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            axs.set_xlabel('Valores')
            axs.set_ylabel('Probabilidade Acumulada')
            plt.show()
    
    def valor_metricas(self, estatisticas_globais = True, metricas_ref = True, alga_signif = 0, conv_str = False):
        #Retorna um pd.Series com as metricas calculadas
        #Esse formato é bom para criar dataframes
        d = {}
        if(estatisticas_globais):
            d['QTD'] = self.__qtd_tot
            d['Soma_Alvo'] = self.__soma_y
            d['Soma_Pred'] = self.__soma_preds
            d['Media_Alvo'] = self.__media_y
            d['Media_Pred'] = self.__media_preds
        d['KS'] = self.__ks
        d['Coef_KS'] = self.__coef_ks
        d['Coef_KS_ref'] = self.__coef_ks_ref
        if(alga_signif > 0):
            str_conv = '%.' + str(alga_signif) + 'g'
            for key in d.keys():
                try:
                    d[key] = float(str_conv % d[key])
                except:
                    d[key] = np.nan
        if(conv_str):
            for key in d.keys():
                try:
                    d[key] = str(d[key])
                except:
                    d[key] = np.nan
        return pd.Series(d, index = d.keys())