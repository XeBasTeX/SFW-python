#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 10:40:26 2021

@author: Bastien (https://github.com/XeBasTeX)
"""

__saveFig__ = False
__saveVid__ = False


import numpy as np
import matplotlib.pyplot as plt
import covenant


# np.random.seed(80)

N_ECH = 2**4  # Taux d'échantillonnage
X_GAUCHE = 0
X_DROIT = 1
GRID = np.linspace(X_GAUCHE, X_DROIT, N_ECH)
X, Y = np.meshgrid(GRID, GRID)
SIGMA = 1e-1

FOND = 5.0
SIGMA_BRUITS = 3e-1
TYPE_BRUITS = 'gauss'


domain = covenant.Domain2D(X_GAUCHE, X_DROIT, N_ECH, GRID, X, Y, SIGMA)
bruits_t = covenant.Bruits(FOND, SIGMA_BRUITS, TYPE_BRUITS)
m_ax0 = covenant.mesure_aleatoire(9, domain)

T_ECH = 100
pile = covenant.pile_aquisition(m_ax0, domain, bruits_t, T_ECH)
y_bar = np.mean(pile, axis=0)
R_y = covenant.covariance_pile(pile, y_bar)
R_x = m_ax0.covariance_kernel(domain)


# Homotopie
def homotopy(acquis, dom, sigma_target, c=2, nIter=10, obj='covar'):
    lambda_k = np.linalg.norm(covenant.phiAdjoint(acquis, dom, obj=obj),
                              np.inf)/(np.sqrt(acquis.shape[0])*np.linalg.norm(acquis) )
    mesure_k = covenant.Mesure2D()
    for k in range(nIter):
        print(f"* lambda_{k} = {lambda_k}")
        (mesure_k, nrj_k) = covenant.SFW(acquis, dom, regul=lambda_k, 
                                         nIter=nIter, obj=obj,
                                         printInline=False)
        residual = covenant.phi(mesure_k, dom, obj=obj) - acquis
        sigma_k = np.std(residual)
        if sigma_k < sigma_target:
            print(f"[!] Condition d'arrêt homotopie : sigma = {sigma_k:.4f}")
            return(mesure_k, nrj_k, lambda_k)
        else:
            mx_eta = np.max(covenant.etak(mesure_k, acquis, dom, lambda_k,
                                          obj=obj))
            lambda_k *= mx_eta / (1 + c)
    print('[!] Fin boucle homotopie')
    return(mesure_k, nrj_k, lambda_k)


# test_obj = 'acquis'
# test_acquis = y_bar

test_obj = 'covar'
test_acquis = R_y

(m_top, nrj_top, lambda_top) = homotopy(test_acquis, domain, SIGMA_BRUITS,
                                        obj=test_obj, nIter=m_ax0.N, c=2)
if m_top.N == 0:
    dist_x_top = np.inf
else:
    try:
        dist_x_top = covenant.wasserstein_metric(m_top, m_ax0)
    except ValueError:
        dist_x_top = np.inf

print(fr'Dist W_1 des x de top : {dist_x_top:.3f}')
print(f'm_top : {m_top.N} Diracs')
print(f'm_ax0 : {m_ax0.N} Diracs')
if m_top.N > 0:
    certificat_V_top = covenant.etak(m_top, test_acquis, domain, lambda_top,
                                 obj=test_obj)
    covenant.plot_results(m_top, m_ax0, domain, bruits_t, y_bar, nrj_top,
                          certificat_V_top, saveFig=__saveFig__, obj=test_obj)


#%% SFW classique

# Pour Q_\lambda(y) et P_\lambda(y_bar) à 3
lambda_regul = 2e-4  # Param de relaxation pour SFW R_y
lambda_regul2 = 1e-1  # Param de relaxation pour SFW y_moy

# # Pour Q_\lambda(y) et P_\lambda(y_bar) à 9
# lambda_regul = 4e-8 # Param de relaxation pour SFW R_y
# lambda_regul2 = 5e-5 # Param de relaxation pour SFW y_moy

# # Pour Q_0(y_0) P_0(y_0)
# lambda_regul = 1e-8 # Param de relaxation pour SFW R_y
# lambda_regul2 = 5e-5 # Param de relaxation pour SFW y_moy

iteration = m_ax0.N


(m_cov, nrj_cov, mes_cov) = covenant.SFW(R_y, domain,
                                         regul=lambda_regul,
                                         nIter=iteration, mesParIter=True,
                                         obj='covar')
(m_moy, nrj_moy, mes_moy) = covenant.SFW(y_bar, domain,
                                         regul=lambda_regul2,
                                         nIter=iteration, mesParIter=True,
                                         obj='acquis')

print(f'm_Mx : {m_cov.N} Diracs')
print(f'm_ax : {m_moy.N} Diracs')
print(f'm_ax0 : {m_ax0.N} Diracs')

certificat_V = covenant.etak(m_cov, R_y, domain, lambda_regul, obj='covar')
certificat_V_moy = covenant.etak(m_moy, y_bar, domain, lambda_regul2,
                                 obj='acquis')

if __saveFig__:
    plt.figure(figsize=(12, 4))
    plt.subplot(121)
    plt.imshow(m_cov.covariance_kernel(domain))
    plt.colorbar()
    plt.title(r'$\Lambda(m_{M,x})$', fontsize=40)
    plt.subplot(122)
    plt.imshow(R_y)
    plt.colorbar()
    plt.title(r'$R_y$', fontsize=40)


# Métrique de déconvolution : distance de Wasserstein
try:
    dist_x_cov = covenant.wasserstein_metric(m_cov, m_ax0)
except ValueError:
    dist_x_cov = np.inf
try:
    dist_x_moy = covenant.wasserstein_metric(m_moy, m_ax0)
except ValueError:
    dist_x_moy = np.inf

print(fr'Dist W_1 des x de Q_\lambda : {dist_x_cov:.3f}')
print(fr'Dist W_1 des x de P_\lambda : {dist_x_moy:.3f}')


# Afficher les résultats
y_simul = m_cov.kernel(domain)
if m_cov.N > 0:
    covenant.plot_results(m_cov, m_ax0, domain, bruits_t, y_bar, nrj_cov,
                          certificat_V, saveFig=__saveFig__)
if m_moy.N > 0:
    covenant.plot_results(m_moy, m_ax0, domain, bruits_t, y_bar, nrj_moy,
                          certificat_V_moy, title='covar-moy-certificat-2d',
                          obj='acquis', saveFig=__saveFig__)

if __saveVid__:
    covenant.gif_pile(pile, m_ax0, y_bar, domain)
    if m_cov.N > 0:
        covenant.gif_results(y_bar, m_ax0, mes_cov, domain)
