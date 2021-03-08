#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 10:40:26 2021

@author: Bastien (https://github.com/XeBasTeX)
"""

__saveFig__ = False
__saveVid__ = False


import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
import covenant


# np.random.seed(90)

N_ECH = 2**4  # Taux d'échantillonnage
X_GAUCHE = 0
X_DROIT = 1
GRID = np.linspace(X_GAUCHE, X_DROIT, N_ECH)
X, Y = np.meshgrid(GRID, GRID)
SIGMA = 1e-1

FOND = 2.0
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

# test_obj = 'acquis'
# test_acquis = y_bar

test_obj = 'covar'
test_acquis = R_y

(m_top, nrj_top, lambda_top) = covenant.homotopy(test_acquis, domain, 
                                                 SIGMA_BRUITS, obj=test_obj,
                                                 nIter=2*m_ax0.N, c=1)

(m_cov, nrj_cov, mes_cov) = covenant.SFW(test_acquis, domain, regul=1e-5,
                                         nIter=m_ax0.N, mesParIter=True,
                                         obj=test_obj, printInline=False)
if m_top.N == 0:
    dist_x_top = np.inf
else:
    try:
        dist_x_top = covenant.wasserstein_metric(m_top, m_ax0)
    except ValueError:
        dist_x_top = np.inf

print(f'm_top : {m_top.N} Diracs')
print(f'm_ax0 : {m_ax0.N} Diracs')
print(fr'Dist W_1 des x de top : {dist_x_top:.3f}')
if m_top.N > 0:
    certificat_V_top = covenant.etak(m_top, test_acquis, domain, lambda_top,
                                 obj=test_obj)
    covenant.plot_results(m_top, m_ax0, domain, bruits_t, y_bar, nrj_top,
                          certificat_V_top, obj=test_obj, saveFig=__saveFig__,
                          title='homotopie-covar-certificat-2d')


#%% SFW sur CBLasso


test_obj = 'covar'
test_acquis = R_y
lambda_regul = 4e-5 # Param de relaxation pour SFW R_y
iteration = m_ax0.N


(m_cbl, sigma_cbl, nrj_cbl) = covenant.concomitant_SFW(test_acquis, domain,
                                          regul=lambda_regul,
                                          nIter=iteration, mesParIter=False,
                                          obj=test_obj, printInline=False)
if m_top.N == 0:
    dist_x_top = np.inf
else:
    try:
        dist_x_top = covenant.wasserstein_metric(m_top, m_ax0)
    except ValueError:
        dist_x_top = np.inf

print(f'm_cbl : {m_cbl.N} Diracs')
print(f'm_ax0 : {m_ax0.N} Diracs')
print(fr'Dist W_1 des x de top : {dist_x_top:.3f}')
if m_top.N > 0:
    certificat_V_cbl = covenant.etak(m_cbl, test_acquis, domain, lambda_regul,
                                     obj=test_obj)
    covenant.plot_results(m_cbl, m_ax0, domain, bruits_t, y_bar, nrj_cbl,
                          certificat_V_cbl, obj=test_obj, saveFig=__saveFig__,
                          title='cbl-covar-certificat-2d')






#%% SFW sur SOFItool

from skimage import io


# Test sur données réelles
pile_sofi = np.array(io.imread('sofi/siemens_star.tiff'), dtype='float64')
pile_sofi_moy = np.mean(pile_sofi, axis=0)
T_ech = pile_sofi.shape[0]
VRAI_N_ECH = pile_sofi.shape[-1]

bas_red = 6
haut_red = 26
reduc = VRAI_N_ECH/(haut_red - bas_red)

emitters_loc = np.fliplr(np.genfromtxt('sofi/emitters.csv', delimiter=','))
emitters_loc /= VRAI_N_ECH
emitters_loc_test = [el for el in emitters_loc
                     if bas_red/VRAI_N_ECH <
                     np.linalg.norm(el, np.inf) <
                     haut_red/VRAI_N_ECH]

emitters_loc_test = np.vstack(emitters_loc_test) - bas_red/VRAI_N_ECH
emitters_loc_test = reduc * emitters_loc_test
m_ax0 = covenant.Mesure2D(np.ones(emitters_loc_test.shape[0]),
                          emitters_loc_test)
# plot_results(m_ax0, domaine, nrj_cov, certif_V, y)

pile_sofi_test = pile_sofi[:, bas_red:haut_red, bas_red:haut_red]
pile_sofi_test = pile_sofi_test / np.max(pile_sofi_test)
pile_sofi_test_moy = np.mean(pile_sofi_test, axis=0)

FWMH = 2.2875 / VRAI_N_ECH
SIFMA = FWMH / (2*np.sqrt(2*np.log(2)))
N_ECH = pile_sofi_test.shape[-1]  # Taux d'échantillonnage
X_GAUCHE = 0
X_DROIT = 1
GRID = np.linspace(X_GAUCHE, X_DROIT, N_ECH)
X, Y = np.meshgrid(GRID, GRID)
domaine = covenant.Domain2D(X_GAUCHE, X_DROIT, N_ECH, GRID, X, Y, SIFMA)

y_bar = np.mean(pile_sofi_test, axis=0)
R_y = covenant.covariance_pile(pile_sofi_test, y_bar)
SIGMA_BRUITS = 1e-5

# test_acquis = R_y
# test_obj = 'covar'

test_obj = 'acquis'
test_acquis = y_bar

(m_top, nrj_top, lambda_top) = covenant.homotopy(test_acquis, domaine, 
                                                 SIGMA_BRUITS, obj=test_obj,
                                                 nIter=5, c=1)
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
    certificat_V_top = covenant.etak(m_top, test_acquis, domaine, lambda_top,
                                 obj=test_obj)
    covenant.plot_results(m_top, m_ax0, domaine, bruits_t, y_bar, nrj_top,
                          certificat_V_top, obj=test_obj, saveFig=__saveFig__,
                          title='homotopie-covar-certificat-2d')


#% SFW classique sur SOFItool

# Pour Q_\lambda(y) et P_\lambda(y_bar) à 3
# lambda_regul = 2e-4  # Param de relaxation pour SFW R_y
# lambda_regul2 = 1e-1  # Param de relaxation pour SFW y_moy

# Pour Q_\lambda(y) et P_\lambda(y_bar) à 9
lambda_regul = 3e-7 # Param de relaxation pour SFW R_y
lambda_regul2 = 1e-4 # Param de relaxation pour SFW y_moy

# # # Pour Q_0(y_0) P_0(y_0)
# lambda_regul = 1e-8 # Param de relaxation pour SFW R_y
# lambda_regul2 = 5e-5 # Param de relaxation pour SFW y_moy

iteration = m_ax0.N


(m_cov, nrj_cov, mes_cov) = covenant.SFW(R_y, domaine,
                                          regul=lambda_regul,
                                          nIter=iteration, mesParIter=True,
                                          obj='covar')
(m_moy, nrj_moy, mes_moy) = covenant.SFW(y_bar, domaine,
                                          regul=lambda_regul2,
                                          nIter=iteration, mesParIter=True,
                                          obj='acquis')

print(f'm_Mx : {m_cov.N} Diracs')
print(f'm_ax : {m_moy.N} Diracs')
print(f'm_ax0 : {m_ax0.N} Diracs')

certificat_V = covenant.etak(m_cov, R_y, domaine, lambda_regul,
                              obj='covar')
certificat_V_moy = covenant.etak(m_moy, y_bar, domaine, lambda_regul2,
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
    covenant.plot_results(m_cov, m_ax0, domaine, bruits_t, y_bar, nrj_cov,
                          certificat_V, saveFig=__saveFig__)
if m_moy.N > 0:
    covenant.plot_results(m_moy, m_ax0, domaine, bruits_t, y_bar, nrj_moy,
                          certificat_V_moy, title='covar-moy-certificat-2d',
                          obj='acquis', saveFig=__saveFig__)

if __saveVid__:
    covenant.gif_pile(pile, m_ax0, y_bar, domain)
    if m_cov.N > 0:
        covenant.gif_results(y_bar, m_ax0, mes_cov, domain)



# #%%

# from scipy.spatial.distance import cdist

# def merge_spikes(mes, tol=2e-1):
#     mat_dist = cdist(mes.x, mes.x)
#     idx_spurious = np.array([])
#     list_x = np.array([])
#     list_a = np.array([])
#     for j in range(mes.N):
#         for i in range(j):
#             if mat_dist[i,j] < tol:
#                 coord = [int(i), int(j)]
#                 idx_spurious = np.append(idx_spurious, np.array(coord))
#     idx_spurious = idx_spurious.reshape((int(len(idx_spurious)/2), 2))
#     idx_spurious = idx_spurious.astype(int)
#     print(idx_spurious)
#     if idx_spurious.size == 0:
#         return mes
#     else:
#         cancelled = []
#         for i in range(mes.N):
#             if i in cancelled or i in idx_spurious[:,0]:
#                 cancelled = np.append(cancelled, i)
#             else:
#                 if list_x.size == 0:
#                     list_x = np.vstack([mes.x[i]])
#                 else:
#                     list_x = np.vstack([list_x, mes.x[i]])
#                 list_a = np.append(list_a, mes.a[i])
#         return covenant.Mesure2D(list_a, list_x)


# m_test = covenant.Mesure2D([0.1,0.2,0.3,0.4], [[0.7,0.5], [0.7,0.5],
#                                                [0.3,0.5],[1.7,1.9]])
# m_cut = merge_spikes(m_ax0)


# plt.figure(figsize=(12, 12))
# plt.scatter(m_ax0.x[:, 0], m_ax0.x[:, 1], marker='x',
#             label='GT spikes', s=400)
# plt.scatter(m_cut.x[:, 0], m_cut.x[:, 1], marker='+',
#             label='Recovered spikes', s=400)
