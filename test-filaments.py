#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 15:30:44 2021

@author: Bastien (https://github.com/XeBasTeX)

Test sur filaments avec acquisitions SOFI inspirées de SMLM
http://bigwww.epfl.ch/smlm/challenge2016/datasets/MT4.N2.HD/Data/data.html

"""


__saveFig__ = False
__saveVid__ = False


import matplotlib.pyplot as plt
import scipy.io
from skimage import io
import numpy as np

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import covenant


pile = np.array(io.imread('sofi_filaments/tubulin_noiseless_lowBg.tif'),
                dtype='float64')[:, 16:32, 16:32]
# pile = np.array(io.imread('sofi_filaments/tubulin_noiseless_lowBg.tif'),
#                 dtype='float64')

# pile = pile / pile.shape[-1]
pile = pile / np.max(pile)**(1/2)
# pile = pile + np.random.normal(0, 1e-1, size=pile.shape)
pile_moy = np.mean(pile, axis=0)

emitters_loc = np.genfromtxt('sofi_filaments/emitters_noiseless_lowBg.csv',
                             delimiter=',')
emitters_loc = np.fliplr(emitters_loc) / 64
m_ax0 = covenant.Mesure2D(np.ones(emitters_loc.shape[0]), emitters_loc)

y_bar = pile_moy
R_y = covenant.covariance_pile(pile, y_bar)
print('[+] Covariance calculée')

#%% Calcul effectif

N_ECH = pile_moy.shape[0]  # Taux d'échantillonnage
X_GAUCHE = 0
X_DROIT = 1
FWMH = 2.2875 / N_ECH
SIGMA = FWMH / (2 * np.sqrt(2*np.log(2)))
domain = covenant.Domain2D(X_GAUCHE, X_DROIT, N_ECH, SIGMA)

q = 2**3
super_domain = domain.super_resolve(q)


iteration = 60
lambda_cov = 1e-7
lambda_moy = 1e-4  # s'arrête à 103 Diracs

# test_obj = 'acquis'
# test_acquis = y_bar

# test_obj = 'covar'
# test_acquis = R_y


# SIGMA_BRUITS = 1e-2
# SIGMA_TARGET = 1.5 * SIGMA_BRUITS

# (m_top, nrj_top, lambda_top) = covenant.homotopy(test_acquis, domain,
#                                                   SIGMA_TARGET,
#                                                   obj=test_obj,
#                                                   nIter=80, c=1)

(m_cov, nrj_cov, mes_cov) = covenant.SFW(R_y, domain, regul=lambda_cov,
                                         nIter=iteration, mesParIter=True,
                                         obj='covar', printInline=True)

(m_moy, nrj_moy, mes_moy) = covenant.SFW(y_bar, domain,
                                         regul=lambda_moy,
                                         nIter=60 , mesParIter=True,
                                         obj='acquis', printInline=False)

print(f'm_cov : {m_cov.N} Diracs')
print(f'm_moy : {m_moy.N} Diracs')
if m_cov.N > 0:
    certificat_V_cov = covenant.etak(m_cov, R_y, domain, lambda_cov,
                                     obj='covar')
    covenant.plot_experimental(m_cov, domain, y_bar, nrj_cov,
                               certificat_V_cov, obj='covar',
                               saveFig=__saveFig__, title='filaments-covar')
if m_moy.N > 0:
    certificat_V_moy = covenant.etak(m_moy, y_bar, domain, lambda_moy,
                                     obj='acquis')
    covenant.plot_experimental(m_moy, domain, y_bar, nrj_moy,
                               certificat_V_moy, obj='acquis',
                               saveFig=__saveFig__, title='filaments-moy')

if __saveVid__:
    covenant.gif_experimental(y_bar, mes_cov, super_domain, cross=True,
                              video='gif', title='filaments-cov')
    covenant.gif_experimental(y_bar, mes_moy, super_domain, cross=True,
                          video='gif', title='filaments-moy')


if m_cov.N > 0:
    covenant.compare_covariance(m_cov, R_y, domain)



# #%% Test depuis la position des émetteurs

# import covenant

# N_ECH = 16  # Taux d'échantillonnage
# X_GAUCHE = 0
# X_DROIT = 1
# FWMH = 2.2875 / N_ECH
# SIGMA = FWMH / (2 * np.sqrt(2*np.log(2)))
# domain = covenant.Domain2D(X_GAUCHE, X_DROIT, N_ECH, SIGMA)

# FOND = 2.0
# SIGMA_BRUITS = 3e-1
# TYPE_BRUITS = 'gauss'
# bruits_t = covenant.Bruits(FOND, SIGMA_BRUITS, TYPE_BRUITS)


# m_ax0 = covenant.Mesure2D([8, 10, 6, 7, 9],
#                  [[0.2, 0.23], [0.90, 0.95], [0.33, 0.82], [0.30, 0.30],
#                   [0.23, 0.38]])

# pile = covenant.pile_aquisition(m_ax0, domain, bruits_t, 500)
# pile = pile
# pile_moy = np.mean(pile, axis=0)

# y_bar = pile_moy
# R_y = covenant.covariance_pile(pile, y_bar)
# print('[+] Covariance calculée')

# (m_cov, nrj_cov, mes_cov) = covenant.SFW(R_y, domain, regul=lambda_cov,
#                                          nIter=iteration, mesParIter=True,
#                                          obj='covar', printInline=True)

# (m_moy, nrj_moy, mes_moy) = covenant.SFW(y_bar, domain,
#                                          regul=lambda_moy,
#                                          nIter=20 , mesParIter=True,
#                                          obj='acquis', printInline=False)

# print(f'm_cov : {m_cov.N} Diracs')
# print(f'm_moy : {m_moy.N} Diracs')
# if m_cov.N > 0:
#     certificat_V_cov = covenant.etak(m_cov, R_y, domain, lambda_cov,
#                                      obj='covar')
#     covenant.plot_experimental(m_cov, domain, y_bar, nrj_cov,
#                                certificat_V_cov, obj='covar',
#                                saveFig=__saveFig__, title='filaments-covar')
# if m_moy.N > 0:
#     certificat_V_moy = covenant.etak(m_moy, y_bar, domain, lambda_moy,
#                                      obj='acquis')
#     covenant.plot_experimental(m_moy, domain, y_bar, nrj_moy,
#                                certificat_V_moy, obj='acquis',
#                                saveFig=__saveFig__, title='filaments-moy')


