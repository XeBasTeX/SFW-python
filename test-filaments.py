#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 15:30:44 2021

@author: Bastien (https://github.com/XeBasTeX)
"""

__saveFig__ = False
__saveVid__ = False


import covenant
import matplotlib.pyplot as plt
import scipy.io
from skimage import io
import numpy as np

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))


# Test sur données PALM/STORM
# http://bigwww.epfl.ch/smlm/challenge2016/datasets/MT4.N2.HD/Data/data.html

pile = np.array(io.imread('sofi_filaments/tubulin_noiseless_lowBg.tif'),
                dtype='float64')[:, 16:32, 16:32]
# pile = np.array(io.imread('sofi_filaments/tubulin_noiseless_lowBg.tif'),
#                 dtype='float64')

# pile = pile / pile.shape[-1]
pile = pile / np.max(pile)
pile_moy = np.mean(pile, axis=0)
y_bar = pile_moy
R_y = covenant.covariance_pile(pile, y_bar)
print('[+] Covariance calculée')

#%% Calcul effectif

N_ECH = pile_moy.shape[0]  # Taux d'échantillonnage
X_GAUCHE = 0
X_DROIT = 1
GRID = np.linspace(X_GAUCHE, X_DROIT, N_ECH)
X, Y = np.meshgrid(GRID, GRID)
FWMH = 2.2875 / N_ECH
SIGMA = FWMH / (2 * np.sqrt(2*np.log(2)))
domain = covenant.Domain2D(X_GAUCHE, X_DROIT, N_ECH, GRID, X, Y, SIGMA)


iteration = 10
lambda_cov = 1e-8
lambda_moy = 1e-4  # s'arrête à 103 Diracs

# test_obj = 'acquis'
# test_acquis = y_bar

# test_obj = 'covar'
# test_acquis = R_y


SIGMA_BRUITS = 1e-2
SIGMA_TARGET = 1.5 * SIGMA_BRUITS

# (m_top, nrj_top, lambda_top) = covenant.homotopy(test_acquis, domain,
#                                                   SIGMA_TARGET,
#                                                   obj=test_obj,
#                                                   nIter=80, c=1)

(m_cov, nrj_cov, mes_cov) = covenant.SFW(R_y, domain, regul=lambda_cov,
                                          nIter=5, mesParIter=True,
                                          obj='covar', printInline=True)

(m_moy, nrj_moy, mes_moy) = covenant.SFW(y_bar, domain,
                                          regul=lambda_moy,
                                          nIter=iteration, mesParIter=True,
                                          obj='acquis', printInline=True)

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


# plt.figure(figsize=(12, 4))
# plt.subplot(121)
# plt.imshow(m_cov.covariance_kernel(domain), cmap='gray')
# plt.colorbar()
# plt.title(r'$\Lambda(m_{M,x})$', fontsize=40)
# plt.subplot(122)
# plt.imshow(R_y, cmap='gray')
# plt.colorbar()
# plt.title(r'$R_y$', fontsize=40)

if __saveVid__:
    covenant.gif_experimental(y_bar, mes_moy, domain, video='gif',
                              title='filaments-moy')
