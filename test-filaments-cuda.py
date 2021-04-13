#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 15:30:44 2021

@author: Bastien https://github.com/XeBasTeX

Test sur filaments avec acquisitions SOFI inspirées de SMLM
http://bigwww.epfl.ch/smlm/challenge2016/datasets/MT4.N2.HD/Data/data.html

"""


__saveFig__ = True
__saveVid__ = False


from skimage import io
import numpy as np
import torch
import pickle

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import cudavenant


# GPU acceleration if needed
device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
print("[Test] Using {} device".format(device))

# Charger pile et cumulants
stream = io.imread('sofi_filaments/tubulin_noiseless_lowBg.tif')
pile = torch.from_numpy(np.array(stream, dtype='float64')) [:,32:,:32]
pile_max = torch.max(pile)
# pile /= torch.sqrt(pile_max)
pile /= pile_max

emitters_loc = np.genfromtxt('sofi_filaments/emitters_noiseless_lowBg.csv',
                             delimiter=',')
emitters_loc = torch.fliplr(torch.from_numpy(emitters_loc))[0:-1:3] / 64
m_ax0 = cudavenant.Mesure2D(torch.ones(emitters_loc.shape[0]),
                            emitters_loc)

# Calculer les cumulants
y_bar_cpu = torch.mean(pile.float(), 0)
y_bar = y_bar_cpu.to(device)
R_y = cudavenant.covariance_pile(pile).to(device)


#%%

N_ECH = y_bar.shape[0]
X_GAUCHE = 0
X_DROIT = 1
FWMH = 2.2875 / N_ECH
SIGMA = FWMH / (2 * np.sqrt(2*np.log(2)))
domain = cudavenant.Domain2D(X_GAUCHE, X_DROIT, N_ECH, SIGMA, dev=device)
domain_cpu = cudavenant.Domain2D(X_GAUCHE, X_DROIT, N_ECH, SIGMA)

q = 2**3
super_domain = domain.super_resolve(q)

lambda_cov = 1e-6
lambda_moy = 1e-2
iteration = 100

(m_cov, nrj_cov, mes_cov) = cudavenant.SFW(R_y, domain, regul=lambda_cov,
                                           nIter=iteration, mesParIter=True,
                                           obj='covar', printInline=True)

(m_moy, nrj_moy, mes_moy) = cudavenant.SFW(y_bar , domain,
                                           regul=lambda_moy,
                                           nIter=iteration, mesParIter=True,
                                           obj='acquis', printInline=True)

print(f'm_cov : {m_cov.N} Diracs')
print(f'm_moy : {m_moy.N} Diracs')

if m_cov.N > 0:
    certificat_V_cov = cudavenant.etak(m_cov, R_y, domain, lambda_cov,
                                       obj='covar').to('cpu')
    m_cov.to('cpu')
    cudavenant.plot_experimental(m_cov, domain_cpu, y_bar_cpu, nrj_cov,
                                 certificat_V_cov, obj='covar',
                                 saveFig=__saveFig__, title='filaments-covar')
if m_moy.N > 0:
    certificat_V_moy = cudavenant.etak(m_moy, y_bar, domain, lambda_moy,
                                       obj='acquis').to('cpu')
    m_moy.to('cpu')
    cudavenant.plot_experimental(m_moy, domain_cpu, y_bar_cpu, nrj_moy,
                                 certificat_V_moy, obj='acquis',
                                 saveFig=__saveFig__, title='filaments-moy')

if __saveVid__:
    cudavenant.gif_experimental(y_bar, mes_cov, super_domain, cross=True,
                                video='gif', title='filaments-cov')
    cudavenant.gif_experimental(y_bar_cpu, mes_moy, super_domain, cross=True,
                                video='gif', title='filaments-moy')


# torch.cuda.empty_cache()

# with open('m_moy.pkl', 'wb') as output:
#     pickle.dump(m_moy, output, pickle.HIGHEST_PROTOCOL)

# #%%

# FOND = 0.01
# SIGMA_BRUITS = 1e-7
# TYPE_BRUITS = 'unif'
# bruits_t = cudavenant.Bruits(FOND, SIGMA_BRUITS, TYPE_BRUITS)

# T_ECH = 100
# pile = cudavenant.pile_aquisition(m_ax0, domain, bruits_t, T_ECH,
#                                   dev=device) #[:,:32, 32:]
# y_bar = pile.mean(0)
# y_bar_cpu = y_bar.to('cpu')
# cov_pile = cudavenant.covariance_pile(pile)
# R_y = cov_pile
