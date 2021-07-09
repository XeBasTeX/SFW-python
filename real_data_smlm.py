# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 07:54:36 2021

@author: blaville
"""

__saveFig__ = False
__saveVid__ = True
__savePickle__ = False


import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import time

from skimage import io
import pickle, imageio
import numpy as np
import torch
import matplotlib.pyplot as plt

import cudavenant


# GPU acceleration if needed
device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
print("[+] Using {} device".format(device))

# Tic 
tic = time.time()

# Charger pile et cumulants
stream = io.imread('real_data_smlm/real_data_smlm.tif')
pile = torch.from_numpy(np.array(stream, dtype='float64'))
pile_max = torch.max(pile)
pile /= pile_max
imageio.mimwrite('real_data_smlm/real_data_smlm_cropped.tif',pile)


# Calculer les cumulants
y_bar_cpu = torch.mean(pile.float(), 0)
y_bar = (y_bar_cpu).to(device)
R_y = cudavenant.covariance_pile(pile).to(device)


N_ECH = y_bar.shape[0]
X_GAUCHE = 0
X_DROIT = 1
FWMH = 351.8 / 100 / N_ECH
SIGMA = FWMH / (2 * np.sqrt(2*np.log(2)))

domain = cudavenant.Domain2D(X_GAUCHE, X_DROIT, N_ECH, SIGMA, dev=device)
domain_cpu = cudavenant.Domain2D(X_GAUCHE, X_DROIT, N_ECH, SIGMA)

q = 2**3
super_domain = domain.super_resolve(q, SIGMA/3)

lambda_cov = 1e-6
lambda_moy = 2e-4
iteration = 50


# #%% Calcul SFW

# (m_cov, nrj_cov, mes_cov) = cudavenant.SFW(R_y, domain, regul=lambda_cov,
#                                            nIter=iteration, mesParIter=True,
#                                            obj='covar', printInline=True)

# (m_moy, nrj_moy, mes_moy) = cudavenant.SFW(y_bar - y_bar.min(), domain,
#                                            regul=lambda_moy,
#                                            nIter=iteration, mesParIter=True,
#                                            obj='acquis', printInline=True)

# print(f'm_cov : {m_cov.N} Diracs')
# print(f'm_moy : {m_moy.N} Diracs')


# if __savePickle__:
#     with open('pickle/real_data_smlm_moy.pkl', 'wb') as output:
#         pickle.dump(m_moy, output, pickle.HIGHEST_PROTOCOL)
#     with open('pickle/real_data_smlmm_cov.pkl', 'wb') as output:
#         pickle.dump(m_cov, output, pickle.HIGHEST_PROTOCOL)

# if m_cov.N > 0:
#     certificat_V_cov = cudavenant.etak(m_cov, R_y, domain, lambda_cov,
#                                        obj='covar').to('cpu')
#     m_cov.to('cpu')
#     cudavenant.plot_experimental(m_cov, domain_cpu, y_bar_cpu, 
#                                  nrj_cov,
#                                  certificat_V_cov, 
#                                  obj='covar',
#                                  saveFig=__saveFig__, 
#                                  title='real-data-smlm-covar')
#     if __saveFig__:
#         m_cov.export(super_domain, title="real_data_smlm_cov")


# if m_moy.N > 0:
#     certificat_V_moy = cudavenant.etak(m_moy, y_bar, domain, lambda_moy,
#                                        obj='acquis').to('cpu')
#     m_moy.to('cpu')
#     cudavenant.plot_experimental(m_moy, domain_cpu, y_bar_cpu, nrj_moy,
#                                  certificat_V_moy, 
#                                  obj='acquis',
#                                  saveFig=__saveFig__, 
#                                  title='real-data-smlm-moy')
#     if __saveFig__:
#         m_moy.export(super_domain, title="real_data_smlm_moy")


# if __saveVid__:
#     cudavenant.gif_experimental(y_bar_cpu, mes_cov, super_domain, 
#                                 cross=False, video='mp4', 
#                                 title='real_data_smlm_cov')
#     cudavenant.gif_experimental(y_bar_cpu, mes_moy, super_domain, 
#                                 cross=False, video='mp4', 
#                                 title='real_data_smlm_moy',
#                                 obj='acquis')


# tac = time.time() - tic
# print(f"[+] Elapsed time: {tac:.2f} seconds")

#%%

# m_cov = cudavenant.divide_and_conquer(pile, domain, obj='covar')
m_moy = cudavenant.divide_and_conquer(pile, domain, obj='acquis',
                                      quadrant_size=64, nIter=150)

if __saveFig__:
    # m_cov.export(super_domain, title="real_data_smlm_cov_reign")
    m_moy.export(super_domain, title="real_data_smlm_moy_reign")

tac = time.time() - tic
print(f"[+] Elapsed time: {tac:.2f} seconds")

