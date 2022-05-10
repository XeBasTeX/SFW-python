# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 13:40:11 2022

@author: basti
"""


__saveFig__ = False
__saveVid__ = False
__savePickle__ = False


from skimage import io
import numpy as np
import torch
import pickle

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import time

import matplotlib.pyplot as plt

import cudavenant


# GPU acceleration if needed
device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
print("[Test] Using {} device".format(device))

# Tic 
tic = time.time()

# Charger pile et cumulants
stream = io.imread('datasets/ma_tirf/angle2.tiff')
pile = torch.from_numpy(np.array(stream, dtype='float64')) [:,83:123,410:450]
# plt.imshow(pile[:,310:350, 460:500].mean(0), cmap='hot')
plt.imsave('fig/reconstruction/mean_gretsi.png', pile.mean(0), cmap='hot')
plt.imshow(pile[0,:])

#%%

pile_max = torch.max(pile)
pile /= pile_max


# Calculer les cumulants
y_bar_cpu = torch.mean(pile.float(), 0)
y_bar = (y_bar_cpu).to(device)
R_y = cudavenant.covariance_pile(pile).to(device)


N_ECH = y_bar.shape[0]
X_GAUCHE = 0
X_DROIT = 1
FWMH = 2.9205 / 1.06 / N_ECH
SIGMA = FWMH / (2 * np.sqrt(2*np.log(2)))
domain = cudavenant.Domain2D(X_GAUCHE, X_DROIT, N_ECH, SIGMA, dev=device)
domain_cpu = cudavenant.Domain2D(X_GAUCHE, X_DROIT, N_ECH, SIGMA)

q = 2**2
super_domain = domain.super_resolve(q, SIGMA/3)

lambda_cov = 1e-6
lambda_moy = 1e-3   
iteration = 85

#%% Calcul SFW

R_y_corr = R_y - torch.eye(N_ECH**2, N_ECH**2) * (torch.diag(R_y).min())
(m_cov, nrj_cov, mes_cov) = cudavenant.SFW(R_y_corr, domain, regul=lambda_cov,
                                           nIter=iteration, mesParIter=True,
                                           obj='covar', printInline=True)

(m_moy, nrj_moy, mes_moy) = cudavenant.SFW(y_bar - y_bar.min(), domain,
                                           regul=lambda_moy,
                                           nIter=iteration, mesParIter=True,
                                           obj='acquis', printInline=True)

print(f'm_cov : {m_cov.N} Diracs')
print(f'm_moy : {m_moy.N} Diracs')


if __savePickle__:
    torch.save(mes_cov, 'saved_objects/mes_cov_gretsi_2.pkl')
    torch.save(mes_moy, 'saved_objects/mes_moy_gretsi_2.pkl')

if m_cov.N > 0:
    certificat_V_cov = cudavenant.etak(m_cov, R_y, domain, lambda_cov,
                                       obj='covar').to('cpu')
    m_cov.to('cpu')
    cudavenant.plot_experimental(m_cov, domain_cpu, y_bar_cpu , 
                                 nrj_cov,
                                 certificat_V_cov, 
                                 obj='covar',
                                 saveFig=__saveFig__, 
                                 title='filaments-covar-global')
if m_moy.N > 0:
    certificat_V_moy = cudavenant.etak(m_moy, y_bar, domain, lambda_moy,
                                       obj='acquis').to('cpu')
    m_moy.to('cpu')
    cudavenant.plot_experimental(m_moy, domain_cpu, y_bar_cpu, nrj_moy,
                                 certificat_V_moy, 
                                 obj='acquis',
                                 saveFig=__saveFig__, 
                                 title='filaments-moy-global')

if __saveVid__:
    cudavenant.gif_experimental(y_bar_cpu, mes_cov, super_domain, 
                                cross=False, video='mp4', 
                                title='filaments-cov')
    cudavenant.gif_experimental(y_bar_cpu, mes_moy, super_domain, 
                                cross=False, video='mp4', 
                                title='filaments-moy', obj='acquis')




# Evacuer données cheloues
ind = torch.where(m_moy.a > 1)
m_moy_threshold = cudavenant.Mesure2D(m_moy.a[ind], m_moy.x[ind])
m_moy_threshold.export(super_domain, title='moy_gretsi', obj='moy')
m_cov.export(super_domain, title='cov_gretsi', obj='cov')

# torch.cuda.empty_cache()

# mes_cov = pickle.load(open("pickle/mes_covar.pkl", "rb" ))
# mes_moy = pickle.load(open("pickle/mes_acquis.pkl", "rb" ))

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
