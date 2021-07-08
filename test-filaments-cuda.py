#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 15:30:44 2021

@author: Bastien https://github.com/XeBasTeX

Test sur filaments avec acquisitions SOFI inspirées de SMLM
http://bigwww.epfl.ch/smlm/challenge2016/datasets/MT4.N2.HD/Data/data.html

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

import cudavenant


# GPU acceleration if needed
device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
print("[Test] Using {} device".format(device))

# Tic 
tic = time.time()

# Charger pile et cumulants
stream = io.imread('sofi_filaments/tubulin_noiseless_highBg.tif')
pile = torch.from_numpy(np.array(stream, dtype='float64')) #[:,20:52,20:52]
pile_max = torch.max(pile)
# pile /= torch.sqrt(pile_max)
pile /= pile_max
pile += torch.normal(0, 5e-2, size=pile.shape)

emitters_loc = np.genfromtxt('sofi_filaments/emitters_noiseless_lowBg.csv',
                             delimiter=',')
emitters_loc = torch.fliplr(torch.from_numpy(emitters_loc))[0:-1:3] / 64
m_ax0 = cudavenant.Mesure2D(torch.ones(emitters_loc.shape[0]),
                            emitters_loc)

# Calculer les cumulants
y_bar_cpu = torch.mean(pile.float(), 0)
y_bar = (y_bar_cpu).to(device)
R_y = cudavenant.covariance_pile(pile).to(device)


N_ECH = y_bar.shape[0]
X_GAUCHE = 0
X_DROIT = 1
FWMH = 2.2875 / N_ECH
SIGMA = FWMH / (2 * np.sqrt(2*np.log(2)))
domain = cudavenant.Domain2D(X_GAUCHE, X_DROIT, N_ECH, SIGMA, dev=device)
domain_cpu = cudavenant.Domain2D(X_GAUCHE, X_DROIT, N_ECH, SIGMA)

q = 2**3
super_domain = domain.super_resolve(q, SIGMA/4)

lambda_cov = 1e-6
lambda_moy = 1e-3
iteration = 160

#%% Calcul SFW

(m_cov, nrj_cov, mes_cov) = cudavenant.SFW(R_y, domain, regul=lambda_cov,
                                           nIter=iteration, mesParIter=True,
                                           obj='covar', printInline=True)

(m_moy, nrj_moy, mes_moy) = cudavenant.SFW(y_bar - y_bar.min(), domain,
                                           regul=lambda_moy,
                                           nIter=iteration, mesParIter=True,
                                           obj='acquis', printInline=True)

print(f'm_cov : {m_cov.N} Diracs')
print(f'm_moy : {m_moy.N} Diracs')


if __savePickle__:
    with open('pickle/m_moy.pkl', 'wb') as output:
        pickle.dump(m_moy, output, pickle.HIGHEST_PROTOCOL)
    with open('pickle/m_cov.pkl', 'wb') as output:
        pickle.dump(m_cov, output, pickle.HIGHEST_PROTOCOL)

if m_cov.N > 0:
    certificat_V_cov = cudavenant.etak(m_cov, R_y, domain, lambda_cov,
                                       obj='covar').to('cpu')
    m_cov.to('cpu')
    cudavenant.plot_experimental(m_cov, domain_cpu, y_bar_cpu , 
                                 nrj_cov[:123],
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
    cudavenant.gif_experimental(y_bar_cpu, mes_moy[:102], super_domain, 
                                cross=False, video='mp4', title='filaments-moy',
                                obj='acquis')


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

#%%

# Evacuer données cheloues
# ind = torch.where(m_moy.a > 0.1)
# m_moy = cudavenant.Mesure2D(m_moy.a[ind], m_moy.x[ind])
# m_cov = mes_cov[147]


# #%% sfw sur chaque image

# N_ECH = y_bar.shape[0]
# X_GAUCHE = 0
# X_DROIT = 1
# FWMH = 2.2875 / N_ECH
# SIGMA = FWMH / (2 * np.sqrt(2*np.log(2)))
# domain = cudavenant.Domain2D(X_GAUCHE, X_DROIT, N_ECH, SIGMA, dev=device)
# domain_cpu = cudavenant.Domain2D(X_GAUCHE, X_DROIT, N_ECH, SIGMA)

# q = 2**2
# super_domain = domain.super_resolve(q, SIGMA/2)

# lambda_moy = 1e-3
# iteration = 120
# m_list = []

# for i in range(5):
#     (m_moy, nrj_moy, mes_moy) = cudavenant.SFW(pile[i,:].float() , domain,
#                                                regul=lambda_moy,
#                                                nIter=iteration, mesParIter=True,
#                                                obj='acquis', printInline=True)
#     print(f'm_moy : {m_moy.N} Diracs')
#     m_list.append(m_moy)

# for i in range(5):
#     certificat_V_moy = cudavenant.etak(m_moy, y_bar, domain, lambda_moy,
#                                        obj='acquis').to('cpu')
#     m_moy.to('cpu')
#     cudavenant.plot_experimental(m_list[i], domain_cpu, pile[i,:].float(), nrj_moy,
#                                  certificat_V_moy, obj='acquis',
#                                  saveFig=False)



tac = time.time() - tic
print(f"Elapsed time: {tac:.2f} seconds")


# nrj_cov = torch.zeros(len(mes_cov))
# for i in range(len(mes_cov)):
#     nrj_cov[i] = mes_cov[i].energie(domain, R_y, lambda_cov)


# #%%

# import matplotlib.pyplot as plt

# super_domain = domain.super_resolve(q, SIGMA/5)

# fig = plt.figure()
# ax2 = fig.add_subplot(111)
# ax2.set_aspect('equal', adjustable='box')
# # ax2.contourf(dom.X, dom.Y, m_list[k].kernel(dom), 100,
# #              cmap='seismic')

# ax2.imshow(m_cov.kernel(super_domain), cmap='hot', interpolation='nearest')

# plt.axis('off')
# plt.tight_layout()
# plt.savefig('cudavenant-final.pdf', format='pdf', dpi=1000,
#             bbox_inches='tight', pad_inches=0.03)

# psf = cudavenant.gaussienne_2D(domain.X, domain.Y, SIGMA, undivide=True).numpy()
# scipy.io.savemat('psf.mat', dict(psf=psf, st=0))

#%%


def SNR(signal, acquisition):
    return 10 * torch.log10(torch.norm(signal) / torch.norm(acquisition - signal))


gt = m_ax0.kernel(domain_cpu)
print(SNR(pile[0,:], y_bar))