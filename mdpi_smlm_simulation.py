# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 09:48:29 2021

@author: basti
"""

__saveFig__ = False
__saveVid__ = False
__savePickle__ = True


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
# stream = io.imread('mdpi_smlm/MT4.N2.HD-2D-Exp.tif')
pile = torch.from_numpy(np.array(stream, dtype='float64'))
pile_max = torch.max(pile)
pile /= pile_max
pile = pile.float()



#%%


# Calculer les cumulants
y_bar_cpu = torch.mean(pile.float(), 0)
y_bar = (y_bar_cpu).to(device)


N_ECH = y_bar.shape[0]
X_GAUCHE = 0
X_DROIT = 1
FWMH = 351.8 / 100 / N_ECH
SIGMA = FWMH / (2 * np.sqrt(2*np.log(2)))

domain = cudavenant.Domain2D(X_GAUCHE, X_DROIT, N_ECH, SIGMA, dev=device)
domain_cpu = cudavenant.Domain2D(X_GAUCHE, X_DROIT, N_ECH, SIGMA)

q = 2**4
super_domain = domain.super_resolve(q, SIGMA/5)

lambda_moy = 2e-4
iteration = 160

N_stack = pile.size(0)
m_ax = cudavenant.Mesure2D()

print(f"[+] Intializing the stack: computation on {device}")

for i in range(N_stack):
    acquisition = pile[i,:] - pile[i,:].min()
    (m_moy, nrj_moy) = cudavenant.SFW(acquisition, domain,
                                      regul=lambda_moy,
                                      nIter=iteration, mesParIter=False,
                                      obj='acquis', printInline=False)
    m_ax += m_moy
    print(i)
    if i % 50 == 0:
        print(f"[+] Algorithm has computed {100*i/N_stack:.2f}%"
              + " of the stack")

print("[+] Algorithm has finished \n")
print(f'm_ax : {m_ax.N} Diracs')


# # Evacuer données cheloues
# ind = torch.where(m_ax.a > 2)
# m_seuil = cudavenant.Mesure2D(m_ax.a[ind], m_ax.x[ind])

if __savePickle__:
    with open('pickle/real_data_smlm_x.pkl', 'wb') as output:
        pickle.dump(m_ax.x, output, pickle.HIGHEST_PROTOCOL)


if __saveFig__:
    m_ax.export(super_domain, title="real_data_smlm_mdpi")

tac = time.time() - tic
print(f"[+] Elapsed time: {tac:.2f} seconds")

