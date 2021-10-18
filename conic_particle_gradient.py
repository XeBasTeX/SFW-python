# -*- coding: utf-8 -*-
"""
Temporary file to test the CPGD

Created on Fri Jul 30 11:03:38 2021

@author: basti
"""

__saveFig__ = False
__saveVid__ = False
__savePickle__ = False


import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import time

import numpy as np
import torch
from scipy import integrate
import matplotlib.pyplot as plt

import cudavenant



# GPU acceleration if needed
device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
print("[+] Using {} device".format(device))

N_ECH = 16
X_GAUCHE = 0
X_DROIT = 1
SIGMA = 0.1

domain = cudavenant.Domain2D(X_GAUCHE, X_DROIT, N_ECH, SIGMA, dev=device)

amp = torch.tensor([1, 2])
xpos = torch.tensor([[0.2, 0.2], [0.7, 0.6]])
m_ax0 = cudavenant.Mesure2D(amp, xpos)

y = m_ax0.kernel(domain)
plt.imshow(y)


def phiAdjointSimps(acquis, mes_pos, domain, noyau='gaussien'):
    eta = np.empty(len(mes_pos))
    if noyau == 'gaussien':
        for i in range(len(mes_pos)):
            x = mes_pos[i]
            eta[i] = integrate.simps(acquis*cudavenant.gaussienne(x-domain), x=domain)
        return eta
    if noyau == 'fourier':
        for i in range(len(mes_pos)):
            raise TypeError("Fourier is not implemented")
        return eta
    else:
        raise TypeError


def gradPhiAdjointSimps(acquis, mes_pos, domain, noyau='gaussien'):
    eta = np.empty(len(mes_pos))
    if noyau == 'gaussien':
        for i in range(len(mes_pos)):
            x = mes_pos[i]
            eta[i] = integrate.simps(acquis*cudavenant.grad_gaussienne(x-domain),
                                     x=domain)
        return eta
    if noyau == 'fourier':
        raise TypeError("Fourier is not implemented")
    else:
        raise TypeError


def CPGD(acquis, dom, λ=1e-5, nIter=5, mes_init=None, mesParIter=False,
        obj='covar', printInline=True):
    """
    This is an implementation of Conic Particle Gradient Descent
    
    Only the mirror retractation is currently implemented
    
    The initialisation is currently restricted to the acquisition grid 
    (you can't initialize with any mesh you choose, even if an adaptative
     grid makes more sense)

    Parameters
    ----------
    acquis : TYPE
        DESCRIPTION.
    dom : TYPE
        DESCRIPTION.
    regul : TYPE, optional
        DESCRIPTION. The default is 1e-5.
    nIter : TYPE, optional
        DESCRIPTION. The default is 5.
    mes_init : TYPE, optional
        DESCRIPTION. The default is None.
    mesParIter : TYPE, optional
        DESCRIPTION. The default is False.
    obj : TYPE, optional
        DESCRIPTION. The default is 'covar'.
    printInline : TYPE, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    None.

    """

    N_init = acquis.size(0)
    α = 0.1
    β = 1
    amp_unif = torch.ones(N_init**2)
    grid_unif = torch.linspace(dom.x_gauche, dom.x_droit,
                               N_init)
    mesh_unif = torch.meshgrid(grid_unif, grid_unif)
    pos_unif = torch.stack((mesh_unif[0].flatten(), mesh_unif[1].flatten()),
                           dim=1)
    µ_k = cudavenant.Mesure2D(amp_unif, pos_unif) # initialising measure
    µ_k.show(dom, acquis)
    
    for k in range(nIter):
        r_k = torch.sqrt(µ_k.a)
        x_k = µ_k.x

        η_k_plus = cudavenant.etak(µ_k, acquis, dom, λ, obj='acquis')
        η_k_plus.requires_grad = True
        grad_η_k_plus = 1

        r_k_plus = r_k * torch.exp(2*α*λ*(η_k_plus.flatten() - 1))
        x_k_plus = x_k + β*λ*grad_η_k_plus

        µ_k = cudavenant.Mesure2D(r_k_plus**2, x_k_plus)
    return µ_k


m_ax = CPGD(y, domain)
