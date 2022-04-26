# -*- coding: utf-8 -*-
"""
Temporary file to test the CPGD

Created on Fri Jul 30 11:03:38 2021

@author: basti
"""

import numpy as np
from skimage import io
# from tqdm import tqdm

import matplotlib.pyplot as plt
# from mpl_toolkits.axes_grid1 import make_axes_locatable
# from matplotlib.animation import FuncAnimation
import torch

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import cudavenant

__saveFig__ = False
__saveVid__ = False
__savePickle__ = False


# GPU acceleration if needed
device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
# print("[+] Using {} device".format(device))


N_ECH = 2**7
X_GAUCHE = 0
X_DROIT = 1
SIGMA = 0.07

domain = cudavenant.Domain2D(X_GAUCHE, X_DROIT, N_ECH, SIGMA, dev=device)
domain_cpu = domain.to('cpu')
domain = domain.to(device)

amp = torch.tensor([1, 2, 3])
xpos = torch.tensor([[0.2, 0.2], [0.7, 0.6], [0.5, 0.8]])
m_ax0 = cudavenant.Mesure2D(amp, xpos)

y0 = m_ax0.kernel(domain)
y0_cpu = y0.to('cpu')
bru = cudavenant.Bruits(0, 1e-5, 'unif')
y = m_ax0.acquisition(domain, N_ECH, bru)
y_cpu = y.to('cpu')
plt.imshow(y_cpu)


# CPGD computation
𝜈, 𝜈_itere, r_itere, θ_itere, nrj = cudavenant.CPGD(y, domain, λ=1e0, α=4e-2, 
                                                    β=1e1, nIter=250, 
                                                    nParticles=8, 
                                                    noyau='gaussien', 
                                                    obj='acquis')

plt.figure()
plt.plot(nrj, '--')
plt.grid()
plt.title('Gradient flow loss', fontsize=22)
plt.xlabel('Number of iterations', fontsize=18)
plt.ylabel(r'$T_\lambda(\nu_k)$', fontsize=20)


plt.figure()
plt.scatter(𝜈.x[:, 0], 𝜈.x[:, 1], label=r'$\nu_\infty$', marker='o',
            c='red')
plt.axis('square')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.colorbar()
plt.legend()


plt.figure()
cont = plt.contourf(domain_cpu.X, domain_cpu.Y, y_cpu, 100)
for c in cont.collections:
    c.set_edgecolor("face")
plt.scatter(m_ax0.x[:, 0], m_ax0.x[:, 1], label='$m_{a_0,x_0}$', marker='o',
            c='white', s=m_ax0.a*domain_cpu.N_ech)
for l in range(len(𝜈.a)):
    plt.plot(θ_itere[:, l, 0], θ_itere[:, l, 1], 'r', linewidth=1.2)
plt.scatter(𝜈.x[:, 0], 𝜈.x[:, 1], label='$\\nu_t$', c='r',
            s=𝜈.a*domain_cpu.N_ech)

plt.axis('square')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.colorbar()
plt.legend(loc=1)


cudavenant.cpgd_anim(y_cpu, m_ax0, 𝜈_itere, θ_itere, domain_cpu, 'mp4')


#%% Curve measure

n_curve = 30
𝛾_grid = torch.linspace(0.23, 0.77, n_curve)
𝛾_x = torch.stack((𝛾_grid, 0.2*torch.ones(n_curve)), dim=1)
𝛾_x = torch.cat((torch.stack((𝛾_grid, 0.8*torch.ones(n_curve)), dim=1), 𝛾_x))
𝛾_x = torch.cat((torch.stack((0.2*torch.ones(n_curve), 𝛾_grid), dim=1), 𝛾_x))
# 𝛾_x = torch.cat((torch.stack((torch.flip(𝛾_grid, [-1]), 𝛾_grid), dim=1),
#                   𝛾_x))
𝛾_a = torch.ones(𝛾_x.shape[0])
R_𝛾 = cudavenant.Mesure2D(𝛾_a, 𝛾_x)

y_curve = R_𝛾.kernel(domain)
y_curve_cpu = y_curve.to('cpu')
plt.imshow(y_curve_cpu)


print(f'[+] Computing CPGD for curves on {device}')
𝜈, 𝜈_itere, r_itere, θ_itere, nrj = cudavenant.CPGD(y_curve, domain, λ=1e1, 
                                                    α=5e-2, β=1e0, nIter=1500, 
                                                    nParticles=25, 
                                                    noyau='gaussien')


plt.figure()
plt.plot(nrj, '--')
plt.grid()
plt.title('Gradient flow loss', fontsize=22)
plt.xlabel('Number of iterations', fontsize=18)
plt.ylabel(r'$T_\lambda(\nu_k)$', fontsize=20)


plt.figure()
plt.scatter(𝜈.x[:, 0], 𝜈.x[:, 1], label=r'$\nu_\infty$', marker='o',
            c='red')
plt.axis('square')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.colorbar()
plt.legend()


plt.figure()
cont = plt.contourf(domain_cpu.X, domain_cpu.Y, y_curve_cpu, 100)
for c in cont.collections:
    c.set_edgecolor("face")
plt.scatter(R_𝛾.x[:, 0], R_𝛾.x[:, 1], label='$m_{a_0,x_0}$', marker='o',
            c='white', s=0.1*R_𝛾.a*domain.N_ech)
for l in range(len(𝜈.a)):
    plt.plot(θ_itere[:, l, 0], θ_itere[:, l, 1], 'r', linewidth=1.2)
plt.scatter(𝜈.x[:, 0], 𝜈.x[:, 1], label='$\\nu_t$', c='r', s=𝜈.a*domain.N_ech)

plt.axis('square')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.colorbar()
plt.legend(loc=1)

# cpgd_anim(y_curve, R_𝛾, 𝜈_itere, θ_itere, domain, 'mp4')

#%% Real data measure


stream = io.imread('datasets/sofi_filaments/tubulin_noiseless_noBg.tif')
pile = torch.from_numpy(np.array(stream, dtype='float64'))  # [:,20:52,20:52]
pile_max = torch.max(pile)
# pile /= torch.sqrt(pile_max)
pile /= pile_max

y_bar_cpu = torch.mean(pile.float(), 0)
y_bar = (y_bar_cpu).to(device)

N_ECH = y_bar.shape[0]
X_GAUCHE = 0
X_DROIT = 1
FWMH = 2.2875 / N_ECH
SIGMA = FWMH / (2 * np.sqrt(2*np.log(2)))
domain = cudavenant.Domain2D(X_GAUCHE, X_DROIT, N_ECH, SIGMA, dev=device)
domain_cpu = cudavenant.Domain2D(X_GAUCHE, X_DROIT, N_ECH, SIGMA)


print(f'[+] Computing CPGD for real data on {device}')
𝜈, 𝜈_itere, r_itere, θ_itere, nrj = cudavenant.CPGD(y_bar, domain, λ=5e1, 
                                                    α=5e-2, β=1e0, nIter=500, 
                                                    nParticles=10, 
                                                    noyau='gaussien')


plt.figure()
plt.plot(nrj, '--')
plt.grid()
plt.title('Gradient flow loss', fontsize=22)
plt.xlabel('Number of iterations', fontsize=18)
plt.ylabel(r'$T_\lambda(\nu_k)$', fontsize=20)


plt.figure()
plt.scatter(𝜈.x[:, 0], 𝜈.x[:, 1], label=r'$\nu_\infty$', marker='o',
            c='red')
plt.axis('square')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.colorbar()
plt.legend()


plt.figure()
cont = plt.contourf(domain_cpu.X, domain_cpu.Y, y_bar_cpu, 100)
for c in cont.collections:
    c.set_edgecolor("face")
for l in range(len(𝜈.a)):
    plt.plot(θ_itere[:, l, 0], θ_itere[:, l, 1], 'r', linewidth=1.2)
plt.scatter(𝜈.x[:, 0], 𝜈.x[:, 1], label='$\\nu_t$', c='r', s=𝜈.a*domain.N_ech)

plt.axis('square')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.colorbar()
plt.legend(loc=1)
