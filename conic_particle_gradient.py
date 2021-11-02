# -*- coding: utf-8 -*-
"""
Temporary file to test the CPGD

Created on Fri Jul 30 11:03:38 2021

@author: basti
"""

from tqdm import tqdm
import cudavenant
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.animation import FuncAnimation
import torch

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))


__saveFig__ = False
__saveVid__ = False
__savePickle__ = False



# GPU acceleration if needed
device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"
# print("[+] Using {} device".format(device))



def cpgd_anim(acquis, mes_zer, m_itere, theta_itere, dom, video='gif', 
              dev=device):
    X = dom.X
    Y = dom.Y
    N_ech = dom.N_ech
    fig = plt.figure(figsize=(10, 10))

    ax = fig.add_subplot(111)
    ax.set_aspect('equal', adjustable='box')
    cont = ax.contourf(X, Y, acquis, 100, cmap='seismic')
    for c in cont.collections:
        c.set_edgecolor("face")
    # divider = make_axes_locatable(ax)  # pour paramétrer colorbar
    # cax = divider.append_axes("right", size="5%", pad=0.15)
    # fig.colorbar(cont, cax=cax)
    ax.set_xlabel('X', fontsize=25)
    ax.set_ylabel('Y', fontsize=25)
    ax.set_title('Acquisition $y$', fontsize=35)

    plt.tight_layout()

    def animate(k):
        if k >= len(𝜈_itere):
            # On fige l'animation pour faire une pause à la fin
            return
        else:
            # if k % 2 == 0:
            #     pass
            print(k)
            ax.clear()
            ax.set_aspect('equal', adjustable='box')

            cont = ax.contourf(X, Y, acquis, 100, cmap='seismic')
            for c in cont.collections:
                c.set_edgecolor("face")
            # divider = make_axes_locatable(ax)  # pour paramétrer colorbar
            # cax = divider.append_axes("right", size="5%", pad=0.25)
            # fig.colorbar(cont, ax=cax)
            for l in range(len(m_itere[k].a)):
                ax.plot(theta_itere[:k, l, 0], theta_itere[:k, l, 1], 'orange', 
                         linewidth=1.4)
            ax.scatter(mes_zer.x[:, 0], mes_zer.x[:, 1], marker='x',
                        s=N_ech, label='GT spikes')
            ax.scatter(m_itere[k].x[:, 0], m_itere[k].x[:, 1], marker='+',
                        s=2*N_ech, c='orange', label='Recovered spikes')
            ax.set_xlabel('X', fontsize=25)
            ax.set_ylabel('Y', fontsize=25)
            ax.set_title(f'Reconstruction at iterate = {k}', fontsize=35)
            ax.legend(loc=1, fontsize=20)
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            # plt.tight_layout()

    anim = FuncAnimation(fig, animate, interval=20, frames=len(m_itere)+3,
                          blit=False)

    plt.draw()
    if video == "mp4":
        anim.save('fig/anim/anim-cpgd-2d.mp4')
    elif video == "gif":
        anim.save('fig/anim/anim-cpgd-2d.gif')
    else:
        raise ValueError('Unknown video format')
    return fig


def phiAdjointSimps(acquis, mes_pos, dom, obj='acquis', noyau='gaussien', 
                    dev=device):
    eta = torch.zeros(len(mes_pos))
    eta = eta.to(dev)
    if noyau == 'gaussien':
        for i in range(len(mes_pos)):
            x = mes_pos[i]
            gauss = cudavenant.gaussienne_2D(x[0] - dom.X, x[1] - dom.Y, SIGMA)
            integ_x = torch.trapz(acquis * gauss, dom.X_grid)
            eta[i] = torch.trapz(integ_x, dom.X_grid)
        return eta
    if noyau == 'fourier':
        for i in range(len(mes_pos)):
            raise TypeError("Fourier is not implemented")
        return eta
    else:
        raise TypeError("Unknown kernel")


def gradPhiAdjointSimps(acquis, mes_pos, dom, obj='acquis', noyau='gaussien',
                        dev=device):
    eta = torch.zeros(len(mes_pos), 2)
    eta = eta.to(dev)
    if noyau == 'gaussien':
        for i in range(len(mes_pos)):
            x = mes_pos[i]
            gauss_x = cudavenant.grad_x_gaussienne_2D(
                x[0] - dom.X, x[1] - dom.Y, x[0] - dom.X, SIGMA)
            integ_x = torch.trapz(acquis * gauss_x, dom.X_grid)
            eta[i, 0] = torch.trapz(integ_x, dom.X_grid)
            gauss_y = cudavenant.grad_y_gaussienne_2D(
                x[0] - dom.X, x[1] - dom.Y, x[1] - dom.Y, SIGMA)
            integ_y = torch.trapz(acquis * gauss_y, dom.X_grid)
            eta[i, 1] = torch.trapz(integ_y, dom.X_grid)
        return eta
    if noyau == 'fourier':
        raise TypeError("Fourier is not implemented")
    else:
        raise TypeError


def etaλk(mesure, acquis, dom, regul, obj='acquis', noyau='gaussien'):
    simul = cudavenant.phi(mesure, dom, obj)
    eta = 1/regul*phiAdjointSimps(acquis - simul, mesure.x, dom, obj)
    # eta = eta - torch.floor(eta) # periodic boundaries
    return eta


def grad_etaλk(mesure, acquis, dom, regul, obj='acquis', noyau='gaussien'):
    if noyau == 'gaussien':
        eta = 1/regul*gradPhiAdjointSimps(acquis - cudavenant.phi(mesure, dom, obj),
                                          mesure.x, dom, obj)
        # eta = eta - torch.floor(eta) # periodic boundaries
        return eta
    if noyau == 'fourier':
        raise TypeError("Fourier is not implemented")
    else:
        raise TypeError("Unkwnon kernel provided")


def CPGD(acquis, dom, λ=1, α=1e-2, β=1e-2, nIter=20, nParticles=5,
         noyau='gaussien', obj='acquis', dev=device):
    """
    This is an implementation of Conic Particle Gradient Descent
    
    Only the mirror retractation is currently implemented
    
    The initialisation is currently restricted to the acquisition grid 
    (you can't initialize with any mesh you choose, even if an adaptative
     grid makes more sense)

    Parameters
    ----------
    acquis : Tensor
        Acquisition input for the specified kernel and obj.
    dom : Domain2D
        Acquisition domain :math:`\mathcal{X}`.
    λ : Float, optional
        Regularisaiton parameter. The default is 1.
    α : Float, optional
        Trade-off parameter for the Wasserstein metric. The default is 1e-2.
    β : Float, optional
        Trade-off parameter for the Fisher-Rao metric. The default is 1e-2.
    nIter : int, optional
        Number of iterations. The default is 20.
    nParticles : int, optional
        Number of particles that makes up the measure ν_0. The default is 5.
    noyau : str, optional
        Class of the kernel applied to the measure. Only the class is 
        currently supported, the Laplace kernel or the Airy function will 
        probably be implemented in a in a future version. The default is 
        'Gaussian'.
    obj : str, optional
        Either 'covar'to reconstruct on covariance either 'acquis'
            to reconstruct. The default is 'acquis'.

    Returns
    -------
    ν_k : Mesure2D
        Reconstructed measure.
    ν_vecteur : List
        List of measures along the loop iterates.
    r_vecteur : List
        List of Dirac amplitudes' along the loop iterates.
    θ_vecteur : List
        List of Dirac positions' along the loop iterates.

    References
    ----------
    [1] Lenaic Chizat. Sparse Optimization on Measures with Over-parameterized 
    Gradient Descent. 2020.
    https://hal.archives-ouvertes.fr/hal-02190822/
    """

    grid_unif = torch.linspace(dom.x_gauche + 0.12, dom.x_droit - 0.12, 
                               nParticles).to(dev)
    mesh_unif = torch.meshgrid(grid_unif, grid_unif)
    θ_0 = torch.stack((mesh_unif[0].flatten(), mesh_unif[1].flatten()),
                      dim=1).to(dev)
    r_0 = 1 * torch.ones(nParticles**2).to(dev)
    ν_0 = cudavenant.Mesure2D(r_0, θ_0, dev=device)
    ν_0 = ν_0.to(dev)

    r_k, θ_k, ν_k = r_0, θ_0, ν_0
    ν_vecteur = [0] * (nIter)
    r_vecteur, θ_vecteur = torch.zeros(
        (nIter, nParticles**2)), torch.zeros((nIter, nParticles**2, 2))
    r_vecteur = r_vecteur.to(dev)
    θ_vecteur = θ_vecteur.to(dev)
    
    loss = torch.zeros(nIter)
    loss[0] = ν_0.energie(dom, acquis, λ, obj=obj)

    # Loop over the gradient descent
    for k in tqdm(range(nIter), 
                  desc=f'[+] Computing CPGD on {device}'):
        # Store iterated measure ν
        ν_vecteur[k] = ν_k.to('cpu')
        r_vecteur[k, :] = r_k
        θ_vecteur[k, :] = θ_k

        grad_r = etaλk(ν_k, acquis, dom, λ, obj) - 1
        grad_θ = grad_etaλk(ν_k, acquis, dom, λ, obj)

        # Update gradient flow
        r_k *= torch.exp(2 * α * λ * grad_r)
        θ_k += β * λ * grad_θ
        ν_k = cudavenant.Mesure2D(r_k, θ_k)

        # NRJ
        loss[k]= ν_k.energie(dom, acquis, λ, obj=obj)
    
    ν_k = ν_k.to('cpu')
    r_vecteur = r_vecteur.to('cpu')
    θ_vecteur = θ_vecteur.to('cpu')
    loss = loss.to('cpu')
    return (ν_k, ν_vecteur, r_vecteur, θ_vecteur, loss)



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
𝜈, 𝜈_itere, r_itere, θ_itere, nrj = CPGD(y, domain, λ=1e0, α=4e-2, β=1e1, 
                                        nIter=500, nParticles=8, 
                                        noyau='gaussien', obj='acquis')

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



# cpgd_anim(y_cpu, m_ax0, 𝜈_itere, θ_itere, domain_cpu, 'mp4')


#%% Curve measure

n_curve = 30
𝛾_grid = torch.linspace(0.2, 0.8, n_curve)
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
𝜈, 𝜈_itere, r_itere, θ_itere, nrj = CPGD(y_curve, domain, λ=1e1, α=5e-2, 
                                         β=1e0, nIter=500, nParticles=15, 
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

# cpgd_anim(y_curve, R_𝛾, 𝜈_itere, θ_itere, domain, 'gif')

