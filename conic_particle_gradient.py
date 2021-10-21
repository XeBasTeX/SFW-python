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
from scipy import integrate
import torch

__saveFig__ = False
__saveVid__ = False
__savePickle__ = False


import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))


# GPU acceleration if needed
device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
print("[+] Using {} device".format(device))

N_ECH = 2**5
X_GAUCHE = 0
X_DROIT = 1
SIGMA = 0.1

domain = cudavenant.Domain2D(X_GAUCHE, X_DROIT, N_ECH, SIGMA, dev=device)

amp = torch.tensor([1, 2, 3])
xpos = torch.tensor([[0.2, 0.2], [0.7, 0.6], [0.5, 0.8]])
m_ax0 = cudavenant.Mesure2D(amp, xpos)

y0 = m_ax0.kernel(domain)
bru = cudavenant.Bruits(0, 1e-3, 'unif')
y = m_ax0.acquisition(domain, N_ECH, bru)
plt.imshow(y)



def cpgd_anim(acquis, mes_zer, m_itere, theta_itere, dom, video='gif'):
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


def phiAdjointSimps(acquis, mes_pos, dom, obj='acquis', noyau='gaussien'):
    eta = torch.zeros(len(mes_pos))
    if noyau == 'gaussien':
        for i in range(len(mes_pos)):
            x = mes_pos[i]
            gauss = cudavenant.gaussienne_2D(x[0] - dom.X, x[1] - dom.Y, SIGMA)
            integ_x = integrate.simps(acquis * gauss, x=dom.X_grid)
            eta[i] = integrate.simps(integ_x, x=dom.X_grid)
        return eta
    if noyau == 'fourier':
        for i in range(len(mes_pos)):
            raise TypeError("Fourier is not implemented")
        return eta
    else:
        raise TypeError("Unknown kernel")


def gradPhiAdjointSimps(acquis, mes_pos, dom, obj='acquis', noyau='gaussien'):
    eta = torch.zeros(len(mes_pos), 2)
    if noyau == 'gaussien':
        for i in range(len(mes_pos)):
            x = mes_pos[i]
            gauss_x = cudavenant.grad_x_gaussienne_2D(
                x[0] - dom.X, x[1] - dom.Y, x[0] - dom.X, SIGMA)
            integ_x = integrate.simps(acquis * gauss_x, x=dom.X_grid)
            eta[i, 0] = integrate.simps(integ_x, x=dom.X_grid)
            gauss_y = cudavenant.grad_y_gaussienne_2D(
                x[0] - dom.X, x[1] - dom.Y, x[1] - dom.Y, SIGMA)
            integ_y = integrate.simps(acquis * gauss_y, x=dom.X_grid)
            eta[i, 1] = integrate.simps(integ_y, x=dom.X_grid)
        return eta
    if noyau == 'fourier':
        raise TypeError("Fourier is not implemented")
    else:
        raise TypeError


def etaλk(mesure, acquis, dom, regul, obj='acquis', noyau='gaussien'):
    eta = 1/regul*phiAdjointSimps(acquis - cudavenant.phi(mesure, dom, obj),
                                  mesure.x, dom, obj)
    return eta


def grad_etaλk(mesure, acquis, dom, regul, obj='acquis', noyau='gaussien'):
    if noyau == 'gaussien':
        eta = 1/regul*gradPhiAdjointSimps(acquis - cudavenant.phi(mesure, dom, obj),
                                          mesure.x, dom, obj)
        return eta
    if noyau == 'fourier':
        raise TypeError("Fourier is not implemented")
    else:
        raise TypeError("Unkwnon kernel provided")


def CPGD(acquis, dom, λ=1, α=1e-2, β=1e-2, nIter=20, nParticles=5,
         noyau='gaussien', obj='acquis'):
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
    λ : TYPE, optional
        DESCRIPTION. The default is 1.
    α : TYPE, optional
        DESCRIPTION. The default is 1e-2.
    β : TYPE, optional
        DESCRIPTION. The default is 1e-2.
    nIter : TYPE, optional
        DESCRIPTION. The default is 20.
    nParticles : TYPE, optional
        DESCRIPTION. The default is 5.
    noyau : TYPE, optional
        DESCRIPTION. The default is 'gaussien'.
    obj : TYPE, optional
        DESCRIPTION. The default is 'acquis'.

    Returns
    -------
    ν_k : TYPE
        DESCRIPTION.
    ν_vecteur : TYPE
        DESCRIPTION.
    r_vecteur : TYPE
        DESCRIPTION.
    θ_vecteur : TYPE
        DESCRIPTION.

    References
    ----------
    [1] Lenaic Chizat. Sparse Optimization on Measures with Over-parameterized 
    Gradient Descent. 2020.
    https://hal.archives-ouvertes.fr/hal-02190822/
    """

    grid_unif = torch.linspace(dom.x_gauche+0.1, dom.x_droit-0.1, nParticles)
    mesh_unif = torch.meshgrid(grid_unif, grid_unif)
    θ_0 = torch.stack((mesh_unif[0].flatten(), mesh_unif[1].flatten()),
                      dim=1)
    r_0 = 1 * torch.ones(nParticles**2)
    ν_0 = cudavenant.Mesure2D(r_0, θ_0)

    r_k, θ_k, ν_k = r_0, θ_0, ν_0
    ν_vecteur = [0] * (nIter)
    r_vecteur, θ_vecteur = torch.zeros(
        (nIter, nParticles**2)), torch.zeros((nIter, nParticles**2, 2))

    # Loop over the gradient descent
    for k in tqdm(range(nIter)):
        # Store iterated measure ν
        ν_vecteur[k] = ν_k
        r_vecteur[k, :] = r_k
        θ_vecteur[k, :] = θ_k

        grad_r = etaλk(ν_k, acquis, dom, λ, obj) - 1
        grad_θ = grad_etaλk(ν_k, acquis, dom, λ, obj)

        # Update gradient flow
        r_k *= torch.exp(2 * α * λ * grad_r)
        θ_k += β * λ * grad_θ
        ν_k = cudavenant.Mesure2D(r_k, θ_k)

    return (ν_k, ν_vecteur, r_vecteur, θ_vecteur)


print('[+] Computing CPGD for points')
𝜈, 𝜈_itere, r_itere, θ_itere = CPGD(y, domain, λ=1e-1, α=4e-2, β=1e0, 
                                    nIter=5, noyau='gaussien')


plt.figure()
cont = plt.contourf(domain.X, domain.Y, y, 100)
for c in cont.collections:
    c.set_edgecolor("face")
plt.scatter(m_ax0.x[:, 0], m_ax0.x[:, 1], label='$m_{a_0,x_0}$', marker='o',
            c='white', s=2*m_ax0.a*domain.N_ech)
for l in range(len(𝜈.a)):
    plt.plot(θ_itere[:, l, 0], θ_itere[:, l, 1], 'r', linewidth=1.2)
plt.scatter(𝜈.x[:, 0], 𝜈.x[:, 1], label='$\\nu_t$', c='r', s=𝜈.a*domain.N_ech)

plt.axis('square')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.colorbar()
plt.legend(loc=1)



# cpgd_anim(y0, m_ax0, 𝜈_itere, θ_itere, domain, 'gif')


#%% Curve measure

n_curve = 30
𝛾_a = torch.ones(2*n_curve)
𝛾_grid = torch.linspace(0.1, 0.9, n_curve)
𝛾_x = torch.stack((𝛾_grid, 𝛾_grid), dim=1)
𝛾_x = torch.cat((torch.stack((torch.flip(𝛾_grid, [-1]), 𝛾_grid), dim=1),
                 𝛾_x))
R_𝛾 = cudavenant.Mesure2D(𝛾_a, 𝛾_x)

y_curve = R_𝛾.kernel(domain)
plt.imshow(y_curve)


print('[+] Computing CPGD for curves')
𝜈, 𝜈_itere, r_itere, θ_itere = CPGD(y_curve, domain, λ=1e-1, α=5e-2, β=1e0, 
                                    nIter=300, nParticles=15, noyau='gaussien')


plt.figure()
cont = plt.contourf(domain.X, domain.Y, y_curve, 100)
for c in cont.collections:
    c.set_edgecolor("face")
plt.scatter(R_𝛾.x[:, 0], R_𝛾.x[:, 1], label='$m_{a_0,x_0}$', marker='o',
            c='white', s=2*R_𝛾.a*domain.N_ech)
# for l in range(len(𝜈.a)):
#     plt.plot(θ_itere[:, l, 0], θ_itere[:, l, 1], 'r', linewidth=1.2)
plt.scatter(𝜈.x[:, 0], 𝜈.x[:, 1], label='$\\nu_t$', c='r', s=𝜈.a*domain.N_ech)

plt.axis('square')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.colorbar()
plt.legend(loc=1)

# cpgd_anim(y_curve, R_𝛾, 𝜈_itere, θ_itere, domain, 'gif')

