# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 18:59:06 2021

@author: basti
"""

import cudavenant
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import time
import torch
import numpy as np
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))


# GPU acceleration if needed
device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
print("[Test] Using {} device".format(device))

N_ECH = 64
X_GAUCHE = 0
X_DROIT = 1
SIGMA = 0.05
domain = cudavenant.Domain2D(X_GAUCHE, X_DROIT, N_ECH, SIGMA, dev=device)
domain_cpu = cudavenant.Domain2D(X_GAUCHE, X_DROIT, N_ECH, SIGMA)

q = 2**2
super_domain = domain.super_resolve(q, SIGMA/5)


class Curve2D:
    def __init__(self, amplitude=None, position=None, dev='cpu'):
        if amplitude is None or position is None:
            amplitude = torch.Tensor().to(dev)
            position = torch.Tensor().to(dev)
        assert(len(amplitude)+1 == len(position)
               or len(amplitude) == len(position))
        if isinstance(amplitude, torch.Tensor) and isinstance(position,
                                                              torch.Tensor):
            self.a = amplitude.to(dev)
            self.x = position.to(dev)
        elif isinstance(amplitude, np.ndarray) and isinstance(position,
                                                              np.ndarray):
            self.a = torch.from_numpy(amplitude).to(dev)
            self.x = torch.from_numpy(position).to(dev)
        elif isinstance(amplitude, list) and isinstance(position, list):
            self.a = torch.tensor(amplitude).to(dev)
            self.x = torch.tensor(position).to(dev)
        else:
            raise TypeError("You provided wrong tensor m8.")
        self.N_approx = len(amplitude)

# The is not self.N because we need to create a super class of curves that
# encompasses the Curve2D class

    def __eq__(self, m):
        if m == 0 and (self.a == [] and self.x == []):
            return True
        if isinstance(m, self.__class__):
            return self.__dict__ == m.__dict__
        return False

    def __ne__(self, m):
        return not self.__eq__(m)

    def __str__(self):
        return(f"𝛾(0) = {self.x[0]} and 𝛾(1) = {self.x[-1]}")

    def to(self, dev):
        """
        Sends the Measure2D object to the `device` component (the processor or 
        the Nvidia graphics card)

        Parameters
        ----------
        dev : str
            Either `cpu`, `cuda` (default GPU) or `cuda:0`, `cuda:1`, etc.

        Returns
        -------
        None.

        """
        return Curve2D(self.a, self.x, dev=dev)


    def kernel(self, dom, noyau='gaussienne', dev=device):
        r"""
        Applies a kernel to the curve measure :math:`m`.
        BE CAREFUL, the current implementation is a messy scheme
        Namely, rather than smartly computing the curve acquisition with
        the Radon measure definition, we interpolate each part of the curve by
        a Dirac on its center. This is a first approximation
        Supported: convolution with Gaussian kernel.

        Parameters
        ----------
        dom: :py:class:`covenant.Domain2D`
            Domain where the acquisition in :math:`\mathrm{L}^2(\mathcal{X})` 
            of :math:`m` lives.
        kernel: str, optional
            Class of the kernel applied to the measurement. Only the classes
            'gaussienne' and 'fourier' are currently supported, the Laplace 
            kernel or the Laplace kernel or the Airy function will probably be 
            implemented in a future version. The default is 'gaussienne'.

        Raises
        ------
        TypeError
            The kernel is not yet implemented.
        NameError
            The kernel is not recognised by the function.

        Returns
        -------
        acquired: Tensor
            Matrix discretizing :math:`\Phi(m)` .

        """

        N_approx = self.N_approx
        x = self.x
        a = self.a
        X_domain = dom.X
        Y_domain = dom.Y
        if dev == 'cuda':
            acquis = torch.cuda.FloatTensor(X_domain.shape).fill_(0)
        else:
            acquis = torch.zeros(X_domain.shape)
        if noyau == 'gaussienne':
            sigma = dom.sigma
            for i in range(0, N_approx):
                x_interpol = (x[i, :] + x[i+1, :])/2
                gaus_decal = cudavenant.gaussienne_2D(X_domain - x_interpol[0],
                                                      Y_domain - x_interpol[1],
                                                      sigma)
                acquis += a[i] * gaus_decal
            return acquis
        if noyau == 'fourier':
            raise TypeError("Not implemented.")
        if noyau == 'laplace':
            raise TypeError("Not implemented.")
        raise NameError("Unknown kernel.")


def phi_curve(m, dom):
    return m.kernel(dom)


def phi_vecteur(a, x, dom):
    m_tmp = Curve2D(a, x)
    return m_tmp.kernel(dom)


def phiAdjoint_curve(acquis, dom):
    N_ech = dom.N_ech
    sigma = dom.sigma
    (X_big, Y_big) = dom.big()
    h_vec = cudavenant.gaussienne_2D(X_big, Y_big, sigma, 
                                     undivide=cudavenant.__normalis_PSF__)
    h_ker = h_vec.reshape(1, 1, N_ech*2-1, N_ech*2-1)
    y_arr = acquis.reshape(1, 1, N_ech, N_ech)
    eta = torch.nn.functional.conv2d(h_ker, y_arr, stride=1)
    eta = torch.flip(torch.squeeze(eta), [1, 0])
    return eta

# Hierher we need gradient: you need to implement autodiff
def etak_curve(mesure, acquis, dom, regul):
    eta = 1/regul*phiAdjoint_curve(acquis - phi_curve(mesure, dom), dom)
    return eta


#%% Simulate curve

N_pts_curve = 50
a_vector = torch.linspace(2, 4, N_pts_curve-1)

t_vector = torch.linspace(0, 1, N_pts_curve)
x_vector = 0.4 * t_vector * torch.cos(2*np.pi*t_vector) + 0.4
y_vector = 0.4 * t_vector * torch.sin(2*np.pi*t_vector) + 0.6
pos_vector = torch.stack((x_vector, y_vector), dim=-1)

µ_𝛾 = Curve2D(a_vector, pos_vector)
y_0 = µ_𝛾.kernel(domain)

cont1 = plt.contourf(domain.X, domain.Y, y_0, 100, cmap='bone')
for c in cont1.collections:
    c.set_edgecolor("face")
plt.colorbar()
plt.plot(µ_𝛾.x[:, 0], µ_𝛾.x[:, 1], c='red', label='$\Gamma=\gamma([0,1])$')
plt.xlabel('$x$', fontsize=18)
plt.ylabel('$y$', fontsize=18)
plt.title('Measure $\mu_\gamma$ and its acquisition on $\mathcal{X}$', fontsize=18)
plt.axis([0, 1, 0, 1])
plt.legend()



#%% Sliding Frank Wolfe renconstruction

(m_sfw, nrj_sfw) = cudavenant.SFW(y_0, domain,
                                  regul=1e-3,
                                  nIter=12, mesParIter=False,
                                  obj='acquis', printInline=False)
m_sfw.show(domain, y_0)
plt.title('SFW Diracs peaks reconstruction, $y_0$ in background')
plt.xlabel('$x$', fontsize=18)
plt.ylabel('$y$', fontsize=18)

plt.figure()
cont1 = plt.contourf(super_domain.X, super_domain.Y, m_sfw.kernel(super_domain), 100, 
                     cmap='bone')
for c in cont1.collections:
    c.set_edgecolor("face")
plt.colorbar()
plt.title('Super-resolved SFW reconstruction', fontsize=18)
plt.xlabel('$x$', fontsize=18)
plt.ylabel('$y$', fontsize=18)

# a_skeleton = torch.abs(torch.diff(m_sfw.a))
# x_skeleton = m_sfw.x
# skeleton = Curve2D(a_skeleton, x_skeleton)

# y_skeleton = skeleton.kernel(domain)
# plt.figure()
# plt.imshow(y_skeleton)


#%% Curve Sliding Frank Wolfe renconstruction


def CSFW(acquis, dom, regul=1e-5, nIter=0, mes_init=None, mesParIter=False,
         dev=device, printInline=True):
    r"""Algorithm Curve Sliding Frank-Wolfe for curve measures reconstruction
    solution du Curve-BLASSO [1].

    Parameters
    ----------
    acquis : ndarray
            Soit l'acquisition moyenne :math:`y` soit la covariance :math:`R_y`.
    dom : Domain2D
        Domaine :math:`\mathcal{X}` sur lequel est défini :math:`m_{a,x}`
        ainsi que l'acquisition :math:`y(x,t)` , etc.
    regul : double, optional
        Paramètre de régularisation :math:`\lambda`. The default is 1e-5.
    nIter : int, optional
        Nombre d'itérations maximum pour l'algorithme. The default is 5.
    mes_init : Mesure2D, optional
        Mesure pour initialiser l'algorithme. Si None est passé en argument,
        l'algorithme initialisera avec la mesure nulle. The default is None.
    mesParIter : boolean, optional
        Vontrôle le renvoi ou non du ndarray mes_vecteur qui contient les
        :math:`k` mesures calculées par SFW. The default is False.
    obj : str, optional
        Soit `covar` pour reconstruire sur la covariance soit `acquis` pour
        reconstruire sur la moyenne. The default is 'covar'.
    printInline : str, optional
        Ouput the log of the optimizer. The default is True.

    Output
    -------
    mesure_k : Mesure2D
            Dernière mesure reconstruite par SFW.
    nrj_vecteur : ndarray
                Vecteur qui donne l'énergie :math:`T_\lambda(m^k)`
                au fil des itérations.
    mes_vecteur : ndarray
                Vecteur des mesures reconstruites au fil des itérations.

    Raises
    ------
    TypeError
        Si l'objectif `obj` n'est pas connu, lève une exception.

    References
    ----------
    """
    N_ech_y = dom.N_ech
    N_grille = dom.N_ech**2
    acquis = acquis.to(dev)
    if mes_init == None:
        mesure_k = Curve2D(dev=dev)
        a_k = torch.Tensor().to(dev)
        x_k = torch.Tensor().to(dev)
        x_k_demi = torch.Tensor().to(dev)
        Nk = 1
    else:
        mesure_k = mes_init
        a_k = mes_init.a
        x_k = mes_init.x
        Nk = 1 # hierher change with SuperCurve2D class
    if mesParIter:
        mes_vecteur = torch.Tensor()
    nrj_vecteur = torch.zeros(nIter)
    N_vecteur = [Nk]
    for k in range(nIter):
        if printInline:
            print('\n' + 'Step number ' + str(k))
        eta_V_k = etak_curve(mesure_k, acquis, dom, regul)
        certif_abs = torch.abs(eta_V_k)
        
        # # Hierher Skeleton estimation step: to be implemented!
        # x_star_index = unravel_index(certif_abs.argmax(), eta_V_k.shape)
        # x_star_tuple = tuple(s / N_ech_y for s in x_star_index)
        # x_star = torch.tensor(x_star_tuple).reshape(1, 2).to(dev)
        # if printInline:
        #     print(fr'* x^* index {x_star_tuple} max ' +
        #           fr'à {certif_abs[x_star_index]:.3e}')

        # # Hierher: stopping condition (step 4), you have to implement it!
        # if torch.abs(eta_V_k[x_star_index]) < 1:
        #     nrj_vecteur[k] = mesure_k.energie(dom, acquis, regul)
        #     if printInline:
        #         print("\n\n---- Stopping condition ----")
        #     if mesParIter:
        #         return(mesure_k, nrj_vecteur[:k], mes_vecteur)
        #     return(mesure_k, nrj_vecteur[:k])

        mesure_k_demi = µ_𝛾
        x_k_demi = pos_vector
        a_param = (torch.ones(N_pts_curve, dtype=torch.float)
                   ).to(dev).detach().requires_grad_(True)

        # # Hierher: Create atom with estimated position
        # mesure_k_demi = Mesure2D()
        # if not x_k.numel():
        #     x_k_demi = x_star
        #     a_param = (torch.ones(Nk+1, dtype=torch.float)
        #                ).to(dev).detach().requires_grad_(True)
        # else:
        #     x_k_demi = torch.cat((x_k, x_star))
        #     uno = torch.tensor([10.0], dtype=torch.float).to(dev).detach()
        #     a_param = torch.cat((a_k, uno))
        #     a_param.requires_grad = True

        # Solve LASSO (step 7)
        if printInline:
            print('* Convex step')
        mse_loss = torch.nn.MSELoss(reduction='sum')
        optimizer = torch.optim.LBFGS([a_param])
        alpha = regul
        n_epoch = 15

        for epoch in range(n_epoch):
            def closure():
                if torch.is_grad_enabled():
                    optimizer.zero_grad()
                outputs = phi_vecteur_curve(a_param, x_k_demi, dom)
                loss = 0.5 * mse_loss(acquis, outputs)
                loss += alpha * a_param.abs().sum()
                if loss.requires_grad:
                    loss.backward()
                del outputs
                return loss

            optimizer.step(closure)

        a_k_demi = a_param.detach().clone()  # pour ne pas copier l'arbre
        del a_param, optimizer, mse_loss

        # print('* x_k_demi : ' + str(np.round(x_k_demi, 2)))
        # print('* a_k_demi : ' + str(np.round(a_k_demi, 2)))
        mesure_k_demi += Mesure2D(a_k_demi, x_k_demi)

        # double non-convex LASSO (step 8)
        if printInline:
            print(f'* Non-convex step with {Nk} curve(s)')
        param = torch.cat((a_k_demi, x_k_demi.reshape(-1)))
        param.requires_grad = True

        mse_loss = torch.nn.MSELoss(reduction='sum')
        optimizer = torch.optim.Adam([param])
        n_epoch = 30

        for epoch in range(n_epoch):
            def closure():
                optimizer.zero_grad()
                x_tmp = param[Nk+1:].reshape(Nk+1, 2)
                fidelity = phi_vecteur(param[:Nk+1], x_tmp, dom)
                loss = 0.5 * mse_loss(acquis, fidelity)
                loss += regul * param[:1].abs().sum()
                loss.backward()
                del fidelity, x_tmp
                return loss

            optimizer.step(closure)

        a_k_plus = param[:int(len(param)/3)].detach().clone()
        x_k_plus = param[int(len(param)/3):].detach().clone().reshape(Nk+1, 2)
        del param, optimizer, mse_loss

        # print('* a_k_plus : ' + str(np.round(a_k_plus, 2)))
        # print('* x_k_plus : ' +  str(np.round(x_k_plus, 2)))

        # Update parameters while discarding small atoms
        mesure_k = Mesure2D(a_k_plus, x_k_plus, dev=dev)
        mesure_k = mesure_k.prune()
        # mesure_k = merge_spikes(mesure_k)
        a_k = mesure_k.a
        x_k = mesure_k.x
        Nk = mesure_k.N

        # Graph and energy
        nrj_vecteur[k] = mesure_k.energie(dom, acquis, regul)
        if printInline:
            print(f'* Energy: {nrj_vecteur[k]:.3e}')
        if mesParIter == True:
            mes_vecteur = np.append(mes_vecteur, [mesure_k])
            torch.save(mes_vecteur, 'saved_objects/mes_curve_test' + '.pkl')
        try:
            if (N_vecteur[-1] == N_vecteur[-2]
                and N_vecteur[-1] == N_vecteur[-3]
                    and N_vecteur[-1] == N_vecteur[-4]):
                if printInline:
                    print('\n[!] Algorithm has finished')
                    print("\n\n---- End of loop up ----")
                if mesParIter:
                    return(mesure_k, nrj_vecteur[:k], mes_vecteur)
                return(mesure_k, nrj_vecteur)
        except IndexError:
            pass
        N_vecteur = np.append(N_vecteur, Nk)

    # End of the computation
    if printInline:
        print("\n\n---- End of loop up ----")
    if mesParIter:
        return(mesure_k, nrj_vecteur, mes_vecteur)
    return(mesure_k, nrj_vecteur)






# N_pts_curve = 50
# a_vector = torch.cat((torch.linspace(50, 60, N_pts_curve//2),
#                       torch.linspace(50, 60, N_pts_curve//2).flip(0)))
# grid_vector = torch.linspace(0.2, 0.8, N_pts_curve)
# x_vector = torch.stack((grid_vector, grid_vector), dim=-1)

# m_ax0 = cudavenant.Mesure2D(a_vector, x_vector)

# y0 = m_ax0.kernel(domain)

# plt.imshow(y0, cmap='bone')
# # plt.plot([N_ECH, 0], [0, 0], c='red')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('Acquisition $y_0$')
# plt.colorbar()

# grad_y0 = torch.diff(y0)
# laplacien_y0 = torch.diff(grad_y0)

# fig = plt.figure(figsize=(16, 12))

# # ax1 = fig.add_subplot(121)
# # divider1 = make_axes_locatable(ax1)
# # cax1 = divider1.append_axes('right', size='5%', pad=0.05)
# # im1 = ax1.imshow(grad_y0, cmap='bone')
# # ax1.set_title('$\\nabla y_0$', fontsize=30)
# # fig.colorbar(im1, cax=cax1, orientation='vertical')

# # ax2 = fig.add_subplot(122)
# # divider2 = make_axes_locatable(ax2)
# # cax2 = divider2.append_axes('right', size='5%', pad=0.05)
# # im2 = ax2.imshow(laplacien_y0, cmap='bone')
# # ax2.set_title('$\Delta y_0$', fontsize=30)
# # fig.colorbar(im1, cax=cax2, orientation='vertical')


# Hierher might be useful
# https://stackoverflow.com/questions/56111340/how-to-calculate-gradients-on-a-tensor-in-pytorch



