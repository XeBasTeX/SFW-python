#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Attemps to provide the covenant package with a GPU acceleration with CUDA.
Provides  the essential functions for simulating nD discrete Radon measures
and reconstructing those measures w.r.t to a provided acquistion

Created on Mon Mar 22 08:49:01 2021

@author: Bastien (https://github.com/XeBasTeX)
"""


__normalis_PSF__ = True


import torch
import ot
import numpy as np

from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt

import pickle

# GPU acceleration if needed
device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
print("[Cudavenant] Using {} device".format(device))



def sum_normalis(X_domain, Y_domain, sigma_g):
    r"""
    Renvoie la somme de toutes les composantes de la PSF discrète, pour que
    l'on puisse normaliser la PSF, de sorte à avoir torch.sum(PSF) = 1

    Parameters
    ----------
    X_domain : Tensor
        Grille des coordonnées X (issue de meshgrid).
    Y_domain : Tensor
        Grille des coordonnées Y (issue de meshgrid).
    sigma_g : double
        :math:`\sigma` paramètre de la gaussienne.

    Returns
    -------
    double
        Somme de toutes les composantes de la PSF

    """
    expo = torch.exp(-(torch.pow(X_domain, 2) +
                torch.pow(Y_domain, 2))/(2*sigma_g**2))
    normalis = sigma_g * (2*np.pi)
    normalis = 1 / normalis
    return torch.sum(normalis * expo)


def gaussienne_2D(X_domain, Y_domain, sigma_g, undivide=__normalis_PSF__):
    r"""
    Gaussienne en 2D centrée en 0 normalisée.

    Parameters
    ----------
    X_domain : Tensor
        Grille des coordonnées X (issue de meshgrid).
    Y_domain : Tensor
        Grille des coordonnées Y (issue de meshgrid).
    sigma_g : double
        :math:`\sigma` paramètre de la gaussienne.
    undivide : str
        Pour savoir si on normalise par les composantes de la PSF.
        The default is True.

    Returns
    -------
    Tensor
        Vecteur discrétisant la gaussienne :math:`h` sur :math:`\mathcal{X}`.

    """
    expo = torch.exp(-(torch.pow(X_domain, 2) + torch.pow(Y_domain, 2))
                  / (2*sigma_g**2))
    normalis = (sigma_g * (2*np.pi))
    normalis = 1 / normalis
    if undivide == True:
        sumistan = torch.sum(expo * normalis)
        return expo * normalis / sumistan
    return expo * normalis


def grad_x_gaussienne_2D(X_domain, Y_domain, X_deriv, sigma_g, normalis=None):
    r"""
    Gaussienne centrée en 0  normalisée. Attention, ne prend pas en compte la
    chain rule derivation.

    Parameters
    ----------
    X_domain : Tensor
        Grille des coordonnées X (issue de meshgrid).
    Y_domain : Tensor
        Grille des coordonnées Y (issue de meshgrid).
    X_deriv : Tensor
        Grille des coordonnées X pour calculer la partie en :math:`x` de la
        dérivée partielle
    sigma_g : double
        :math:`\sigma` paramètre de la gaussienne.

    Returns
    -------
    Tensor
        Vecteur discrétisant la première dérivée partielle de la gaussienne
        :math:`\partial_1 h` sur :math:`\mathcal{X}`.

    """
    expo = gaussienne_2D(X_domain, Y_domain, sigma_g, normalis)
    cst_deriv = sigma_g**2
    carre = - X_deriv
    return carre * expo / cst_deriv


def grad_y_gaussienne_2D(X_domain, Y_domain, Y_deriv, sigma_g, normalis=None):
    r"""
    Gaussienne centrée en 0  normalisée. Attention, ne prend pas en compte la
    chain rule derivation.

    Parameters
    ----------
    X_domain : Tensor
        Grille des coordonnées X (issue de meshgrid).
    Y_domain : Tensor
        Grille des coordonnées Y (issue de meshgrid).
    Y_deriv : Tensor
        Grille des coordonnées Y pour calculer la partie en :math:`y` de la
        dérivée partielle
    sigma_g : double
        :math:`\sigma` paramètre de la gaussienne.

    Returns
    -------
    Tensor
        Vecteur discrétisant la première dérivée partielle de la gaussienne
        :math:`\partial_2 h` sur :math:`\mathcal{X}`.

    """
    expo = gaussienne_2D(X_domain, Y_domain, sigma_g, normalis)
    cst_deriv = sigma_g**2
    carre = - Y_deriv
    return carre * expo / cst_deriv


def grad_gaussienne(X_domain, Y_domain, sigma_g, normalis=None):
    """
    Gradient of the Gaussian function centred in 0.

    Parameters
    ----------
    X_domain : Tensor
        Grille des coordonnées X (issue de meshgrid).
    Y_domain : Tensor
        Grille des coordonnées Y (issue de meshgrid).
    sigma_g : double
        :math:`\sigma` paramètre de la gaussienne.
    normalis : boolean, optional
        Enables or not the normalisation of the gaussian PSF. 
        The default is None.

    Returns
    -------
    Tensor
        Vecteur discrétisant le gradient de la gaussienne :math:`h` sur 
        :math:`\mathcal{X}`.

    """
    gauss = gaussienne_2D(X_domain, Y_domain, sigma_g)
    return((- (X_domain + Y_domain) / sigma_g**2) * gauss)


# def indicator_function_list(elements, X_domain, Y_domain, width):
#     for x_coord in X_domain:
#         for y_coord in Y_domain:
#             if (x_coord - width <= element[0] <= x_coord + width and 
#                 y_coord - width <= element[1] <= y_coord + width):
#                 return 1


class Domain2D:
    def __init__(self, gauche, droit, ech, sigma_psf, dev='cpu'):
        '''Hierher ne marche que pour des grilles carrées'''
        grille = torch.linspace(gauche, droit, ech)
        X_domain, Y_domain = torch.meshgrid(grille, grille)
        self.x_gauche = gauche
        self.x_droit = droit
        self.N_ech = ech
        self.X_grid = grille.to(dev)
        self.X = X_domain.to(dev)
        self.Y = Y_domain.to(dev)
        self.sigma = sigma_psf
        self.dev = dev

    def get_domain(self):
        return(self.X, self.Y)

    def size(self):
        return self.N_ech

    def compute_square_mesh(self):
        r"""
        Renvoie les grilles discrétisant le domaine :math:`\mathcal{X}` à N_ech

        Returns
        -------
        Tensor
            Les grilles coordonnées de x et des y.

        """
        return torch.meshgrid(self.X_grid)

    def big(self):
        """
        Renvoie les grilles adaptées pour le calcul de la convolution discrètes
        (par exemple dans phiAdjoint) entre deux matrices

        Returns
        -------
        X_big : Tensor
                Matrice coordonnées des x
        Y_big : Tensor
                Matrice coordonnées des y

        """
        grid_big = torch.linspace(self.x_gauche-self.x_droit, self.x_droit,
                               2*self.N_ech - 1).to(device)
        X_big, Y_big = torch.meshgrid(grid_big, grid_big)
        return(X_big, Y_big)

    def biggy(self):
        """
        Renvoie les grilles adaptées pour le calcul de la convolution discrètes
        (par exemple dans phiAdjoint) entre deux matrices

        Returns
        -------
        X_big : Tensor
                Matrice coordonnées des x
        Y_big : Tensor
                Matrice coordonnées des y

        """
        grid_big = torch.linspace(self.x_gauche-self.x_droit, self.x_droit,
                                  2*self.N_ech)
        X_big, Y_big = torch.meshgrid(grid_big, grid_big)
        return(X_big, Y_big)


    def reverse(self):
        """
        Renvoie les grilles adaptées pour le calcul de la convolution discrètes
        (par exemple dans phiAdjoint) entre deux matrices

        Returns
        -------
        X_big : Tensor
                Matrice coordonnées des x
        Y_big : Tensor
                Matrice coordonnées des y

        """
        grid_big = torch.linspace(self.x_gauche-self.x_droit, self.x_droit,
                                  self.N_ech)
        X_big, Y_big = torch.meshgrid(grid_big, grid_big)
        return(X_big, Y_big)


    def super_resolve(self, q=4, sigma=None):
        super_N_ech = q * self.N_ech
        if sigma == None:
            super_sigma = self.sigma
        else:
            super_sigma = sigma
        return Domain2D(self.x_gauche, self.x_droit, super_N_ech, super_sigma,
                        dev=self.dev)


class Bruits:
    def __init__(self, fond, niveau, type_de_bruits):
        self.fond = fond
        self.niveau = niveau
        self.type = type_de_bruits

    def get_fond(self):
        return self.fond

    def get_nv(self):
        return self.niveau


class Mesure2D:
    def __init__(self, amplitude=None, position=None, dev='cpu'):
        if amplitude is None or position is None:
            amplitude = torch.Tensor().to(dev)
            position = torch.Tensor().to(dev)
        assert(len(amplitude) == len(position) or len(amplitude) == len(position))
        if isinstance(amplitude, torch.Tensor) and isinstance(position,
                                                              torch.Tensor):
            self.a = amplitude.to(dev)
            self.x = position.to(dev)
        elif isinstance(amplitude, np.ndarray) and isinstance(position,
                                                            np.ndarray):
            self.a = torch.from_numpy(amplitude).to(dev)
            self.x = torch.from_numpy(position).to(dev)
        else:
            raise TypeError("Gros chien de la casse formate ta bouse")
        self.N = len(amplitude)

    def __add__(self, m):
        '''Hieher : il faut encore régler l'addition pour les mesures au même
        position, ça vaudrait le coup de virer les duplicats'''
        a_new = torch.cat((self.a, m.a))
        x_new = torch.cat((self.x, m.x))
        return Mesure2D(a_new, x_new, dev=device)

    def __eq__(self, m):
        if m == 0 and (self.a == [] and self.x == []):
            return True
        if isinstance(m, self.__class__):
            return self.__dict__ == m.__dict__
        return False

    def __ne__(self, m):
        return not self.__eq__(m)

    def __str__(self):
        '''Donne les infos importantes sur la mesure'''
        return(f"{self.N} δ-pics \nAmplitudes : {self.a}" +
               f"\nPositions : {self.x}")

    def to(self, dev):
        """
        Envoie l'objet Mesure2D sur le composant `device` (le processeur ou 
        la carte graphique Nvidia)

        Parameters
        ----------
        dev : str
            Either `cpu`, `cuda` (GPU by default) or `cuda:0`, `cuda:1`, etc.

        Returns
        -------
        None.

        """
        # self.a = self.a.to(dev)
        # self.x = self.x.to(dev)
        return Mesure2D(self.a, self.x, dev=dev)

    def kernel(self, dom, noyau='gaussienne', dev=device):
        r"""
        Applique un noyau à la mesure discrète :math:`m`.
        Pris en charge : convolution  à noyau gaussien.

        Parameters
        ----------
        dom : :py:class:`covenant.Domain2D`
            Domaine sur lequel va être calculé l'acquisition de :math:`m` dans
            :math:`\\mathrm{L}^2(\\mathcal{X})`.
        noyau : str, optional
            Classe du noyau appliqué sur la mesure. Seule la classe
            'gaussienne' est pour l'instant prise en charge, le noyau de
            Laplace ou la fonction d'Airy seront probablement implémentées
            dans une prochaine version. The default is 'gaussienne'.

        Raises
        ------
        TypeError
            Le noyau n'est pas encore implémenté.
        NameError
            Le noyau n'est pas reconnu par la fonction.

        Returns
        -------
        acquis : Tensor
            Matrice discrétisant :math:`\Phi(m)` .

        """
        N = self.N
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
            for i in range(0, N):
                acquis += a[i] * gaussienne_2D(X_domain - x[i, 0],
                                               Y_domain - x[i, 1],
                                               sigma)
            return acquis
        if noyau == 'fourier':
            f_c = dom.sigma
            f_c_vec = torch.arange(- f_c, f_c + 1)
            imag = torch.complex(torch.tensor(0), torch.tensor(1))
            acquis = a * torch.exp(-2 * imag * np.pi * f_c_vec * x)
            return acquis
        if noyau == 'laplace':
            raise TypeError("Pas implémenté.")
        # if noyau == 'indicator':
        #     pixel_size = dom.x_gauche / N
        #     for i in range(0, N):
        #         acquis += a[i] * indicator_function_list(x[i,:], X_domain, 
        #                                                  Y_domain, pixel_size)
        raise NameError("Unknown kernel.")

    def cov_kernel(self, dom):
        r"""
        Noyau de covariance associée à la mesure :math:`m` sur le domaine `dom`

        Parameters
        ----------
        dom : Domain2D
            Domaine :math:`\mathcal{X}` sur lequel va être calculé
            l'acquisition de :math:`m` dans
            :math:`\\mathrm{L}^2(\\mathcal{X}^2)`.

        Returns
        -------
        acquis : Tensor
            Matrice discrétisant :math:`\Lambda(m)` .

        """
        X_domain = dom.X
        Y_domain = dom.Y
        sigma = dom.sigma
        N_ech = dom.N_ech
        N = self.N
        x = self.x
        amp = self.a
        if device == 'cuda':
            acquis = torch.cuda.FloatTensor(N_ech**2, N_ech**2).fill_(0)
        else:
            acquis = torch.zeros(N_ech**2, N_ech**2)
        for i in range(0, N):
            noyau = gaussienne_2D(X_domain - x[i, 0],
                                  Y_domain - x[i, 1],
                                  sigma)
            noyau_re = noyau.reshape(-1)
            acquis += amp[i] * torch.outer(noyau_re, noyau_re)
        return acquis


    def acquisition(self, dom, echantillo, bru):
        r"""
        Simule une acquisition pour la mesure.

        Parameters
        ----------
        dom : Domain2D
            Domaine sur lequel va être calculé l'acquisition de :math:`m` dans
            :math:`\mathrm{L}^2(\mathcal{X})`.
        echantillo : double
            Format de la matrice carrée des bruits. Hierher àimplémenter sur
            des domaines en rectangle.
        bru : Bruits
            L'objet contient toutes les valeurs définissant le bruit à simuler.

        Raises
        ------
        NameError
            Le type de bruit spécifié n'est pas reconnu par la fonction.

        Returns
        -------
        acquis : Tensor
            Vecteur discrétisant l'acquisition :math:`\Phi(m)`.

        """
        fond = bru.fond
        nv = bru.niveau
        type_de_bruits = bru.type
        if type_de_bruits == 'unif':
            w = (nv * torch.rand((echantillo, echantillo))).to(device)
            simul = self.kernel(dom, noyau='gaussienne')
            acquis = simul + w + fond
            return acquis
        if type_de_bruits == 'gauss':
            simul = self.kernel(dom, noyau='gaussienne')
            w = torch.normal(0, nv, size=(echantillo, echantillo)).to(device)
            acquis = simul + w + fond
            return acquis
        if type_de_bruits == 'poisson':
            w = nv*torch.poisson(dom.X)
            simul = w*self.kernel(dom, noyau='gaussienne')
            acquis = simul + fond
        raise NameError("Unknown type of noise")


    def graphe(self, dom, lvl=50):
        '''Trace le contourf de la mesure'''
        # f = plt.figure()
        X_domain = dom.X
        Y_domain = dom.Y
        plt.contourf(X_domain, Y_domain, self.kernel(dom),
                     lvl, label='$y_0$', cmap='hot')
        plt.xlabel('X', fontsize=18)
        plt.ylabel('Y', fontsize=18)
        plt.title('$\Phi y$', fontsize=18)
        plt.colorbar()
        plt.grid()
        plt.show()


    def tv(self):
        r"""
        Renvoie 0 si la mesure est vide : hierher à vérifier.

        Returns
        -------
        double
            :math:`|m|(\mathcal{X})`, norme TV de la mesure.

        """
        try:
            return torch.linalg.norm(self.a, ord=1)
        except ValueError:
            return 0

    def export(self, dom, obj='covar', dev=device, title=None, legend=False):
        r"""
        Export the measure plotted through a kernel

        Parameters
        ----------
        dom : Domain2D
            Domaine sur lequel va être calculée l'acquisition de :math:`m` dans
            :math:`\mathrm{L}^2(\mathcal{X})`.
        obj : str, optional
            Soit 'covar' pour reconstruire sur la covariance soit 'acquis'
            pour reconstruire sur la moyenne. The default is 'covar'.
        dev: str
            Either `cpu`, `cuda` (GPU par défaut) or `cuda:0`, `cuda:1`, etc.
        title : str, optional
            Title of the ouput image file. The default is None.
        legend : boolean, optional
            Boolean: should a legend be written. The default is True.

        Raises
        ------
        TypeError
            DESCRIPTION.

        Returns
        -------
        None.

        """
        result = self.kernel(dom, dev=dev) # Really fast if you have 
                                            # CUDA enabled
        if dev != 'cpu':
            result = result.to('cpu')
        if title is None:
            title = 'fig/reconstruction/experimentals.png'
        elif isinstance(title, str):
            title = 'fig/reconstruction/' + title + '.png'
        else:
            raise TypeError("You ought to give a str type name for the plot")
        if legend == True:
            if obj == 'covar':
                plt.title(r'Reconstruction $\Phi(m_{M,x})$ ' +
                          f'with N = {m.N}', fontsize=20)
            elif obj == 'acquis':
                plt.title(r'Reconstruction $\Phi(m_{a,x})$ ' +
                          f'with N = {m.N}', fontsize=20)
            plt.imshow(result, cmap='hot')
            plt.colorbar()
            plt.xlabel('X', fontsize=18)
            plt.ylabel('Y', fontsize=18)
            plt.colorbar()
            plt.show()
            plt.savefig(title, format='pdf', dpi=1000,
                    bbox_inches='tight', pad_inches=0.03)
        else:
            plt.imsave(title, result, cmap='hot', vmax=13)
            print(f"[+] {self} saved")


    def show(self, dom, acquis):
        """
        HIERHER: write it

        Parameters
        ----------
        dom : TYPE
            DESCRIPTION.
        acquis : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        plt.figure()
        cont1 = plt.contourf(dom.X, dom.Y, acquis, 100, cmap='gray')
        for c in cont1.collections:
            c.set_edgecolor("face")
        plt.colorbar()
        plt.scatter(self.x[:, 0], self.x[:, 1], marker='+', c='orange',
                    label='Spikes', s=dom.N_ech*self.a)
        plt.legend()


    def energie(self, dom, acquis, regul, obj='covar', bruits='gauss'):
        r"""
        Énergie de la mesure pour le problème Covenant
        :math:`(\mathcal{Q}_\lambda (y))` ou BLASSO sur
        l'acquisition moyenne :math:`(\mathcal{P}_\lambda (\overline{y}))`.

        Parameters
        ----------
        dom : Domain2D
            Domaine sur lequel va être calculée l'acquisition de :math:`m` dans
            :math:`\mathrm{L}^2(\mathcal{X})`.
        acquis : Tensor
            Vecteur de l'observation, pour comparer à l'action de l'opérateur
            adéquat sur la mesure :math:`m`.
        regul : double, optional
            Paramètre de régularisation `\lambda`.
        obj : str, optional
            Soit 'covar' pour reconstruire sur la covariance soit 'acquis'
            pour reconstruire sur la moyenne. The default is 'covar'.
        bruits : str, optional
            L'objet contient toutes les valeurs définissant le bruit à simuler.


        Raises
        ------
        NameError
            Le noyau n'est pas reconnu par la fonction.

        Returns
        -------
        double
            Évaluation de l'énergie :math:`T_\lambda` pour la mesure :math:`m`.

        """
        # normalis = torch.numel(acquis)
        normalis = 1
        if bruits == 'poisson':
            if obj == 'covar':
                R_nrj = self.cov_kernel(dom)
                attache = 0.5 * torch.linalg.norm(acquis - R_nrj)**2 / normalis
                parcimonie = regul*self.tv()
                return attache + parcimonie
            if obj == 'acquis':
                simul = self.kernel(dom)
                attache = 0.5 * torch.linalg.norm(acquis - simul)**2 / normalis
                parcimonie = regul*self.tv()
                return attache + parcimonie
        elif bruits in ('gauss', 'unif'):
            if obj == 'covar':
                R_nrj = self.cov_kernel(dom)
                attache = 0.5 * torch.linalg.norm(acquis - R_nrj)**2 / normalis
                parcimonie = regul*self.tv()
                return attache + parcimonie
            if obj == 'acquis':
                simul = self.kernel(dom)
                attache = 0.5 * torch.linalg.norm(acquis - simul)**2 / normalis
                parcimonie = regul*self.tv()
                return attache + parcimonie
            raise NameError("Unknown kernel")
        raise NameError("Unknown noise")

    def prune(self, tol=1e-4):
        r"""
        Retire les :math:`\delta`-pics avec une très faible amplitude (qui
        s'interpète comme des artefacts numériques).

        Parameters
        ----------
        tol : double, optional
            Tolérance en-dessous de laquelle les mesures de Dirac
            ne sont pas conservées. The default is 1e-3.

        Returns
        -------
        m : Mesure2D
            Mesure discrète sans les :math:`\delta`-pics de faibles amplitudes.


        """
        # nnz = np.count_nonzero(self.a)
        nnz_a = self.a.clone().detach()
        nnz = nnz_a > tol
        nnz_a = nnz_a[nnz]
        nnz_x = self.x.clone().detach()
        nnz_x = nnz_x[nnz]
        m = Mesure2D(nnz_a, nnz_x, dev=device)
        return m


# def merge_spikes(mes, tol=1e-3):
#     """
#     Retire les :math:`\delta`-pic doublons, sans considération sur l'amplitude.

#     Parameters
#     ----------
#     mes : Mesure2D
#         Mesure dont on veut retirer les :math:`\delta`-pics doublons.
#     tol : double, optional
#         Tolérance pour la distance entre les points. The default is 1e-3.

#     Returns
#     -------
#     new_mes : Mesure2D
#         Mesure sans les :math:`\delta`-pics doublons.

#     """
#     mat_dist = cdist(mes.x, mes.x)
#     idx_spurious = np.array([])
#     list_x = np.array([])
#     list_a = np.array([])
#     for j in range(mes.N):
#         for i in range(j):
#             if mat_dist[i, j] < tol:
#                 coord = [int(i), int(j)]
#                 idx_spurious = np.append(idx_spurious, np.array(coord))
#     idx_spurious = idx_spurious.reshape((int(len(idx_spurious)/2), 2))
#     idx_spurious = idx_spurious.astype(int)
#     if idx_spurious.size == 0:
#         return mes
#     else:
#         cancelled = []
#         for i in range(mes.N):
#             if i in cancelled or i in idx_spurious[:, 0]:
#                 cancelled = np.append(cancelled, i)
#             else:
#                 if list_x.size == 0:
#                     list_x = np.vstack([mes.x[i]])
#                 else:
#                     list_x = np.vstack([list_x, mes.x[i]])
#                 list_a = np.append(list_a, mes.a[i])
#         return Mesure2D(list_a, list_x)


def mesure_aleatoire(N, dom):
    r"""
    Créé une mesure aléatoire de N :math:`\delta`-pics d'amplitudes aléatoires
    comprises entre 0,5 et 1,5.

    Parameters
    ----------
    N : int
        Nombre de :math:`\delta`-pics à mettre dans la mesure discrète.
    dom : Domaine2D
        Domaine où les :math:`\delta`-pics vont être disposés.

    Returns
    -------
    m : :class:`Mesure2D`
        Mesure discrète composée de N :math:`\delta`-pics distincts.

    """
    x = dom.x_gauche + torch.rand(N, 2) * (dom.x_droit - dom.x_gauche)
    a = 0.5 + torch.rand(N)
    return Mesure2D(a, x)


def phi(m, dom, obj='covar'):
    r"""
    Calcule le résultat d'un opérateur d'acquisition à partir de la mesure m

    Parameters
    ----------
    m : Mesure2D
        Mesure discrète sur :math:`\mathcal{X}`.
    dom : Domain2D
        Domaine :math:`\mathcal{X}` sur lequel va être calculée l'acquisition
        de :math:`m` dans :math:`\mathrm{L}^2(\mathcal{X})` si l'objectif est
        l'acquisition ou :math:`\mathrm{L}^2(\mathcal{X^2})` si l'objectif est
        la covariance.
    obj : str, optional
        Donne l'objectif de l'opérateur. The default is 'covar'.

    Raises
    ------
    TypeError
        L'objectif de l'opérateur (ou son noyau) n'est pas reconnu.

    Returns
    -------
    Tensor
        Renvoie :math:`\Phi(m)` si l'objectif est l'acquisition, et
        :math:`\Lambda(m)` si l'objectif est la covariance.

    """
    if obj == 'covar':
        return m.cov_kernel(dom)
    if obj == 'acquis':
        return m.kernel(dom)
    raise NameError('Unknown BLASSO target.')


def phi_vecteur(a, x, dom, obj='covar'):
    r"""
    Créer le résultat d'un opérateur d'acquisition à partir des vecteurs
    a et x qui forme une mesure m_tmp

    Parameters
    ----------
    a : array_like
        Vecteur des luminosités, même nombre d'éléments que :math:`x`.
    x : array_like
        Vecteur des positions 2D, même nombre d'éléments que :math:`a`.
    dom : Domain2D
        Domaine :math:`\mathcal{X}` sur lequel va être intégré l'adjoint
        de :math:`m` dans :math:`\mathrm{L}^2(\mathcal{X})` si l'objectif est
        l'acquisition ou :math:`\mathrm{L}^2(\mathcal{X^2})` si l'objectif est
        la covariance.
    obj : str, optional
        Soit 'covar' pour reconstruire sur la covariance soit 'acquis' pour
        reconstruire sur la moyenne. The default is 'covar'.

    Raises
    ------
    TypeError
        L'objectif de l'opérateur (ou son noyau) n'est pas reconnu.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    if obj == 'covar':
        m_tmp = Mesure2D(a, x)
        return m_tmp.cov_kernel(dom)
    if obj == 'acquis':
        m_tmp = Mesure2D(a, x)
        return m_tmp.kernel(dom)
    raise TypeError('Unknown BLASSO target.')


def phiAdjoint(acquis, dom, obj='covar'):
    r"""
    Hierher débugger ; taille_x et taille_y pas implémenté

    Parameters
    ----------
    acquis : Tensor
        Soit l'acquisition moyenne :math:`y` soit la covariance :math:`R_y`.
    dom : Domain2D
        Domaine :math:`\mathcal{X}` sur lequel va être intégré l'adjoint
        de :math:`m` dans :math:`\mathrm{L}^2(\mathcal{X})` si l'objectif est
        l'acquisition ou :math:`\mathrm{L}^2(\mathcal{X^2})` si l'objectif est
        la covariance.
    obj : str, optional
        Soit 'covar' pour reconstruire sur la covariance soit 'acquis' pour
        reconstruire sur la moyenne. The default is 'covar'.

    Raises
    ------
    TypeError
        L'objectif de l'opérateur (ou son noyau) n'est pas reconnu.

    Returns
    -------
    eta : Tensor
        Fonction continue, élément de :math:`\mathscr{C}(\mathcal{X})`,
        discrétisée. Utile pour calculer la discrétisation du certificat
        :math:`\eta` associé à une mesure.

    """
    N_ech = dom.N_ech
    sigma = dom.sigma
    if obj == 'covar':
        (X_big, Y_big) = dom.big()
        h_vec = gaussienne_2D(X_big, Y_big, sigma, undivide=__normalis_PSF__)
        h_vec2 = torch.pow(h_vec, 2)
        h_ker = h_vec2.reshape(1, 1, N_ech*2-1 , N_ech*2-1)
        diacquis = torch.diag(torch.abs(acquis)).reshape(N_ech, N_ech)
        y_arr = diacquis.reshape(1, 1, N_ech , N_ech)
        eta = torch.nn.functional.conv2d(h_ker, y_arr, stride=1)
        eta = torch.flip(torch.squeeze(eta), [1, 0])
        return eta
    if obj == 'acquis':
        (X_big, Y_big) = dom.big()
        h_vec = gaussienne_2D(X_big, Y_big, sigma, undivide=__normalis_PSF__)
        h_ker = h_vec.reshape(1, 1, N_ech*2-1 , N_ech*2-1)
        y_arr = acquis.reshape(1, 1, N_ech , N_ech)
        eta = torch.nn.functional.conv2d(h_ker, y_arr, stride=1)
        eta = torch.flip(torch.squeeze(eta), [1, 0])
        return eta
    raise TypeError


def etak(mesure, acquis, dom, regul, obj='covar'):
    r"""Certificat dual :math:`\eta_\lambda` associé à la mesure
    :math:`m_\lambda`. Ce certificat permet de donner une approximation
    (sur la grille) de la position du Dirac de plus forte intensité du
    résidu.

    Parameters
    ----------
    mesure : Mesure2D
        Mesure discrète :math:`m` dont on veut obtenir le certificat duaL.
    acquis : Tensor
        Soit l'acquisition moyenne :math:`y` soit la covariance :math:`R_y`.
    dom : Domain2D
        Domaine :math:`\mathcal{X}` sur lequel va être intégré l'adjoint
        de :math:`m` dans :math:`\mathrm{L}^2(\mathcal{X})` si l'objectif est
        l'acquisition ou :math:`\mathrm{L}^2(\mathcal{X^2})` si l'objectif est
        la covariance.
    regul : double
        Paramètre de régularisation `\lambda`.
    obj : str, optional
        Soit 'covar' pour reconstruire sur la covariance soit 'acquis' pour
        reconstruire sur la moyenne. The default is 'covar'.

    Returns
    -------
    eta : Tensor
        Fonction continue, élément de :math:`\mathscr{C}(\mathcal{X})`,
        discrétisée. Certificat dual :math:`\eta` associé à la mesure.

    Notes
    -------
    Si l'objectif est la covariance :

    .. math:: \eta_\mathcal{Q} = \Lambda^\ast(\Lambda m - R_y)

    Si l'objectif est la moyenne :

    .. math:: \eta_\mathcal{P} = \Phi^\ast(\Phi m - \bar{y})

    """
    eta = 1/regul*phiAdjoint(acquis - phi(mesure, dom, obj), dom, obj)
    return eta


def pile_aquisition(m, dom, bru, T_ech, dev='cpu'):
    r"""Construit une pile d'acquisition à partir d'une mesure. Correspond à
    l'opérateur $\vartheta(\mu)$

    Parameters
    ----------
    m : Mesure2D
        Mesure discrète sur :math:`\mathcal{X}`.
    dom : Domain2D
        Domaine :math:`\mathcal{X}` sur lequel est défini :math:`m_{a,x}`
        ainsi que l'acquisition :math:`y(x,t)` , etc.
    bru : Bruits
        Contient les paramètres caractérisant le bruit qui pollue les données.
    T_ech : double
        Nombre d'images à générer.

    Returns
    -------
    acquis_temporelle : Tensor
        Vecteur 3D, la première dimension est celle du temps et les deux autres
        sont celles de l'espace :math:`\mathcal{X}`.

    """
    N_mol = len(m.a)
    taille = dom.N_ech
    if device == 'cuda':
        acquis_temporelle = torch.cuda.FloatTensor(T_ech,
                                                   taille,
                                                   taille).fill_(0)
    else:
        acquis_temporelle = torch.zeros(T_ech, taille, taille)
    for t in range(T_ech):
        a_tmp = torch.rand(N_mol).to(dev) * m.a
        m_tmp = Mesure2D(a_tmp, m.x)
        acquis_temporelle[t, :] = m_tmp.acquisition(dom, taille, bru)
    return acquis_temporelle


def covariance_pile(stack):
    r"""Calcule la covariance de y(x,t) à partir de la pile et de sa moyenne

    Parameters
    ----------
    stack : Tensor
        Matrice 3D, la première dimension est celle du temps et les deux autres
        sont celles de l'espace :math:`\mathcal{X}`. Aussi noté :math:`y(x,t)`.

    Returns
    -------
    R_y : Tensor
        Matrice de covariance de la pile d'acquisition.

    Notes
    -------
    La moyenne est donnée par :math:`x \in \mathcal{X}` :

    .. math:: \overline{y}(x) = \int_0^T y(x,t) \,\mathrm{d}t.

    La covariance est donnée par :math:`u,v \in \mathcal{X}` :

    .. math:: R_y(u,v) = \int_0^T (y(u,t) - \overline{y}(u))(y(v,t) -
                                            \overline{y}(v))\,\mathrm{d}t.

    """
    stack = stack.float()
    stack_re = stack.reshape(stack.shape[0],-1)
    stack_re -= stack_re.mean(0, keepdim=True)
    return 1/(stack_re.shape[0]-1) * stack_re.T @ stack_re


def unravel_index(indices, shape):
    r"""Converts flat indices into unraveled coordinates in a target shape.

    This is a `torch` implementation of `numpy.unravel_index`.

    Args:
        indices: A tensor of indices, (*, N).
        shape: The targeted shape, (D,).

    Returns:
        unravel coordinates, (*, N, D).
    """
    shape = torch.tensor(shape)
    indices = indices % shape.prod()  # prevent out-of-bounds indices

    coord = torch.zeros(indices.size() + shape.size(), dtype=int)

    for i, dim in enumerate(reversed(shape)):
        coord[..., i] = indices % dim
        indices = indices // dim

    coord_tensor = coord.flip(-1)
    return (coord_tensor.tolist()[0], coord_tensor.tolist()[1])


# Le fameux algo de Sliding Frank Wolfe
def SFW(acquis, dom, regul=1e-5, nIter=5, mes_init=None, mesParIter=False,
        obj='covar', dev=device, printInline=True):
    r"""Algorithme de Sliding Frank-Wolfe pour la reconstruction de mesures
    solution du BLASSO [1].

    Si l'objectif est la covariance :

    .. math:: \mathrm{argmin}_{m \in \mathcal{M(X)}} {T}_\lambda(m) =
        \lambda |m|(\mathcal{X}) + \dfrac{1}{2} ||R_y - \Lambda (m)||^2_2.
            \quad (\mathcal{Q}_\lambda (y))

    Si l'objectif est la moyenne :

    .. math:: \mathrm{argmin}_{m \in \mathcal{M(X)}} {S}_\lambda(m) =
        \lambda |m|(\mathcal{X}) +
            \dfrac{1}{2} ||\overline{y} - \Phi (m)||^2_2.
                \quad (\mathcal{P}_\lambda (\overline{y}))


    Paramètres
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

    Sorties
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

    Références
    ----------
    [1] Quentin Denoyelle, Vincent Duval, Gabriel Peyré, Emmanuel Soubies.
    The Sliding Frank-Wolfe Algorithm and its Application to Super-Resolution
    Microscopy. Inverse Problems, IOP Publishing, In press
    https://hal.archives-ouvertes.fr/hal-01921604
    """
    N_ech_y = dom.N_ech  # hierher à adapter
    N_grille = dom.N_ech**2
    acquis = acquis.to(dev)
    if obj == 'covar':
        N_grille = N_grille**2
    if mes_init == None:
        mesure_k = Mesure2D(dev=dev)
        a_k = torch.Tensor().to(dev)
        x_k = torch.Tensor().to(dev)
        x_k_demi = torch.Tensor().to(dev)
        Nk = 0
    else:
        mesure_k = mes_init
        a_k = mes_init.a
        x_k = mes_init.x
        Nk = mesure_k.N
    if mesParIter:
        mes_vecteur = torch.Tensor()
    nrj_vecteur = torch.zeros(nIter)
    N_vecteur = [Nk]
    for k in range(nIter):
        if printInline:
            print('\n' + 'Étape numéro ' + str(k))
        eta_V_k = etak(mesure_k, acquis, dom, regul, obj)
        certif_abs = torch.abs(eta_V_k)
        x_star_index = unravel_index(certif_abs.argmax(), eta_V_k.shape)
        x_star_tuple = tuple(s / N_ech_y for s in x_star_index)
        x_star = torch.tensor(x_star_tuple).reshape(1,2).to(dev)
        if printInline:
            print(fr'* x^* index {x_star_tuple} max ' +
                  fr'à {certif_abs[x_star_index]:.3e}')

        # Condition d'arrêt (étape 4)
        if torch.abs(eta_V_k[x_star_index]) < 1:
            nrj_vecteur[k] = mesure_k.energie(dom, acquis, regul, obj=obj)
            if printInline:
                print("\n\n---- Condition d'arrêt ----")
            if mesParIter:
                return(mesure_k, nrj_vecteur[:k], mes_vecteur)
            return(mesure_k, nrj_vecteur[:k])

        # Création du x positions estimées
        mesure_k_demi = Mesure2D()
        if not x_k.numel():
            x_k_demi = x_star
            a_param = (torch.ones(Nk+1, dtype=torch.float)
                       ).to(dev).detach().requires_grad_(True)
        else:
            x_k_demi = torch.cat((x_k, x_star))
            uno = torch.tensor([10.0], dtype=torch.float).to(dev).detach()
            a_param = torch.cat((a_k, uno))
            a_param.requires_grad = True

        # On résout LASSO (étape 7)
        mse_loss = torch.nn.MSELoss(reduction='sum')
        optimizer = torch.optim.LBFGS([a_param])
        alpha = regul
        n_epoch = 15

        for epoch in range(n_epoch):
            def closure():
                if torch.is_grad_enabled():
                    optimizer.zero_grad()
                outputs = phi_vecteur(a_param, x_k_demi, dom, obj)
                loss = 0.5 * mse_loss(acquis, outputs)
                loss += alpha * a_param.abs().sum()
                if loss.requires_grad:
                    loss.backward()
                del outputs
                return loss

            optimizer.step(closure)

        a_k_demi = a_param.detach().clone() # pour ne pas copier l'arbre
        del a_param, optimizer, mse_loss

        # print('* x_k_demi : ' + str(np.round(x_k_demi, 2)))
        # print('* a_k_demi : ' + str(np.round(a_k_demi, 2)))
        if printInline:
            print('* Optim convexe')
        mesure_k_demi += Mesure2D(a_k_demi, x_k_demi)

        # On résout double LASSO non-convexe (étape 8)
        param = torch.cat((a_k_demi, x_k_demi.reshape(-1)))
        param.requires_grad = True

        mse_loss = torch.nn.MSELoss(reduction='sum')
        if obj == 'acquis':
            optimizer = torch.optim.Adam([param])
            n_epoch = 30
        else:
            optimizer = torch.optim.Adam([param])
            n_epoch = 80

        for epoch in range(n_epoch):
            def closure():
                optimizer.zero_grad()
                x_tmp = param[Nk+1:].reshape(Nk+1, 2)
                fidelity = phi_vecteur(param[:Nk+1], x_tmp, dom, obj)
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

        # Mise à jour des paramètres avec retrait des Dirac nuls
        mesure_k = Mesure2D(a_k_plus, x_k_plus, dev=dev)
        mesure_k = mesure_k.prune()
        # mesure_k = merge_spikes(mesure_k)
        a_k = mesure_k.a
        x_k = mesure_k.x
        Nk = mesure_k.N
        if printInline:
            print(f'* Optim non-convexe : {Nk} δ-pics')

        # Graphe et énergie
        nrj_vecteur[k] = mesure_k.energie(dom, acquis, regul, obj)
        if printInline:
            print(f'* Énergie : {nrj_vecteur[k]:.3e}')
        if mesParIter == True:
            mes_vecteur = np.append(mes_vecteur, [mesure_k])
            with open('pickle/mes_' + obj + '.pkl', 'wb') as output:
                pickle.dump(mes_vecteur, output, pickle.HIGHEST_PROTOCOL)
        try:
            if (N_vecteur[-1] == N_vecteur[-2]
                and N_vecteur[-1] == N_vecteur[-3]
                and N_vecteur[-1] == N_vecteur[-4]):
                if printInline:
                    print('\n[!] Algorithme a optimisé')
                    print("\n\n---- Fin de la boucle ----")
                if mesParIter:
                    return(mesure_k, nrj_vecteur[:k], mes_vecteur)
                return(mesure_k, nrj_vecteur)
        except IndexError:
            pass
        N_vecteur = np.append(N_vecteur, Nk)
        # del (a_k_plus, x_k_plus, optimizer, mse_loss, x_k_demi, a_k_demi, 
        #      a_param)
        # torch.cuda.empty_cache()

    # Fin des itérations
    if printInline:
        print("\n\n---- Fin de la boucle ----")
    if mesParIter:
        return(mesure_k, nrj_vecteur, mes_vecteur)
    return(mesure_k, nrj_vecteur)


def non_convex_step(acquis, dom, regul, a_k_demi, x_k_demi, obj='covar'):
    """
    Step 8 of the SFW, to optimize both amplitude and position with fixed
    number of δ-measures. It's a hard non-convex optimization.
    
    Hierher : ne marche pas

    Parameters
    ----------
    acquis : ndarray
            Soit l'acquisition moyenne :math:`y` soit la covariance :math:`R_y`.
    dom : Domain2D
        Domaine :math:`\mathcal{X}` sur lequel est défini :math:`m_{a,x}`
        ainsi que l'acquisition :math:`y(x,t)` , etc.
    regul : double, optional
        Paramètre de régularisation :math:`\lambda`. The default is 1e-5.
    a_k_demi : ndarray
        Vecteur des amplitudes/luminosités avant optimisation non-convexe.
    x_k_demi : ndarray
        Vecteur des positions de la mesure avant optimisation non-convexe.
    obj : str, optional
        Soit `covar` pour reconstruire sur la covariance soit `acquis` pour
        reconstruire sur la moyenne. The default is 'covar'.

    Returns
    -------
    a_k_plus : ndarray
        Vecteur des amplitudes/luminosités solution de l'étape 8 du SFW.
    x_k_plus : ndarray
        Vecteur des positions solution de l'étape 8 du SFW.

    """
    
    Nk = len(a_k_demi)
    param = torch.cat((a_k_demi, x_k_demi.reshape(-1)))
    param.requires_grad = True

    mse_loss = torch.nn.MSELoss(reduction='sum')
    if obj == 'acquis':
        optimizer = torch.optim.Adam([param])
        n_epoch = 30
    else:
        optimizer = torch.optim.Adam([param])
        n_epoch = 80

    for epoch in range(n_epoch):
        def closure():
            optimizer.zero_grad()
            x_tmp = param[Nk+1:].reshape(Nk+1, 2)
            fidelity = phi_vecteur(param[:Nk+1], x_tmp, dom, obj)
            loss = 0.5 * mse_loss(acquis, fidelity)
            loss += regul * param[:1].abs().sum()
            loss.backward()
            del fidelity, x_tmp
            return loss

        optimizer.step(closure)

    a_k_plus = param[:int(len(param)/3)].detach().clone()
    x_k_plus = param[int(len(param)/3):].detach().clone().reshape(Nk+1, 2)
    return (Mesure2D(a_k_plus, x_k_plus))


def divide_and_conquer(stack, dom, quadrant_size=32, regul=1e-5, 
                       nIter=80, obj='covar', printInline=True):
    """
    Hierher : CUDA support

    Parameters
    ----------
    stack : Tensor
        L'acquisition dynamique :math:`y(x,t)`.
    dom : Domain2D
        Domaine :math:`\mathcal{X}` sur lequel est défini :math:`m_{a,x}`
        ainsi que l'acquisition :math:`y(x,t)` , etc.
    quadrant_size : int, optional
        Size of the quadrant to divide the domain. Only suitable to power of 2.
        The default is 32.
    regul : TYPE, optional
        DESCRIPTION. The default is 1e-5.
    nIter : int, optional
        Nombre d'itérations maximum pour l'algorithme. The default is 5.
    obj : str, optional
        Soit `covar` pour reconstruire sur la covariance soit `acquis` pour
        reconstruire sur la moyenne. The default is 'covar'.
    printInline : str, optional
        Ouput the log: final result of each conquer step. The default is False.


    Returns
    -------
    m_tot : TYPE
        DESCRIPTION.

    """
    m_tot = Mesure2D()
    GLOBAL_N_ECH = stack.size(-1)
    q = int(GLOBAL_N_ECH / quadrant_size)
    if printInline:
        print(f'\n[+] Beginning loop with {quadrant_size} × {quadrant_size} ' +
              f'samples from the {GLOBAL_N_ECH} × {GLOBAL_N_ECH} image')
    for i in range(q):
        for j in range(q):
            x_ht_gche = quadrant_size * i
            x_ht_droit = x_ht_gche + quadrant_size  # commande la ligne
            x_bas_gche = quadrant_size * j
            x_bas_droite = x_bas_gche + quadrant_size  # commande la colonne

            y_tmp = stack[:, x_ht_gche:x_ht_droit, x_bas_gche:x_bas_droite]
            y_tmp = y_tmp / y_tmp.max()
            pile_moy_tmp = y_tmp.mean(0)

            domain = Domain2D(0, 1, quadrant_size, dom.sigma)
            if obj == 'covar':
                R_y_tmp = covariance_pile(y_tmp)
                (m_tmp, nrj_tmp) = SFW(R_y_tmp, domain,
                                       regul=regul,
                                       nIter=nIter,
                                       obj='covar',
                                       printInline=True)
            if obj == 'acquis':
                y_bar_tmp = pile_moy_tmp.float()
                (m_tmp, nrj_tmp) = SFW(y_bar_tmp - y_bar_tmp.min(), 
                                       domain,
                                       regul=regul,
                                       nIter=nIter,
                                       obj='acquis',
                                       printInline=False)

            print(f'* quadrant ({i},{j}) = [{x_ht_gche},{x_ht_droit}] x ' +
                  f'[{x_bas_gche},{x_bas_droite}] : {m_tmp.N} δ-peaks')

            x_redress = (m_tmp.x[:, 0]*quadrant_size
                         + x_ht_gche)/GLOBAL_N_ECH
            y_redress = (m_tmp.x[:, 1]*quadrant_size
                          + x_bas_gche)/GLOBAL_N_ECH
            pos = torch.tensor((list(zip(x_redress, y_redress))))
            m_rajout = Mesure2D(m_tmp.a, pos)
            m_tot += m_rajout
    if printInline:
        print(f'[+] End of optmization: m_tot = {m_tot.N} δ-peaks')
    return m_tot


def wasserstein_metric(mes, m_zer, p_wasser=1):
    """
    Retourne la :math:`p`-distance de Wasserstein partiel (Partial Gromov-
    Wasserstein) pour des poids égaux (pas de prise en compte de la
    luminosité)

    Parameters
    ----------
    mes : Mesure2D
        A measure :math:`m` (typically the reconstructed one).
    m_zer : Mesure2D
        A measure :math:`m_0` (typically the reference/ground-truth one).
    p_wasser : int, optional
        Order of the Wasserstein distance :math:`\mathcal{W}^p`. Classic values
        are :math:`p = 1,2`. The default is 1.

    Raises
    ------
    ValueError
        The order :math:`p` you choose is not supported by the `ot` package.

    Returns
    -------
    W : float64
        The scalar value :math:`\mathcal{W}^p(m,m_0)`.

    """
    if p_wasser == 1:
        M = ot.dist(mes.x, m_zer.x, metric='euclidean')
    elif p_wasser == 2:
        M = ot.dist(mes.x, m_zer.x)
    else:
        raise ValueError('Unknown p for W_p computation')
    a = ot.unif(mes.N)
    b = ot.unif(m_zer.N)
    W = ot.emd2(a, b, M)
    return W


def cost_matrix_wasserstein(mes, m_zer, p_wasser=1):
    """
    Compute the cost matrix of order :math:`p` in the optimal transport 
    problem.

    Parameters
    ----------
    mes : Mesure2D
        A measure :math:`m` (typically the reconstructed one).
    m_zer : Mesure2D
        A measure :math:`m_0` (typically the reference/ground-truth one).
    p_wasser : int, optional
        Order of the Wasserstein distance :math:`\mathcal{W}^p`. Classic values
        are :math:`p = 1,2`. The default is 1.

    Raises
    ------
    ValueError
        The order :math:`p` you choose is not supported by the `ot` package.

    Returns
    -------
    M : ndarray
        The matrix of transport cost (known as the transport plan 
        :math:`\gamma` in the litterature).

    """
    if p_wasser == 1:
        M = ot.dist(mes.x, m_zer.x, metric='euclidean')
        return M
    elif p_wasser == 2:
        M = ot.dist(mes.x, m_zer.x)
        return M
    raise ValueError('Unknown p for W_p computation')


def SNR(signal, acquisition):
    """
    Compute the Signal to Noise Ratio of the stack.

    Parameters
    ----------
    signal : Tensor
        The ground-truth signal.
    acquisition : Tensor
        The observed acquisition.

    Returns
    -------
    double
        SNR of the stack.

    """
    return 10 * torch.log10(torch.norm(signal) / torch.norm(acquisition - signal))


def plot_results(m, m_zer, dom, bruits, y, nrj, certif, q=4, title=None,
                 saveFig=False, obj='covar'):
    '''Affiche tous les graphes importants pour la mesure m'''
    if m.a.numel() > 0:
        fig = plt.figure(figsize=(15, 12))
        # fig = plt.figure(figsize=(13,10))
        # type_de_bruits = bruits.type
        # niveau_bruits = bruits.niveau
        # fig.suptitle(fr'Reconstruction en bruits {type_de_bruits} ' +
        #              fr'pour $\lambda = {lambda_regul:.0e}$ ' +
        #              fr'et $\sigma_B = {niveau_bruits:.0e}$', fontsize=20)
        diss = wasserstein_metric(m, m_zer)
        fig.suptitle(f'Reconstruction {obj} : ' +
                     r'$\mathcal{{W}}_1(m, m_{a_0,x_0})$' +
                     f' = {diss:.3e}', fontsize=22)

        plt.subplot(221)
        cont1 = plt.contourf(dom.X, dom.Y, y, 100, cmap='gray')
        for c in cont1.collections:
            c.set_edgecolor("face")
        plt.colorbar()
        plt.scatter(m_zer.x[:, 0], m_zer.x[:, 1], marker='x',
                    label='GT spikes')
        plt.scatter(m.x[:, 0], m.x[:, 1], marker='+',
                    label='Recovered spikes')
        plt.legend()

        plt.xlabel('X', fontsize=18)
        plt.ylabel('Y', fontsize=18)
        plt.title(r'Temporal mean $\overline{y}$', fontsize=20)

        plt.subplot(222)
        super_dom = dom.super_resolve(q)
        cont2 = plt.contourf(super_dom.X,
                             super_dom.Y,
                             m.kernel(super_dom, dev='cpu'),
                             100, cmap='gray')
        for c in cont2.collections:
            c.set_edgecolor("face")
        plt.xlabel('X', fontsize=18)
        plt.ylabel('Y', fontsize=18)
        if obj == 'covar':
            plt.title(r'Reconstruction $\Lambda(m_{M,x})$ ' +
                      f'à N = {m.N}', fontsize=20)
        elif obj == 'acquis':
            plt.title(r'Reconstruction $\Phi(m_{a,x})$ ' +
                      f'with N = {m.N}', fontsize=20)
        plt.colorbar()

        plt.subplot(223)
        cont3 = plt.contourf(dom.X, dom.Y, certif, 100, cmap='gray')
        for c in cont3.collections:
            c.set_edgecolor("face")
        plt.xlabel('X', fontsize=18)
        plt.ylabel('Y', fontsize=18)
        plt.title(r'Certificate $\eta_\lambda$', fontsize=20)
        plt.colorbar()

        plt.subplot(224)
        plt.plot(nrj, 'o--', color='black', linewidth=2.5)
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        plt.xlabel('Itération', fontsize=18)
        plt.ylabel(r'$T_\lambda(m)$', fontsize=20)
        if obj == 'covar':
            plt.title(r'BLASSO energy $\mathcal{Q}_\lambda(y)$', fontsize=20)
        elif obj == 'acquis':
            plt.title(r'BLASSO energy $\mathcal{P}_\lambda(\overline{y})$',
                      fontsize=20)
        plt.grid()

        if title is None:
            title = 'fig/covar-certificat-2d.pdf'
        elif isinstance(title, str):
            title = 'fig/' + title + '.pdf'
        else:
            raise TypeError("You ought to give a str type name for the plot")
        if saveFig:
            plt.savefig(title, format='pdf', dpi=1000,
                        bbox_inches='tight', pad_inches=0.03)


def plot_experimental(m, dom, acquis, nrj, certif, q=4, title=None,
                      saveFig=False, obj='covar'):
    '''Affiche tous les graphes importants pour la mesure m sans connaissance
    de la mesure vérité-terrain.'''
    if m.a.numel() > 0:
        fig = plt.figure(figsize=(15, 12))
        fig.suptitle(f'Reconstruction {obj}', fontsize=22)

        plt.subplot(221)
        cont1 = plt.contourf(dom.X, dom.Y, acquis, 100, cmap='gray')
        for c in cont1.collections:
            c.set_edgecolor("face")
        plt.colorbar()
        plt.scatter(m.x[:, 0], m.x[:, 1], marker='+', c='orange',
                    label='Recovered spikes', s=dom.N_ech*m.a/8)
        plt.legend()

        plt.xlabel('X', fontsize=18)
        plt.ylabel('Y', fontsize=18)
        plt.title(r'Temporal mean $\overline{y}$', fontsize=20)

        plt.subplot(222)
        super_dom = dom.super_resolve(q)
        cont2 = plt.contourf(super_dom.X, super_dom.Y,
                             m.kernel(super_dom, dev='cpu'),
                             100, cmap='gray')
        for c in cont2.collections:
            c.set_edgecolor("face")
        plt.xlabel('X', fontsize=18)
        plt.ylabel('Y', fontsize=18)
        if obj == 'covar':
            plt.title(r'Reconstruction $\Lambda(m_{M,x})$ ' +
                      f'à N = {m.N}', fontsize=20)
        elif obj == 'acquis':
            plt.title(r'Reconstruction $\Phi(m_{a,x})$ ' +
                      f'à N = {m.N}', fontsize=20)
        plt.colorbar()

        plt.subplot(223)
        cont3 = plt.contourf(dom.X, dom.Y, certif, 100, cmap='gray')
        for c in cont3.collections:
            c.set_edgecolor("face")
        plt.xlabel('X', fontsize=18)
        plt.ylabel('Y', fontsize=18)
        plt.title(r'Certificate $\eta_\lambda$', fontsize=20)
        plt.colorbar()

        plt.subplot(224)
        plt.plot(nrj, 'o--', color='black', linewidth=2.5)
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        plt.xlabel('Itération', fontsize=18)
        plt.ylabel(r'$T_\lambda(m)$', fontsize=20)
        if obj == 'covar':
            plt.title(r'BLASSO energy $\mathcal{Q}_\lambda(y)$', fontsize=20)
        elif obj == 'acquis':
            plt.title(r'BLASSO energy $\mathcal{P}_\lambda(\overline{y})$',
                      fontsize=20)
        plt.grid()

        if title is None:
            title = 'fig/experimentals.pdf'
        elif isinstance(title, str):
            title = 'fig/' + title + '.pdf'
        else:
            raise TypeError("You ought to give a str type name for the plot")
        if saveFig:
            plt.savefig(title, format='pdf', dpi=1000,
                        bbox_inches='tight', pad_inches=0.03)


def plot_reconstruction(m, dom, acquis, q=4, title=None, saveFig=False,
                        obj='covar'):
    '''Affiche que 2 graphes importants pour la mesure m'''
    if m.a.numel() > 0:
        fig = plt.figure(figsize=(15, 6))
        fig.suptitle(f'Reconstruction {obj}', fontsize=22)

        plt.subplot(121)
        cont1 = plt.contourf(dom.X, dom.Y, acquis, 100, cmap='gray')
        for c in cont1.collections:
            c.set_edgecolor("face")
        plt.colorbar()
        plt.scatter(m.x[:, 0], m.x[:, 1], marker='+', c='orange',
                    label='Recovered spikes')
        plt.legend()

        plt.xlabel('X', fontsize=18)
        plt.ylabel('Y', fontsize=18)
        plt.title(r'Temporal mean $\overline{y}$', fontsize=20)

        plt.subplot(122)
        super_dom = dom.super_resolve(q)
        cont2 = plt.contourf(super_dom.X, super_dom.Y, m.kernel(super_dom),
                             100, cmap='gray')
        for c in cont2.collections:
            c.set_edgecolor("face")
        plt.xlabel('X', fontsize=18)
        plt.ylabel('Y', fontsize=18)
        if obj == 'covar':
            plt.title(r'Reconstruction $\Lambda(m_{M,x})$ ' +
                      f'à N = {m.N}', fontsize=20)
        elif obj == 'acquis':
            plt.title(r'Reconstruction $\Phi(m_{a,x})$ ' +
                      f'à N = {m.N}', fontsize=20)
        plt.colorbar()
        if title is None:
            title = 'fig/experimentals.pdf'
        elif isinstance(title, str):
            title = 'fig/' + title + '.pdf'
        else:
            raise TypeError("You ought to give a str type name for the plot")
        if saveFig:
            plt.savefig(title, format='pdf', dpi=1000,
                        bbox_inches='tight', pad_inches=0.03)


def gif_pile(pile_acquis, m_zer, y_moy, dom, video='gif', title=None):
    '''Hierher à terminer de débogger'''
    # ax, fig = plt.subplots(figsize=(10,10))
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    # ax.set(xlim=(0, 1), ylim=(0, 1))
    cont_pile = ax.contourf(dom.X, dom.Y, y_moy, 100, cmap='seismic')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.15)
    fig.colorbar(cont_pile, cax=cax)
    # plt.tight_layout()

    def animate(k):
        if k >= len(pile_acquis):
            # On fige l'animation pour faire une pause à la fin
            return
        ax.clear()
        ax.set_aspect('equal', adjustable='box')
        ax.contourf(dom.X, dom.Y, pile_acquis[k, :], 100, cmap='seismic')
        ax.scatter(m_zer.x[:, 0], m_zer.x[:, 1], marker='x',
                   s=2*dom.N_ech, label='GT spikes')
        ax.set_xlabel('X', fontsize=25)
        ax.set_ylabel('Y', fontsize=25)
        ax.set_title(f'Acquisition numéro = {k}', fontsize=30)
        # ax.legend(loc=1, fontsize=20)
        # ax.tight_layout()

    anim = FuncAnimation(fig, animate, interval=400,
                         frames=len(pile_acquis)+3,
                         blit=False)
    plt.draw()

    if title is None:
        title = 'fig/anim/anim-pile-2d'
    elif isinstance(title, str):
        title = 'fig/anim/' + title
    else:
        raise TypeError("You ought to give a str type name for the video file")

    if video == "mp4":
        anim.save(title + '.mp4')
    elif video == "gif":
        anim.save(title + '.gif')
    else:
        raise ValueError('Unknown video format')
    return fig


def gif_results(acquis, m_zer, m_list, dom, step=1000, video='gif',
                title=None):
    '''Montre comment la fonction SFW ajoute ses Dirac'''
    fig = plt.figure(figsize=(20, 10))

    ax1 = fig.add_subplot(121)
    ax1.set_aspect('equal', adjustable='box')
    cont = ax1.contourf(dom.X, dom.Y, acquis, 100, cmap='seismic')
    divider = make_axes_locatable(ax1)  # pour paramétrer colorbar
    cax = divider.append_axes("right", size="5%", pad=0.15)
    fig.colorbar(cont, cax=cax)
    ax1.set_xlabel('X', fontsize=25)
    ax1.set_ylabel('Y', fontsize=25)
    ax1.set_title(r'Moyenne $\overline{y}$', fontsize=35)

    ax2 = fig.add_subplot(122)
    # ax2.set(xlim=(0, 1), ylim=(0, 1))
    cont_sfw = ax2.contourf(dom.X, dom.Y, acquis, 100, cmap='seismic')
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="5%", pad=0.15)
    fig.colorbar(cont_sfw, cax=cax)
    ax2.contourf(dom.X, dom.Y, acquis, 100, cmap='seismic')
    ax2.scatter(m_zer.x[:, 0], m_zer.x[:, 1], marker='x',
                s=dom.N_ech, label='Hidden spikes')
    ax2.scatter(m_list[0].x[:, 0], m_list[0].x[:, 1], marker='+',
                s=2*dom.N_ech, c='g', label='Recovered spikes')
    ax2.legend(loc=1, fontsize=20)
    plt.tight_layout()

    def animate(k):
        if k >= len(m_list):
            # On fige l'animation pour faire une pause à la pause
            return
        ax2.clear()
        ax2.set_aspect('equal', adjustable='box')
        ax2.contourf(dom.X, dom.Y, m_list[k].kernel(dom), 100,
                     cmap='seismic')
        ax2.scatter(m_zer.x[:, 0], m_zer.x[:, 1], marker='x',
                    s=4*dom.N_ech, label='GT spikes')
        ax2.scatter(m_list[k].x[:, 0], m_list[k].x[:, 1], marker='+',
                    s=8*dom.N_ech, c='g', label='Recovered spikes')
        ax2.set_xlabel('X', fontsize=25)
        ax2.set_ylabel('Y', fontsize=25)
        ax2.set_title(f'Reconstruction itération = {k}', fontsize=35)
        ax2.legend(loc=1, fontsize=20)
        plt.tight_layout()

    anim = FuncAnimation(fig, animate, interval=step, frames=len(m_list)+3,
                         blit=False)

    plt.draw()

    if title is None:
        title = 'fig/anim/anim-sfw-covar-2d'
    elif isinstance(title, str):
        title = 'fig/anim/' + title
    else:
        raise TypeError("You ought to give a str type name for the video file")

    if video == "mp4":
        anim.save(title + '.mp4')
    elif video == "gif":
        anim.save(title + '.gif')
    else:
        raise ValueError('Unknown video format')
    return fig


def gif_experimental(acquis, m_list, dom, step=100, cross=True, video='gif',
                     title=None, obj='covar'):
    '''Montre comment la fonction SFW ajoute ses Dirac'''
    fig = plt.figure(figsize=(20, 10))

    ax1 = fig.add_subplot(121)
    ax1.set_aspect('equal', adjustable='box')
    # cont = ax1.contourf(dom.X, dom.Y, acquis, 100, cmap='seismic')
    cont = ax1.imshow(acquis, cmap='hot', interpolation='nearest')
    divider = make_axes_locatable(ax1)  # pour paramétrer colorbar
    cax = divider.append_axes("right", size="5%", pad=0.15)
    fig.colorbar(cont, cax=cax)
    ax1.set_xlabel('X', fontsize=25)
    ax1.set_ylabel('Y', fontsize=25)
    ax1.set_title(r'Temporal mean $\overline{y}$', fontsize=35)

    ax2 = fig.add_subplot(122)
    # cont_sfw = ax2.contourf(dom.X, dom.Y, acquis, 100, cmap='seismic')
    cont_sfw = ax2.imshow(acquis, cmap='hot', interpolation='nearest')
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="5%", pad=0.15)
    fig.colorbar(cont_sfw, cax=cax)
    # ax2.contourf(dom.X, dom.Y, acquis, 100, cmap='seismic')
    ax2.imshow(acquis, cmap='hot')
    if cross:
        taille = dom.N_ech
        ax2.scatter(taille*m_list[0].x[:, 0], taille * (1-m_list[0].x[:, 1]),
                    marker='+', s=dom.N_ech*m_list[0].a/10, c='orange',
                    label='Recovered spikes')
        ax2.legend(loc=1, fontsize=20)
    plt.tight_layout()

    def animate(k):
        if k >= len(m_list):
            # On fige l'animation pour faire une pause à la pause
            return

        ax2.clear()
        ax2.set_aspect('equal', adjustable='box')
        # ax2.contourf(dom.X, dom.Y, m_list[k].kernel(dom), 100,
        #              cmap='seismic')
        ax2.imshow(m_list[k].kernel(dom), cmap='hot', interpolation='nearest')
        if cross:
            ax2.scatter(taille*(m_list[k].x[:, 1]), taille*m_list[k].x[:, 0],
                        marker='+', s=dom.N_ech*m_list[k].a/10, c='orange',
                        label='Recovered spikes')
            # ax2.set_ylim([0, 1])
            # ax2.set_xlim([0, 1])
            ax2.legend(loc=1, fontsize=20)
        ax2.set_xlabel('X', fontsize=25)
        ax2.set_ylabel('Y', fontsize=25)
        if obj =='covar':
            ax2.set_title(f'$\Lambda(m_{{M,x}})$ at iterate {k}', fontsize=35)
        elif obj =='acquis':
            ax2.set_title(f'$\Phi(m_{{a,x}})$ at iterate {k}', fontsize=35)
        else:
            raise TypeError("Unknown model objective")
        plt.tight_layout()

    anim = FuncAnimation(fig, animate, interval=step, frames=len(m_list)+3,
                         blit=False)

    plt.draw()

    if title is None:
        title = 'fig/anim/anim-sfw-covar-2d'
    elif isinstance(title, str):
        title = 'fig/anim/' + title
    else:
        raise TypeError("You ought to give a str type name for the video file")

    if video == "mp4":
        anim.save(title + '.mp4')
    elif video == "gif":
        anim.save(title + '.gif')
    else:
        raise ValueError('Unknown video format')
    return fig


def compare_covariance(m, covariance, dom, saveFig=True):
    fig = plt.figure(figsize=(12, 4))
    ax1 = fig.add_subplot(121)
    lambada = ax1.imshow(m.covariance_kernel(dom))
    fig.colorbar(lambada)
    ax1.set_title(r'$\Lambda(m_{M,x})$', fontsize=40)
    ax2 = fig.add_subplot(122)
    ax2.imshow(covariance)
    fig.colorbar(lambada)
    ax2.set_title(r'$R_y$', fontsize=40)
    if saveFig:
        plt.savefig('fig/compared_covariances.pdf', format='pdf', dpi=1000,
                    bbox_inches='tight', pad_inches=0.03)


def trace_ground_truth(m_ax0, reduc=2, saveFig=True):
    plt.scatter(m_ax0.x[0:-1:reduc, 0], 1 - m_ax0.x[0:-1:reduc, 1],
                marker='x', s=0.001)
    plt.title('Ground-truth position of ẟ-peaks', fontsize=20)
    plt.xlabel('x', fontsize=20)
    plt.ylabel('y', fontsize=20)
    if saveFig:
        plt.savefig('fig/gt.pdf', format='pdf', dpi=1000,
                    bbox_inches='tight', pad_inches=0.03)




if __name__ == 'main':
    N_ECH = 32
    X_GAUCHE = 0
    X_DROIT = 1
    # SIGMA = 1e-1
    SIGMA = 3 # en fait c'est la freq de coupure HIERHER à corriger
    domain = Domain2D(X_GAUCHE, X_DROIT, N_ECH, SIGMA, dev=device)
    domain_cpu = Domain2D(X_GAUCHE, X_DROIT, N_ECH, SIGMA)
    
    a = torch.Tensor([1,2])
    x = torch.Tensor([[0.1, 0.5], [0.7, 0.2]])
    x2 = torch.Tensor([[0.3, 0.4], [0.5, 0.5]])
    
    a = torch.tensor([8, 10, 6, 7, 9])
    x = torch.tensor([[0.2, 0.23], [0.90, 0.95],
                      [0.33, 0.82], [0.33, 0.30],
                      [0.23, 0.38]])
    
    m = Mesure2D(a, x, dev=device)
    m_cpu = Mesure2D(a, x, dev='cpu')
    # m2 = Mesure2D(a, x2)
    
    # plt.figure()
    # plt.imshow(m.kernel(domain).to('cpu'), extent=[0,1,1,0])
    # plt.title('$\Phi(m)$', fontsize=28)
    # plt.colorbar()
    
    # plt.figure()
    # plt.imshow(m.cov_kernel(domain).to('cpu'), extent=[0,1,1,0])
    # plt.colorbar()
    # plt.title('$\Lambda(m)$', fontsize=28)
    
    
    FOND = 5e-2
    SIGMA_BRUITS = 1e-2
    TYPE_BRUITS = 'gauss'
    bruits_t = Bruits(FOND, SIGMA_BRUITS, TYPE_BRUITS)
    
    
    T_ECH = 500
    pile = pile_aquisition(m, domain, bruits_t, T_ECH, dev=device)
    pile = pile / torch.max(pile)
    y_bar = pile.mean(0)
    y_bar_cpu = y_bar.to('cpu')
    cov_pile = covariance_pile(pile)
    R_y = cov_pile
    
    # plt.figure()
    # plt.scatter(x[:,0], x[:,1])
    # plt.imshow(cov_pile)
    # plt.colorbar()
    # plt.title('$R_y$', fontsize=28)
    
    
    lambda_cov = 1e-6
    lambda_moy = 1e-5
    iteration = m.N
    
    (m_cov, nrj_cov, mes_cov) = SFW(R_y, domain, regul=lambda_cov,
                                              nIter=iteration, mesParIter=True,
                                              obj='covar', printInline=True)
    
    (m_moy, nrj_moy, mes_moy) = SFW(y_bar , domain,
                                    regul=lambda_moy,
                                    nIter=iteration, mesParIter=True,
                                    obj='acquis', printInline=True)
    

    
    print(f'm_cov : {m_cov.N} δ-pics')
    print(f'm_moy : {m_moy.N} δ-pics')
    # print(m_moy)
    
    
    if m_cov.N > 0:
        certificat_V_cov = etak(m_cov, R_y, domain, lambda_cov,
                                obj='covar').to('cpu')
        m_cov.to('cpu')
        plot_results(m_cov, m_cpu, domain_cpu, bruits_t, y_bar_cpu, nrj_cov,
                      certificat_V_cov, obj='covar')
    if m_moy.N > 0:
        certificat_V_moy = etak(m_moy, y_bar, domain, lambda_moy,
                                obj='acquis').to('cpu')
        m_moy.to('cpu')
        plot_results(m_moy, m_cpu, domain_cpu, bruits_t, y_bar_cpu, nrj_moy,
                      certificat_V_moy, obj='acquis')





# # Calcul certificat P_\lambda
# (X_big, Y_big) = domain.big()
# h_vec = gaussienne_2D(X_big, Y_big, SIGMA, undivide=__normalis_PSF__)
# h_ker = h_vec.reshape(1, 1, N_ECH*2-1 , N_ECH*2-1)
# y_arr = y_bar.reshape(1, 1, N_ECH , N_ECH)
# etas = torch.nn.functional.conv2d(h_ker, y_arr, stride=1)
# etap = torch.flip(torch.squeeze(etas), [0, 1])

# plt.figure()
# plt.imshow(etap, extent=[0,1,1,0])
# plt.colorbar()
# plt.title('$\eta_\lambda^{\mathcal{P}}$', fontsize=28)


# # # Calcul certificat Q_\lambda
# # (X_big, Y_big) = domain.biggy()
# # h_vec = gaussienne_2D(X_big, Y_big, SIGMA,
# #                       undivide=__normalis_PSF__).reshape(-1).cuda()
# # lambda_vec = torch.outer(h_vec, h_vec)[:-1,:-1]
# # h_ker = lambda_vec.reshape(1, 1, 4*N_ECH**2 -1 , 4*N_ECH**2-1)
# # R_y_arr = cov_pile.reshape(1, 1, N_ECH**2 , N_ECH**2).cuda()
# # eta = torch.nn.functional.conv2d(h_ker, R_y_arr, stride=1)
# # eta = torch.diagonal(torch.squeeze(eta)).cpu()
# # plt.imshow(eta.reshape(N_ECH, N_ECH))


# (Xt, Yt) = domain.reverse()
# h_vec = gaussienne_2D(Xt, Yt, SIGMA, undivide=__normalis_PSF__).reshape(-1)
# lambda_vec = torch.outer(h_vec, h_vec)
# R_y_arr = cov_pile
# eta = R_y_arr
# output = torch.diag(torch.abs(eta)).reshape(N_ECH, N_ECH)

# plt.figure()
# plt.imshow(output, extent=[0,1,1,0])
# plt.colorbar()
# plt.title('$\eta_\lambda^{\mathcal{Q}}$', fontsize=28)





# #%% Etape 7


# x_k_demi = torch.tensor([[0.1,0.2], [0.3,0.2]])
# a_k_demi = torch.tensor([1, 4.56])
# m_tmp = Mesure2D(a_k_demi, x_k_demi)
# acquis = m_tmp.kernel(domain)
# # acquis = m_tmp.cov_kernel(domain)

# N_grille = acquis.numel()
# regul = 1e-5


# a_param = torch.tensor([3,3], dtype=torch.float, requires_grad=True)

# mse_loss = torch.nn.MSELoss(reduction='sum')
# optimizer = torch.optim.LBFGS([a_param], lr=1)
# alpha = regul
# n_epoch = 10

# for epoch in range(n_epoch):
#     def closure():
#         optimizer.zero_grad()
#         outputs = phi_vecteur(a_param, x_k_demi, domain, obj='acquis')
#         loss = 0.5 * mse_loss(acquis, outputs)

#         loss += alpha * a_param.abs().sum()
#         loss.backward()
#         return loss

#     optimizer.step(closure)

# print(a_param)



# #%% Etape 8

# a_k_demi = torch.tensor([1])
# x_k_demi = torch.tensor([[0.1,0.2]])

# m_tmp = Mesure2D(a_k_demi, x_k_demi)
# acquis = m_tmp.kernel(domain)
# acquis = m_tmp.cov_kernel(domain)


# N_grille = acquis.numel()
# regul = 1e-5


# a_param = torch.tensor([1.3], dtype=torch.float)
# x_param = torch.tensor([[0.085, 0.21]], dtype=torch.float)

# param = torch.cat((a_param, x_param.reshape(-1)))
# param.requires_grad = True

# with torch.no_grad():
#     param[1:] = param[1:].clamp(0, +1)

# mse_loss = torch.nn.MSELoss(reduction='sum')
# optimizer = torch.optim.Adam([param])
# alpha = regul
# n_epoch = 10

# for epoch in range(n_epoch):
#     def closure():
#         optimizer.zero_grad()
#         x_tmp = param[1:].reshape(1,2)
#         fidelity = phi_vecteur(param[:1], x_tmp, domain, obj='covar')
#         loss = 0.5 * mse_loss(acquis, fidelity)

#         loss += alpha * param[:1].abs().sum()
#         loss.backward()
#         return loss

#     optimizer.step(closure)

# print(param)

# #%%

# plt.figure()
# plt.subplot(121)
# plt.imshow(cov_pile.to('cpu'))
# plt.colorbar()
# plt.subplot(122)
# plt.imshow(phi(m_cpu, domain_cpu))


# plt.figure()
# plt.imshow(cov_pile.to('cpu') - phi(m_cpu, domain_cpu));plt.colorbar()

# plt.figure()
# plt.imshow(phiAdjoint(cov_pile.to('cpu') - phi(m_cpu, domain_cpu), 
#                       domain_cpu))
# plt.colorbar()


# Biblio
# http://sagecal.sourceforge.net/pytorch/index.html
# LazyTensor https://www.kernel-operations.io/keops/_auto_tutorials/a_LazyTensors/plot_lazytensors_a.html
# 
