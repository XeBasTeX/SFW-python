#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Attemps to provide the covenant package with a GPU acceleration with CUDA. 
Provides  the essential functions for simulating nD discrete Radon measures
and reconstructing those measures w.r.t to a provided acquistion

Created on Mon Mar 22 08:49:01 2021

@author: Bastien (https://github.com/XeBasTeX)
"""


__normalis_PSF__ = False


import torch
import numpy as np
import matplotlib.pyplot as plt
import ot



def sum_normalis(X_domain, Y_domain, sigma_g):
    """
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
    """
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
        sum_normalis = torch.sum(expo * normalis)
        return expo * normalis / sum_normalis
    else:
        return expo * normalis


def grad_x_gaussienne_2D(X_domain, Y_domain, X_deriv, sigma_g, normalis=None):
    """
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
    """
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


class Domain2D:
    def __init__(self, gauche, droit, ech, sigma_psf):
        '''Hierher ne marche que pour des grilles carrées'''
        grille = torch.linspace(gauche, droit, ech)
        X_domain, Y_domain = torch.meshgrid(grille, grille)
        self.x_gauche = gauche
        self.x_droit = droit
        self.N_ech = ech
        self.X_grid = grille
        self.X = X_domain
        self.Y = Y_domain
        self.sigma = sigma_psf

    def get_domain(self):
        return(self.X, self.Y)

    def compute_square_mesh(self):
        """
        Renvoie les grilles discrétisant le domaine :math:`\mathcal{X}` à N_ech

        Returns
        -------
        Tensor
            Les grilles coordonnées de x et des y.

        """
        return torch.meshrgrid(self.X_grid)

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
                               2*self.N_ech - 1)
        X_big, Y_big = torch.meshgrid(grid_big, grid_big)
        return(X_big, Y_big)

    def super_resolve(self, q=4):
        super_N_ech = q * self.N_ech
        return Domain2D(self.x_gauche, self.x_droit, super_N_ech, self.sigma)


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
    def __init__(self, amplitude=None, position=None):
        if amplitude is None or position is None:
            amplitude = torch.Tensor()
            position = torch.Tensor()
        if len(amplitude) != len(position):
            raise ValueError('Pas le même nombre')
        if isinstance(amplitude, torch.Tensor) and isinstance(position,
                                                              torch.Tensor):
            self.a = amplitude
            self.x = position
        elif isinstance(amplitude, np.ndarray) and isinstance(position,
                                                            np.ndarray):
            self.a = torch.from_numpy(amplitude)
            self.x = torch.from_numpy(position)
        else:
            raise TypeError("Gros chien de la casse formate ta bouse")
        self.N = len(amplitude)

    def __add__(self, m):
        '''Hieher : il faut encore régler l'addition pour les mesures au même
        position, ça vaudrait le coup de virer les duplicats'''
        a_new = torch.cat((self.a, m.a))
        x_new = torch.cat((self.x, m.x))
        return Mesure2D(a_new, x_new)

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
        return(f"{self.N} Diracs \nAmplitudes : {self.a}" +
               f"\nPositions : {self.x}")

    def kernel(self, dom, noyau='gaussienne'):
        """
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
        X_domain = dom.X
        Y_domain = dom.Y
        sigma = dom.sigma
        N = self.N
        x = self.x
        a = self.a
        acquis = X_domain*0
        if noyau == 'gaussienne':
            for i in range(0, N):
                acquis += a[i]*gaussienne_2D(X_domain - x[i, 0],
                                             Y_domain - x[i, 1], sigma)
            return acquis
        if noyau == 'laplace':
            raise TypeError("Pas implémenté.")
        if noyau == 'fourier':
            raise TypeError("Pas implémenté.")
        raise NameError("Unknown kernel.")

    def cov_kernel(self, dom):
        """
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
        acquis = torch.zeros(N_ech**2, N_ech**2)
        for i in range(0, N):
            noyau = gaussienne_2D(X_domain - x[i, 0], Y_domain - x[i, 1], 
                                  sigma)
            noyau_re = noyau.reshape(-1)
            acquis += amp[i] * torch.outer(noyau_re, noyau_re)
        return acquis


    def acquisition(self, dom, echantillo, bru):
        """
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
            w = nv*np.random.random_sample((echantillo, echantillo))
            simul = self.kernel(dom, noyau='gaussienne')
            acquis = simul + w + fond
            return acquis
        if type_de_bruits == 'gauss':
            simul = self.kernel(dom, noyau='gaussienne')
            w = torch.normal(0, nv, size=(echantillo, echantillo))
            acquis = simul + w + fond
            return acquis
        if type_de_bruits == 'poisson':
            w = nv*torch.poisson(dom.X)
            simul = w*self.kernel(dom, noyau='gaussienne')
            acquis = simul + fond
        raise NameError("Unknown type of noise")

    def graphe(self, X_domain, Y_domain, lvl=50):
        '''Trace le contourf de la mesure'''
        # f = plt.figure()
        plt.contourf(X_domain, Y_domain, self.kernel(X_domain, Y_domain),
                     lvl, label='$y_0$', cmap='hot')
        plt.xlabel('X', fontsize=18)
        plt.ylabel('Y', fontsize=18)
        plt.title('Acquisition y', fontsize=18)
        plt.colorbar()
        plt.grid()
        plt.show()

    def tv(self):
        """
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

    def energie(self, dom, acquis, regul, obj='covar', bruits='gauss'):
        """
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
        if bruits == 'poisson':
            normalis = torch.numel(acquis)
            if obj == 'covar':
                R_nrj = self.covariance_kernel(dom)
                attache = 0.5 * torch.linalg.norm(acquis - R_nrj)**2 / normalis
                parcimonie = regul*self.tv()
                return attache + parcimonie
            if obj == 'acquis':
                simul = self.kernel(dom)
                attache = 0.5 * torch.linalg.norm(acquis - simul)**2 / normalis
                parcimonie = regul*self.tv()
                return attache + parcimonie
        elif bruits in ('gauss', 'unif'):
            normalis = torch.numel(acquis)
            if obj == 'covar':
                R_nrj = self.covariance_kernel(dom)
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

    def prune(self, tol=1e-3):
        """
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
        m = Mesure2D(nnz_a, nnz_x)
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
    """
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
    """
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
        return m.covariance_kernel(dom)
    if obj == 'acquis':
        return m.kernel(dom)
    raise NameError('Unknown BLASSO target.')


def phi_vecteur(a, x, dom, obj='covar'):
    """
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
        return m_tmp.covariance_kernel(dom)
    if obj == 'acquis':
        m_tmp = Mesure2D(a, x)
        return m_tmp.kernel(dom)
    raise TypeError('Unknown BLASSO target.')


# def phiAdjointSimps(acquis, dom, obj='covar'):
#     '''Hierher débugger ; taille_x et taille_y pas implémenté
#     Partie torch à faire'''
#     X_domain = dom.X
#     Y_domain = dom.Y
#     taille_y = len(X_domain)
#     taille_x = len(X_domain[0])
#     ech_x_square = X_domain.size
#     ech_y_square = Y_domain.size
#     eta = np.zeros(np.shape(X_domain))
#     if obj == 'covar':
#         for i in range(taille_x):
#             for j in range(taille_y):
#                 x_decal = X_domain[i, j]
#                 y_decal = Y_domain[i, j]
#                 convol = gaussienne_2D(x_decal - X_domain, y_decal - Y_domain,
#                                        dom.sigma)
#                 noyau = np.outer(convol, convol)
#                 X_range_integ = np.linspace(dom.x_gauche, dom.x_droit,
#                                             ech_x_square)
#                 Y_range_integ = np.linspace(dom.x_gauche, dom.x_droit,
#                                             ech_y_square)
#                 integ_x = integrate.trapz(acquis * noyau, x=X_range_integ)
#                 eta[i, j] = integrate.trapz(integ_x, x=Y_range_integ)
#         return eta
#     if obj == 'acquis':
#         for i in range(taille_x):
#             for j in range(taille_y):
#                 x_decal = X_domain[i, j]
#                 y_decal = Y_domain[i, j]
#                 gauss = gaussienne_2D(x_decal - X_domain, y_decal - Y_domain,
#                                       dom.sigma)
#                 integ_x = integrate.simps(acquis * gauss, x=dom.X_grid)
#                 eta[i, j] = integrate.simps(integ_x, x=dom.X_grid)
#         return eta
#     raise TypeError


# def phiAdjoint(acquis, dom, obj='covar'):
#     """
#     Hierher débugger ; taille_x et taille_y pas implémenté

#     Parameters
#     ----------
#     acquis : Tensor 
#         Soit l'acquisition moyenne :math:`y` soit la covariance :math:`R_y`.
#     dom : Domain2D
#         Domaine :math:`\mathcal{X}` sur lequel va être intégré l'adjoint 
#         de :math:`m` dans :math:`\mathrm{L}^2(\mathcal{X})` si l'objectif est 
#         l'acquisition ou :math:`\mathrm{L}^2(\mathcal{X^2})` si l'objectif est 
#         la covariance.
#     obj : str, optional
#         Soit 'covar' pour reconstruire sur la covariance soit 'acquis' pour
#         reconstruire sur la moyenne. The default is 'covar'.

#     Raises
#     ------
#     TypeError
#         L'objectif de l'opérateur (ou son noyau) n'est pas reconnu.

#     Returns
#     -------
#     eta : Tensor
#         Fonction continue, élément de :math:`\mathscr{C}(\mathcal{X})`,
#         discrétisée. Utile pour calculer la discrétisation du certificat 
#         :math:`\eta` associé à une mesure.

#     """
#     N_ech = dom.N_ech
#     eta = np.zeros(np.shape(dom.X))
#     if obj == 'covar':
#         (X_big, Y_big) = dom.big()
#         sigma = dom.sigma
#         h_vec = gaussienne_2D(X_big, Y_big, sigma, undivide=__normalis_PSF__)
#         convol_row = scipy.signal.convolve2d(acquis, h_vec, 'same').T/N_ech**2
#         adj = np.diag(scipy.signal.convolve2d(convol_row, h_vec, 'same'))
#         eta = adj.reshape(N_ech, N_ech) / N_ech**2
#         return eta
#     if obj == 'acquis':
#         (X_big, Y_big) = dom.big()
#         sigma = dom.sigma
#         out = gaussienne_2D(X_big, Y_big, sigma, undivide=__normalis_PSF__)
#         eta = scipy.signal.convolve2d(out, acquis, mode='valid') / N_ech**2
#         return eta
#     raise TypeError


# def etak(mesure, acquis, dom, regul, obj='covar'):
#     r"""Certificat dual :math:`\eta_\lambda` associé à la mesure 
#     :math:`m_\lambda`. Ce certificat permet de donner une approximation 
#     (sur la grille) de la  position du Dirac de plus forte intensité du 
#     résidu.

#     Parameters
#     ----------
#     mesure : Mesure2D
#         Mesure discrète :math:`m` dont on veut obtenir le certificat duaL.
#     acquis : Tensor 
#         Soit l'acquisition moyenne :math:`y` soit la covariance :math:`R_y`.
#     dom : Domain2D
#         Domaine :math:`\mathcal{X}` sur lequel va être intégré l'adjoint 
#         de :math:`m` dans :math:`\mathrm{L}^2(\mathcal{X})` si l'objectif est 
#         l'acquisition ou :math:`\mathrm{L}^2(\mathcal{X^2})` si l'objectif est 
#         la covariance.
#     regul : double
#         Paramètre de régularisation `\lambda`.
#     obj : str, optional
#         Soit 'covar' pour reconstruire sur la covariance soit 'acquis' pour
#         reconstruire sur la moyenne. The default is 'covar'.

#     Returns
#     -------
#     eta : Tensor
#         Fonction continue, élément de :math:`\mathscr{C}(\mathcal{X})`,
#         discrétisée. Certificat dual :math:`\eta` associé à la mesure.

#     Notes
#     -------
#     Si l'objectif est la covariance :

#     .. math:: \eta_\mathcal{Q} = \Lambda^\ast(\Lambda m - R_y)

#     Si l'objectif est la moyenne :

#     .. math:: \eta_\mathcal{P} = \Phi^\ast(\Phi m - \bar{y})

#     """
#     eta = 1/regul*phiAdjoint(acquis - phi(mesure, dom, obj), dom, obj)
#     return eta


def pile_aquisition(m, dom, bru, T_ech):
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
    acquis_temporelle = torch.zeros(T_ech, taille, taille)
    for t in range(T_ech):
        a_tmp = torch.rand(N_mol) * m.a
        m_tmp = Mesure2D(a_tmp, m.x)
        acquis_temporelle[t, :] = m_tmp.acquisition(dom, taille, bru)
    return acquis_temporelle


def covariance_pile(stack):
    """Calcule la covariance de y(x,t) à partir de la pile et de sa moyenne

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
    stack_re = stack.reshape(stack.shape[0],-1)
    stack_re -= stack_re.mean(0, keepdim=True)
    return 1/(stack_re.shape[0]-1) * stack_re.T @ stack_re








N_ECH = 64
X_GAUCHE = 0
X_DROIT = 1
FWMH = 2.2875 / N_ECH
SIGMA = FWMH / (2 * np.sqrt(2*np.log(2)))
domain = Domain2D(X_GAUCHE, X_DROIT, N_ECH, SIGMA)

a = torch.Tensor([1,2])
x = torch.Tensor([[0.1, 0.2], [0.8, 0.7]])
x2 = torch.Tensor([[0.3, 0.4], [0.5, 0.5]])

m = Mesure2D(a,x)
m2 = Mesure2D(a,x2)

plt.imshow(m.kernel(domain))
plt.figure()
plt.imshow(m.cov_kernel(domain))

FOND = 2.0
SIGMA_BRUITS = 3e-1
TYPE_BRUITS = 'gauss'


bruits_t = Bruits(FOND, SIGMA_BRUITS, TYPE_BRUITS)

T_ECH = 50
pile = pile_aquisition(m, domain, bruits_t, T_ECH)
y_bar = pile.mean(0)
cov_pile = covariance_pile(pile)

plt.figure()
plt.imshow(cov_pile)

