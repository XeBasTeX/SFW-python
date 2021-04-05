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


# # GPU acceleration if needed
# # Currently ONLY on CPU, you might want to put cudavenant in GPU also
# dtype = torch.cuda.float if torch.cuda.is_available() else torch.float


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
                               2*self.N_ech - 1)
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
        assert(len(amplitude) == len(position))
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


def phiAdjoint(acquis, dom, obj='covar'):
    """
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
        # (Xt, Yt) = domain.get_domain()
        # h_vec = gaussienne_2D(Xt, Yt, SIGMA,
        #                       undivide=__normalis_PSF__).reshape(-1)
        # lambda_vec = torch.outer(h_vec, h_vec)
        # eta = torch.fft.ifft2(torch.fft.fft2(lambda_vec) @\
        #                       torch.fft.fft2(acquis))
        eta = acquis
        eta = torch.diag(torch.abs(eta)).reshape(N_ECH, N_ECH)
        return eta
    if obj == 'acquis':
        (X_big, Y_big) = domain.big()
        h_vec = gaussienne_2D(X_big, Y_big, sigma, undivide=__normalis_PSF__)
        h_ker = h_vec.reshape(1, 1, N_ech*2-1 , N_ech*2-1)
        y_arr = y_bar.reshape(1, 1, N_ech , N_ech)
        eta = torch.nn.functional.conv2d(h_ker, y_arr, stride=1)
        eta = torch.squeeze(eta)
        return eta
    raise TypeError


def etak(mesure, acquis, dom, regul, obj='covar'):
    r"""Certificat dual :math:`\eta_\lambda` associé à la mesure 
    :math:`m_\lambda`. Ce certificat permet de donner une approximation 
    (sur la grille) de la  position du Dirac de plus forte intensité du 
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








N_ECH = 32
X_GAUCHE = 0
X_DROIT = 1
FWMH = 2.2875 / N_ECH
SIGMA = 0.15
domain = Domain2D(X_GAUCHE, X_DROIT, N_ECH, SIGMA)

a = torch.Tensor([1,2])
x = torch.Tensor([[0.1, 0.5], [0.7, 0.2]])
x2 = torch.Tensor([[0.3, 0.4], [0.5, 0.5]])

m = Mesure2D(a,x)
m2 = Mesure2D(a,x2)

plt.imshow(m.kernel(domain), extent=[0,1,1,0])
plt.title('$\Phi(m)$', fontsize=28)
plt.colorbar()

plt.figure()
plt.imshow(m.cov_kernel(domain), extent=[0,1,1,0])
plt.colorbar()
plt.title('$\Lambda(m)$', fontsize=28)


FOND = 0.0
SIGMA_BRUITS = 0.000001e-1
TYPE_BRUITS = 'gauss'


bruits_t = Bruits(FOND, SIGMA_BRUITS, TYPE_BRUITS)

T_ECH = 50
pile = pile_aquisition(m, domain, bruits_t, T_ECH)
y_bar = pile.mean(0)
cov_pile = covariance_pile(pile)

plt.figure()
plt.scatter(x[:,0], x[:,1])
plt.imshow(cov_pile)
plt.colorbar()
plt.title('$R_y$', fontsize=28)








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

    return coord.flip(-1)


# Le fameux algo de Sliding Frank Wolfe
def SFW(acquis, dom, regul=1e-5, nIter=5, mes_init=None, mesParIter=False,
        obj='covar', printInline=True):
    """Algorithme de Sliding Frank-Wolfe pour la reconstruction de mesures
    solution du BLASSO [1].

    Si l'objectif est la covariance :

    .. math:: \mathrm{argmin}_{m \in \mathcal{M(X)}} {T}_\lambda(m) = \\
        \lambda |m|(\mathcal{X}) + \dfrac{1}{2} ||R_y - \Lambda (m)||^2. \\
            \quad (\mathcal{Q}_\lambda (y))

    Si l'objectif est la moyenne :

    .. math:: \mathrm{argmin}_{m \in \mathcal{M(X)}} {S}_\lambda(m) = \\
        \lambda |m|(\mathcal{X}) + \\
            \dfrac{1}{2} ||\overline{y} - \Phi (m)||^2.\\
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
    if obj == 'covar':
        N_grille = N_grille**2
    if mes_init == None:
        mesure_k = Mesure2D()
        a_k = torch.Tensor()
        x_k = torch.Tensor()
        x_k_demi = torch.Tensor()
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
        x_star = x_star_index / N_ech_y # passer de l'idx à xstar
        if printInline:
            print(fr'* x^* index {x_star} max ' +
                  fr'à {certif_abs[x_star_index]:.2f}')

        # Condition d'arrêt (étape 4)
        if torch.abs(eta_V_k[x_star_index]) < 1:
            nrj_vecteur[k] = mesure_k.energie(dom, acquis,
                                              regul, obj=obj)
            if printInline:
                print("\n\n---- Condition d'arrêt ----")
            if mesParIter:
                return(mesure_k, nrj_vecteur[:k], mes_vecteur)
            return(mesure_k, nrj_vecteur[:k])

        # Création du x positions estimées
        mesure_k_demi = Mesure2D()
        if x_k.size == 0:
            x_k_demi = torch.vstack(x_star)
            lasso_guess = torch.ones(Nk+1)
        else:
            x_k_demi = torch.vstack(x_k, x_star)
            lasso_guess = torch.cat((a_k, [1.0]))

        # On résout LASSO (étape 7)
        def lasso(aa):
            difference = acquis - phi_vecteur(aa, x_k_demi, dom, obj)
            attache = 0.5*np.linalg.norm(difference)**2/N_grille
            parcimonie = regul*np.linalg.norm(aa, 1)
            return attache + parcimonie

        def grad_lasso(params):
            aa = params
            xx = x_k_demi
            N = len(aa)
            partial_a = N*[0]
            residual = acquis - phi_vecteur(aa, xx, dom, obj)
            if obj == 'covar':
                for i in range(N):
                    gauss = gaussienne_2D(dom.X - xx[i, 0], dom.Y - xx[i, 1],
                                          dom.sigma)
                    cov_gauss = np.outer(gauss, gauss)
                    normalis = dom.N_ech**4
                    partial_a[i] = regul - np.sum(cov_gauss*residual)/normalis
                return partial_a
            elif obj == 'acquis':
                for i in range(N):
                    gauss = gaussienne_2D(dom.X - xx[i, 0], dom.Y - xx[i, 1],
                                          dom.sigma)
                    normalis = dom.N_ech**2
                    partial_a[i] = regul - np.sum(gauss*residual)/normalis
                return partial_a
            else:
                raise TypeError('Unknown BLASSO target.')

        res = scipy.optimize.minimize(lasso, lasso_guess,
                                      jac=grad_lasso,
                                      options={'disp': __deboggage__})
        a_k_demi = res.x

        # print('* x_k_demi : ' + str(np.round(x_k_demi, 2)))
        # print('* a_k_demi : ' + str(np.round(a_k_demi, 2)))
        if printInline:
            print('* Optim convexe')
        mesure_k_demi += Mesure2D(a_k_demi, x_k_demi)

        # On résout double LASSO non-convexe (étape 8)
        def lasso_double(params):
            a_p = params[:int(len(params)/3)]
            x_p = params[int(len(params)/3):]
            x_p = x_p.reshape((len(a_p), 2))
            attache = 0.5*np.linalg.norm(acquis - phi_vecteur(a_p, x_p, dom,
                                                              obj))**2/N_grille
            parcimonie = regul*np.linalg.norm(a_p, 1)
            return attache + parcimonie

        def grad_lasso_double(params):
            a_p = params[:int(len(params)/3)]
            x_p = params[int(len(params)/3):]
            x_p = x_p.reshape((len(a_p), 2))
            N = len(a_p)
            partial_a = N*[0]
            partial_x = 2*N*[0]
            residual = acquis - phi_vecteur(a_p, x_p, dom, obj)
            if obj == 'covar':
                for i in range(N):
                    gauss = gaussienne_2D(dom.X - x_p[i, 0], dom.Y - x_p[i, 1],
                                          dom.sigma)
                    cov_gauss = np.outer(gauss, gauss)
                    partial_a[i] = regul - np.sum(cov_gauss*residual)/N_grille

                    gauss_der_x = grad_x_gaussienne_2D(dom.X - x_p[i, 0],
                                                       dom.Y - x_p[i, 1],
                                                       dom.X - x_p[i, 0],
                                                       dom.sigma)
                    cov_der_x = np.outer(gauss_der_x, gauss)
                    partial_x[2*i] = 2*a_p[i] * \
                        np.sum(cov_der_x*residual) / (N_grille)
                    gauss_der_y = grad_y_gaussienne_2D(dom.X - x_p[i, 0],
                                                       dom.Y - x_p[i, 1],
                                                       dom.Y - x_p[i, 1],
                                                       dom.sigma)
                    cov_der_y = np.outer(gauss_der_y, gauss)
                    partial_x[2*i+1] = 2*a_p[i] * \
                        np.sum(cov_der_y*residual) / (N_grille)

                return(partial_a + partial_x)
            elif obj == 'acquis':
                for i in range(N):
                    integ = np.sum(residual*gaussienne_2D(dom.X - x_p[i, 0],
                                                          dom.Y - x_p[i, 1],
                                                          dom.sigma))
                    partial_a[i] = regul - integ/N_grille

                    grad_gauss_x = grad_x_gaussienne_2D(dom.X - x_p[i, 0],
                                                        dom.Y - x_p[i, 1],
                                                        dom.X - x_p[i, 0],
                                                        dom.sigma)
                    integ_x = np.sum(residual*grad_gauss_x) / (N_grille)
                    partial_x[2*i] = a_p[i] * integ_x
                    grad_gauss_y = grad_y_gaussienne_2D(dom.X - x_p[i, 0],
                                                        dom.Y - x_p[i, 1],
                                                        dom.Y - x_p[i, 1],
                                                        dom.sigma)
                    integ_y = np.sum(residual*grad_gauss_y) / (N_grille)
                    partial_x[2*i+1] = a_p[i] * integ_y

                return(partial_a + partial_x)
            else:
                raise TypeError('Unknown BLASSO target.')

        # On met la graine au format array pour scipy...minimize
        # il faut en effet que ça soit un vecteur
        initial_guess = np.append(a_k_demi, np.reshape(x_k_demi, -1))
        a_part = list(zip([0]*(Nk+1), [10000]*(Nk+1)))
        x_part = list(zip([0]*2*(Nk+1), [1.001]*2*(Nk+1)))
        bounds_bfgs = a_part + x_part
        tes = scipy.optimize.minimize(lasso_double, initial_guess,
                                       bounds=bounds_bfgs,
                                      options={'disp': True})
        a_k_plus = (tes.x[:int(len(tes.x)/3)])
        x_k_plus = (tes.x[int(len(tes.x)/3):]).reshape((len(a_k_plus), 2))
        # print('* a_k_plus : ' + str(np.round(a_k_plus, 2)))
        # print('* x_k_plus : ' +  str(np.round(x_k_plus, 2)))

        # Mise à jour des paramètres avec retrait des Dirac nuls
        mesure_k = Mesure2D(a_k_plus, x_k_plus)
        mesure_k = mesure_k.prune()
        # mesure_k = merge_spikes(mesure_k)
        a_k = mesure_k.a
        x_k = mesure_k.x
        Nk = mesure_k.N
        if printInline:
            print(f'* Optim non-convexe : {Nk} Diracs')

        # Graphe et énergie
        nrj_vecteur[k] = mesure_k.energie(dom, acquis, regul, obj)
        if printInline:
            print(f'* Énergie : {nrj_vecteur[k]:.3e}')
        if mesParIter == True:
            mes_vecteur = np.append(mes_vecteur, [mesure_k])
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

    # Fin des itérations
    if printInline:
        print("\n\n---- Fin de la boucle ----")
    if mesParIter:
        return(mesure_k, nrj_vecteur, mes_vecteur)
    return(mesure_k, nrj_vecteur)




# https://stackoverflow.com/questions/50621786/lbfgs-never-converges-in-large-dimensions-in-pytorch
# Optim : https://discuss.pytorch.org/t/use-pytorch-optimizer-to-minimize-a-user-function/66712/8

































def wasserstein_metric(mes, m_zer, p_wasser=1):
    '''Retourne la p--distance de Wasserstein partiel (Partial Gromov-
    Wasserstein) pour des poids égaux (pas de prise en compte de la 
    luminosité)'''
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
    if p_wasser == 1:
        M = ot.dist(mes.x, m_zer.x, metric='euclidean')
        return M
    elif p_wasser == 2:
        M = ot.dist(mes.x, m_zer.x)
        return M
    raise ValueError('Unknown p for W_p computation')

