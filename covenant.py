#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Provides the essential functions for simulating 2D discrete Radon measures
and reconstructing those measures w.r.t to a provided acquistion

@author: Bastien Laville (https://github.com/XeBasTeX)
"""

__author__ = 'Bastien'
__team__ = 'Morpheme'
__deboggage__ = False


import numpy as np
from scipy import integrate
import scipy
import scipy.signal
import ot

from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt


def gaussienne_2D(X_domain, Y_domain, sigma_g):
    """
    Gaussienne en 2D centrée en 0 normalisée.

    Parameters
    ----------
    X_domain : ndarray
        Grille des coordonnées X (issue de meshgrid).
    Y_domain : ndarray
        Grille des coordonnées Y (issue de meshgrid).
    sigma_g : double
        :math:`\sigma` paramètre de la gaussienne.

    Returns
    -------
    ndarray
        Vecteur discrétisant la gaussienne :math:`h` sur :math:`\mathcal{X}`.

    """
    expo = np.exp(-(np.power(X_domain, 2) +
                    np.power(Y_domain, 2))/(2*sigma_g**2))
    normalis = sigma_g * (2*np.pi)
    # normalis = 1
    return expo/normalis


def grad_x_gaussienne_2D(X_domain, Y_domain, X_deriv, sigma_g):
    """
    Gaussienne centrée en 0  normalisée. Attention, ne prend pas en compte la 
    chain rule derivation.

    Parameters
    ----------
    X_domain : ndarray
        Grille des coordonnées X (issue de meshgrid).
    Y_domain : ndarray
        Grille des coordonnées Y (issue de meshgrid).
    X_deriv : ndarray
        Grille des coordonnées X pour calculer la partie en :math:`x` de la
        dérivée partielle
    sigma_g : double
        :math:`\sigma` paramètre de la gaussienne.

    Returns
    -------
    ndarray
        Vecteur discrétisant la première dérivée partielle de la gaussienne 
        :math:`\partial_1 h` sur :math:`\mathcal{X}`.

    """
    expo = np.exp(-(np.power(X_domain, 2) +
                    np.power(Y_domain, 2))/(2*sigma_g**2))
    normalis = sigma_g**3 * (2*np.pi)
    carre = - X_deriv
    return carre * expo / normalis


def grad_y_gaussienne_2D(X_domain, Y_domain, Y_deriv, sigma_g):
    """
    Gaussienne centrée en 0  normalisée. Attention, ne prend pas en compte la 
    chain rule derivation.

    Parameters
    ----------
    X_domain : ndarray
        Grille des coordonnées X (issue de meshgrid).
    Y_domain : ndarray
        Grille des coordonnées Y (issue de meshgrid).
    Y_deriv : ndarray
        Grille des coordonnées Y pour calculer la partie en :math:`y` de la
        dérivée partielle
    sigma_g : double
        :math:`\sigma` paramètre de la gaussienne.

    Returns
    -------
    ndarray
        Vecteur discrétisant la première dérivée partielle de la gaussienne 
        :math:`\partial_2 h` sur :math:`\mathcal{X}`.

    """
    expo = np.exp(-(np.power(X_domain, 2) +
                    np.power(Y_domain, 2))/(2*sigma_g**2))
    normalis = sigma_g**3 * (2*np.pi)
    carre = - Y_deriv
    return carre * expo / normalis


def ideal_lowpass(carre, f_c):
    '''Passe-bas idéal de fréquence de coupure f_c'''
    return np.sin((2*f_c + 1)*np.pi*carre)/np.sin(np.pi*carre)


class Domain2D:
    def __init__(self, gauche, droit, ech, grille, X_domain, Y_domain,
                 sigma_psf):
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
        ndarray
            Les grilles coordonnées de x et des y.

        """
        return np.meshrgrid(self.X_grid)


    def big(self):
        """
        Renvoie les grilles adaptées pour le calcul de la convolution discrètes 
        (par exemple dans phiAdjoint) entre deux matrices

        Returns
        -------
        X_big : ndarray
                Matrice coordonnées des x
        Y_big : ndarray
                Matrice coordonnées des y

        """
        grid_big = np.linspace(self.x_gauche-self.x_droit, self.x_droit,
                               2*self.N_ech - 1)
        X_big, Y_big = np.meshgrid(grid_big, grid_big)
        return(X_big, Y_big)


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
            amplitude = []
            position = []
        if len(amplitude) != len(position):
            raise ValueError('Pas le même nombre')
        if isinstance(amplitude, np.ndarray) and isinstance(position,
                                                            np.ndarray):
            self.a = amplitude
            self.x = position
        else:
            self.x = np.array(position)
            self.a = np.array(amplitude)
        self.N = len(amplitude)


    def __add__(self, m):
        '''Hieher : il faut encore régler l'addition pour les mesures au même
        position, ça vaudrait le coup de virer les duplicats'''
        a_new = np.append(self.a, m.a)
        x_new = np.array(list(self.x) + list(m.x))
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
        amplitudes = np.round(self.a, 3)
        positions = np.round(self.x, 3)
        return(f"{self.N} Diracs \nAmplitudes : {amplitudes}" +
               f"\nPositions : {positions}")


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
        acquis : ndarray
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
        raise NameError("Unknown kernel.")

    def covariance_kernel(self, dom):
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
        acquis : ndarray
            Matrice discrétisant :math:`\Lambda(m)` .

        """
        X_domain = dom.X
        Y_domain = dom.Y
        sigma = dom.sigma
        N = self.N
        x = self.x
        a = self.a
        taille_x = np.size(X_domain)
        taille_y = np.size(Y_domain)
        acquis = np.zeros((taille_x, taille_y))
        for i in range(0, N):
            noyau_u = gaussienne_2D(X_domain - x[i, 0], Y_domain - x[i, 1],
                                    sigma)
            noyau_v = gaussienne_2D(X_domain - x[i, 0], Y_domain - x[i, 1],
                                    sigma)
            acquis += a[i]*np.outer(noyau_u, noyau_v)
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
        acquis : ndarray
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
            w = np.random.normal(0, nv, size=(echantillo, echantillo))
            simul = self.kernel(dom, noyau='gaussienne')
            acquis = simul + w + fond
            return acquis
        if type_de_bruits == 'poisson':
            w = np.random.poisson(nv, size=(echantillo, echantillo))
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
            return np.linalg.norm(self.a, 1)
        except ValueError:
            return 0

    def energie(self, dom, acquis, regul, obj='covar',
                bruits='gauss'):
        """
        Énergie de la mesure pour le problème Covenant 
        :math:`(\mathcal{Q}_\lambda (y))` ou BLASSO sur
        l'acquisition moyenne :math:`(\mathcal{P}_\lambda (\overline{y}))`.

        Parameters
        ----------
        dom : Domain2D
            Domaine sur lequel va être calculée l'acquisition de :math:`m` dans
            :math:`\mathrm{L}^2(\mathcal{X})`.
        acquis : ndarray
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
            normalis = acquis.size
            if obj == 'covar':
                R_nrj = self.covariance_kernel(dom)
                attache = 0.5*np.linalg.norm(acquis - R_nrj)**2/normalis
                parcimonie = regul*self.tv()
                return attache + parcimonie
            if obj == 'acquis':
                simul = self.kernel(dom)
                attache = 0.5*np.linalg.norm(acquis - simul)**2/normalis
                parcimonie = regul*self.tv()
                return attache + parcimonie
        elif bruits in ('gauss', 'unif'):
            normalis = acquis.size
            if obj == 'covar':
                R_nrj = self.covariance_kernel(dom)
                attache = 0.5*np.linalg.norm(acquis - R_nrj)**2/normalis
                parcimonie = regul*self.tv()
                return attache + parcimonie
            if obj == 'acquis':
                simul = self.kernel(dom)
                attache = 0.5*np.linalg.norm(acquis - simul)**2/normalis
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
        nnz_a = np.array(self.a)
        nnz = nnz_a > tol
        nnz_a = nnz_a[nnz]
        nnz_x = np.array(self.x)
        nnz_x = nnz_x[nnz]
        m = Mesure2D(nnz_a, nnz_x)
        return m


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
    x = np.round(dom.x_gauche + np.random.rand(N, 2)*
                 (dom.x_droit - dom.x_gauche), 2)
    a = np.round(0.5 + np.random.rand(1, N), 2)[0]
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
    ndarray
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


def phiAdjointSimps(acquis, dom, obj='covar'):
    '''Hierher débugger ; taille_x et taille_y pas implémenté'''
    X_domain = dom.X
    Y_domain = dom.Y
    taille_y = len(X_domain)
    taille_x = len(X_domain[0])
    ech_x_square = X_domain.size
    ech_y_square = Y_domain.size
    eta = np.zeros(np.shape(X_domain))
    if obj == 'covar':
        for i in range(taille_x):
            for j in range(taille_y):
                x_decal = X_domain[i, j]
                y_decal = Y_domain[i, j]
                convol = gaussienne_2D(x_decal - X_domain, y_decal - Y_domain,
                                       dom.sigma)
                noyau = np.outer(convol, convol)
                X_range_integ = np.linspace(dom.x_gauche, dom.x_droit,
                                            ech_x_square)
                Y_range_integ = np.linspace(dom.x_gauche, dom.x_droit,
                                            ech_y_square)
                integ_x = integrate.trapz(acquis * noyau, x=X_range_integ)
                eta[i, j] = integrate.trapz(integ_x, x=Y_range_integ)
        return eta
    if obj == 'acquis':
        for i in range(taille_x):
            for j in range(taille_y):
                x_decal = X_domain[i, j]
                y_decal = Y_domain[i, j]
                gauss = gaussienne_2D(x_decal - X_domain, y_decal - Y_domain,
                                      dom.sigma)
                integ_x = integrate.simps(acquis * gauss, x=dom.X_grid)
                eta[i, j] = integrate.simps(integ_x, x=dom.X_grid)
        return eta
    raise TypeError


def phiAdjoint(acquis, dom, obj='covar'):
    """
    Hierher débugger ; taille_x et taille_y pas implémenté

    Parameters
    ----------
    acquis : ndarray 
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
    eta : ndarray
        Fonction continue, élément de :math:`\mathscr{C}(\mathcal{X})`,
        discrétisée. Utile pour calculer la discrétisation du certificat 
        :math:`\eta` associé à une mesure.

    """
    N_ech = dom.N_ech
    eta = np.zeros(np.shape(dom.X))
    if obj == 'covar':
        (X_big, Y_big) = dom.big()
        sigma = dom.sigma
        h_vec = gaussienne_2D(X_big, Y_big, sigma)
        convol_row = scipy.signal.convolve2d(acquis, h_vec, 'same').T/N_ech**2
        adj = np.diag(scipy.signal.convolve2d(convol_row, h_vec, 'same'))
        eta = adj.reshape(N_ech, N_ech)/N_ech**2
        return eta
    if obj == 'acquis':
        (X_big, Y_big) = dom.big()
        sigma = dom.sigma
        out = gaussienne_2D(X_big, Y_big, sigma)
        eta = scipy.signal.convolve2d(out, acquis, mode='valid')/N_ech**2
        return eta
    raise TypeError


def etak(mesure, acquis, dom, regul, obj='covar'):
    r"""Certificat dual :math:`\eta` associé à la mesure :math:`m`. Ce 
    certificat permet de donner une approximation (sur la grille) de la 
    position du Dirac de plus forte intensité qui engendrerait acquis par
    le noyau `obj`.

    Parameters
    ----------
    mesure : Mesure2D
        Mesure discrète :math:`m` dont on veut obtenir le certificat duaL.
    acquis : ndarray 
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
    eta : ndarray
        Fonction continue, élément de :math:`\mathscr{C}(\mathcal{X})`,
        discrétisée. Certificat dual :math:`\eta` associé à la mesure.

    Notes
    -------
    Si l'objectif est la covariance :

    .. math:: \eta_\mathcal{Q} = \Lambda^\ast(\Lambda m - R_y)

    Si l'objectif est la moyenne :

    .. math:: \eta_\mathcal{P} = \Phi^\ast(\Phi m - \bar{y})

    """
    eta = 1/regul*phiAdjointSimps(acquis - phi(mesure, dom, obj), dom, obj)
    return eta


def pile_aquisition(m, dom, bru, T_ech):
    r'''Construit une pile d'acquisition à partir d'une mesure.
    Correspond à l'opérateur $\vartheta(\mu)$ '''
    N_mol = len(m.a)
    taille = dom.N_ech
    acquis_temporelle = np.zeros((T_ech, taille, taille))
    for t in range(T_ech):
        a_tmp = (np.random.rand(N_mol))*m.a
        m_tmp = Mesure2D(a_tmp, m.x)
        acquis_temporelle[t, :] = m_tmp.acquisition(dom, taille, bru)
    return acquis_temporelle


def covariance_pile(stack, stack_mean):
    '''Calcule la covariance de y(x,t) à partir de la pile et de sa moyenne'''
    covar = np.zeros((len(stack[0])**2, len(stack[0])**2))
    for i in range(len(stack)):
        covar += np.outer(stack[i, :] - stack_mean, stack[i, :] - stack_mean)
    return covar/len(stack-1)   # T-1 ou T hierher ?


# Le fameux algo de Sliding Frank Wolfe
def SFW(acquis, dom, regul=1e-5, nIter=5, mesParIter=False,
        obj='covar', printInline=True):
    """
    Algorithme de Sliding Frank-Wolfe pour la reconstruction de mesures
    solution du BLASSO [1,2].

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
    [1] Denoyelle
    [2] De Castro
    """
    N_ech_y = dom.N_ech  # hierher à adapter
    N_grille = dom.N_ech**2
    if obj == 'covar':
        N_grille = N_grille**2
    # X_domain = dom.X
    # Y_domain = dom.Y
    a_k = np.empty((0, 0))
    x_k = np.empty((0, 0))
    mesure_k = Mesure2D()
    x_k_demi = np.empty((0, 0))
    Nk = 0                      # Taille de la mesure discrète
    if mesParIter:
        mes_vecteur = np.array([])
    nrj_vecteur = np.zeros(nIter)
    for k in range(nIter):
        if printInline: print('\n' + 'Étape numéro ' + str(k))
        eta_V_k = etak(mesure_k, acquis, dom, regul, obj)
        certif_abs = np.abs(eta_V_k)
        x_star_index = np.unravel_index(np.argmax(certif_abs, axis=None),
                                        eta_V_k.shape)
        # passer de l'idx à xstar
        x_star = np.array(x_star_index)[::-1]/N_ech_y
        if printInline: print(fr'* x^* index {x_star} max ' +
                              fr'à {np.round(certif_abs[x_star_index], 2)}')

        # Condition d'arrêt (étape 4)
        if np.abs(eta_V_k[x_star_index]) < 1:
            nrj_vecteur[k] = mesure_k.energie(dom, acquis,
                                              regul, obj=obj)
            if printInline: print("\n\n---- Condition d'arrêt ----")
            if mesParIter:
                return(mesure_k, nrj_vecteur[:k], mes_vecteur)
            return(mesure_k, nrj_vecteur[:k])

        # Création du x positions estimées
        mesure_k_demi = Mesure2D()
        if x_k.size == 0:
            x_k_demi = np.vstack([x_star])
            lasso_guess = np.ones(Nk+1)
        else:
            x_k_demi = np.vstack([x_k, x_star])
            lasso_guess = np.concatenate((a_k, [1.0]))

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
        if printInline: print('* Optim convexe')
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
        a_part = list(zip([0]*(Nk+1), [30]*(Nk+1)))
        x_part = list(zip([0]*2*(Nk+1), [1.001]*2*(Nk+1)))
        bounds_bfgs = a_part + x_part
        tes = scipy.optimize.minimize(lasso_double, initial_guess,
                                      method='L-BFGS-B',
                                      jac=grad_lasso_double,
                                      bounds=bounds_bfgs,
                                      options={'disp': __deboggage__})
        a_k_plus = (tes.x[:int(len(tes.x)/3)])
        x_k_plus = (tes.x[int(len(tes.x)/3):]).reshape((len(a_k_plus), 2))
        # print('* a_k_plus : ' + str(np.round(a_k_plus, 2)))
        # print('* x_k_plus : ' +  str(np.round(x_k_plus, 2)))

        # Mise à jour des paramètres avec retrait des Dirac nuls
        mesure_k = Mesure2D(a_k_plus, x_k_plus)
        mesure_k = mesure_k.prune()
        a_k = mesure_k.a
        x_k = mesure_k.x
        Nk = mesure_k.N
        if printInline: print(f'* Optim non-convexe : {Nk} Diracs')

        # Graphe et énergie
        nrj_vecteur[k] = mesure_k.energie(dom, acquis, regul, obj)
        if printInline: print(f'* Énergie : {nrj_vecteur[k]:.3f}')
        if mesParIter == True:
            mes_vecteur = np.append(mes_vecteur, [mesure_k])

    # Fin des itérations
    if printInline: print("\n\n---- Fin de la boucle ----")
    if mesParIter:
        return(mesure_k, nrj_vecteur, mes_vecteur)
    return(mesure_k, nrj_vecteur)


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


def partial_wasserstein_metric(mes, m_zer):
    '''Retourne la 2--distance de Wasserstein partiel (Partial Gromov-
    Wasserstein) pour des poids égaux (pas de prise en compte de la 
    luminosité)'''
    M = scipy.spatial.distance.cdist(mes.x, m_zer.x)
    p = ot.unif(mes.N)
    q = ot.unif(m_zer.N)
    # masse = min(np.linalg.norm(p, 1),np.linalg.norm(q, 1))
    # en fait la masse pour les deux = 1
    masse = 1
    w, log = ot.partial.partial_wasserstein(p, q, M, m=masse, log=True)
    return log['partial_w_dist']


def plot_results(m, m_zer, dom, bruits, y, nrj, certif, title=None,
                 saveFig=False, obj='covar'):
    '''Affiche tous les graphes importants pour la mesure m'''
    if m.a.size > 0:
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
                     f' = {diss:.3f}', fontsize=22)

        plt.subplot(221)
        cont1 = plt.contourf(dom.X, dom.Y, y, 100, cmap='gray')
        for c in cont1.collections:
            c.set_edgecolor("face")
        plt.colorbar()
        plt.scatter(m_zer.x[:, 0], m_zer.x[:, 1], marker='x',
                    label='GT spikes')
        plt.scatter(m.x[:, 0], m.x[:, 1], marker='+',
                    label='Recovered spikes')
        plt.legend(loc=2)

        plt.xlabel('X', fontsize=18)
        plt.ylabel('Y', fontsize=18)
        plt.title(r'Temporal mean $\overline{y}$', fontsize=20)

        plt.subplot(222)
        cont2 = plt.contourf(dom.X, dom.Y, m.kernel(dom), 100, cmap='gray')
        for c in cont2.collections:
            c.set_edgecolor("face")
        plt.xlabel('X', fontsize=18)
        plt.ylabel('Y', fontsize=18)
        if obj == 'covar':
            plt.title(r'Reconstruction $m_{M,x}$ ' +
                      f'à N = {m.N}', fontsize=20)
        elif obj == 'acquis':
            plt.title(r'Reconstruction $m_{a,x}$ ' +
                      f'à N = {m.N}', fontsize=20)
        plt.colorbar()

        plt.subplot(223)
        cont3 = plt.contourf(dom.X, dom.Y, certif, 100, cmap='gray')
        for c in cont3.collections:
            c.set_edgecolor("face")
        plt.xlabel('X', fontsize=18)
        plt.ylabel('Y', fontsize=18)
        plt.title(r'Certificate $\eta_V$', fontsize=20)
        plt.colorbar()

        plt.subplot(224)
        plt.plot(nrj, 'o--', color='black', linewidth=2.5)
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

    anim = FuncAnimation(fig, animate, interval=400, frames=len(pile_acquis)+3,
                         blit=False)

    plt.draw()

    if title is None:
        title = 'fig/anim-pile-2d'
    elif isinstance(title, str):
        title = 'fig/' + title
    else:
        raise TypeError("You ought to give a str type name for the video file")

    if video == "mp4":
        anim.save(title + '.mp4')
    elif video == "gif":
        anim.save(title + '.gif')
    else:
        raise ValueError('Unknown video format')
    return fig


def gif_results(acquis, m_zer, m_list, dom, video='gif', title=None):
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

    anim = FuncAnimation(fig, animate, interval=1000, frames=len(m_list)+3,
                         blit=False)

    plt.draw()

    if title is None:
        title = 'fig/anim-sfw-covar-2d'
    elif isinstance(title, str):
        title = 'fig/' + title
    else:
        raise TypeError("You ought to give a str type name for the video file")

    if video == "mp4":
        anim.save(title + '.mp4')
    elif video == "gif":
        anim.save(title + '.gif')
    else:
        raise ValueError('Unknown video format')
    return fig

