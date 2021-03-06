#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 17:01:05 2021

@author: Bastien (https://github.com/XeBasTeX)
"""

__author__ = 'Bastien'
__team__ = 'Morpheme'
__saveFig__ = False
__saveVid__ = False
__deboggage__ = False


import numpy as np
from scipy import integrate
import scipy.signal
import scipy
import ot

from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt

import os
from skimage import io


os.chdir(os.path.dirname(os.path.abspath(__file__)))

sigma = 5e-2


def gaussienne(carre, sigma_g=sigma):
    '''Gaussienne centrée en 0'''
    expo = np.exp(-np.power(carre,2)/(2*sigma_g**2))
    return np.sqrt(2*np.pi*sigma_g**2) * expo


def gaussienne_2D(X_domain, Y_domain, sigma_g=sigma):
    '''Gaussienne centrée en 0'''
    expo = np.exp(-(np.power(X_domain,2) +
                    np.power(Y_domain,2))/(2*sigma_g**2))
    normalis = np.sqrt(2*np.pi*sigma_g**2)
    return normalis*expo


# def gaussienne(carre, sigma_g=sigma):
#     '''Gaussienne centrée en 0'''
#     expo = np.exp(-np.power(carre,2)/(2*sigma_g**2))
#     normalis = sigma_g * (2*np.pi)
#     return expo * normalis


# def gaussienne_2D(X_domain, Y_domain, sigma_g=sigma):
#     '''Gaussienne centrée en 0'''
#     expo = np.exp(-(np.power(X_domain,2) +
#                     np.power(Y_domain,2))/(2*sigma_g**2))
#     normalis = sigma_g * (2*np.pi)
#     return expo * normalis


def grad_x_gaussienne_2D(X_domain, Y_domain, X_deriv, sigma_g=sigma):
    '''Gaussienne centrée en 0. Attention, ne prend pas en compte la chain 
    rule derivation'''
    expo = np.exp(-(np.power(X_domain, 2) +
                    np.power(Y_domain, 2))/(2*sigma_g**2))
    normalis = sigma_g**3 * (2*np.pi)
    carre = - X_deriv
    return carre * expo / normalis


def grad_y_gaussienne_2D(X_domain, Y_domain, Y_deriv, sigma_g=sigma):
    '''Gaussienne centrée en 0. Attention, ne prend pas en compte la chain 
    rule derivation'''
    expo = np.exp(-(np.power(X_domain, 2) +
                    np.power(Y_domain, 2))/(2*sigma_g**2))
    normalis = sigma_g**3 * (2*np.pi)
    carre = - Y_deriv
    return carre * expo / normalis


# Domaine
class Domain:
    def __init__(self, gauche, droit, ech, grille, X_domain, Y_domain):
        self.x_gauche = gauche
        self.x_droit = droit
        self.N_ech = ech
        self.X_grid = grille
        self.X = X_domain
        self.Y = Y_domain


    def get_domain(self):
        return(self.X, self.Y)


    def compute_square_mesh(self):
        return np.meshrgrid(self.X_grid)


    def big(self):
        grid_big = np.linspace(self.x_gauche-self.x_droit, self.x_droit,
                                 2*self.N_ech)
        X_big, Y_big = np.meshgrid(grid_big, grid_big)
        return(X_big, Y_big)


    def certif(self):
        X_grid_certif = np.linspace(self.x_gauche, self.x_droit, self.N_ech+1)
        X_certif, Y_certif = np.meshgrid(X_grid_certif, X_grid_certif)
        return(X_certif, Y_certif)


# Bruits
class Bruits2D:
    def __init__(self, fond, niveau, type_de_bruits):
        self.fond = fond
        self.niveau = niveau
        self.type = type_de_bruits


    def get_fond(self):
        return self.fond


    def get_nv(self):
        return self.niveau


# Mesure 2D
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


    def kernel(self, X_domain, Y_domain, noyau='gaussienne'):
        '''Applique un noyau à la mesure discrète.
        Pris en charge : convolution  à noyau gaussien.'''
        N = self.N
        x = self.x
        a = self.a
        acquis = X_domain*0
        if noyau == 'gaussienne':
            for i in range(0,N):
                D = (np.sqrt(np.power(X_domain - x[i,0],2)
                             + np.power(Y_domain - x[i,1],2)))
                acquis += a[i]*gaussienne(D)
            return acquis
        if noyau == 'laplace':
            raise TypeError("Pas implémenté")
        raise NameError("Unknown kernel")


    def covariance_kernel(self, X_domain, Y_domain):
        '''Noyau de covariance associée à la mesure.
        Pris en charge : convolution  à noyau gaussien.'''
        N = self.N
        x = self.x
        a = self.a
        taille_x = np.size(X_domain)
        taille_y = np.size(Y_domain)
        acquis = np.zeros((taille_x, taille_y))
        for i in range(0, N):
            D = (np.sqrt(np.power(X_domain - x[i, 0], 2) +
                         np.power(Y_domain - x[i, 1], 2)))
            noyau_u = gaussienne(D)
            noyau_v = gaussienne(D)
            acquis += a[i]*np.outer(noyau_u, noyau_v)
        return acquis


    def acquisition(self, X_domain, Y_domain, echantillo, fond, nv,
                    bruits='unif'):
        '''Simule une acquisition pour la mesure avec un bruit de fond et
        un bruit poisson ou gaussien de niveau nv'''
        if bruits == 'unif':
            w = nv*np.random.random_sample((echantillo, echantillo))
            simul = self.kernel(X_domain, Y_domain, noyau='gaussienne')
            acquis = simul + w + fond
            return acquis
        if bruits == 'gauss':
            w = np.random.normal(0, nv, size=(echantillo, echantillo))
            simul = self.kernel(X_domain, Y_domain, noyau='gaussienne')
            acquis = simul + w + fond
            return acquis
        if bruits == 'poisson':
            w = np.random.poisson(nv, size=(echantillo,echantillo))
            simul = w*self.kernel(X_domain, Y_domain, noyau='gaussienne')
            acquis = simul + fond
        raise TypeError


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


    def torus(self, X_domain, current_fig=False, subplot=False):
        '''Mesure placée sur le tore si dim(cadre) = 1'''
        echantillo = len(X_domain)
        if subplot:
            ax = current_fig.add_subplot(224, projection='3d')
        else:
            current_fig = plt.figure()
            ax = current_fig.add_subplot(111, projection='3d')
        theta = np.linspace(0, 2*np.pi, echantillo)
        y_torus = np.sin(theta)
        x_torus = np.cos(theta)

        a_x_torus = np.sin(2*np.pi*np.array(self.x))
        a_y_torus = np.cos(2*np.pi*np.array(self.x))
        a_z_torus = self.a

        # ax.plot((0,0),(0,0), (0,1), '-k', label='z-axis')
        ax.plot(x_torus,y_torus, 0, '-k', label=r'$\mathbb{S}_1$')
        ax.plot(a_x_torus,a_y_torus, a_z_torus,
                'o', label=r'$m_{a,x}$', color='orange')

        for i in range(self.N):
            ax.plot((a_x_torus[i],a_x_torus[i]), (a_y_torus[i],a_y_torus[i]),
                    (0,a_z_torus[i]), '--r', color='orange')

        ax.set_xlabel('$X$')
        ax.set_ylabel('$Y$')
        ax.set_zlabel('$Amplitude$')
        ax.set_title(r'Mesure sur $\mathbb{S}_1$', fontsize=20)
        ax.legend()


    def tv(self):
        '''Renvoie 0 si la mesure est vide : hierher à vérifier'''
        try:
            return np.linalg.norm(self.a, 1)
        except ValueError:
            return 0


    def energie(self, X_domain, Y_domain, acquis, regul, obj='covar',
                bruits='gauss'):
        '''énergie de la mesure pour le problème Covenant ou BLASSO sur
        l'acquisition moyenne '''
        if bruits == 'poisson':
            normalis = acquis.size
            if obj == 'covar':
                R_nrj = self.covariance_kernel(X_domain, Y_domain)
                attache = 0.5*np.linalg.norm(acquis - R_nrj)**2/normalis
                parcimonie = regul*self.tv()
                return attache + parcimonie
            if obj == 'acquis':
                simul = self.kernel(X_domain, Y_domain)
                attache = 0.5*np.linalg.norm(acquis - simul)**2/normalis
                parcimonie = regul*self.tv()
                return attache + parcimonie
        elif bruits in ('gauss', 'unif'):
            normalis = 1
            if obj == 'covar':
                R_nrj = self.covariance_kernel(X_domain, Y_domain)
                attache = 0.5*np.linalg.norm(acquis - R_nrj)**2/normalis
                parcimonie = regul*self.tv()
                return attache + parcimonie
            if obj == 'acquis':
                simul = self.kernel(X_domain, Y_domain)
                attache = 0.5*np.linalg.norm(acquis - simul)**2/normalis
                parcimonie = regul*self.tv()
                return attache + parcimonie
            raise TypeError("Unknown kernel")
        raise TypeError("Unknown noise")


    def prune(self, tol=1e-3):
        '''Retire les Dirac avec zéro d'amplitude'''
        # nnz = np.count_nonzero(self.a)
        nnz_a = np.array(self.a)
        nnz = nnz_a > tol
        nnz_a = nnz_a[nnz]
        nnz_x = np.array(self.x)
        nnz_x = nnz_x[nnz]
        m = Mesure2D(nnz_a, nnz_x)
        return m


    def prune_spurious(self, tol=1e-2):
        a = np.array([])
        x = np.array([])
        for i in range(self.N):
          for j in range(self.N):
              if i != j and np.linalg.norm(x[i] - x[j]) < tol:
                  a = np.delete(self.a[i], 1, 0)
                  x = np.delete(self.x[i], 1, 0)
        self.a = a
        self.x = x
        self.N = len(a)


def mesure_aleatoire(N):
    '''Créé une mesure aléatoire de N pics d'amplitudes aléatoires comprises
    entre 0,5 et 1,5'''
    x = np.round(np.random.rand(N,2), 2)
    a = np.round(0.5 + np.random.rand(1,N), 2)[0]
    return Mesure2D(a, x)

# m = Mesure([1,1.1,0,1.5],[2,88,3,8])
# print(m.prune())

def phi(m, dom, obj='covar'):
    '''Créé le résultat d'un opérateur d'acquisition à partir de la mesure m'''
    X_domain = dom.X
    Y_domain = dom.Y
    if obj == 'covar':
        return m.covariance_kernel(X_domain, Y_domain)
    if obj == 'acquis':
        return m.kernel(X_domain, Y_domain)
    raise TypeError('Unknown BLASSO target')


def phi_vecteur(a, x, dom, obj='covar'):
    '''créé le résultat d'un opérateur d'acquisition à partir des vecteurs
    a et x qui forme une mesure m_tmp'''
    X_domain = dom.X
    Y_domain = dom.Y
    if obj == 'covar':
        m_tmp = Mesure2D(a, x)
        return m_tmp.covariance_kernel(X_domain, Y_domain)
    if obj == 'acquis':
        m_tmp = Mesure2D(a, x)
        return m_tmp.kernel(X_domain, Y_domain)
    raise TypeError('Unknown BLASSO target')
   

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
                x_decal = X_domain[i,j]
                y_decal = Y_domain[i,j]
                convol = gaussienne_2D(x_decal - X_domain, y_decal - Y_domain)
                noyau = np.outer(convol, convol)
                X_range_integ = np.linspace(dom.x_gauche, dom.x_droit,
                                            ech_x_square)
                Y_range_integ = np.linspace(dom.x_gauche, dom.x_droit,
                                            ech_y_square)
                integ_x = integrate.trapz(acquis * noyau, x=X_range_integ)
                eta[i,j] = integrate.trapz(integ_x, x=Y_range_integ)
        return eta
    if obj == 'acquis':
        for i in range(taille_x):
            for j in range(taille_y):
                x_decal = X_domain[i,j]
                y_decal = Y_domain[i,j]
                D_decal = (np.sqrt(np.power(x_decal-X_domain,2)
                                   + np.power(y_decal-Y_domain,2)))
                integ_x = integrate.simps(acquis * gaussienne(D_decal),
                                          x=dom.X_grid)
                eta[i,j] = integrate.simps(integ_x, x=dom.X_grid)
        return eta
    raise TypeError


def phiAdjoint(acquis, dom, obj='covar'):
    '''Hierher débugger ; taille_x et taille_y pas implémenté'''
    N_ech = dom.N_ech
    eta = np.zeros(np.shape(dom.X))
    if obj == 'covar':
        (X_big, Y_big) = dom.big()
        h_vec = gaussienne_2D(X_big, Y_big)
        convol_row = scipy.signal.convolve(acquis, h_vec, 'same').T
        adj = np.diag(scipy.signal.convolve(convol_row, h_vec, 'same'))
        eta = adj.reshape(N_ech, N_ech)/N_ech**4
        return eta
    if obj == 'acquis':
        (X_big, Y_big) = dom.big()
        out = gaussienne_2D(X_big, Y_big)
        eta = scipy.signal.convolve2d(out, acquis, mode='valid')/(N_ech**2)
        return eta
    raise TypeError


def etak(mesure, acquis, dom, regul, obj='covar'):
    r'''Certificat $\eta$ assicé à la mesure'''
    # eta = 1/regul*phiAdjointSimps(acquis - phi(mesure, dom, obj),
    #                          dom, obj)
    eta = 1/regul*phiAdjointSimps(acquis - phi(mesure, dom, obj), dom, obj)
    return eta


def pile_aquisition(m, dom, fond, nv, bruits='gauss'):
    r'''Construit une pile d'acquisition à partir d'une mesure.
    Correspond à l'opérateur $\vartheta(\mu)$ '''
    N_mol = len(m.a)
    taille = dom.N_ech
    acquis_temporelle = np.zeros((T_ech, taille, taille))
    for t in range(T_ech):
        a_tmp = (np.random.rand(N_mol))*m.a
        m_tmp = Mesure2D(a_tmp, m.x)
        acquis_temporelle[t,:] = m_tmp.acquisition(dom.X, dom.Y, taille, fond,
                                                   nv, bruits)
    return acquis_temporelle


def covariance_pile(stack, stack_mean):
    '''Calcule la covariance de y(x,t) à partir de la pile et de sa moyenne'''
    covar = np.zeros((len(stack[0])**2, len(stack[0])**2))
    for i in range(len(stack)):
        covar += np.outer(stack[i,:] - stack_mean, stack[i,:] - stack_mean)
    return covar/len(stack-1)   # T-1 ou T hierher ?


# Le fameux algo de Sliding Frank Wolfe
def SFW(acquis, dom, regul, nIter=5, mesParIter=False, obj='covar'):
    '''y acquisition et nIter nombre d'itérations
    mesParIter est un booléen qui contrôle le renvoi du vecteur mesParIter
    un vecteur qui contient les mesure_k, mesure à la k-ième itération'''
    N_ech_y = dom.N_ech # hierher à adapter
    N_grille = dom.N_ech**2
    if obj == 'covar':
        N_grille = N_grille**2
    X_domain = dom.X
    Y_domain = dom.Y
    a_k = np.empty((0,0))
    x_k = np.empty((0,0))
    mesure_k = Mesure2D()
    x_k_demi = np.empty((0,0))
    Nk = 0                      # Taille de la mesure discrète
    if mesParIter:
        mes_vecteur = np.array([])
    nrj_vecteur = np.zeros(nIter)

    print(f'---- Début de la boucle : {obj} ----')
    for k in range(nIter):
        print('\n' + 'Étape numéro ' + str(k))
        eta_V_k = etak(mesure_k, acquis, dom, regul, obj)
        certif_abs = np.abs(eta_V_k)
        x_star_index = np.unravel_index(np.argmax(certif_abs, axis=None),
                                        eta_V_k.shape)
        x_star = np.array(x_star_index)[::-1]/N_ech_y # hierher passer de l'idx à xstar
        print(fr'* x^* index {x_star} max ' +
              fr'à {np.round(certif_abs[x_star_index], 2)}')

        # Condition d'arrêt (étape 4)
        if np.abs(eta_V_k[x_star_index]) < 1:
            nrj_vecteur[k] = mesure_k.energie(X_domain, Y_domain, acquis,
                                               regul, obj)
            # print(f'* Énergie : {nrj_vecteur[k]:.3f}')
            print("\n\n---- Condition d'arrêt ----")
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
                    gauss = gaussienne_2D(dom.X - xx[i,0], dom.Y - xx[i,1])
                    cov_gauss = np.outer(gauss, gauss)
                    normalis = dom.N_ech**4
                    partial_a[i] = regul - np.sum(cov_gauss*residual)/normalis
                return partial_a
            elif obj == 'acquis':
                for i in range(N):
                    gauss = gaussienne_2D(dom.X - xx[i,0], dom.Y - xx[i,1])
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
        print('* Optim convexe')
        mesure_k_demi += Mesure2D(a_k_demi, x_k_demi)

        # On résout double LASSO non-convexe (étape 8)
        def lasso_double(params):
            a_p = params[:int(len(params)/3)]
            x_p = params[int(len(params)/3):]
            # Bout de code immonde, à corriger !
            x_p = x_p.reshape((len(a_p), 2))
            attache = 0.5*np.linalg.norm(acquis - phi_vecteur(a_p, x_p, dom,
                                                              obj))**2
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
                    gauss = gaussienne_2D(dom.X - x_p[i, 0], dom.Y - x_p[i, 1])
                    cov_gauss = np.outer(gauss, gauss)
                    partial_a[i] = regul - np.sum(cov_gauss*residual)/N_grille

                    gauss_der_x = grad_x_gaussienne_2D(dom.X - x_p[i, 0],
                                                       dom.Y - x_p[i, 1],
                                                       dom.X - x_p[i, 0])
                    cov_der_x = np.outer(gauss_der_x, gauss)
                    partial_x[2*i] = 2*a_p[i]*np.sum(cov_der_x*residual)
                    gauss_der_y = grad_y_gaussienne_2D(dom.X - x_p[i, 0],
                                                       dom.Y - x_p[i, 1],
                                                       dom.Y - x_p[i, 1])
                    cov_der_y = np.outer(gauss_der_y, gauss)
                    partial_x[2*i+1] = 2*a_p[i]*np.sum(cov_der_y*residual)
                return(partial_a + partial_x)
            elif obj == 'acquis':
                for i in range(N):
                    integ = np.sum(residual*gaussienne_2D(dom.X - x_p[i, 0],
                                                          dom.Y - x_p[i, 1]))
                    partial_a[i] = regul - integ/N_grille

                    grad_gauss_x = grad_x_gaussienne_2D(dom.X - x_p[i, 0],
                                                        dom.Y - x_p[i, 1],
                                                        dom.X - x_p[i, 0])
                    integ_x = np.sum(residual*grad_gauss_x)
                    partial_x[2*i] = a_p[i] * integ_x
                    grad_gauss_y = grad_y_gaussienne_2D(dom.X - x_p[i, 0],
                                                        dom.Y - x_p[i, 1],
                                                        dom.Y - x_p[i, 1])
                    integ_y = np.sum(residual*grad_gauss_y)
                    partial_x[2*i+1] = a_p[i] * integ_y

                return(partial_a + partial_x)
            else:
                raise TypeError('Unknown BLASSO target.')

        # On met la graine au format array pour scipy...minimize
        # il faut en effet que ça soit un vecteur
        initial_guess = np.append(a_k_demi, np.reshape(x_k_demi, -1))
        a_part = list(zip([0]*(Nk+1), [10]*(Nk+1)))
        x_part = list(zip([0]*2*(Nk+1), [1]*2*(Nk+1)))
        bounds_bfgs = a_part + x_part
        tes = scipy.optimize.minimize(lasso_double, initial_guess,
                                      method='L-BFGS-B',
                                      jac=grad_lasso_double,
                                      bounds=bounds_bfgs,
                                      options={'disp': __deboggage__})
        a_k_plus = (tes.x[:int(len(tes.x)/3)])
        x_k_plus = (tes.x[int(len(tes.x)/3):]).reshape((len(a_k_plus), 2))
        # print('* a_k_plus : ' + str(np.round(a_k_plus, 2)))
        # print('* x_k_plus : ' + str(np.round(x_k_plus, 2)))

        # Mise à jour des paramètres avec retrait des Dirac nuls
        mesure_k = Mesure2D(a_k_plus, x_k_plus)
        mesure_k = mesure_k.prune(tol=1e-3)
        a_k = mesure_k.a
        x_k = mesure_k.x
        Nk = mesure_k.N
        print(f'* Optim non-convexe : {Nk} Diracs')

        # Graphe et énergie
        nrj_vecteur[k] = mesure_k.energie(X_domain, Y_domain, acquis, regul,
                                          obj)
        print(f'* Énergie : {nrj_vecteur[k]:.3f}')
        if mesParIter == True:
            mes_vecteur = np.append(mes_vecteur, [mesure_k])

    # Fin des itérations
    print("\n\n---- Fin de la boucle ----")
    if mesParIter:
        return(mesure_k, nrj_vecteur, mes_vecteur)
    return(mesure_k, nrj_vecteur)


def wasserstein_metric(mes, m_zer, p_wasser=1):
    '''Retourne la 2--distance de Wasserstein partiel (Partial Gromov-
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
    '''Retourne la 1--distance de Wasserstein partiel (Partial Gromov-
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


def plot_results(m, dom, nrj, certif, moy, title=None, obj='covar'):
    '''Affiche tous les graphes importants pour la mesure m'''
    if m.a.size > 0:
        fig = plt.figure(figsize=(15,12))
        # fig = plt.figure(figsize=(13,10))
        diss = wasserstein_metric(m, m_ax0)
        fig.suptitle(f'Reconstruction {obj} : ' + 
                     r'$\mathcal{{W}}_1(m, m_{a_0,x_0})$' +
                     f' = {diss:.3f}', fontsize=22)

        plt.subplot(221)
        cont1 = plt.contourf(dom.X, dom.Y, moy, 100, cmap='seismic')
        for c in cont1.collections:
            c.set_edgecolor("face")
        plt.colorbar()
        plt.scatter(m_ax0.x[:,0], m_ax0.x[:,1], marker='x',
                    label='GT spikes')
        plt.scatter(m.x[:,0], m.x[:,1], marker='+',
                    label='Recovered spikes')
        plt.legend(loc=2)

        plt.xlabel('X', fontsize=18)
        plt.ylabel('Y', fontsize=18)
        plt.title(r'Temporal mean $\overline{y}$', fontsize=20)

        plt.subplot(222)
        cont2 = plt.contourf(dom.X, dom.Y, m.kernel(dom.X, dom.Y),
                             100, cmap='seismic')
        for c in cont2.collections:
            c.set_edgecolor("face")
        plt.xlabel('X', fontsize=18)
        plt.ylabel('Y', fontsize=18)
        if obj == 'covar':
            plt.title(r'Reconstruction $m_{M,x}$ ' +
                      f'of N = {m.N}', fontsize=20)
        elif obj == 'acquis':
            plt.title(r'Reconstruction $m_{a,x}$ ' +
                      f'of N = {m.N}', fontsize=20)
        plt.colorbar()

        plt.subplot(223)
        if obj == 'acquis':
            X_cer, Y_cer = dom.certif()
            cont3 = plt.contourf(X_cer, Y_cer, certif, 100, cmap='seismic')
        else:
            cont3 = plt.contourf(dom.X, dom.Y, certif, 100, cmap='seismic')
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
            title = 'fig/test-covar-certificat-2d.pdf'
        elif isinstance(title, str):
            title = 'fig/' + title + '.pdf'
        else:
            raise TypeError("You ought to give a str type name for the plot")
        if __saveFig__:
            plt.savefig(title, format='pdf', dpi=1000,
                        bbox_inches='tight', pad_inches=0.03)


def gif_pile(pile_acquis, acquis, m_zer, dom, video='gif', title=None):
    '''Hierher à terminer de débogger'''
    # ax, fig = plt.subplots(figsize=(10,10))
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    # ax.set(xlim=(0, 1), ylim=(0, 1))
    cont_pile = ax.contourf(dom.X, dom.Y, acquis, 100, cmap='seismic')
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
        title = 'fig/anim-pile-test-2d'
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
    divider = make_axes_locatable(ax1) # pour paramétrer colorbar
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
    ax2.scatter(m_zer.x[:,0], m_zer.x[:,1], marker='x',
               s=dom.N_ech, label='Hidden spikes')
    ax2.scatter(m_list[0].x[:,0], m_list[0].x[:,1], marker='+',
                      s=2*dom.N_ech, c='g', label='Recovered spikes')
    ax2.legend(loc=1, fontsize=20)
    plt.tight_layout()

    def animate(k):
        if k >= len(m_list):
            # On fige l'animation pour faire une pause à la pause
            return
        ax2.clear()
        ax2.set_aspect('equal', adjustable='box')
        ax2.contourf(dom.X, dom.Y, m_list[k].kernel(dom.X,dom.Y), 100,
                             cmap='seismic')
        ax2.scatter(m_zer.x[:,0], m_zer.x[:,1], marker='x',
               s=4*dom.N_ech, label='GT spikes')
        ax2.scatter(m_list[k].x[:,0], m_list[k].x[:,1], marker='+',
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
        title = 'fig/anim-sfw-covar-test-2d'
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


# Partie traitement données réelles

def quadrant_divide(pile):
    '''Divise la partie X,Y de la pile en quadrant
    hierher : débugger la récursivité'''
    forme = pile[0,:].shape
    if forme[0] > 16 or forme[1] > 16:
        limx = int(forme[0]/2)
        limy = int(forme[1]/2)
        quadrant_vecteur = np.array([pile[:,:limx,:limy], pile[:,:limx,limy:],
                                    pile[:,limx:,:limy], pile[:,limx:,limy:]])
        return quadrant_vecteur
    else:
        return pile


def recollons_un_quadrant(quadr):
    '''à partir d'une liste de 4 quadrants il reconstruit la pile de base'''
    pile = np.empty((quadr.shape[1], 2*quadr.shape[2], 2*quadr.shape[3]))
    for t in range(quadr.shape[1]):
        stack_col1 = np.hstack((quadr[0,t,:], quadr[1,t,:]))
        stack_col2 = np.hstack((quadr[2,t,:], quadr[3,t,:]))
        stack_lin = np.vstack((stack_col1, stack_col2))
        pile[t,:] = stack_lin
    return pile


def divide(pile, tol=16):
    return


def conquer(sous_piles):
    '''Etape du règne pour l'instant à \lambda constant' '''
    for qdrt in sous_piles:
        pass
    return


# Test sur données réelles
pile_sofi = np.array(io.imread('sofi/siemens_star.tiff'), dtype='float64')
pile_sofi_moy = np.mean(pile_sofi, axis=0)
T_ech = pile_sofi.shape[0]
VRAI_N_ECH = pile_sofi.shape[-1]

bas_red = 6
haut_red = 26
reduc = VRAI_N_ECH/(haut_red - bas_red)

emitters_loc = np.fliplr(np.genfromtxt('sofi/emitters.csv', delimiter=','))
emitters_loc /= VRAI_N_ECH
emitters_loc_test = [el for el in emitters_loc
                     if bas_red/VRAI_N_ECH <
                     np.linalg.norm(el, np.inf) <
                     haut_red/VRAI_N_ECH]

emitters_loc_test = np.vstack(emitters_loc_test) - bas_red/VRAI_N_ECH
emitters_loc_test = reduc * emitters_loc_test
m_ax0 = Mesure2D(np.ones(emitters_loc_test.shape[0]), emitters_loc_test)
# plot_results(m_ax0, domaine, nrj_cov, certif_V, y)

pile_sofi_test = pile_sofi[:, bas_red:haut_red, bas_red:haut_red]
pile_sofi_test = pile_sofi_test / np.max(pile_sofi_test)
pile_sofi_test_moy = np.mean(pile_sofi_test, axis=0)


N_ECH = pile_sofi_test.shape[-1]  # Taux d'échantillonnage
X_GAUCHE = 0
X_DROIT = 1
GRID = np.linspace(X_GAUCHE, X_DROIT, N_ECH)
X, Y = np.meshgrid(GRID, GRID)
domaine = Domain(X_GAUCHE, X_DROIT, N_ECH, GRID, X, Y)

y_bar = np.mean(pile_sofi_test, axis=0)
R_y = covariance_pile(pile_sofi_test, y_bar)


FWMH = 2.2875 / VRAI_N_ECH
sigma = FWMH / (2*np.sqrt(2*np.log(2)))
lambda_regul = 3e-9  # Param de relaxation pour Q_\lambda(y)
lambda_regul2 = 3e-5  # Param de relaxation pour P_\lambda(y_bar)
iteration = m_ax0.N


# Reconstruction
(m_cov, nrj_cov, mes_cov) = SFW(R_y, domaine, lambda_regul, nIter=iteration, 
                                mesParIter=True)
(m_moy, nrj_moy, mes_moy) = SFW(y_bar, domaine, lambda_regul2, nIter=iteration,
                                mesParIter=True, obj='acquis')

print(f'm_Mx : {m_cov.N} Diracs')
print(f'm_ax : {m_moy.N} Diracs')
print(f'm_ax0 : {m_ax0.N} Diracs')

# Métrique de déconvolution : distance de Wasserstein
try:
    dist_x_cov = wasserstein_metric(m_cov, m_ax0)
except ValueError:
    print('[!] Attention Cov Dirac nul')
    dist_x_cov = np.inf
try:
    dist_x_moy = wasserstein_metric(m_moy, m_ax0)
except ValueError:
    print('[!] Attention Moy Dirac nul')
    dist_x_moy = np.inf


print(fr'Dist W_1 des x de Q_\lambda : {dist_x_cov:.3f}')
print(fr'Dist W_1 des x de P_\lambda : {dist_x_moy:.3f}')

if m_cov.a.size > 0:
    certif_V = etak(m_cov, R_y, domaine, lambda_regul)
    plot_results(m_cov, domaine, nrj_cov, certif_V, y_bar)
if m_moy.a.size > 0:
    certificat_V_moy = etak(m_moy, y_bar, domaine, lambda_regul2,
                        obj='acquis')
    plot_results(m_moy, domaine, nrj_moy, certificat_V_moy, y_bar,
                  title='test-moy-certificat-2d', obj='acquis')

if __saveVid__ and m_cov.a.size > 0:
    gif_results(y_bar, m_ax0, mes_cov, domaine)

#%%

import covenant


SIGMA = sigma
domain = covenant.Domain2D(X_GAUCHE, X_DROIT, N_ECH, GRID, X, Y, SIGMA)

(m_cov, nrj_cov, mes_cov) = covenant.SFW(R_y, domain,
                                          regul=lambda_regul,
                                          nIter=iteration, mesParIter=True,
                                          obj='covar')
(m_moy, nrj_moy, mes_moy) = covenant.SFW(y_bar, domain,
                                          regul=lambda_regul2,
                                          nIter=iteration, mesParIter=True,
                                          obj='acquis')

print(f'm_Mx : {m_cov.N} Diracs')
print(f'm_ax : {m_moy.N} Diracs')
print(f'm_ax0 : {m_ax0.N} Diracs')


#%%

import covenant


mes_nulle_tst = Mesure2D()
tst_marche = etak(mes_nulle_tst, R_y, domaine, lambda_regul)
mes_nulle = covenant.Mesure2D()
tst = covenant.etak(mes_nulle, R_y, domain, lambda_regul)


plt.figure()
plt.subplot(121)
plt.imshow(tst)
plt.colorbar()
plt.subplot(122)
plt.imshow(tst_marche)
plt.colorbar()


# # Partie test quadrants
# quadr = quadrant_divide(pile_sofi_test)
# recol = recollons_un_quadrant(quadr)


# plt.figure()
# plt.imshow(recol[0])
# plt.figure()
# plt.imshow(pile_sofi_test[0])


# y = np.mean(quadr[0,:], axis=0) 
# R_y = covariance_pile(quadr[0,:], y)


# N_ECH = y.shape[1] # Taux d'échantillonnage
# X_GAUCHE = 0
# X_DROIT = 1/4
# GRID = np.linspace(X_GAUCHE, X_DROIT, N_ECH)
# X, Y = np.meshgrid(GRID, GRID)
# domaine = Domain(X_GAUCHE, X_DROIT, N_ECH, GRID, X, Y)

# #%%

# emitters_loc = np.fliplr(np.genfromtxt('sofi/emitters.csv', delimiter=','))
# emitters_loc /= VRAI_N_ECH
# m_ax0 = Mesure2D(np.ones(emitters_loc_test.shape[0]), emitters_loc_test)


# cont1 = plt.contourf(domaine.X, domaine.Y, y, 100, cmap='seismic')
# for c in cont1.collections:
#     c.set_edgecolor("face")
# plt.colorbar()
# plt.scatter(m_ax0.x[:,0], m_ax0.x[:,1], marker='x',
#             label='GT spikes')


