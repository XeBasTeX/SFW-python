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
import scipy

from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt

import os
from skimage import io


os.chdir(os.path.dirname(os.path.abspath(__file__)))

N_ech = 2**4 # Taux d'échantillonnage
xgauche = 0
xdroit = 1
X_grid = np.linspace(xgauche, xdroit, N_ech)
X, Y = np.meshgrid(X_grid, X_grid)

sigma = 1e-1 # écart-type de la PSF

def gaussienne(domain, sigma_g=sigma):
    '''Gaussienne centrée en 0'''
    expo = np.exp(-np.power(domain,2)/(2*sigma_g**2))
    return np.sqrt(2*np.pi*sigma_g**2) * expo


def gaussienne_2D(X_domain, Y_domain, sigma_g=sigma):
    '''Gaussienne centrée en 0'''
    expo = np.exp(-(np.power(X_domain,2) +
                    np.power(Y_domain,2))/(2*sigma_g**2))
    normalis = np.sqrt(2*np.pi*sigma_g**2)
    return normalis*expo


def ideal_lowpass(domain, fc):
    '''Passe-bas idéal de fréquence de coupure f_c'''
    return np.sin((2*fc + 1)*np.pi*domain)/np.sin(np.pi*domain)


class Domain:
    def __init__(self, gauche, droit, ech, grid, X_domain, Y_domain):
        self.x_gauche = gauche
        self.x_droit = droit
        self.N_ech = ech
        self.X_grid = grid
        self.X = X_domain
        self.Y = Y_domain


class Bruits2D:
    def __init__(self, fond, niveau, type_bruits):
        self.fond = fond
        self.niveau = niveau
        self.type = type_bruits


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
        '''Noyau de covariance associée à la mesure'''
        N = self.N
        x = self.x
        a = self.a
        taille_x = np.size(X_domain)
        taille_y = np.size(Y_domain)
        acquis = np.zeros((taille_x, taille_y))
        for i in range(0, N):
            D = (np.sqrt(np.power(X_domain - x[i,0],2) +
                         np.power(Y_domain - x[i,1],2)))
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
            if obj == 'covar':
                R_nrj = self.covariance_kernel(X_domain, Y_domain)
                attache = 0.5*np.linalg.norm(acquis - R_nrj)
                parcimonie = regul*self.tv()
                return attache + parcimonie
            if obj == 'acquis':
                simul = self.kernel(X_domain, Y_domain)
                attache = 0.5*np.linalg.norm(acquis - simul)
                parcimonie = regul*self.tv()
                return attache + parcimonie
        elif bruits in ('gauss', 'unif'):
            if obj == 'covar':
                R_nrj = self.covariance_kernel(X_domain, Y_domain)
                attache = 0.5*np.linalg.norm(acquis - R_nrj)
                parcimonie = regul*self.tv()
                return attache + parcimonie
            if obj == 'acquis':
                simul = self.kernel(X_domain, Y_domain)
                attache = 0.5*np.linalg.norm(acquis - simul)
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


def mesureAleatoire(N):
    '''Créé une mesure aléatoire de N pics d'amplitudes aléatoires comprises
    entre 0,5 et 1,5'''
    x = np.round(np.random.rand(N,2), 2)
    a = np.round(0.5 + np.random.rand(1,N), 2)[0]
    return Mesure2D(a, x)

# m = Mesure([1,1.1,0,1.5],[2,88,3,8])
# print(m.prune())

def phi(m, dom, obj='covar'):
    X_domain = dom.X
    Y_domain = dom.Y
    '''créé le résultat d'un opérateur d'acquisition à partir de la mesure m'''
    if obj == 'covar':
        return m.covariance_kernel(X_domain, Y_domain)
    if obj == 'acquis':
        return m.kernel(X_domain, Y_domain)
    raise TypeError


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
    raise TypeError


def phiAdjoint(acquis, dom, obj='covar'):
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
                integ_x = integrate.simps(acquis*noyau, x=X_range_integ)
                eta[i,j] = integrate.simps(integ_x, x=Y_range_integ)
        return eta
    if obj == 'acquis':
        for i in range(taille_x):
            for j in range(taille_y):
                x_decal = X_domain[i,j]
                y_decal = Y_domain[i,j]
                D_decal = (np.sqrt(np.power(x_decal-X_domain,2)
                                   + np.power(y_decal-Y_domain,2)))
                integ_x = integrate.simps(acquis*gaussienne(D_decal),
                                          x=dom.X_grid)
                eta[i,j] = integrate.simps(integ_x, x=dom.X_grid)
        return eta
    raise TypeError


def etak(mesure, acquis, dom, regul, obj='covar'):
    r'''Certificat $\eta$ assicé à la mesure'''
    eta = 1/regul*phiAdjoint(acquis - phi(mesure, dom, obj),
                             dom, obj)
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

def SFW(acquis, dom, regul=1e-5, nIter=5, mesParIter=False, obj='covar'):
    '''y acquisition et nIter nombre d'itérations
    mesParIter est un booléen qui contrôle le renvoi du vecteur mesParIter
    un vecteur qui contient les mesure_k, mesure à la k-ième itération'''
    N_ech_y = dom.N_ech # hierher à adapter
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
    for k in range(nIter):
        print('\n' + 'Étape numéro ' + str(k))
        eta_V_k = etak(mesure_k, acquis, dom, regul, obj)
        x_star_index = np.unravel_index(np.argmax(np.abs(eta_V_k), axis=None),
                                        eta_V_k.shape)
        x_star = np.array(x_star_index)[::-1]/N_ech_y # hierher passer de l'idx à xstar
        print(fr'* x^* index {x_star} max ' +
              fr'à {np.round(eta_V_k[x_star_index], 2)}')

        # Condition d'arrêt (étape 4)
        if np.abs(eta_V_k[x_star_index]) < 1:
            nrj_vecteur[k:] = mesure_k.energie(X_domain, Y_domain, acquis,
                                               regul, obj)
            # print(f'* Énergie : {nrj_vecteur[k]:.3f}')
            print("\n\n---- Condition d'arrêt ----")
            if mesParIter:
                return(mesure_k, nrj_vecteur, mes_vecteur)
            return(mesure_k, nrj_vecteur)

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
            attache = 0.5*np.linalg.norm(difference)
            parcimonie = regul*np.linalg.norm(aa, 1)
            return attache + parcimonie
        res = scipy.optimize.minimize(lasso, lasso_guess,
                                      options={'disp': __deboggage__})
        a_k_demi = res.x

        print('* x_k_demi : ' + str(x_k_demi))
        print('* a_k_demi : ' + str(a_k_demi))
        mesure_k_demi += Mesure2D(a_k_demi, x_k_demi)

        # On résout double LASSO non-convexe (étape 8)
        def lasso_double(params):
            a_p = params[:int(len(params)/3)]
            x_p = params[int(len(params)/3):]
            # Bout de code immonde, à corriger !
            x_p = x_p.reshape((len(a_p), 2))
            attache = 0.5*np.linalg.norm(acquis - phi_vecteur(a_p, x_p,
                                                              dom, obj))
            parcimonie = regul*np.linalg.norm(a_p, 1)
            return attache + parcimonie

        # On met la graine au format array pour scipy...minimize
        # il faut en effet que ça soit un vecteur
        initial_guess = np.append(a_k_demi, np.reshape(x_k_demi, -1))
        res = scipy.optimize.minimize(lasso_double, initial_guess,
                                      method='BFGS',
                                      options={'disp': __deboggage__})
        a_k_plus = (res.x[:int(len(res.x)/3)])
        x_k_plus = (res.x[int(len(res.x)/3):]).reshape((len(a_k_plus), 2))
        print('* a_k_plus : ' + str(np.round(a_k_plus, 2)))
        print('* x_k_plus : ' +  str(x_k_plus))

        # Mise à jour des paramètres avec retrait des Dirac nuls
        mesure_k = Mesure2D(a_k_plus, x_k_plus)
        mesure_k = mesure_k.prune(tol=1e-3)
        a_k = mesure_k.a
        x_k = mesure_k.x
        Nk = mesure_k.N

        # Graphe et énergie
        nrj_vecteur[k] = mesure_k.energie(X_domain, Y_domain, acquis, regul,
                                          obj)
        print(f'* Energie : {nrj_vecteur[k]:.3f}')
        if mesParIter == True:
            mes_vecteur = np.append(mes_vecteur, [mesure_k])

    # Fin des itérations
    print("\n\n---- Fin de la boucle ----")
    if mesParIter:
        return(mesure_k, nrj_vecteur, mes_vecteur)
    return(mesure_k, nrj_vecteur)


def plot_results(m, dom, nrj, certif, moy, title=None, obj='covar'):
    '''Affiche tous les graphes importants pour la mesure m'''
    if m.a.size > 0:
        # fig = plt.figure(figsize=(15,12))
        fig = plt.figure(figsize=(13,10))
        fig.suptitle(r'Reconstruction', fontsize=20)

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
                      f'à N = {m.N}', fontsize=20)
        elif obj == 'acquis':
            plt.title(r'Reconstruction $m_{a,x}$ ' +
                      f'à N = {m.N}', fontsize=20)
        plt.colorbar()

        plt.subplot(223)
        # plt.pcolormesh(X, Y, certificat_V, shading='gouraud', cmap='seismic')
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
            title = 'fig/covar-certificat-2d.pdf'
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
    fig = plt.figure(figsize=(8,8))
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
        ax.contourf(dom.X, dom.Y, pile_acquis[k,:], 100,cmap='seismic')
        ax.scatter(m_zer.x[:,0], m_zer.x[:,1], marker='x',
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

# Partie traitement données réelles

def quadrant_divide(pile, tol=16):
    '''Divise récursivement la partie X,Y de la pile en quadrant
    hierher : débugger la récursivité'''
    forme = pile[0,:].shape
    quadrant_vecteur = pile
    while forme[0] > 16 or forme[1] > 16:
        limx = int(forme[0]/2)
        limy = int(forme[1]/2)
        quadrant_vecteur = np.array([pile[:,:limx,:limy], pile[:,:limx,limy:],
                                    pile[:,limx:,:limy], pile[:,limx:,limy:]])
        return quadrant_divide(quadrant_vecteur, tol)
    return quadrant_vecteur


def recollons_un_quadrant(quadrant):
    '''à partir d'une liste de 4 quadrants il reconstruit la pile de base'''
    pile = np.empty((quadr.shape[1], 2*quadr.shape[2], 2*quadr.shape[3]))
    for t in range(quadr.shape[1]):
        stack_col1 = np.hstack((quadr[0,t,:], quadr[1,t,:]))
        stack_col2 = np.hstack((quadr[2,t,:], quadr[3,t,:]))
        stack_lin = np.vstack((stack_col1, stack_col2))
        pile[t,:] = stack_lin
    return pile

m_ax0 = Mesure2D([8,10,6],[[0.2,0.23],[0.80,0.35],[0.33,0.82]])


# Test sur données réelles
pile_sofi = np.array(io.imread('sofi/siemens_star.tiff'), dtype='float64')
pile_sofi_moy = np.mean(pile_sofi, axis=0) 

T_ech = pile_sofi.shape[0]

pile_sofi_test = pile_sofi[:3,:32,:32]
quadr = quadrant_divide(pile_sofi_test)
recol = recollons_un_quadrant(quadr)


plt.figure()
plt.imshow(recol[0])
plt.figure()
plt.imshow(pile_sofi_test[0])


# y = pile_sofi_moy  
# R_y = covariance_pile(pile_sofi, pile_sofi_moy)
# R_x = m_ax0.covariance_kernel(X, Y)

# # Param régul
# lambda_regul = 5e-6 # Param de relaxation pour SFW R_y
# lambda_regul2 = 8e-2 # Param de relaxation pour SFW y_moy

# iteration = m_ax0.N + 1

# (m_cov, nrj_cov, mes_cov) = SFW(R_y, regul=lambda_regul, nIter=iteration,
#                                   mesParIter=True, obj='covar')
# (m_moy, nrj_moy, mes_moy) = SFW(y, regul=lambda_regul2, nIter=iteration,
#                                   mesParIter=True, obj='acquis')

# print('On a retrouvé m_Mx = ' + str(m_cov))
# certificat_V = etak(m_cov, R_y, X, Y, lambda_regul, obj='covar')
# print('On voulait retrouver m_ax0 = ' + str(m_ax0))

# print('On a retrouvé m_ax = ' + str(m_moy))
# certificat_V_moy = etak(m_moy, y, X, Y, lambda_regul2, obj='acquis')
# print('On voulait retrouver m_ax0 = ' + str(m_ax0))


# true_pos =  np.sort(m_ax0.x, axis=0)
# dist_x_cov = str(np.linalg.norm(true_pos - np.sort(m_cov.x,axis=0)))
# dist_x_moy = str(np.linalg.norm(true_pos - np.sort(m_moy.x,axis=0)))
# print('Dist L^2 des x de Q_\lambda : ' + dist_x_cov)
# print('Dist L^2 des x de P_\lambda : ' + dist_x_moy)


# y_simul = m_cov.kernel(X,Y)

# if m_cov.a.size > 0:
#     plot_results(m_cov, y, nrj_cov, certificat_V)
# if m_moy.a.size > 0:
#     plot_results(m_moy, y, nrj_moy, certificat_V_moy, 
#                   title='covar-moy-certificat-2d', obj='acquis')



