#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 15:44:06 2020

@author: Bastien (https://github.com/XeBasTeX)
"""


__author__ = 'Bastien Laville'
__team__ = 'Morpheme'
__saveFig__ = True
__saveVid__ = True
__deboggage__ = True


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import integrate
import scipy
# import seaborn as sns


np.random.seed(80)
sigma = 1e-1 # écart-type de la PSF
# lambda_regul = 1e-3 # Param de relaxation
niveaubruits = 1e-2 # sigma du bruit

N_ech = 30
xgauche = 0
xdroit = 1
X_grid = np.linspace(xgauche, xdroit, N_ech)
X, Y = np.meshgrid(X_grid, X_grid)


class Mesure2D:
    
    def __init__(self, amplitude=[], position=[]):
        if len(amplitude) != len(position):
            raise ValueError('Pas le même nombre')
        if isinstance(amplitude, np.ndarray) and isinstance(position, np.ndarray):
            self.a = amplitude
            self.x = position
        else:
            self.x = np.array(position)
            self.a = np.array(amplitude)
        self.N = len(amplitude)


    def __add__(self, m):
        '''Hieher : il faut encore régler laddition pour les mesures au même position'''
        '''ça vaudrait le coup de virer les duplicats'''
        a_new = np.append(self.a, m.a)
        x_new = np.array(list(self.x) + list(m.x))
        return Mesure2D(a_new, x_new)


    def __eq__(self, m):
        if m == 0 and (self.a == [] and self.x == []):
            return True
        elif isinstance(m, self.__class__):
            return self.__dict__ == m.__dict__
        else:
            return False


    def __ne__(self, m):
        return not self.__eq__(m)


    def __str__(self):
        amplitudes = np.round(self.a, 3)
        positions = np.round(self.x, 3)
        return(f"{self.N} Diracs \nAmplitudes : {amplitudes}\nPositions : {positions}")


    def kernel(self, X_domain, Y_domain, noyau='gaussien'):
        '''Applique un noyau à la mesure discrète. Exemple : convol'''
        N = self.N
        x = self.x
        a = self.a
        acquis = X_domain*0
        if noyau == 'gaussien':
            for i in range(0,N):
                D = np.sqrt(np.power(X_domain - x[i,0],2) + np.power(Y_domain - x[i,1],2))
                acquis += a[i]*gaussienne(D)
            return acquis
        elif noyau == 'double_gaussien':
            for i in range(0,N):
                D = np.sqrt(np.power(X_domain - x[i,0],2) + np.power(Y_domain - x[i,1],2))
                acquis += a[i]*double_gaussienne(D)
            return acquis
        else:
            return acquis


    def acquisition(self, X_domain, Y_domain, N_ech, nv):
        w = nv*np.random.random_sample((N_ech, N_ech))
        acquis = self.kernel(X_domain, Y_domain, noyau='gaussien') + w
        return acquis


    def graphe(self, X_domain, Y_domain, lvl=50):
        # f = plt.figure()
        plt.contourf(X_domain, Y_domain, self.kernel(X_domain, Y_domain), 
                     lvl, label='$y_0$', cmap='hot')
        plt.xlabel('X', fontsize=18)
        plt.ylabel('Y', fontsize=18)
        plt.title('Acquisition y', fontsize=18)
        plt.colorbar();
        plt.grid()
        plt.show()
        return


    def torus(self, X, current_fig=False, subplot=False):
        N_ech = len(X)
        if subplot == True:
            ax = current_fig.add_subplot(224, projection='3d')
        else:
            current_fig = plt.figure()
            ax = current_fig.add_subplot(111, projection='3d')
        theta = np.linspace(0, 2*np.pi, N_ech)
        y_torus = np.sin(theta)
        x_torus = np.cos(theta)

        a_x_torus = np.sin(2*np.pi*np.array(self.x))
        a_y_torus = np.cos(2*np.pi*np.array(self.x))
        a_z_torus = self.a

        # ax.plot((0,0),(0,0), (0,1), '-k', label='z-axis')
        ax.plot(x_torus,y_torus, 0, '-k', label='$\mathbb{S}_1$')
        ax.plot(a_x_torus,a_y_torus, a_z_torus,
                'o', label='$m_{a,x}$', color='orange')

        for i in range(self.N):
            ax.plot((a_x_torus[i],a_x_torus[i]),(a_y_torus[i],a_y_torus[i]),
                    (0,a_z_torus[i]), '--r', color='orange')

        ax.set_xlabel('$X$')
        ax.set_ylabel('$Y$')
        ax.set_zlabel('$Amplitude$')
        ax.set_title('Mesure sur $\mathbb{S}_1$', fontsize=20)
        ax.legend()
        return 


    def tv(self):
        '''Renvoie 0 si la mesure est vide : hierher à vérifier'''
        try:
            return np.linalg.norm(self.a, 1)
        except ValueError:
            return 0


    def energie(self, X_domain, Y_domain, y, regul):
        attache = 0.5*np.linalg.norm(y - self.kernel(X_domain, Y_domain))
        parcimonie = regul*self.tv()
        return(attache + parcimonie)


    def prune(self, tol=1e-3):
        '''Retire les dirac avec zéro d'amplitude'''
        # nnz = np.count_nonzero(self.a)
        nnz_a = np.array(self.a)
        nnz = nnz_a > tol
        nnz_a = nnz_a[nnz]
        nnz_x = np.array(self.x)
        nnz_x = nnz_x[nnz]
        m = Mesure2D(nnz_a, nnz_x)
        return m


def mesureAleatoire(N):
    x = np.round(np.random.rand(N,2), 2)
    a = np.round(0.5 + np.random.rand(1,N), 2)[0]
    return Mesure2D(a, x)

# m = Mesure([1,1.1,0,1.5],[2,88,3,8])
# print(m.prune())

def phi(m, X_domain, Y_domain):
    return m.kernel(X_domain, Y_domain)


def phi_vecteur(a, x, X_domain, Y_domain):
    '''shape est un entier indiquant le taille à viser avec le 
        padding'''
    m_tmp = Mesure2D(a, x)
    return(m_tmp.kernel(X_domain, Y_domain))


def phiAdjoint(y, X_domain, Y_domain, noyau='gaussien'):
    taille_y = len(y)
    eta = np.empty(np.shape(y))
    for i in range(taille_y):
        for j in range(taille_y):
            x_decal = X_domain[i,j]
            y_decal = Y_domain[i,j]
            D_decal = np.sqrt(np.power(x_decal-X_domain,2) + np.power(y_decal-Y_domain,2))
            integ_x = integrate.simps(y*gaussienne(D_decal), x=X_grid)
            eta[i,j] = integrate.simps(integ_x, x=X_grid)
    return eta


def etaW(x, N, sigma, noyau='gaussien'):
    '''Certificat \eta_W dans le cas gaussien, avec formule analytique'''
    x = x/sigma # Normalisation
    tmp = 0
    for k in range(1,N+1):
        tmp += (x**(2*k))/(2**(2*k)*np.math.factorial(k))
    eta = np.exp(-x**2/4)*tmp
    return eta


def etaWx0(x, x0, mesure, sigma, noyau='gaussien'):
    '''Certificat \eta_W dans le cas gaussien'''
    N = mesure.N
    x = x/sigma # Normalisation
    tmp = 0
    for k in range(N):
        tmp += ((x - x0)**(2*k))/((2*sigma)**(2*k)*np.math.factorial(k))
    eta = np.exp(-(x - x0)**2/4)*tmp
    return eta


# plt.plot(X,etaW(X,5,0.1))


def etak(mesure, y, X_domain, Y_domain, regul):
    eta = 1/regul*phiAdjoint(y - phi(mesure, X_domain, Y_domain), X_domain, Y_domain)
    return eta

def gaussienne(domain, sigma=1e-1):
    '''Gaussienne centrée en 0'''
    return np.sqrt(2*np.pi*sigma**2)*np.exp(-np.power(domain,2)/(2*sigma**2))


def double_gaussienne(domain):
    '''Gaussienne au carré centrée en 0'''
    return np.power(gaussienne(domain),2)


def ideal_lowpass(domain, fc):
    '''Passe-bas idéal de fréquence de coupure f_c'''
    return np.sin((2*fc + 1)*np.pi*domain)/np.sin(np.pi*domain)



# Le fameux algo de Sliding Frank Wolfe

def SFW(y, regul=1e-5, nIter=5, mesParIter=False):
    '''y acquisition et nIter nombre d'itérations
    mesParIter est un booléen qui contrôle le renvoi du vecteur mesParIter
    un vecteur qui contient mesure_k la mesure de la k-ième itération'''
    N_ech_y = len(y)
    a_k = np.empty((0,0))
    x_k = np.empty((0,0))
    mesure_k = Mesure2D(a_k, x_k)    # Mesure discrète vide
    Nk = 0                      # Taille de la mesure discrète
    if mesParIter == True:
        mes_vecteur = np.array([])
    nrj_vecteur = np.zeros(nIter)
    for k in range(nIter):
        print('\n' + 'Étape numéro ' + str(k))
        eta_V_k = etak(mesure_k, y, X, Y, regul)
        x_star_index = np.unravel_index(np.argmax(np.abs(eta_V_k), axis=None), eta_V_k.shape)
        x_star = np.array(x_star_index)/N_ech_y # hierher passer de l'idx à xstar
        print(f'* x^* index {x_star} max à {np.round(eta_V_k[x_star_index], 2)}')
        
        # Condition d'arrêt (étape 4)
        if np.abs(eta_V_k[x_star_index]) < 1:
            nrj_vecteur[k] = mesure_k.energie(X, Y, y, regul)
            print(f'* Énergie : {nrj_vecteur[k]:.3f}')
            print("\n\n---- Condition d'arrêt ----")
            if mesParIter == True:
                return(mesure_k, nrj_vecteur, mes_vecteur)
            else:
                return(mesure_k, nrj_vecteur)
        else:
            mesure_k_demi = Mesure2D()
            if x_k.size == 0:
                x_k_demi = np.vstack([x_star])
            else:
                x_k_demi = np.vstack([x_k, x_star])

            # On résout LASSO (étape 7)
            def lasso(a):
                attache = 0.5*np.linalg.norm(y - phi_vecteur(a,x_k_demi,X,Y))
                parcimonie = regul*np.linalg.norm(a, 1)
                return(attache + parcimonie)
            res = scipy.optimize.minimize(lasso, np.ones(Nk+1))
            a_k_demi = res.x
            print('* a_k_demi : ' + str(np.round(a_k_demi, 2)))
            mesure_k_demi += Mesure2D(a_k_demi,x_k_demi)
            print('* Mesure_k_demi : ' +  str(mesure_k_demi))

            # On résout double LASSO non-convexe (étape 8)
            def lasso_double(params):
                a_p = params[:int(len(params)/3)] # Bout de code immonde, à corriger !
                x_p = params[int(len(params)/3):]
                x_p = x_p.reshape((len(a_p), 2))
                attache = 0.5*np.linalg.norm(y - phi_vecteur(a_p,x_p,X,Y))
                parcimonie = regul*np.linalg.norm(a_p, 1)
                return(attache + parcimonie)

            # On met la graine au format array pour scipy...minimize
            # il faut en effet que ça soit un vecteur
            initial_guess = np.append(a_k_demi, np.reshape(x_k_demi, -1))
            res = scipy.optimize.minimize(lasso_double, initial_guess,
                                          method='BFGS',
                                          options={'disp': True})
            a_k_plus = (res.x[:int(len(res.x)/3)])
            x_k_plus = (res.x[int(len(res.x)/3):]).reshape((len(a_k_plus), 2))

            # Mise à jour des paramètres avec retrait des Dirac nuls
            mesure_k = Mesure2D(a_k_plus, x_k_plus)
            mesure_k = mesure_k.prune(tol=1e-2)
            a_k = mesure_k.a
            x_k = mesure_k.x
            Nk = mesure_k.N

            # Graphe et énergie
            # mesure_k.graphe()
            nrj_vecteur[k] = mesure_k.energie(X, Y, y, regul)
            print(f'* Energie : {nrj_vecteur[k]:.3f}')
            if mesParIter == True:
                mes_vecteur = np.append(mes_vecteur, [mesure_k])
            
    print("\n\n---- Fin de la boucle ----")
    if mesParIter == True:
        return(mesure_k, nrj_vecteur, mes_vecteur)
    else:
        return(mesure_k, nrj_vecteur)


def plot_results(m):
    if m.a.size > 0:
        fig = plt.figure(figsize=(15,12))
        fig.suptitle(f'Reconstruction pour $\lambda = {lambda_regul:.0e}$ ' + 
                     f'et $\sigma_B = {niveaubruits:.0e}$', fontsize=20)

        plt.subplot(221)
        cont1 = plt.contourf(X, Y, y, 100, cmap='seismic')
        for c in cont1.collections:
            c.set_edgecolor("face")
        plt.colorbar();
        plt.scatter(m_ax0.x[:,0], m_ax0.x[:,1], marker='x',
                    label='GT spikes')
        plt.scatter(m.x[:,0], m.x[:,1], marker='+',
                    label='Recovered spikes')
        plt.legend(loc=2)

        plt.xlabel('X', fontsize=18)
        plt.ylabel('Y', fontsize=18)
        plt.title('Acquisition $y = \Phi m_{a_0,x_0} + w$', fontsize=20)

        plt.subplot(222)
        cont2 = plt.contourf(X, Y, m.kernel(X, Y), 100, cmap='seismic')
        for c in cont2.collections:
            c.set_edgecolor("face")
        plt.xlabel('X', fontsize=18)
        plt.ylabel('Y', fontsize=18)
        plt.title('Reconstruction $m_{a,x}$', fontsize=20)
        plt.colorbar();

        plt.subplot(223)
        cont3 = plt.contourf(X, Y, certificat_V, 100, cmap='seismic')
        for c in cont3.collections:
            c.set_edgecolor("face")
        plt.xlabel('X', fontsize=18)
        plt.ylabel('Y', fontsize=18)
        plt.title('Certificate $\eta_V$', fontsize=20)
        plt.colorbar();

        plt.subplot(224)
        plt.plot(nrj_sfw, 'o--', color='black', linewidth=2.5)
        plt.xlabel('Itération', fontsize=18)
        plt.ylabel('$T_\lambda(m)$', fontsize=20)
        plt.title('BLASSO energy $T_\lambda(m)$', fontsize=20)
        plt.grid()
        if __saveFig__ == True:
            plt.savefig('fig/dirac-certificat-2d.pdf', format='pdf', dpi=1000,
                        bbox_inches='tight', pad_inches=0.03)


def gif_results(y, m_ax0, m_sfw, video='gif'):
    fig = plt.figure(figsize=(20, 10))

    ax1 = fig.add_subplot(121)
    ax1.set_aspect('equal', adjustable='box')
    cont = ax1.contourf(X, Y, y, 100, cmap='seismic')
    divider = make_axes_locatable(ax1) # pour paramétrer colorbar
    cax = divider.append_axes("right", size="5%", pad=0.15)
    fig.colorbar(cont, cax=cax)
    ax1.set_xlabel('X', fontsize=25)
    ax1.set_ylabel('Y', fontsize=25)
    ax1.set_title('Acquisition $y$', fontsize=35)

    ax2 = fig.add_subplot(122)
    # ax2.set(xlim=(0, 1), ylim=(0, 1))
    cont_sfw = ax2.contourf(X, Y, y, 100, cmap='seismic')
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="5%", pad=0.15)
    fig.colorbar(cont_sfw, cax=cax)
    ax2.contourf(X, Y, y, 100, cmap='seismic')
    ax2.scatter(m_ax0.x[:,0], m_ax0.x[:,1], marker='x',
               s=N_ech, label='Hidden spikes')
    ax2.scatter(mes_sfw[0].x[:,0], mes_sfw[0].x[:,1], marker='+',
                      s=2*N_ech, c='g', label='Recovered spikes')
    ax2.legend(loc=1, fontsize=20)
    plt.tight_layout()

    def animate(k):
        if k >= len(mes_sfw):
            # On fige l'animation pour faire une pause à la pause
            return
        else:
            ax2.clear()
            ax2.set_aspect('equal', adjustable='box')
            ax2.contourf(X, Y, mes_sfw[k].kernel(X,Y), 100,
                                 cmap='seismic')
            ax2.scatter(m_ax0.x[:,0], m_ax0.x[:,1], marker='x',
                   s=N_ech, label='GT spikes')
            ax2.scatter(mes_sfw[k].x[:,0], mes_sfw[k].x[:,1], marker='+',
                          s=2*N_ech, c='g', label='Recovered spikes')
            ax2.set_xlabel('X', fontsize=25)
            ax2.set_ylabel('Y', fontsize=25)
            ax2.set_title(f'Reconstruction itération = {k}', fontsize=35)
            ax2.legend(loc=1, fontsize=20)
            plt.tight_layout()

    anim = FuncAnimation(fig, animate, interval=1000, frames=len(mes_sfw)+3,
                         blit=False)

    plt.draw()
    if video == "mp4":
        anim.save('fig/anim-sfw-2d.mp4')
    elif video == "gif":
        anim.save('fig/anim-sfw-2d.gif')
    else:
        raise ValueError('Unknown video format')
    return fig


# m_ax0 = Mesure2D([0.5,1,0.8],[[0.25,0.25],[0.75,0.75],[0.25,0.35]])
m_ax0 = mesureAleatoire(9)
y = m_ax0.acquisition(X, Y, N_ech, niveaubruits)

lambda_regul = 9e-5 # Param de relaxation
iteration = 11

# (m_sfw, nrj_sfw) = SFW(y, regul=lambda_regul, nIter=iteration)
(m_sfw, nrj_sfw, mes_sfw) = SFW(y, regul=lambda_regul, nIter=iteration,
                                 mesParIter=True)
print('On a retrouvé m_ax = ' + str(m_sfw))
certificat_V = etak(m_sfw, y, X, Y, lambda_regul)
print('On voulait retrouver m_ax0 = ' + str(m_ax0))

y_simul = m_sfw.kernel(X,Y)

if __saveFig__ == True:
    plot_results(m_sfw)
if __saveVid__ == True and m_sfw.a.size > 0:
    gif_results(y, m_ax0, m_sfw)


# #%% Wasserstein 2D
# # Il faudrait utiliser sliced_wasserstein
# # Mais la dernière version n'est pas dispo
# J'ai essayé gudhi, c'était tout sauf concluant
