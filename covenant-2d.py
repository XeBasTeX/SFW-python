#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 10:34:43 2020

@author: Bastien (https://github.com/XeBasTeX)
"""

__author__ = 'Bastien'
__team__ = 'Morpheme'
__saveFig__ = False
__saveVid__ = False
__deboggage__ = True


import numpy as np
from scipy import integrate
import scipy

from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt


np.random.seed(80)
N_ech = 2**4 # Taux d'échantillonnage
xgauche = 0
xdroit = 1
X_grid = np.linspace(xgauche, xdroit, N_ech)
X, Y = np.meshgrid(X_grid, X_grid)

sigma = 1e-1 # écart-type de la PSF
niveaubruits = 1e0 # sigma du bruit


def gaussienne(domain, sigma=sigma):
    '''Gaussienne centrée en 0'''
    return np.sqrt(2*np.pi*sigma**2)*np.exp(-np.power(domain,2)/(2*sigma**2))


def gaussienne2D(X_domain, Y_domain, sigma=sigma):
    '''Gaussienne centrée en 0'''
    expo = np.exp(-(np.power(X_domain,2) + 
                    np.power(Y_domain,2))/(2*sigma**2))
    normalis = np.sqrt(2*np.pi*sigma**2)
    return normalis*expo


def ideal_lowpass(domain, fc):
    '''Passe-bas idéal de fréquence de coupure f_c'''
    return np.sin((2*fc + 1)*np.pi*domain)/np.sin(np.pi*domain)


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
        '''Hieher : il faut encore régler laddition pour les mesures au même'''
        '''position, ça vaudrait le coup de virer les duplicats'''
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
        else:
            raise NameError("Unknown kernel")

    def covariance_kernel(self, X_domain, Y_domain):
        N = self.N
        x = self.x
        a = self.a
        taille_x = np.size(X_domain)
        taille_y = np.size(Y_domain)
        acquis = np.zeros((taille_x, taille_y))
        for i in range(0, N):
            D = np.sqrt(np.power(X_domain - x[i,0],2) + np.power(Y_domain - x[i,1],2))
            noyau_u = gaussienne(D)
            noyau_v = gaussienne(D)
            acquis += a[i]*np.outer(noyau_u, noyau_v)
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
            ax.plot((a_x_torus[i],a_x_torus[i]), (a_y_torus[i],a_y_torus[i]),
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


    def energie(self, X_domain, Y_domain, acquis, regul, obj='covar'):
        if obj == 'covar':
            R_nrj = self.covariance_kernel(X_domain, Y_domain)
            attache = 0.5*np.linalg.norm(acquis - R_nrj)
            parcimonie = regul*self.tv()
            return(attache + parcimonie)
        elif obj == 'acquis':
            attache = 0.5*np.linalg.norm(acquis - self.kernel(X_domain, Y_domain))
            parcimonie = regul*self.tv()
            return(attache + parcimonie)
        else:
            raise TypeError


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
    x = np.round(np.random.rand(N,2), 2)
    a = np.round(0.5 + np.random.rand(1,N), 2)[0]
    return Mesure2D(a, x)

# m = Mesure([1,1.1,0,1.5],[2,88,3,8])
# print(m.prune())

def phi(m, X_domain, Y_domain, obj='covar'):
    if obj == 'covar':
        return m.covariance_kernel(X_domain, Y_domain)
    elif obj == 'acquis':
        return m.kernel(X_domain, Y_domain)
    else:
        raise TypeError


def phi_vecteur(a, x, X_domain, Y_domain, obj='covar'):
    '''shape est un entier indiquant le taille à viser avec le 
        padding'''
    if obj == 'covar':
        m_tmp = Mesure2D(a, x)
        return(m_tmp.covariance_kernel(X_domain, Y_domain))
    elif obj == 'acquis':
        m_tmp = Mesure2D(a, x)
        return(m_tmp.kernel(X_domain, Y_domain))
    else:
        raise TypeError


def phiAdjoint(acquis, X_domain, Y_domain, obj='covar'):
    '''Hierher débugger ; taille_x et taille_y pas implémenté'''
    taille_y = len(X_domain)
    taille_x = len(X_domain[0])
    eta = np.zeros(np.shape(X_domain))
    if obj == 'covar':
        for i in range(taille_x):
            for j in range(taille_y):
                x_decal = X_domain[i,j]
                y_decal = Y_domain[i,j]
                convol = gaussienne2D(x_decal - X, y_decal - Y)
                noyau = np.outer(convol, convol)
                X_range_integ = np.linspace(xgauche, xdroit, N_ech**2)
                Y_range_integ = np.linspace(xgauche, xdroit, N_ech**2)
                integ_x = integrate.simps(acquis*noyau, x=X_range_integ)
                eta[i,j] = integrate.simps(integ_x, x=Y_range_integ)
        return eta
    elif obj == 'acquis':
        for i in range(taille_x):
            for j in range(taille_y):
                x_decal = X_domain[i,j]
                y_decal = Y_domain[i,j]
                D_decal = np.sqrt(np.power(x_decal-X_domain,2) + np.power(y_decal-Y_domain,2))
                integ_x = integrate.simps(acquis*gaussienne(D_decal), x=X_grid)
                eta[i,j] = integrate.simps(integ_x, x=X_grid)
        return eta
    else:
        raise TypeError


def etak(mesure, acquis, X_domain, Y_domain, regul, obj='covar'):
    eta = 1/regul*phiAdjoint(acquis - phi(mesure, X_domain, Y_domain, obj), 
                             X_domain, Y_domain, obj)
    return eta


def pile_aquisition(m):
    '''Construit une pile d'acquisition à partir d'une mesure.'''
    N_mol = len(m.a)
    acquis_temporelle = np.zeros((T_ech, N_ech, N_ech))
    for t in range(T_ech):
        a_tmp = (np.random.random(N_mol))*m.a
        m_tmp = Mesure2D(a_tmp, m.x)
        acquis_temporelle[t,:] = m_tmp.acquisition(X,Y,N_ech,niveaubruits)
    return(acquis_temporelle)


def covariance_pile(stack, stack_mean):
    covar = np.zeros((len(stack[0])**2, len(stack[0])**2))
    for i in range(len(stack)):
        covar += np.outer(stack[i,:] - stack_mean, stack[i,:] - stack_mean)
    return covar/len(stack-1)   # T-1 ou T hierher ?

# J'ai essayé plusieurs versions de calcul mais c'est tjs la même chose :
# la cov que je trouve est la même...

# def covariance_pile_2(stack, stack_mean):
#     covar = np.zeros((len(stack[0])**2, len(stack[0])**2))
#     for i in range(len(stack)):
#         yt = (stack[i,:] - stack_mean).reshape(-1)
#         covar += np.outer(yt, yt.T)
#     return covar/len(stack-1)   # T-1 ou T hierher ?

# def covariance_pile(stack, stack_mean):
#     taille = len(stack[0])**2
#     covar = np.zeros((taille, taille))
#     stack_mean = np.reshape(stack_mean, -1)
#     for i in range(len(stack)):
#         stick_stack = np.reshape(stack[i,:], -1)
#         ps = np.outer(stick_stack - stack_mean, stick_stack - stack_mean)
#         covar += np.reshape(ps, (taille, taille))
#     return covar/len(stack-1)   # T-1 ou T hierher ?


# Le fameux algo de Sliding Frank Wolfe

def SFW(acquis, regul=1e-5, nIter=5, mesParIter=False, obj='covar'):
    '''y acquisition et nIter nombre d'itérations
    mesParIter est un booléen qui contrôle le renvoi du vecteur mesParIter
    un vecteur qui contient mesure_k la mesure de la k-ième itération'''
    N_ech_y = N_ech # hierher à adapter
    a_k = np.empty((0,0))
    x_k = np.empty((0,0))
    mesure_k = Mesure2D()
    # mesure_k = Mesure2D(a_k, x_k)    # Mesure discrète vide
    Nk = 0                      # Taille de la mesure discrète
    if mesParIter == True:
        mes_vecteur = np.array([])
    nrj_vecteur = np.zeros(nIter)
    for k in range(nIter):
        print('\n' + 'Étape numéro ' + str(k))
        eta_V_k = etak(mesure_k, acquis, X, Y, regul, obj)
        x_star_index = np.unravel_index(np.argmax(np.abs(eta_V_k), axis=None), 
                                        eta_V_k.shape)
        x_star = np.array(x_star_index)/N_ech_y # hierher passer de l'idx à xstar
        print(f'* x^* index {x_star} max à {np.round(eta_V_k[x_star_index], 2)}')
        
        # Condition d'arrêt (étape 4)
        if np.abs(eta_V_k[x_star_index]) < 1:
            nrj_vecteur[k:] = mesure_k.energie(X, Y, acquis, regul, obj)
            # print(f'* Énergie : {nrj_vecteur[k]:.3f}')
            print("\n\n---- Condition d'arrêt ----")
            if mesParIter == True:
                return(mesure_k, nrj_vecteur, mes_vecteur)
            else:
                return(mesure_k, nrj_vecteur)
        else:
            mesure_k_demi = Mesure2D()
            if x_k.size == 0:
                x_k_demi = np.vstack([x_star])
                lasso_guess = np.ones(Nk+1)
            else:
                x_k_demi = np.vstack([x_k, x_star])
                lasso_guess = np.concatenate((a_k, [1.0]))

            # On résout LASSO (étape 7)
            def lasso(a):
                attache = 0.5*np.linalg.norm(acquis - phi_vecteur(a,
                                                                  x_k_demi,
                                                                  X, Y, obj))
                parcimonie = regul*np.linalg.norm(a, 1)
                return(attache + parcimonie)
            # lasso_guess = np.ones(Nk+1)
            res = scipy.optimize.minimize(lasso, lasso_guess,
                                          options={'disp': False})
            a_k_demi = res.x
            # print('* a_k_demi : ' + str(np.round(a_k_demi, 2)))
            print('* a_k_demi : ' + str(a_k_demi))
            mesure_k_demi += Mesure2D(a_k_demi, x_k_demi)
            # print('* Mesure_k_demi : ' +  str(mesure_k_demi))

            # On résout double LASSO non-convexe (étape 8)
            def lasso_double(params):
                a_p = params[:int(len(params)/3)] # Bout de code immonde, à corriger !
                x_p = params[int(len(params)/3):]
                x_p = x_p.reshape((len(a_p), 2))
                attache = 0.5*np.linalg.norm(acquis - phi_vecteur(a_p,x_p,
                                                                  X,Y,obj))
                parcimonie = regul*np.linalg.norm(a_p, 1)
                return(attache + parcimonie)

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
            # mesure_k.graphe()
            nrj_vecteur[k] = mesure_k.energie(X, Y, acquis, regul, obj)
            print(f'* Energie : {nrj_vecteur[k]:.3f}')
            if mesParIter == True:
                mes_vecteur = np.append(mes_vecteur, [mesure_k])
            
    print("\n\n---- Fin de la boucle ----")
    if mesParIter == True:
        return(mesure_k, nrj_vecteur, mes_vecteur)
    else:
        return(mesure_k, nrj_vecteur)


def plot_results(m, nrj, certif, title=None):
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
        plt.title('Temporal mean $\overline{y}$', fontsize=20)

        plt.subplot(222)
        cont2 = plt.contourf(X, Y, m.kernel(X, Y), 100, cmap='seismic')
        for c in cont2.collections:
            c.set_edgecolor("face")
        plt.xlabel('X', fontsize=18)
        plt.ylabel('Y', fontsize=18)
        plt.title('Reconstruction $m_{a,x}$', fontsize=20)
        plt.colorbar();

        plt.subplot(223)
        # plt.pcolormesh(X, Y, certificat_V, shading='gouraud', cmap='seismic')
        cont3 = plt.contourf(X, Y, certif, 100, cmap='seismic')
        for c in cont3.collections:
            c.set_edgecolor("face")
        plt.xlabel('X', fontsize=18)
        plt.ylabel('Y', fontsize=18)
        plt.title('Certificate $\eta_V$', fontsize=20)
        plt.colorbar();

        plt.subplot(224)
        plt.plot(nrj, 'o--', color='black', linewidth=2.5)
        plt.xlabel('Itération', fontsize=18)
        plt.ylabel('$T_\lambda(m)$', fontsize=20)
        plt.title('BLASSO energy $\mathcal{Q}_\lambda(y)$', fontsize=20)
        plt.grid()
        
        if title == None:
            title = 'fig/covar-certificat-2d.pdf'
        elif isinstance(title, str):
            title = 'fig/' + title + '.pdf'
        else:
            raise TypeError("You ought to give a str type name for the plot")
        if __saveFig__ == True:
            plt.savefig(title, format='pdf', dpi=1000,
                        bbox_inches='tight', pad_inches=0.03)

def gif_pile(pile_acquis, m_zer, video='gif', title=None):
    '''Hierher à terminer de débogger'''
    # ax, fig = plt.subplots(figsize=(10,10))
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    # ax.set(xlim=(0, 1), ylim=(0, 1))
    cont_pile = ax.contourf(X, Y, y, 100, cmap='seismic')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.15)
    fig.colorbar(cont_pile, cax=cax)
    # plt.tight_layout()

    def animate(k):
        if k >= len(pile_acquis):
            # On fige l'animation pour faire une pause à la fin
            return
        else:
            ax.clear()
            ax.set_aspect('equal', adjustable='box')
            ax.contourf(X, Y, pile_acquis[k,:], 100,cmap='seismic')
            ax.scatter(m_zer.x[:,0], m_zer.x[:,1], marker='x',
                   s=2*N_ech, label='GT spikes')
            ax.set_xlabel('X', fontsize=25)
            ax.set_ylabel('Y', fontsize=25)
            ax.set_title(f'Acquisition numéro = {k}', fontsize=30)
            # ax.legend(loc=1, fontsize=20)
            # ax.tight_layout()

    anim = FuncAnimation(fig, animate, interval=400, frames=len(pile_acquis)+3,
                         blit=False)

    plt.draw()
    
    if title == None:
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

def gif_results(acquis, m_ax0, m_cov, video='gif', title=None):
    fig = plt.figure(figsize=(20, 10))

    ax1 = fig.add_subplot(121)
    ax1.set_aspect('equal', adjustable='box')
    cont = ax1.contourf(X, Y, acquis, 100, cmap='seismic')
    divider = make_axes_locatable(ax1) # pour paramétrer colorbar
    cax = divider.append_axes("right", size="5%", pad=0.15)
    fig.colorbar(cont, cax=cax)
    ax1.set_xlabel('X', fontsize=25)
    ax1.set_ylabel('Y', fontsize=25)
    ax1.set_title('Moyenne $\overline{y}$', fontsize=35)

    ax2 = fig.add_subplot(122)
    # ax2.set(xlim=(0, 1), ylim=(0, 1))
    cont_sfw = ax2.contourf(X, Y, acquis, 100, cmap='seismic')
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="5%", pad=0.15)
    fig.colorbar(cont_sfw, cax=cax)
    ax2.contourf(X, Y, acquis, 100, cmap='seismic')
    ax2.scatter(m_ax0.x[:,0], m_ax0.x[:,1], marker='x',
               s=N_ech, label='Hidden spikes')
    ax2.scatter(mes_cov[0].x[:,0], mes_cov[0].x[:,1], marker='+',
                      s=2*N_ech, c='g', label='Recovered spikes')
    ax2.legend(loc=1, fontsize=20)
    plt.tight_layout()

    def animate(k):
        if k >= len(mes_cov):
            # On fige l'animation pour faire une pause à la pause
            return
        else:
            ax2.clear()
            ax2.set_aspect('equal', adjustable='box')
            ax2.contourf(X, Y, mes_cov[k].kernel(X,Y), 100,
                                 cmap='seismic')
            ax2.scatter(m_ax0.x[:,0], m_ax0.x[:,1], marker='x',
                   s=N_ech, label='GT spikes')
            ax2.scatter(mes_cov[k].x[:,0], mes_cov[k].x[:,1], marker='+',
                          s=2*N_ech, c='g', label='Recovered spikes')
            ax2.set_xlabel('X', fontsize=25)
            ax2.set_ylabel('Y', fontsize=25)
            ax2.set_title(f'Reconstruction itération = {k}', fontsize=35)
            ax2.legend(loc=1, fontsize=20)
            plt.tight_layout()

    anim = FuncAnimation(fig, animate, interval=1000, frames=len(mes_cov)+3,
                         blit=False)

    plt.draw()
    
    if title == None:
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


# m_ax0 = Mesure2D([5,10,8],[[0.25,0.25],[0.65,0.35],[0.35,0.80]])
m_ax0 = Mesure2D([10],[[0.65,0.35]])
# m_ax0 = mesureAleatoire(3)
# y = m_ax0.acquisition(X, Y, N_ech, niveaubruits)

T_ech = 50        # Il faut mettre vraiment bcp d'échantillons pour R_x=R_y !
T_acquis = 1
T = np.linspace(0, T_acquis, T_ech)

pile = pile_aquisition(m_ax0)   
pile_moy = np.mean(pile, axis=0) 
y = pile_moy  
R_y = covariance_pile(pile, pile_moy)
R_x = m_ax0.covariance_kernel(X, Y)


plt.figure(figsize=(15,5))
plt.subplot(121)
plt.imshow(R_x)
plt.colorbar()
plt.title('$R_x$', fontsize=40)
plt.subplot(122)
plt.imshow(R_y)
plt.colorbar()
plt.title('$R_y$', fontsize=40)
if __saveFig__ == True:
    plt.savefig('fig/R_x-R_y-2d.pdf', format='pdf', dpi=1000,
                bbox_inches='tight', pad_inches=0.03)

#%%
lambda_regul = 2e-5 # Param de relaxation pour SFW R_y
lambda_regul2 = 1e-2 # Param de relaxation pour SFW y_moy
iteration = 4

(m_cov, nrj_cov, mes_cov) = SFW(R_y, regul=lambda_regul, nIter=iteration,
                                  mesParIter=True, obj='covar')
# (m_moy, nrj_moy, mes_moy) = SFW(y, regul=lambda_regul2, nIter=iteration,
#                                   mesParIter=True, obj='acquis')

print('On a retrouvé m_Mx = ' + str(m_cov))
certificat_V = etak(m_cov, R_y, X, Y, lambda_regul, obj='covar')
print('On voulait retrouver m_ax0 = ' + str(m_ax0))

# print('On a retrouvé m_ax = ' + str(m_moy))
# certificat_V_moy = etak(m_moy, y, X, Y, lambda_regul2, obj='acquis')
# print('On voulait retrouver m_ax0 = ' + str(m_ax0))

# y_simul = m_cov.kernel(X,Y)

if m_cov.a.size > 0:
    plot_results(m_cov, nrj_cov, certificat_V)
# if m_moy.a.size > 0:
#     plot_results(m_moy, nrj_moy, certificat_V_moy, 
#                   title='covar-moy-certificat-2d')
# if __saveVid__ == True:
#     if m_cov.a.size > 0:
#         gif_results(y, m_ax0, m_cov)
#     gif_pile(pile, m_ax0)


