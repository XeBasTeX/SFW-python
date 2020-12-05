#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 15:44:06 2020

@author: Bastien (https://github.com/XeBasTeX)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import scipy


sigma = 1e-1 # écart-type de la PSF
# lambda_regul = 1e-3 # Param de relaxation
niveaubruits = 1e-3 # sigma du bruit

N_ech = 100
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
        return(f"{self.N} molécules d'amplitudes : {amplitudes} --- Positions : {positions}")


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
        return np.linalg.norm(self.a, 1)


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
    x = np.round(np.random.rand(1,N), 2)
    a = np.round(np.random.rand(1,N), 2)
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


# Le fameux algo de Sliding Frank Wolfe

def SFW(y, regul=1e-5, nIter=5):
    '''y acquisition et nIter nombre d'itérations'''
    N_ech_y = len(y)
    a_k = np.empty((0,0))
    x_k = np.empty((0,0))
    mesure_k = Mesure2D(a_k, x_k)    # Msure discrète vide
    Nk = 0                      # Taille de la mesure discrète
    nrj_vecteur = np.zeros(nIter)
    for k in range(nIter):
        print('\n' + 'Etape numéro ' + str(k))
        eta_V_k = etak(mesure_k, y, X, Y, regul)
        x_star_index = np.unravel_index(np.argmax(np.abs(eta_V_k), axis=None), eta_V_k.shape)
        x_star = np.array(x_star_index)/N_ech_y # hierher passer de l'idx à xstar
        print(f'* x^* index {x_star} max à {np.round(eta_V_k[x_star_index], 2)}')
        
        # Condition d'arrêt (étape 4)
        if np.abs(eta_V_k[x_star_index]) < 1:
            nrj_vecteur[k] = mesure_k.energie(X, Y, y, regul)
            print(f'* Energie : {nrj_vecteur[k]:.3f}')
            print("\n\n---- Condition d'arrêt ----")
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
            # lasso = lambda a : 0.5*np.linalg.norm(y - phi_vecteur(a,x_k_demi,len(y))) + regul*np.linalg.norm(a, 1)
            res = scipy.optimize.minimize(lasso, np.ones(Nk+1)) # renvoie un array (et pas une liste)
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
            # Il faut en effet que ça soit un vecteur
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
            nrj_vecteur[k] = 0.5*np.linalg.norm(y - mesure_k.kernel(X,Y)) + regul*mesure_k.tv()
            print(f'* Energie : {nrj_vecteur[k]:.3f}')
            
    print("\n\n---- Fin de la boucle ----")
    return(mesure_k, nrj_vecteur)


N_ech = 20
xgauche = 0
xdroit = 1
X_grid = np.linspace(xgauche, xdroit, N_ech)
X, Y = np.meshgrid(X_grid, X_grid)

m_ax0 = Mesure2D([0.5,1,0.8],[[0.25,0.25],[0.75,0.75],[0.25,0.35]])
y = m_ax0.acquisition(X, Y, N_ech, niveaubruits)


lambda_regul = 1e-5 # Param de relaxation
(m_sfw, nrj_sfw) = SFW(y, regul=lambda_regul, nIter=5)
print('On a retrouvé m_ax = ' + str(m_sfw))
certificat_V = etak(m_sfw, y, X, Y, lambda_regul)
print('On voulait retrouver m_ax0 = ' + str(m_ax0))


fig = plt.figure(figsize=(12,12))
fig.suptitle(f'Reconstruction pour $\lambda = {lambda_regul:.0e}$ ' + 
             f'et $\sigma_B = {niveaubruits:.0e}$', fontsize=20)

plt.subplot(221)
plt.contourf(X, Y, y, 50, cmap='hot')
plt.xlabel('X', fontsize=18)
plt.ylabel('Y', fontsize=18)
plt.title('Acquisition $y = \Phi m_{a_0,x_0} + w$', fontsize=20)
plt.colorbar();
# plt.grid()

plt.subplot(222)
plt.contourf(X, Y, m_sfw.kernel(X, Y), 50, cmap='hot')
plt.xlabel('X', fontsize=18)
plt.ylabel('Y', fontsize=18)
plt.title('Reconstruction de $m$', fontsize=20)
plt.colorbar();
# plt.grid()

plt.subplot(223)
plt.contourf(X, Y, certificat_V, 50, cmap='hot')
plt.xlabel('X', fontsize=18)
plt.ylabel('Y', fontsize=18)
plt.title('Certificat $\eta_V$', fontsize=20)
plt.colorbar();

plt.subplot(224)
plt.plot(nrj_sfw, 'o--', color='black', linewidth=2.5)
plt.xlabel('Itération', fontsize=18)
plt.ylabel('$T_\lambda$(m)', fontsize=20)
plt.title('Décroissance énergie', fontsize=20)
plt.grid()


