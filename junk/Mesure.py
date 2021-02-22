#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 17:12:54 2020

@author: Bastien (https://github.com/XeBasTeX)
"""


# Je vais tout trier ici ! On va faire une classe pour nD

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
# import scipy


class Mesure2D:
    
    def __init__(self, amplitude, position):
        if len(amplitude) != len(position):
            raise ValueError('Pas le même nombre')
        if isinstance(amplitude, list):
            self.a = np.array(amplitude)
        if isinstance(position, list):
            self.x = np.array(position)
        else:
            self.a = amplitude
            self.x = position
        self.N = len(amplitude)


    def __add__(self, m):
        '''Hieher : il faut encore régler laddition pour les mesures au même position'''
        '''ça vaudrait le coup de virer les duplicats'''
        return Mesure2D(self.a + m.a, self.x + m.x)


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
        amplitudes = np.round(self.a, 2)
        positions = np.round(self.x, 2)
        return(f"{self.N} molécules d'amplitudes : {amplitudes} --- Positions : {positions}")


    def kernel(self, X_domain, Y_domain, noyau='gaussien'):
        '''Applique un noyau à la mesure discrète. Exemple : convol'''
        N = self.N
        x = self.x
        a = self.a
        acquis = X_domain*0
        if noyau == 'gaussien':
            for i in range(0,N):
                D = np.sqrt(np.power(X_domain - x[i,1],2) + np.power(Y_domain - x[i,1],2))
                acquis += a[i]*gaussienne(D)
            return acquis
        elif noyau == 'double_gaussien':
            for i in range(0,N):
                D = np.sqrt(np.power(X_domain - x[i,1],2) + np.power(Y_domain - x[i,1],2))
                acquis += a[i]*double_gaussienne(D)
            return acquis
        else:
            return acquis


    def acquisition(self, X_domain, Y_domain, N_ech, nv):
        w = nv*np.random.random_sample((N_ech, N_ech))
        acquis = self.kernel(X_domain, Y_domain, noyau='gaussien') + w
        return acquis


    def graphe(self, X_domain, Y_domain, lvl=50):
        f = plt.figure()
        plt.contourf(X_domain, Y_domain, self.kernel(X_domain, Y_domain), 
                     lvl, label='$y_0$', cmap='hot')
        plt.xlabel('X', fontsize=18)
        plt.ylabel('Y', fontsize=18)
        plt.title('Acquisition y', fontsize=18)
        plt.colorbar();
        plt.grid()
        return f


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


    def energie(X_domain, Y_domain, y, regul, self):
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
    x = (np.round(np.random.rand(1,N), 2)).tolist()[0]
    a = (np.round(np.random.rand(1,N), 2)).tolist()[0]
    return Mesure2D(a, x)

# m = Mesure([1,1.1,0,1.5],[2,88,3,8])
# print(m.prune())

def phi(m, domain):
    return m.kernel(domain)

def phi_vecteur(a, x, domain):
    '''shape est un entier indiquant le taille à viser avec le 
        padding'''
    m_tmp = Mesure2D(a, x)
    return(m_tmp.kernel(domain))



def phiAdjoint(y, domain, noyau='gaussien'):
    taille_y = np.size(y)
    eta = np.empty(np.size(y))
    for i in range(taille_y):
        x = domain[i]
        eta[i] = integrate.simps(y*gaussienne(x-domain),x=domain)
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


def etak(mesure, X, y, regul):
    eta = 1/regul*phiAdjoint(y - phi(mesure, X), X)
    return eta

def gaussienne(domain, sigma=1e-1):
    '''Gaussienne centrée en 0'''
    return np.sqrt(2*np.pi*sigma**2)*np.exp(-np.power(domain,2)/(2*sigma**2))


def double_gaussienne(domain):
    '''Gaussienne au carré centrée en 0'''
    return np.power(gaussienne(domain),2)



N_ech = 50
xgauche = 0
xdroit = 1
X_grid = np.linspace(xgauche, xdroit, N_ech)
X, Y = np.meshgrid(X_grid, X_grid)
# D = np.sqrt(np.power(X, 2) + np.power(Y, 2)) 

# plt.contourf(X, Y, gaussienne(D))
# plt.colorbar();

m = Mesure2D([2,1],[[0.25,0.25],[0.75,0.75]])
m.graphe(X, Y)
# y_2d = m.kernel(X, Y)
# plt.figure()
# plt.contourf(X, Y, y_2d, 50, cmap='hot')
# plt.colorbar();

