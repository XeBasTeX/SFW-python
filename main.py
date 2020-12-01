#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 11:32:17 2020

@author: Bastien (https://github.com/XeBasTeX)

Cas 1D sans bruit pour SFW
"""


__author__ = 'Bastien'


import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from scipy import integrate
import scipy
# from sklearn import linear_model


np.random.seed(2)
N_ech = 100
xgauche = 0
xdroit = 1
X = np.linspace(xgauche, xdroit, N_ech)

sigma = 1e-1
regul = 1e-5
niveaubruits = 1e-5


def gaussienne(X):
    '''Gaussienne centrée en 0'''
    return np.sqrt(2*np.pi*sigma**2)*np.exp(-np.power(X,2)/(2*sigma**2))


class Mesure:
    def __init__(self, amplitude, position):
        if len(amplitude) != len(position):
            raise ValueError('Pas le même nombre')
        self.a = amplitude
        self.x = position
        self.N = len(amplitude)


    def __add__(self, m):
        '''ça vaudrait le coup de virer les duplicats'''
        return Mesure(self.a + m.a, self.x + m.x)


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
        return(f'Amplitude : {amplitudes} --- Position : {positions}')


    def graphe(self):
        plt.figure()
        plt.plot(X,y, label='$y_0$')
        plt.stem(self.x, self.a, label='$m_{a,x}$', linefmt='C1--', 
                  markerfmt='C1o', use_line_collection=True, basefmt=" ")
        plt.grid()
        return


    def kernel(self, X, noyau='gaussien'):
        '''Applique un noyau à la mesure discrète. Exemple : convol'''
        N = self.N
        x = self.x
        a = self.a
        acquis = X*0
        if noyau == 'gaussien':
            for i in range(0,N):
                acquis += a[i]*gaussienne(X - x[i])
        return acquis


    def acquisition(self, nv):
        w = nv*np.random.random_sample((N_ech))
        acquis = self.kernel(X, noyau='gaussien') + w
        return acquis


    def tv(self):
        return np.linalg.norm(self.a, 1)


    def energie(X, y, regul, self):
        attache = 0.5*np.linalg.norm(y - self.kernel(X))
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
        m = Mesure(nnz_a.tolist(), nnz_x.tolist())
        return m


# def mesureAleatoire(N):
#     x = np.round(np.random.rand(1,N), 2)
#     a = np.round(np.random.rand(1,N), 2)
#     return Mesure(a, x)

# m = Mesure([1,1.1,0,1.5],[2,88,3,8])
# print(m.prune())

def phi(m, X):
    return m.kernel(X)


def phi_vecteur(a, x, shape=0):
    '''shape est un entier indiquant le taille à viser avec le 
        padding'''
    acquis = 0
    # HIERHER il faut débuguer, quand c'est un seul élément ça merde
    # if len(a) != len(x):
    #     print('a = ' + str(a))
    #     print('x = ' + str(x))
    #     raise ValueError('Pas le même nombre de Dirac')
    acquis = a*gaussienne(x)
    if shape > 0:
        if shape % 2 == 0:
            if len(a) % 2 == 0:
                shape_a = int((shape - len(a))/2)
                observ = np.pad(acquis, (shape_a, shape_a), 'constant')
            else:
                shape_a = int((shape - len(a))/2)
                observ = np.pad(acquis, (shape_a+1, shape_a), 'constant')
        else:
            raise ValueError('Choisis mieux ton échantillonnage')
        return observ
    else:
        return acquis


def phi_vecteur_2(a, x, domain):
    '''shape est un entier indiquant le taille à viser avec le 
        padding'''
    m_tmp = Mesure(a, x)
    return(m_tmp.kernel(X))



def phiAdjoint(y, X, noyau='gaussien'):
    eta = np.empty(N_ech)
    for i in range(N_ech):
        x = X[i]
        eta[i] = integrate.simps(y*gaussienne(x-X),x=X)
    return eta


def etaW(x, N, sigma, noyau='gaussien'):
    '''Certificat \eta_W dans le cas gaussien'''
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


def etak(mesure, y, regul):
    eta = 1/regul*phiAdjoint(y - phi(mesure, X), X)
    return eta


# m_ax0 = Mesure([1.3,0.8,1.4], [0.3,0.37,0.7])
m_ax0 = Mesure([1.3,0.8,1.4], [0.3,0.37,0.7])
y = m_ax0.acquisition(niveaubruits)

# m_ax = Mesure([1.31,0.79,1.39], [0.3,0.37,0.7])
# certificat_V = etak(m_ax, y, regul)
# certificat_W_x0 = etaWx0(X, 0.5, m_ax, sigma)


# GRAPHIQUES

# # plt.plot(X, certificat_W_x0)
# plt.figure('Dirac')
# plt.plot(X,y, label='$y_0$')
# plt.stem(m_ax0.x, m_ax0.a, label='$m_{a_0,x_0}$', linefmt='C1--', 
#          markerfmt='C1o', use_line_collection=True, basefmt=" ")
# # for i in range(m_ax0.N):
# #     plt.vlines(m_ax0.x[i], 0, m_ax0.a[i], colors='orange', linestyles='dashed')


# plt.xlabel('$x$', fontsize=18)
# plt.ylabel('Luminosité', fontsize=18)
# plt.title('$m_{a_0,x_0}$ et son acquisition $y_0$', fontsize=20)
# plt.legend()
# # plt.grid()    
# # plt.savefig('fig/dirac.pdf', format='pdf', dpi=1000,
# #         bbox_inches='tight', pad_inches=0.03)
# plt.grid()
# plt.show()



# plt.figure('Certificat')
# # plt.scatter(m_ax.x, m_ax.a, color='orange', label='$m_{a,x}$')
# # for i in range(m_ax.N):
# #     plt.vlines(m_ax.x[i], 0, m_ax.a[i], colors='orange', linestyles='dashed')
# plt.plot(X, certificat_V, 'r', label='$\eta_V$')
# plt.stem(m_ax.x, m_ax.a, label='$m_{a,x}$', linefmt='C1:', 
#          markerfmt='C1o', use_line_collection=True, basefmt=" ")
# plt.axhline(y=1, color='gray', linestyle='--')
# plt.axhline(y=-1, color='gray', linestyle='--')

# plt.title('$m_{{a,x}}$ et certificat $\eta_V$',
#           fontsize=20)
# plt.xlabel('$x$', fontsize=18)
# plt.ylabel(f'$\eta_V$ à $\lambda=${regul:.1e}', fontsize=18)
# plt.grid()
# plt.legend(loc=4)



def SFW(y, regul=1e-5, nIter=5):
    '''y acquisition et nIter nombre d'itérations'''
    N_ech = len(y)
    a_k = []
    x_k = []
    mesure_k = Mesure(a_k, x_k)    # Msure discrète vide
    Nk = 0                      # Taille de la mesure discrète
    nrj_vecteur = np.zeros(nIter)
    for k in range(nIter):
        print('\n' + 'Etape numéro ' + str(k))
        eta_V_k = etak(mesure_k, y, regul)
        
        x_star_index = np.argmax(np.abs(eta_V_k))
        x_star = x_star_index/N_ech
        print('* x^* = ' + str(x_star) + ' max à ' + str(np.round(eta_V_k[x_star_index], 2)))
        
        # Condition d'arrêt (attention x_star est un indice)
        if np.abs(eta_V_k[x_star_index]) < 1:
            print("\n\n---- Condition d'arrêt ----")
            return(mesure_k, nrj_vecteur)
        else:
            mesure_k_demi = Mesure([],[])
            x_k_demi = x_k + [x_star]
            
            # On résout LASSO (étape 7)
            def lasso(a):
                attache = 0.5*np.linalg.norm(y - phi_vecteur_2(a,x_k_demi,len(y)))
                parcimonie = regul*np.linalg.norm(a, 1)
                return(attache + parcimonie)
            # lasso = lambda a : 0.5*np.linalg.norm(y - phi_vecteur(a,x_k_demi,len(y))) + regul*np.linalg.norm(a, 1)
            res = scipy.optimize.minimize(lasso, np.ones(Nk+1)) # renvoie un array (et pas une liste)
            a_k_demi = res.x.tolist()
            print('* a_k_demi : ' + str(np.round(a_k_demi, 2)))
            mesure_k_demi += Mesure(a_k_demi,x_k_demi)
            print('* Mesure_k_demi : ' +  str(mesure_k_demi))
            # lasso.fit(phi_vecteur()) 
            # mesure_k_demi += Mesure([a_k_demi],[mesure_k[i]])
            
            # On résout double LASSO (étape 8)
            def lasso_double(params):
                a = params[:int(len(params)/2)]
                x = params[int(len(params)/2):]
                attache = 0.5*np.linalg.norm(y - phi_vecteur_2(a,x,len(y)))
                parcimonie = regul*np.linalg.norm(a, 1)
                return(attache + parcimonie)

            initial_guess = a_k_demi + x_k_demi
            res = scipy.optimize.minimize(lasso_double, initial_guess,
                                          method='BFGS',
                                          options={'disp': True})
            a_k_plus = (res.x[:int(len(res.x)/2)]).tolist()
            x_k_plus = (res.x[int(len(res.x)/2):]).tolist()

            # Mise à jour des paramètres avec retrait des Dirac nuls
            mesure_k = Mesure(a_k_plus, x_k_plus)
            mesure_k = mesure_k.prune()
            a_k = mesure_k.a
            x_k = mesure_k.x
            Nk = mesure_k.N

            # Graphe et énergie
            # mesure_k.graphe()
            nrj_vecteur[k] = 0.5*np.linalg.norm(y - mesure_k.kernel(X)) + regul*mesure_k.tv()
            print(f'* Energie : {nrj_vecteur[k]:.3f}')
            
    return(mesure_k, nrj_vecteur)


(m_sfw, nrj_sfw) = SFW(y, regul=1e-4, nIter=5)
print(m_sfw)
certificat_V = etak(m_sfw, y, regul)

if m_sfw != 0:
    plt.figure(figsize=(21,4))
    plt.subplot(131)
    plt.plot(X,y, label='$y_0$')
    plt.stem(m_sfw.x, m_sfw.a, label='$m_{a,x}$', linefmt='C1--', 
              markerfmt='C1o', use_line_collection=True, basefmt=" ")
    plt.stem(m_ax0.x, m_ax0.a, label='$m_{a_0,x_0}$', linefmt='C2--', 
              markerfmt='C2o', use_line_collection=True, basefmt=" ")
    plt.xlabel('$x$', fontsize=18)
    plt.ylabel(f'Lumino à $\lambda=${regul:.1e}', fontsize=18)
    plt.title('$m_{a,x}$ contre la mesure VT', fontsize=20)
    plt.grid()
    plt.legend()
    
    plt.subplot(132)
    plt.plot(X, certificat_V,'r')
    plt.axhline(y=1, color='gray', linestyle='--')
    plt.axhline(y=-1, color='gray', linestyle='--')
    plt.xlabel('$x$', fontsize=18)
    plt.ylabel(f'Lumino à $\lambda=${regul:.1e}', fontsize=18)
    plt.title('Certificat $\eta$ mesure finale', fontsize=20)
    plt.grid()
    
    plt.subplot(133)
    plt.plot(nrj_sfw, 'o--')
    plt.xlabel('Itération', fontsize=18)
    plt.ylabel('$\mathcal{J}_\lambda$(m)', fontsize=20)
    plt.title('Décroissance énergie', fontsize=20)
    plt.grid()


# #%%
# N=10
# M=100

# U = np.random.random((M,N))
# m = np.random.random(M)
# t = cp.Variable(M)
# x = cp.Variable(N)

# prob = cp.Problem(cp.Minimize(cp.sum(t)), [-t<=U@x-m, U@x-m<=t])
# optimal_value = prob.solve()
# print("t=",t.value)
# print("x=",x.value)
# print("val=",optimal_value)