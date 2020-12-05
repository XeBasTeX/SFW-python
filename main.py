#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 11:32:17 2020

@author: Bastien Laville (https://github.com/XeBasTeX)

Cas 1D pour SFW
"""


__author__ = 'Bastien'
__team__ = 'Morpheme'
__saveFig__ = True
__deboggage__ = True


import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import scipy
# from mpl_toolkits.mplot3d import Axes3D
# import cvxpy as cp
# from sklearn import linear_model


# np.random.seed(0)
N_ech = 100 # Taux d'échantillonnage
xgauche = 0
xdroit = 1
X = np.linspace(xgauche, xdroit, N_ech)

sigma = 1e-1 # écart-type de la PSF
lambda_regul = 8e-3 # Param de relaxation
niveaubruits = 1e-2 # sigma du bruit


def gaussienne(domain):
    '''Gaussienne centrée en 0'''
    return np.sqrt(2*np.pi*sigma**2)*np.exp(-np.power(domain,2)/(2*sigma**2))


def double_gaussienne(domain):
    '''Gaussienne au carré centrée en 0'''
    return np.power(gaussienne(domain),2)


def ideal_lowpass(domain, fc):
    '''Passe-bas idéal de fréquence de coupure f_c'''
    return np.sin((2*fc + 1)*np.pi*domain)/np.sin(np.pi*domain)


class Mesure:
    def __init__(self, amplitude, position, typeDomaine='segment'):
        if len(amplitude) != len(position):
            raise ValueError('Pas le même nombre')
        self.a = amplitude
        self.x = position
        self.N = len(amplitude)
        self.Xtype = typeDomaine


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


    def torus(self, current_fig=False, subplot=False):
        if subplot == True:
            ax = fig.add_subplot(224, projection='3d')
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
        elif noyau == 'double_gaussien':
            for i in range(0,N):
                acquis += a[i]*double_gaussienne(X - x[i])
            return acquis
        else:
            return acquis


    def acquisition(self, nv):
        '''Opérateur d'acquistions ainsi que bruiateg blanc gaussien
            de DSP nv (pour niveau).'''
        w = nv*np.random.random_sample((N_ech))
        acquis = self.kernel(X, noyau='gaussien') + w
        return acquis


    def tv(self):
        '''Norme TV de la mesure'''
        return np.linalg.norm(self.a, 1)


    def energie(self, X, y, regul):
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


    def deltaSeparation(self):
        '''Donne la plus petite distance entre Diracs de la mesure considérée
        selon l'option (tore ou segment)'''
        diff = np.inf
        if self.Xtype == 'segment':
            for i in range(self.N-1): 
                for j in range(i+1,self.N): 
                    if abs(self.x[i]-self.x[j]) < diff: 
                        diff = abs(self.x[i] - self.x[j]) 
            return diff
        elif self.xtype == 'tore':
            print('à def')
        else:
            raise TypeError


def mesureAleatoire(N):
    x = (np.round(np.random.rand(1,N), 2)).tolist()[0]
    a = (np.round(np.random.rand(1,N), 2)).tolist()[0]
    return Mesure(a, x)

# m = Mesure([1,1.1,0,1.5],[2,88,3,8])
# print(m.prune())

def phi(m, domain):
    return m.kernel(domain)

def phi_vecteur(a, x, domain):
    '''shape est un entier indiquant le taille à viser avec le 
        padding'''
    m_tmp = Mesure(a, x)
    return(m_tmp.kernel(domain))



def phiAdjoint(y, domain, noyau='gaussien'):
    eta = np.empty(N_ech)
    for i in range(N_ech):
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
    tmp = np.zeros(x.size)
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

# certificat_V = etak(m_ax, y, regul)
certificat_W_x0 = etaWx0(X, 0.5, m_ax0, sigma)
plt.plot(X, certificat_W_x0)
plt.grid()




#%

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
    N_ech_y = len(y)
    a_k = []
    x_k = []
    mesure_k = Mesure(a_k, x_k)    # Msure discrète vide
    Nk = 0                      # Taille de la mesure discrète
    nrj_vecteur = np.zeros(nIter)
    for k in range(nIter):
        print('\n' + 'Etape numéro ' + str(k))
        eta_V_k = etak(mesure_k, y, regul)
        x_star_index = np.argmax(np.abs(eta_V_k))
        x_star = x_star_index/N_ech_y
        print(f'* x^* = {x_star} max à {np.round(eta_V_k[x_star_index], 2)}')
        
        # Condition d'arrêt (étape 4)
        if np.abs(eta_V_k[x_star_index]) < 1:
            nrj_vecteur[k] = mesure_k.energie(X, y, regul)
            print(f'* Energie : {nrj_vecteur[k]:.3f}')
            print("\n\n---- Condition d'arrêt ----")
            return(mesure_k, nrj_vecteur)
        else:
            mesure_k_demi = Mesure([],[])
            x_k_demi = x_k + [x_star]

            # On résout LASSO (étape 7)
            def lasso(a):
                attache = 0.5*np.linalg.norm(y - phi_vecteur(a,x_k_demi,X))
                parcimonie = regul*np.linalg.norm(a, 1)
                return(attache + parcimonie)
            # lasso = lambda a : 0.5*np.linalg.norm(y - phi_vecteur(a,x_k_demi,len(y))) + regul*np.linalg.norm(a, 1)
            res = scipy.optimize.minimize(lasso, np.ones(Nk+1)) # renvoie un array (et pas une liste)
            a_k_demi = res.x.tolist()
            print('* a_k_demi : ' + str(np.round(a_k_demi, 2)))
            mesure_k_demi += Mesure(a_k_demi,x_k_demi)
            print('* Mesure_k_demi : ' +  str(mesure_k_demi))

            # On résout double LASSO non-convexe (étape 8)
            def lasso_double(params):
                a = params[:int(len(params)/2)]
                x = params[int(len(params)/2):]
                attache = 0.5*np.linalg.norm(y - phi_vecteur(a,x,X))
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
            
    print("\n\n---- Fin de la boucle ----")
    return(mesure_k, nrj_vecteur)



if __name__ == '__main__':
    (m_sfw, nrj_sfw) = SFW(y, regul=lambda_regul, nIter=5)
    print('On a retrouvé m_ax ' + str(m_sfw))
    certificat_V = etak(m_sfw, y, lambda_regul)
    print('On voulait retrouver m_ax0 ' + str(m_ax0))
    
    if m_sfw != 0:
        # plt.figure(figsize=(21,4))
        fig = plt.figure(figsize=(15,12))
        plt.subplot(221)
        plt.plot(X,y, label='$y$', linewidth=1.7)
        plt.stem(m_sfw.x, m_sfw.a, label='$m_{a,x}$', linefmt='C1--', 
                  markerfmt='C1o', use_line_collection=True, basefmt=" ")
        plt.stem(m_ax0.x, m_ax0.a, label='$m_{a_0,x_0}$', linefmt='C2--', 
                  markerfmt='C2o', use_line_collection=True, basefmt=" ")
        # plt.xlabel('$x$', fontsize=18)
        plt.ylabel(f'Lumino à $\sigma_B=${niveaubruits:.1e}', fontsize=18)
        plt.title('$m_{a,x}$ contre la VT $m_{a_0,x_0}$', fontsize=20)
        plt.grid()
        plt.legend()
        
        plt.subplot(222)
        plt.plot(X, certificat_V, 'r', linewidth=2)
        plt.axhline(y=1, color='gray', linestyle='--', linewidth=2.5)
        plt.axhline(y=-1, color='gray', linestyle='--', linewidth=2.5)
        # plt.xlabel('$x$', fontsize=18)
        plt.ylabel(f'Amplitude à $\lambda=${lambda_regul:.1e}', fontsize=18)
        plt.title('Certificat $\eta_V$ de $m_{a,x}$', fontsize=20)
        plt.grid()
        
        plt.subplot(223)
        plt.plot(nrj_sfw, 'o--', color='black', linewidth=2.5)
        plt.xlabel('Itération', fontsize=18)
        plt.ylabel('$T_\lambda$(m)', fontsize=20)
        plt.title('Décroissance énergie', fontsize=20)
        plt.grid()
        
        m_sfw.torus(current_fig=fig, subplot=True)
        
        if __saveFig__ == True:
            plt.savefig('fig/dirac-certificat.pdf', format='pdf', dpi=1000,
            bbox_inches='tight', pad_inches=0.03)



# Playground

# # Tester une autre convolution
# # Conlusion du test : le np convolve et ma méthode donnent les mêmes résultats
# # le np convolve est un poil plus rapide, mais il doit faire appel à find_x_on_grid 
# # du reste je ne suis pas convaincu que ça soit la meilleure méthode
# # ça oblige à projeter la position du dirac sur la grille.
# # Avec ma méthode, on reste dans le continu.


# m_test = Mesure([1],[0.5])
# X_dble = np.linspace(xgauche-xdroit, xdroit, N_ech)/2

# def find_x_on_grid(domain, m):
#     positions = m.x
#     amplitudes = m.a
#     x_on_grid = np.zeros(len(domain))
#     for j in range(len(positions)):
#         idx = min(range(len(domain)), key=lambda i: abs(domain[i]-positions[j]))
#         x_on_grid[idx] = amplitudes[j]
#     return np.array(x_on_grid)

# positions_on_grid = find_x_on_grid(X, m_test)
# # y_test = np.abs(np.fft.ifft(np.fft.fft(positions_on_grid)*np.fft.fft(gaussienne(X))))
# np_test = np.convolve(positions_on_grid, gaussienne(X_dble), mode="same")

# plt.figure()
# plt.plot(gaussienne(X_dble), '--')
# plt.plot(np_test, 'o')
# plt.plot(m_test.kernel(X), '^')
# print('Erreur L^2 : ' + str(np.linalg.norm(np_test - m_test.kernel(X))))

