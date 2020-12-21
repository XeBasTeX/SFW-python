#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 21:57:58 2020

@author: Bastien (https://github.com/XeBasTeX)
"""

__author__ = 'Bastien'
__team__ = 'Morpheme'
__saveFig__ = True
__deboggage__ = True


import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.stats import wasserstein_distance
import scipy


# T_ech = 60
# T_acquis = 30
# T = np.linspace(0, T_acquis, T_ech)
# N_mol = 15
# acquis_temporelle = np.zeros((N_mol, T_ech))
# for i in range(N_mol):
#     phase = np.pi*np.random.rand()
#     acquis_temporelle[i,] = np.sin(T + phase)
#     plt.plot(acquis_temporelle[i,:])

# positions_mesures = np.random.rand(1, N_mol)[0]
# amplitudes_moyennes = 0


np.random.seed(0)
N_ech = 2**6 # Taux d'échantillonnage
xgauche = 0
xdroit = 1
X = np.linspace(xgauche, xdroit, N_ech)

sigma = 1e-1 # écart-type de la PSF
lambda_regul = 1e-4 # Param de relaxation
niveaubruits = 5e-2 # sigma du bruit


def gaussienne(domain):
    '''Gaussienne centrée en 0'''
    return np.sqrt(2*np.pi*sigma**2)*np.exp(-np.power(domain,2)/(2*sigma**2))


def double_gaussienne(domain):
    '''Gaussienne au carré centrée en 0'''
    return np.power(gaussienne(domain),2)


def ideal_lowpass(domain, fc):
    '''Passe-bas idéal de fréquence de coupure f_c'''
    return np.sin((2*fc + 1)*np.pi*domain)/np.sin(np.pi*domain)


def covariance_pile(stack, stack_mean):
    covar = np.zeros((len(stack), len(stack[0])))
    for i in range(len(stack)):
        covar += np.outer(stack - stack_mean, stack - stack_mean)
    return covar


class Mesure:
    def __init__(self, amplitude, position, typeDomaine='segment'):
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
        return Mesure(a_new, x_new)


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


    def covariance_kernel(self, domain, u, v):
        N = self.N
        x = self.x
        a = self.a
        acquis = np.zeros((N_ech, N_ech))
        for i in range(0, N):
            noyau_u = gaussienne(u - x[i])
            noyau_v = gaussienne(v - x[i])
            acquis += a[i]*np.outer(noyau_u, noyau_v)
        return acquis


    def acquisition(self, nv):
        '''Opérateur d'acquistions ainsi que bruit blanc gaussien
            de DSP nv (pour niveau).'''
        w = nv*np.random.random_sample((N_ech))
        acquis = self.kernel(X, noyau='gaussien') + w
        return acquis


    def tv(self):
        '''Norme TV de la mesure. Renvoie 0 si la mesure est vide : 
            hierher à vérifier'''
        return np.linalg.norm(self.a, 1)


    def energie(self, X, r_acquis, regul):
        attache = 0.5*np.linalg.norm(r_acquis - self.covariance_kernel(X,X,X))
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
    x = np.round(np.random.rand(1,N), 2)
    a = np.round(np.random.rand(1,N), 2)
    return Mesure(a, x)


def phi(m, domain):
    '''Opérateur \Lambda de somulation de la covariance'''
    return m.covariance_kernel(domain, domain, domain)


def phi_vecteur(a, x, domain):
    '''shape est un entier indiquant le taille à viser avec le 
        padding'''
    m_tmp = Mesure(a, x)
    return(m_tmp.covariance_kernel(domain, domain, domain))


def phiAdjoint(cov_acquis, u_domain, v_domain, noyau='gaussien'):
    taille_y = len(cov_acquis)
    eta = np.empty(np.shape(cov_acquis))
    for i in range(taille_y):
        for j in range(taille_y):
            x_decal = u_domain[i,j]
            y_decal = v_domain[i,j]
            D_decal = np.sqrt(np.power(x_decal-u_domain,2) + np.power(y_decal-v_domain,2))
            integ_x = integrate.simps(cov_acquis*gaussienne(D_decal), x=X)
            eta[i,j] = integrate.simps(integ_x, x=X)
    return eta

def etak(mesure, cov_acquis, regul):
    eta = 1/regul*phiAdjoint(cov_acquis - phi(mesure, X), X_u, X_v)
    return eta


# m_ax0 = Mesure([1.3,0.8,1.4], [0.3,0.37,0.7])
m_ax0 = Mesure([1.3,0.8,1.4], [0.2,0.37,0.7])
y = m_ax0.acquisition(niveaubruits)

R_y = m_ax0.covariance_kernel(X,X,X)
X_u, X_v = np.meshgrid(X, X)
# opadj = phiAdjoint(R_y, X_u, X_v)


def SFW(acquis, regul=1e-5, nIter=5, mesParIter=False):
    '''y acquisition et nIter nombre d'itérations
    mesParIter est un booléen qui contrôle le renvoi du vecteur mesParIter
    un vecteur qui contient mesure_k la mesure de la k-ième itération'''
    N_ech_y = len(R_y)
    a_k = np.array([])
    x_k = np.array([])
    mesure_k = Mesure(a_k, x_k)    # Mesure discrète vide
    Nk = 0                      # Taille de la mesure discrète
    if mesParIter == True:
        mes_vecteur = np.array([])
    nrj_vecteur = np.zeros(nIter)
    for k in range(nIter):
        print('\n' + 'Étape numéro ' + str(k))
        eta_V_k = etak(mesure_k, R_y, regul)
        x_star_index = np.unravel_index(np.argmax(np.abs(eta_V_k), axis=None),
                                        eta_V_k.shape)
        x_star = np.array(x_star_index)/N_ech_y # hierher passer de l'idx à xstar
        print(f'* x^* index {x_star} max à {np.round(eta_V_k[x_star_index], 2)}')
        
        # Condition d'arrêt (étape 4)
        if np.abs(eta_V_k[x_star_index]) < 1:
            nrj_vecteur[k] = mesure_k.energie(X, R_y, regul)
            print(f'* Énergie : {nrj_vecteur[k]:.3f}')
            print("\n\n---- Condition d'arrêt ----")
            if mesParIter == True:
                return(mesure_k, nrj_vecteur, mes_vecteur)
            else:
                return(mesure_k, nrj_vecteur)
        else:
            mesure_k_demi = Mesure([], [])
            # Hierher ! c'est pas très propre !
            # On utilise la symétrie de la cov pour récup la position du Dirac
            if x_k.size == 0:
                x_k_demi = np.array([x_star[0]])
            else:
                x_k_demi = np.append(x_k, x_star[0])

            # On résout LASSO (étape 7)
            def lasso(a):
                attache = 0.5*np.linalg.norm(R_y - phi_vecteur(a,x_k_demi,X))
                parcimonie = regul*np.linalg.norm(a, 1)
                return(attache + parcimonie)
            res = scipy.optimize.minimize(lasso, np.ones(Nk+1))
            a_k_demi = res.x
            print('* a_k_demi : ' + str(np.round(a_k_demi, 2)))
            mesure_k_demi += Mesure(a_k_demi,x_k_demi)
            print('* Mesure_k_demi : ' +  str(mesure_k_demi))

            # On résout double LASSO non-convexe (étape 8)
            def lasso_double(params):
                a_p = params[:int(np.ceil(len(params)/2))] # Bout de code immonde, à corriger !
                x_p = params[int(np.ceil(len(params)/2)):]
                attache = 0.5*np.linalg.norm(R_y - phi_vecteur(a_p, x_p, X))
                parcimonie = regul*np.linalg.norm(a_p, 1)
                return(attache + parcimonie)

            # On met la graine au format array pour scipy...minimize
            # il faut en effet que ça soit un vecteur
            initial_guess = np.append(a_k_demi, np.reshape(x_k_demi, -1))
            res = scipy.optimize.minimize(lasso_double, initial_guess,
                                          method='BFGS',
                                          options={'disp': True})
            a_k_plus = res.x[:int(np.ceil(len(res.x)/2))]
            x_k_plus = res.x[int(np.ceil(len(res.x)/2)):]

            # Mise à jour des paramètres avec retrait des Dirac nuls
            mesure_k = Mesure(a_k_plus, x_k_plus)
            mesure_k = mesure_k.prune(tol=1e-2)
            a_k = mesure_k.a
            x_k = mesure_k.x
            Nk = mesure_k.N

            # Graphe et énergie
            # mesure_k.graphe()
            # attache = 0.5*np.linalg.norm(R_y - mesure_k.covariance_kernel(X, X, X))
            # nrj_vecteur[k] = attache + regul*mesure_k.tv()
            nrj_vecteur[k] = mesure_k.energie(X, R_y, regul)
            print(f'* Energie : {nrj_vecteur[k]:.3f}')
            if mesParIter == True:
                mes_vecteur = np.append(mes_vecteur, [mesure_k])
            
    print("\n\n---- Fin de la boucle ----")
    if mesParIter == True:
        return(mesure_k, nrj_vecteur, mes_vecteur)
    else:
        return(mesure_k, nrj_vecteur)



if __name__ == '__main__' and True==1:
    (m_sfw, nrj_sfw) = SFW(R_y, regul=lambda_regul, nIter=4)
    print('On a retrouvé m_ax, ' + str(m_sfw))
    certificat_V = etak(m_sfw, R_y, lambda_regul)
    print('On voulait retrouver m_ax0, ' + str(m_ax0))
    
    if m_sfw.a.size > 0:
        wasser = wasserstein_distance(m_sfw.x, m_ax0.x, m_sfw.a, m_ax0.a)
        print(f'2-distance de Wasserstein : {wasser}')
    
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
        cont = plt.contourf(X, X, certificat_V, 100, cmap='seismic')
        for c in cont.collections:
            c.set_edgecolor("face")
        plt.colorbar();
        plt.scatter(m_ax0.x, m_ax0.x, marker='x', label='True spikes')
        plt.scatter(m_sfw.x, m_sfw.x, marker='+', label='Recovered spikes')
        plt.legend(loc=2)
        # plt.xlabel('$x$', fontsize=18)
        plt.ylabel(f'Amplitude à $\lambda=${lambda_regul:.1e}', fontsize=18)
        plt.title('Certificat $\eta_V$ de $m_{a,x}$', fontsize=20)
        plt.grid()
        
        plt.subplot(223)
        plt.plot(nrj_sfw, 'o--', color='black', linewidth=2.5)
        plt.xlabel('Itération', fontsize=18)
        plt.ylabel('$T_\lambda(m)$', fontsize=20)
        plt.title('Décroissance énergie', fontsize=20)
        plt.grid()
        
        m_sfw.torus(current_fig=fig, subplot=True)
        
        if __saveFig__ == True:
            plt.savefig('fig/covar-certificat.pdf', format='pdf', dpi=1000,
            bbox_inches='tight', pad_inches=0.03)


