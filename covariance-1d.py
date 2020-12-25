#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 21:57:58 2020

@author: Bastien (https://github.com/XeBasTeX)
"""

__author__ = 'Bastien'
__team__ = 'Morpheme'
__saveFig__ = False
__deboggage__ = True


import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.stats import wasserstein_distance
import scipy



np.random.seed(0)
N_ech = 2**6 # Taux d'échantillonnage
xgauche = 0
xdroit = 1
X = np.linspace(xgauche, xdroit, N_ech)
X_u, X_v = np.meshgrid(X, X)

sigma = 1e-1 # écart-type de la PSF
lambda_regul = 2.5e-5 # Param de relaxation pour SFW R_y
lambda_regul2 = 2e-3 # Param de relaxation pour SFW y_moy
niveaubruits = 1e-1 # sigma du bruit


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
        plt.plot(X, self.kernel(X), label='$y_0$')
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


    def energie(self, X, acquis, regul, obj='covar'):
        if obj == 'covar':
            attache = 0.5*np.linalg.norm(acquis - self.covariance_kernel(X,X,X))
            parcimonie = regul*self.tv()
            return(attache + parcimonie)
        elif obj == 'acquis':
            attache = 0.5*np.linalg.norm(acquis - self.kernel(X))
            parcimonie = regul*self.tv()
            return(attache + parcimonie)
        else:
            raise TypeError


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


def torus(ax, m, titre, couleur):
        theta = np.linspace(0, 2*np.pi, N_ech)
        y_torus = np.sin(theta)
        x_torus = np.cos(theta)

        a_x_torus = np.sin(2*np.pi*np.array(m.x))
        a_y_torus = np.cos(2*np.pi*np.array(m.x))
        a_z_torus = m.a

        # ax.plot((0,0),(0,0), (0,1), '-k', label='z-axis')
        ax.plot(x_torus,y_torus, 0, '-k', label='$\mathbb{S}_1$')
        ax.plot(a_x_torus,a_y_torus, a_z_torus,
                'o', label=titre, color=couleur)

        for i in range(m.N):
            ax.plot((a_x_torus[i],a_x_torus[i]),(a_y_torus[i],a_y_torus[i]),
                    (0,a_z_torus[i]), '--r', color=couleur)

        ax.set_xlabel('$X$')
        ax.set_ylabel('$Y$')
        ax.set_zlabel('$Amplitude$')
        ax.set_title('Mesure sur $\mathbb{S}_1$', fontsize=20)
        ax.legend()
        return 


def phi(m, domain, obj='covar'):
    '''Opérateur \Lambda de somulation de la covariance'''
    if obj == 'covar':
        return m.covariance_kernel(domain, domain, domain)
    elif obj == 'acquis':
        return m.kernel(domain)
    else:
        raise TypeError


def phi_vecteur(a, x, domain, obj='covar'):
    '''shape est un entier indiquant le taille à viser avec le 
        padding'''
    if obj == 'covar':
        m_tmp = Mesure(a, x)
        return(m_tmp.covariance_kernel(domain, domain, domain))
    elif obj == 'acquis':
        m_tmp = Mesure(a, x)
        return(m_tmp.kernel(domain))
    else:
        raise TypeError


def phiAdjoint(cov_acquis, domain, obj='covar'):
    taille_y = len(cov_acquis)
    if obj == 'covar':
        eta = np.empty(taille_y)
        for i in range(taille_y):
            x_decal = domain[i]
            integ_x = integrate.simps(cov_acquis*gaussienne(x_decal - domain),
                                      x=domain)
            eta[i] = integrate.simps(integ_x, x=domain) # Deux intégrations car eta 
                                                    # \in C(X,R) 
        return eta
    if obj == 'acquis':
        eta = np.empty(taille_y)
        for i in range(taille_y):
            x = domain[i]
            eta[i] = integrate.simps(cov_acquis*gaussienne(x-domain),x=domain)
        return eta
    else:
        raise TypeError


def etak(mesure, cov_acquis, regul, obj='covar'):
    eta = 1/regul*phiAdjoint(cov_acquis - phi(mesure, X, obj),
                             X, obj)
    return eta


def pile_aquisition(m):
    '''Construit une pile d'acquisition à partir d'une mesure.'''
    N_mol = len(m.a)
    acquis_temporelle = np.zeros((T_ech, N_ech))
    for t in range(T_ech):
        a_tmp = (np.random.random(N_mol))*m.a
        m_tmp = Mesure(a_tmp, m.x)
        acquis_temporelle[t,:] = m_tmp.acquisition(niveaubruits)
    return(acquis_temporelle)


def covariance_pile(stack, stack_mean):
    covar = np.zeros((len(stack[0]), len(stack[0])))
    for i in range(len(stack)):
        covar += np.outer(stack[i,:] - stack_mean, stack[i,:] - stack_mean)
    return covar/len(stack-1)   # T-1 ou T hierher ?


# m_ax0 = Mesure([1.3,0.8,1.4], [0.3,0.37,0.7])
m_ax0 = Mesure([1.3,0.8,1.4], [0.2,0.4,0.7])
# y = m_ax0.acquisition(niveaubruits)

T_ech = 500        # Il faut mettre vraiment bcp d'échantillons !
T_acquis = 1
T = np.linspace(0, T_acquis, T_ech)

pile = pile_aquisition(m_ax0)   
pile_moy = np.mean(pile, axis=0) 
y = pile_moy  
R_y = covariance_pile(pile, pile_moy)
R_x = m_ax0.covariance_kernel(X,X,X)


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
    plt.savefig('fig/R_x-R_y-1d.pdf', format='pdf', dpi=1000,
                bbox_inches='tight', pad_inches=0.03)



def SFW(acquis, regul=1e-5, nIter=5, mesParIter=False, obj='covar'):
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
        eta_V_k = etak(mesure_k, acquis, regul, obj)
        x_star_index = np.unravel_index(np.argmax(np.abs(eta_V_k), axis=None),
                                        eta_V_k.shape)
        x_star = np.array(x_star_index)/N_ech_y # hierher passer de l'idx à xstar
        print(f'* x^* index {x_star} max à {np.round(eta_V_k[x_star_index], 2)}')
        
        # Condition d'arrêt (étape 4)
        if np.abs(eta_V_k[x_star_index]) < 1:
            nrj_vecteur[k:] = mesure_k.energie(X, acquis, regul, obj)
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
                attache = 0.5*np.linalg.norm(acquis - phi_vecteur(a, x_k_demi, 
                                                                  X, obj))
                parcimonie = regul*np.linalg.norm(a, 1)
                return(attache + parcimonie)
            res = scipy.optimize.minimize(lasso, np.ones(Nk+1))
            a_k_demi = res.x
            print('* a_k_demi : ' + str(np.round(a_k_demi, 2)))
            mesure_k_demi += Mesure(a_k_demi, x_k_demi)
            print('* Mesure_k_demi : ' +  str(mesure_k_demi))

            # On résout double LASSO non-convexe (étape 8)
            def lasso_double(params):
                a_p = params[:int(np.ceil(len(params)/2))] # Bout de code immonde, à corriger !
                x_p = params[int(np.ceil(len(params)/2)):]
                attache = 0.5*np.linalg.norm(acquis - phi_vecteur(a_p, x_p,
                                                                  X, obj))
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
            nrj_vecteur[k] = mesure_k.energie(X, acquis, regul, obj)
            print(f'* Energie : {nrj_vecteur[k]:.3f}')
            if mesParIter == True:
                mes_vecteur = np.append(mes_vecteur, [mesure_k])
            
    print("\n\n---- Fin de la boucle ----")
    if mesParIter == True:
        return(mesure_k, nrj_vecteur, mes_vecteur)
    else:
        return(mesure_k, nrj_vecteur)



if __name__ == '__main__' and True==1:
    (m_sfw, nrj_sfw) = SFW(R_y, regul=lambda_regul, nIter=5)
    print('On a retrouvé m_ax, ' + str(m_sfw))
    certificat_V = etak(m_sfw, R_y, lambda_regul)
    print('On voulait retrouver m_ax0, ' + str(m_ax0))
    (m_moy, nrj_moy) = SFW(y, regul=lambda_regul2, nIter=5, obj='acquis')
    print('On a retrouvé m_ax, ' + str(m_moy))
    certificat_V_moy = etak(m_moy, y, lambda_regul2, obj='acquis')
    print('On voulait retrouver m_ax0, ' + str(m_ax0))
    
    if m_sfw.a.size > 0:
        wasser = wasserstein_distance(m_sfw.x, m_ax0.x, m_sfw.a, m_ax0.a)
        print(f'2-distance de Wasserstein : W_2(m_sfw,m_ax0) = {wasser}')
    
        # plt.figure(figsize=(21,4))
        fig = plt.figure(figsize=(15,12))
        
        plt.subplot(221)
        plt.plot(X, pile_moy, label='$\overline{y}$', linewidth=1.7)
        plt.stem(m_sfw.x, m_sfw.a, label='$m_{a,x}$', linefmt='C3--', 
                  markerfmt='C3o', use_line_collection=True, basefmt=" ")
        plt.stem(m_ax0.x, m_ax0.a, label='$m_{a_0,x_0}$', linefmt='C2--', 
                  markerfmt='C2o', use_line_collection=True, basefmt=" ")
        # plt.xlabel('$x$', fontsize=18)
        plt.ylabel(f'Lumino à $\sigma_B=${niveaubruits:.1e}', fontsize=18)
        plt.title('$m_{M,x}$ contre la VT $m_{a_0,x_0}$', fontsize=20)
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
        plt.ylabel('$\mathcal{J}_\lambda(m)$', fontsize=20)
        plt.title('Décroissance énergie', fontsize=20)
        plt.grid()
        
        m_sfw.torus(current_fig=fig, subplot=True)
        
        if __saveFig__ == True:
            plt.savefig('fig/covar-moy-certificat-1d.pdf', format='pdf', 
                        dpi=1000, bbox_inches='tight', pad_inches=0.03)
    
    if m_moy.a.size > 0:
        wasser = wasserstein_distance(m_moy.x, m_ax0.x, m_moy.a, m_ax0.a)
        print(f'2-distance de Wasserstein : W_2(m_moy,m_ax0) = {wasser}')
    
        fig = plt.figure(figsize=(15,12))
        
        plt.subplot(221)
        plt.plot(X, pile_moy, label='$\overline{y}$', linewidth=1.7)
        plt.stem(m_moy.x, m_moy.a, label='$m_{a,x}$', linefmt='C1--', 
                  markerfmt='C1o', use_line_collection=True, basefmt=" ")
        plt.stem(m_ax0.x, m_ax0.a, label='$m_{a_0,x_0}$', linefmt='C2--', 
                  markerfmt='C2o', use_line_collection=True, basefmt=" ")
        plt.stem(m_sfw.x, m_sfw.a, label='$m_{M,x}$', linefmt='C1--', 
                  markerfmt='C3o', use_line_collection=True, basefmt=" ")
        # plt.xlabel('$x$', fontsize=18)
        plt.ylabel(f'Lumino à $\sigma_B=${niveaubruits:.1e}', fontsize=18)
        plt.title('$m_{a,x}$ et $m_{M,x}$ contre la VT $m_{a_0,x_0}$',
                  fontsize=20)
        plt.grid()
        plt.legend()
        
        plt.subplot(222)
        plt.plot(X, certificat_V_moy, 'r', linewidth=2)
        plt.axhline(y=1, color='gray', linestyle='--', linewidth=2.5)
        plt.axhline(y=-1, color='gray', linestyle='--', linewidth=2.5)
        # plt.xlabel('$x$', fontsize=18)
        plt.ylabel(f'Amplitude à $\lambda=${lambda_regul:.1e}', fontsize=18)
        plt.title('Certificat $\eta_V$ de $m_{a,x}$', fontsize=20)
        plt.grid()
        
        plt.subplot(223)
        plt.plot(nrj_moy, 'o--', color='black', linewidth=2.5)
        plt.xlabel('Itération', fontsize=18)
        plt.ylabel('$T_\lambda(m)$', fontsize=20)
        plt.title('Décroissance énergie', fontsize=20)
        plt.grid()
        
        ax = fig.add_subplot(224, projection='3d')
        torus(ax, m_sfw, '$m_{a,x}$', 'red')
        torus(ax, m_moy, '$m_{M,x}$', 'orange')
        
        if __saveFig__ == True:
            plt.savefig('fig/covar-moy-sfw-certificat-1d.pdf', format='pdf', 
                        dpi=1000, bbox_inches='tight', pad_inches=0.03)


