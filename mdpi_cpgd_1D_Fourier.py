# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 10:23:27 2021

@author: blaville
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
# from mpl_toolkits.mplot3d import Axes3D
# import cvxpy as cp
# from sklearn import linear_model


np.random.seed(90)
# N_ech = 2048*8 # Taux d'échantillonnage
N_ech = 1000
xgauche = 0
xdroit = 1
X = np.linspace(xgauche, xdroit, N_ech)
X_big = np.linspace(xgauche-xdroit, xdroit, 2*N_ech)
X_certif = np.linspace(xgauche, xdroit, N_ech+1)

sigma = 1e-1 # écart-type de la PSF
lambda_regul = 1e-1 # Param de relaxation
niveaubruits = 1e-2 # sigma du bruit

F_C = 6
pas = 0.7 / F_C;
x0 = np.array([0.5-pas,0.5,0.5+pas])


def gaussienne(domain):
    '''Gaussienne centrée en 0'''
    return np.sqrt(2*np.pi*sigma**2)*np.exp(-np.power(domain,2)/(2*sigma**2))


def grad_gaussienne(domain):
    '''Gaussienne centrée en 0'''
    return - (domain * np.sqrt(2*np.pi) / sigma *
              np.exp(-np.power(domain,2)/(2*sigma**2)))


def double_gaussienne(domain):
    '''Gaussienne au carré centrée en 0'''
    return np.power(gaussienne(domain),2)


def ideal_lowpass(domain, fc):
    '''Passe-bas idéal de fréquence de coupure f_c'''
    return np.sin((2*fc + 1)*np.pi*domain)/np.sin(np.pi*domain)


def fourier_measurements(x, fc):
    ii = np.complex(0,1)
    x = np.array(x)
    fc_vect = np.arange(-fc, fc+1)
    result = np.exp(-2* ii * np.pi * (fc_vect[:,None] @ x[:,None].T))
    return result


def grad_fourier_measurements(x, fc):
    ii = np.complex(0,1)
    result = - 2* ii * np.pi * fourier_measurements(x, fc)
    return result


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


    def kernel(self, X, noyau='gaussien'):
        '''Applique un noyau à la mesure discrète. Exemple : convol'''
        N = self.N
        x = self.x
        a = self.a
        acquis = X*0
        if noyau == 'gaussien':
            for i in range(0,N):
                acquis += a[i] * gaussienne(X - x[i])
            return acquis
        elif noyau == 'double_gaussien':
            for i in range(0,N):
                acquis += a[i] * double_gaussienne(X - x[i])
            return acquis
        elif noyau == 'gaussien_der':
            for i in range(0,N):
                acquis += a[i] * grad_gaussienne(X - x[i])
            return acquis
        elif noyau == 'fourier':
            a = np.array(a)
            acquis = fourier_measurements(x, F_C) @ a
            return acquis
        elif noyau == 'fourier_der':
            a = np.array(a)
            acquis = grad_fourier_measurements(x, F_C) @ a
            return acquis
        else:
            raise TypeError("Unknown kernel")


    def acquisition(self, nv, noyau='gaussien'):
        '''Opérateur d'acquistions ainsi que bruiateg blanc gaussien
            de DSP nv (pour niveau).'''
        acquis = self.kernel(X, noyau=noyau)
        if nv == 0:
            return acquis
        if noyau == 'fourier':
            w = np.fft.fft(np.random.randn(acquis.shape[0]))
            w = np.fft.fftshift(w)
            w /= np.linalg.norm(w)
            acquis += nv * w
            return acquis
        if noyau == 'gaussien':
            w = nv*np.random.random_sample((N_ech))
            acquis += w
            return acquis
        else:
            raise TypeError("Unknown kernel")


    def tv(self):
        '''Norme TV de la mesure'''
        return np.linalg.norm(self.a, 1)


    def energie(self, X, acquis, regul, noyau='gaussien'):
        attache = 0.5*np.linalg.norm(acquis - self.kernel(X, noyau=noyau))/len(acquis)
        parcimonie = regul*self.tv()
        return(attache + parcimonie)


    def prune(self, tol=1e-3):
        '''Retire les dirac avec zéro d'amplitude'''
        # nnz = np.count_nonzero(self.a)
        nnz_a = np.array(self.a)
        nnz = np.abs(nnz_a) > tol
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


def phi(m, domain, noyau='gaussien'):
    return m.kernel(domain, noyau=noyau)


def phi_der(m, domain, noyau='gaussien'):
    if noyau == 'gaussien':
        return m.kernel(domain, noyau='gaussien_der')
    if noyau == 'fourier':
        return m.kernel(domain, noyau='fourier_der')
    raise TypeError


def phi_vecteur(a, x, domain, noyau='gaussien'):
    '''shape est un entier indiquant le taille à viser avec le 
        padding'''
    m_tmp = Mesure(a, x)
    return(m_tmp.kernel(domain, noyau=noyau))


def phiAdjointSimps(acquis, domain, noyau='gaussien'):
    eta = np.empty(N_ech)
    if noyau == 'gaussien':
        for i in range(N_ech):
            x = domain[i]
            eta[i] = integrate.simps(acquis*gaussienne(x-domain),x=domain)
        return eta
    if noyau == 'fourier':
        return
    else:
        raise TypeError


def phiAdjoint(acquis, domain, noyau='gaussien'):
    if noyau == 'gaussien':
        return np.convolve(gaussienne(domain),y,'valid')/N_ech
    if noyau == 'fourier':
        cont_fct = fourier_measurements(domain, F_C).T @ acquis
        return np.flip(np.real(cont_fct))
    else:
        raise TypeError


def gradPhiAdjoint(acquis, domain, noyau='gaussien'):
    if noyau == 'gaussien':
        return np.convolve(grad_gaussienne(domain),y,'valid')/N_ech
    if noyau == 'fourier':
        cont_fct = grad_fourier_measurements(domain, F_C).T @ acquis
        return np.flip(np.real(cont_fct))
    else:
        raise TypeError


def etak(mesure, acquis, regul, noyau='gaussien'):
    if noyau == 'gaussien' or noyau == 'gaussien_der':
        eta = 1/regul*phiAdjoint(acquis - phi(mesure, X, noyau=noyau),
                                 X_big, noyau=noyau)
        return eta
    if noyau == 'fourier' or noyau == 'fourier_der':
        eta = 1/regul*phiAdjoint(acquis - phi(mesure, X, noyau=noyau),
                                 X, noyau=noyau)
        return eta


def grad_etak(mesure, acquis, regul, noyau='gaussien'):
    if noyau == 'gaussien':
        eta = 1/regul*phiAdjoint(acquis - phi(mesure, X, noyau=noyau),
                                 X_big, noyau=noyau)
        return eta
    if noyau == 'fourier':
        eta = 1/regul*phiAdjoint(acquis - phi(mesure, X, noyau=noyau),
                                 X, noyau=noyau)
        return eta


a0 = [1, 1, -1]
m_ax0 = Mesure(a0, x0)
niveaubruits = 0.12

noy = 'gaussien'
y0 = m_ax0.kernel(X, noyau=noy)
y = m_ax0.acquisition(niveaubruits, noyau=noy)
certes0 = phiAdjoint(y0, X, noyau=noy)
certes = phiAdjoint(y, X, noyau=noy)

if noy == 'fourier':
    plt.plot(X, certes0, label='$\Phi^* y_0$')
    plt.plot(X, certes, label='$\Phi^* y$')
    plt.stem(x0, 10*np.array(a0), basefmt=" ", use_line_collection=True,
             linefmt='k--', markerfmt='ko')
    plt.legend()
    plt.grid()
elif noy == 'gaussien':
    plt.plot(X, y0, label='$\Phi^* y_0$')
    plt.plot(X, y, label='$\Phi^* y$')
    plt.stem(x0, np.array(a0), basefmt=" ", use_line_collection=True,
             linefmt='k--', markerfmt='ko')
    plt.legend()
    plt.grid()
else:
    raise TypeError("Unknown kernel")



def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def CPGD(acquis, domain, λ=0.6, α=5e-3, β=5e-3, nIter=50, nParticles=20, 
         noyau='fourier'):

    ss = lambda r : np.abs(r) * r

    θ_0 = np.linspace(xgauche, xdroit, nParticles)
    r_0 = 0.5 * np.ones(nParticles)
    𝜈_0 = Mesure(r_0, θ_0)
    grad_r, grad_θ = np.zeros(nParticles), np.zeros(nParticles)

    r_k, θ_k, 𝜈_k = r_0, θ_0, 𝜈_0
    𝜈_vecteur = [0] * (nIter)
    (r_vecteur, θ_vecteur) = (np.zeros((nIter, nParticles)),
                              np.zeros((nIter, nParticles)))

    # Loop over the gradient descent
    for k in range(nIter):
        r_vecteur[k,:] = r_k
        θ_vecteur[k,:] = θ_k
        𝜈_vecteur[k] = 𝜈_k
        for l in range(nParticles):
            t_k = min(range(len(acquis)), key=lambda i: abs(acquis[i]-r_k[l]))
            simul_r = np.sum(ss(r_k) * fourier_measurements(θ_k[l] - θ_k, F_C))
            obs_r = np.sum(fourier_measurements(θ_k[l] - domain, F_C).T * ss(acquis)) / nParticles
            grad_r[l] = np.sign(r_k[l]) * np.real(simul_r - obs_r) + λ

            simul_der_r = np.sum(ss(r_k) * grad_fourier_measurements(θ_k[l] - θ_k, F_C))
            obs_der_r = np.sum(grad_fourier_measurements(θ_k[l] - domain, F_C).T * ss(acquis)) / nParticles
            grad_θ[l] = np.sign(r_k[l]) * np.real(simul_der_r - obs_der_r)

        # Update gradient flow
        r_k *= np.exp(2 * α * grad_r)
        θ_k += β * λ * grad_θ

        # Store iterated measure 𝜈
        𝜈_k = Mesure(r_k, θ_k)
    return (𝜈_k, 𝜈_vecteur, r_vecteur, θ_vecteur)


𝜈, 𝜈_itere, r_itere, θ_itere = CPGD(y0, X, noyau=noy)


print('On a retrouvé m_ax = ' + str(𝜈))
print('On voulait retrouver m_ax0 = ' + str(m_ax0))

plt.figure()
plt.stem(𝜈.x, 𝜈.a, label='$\\nu_t$', linefmt='r--',
         markerfmt='ro', use_line_collection=True, basefmt=" ")
plt.stem(m_ax0.x, m_ax0.a, label='$m_{a_0,x_0}$', linefmt='k--',
         markerfmt='ko', use_line_collection=True, basefmt=" ")
for l in range(len(𝜈.a)):
    plt.plot(θ_itere[:,l], r_itere[:,l], 'b', linewidth=1)
plt.legend()
plt.plot((-1, 2), (0, 0), 'k-', linewidth=1)
plt.xlim([-0.1,1.1])




