#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 15:30:44 2021

@author: Bastien (https://github.com/XeBasTeX)

Test sur filaments avec acquisitions SOFI inspirées de SMLM
http://bigwww.epfl.ch/smlm/challenge2016/datasets/MT4.N2.HD/Data/data.html

"""


__saveFig__ = False
__saveVid__ = False


import matplotlib.pyplot as plt
import scipy.io
from skimage import io
import numpy as np

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import covenant


pile = np.array(io.imread('sofi_filaments/tubulin_noiseless_lowBg.tif'),
                dtype='float64')
# pile = np.array(io.imread('sofi_filaments/tubulin_noiseless_lowBg.tif'),
#                 dtype='float64')

pile_moy = np.mean(pile, axis=0)
# pile = pile / pile.shape[-1]
y = pile[:, 16:32, 16:32] / 1000
# pile = pile + np.random.normal(0, 1e-1, size=pile.shape)

emitters_loc = np.genfromtxt('sofi_filaments/emitters_noiseless_lowBg.csv',
                             delimiter=',')
emitters_loc = np.fliplr(emitters_loc) / 64
m_ax0 = covenant.Mesure2D(np.ones(emitters_loc.shape[0]), emitters_loc)

y_bar = np.mean(y, axis=0)
R_y = covenant.covariance_pile(y, y_bar)
print('[+] Covariance calculée')

#%% Calcul effectif

N_ECH = 16
X_GAUCHE = 0
X_DROIT = 1
FWMH = 2.2875 / 64 # c le bon fwmh
SIGMA = FWMH / (2 * np.sqrt(2*np.log(2)))
domain = covenant.Domain2D(X_GAUCHE, X_DROIT, N_ECH, SIGMA)

q = 2**3
super_domain = domain.super_resolve(q)


iteration = 150
lambda_cov = 1e-7
lambda_moy = 1e-4  # s'arrête à 103 Diracs

# test_obj = 'acquis'
# test_acquis = y_bar

# test_obj = 'covar'
# test_acquis = R_y


# SIGMA_BRUITS = 1e-2
# SIGMA_TARGET = 1.5 * SIGMA_BRUITS

# (m_top, nrj_top, lambda_top) = covenant.homotopy(test_acquis, domain,
#                                                   SIGMA_TARGET,
#                                                   obj=test_obj,
#                                                   nIter=80, c=1)

(m_cov, nrj_cov, mes_cov) = covenant.SFW(R_y, domain, regul=lambda_cov,
                                         nIter=iteration, mesParIter=True,
                                         obj='covar', printInline=True)

(m_moy, nrj_moy, mes_moy) = covenant.SFW(kernel_tmp[16:32, 16:32] , domain,
                                         regul=lambda_moy,
                                         nIter=iteration , mesParIter=True,
                                         obj='acquis', printInline=True)

print(f'm_cov : {m_cov.N} Diracs')
print(f'm_moy : {m_moy.N} Diracs')
if m_cov.N > 0:
    certificat_V_cov = covenant.etak(m_cov, R_y, domain, lambda_cov,
                                     obj='covar')
    covenant.plot_experimental(m_cov, domain, y_bar, nrj_cov,
                               certificat_V_cov, obj='covar',
                               saveFig=__saveFig__, title='filaments-covar')
if m_moy.N > 0:
    certificat_V_moy = covenant.etak(m_moy, y_bar, domain, lambda_moy,
                                     obj='acquis')
    covenant.plot_experimental(m_moy, domain, y_bar, nrj_moy,
                               certificat_V_moy, obj='acquis',
                               saveFig=__saveFig__, title='filaments-moy')

if __saveVid__:
    covenant.gif_experimental(y_bar, mes_cov, super_domain, cross=True,
                              video='gif', title='filaments-cov')
    covenant.gif_experimental(y_bar, mes_moy, super_domain, cross=True,
                          video='gif', title='filaments-moy')


if m_cov.N > 0:
    covenant.compare_covariance(m_cov, R_y, domain)


m_tot = covenant.Mesure2D()



#%% Diviser pour régner

# class Quadrant:
#      def __init__(self, cut_domain, local_coord, empty):
#          quad = []
#          for i in range(len(empty)):
#              quad += [[cut_domain[i], local_coord[i], empty[i]]]
#          self.quadrant = quad

#      def conquer(self, obj='covar'):
#          m_quadrant = []
#          for i in range(len(self.quadrant)):
#             if self.quad[i, -1] == False:
#                 y_cut = self.quad[i, 0]
#                 if 'obj' == 'covar':
#                     y_cut = y_cut / np.max(y_cut)**(1/2)
#                     pile_moy = np.mean(y_cut, axis=0)
#                     y_bar = pile_moy
#                     R_y = covenant.covariance_pile(y_cut, y_bar)
#                     (m_cov, nrj_cov) = covenant.SFW(R_y, domain,
#                                                     regul=lambda_cov,
#                                                     nIter=iteration,
#                                                     obj='covar')
#                     m_cov.a = m_cov.a * np.max(y_cut)**(1/2)
#                     m_quadrant += m_cov
#                 elif 'obj' == 'acquis':
#                     y_cut = y_cut / np.max(y_cut)
#                     y_bar = pile_moy
#                     (m_cov, nrj_cov, mes_cov) = covenant.SFW(y_bar, domain,
#                                                              regul=lambda_moy,
#                                                              nIter=iteration,
#                                                              obj='moy')
#                     m_cov.a = m_cov.a * np.max(y_cut)
#                     m_quadrant += m_cov
#          return(m_quadrant)


# # def split(array, nrows, ncols):
# #     """Split a matrix into sub-matrices."""

# #     r, h = array.shape
# #     return (array.reshape(h//nrows, nrows, -1, ncols)
# #                  .swapaxes(1, 2)
# #                  .reshape(-1, nrows, ncols))

# def split(mat):
#     N_0, N_1 = mat.shape[0], mat.shape[1]
#     return (mat[:N_0//2, :N_1//2], mat[N_0//2:, :N_1//2], mat[:N_0//2, N_1//2:],
#             mat[N_0//2:, N_1//2:])

# def gen_list_dom(array, split_factor=2):
#     quad_1, quad_2, quad_3, quad_4 = split(array)
#     list_domain = [quad_1, quad_2, quad_3, quad_4]
#     return list_domain

# exemple = scipy.linalg.hilbert(32)
# list_dom = gen_list_dom(exemple)
# local_coord = []



# # decoupage = Quadrant()

y_big = np.mean(pile, axis=0)
N_ECH = 64
X_GAUCHE = 0
X_DROIT = 1
FWMH = 2.2875 / N_ECH
SIGMA = FWMH / (2 * np.sqrt(2*np.log(2)))
domain_big = covenant.Domain2D(X_GAUCHE, X_DROIT, N_ECH, SIGMA)

#%% Grindhouse : là où ça devient sale. Tout à la main comme un sagouin


x_ht_gche = 16 * 3
x_ht_droit = x_ht_gche + 16 # commande la ligne
x_bas_gche = 16 * 0
x_bas_droite = x_bas_gche + 16# commande la colonne


y = pile[:, x_ht_gche:x_ht_droit, x_bas_gche:x_bas_droite]
y = y / np.max(y)**(1/2)
pile_moy = np.mean(y, axis=0)
y_bar = pile_moy
R_y = covenant.covariance_pile(y, y_bar)

plt.imshow(y_bar)
plt.colorbar()

#%% 

iteration = 40
N_ECH = pile_moy.shape[0]  # Taux d'échantillonnage
X_GAUCHE = 0
X_DROIT = 1
FWMH = 2.2875 / N_ECH
SIGMA = FWMH / (2 * np.sqrt(2*np.log(2)))
domain = covenant.Domain2D(X_GAUCHE, X_DROIT, N_ECH, SIGMA)

(m_cov, nrj_cov) = covenant.SFW(R_y, domain, regul=lambda_cov,
                                         nIter=iteration,
                                         obj='covar', printInline=True)

print(f'm_cov : {m_cov.N} Diracs')
if m_cov.N > 0:
    certificat_V_cov = covenant.etak(m_cov, R_y, domain, lambda_cov,
                                     obj='covar')
    covenant.plot_experimental(m_cov, domain, y_bar, nrj_cov,
                               certificat_V_cov, obj='covar',
                               saveFig=False, title='filaments-covar')


#%%

plt.imshow(y_big)
plt.scatter((64*m_cov.x[:, 0])/4 + x_bas_gche, (64*m_cov.x[:, 1])/4 + x_ht_gche, 
            marker='+', c='g',
            label='Recovered spikes')

#%%

x_redress = m_cov.x[:, 0]/4  + x_bas_gche/64
y_reddress = m_cov.x[:, 1]/4 + x_ht_gche/64
pos = np.array(list(zip(x_redress, y_reddress)))
m_tmp = covenant.Mesure2D(m_cov.a, pos)
m_previous = m_tot
m_tot += m_tmp

plt.imshow(m_tot.kernel(domain_big))

# Si le carré n'est pas dans ce découpage il est vide
# Carré 1 : 18
# Carré 3 : 27
# Carré 5 : ? 20
# Carré 6 : 43
# Carré 7 : 25
# Carré 9 : 29
# Carré 10 pas fait

#%% Save the m_tot

import pickle

with open('m_tot.pkl', 'wb') as output:
    pickle.dump(m_tot, output, pickle.HIGHEST_PROTOCOL)


#%% Retrieve the m_plot

import pickle

objects = []
with (open("m_tot.pkl", "rb")) as openfile:
    while True:
        try:
            objects.append(pickle.load(openfile))
        except EOFError:
            break

m_Mx = objects[0]
covenant.plot_reconstruction(m_Mx, domain_big, y_big, obj='covar',
                             saveFig=True, title='filaments-covar-global')

#%% Vérifier qu'on a le bon sigma

N_ECH = 64 # Taux d'échantillonnage
X_GAUCHE = 0
X_DROIT = 1
FWMH = 2.2875  / 64
SIGMA = FWMH / (2 * np.sqrt(2*np.log(2)))
domain_tmp = covenant.Domain2D(X_GAUCHE, X_DROIT, N_ECH, SIGMA)

# On ne prend que 1 mes de dirac sur 10 (ou sur 20)
m_tmp = covenant.Mesure2D(np.ones(emitters_loc[0:-1:5, :].shape[0]),
                          emitters_loc[0:-1:5, :])
m_tmp_2 = covenant.Mesure2D(np.ones(emitters_loc[0:-1:20, :].shape[0]),
                          emitters_loc[0:-1:20, :])
kernel_tmp = m_tmp.kernel(domain_tmp)

fig = plt.figure(figsize=(12, 4))
plt.subplot(121)
plt.imshow(pile_moy)
plt.title(r'Mean $\overline{y}$', fontsize=30)
plt.colorbar()
plt.subplot(122)
plt.title(r'Kernel $m_{a_0, x_0}$', fontsize=30)
plt.imshow(kernel_tmp)
plt.colorbar()

covar_tmp = m_tmp_2.covariance_kernel(domain_tmp)

fig2 = plt.figure(figsize=(12, 4))
plt.imshow(covar_tmp[:600, :600])
plt.title(r'$\Lambda(m_{M,x})$', fontsize=30)
plt.colorbar()


# np.linalg.norm(kernel_tmp - pile_moy)