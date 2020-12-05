#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 16:51:33 2020

@author: Bastien (https://github.com/XeBasTeX)
"""


from Mesure import *
import numpy as np
import matplotlib.pyplot as plt


T_ech = 60
T_acquis = 30
T = np.linspace(0, T_acquis, T_ech)
N_mol = 15
acquis_temporelle = np.zeros((N_mol, T_ech))
for i in range(N_mol):
    phase = np.pi*np.random.rand()
    acquis_temporelle[i,] = np.sin(T + phase)
    plt.plot(acquis_temporelle[i,:])

positions_mesures = np.random.rand(1, N_mol)[0]
amplitudes_moyennes = 0


