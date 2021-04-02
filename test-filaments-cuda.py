#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 15:30:44 2021

@author: Bastien https://github.com/XeBasTeX

Test sur filaments avec acquisitions SOFI inspirées de SMLM
http://bigwww.epfl.ch/smlm/challenge2016/datasets/MT4.N2.HD/Data/data.html

"""


__saveFig__ = False
__saveVid__ = False


import matplotlib.pyplot as plt
import scipy.io
from skimage import io
import numpy as np
import torch

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import cudavenant


# Initialiser torch
device = "cuda" if torch.cuda.is_available() else "cpu"
print("[Cudavenant] Using {} device".format(device))

# Charger pile et cumulants
stream = io.imread('sofi_filaments/tubulin_noiseless_lowBg.tif')
pile = torch.from_numpy(np.array(stream, dtype='float64'))

y_torch = torch.mean(pile, 0)
R_y_torch = cudavenant.torch_cov(pile)









