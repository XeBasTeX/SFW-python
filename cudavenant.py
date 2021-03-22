#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Attemps to provide the covenant package with a GPU acceleration

Created on Mon Mar 22 08:49:01 2021

@author: Bastien (https://github.com/XeBasTeX)
"""

import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

