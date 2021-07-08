# -*- coding: utf-8 -*-
"""
This little script generates a big *tif file from a lot of individual *tif
The sole *tif file is better suited for the cudavenant project.

Created on Wed Jul  7 08:13:40 2021

@author: blaville
"""

import glob
import tifffile

with tifffile.TiffWriter('real_data_smlm.tif') as stack:
    for filename in glob.glob('*.tif'):
        stack.save(
            tifffile.imread(filename), 
            photometric='minisblack', 
            contiguous=True
        )
