#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 13:03:41 2020

@author: mlampert
"""

from openpiv import tools, process, validation, filters, scaling 

import numpy as np
import matplotlib.pyplot as plt

import os
import copy

import flap
import flap_nstx
from flap_nstx.analysis import *

flap_nstx.register()
import flap_mdsplus
flap_mdsplus.register('NSTX_MDSPlus')
thisdir = os.path.dirname(os.path.realpath(__file__))
fn = os.path.join(thisdir,"../flap_nstx.cfg")
flap.config.read(file_name=fn)  
import scipy

exp_id=139901
time_range=[0.307,0.308]

flap.get_data('NSTX_GPI', exp_id=exp_id,name='',object_name='GPI')
d=flap.slice_data('GPI', slicing={'Time':flap.Intervals(time_range[0],time_range[1])})
sample0=d.coordinate('Sample')[0][0,0,0]
frame_a=np.asarray(flap.slice_data('GPI', slicing={'Sample':sample0}, output_name='GPI_FRAME1').data, dtype='float32')
frame_b=np.asarray(flap.slice_data('GPI', slicing={'Sample':sample0+1}, output_name='GPI_FRAME2').data, dtype='float32')

winsize = 10 # pixels
searchsize = 10  # pixels, search in image B
overlap = 5 # pixels
dt = 2.5e-6 # sec


u0, v0, sig2noise = process.extended_search_area_piv(frame_a.astype(np.int32), 
                                                     frame_b.astype(np.int32), 
                                                     window_size=winsize, 
                                                     overlap=overlap, 
                                                     dt=dt, 
                                                     search_area_size=searchsize, 
                                                     sig2noise_method='peak2peak' )

x, y = process.get_coordinates(image_size=frame_a.shape, 
                               window_size=winsize, 
                               overlap=overlap)
u1, v1, mask = validation.sig2noise_val( u0, v0, sig2noise, threshold = 1.3 )
u2, v2 = filters.replace_outliers( u1, v1, method='localmean', max_iter=10, kernel_size=2)
x, y, u3, v3 = scaling.uniform(x, y, u2, v2, scaling_factor = 1 )
tools.save(x, y, u3, v3, mask, 'exp1_001.txt' )
tools.display_vector_field('exp1_001.txt', scale=100, width=0.0025)