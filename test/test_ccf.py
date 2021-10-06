#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 17:01:34 2020

@author: mlampert
"""
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

import matplotlib.pyplot as plt
import numpy as np
import scipy

nstx_gpi_generate_synthetic_data(exp_id=1, time=0.0001, output_name='test', 
                                 poloidal_velocity=10e3, radial_velocity=5e3, 
                                 poloidal_size=0.02, radial_size=0.03, start_position=[1.5, 0.3], 
                                 amplitude=1., gaussian=True, add_background=False)

flap.slice_data('test', slicing={'Sample':2}, output_name='test_0')
flap.slice_data('test', slicing={'Sample':3}, output_name='test_1')
#flap.ccf('test_0', 'test_1', 
#         coordinate=['Image x'], 
#         options={'Resolution':1, 'Range':[-63,63], 'Trend removal':None, 'Normalize':True, 'Interval_n': 1}, 
#         output_name='test_01_correlation')

flap.ccf('test_0', 'test_1', 
         coordinate=['Image x', 'Image y'], 
         options={'Resolution':1, 'Range':[[-63,63],[-79,79]], 'Trend removal':None, 'Normalize':True, 'Interval_n': 1}, 
         output_name='test_01_correlation')
flap.plot('test_01_correlation', plot_type='contour', axes=['Image x lag','Image y lag'])