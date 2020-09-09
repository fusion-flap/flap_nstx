#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 13:16:15 2020

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
fn = os.path.join(thisdir,"flap_nstx.cfg")
flap.config.read(file_name=fn)  

import matplotlib.pyplot as plt
import numpy as np
import scipy

exp_id=139901
time_range=[0.300,0.310]
test=False
n_lag=40
sample_time=2.5e-6
time_window=0.001
n_frames=int(time_window/sample_time)
correlation_evolution=np.zeros([n_frames+1,n_lag+1])
time_ranges=np.asarray([np.arange(0,n_lag+1)/(n_lag)*(time_range[1]-time_range[0])+time_range[0],
                        np.arange(0,n_lag+1)/(n_lag)*(time_range[1]-time_range[0])+time_range[0]+time_window]).T
flap.get_data('NSTX_GPI', exp_id=exp_id,name='',object_name='GPI')

for j in range(n_lag+1):
    sample0=flap.get_data_object_ref('GPI').slice_data(slicing={'Time':time_ranges[j,0]}).coordinate('Sample')[0][0,0]
    d=flap.slice_data('GPI', 
                      slicing={'Sample':flap.Intervals(sample0,sample0+n_frames)}, 
                      output_name='GPI_SLICED')
    d.data=d.data/np.mean(d.data, axis=0)
    d=flap_nstx.analysis.detrend_multidim('GPI_SLICED', 
                                          order=1, 
                                          coordinates=['Image x', 'Image y'], 
                                          output_name='GPI_SLICED_DETREND')
    
    time=flap.get_data_object_ref('GPI_SLICED_DETREND').coordinate('Time')[0][:,0,0]
    time=time-time[0]
    
    frame1=np.asarray(flap.slice_data('GPI_SLICED_DETREND', slicing={'Sample':sample0}, output_name='GPI_FRAME1').data, dtype='float32')

    for i in range(0,n_frames-1):
        frame2=np.asarray(flap.slice_data('GPI_SLICED_DETREND', slicing={'Sample':sample0+i}, output_name='GPI_FRAME2').data, dtype='float32')
        corr=np.sum(frame1*frame2)/np.sqrt(np.sum(frame1*frame1)*np.sum(frame2*frame2))
        if test:
            plt.contourf(corr.data.T)
            plt.pause(0.001)
            plt.cla()
        correlation_evolution[i,j]=corr#.data[63,79]

plt.figure()
plt.contourf(time_ranges[0:-1,0]*1000., time*1000., correlation_evolution[:,0:-1], levels=51)

plt.ylabel('Time difference between frames [ms]')
plt.xlabel('First frame time [ms]')
plt.colorbar()