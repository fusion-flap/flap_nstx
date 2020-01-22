#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 16:46:27 2020

@author: mlampert
"""
import os
import copy
import flap
import flap_nstx
flap_nstx.register()
import flap_mdsplus
flap_mdsplus.register('NSTX_MDSPlus')
    
thisdir = os.path.dirname(os.path.realpath(__file__))
fn = os.path.join(thisdir,"flap_nstx.cfg")
flap.config.read(file_name=fn) 

import matplotlib.pyplot as plt
from matplotlib import path as pltPath
import numpy as np
from scipy import interpolate
from scipy.signal import correlate
order=0

flap.get_data('NSTX_GPI', exp_id=139901, name='', object_name='GPI_RAW')
d11=flap.slice_data('GPI_RAW',slicing={'Time':0.324}, output_name='GPI_RAW_SLICED')
d12=flap.slice_data('GPI_RAW',slicing={'Time':0.324+2.5e-6}, output_name='GPI_RAW_SLICED_2')
trend1=flap_nstx.analysis.detrend_multidim(data_object='GPI_RAW_SLICED', coordinates=['Image x','Image y'], order=order, output_name='GPI_RAW_SLICED_DETREND', return_trend=True)
trend2=flap_nstx.analysis.detrend_multidim(data_object='GPI_RAW_SLICED_2', coordinates=['Image x','Image y'], order=order, output_name='GPI_RAW_SLICED_2_DETREND', return_trend=True)

d=flap.get_data_object('GPI_RAW_SLICED')
d1=flap.get_data_object_ref('GPI_RAW_SLICED_DETREND')
d2=flap.get_data_object_ref('GPI_RAW_SLICED_2_DETREND')


plt.figure()
flap.plot('GPI_RAW_SLICED', plot_type='contour', axes=['Image x', 'Image y'], plot_options={'levels':51})
plt.title('139901 GPI @ 0.324s')
plt.savefig('139901_0.324s_orig_frame.pdf')

plt.figure()
flap.plot('GPI_RAW_SLICED_DETREND', plot_type='contour', axes=['Image x', 'Image y'], plot_options={'levels':51})
plt.title('139901 GPI '+str(order)+' detrend @ 0.324s')
plt.savefig('139901_0.324s_'+str(order)+'_detrend.pdf')

plt.figure()
flap.plot('GPI_RAW_SLICED_2_DETREND', plot_type='contour', axes=['Image x', 'Image y'], plot_options={'levels':51})
plt.title('139901 GPI detrend @ 0.3240025s')
plt.savefig('139901_0.3240025s_'+str(order)+'_detrend.pdf')
if order != 0:
    plt.figure()
    d.data=trend2
    d.plot(plot_type='contour', axes=['Image x', 'Image y'], plot_options={'levels':51})
    plt.title('139901 GPI trend @ 0.324s')
    plt.savefig('139901_0.324s_'+str(order)+'ord_trend.pdf')

corr=correlate(d1.data,d2.data)

plt.figure()
plt.title('139901 0.324-0.3240025 2D correlation')
plt.contourf(np.arange(-63,64),np.arange(-79,80),corr.T, levels=51)
plt.xlabel('X shift [pix]')
plt.ylabel('Y shift [pix]')
plt.savefig('139901_0.324s_subs_correlation_'+str(order)+'ord_detrend.pdf')

corr=correlate(np.asarray(d11.data,dtype='float64'),np.asarray(d12.data,dtype='float64'))

plt.figure()
plt.title('139901 0.324-0.3240025 2D correlation no detrend')
plt.contourf(np.arange(-63,64),np.arange(-79,80),corr.T, levels=51)
plt.xlabel('X shift [pix]')
plt.ylabel('Y shift [pix]')
plt.savefig('139901_0.324s_subs_correlation.pdf')