#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 12:21:36 2022

@author: mlampert
"""
#This plot out the spectrogram needed for the 
import os

import scipy
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import flap
import flap_nstx

thisdir = os.path.dirname(os.path.realpath(__file__))
fn = os.path.join(thisdir,"../flap_nstx.cfg")
flap.config.read(file_name=fn)
flap_nstx.register()

tmin = 0.0
tmax = 1.0
fmin = 0
fmax = 100

shot=139026

data_object=flap.get_data('NSTX_MDSPlus',
                          name='\\OPS_PC::'+'\\bdot_l1dmivvhf8_raw'.upper(),
                          exp_id=shot,
                          object_name='BDOT_SIGNAL')
data_object_trimmed=data_object.slice_data(slicing={'Time':flap.Intervals(tmin,tmax)})
# compute spectrogram
x=data_object_trimmed.data
fs=1/(data_object_trimmed.coordinate('Time')[0][1]-data_object_trimmed.coordinate('Time')[0][0])


nperseg = 2**12
f, t, Sxx = scipy.signal.spectrogram(x,fs,nperseg=nperseg,noverlap=nperseg//2,nfft=nperseg*4)
time = t+data_object_trimmed.coordinate('Time')[0][0]

plt.figure()
ax = plt.subplot(111)
vmin=np.min(Sxx)*1e6; vmax=np.max(Sxx)/1e2
#Sxx=25+np.log(Sxx)
#print(Sxx)
plt.pcolormesh(time, f/1e3, Sxx, norm=mpl.colors.LogNorm(vmin=vmin,vmax=vmax))#, cmap='binary')
ax.set_ylim(fmin,fmax)
ax.set_xlim(tmin,tmax)
f_list=[17+4,31+8,44+12,57+16]
#for f in f_list:
#    ax.plot(t_range,[f]*len(t_range),color='red')
ax.set_title('NSTX ' +str(shot)+' Magnetic')
ax.ticklabel_format(useOffset=False, style='plain')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Frequency (kHz)')


