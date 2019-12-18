#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 17:21:19 2019

@author: mlampert
"""


import os

#Importing and setting up the FLAP environment
try:
    flap
except:
    import flap
try:
    flap_nstx
except:
    import flap_nstx
    flap_nstx.register()

thisdir = os.path.dirname(os.path.realpath(__file__))
fn = os.path.join(thisdir,"flap_nstx.cfg")
flap.config.read(file_name=fn)

import matplotlib.style as pltstyle
import matplotlib.pyplot as plt

import numpy as np

#from scipy.signal import find_peaks    #This is not working as reliably as the CWT
from scipy.signal import find_peaks_cwt
    
def find_filaments(data_object=None,      #FLAP data objectCould be set instead of exp_id and time_range 
                   exp_id=None,           #Shot number
                   time_range=None,       #Time range for the filament finding
                   frange=[0.1e3,100e3],  #Frequency range to pre-condition the data
                   pixel=[10,40],         #The pixel to find the peak in.                   
                   vertical_sum=False,    #Sum up all the pixels vertically
                   width_range=[1,30],    #The width range for the CWT peak finding algorithm
                   cache_data=False,      #Try to gather the cached data (exp_id, timerange input)
                   return_index=False,     #Return the peak times instaed of the peak indices
                   test=False):           #Plot the resulting data along with the peaks
    
    #Read signal
    if data_object is None:
        data_object='GPI'
        if time_range is None:
            print('The time range needs to set for the calculation.')
            return
        else:    
            if (type(time_range) is not list and len(time_range) != 2):
                raise TypeError('time_range needs to be a list with two elements.')
        if exp_id is not None:
            print("\n------- Reading NSTX GPI data --------")
            if cache_data:
                try:
                    d=flap.get_data_object_ref(exp_id=exp_id,object_name=data_object)
                except:
                    print('Data is not cached, it needs to be read.')
                    d=flap.get_data('NSTX_GPI',exp_id=exp_id,name='',object_name=data_object)
            else:
                d=flap.get_data('NSTX_GPI',exp_id=exp_id,name='',object_name=data_object)
        else:
            raise ValueError('The experiment ID needs to be set.')
        
        
        #Filter signal to HPF 100Hz
        if vertical_sum:
            slicing={'Time':flap.Intervals(time_range[0],time_range[1]),
                     'Image x':pixel[0],}
            summing={'Image y':'Mean'}
        else:
            slicing={'Time':flap.Intervals(time_range[0],time_range[1]),
                     'Image x':pixel[0],
                     'Image y':pixel[1],}
            summing=None
    else:
        #try:
        #    flap.get_data_object_ref(data_object)
        #except:
        #    raise ValueError('The data object, '+data_object+' does not exist.')
        if vertical_sum:
            slicing={'Image x':pixel[0]}
            summing={'Image y':'Mean'}
        else:
            slicing={'Image x':pixel[0],
                     'Image y':pixel[1]}
            summing=None
    try:            
        flap.slice_data(data_object, 
                        slicing=slicing,
                        summing=summing,
                        output_name='GPI_SLICED')
    except:
        raise ValueError('The data object '+data_object+' does not exist.')
    d=flap.filter_data('GPI_SLICED',
                       coordinate='Time',
                       options={'Type':'Bandpass',
                                'f_low':frange[0],
                                'f_high':frange[1],
                                'Design':'Chebyshev II'},
                       output_name='GPI_FILTERED')

    #ind=find_peaks(d.data, distance=25, threshold=threshold)[0]         #This method needs quite a lot of tinkering, it is deprecated
    ind=find_peaks_cwt(d.data, np.arange(width_range[0],width_range[1])) #This method is working quite well without any data preconditioning except the filtering
    
    if return_index:
        return ind
    else:
        return d.coordinate('Time')[0][ind]
    
    if test:
        plt.figure()
        plt.plot(d.coordinate('Time')[0],d.data)
        plt.scatter(d.coordinate('Time')[0][ind],d.data[ind], color='red')