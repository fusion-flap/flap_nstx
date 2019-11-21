#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 14:58:31 2019

@author: mlampert
"""


import os

#import copy

import flap
import flap_nstx
from flap_nstx.analysis.nstx_gpi_tools import calculate_nstx_gpi_norm_coeff
from flap_nstx.analysis.nstx_gpi_tools import calculate_nstx_gpi_reference

import flap_mdsplus

import matplotlib.style as pltstyle
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import numpy as np

publication=False

if publication:

    plt.rcParams['lines.linewidth'] = 4
    plt.rcParams['axes.linewidth'] = 4
    plt.rcParams['axes.labelsize'] = 28  # 28 for paper 35 for others
    plt.rcParams['axes.titlesize'] = 28
    plt.rcParams['xtick.labelsize'] = 30
    plt.rcParams['xtick.major.size'] = 10
    plt.rcParams['xtick.major.width'] = 4
    plt.rcParams['xtick.major.size'] = 6
    plt.rcParams['xtick.minor.width'] = 2
    plt.rcParams['xtick.minor.size'] = 4
    plt.rcParams['ytick.labelsize'] = 30
    plt.rcParams['ytick.major.width'] = 4
    plt.rcParams['ytick.major.size'] = 6
    plt.rcParams['ytick.minor.width'] = 2
    plt.rcParams['ytick.minor.size'] = 4
    plt.rcParams['legend.fontsize'] = 28

else:
    pltstyle.use('default')

flap_nstx.register()
flap_mdsplus.register('NSTX_MDSPlus')

# The following code is deprecated. The gas cloud has a certain time evolution which is
# averaged. This should only be used for quiescent time ranges, not for e.g. ELMs.

def calculate_nstx_gpi_crosspower(exp_id=None,
                                  time_range=None,
                                  normalize_signal=False, #Normalize the amplitude for the average frame for the entire time range
                                  reference_pixel=None,
                                  reference_position=None,
                                  reference_flux=None,
                                  reference_area=None,      #In the unit of the reference, [psi,z] if reference_flux is not None
                                  fres=1.,                  #in KHz due to data being in ms
                                  flog=False,
                                  frange=None,
                                  interval_n=8.,
                                  filename=None,
                                  options=None,
                                  cache_data=False,
                                  normalize=False,           #Calculate coherency if True
                                  plot=False,
                                  plot_phase=False
                                  ):
    
    #139901 [300,307]
    
    #This function returns the crosspower between a single signal and all the other signals in the GPI.
    #A separate function is dedicated for multi channel reference channel 
    #e.g 3x3 area and 64x80 resulting 3x3x64x80 cross power spectra
    
    #Read data from the cine file
    if time_range is None:
        print('The time range needs to set for the calculation.')
        print('There is no point of calculating the entire time range.')
        return
    else:    
        if (type(time_range) is not list and len(time_range) != 2):
            raise TypeError('time_range needs to be a list with two elements.')
    if exp_id is not None:
        print("\n------- Reading NSTX GPI data --------")
        if cache_data:
            try:
                d=flap.get_data_object_ref(exp_id=exp_id,object_name='GPI')
            except:
                print('Data is not cached, it needs to be read.')
                d=flap.get_data('NSTX_GPI',exp_id=exp_id,name='',object_name='GPI')
        else:
            d=flap.get_data('NSTX_GPI',exp_id=exp_id,name='',object_name='GPI')
    else:
        raise ValueError('The experiment ID needs to be set.')
    if reference_flux is not None:   
        d.add_coordinate(coordinates='Flux r',exp_id=exp_id)
    
    #Normalize the data for the maximum cloud distribution
    if normalize_signal:
        normalizer=calculate_nstx_gpi_norm_coeff(exp_id=exp_id,             # Experiment ID
                                                 f_high=1e2,            # Low pass filter frequency in Hz
                                                 design='Chebyshev II',    # IIR filter design (from scipy)
                                                 test=False,               # Testing input
                                                 filter_data=True,         # IIR LPF the data
                                                 time_range=None,          # Timer range for the averaging in ms [t1,t2]
                                                 calc_around_max=False,    # Calculate the average around the maximum of the GPI signal
                                                 time_window=50.,          # The time window for the calc_around_max calculation
                                                 cache_data=True,          #
                                                 verbose=False,
                                                 )
        d.data = d.data/normalizer.data #This should be checked to some extent, it works with smaller matrices
    
#    for index_x in range(d.data.shape[0]):
#        for index_y in range(d.data.shape[1]):
#            d.data[:,index_x,index_y] /= normalizer.data[index_x,index_y]
    
    #Calculate the crosspower spectra for the timerange between the reference pixel and all the other pixels
    if reference_pixel is None and reference_position is None and reference_flux is None:
        calculate_apsd=True
        print('No reference is defined, returning autopower spectra.')
    else:
        calculate_apsd=False
        reference_signal=calculate_nstx_gpi_reference('GPI', exp_id=exp_id,
                                                      time_range=time_range,
                                                      reference_pixel=reference_pixel,
                                                      reference_area=reference_area,
                                                      reference_position=reference_position,
                                                      reference_flux=reference_flux,
                                                      output_name='GPI_REF')
    
    flap.slice_data('GPI',exp_id=exp_id,
                    slicing={'Time':flap.Intervals(time_range[0],time_range[1])},
                    output_name='GPI_SLICED')
    if calculate_apsd:
        object_name='GPI_APSD'
        flap.apsd('GPI_SLICED',exp_id=exp_id,
                  coordinate='Time',
                  options={'Resolution':fres,
                           'Range':frange,
                           'Logarithmic':flog,
                           'Interval_n':interval_n,
                           'Hanning':False,
                           'Trend removal':None,
                           },
                   output_name=object_name)
        if plot:
            flap.plot(object_name, exp_id=exp_id,
                      plot_type='animation', 
                      axes=['Image x', 'Image y', 'Frequency'], 
                      options={'Plot units': {'Frequency':'kHz'}})
    else:
        object_name='GPI_CPSD'
        flap.cpsd('GPI_SLICED',exp_id=exp_id,
                  ref=reference_signal,
                  coordinate='Time',
                  options={'Resolution':fres,
                           'Range':frange,
                           'Logarithmic':flog,
                           'Interval_n':interval_n,
                           'Hanning':False,
                           'Normalize':normalize,
                           'Trend removal':None,
                           },
                   output_name=object_name)
        flap.abs_value('GPI_CPSD',exp_id=exp_id,
                       output_name='GPI_CPSD_ABS')
        flap.phase('GPI_CPSD',exp_id=exp_id,
                   output_name='GPI_CPSD_PHASE')
        if plot:
            if plot_phase:
                flap.plot('GPI_CPSD_PHASE', exp_id=exp_id,
                          plot_type='animation', 
                          axes=['Image x', 'Image y', 'Frequency'], 
                          options={'Plot units': {'Frequency':'kHz'}})
            else:
                flap.plot('GPI_CPSD_ABS', exp_id=exp_id,
                          plot_type='animation', 
                          axes=['Image x', 'Image y', 'Frequency'], 
                          options={'Plot units': {'Frequency':'kHz'}})
        
    #Calculate the cross-correlation functions for the timerange between the reference pixel and all the other pixels
    #Save all the data and the settings. The filename should resembe the settings.
    
def calculate_nstx_gpi_crosscorrelation(exp_id=None,
                                        time_range=None,
                                        add_flux=None,
                                        reference_pixel=None,
                                        reference_flux=None,
                                        reference_position=None,
                                        reference_area=None,
                                        filter_low=None,
                                        filter_high=None,
                                        filter_design='Chebyshev II',
                                        trend=['Poly',2],
                                        frange=None,
                                        taurange=[-500e-6,500e-6],
                                        taures=2.5e-6,
                                        interval_n=11,
                                        filename=None,
                                        options=None,
                                        cache_data=False,
                                        normalize_signal=False,
                                        normalize=True,           #Calculate correlation if True (instead of covariance)
                                        plot=False,
                                        plot_acf=False
                                       ):
    
    if time_range is None:
        print('The time range needs to set for the calculation.')
        print('There is no point of calculating the entire time range.')
        return
    else:    
        if (type(time_range) is not list and len(time_range) != 2):
            raise TypeError('time_range needs to be a list with two elements.')
    if exp_id is not None:
        print("\n------- Reading NSTX GPI data --------")
        if cache_data:
            try:
                d=flap.get_data_object_ref(exp_id=exp_id,object_name='GPI')
            except:
                print('Data is not cached, it needs to be read.')
                d=flap.get_data('NSTX_GPI',exp_id=exp_id,name='',object_name='GPI')
        else:
            d=flap.get_data('NSTX_GPI',exp_id=exp_id,name='',object_name='GPI')
    else:
        raise ValueError('The experiment ID needs to be set.')
        
    if reference_flux is not None or add_flux:
        d.add_coordinate(coordinates='Flux r',exp_id=exp_id)
    
    #Normalize the data for the maximum cloud distribution
    if normalize_signal:
        normalizer=calculate_nstx_gpi_norm_coeff(exp_id=exp_id,             # Experiment ID
                                                 f_high=1e2,            # Low pass filter frequency in Hz
                                                 design=filter_design,    # IIR filter design (from scipy)
                                                 test=False,               # Testing input
                                                 filter_data=True,         # IIR LPF the data
                                                 time_range=None,          # Timer range for the averaging in ms [t1,t2]
                                                 calc_around_max=False,    # Calculate the average around the maximum of the GPI signal
                                                 time_window=50.,          # The time window for the calc_around_max calculation
                                                 cache_data=True,          #
                                                 verbose=False,
                                                 )
        d.data = d.data/normalizer.data #This should be checked to some extent, it works with smaller matrices
    
    #SLicing data to the input time range    
    flap.slice_data('GPI',exp_id=exp_id,
                    slicing={'Time':flap.Intervals(time_range[0],time_range[1])},
                    output_name='GPI_SLICED')
    
    #Filtering the signal since we are in time-space not frequency space
    if frange is not None:
        filter_low=frange[0]
        filter_high=frange[1]
    if filter_low is not None or filter_high is not None:
        if filter_low is not None and filter_high is None:
            filter_type='Highpass'
        if filter_low is None and filter_high is not None:
            filter_type='Lowpass'
        if filter_low is not None and filter_high is not None:
            filter_type='Bandpass'

        flap.filter_data('GPI_SLICED',exp_id=exp_id,
                         coordinate='Time',
                         options={'Type':filter_type,
                                  'f_low':filter_low,
                                  'f_high':filter_high, 
                                  'Design':filter_design},
                         output_name='GPI_SLICED_FILTERED')

    if reference_pixel is None and reference_position is None and reference_flux is None:
        calculate_acf=True
    else:
        calculate_acf=False
                
    if not calculate_acf:
        calculate_nstx_gpi_reference('GPI_SLICED_FILTERED', exp_id=exp_id,
                                     reference_pixel=reference_pixel,
                                     reference_area=reference_area,
                                     reference_position=reference_position,
                                     reference_flux=reference_flux,
                                     output_name='GPI_REF')
        
        flap.ccf('GPI_SLICED_FILTERED',exp_id=exp_id,
                  ref='GPI_REF',
                  coordinate='Time',
                  options={'Resolution':taures,
                           'Range':taurange,
                           'Trend':trend,
                           'Interval':interval_n,
                           'Normalize':normalize,
                           },
                   output_name='GPI_CCF')
        
    if plot:
        if not plot_acf:
            object_name='GPI_CCF'
        else:
            object_name='GPI_ACF'
            
            flap.plot(object_name, exp_id=exp_id,
                      plot_type='animation', 
                      axes=['Image x', 'Image y', 'Time lag'], 
                      options={'Plot units': {'Time lag':'us'}, 
                               'Z range':[0,1]},)     

thisdir = os.path.dirname(os.path.realpath(__file__))
fn = os.path.join(thisdir,"flap_nstx.cfg")
flap.config.read(file_name=fn)