#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 14:58:31 2019

@author: mlampert
"""

#Core imports
import os

#FLAP imports
import flap

import flap_nstx
flap_nstx.register()

import flap_mdsplus
flap_mdsplus.register('NSTX_MDSPlus')

thisdir = os.path.dirname(os.path.realpath(__file__))
fn = os.path.join(thisdir,"../flap_nstx.cfg")
flap.config.read(file_name=fn)  
    
#Scientific library imports
import matplotlib.style as pltstyle
import matplotlib.pyplot as plt

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
                                  plot_phase=False,
                                  axes=['Image x', 'Image y', 'Frequency'],
                                  hanning=True,
                                  colormap=None,
                                  video_saving_only=False,
                                  video_filename=None,
                                  save_video=False,
                                  comment=None,
                                  zlog=False,
                                  save_for_paraview=False
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
        normalizer=flap_nstx.tools.calculate_nstx_gpi_norm_coeff(exp_id=exp_id,             # Experiment ID
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
    
    #Calculate the crosspower spectra for the timerange between the reference pixel and all the other pixels
    if reference_pixel is None and reference_position is None and reference_flux is None:
        calculate_apsd=True
        print('No reference is defined, returning autopower spectra.')
    else:
        calculate_apsd=False
        reference_signal=flap_nstx.tools.calculate_nstx_gpi_reference('GPI', exp_id=exp_id,
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
        d=flap.apsd('GPI_SLICED',exp_id=exp_id,
                  coordinate='Time',
                  options={'Resolution':fres,
                           'Range':frange,
                           'Logarithmic':flog,
                           'Interval_n':interval_n,
                           'Hanning':hanning,
                           'Trend removal':None,
                           },
                   output_name=object_name)
    else:
        object_name='GPI_CPSD'
        flap.cpsd('GPI_SLICED',exp_id=exp_id,
                  ref=reference_signal,
                  coordinate='Time',
                  options={'Resolution':fres,
                           'Range':frange,
                           'Logarithmic':flog,
                           'Interval_n':interval_n,
                           'Hanning':hanning,
                           'Normalize':normalize,
                           'Trend removal':None,
                           },
                   output_name=object_name)
        flap.abs_value(object_name,exp_id=exp_id,
                       output_name='GPI_CPSD_ABS')
        flap.phase(object_name,exp_id=exp_id,
                   output_name='GPI_CPSD_PHASE')    
    if not save_video:
        if plot:
            if calculate_apsd:
                object_name='GPI_APSD'
            else:
                if plot_phase:
                    object_name='GPI_CPSD_PHASE'
                else:
                    object_name='GPI_CPSD_ABS'
            flap.plot(object_name, exp_id=exp_id,
                      plot_type='animation', 
                      axes=axes, 
                      options={'Force axes':True,
                               'Colormap':colormap,
                               'Plot units':{'Device R':'mm',
                                             'Device z':'mm',
                                             'Frequency':'kHz'},
                               'Log z':zlog})
    else:
        if video_filename is None:
            if time_range is not None:
                video_filename='NSTX_GPI_'+str(exp_id)
                if calculate_apsd:
                    video_filename+='_APSD'
                else:
                    video_filename+='_CPSD'
                    if plot_phase:
                        video_filename+='_PHASE'
                    else:
                        video_filename+='_ABS'
                video_filename+='_'+str(time_range[0])+'_'+str(time_range[1])
                if reference_pixel is not None:
                    video_filename+='_PIX_'+str(reference_pixel[0])+'_'+str(reference_pixel[1])
                if reference_position is not None:
                    video_filename+='_POS_'+str(reference_position[0])+'_'+str(reference_position[1])
                if reference_flux is not None:
                    video_filename+='_FLX_'+str(reference_flux[0])+'_'+str(reference_flux[1])
                video_filename+='_FRES_'+str(fres)
                if comment is not None:
                    video_filename+=comment
                video_filename+='.mp4'            
            else:
                video_filename='NSTX_GPI_CPSD_'+str(exp_id)+'_FULL.mp4'      
        if video_saving_only:
            import matplotlib
            current_backend=matplotlib.get_backend()
            matplotlib.use('agg')
            waittime=0.
        else:
            waittime=1.
            
        if calculate_apsd:
            object_name='GPI_APSD'
        else:
            if plot_phase:
                object_name='GPI_CPSD_PHASE'
            else:
                object_name='GPI_CPSD_ABS'
                
        flap.plot(object_name, exp_id=exp_id,
                  plot_type='anim-contour', 
                  axes=axes, 
                  options={'Force axes':True,
                           'Colormap':colormap,
                           'Plot units':{'Device R':'mm',
                                         'Device z':'mm',
                                         },
                           'Waittime':waittime,
                           'Video file':video_filename,
                           'Video format':'mp4',
                           'Log z':zlog})
                           
            
        if video_saving_only:
            import matplotlib
            matplotlib.use(current_backend)

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
                                        plot_acf=False,
                                        axes=['Image x', 'Image y', 'Time lag']
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
        normalizer=flap_nstx.tools.calculate_nstx_gpi_norm_coeff(exp_id=exp_id,             # Experiment ID
                                                                     f_high=1e2,                # Low pass filter frequency in Hz
                                                                     design=filter_design,      # IIR filter design (from scipy)
                                                                     test=False,                # Testing input
                                                                     filter_data=True,          # IIR LPF the data
                                                                     time_range=None,           # Timer range for the averaging in ms [t1,t2]
                                                                     calc_around_max=False,     # Calculate the average around the maximum of the GPI signal
                                                                     time_window=50.,           # The time window for the calc_around_max calculation
                                                                     cache_data=True,           
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
        flap_nstx.tools.calculate_nstx_gpi_reference('GPI_SLICED_FILTERED', exp_id=exp_id,
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
                      axes=axes, 
                      options={'Plot units': {'Time lag':'us'}, 
                               'Z range':[0,1]},)     