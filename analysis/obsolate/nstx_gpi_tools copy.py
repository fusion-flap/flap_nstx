#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 13:37:37 2019

@author: mlampert
"""

import os
import flap
import flap_nstx
import matplotlib.pyplot as plt

import numpy as np

flap_nstx.register()
def calculate_nstx_gpi_norm_coeff(exp_id=None,              # Experiment ID
                                  f_high=1e2,               # Low pass filter frequency in Hz
                                  design='Chebyshev II',    # IIR filter design (from scipy)
                                  test=False,               # Testing input
                                  filter_data=True,         # IIR LPF the data
                                  time_range=None,          # Timer range for the averaging in ms [t1,t2]
                                  calc_around_max=False,    # Calculate the average around the maximum of the GPI signal
                                  time_window=50.,          # The time window for the calc_around_max calculation
                                  cache_data=True,          #
                                  verbose=False,
                                  output_name='GPI_NORMALIZER',
                                  #add_flux_r=False,
                                  ):
    
    #This function calculates the GPI normalizer image with which all the GPI
    #images should be divided. Returns a flap data object. The inputs are 
    #expained next to the inputs.

    normalizer_options={'LP freq':f_high,
                        'Filter design': design,
                        'Filter data':filter_data,
                        'Time range':time_range,
                        'Calc around max': calc_around_max,
                        'Time window': time_window,
                        #'Flux R': add_flux_r,
                        }

    if not cache_data:
        flap.delete_data_object(output_name,
                                'GPI_*_FILTERED_*',
                                'GPI_MEAN')

    if calc_around_max and time_range is not None:
        print('Both calc_around_max and time_range cannot be set.')
        print('Setting calc_around_max to False')
        calc_around_max=False

    #Get the data from the cine file
    if exp_id is not None:
        try:
            flap.get_data_object_ref(exp_id=exp_id,object_name='GPI')
        except:
            if verbose or test:
                print('Data is not cached, it needs to be read.')
                print("\n------- Reading NSTX GPI data --------")
            flap.get_data('NSTX_GPI',exp_id=exp_id,name='',object_name='GPI')
        object_name='GPI'
        #if add_flux_r:
        #    flap.add_coordinate(object_name, exp_id=exp_id, coordinates='Flux r')
    else:
        raise ValueError('The experiment ID needs to be set.')
        
    if time_range is not None:
        sliced_object_name=object_name+'_'+str(time_range[0])+'_'+str(time_range[1])
        try:
            flap.get_data_object_ref(exp_id=exp_id,object_name=sliced_object_name)
        except:
            flap.slice_data(object_name,exp_id=exp_id,
                            slicing={'Time':flap.Intervals(time_range[0],time_range[1])}, 
                            output_name=object_name+'_'+str(time_range[0])+'_'+str(time_range[1])
                            )
        object_name=sliced_object_name
        
        
    #Highpass filter the data to get rid of the spikes
    if filter_data:
        filtered_data_object_name=object_name+'_FILTERED_LP_'+str(f_high)+'_'+design.replace(' ','')
        try:
            flap.get_data_object_ref(exp_id=exp_id,object_name=filtered_data_object_name)
        except:
            if verbose or test:
                print('Filtered data is not cached, it needs to be filtered.')
                print("\n------- Filtering NSTX GPI data --------")
            flap.filter_data(object_name,exp_id=exp_id,
                             coordinate='Time',
                             options={'Type':'Lowpass',
                                      'f_high':f_high, 
                                      'Design':design},
                                      output_name=filtered_data_object_name)
        object_name=filtered_data_object_name
        
    if calc_around_max and time_range is None:
        #Calculate the average image for a time window around the maximum signal
        d=flap.slice_data(object_name,
                          summing={'Image x':'Mean','Image y':'Mean'},
                          output_name='GPI_MEAN')

        max_time_index=np.argmax(d.data)
        max_time=d.coordinate('Time')[0][max_time_index]
        flap.slice_data(object_name,exp_id=exp_id,
                        slicing={'Time':flap.Intervals(max_time-time_window,max_time+time_window)},
                        summing={'Time':'Mean'},
                        output_name=output_name)
    else:
        #Calculate the average image for the entire shot
        d=flap.slice_data(object_name,exp_id=exp_id,
                        summing={'Time':'Mean'},
                        output_name=output_name)
    object_name=output_name
    
    d.info['Normalizer options']=normalizer_options
    
    if test:
        plt.figure()
#        if False:
#        if add_flux_r:
#            flap.plot(object_name, 
#                      axes=['Flux r', 'Device z'], 
#                      exp_id=exp_id, 
#                      plot_type='contour',
#                      plot_options={'levels':21}
#                      )
#        else:
        flap.plot(object_name, 
                  axes=['Device R', 'Device z'], 
                  exp_id=exp_id, 
                  plot_type='contour',
                  plot_options={'levels':21}
                  )
    return d

def calculate_nstx_gpi_reference(object_name=None,
                                 exp_id=None,
                                 time_range=None,
                                 reference_pixel=None,
                                 reference_area=None,
                                 reference_position=None,
                                 reference_flux=None,
                                 filter_low=None,
                                 filter_high=None,
                                 filter_design='Chebyshev II',
                                 output_name=None
                                 ):
    try:
        input_object=flap.get_data_object_ref(object_name, exp_id=exp_id)
    except:
        raise IOError('The given object_name doesn\'t exist in the FLAP storage.')
    if output_name is None:
        output_name=object_name+'_REF'
        
    if reference_pixel is None and reference_position is None and reference_flux is None:
        raise ValueError('There is no reference given. Please set reference_pixel or reference_position or reference_flux.')
        
    if filter_low is not None or filter_high is not None:
        if filter_low is not None and filter_high is None:
            filter_type='Highpass'
        if filter_low is None and filter_high is not None:
            filter_type='Lowpass'
        if filter_low is not None and filter_high is None:
            filter_type='Bandpass'
        
        flap.filter_data(object_name,exp_id=exp_id,
                         coordinate='Time',
                         options={'Type':filter_type,
                                  'f_low':filter_low,
                                  'f_high':filter_high, 
                                  'Design':filter_design},
                         output_name=object_name+'_FILTERED')
        object_name=object_name+'_FILTERED'
    slicing_dict={}
    
    if time_range is not None:
        slicing_dict['Time']=flap.Intervals(time_range[0],time_range[1])
        
    if reference_pixel is not None:
        #Single pixel correlation
        if reference_area is None:
            slicing_dict['Image x']=reference_pixel[0]
            slicing_dict['Image y']=reference_pixel[1]
            summing_dict=None
        else:
            if type(reference_area) is not list:
                reference_area=[reference_area,reference_area]
            #Handling the edges:
            if reference_pixel[0]-reference_area[0] < 0:
                reference_pixel[0]=reference_area[0]
            if reference_pixel[1]-reference_area[0] < 0:
                reference_pixel[1]=reference_area[0]
                
            if reference_pixel[0]+reference_area[1] > input_object.data.shape[1]:
                reference_pixel[0]=input_object.data.shape[1]-reference_area[1]
            if reference_pixel[1]+reference_area[1] > input_object.data.shape[2]:
                reference_pixel[1]=input_object.data.shape[2]-reference_area[1]
                
            slicing_dict['Image x']=flap.Intervals(reference_pixel[0]-reference_area[0],
                                                   reference_pixel[0]+reference_area[0])
            slicing_dict['Image y']=flap.Intervals(reference_pixel[1]-reference_area[1],
                                                   reference_pixel[1]+reference_area[1])
            summing_dict={'Image x':'Mean', 'Image y':'Mean'}

    if reference_position is not None:
        if reference_area is None:
            try:
                slicing_dict['Device R']=reference_position[0]
                slicing_dict['Device z']=reference_position[1]
                summing_dict=None
            except:
                raise ValueError('Reference position is outside the measurement range.')
        else:
            if type(reference_area) is not list:
                reference_area=[reference_area,reference_area]
            try:
            #Multiple pixel correlation (averaged)
                slicing_dict['Device R']=flap.Intervals(reference_position[0]-reference_area[0],
                                                        reference_position[0]+reference_area[0])
                slicing_dict['Device z']=flap.Intervals(reference_position[1]-reference_area[1],
                                                        reference_position[1]+reference_area[1])
                summing_dict={'Device R':'Mean', 'Device z':'Mean'}
            except:
                raise ValueError('Reference position is outside the measurement range.')
                
    if reference_flux is not None:
        if len(reference_flux) != 2:
            raise ValueError('The reference position needs to be a 2 element list (Psi,z).')
        if reference_area is None:
            try:
                slicing_dict['Flux r']=flap.Intervals(reference_flux[0])
                slicing_dict['Device z']=flap.Intervals(reference_flux[1])
                summing_dict=None
            except:
                raise ValueError('Reference position is outside the measurement range.')
        else:
            if len(reference_area) !=2:
                 raise ValueError('The reference area needs to be a 2 element list (Psi,z).')
            try:
            #Multiple pixel correlation (averaged)
                slicing_dict['Flux r']=flap.Intervals(reference_flux[0]-reference_area[0],
                                                      reference_flux[0]+reference_area[0])
                slicing_dict['Device z']=flap.Intervals(reference_flux[1]-reference_area[1],
                                                        reference_flux[1]+reference_area[1])
                summing_dict={'Flux r':'Mean', 'Device z':'Mean'}  
            except:
                raise ValueError('Reference position is outside the measurement range.')    
                
    reference_signal=flap.slice_data(object_name, exp_id=exp_id,
                                     slicing=slicing_dict,
                                     summing=summing_dict,
                                     output_name=output_name)
    return reference_signal
        
thisdir = os.path.dirname(os.path.realpath(__file__))
fn = os.path.join(thisdir,"flap_nstx.cfg")
flap.config.read(file_name=fn)       