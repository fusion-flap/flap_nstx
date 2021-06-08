#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  5 01:50:16 2021

@author: mlampert
"""
#Core modules
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

#Scientific modules

import matplotlib.pyplot as plt

import numpy as np
import scipy

#Plot settings for publications
publication=False

def calculate_tde_velocity(data_object,                 #Shot number
                           time_range=None,             #Time range of the calculation
                           time_res=None,             #Time resolution of the calculation
                           xrange=None,               #Range of the calculation (Default: entire data range)
                           yrange=None,              #Range of the calculation (Default: entire data range)
                           
                           x_direction=False,                #Calculate the radial velocity
                           y_direction=False,              #Calculate the poloidal velocity
                           
                           filter_data=True,
                           f_low=None,                  #Highpass filtering of the signal (Bandpass if f_high is set, as well.)
                           f_high=None,                 #Lowpass filtering of the signal (Bandpass if f_low is set, as well.)
                           
                           taurange=None,               #Range of the CCF time delay calculation (Default: +-time_res)
                           taures=None,               #Resoluion of the CCF calculation (Default: sampling time)
                           correct_acf_peak=False,
                           interval_n=1,                #Number of intervals the signal is split to
                           
                           
                           time_delay=False,            # Show the time delay instead of the velocity
                           
                           cache_data=True,             #Cache the signal
                           
                           pdf=True,                    #Save the plots into a pdf
                           correlation_threshold=0.6,   #Threshold for the correlation calculation
                           correlation_length=False,
                           
                           plot=True,
                           nocalc=False,                 #Load the results from a file
                           save_data=False,
                           ):
    
    """
    The code calculates a 1D-1D (pol/rad vs. rad/pol vs. time) velocity evolution.
    It slices the data to the given ranges and then calculates the velocity
    distribution in the range with cross-correlation based time delay estimation.
    vica versa. It's time resolution is limited to around 40frames.
    If poloidal velocity is needed, it's radial distribution is calculated and
    """

    if type(data_object) is str:
        d=flap.get_data_object_ref(data_object)
        flap.add_data_object(d,'DATA')
    else:
        try:
            d=copy.deepcopy(data_object)
            flap.add_data_object(d, 'DATA')
        except:
            raise ValueError('Input object needs to be either string or flap.DataObject')
            
    exp_id=d.exp_id
    
    if time_range is None:
        time_range=[d.coordinate('Time')[0].min(),
                    d.coordinate('Time')[0].max()]
        
    if xrange is None:
        xrange=[d.coordinate('Image x')[0].min(),
                d.coordinate('Image x')[0].max()]
        
    if yrange is None:
        yrange=[d.coordinate('Image y')[0].min(),
                d.coordinate('Image y')[0].max()]

    if time_res is None:
        time_res=time_range[1]-time_range[0]/2
    
    if taurange is None:
        taurange = [-time_res/interval_n/2,time_res/interval_n/2]
    
    # if taures is None:
    #     taures=time_res
        
    if time_res/interval_n < (taurange[1]-taurange[0]):
        raise ValueError('The time resolution divided by the interval number is smaller than the taurange of the CCF calculation.')
        
    flap.slice_data('DATA', exp_id=exp_id, 
                    slicing={'Image x':flap.Intervals(xrange[0],xrange[1]),
                             'Image y':flap.Intervals(yrange[0],yrange[1]),
                             'Time':flap.Intervals(time_range[0],time_range[1])}, 
                    output_name='DATA_SLICED_FULL')
    
    passed_object_name='DATA_SLICED_FULL'
    
    if filter_data:
        print("*** Filtering the data ***")
        if f_low is not None and f_high is None:
            d=flap.filter_data('DATA_SLICED_FULL',
                               exp_id=exp_id,
                               coordinate='Time',
                               options={'Type':'Highpass',
                                        'f_low':f_low,
                                        'Design':'Chebyshev II'},
                               output_name='DATA_FILTERED')

        if f_low is None and f_high is not None:
            d=flap.filter_data('DATA_SLICED_FULL',
                               exp_id=exp_id,
                               coordinate='Time',
                               options={'Type':'Lowpass',
                                        'f_high':f_high,
                                        'Design':'Chebyshev II'},
                               output_name='DATA_FILTERED')

        if f_low is not None and f_high is not None:
            d=flap.filter_data('DATA_SLICED_FULL',
                               exp_id=exp_id,
                               coordinate='Time',
                               options={'Type':'Bandpass',
                                        'f_low':f_low,
                                        'f_high':f_high,
                                        'Design':'Chebyshev II'},
                               output_name='DATA_FILTERED')
        passed_object_name='DATA_FILTERED'
        
        d.data=np.asarray(d.data,dtype='float32')
        
    n_time=int((time_range[1]-time_range[0])/time_res)
    time_window_vector=np.linspace(time_range[0]+time_res/2,time_range[1]-time_res/2,n_time)
    
        
    if x_direction:
        velocity_matrix=np.zeros([yrange[1]-yrange[0],n_time])
        correlation_length_matrix=np.zeros([yrange[1]-yrange[0],n_time])
        
    if y_direction:
        velocity_matrix=np.zeros([xrange[1]-xrange[0],n_time])
        correlation_length_matrix=np.zeros([xrange[1]-xrange[0],n_time])
        
    for i_time in range(n_time):
        time_window=[time_range[0]+i_time*time_res,
                     time_range[0]+(i_time+1)*time_res]
        
        if x_direction:
            slicing={'Time':flap.Intervals(time_window[0],time_window[1]),
                     'Image x':np.mean(xrange)}
            
        if y_direction:
            slicing={'Time':flap.Intervals(time_window[0],time_window[1]),
                     'Image y':np.mean(yrange)}

        flap.slice_data(passed_object_name,
                        exp_id=exp_id,
                        slicing=slicing, 
                        output_name='DATA_WINDOW_2')
        
        flap.slice_data(passed_object_name,
                        exp_id=exp_id,
                        slicing={'Time':flap.Intervals(time_window[0],time_window[1])},
                        output_name='DATA_WINDOW')
        
        if x_direction:
            slicing_range=yrange[1]-yrange[0]
            
        if y_direction:
            slicing_range=xrange[1]-xrange[0]
            
        for j_range in range(slicing_range):
            
            if x_direction:
                slicing={'Image y':yrange[0]+j_range}
            if y_direction:
                slicing={'Image x':xrange[0]+j_range}
            flap.slice_data('DATA_WINDOW',
                            exp_id=exp_id,
                            slicing=slicing,
                            output_name='DATA_WINDOW_1')
            
            flap.slice_data('DATA_WINDOW_2',
                            exp_id=exp_id, 
                            slicing=slicing, 
                            output_name='DATA_WINDOW_REF_12')
            
            ccf=flap.ccf('DATA_WINDOW_1',exp_id=exp_id,
                         ref='DATA_WINDOW_REF_12',
                         coordinate='Time',
                         options={'Resolution':taures,
                                  'Range':taurange,
                                  'Trend':['Poly',2],
                                  'Interval':interval_n,
                                  'Correct ACF peak':correct_acf_peak,
                                  'Normalize':True,
                                  },
                         output_name='DATA_CCF_SLICE')
            
            time_index=ccf.get_coordinate_object('Time lag').dimension_list[0]
            index_time=[0]*2
            index_time[time_index]=Ellipsis
            ccf_time_lag=ccf.coordinate('Time lag')[0][tuple(index_time)] #Time lag is the second coordinate in the dimension list.
            ccf_max_time_lag=np.zeros(ccf.data.shape[0])
            ccf_max_correlation=np.zeros(ccf.data.shape[0])
            
            maxrange=Ellipsis
            time_lag_vector=ccf.coordinate('Time lag')[0][0,:]
            
            if x_direction:
                displacement_vector=ccf.coordinate('Device R')[0][:,0]            
                
            if y_direction:
                displacement_vector=ccf.coordinate('Device z')[0][:,0]
                
            for i_shape_1 in range(ccf.data.shape[0]):
                max_time_ind=np.argmax(ccf.data[i_shape_1,maxrange])
                ind_window=3
                ind=[max_time_ind-ind_window,max_time_ind+ind_window+1]
                if ind[0] < 0:
                    ind[0]=0
                if ind[1] >= ccf_time_lag.shape[0]:
                    ind[1]=ccf_time_lag.shape[0]-1
                    
                indrange=slice(ind[0],ind[1])
                #Fitting a second order polynom on the peak
                coeff=np.polyfit(time_lag_vector[indrange],
                                 ccf.data[i_shape_1,indrange],2)
                
                max_correlation=coeff[2]-coeff[1]**2/(4*coeff[0])
                max_time_lag=-coeff[1]/(2*coeff[0])
                if max_correlation < correlation_threshold:
                    ccf_max_time_lag[i_shape_1]=np.nan
                else:
                    ccf_max_time_lag[i_shape_1]=max_time_lag
                ccf_max_correlation[i_shape_1]=max_correlation
                # p0 is the initial guess for the fitting coefficients (A, mu and sigma above)
                
            ind_not_nan=np.logical_not(np.isnan(ccf_max_time_lag))
            # displacement=(displacement_vector-displacement_vector[0])
            # time_lag=(ccf_max_time_lag-ccf_max_time_lag[0])
            #DIRECT calculation:
            try:
            #if True:
                # coeff=np.polyfit(time_lag[ind_not_nan], displacement[ind_not_nan], 1)
                coeff=np.polyfit(ccf_max_time_lag[ind_not_nan], displacement_vector[ind_not_nan], 1)
                velocity=coeff[0]            
            except:
                 coeff=[0,0]
                 velocity=0.
            
            if time_delay:
                velocity_matrix[j_range,i_time]=np.mean(ccf_max_time_lag[ind_not_nan][1:-1]-ccf_max_time_lag[ind_not_nan][0:-2])
            else:        
                velocity_matrix[j_range,i_time]=velocity
                
            if correlation_length: #DEPRECATED, DOESN'T WORK RELIABLY
                try:
                    def gauss(x, *p):
                        A, mu, sigma = p
                        return A*np.exp(-(x-mu)**2/(2.*sigma**2))
                    
                    coeff, var_matrix = scipy.optimize.curve_fit(gauss, 
                                                                 displacement_vector, 
                                                                 ccf_max_correlation, 
                                                                 p0=[np.max(ccf_max_correlation), np.argmin(np.abs(ccf_max_correlation) - np.max(ccf_max_correlation)), 0.2]) 
                    correlation_length_matrix[j_range,i_time]=2.3548*coeff[2]
                except:
                    correlation_length_matrix[j_range,i_time]=0.

        print('Calculation done: '+str((i_time+1)/n_time*100)+'%')
        
    r_coordinate_vector=flap.slice_data(passed_object_name,
                                        exp_id=exp_id,
                                        slicing={'Time':flap.Intervals(time_window[0],time_window[1]),
                                                 'Image y':np.mean(yrange)}, 
                                        output_name='DATA_WINDOW_Y').coordinate('Device R')[0][0,:]

    z_coordinate_vector=flap.slice_data(passed_object_name,
                                        exp_id=exp_id,
                                        slicing={'Time':    flap.Intervals(time_window[0],time_window[1]),
                                                 'Image x':np.mean(xrange)}, 
                                        output_name='DATA_WINDOW_Y').coordinate('Device z')[0][0,:]
    coord=[]
    
    coord.append(copy.deepcopy(flap.Coordinate(name='Time',
                                            unit='s',
                                            mode=flap.CoordinateMode(equidistant=True),
                                            start=time_window_vector[0],
                                            step=time_window_vector[1]-time_window_vector[0],
                                            dimension_list=[1]
                                            )))
    
    coord.append(copy.deepcopy(flap.Coordinate(name='Device R',
                                            unit='m',
                                            mode=flap.CoordinateMode(equidistant=True),
                                            start=r_coordinate_vector[0],
                                            step=r_coordinate_vector[1]-r_coordinate_vector[0],
                                            dimension_list=[0]
                                            )))
        
    coord.append(copy.deepcopy(flap.Coordinate(name='Device z',
                                            unit='m',
                                            mode=flap.CoordinateMode(equidistant=True),
                                            start=z_coordinate_vector[0],
                                            step=z_coordinate_vector[1]-z_coordinate_vector[0],
                                            dimension_list=[1]
                                            )))
            
    if y_direction:
        data_title='Estimated poloidal velocity'
    if x_direction:
        data_title='Estimated radial velocity'
    d_vel = flap.DataObject(data_array=velocity_matrix,
                            data_unit=flap.Unit(name='Velocity',unit='m/s'),
                            coordinates=coord,
                            exp_id=exp_id,
                            data_title=data_title,
                            info='',
                            data_source="NSTX_GPI")
    if x_direction:
        flap.add_data_object(d_vel, 'DATA_RAD_VELOCITY')
    if y_direction:
        flap.add_data_object(d_vel, 'DATA_POL_VELOCITY')
        
    if correlation_length:
        d_len = flap.DataObject(data_array=correlation_length_matrix,
                                data_unit=flap.Unit(name='Correlation length',unit='m'),
                                coordinates=coord,
                                exp_id=exp_id,
                                data_title=data_title,
                                info='',
                                data_source="NSTX_GPI")
        flap.add_data_object(d_len, 'DATA_POL_CORR_LEN')
    if plot:
        plt.figure()
        if y_direction:
            flap.plot('DATA_POL_VELOCITY', 
                      exp_id=exp_id,
                      plot_type='contour', 
                      axes=['Time', 'Device R'],
                      plot_options={'levels':51}, 
                      options={'Colormap':'gist_ncar'})
            plt.title('Poloidal velocity of '+str(exp_id)+' @ ['+str(time_range[0])+','+str(time_range[1])+']s')
            
            if correlation_length:
                plt.figure()
                flap.plot('DATA_POL_CORR_LEN',
                          exp_id=exp_id,
                          plot_type='contour', 
                          axes=['Time', 'Device R'],
                          plot_options={'levels':51}, 
                          options={'Colormap':'gist_ncar','Z range':[0,0.4]})
                plt.title('Poloidal correlation length of '+str(exp_id)+' @ ['+str(time_range[0])+','+str(time_range[1])+']s')
                
        if x_direction:
            flap.plot('DATA_RAD_VELOCITY', 
                      exp_id=exp_id,
                      plot_type='contour', 
                      axes=['Time', 'Device z'],
                      plot_options={'levels':51}, 
                      options={'Colormap':'gist_ncar'})
            plt.title('Radial velocity of '+str(exp_id)+' @ ['+str(time_range[0])+','+str(time_range[1])+']s')
            
            if correlation_length:
                plt.figure()
                flap.plot('DATA_RAD_CORR_LEN', 
                          exp_id=exp_id,
                          plot_type='contour', 
                          axes=['Time', 'Device z'],
                          plot_options={'levels':51}, 
                          options={'Colormap':'gist_ncar','Z range':[0,0.4]})
                plt.title('Radial correlation length of '+str(exp_id)+' @ ['+str(time_range[0])+','+str(time_range[1])+']s')
    if save_data:
        if x_direction:
            filename=flap_nstx.analysis.filename(exp_id=exp_id, 
                                                 time_range=time_range,
                                                 purpose='TDE radial velocity',
                                                 extension='pickle')
            flap.save(flap.get_data_object_ref('DATA_RAD_VELOCITY'), filename=filename)
        if y_direction:
            filename=flap_nstx.analysis.filename(exp_id=exp_id, 
                                                 time_range=time_range,
                                                 purpose='TDE poloidal velocity',
                                                 extension='pickle')
            flap.save(flap.get_data_object_ref('DATA_POL_VELOCITY'), filename=filename)