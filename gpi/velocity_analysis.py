#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 16:31:33 2019

@author: mlampert
"""
#Core modules
import os
import copy
import cv2

import flap
import flap_nstx
flap_nstx.register()
import flap_mdsplus
flap_mdsplus.register('NSTX_MDSPlus')

thisdir = os.path.dirname(os.path.realpath(__file__))
fn = os.path.join(thisdir,"../flap_nstx.cfg")
flap.config.read(file_name=fn)

#Scientific modules
import matplotlib.style as pltstyle
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import numpy as np
import scipy
import pickle
#Plot settings for publications
publication=False

if publication:

    plt.rcParams['lines.linewidth'] = 4
    plt.rcParams['axes.linewidth'] = 4
    plt.rcParams['axes.labelsize'] = 28
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



def calculate_nstx_gpi_avg_velocity(exp_id=None,
                                    time_range=None,
                                    xrange=[0,63],
                                    yrange=[10,70],
                                    frange=[1e3,100e3],
                                    taurange=[-500e-6,500e-6],
                                    taures=2.5e-6,
                                    interval_n=1,
                                    radial=False,
                                    poloidal=False,
                                    cache_data=True,
                                    pdf=True,
                                    ):
    
    """ The code calculates the fluctuation velocity from the slope of the time
    lag between a reference channel (middle of the range) and all the other pixels.
    The data is filtered to frange first, then the cross-correlation functions are 
    calculated between a row or line of pixels crossing the average x or y range 
    depending on the radial/poloidal setting. The velocity is calculated from the 
    channel separation and the slope of the fitted line.
    
    INPUTs:
        exp_id: shot number
        time_range: time range of the calculation in seconds
        xrange: the horizontal range for calculating the cross-correlation functions
        yrange: the vertical range for calculating the cross-correlation functions
        frange: frequency range of the data filtering in Hz
        taurange: the time lag range of the cross-correlation function calculations
        taured: the resolution of the time lag calculation.
        interval_n: the number of intervals the time_range is divided to
        radial: set True if radial velocity is to be calculated
        poloidal: set True if poloidal velocity is to be calculated
        cache_data: try to read the data objects from earlier calculations
        pdf: export the resulting plots into a pdf (NOTIMPLEMENTED)
        
    
    Caveats:
        - No correlation coefficient check is performed, this is only a crude calculation.
        It gives reasonable results, but random noise can influence the outcome
        to some extent.
        - Only pixel coordinates are considered for the directions, slicing
        along spatial or magnetic coordinates would require the cross-correlation
        functions to be calculated between each and other pixels which is not
        a memory efficient process.
    """
    
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
        
    try:
        d.get_coordinate_object('Flux r')
    except:
        d.add_coordinate(coordinates='Flux r',exp_id=exp_id)
        average_flux_coordinates=np.mean(d.coordinate('Flux r')[0], axis=0)
        d.coordinates.append(copy.deepcopy(flap.Coordinate(name='Flux r avg',
                                                           unit='',
                                                           mode=flap.CoordinateMode(equidistant=False),
                                                           values=average_flux_coordinates,
                                                           shape=average_flux_coordinates.shape,
                                                           dimension_list=[1,2]
                                                           )))
    flap.slice_data('GPI', exp_id=exp_id, 
                    slicing={'Image x':flap.Intervals(xrange[0],xrange[1]),
                             'Image y':flap.Intervals(yrange[0],yrange[1]),
                             'Time':flap.Intervals(time_range[0],time_range[1])}, 
                    output_name='GPI_SLICED_FULL')

    print("*** Filtering the data ***")
    flap.filter_data('GPI_SLICED_FULL',exp_id=exp_id,
                     coordinate='Time',
                     options={'Type':'Bandpass',
                              'f_low':frange[0],
                              'f_high':frange[1],
                              'Design':'Chebyshev II'},
                     output_name='GPI_FILTERED')
    if poloidal:
        flap.slice_data('GPI_FILTERED', slicing={'Image y':int(np.mean(yrange))}, output_name='GPI_REF')
    if radial:
        flap.slice_data('GPI_FILTERED', slicing={'Image x':int(np.mean(xrange))}, output_name='GPI_REF')
    print("*** Doing the crosscorrelation function calculation ***")
    ccf=flap.ccf('GPI_FILTERED',exp_id=exp_id,
                 ref='GPI_REF',
                 coordinate='Time',
                 options={'Resolution':taures,
                          'Range':taurange,
                          'Trend':['Poly',2],
                          'Interval':interval_n,
                          'Normalize':True,
                          },
                 output_name='GPI_CCF_SLICE')
    time_index=ccf.get_coordinate_object('Time lag').dimension_list[0]
    index_time=[0]*4
    index_time[time_index]=Ellipsis
    ccf_time_lag=ccf.coordinate('Time lag')[0][tuple(index_time)] #Time lag is the second coordinate in the dimension list.
    ccf_max_time_lag=np.zeros(ccf.data.shape)[:,:,0,:]
    maxrange=Ellipsis
    time_lags=ccf.coordinate('Time lag')[0]
    for i_shape_1 in range(ccf.data.shape[0]):
        for j_shape_2 in range(ccf.data.shape[1]):
            for k_shape_3 in range(ccf.data.shape[3]):
                max_time_ind=np.argmax(ccf.data[i_shape_1,j_shape_2,maxrange,k_shape_3])
                ind=[max_time_ind-3,max_time_ind+4]
                if max_time_ind-3 < 0:
                    ind[0]=0
                if max_time_ind+4 >= ccf_time_lag.shape[0]:
                    ind[1]=ccf_time_lag.shape[0]-1
                indrange=slice(ind[0],ind[1])
                #Fitting a second order polynom on the peak
                coeff=np.polyfit(time_lags[i_shape_1,j_shape_2,indrange,k_shape_3],ccf.data[i_shape_1,j_shape_2,indrange,k_shape_3],2)
                ccf_max_time_lag[i_shape_1,j_shape_2, k_shape_3]=-coeff[1]/(2*coeff[0])
                #ccf_max_time_lag[i_shape_1,j_shape_2, k_shape_3]=time_lags[i_shape_1,j_shape_2,max_time_ind,k_shape_3]
    ccf_max=flap.slice_data('GPI_CCF_SLICE', slicing={'Time lag': 0}, output_name='GPI_CCF_MAX')
    ccf_max.data=ccf_max_time_lag
    plt.figure()
    if poloidal:
        velocity_data=np.zeros(xrange[1]-xrange[0])
        time_lag=np.zeros(xrange[1]-xrange[0])
        for i_x in range(xrange[1]-xrange[0]):
            x_coordinate=xrange[0]+i_x
            ccf_slice=flap.slice_data('GPI_CCF_MAX', slicing={'Image x (Ref)':x_coordinate, 
                                                              'Image x':x_coordinate}, 
                                      output_name='GPI_CCF_MAX_SLICE')
            z_coord=ccf_slice.coordinate('Device z')[0]
            #displacement=np.sqrt((r_coord-r_coord[0])**2 +(z_coord-z_coord[0])**2)
            displacement=(z_coord-z_coord[0])
            time_lag=(ccf_slice.data-ccf_slice.data[0])
            coeff=np.polyfit(time_lag, displacement, 1)
            #plt.plot(time_lag, displacement)
            #plt.pause(0.1)
            velocity_data[i_x]=coeff[0]
        plt.cla()
        plt.plot(flap.get_data_object('GPI_REF').coordinate('Flux r avg')[0][0,:], velocity_data)
        plt.xlabel('PSI norm')
        plt.ylabel('v_pol [m/s]')
        plt.title('PSI_norm vs. v_pol '+str(exp_id)+' @ ['+str(time_range[0])+','+str(time_range[1])+']s')
    if radial:
        velocity_data=np.zeros(yrange[1]-yrange[0])
        time_lag=np.zeros(yrange[1]-yrange[0])
        for i_y in range(yrange[1]-yrange[0]):
            y_coordinate=yrange[0]+i_y
            ccf_slice=flap.slice_data('GPI_CCF_MAX', slicing={'Image y (Ref)':y_coordinate, 
                                                              'Image y':y_coordinate}, 
                                      output_name='GPI_CCF_MAX_SLICE')
            r_coord=ccf_slice.coordinate('Device R')[0]
            #displacement=np.sqrt((r_coord-r_coord[0])**2 +(z_coord-z_coord[0])**2)
            displacement=(r_coord-r_coord[0])
            time_lag=(ccf_slice.data-ccf_slice.data[0])
            coeff=np.polyfit(time_lag, displacement, 1)
            #plt.plot(time_lag, displacement)
            #plt.pause(0.1)
            velocity_data[i_y]=coeff[0]
        plt.cla()
        plt.plot(flap.get_data_object('GPI_REF').coordinate('Device z')[0][0,:], velocity_data)
        plt.xlabel('z [m]')
        plt.ylabel('v_rad [m/s]')
        plt.title('z vs. v_rad '+str(exp_id)+' @ ['+str(time_range[0])+','+str(time_range[1])+']s')

def calculate_nstx_gpi_tde_velocity(exp_id=None,                 #Shot number
                                       time_range=None,             #Time range of the calculation
                                       time_res=100e-6,             #Time resolution of the calculation
                                       xrange=[0,63],               #Range of the calculation
                                       yrange=[10,70],              #Range of the calculation
                                       filter_data=True,
                                       f_low=None,                  #Highpass filtering of the signal (Bandpass if f_high is set, as well.)
                                       f_high=None,                 #Lowpass filtering of the signal (Bandpass if f_low is set, as well.)
                                       taurange=None,               #Range of the CCF time delay calculation
                                       taures=2.5e-6,               #Resoltuion of the CCF calculation
                                       interval_n=1,                #Number of intervals the signal is split to
                                       radial=False,                #Calculate the radial velocity
                                       poloidal=False,              #Calculate the poloidal velocity
                                       time_delay=False,            # Show the time delay instead of the velocity
                                       
                                       cache_data=True,             #Cache the signal
                                       pdf=True,                    #Save the plots into a pdf
                                       correlation_threshold=0.6,   #Threshold for the correlation calculation
                                       correlation_length=False,
                                       add_flux=False,
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
    if time_res is None:
        raise ValueError('The time resolution needs to be set for the calculation.')
    if taurange is None:
        taurange = [-time_res/interval_n/2,time_res/interval_n/2]
    if time_res/interval_n < (taurange[1]-taurange[0]):
        raise ValueError('The time resolution divided by the interval number is smaller than the taurange of the CCF calculation.')
    if add_flux:
        try:
            d.get_coordinate_object('Flux r')
        except:
            d.add_coordinate(coordinates='Flux r',exp_id=exp_id)
            average_flux_coordinates=np.mean(d.coordinate('Flux r')[0], axis=0)
            d.coordinates.append(copy.deepcopy(flap.Coordinate(name='Flux r avg',
                                                               unit='',
                                                               mode=flap.CoordinateMode(equidistant=False),
                                                               values=average_flux_coordinates,
                                                               shape=average_flux_coordinates.shape,
                                                               dimension_list=[1,2]
                                                               )))
    flap.slice_data('GPI', exp_id=exp_id, 
                    slicing={'Image x':flap.Intervals(xrange[0],xrange[1]),
                             'Image y':flap.Intervals(yrange[0],yrange[1]),
                             'Time':flap.Intervals(time_range[0],time_range[1])}, 
                    output_name='GPI_SLICED_FULL')
    if filter_data:
        print("*** Filtering the data ***")
        if f_low is not None and f_high is None:
            d=flap.filter_data('GPI_SLICED_FULL',
                               exp_id=exp_id,
                               coordinate='Time',
                               options={'Type':'Highpass',
                                        'f_low':f_low,
                                        'Design':'Chebyshev II'},
                               output_name='GPI_FILTERED')

        if f_low is None and f_high is not None:
            d=flap.filter_data('GPI_SLICED_FULL',
                               exp_id=exp_id,
                               coordinate='Time',
                               options={'Type':'Lowpass',
                                        'f_high':f_high,
                                        'Design':'Chebyshev II'},
                               output_name='GPI_FILTERED')
        if f_low is not None and f_high is not None:
            d=flap.filter_data('GPI_SLICED_FULL',
                               exp_id=exp_id,
                               coordinate='Time',
                               options={'Type':'Bandpass',
                                        'f_low':f_low,
                                        'f_high':f_high,
                                        'Design':'Chebyshev II'},
                               output_name='GPI_FILTERED')

        d.data=np.asarray(d.data,dtype='float32')
        
    n_time=int((time_range[1]-time_range[0])/time_res)
    time_window_vector=np.linspace(time_range[0]+time_res/2,time_range[1]-time_res/2,n_time)
    if poloidal:
        velocity_matrix=np.zeros([xrange[1]-xrange[0],n_time])
        correlation_length_matrix=np.zeros([xrange[1]-xrange[0],n_time])
    if radial:
        velocity_matrix=np.zeros([yrange[1]-yrange[0],n_time])
        correlation_length_matrix=np.zeros([yrange[1]-yrange[0],n_time])

    for i_time in range(n_time):
        time_window=[time_range[0]+i_time*time_res,
                     time_range[0]+(i_time+1)*time_res]
        if poloidal:
            slicing={'Time':flap.Intervals(time_window[0],time_window[1]),
                     'Image y':np.mean(yrange)}
        if radial:
            slicing={'Time':flap.Intervals(time_window[0],time_window[1]),
                     'Image x':np.mean(xrange)}
        flap.slice_data('GPI_FILTERED',
                        exp_id=exp_id,
                        slicing=slicing, 
                        output_name='GPI_WINDOW_2')
        flap.slice_data('GPI_FILTERED',
                        exp_id=exp_id,
                        slicing={'Time':flap.Intervals(time_window[0],time_window[1])},
                        output_name='GPI_WINDOW')
        if poloidal:
            slicing_range=xrange[1]-xrange[0]
        if radial:
            slicing_range=yrange[1]-yrange[0]
        for j_range in range(slicing_range):
            if poloidal:
                slicing={'Image x':xrange[0]+j_range}
            if radial:
                slicing={'Image y':yrange[0]+j_range}
            flap.slice_data('GPI_WINDOW',
                            exp_id=exp_id,
                            slicing=slicing,
                            output_name='GPI_WINDOW_1')
            flap.slice_data('GPI_WINDOW_2',
                            exp_id=exp_id, 
                            slicing=slicing, 
                            output_name='GPI_WINDOW_REF_12')
            ccf=flap.ccf('GPI_WINDOW_1',
                         exp_id=exp_id,
                         ref='GPI_WINDOW_REF_12',
                         coordinate='Time',
                         options={'Resolution':taures,
                                  'Range':taurange,
                                  'Trend':['Poly',2],
                                  'Interval':interval_n,
                                  'Correct ACF peak':True,
                                  'Normalize':True,
                                  },
                         output_name='GPI_CCF_SLICE')
            time_index=ccf.get_coordinate_object('Time lag').dimension_list[0]
            index_time=[0]*2
            index_time[time_index]=Ellipsis
            ccf_time_lag=ccf.coordinate('Time lag')[0][tuple(index_time)] #Time lag is the second coordinate in the dimension list.
            ccf_max_time_lag=np.zeros(ccf.data.shape[0])
            ccf_max_correlation=np.zeros(ccf.data.shape[0])
            maxrange=Ellipsis
            time_lag_vector=ccf.coordinate('Time lag')[0][0,:]
            
            if poloidal:
                displacement_vector=ccf.coordinate('Device z')[0][:,0]
            if radial:
                displacement_vector=ccf.coordinate('Device R')[0][:,0]
                
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
                coeff=np.polyfit(time_lag_vector[indrange],ccf.data[i_shape_1,indrange],2)
                max_correlation=coeff[2]-coeff[1]**2/(4*coeff[0])
                max_time_lag=-coeff[1]/(2*coeff[0])
                if max_correlation < correlation_threshold:
                    ccf_max_time_lag[i_shape_1]=np.nan
                else:
                    ccf_max_time_lag[i_shape_1]=max_time_lag
                ccf_max_correlation[i_shape_1]=max_correlation
                # p0 is the initial guess for the fitting coefficients (A, mu and sigma above)
                
            ind_not_nan=np.logical_not(np.isnan(ccf_max_time_lag))
            displacement=(displacement_vector-displacement_vector[0])
            time_lag=(ccf_max_time_lag-ccf_max_time_lag[0])
            #DIRECT calculation:
            try:
                coeff=np.polyfit(time_lag[ind_not_nan], displacement[ind_not_nan], 1)
                velocity=coeff[0]            
            except:
                coeff=[0,0]
                velocity=0.
            
            if time_delay:
                velocity_matrix[j_range,i_time]=np.mean(time_lag)
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
        
    r_coordinate_vector=flap.slice_data('GPI_FILTERED',
                                        exp_id=exp_id,
                                        slicing={'Time':flap.Intervals(time_window[0],time_window[1]),
                                                 'Image y':np.mean(yrange)}, 
                                        output_name='GPI_WINDOW_Y').coordinate('Device R')[0][0,:]
    if add_flux:
        flux_coordinate_vector=flap.slice_data('GPI_FILTERED',
                                               exp_id=exp_id,
                                               slicing={'Time':flap.Intervals(time_window[0],time_window[1]),
                                                        'Image y':np.mean(yrange)}, 
                                               output_name='GPI_WINDOW_Y').coordinate('Flux r avg')[0][0,:]
        z_coordinate_vector=flap.slice_data('GPI_FILTERED',
                                            exp_id=exp_id,
                                            slicing={'Time':flap.Intervals(time_window[0],time_window[1]),
                                                     'Image x':np.mean(xrange)}, 
                                            output_name='GPI_WINDOW_Y').coordinate('Flux r avg')[0][0,:]
    else:
        z_coordinate_vector=flap.slice_data('GPI_FILTERED',
                                            exp_id=exp_id,
                                            slicing={'Time':    flap.Intervals(time_window[0],time_window[1]),
                                                     'Image x':np.mean(xrange)}, 
                                            output_name='GPI_WINDOW_Y').coordinate('Device z')[0][0,:]
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
    if add_flux:
        coord.append(copy.deepcopy(flap.Coordinate(name='Flux r',
                                                unit='a.u.',
                                                mode=flap.CoordinateMode(equidistant=False),
                                                values=flux_coordinate_vector,
                                                shape=flux_coordinate_vector.shape,
                                                dimension_list=[0]
                                                )))
            
    if poloidal:
        data_title='Estimated poloidal velocity'
    if radial:
        data_title='Estimated radial velocity'
        
    d_vel = flap.DataObject(data_array=velocity_matrix,
                            data_unit=flap.Unit(name='Velocity',unit='m/s'),
                            coordinates=coord,
                            exp_id=exp_id,
                            data_title=data_title,
                            info='',
                            data_source="NSTX_GPI")
    if radial:
        flap.add_data_object(d_vel, 'NSTX_GPI_RAD_VELOCITY')
    if poloidal:
        flap.add_data_object(d_vel, 'NSTX_GPI_POL_VELOCITY')
    if correlation_length:
        d_len = flap.DataObject(data_array=correlation_length_matrix,
                                data_unit=flap.Unit(name='Correlation length',unit='m'),
                                coordinates=coord,
                                exp_id=exp_id,
                                data_title=data_title,
                                info='',
                                data_source="NSTX_GPI")
        flap.add_data_object(d_len, 'NSTX_GPI_POL_CORR_LEN')
    if plot:
        plt.figure()
        if poloidal:
            flap.plot('NSTX_GPI_POL_VELOCITY', 
                      exp_id=exp_id,
                      plot_type='contour', 
                      axes=['Time', 'Device R'],
                      plot_options={'levels':51}, 
                      options={'Colormap':'gist_ncar'})
            plt.title('Poloidal velocity of '+str(exp_id)+' @ ['+str(time_range[0])+','+str(time_range[1])+']s')
            
            if correlation_length:
                plt.figure()
                flap.plot('NSTX_GPI_POL_CORR_LEN',
                          exp_id=exp_id,
                          plot_type='contour', 
                          axes=['Time', 'Device R'],
                          plot_options={'levels':51}, 
                          options={'Colormap':'gist_ncar','Z range':[0,0.4]})
                plt.title('Poloidal correlation length of '+str(exp_id)+' @ ['+str(time_range[0])+','+str(time_range[1])+']s')
                
        if radial:
            flap.plot('NSTX_GPI_RAD_VELOCITY', 
                      exp_id=exp_id,
                      plot_type='contour', 
                      axes=['Time', 'Device z'],
                      plot_options={'levels':51}, 
                      options={'Colormap':'gist_ncar'})
            plt.title('Radial velocity of '+str(exp_id)+' @ ['+str(time_range[0])+','+str(time_range[1])+']s')
            
            if correlation_length:
                plt.figure()
                flap.plot('NSTX_GPI_RAD_CORR_LEN', 
                          exp_id=exp_id,
                          plot_type='contour', 
                          axes=['Time', 'Device z'],
                          plot_options={'levels':51}, 
                          options={'Colormap':'gist_ncar','Z range':[0,0.4]})
                plt.title('Radial correlation length of '+str(exp_id)+' @ ['+str(time_range[0])+','+str(time_range[1])+']s')
    if save_data:
        if radial:
            filename=flap_nstx.tools.filename(exp_id=exp_id, 
                                                 time_range=time_range,
                                                 purpose='TDE radial velocity',
                                                 extension='pickle')
            flap.save(flap.get_data_object_ref('NSTX_GPI_RAD_VELOCITY'), filename=filename)
        if poloidal:
            filename=flap_nstx.tools.filename(exp_id=exp_id, 
                                                 time_range=time_range,
                                                 purpose='TDE poloidal velocity',
                                                 extension='pickle')
            flap.save(flap.get_data_object_ref('NSTX_GPI_POL_VELOCITY'), filename=filename)
            
            
def calculate_nstx_gpi_filament_velocity(exp_id=None,                           #Shot number
                                         time_range=None,                       #The time range for the calculation
                                         #Inputs for preprocessing
                                         ref_pixel=[10,40],                     #Reference pixel for finding the filament peaks
                                         xrange=[0,32],                         #The xrange for the calculation in image x, if x_average, then the averaging range (also the range for peak finding)
                                         x_average=False,                       #Averaging in x direction, not effective when radial is set
                                         yrange=[10,70],                        #The yrange for the calculation in image y, if y_average, then the averaging range (also the range for peak finding)
                                         y_average=False,                       #Averaging in y direction, not effective when poloidal is set
                                         normalize=False,                       #Normalize the signal (should be implemented differently for poloidal and vertical)
                                         #Peak finding inputs:
                                         width_range=[1,30],                    #Setting for the peak finding algorithm
                                         vertical_sum=False,                    #Setting for the peak finding algorithm
                                         horizontal_sum=False,                  #Setting for the peak finding algorithm
                                         radial=False,                          #Calculate the radial velocity
                                         poloidal=False,                        #Calculate the poloidal velocity
                                         #Temporal maximum finding inputs
                                         temporal_maximum=False,                #Calculate the maximum time shift of the time traces of the signal instead of maximum position (not time resolved)
                                         taures=10e-6,                          #Time resolution to find the filaments around the found peak
                                         maximum_shift_avg=False,               #Calculate the velocities from the maximum averaged time shift (NOT WORKING FOR RADIAL)
                                         maximum_shift=False,                   #Calculate the valocites with fit_window shifted times (gives scattered results) (DEPRECATED)
                                         fit_window=5,                          #Window for the linear fit of the filament (+- value, FYI: 1 is too low)
                                         #Spatial maximum finding inputs
                                         spatial_maximum=False,                 #Calculate the velocities based on a spatial maximum movement of the filaments (structure tracking)
                                         spatial_correlation_thres=0.5,         #Correlation threshold for the spatial correlation calculation.
                                         remove_interlaced_structures=True,
                                         #Test options:
                                         test=False,                            #Plot the results
                                         ploterror=False,                       #Plot the errors during the test.
                                         #Output options:
                                         return_times=False,                    #Return the maximum times and displacement vector (for debugging mainly).
                                         cache_data=True,                       #Cache the data or try to open is from cache
                                         pdf=True,                              #Print the results into a PDF
                                         nocalc=False,                          #Restore the results from the pickle file
                                         ):
    
    """
    The code calculates the velocity of filaments for a certain period of time
    in the GPI signal. First it finds the filaments based on a wavelet peak
    finder algorithm from scipy (find_peaks_cwt), then fits the slope of the
    peaks with linear regression.
    """
    
    #Handle input variable errors
    if time_range is None:
        print('The time range needs to be set for the calculation.')
        return
    else:    
        if (type(time_range) is not list and len(time_range) != 2):
            raise TypeError('time_range needs to be a list with two elements.')
            
    if exp_id is None:
        raise ValueError('The experiment ID needs to be set.')
    if temporal_maximum and maximum_shift+maximum_shift_avg != 1:
        raise IOError('Only one type of calculation should be set.')
    if spatial_maximum+temporal_maximum != 1:
        raise IOError('Only one type of calculation should be set.')
    if radial+poloidal != 1:
        raise IOError('Either radial or poloidal can be calculated, not both at the same time.')
        
    #Read the data
    print("\n------- Reading NSTX GPI data --------")
    if cache_data:
        try:
            d=flap.get_data_object_ref(exp_id=exp_id,object_name='GPI')
        except:
            print('Data is not cached, it needs to be read.')
            d=flap.get_data('NSTX_GPI',exp_id=exp_id,name='',object_name='GPI')
    else:
        d=flap.get_data('NSTX_GPI',exp_id=exp_id,name='',object_name='GPI')

    comment='ref_'+str(ref_pixel[0])+'_'+str(ref_pixel[1])
    if radial:
        comment+='_rad'
    if poloidal:
        comment+='_pol'
    if maximum_shift_avg:
        comment+='_max_shift_avg'
    if maximum_shift:
        comment+='_max_shift'
    pickle_filename=flap_nstx.tools.filename(exp_id=exp_id, 
                                             time_range=time_range,
                                             purpose='filament velocity trace',
                                             comment=comment,
                                             extension='pickle')
    if os.path.exists(pickle_filename) and nocalc:
        try:
            pickle.load(open(pickle_filename, 'rb'))
        except:
            print('The pickle file cannot be loaded. Recalculating the results.')
            nocalc=False
    elif nocalc:
        print('The pickle file cannot be loaded. Recalculating the results.')
        nocalc=False  
        
    if not nocalc:
        #Slice it to the time range
        flap.slice_data('GPI', slicing={'Time':flap.Intervals(time_range[0],time_range[1])}, output_name='GPI_SLICED')
        #Find the peaks of the filaments
        filament_times=flap_nstx.analysis.find_filaments('GPI_SLICED', 
                                                         ref_pixel=ref_pixel,
                                                         horizontal_sum=horizontal_sum,
                                                         xrange=xrange,
                                                         vertical_sum=vertical_sum,
                                                         yrange=yrange,
                                                         width_range=width_range,
                                                         normalize=normalize,
                                                         test=test)
        flap.slice_data('GPI', 
                        slicing={'Time':flap.Intervals(time_range[0]-taures*80,
                                                       time_range[1]+taures*80)},
                        output_name='GPI_SLICED')
        #Find the time lags between radially or poloidally adjacent pixels
        n_filaments=len(filament_times)
 
    if poloidal:
        displacement=d.coordinate('Device z')[0][0,ref_pixel[0],yrange[0]:yrange[1]]
        displacement-=displacement[0]
        #First approach: maximum shift calculation and line fitting in the taures
        if maximum_shift_avg or maximum_shift:
            if not nocalc:
                maximum_times=np.zeros([n_filaments,yrange[1]-yrange[0]])
                if maximum_shift_avg:
                    velocity_matrix=np.zeros([n_filaments,3]) #[:,0]=times, [:,1]=velocities,[:,2]=velocity error
                    velocity_matrix[:,0]=filament_times
                else:
                    velocity_matrix=np.zeros([n_filaments,(yrange[1]-yrange[0]),3]) #[:,0]=times, [:,1]=velocities,[:,2]=velocity error
                for index_filaments in range(n_filaments):
                    #Above the reference pixels:
                    maximum_times[index_filaments,ref_pixel[1]-yrange[0]]=filament_times[index_filaments]
                    for index_pol_channel in range(ref_pixel[1],yrange[1]):
                        if index_pol_channel == ref_pixel[1]:                          #This is necessary, because parabola fitting is also necessary for the peak times, not just the neighboring pixels.
                            filament_time_range=[filament_times[index_filaments]-taures,
                                                 filament_times[index_filaments]+taures]
                        else:
                            filament_time_range=[maximum_times[index_filaments,index_pol_channel-yrange[0]-1]-taures,
                                                 maximum_times[index_filaments,index_pol_channel-yrange[0]-1]+taures]
                        #print(filament_time_range[0],filament_time_range[1],ref_pixel[0],index_pol_channel)
                        
                        if x_average:
                            slicing={'Time':flap.Intervals(filament_time_range[0],filament_time_range[1]),
                                     'Image x':flap.Intervals(xrange[0],xrange[1]),
                                     'Image y':index_pol_channel}
                            summing={'Image x':'Mean'}
                        else:
                            slicing={'Time':flap.Intervals(filament_time_range[0],filament_time_range[1]),
                                     'Image x':ref_pixel[0],
                                     'Image y':index_pol_channel}
                            summing=None
                            
                        d=flap.slice_data('GPI_SLICED', slicing=slicing, summing=summing)
                        coeff=np.polyfit(d.coordinate('Time')[0][:],d.data,2)
                        max_time=-coeff[1]/(2*coeff[0])
                        if max_time < filament_time_range[0] or max_time > filament_time_range[1]:
                            max_time=d.coordinate('Time')[0][np.argmax(d.data)]
                        maximum_times[index_filaments,index_pol_channel-yrange[0]]=max_time
    
                    #Below the reference pixels
                    for index_pol_channel in range(ref_pixel[1]-1,yrange[0]-1,-1):
                        
                        filament_time_range=[maximum_times[index_filaments,index_pol_channel-yrange[0]+1]-taures,
                                             maximum_times[index_filaments,index_pol_channel-yrange[0]+1]+taures]
                        if x_average:
                            slicing={'Time':flap.Intervals(filament_time_range[0],filament_time_range[1]),
                                     'Image x':flap.Intervals(xrange[0],xrange[1]),
                                     'Image y':index_pol_channel}
                            summing={'Image x':'Mean'}
                        else:
                            slicing={'Time':flap.Intervals(filament_time_range[0],filament_time_range[1]),
                                     'Image x':ref_pixel[0],
                                     'Image y':index_pol_channel}
                            summing=None
                            
                        d=flap.slice_data('GPI_SLICED', slicing=slicing, summing=summing)
                        coeff=np.polyfit(d.coordinate('Time')[0][:],d.data,2)
                        max_time=-coeff[1]/(2*coeff[0])
                        if max_time < filament_time_range[0] or max_time > filament_time_range[1]:
                            max_time=d.coordinate('Time')[0][np.argmax(d.data)]
                        maximum_times[index_filaments,index_pol_channel-yrange[0]]=max_time
                    if maximum_shift:
                        for index_pol_channel in range(yrange[0]+fit_window,yrange[1]-fit_window): 
                           coeff,cov=np.polyfit(displacement[index_pol_channel-fit_window-yrange[0]:index_pol_channel-yrange[0]+fit_window+1],
                                                maximum_times[index_filaments,index_pol_channel-fit_window-yrange[0]:index_pol_channel+fit_window+1-yrange[0]],1,cov=True)
                           velocity_matrix[index_filaments,index_pol_channel-yrange[0],0]=np.mean(maximum_times[index_filaments,index_pol_channel-fit_window-yrange[0]:index_pol_channel+fit_window+1-yrange[0]])
                           velocity_matrix[index_filaments,index_pol_channel-yrange[0],1]=1/coeff[0]
                           velocity_matrix[index_filaments,index_pol_channel-yrange[0],2]=np.sqrt(cov[0][0])/coeff[0]**2
                    else:
                        coeff,cov=np.polyfit(displacement,maximum_times[index_filaments,:],1,cov=True)
                        velocity_matrix[index_filaments,1]=1/coeff[0]
                        velocity_matrix[index_filaments,2]=np.sqrt(cov[0][0])/coeff[0]**2
                if maximum_shift:
                    velocity_matrix=velocity_matrix[:,fit_window:-fit_window,:] #No data in the beginning and at the end due to the 3 point linear fit
                    velocity_matrix=velocity_matrix.reshape([n_filaments*(yrange[1]-yrange[0]-2*fit_window),3])
                    sorted_ind=np.argsort(velocity_matrix[:,0])
                    velocity_matrix=velocity_matrix[sorted_ind,:]
                pickle_object=[ref_pixel,velocity_matrix,maximum_times,'ref_pix,velocity_matrix,maximum_times']
                pickle.dump(pickle_object,open(pickle_filename, 'wb'))
            else: #NOCALC
                print('--- Loading data from pickle file ---')
                ref_pixel, velocity_matrix, maximum_times, comment = pickle.load(open(pickle_filename, 'rb'))
                
            if test:
                plt.figure()
                flap.plot('GPI', exp_id, 
                          slicing={'Time':flap.Intervals(time_range[0],time_range[1]), 
                                   'Image x':ref_pixel[0]}, 
                          plot_type='contour', 
                          axes=['Time', 'Image y'], 
                          plot_options={'levels':51})
                for i in range(len(maximum_times[:,0])):
                    plt.plot(maximum_times[i,:],np.arange(yrange[0],yrange[1]))
                plt.figure()
                plt.scatter(velocity_matrix[:,0],velocity_matrix[:,1]/1000)
                if ploterror:
                    plt.errorbar(velocity_matrix[:,0],
                                 velocity_matrix[:,1]/1000, 
                                 yerr=velocity_matrix[:,2]/1000,
                                 marker='o')
                plt.xlabel('Time [s]')
                plt.ylabel('Poloidal velocity [km/s]')
                plt.ylim([-50,50])
                plt.show()
            if return_times:
                return velocity_matrix, maximum_times, displacement
            else:
                return velocity_matrix
            #Profit.
    if radial:
        displacement=d.coordinate('Device R')[0][0,xrange[0]:xrange[1],ref_pixel[1]]
        displacement-=displacement[0]
        if maximum_shift_avg or maximum_shift:
            if not nocalc:
                maximum_times=np.zeros([n_filaments,xrange[1]-xrange[0]])
                if maximum_shift_avg:
                    velocity_matrix=np.zeros([n_filaments,3]) #[:,0]=times, [:,1]=velocities,[:,2]=velocity error
                    velocity_matrix[:,0]=filament_times
                else:
                    velocity_matrix=np.zeros([n_filaments,(xrange[1]-xrange[0]),3]) #[:,0]=times, [:,1]=velocities,[:,2]=velocity error
                for index_filaments in range(n_filaments):
                    #Above the reference pixels:
                    maximum_times[index_filaments,ref_pixel[0]-xrange[0]]=filament_times[index_filaments]
                    for index_rad_channel in range(ref_pixel[0],xrange[1]):
                        if index_rad_channel == ref_pixel[0]:                          #This is necessary, because parabola fitting is also necessary for the peak times, not just the neighboring pixels.
                            filament_time_range=[filament_times[index_filaments]-taures,
                                                 filament_times[index_filaments]+taures]
                        else:
                            filament_time_range=[maximum_times[index_filaments,index_rad_channel-xrange[0]-1]-taures,
                                                 maximum_times[index_filaments,index_rad_channel-xrange[0]-1]+taures]
                        #print(filament_time_range[0],filament_time_range[1],ref_pixel[0],index_rad_channel)
                        
                        if y_average:
                            slicing={'Time':flap.Intervals(filament_time_range[0],filament_time_range[1]),
                                     'Image x':index_rad_channel,
                                     'Image y':flap.Intervals(yrange[0],yrange[1])}
                            summing={'Image x':'Mean'}
                        else:
                            slicing={'Time':flap.Intervals(filament_time_range[0],filament_time_range[1]),
                                     'Image x':index_rad_channel,
                                     'Image y':ref_pixel[1]}
                            summing=None
                            
                        d=flap.slice_data('GPI_SLICED', slicing=slicing, summing=summing)
                        coeff=np.polyfit(d.coordinate('Time')[0][:],d.data,2)
                        max_time=-coeff[1]/(2*coeff[0])
                        if max_time < filament_time_range[0] or max_time > filament_time_range[1]:
                            max_time=d.coordinate('Time')[0][np.argmax(d.data)]
                        maximum_times[index_filaments,index_rad_channel-xrange[0]]=max_time
    
                    #Below the reference pixels
                    for index_rad_channel in range(ref_pixel[0]-1,xrange[0]-1,-1):
                        
                        filament_time_range=[maximum_times[index_filaments,index_rad_channel-xrange[0]+1]-taures,
                                             maximum_times[index_filaments,index_rad_channel-xrange[0]+1]+taures]
                        if y_average:
                            slicing={'Time':flap.Intervals(filament_time_range[0],filament_time_range[1]),
                                     'Image x':index_rad_channel,
                                     'Image y':flap.Intervals(yrange[0],yrange[1])}
                            summing={'Image x':'Mean'}
                        else:
                            slicing={'Time':flap.Intervals(filament_time_range[0],filament_time_range[1]),
                                     'Image x':index_rad_channel,
                                     'Image y':ref_pixel[1]}
                            summing=None
                            
                        d=flap.slice_data('GPI_SLICED', slicing=slicing, summing=summing)
                        coeff=np.polyfit(d.coordinate('Time')[0][:],d.data,2)
                        max_time=-coeff[1]/(2*coeff[0])
                        if max_time < filament_time_range[0] or max_time > filament_time_range[1]:
                            max_time=d.coordinate('Time')[0][np.argmax(d.data)]
                        maximum_times[index_filaments,index_rad_channel-xrange[0]]=max_time
                    if maximum_shift:
                        for index_rad_channel in range(xrange[0]+fit_window,xrange[1]-fit_window): 
                           coeff,cov=np.polyfit(displacement[index_rad_channel-fit_window-xrange[0]:index_rad_channel-xrange[0]+fit_window+1],
                                                maximum_times[index_filaments,index_rad_channel-fit_window-xrange[0]:index_rad_channel+fit_window+1-xrange[0]],1,cov=True)
                           velocity_matrix[index_filaments,index_rad_channel-xrange[0],0]=np.mean(maximum_times[index_filaments,index_rad_channel-fit_window-xrange[0]:index_rad_channel+fit_window+1-xrange[0]])
                           velocity_matrix[index_filaments,index_rad_channel-xrange[0],1]=1/coeff[0]
                           velocity_matrix[index_filaments,index_rad_channel-xrange[0],2]=np.sqrt(cov[0][0])/coeff[0]**2
                    else:
                        coeff,cov=np.polyfit(displacement,maximum_times[index_filaments,:],1,cov=True)
                        velocity_matrix[index_filaments,1]=1/coeff[0]
                        velocity_matrix[index_filaments,2]=np.sqrt(cov[0][0])/coeff[0]**2
                if maximum_shift:
                    velocity_matrix=velocity_matrix[:,fit_window:-fit_window,:] #No data in the beginning and at the end due to the 3 point linear fit
                    velocity_matrix=velocity_matrix.reshape([n_filaments*(xrange[1]-xrange[0]-2*fit_window),3])
                    sorted_ind=np.argsort(velocity_matrix[:,0])
                    velocity_matrix=velocity_matrix[sorted_ind,:]
                pickle_object=[ref_pixel,velocity_matrix,maximum_times,'ref_pix,velocity_matrix,maximum_times']
                pickle.dump(pickle_object,open(pickle_filename, 'wb'))
            else: #NOCALC
                print('--- Loading data from pickle file ---')
                ref_pixel, velocity_matrix, maximum_times, comment = pickle.load(open(pickle_filename, 'rb'))
                
            if test:
                plt.figure()
                flap.plot('GPI', exp_id, 
                          slicing={'Time':flap.Intervals(time_range[0],time_range[1]), 
                                   'Image y':ref_pixel[1]}, 
                          plot_type='contour', 
                          axes=['Time', 'Image x'], 
                          plot_options={'levels':51})
                for i in range(len(maximum_times[:,0])):
                    plt.plot(maximum_times[i,:],np.arange(xrange[0],xrange[1]))
                plt.figure()
                plt.scatter(velocity_matrix[:,0],velocity_matrix[:,1]/1000)
                if ploterror:
                    plt.errorbar(velocity_matrix[:,0],
                                 velocity_matrix[:,1]/1000, 
                                 yerr=velocity_matrix[:,2]/1000,
                                 marker='o')
                plt.xlabel('Time [s]')
                plt.ylabel('Radial velocity [km/s]')
                plt.ylim([-50,50])
                plt.show()
            if return_times:
                return velocity_matrix, maximum_times, displacement
            else:
                return velocity_matrix
            #Profit.
          
    
def average_gpi_turbulence_velocity(exp_id=None,
                                    time_range=None,
                                    cache_data=True,
                                    nocalc=False):
    
    #Get GPI data
    if not nocalc:
        slicing={'Time':flap.Intervals(time_range[0],time_range[1])}
    #Read data

        print("\n------- Reading NSTX GPI data --------")
    if cache_data:
        try:
            flap.get_data_object('GPI',exp_id=exp_id)
        except:
            print('Data is not cached, it needs to be read.')
            flap.get_data('NSTX_GPI',exp_id=exp_id,
                          name='',
                          object_name='GPI')
    else:
        flap.get_data('NSTX_GPI',exp_id=exp_id,
                      name='',
                      object_name='GPI')
  
    flap.slice_data('GPI',exp_id=exp_id, 
                    slicing=slicing,
                    output_name='GPI_SLICED_FULL')
    nx=10
    ny=10
    nx_ref=5
    ny_ref=6
    max_corr=np.zeros([5,nx_ref,ny_ref,2*nx+1,2*ny+1])
    for time_shift in range(1,5):
        print('Time shift: '+str(time_shift))
        for i_pixel_x_1 in range(nx_ref):
            for j_pixel_y_1 in range(ny_ref):
                reference_pixel=[i_pixel_x_1*nx+nx,j_pixel_y_1*ny+ny]
                print(reference_pixel)
                d_ref=flap.slice_data('GPI_SLICED_FULL',
                                      exp_id=exp_id,
                                      slicing={'Image x':reference_pixel[0],
                                               'Image y':reference_pixel[1]},
                                      output_name='GPI_SLICED_REF').data
                d_ref=d_ref-np.mean(d_ref)
                for i_pixel_x_2 in range(-nx,nx+1,1):
                    for j_pixel_y_2 in range(-ny,ny+1,1):
                            d_pix=flap.slice_data('GPI_SLICED_FULL',
                                                    exp_id=exp_id,
                                                    slicing={'Image x':reference_pixel[0]+i_pixel_x_2,
                                                             'Image y':reference_pixel[1]+j_pixel_y_2},
                                                    output_name='GPI_SLICED_PIX').data
                            d_pix_new=np.roll(d_pix,time_shift)
                            d_pix_new=d_pix_new-np.mean(d_pix_new)
                            max_corr[time_shift-1,i_pixel_x_1,j_pixel_y_1,i_pixel_x_2+nx,j_pixel_y_2+ny]=np.sum(d_pix_new*d_ref)/np.sqrt(np.sum(d_pix_new*d_pix_new)*np.sum(d_ref*d_ref))
    return max_corr
    
