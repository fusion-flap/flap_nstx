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
from flap_nstx.analysis import nstx_gpi_one_frame_structure_finder

thisdir = os.path.dirname(os.path.realpath(__file__))
fn = os.path.join(thisdir,"flap_nstx.cfg")
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
        a memory efficient process
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

def calculate_nstx_gpi_smooth_velocity(exp_id=None,                 #Shot number
                                       time_range=None,             #Time range of the calculation
                                       time_res=100e-6,             #Time resolution of the calculation
                                       xrange=[0,63],               #Range of the calculation
                                       yrange=[10,70],              #Range of the calculation
                                       filter_data=True,
                                       f_low=10e3,                  #Highpass filtering of the signal
                                       taurange=None,               #Range of the CCF time delay calculation
                                       taures=2.5e-6,               #Resoltuion of the CCF calculation
                                       interval_n=1,                #Number of intervals the signal is split to
                                       radial=False,                #Calculate the radial velocity
                                       poloidal=False,              #Calculate the poloidal velocity
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
        d=flap.filter_data('GPI_SLICED_FULL',
                           exp_id=exp_id,
                           coordinate='Time',
                           options={'Type':'Highpass',
                                    'f_low':f_low,
                                    'Design':'Chebyshev II'},
                           output_name='GPI_FILTERED')
        d.data=np.asarray(d.data,dtype='float32')
        #THIS IS WHERE THE CODE STARTS TO BE VERY DIFFERENT FROM THE LAST ONE.
    
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
                        slicing=slicing, 
                        output_name='GPI_WINDOW_2')
        flap.slice_data('GPI_FILTERED',
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
                            slicing=slicing,
                            output_name='GPI_WINDOW_1')
            flap.slice_data('GPI_WINDOW_2', 
                            slicing=slicing, 
                            output_name='GPI_WINDOW_REF_12')
            ccf=flap.ccf('GPI_WINDOW_1',exp_id=exp_id,
                         ref='GPI_WINDOW_REF_12',
                         coordinate='Time',
                         options={'Resolution':taures,
                                  'Range':taurange,
                                  'Trend':['Poly',2],
                                  'Interval':interval_n,
                                  'Normalize':False,
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
                                        slicing={'Time':flap.Intervals(time_window[0],time_window[1]),
                                                 'Image y':np.mean(yrange)}, 
                                        output_name='GPI_WINDOW_Y').coordinate('Device R')[0][0,:]
    if add_flux:
        flux_coordinate_vector=flap.slice_data('GPI_FILTERED',
                                               slicing={'Time':flap.Intervals(time_window[0],time_window[1]),
                                                        'Image y':np.mean(yrange)}, 
                                               output_name='GPI_WINDOW_Y').coordinate('Flux r avg')[0][0,:]
        z_coordinate_vector=flap.slice_data('GPI_FILTERED',
                                            slicing={'Time':flap.Intervals(time_window[0],time_window[1]),
                                                     'Image x':np.mean(xrange)}, 
                                            output_name='GPI_WINDOW_Y').coordinate('Flux r avg')[0][0,:]
    else:
        z_coordinate_vector=flap.slice_data('GPI_FILTERED',
                                        slicing={'Time':flap.Intervals(time_window[0],time_window[1]),
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
            filename=flap_nstx.analysis.filename(exp_id=exp_id, 
                                                 time_range=time_range,
                                                 purpose='TDE radial velocity',
                                                 extension='pickle')
            flap.save(flap.get_data_object_ref('NSTX_GPI_RAD_VELOCITY'), filename=filename)
        if poloidal:
            filename=flap_nstx.analysis.filename(exp_id=exp_id, 
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
    pickle_filename=flap_nstx.analysis.filename(exp_id=exp_id, 
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
            
def calculate_nstx_gpi_avg_frame_velocity(exp_id=None,                          #Shot number
                                          time_range=None,                      #The time range for the calculation
                                          data_object=None,                     #Input data object if available from outside (e.g. generated sythetic signal)
                                          x_range=[0,63],                       #X range for the calculation
                                          y_range=[0,79],                       #Y range for the calculation
                                          
                                          #Normalizer inputs
                                          normalize='roundtrip',                #Normalization options, 
                                                                                  #None: no normalization
                                                                                  #'roundtrip': zero phase LPF IIR filter
                                                                                  #'halved': different normalzation for before and after the ELM
                                                                                  #'simple': simple low-pass filtered normalization
                                          normalize_f_kernel='Elliptic',        #The kernel for filtering the gas cloud
                                          normalize_f_high=1e3,                 #High pass frequency for the normalizer data
                                          
                                          #Inputs for velocity pre processing
                                          normalize_for_velocity=False,         #Normalize the signal for velocity processing
                                          subtraction_order_for_velocity=1,     #Order of the 2D polynomial for background subtraction
                                          
                                          #Inputs for velocity processing
                                          flap_ccf=True,                        #Calculate the cross-correlation functions with flap instead of CCF
                                          correlation_threshold=0.6,            #Threshold for the maximum of the cross-correlation between the two frames (calculated with the actual maximum, not the fit value)
                                          frame_similarity_threshold=0.0,       #Similarity threshold between the two subsequent frames (CCF at zero lag) DEPRECATED when an ELM is present
                                          valid_frame_thres=5,                  #The number of consequtive frames to consider the calculation valid.
                                          velocity_threshold=None,              #Velocity threshold for the calculation. Abova values are considered np.nan.
                                          parabola_fit=True,                    #Fit a parabola on top of the cross-correlation function (CCF) peak
                                          fitting_range=5,                      #Fitting range of the peak of CCF 
                                          gpi_plane_calculation=True,
                                          
                                          #Input for size pre-processing
                                          skip_structure_calculation=False,     #Self explanatory
                                          normalize_for_size=True,              #Normalize the signal for the size calculation
                                          subtraction_order_for_size=None,      #Polynomial subtraction order
                                          remove_interlaced_structures=True,
                                          #Inputs for size processing
                                          nlevel=51,                            #Number of contour levels for the structure size and velocity calculation.
                                          filter_level=3,                       #Number of embedded paths to be identified as an individual structure
                                          global_levels=False,                  #Set for having structure identification based on a global intensity level.
                                          levels=None,                          #Levels of the contours for the entire dataset. If None, it equals data.min(),data.max() divided to nlevel intervals.
                                          threshold_coeff=1.0,                  #Variance multiplier threshold for size determination
                                          weighting='intensity',                #Weighting of the results based on the 'number' of structures, the 'intensity' of the structures or the 'area' of the structures (options are in '')
                                          maxing='intensity',                   #Return the properties of structures which have the largest "area" or "intensity"
                                          velocity_base='cog',                  #Base of calculation of the advanced velocity, available options: 
                                                                                #Center of gravity: 'cog'; Geometrical center: 'centroid'; Ellipse center: 'center'
                                          #Plot options:
                                          plot=True,                            #Plot the results
                                          pdf=False,                            #Print the results into a PDF
                                          plot_gas=True,                        #Plot the gas cloud parameters on top of the other results from the structure size calculation
                                          plot_error=False,                     #Plot the errorbars of the velocity calculation based on the line fitting and its RMS error
                                          error_window=4.,                      #Plot the average signal with the error bars calculated from the normalized variance.
                                          overplot_str_vel=True,                #Overplot the velocity calculated from the structure onto the one from CCF
                                          overplot_average=True,
                                          plot_scatter=True,
                                          structure_video_save=False,           #Save the video of the overplot ellipses
                                          structure_pdf_save=False,             #Save the struture finding algorithm's plot output into a PDF (can create very large PDF's, the number of pages equals the number of frames)
                                          structure_pixel_calc=False,           #Calculate and plot the structure sizes in pixels
                                          plot_time_range=None,                 #Plot the results in a different time range than the data is read from
                                          plot_for_publication=False,           #Modify the plot sizes to single column sizes and golden ratio axis ratios
                                          
                                          #File input/output options
                                          filename=None,                        #Filename for restoring data
                                          save_results=True,                    #Save the results into a .pickle file to filename+.pickle
                                          nocalc=True,                          #Restore the results from the .pickle file from filename+.pickle
                                          
                                          #Output options:
                                          return_results=False,                 #Return the results if set.
                                          cache_data=True,                      #Cache the data or try to open is from cache
                                          
                                          #Test options
                                          test=False,                           #Test the results
                                          test_structures=False,                #Test the structure size calculation
                                          test_gas_cloud=False,                 #Test the gas cloud property determination
                                          test_histogram=False,                 #Plot the poloidal velocity histogram
                                          ):
    
    """
    Calculate frame by frame average frame velocity of the NSTX GPI signal. The
    code takes subsequent frames, calculates the 2D correlation function between
    the two and finds the maximum. Based on the pixel shift and the sampling
    time of the signal, the radial and poloidal velocity is calculated.
    The code assumes that the structures present in the subsequent frames are
    propagating with the same velocity. If there are multiple structures
    propagating in e.g. different direction or with different velocities, their
    effects are averaged over.
    """
    
    #Constants for the calculation
    #Using the spatial calibration to find the actual velocities.
    coeff_r=np.asarray([3.7183594,-0.77821046,1402.8097])/1000. #The coordinates are in meters, the coefficients are in mm
    coeff_z=np.asarray([0.18090118,3.0657776,70.544312])/1000.  #The coordinates are in meters, the coefficients are in mm
    
    #Input error handling
    if exp_id is None and data_object is None:
        raise ValueError('Either exp_id or data_object needs to be set for the calculation.')
    if data_object is None:
        if time_range is None and filename is None:
            raise ValueError('It takes too much time to calculate the entire shot, please set a time_range.')
        else:    
            if type(time_range) is not list and filename is None:
                raise TypeError('time_range is not a list.')
            if filename is None and len(time_range) != 2:
                raise ValueError('time_range should be a list of two elements.')
        
    if weighting not in ['number', 'area', 'intensity']:
        raise ValueError("Weighting can only be by the 'number', 'area' or 'intensity' of the structures.")
    if maxing not in ['area', 'intensity']:
        raise ValueError("Maxing can only be by the 'area' or 'intensity' of the structures.")
    if velocity_base not in ['cog', 'center', 'centroid']:
        raise ValueError("The base of the velocity can only be 'cog', 'center', 'centroid'")
        
    if (fitting_range*2+1 > np.abs(x_range[1]-x_range[0]) or 
        fitting_range*2+1 > np.abs(y_range[1]-y_range[0])):
        raise ValueError('The fitting range for the parabola is too large for the given coordinate range.')
        
    if correlation_threshold is not None and not flap_ccf:
        print('Correlation threshold is not taken into account if not calculated with FLAP.')
        
    if frame_similarity_threshold is not None and not flap_ccf:
        print('Correlation threshold is not taken into account if not calculated with FLAP.')
    
    if data_object is not None and type(data_object) == str:
        if exp_id is None:
            exp_id='*'
        d=flap.get_data_object(data_object,exp_id=exp_id)
        time_range=[d.coordinate('Time')[0][0,0,0],
                    d.coordinate('Time')[0][-1,0,0]]
        exp_id=d.exp_id
        flap.add_data_object(d, 'GPI_SLICED_FULL')
    
    if parabola_fit:
        comment='pfit_o'+str(subtraction_order_for_velocity)+\
                '_fst_'+str(frame_similarity_threshold)
    else:
        comment='max_o'+str(subtraction_order_for_velocity)+\
                '_fst_'+str(frame_similarity_threshold)
                
    if normalize_for_size:
        comment+='_ns'
    if normalize_for_velocity:
        comment+='_nv'
    if remove_interlaced_structures:
        comment+='_nointer'
               
    if filename is None:
        wd=flap.config.get_all_section('Module NSTX_GPI')['Working directory']
        filename=flap_nstx.analysis.filename(exp_id=exp_id,
                                             working_directory=wd+'/processed_data',
                                             time_range=time_range,
                                             purpose='ccf velocity',
                                             comment=comment)
        filename_was_none=True
    else:
        filename_was_none=False
        
    pickle_filename=filename+'.pickle'
    if os.path.exists(pickle_filename) and nocalc:
        try:
            pickle.load(open(pickle_filename, 'rb'))
        except:
            print('The pickle file cannot be loaded. Recalculating the results.')
            nocalc=False
    elif nocalc:
        print(pickle_filename)
        print('The pickle file cannot be loaded. Recalculating the results.')
        nocalc=False

    if not test and not test_structures and not test_gas_cloud and not structure_pdf_save:
        import matplotlib
        matplotlib.use('agg')
    import matplotlib.pyplot as plt
    
    if structure_pdf_save:
        filename=flap_nstx.analysis.filename(exp_id=exp_id,
                                             working_directory=wd+'/plots',
                                             time_range=time_range,
                                             purpose='found structures',
                                             comment=comment,
                                             extension='pdf')
        pdf_structures=PdfPages(filename)
    if not nocalc:
        slicing={'Time':flap.Intervals(time_range[0],time_range[1]),
                 'Image x':flap.Intervals(x_range[0],x_range[1]),
                 'Image y':flap.Intervals(y_range[0],y_range[1])}
        #Read data
        if data_object is None:
            print("\n------- Reading NSTX GPI data --------")
            if cache_data:
                try:
                    d=flap.get_data_object('GPI',exp_id=exp_id)
                except:
                    print('Data is not cached, it needs to be read.')
                    d=flap.get_data('NSTX_GPI',exp_id=exp_id,
                                    name='',
                                    object_name='GPI')
            else:
                d=flap.get_data('NSTX_GPI',exp_id=exp_id,
                                name='',
                                object_name='GPI')
          
            d=flap.slice_data('GPI',exp_id=exp_id, 
                              slicing=slicing,
                              output_name='GPI_SLICED_FULL')
            
        object_name_ccf_velocity='GPI_SLICED_FULL'
        object_name_str_size='GPI_SLICED_FULL'
        object_name_str_vel='GPI_SLICED_FULL'
        
        #Normalize data for size calculation
        print("**** Calculating the gas cloud ****")

        normalizer_object_name='GPI_LPF_INTERVAL'

        slicing_for_filtering=copy.deepcopy(slicing)
        slicing_for_filtering['Time']=flap.Intervals(time_range[0]-1/normalize_f_high*10,
                                                     time_range[1]+1/normalize_f_high*10)
        if data_object is None:
            flap.slice_data('GPI',
                            exp_id=exp_id,
                            slicing=slicing_for_filtering,
                            output_name='GPI_SLICED_FOR_FILTERING')            
        
        if normalize == 'simple':
            flap.filter_data('GPI_SLICED_FOR_FILTERING',
                             exp_id=exp_id,
                             coordinate='Time',
                             options={'Type':'Lowpass',
                                      'f_high':normalize_f_high,
                                      'Design':normalize_f_kernel},
                             output_name=normalizer_object_name)
            coefficient=flap.slice_data(normalizer_object_name,
                                        exp_id=exp_id,
                                        slicing=slicing,
                                        output_name='GPI_GAS_CLOUD').data
            
        elif normalize == 'roundtrip':
            norm_obj=flap.filter_data('GPI_SLICED_FOR_FILTERING',
                                         exp_id=exp_id,
                                         coordinate='Time',
                                         options={'Type':'Lowpass',
                                                  'f_high':normalize_f_high,
                                                  'Design':normalize_f_kernel},
                                         output_name=normalizer_object_name)
            
            norm_obj.data=np.flip(norm_obj.data,axis=0)
            norm_obj=flap.filter_data(normalizer_object_name,
                                         exp_id=exp_id,
                                         coordinate='Time',
                                         options={'Type':'Lowpass',
                                                  'f_high':normalize_f_high,
                                                  'Design':normalize_f_kernel},
                                         output_name=normalizer_object_name)
            
            norm_obj.data=np.flip(norm_obj.data,axis=0)                
            coefficient=flap.slice_data(normalizer_object_name,
                                        exp_id=exp_id,
                                        slicing=slicing,
                                        output_name='GPI_GAS_CLOUD').data
            
        elif normalize == 'halved':
            #Find the peak of the signal (aka ELM time as of GPI data)
            data_obj=flap.get_data_object_ref(object_name_str_size).slice_data(summing={'Image x':'Mean', 'Image y':'Mean'})
            ind_peak=np.argmax(data_obj.data)
            data_obj_reverse=copy.deepcopy(flap.get_data_object('GPI_SLICED_FOR_FILTERING'))
            data_obj_reverse.data=np.flip(data_obj_reverse.data, axis=0)
            flap.add_data_object(data_obj_reverse,'GPI_SLICED_FOR_FILTERING_REV')
            
            normalizer_object_name_reverse='GPI_LPF_INTERVAL_REV'
            flap.filter_data('GPI_SLICED_FOR_FILTERING',
                             exp_id=exp_id,
                             coordinate='Time',
                             options={'Type':'Lowpass',
                                      'f_high':normalize_f_high,
                                      'Design':normalize_f_kernel},
                             output_name=normalizer_object_name)
            coefficient1_sliced=flap.slice_data(normalizer_object_name,
                                         exp_id=exp_id,
                                         slicing=slicing)
            
            coefficient2=flap.filter_data('GPI_SLICED_FOR_FILTERING_REV',
                                         exp_id=exp_id,
                                         coordinate='Time',
                                         options={'Type':'Lowpass',
                                                  'f_high':normalize_f_high,
                                                  'Design':normalize_f_kernel},
                                         output_name=normalizer_object_name_reverse)
            coefficient2.data=np.flip(coefficient2.data, axis=0)
            coefficient2_sliced=flap.slice_data(normalizer_object_name_reverse,
                                                exp_id=exp_id,
                                                slicing=slicing)
            
            coeff1_first_half=coefficient1_sliced.data[:ind_peak-4,:,:]
            coeff2_second_half=coefficient2_sliced.data[ind_peak-4:,:,:]
            coefficient=np.append(coeff1_first_half,coeff2_second_half, axis=0)
            coefficient_dataobject=copy.deepcopy(coefficient1_sliced)
            coefficient_dataobject.data=coefficient
            flap.add_data_object(coefficient_dataobject, 'GPI_GAS_CLOUD')
            
        #Global gas levels
        if normalize is not None:
            gas_min=flap.get_data_object_ref('GPI_GAS_CLOUD', exp_id=exp_id).data.min()
            gas_max=flap.get_data_object_ref('GPI_GAS_CLOUD', exp_id=exp_id).data.max()
            gas_levels=np.arange(nlevel)/(nlevel-1)*(gas_max-gas_min)+gas_min            
            
        if normalize_for_size:
            data_obj=flap.get_data_object(object_name_str_size)
            data_obj.data = data_obj.data/coefficient
            flap.add_data_object(data_obj, 'GPI_SLICED_DENORM_STR_SIZE')
            object_name_str_size='GPI_SLICED_DENORM_STR_SIZE'
            
        if normalize_for_velocity:
            data_obj=flap.get_data_object(object_name_str_vel)
            data_obj.data = data_obj.data/coefficient
            flap.add_data_object(data_obj, 'GPI_SLICED_DENORM_STR_VEL')
            object_name_str_vel='GPI_SLICED_DENORM_STR_VEL'

            data_obj=flap.get_data_object(object_name_ccf_velocity)
            data_obj.data = data_obj.data/coefficient
            flap.add_data_object(data_obj, 'GPI_SLICED_DENORM_CCF_VEL')
            object_name_ccf_velocity='GPI_SLICED_DENORM_CCF_VEL'
            
        #Subtract trend from data
        if subtraction_order_for_velocity is not None:
            print("*** Subtracting the trend of the data ***")
            d=flap_nstx.analysis.detrend_multidim(object_name_ccf_velocity,
                                                  exp_id=exp_id,
                                                  order=subtraction_order_for_velocity, 
                                                  coordinates=['Image x', 'Image y'], 
                                                  output_name='GPI_DETREND_VEL')
            object_name_ccf_velocity='GPI_DETREND_VEL'
            
        if subtraction_order_for_size is not None:
            print("*** Subtracting the trend of the data ***")
            d=flap_nstx.analysis.detrend_multidim(object_name_str_size,
                                                  exp_id=exp_id,
                                                  order=subtraction_order_for_size, 
                                                  coordinates=['Image x', 'Image y'], 
                                                  output_name='GPI_DETREND_SIZE')
            object_name_str_size='GPI_DETREND_SIZE'
            d=flap_nstx.analysis.detrend_multidim(object_name_str_size,
                                                  exp_id=exp_id,
                                                  order=subtraction_order_for_size, 
                                                  coordinates=['Image x', 'Image y'], 
                                                  output_name='GPI_DETREND_SIZE')
            object_name_str_vel='GPI_DETREND_SIZE'
            
        
        if global_levels:
            if levels is None:
                d=flap.get_data_object_ref(object_name_str_size)
                min_data=d.data.min()
                max_data=d.data.max()
                levels=np.arange(nlevel)/(nlevel-1)*(max_data-min_data)+min_data

        thres_obj_str_size=flap.slice_data(object_name_str_size,
                                           summing={'Image x':'Mean',
                                                    'Image y':'Mean'},
                                                    output_name='GPI_SLICED_TIMETRACE')
        intensity_thres_level_str_size=np.sqrt(np.var(thres_obj_str_size.data))*threshold_coeff+np.mean(thres_obj_str_size.data)
        thres_obj_str_vel=flap.slice_data(object_name_str_size,
                                           summing={'Image x':'Mean',
                                                    'Image y':'Mean'},
                                                    output_name='GPI_SLICED_TIMETRACE')
        intensity_thres_level_str_vel=np.sqrt(np.var(thres_obj_str_vel.data))*threshold_coeff+np.mean(thres_obj_str_vel.data)
        """
            VARIABLE DEFINITION
        """
        #Calculate correlation between subsequent frames in the data
        #Setting the variables for the calculation
        time_dim=d.get_coordinate_object('Time').dimension_list[0]
        n_frames=d.data.shape[time_dim]
        time=d.coordinate('Time')[0][:,0,0]
        sample_time=time[1]-time[0]
        sample_0=flap.get_data_object_ref(object_name_ccf_velocity).coordinate('Sample')[0][0,0,0]
        dalpha=flap.get_data_object_ref('GPI_SLICED_FULL').slice_data(summing={'Image x':'Mean', 'Image y':'Mean'}).data,
        
        frame_properties={'Shot':exp_id,
                          'Time':time[1:-1],
                          'GPI Dalpha':dalpha,
                          'Correlation max':np.zeros(len(time)-2),
                          'Frame similarity':np.zeros(len(time)-2),

                          'GC size':np.zeros([len(time)-2,2]),
                          'GC area':np.zeros(len(time)-2),                                   
                          'GC angle':np.zeros(len(time)-2),
                          'GC elongation':np.zeros(len(time)-2),
                          'GC centroid':np.zeros([len(time)-2,2]),
                          'GC position':np.zeros([len(time)-2,2]),
                          'GC COG':np.zeros([len(time)-2,2]),
                          
                          'Velocity ccf':np.zeros([len(time)-2,2]),
                          'Velocity str avg':np.zeros([len(time)-2,2]),
                          'Velocity str max':np.zeros([len(time)-2,2]),
                          
                          'Size avg':np.zeros([len(time)-2,2]),
                          'Size max':np.zeros([len(time)-2,2]),
                          'Position avg':np.zeros([len(time)-2,2]),
                          'Position max':np.zeros([len(time)-2,2]),
                          'Area avg':np.zeros(len(time)-2),
                          'Area max':np.zeros(len(time)-2),
                          'Elongation avg':np.zeros(len(time)-2),
                          'Elongation max':np.zeros(len(time)-2),                          
                          'Angle avg':np.zeros(len(time)-2),
                          'Angle max':np.zeros(len(time)-2),                          
                          'Centroid avg':np.zeros([len(time)-2,2]),
                          'Centroid max':np.zeros([len(time)-2,2]),                          
                          'COG avg':np.zeros([len(time)-2,2]),
                          'COG max':np.zeros([len(time)-2,2]),

                          'Frame COG':np.zeros([len(time)-2,2]),
                          'Str number':np.zeros(len(time)-2),
                         }
        ccf_data=np.zeros([n_frames-1,
                           (x_range[1]-x_range[0])*2+1,
                           (y_range[1]-y_range[0])*2+1])
        coord=[]
        coord.append(copy.deepcopy(flap.Coordinate(name='Sample',
                                                   unit='n.a.',
                                                   mode=flap.CoordinateMode(equidistant=True),
                                                   start=sample_0,
                                                   step=1,
                                                   dimension_list=[0]
                                                   )))
        
        coord.append(copy.deepcopy(flap.Coordinate(name='Time',
                                                   unit='s',
                                                   mode=flap.CoordinateMode(equidistant=True),
                                                   start=flap.get_data_object_ref(object_name_ccf_velocity).coordinate('Time')[0][0,0,0],
                                                   step=sample_time,
                                                   dimension_list=[0]
                                                   )))
        coord.append(copy.deepcopy(flap.Coordinate(name='Image x',
                                                   unit='Pixel',
                                                   mode=flap.CoordinateMode(equidistant=True),
                                                   start=-x_range[1],
                                                   step=1,
                                                   shape=[],
                                                   dimension_list=[1]
                                                   )))
        coord.append(copy.deepcopy(flap.Coordinate(name='Image y',
                                                   unit='Pixel',
                                                   mode=flap.CoordinateMode(equidistant=True),
                                                   start=-y_range[1],
                                                   step=1,
                                                   shape=[],
                                                   dimension_list=[2]
                                                   )))
        ccf_object = flap.DataObject(data_array=ccf_data,
                                     data_unit=flap.Unit(name='Signal',unit='Digit'),
                                     coordinates=coord,
                                     exp_id=exp_id,
                                     data_title='',
                                     data_source="NSTX_GPI")
        
        flap.add_data_object(ccf_object, 'GPI_CCF_F_BY_F')
        #Velocity is calculated between the two subsequent frames, the velocity time is the second frame's time
        delta_index=np.zeros(2)       
       
        #Inicializing for frame handling
        frame2=None
        frame2_vel=None
        structures2_vel=None
        invalid_correlation_frame_counter=0.
        
        if test or test_structures or test_gas_cloud or structure_pdf_save:
            plt.figure()
            
        for i_frames in range(0,n_frames-2):
            """
            STRUCTURE VELOCITY CALCULATION BASED ON CCF CALCULATION
            """
            
            slicing_frame1={'Sample':sample_0+i_frames}
            slicing_frame2={'Sample':sample_0+i_frames+1}
            if frame2 is None:
                frame1=flap.slice_data(object_name_ccf_velocity,
                                       exp_id=exp_id,
                                       slicing=slicing_frame1, 
                                       output_name='GPI_FRAME_1')
                #frame1.add_coordinate()
            else:
                frame1=copy.deepcopy(frame2)
                flap.add_data_object(frame1,'GPI_FRAME_1')
                
            frame2=flap.slice_data(object_name_ccf_velocity, 
                                   exp_id=exp_id,
                                   slicing=slicing_frame2,
                                   output_name='GPI_FRAME_2')

            frame1.data=np.asarray(frame1.data, dtype='float64')
            frame2.data=np.asarray(frame2.data, dtype='float64')                                
            
#                fft = np.fft.fftn(proc_array,axes=correlation_dimensions)
            if flap_ccf:
                flap.ccf('GPI_FRAME_2', 'GPI_FRAME_1', 
                         coordinate=['Image x', 'Image y'], 
                         options={'Resolution':1, 
                                  'Range':[[-(x_range[1]-x_range[0]),(x_range[1]-x_range[0])],
                                           [-(y_range[1]-y_range[0]),(y_range[1]-y_range[0])]], 
                                  'Trend removal':None, 
                                  'Normalize':True, 
                                  'Interval_n': 1}, 
                         output_name='GPI_FRAME_12_CCF')
    
                ccf_object.data[i_frames,:,:]=flap.get_data_object_ref('GPI_FRAME_12_CCF').data
                
                if test:
                    print('Maximum correlation:'+str(ccf_object.data[i_frames,:,:].max()))
            else: #DEPRECATED (only to be used when the FLAP 2D correlation is not working for some reason.)
                frame1=np.asarray(d.data[i_frames,:,:],dtype='float64')
                frame2=np.asarray(d.data[i_frames+1,:,:],dtype='float64')
                ccf_object.data[i_frames,:,:]=scipy.signal.correlate2d(frame2,frame1, mode='full')
                
            
            max_index=np.asarray(np.unravel_index(ccf_object.data[i_frames,:,:].argmax(), ccf_object.data[i_frames,:,:].shape))
            
            #Fit a 2D polinomial on top of the peak
            if parabola_fit:
                area_max_index=tuple([slice(max_index[0]-fitting_range,
                                            max_index[0]+fitting_range+1),
                                      slice(max_index[1]-fitting_range,
                                            max_index[1]+fitting_range+1)])
                #Finding the peak analytically
                try:
                    coeff=flap_nstx.analysis.polyfit_2D(values=ccf_object.data[i_frames,:,:][area_max_index],order=2)
                    index=[0,0]
                    index[0]=(2*coeff[2]*coeff[3]-coeff[1]*coeff[4])/(coeff[4]**2-4*coeff[2]*coeff[5])
                    index[1]=(-2*coeff[5]*index[0]-coeff[3])/coeff[4]
                except:
                    index=[fitting_range,fitting_range]
                if (index[0] < 0 or 
                    index[0] > 2*fitting_range or 
                    index[1] < 0 or 
                    index[1] > 2*fitting_range):
                    
                    index=[fitting_range,fitting_range]
                #if ccf_object.data[i_frames,:,:].max() > correlation_threshold and flap_ccf:              
                
                delta_index=[index[0]+max_index[0]-fitting_range-ccf_object.data[i_frames,:,:].shape[0]//2,
                             index[1]+max_index[1]-fitting_range-ccf_object.data[i_frames,:,:].shape[1]//2]
                if test:
                    plt.contourf(ccf_object.data[i_frames,:,:].T, levels=np.arange(0,51)/25-1)
                    plt.scatter(index[0]+max_index[0]-fitting_range,
                                index[1]+max_index[1]-fitting_range)

            else: #if not parabola_fit:
                
                delta_index=[max_index[0]-ccf_object.data[i_frames,:,:].shape[0]//2,
                             max_index[1]-ccf_object.data[i_frames,:,:].shape[1]//2]
            
            frame_properties['Correlation max'][i_frames]=ccf_object.data[i_frames,:,:].max()
            #Checking the threshold of the correlation     
#            if ccf_object.data[i_frames,:,:].max() < correlation_threshold and flap_ccf:
#                print('Correlation threshold '+str(correlation_threshold)+' is not reached.')
#                delta_index=[np.nan,np.nan]
                
            frame_properties['Frame similarity'][i_frames]=ccf_object.data[i_frames,:,:][tuple(np.asarray(ccf_object.data[i_frames,:,:].shape)[:]//2)]
                
            if ccf_object.data[i_frames,:,:].max() < correlation_threshold and flap_ccf:
                invalid_correlation_frame_counter+=1
            else:
                invalid_correlation_frame_counter=0.
            #Calculating the radial and poloidal velocity from the correlation map.
            frame_properties['Velocity ccf'][i_frames,0]=(coeff_r[0]*delta_index[0]+
                                                          coeff_r[1]*delta_index[1])/sample_time
            frame_properties['Velocity ccf'][i_frames,1]=(coeff_z[0]*delta_index[0]+
                                                          coeff_z[1]*delta_index[1])/sample_time       
            if not skip_structure_calculation:        
                """
                STRUCTURE SIZE CALCULATION AND MANIPULATION BASED ON STRUCTURE FINDING
                """
                
                """
                GAS CLOUD CALCULATION
                """
                if normalize_for_size or normalize_for_velocity or plot_gas:
                    flap.slice_data('GPI_GAS_CLOUD',
                                    exp_id=exp_id,
                                    slicing=slicing_frame2,
                                    output_name='GPI_GAS_CLOUD_SLICED')
                    gas_cloud_structure=nstx_gpi_one_frame_structure_finder(data_object='GPI_GAS_CLOUD_SLICED',
                                                                            exp_id=exp_id,
                                                                            filter_level=filter_level,
                                                                            nlevel=nlevel,
                                                                            levels=gas_levels,
                                                                            spatial=not structure_pixel_calc,
                                                                            pixel=structure_pixel_calc,
                                                                            remove_interlaced_structures=True,
                                                                            test_result=test_gas_cloud)
                    n_gas_structures=len(gas_cloud_structure)
                    for gas_structure in gas_cloud_structure:
                        frame_properties['GC size'][i_frames,:]+=gas_structure['Size'][:]/n_gas_structures
                        frame_properties['GC centroid'][i_frames,:]+=gas_structure['Centroid'][:]/n_gas_structures
                        frame_properties['GC position'][i_frames,:]+=gas_structure['Center'][:]/n_gas_structures
                        frame_properties['GC COG'][i_frames,:]+=gas_structure['Center of gravity'][:]/n_gas_structures
                        frame_properties['GC area'][i_frames]+=gas_structure['Area']/n_gas_structures
                        frame_properties['GC elongation'][i_frames]+=gas_structure['Elongation']/n_gas_structures
                        frame_properties['GC angle'][i_frames]+=gas_structure['Angle']/n_gas_structures
                        
                if invalid_correlation_frame_counter < valid_frame_thres:        
                        
                    if frame2_vel is None:
                        frame1_vel=flap.slice_data(object_name_str_vel,
                                                    exp_id=exp_id,
                                                    slicing=slicing_frame1, 
                                                    output_name='GPI_FRAME_1_VEL')
                    else:
                        frame1_vel=copy.deepcopy(frame2_vel)
                        
                    frame2_vel=flap.slice_data(object_name_str_vel, 
                                                exp_id=exp_id,
                                                slicing=slicing_frame2,
                                                output_name='GPI_FRAME_2_VEL')
                    
                    frame1_vel.data=np.asarray(frame1_vel.data, dtype='float64')
                    frame2_vel.data=np.asarray(frame2_vel.data, dtype='float64')
                    
                    frame2_size=flap.slice_data(object_name_str_size, 
                                                exp_id=exp_id,
                                                slicing=slicing_frame2,
                                                output_name='GPI_FRAME_2_SIZE')
                    if structures2_vel is None:
                        flap.get_data_object('GPI_FRAME_1_VEL', exp_id=exp_id)
                        structures1_vel=nstx_gpi_one_frame_structure_finder(data_object='GPI_FRAME_1_VEL',
                                                                            exp_id=exp_id,
                                                                            filter_level=filter_level,
                                                                            threshold_level=intensity_thres_level_str_vel,
                                                                            nlevel=nlevel,
                                                                            levels=levels,
                                                                            spatial=not structure_pixel_calc,
                                                                            remove_interlaced_structures=True,
                                                                            pixel=structure_pixel_calc,
                                                                            test_result=test_structures)
                    else:
                        structures1_vel=copy.deepcopy(structures2_vel)
                        
                    if structure_video_save or structure_pdf_save:
                        plt.cla()
                        test_structures=True
                    structures2_vel=nstx_gpi_one_frame_structure_finder(data_object='GPI_FRAME_2_VEL',
                                                                        exp_id=exp_id,
                                                                        filter_level=filter_level,
                                                                        threshold_level=intensity_thres_level_str_vel,
                                                                        nlevel=nlevel,
                                                                        levels=levels,
                                                                        spatial=not structure_pixel_calc,
                                                                        remove_interlaced_structures=remove_interlaced_structures,
                                                                        pixel=structure_pixel_calc,
                                                                        test_result=test_structures)
                    if not structure_video_save:
                        plt.pause(0.1)
                        if structure_pdf_save:
                            try:
                                plt.show()
                                pdf_structures.savefig()
                            except:
                                print('asdf')
                    else:
                        test_structures=False
                        fig = plt.gcf()
                        fig.canvas.draw()
                        # Get the RGBA buffer from the figure
                        w,h = fig.canvas.get_width_height()
                        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                        if buf.shape[0] == h*2 * w*2 * 3:
                            buf.shape = ( h*2, w*2, 3 ) #ON THE MAC'S INTERNAL SCREEN, THIS COULD HAPPEN NEEDS A MORE ELEGANT FIX
                        else:
                            buf.shape = ( h, w, 3 )
                        buf = cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR)
                        try:
                            video
                        except NameError:
                            height = buf.shape[0]
                            width = buf.shape[1]
                            video_codec_code='mp4v'
                            filename='NSTX_GPI_'+str(exp_id)+'fit_structures.mp4'
                            video = cv2.VideoWriter(filename,  
                                                    cv2.VideoWriter_fourcc(*video_codec_code), 
                                                    float(24), 
                                                    (width,height),
                                                    isColor=True)
                        video.write(buf)
                    structures2_size=nstx_gpi_one_frame_structure_finder(data_object='GPI_FRAME_2_SIZE',
                                                                         exp_id=exp_id,
                                                                         filter_level=filter_level,
                                                                         threshold_level=intensity_thres_level_str_size,
                                                                         nlevel=nlevel,
                                                                         levels=levels,
                                                                         remove_interlaced_structures=remove_interlaced_structures,
                                                                         spatial=True,
                                                                         test_result=test_structures)
                    
                    if structures1_vel is not None and len(structures1_vel) != 0:
                        valid_structure1_vel=True
                    else:
                        valid_structure1_vel=False            
                    if structures2_vel is not None and len(structures2_vel) != 0:
                        valid_structure2_vel=True
                    else:
                        valid_structure2_vel=False
                        
                    if structures2_size is not None and len(structures2_size) != 0:
                        valid_structure2_size=True
                    else:
                        valid_structure2_size=False
                else:
                    valid_structure1_vel=False
                    valid_structure2_vel=False
                    valid_structure2_size=False
                    print('Invalid consequtive number of frames: '+str(invalid_correlation_frame_counter))
                        
                """Structure size calculation based on the contours"""    
                #Crude average size calculation
                if valid_structure2_size:
                    #Calculating the average properties of the structures present in one frame
                    n_str2=len(structures2_size)
                    areas=np.zeros(len(structures2_size))
                    intensities=np.zeros(len(structures2_size))
                    for i_str2 in range(n_str2):
                        #Average size calculation based on the number of structures
                        areas[i_str2]=structures2_size[i_str2]['Area']
                        intensities[i_str2]=structures2_size[i_str2]['Intensity']
                        
                    #Calculating the averages based on the input setting
                    if weighting == 'number':
                        weight=1./n_str2
                    elif weighting == 'intensity':
                        weight=intensities/np.sum(intensities)
                    elif weighting == 'area':
                        weight=areas/np.sum(areas)
                        
                    for i_str2 in range(n_str2):   
                        #Quantities from Ellipse fitting
                        frame_properties['Size avg'][i_frames,:]+=structures2_size[i_str2]['Size']*weight[i_str2]
                        frame_properties['Angle avg'][i_frames]+=structures2_size[i_str2]['Angle']*weight[i_str2]
                        frame_properties['Elongation avg'][i_frames]+=structures2_size[i_str2]['Elongation']*weight[i_str2]
                        frame_properties['Position avg'][i_frames,:]+=structures2_size[i_str2]['Center']*weight[i_str2]
                        #Quantities from polygons
                        frame_properties['Area avg'][i_frames]+=structures2_size[i_str2]['Area']*weight[i_str2]
                        frame_properties['Centroid avg'][i_frames,:]+=structures2_size[i_str2]['Centroid']*weight[i_str2]
                        frame_properties['COG avg'][i_frames,:]+=structures2_size[i_str2]['Center of gravity']*weight[i_str2]
    
                    #The number of structures in a frame
                    frame_properties['Str number'][i_frames]=n_str2
                    
                    #Calculating the properties of the structure having the maximum area or intensity
                    if maxing == 'area':
                        ind_max=np.argmax(areas)
                    elif maxing == 'intensity':
                        ind_max=np.argmax(intensities)
                        
                    #Properties of the max structure:
                    frame_properties['Size max'][i_frames,:]=structures2_size[ind_max]['Size']
                    frame_properties['Area max'][i_frames]=structures2_size[ind_max]['Area']
                    frame_properties['Angle max'][i_frames]=structures2_size[ind_max]['Angle']
                    frame_properties['Elongation max'][i_frames]=structures2_size[ind_max]['Elongation']
                    frame_properties['Position max'][i_frames,:]=structures2_size[ind_max]['Center']
                    frame_properties['Centroid max'][i_frames,:]=structures2_size[ind_max]['Centroid']
                    frame_properties['COG max'][i_frames,:]=structures2_size[ind_max]['Center of gravity']
    
                    
                    #The center of gravity for the entire frame
                    x_coord=frame2_size.coordinate('Device R')[0]
                    y_coord=frame2_size.coordinate('Device z')[0]
                    frame_properties['Frame COG'][i_frames,:]=np.asarray([np.sum(x_coord*frame2_size.data)/np.sum(frame2_size.data),
                                                      np.sum(y_coord*frame2_size.data)/np.sum(frame2_size.data)])
                else:
                    #Setting np.nan if no structure is available
                    frame_properties['Size avg'][i_frames,:]=[np.nan,np.nan]
                    frame_properties['Area avg'][i_frames]=np.nan
                    frame_properties['Angle avg'][i_frames]=np.nan
                    frame_properties['Elongation avg'][i_frames]=np.nan
                    frame_properties['Position avg'][i_frames,:]=[np.nan,np.nan]
                    frame_properties['COG avg'][i_frames,:]=[np.nan,np.nan]
                    frame_properties['Centroid avg'][i_frames,:]=[np.nan,np.nan]
                    
                    frame_properties['Size max'][i_frames,:]=[np.nan,np.nan]
                    frame_properties['Area max'][i_frames]=np.nan
                    frame_properties['Angle max'][i_frames]=np.nan
                    frame_properties['Elongation max'][i_frames]=np.nan
                    frame_properties['Position max'][i_frames,:]=[np.nan,np.nan]
                    frame_properties['COG max'][i_frames,:]=[np.nan,np.nan]
                    frame_properties['Centroid max'][i_frames,:]=[np.nan,np.nan]
                    
                    frame_properties['Str number'][i_frames]=0.
                    frame_properties['Frame COG'][i_frames,:]=[np.nan,np.nan]
                    
                """Velocity calculation based on the contours"""
    
                if valid_structure1_vel and valid_structure2_vel:          
                    #if multiple structures merge into one, then previous position is their average
                    #if one structure is split up into to, then the previous position is the old's position
                    #Current velocity is the position change and sampling time's ratio
                    n_str1=len(structures1_vel)
                    n_str2=len(structures2_vel)
                    
                    prev_str_number=np.zeros([n_str2,n_str1])
                    prev_str_intensity=np.zeros([n_str2,n_str1])
                    prev_str_area=np.zeros([n_str2,n_str1])
                    prev_str_pos=np.zeros([n_str2,n_str1,2])
                    
                    current_str_area=np.zeros(n_str2)
                    current_str_intensity=np.zeros(n_str2)
                    
                    current_str_vel=np.zeros([n_str2,2])
                    if velocity_base == 'cog':
                        vel_base_key='Center of gravity'
                    elif velocity_base == 'center':
                        vel_base_key='Center'
                    elif velocity_base == 'centroid':
                        vel_base_key='Centroid'
                    elif velocity_base == 'frame cog':
                        vel_base_key='Centroid'                             #Just error handling, not actually used for the results
    
                    for i_str2 in range(n_str2):
                        for i_str1 in range(n_str1):
                            if structures2_vel[i_str2]['Half path'].intersects_path(structures1_vel[i_str1]['Half path']):
                                prev_str_number[i_str2,i_str1]=1.
                                prev_str_intensity[i_str2,i_str1]=structures1_vel[i_str1]['Intensity']
                                prev_str_area[i_str2,i_str1]=structures1_vel[i_str1]['Area']
                                prev_str_pos[i_str2,i_str1,:]=structures1_vel[i_str1][vel_base_key]
                                
                        if np.sum(prev_str_number[i_str2,:]) > 0:
    
                            current_str_intensity[i_str2]=structures2_vel[i_str2]['Intensity']
                            current_str_area[i_str2]=structures2_vel[i_str2]['Area']
                            
                            if weighting == 'number':
                                weight=1./np.sum(prev_str_number[i_str2,:])
                            elif weighting == 'intensity':
                                weight=prev_str_intensity[i_str2,:]/np.sum(prev_str_intensity[i_str2,:])
                            elif weighting == 'area':
                                weight=prev_str_area[i_str2,:]/np.sum(prev_str_area[i_str2,:])
                                
                            prev_str_pos_avg=np.asarray([np.sum(prev_str_pos[i_str2,:,0]*weight),
                                                         np.sum(prev_str_pos[i_str2,:,1]*weight)])
                            current_str_pos=structures2_vel[i_str2][vel_base_key]                        
                            current_str_vel[i_str2,:]=(current_str_pos-prev_str_pos_avg)/sample_time
    
                            
                    #Criterium for validity of the velocity
                    #If the structures are not overlapping, then the velocity cannot be valid.
                    ind_valid=np.where(np.sum(prev_str_number,axis=1) > 0)
                    
                    if np.sum(prev_str_number) != 0:
                        if weighting == 'number':
                            weight=1./n_str2
                        elif weighting == 'intensity':
                            weight=current_str_intensity/np.sum(current_str_intensity)
                        elif weighting == 'area':
                            weight=current_str_area/np.sum(current_str_area)
                            
                        #Calculating the average based on the number of valid moving structures
                        frame_properties['Velocity str avg'][i_frames,0]=np.sum(current_str_vel[ind_valid,0]*weight[ind_valid])
                        frame_properties['Velocity str avg'][i_frames,1]=np.sum(current_str_vel[ind_valid,1]*weight[ind_valid])
    
                    #Calculating the properties of the structure having the maximum area or intensity
                        if maxing == 'area':
                            ind_max=np.argmax(current_str_area[ind_valid])
                        elif maxing == 'intensity':
                            ind_max=np.argmax(current_str_intensity[ind_valid])
    
                        frame_properties['Velocity str max'][i_frames,:]=current_str_vel[ind_max,:]
                        
                        if abs(np.mean(current_str_vel[:,0])) > 10e3:
                            print('Structure velocity over 10km/s')
                            print('Current structure velocity: '+str(current_str_vel))
                            print('Position difference: '+str(structures2_vel[i_str2]['Center']-prev_str_pos[i_str2,:]))
                            
                        if velocity_threshold is not None:
                            if (abs(frame_properties['Velocity str max'][i_frames,0]) > velocity_threshold or
                                abs(frame_properties['Velocity str max'][i_frames,1]) > velocity_threshold):
                                print('Velocity validity threshold reached. Setting np.nan as velocity.')
                                frame_properties['Velocity str max'][i_frames,:]=[np.nan,np.nan]
                    else:
                        frame_properties['Velocity str avg'][i_frames,:]=[np.nan,np.nan]
                        frame_properties['Velocity str max'][i_frames,:]=[np.nan,np.nan]
                else:
                    frame_properties['Velocity str avg'][i_frames,:]=[np.nan,np.nan]
                    frame_properties['Velocity str max'][i_frames,:]=[np.nan,np.nan]
    
        if (structure_video_save):
            cv2.destroyAllWindows()
            video.release()  
            del video
        if structure_pdf_save:
            pdf_structures.close()
        #Saving results into a pickle file
        
        pickle.dump(frame_properties,open(pickle_filename, 'wb'))
        if test:
            plt.close()
    else:
        print('--- Loading data from the pickle file ---')
        frame_properties=pickle.load(open(pickle_filename, 'rb'))
        
    if not filename_was_none and not time_range is None:
        sample_time=frame_properties['Time'][1]-frame_properties['Time'][0]
        if time_range[0] < frame_properties['Time'][0]-sample_time or time_range[1] > frame_properties['Time'][-1]+sample_time:
            raise ValueError('Please run the calculation again with the timerange. The pickle file doesn\'t have the desired range')
    if time_range is None:
        time_range=[frame_properties['Time'][0],frame_properties['Time'][-1]]
    if gpi_plane_calculation:
        coeff_r=np.asarray([3.7183594,-0.77821046,1402.8097])/1000. #The coordinates are in meters, the coefficients are in mm
        coeff_z=np.asarray([0.18090118,3.0657776,70.544312])/1000.  #The coordinates are in meters, the coefficients are in mm
        coeff_r_new=3./800.
        coeff_z_new=3./800.
        det=coeff_r[0]*coeff_z[1]-coeff_z[0]*coeff_r[1]
        
        for key in ['Velocity ccf','Velocity str max','Velocity str avg','Size max','Size avg']:
            orig=copy.deepcopy(frame_properties[key])
            frame_properties[key][:,0]=coeff_r_new/det*(coeff_z[1]*orig[:,0]-coeff_r[1]*orig[:,1])
            frame_properties[key][:,1]=coeff_z_new/det*(-coeff_z[0]*orig[:,0]+coeff_r[0]*orig[:,1])
        
        frame_properties['Elongation max'][:]=(frame_properties['Size max'][:,0]-frame_properties['Size max'][:,1])/(frame_properties['Size max'][:,0]+frame_properties['Size max'][:,1])
        frame_properties['Elongation avg'][:]=(frame_properties['Size avg'][:,0]-frame_properties['Size avg'][:,1])/(frame_properties['Size avg'][:,0]+frame_properties['Size avg'][:,1])
    
    #Calculating the distance from the separatrix
    frame_properties['Separatrix dist avg']=np.zeros(frame_properties['Position avg'].shape[0])
    frame_properties['Separatrix dist max']=np.zeros(frame_properties['Position max'].shape[0])
    
    elm_time=(frame_properties['Time'][-1]+frame_properties['Time'][0])/2
    
    R_sep=flap.get_data('NSTX_MDSPlus',
                        name='\EFIT02::\RBDRY',
                        exp_id=exp_id,
                        object_name='SEP R OBJ').slice_data(slicing={'Time':elm_time}).data
    z_sep=flap.get_data('NSTX_MDSPlus',
                        name='\EFIT02::\ZBDRY',
                        exp_id=exp_id,
                        object_name='SEP Z OBJ').slice_data(slicing={'Time':elm_time}).data
    sep_GPI_ind=np.where(np.logical_and(R_sep > coeff_r[2],
                                              np.logical_and(z_sep > coeff_z[2],
                                                             z_sep < coeff_z[2]+79*coeff_z[0]+64*coeff_z[1])))
    try:
        sep_GPI_ind=np.asarray(sep_GPI_ind[0])
        sep_GPI_ind=np.insert(sep_GPI_ind,0,sep_GPI_ind[0]-1)
        sep_GPI_ind=np.insert(sep_GPI_ind,len(sep_GPI_ind),sep_GPI_ind[-1]+1)            
        z_sep_GPI=z_sep[(sep_GPI_ind)]
        R_sep_GPI=R_sep[sep_GPI_ind]
        GPI_z_vert=coeff_z[0]*np.arange(80)/80*64+coeff_z[1]*np.arange(80)+coeff_z[2]
        R_sep_GPI_interp=np.interp(GPI_z_vert,np.flip(z_sep_GPI),np.flip(R_sep_GPI))
        z_sep_GPI_interp=GPI_z_vert
        
        for key in ['max','avg']:
            for ind_time in range(len(frame_properties['Position '+key][:,0])):
                frame_properties['Separatrix dist '+key][ind_time]=np.min(np.sqrt((frame_properties['Position '+key][ind_time,0]-R_sep_GPI_interp)**2 + 
                                                                          (frame_properties['Position '+key][ind_time,1]-z_sep_GPI_interp)**2))
                ind_z_min=np.argmin(np.abs(z_sep_GPI-frame_properties['Position '+key][ind_time,1]))
                if z_sep_GPI[ind_z_min] >= frame_properties['Position '+key][ind_time,1]:
                    ind1=ind_z_min
                    ind2=ind_z_min+1
                else:
                    ind1=ind_z_min-1
                    ind2=ind_z_min
                    
                radial_distance=frame_properties['Position '+key][ind_time,0]-((frame_properties['Position '+key][ind_time,1]-z_sep_GPI[ind2])/(z_sep_GPI[ind1]-z_sep_GPI[ind2])*(R_sep_GPI[ind1]-R_sep_GPI[ind2])+R_sep_GPI[ind2])
                if radial_distance < 0:
                    frame_properties['Separatrix dist '+key][ind_time]*=-1
    except:
        pass
    
    #Plotting the results
    if plot or pdf:
        #This is a bit unusual here, but necessary due to the structure size calculation based on the contours which are not plot
        if plot:
            import matplotlib
            matplotlib.use('QT5Agg')
            import matplotlib.pyplot as plt
        else:
            import matplotlib
            matplotlib.use('agg')
            import matplotlib.pyplot as plt     
        
        nan_ind=np.where(frame_properties['Correlation max'] < correlation_threshold)
        frame_properties['Velocity ccf'][nan_ind,0] = np.nan
        if plot_time_range is not None:
            if plot_time_range[0] < time_range[0] or plot_time_range[1] > time_range[1]:
                raise ValueError('The plot time range is not in the interval of the original time range.')
            time_range=plot_time_range
            
        plot_index=np.logical_and(np.logical_not(np.isnan(frame_properties['Velocity ccf'][:,0])),
                                  np.logical_and(frame_properties['Time'] >= time_range[0],
                                                 frame_properties['Time'] <= time_range[1]))
        
        plot_index_structure=np.logical_and(np.logical_not(np.isnan(frame_properties['Elongation avg'])),
                                            np.logical_and(frame_properties['Time'] >= time_range[0],
                                                           frame_properties['Time'] <= time_range[1]))

        #Plotting the radial velocity
        if pdf:
            wd=flap.config.get_all_section('Module NSTX_GPI')['Working directory']
            filename=flap_nstx.analysis.filename(exp_id=exp_id,
                                                 working_directory=wd+'/plots',
                                                 time_range=time_range,
                                                 purpose='ccf velocity',
                                                 comment=comment+'_ct_'+str(correlation_threshold))
            pdf_filename=filename+'.pdf'
            pdf_pages=PdfPages(pdf_filename)
        if plot_for_publication:
            figsize=(8.5/2.54, 
                     8.5/2.54/1.618*1.1)
            plt.rc('font', family='serif', serif='Helvetica')
            labelsize=9
            linewidth=0.5
            major_ticksize=2
            plt.rc('text', usetex=False)
            plt.rcParams['pdf.fonttype'] = 42
            plt.rcParams['ps.fonttype'] = 42
            plt.rcParams['lines.linewidth'] = linewidth
            plt.rcParams['axes.linewidth'] = linewidth
            plt.rcParams['axes.labelsize'] = labelsize
            plt.rcParams['axes.titlesize'] = labelsize
            
            plt.rcParams['xtick.labelsize'] = labelsize
            plt.rcParams['xtick.major.size'] = major_ticksize
            plt.rcParams['xtick.major.width'] = linewidth
            plt.rcParams['xtick.minor.width'] = linewidth/2
            plt.rcParams['xtick.minor.size'] = major_ticksize/2
            
            plt.rcParams['ytick.labelsize'] = labelsize
            plt.rcParams['ytick.major.width'] = linewidth
            plt.rcParams['ytick.major.size'] = major_ticksize
            plt.rcParams['ytick.minor.width'] = linewidth/2
            plt.rcParams['ytick.minor.size'] = major_ticksize/2
            plt.rcParams['legend.fontsize'] = labelsize
        else:
            figsize=None
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(frame_properties['Time'][plot_index], 
                 frame_properties['Velocity ccf'][plot_index,0])
        if plot_scatter:
            ax.scatter(frame_properties['Time'][plot_index], 
                        frame_properties['Velocity ccf'][plot_index,0], 
                        s=5, 
                        marker='o')
        if overplot_str_vel:
            ax.plot(frame_properties['Time'][plot_index],
                    frame_properties['Velocity str max'][plot_index,0], 
                    linewidth=0.3,
                    color='red')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('v_rad[m/s]')
        ax.set_xlim(time_range)
        ax.set_title('Radial velocity of '+str(exp_id))
        if plot_for_publication:
            x1,x2=ax.get_xlim()
            y1,y2=ax.get_ylim()
            ax.set_aspect((x2-x1)/(y2-y1)/1.618)
            ax.set_title('Rad vel ccf')
        fig.tight_layout()
        if pdf:
            pdf_pages.savefig()
        
        #Plotting the poloidal velocity
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(frame_properties['Time'][plot_index], 
                 frame_properties['Velocity ccf'][plot_index,1]) 
        if plot_scatter:
            ax.scatter(frame_properties['Time'][plot_index], 
                        frame_properties['Velocity ccf'][plot_index,1], 
                        s=5, 
                        marker='o')
        if overplot_str_vel:
            ax.plot(frame_properties['Time'][plot_index],
                    frame_properties['Velocity str max'][plot_index,1], 
                    linewidth=0.3,
                    color='red')

        ax.set_xlabel('Time [s]')
        ax.set_ylabel('v_pol[m/s]')
        ax.set_title('Poloidal velocity of '+str(exp_id))
        ax.set_xlim(time_range)
        if plot_for_publication:
            x1,x2=ax.get_xlim()
            y1,y2=ax.get_ylim()
            ax.set_aspect((x2-x1)/(y2-y1)/1.618)
            ax.set_title('Pol vel ccf')
        fig.tight_layout()
        if pdf:
            pdf_pages.savefig()

        #Plotting the radial velocity from the structures
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(frame_properties['Time'][plot_index_structure],
                frame_properties['Velocity str max'][plot_index_structure,0], 
                color='red')     
        if plot_scatter:
            ax.scatter(frame_properties['Time'][plot_index_structure], 
                        frame_properties['Velocity str max'][plot_index_structure,0], 
                        s=5, 
                        marker='o',
                        color='red')
        if overplot_average:
            ax.plot(frame_properties['Time'][plot_index_structure],
                    frame_properties['Velocity str avg'][plot_index_structure,0])     
            if plot_scatter:
                ax.scatter(frame_properties['Time'][plot_index_structure], 
                            frame_properties['Velocity str avg'][plot_index_structure,0], 
                            s=5, 
                            marker='o')

        ax.set_xlabel('Time [s]')
        ax.set_ylabel('v_rad[m/s]')
        ax.set_title('Radial velocity of '+str(exp_id))
        ax.set_xlim(time_range)
        if plot_for_publication:
            x1,x2=ax.get_xlim()
            y1,y2=ax.get_ylim()
            ax.set_aspect((x2-x1)/(y2-y1)/1.618)
            ax.set_title('Rad vel str')
        fig.tight_layout()
        if pdf:
            pdf_pages.savefig()
        
        #Plotting the poloidal velocity from the structures
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(frame_properties['Time'][plot_index_structure],
                frame_properties['Velocity str max'][plot_index_structure,1], 
                color='red')     
        if plot_scatter:
            ax.scatter(frame_properties['Time'][plot_index_structure], 
                        frame_properties['Velocity str max'][plot_index_structure,1], 
                        s=5, 
                        marker='o',
                        color='red')
        if overplot_average:
            ax.plot(frame_properties['Time'][plot_index_structure],
                    frame_properties['Velocity str avg'][plot_index_structure,1])     
            if plot_scatter:
                ax.scatter(frame_properties['Time'][plot_index_structure], 
                           frame_properties['Velocity str avg'][plot_index_structure,1], 
                           s=5, 
                           marker='o')

        ax.set_xlabel('Time [s]')
        ax.set_ylabel('v_pol[m/s]')
        ax.set_title('Poloidal velocity of '+str(exp_id))
        ax.set_xlim(time_range)
        if plot_for_publication:
            x1,x2=ax.get_xlim()
            y1,y2=ax.get_ylim()
            ax.set_aspect((x2-x1)/(y2-y1)/1.618)
            ax.set_title('Pol vel str')
        fig.tight_layout()
        if pdf:
            pdf_pages.savefig()

        
        #Plotting the correlation coefficients
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(frame_properties['Time'], 
                 frame_properties['Frame similarity'])
        if plot_scatter:
            ax.scatter(frame_properties['Time'][plot_index], 
                     frame_properties['Frame similarity'][plot_index],
                     s=5)
        ax.plot(frame_properties['Time'],
                frame_properties['Correlation max'], 
                color='red')
        if plot_scatter:
            ax.scatter(frame_properties['Time'][plot_index],
                       frame_properties['Correlation max'][plot_index],
                       s=5,
                       color='red')   
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Correlation coefficient')
        ax.set_title('Maximum correlation (red) and \nframe similarity (blue) of '+str(exp_id))
        ax.set_xlim(time_range)
        ax.set_ylim([0,1])
        if plot_for_publication:
            x1,x2=ax.get_xlim()
            y1,y2=ax.get_ylim()
            ax.set_aspect((x2-x1)/(y2-y1)/1.618)
            ax.set_title('Max corr')
        fig.tight_layout()
        if pdf:
            pdf_pages.savefig()

        #Plotting the correlation coefficients
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(frame_properties['Time'], 
                frame_properties['GPI Dalpha'][0][1:-1])
        if plot_scatter:
            ax.scatter(frame_properties['Time'][plot_index], 
                       frame_properties['GPI Dalpha'][0][1:-1][plot_index],
                       s=5)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Average GPI signal')
        ax.set_title('Average GPI signal of '+str(exp_id))
        ax.set_xlim(time_range)
        
        if plot_for_publication:
            x1,x2=ax.get_xlim()
            y1,y2=ax.get_ylim()
            ax.set_aspect((x2-x1)/(y2-y1)/1.618)
            ax.set_title('Avg gpi')
        fig.tight_layout()
        if pdf:
            pdf_pages.savefig()            

        #Plotting the radial size
        fig, ax = plt.subplots(figsize=figsize)
        #Maximum
        ax.plot(frame_properties['Time'][plot_index_structure], 
                 frame_properties['Size max'][:,0][plot_index_structure],
                 color='red')
        if plot_scatter:
            ax.scatter(frame_properties['Time'][plot_index_structure], 
                        frame_properties['Size max'][:,0][plot_index_structure], 
                        s=5, 
                        marker='o', 
                        color='red')
        #Average
        if overplot_average:
            ax.plot(frame_properties['Time'][plot_index_structure], 
                     frame_properties['Size avg'][:,0][plot_index_structure]) 
            if plot_scatter:
                ax.scatter(frame_properties['Time'][plot_index_structure], 
                            frame_properties['Size avg'][:,0][plot_index_structure], 
                            s=5, 
                            marker='o')
        if plot_gas:
            ax.plot(frame_properties['Time'], 
                     frame_properties['GC size'][:,0]) 
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Radial size [m]')
        ax.set_title('Average (blue) and '+maxing+' maximum (red) radial\n size of structures of '+str(exp_id))
        ax.set_xlim(time_range)
        if plot_for_publication:
            x1,x2=ax.get_xlim()
            y1,y2=ax.get_ylim()
            ax.set_aspect((x2-x1)/(y2-y1)/1.618)
            ax.set_title('Rad size')
        fig.tight_layout()
        if pdf:
            pdf_pages.savefig()
            
        #Plotting the poloidal size
        fig, ax = plt.subplots(figsize=figsize)
        #Maximum
        ax.plot(frame_properties['Time'][plot_index_structure], 
                 frame_properties['Size max'][:,1][plot_index_structure], 
                 color='red') 
        if plot_scatter:
            ax.scatter(frame_properties['Time'][plot_index_structure], 
                        frame_properties['Size max'][:,1][plot_index_structure], 
                        s=5, 
                        marker='o', 
                        color='red')
        #Average
        if overplot_average:
            ax.plot(frame_properties['Time'][plot_index_structure], 
                     frame_properties['Size avg'][:,1][plot_index_structure]) 
            if plot_scatter:
                ax.scatter(frame_properties['Time'][plot_index_structure], 
                            frame_properties['Size avg'][:,1][plot_index_structure], 
                            s=5, 
                            marker='o')
        if plot_gas:
            ax.plot(frame_properties['Time'], 
                     frame_properties['GC size'][:,1]) 
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Poloidal size [m]')
        ax.set_title('Average (blue) and '+maxing+' maximum (red) poloidal\n size of structures of '+str(exp_id))    
        ax.set_xlim(time_range)
        if plot_for_publication:
            x1,x2=ax.get_xlim()
            y1,y2=ax.get_ylim()
            ax.set_aspect((x2-x1)/(y2-y1)/1.618)
            ax.set_title('Pol size')
        fig.tight_layout()
        if pdf:
            pdf_pages.savefig()    
            
        #Plotting the radial position of the fit ellipse
        fig, ax = plt.subplots(figsize=figsize)
        #Maximum
        ax.plot(frame_properties['Time'][plot_index_structure], 
                 frame_properties['Position max'][:,0][plot_index_structure], 
                 color='red') 
        if plot_scatter:
            ax.scatter(frame_properties['Time'][plot_index_structure], 
                        frame_properties['Position max'][:,0][plot_index_structure], 
                        s=5, 
                        marker='o', 
                        color='red')
        #Average
        if overplot_average:
            ax.plot(frame_properties['Time'][plot_index_structure], 
                     frame_properties['Position avg'][:,0][plot_index_structure]) 
            if plot_scatter:
                ax.scatter(frame_properties['Time'][plot_index_structure], 
                            frame_properties['Position avg'][:,0][plot_index_structure], 
                            s=5, 
                            marker='o')
        if plot_gas:
            ax.plot(frame_properties['Time'], 
                     frame_properties['GC position'][:,0]) 
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Radial position [m]')            
        ax.set_title('Average (blue) and '+maxing+' maximum (red) radial\n position of structures of '+str(exp_id))   
        ax.set_xlim(time_range)
        if plot_for_publication:
            x1,x2=ax.get_xlim()
            y1,y2=ax.get_ylim()
            ax.set_aspect((x2-x1)/(y2-y1)/1.618)
            ax.set_title('Rad pos')
        fig.tight_layout()
        if pdf:
            pdf_pages.savefig()                  

        #Plotting the radial centroid of the half path
        fig, ax = plt.subplots(figsize=figsize)
        #Maximum
        ax.plot(frame_properties['Time'][plot_index_structure], 
                 frame_properties['Centroid max'][plot_index_structure,0], 
                 color='red')
        if plot_scatter:
            ax.scatter(frame_properties['Time'][plot_index_structure], 
                        frame_properties['Centroid max'][plot_index_structure,0], 
                        s=5, 
                        marker='o', 
                        color='red')
        #Average
        if overplot_average:
            ax.plot(frame_properties['Time'][plot_index_structure], 
                     frame_properties['Centroid avg'][plot_index_structure,0]) 
            if plot_scatter:
                ax.scatter(frame_properties['Time'][plot_index_structure], 
                            frame_properties['Centroid avg'][plot_index_structure,0], 
                            s=5, 
                            marker='o')
        if plot_gas:
            ax.plot(frame_properties['Time'], 
                     frame_properties['GC centroid'][:,0]) 
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Radial centroid [m]')
        ax.set_title('Average (blue) and '+maxing+' maximum (red) radial\n centroid of structures of '+str(exp_id))   
        ax.set_xlim(time_range)
        if plot_for_publication:
            x1,x2=ax.get_xlim()
            y1,y2=ax.get_ylim()
            ax.set_aspect((x2-x1)/(y2-y1)/1.618)
            ax.set_title('Rad centroid')
        fig.tight_layout()
        if pdf:
            pdf_pages.savefig() 

        #Plotting the radial COG of the structure
        fig, ax = plt.subplots(figsize=figsize)
        #Maximum
        ax.plot(frame_properties['Time'][plot_index_structure], 
                 frame_properties['COG max'][plot_index_structure,0], 
                 color='red') 
        if plot_scatter:
            ax.scatter(frame_properties['Time'][plot_index_structure], 
                        frame_properties['COG max'][plot_index_structure,0], 
                        s=5, 
                        marker='o', 
                        color='red')            
        #Average
        if overplot_average:
            ax.plot(frame_properties['Time'][plot_index_structure], 
                     frame_properties['COG avg'][plot_index_structure,0]) 
            if plot_scatter:
                ax.scatter(frame_properties['Time'][plot_index_structure], 
                            frame_properties['COG avg'][plot_index_structure,0], 
                            s=5, 
                            marker='o')
        if plot_gas:
            ax.plot(frame_properties['Time'], 
                     frame_properties['GC COG'][:,0]) 
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Radial COG [m]')
        ax.set_title('Average (blue) and '+maxing+' maximum (red) radial\n center of gravity of structures of '+str(exp_id))   
        ax.set_xlim(time_range)
        if plot_for_publication:
            x1,x2=ax.get_xlim()
            y1,y2=ax.get_ylim()
            ax.set_aspect((x2-x1)/(y2-y1)/1.618)
            ax.set_title('Rad COG')
        fig.tight_layout()
        if pdf:
            pdf_pages.savefig()      
            
        #Plotting the poloidal position of the fit ellipse
        fig, ax = plt.subplots(figsize=figsize)
        #Maximum
        ax.plot(frame_properties['Time'][plot_index_structure], 
                 frame_properties['Position max'][:,1][plot_index_structure], 
                 color='red')
        if plot_scatter:
            ax.scatter(frame_properties['Time'][plot_index_structure], 
                        frame_properties['Position max'][:,1][plot_index_structure], 
                        s=5, 
                        marker='o', 
                        color='red')
        #Average
        if overplot_average:
            ax.plot(frame_properties['Time'][plot_index_structure], 
                     frame_properties['Position avg'][:,1][plot_index_structure]) 
            if plot_scatter:
                ax.scatter(frame_properties['Time'][plot_index_structure], 
                            frame_properties['Position avg'][:,1][plot_index_structure], 
                            s=5, 
                            marker='o')
        if plot_gas:
            ax.plot(frame_properties['Time'], 
                     frame_properties['GC position'][:,1])         
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Poloidal position [m]')
        ax.set_title('Average (blue) and '+maxing+' maximum (red) poloidal\n position of structures of '+str(exp_id))   
        ax.set_xlim(time_range)
        if plot_for_publication:
            x1,x2=ax.get_xlim()
            y1,y2=ax.get_ylim()
            ax.set_aspect((x2-x1)/(y2-y1)/1.618)
            ax.set_title('Pol pos')
        fig.tight_layout()
        if pdf:
            pdf_pages.savefig()                               
            
        #Plotting the poloidal centroid of the half path
        fig, ax = plt.subplots(figsize=figsize)
        #Maximum
        ax.plot(frame_properties['Time'][plot_index_structure], 
                 frame_properties['Centroid max'][plot_index_structure,1], 
                 color='red')
        if plot_scatter:
            ax.scatter(frame_properties['Time'][plot_index_structure], 
                        frame_properties['Centroid max'][plot_index_structure,1], 
                        s=5, 
                        marker='o', 
                        color='red')
        #Average      
        if overplot_average:
            ax.plot(frame_properties['Time'][plot_index_structure], 
                     frame_properties['Centroid avg'][plot_index_structure,1]) 
            if plot_scatter:
                ax.scatter(frame_properties['Time'][plot_index_structure], 
                            frame_properties['Centroid avg'][plot_index_structure,1], 
                            s=5, 
                            marker='o')
        if plot_gas:
            ax.plot(frame_properties['Time'], 
                     frame_properties['GC centroid'][:,1]) 
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Poloidal centroid [m]')
        ax.set_title('Average (blue) and '+maxing+' maximum (red) poloidal\n centroid of structures of '+str(exp_id))   
        ax.set_xlim(time_range)
        if plot_for_publication:
            x1,x2=ax.get_xlim()
            y1,y2=ax.get_ylim()
            ax.set_aspect((x2-x1)/(y2-y1)/1.618)
            ax.set_title('Pol centroid')
        fig.tight_layout()
        if pdf:
            pdf_pages.savefig()          
            
        #Plotting the poloidal COG of the structure
        fig, ax = plt.subplots(figsize=figsize)
        #Maximum
        ax.plot(frame_properties['Time'][plot_index_structure], 
                 frame_properties['COG max'][plot_index_structure,1], 
                 color='red')
        if plot_scatter:
            ax.scatter(frame_properties['Time'][plot_index_structure], 
                        frame_properties['COG max'][plot_index_structure,1], 
                        s=5, 
                        marker='o', 
                        color='red')
        #Average     
        if overplot_average:
            ax.plot(frame_properties['Time'][plot_index_structure], 
                     frame_properties['COG avg'][plot_index_structure,1]) 
            if plot_scatter:
                ax.scatter(frame_properties['Time'][plot_index_structure], 
                            frame_properties['COG avg'][plot_index_structure,1], 
                            s=5, 
                            marker='o')
        if plot_gas:
            ax.plot(frame_properties['Time'], 
                     frame_properties['GC COG'][:,1]) 
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Poloidal COG [m]')
        ax.set_title('Average (blue) and '+maxing+' maximum (red) radial\n center of gravity of structures of '+str(exp_id))   
        ax.set_xlim(time_range)
        if plot_for_publication:
            x1,x2=ax.get_xlim()
            y1,y2=ax.get_ylim()
            ax.set_aspect((x2-x1)/(y2-y1)/1.618)
            ax.set_title('Pol COG')
        fig.tight_layout()
        if pdf:
            pdf_pages.savefig()
            
        #Plotting the radial size
        fig, ax = plt.subplots(figsize=figsize)
        #Maximum
        ax.plot(frame_properties['Time'][plot_index_structure], 
                 frame_properties['Separatrix dist max'][plot_index_structure],
                 color='red')
        if plot_scatter:
            ax.scatter(frame_properties['Time'][plot_index_structure], 
                        frame_properties['Separatrix dist max'][plot_index_structure], 
                        s=5, 
                        marker='o', 
                        color='red')
        #Average
        if overplot_average:
            ax.plot(frame_properties['Time'][plot_index_structure], 
                     frame_properties['Separatrix dist avg'][plot_index_structure]) 
            if plot_scatter:
                ax.scatter(frame_properties['Time'][plot_index_structure], 
                            frame_properties['Separatrix dist avg'][plot_index_structure], 
                            s=5, 
                            marker='o')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Separatrix dist [m]')
        ax.set_title('Average (blue) and '+maxing+' maximum (red) separatrix \n distance of structures of '+str(exp_id))
        ax.set_xlim(time_range)
        if plot_for_publication:
            x1,x2=ax.get_xlim()
            y1,y2=ax.get_ylim()
            ax.set_aspect((x2-x1)/(y2-y1)/1.618)
            ax.set_title('Sep dist')
        fig.tight_layout()
        if pdf:
            pdf_pages.savefig()
            
        #Plotting the elongation
        fig, ax = plt.subplots(figsize=figsize)
        #Maximum
        ax.plot(frame_properties['Time'][plot_index_structure], 
                 frame_properties['Elongation max'][plot_index_structure], 
                 color='red') 
        if plot_scatter:
            ax.scatter(frame_properties['Time'][plot_index_structure], 
                        frame_properties['Elongation max'][plot_index_structure], 
                        s=5, 
                        marker='o', 
                        color='red')
        #Average
        if overplot_average:
            ax.plot(frame_properties['Time'][plot_index_structure], 
                     frame_properties['Elongation avg'][plot_index_structure]) 
            if plot_scatter:
                ax.scatter(frame_properties['Time'][plot_index_structure], 
                            frame_properties['Elongation avg'][plot_index_structure], 
                            s=5, 
                            marker='o')
        if plot_gas:
            ax.plot(frame_properties['Time'], 
                     frame_properties['GC elongation'])         
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Elongation')
        ax.set_title('Average (blue) and '+maxing+' maximum (red) elongation\n of structures of '+str(exp_id))   
        ax.set_xlim(time_range)
        if plot_for_publication:
            x1,x2=ax.get_xlim()
            y1,y2=ax.get_ylim()
            ax.set_aspect((x2-x1)/(y2-y1)/1.618)
            ax.set_title('Elongation')
        fig.tight_layout()
        if pdf:
            pdf_pages.savefig()                
        
        #Plotting the angle
        fig, ax = plt.subplots(figsize=figsize)
        #Maximum
        ax.plot(frame_properties['Time'][plot_index_structure], 
                 frame_properties['Angle max'][plot_index_structure], 
                 color='red') 
        if plot_scatter:
            ax.scatter(frame_properties['Time'][plot_index_structure], 
                        frame_properties['Angle max'][plot_index_structure], 
                        s=5, 
                        marker='o', 
                        color='red')
        #Average           
        if overplot_average:
            ax.plot(frame_properties['Time'][plot_index_structure], 
                     frame_properties['Angle avg'][plot_index_structure]) 
            if plot_scatter:
                ax.scatter(frame_properties['Time'][plot_index_structure], 
                            frame_properties['Angle avg'][plot_index_structure], 
                            s=5, 
                            marker='o')
        if plot_gas:
            ax.plot(frame_properties['Time'], 
                     frame_properties['GC angle'])   
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Angle [rad]')
        ax.set_title('Average (blue) and '+maxing+' maximum (red) angle\n of structures of '+str(exp_id))   
        ax.set_xlim(time_range)
        if plot_for_publication:
            x1,x2=ax.get_xlim()
            y1,y2=ax.get_ylim()
            ax.set_aspect((x2-x1)/(y2-y1)/1.618)
            ax.set_title('Angle')
        fig.tight_layout()
        if pdf:
            pdf_pages.savefig()
            
        #Plotting the area
        fig, ax = plt.subplots(figsize=figsize)
        #Maximum
        ax.plot(frame_properties['Time'][plot_index_structure], 
                frame_properties['Area max'][plot_index_structure], 
                color='red') 
        if plot_scatter:
            ax.scatter(frame_properties['Time'][plot_index_structure], 
                        frame_properties['Area max'][plot_index_structure], 
                        s=5, 
                        marker='o', 
                        color='red')
        #Average
        if overplot_average:
            ax.plot(frame_properties['Time'][plot_index_structure], 
                     frame_properties['Area avg'][plot_index_structure]) 
            if plot_scatter:
                ax.scatter(frame_properties['Time'][plot_index_structure], 
                            frame_properties['Area avg'][plot_index_structure], 
                            s=5, 
                            marker='o')
        if plot_gas:
            ax.plot(frame_properties['Time'], 
                     frame_properties['GC area'])   
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Area [m2]')
        ax.set_title('Average (blue) and '+maxing+' maximum (red) area\n of structures of '+str(exp_id))   
        ax.set_xlim(time_range)
        if plot_for_publication:
            x1,x2=ax.get_xlim()
            y1,y2=ax.get_ylim()
            ax.set_aspect((x2-x1)/(y2-y1)/1.618)
            ax.set_title('Area')
        fig.tight_layout()
        if pdf:
            pdf_pages.savefig()
            
        #Plotting the number of structures 
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(frame_properties['Time'][:], 
                 frame_properties['Str number'][:]) 
        if plot_scatter:
            ax.scatter(frame_properties['Time'][:], 
                        frame_properties['Str number'][:], 
                        s=5, 
                        marker='o')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Str number')
        ax.set_title('Number of structures vs. time of '+str(exp_id))
        ax.set_xlim(time_range)
        if plot_for_publication:
            x1,x2=ax.get_xlim()
            y1,y2=ax.get_ylim()
            ax.set_aspect((x2-x1)/(y2-y1)/1.618)
            ax.set_title('Str num')
        fig.tight_layout()
        if pdf:
            pdf_pages.savefig()
           
        #Plotting the frame radial center of gravity
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(frame_properties['Time'][plot_index_structure], 
                 frame_properties['Frame COG'][plot_index_structure,0]) 
        ax.scatter(frame_properties['Time'][plot_index_structure], 
                    frame_properties['Frame COG'][plot_index_structure,0], 
                    s=5, 
                    marker='o', 
                    color='red')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Radial frame COG [m]')
        ax.set_title('Radial center of gravity of the frame vs. time of '+str(exp_id))
        ax.set_xlim(time_range)
        if plot_for_publication:
            x1,x2=ax.get_xlim()
            y1,y2=ax.get_ylim()
            ax.set_aspect((x2-x1)/(y2-y1)/1.618)
            ax.set_title('Rad frame COG')
        fig.tight_layout()
        if pdf:
            pdf_pages.savefig()
            
        #Plotting the frame poloidal center of gravity
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(frame_properties['Time'][plot_index_structure], 
                 frame_properties['Frame COG'][plot_index_structure,1]) 
        ax.scatter(frame_properties['Time'][plot_index_structure], 
                    frame_properties['Frame COG'][plot_index_structure,1], 
                    s=5, 
                    marker='o', 
                    color='red')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Poloidal frame COG [m]')
        ax.set_title('Poloidal center of gravity of the frame vs. time of '+str(exp_id))
        ax.set_xlim(time_range)
        if plot_for_publication:
            x1,x2=ax.get_xlim()
            y1,y2=ax.get_ylim()
            ax.set_aspect((x2-x1)/(y2-y1)/1.618)
            ax.set_title('Pol frame COG')
        fig.tight_layout()
        if pdf:
            pdf_pages.savefig()            
                                
        if pdf:
           pdf_pages.close()
                
        if test_histogram:
            #Plotting the velocity histogram if test is set.
            hist,bin_edge=np.histogram(frame_properties['Velocity'][:,1], bins=(np.arange(100)/100.-0.5)*24000.)
            bin_edge=(bin_edge[:99]+bin_edge[1:])/2
            plt.figure()
            plt.plot(bin_edge,hist)
            plt.title('Poloidal velocity histogram')
            plt.xlabel('Poloidal velocity [m/s]')
            plt.ylabel('Number of points')
    if plot_for_publication:
        import matplotlib.style as pltstyle
        pltstyle.use('default')
    if return_results:
        return frame_properties
    
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
    

def calculate_nstx_gpi_angular_velocity(exp_id=None,                                #Shot number
                                        time_range=None,                            #The time range for the calculation
                                        data_object=None,                           #Input data object if available from outside (e.g. generated sythetic signal)
                                        x_range=[0,63],                             #X range for the calculation
                                        y_range=[0,79],                             #Y range for the calculation
                                            
                                        #Normalizer inputs
                                        normalize_f_kernel='Elliptic',              #The kernel for filtering the gas cloud
                                        normalize_f_high=1e3,                       #High pass frequency for the normalizer data
                                                                                  
                                                                                    #Inputs for velocity pre processing
                                        normalize_for_velocity=False,               #Normalize the signal for velocity processing
                                        subtraction_order_for_velocity=1,           #Order of the 2D polynomial for background subtraction
                                        nlevel=51,
                                        
                                        #Inputs for velocity processing
                                        correlation_threshold=0.6,                  #Threshold for the maximum of the cross-correlation between the two frames (calculated with the actual maximum, not the fit value)
                                        frame_similarity_threshold=0.0,             #Similarity threshold between the two subsequent frames (CCF at zero lag) DEPRECATED when an ELM is present
                                        valid_frame_thres=5,                        #The number of consequtive frames to consider the calculation valid.
                                        fitting_range=5,                            #Fitting range of the peak of CCF 
                                        
                                        #Options for angular velocity processing
                                        zero_padding_scale=2,
                                        upsample_factor=20,
                                        sigma_low=3,                                #Gaussian blurring bandpass highpass sigma
                                        sigma_high=None,                            #Gaussian blurring bandpass lowpass sigma, should be higher than sigma_low
                                        calculate_retransformation=False,           #Retransform the image from the rotation and calculate the velocity with that (produces noisy output)
                                        #Plot options:
                                        plot=True,                                  #Plot the results
                                        pdf=False,                                  #Print the results into a PDF
                                        plot_gas=True,                              #Plot the gas cloud parameters on top of the other results from the structure size calculation
                                        plot_error=False,                           #Plot the errorbars of the velocity calculation based on the line fitting and its RMS error
                                        error_window=4.,                            #Plot the average signal with the error bars calculated from the normalized variance.
                                        plot_scatter=True,
                                        plot_time_range=None,                       #Plot the results in a different time range than the data is read from
                                        plot_for_publication=False,                 #Modify the plot sizes to single column sizes and golden ratio axis ratios
                                          
                                        #File input/output options
                                        filename=None,                              #Filename for restoring data
                                        save_results=True,                          #Save the results into a .pickle file to filename+.pickle
                                        nocalc=True,                                #Restore the results from the .pickle file from filename+.pickle
                                          
                                        #Output options:
                                        return_results=False,                       #Return the results if set.
                                        cache_data=True,                            #Cache the data or try to open is from cache
                                          
                                        #Test options
                                        test=False,                                 #Test the results
                                        ):                                          #Sad face right there :D
    
    from skimage.registration import phase_cross_correlation
    from skimage.transform import warp_polar, rotate, rescale
    #from skimage.util import img_as_float                                      #It's questionable if this has to be used or not

    from skimage.filters import difference_of_gaussians
    
    """
    Calculate frame by frame average frame velocity of the NSTX GPI signal. The
    code takes subsequent frames, calculates the 2D correlation function between
    the two and finds the maximum. Based on the pixel shift and the sampling
    time of the signal, the radial and poloidal velocity is calculated.
    The code assumes that the structures present in the subsequent frames are
    propagating with the same velocity. If there are multiple structures
    propagating in e.g. different direction or with different velocities, their
    effects are averaged over.
    """
    
    #Constants for the calculation
    #Using the spatial calibration to find the actual velocities.
    coeff_r_new=3./800.
    coeff_z_new=3./800.
    
    #Input error handling
    if exp_id is None and data_object is None:
        raise ValueError('Either exp_id or data_object needs to be set for the calculation.')
    if data_object is None:
        if time_range is None and filename is None:
            raise ValueError('It takes too much time to calculate the entire shot, please set a time_range.')
        else:    
            if type(time_range) is not list and filename is None:
                raise TypeError('time_range is not a list.')
            if filename is None and len(time_range) != 2:
                raise ValueError('time_range should be a list of two elements.')
        
    if (fitting_range*2+1 > np.abs(x_range[1]-x_range[0]) or 
        fitting_range*2+1 > np.abs(y_range[1]-y_range[0])):
        raise ValueError('The fitting range for the parabola is too large for the given coordinate range.')
        
   
    if data_object is not None and type(data_object) == str:
        if exp_id is None:
            exp_id='*'
        d=flap.get_data_object(data_object,exp_id=exp_id)
        time_range=[d.coordinate('Time')[0][0,0,0],
                    d.coordinate('Time')[0][-1,0,0]]
        exp_id=d.exp_id
        flap.add_data_object(d, 'GPI_SLICED_FULL')
    
    comment='pfit_o'+str(subtraction_order_for_velocity)+\
            '_fst_'+str(frame_similarity_threshold)

               
    if filename is None:
        wd=flap.config.get_all_section('Module NSTX_GPI')['Working directory']
        filename=flap_nstx.analysis.filename(exp_id=exp_id,
                                             working_directory=wd+'/processed_data',
                                             time_range=time_range,
                                             purpose='ccf ang velocity',
                                             comment=comment)
        filename_was_none=True
    else:
        filename_was_none=False
        
    pickle_filename=filename+'.pickle'
    if os.path.exists(pickle_filename) and nocalc:
        try:
            pickle.load(open(pickle_filename, 'rb'))
        except:
            print('The pickle file cannot be loaded. Recalculating the results.')
            nocalc=False
    elif nocalc:
        print(pickle_filename)
        print('The pickle file cannot be loaded. Recalculating the results.')
        nocalc=False

    if not test:
        import matplotlib
        matplotlib.use('agg')
    import matplotlib.pyplot as plt
           
    if not nocalc:
        slicing={'Time':flap.Intervals(time_range[0],time_range[1]),
                 'Image x':flap.Intervals(x_range[0],x_range[1]),
                 'Image y':flap.Intervals(y_range[0],y_range[1])}
        #Read data
        if data_object is None:
            print("\n------- Reading NSTX GPI data --------")
            if cache_data:
                try:
                    d=flap.get_data_object('GPI',exp_id=exp_id)
                except:
                    print('Data is not cached, it needs to be read.')
                    d=flap.get_data('NSTX_GPI',exp_id=exp_id,
                                    name='',
                                    object_name='GPI')
            else:
                d=flap.get_data('NSTX_GPI',exp_id=exp_id,
                                name='',
                                object_name='GPI')
          
            d=flap.slice_data('GPI',exp_id=exp_id, 
                              slicing=slicing,
                              output_name='GPI_SLICED_FULL')
            
        object_name_ccf_velocity='GPI_SLICED_FULL'
        
        #Normalize data for size calculation
        print("**** Calculating the gas cloud ****")

        normalizer_object_name='GPI_LPF_INTERVAL'

        slicing_for_filtering=copy.deepcopy(slicing)
        slicing_for_filtering['Time']=flap.Intervals(time_range[0]-1/normalize_f_high*10,
                                                     time_range[1]+1/normalize_f_high*10)
        if data_object is None:
            flap.slice_data('GPI',
                            exp_id=exp_id,
                            slicing=slicing_for_filtering,
                            output_name='GPI_SLICED_FOR_FILTERING')            
        #Roundtrip normalization
        
        norm_obj=flap.filter_data('GPI_SLICED_FOR_FILTERING',
                                     exp_id=exp_id,
                                     coordinate='Time',
                                     options={'Type':'Lowpass',
                                              'f_high':normalize_f_high,
                                              'Design':normalize_f_kernel},
                                     output_name=normalizer_object_name)
        
        norm_obj.data=np.flip(norm_obj.data,axis=0)
        norm_obj=flap.filter_data(normalizer_object_name,
                                  exp_id=exp_id,
                                  coordinate='Time',
                                  options={'Type':'Lowpass',
                                           'f_high':normalize_f_high,
                                           'Design':normalize_f_kernel},
                                 output_name=normalizer_object_name)
        
        norm_obj.data=np.flip(norm_obj.data,axis=0)                
        coefficient=flap.slice_data(normalizer_object_name,
                                    exp_id=exp_id,
                                    slicing=slicing,
                                    output_name='GPI_GAS_CLOUD').data

        data_obj=flap.get_data_object(object_name_ccf_velocity)
        data_obj.data = data_obj.data/coefficient
        flap.add_data_object(data_obj, 'GPI_SLICED_DENORM_CCF_VEL')
        object_name_ccf_velocity='GPI_SLICED_DENORM_CCF_VEL'
            
        #Subtract trend from data
        if subtraction_order_for_velocity is not None:
            print("*** Subtracting the trend of the data ***")
            d=flap_nstx.analysis.detrend_multidim(object_name_ccf_velocity,
                                                  exp_id=exp_id,
                                                  order=subtraction_order_for_velocity, 
                                                  coordinates=['Image x', 'Image y'], 
                                                  output_name='GPI_DETREND_VEL')
            object_name_ccf_velocity='GPI_DETREND_VEL'
        """
            VARIABLE DEFINITION
        """
        #Calculate correlation between subsequent frames in the data
        #Setting the variables for the calculation
        time_dim=d.get_coordinate_object('Time').dimension_list[0]
        n_frames=d.data.shape[time_dim]
        time=d.coordinate('Time')[0][:,0,0]
        sample_time=time[1]-time[0]
        sample_0=flap.get_data_object_ref(object_name_ccf_velocity).coordinate('Sample')[0][0,0,0]
        dalpha=flap.get_data_object_ref('GPI_SLICED_FULL').slice_data(summing={'Image x':'Mean', 'Image y':'Mean'}).data,
        
        frame_properties={'Shot':exp_id,
                          'Time':time[1:-1],
                          'GPI Dalpha':dalpha,
                          'Correlation max':np.zeros(len(time)-2),
                          'Frame similarity':np.zeros(len(time)-2),
                          
                          'Velocity ccf FLAP':np.zeros([len(time)-2,2]),
                          'Angular velocity ccf FLAP':np.zeros([len(time)-2]),
                          'Angular velocity ccf FLAP log':np.zeros([len(time)-2]),
                          'Expansion velocity ccf FLAP':np.zeros([len(time)-2]),
                          
                          'Velocity ccf':np.zeros([len(time)-2,2]),
                          'Velocity ccf skim':np.zeros([len(time)-2,2]),
                          'Angular velocity ccf':np.zeros([len(time)-2]),
                          'Angular velocity ccf log':np.zeros([len(time)-2]),
                          'Expansion velocity ccf':np.zeros([len(time)-2]),
                         }
        ccf_data=np.zeros([n_frames-1,
                           (x_range[1]-x_range[0])*2+1,
                           (y_range[1]-y_range[0])*2+1])
        coord=[]
        coord.append(copy.deepcopy(flap.Coordinate(name='Sample',
                                                   unit='n.a.',
                                                   mode=flap.CoordinateMode(equidistant=True),
                                                   start=sample_0,
                                                   step=1,
                                                   dimension_list=[0]
                                                   )))
        
        coord.append(copy.deepcopy(flap.Coordinate(name='Time',
                                                   unit='s',
                                                   mode=flap.CoordinateMode(equidistant=True),
                                                   start=flap.get_data_object_ref(object_name_ccf_velocity).coordinate('Time')[0][0,0,0],
                                                   step=sample_time,
                                                   dimension_list=[0]
                                                   )))
        coord.append(copy.deepcopy(flap.Coordinate(name='Image x',
                                                   unit='Pixel',
                                                   mode=flap.CoordinateMode(equidistant=True),
                                                   start=-x_range[1],
                                                   step=1,
                                                   shape=[],
                                                   dimension_list=[1]
                                                   )))
        coord.append(copy.deepcopy(flap.Coordinate(name='Image y',
                                                   unit='Pixel',
                                                   mode=flap.CoordinateMode(equidistant=True),
                                                   start=-y_range[1],
                                                   step=1,
                                                   shape=[],
                                                   dimension_list=[2]
                                                   )))
        
        ccf_object = flap.DataObject(data_array=ccf_data,
                                     data_unit=flap.Unit(name='Signal',unit='Digit'),
                                     coordinates=coord,
                                     exp_id=exp_id,
                                     data_title='',
                                     data_source="NSTX_GPI")
        
        flap.add_data_object(ccf_object, 'GPI_CCF_F_BY_F')        
        
        ccf_data_polar=np.zeros([360,64])
        
        coord_polar=[]
        
        coord_polar.append(copy.deepcopy(flap.Coordinate(name='Angle',
                                                         unit='Pixel',
                                                         mode=flap.CoordinateMode(equidistant=True),
                                                         start=0,
                                                         step=1,
                                                         shape=[],
                                                         dimension_list=[0]
                                                         )))
        
        coord_polar.append(copy.deepcopy(flap.Coordinate(name='Radius',
                                                         unit='Pixel',
                                                         mode=flap.CoordinateMode(equidistant=True),
                                                         start=0,
                                                         step=1,
                                                         shape=[],
                                                         dimension_list=[1]
                                                         )))
        
        frame1_polar_fft_object = flap.DataObject(data_array=ccf_data_polar,
                                                  data_unit=flap.Unit(name='Signal',unit='Digit'),
                                                  coordinates=coord_polar,
                                                  exp_id=exp_id,
                                                  data_title='',
                                                  data_source="NSTX_GPI")
            
        frame2_polar_fft_object = copy.deepcopy(frame1_polar_fft_object)
        
        flap.add_data_object(frame1_polar_fft_object, 'GPI_FRAME_1_FFT_POLAR')
        flap.add_data_object(frame2_polar_fft_object, 'GPI_FRAME_2_FFT_POLAR')
        
        #Velocity is calculated between the two subsequent frames, the velocity time is the second frame's time
        delta_index_flap=np.zeros(2)       
       
        
        
        #Inicializing for frame handling
        frame2=None
        invalid_correlation_frame_counter=0.
        
        if test:
            plt.figure()
            
        for i_frames in range(0,n_frames-2):
            """
            STRUCTURE VELOCITY CALCULATION BASED ON FLAP CCF CALCULATION
            """

            slicing_frame1={'Sample':sample_0+i_frames}
            slicing_frame2={'Sample':sample_0+i_frames+1}
            if frame2 is None:
                frame1=flap.slice_data(object_name_ccf_velocity,
                                       exp_id=exp_id,
                                       slicing=slicing_frame1, 
                                       output_name='GPI_FRAME_1')
                frame1.add_coordinate()
            else:
                frame1=copy.deepcopy(frame2)
                flap.add_data_object(frame1,'GPI_FRAME_1')
                
            frame2=flap.slice_data(object_name_ccf_velocity, 
                                   exp_id=exp_id,
                                   slicing=slicing_frame2,
                                   output_name='GPI_FRAME_2')

            frame1.data=np.asarray(frame1.data, dtype='float64')
            frame2.data=np.asarray(frame2.data, dtype='float64')
                           
            flap.ccf('GPI_FRAME_2', 'GPI_FRAME_1', 
                     coordinate=['Image x', 'Image y'], 
                     options={'Resolution':1, 
                              'Range':[[-(x_range[1]-x_range[0]),(x_range[1]-x_range[0])],
                                       [-(y_range[1]-y_range[0]),(y_range[1]-y_range[0])]], 
                              'Trend removal':None, 
                              'Normalize':True, 
                              'Interval_n': 1}, 
                     output_name='GPI_FRAME_12_CCF')

            ccf_object.data[i_frames,:,:]=flap.get_data_object_ref('GPI_FRAME_12_CCF').data                        
            
            if test:
                print('Maximum correlation:'+str(ccf_object.data[i_frames,:,:].max()))
                
            
            max_index=np.asarray(np.unravel_index(ccf_object.data[i_frames,:,:].argmax(), ccf_object.data[i_frames,:,:].shape))
            
            #Fit a 2D polinomial on top of the peak

            area_max_index=tuple([slice(max_index[0]-fitting_range,
                                        max_index[0]+fitting_range+1),
                                  slice(max_index[1]-fitting_range,
                                        max_index[1]+fitting_range+1)])
            #Finding the peak analytically
            try:
                coeff=flap_nstx.analysis.polyfit_2D(values=ccf_object.data[i_frames,:,:][area_max_index],order=2)
                index=[0,0]
                index[0]=(2*coeff[2]*coeff[3]-coeff[1]*coeff[4])/(coeff[4]**2-4*coeff[2]*coeff[5])
                index[1]=(-2*coeff[5]*index[0]-coeff[3])/coeff[4]
            except:
                index=[fitting_range,fitting_range]
            if (index[0] < 0 or 
                index[0] > 2*fitting_range or 
                index[1] < 0 or 
                index[1] > 2*fitting_range):
                
                index=[fitting_range,fitting_range]           
            
            delta_index_flap=[index[0]+max_index[0]-fitting_range-ccf_object.data[i_frames,:,:].shape[0]//2,
                              index[1]+max_index[1]-fitting_range-ccf_object.data[i_frames,:,:].shape[1]//2]
            if test:
                plt.contourf(ccf_object.data[i_frames,:,:].T, levels=np.arange(0,51)/25-1)
                plt.scatter(index[0]+max_index[0]-fitting_range,
                            index[1]+max_index[1]-fitting_range)
            
            frame_properties['Correlation max'][i_frames]=ccf_object.data[i_frames,:,:].max()
                
            frame_properties['Frame similarity'][i_frames]=ccf_object.data[i_frames,:,:][tuple(np.asarray(ccf_object.data[i_frames,:,:].shape)[:]//2)]
                
            if ccf_object.data[i_frames,:,:].max() < correlation_threshold:
                invalid_correlation_frame_counter+=1
            else:
                invalid_correlation_frame_counter=0.
            #Calculating the radial and poloidal velocity from the correlation map.
            frame_properties['Velocity ccf FLAP'][i_frames,0]=coeff_r_new*delta_index_flap[0]/sample_time
            frame_properties['Velocity ccf FLAP'][i_frames,1]=coeff_z_new*delta_index_flap[1]/sample_time
            
            """            
            STRUCTURE ANGULAR VELOCITY CALCULATION BASED ON FLAP CCF
            """
            
            radius=min([frame1.data.shape[0],frame1.data.shape[1]])
            if i_frames == 0 :
                frame1_filtered = difference_of_gaussians(frame1.data, sigma_low, sigma_high)
                
            frame2_filtered = difference_of_gaussians(frame2.data, sigma_low, sigma_high)
            #The calculations only need to be done for the first frame in the first stop
            #In the following steps the first calculation is the same as the second in the
            #previous step
            
            shape=np.asarray(frame1_filtered.shape)
            if i_frames == 0 :
                frame1_zero_padded=np.zeros(np.asarray(shape)*(2*zero_padding_scale+1))
                frame1_zero_padded[shape[0]*zero_padding_scale:shape[0]*(zero_padding_scale+1),
                                   shape[1]*zero_padding_scale:shape[1]*(zero_padding_scale+1)]=frame1_filtered
            
            
            frame2_zero_padded=np.zeros(np.asarray(shape)*(2*zero_padding_scale+1))
            frame2_zero_padded[shape[0]*zero_padding_scale:shape[0]*(zero_padding_scale+1),
                               shape[1]*zero_padding_scale:shape[1]*(zero_padding_scale+1)]=frame2_filtered
                               
            if i_frames == 0 :
                frame1_fft=np.absolute( np.fft.fftshift( np.fft.fftn(frame1_zero_padded, axes=[0,1])))
            frame2_fft=np.absolute( np.fft.fftshift( np.fft.fftn(frame2_zero_padded, axes=[0,1])))
            
            if i_frames == 0 :
                frame1_fft_polar_log = warp_polar(frame1_fft, scaling='log',)
            else:
                frame1_fft_polar_log = copy.deepcopy(frame2_fft_polar_log)      #No error here, in the 2nd step this object exists
            frame2_fft_polar_log = warp_polar(frame2_fft, scaling='log',)
            
            if i_frames == 0 :
                frame1_fft_polar = warp_polar(frame1_fft, radius=radius)
            else:
                frame1_fft_polar = copy.deepcopy(frame2_fft_polar)              #No error here, in the 2nd step this object exists
            frame2_fft_polar = warp_polar(frame2_fft, radius=radius)
            
            """
            HIJACKING THE DATA FOR FLAP CALCULATION
            """
            for i_log_or_not in range(2):
                if i_log_or_not == 0:
#                    frame1_polar_fft_object.data=frame1_fft_polar-np.mean(frame1_fft_polar)
#                    frame2_polar_fft_object.data=frame2_fft_polar-np.mean(frame2_fft_polar)
                    frame1_polar_fft_object.data=frame1_fft_polar
                    frame2_polar_fft_object.data=frame2_fft_polar
                else:
#                    frame1_polar_fft_object.data=frame1_fft_polar_log-np.mean(frame1_fft_polar_log)
#                    frame2_polar_fft_object.data=frame2_fft_polar_log-np.mean(frame2_fft_polar_log)
                    frame1_polar_fft_object.data=frame1_fft_polar_log
                    frame2_polar_fft_object.data=frame2_fft_polar_log
                
                ccf_object_polar=flap.ccf('GPI_FRAME_1_FFT_POLAR', 
                                          'GPI_FRAME_2_FFT_POLAR', 
                                          coordinate=['Angle', 'Radius'], 
                                          options={'Resolution':1, 
                                                   'Range':[[-359,359],
                                                             [-radius+1,radius-1]], 
                                                   'Trend removal':None, 
                                                   'Normalize':True, 
                                                   'Interval_n': 1}, 
                                                   output_name='GPI_FRAME_12_CCF_POLAR')
                if test:
                    plt.cla()

                    flap.plot('GPI_FRAME_12_CCF_POLAR', plot_type='contour', axes=['Angle lag', 'Radius lag'])
                    plt.pause(0.1)
                    plt.show()
                    #flap.plot('GPI_FRAME_2_FFT_POLAR', plot_type='contour', axes=['Angle', 'Radius'])
                
                max_index=np.asarray(np.unravel_index(ccf_object_polar.data.argmax(), ccf_object_polar.data.shape))
                
                #Fit a 2D polinomial on top of the peak
    
                area_max_index=tuple([slice(max_index[0]-fitting_range,
                                            max_index[0]+fitting_range+1),
                                      slice(max_index[1]-fitting_range,
                                            max_index[1]+fitting_range+1)])

                #Finding the peak analytically
                try:
                    coeff=flap_nstx.analysis.polyfit_2D(values=ccf_object_polar.data[area_max_index],order=2)
                    index=[0,0]
                    index[0]=(2*coeff[2]*coeff[3]-coeff[1]*coeff[4])/(coeff[4]**2-4*coeff[2]*coeff[5])
                    index[1]=(-2*coeff[5]*index[0]-coeff[3])/coeff[4]
                except:
                    index=[fitting_range,fitting_range]
                
                delta_index_flap=[index[0]+max_index[0]-fitting_range-ccf_object_polar.data.shape[0]//2,
                                  index[1]+max_index[1]-fitting_range-ccf_object_polar.data.shape[1]//2]
                if test:
                    plt.scatter(delta_index_flap[0],delta_index_flap[1], color='red')
                    #plt.pause(0.5)
                if i_log_or_not == 0:
                    frame_properties['Angular velocity ccf FLAP'][i_frames]=delta_index_flap[0]/180.*np.pi/sample_time #dphi/dt
                else:
                    klog = radius / np.log(radius)
                    shift_scale_log = (np.exp(delta_index_flap[1] / klog))
                    frame_properties['Expansion velocity ccf FLAP'][i_frames]=shift_scale_log-1.
                    frame_properties['Angular velocity ccf FLAP log'][i_frames]=delta_index_flap[0]/180.*np.pi/sample_time #dphi/dt
                    
            """
            END OF HIJACKING
            """
            
            shift_rot, error, phasediff = phase_cross_correlation(frame1_fft_polar, 
                                                                  frame2_fft_polar,
                                                                  upsample_factor=upsample_factor)
            shiftr, shiftc = shift_rot[:2]
            
            shift_rot_log, error_log, phasediff = phase_cross_correlation(frame1_fft_polar_log, 
                                                                          frame2_fft_polar_log,
                                                                          upsample_factor=upsample_factor)
            shiftr_log, shiftc_log = shift_rot_log[:2]
            
            # Calculate scale factor from translation
            klog = radius / np.log(radius)
            shift_scale_log = (np.exp(shiftc_log / klog))
            
            frame_properties['Angular velocity ccf'][i_frames]=shiftr/180.*np.pi/sample_time #dphi/dt
            frame_properties['Angular velocity ccf log'][i_frames]=shiftr_log/180.*np.pi/sample_time #dphi/dt
            frame_properties['Expansion velocity ccf'][i_frames]=shift_scale_log-1.
            
            if calculate_retransformation:
                frame1_retransformed_scaled=rescale(rotate(frame2.data,-shiftr),1/shift_scale_log)
                frame1_retransformed=np.zeros(frame1.data.shape)
                
                x_addon=0
                y_addon=0
                
                ret_shape=frame1_retransformed.shape
                scaled_shape=frame1_retransformed_scaled.shape
                
                if (ret_shape[0]-scaled_shape[0])//2-ret_shape[0]+(ret_shape[0]-scaled_shape[0])//2 != -scaled_shape[0]:
                    x_addon=1
                    
                if (ret_shape[1]-scaled_shape[1])//2-ret_shape[1]+(ret_shape[1]-scaled_shape[1])//2 != -scaled_shape[1]:
                    y_addon=1
                    
                if shift_scale_log > 1:        
                    frame1_retransformed[(ret_shape[0]-scaled_shape[0])//2+x_addon:ret_shape[0]-(ret_shape[0]-scaled_shape[0])//2,
                                         (ret_shape[1]-scaled_shape[1])//2+y_addon:ret_shape[1]-(ret_shape[1]-scaled_shape[1])//2]=frame1_retransformed_scaled
                else:
                    frame1_retransformed = frame1_retransformed_scaled[(scaled_shape[0]-ret_shape[0])//2+x_addon:scaled_shape[0]-(scaled_shape[0]-ret_shape[0])//2,
                                                                       (scaled_shape[1]-ret_shape[1])//2+y_addon:scaled_shape[1]-(scaled_shape[1]-ret_shape[1])//2]
    
                shift_tra, error, phasediff = phase_cross_correlation(frame1.data, 
                                                                      frame1_retransformed,
                                                                      upsample_factor=upsample_factor)
                
                frame_properties['Velocity ccf'][i_frames,0] = -1 * coeff_r_new * shift_tra[0] / sample_time
                frame_properties['Velocity ccf'][i_frames,1] = -1 * coeff_z_new * shift_tra[1] / sample_time
            
            shift_tra, error, phasediff = phase_cross_correlation(frame1.data, 
                                                                  frame2.data,
                                                                  upsample_factor=upsample_factor)
            
            frame_properties['Velocity ccf skim'][i_frames,0] = -1 * coeff_r_new * shift_tra[0] / sample_time
            frame_properties['Velocity ccf skim'][i_frames,1] = -1 * coeff_z_new * shift_tra[1] / sample_time
            
        pickle.dump(frame_properties,open(pickle_filename, 'wb'))
        if test:
            plt.close()
    else:
        print('--- Loading data from the pickle file ---')
        frame_properties=pickle.load(open(pickle_filename, 'rb'))
        
        
    if not filename_was_none and not time_range is None:
        sample_time=frame_properties['Time'][1]-frame_properties['Time'][0]
        if time_range[0] < frame_properties['Time'][0]-sample_time or time_range[1] > frame_properties['Time'][-1]+sample_time:
            raise ValueError('Please run the calculation again with the timerange. The pickle file doesn\'t have the desired range')
    if time_range is None:
        time_range=[frame_properties['Time'][0],frame_properties['Time'][-1]]

    #Plotting the results
    if plot or pdf:
        #This is a bit unusual here, but necessary due to the structure size calculation based on the contours which are not plot
        if plot:
            import matplotlib
            matplotlib.use('QT5Agg')
            import matplotlib.pyplot as plt
        else:
            import matplotlib
            matplotlib.use('agg')
            import matplotlib.pyplot as plt     
        
        nan_ind=np.where(frame_properties['Correlation max'] < correlation_threshold)
        frame_properties['Velocity ccf FLAP'][nan_ind,:] = [np.nan,np.nan]
        frame_properties['Angular velocity ccf'][nan_ind] = np.nan
        frame_properties['Angular velocity ccf FLAP'][nan_ind] = np.nan
        frame_properties['Angular velocity ccf FLAP log'][nan_ind] = np.nan
        frame_properties['Expansion velocity ccf'][nan_ind] = np.nan
        frame_properties['Expansion velocity ccf FLAP'][nan_ind] = np.nan
        
        if plot_time_range is not None:
            if plot_time_range[0] < time_range[0] or plot_time_range[1] > time_range[1]:
                raise ValueError('The plot time range is not in the interval of the original time range.')
            time_range=plot_time_range
            
        plot_index=np.logical_and(np.logical_not(np.isnan(frame_properties['Velocity ccf'][:,0])),
                                  np.logical_and(frame_properties['Time'] >= time_range[0],
                                                 frame_properties['Time'] <= time_range[1]))

        #Plotting the radial velocity
        if pdf:
            wd=flap.config.get_all_section('Module NSTX_GPI')['Working directory']
            filename=flap_nstx.analysis.filename(exp_id=exp_id,
                                                 working_directory=wd+'/plots',
                                                 time_range=time_range,
                                                 purpose='ccf ang velocity',
                                                 comment=comment+'_ct_'+str(correlation_threshold))
            pdf_filename=filename+'.pdf'
            pdf_pages=PdfPages(pdf_filename)
            
        if plot_for_publication:
            figsize=(8.5/2.54, 
                     8.5/2.54/1.618*1.1)
            plt.rc('font', family='serif', serif='Helvetica')
            labelsize=9
            linewidth=0.5
            major_ticksize=2
            plt.rc('text', usetex=False)
            plt.rcParams['pdf.fonttype'] = 42
            plt.rcParams['ps.fonttype'] = 42
            plt.rcParams['lines.linewidth'] = linewidth
            plt.rcParams['axes.linewidth'] = linewidth
            plt.rcParams['axes.labelsize'] = labelsize
            plt.rcParams['axes.titlesize'] = labelsize
            
            plt.rcParams['xtick.labelsize'] = labelsize
            plt.rcParams['xtick.major.size'] = major_ticksize
            plt.rcParams['xtick.major.width'] = linewidth
            plt.rcParams['xtick.minor.width'] = linewidth/2
            plt.rcParams['xtick.minor.size'] = major_ticksize/2
            
            plt.rcParams['ytick.labelsize'] = labelsize
            plt.rcParams['ytick.major.width'] = linewidth
            plt.rcParams['ytick.major.size'] = major_ticksize
            plt.rcParams['ytick.minor.width'] = linewidth/2
            plt.rcParams['ytick.minor.size'] = major_ticksize/2
            plt.rcParams['legend.fontsize'] = labelsize
        else:
            figsize=None
            
        """
        RADIAL AND POLOIDAL PLOTTING FROM FLAP
        """
            
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(frame_properties['Time'][plot_index], 
                 frame_properties['Velocity ccf FLAP'][plot_index,0],
                 label='Velocity ccf FLAP')
        if calculate_retransformation:
            ax.plot(frame_properties['Time'][plot_index], 
                    frame_properties['Velocity ccf'][plot_index,0],
                    label='Velocity ccf recon')
        ax.plot(frame_properties['Time'][plot_index], 
                 frame_properties['Velocity ccf skim'][plot_index,0],
                 label='Velocity ccf skim')
        plt.legend()
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('v_rad[m/s]')
        ax.set_xlim(time_range)
        ax.set_title('Radial velocity of '+str(exp_id)+' with FLAP')
        if plot_for_publication:
            x1,x2=ax.get_xlim()
            y1,y2=ax.get_ylim()
            ax.set_aspect((x2-x1)/(y2-y1)/1.618)
            ax.set_title('Rad vel ccf')
        fig.tight_layout()
        if pdf:
            pdf_pages.savefig()        
            
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(frame_properties['Time'][plot_index], 
                 frame_properties['Velocity ccf FLAP'][plot_index,1],
                 label='Velocity ccf FLAP')
        if calculate_retransformation:
            ax.plot(frame_properties['Time'][plot_index], 
                     frame_properties['Velocity ccf'][plot_index,1],
                     label='Velocity ccf recon')
        ax.plot(frame_properties['Time'][plot_index], 
                 frame_properties['Velocity ccf skim'][plot_index,1],
                 label='Velocity ccf skim')
        plt.legend()
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('v_pol[m/s]')
        ax.set_xlim(time_range)
        ax.set_title('Poloidal velocity of '+str(exp_id))
        if plot_for_publication:
            x1,x2=ax.get_xlim()
            y1,y2=ax.get_ylim()
            ax.set_aspect((x2-x1)/(y2-y1)/1.618)
            ax.set_title('Pol vel ccf')
        fig.tight_layout()
        if pdf:
            pdf_pages.savefig()       
        
        
        """
        ANGULAR AND EXPANSION VELOCITY PLOTTING FROM SKIMAGE
        """
            
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(frame_properties['Time'][plot_index], 
                 frame_properties['Angular velocity ccf'][plot_index],
                 label='Angular velocity ccf')
        ax.plot(frame_properties['Time'][plot_index], 
                 frame_properties['Angular velocity ccf log'][plot_index],
                 label='Angular velocity ccf log',
                 color='green')
        ax.plot(frame_properties['Time'][plot_index], 
                 frame_properties['Angular velocity ccf FLAP'][plot_index],
                 label='Angular velocity ccf FLAP',
                 color='red')
        ax.plot(frame_properties['Time'][plot_index], 
                 frame_properties['Angular velocity ccf FLAP log'][plot_index],
                 label='Angular velocity ccf FLAP log',
                 color='orange')
        plt.legend()
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('omega[1/s]')
        ax.set_xlim(time_range)
        ax.set_title('Angular velocity of '+str(exp_id))
        if plot_for_publication:
            x1,x2=ax.get_xlim()
            y1,y2=ax.get_ylim()
            ax.set_aspect((x2-x1)/(y2-y1)/1.618)
            ax.set_title('Ang vel CCF skimage')
        fig.tight_layout()
        if pdf:
            pdf_pages.savefig()            
            
        """
        ANGULAR AND EXPANSION VELOCITY PLOTTING FROM SKIMAGE
        """
            
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(frame_properties['Time'][plot_index], 
                 frame_properties['Expansion velocity ccf'][plot_index], 
                 label='Expansion velocity ccf ',
                 )
        ax.plot(frame_properties['Time'][plot_index], 
                frame_properties['Expansion velocity ccf FLAP'][plot_index], 
                label='Expansion velocity ccf FLAP',
                color='red')
        plt.legend()
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('v_exp[1/s]')
        ax.set_xlim(time_range)
        ax.set_title('Expansion velocity of '+str(exp_id)+' with FLAP and skimage')
        if plot_for_publication:
            x1,x2=ax.get_xlim()
            y1,y2=ax.get_ylim()
            ax.set_aspect((x2-x1)/(y2-y1)/1.618)
            ax.set_title('Exp vel CCF FLAP')
        fig.tight_layout()
        if pdf:
            pdf_pages.savefig()     
            
    if pdf:
        pdf_pages.close()