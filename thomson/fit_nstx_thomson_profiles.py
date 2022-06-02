#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 17:45:24 2021

@author: mlampert
"""

import os
import copy

#FLAP imports and settings
import flap
import flap_nstx
import flap_mdsplus

flap_nstx.register()
flap_mdsplus.register('NSTX_MDSPlus')

thisdir = os.path.dirname(os.path.realpath(__file__))
fn = os.path.join(thisdir,"../flap_nstx.cfg")
flap.config.read(file_name=fn)

#Scientific imports
import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline
from scipy.integrate import quad

import matplotlib.pyplot as plt
#Other necessary imports

def get_fit_nstx_thomson_profiles(exp_id=None,                                      #Shot number
                                  pressure=False,                                   #Return the pressure profile paramenters
                                  temperature=False,                                #Return the temperature profile parameters
                                  density=False,                                    #Return the density profile parameters
                                  spline_data=False,                                #Calculate the results from the spline data (no error is going to be taken into account)
                                  
                                  modified_tanh=False,
                                  
                                  device_coordinates=False,                          #Calculate the results as a function of device coordinates
                                  radial_range=None,                                #Radial range of the pedestal (only works when the device coorinates is set)
                                  
                                  flux_coordinates=False,                           #Calculate the results in flux coordinates
                                  flux_range=None,                                  #The normalaized flux coordinates range for returning the results
                                  outboard_only=True,
                                  
                                  force_overlap=False,                          #Shifts the inboard and outboard profiles of the TS to match 
                                  max_iter=1200,                                #Maximum iteration for the shifting
                                  max_err=1e-5,                                 #difference between iteration steps to be reached
                                  
                                  output_name=None,
                                  return_parameters=False,
                                  plot_time=None,
                                  pdf_object=None,
                                  
                                  test=False,
                                  ):
    """
    
    Returns a dataobject which has the largest corresponding gradient based on the tanh fit.
    
    Fitting is based on publication https://aip.scitation.org/doi/pdf/10.1063/1.4961554
    The linear background is not usitlized, instead of the mtanh, only tanh is used.
    """
    
    
    if ((device_coordinates and flux_range is not None) or
        (flux_coordinates and radial_range is not None)):
        raise ValueError('When flux or device coordinates are set, only flux or radial range can be set! Returning...')
        
    d=flap.get_data('NSTX_THOMSON', 
                    exp_id=exp_id,
                    name='',
                    object_name='THOMSON_DATA', 
                    options={'pressure':pressure,
                             'temperature':temperature,
                             'density':density,
                             'spline_data':False,
                             'add_flux_coordinates':True,
                             'force_mdsplus':False})
    
    if flux_coordinates:
        r_coord_name='Flux r'
    if device_coordinates:
        r_coord_name='Device R'
    time=d.coordinate('Time')[0][0,:]
    thomson_profile={'Time':time,
                     'Data':d.data,
                     'Device R':d.coordinate('Device R')[0],
                     'Flux r':d.coordinate('Flux r')[0],
                     'Fit parameters':np.zeros([time.shape[0],5]),
                     'Fit parameter errors':np.zeros([time.shape[0],5]),
                     'a':np.zeros(time.shape),
                     'Height':np.zeros(time.shape),
                     'Width':np.zeros(time.shape),
                     'Global gradient':np.zeros(time.shape),
                     'Position':np.zeros(time.shape),
                     'Position r':np.zeros(time.shape),
                     'SOL offset':np.zeros(time.shape),
                     'Max gradient':np.zeros(time.shape),
                     'Value at max':np.zeros(time.shape),
                     
                     'Error':{'Height':np.zeros(time.shape),
                              'SOL offset':np.zeros(time.shape),
                              'Position':np.zeros(time.shape),
                              'Position r':np.zeros(time.shape),
                              'Width':np.zeros(time.shape),
                              'Global gradient':np.zeros(time.shape),
                              'Max gradient':np.zeros(time.shape),
                              'Value at max':np.zeros(time.shape),
                              },
                     }
    if modified_tanh:
        thomson_profile['Slope']=np.zeros(time.shape)
        thomson_profile['Error']['Slope']=np.zeros(time.shape)
        
    if test:
        plt.figure()
    if flux_range is not None:
        x_range=flux_range
    if radial_range is not None:
        x_range=radial_range

#    def mtanh_fit_function(r, b_height, b_sol, b_pos, b_width, b_slope):           #This version of the code is not working due to the b_slope linear dependence
#        def mtanh(x,b_slope):
#            return ((1+b_slope*x)*np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
#        return (b_height-b_sol)/2*(mtanh((b_pos-r)/(2*b_width),b_slope)+1)+b_sol

    if not modified_tanh:
        def tanh_fit_function(r, b_height, b_sol, b_pos, b_width):
            def tanh(x):
                return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
            return (b_height-b_sol)/2*(tanh((b_pos-r)/(2*b_width))+1)+b_sol
    else:
        def tanh_fit_function(x, b_height, b_sol, b_pos, b_width, b_slope):
            x_mod=2*(x - b_pos)/b_width
            return (b_height+b_sol)/2 + (b_height-b_sol)/2*((1 - b_slope*x_mod)*np.exp(-x_mod) - np.exp(x_mod))/(np.exp(x_mod) + np.exp(-x_mod))
            
    for i_time in range(len(time)):
        if r_coord_name =='Flux r':
            x_data=d.coordinate('Flux r')[0][:,i_time]
            y_data=d.data[:,i_time]
            y_data_error=d.error[:,i_time]
            if outboard_only or force_overlap:
                rmaxis=flap.get_data('NSTX_MDSPlus',
                                     name='\EFIT02::\RMAXIS',
                                     exp_id=exp_id,
                                     object_name='RMAXIS')
                
                ind_time_efit=np.argmin(np.abs(rmaxis.coordinate('Time')[0]-time[i_time]))
                r_maxis_cur=rmaxis.data[ind_time_efit]
                ind_maxis=np.argmin(np.abs(d.coordinate('Device R')[0][:,i_time]-r_maxis_cur))
                
            if outboard_only:
                rmaxis=flap.get_data('NSTX_MDSPlus',
                                     name='\EFIT02::\RMAXIS',
                                     exp_id=exp_id,
                                     object_name='RMAXIS')
                
                ind_time_efit=np.argmin(np.abs(rmaxis.coordinate('Time')[0]-time[i_time]))
                r_maxis_cur=rmaxis.data[ind_time_efit]
                ind_maxis=np.argmin(np.abs(d.coordinate('Device R')[0][:,i_time]-r_maxis_cur))
                
                x_data=x_data[ind_maxis:]
                y_data=y_data[ind_maxis:]
                y_data_error=d.error[ind_maxis:,i_time]
                
            elif force_overlap:
                x_data_in=x_data[:ind_maxis]
                x_data_out=x_data[ind_maxis:]
                y_data_in=y_data[:ind_maxis]
                y_data_out=y_data[ind_maxis:]
                
                if not temperature:                                                 #Only temperature is a flux function on NSTX
                    d2=flap.get_data('NSTX_THOMSON', 
                                    exp_id=exp_id,
                                    name='',
                                    object_name='THOMSON_DATA', 
                                    options={'pressure':pressure,
                                             'temperature':temperature,
                                             'density':density,
                                             'spline_data':False,
                                             'add_flux_coordinates':True,
                                             'force_mdsplus':False})
                    
                    temp_data_in=d2.data[:,i_time][:ind_maxis]
                    temp_data_out=d2.data[:,i_time][ind_maxis:]
                else:
                    temp_data_in=y_data[:ind_maxis]
                    temp_data_out=y_data[ind_maxis:]
                    
                psi_shift_0=max(x_data_out[2:]-x_data_out[:-2])
                psi_shift_1=-psi_shift_0
                for ind_iter in range(max_iter):
                    
                    s_in_neg=UnivariateSpline(x_data_in-psi_shift_0, temp_data_in)
                    s_out_neg=UnivariateSpline(x_data_out+psi_shift_0, temp_data_out)
                    
                    integral_difference_neg = s_out_neg.integral(0, np.inf) - s_in_neg.integral(0, np.inf)
                    
                    s_in_pos=UnivariateSpline(x_data_in-psi_shift_1, temp_data_in)
                    s_out_pos=UnivariateSpline(x_data_out+psi_shift_1, temp_data_out)
                    
                    integral_difference_pos = s_out_pos.integral(0, np.inf) - s_in_pos.integral(0, np.inf)
                    
                    if integral_difference_pos > integral_difference_neg:
                        psi_shift=None
                        
                x_data=np.concatenate([x_data_in,x_data_out])
                y_data=np.concatenate([y_data_in,y_data_out])
                sort_ind=np.argsort(x_data)
                x_data=x_data[sort_ind]
                y_data=y_data[sort_ind]
                
                
                
        else:
            #By default it's outboard only, the full profile is not tanh
            ind_max=np.argmax(d.data[:,i_time])
            x_data=d.coordinate('Device R')[0][ind_max:,i_time]            
            y_data=d.data[ind_max:,i_time]
            y_data_error=d.error[ind_max:,i_time]
        
        if np.sum(np.isinf(x_data)) != 0:
            continue
        
        #Further adjustment based on the set x_range
        ind_coord=np.where(np.logical_and(x_data > x_range[0],
                                          x_data <= x_range[1]))
        x_data=x_data[ind_coord]
        y_data=y_data[ind_coord]
        y_data_error=y_data_error[ind_coord]
        
        try:
            if not modified_tanh:
                p0=[y_data[0],                                      #b_height
                    y_data[-1],                                     #b_sol
                    (x_data[0]+x_data[-1])/2,                       #b_pos
                    np.abs((x_data[-1]-x_data[0])/2),                      #b_width
                    #(y_data[0]-y_data[-1])/(x_data[0]-x_data[-1]), #b_slope this is supposed to be some kind of linear modification to the 
                                                                    #tanh function called mtanh. It messes up the fitting quite a bit and it's not useful at all.
                    ]
    
            else:
                p0=[y_data[0],                                      #b_height
                    y_data[-1],                                     #b_sol
                    (x_data[0]+x_data[-1])/2,                       #b_pos
                    np.abs((x_data[-1]-x_data[0])/2),                  #b_width
                    (y_data[0]-y_data[-1])/(x_data[0]-x_data[-1]),  #b_slope this is supposed to be some kind of linear modification to the 
                                                                    #tanh function called mtanh. It messes up the fitting quite a bit and it's not useful at all.
                    
                    ]
        except:
            print('Missing TS data for shot '+str(exp_id)+', time: '+ str(time[i_time]))
            
        try:
            popt, pcov = curve_fit(tanh_fit_function, 
                                   x_data, 
                                   y_data, 
                                   sigma=y_data_error,
                                   p0=p0)
            perr = np.sqrt(np.diag(pcov))
            successful_fitting=True
        except:
            if modified_tanh:
                popt=[np.nan,np.nan,np.nan,np.nan,np.nan]
                perr=[np.nan,np.nan,np.nan,np.nan,np.nan]
            else:
                popt=[np.nan,np.nan,np.nan,np.nan]
                perr=[np.nan,np.nan,np.nan,np.nan]
            successful_fitting=False
        
        if test or (plot_time is not None and np.abs(plot_time-time[i_time]) < 1e-3):
            plt.cla()
            if successful_fitting:
                color='tab:blue'
            else:
                color='red'
            plt.scatter(x_data,
                        y_data,
                        color=color)
            plt.errorbar(x_data,
                         y_data,
                         yerr=y_data_error,
                         marker='o', 
                         color=color,
                         ls='')
            plt.plot(x_data,tanh_fit_function(x_data,*popt), color=color)
            
            if flux_coordinates:
                xlabel='PSI_norm'
            else:
                xlabel='Device R [m]'
                
            if temperature:
                profile_string='temperature'
                ylabel='Temperature [keV]'
            
            elif density:
                profile_string='density'
                ylabel='Density [1/m3]'
            elif pressure:
                profile_string='pressure'
                ylabel='Pressure [kPa]'
                
            time_string=' @ '+str(time[i_time])
            plt.title('Fit '+profile_string+' profile of '+str(exp_id)+time_string)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.pause(1.0)

            if pdf_object is not None:
                pdf_object.savefig()
        else:
            pass
        
        if modified_tanh:
            thomson_profile['Fit parameters'][i_time,:]=popt
            thomson_profile['Fit parameter errors'][i_time,:]=perr
        else:
            thomson_profile['Fit parameters'][i_time,0:4]=popt
            thomson_profile['Fit parameter errors'][i_time,0:4]=perr
            
        thomson_profile['Height'][i_time]=popt[0]
        thomson_profile['SOL offset'][i_time]=popt[1]
        thomson_profile['Position'][i_time]=popt[2]
        
        try:
        #if True:
            thomson_profile['Position r'][i_time]=np.interp(popt[2],
                                                            d.coordinate('Flux r')[0][np.argmin(d.coordinate('Flux r')[0][:,i_time]):,i_time],
                                                            d.coordinate('Device R')[0][np.argmin(d.coordinate('Flux r')[0][:,i_time]):,i_time])
        except:
            print('Interpolation failed.')
            thomson_profile['Position r'][i_time]=np.nan
            
        thomson_profile['Width'][i_time]=popt[3]
        thomson_profile['Max gradient']=(thomson_profile['SOL offset']-thomson_profile['Height']/(4*thomson_profile['Width']))
        thomson_profile['Value at max']=(thomson_profile['SOL offset']+thomson_profile['Height'])/2.
        thomson_profile['Global gradient']=(thomson_profile['SOL offset']-thomson_profile['Height']/(4*thomson_profile['Width']))
        
        if modified_tanh:
            thomson_profile['Slope'][i_time]=popt[4]

        thomson_profile['Error']['Height'][i_time]=perr[0]
        thomson_profile['Error']['SOL offset'][i_time]=perr[1]
        thomson_profile['Error']['Position'][i_time]=perr[2]
        thomson_profile['Error']['Width'][i_time]=perr[3]
        thomson_profile['Error']['Value at max'][i_time]=(perr[0]+perr[1])/2
        thomson_profile['Error']['Global gradient'][i_time]=(perr[0]/popt[3]+
                                                             perr[1]/popt[3]+
                                                             np.abs((-popt[1]+popt[0])/popt[3]**2)*perr[3])
        thomson_profile['Error']['Max gradient'][i_time]=(np.abs(1/(4*popt[3])*perr[1])+
                                                          np.abs(1/(4*popt[3])*perr[0]))
        
        if modified_tanh:
            thomson_profile['Error']['Slope'][i_time]=perr[4]

    return thomson_profile
        
    
