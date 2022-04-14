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
import matplotlib.pyplot as plt
#Other necessary imports

def get_fit_nstx_thomson_profiles(exp_id=None,                                      #Shot number
                                  pressure=False,                                   #Return the pressure profile paramenters
                                  temperature=False,                                #Return the temperature profile parameters
                                  density=False,                                    #Return the density profile parameters
                                  spline_data=False,                                #Calculate the results from the spline data (no error is going to be taken into account)
                                  device_coordinates=False,                          #Calculate the results as a function of device coordinates
                                  radial_range=None,                                #Radial range of the pedestal (only works when the device coorinates is set)
                                  flux_coordinates=False,                           #Calculate the results in flux coordinates
                                  flux_range=None,                                  #The normalaized flux coordinates range for returning the results
                                  test=False,
                                  output_name=None,
                                  return_parameters=False,
                                  plot_time=None,
                                  pdf_object=None,
                                  modified_tanh=False,
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
        x_data=d.coordinate(r_coord_name)[0][:,i_time]
        y_data=d.data[:,i_time]
        y_data_error=d.error[:,i_time]
        if r_coord_name=='Flux r':
            x_data=x_data[np.argmin(x_data):]
            
        if np.sum(np.isinf(x_data)) != 0:
            continue

        ind_coord=np.where(np.logical_and(x_data > x_range[0],
                                          x_data <= x_range[1]))
        x_data=x_data[ind_coord]
        y_data=y_data[ind_coord]
        y_data_error=y_data_error[ind_coord]
        try:
            if not modified_tanh:
                p0=[y_data[0],                                      #b_height
                    y_data[-1],                                     #b_sol
                    x_data[0],                                      #b_pos
                    (x_data[-1]-x_data[0])/2.,                      #b_width
                    #(y_data[0]-y_data[-1])/(x_data[0]-x_data[-1]), #b_slope this is supposed to be some kind of linear modification to the 
                                                                    #tanh function called mtanh. It messes up the fitting quite a bit and it's not useful at all.
                    ]
    
            else:
                p0=[y_data[0],                                      #b_height
                    y_data[-1],                                     #b_sol
                    x_data[0],                                      #b_pos
                    (x_data[-1]-x_data[0])/2.,                      #b_width
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
        except:
            if modified_tanh:
                popt=[np.nan,np.nan,np.nan,np.nan,np.nan]
                perr=[np.nan,np.nan,np.nan,np.nan,np.nan]
            else:
                popt=[np.nan,np.nan,np.nan,np.nan]
                perr=[np.nan,np.nan,np.nan,np.nan]
        
        if test or (plot_time is not None and np.abs(plot_time-time[i_time]) < 1e-3):
            plt.cla()
            plt.scatter(x_data,
                        y_data,
                        color='tab:blue')
            plt.errorbar(x_data,
                         y_data,
                         yerr=y_data_error,
                         marker='o', 
                         color='tab:blue',
                         ls='')
            plt.plot(x_data,tanh_fit_function(x_data,*popt))
            
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
            
        thomson_profile['Height'][i_time]=popt[0]
        thomson_profile['SOL offset'][i_time]=popt[1]
        thomson_profile['Position'][i_time]=popt[2]
        #try:
        if True:
            thomson_profile['Position r'][i_time]=np.interp(popt[2],
                                                            d.coordinate('Flux r')[0][np.argmin(d.coordinate('Flux r')[0][:,i_time]):,i_time],
                                                            d.coordinate('Device R')[0][np.argmin(d.coordinate('Flux r')[0][:,i_time]):,i_time])
        #except:
        #    print('Interpolation failed.')
        #    thomson_profile['Position r'][i_time]=np.nan
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
            

    coord = []
    
    coord.append(copy.deepcopy(flap.Coordinate(name='Time',
                               unit='s',
                               mode=flap.CoordinateMode(equidistant=True),
                               start=time[0],
                               step=time[1]-time[0],
                               #shape=time_arr.shape,
                               dimension_list=[0]
                               )))
        
    coord.append(copy.deepcopy(flap.Coordinate(name='Sample',
                               unit='n.a.',
                               mode=flap.CoordinateMode(equidistant=True),
                               start=0,
                               step=1,
                               dimension_list=[0]
                               )))
    if device_coordinates:
        grad_unit='/m'
    if flux_coordinates:
        grad_unit='/psi'
    if pressure:
        data_unit = flap.Unit(name='Pressure gradient',unit='kPa'+grad_unit)
    elif temperature:
        data_unit = flap.Unit(name='Temperature gradient',unit='keV'+grad_unit)
    elif density:
        data_unit = flap.Unit(name='Density gradient',unit='m-3'+grad_unit)
        
    if spline_data:
        data_title='NSTX Thomson gradient'
    else:
        data_title='NSTX Thomson gradient spline'
    d = flap.DataObject(exp_id=exp_id,
                        data_array=thomson_profile['Max gradient'],
                        data_unit=data_unit,
                        coordinates=coord, 
                        data_title=data_title)
        
    if output_name is not None:
        flap.add_data_object(d,output_name)
    if not return_parameters:
        return d
    else:
        return thomson_profile
        
    
