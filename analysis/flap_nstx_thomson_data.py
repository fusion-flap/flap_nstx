#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 18:24:43 2020

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
fn = os.path.join(thisdir,"flap_nstx.cfg")
flap.config.read(file_name=fn)

#Scientific imports
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
#Other necessary imports
import MDSplus as mds
import pickle

def flap_nstx_thomson_data(exp_id=None,
                           force_mdsplus=False,
                           pressure=False,
                           temperature=False,
                           density=False,
                           spline_data=False,
                           add_flux_coordinates=True,
                           output_name=None,
                           test=False):
    
    
    """
    Returns the Thomson scattering processed data from the MDSplus tree as
    a dictionary containing all the necessary parameters. The description of
    the dictionary can be seen below.
    """
    
    if exp_id is None:
        raise TypeError('exp_id must be set.')
        
    wd=flap.config.get_all_section('Module NSTX_GPI')['Local datapath']
    filename=wd+'/'+str(exp_id)+'/nstx_mdsplus_thomson_'+str(exp_id)+'.pickle'
    
    if not os.path.exists(filename) or force_mdsplus:
        conn = mds.Connection('skylark.pppl.gov:8501')
        conn.openTree('activespec', exp_id)
        
        mdsnames=['ts_times',           #The time vector of the measurement (60Hz measurement with the Thomson)
                  'FIT_RADII',          #Radius of the measurement                  
                  'FIT_R_WIDTH',        #N/A (proably error of the radious)
                  'FIT_TE',             #Electron temperature profile numpy array([radius,time])
                  'FIT_TE_ERR',         #The error for Te (symmetric)
                  'FIT_NE',             #Electron density profile numpy array([radius,time])
                  'FIT_NE_ERR',         #The error for ne (symmetric)
                  'FIT_PE',             #Electron pressure profile numpy array([radius,time])
                  'FIT_PE_ERR',         #The error for pe (symmetric)
                  'SPLINE_RADII',       #Spline fit of the previous results (4times interpolation compared to the previous ones)
                  'SPLINE_NE',          #Spline fit ne without error
                  'SPLINE_PE',          #Spline fit pe without error
                  'SPLINE_TE',          #Spline fit Te without error
                  'TS_LD',              #N/A
                  'LASER_ID',           #ID of the Thomson laser
                  'VALID',              #Validity of the measurement
                  'DATEANALYZED',       #The date when the analysis was done for the data
                  'COMMENT']            #Comment for the analysis

        thomson={}
        for name in mdsnames:
            thomson[name]=conn.get('\TS_BEST:'+name).data()
            if name == 'ts_times' and type(thomson[name]) is str:
                raise ValueError('No Thomson data available.')
        
        thomson['FIT_R_WIDTH'] /= 100.
        thomson['FIT_RADII'] /= 100.
        thomson['SPLINE_RADII'] /= 100.
        
        thomson['FIT_NE'] *= 10e6
        thomson['FIT_NE_ERR'] *= 10e6
        thomson['SPLINE_NE'] *= 10e6
        
        conn.closeAllTrees()
        conn.disconnect()
        try:
            pickle.dump(thomson,open(filename, 'wb'))
        except:
            raise IOError('The path '+filename+' cannot be accessed. Pickle file cannot be created.')
    else:
        thomson=pickle.load(open(filename, 'rb'))
    
    thomson_time=thomson['ts_times']
    
    coord = []

    coord.append(copy.deepcopy(flap.Coordinate(name='Time',
                               unit='s',
                               mode=flap.CoordinateMode(equidistant=True),
                               start=thomson_time[0],
                               step=thomson_time[1]-thomson_time[0],
                               #shape=time_arr.shape,
                               dimension_list=[1]
                               )))
    
    coord.append(copy.deepcopy(flap.Coordinate(name='Sample',
                               unit='n.a.',
                               mode=flap.CoordinateMode(equidistant=True),
                               start=0,
                               step=1,
                               dimension_list=[1]
                               )))
    if spline_data:
        thomson_r_coord=thomson['SPLINE_RADII']
        if pressure:
            data_arr=thomson['SPLINE_PE']
            data_arr_err=None
            data_unit = flap.Unit(name='Pressure',unit='kPa')
        elif temperature:
            data_arr=thomson['SPLINE_TE']
            data_arr_err=None
            data_unit = flap.Unit(name='Temperature',unit='keV')
        elif density:
            data_arr=thomson['SPLINE_NE']
            data_arr_err=None
            data_unit = flap.Unit(name='Density',unit='m-3')
    else:
        thomson_r_coord=thomson['FIT_RADII']       
        if pressure:
            data_arr=thomson['FIT_PE']
            data_arr_err=thomson['FIT_PE_ERR']
            data_unit = flap.Unit(name='Pressure',unit='kPa')
        elif temperature:
            data_arr=thomson['FIT_TE']
            data_arr_err=thomson['FIT_TE_ERR']
            data_unit = flap.Unit(name='Temperature',unit='keV')
        elif density:
            data_arr=thomson['FIT_NE']
            data_arr_err=thomson['FIT_NE_ERR']
            data_unit = flap.Unit(name='Density',unit='m-3')
    
    coord.append(copy.deepcopy(flap.Coordinate(name='Device R',
                                               unit='m',
                                               mode=flap.CoordinateMode(equidistant=False),
                                               values=thomson_r_coord,
                                               shape=thomson_r_coord.shape,
                                               dimension_list=[0]
                                               )))
    if test:
        plt.figure()
    if add_flux_coordinates:
        try:
            psi_rz_obj=flap.get_data('NSTX_MDSPlus',
                                     name='\EFIT02::\PSIRZ',
                                     exp_id=exp_id,
                                     object_name='PSIRZ_FOR_COORD')
            psi_mag=flap.get_data('NSTX_MDSPlus',
                                  name='\EFIT02::\SSIMAG',
                                  exp_id=exp_id,
                                  object_name='SSIMAG_FOR_COORD')
            psi_bdry=flap.get_data('NSTX_MDSPlus',
                                   name='\EFIT02::\SSIBRY',
                                   exp_id=exp_id,
                                   object_name='SSIBRY_FOR_COORD')
        except:
            raise ValueError("The PSIRZ MDSPlus node cannot be reached.")
            
        psi_values=psi_rz_obj.data[:,:,32]
        psi_t_coord=psi_rz_obj.coordinate('Time')[0][:,0,0]
        psi_r_coord=psi_rz_obj.coordinate('Device R')[0][:,:,32]            #midplane is the middle coordinate in the array
        
        #Do the interpolation
        psi_values_spat_interpol=np.zeros([thomson_r_coord.shape[0],
                                           psi_t_coord.shape[0]])
        
        for index_t in range(psi_t_coord.shape[0]):
            norm_psi_values=(psi_values[index_t,:]-psi_mag.data[index_t])/(psi_bdry.data[index_t]-psi_mag.data[index_t])
            norm_psi_values[np.isnan(norm_psi_values)]=0.
            psi_values_spat_interpol[:,index_t]=np.interp(thomson_r_coord,psi_r_coord[index_t,:],norm_psi_values)
        
        psi_values_total_interpol=np.zeros(data_arr.shape)

        for index_r in range(data_arr.shape[0]):
            psi_values_total_interpol[index_r,:]=np.interp(thomson_time,psi_t_coord,psi_values_spat_interpol[index_r,:])
            
        if test:
            for index_t in range(len(thomson_time)):
                plt.cla()
                plt.plot(thomson_r_coord,psi_values_total_interpol[:,index_t])
                plt.pause(0.5)
            
        psi_values_total_interpol[np.isnan(psi_values_total_interpol)]=0.
        
        coord.append(copy.deepcopy(flap.Coordinate(name='Flux r',
                                   unit='',
                                   mode=flap.CoordinateMode(equidistant=False),
                                   values=psi_values_total_interpol,
                                   shape=psi_values_total_interpol.shape,
                                   dimension_list=[0,1]
                                   )))
    if test:
        plt.plot(psi_values_total_interpol, data_arr)
    d = flap.DataObject(data_array=data_arr,
                        error=data_arr_err,
                        data_unit=data_unit,
                        coordinates=coord, 
                        exp_id=exp_id,
                        data_title='NSTX Thomson data')
    
    if output_name is not None:
        flap.add_data_object(d, output_name)
    return d

def get_nstx_thomson_gradient(exp_id=None,
                              pressure=False,
                              temperature=False,
                              density=False,
                              r_pos=None,
                              spline_data=False,
                              output_name=None,
                              device_coordinates=True,
                              flux_coordinates=False):
    
    #Data is RADIUS x TIME
    if pressure+density+temperature != 1:
        raise ValueError('Only one of the inputs should be set (pressure, temperature, density)!')
    if device_coordinates+flux_coordinates !=1:
        raise ValueError('Either device_coordinates or flux_coordinates can be set, not both.')
        
    thomson=flap_nstx_thomson_data(exp_id, 
                                   pressure=pressure,
                                   temperature=temperature,
                                   density=density,
                                   output_name='THOMSON_FOR_GRADIENT')
    thomson_spline=flap_nstx_thomson_data(exp_id, 
                                          pressure=pressure,
                                          temperature=temperature,
                                          density=density,
                                          spline_data=True,
                                          output_name=None)
    if device_coordinates:
        radial_coordinate=thomson.coordinate('Device R')[0][:,0]
        spline_radial_coordinate=thomson_spline.coordinate('Device R')[0][:,0]
    if flux_coordinates:
        radial_coordinate=thomson.coordinate('Flux r')[0][:,0]
        spline_radial_coordinate=thomson_spline.coordinate('Flux r')[0][:,0]
        
    time_vector=thomson.coordinate('Time')[0][0,:]
    data=thomson.data
    error=thomson.error
    interp_data=thomson_spline.data
    
    #Calculation of the numerical gradient and interpolating the values for the given r_pos
    data_gradient=np.asarray([(data[2:,i]-data[:-2,i])/(2*(radial_coordinate[2:]-radial_coordinate[:-2])) for i in range(len(time_vector))]).T
    data_gradient_error=np.asarray([(np.abs(error[2:,i])+np.abs(error[:-2,i]))/(2*(radial_coordinate[2:]-radial_coordinate[:-2])) for i in range(len(time_vector))]).T
    interp_data_gradient=np.asarray([(interp_data[2:,i]-interp_data[:-2,i])/(2*(spline_radial_coordinate[2:]-spline_radial_coordinate[:-2])) for i in range(len(time_vector))]).T
    
    #Interpolation for the r_pos
    if r_pos is not None:
        r_pos_gradient=np.asarray([np.interp(r_pos, radial_coordinate[1:-1], data_gradient[:,i]) for i in range(len(time_vector))])
        r_pos_gradient_spline=np.asarray([np.interp(r_pos, spline_radial_coordinate[1:-1], interp_data_gradient[:,i]) for i in range(len(time_vector))])
    
        ind_r=np.argmin(np.abs(radial_coordinate[1:-1]-r_pos))
        if radial_coordinate[ind_r] < r_pos:
            R1=radial_coordinate[1:-1][ind_r]
            R2=radial_coordinate[1:-1][ind_r+1]
            ind_R1=ind_r
            ind_R2=ind_r+1
        else:
            R1=radial_coordinate[1:-1][ind_r-1]
            R2=radial_coordinate[1:-1][ind_r]
            ind_R1=ind_r-1
            ind_R2=ind_r
        #Result of error propagation (basically average biased error between the two neighboring radii)
        r_pos_gradient_error=np.abs((r_pos-R1)/(R2-R1))*data_gradient_error[ind_R2,:]+\
                             np.abs((r_pos-R2)/(R2-R1))*data_gradient_error[ind_R1,:]
            
        coord = []
    
        coord.append(copy.deepcopy(flap.Coordinate(name='Time',
                                   unit='s',
                                   mode=flap.CoordinateMode(equidistant=True),
                                   start=time_vector[0],
                                   step=time_vector[1]-time_vector[0],
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
            
        if not spline_data:
            d = flap.DataObject(exp_id=exp_id,
                                data_array=r_pos_gradient,
                                error=r_pos_gradient_error,
                                data_unit=data_unit,
                                coordinates=coord, 
                                data_title='NSTX Thomson gradient')
        else:
            d = flap.DataObject(exp_id=exp_id,
                                data_array=r_pos_gradient_spline,
                                data_unit=data_unit,
                                coordinates=coord, 
                                data_title='NSTX Thomson gradient spline')
    else:
        coord = []
    
        coord.append(copy.deepcopy(flap.Coordinate(name='Time',
                                   unit='s',
                                   mode=flap.CoordinateMode(equidistant=True),
                                   start=time_vector[0],
                                   step=time_vector[1]-time_vector[0],
                                   #shape=time_arr.shape,
                                   dimension_list=[1]
                                   )))
        
        coord.append(copy.deepcopy(flap.Coordinate(name='Sample',
                                   unit='n.a.',
                                   mode=flap.CoordinateMode(equidistant=True),
                                   start=0,
                                   step=1,
                                   dimension_list=[1]
                                   )))
        if pressure:
            data_unit = flap.Unit(name='Pressure gradient',unit='kPa/m')
        elif temperature:
            data_unit = flap.Unit(name='Temperature gradient',unit='keV/m')
        elif density:
            data_unit = flap.Unit(name='Density gradient',unit='m-3/m')
        if device_coordinates:
            radial_coordinate_name='Device R'
            radial_unit='m'
        if flux_coordinates:
            radial_coordinate_name='Flux r'    
            radial_unit=''
        if not spline_data:
            coord.append(copy.deepcopy(flap.Coordinate(name=radial_coordinate_name,
                         unit=radial_unit,
                         mode=flap.CoordinateMode(equidistant=False),
                         values=radial_coordinate[1:-1],
                         shape=radial_coordinate[1:-1].shape,
                         dimension_list=[0]
                         )))
            d = flap.DataObject(exp_id=exp_id,
                                data_array=data_gradient,
                                error=data_gradient_error,
                                data_unit=data_unit,
                                coordinates=coord, 
                                data_title='NSTX Thomson gradient')
        else:
            coord.append(copy.deepcopy(flap.Coordinate(name=radial_coordinate_name,
                                       unit=radial_unit,
                                       mode=flap.CoordinateMode(equidistant=False),
                                       values=spline_radial_coordinate[1:-1],
                                       shape=spline_radial_coordinate[1:-1].shape,
                                       dimension_list=[0]
                                       )))
            
            d = flap.DataObject(exp_id=exp_id,
                                data_array=interp_data_gradient,
                                data_unit=data_unit,
                                coordinates=coord, 
                                data_title='NSTX Thomson gradient spline')
        
    if output_name is not None:
        flap.add_data_object(d,output_name)
    return d

def fit_nstx_thomson_profiles(exp_id=None,
                              pressure=False,
                              temperature=False,
                              density=False,
                              spline_data=False,
                              device_coordinates=True,
                              flux_coordinates=False,
                              test=False,):
    """
    Fitting is based on publication https://aip.scitation.org/doi/pdf/10.1063/1.4961554
    """
    def mtanh_fit_function(r, b_height, b_sol, b_pos, b_width, b_slope):
        def mtanh(x,b_slope):
            return ((1+b_slope*x)*np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
        return (b_height-b_sol)*(mtanh((b_pos-r)/(2*b_width),b_slope)+1)
    
    
    d=flap_nstx_thomson_data(exp_id=exp_id,
                             force_mdsplus=False,
                             pressure=pressure,
                             temperature=temperature,
                             density=density,
                             spline_data=False,
                             add_flux_coordinates=True,
                             output_name='THOMSON_DATA')
    if flux_coordinates:
        r_coord_name='Flux r'
    if device_coordinates:
        r_coord_name='Device R'
    time=d.coordinate('Time')[0][0,:]
    thomson_profile_parameters={'Height':np.zeros(time.shape),
                                'Width':np.zeros(time.shape),
                                'Slope':np.zeros(time.shape),
                                'Position':np.zeros(time.shape),
                                'SOL offset':np.zeros(time.shape),
                                }
    if test:
        plt.figure()
    for i_time in range(len(time)):
        x_data=d.coordinate(r_coord_name)[0][:,i_time]
        y_data=d.data[:,i_time]
        popt, pcov = curve_fit(mtanh_fit_function, x_data, y_data)
        if test:
            plt.cla()
            plt.scatter(x_data,y_data)
            plt.plot(x_data,mtanh_fit_function(x_data,*popt))
            plt.pause(0.1)
        thomson_profile_parameters['Height'][i_time]=popt[0]
        thomson_profile_parameters['SOL offset'][i_time]=popt[1]
        thomson_profile_parameters['Position'][i_time]=popt[2]
        thomson_profile_parameters['Width'][i_time]=popt[3]
        thomson_profile_parameters['Slope'][i_time]=popt[4]
        
    return thomson_profile_parameters