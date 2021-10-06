#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 17:44:53 2021

@author: mlampert
"""

import os
import copy

#FLAP imports and settings
import flap
import flap_nstx
import flap_mdsplus

flap_nstx.register('NSTX_GPI')
flap_nstx.register('NSTX_THOMSON')

flap_mdsplus.register('NSTX_MDSPlus')

thisdir = os.path.dirname(os.path.realpath(__file__))
fn = os.path.join(thisdir,"../flap_nstx.cfg")
flap.config.read(file_name=fn)

#Scientific imports
import numpy as np

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
        
    # thomson=flap_nstx_thomson_data(exp_id, 
    #                                pressure=pressure,
    #                                temperature=temperature,
    #                                density=density,
    #                                output_name='THOMSON_FOR_GRADIENT')
    thomson=flap.get_data('NSTX_THOMSON', 
                          exp_id=exp_id,
                          object_name='THOMSON_FOR_GRADIENT', 
                          options={'pressure':pressure,
                                   'temperature':temperature,
                                   'density':density,
                                   'force_mdsplus':True})
    # thomson_spline=flap_nstx_thomson_data(exp_id, 
    #                                       pressure=pressure,
    #                                       temperature=temperature,
    #                                       density=density,
    #                                       spline_data=True,
    #                                       output_name=None)
    
    thomson_spline=flap.get_data('NSTX_THOMSON', 
                                 exp_id=exp_id,
                                 object_name=None, 
                                 options={'pressure':pressure,
                                          'temperature':temperature,
                                          'density':density,
                                          'spline_data':True,
                                          'force_mdsplus':True})
    
    
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
