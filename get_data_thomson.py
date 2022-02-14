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
import flap_mdsplus

flap_mdsplus.register('NSTX_MDSPlus')

thisdir = os.path.dirname(os.path.realpath(__file__))
fn = os.path.join(thisdir,"flap_nstx.cfg")
flap.config.read(file_name=fn)

#Scientific imports
import numpy as np

import matplotlib.pyplot as plt
#Other necessary imports
import MDSplus as mds
import pickle

def get_data_thomson(exp_id=None, 
                     data_name=None, 
                     no_data=False, 
                     options=None, 
                     coordinates=None, 
                     data_source=None):
        
    default_options = {'Temperature': False,
                       'Density': False,
                       'Pressure': False,
                       'test': False,
                       
                       'output_name': None,
                       'add_flux_coordinates': False,
                       'spline_data': False,
                       'force_mdsplus':False
                       }
    
    _options = flap.config.merge_options(default_options,options,data_source=data_source)
    
    temperature=_options['Temperature']
    density=_options['Density']
    pressure=_options['Pressure']
    test=_options['test']
    output_name=_options['output_name']
    add_flux_coordinates=_options['add_flux_coordinates']
    spline_data=_options['spline_data']
    force_mdsplus=_options['force_mdsplus']
    
    """
    Returns the Thomson scattering processed data from the MDSplus tree as
    a dictionary containing all the necessary parameters. The description of
    the dictionary can be seen below.
    """
    if pressure+temperature+density != 1:
        raise ValueError('Either pressure or temperature or density can be set, neither none, nor more than one.')
    if exp_id is None:
        raise TypeError('exp_id must be set.')
        
    wd=flap.config.get_all_section('Module NSTX_GPI')['Local datapath']
    filename=wd+'/'+str(exp_id)+'/nstx_mdsplus_thomson_'+str(exp_id)+'.pickle'
    
    if not os.path.exists(filename) or force_mdsplus:
        conn = mds.Connection('skylark.pppl.gov:8501')
        conn.openTree('activespec', exp_id)
        
        mdsnames=['ts_times',           #The time vector of the measurement (60Hz measurement with the Thomson)
                  'FIT_RADII',          #Radius of the measurement                  
                  'FIT_R_WIDTH',        
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
        
        thomson['FIT_NE'] *= 1e6
        thomson['FIT_NE_ERR'] *= 1e6
        thomson['SPLINE_NE'] *= 1e6
        
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

def add_coordinate_thomson(data_object,
                           coordinates,
                           exp_id=None,
                           options=None):
    raise NotImplementedError("New coordinates need to be added, everything else is added to the FLAP object as default.")