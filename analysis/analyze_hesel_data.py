#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 16:23:01 2021

@author: mlampert
"""

from flap_nstx.analysis import generate_displaced_gaussian, generate_displaced_random_noise, calculate_nstx_gpi_frame_by_frame_velocity, calculate_sde_velocity, calculate_sde_velocity_distribution

#Core modules
import os
import copy
import h5py
import scipy

import flap
import flap_nstx
flap_nstx.register()
import flap_mdsplus
flap_mdsplus.register('NSTX_MDSPlus')

thisdir = os.path.dirname(os.path.realpath(__file__))
fn = os.path.join(thisdir,"flap_nstx.cfg")
flap.config.read(file_name=fn)


import numpy as np


def read_flap_hesel_data(output_name=None):
    f=h5py.File('/Users/mlampert/work/NSTX_workspace/AUG_HESEL_files/n_e__nHESEL_AUG_00000.h5', 'r')
    
    x_coord=np.asarray(list(f['axes']['x_axis']))
    y_coord=np.asarray(list(f['axes']['x_axis']))
    t_coord=np.asarray(list(f['axes']['t_axis']))
    data=np.asarray(list(f['fields']['n_e'])) #t, x, y
    coord=[]
    coord.append(copy.deepcopy(flap.Coordinate(name='Time',
                               unit='s',
                               mode=flap.CoordinateMode(equidistant=True),
                               start=t_coord[0],
                               step=t_coord[1]-t_coord[0],
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
    
    coord.append(copy.deepcopy(flap.Coordinate(name='Device R',
                               unit='m',
                               mode=flap.CoordinateMode(equidistant=True),
                               start=x_coord[0],
                               step=x_coord[1]-x_coord[0],
                               dimension_list=[1]
                               )))
    coord.append(copy.deepcopy(flap.Coordinate(name='Device z',
                               unit='m',
                               mode=flap.CoordinateMode(equidistant=True),
                               start=y_coord[0],
                               step=y_coord[1]-y_coord[0],
                               dimension_list=[2]
                               )))
    
    coord.append(copy.deepcopy(flap.Coordinate(name='Image x',
                               unit='pix',
                               mode=flap.CoordinateMode(equidistant=True),
                               start=0,
                               step=1,
                               dimension_list=[1]
                               )))
    coord.append(copy.deepcopy(flap.Coordinate(name='Image y',
                               unit='pix',
                               mode=flap.CoordinateMode(equidistant=True),
                               start=0,
                               step=1,
                               dimension_list=[2]
                               )))    
    
    data_unit = flap.Unit(name='Density',unit='m-3')
    d = flap.DataObject(data_array=data,
                        error=None,
                        data_unit=data_unit,
                        coordinates=coord, 
                        exp_id=0,
                        data_title='AUG HESEL DATA')
    if output_name is not None:
        flap.add_data_object(d, output_name)
    return d

def analyze_hesel_data():
    try:
        d=flap.get_data_object_ref('HESEL_DATA')
    except:
        d=read_flap_hesel_data(output_name='HESEL_DATA')
    
    a=calculate_sde_velocity(d, time_range=[0,100e-6], filename='sde_code_trial', nocalc=False, correct_acf_peak=True, subtraction_order=4)
    
def analyze_hesel_data_2D():
    try:
        d=flap.get_data_object_ref('HESEL_DATA')
    except:
        d=read_flap_hesel_data(output_name='HESEL_DATA')
        
    result=calculate_sde_velocity_distribution(d, 
                                               time_range=[0,100e-6], 
                                               x_step=20, x_res=40, y_step=40, y_res=40, 
                                               filename='sde_code_trial', 
                                               nocalc=False, 
                                               correct_acf_peak=True, 
                                               subtraction_order=1, 
                                               normalize=None,
                                               correlation_threshold=0.7, 
                                               return_data_object=True, 
                                               return_displacement=True)
    
    flap.add_data_object(result[0], 'X_OBJ')
    flap.add_data_object(result[1], 'Y_OBJ')
    
    """
    
    There should be a check for the overplotted object, whether the time axis 
    is the same there as for the data. If not, they need to be matched.
    
    """
    
    oplot_options={'arrow':{'DISP':{'Data object X':'X_OBJ',
                                    'Data object Y':'Y_OBJ',
                                    'Plot':True,
                                    'width':0.0001,
                                    'color':'red',
                                    }}}
    
    d.plot(plot_type='animation', 
           slicing={'Time':flap.Intervals(0,100e-6)},
           axes=['Device R', 'Device z', 'Time'],
           options={'Wait':0.0, 
                'Clear':False,
                'Overplot options':oplot_options,
                'Equal axes': True,
                'Plot units':{'Time':'s',
                              'Device R':'m',
                              'Device z':'m'}
                }
       )