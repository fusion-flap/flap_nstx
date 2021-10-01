#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 12:21:23 2021

@author: mlampert
"""


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
    
    coord.append(copy.deepcopy(flap.Coordinate(name='Device x',
                               unit='m',
                               mode=flap.CoordinateMode(equidistant=True),
                               start=x_coord[0],
                               step=x_coord[1]-x_coord[0],
                               dimension_list=[1]
                               )))
    coord.append(copy.deepcopy(flap.Coordinate(name='Device y',
                               unit='m',
                               mode=flap.CoordinateMode(equidistant=True),
                               start=y_coord[0],
                               step=y_coord[1]-y_coord[0],
                               dimension_list=[2]
                               )))
    
    coord.append(copy.deepcopy(flap.Coordinate(name='Image x',
                               unit='m',
                               mode=flap.CoordinateMode(equidistant=True),
                               start=0,
                               step=1,
                               dimension_list=[1]
                               )))
    coord.append(copy.deepcopy(flap.Coordinate(name='Image y',
                               unit='m',
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