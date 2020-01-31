#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 14:45:09 2020

@author: mlampert
"""

import os
import copy

import flap
import flap_nstx
flap_nstx.register()
import numpy as np
import scipy
import matplotlib.pyplot as plt

thisdir = os.path.dirname(os.path.realpath(__file__))
fn = os.path.join(thisdir,"flap_nstx.cfg")
flap.config.read(file_name=fn)

def nstx_gpi_generate_synthetic_data(exp_id=None,                               #Artificial exp_id, should be starting from zero and not the one which is used by e.g. background_shot
                                     time=None,                                 #Time to be simulated in seconds
                                     sampling_time=2.5e-6,                      #The sampling time of the diagnostic
                                     #General parameters
                                     n_structures=3,
                                     amplitude=0.5,                             #Amplitude of the structure relative to the background.
                                     add_background=True,
                                     background_shot=139901,                    #The original of the background of the simulated signal.
                                     background_time_range=[0.31,0.32],         #The time range of the background in the background_shot.
                                     poloidal_velocity=1e3,                     #The poloidal velocity of the structures, can be a list [n_structure]
                                     start_position=[1.41,0.195],
                                     radial_size=0.3,                           #The radial size of the structures.
                                     poloidal_size=0.1,                         #The poloidal size of the structures.
                                     #Parameters for a gaussian object
                                     gaussian=False,
                                     radial_velocity=1e2,                       #The radial velocity of the structures, can be a list [n_structure]
                                     poloidal_size_velocity=0.,                 #The velocity of the size change in mm/ms
                                     radial_size_velocity=0.,
                                     rotation=False,                            #Set rotation for the structure.
                                     rotation_frequency=None,                   #Set the frequency of the rotation of the structure.
                                     #Parameters for sinusoidal object
                                     sinusoidal=False,
                                     waveform_divider=1,
                                     y_lambda=0.05,                             #Wavelength in the y direction
                                     output_name=None,                          #Output name of the generated flap.data_object
                                     test=False,                                #Testing/debugging switch (mainly plotting and printing error messages)
                                     ):
    if rotation_frequency is None:
        rotation_frequency=0
    
    n_time=int(time/sampling_time)
    data_arr=np.zeros([n_time,64,80])
    
    background=np.zeros([64,80])
    if add_background:
        background=flap.get_data('NSTX_GPI', exp_id=139901, name='', object_name='GPI_RAW').slice_data(slicing={'Time':flap.Intervals(background_time_range[0],background_time_range[1])},summing={'Time':'Mean'})
        amplitude=amplitude*background.data.max()
    
    #Spatial positions
    coeff_r=np.asarray([3.7183594,-0.77821046,1402.8097])/1000. #The coordinates are in meters
    coeff_z=np.asarray([0.18090118,3.0657776,70.544312])/1000. #The coordinates are in meters
    
    r_coordinates=np.zeros([64,80])
    z_coordinates=np.zeros([64,80])
    
    for i_x in range(64):
        for i_y in range(80):
            r_coordinates[i_x,i_y]=coeff_r[0]*i_x+coeff_r[1]*i_y+coeff_r[2]
            z_coordinates[i_x,i_y]=coeff_z[0]*i_x+coeff_z[1]*i_y+coeff_z[2]
    if gaussian:            
        r0=start_position
        
        for i_frames in range(n_time):
            for i_structures in range(n_structures):
                cur_time=i_frames * sampling_time
                rot_arg=2*np.pi*rotation_frequency*cur_time
                a=(np.cos(rot_arg)/(radial_size+radial_size_velocity*cur_time))**2+\
                  (np.sin(rot_arg)/(poloidal_size+poloidal_size_velocity*cur_time))**2
                b=-0.5*np.sin(2*rot_arg)/(radial_size+radial_size_velocity*cur_time)**2+\
                   0.5*np.sin(2*rot_arg)/(poloidal_size+poloidal_size_velocity*cur_time)**2
                c=(np.sin(rot_arg)/(radial_size+radial_size_velocity*cur_time))**2+\
                  (np.cos(rot_arg)/(poloidal_size+poloidal_size_velocity*cur_time))**2
                x0=r0[i_structures,0]+radial_velocity[i_structures]*cur_time
                y0=r0[i_structures,1]+poloidal_velocity[i_structures]*cur_time
                frame=np.zeros([64,80])
                for j_vertical in range(80):
                    for k_radial in range(64):
                        x=r_coordinates[k_radial,j_vertical]
                        y=z_coordinates[k_radial,j_vertical]                        
                        if (x > x0+radial_size*2 or
                            x < x0-radial_size*2 or
                            y > y0+radial_size*2 or
                            y < y0-radial_size*2):
                            frame[k_radial,j_vertical]=0.
                        else:
                            frame[k_radial,j_vertical]=(amplitude[i_structures]*np.exp(-0.5*(a*(x-x0)**2 + 
                                                                                                          2*b*(x-x0)*(y-y0) + 
                                                                                                          c*(y-y0)**2))
                                                               +background.data[k_radial,j_vertical])
                data_arr[i_frames,:,:]+=frame
                
    if sinusoidal:
        x0=start_position[0]
        y0=start_position[1]
        ky=2*np.pi/poloidal_size
        omega=poloidal_velocity*ky
        phi0=0
        for i_frames in range(n_time):
            cur_time=i_frames * sampling_time
            for j_vertical in range(80):
                for k_radial in range(64):
                    x=r_coordinates[k_radial,j_vertical]
                    y=z_coordinates[k_radial,j_vertical]
                    if np.abs(x-x0)*2 > radial_size: 
                        A=1/np.sqrt(2*np.pi*radial_size)*np.exp(-0.5*((x-x0)/radial_size)**4)
                    else:
                        A=1/np.sqrt(2*np.pi*radial_size)*np.exp(-0.5*((x-x0)/radial_size)**2)
                    arg=ky*y-omega*cur_time+phi0
                    division=(scipy.signal.square(arg/waveform_divider, duty=0.5/waveform_divider)+1)/2.
                    data_arr[i_frames,k_radial,j_vertical]=amplitude*A*np.sin(arg)*division
                    data_arr[i_frames,k_radial,j_vertical]+=background.data[k_radial,j_vertical]
    #Adding the coordinates to the data object:
    
    
    coord = [None]*6
    coord[0]=(copy.deepcopy(flap.Coordinate(name='Time',
                                            unit='s',
                                            mode=flap.CoordinateMode(equidistant=True),
                                            start=0.,
                                            step=sampling_time,
                                            #shape=time_arr.shape,
                                            dimension_list=[0]
                                            )))
    coord[1]=(copy.deepcopy(flap.Coordinate(name='Sample',
                                            unit='n.a.',
                                            mode=flap.CoordinateMode(equidistant=True),
                                            start=0,
                                            step=1,
                                            dimension_list=[0]
                                            )))
    coord[2]=(copy.deepcopy(flap.Coordinate(name='Image x',
                                            unit='Pixel',
                                            mode=flap.CoordinateMode(equidistant=True),
                                            start=0,
                                            step=1,
                                            shape=[],
                                            dimension_list=[1]
                                            )))
    coord[3]=(copy.deepcopy(flap.Coordinate(name='Image y',
                                            unit='Pixel',
                                            mode=flap.CoordinateMode(equidistant=True),
                                            start=0,
                                            step=1,
                                            shape=[],
                                            dimension_list=[2]
                                            )))
    coord[4]=(copy.deepcopy(flap.Coordinate(name='Device R',
                                            unit='m',
                                            mode=flap.CoordinateMode(equidistant=False),
                                            values=r_coordinates,
                                            shape=r_coordinates.shape,
                                            dimension_list=[1,2]
                                            )))
    
    coord[5]=(copy.deepcopy(flap.Coordinate(name='Device z',
                                           unit='m',
                                           mode=flap.CoordinateMode(equidistant=False),
                                           values=z_coordinates,
                                           shape=z_coordinates.shape,
                                           dimension_list=[1,2]
                                           )))
    _options={}
    _options["Trigger time [s]"]=0.
    _options["FPS"]=1/sampling_time
    _options["Sample time [s]"]=sampling_time
    _options["Exposure time [s]"]=2.1e-6
    _options["X size"]=64
    _options["Y size"]=80
    _options["Bits"]=32
    
    d = flap.DataObject(data_array=data_arr,
                        data_unit=flap.Unit(name='Signal',unit='Digit'),
                        coordinates=coord,
                        exp_id=exp_id,
                        data_title='Simulated signal',
                        info={'Options':_options},
                        data_source="NSTX_GPI")
    if output_name is not None:
        flap.add_data_object(d,output_name)
    
    return d
    