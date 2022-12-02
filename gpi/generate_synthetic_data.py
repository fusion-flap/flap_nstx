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

from skimage.util import img_as_float
from skimage.transform import warp_polar, rotate, rescale, warp

from imageio import imread

thisdir = os.path.dirname(os.path.realpath(__file__))
fn = os.path.join(thisdir,"../flap_nstx.cfg")
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
        background=flap.get_data('NSTX_GPI',
                                 exp_id=139901,
                                 name='',
                                 object_name='GPI_RAW')
        background=background.slice_data(slicing={'Time':flap.Intervals(background_time_range[0],
                                                                        background_time_range[1])},
                                         summing={'Time':'Mean'})
        amplitude=amplitude*background.data.max()

    #Spatial positions
    coeff_r=np.asarray([3.75,0.,1402.8097])/1000. #The coordinates are in meters
    coeff_z=np.asarray([0.,3.75,70.544312])/1000. #The coordinates are in meters

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
        ky=np.pi/poloidal_size
        omega=poloidal_velocity*ky
        phi0=0
        for i_frames in range(n_time):
            cur_time=i_frames * sampling_time
            for j_vertical in range(80):
                for k_radial in range(64):
                    x=r_coordinates[k_radial,j_vertical]
                    y=z_coordinates[k_radial,j_vertical]
                    A=1/np.sqrt(2*np.pi*radial_size)*np.exp(-0.5*(np.abs(x-(x0+radial_velocity*cur_time))/(radial_size/2.355))**2)
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
                                            mode=flap.CoordinateMode(equidistant=True),
                                            start=coeff_r[2],
                                            step=coeff_r[0],
                                            shape=[],
                                            dimension_list=[1]
                                            )))

    coord[5]=(copy.deepcopy(flap.Coordinate(name='Device z',
                                            unit='m',
                                            mode=flap.CoordinateMode(equidistant=True),
                                            start=coeff_z[2],
                                            step=coeff_z[1],
                                            shape=[],
                                            dimension_list=[2]
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


def generate_displaced_gaussian(exp_id=0,
                                displacement=[0,0],
                                size=[10,10],
                                size_velocity=[0,0],
                                r0=[32,40],
                                frame_size=[64,80],
                                sampling_time=2.5e-6,
                                rotation_frequency=None,
                                angular_velocity=None,
                                angle_per_frame=0.,
                                circular=False,
                                amplitude=1,
                                output_name=None,
                                test=False,
                                add_background=False,
                                background_time_range=[0.31,0.32],
                                cutoff=100,
                                n_frames=3,
                                use_image_instead=False,
                                convert_to_ellipse=False,
                                noise_level=None,
                                ):
    try:
        if len(displacement) != 2:
            raise TypeError('The displacement should be a two element vector.')
    except:
        raise TypeError('The displacement should be a two element vector.')

    #Spatial positions
    coeff_r=np.asarray([3.75,0.,1402.8097])/1000. #The coordinates are in meters
    coeff_z=np.asarray([0.,3.75,70.544312])/1000. #The coordinates are in meters

    data_arr=np.zeros([n_frames,frame_size[0],frame_size[1]])
    background=np.zeros([frame_size[0],frame_size[1]])
    if add_background:
        background=flap.get_data('NSTX_GPI',
                                 exp_id=139901,
                                 name='',
                                 object_name='GPI_RAW')
        background=background.slice_data(slicing={'Time':flap.Intervals(background_time_range[0],
                                                                        background_time_range[1])},
                                         summing={'Time':'Mean'}).data
        amplitude=amplitude*background.max()

    if rotation_frequency is not None:
        angle_per_frame=2*np.pi*rotation_frequency*sampling_time

    if angular_velocity is not None:
        angle_per_frame=angular_velocity*sampling_time*180./np.pi

    size_arg_x=size[0]/(2*np.sqrt(2*np.log(2)))
    size_arg_y=size[1]/(2*np.sqrt(2*np.log(2)))

    for i_frames in range(n_frames):

        rot_arg=-angle_per_frame*i_frames/180.*np.pi


        a=(np.cos(rot_arg)/size_arg_x)**2+\
          (np.sin(rot_arg)/size_arg_y)**2
        b=-0.5*np.sin(2*rot_arg)/size_arg_x**2+\
           0.5*np.sin(2*rot_arg)/size_arg_y**2
        c=(np.sin(rot_arg)/size_arg_x)**2+\
          (np.cos(rot_arg)/size_arg_y)**2

        x0=r0[0]+displacement[0]*i_frames
        y0=r0[1]+displacement[1]*i_frames

        frame=np.zeros([frame_size[0],
                        frame_size[1]])

        for j_vertical in range(frame_size[1]):
            for k_radial in range(frame_size[0]):
                x=k_radial
                y=j_vertical
                if (x > x0+size[0]*cutoff or
                    x < x0-size[0]*cutoff or
                    y > y0+size[1]*cutoff or
                    y < y0-size[1]*cutoff):
                    frame[k_radial,j_vertical]=background[k_radial,j_vertical]
                else:
                    frame[k_radial,j_vertical]=(amplitude*np.exp(-0.5*(a*(x-x0)**2 +
                                                                     2*b*(x-x0)*(y-y0) +
                                                                       c*(y-y0)**2))
                                                       +background[k_radial,j_vertical])
        size_arg_x *= (1+size_velocity[0])
        size_arg_y *= (1+size_velocity[1])
#        frame = rescale(frame, (1+size_velocity[1])**i_frames)[0:frame.shape[0],0:frame.shape[1]]
 #       frame = img_as_float(frame)
        if convert_to_ellipse:
            half_int=(np.max(frame)+np.min(frame))/2
            ind=np.where(frame>half_int)
            frame[ind]=255.
            ind=np.where(frame<=half_int)
            frame[ind]=0.
        data_arr[i_frames,:,:]+=frame

        if noise_level is not None:
            data_arr[i_frames,:,:] *= (((np.random.rand(frame_size[0],frame_size[1])-0.5)*2)*noise_level+1)

    if use_image_instead:
        image_path='/Users/mlampert/work/NSTX_workspace/horse.png'
        image = imread(image_path)[:,:,0]
        image = img_as_float(image)
        frame_size=[image.shape[0],image.shape[1]]
        data_arr=np.zeros([n_frames,frame_size[0],frame_size[1]])

        rotate_1 = rescale(rotate(image, angle_per_frame), 1+size_velocity[0])[0:image.shape[0],0:image.shape[1]]
        rotate_1 = img_as_float(rotate_1)

        rotate_2 = rescale(rotate(rotate_1, angle_per_frame), 1+size_velocity[0])[0:image.shape[0],0:image.shape[1]]
        rotate_2 = img_as_float(rotate_2)

        data_arr[0,:,:]=image
        data_arr[1,:,:]=rotate_1
        data_arr[2,:,:]=rotate_2


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
                                            mode=flap.CoordinateMode(equidistant=True),
                                            start=coeff_r[2],
                                            step=coeff_r[0],
                                            shape=[],
                                            dimension_list=[1]
                                            )))

    coord[5]=(copy.deepcopy(flap.Coordinate(name='Device z',
                                            unit='m',
                                            mode=flap.CoordinateMode(equidistant=True),
                                            start=coeff_z[2],
                                            step=coeff_z[1],
                                            shape=[],
                                            dimension_list=[2]
                                            )))

    _options={}
    _options["Trigger time [s]"]=0.
    _options["FPS"]=1/sampling_time
    _options["Sample time [s]"]=sampling_time
    _options["Exposure time [s]"]=2.1e-6
    _options["X size"]=frame_size[0]
    _options["Y size"]=frame_size[1]
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


def generate_displaced_random_noise(exp_id=0,
                                    displacement=[0,0],
                                    frame_size=[64,80],
                                    sampling_time=2.5e-6,
                                    circular=False,
                                    amplitude_range=[0,4095],
                                    output_name=None,
                                    test=False,
                                    n_frame=3):

    data_arr=np.zeros([n_frame,frame_size[0],frame_size[1]])
    data_arr[0,:,:]=(np.random.rand(frame_size[0],frame_size[1])*(amplitude_range[1]-amplitude_range[0])+amplitude_range[0]).astype(int)

    for i in range(1,n_frame):

        data_arr[i,:,:]=(np.random.rand(frame_size[0],frame_size[1])*(amplitude_range[1]-amplitude_range[0])+amplitude_range[0]).astype(int)
        if circular:
            data_arr[i,:,:]=np.roll(data_arr[i-1,:,:], displacement[0],axis=1)
            data_arr[i,:,:]=np.roll(data_arr[i,:,:], displacement[1],axis=2)
        else:
            if displacement[0] < 0 and displacement[1] < 0:
                data_arr[i,:frame_size[0]-abs(displacement[0]),:frame_size[1]-abs(displacement[1])] = data_arr[i-1,abs(displacement[0]):,abs(displacement[1]):]

            elif displacement[0] >= 0 and displacement[1] < 0:
                data_arr[i,displacement[0]:,:frame_size[1]-abs(displacement[1])] = data_arr[i-1,:frame_size[0]-displacement[0],abs(displacement[1]):]

            elif displacement[0] < 0 and displacement[1] >= 0:
                data_arr[i,:frame_size[0]-abs(displacement[0]),displacement[1]:] = data_arr[i-1,abs(displacement[0]):,:frame_size[1]-displacement[1]]

            elif displacement[0] >= 0 and displacement[1] >= 0:
                data_arr[i,displacement[0]:,displacement[1]:] = data_arr[i-1,:frame_size[0]-displacement[0],:frame_size[1]-displacement[1]]


    coord = [None]*4
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
