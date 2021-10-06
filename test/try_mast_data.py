#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 14:33:43 2021

@author: mlampert
"""
import numpy as np
import sys
sys.path.append('FILEPATH/flap')
sys.path.append('FILEPATH/reading_MAST_data')

import copy

import flap
import reading_MAST_data as rmd
from scipy.interpolate import griddata
import flap_nstx
from flap_nstx.analysis import calculate_nstx_gpi_frame_by_frame_velocity
from flap_nstx.analysis import calculate_sde_velocity
flap_nstx.register()

filename = '/Users/mlampert/work/NSTX_workspace/xbt029504.nc'
sample_grid = [4, 8]
new_grid_size = [40, 80]
time_bounds = [0.321, 0.322]

channels_data, time_ax, chn_as = rmd.hdf5_to_flap(filename, calib_options={'Choose calibration file':False})
timeslice = channels_data.slice_data(slicing={'Time': flap.Intervals(0.321, 0.322)}) #(0.321, 0.322)
timeslice_data = np.array(timeslice.data)
timeslice_data -= np.mean(timeslice_data)

def meshgrid_for_test(grid_lim_list, sample_grid_list):
    x = np.linspace(0.5, sample_grid_list[0]-0.5, grid_lim_list[0])
    y = np.linspace(0.5, sample_grid_list[1]-0.5, grid_lim_list[1])
    xx, yy = np.meshgrid(x, y, sparse=True)
    mesh_coords = [xx, yy]
    d1_coords = [x, y]
    return mesh_coords, d1_coords

def interpolate_data(input_data):
    global new_grid_size, sample_grid
    xx, yy = meshgrid_for_test(new_grid_size, sample_grid)[0]
    
    def gen_sample_coordinates():
        x, y = meshgrid_for_test(sample_grid, sample_grid)[1]
        coord_list = []
        for i in x:
            for j in y:
                coord_list.append(np.array([i, j]))
        return np.array(coord_list)
    
    coord_list = gen_sample_coordinates()
    interpolated_data = []
    for i in range(len(input_data[0][0])):
        grid_inter = np.array(griddata(coord_list, input_data[:, :, i].flatten(), (xx, yy), method='cubic'))
        interpolated_data.append(grid_inter)
    return np.array(interpolated_data)


interpolated_timeslice = interpolate_data(timeslice_data)

x_radial = np.linspace(0, sample_grid[0], new_grid_size[0])
y_poloidal = np.linspace(0, sample_grid[1], new_grid_size[1])

flap_time_coordinates = timeslice.get_coordinate_object('Time')

coord=[None] * 6
coord[0]=flap_time_coordinates
coord[0].dimension_list=[0]
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
                                        unit='cm', 
                                        values=x_radial, 
                                        dimension_list=[1], 
                                        shape=np.shape(x_radial),
                                        mode=flap.CoordinateMode(equidistant=False))))

coord[5]=(copy.deepcopy(flap.Coordinate(name='Device z', 
                                        unit='cm', 
                                        values=y_poloidal, 
                                        dimension_list=[2], 
                                        shape=np.shape(y_poloidal),
                                        mode=flap.CoordinateMode(equidistant=False))))

d = flap.DataObject(data_array=interpolated_timeslice,
                    data_unit=flap.Unit(name='Signal',unit='Digit'),
                    coordinates=coord,
                    exp_id=29504,
                    data_title='MAST BES DATA',
                    #info={'Options':_options},
                    data_source="NSTX_GPI")

flap.add_data_object(d, 'MAST_TRIAL')
print('calculating')
calculate_nstx_gpi_frame_by_frame_velocity(data_object=d, 
                                            normalize_for_velocity=False, 
                                            normalize_for_size=False, 
                                            correlation_threshold=0.0, 
                                            skip_structure_calculation=True, 
                                            plot=True, 
                                            pdf=True, 
                                            correct_acf_peak=False,
                                            overplot_str_vel=False, 
                                            overplot_average=False, 
                                            plot_scatter=False, 
                                            save_results=False)
# calculate_sde_velocity(d, 
#                        time_range=[0.321,0.322], 
#                        normalize_type='subtraction', 
#                        normalize_f_high=1e3, 
#                        filename='mast_trial', 
#                        return_pixel_displacement=True)