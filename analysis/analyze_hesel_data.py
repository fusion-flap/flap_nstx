#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 16:23:01 2021

@author: mlampert
"""

from flap_nstx.analysis import generate_displaced_gaussian, generate_displaced_random_noise, calculate_nstx_gpi_avg_frame_velocity

def analyze_hesel_data():
    d=read_flap_hesel_data(output_name='HESEL_DATA')
    calculate_nstx_gpi_avg_frame_velocity(exp_id=0, 
                                          data_object='HESEL_DATA', 
                                          normalize_for_velocity=True,
                                          normalize_for_size=False, 
                                          subtraction_order_for_velocity=4, 
                                          skip_structure_calculation=True, 
                                          pdf=True, 
                                          plot_gas=False, 
                                          overplot_str_vel=False, 
                                          plot_scatter=False)