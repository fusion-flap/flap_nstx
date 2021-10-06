#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 12:06:41 2021

@author: mlampert
"""
# -*- coding: utf-8 -*-

"""
Created on Sat Jul 24 13:22:00 2019

@author: Lampert

This is the flap module for NSTX GPI diagnostic
Needs pims package installed
e.g.: conda install -c conda-forge pims

"""

#FLAP imports
import flap

if (flap.VERBOSE):
    print("Importing flap_nstx")
from flap_nstx import get_data_gpi, add_coordinate_gpi  
from flap_nstx import get_data_thomson, add_coordinate_thomson

def register(data_source='NSTX_GPI'):
    if (flap.VERBOSE):
        print("Importing flap_nstx for "+data_source)
    if data_source == 'NSTX_GPI':
        flap.register_data_source('NSTX_GPI', get_data_func=get_data_gpi, add_coord_func=add_coordinate_gpi)
    if data_source == 'NSTX_THOMSON':
        pass
        flap.register_data_source('NSTX_THOMSON', get_data_func=get_data_thomson, add_coord_func=add_coordinate_thomson)