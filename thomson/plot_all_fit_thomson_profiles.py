#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 21:47:03 2020

@author: mlampert
"""
import os
import copy

import pandas
import time

import numpy as np
import pickle

import flap
import flap_nstx

thisdir = os.path.dirname(os.path.realpath(__file__))
fn = os.path.join(thisdir,"../flap_nstx.cfg")
flap.config.read(file_name=fn)
flap_nstx.register()

from flap_nstx.thomson import get_elms_with_thomson_profile

from flap_nstx.gpi import calculate_nstx_gpi_frame_by_frame_velocity, calculate_nstx_gpi_smooth_velocity
from flap_nstx.thomson import get_nstx_thomson_gradient, get_fit_nstx_thomson_profiles, get_elms_with_thomson_profile

from matplotlib.backends.backend_pdf import PdfPages




"""
NOT WORKING, SHOULD BE DEVELOPED
"""

    
    
    

def plot_all_fit_thomson_profiles(thomson_time_window=None): #One time run code, no need to have inputs
    
    def tanh_fit_function(r, b_height, b_sol, b_pos, b_width):
        def tanh(x):
            return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
        return (b_height-b_sol)/2*(tanh((b_pos-r)/(2*b_width))+1)+b_sol
    
    elms_with_thomson_before=get_elms_with_thomson_profile(before=True,
                                                           time_window=thomson_time_window)
    elms_with_thomson_after=get_elms_with_thomson_profile(after=True,
                                                          time_window=thomson_time_window)
        
    database_file='/Users/mlampert/work/NSTX_workspace/ELM_findings_mlampert_velocity_good.csv'
    db=pandas.read_csv(database_file, index_col=0)
    elm_index=list(db.index)
    
    elms_with_thomson_before=get_elms_with_thomson_profile(before=True,
                                                    time_window=thomson_time_window)
    elms_with_thomson_after=get_elms_with_thomson_profile(after=True,
                                                    time_window=thomson_time_window)
        
    database_file='/Users/mlampert/work/NSTX_workspace/ELM_findings_mlampert_velocity_good.csv'
    db=pandas.read_csv(database_file, index_col=0)
    elm_index=list(db.index)
    nwin=int(elm_duration/2.5e-6)
    
    for i in range(2):  #Doing the before and after calculation
        if i == 0:
            elms_with_thomson = elms_with_thomson_before
            scaling_db_file = scaling_db_file_before
        else:
            elms_with_thomson = elms_with_thomson_after
            scaling_db_file=scaling_db_file_after
            
        gradient={'Pressure':[],
                  'Density':[],
                  'Temperature':[]}

        gpi_results_max=copy.deepcopy(gpi_results_avg)
        for elm in elms_with_thomson:
    
            elm_time=db.loc[elm_index[elm['index_elm']]]['ELM time']/1000.
            shot=int(db.loc[elm_index[elm['index_elm']]]['Shot'])
            
            graddata=get_fit_nstx_thomson_profiles(exp_id=elm['shot'],
                                                   temperature=True,
                                                   spline_data=False, 
                                                   flux_coordinates=True, 
                                                   flux_range=[0.5,1.4],
                                                   return_parameters=True)