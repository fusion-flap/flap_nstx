#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 23:06:17 2020

@author: mlampert
"""

#import numpy as np
#import matplotlib as plt
#import math
import pandas
import time

import os
import flap
import flap_nstx

from flap_nstx.analysis import calculate_nstx_gpi_frame_by_frame_velocity, calculate_nstx_gpi_smooth_velocity, calculate_nstx_gpi_angular_velocity
from flap_nstx.analysis import nstx_gpi_velocity_analysis_spatio_temporal_displacement

thisdir = os.path.dirname(os.path.realpath(__file__))
fn = os.path.join(thisdir,"flap_nstx.cfg")
flap.config.read(file_name=fn)
flap_nstx.register()

def calculate_all_nstx_gpi_avg_frame_by_frame_velocity(elm_time_window=0.6e-3):
    database_file='/Users/mlampert/work/NSTX_workspace/db/ELM_findings_mlampert_velocity_good.csv'
    db=pandas.read_csv(database_file, index_col=0)
    elm_index=(list(db.index))
    elm=0.
    failed_elms=[]
    number_of_failed_elms=0
    for index_elm in range(len(elm_index)):
        #preprocess velocity results, tackle with np.nan and outliers
        shot=int(db.loc[elm_index[index_elm]]['Shot'])
        #define ELM time for all the cases
        #elm_time=db.loc[elm_index[index_elm]]['ELM time']/1000.
        elm_time=db.loc[elm_index[index_elm]]['Time prec']
        flap.delete_data_object('*')
        print('Calculating '+str(shot)+ ' at '+str(elm_time))
        elm=elm+1
        start_time=time.time()
        status=db.loc[elm_index[index_elm]]['OK/NOT OK']
        if status != 'NO':
            #try:
            if True:
                calculate_nstx_gpi_frame_by_frame_velocity(exp_id=shot, 
                                                          time_range=[elm_time-elm_time_window,elm_time+elm_time_window], 
                                                          plot=False,
                                                          subtraction_order_for_velocity=1,
                                                          skip_structure_calculation=False,
                                                          correlation_threshold=0.,
                                                          pdf=True, 
                                                          nlevel=51, 
                                                          nocalc=False, 
                                                          filter_level=5, 
                                                          normalize_for_size=True,
                                                          normalize_for_velocity=True,
                                                          remove_interlaced_structures=False,
                                                          flap_ccf=True,
                                                          threshold_coeff=1.,
                                                          normalize_f_high=1e3, 
                                                          normalize='roundtrip', 
                                                          velocity_base='cog', 
                                                          return_results=False, 
                                                          plot_gas=True)
#            except:
#                print('Calculating '+str(shot)+ ' at '+str(elm_time)+' failed.')
#                failed_elms.append({'Shot':shot,'Time':elm_time})
#                number_of_failed_elms+=1
        one_time=time.time()-start_time
        rem_time=one_time*(len(elm_index)-index_elm)
        print('Remaining time from the calculation:'+str(rem_time/3600.)+'hours.')
        print(failed_elms,number_of_failed_elms)
            
def calculate_all_nstx_gpi_smooth_velocity():
    database_file='/Users/mlampert/work/NSTX_workspace/db/ELM_findings_mlampert_velocity_good.csv'
    db=pandas.read_csv(database_file, index_col=0)
    elm_index=list(db.index)
    elm=0.
    failed_elms=[]
    number_of_failed_elms=0
    
    for index_elm in range(len(elm_index)):
        #preprocess velocity results, tackle with np.nan and outliers
        shot=int(db.loc[elm_index[index_elm]]['Shot'])
        #define ELM time for all the cases
        elm_time=db.loc[elm_index[index_elm]]['ELM time']/1000.
        
        flap.delete_data_object('*')

        print('Calculating '+str(shot)+ ' at '+str(elm_time))
        elm=elm+1
        start_time=time.time()
        try:
            elm_time_range=[elm_time-2e-3,elm_time+2e-3]
            calculate_nstx_gpi_smooth_velocity(exp_id=shot, 
                                               time_range=elm_time_range,
                                               radial=True,
                                               time_res=100e-6,
                                               plot=False,
                                               save_data=True)
    
            calculate_nstx_gpi_smooth_velocity(exp_id=shot, 
                                               time_range=elm_time_range,
                                               poloidal=True,
                                               time_res=100e-6,
                                               plot=False,
                                               save_data=True)

        except:
            print('Calculating '+str(shot)+ ' at '+str(elm_time)+' failed.')
            failed_elms.append({'Shot':shot,'Time':elm_time})
            number_of_failed_elms+=1
        print(failed_elms,number_of_failed_elms)
        finish_time=time.time()
        rem_time=(finish_time-start_time)*(len(elm_index)-index_elm+1)
        print('Remaining time from the calculation:'+str(rem_time/3600.)+'hours.')

def calculate_all_nstx_gpi_sz_velocity():
    database_file='/Users/mlampert/work/NSTX_workspace/db/ELM_findings_mlampert_velocity_good.csv'
    db=pandas.read_csv(database_file, index_col=0)
    elm_index=list(db.index)
    elm=0.
    failed_elms=[]
    number_of_failed_elms=0
    
    for index_elm in range(len(elm_index)):
        #preprocess velocity results, tackle with np.nan and outliers
        shot=int(db.loc[elm_index[index_elm]]['Shot'])
        #define ELM time for all the cases
        elm_time=db.loc[elm_index[index_elm]]['ELM time']/1000.
        
        flap.delete_data_object('*')

        print('Calculating '+str(shot)+ ' at '+str(elm_time))
        elm=elm+1
        start_time=time.time()
        #try:
        if True:
            elm_time_range=[elm_time-1e-3,elm_time+1e-3]
            nstx_gpi_velocity_analysis_spatio_temporal_displacement(exp_id=shot, 
                                                                    time_range=elm_time_range, 
                                                                    x_range=[10,15], 
                                                                    y_range=[40,45], 
                                                                    plot=False, 
                                                                    pdf=False, 
                                                                    nocalc=False)

#        except:
#            print('Calculating '+str(shot)+ ' at '+str(elm_time)+' failed.')
#            failed_elms.append({'Shot':shot,'Time':elm_time})
#            number_of_failed_elms+=1
            
        print(failed_elms,number_of_failed_elms)
        finish_time=time.time()
        rem_time=(finish_time-start_time)*(len(elm_index)-index_elm+1)
        print('Remaining time from the calculation:'+str(rem_time/3600.)+'hours.')
        
def calculate_all_nstx_gpi_angular_velocity(elm_time_window=0.6e-3):
    database_file='/Users/mlampert/work/NSTX_workspace/db/ELM_findings_mlampert_velocity_good.csv'
    db=pandas.read_csv(database_file, index_col=0)
    elm_index=(list(db.index))
    elm=0.
    failed_elms=[]
    number_of_failed_elms=0
    for index_elm in range(len(elm_index)):
        #preprocess velocity results, tackle with np.nan and outliers
        shot=int(db.loc[elm_index[index_elm]]['Shot'])
        #define ELM time for all the cases
        elm_time=db.loc[elm_index[index_elm]]['Time prec']
        flap.delete_data_object('*')
        print('Calculating '+str(shot)+ ' at '+str(elm_time))
        elm=elm+1
        start_time=time.time()
        status=db.loc[elm_index[index_elm]]['OK/NOT OK']
        if status != 'NO':
            #try:
            if True:
                calculate_nstx_gpi_angular_velocity(exp_id=shot, 
                                                    time_range=[elm_time-elm_time_window,elm_time+elm_time_window],
                                                    subtraction_order_for_velocity=4,
                                                    nocalc=False, 
                                                    correlation_threshold=0., 
                                                    plot=False, 
                                                    pdf=True)
#            except:
#                print('Calculating '+str(shot)+ ' at '+str(elm_time)+' failed.')
#                failed_elms.append({'Shot':shot,'Time':elm_time})
#                number_of_failed_elms+=1
        one_time=time.time()-start_time
        rem_time=one_time*(len(elm_index)-index_elm)
        print('Remaining time from the calculation:'+str(rem_time/3600.)+'hours.')
        print(failed_elms,number_of_failed_elms)