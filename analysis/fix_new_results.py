#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 14:31:29 2020

@author: mlampert
"""

import pandas
import time

import numpy as np
import pickle

import os
import flap
import flap_nstx

from flap_nstx.analysis import calculate_nstx_gpi_avg_frame_velocity, calculate_nstx_gpi_smooth_velocity, flap_nstx_thomson_data

thisdir = os.path.dirname(os.path.realpath(__file__))
fn = os.path.join(thisdir,"flap_nstx.cfg")
flap.config.read(file_name=fn)
flap_nstx.register()

def fix_new_results():
    database_file='/Users/mlampert/work/NSTX_workspace/db/ELM_findings_mlampert_velocity_good.csv'
    db=pandas.read_csv(database_file, index_col=0)
    elm_index=list(db.index)
    for index_elm in range(len(elm_index)):
        #preprocess velocity results, tackle with np.nan and outliers
        shot=int(db.loc[elm_index[index_elm]]['Shot'])
        #define ELM time for all the cases
        elm_time=db.loc[elm_index[index_elm]]['ELM time']/1000.
        wd=flap.config.get_all_section('Module NSTX_GPI')['Working directory']
        filename_wo_str=flap_nstx.analysis.filename(exp_id=shot,
                                                         working_directory=wd+'/processed_data',
                                                         time_range=[elm_time-2e-3,elm_time+2e-3],
                                                         comment='ccf_velocity_pfit_o1_fst_0.0_nv',
                                                         extension='pickle')
        filename_with_str=flap_nstx.analysis.filename(exp_id=shot,
                                                         working_directory=wd+'/processed_data',
                                                         time_range=[elm_time-2e-3,elm_time+2e-3],
                                                         comment='ccf_velocity_pfit_o1_ct_0.6_fst_0.0_ns_nv',
                                                         extension='pickle')
        try:
        #if True:
            velocity_wo_str=pickle.load(open(filename_wo_str,'rb'))
            velocity_with_str=pickle.load(open(filename_with_str,'rb'))
            for key in velocity_wo_str.keys():
                if key not in ['Velocity ccf', 'Frame similarity', 'Correlation max', 'GPI Dalpha']:
                    velocity_wo_str[key]=velocity_with_str[key]
            velocity_wo_str['GPI Dalpha']=velocity_wo_str['GPI Dalpha'][0].data
            pickle.dump(velocity_wo_str, open(filename_wo_str.replace('_nv', '_ns_nv'), 'wb'))
            
            velocity_with_str['GPI Dalpha']=velocity_wo_str['GPI Dalpha']
            #print(velocity_with_str['GPI Dalpha'])
            pickle.dump(velocity_with_str, open(filename_with_str, 'wb'))
        except:
            print('Lofasz')