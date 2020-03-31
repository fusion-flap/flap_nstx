#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 14:01:47 2020

@author: mlampert
"""

import os
import pandas
import copy

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pickle
import numpy as np
from scipy.signal import correlate

#Flap imports
import flap
import flap_nstx
flap_nstx.register()
import flap_mdsplus

#Setting up FLAP
flap_mdsplus.register('NSTX_MDSPlus')    
thisdir = os.path.dirname(os.path.realpath(__file__))
fn = os.path.join(thisdir,"flap_nstx.cfg")
flap.config.read(file_name=fn) 

def calculate_nstx_gpi_correlation_matrix(window_average=0.5e-3,
                                          sampling_time=2.5e-6
                                          ):
    database_file='/Users/mlampert/work/NSTX_workspace/ELM_findings_mlampert_velocity_good.csv'
    db=pandas.read_csv(database_file, index_col=0)
    elm_index=list(db.index)
    
    pearson_keys=['Velocity ccf',       #0,1
                  'Velocity str avg',   #2,3
                  'Size avg',           #4,5
                  'Position avg',       #6,7
                  'Area avg',           #8
                  'Elongation avg',     #9
                  'Angle avg']          #10
    nelm=0.
    pearson_matrix=np.zeros([11,11,2]) #[velocity ccf, velocity str svg]
    nwin=int(window_average/sampling_time)
    plt.figure()
    for index_elm in range(len(elm_index)):
        #preprocess velocity results, tackle with np.nan and outliers
        shot=int(db.loc[elm_index[index_elm]]['Shot'])
        #define ELM time for all the cases
        elm_time=db.loc[elm_index[index_elm]]['ELM time']/1000.
        wd=flap.config.get_all_section('Module NSTX_GPI')['Working directory']
        filename=wd+'/'+db.loc[elm_index[index_elm]]['Filename']+'.pickle'
        status=db.loc[elm_index[index_elm]]['OK/NOT OK']
        
        if status != 'NO':
            velocity_results=pickle.load(open(filename, 'rb'))
            time=velocity_results['Time']
            elm_time_interval_ind=np.where(np.logical_and(time >= elm_time-window_average,
                                                          time <= elm_time+window_average))
            elm_time=(time[elm_time_interval_ind])[np.argmin(velocity_results['Frame similarity'][elm_time_interval_ind])]
            elm_time_ind=np.argmin(np.abs(time-elm_time))
            ind=slice(elm_time_ind-nwin,elm_time_ind+nwin)
            for ind_first in range(11):
                for ind_second in range(11):
                    if ind_first <= 7:
                        a=velocity_results[pearson_keys[ind_first//2]][ind,np.mod(ind_first,2)]
                        ind_nan=np.isnan(a)
                        a[ind_nan]=0.
                    else:
                        a=velocity_results[pearson_keys[ind_first-4]][ind]
                    if ind_second <= 7:
                        b=velocity_results[pearson_keys[ind_second//2]][ind,np.mod(ind_second,2)]
                    else:
                        b=velocity_results[pearson_keys[ind_second-4]][ind]
                    ind_nan=np.isnan(a)
                    a[ind_nan]=0.
                    ind_nan=np.isnan(b)
                    b[ind_nan]=0.
                    a-=np.mean(a)
                    b-=np.mean(b)
                    cur_pear=np.sum(a*b)/(np.sqrt(np.sum(a**2))*np.sqrt(np.sum(b**2)))
                    if cur_pear == np.nan:
                        cur_pear=0.
                        
                    pearson_matrix[ind_first,ind_second]+=cur_pear
            nelm+=1
    pearson_matrix/=nelm
    
    data = pearson_matrix[:,:,0]
    data[10,10]=-1   
    cs=plt.matshow(data, cmap='seismic')
    plt.xticks(ticks=np.arange(11), labels=['Velocity ccf R',       #0,1
                                        'Velocity ccf z',       #0,1
                                        'Velocity str avg R',   #2,3
                                        'Velocity str avg z',   #2,3
                                        'Size avg R',           #4,5
                                        'Size avg z',           #4,5
                                        'Position avg R',       #6,7
                                        'Position avg z',       #6,7
                                        'Area avg',           #8
                                        'Elongation avg',     #9
                                        'Angle avg'], rotation='vertical')
    plt.yticks(ticks=np.arange(11), labels=['Velocity ccf R',       #0,1
                                        'Velocity ccf z',       #0,1
                                        'Velocity str avg R',   #2,3
                                        'Velocity str avg z',   #2,3
                                        'Size avg R',           #4,5
                                        'Size avg z',           #4,5
                                        'Position avg R',       #6,7
                                        'Position avg z',       #6,7
                                        'Area avg',           #8
                                        'Elongation avg',     #9
                                        'Angle avg'])
    plt.colorbar()
    plt.show()
    
    return pearson_matrix