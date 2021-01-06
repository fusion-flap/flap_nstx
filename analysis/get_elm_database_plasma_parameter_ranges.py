#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 19:08:46 2020

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


from matplotlib.backends.backend_pdf import PdfPages
thisdir = os.path.dirname(os.path.realpath(__file__))
fn = os.path.join(thisdir,'flap_nstx.cfg')
flap.config.read(file_name=fn)
flap_nstx.register()

def get_elm_database_plasma_parameter_ranges():
    database_file='/Users/mlampert/work/NSTX_workspace/db/ELM_findings_mlampert_velocity_good.csv'
    db=pandas.read_csv(database_file, index_col=0)
    elm_index=list(db.index)

    density=[]
    for elm_ind in elm_index:
            
        elm_time=db.loc[elm_ind]['ELM time']/1000.
        shot=int(db.loc[elm_ind]['Shot'])
        
        d=flap_nstx_thomson_data(exp_id=shot, density=True, add_flux_coordinates=True, output_name='DENSITY')
        elm_index=np.argmin(np.abs(d.coordinate('Time')[0][1,:]-elm_time))
        
        #goodind=np.where(np.logical_and(d.coordinate('Flux r')[0][:,elm_index] < 1.0, d.coordinate('Flux r')[0][:,elm_index] > 0))
        dR = (d.coordinate('Device R')[0][:,:]-np.insert(d.coordinate('Device R')[0][0:-1,:],0,0,axis=0))
        LID=np.trapz(d.data[:,:], d.coordinate('Device R')[0][:,:], axis=0)/(np.max(d.coordinate('Device R')[0][:,:],axis=0)-np.min(d.coordinate('Device R')[0][:,:],axis=0))
        #LID=np.sum(((d.data[:,:])[:,:])*dR,axis=0)/np.sum(dR)
        density.append(LID[elm_index])
    print(np.sort(density))