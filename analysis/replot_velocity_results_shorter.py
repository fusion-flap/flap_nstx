#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 13:53:41 2020

@author: mlampert
"""

import flap
import pandas
import flap_nstx

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.lines as mlines

import numpy as np

import os
import math
flap_nstx.register()
database_file='/Users/mlampert/work/NSTX_workspace/ELM_findings_mlampert_velocity_good.csv'
db=pandas.read_csv(database_file, index_col=0)
elm_index=list(db.index)
elm=0.
failed_elms=[]
number_of_failed_elms=0
for index in elm_index:

    flap.delete_data_object('*')
    shot=int(db.loc[index]['Shot'])
    elm_time=db.loc[index]['ELM time']/1000.
    wd=flap.config.get_all_section('Module NSTX_GPI')['Working directory']
    filename=wd+'/'+db.loc[index]['Filename']
    status=db.loc[index]['OK/NOT OK']
    try:
        pickle.load(open(filename+'.pickle', 'rb'))
    except:
        print(filename)
    if status != 'NO' and shot == 141321:
        print('Calculating '+str(shot)+ ' at '+str(elm_time))
#        try:
        calculate_nstx_gpi_avg_frame_velocity(exp_id=shot, 
                                              #time_range=[elm_time-0.5e-3,elm_time+0.5e-3], 
                                              plot=False,
                                              subtraction_order_for_velocity=1, 
                                              pdf=True, 
                                              nlevel=51, 
                                              nocalc=True, 
                                              filter_level=3, 
                                              normalize_for_size=True,
                                              normalize_for_velocity=False,
                                              threshold_coeff=1.,
                                              normalize_f_high=1e3, 
                                              normalize='roundtrip', 
                                              velocity_base='cog', 
                                              return_results=False,
                                              filename=filename,
                                              plot_gas=True)
#        except:
#            print('Calculating '+str(shot)+ ' at '+str(elm_time)+' failed.')
#            failed_elms.append({'Shot':shot,'Time':elm_time})
#            number_of_failed_elms+=1
print(failed_elms,number_of_failed_elms)