#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 02:02:38 2020

@author: mlampert
"""
#One time run code
def fix_non_norm_vel_files():
    database_file='/Users/mlampert/work/NSTX_workspace/ELM_findings_mlampert_velocity_good.csv'
    db=pandas.read_csv(database_file, index_col=0)
    elm_index=list(db.index)
    for index_elm in range(len(elm_index)):
        #preprocess velocity results, tackle with np.nan and outliers
        shot=int(db.loc[elm_index[index_elm]]['Shot'])
        #define ELM time for all the cases
        elm_time=db.loc[elm_index[index_elm]]['ELM time']/1000.
        status=db.loc[elm_index[index_elm]]['OK/NOT OK']
        if status != 'NO':
            wd=flap.config.get_all_section('Module NSTX_GPI')['Working directory']
            filename_norm=flap_nstx.analysis.filename(exp_id=shot,
                                                      working_directory=wd,
                                                      time_range=[elm_time-2e-3,elm_time+2e-3],
                                                      comment='ccf_velocity_pfit_o1_ct_0.6_fst_0.0_ns_nv',
                                                      extension='pickle')
            
            filename_non_norm=wd+'/'+db.loc[elm_index[index_elm]]['Filename']+'.pickle'
            velocity_results_non_norm=pickle.load(open(filename_non_norm, 'rb'))
            velocity_results_norm=pickle.load(open(filename_norm, 'rb'))
            velocity_results_non_norm['GPI Dalpha']=velocity_results_norm['GPI Dalpha']
            pickle.dump(velocity_results_non_norm,open(filename_non_norm,'wb'))