#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 02:05:09 2021

@author: mlampert
"""

from flap_nstx.analysis import flap_nstx_thomson_data
import os
import numpy as np

def prepare_profiles_for_renate(exp_id=139901,
                                time=0.325):
    d_temp=flap_nstx_thomson_data(exp_id=exp_id,
                                  force_mdsplus=True,
                                  temperature=True,
                                  density=False,
                                  add_flux_coordinates=True,
                                  output_name='Temperature',
                                  test=False)
    
    d_dens=flap_nstx_thomson_data(exp_id=exp_id,
                                  force_mdsplus=True,
                                  temperature=True,
                                  density=False,
                                  add_flux_coordinates=True,
                                  output_name='Temperature',
                                  test=False)
    
    
    d_temp_slice=d_temp.slice_data(slicing={'Time':time})
    d_dens_slice=d_dens.slice_data(slicing={'Time':time})
    print(d_temp_slice)
    psi_norm=d_dens_slice.coordinate('Flux r')[0]
    
    Z_eff=np.zeros(len(d_temp_slice.data))
    Z_eff[:]=3.08
    
    q_eff=np.zeros(len(d_temp_slice.data))
    q_eff[:]=6
    
    text_file=open('nt_NSTXU'+str(exp_id)+'_'+str(int(time*1e3))+'.txt', 'wt')
    for i in range(len(psi_norm)):
        text_file.write(str(psi_norm[i])+'\t\t'+str(d_temp_slice.data[i])+'\t\t'+str(d_dens_slice.data[i])+'\t\t'+str(Z_eff[i])+'\t\t'+str(q_eff[i])+'\n')
    text_file.close()