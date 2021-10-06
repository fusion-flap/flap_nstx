#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 16:45:03 2019

@author: mlampert
"""

import os
import copy

import flap
import flap_nstx
#from flap_nstx.analysis.nstx_gpi_tools import calculate_nstx_gpi_norm_coeff
#from flap_nstx.analysis.nstx_gpi_tools import calculate_nstx_gpi_reference

import flap_mdsplus


flap_nstx.register()
flap_mdsplus.register('NSTX_MDSPlus')

thisdir = os.path.dirname(os.path.realpath(__file__))
fn = os.path.join(thisdir,"../flap_nstx.cfg")
flap.config.read(file_name=fn)

import matplotlib.style as pltstyle
import matplotlib.pyplot as plt
import numpy as np

publication=False

if publication:

    plt.rcParams['lines.linewidth'] = 4
    plt.rcParams['axes.linewidth'] = 4
    plt.rcParams['axes.labelsize'] = 28  # 28 for paper 35 for others
    plt.rcParams['axes.titlesize'] = 28
    plt.rcParams['xtick.labelsize'] = 30
    plt.rcParams['xtick.major.size'] = 10
    plt.rcParams['xtick.major.width'] = 4
    plt.rcParams['xtick.major.size'] = 6
    plt.rcParams['xtick.minor.width'] = 2
    plt.rcParams['xtick.minor.size'] = 4
    plt.rcParams['ytick.labelsize'] = 30
    plt.rcParams['ytick.major.width'] = 4
    plt.rcParams['ytick.major.size'] = 6
    plt.rcParams['ytick.minor.width'] = 2
    plt.rcParams['ytick.minor.size'] = 4
    plt.rcParams['legend.fontsize'] = 28

else:
    pltstyle.use('default')

def export_gpi_data_to_paraview(exp_id=None,
                                time_range=None,
                                filename=None):

    if filename is None:
        filename='GPI_FOR_PARAVIEW_'+str(exp_id)+'_'+str(time_range[0])+'_'+str(time_range[1])+'.csv'
    
    d=flap.get_data('NSTX_GPI', exp_id=exp_id, name='', object_name='GPI')
    time=d.coordinate('Time')[0]
    ind=np.where(np.logical_and(time[:,0,0]>=time_range[0], time[:,0,0]<=time_range[1]))
    x=d.coordinate('Device R')[0][ind,:,:].flatten()
    y=d.coordinate('Device z')[0][ind,:,:].flatten()
    t=time[ind,:,:].flatten()
    data=d.data[ind,:,:].flatten()
    np.savetxt(filename, np.asarray([[x],[y],[10000*t],[data]])[:,0,:].T, delimiter=",", header='x [m], y [m], t [ms], data [a.u.]')