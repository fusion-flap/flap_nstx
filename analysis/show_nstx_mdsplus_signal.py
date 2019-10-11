#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 14:26:41 2019

@author: mlampert
"""

import os

import flap
import flap_nstx
import flap_mdsplus
import matplotlib.pyplot as plt


flap_nstx.register()
flap_mdsplus.register('NSTX_MDSPlus')

def show_nstx_mdsplus_signal(exp_id=None, time_range=None, tree=None, node=None,
                             new_plot=False, yrange=None, smooth=None):
    
    if (exp_id is None and time_range is None):
        print('The correct way to call the code is the following:\n')
        print('show_nstx_mdsplus_signal(exp_id=141918, time_range=[250.,260.], tree=\'wf\', node=\'\\IP\'')
        print('INPUTs: \t Description: \t\t\t\t Type: \t\t\t Default values: \n')
        print('exp_id: \t The shot number. \t\t\t int \t\t\t Default: None')
        print('time_range: \t The time range in ms. \t\t\t [float,float] \t\t Default: None')
        print('tree: \t\t Name of the MDSplus tree. \t\t string \t\t Default: None')
        print('node: \t\t Name of the MDSplus tree node. \t string \t\t Default: None')
        return
    
    if (tree is None) or (node is None):
        raise ValueError('The MDSplus tree and the node both needs to be given as an input.')
        
    if time_range is None:
        print('time_range is None, the entire shot is plot.')
    else:    
        if (type(time_range) is not list and len(time_range) != 2):
            raise TypeError('time_range needs to be a list with two elements.')
        time_range=[time_range[0]/1000., time_range[1]/1000.]    

    plot_options={'X range': time_range,
                  'Y range': yrange}

    object_name=str(exp_id)+'_mds_t'+tree+'_n'+node
    d=flap.get_data('NSTX_MDSPlus',
                    name='\\'+tree+'::'+node,
                    exp_id=exp_id,
                    object_name=object_name
                    )
    if (len(d.data.shape) > 1):
        raise ValueError('The read data are not a single channel time series. Returning...')
    if new_plot:
        plt.figure()
    else:
        plt.cla()
    flap.plot(object_name, options=plot_options)
    #flap.plot(object_name)
    
    
thisdir = os.path.dirname(os.path.realpath(__file__))
fn = os.path.join(thisdir,"flap_nstx.cfg")
flap.config.read(file_name=fn)   