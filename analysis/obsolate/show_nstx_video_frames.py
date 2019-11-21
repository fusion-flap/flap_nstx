#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 11:17:42 2019

@author: mlampert
"""


import os

import flap
import flap_nstx
import flap_mdsplus
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

flap_nstx.register()
flap_mdsplus.register('NSTX_MDSPlus')

def show_nstx_video_frames(exp_id=None, 
                           time_range=None,
                           n_frame=20,
                           logz=False,
                           z_range=[0,512],
                           plot_filtered=False, 
                           cache_data=False, 
                           plot_efit=False, 
                           flux_coordinates=False,
                           device_coordinates=False,
                           new_plot=True,
                           save_video=False,
                           video_saving_only=False,
                           colormap='gist_ncar',
                           ):
    
    if time_range is None:
        print('time_range is None, the entire shot is plotted.')
        slicing_range=None
    else:    
        if (type(time_range) is not list and len(time_range) != 2):
            raise TypeError('time_range needs to be a list with two elements.')
        #time_range=[time_range[0]/1000., time_range[1]/1000.] 
        slicing_range={'Time':flap.Intervals(time_range[0],time_range[1])}
        
    if not cache_data: #This needs to be enhanced to actually cache the data no matter what
        flap.delete_data_object('*')
    if exp_id is not None:
        print("\n------- Reading NSTX GPI data --------")
        if cache_data:
            try:
                d=flap.get_data_object_ref(exp_id=exp_id,object_name='GPI')
            except:
                print('Data is not cached, it needs to be read.')
                d=flap.get_data('NSTX_GPI',exp_id=exp_id,name='',object_name='GPI')
        else:
            d=flap.get_data('NSTX_GPI',exp_id=exp_id,name='',object_name='GPI')
        object_name='GPI'
    else:
        raise ValueError('The experiment ID needs to be set.')
    if plot_filtered:
        print("**** Filtering GPI")
        object_name='GPI_FILTERED'
        try:
            flap.get_data_object_ref(object_name, exp_id=exp_id)
        except:
            flap.filter_data('GPI',exp_id=exp_id,output_name='GPI_FILTERED',coordinate='Time',
                             options={'Type':'Highpass',
                                      'f_low':1e2/1e3,
                                      'Design':'Chebyshev II'}) #Data is in milliseconds    
    if n_frame == 30:
        ny=6
        nx=5
        gs=GridSpec(nx,ny)
        for index_grid_x in range(nx):
            for index_grid_y in range(ny):
                print(time_range[0]+(time_range[1]-time_range[0])/(n_frame-1)*(index_grid_x*ny+index_grid_y))
                plt.subplot(gs[index_grid_x,index_grid_y])
                time=time_range[0]+(time_range[1]-time_range[0])/(n_frame-1)*(index_grid_x*ny+index_grid_y)
                flap.plot(object_name,plot_type='contour',
                          exp_id=exp_id,
                          slicing={'Time':time},
                          axes=['Device R','Device z','Time'],
                          options={'Z range':z_range,
                                   'Interpolation': 'Closest value',
                                   'Clear':False,
                                   'Colormap':colormap,
                                   })
                
                #plt.title(str(exp_id)+' @ '+str(time_range[0]+(time_range[1]-time_range[0])/(n_frame-1)*(index_grid_x*4+index_grid_y))+'ms')
                plt.title(str(exp_id)+' @ '+f"{time:.4f}"+'ms')
                frame1 = plt.gca()
                frame1.axis('equal')
                if index_grid_y != 0:
                    frame1.axes.get_yaxis().set_visible(False)
                if index_grid_x != nx-1:
                    frame1.axes.get_xaxis().set_visible(False)                    
                
    plt.savefig('NSTX_GPI_video_frames_'+str(exp_id)+'_'+str(time_range[0])+'_'+str(time_range[1])+'_nf_'+str(n_frame)+'.pdf')
    
thisdir = os.path.dirname(os.path.realpath(__file__))
fn = os.path.join(thisdir,"flap_nstx.cfg")
flap.config.read(file_name=fn)      
    