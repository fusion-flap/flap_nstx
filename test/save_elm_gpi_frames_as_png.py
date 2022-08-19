#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 15:14:45 2021

@author: mlampert
"""
#import numpy as np
#import matplotlib as plt
#import math
import pandas

import os
import flap
import flap_nstx
import numpy as np

from flap_nstx.gpi import calculate_nstx_gpi_frame_by_frame_velocity, calculate_nstx_gpi_tde_velocity, calculate_nstx_gpi_angular_velocity
from flap_nstx.gpi import nstx_gpi_velocity_analysis_spatio_temporal_displacement

import imageio
import scipy

thisdir = os.path.dirname(os.path.realpath(__file__))
fn = os.path.join(thisdir,"../flap_nstx.cfg")
flap.config.read(file_name=fn)
flap_nstx.register()

def save_elm_gpi_frames_as_png(upsample_size=[64,80],
                               elm_time_window=250e-6,
                               pre_process=False):
    database_file='/Users/mlampert/work/NSTX_workspace/db/ELM_findings_mlampert_velocity_good.csv'
    db=pandas.read_csv(database_file, index_col=0)
    elm_index=(list(db.index))
    wd=flap.config.get_all_section('Module NSTX_GPI')['Working directory']
    for index_elm in range(len(elm_index)):
        #preprocess velocity results, tackle with np.nan and outliers
        shot=int(db.loc[elm_index[index_elm]]['Shot'])
        #define ELM time for all the cases
        #elm_time=db.loc[elm_index[index_elm]]['ELM time']/1000.
        elm_time=db.loc[elm_index[index_elm]]['Time prec']
        
        time_range=[elm_time-elm_time_window,elm_time+elm_time_window]
        flap.delete_data_object('*')
        print('Calculating '+str(shot)+ ' at '+str(elm_time))
        status=db.loc[elm_index[index_elm]]['OK/NOT OK']
        if status != 'NO' and shot >= 139877:
            d=flap.get_data('NSTX_GPI',exp_id=shot,
                            name='',
                            object_name='GPI')
                
            slicing={'Time':flap.Intervals(time_range[0],time_range[1])}
            d=flap.slice_data('GPI',exp_id=shot, 
                              slicing=slicing,
                              output_name='GPI_SLICED_FULL')
            ind_elm=np.argmin(np.sum(d.data[0:-1,:,:]*d.data[1:,:,:],axis=(1,2)) /
                              (np.sqrt(np.sum(d.data[0:-1,:,:],axis=(1,2))**2) * np.sqrt(np.sum(d.data[1:,:,:],axis=(1,2))**2)))
            if pre_process:
                slicing_for_filtering={'Time':flap.Intervals(time_range[0]-1/1e3*10,
                                                             time_range[1]+1/1e3*10)}
                flap.slice_data('GPI',
                            exp_id=shot,
                            slicing=slicing_for_filtering,
                            output_name='GPI_SLICED_FOR_FILTERING')
                coefficient=flap_nstx.gpi.normalize_gpi('GPI_SLICED_FOR_FILTERING',
                                                        exp_id=shot,
                                                        slicing_time=slicing,
                                                        normalize='roundtrip',
                                                        normalize_f_high=1e3,
                                                        normalize_f_kernel='Elliptic',
                                                        normalizer_object_name='GPI_LPF_INTERVAL',
                                                        output_name='GPI_GAS_CLOUD')
                
                d.data = d.data/coefficient
                # flap.add_data_object(d, 'GPI_SLICED_FILTERED')
                # d=flap_nstx.tools.detrend_multidim('GPI_SLICED_FILTERED',
                #                                    exp_id=shot,
                #                                    order=4, 
                #                                    coordinates=['Image x', 'Image y'], 
                #                                    output_name='GPI_DETREND_CCF_VEL')
                d.data = np.round(d.data/d.data.max()*255).astype('uint8')
            else:
                d.data = np.round(d.data/4095.*255).astype('uint8')
            for i_time in range(len(d.data[:,0,0])):
                time=(time_range[0]+2.5e-6*i_time)*1e3
                filename=wd+'/plots/ML_filament_finding_all/gpi_elm_filament_'+str(shot)+'_'+f'{time:.3f}'+'ms_'+str(i_time)+'.png'
                data=np.rot90(d.data[i_time,:,:], k=1, axes=(0, 1))
                if upsample_size != [64,80]:
                    d_upsample=scipy.ndimage.zoom(data, [upsample_size[0]/64.,upsample_size[1]/80.], order=3)
                    imageio.imwrite(filename, d_upsample)
                else:
                    imageio.imwrite(filename, data)
            try:
                for i_time in range(25):
                    time=(time_range[0]+(ind_elm+i_time-10)*2.5e-6)*1e3
                    filename=wd+'/plots/ML_filament_finding_around_ELM/gpi_elm_filament_'+str(shot)+'_'+f'{time:.3f}'+'ms_'+str(i_time)+'.png'
                    data=np.rot90(d.data[i_time+ind_elm-10,:,:], k=1, axes=(0, 1))
                    if upsample_size != [64,80]:
                        d_upsample=scipy.ndimage.zoom(data, [upsample_size[0]/64.,upsample_size[1]/80.], order=3)
                        imageio.imwrite(filename, d_upsample)
                    else:
                        imageio.imwrite(filename, data)
            except:
                print('Shot '+str(shot)+' failed.')
                pass
        percent=(index_elm+1)/len(elm_index)*100.
        print('Finished '+f'{percent:.3f}'+'% of the file saving')