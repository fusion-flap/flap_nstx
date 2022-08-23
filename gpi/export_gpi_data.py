#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 00:10:23 2022

@author: mlampert
"""
#Core modules
import os

import warnings
warnings.filterwarnings("ignore")

import flap
import flap_nstx
flap_nstx.register('NSTX_GPI')

import flap_mdsplus
flap_mdsplus.register('NSTX_MDSPlus')

thisdir = os.path.dirname(os.path.realpath(__file__))
fn = os.path.join(thisdir,"../flap_nstx.cfg")
flap.config.read(file_name=fn)

wd=flap.config.get_all_section('Module NSTX_GPI')['Working directory']

import scipy
import pickle

def export_gpi_data(exp_id=None,
                    time_range=None,
                    gaussian_blur=True,
                    normalize=True,
                    normalize_f_high=1e3,
                    subtraction_order=2,
                    output_format='pickle',
                    filename=None,
                    ):

    data_obj=flap.get_data('NSTX_GPI',exp_id=exp_id, name='GPI', object_name='GPI_RAW')
    slicing_original={'Time':flap.Intervals(time_range[0],
                                             time_range[1])}
    comment=''
    data_obj_sliced=flap.slice_data('GPI_RAW', slicing=slicing_original)
    if normalize:
        slicing_for_filtering_only={'Time':flap.Intervals(time_range[0]-10/normalize_f_high,
                                                          time_range[1]+10/normalize_f_high)}
        data_obj_for_filtering=data_obj.slice_data(slicing=slicing_for_filtering_only)
        flap.add_data_object(data_obj_for_filtering,
                             'GPI_SLICED_FOR_FILTERING')

        flap_nstx.gpi.normalize_gpi('GPI_SLICED_FOR_FILTERING',
                                    exp_id=139901,
                                    slicing_time=slicing_original,
                                    normalizer_object_name='GPI_LPF_INTERVAL',
                                    output_name='GPI_NORMALIZED',
                                    return_object='div norm')
        comment+='_norm'

    if subtraction_order is not None:
        data_obj_sliced=flap_nstx.tools.detrend_multidim('GPI_NORMALIZED',
                                                         exp_id=exp_id,
                                                         order=subtraction_order,
                                                         coordinates=['Image x', 'Image y'],
                                                         output_name='GPI_DETREND')
        comment+='_detrend_o'+str(subtraction_order)

    data=data_obj_sliced.data
    time=data_obj_sliced.coordinate('Time')[0][:,0,0]
    r_coord=data_obj_sliced.coordinate('Device R')[0][0,:,:]
    z_coord=data_obj_sliced.coordinate('Device z')[0][0,:,:]

    if gaussian_blur:
        for ind_frames in range(len(data_obj_sliced.data[:,0,0])):
            data_obj_sliced.data[ind_frames,:,:] = scipy.ndimage.median_filter(data_obj_sliced.data[ind_frames,:,:], 5)
        comment+='_gb5'

    if output_format == 'pickle':
        if filename==None:
            filename=flap_nstx.tools.filename(exp_id=exp_id,
                                     time_range=time_range,
                                     working_directory=wd+'/processed_data',
                                     comment=comment,
                                     purpose='gpi_data',
                                     extension='pickle')

        pickle.dump({'Time':time,
                     'Data':data,
                     'R':r_coord,
                     'z':z_coord,},
                    open(filename,'wb'))