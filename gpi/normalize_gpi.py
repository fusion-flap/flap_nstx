#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 14:48:59 2021

@author: mlampert
"""

import copy
import flap
import numpy as np

def normalize_gpi(input_object,
                  exp_id=None,
                  normalize='roundtrip',
                  normalize_f_high=1e3,
                  normalize_f_kernel='Elliptic',

                  normalizer_object_name=None,
                  slicing_time=None,
                  output_name=None,

                  return_object='coefficient',                                  #Can be 'coefficint' or 'sub norm', 'div norm' for subtracted normalized and divided normalized
                  ):

    if normalize == 'simple':
        flap.filter_data(input_object,
                         exp_id=exp_id,
                         coordinate='Time',
                         options={'Type':'Lowpass',
                                  'f_high':normalize_f_high,
                                  'Design':normalize_f_kernel},
                         output_name=normalizer_object_name)
        coefficient=flap.slice_data(normalizer_object_name,
                                    exp_id=exp_id,
                                    slicing=slicing_time,
                                    output_name=output_name).data

    elif normalize == 'roundtrip':
        norm_obj=flap.filter_data(input_object,
                                  exp_id=exp_id,
                                  coordinate='Time',
                                  options={'Type':'Lowpass',
                                           'f_high':normalize_f_high,
                                           'Design':normalize_f_kernel},
                                  output_name=normalizer_object_name)

        norm_obj.data=np.flip(norm_obj.data,axis=0)
        norm_obj=flap.filter_data(normalizer_object_name,
                                  exp_id=exp_id,
                                  coordinate='Time',
                                  options={'Type':'Lowpass',
                                           'f_high':normalize_f_high,
                                           'Design':normalize_f_kernel},
                                  output_name=normalizer_object_name)

        norm_obj.data=np.flip(norm_obj.data,axis=0)
        coefficient=flap.slice_data(normalizer_object_name,
                                    exp_id=exp_id,
                                    slicing=slicing_time,
                                    output_name=output_name).data

    elif normalize == 'halved':
        #Find the peak of the signal (aka ELM time as of GPI data)
        data_obj=flap.get_data_object_ref(input_object).slice_data(summing={'Image x':'Mean', 'Image y':'Mean'})
        ind_peak=np.argmax(data_obj.data)
        data_obj_reverse=copy.deepcopy(flap.get_data_object('SLICED_FOR_FILTERING'))
        data_obj_reverse.data=np.flip(data_obj_reverse.data, axis=0)
        flap.add_data_object(data_obj_reverse,'SLICED_FOR_FILTERING_REV')

        normalizer_object_name_reverse='LPF_INTERVAL_REV'
        flap.filter_data(input_object,
                         exp_id=exp_id,
                         coordinate='Time',
                         options={'Type':'Lowpass',
                                  'f_high':normalize_f_high,
                                  'Design':normalize_f_kernel},
                         output_name=normalizer_object_name)

        coefficient1_sliced=flap.slice_data(normalizer_object_name,
                                     exp_id=exp_id,
                                     slicing=slicing_time)

        coefficient2=flap.filter_data('SLICED_FOR_FILTERING_REV',
                                     exp_id=exp_id,
                                     coordinate='Time',
                                     options={'Type':'Lowpass',
                                              'f_high':normalize_f_high,
                                              'Design':normalize_f_kernel},
                                     output_name=normalizer_object_name_reverse)

        coefficient2.data=np.flip(coefficient2.data, axis=0)
        coefficient2_sliced=flap.slice_data(normalizer_object_name_reverse,
                                            exp_id=exp_id,
                                            slicing=slicing_time)

        coeff1_first_half=coefficient1_sliced.data[:ind_peak-4,:,:]
        coeff2_second_half=coefficient2_sliced.data[ind_peak-4:,:,:]
        coefficient=np.append(coeff1_first_half,coeff2_second_half, axis=0)
        coefficient_dataobject=copy.deepcopy(coefficient1_sliced)
        coefficient_dataobject.data=coefficient
        flap.add_data_object(coefficient_dataobject, output_name)

    elif normalize is None:
        return None

    if return_object == 'coefficient':
        return coefficient
    else:
        data_obj_orig=flap.get_data_object_ref(input_object)
        data_obj_normalized=data_obj_orig.slice_data(slicing=slicing_time)

        if return_object == 'sub norm':
            data_obj_normalized.data=data_obj_normalized.data-coefficient
        if return_object == 'div norm':
            data_obj_normalized.data=data_obj_normalized.data/coefficient

        if output_name is not None:
            flap.add_data_object(data_obj_normalized, output_name)

        return data_obj_normalized