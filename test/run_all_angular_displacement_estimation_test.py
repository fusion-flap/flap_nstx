#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 15:07:10 2022

@author: mlampert
"""
import flap_nstx
flap_nstx.register('NSTX_GPI')

from flap_nstx.test import test_angular_displacement_estimation
import time
import numpy as np

def run_all_angular_displacement_estimation_test():
    i_calc=0
    for method in ['watershed', 'contour']:#, 'ccf']:
        for angle_method in ['angle', 'ALI']:
            for test in [[True,False,False,False],
                         [False,False,True,False],
                         [False,False,False,True]]:
                i_calc+=1
                start_time=time.time()
                print(type(test_angular_displacement_estimation))
                test_angular_displacement_estimation(size_angle=test[0],
                                                     frame_size_angle=test[2],
                                                     elongation_angle=test[3],

                                                     n_angle=18,
                                                     angle_range=[-85,85],

                                                     n_size=9,
                                                     size_range=[6,33],

                                                     n_scale=10,
                                                     scale_range=[0.01,0.19],

                                                     n_elongation=10,
                                                     elongation_range=[0.1,1.],

                                                     frame_size_range=[8,200],
                                                     frame_size_step=8,

                                                     nocalc=False,
                                                     pdf=True,
                                                     plot=False,

                                                     method=method,
                                                     angle_method=angle_method,
                                                     )
                finish_time=time.time()
                rem_time=(finish_time-start_time)*(18-i_calc)

                print('\n**********\nRemaining time from the full calculation: '+str(int(np.mod(rem_time,60)))+'min.\n **********\n')