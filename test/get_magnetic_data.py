#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 15:59:14 2022

@author: mlampert
"""

import os

import scipy
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import flap
import flap_nstx

thisdir = os.path.dirname(os.path.realpath(__file__))
fn = os.path.join(thisdir,"../flap_nstx.cfg")
flap.config.read(file_name=fn)
flap_nstx.register()

shotlist=[139022, 139026, 139027, 139033, 139034, 139045,139046,
          139047,139048,139049, 139050, 139053, 139054, 139055, 139056, 139057, 139058]
nodenames=['\\bdot_l1dmivvhf1_raw',
           '\\bdot_l1dmivvhf2_raw',
           '\\bdot_l1dmivvhf3_raw',
           '\\bdot_l1dmivvhf4_raw',
           '\\bdot_l1dmivvhf5_raw',
           '\\bdot_l1dmivvhf6_raw',
           '\\bdot_l1dmivvhf7_raw',
           '\\bdot_l1dmivvhf8_raw',
           '\\bdot_l1dmivvhf9_raw',
           '\\bdot_l1dmivvhf10_raw',
           '\\bdot_l1dmivvhf11_raw',
           '\\bdot_l1dmivvhf12_raw',
           
           '\\bdot_l1dmivvhn1_raw',
           '\\bdot_l1dmivvhn2_raw',
           '\\bdot_l1dmivvhn3_raw',
           '\\bdot_l1dmivvhn4_raw',
           '\\bdot_l1dmivvhn5_raw',
           '\\bdot_l1dmivvhn6_raw',
           '\\bdot_l1dmivvhn7_raw',
           '\\bdot_l1dmivvhn8_raw',
           '\\bdot_l1dmivvhn9_raw',
           '\\bdot_l1dmivvhn10_raw',
           '\\bdot_l1dmivvhn11_raw',
           '\\bdot_l1dmivvhn12_raw',
           '\\bdot_l1dmivvhn13_raw',
           '\\bdot_l1dmivvhn14_raw',
           '\\bdot_l1dmivvhn15_raw',
           '\\bdot_l1dmivvhn16_raw',]

for shot in shotlist:
    for node in nodenames:
        try:
            data_object=flap.get_data('NSTX_MDSPlus',
                              name='\OPS_PC::'+node,
                              exp_id=shot,
                              object_name='BDOT_SIGNAL')
            flap.delete_data_object('BDOT_SIGNAL')
        except:
            print(shot,node,'Failed')
