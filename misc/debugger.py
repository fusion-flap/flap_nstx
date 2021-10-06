#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 23:33:15 2021

@author: mlampert
"""

def debugger():
    calculate_nstx_gpi_frame_by_frame_velocity(exp_id=139901, time_range=[0.3245,0.3255], normalize_for_velocity=True, skip_structure_calculation=False, normalize_for_size=True, plot=False, nocalc=False, subtraction_order_for_velocity=4, test=False, str_finding_method='watershed')