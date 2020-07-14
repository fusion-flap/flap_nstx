#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 12:26:32 2020

@author: mlampert
"""

averaging='before_after'
plot_elm_properties_vs_gradient_before_vs_after(plot=False, 
                                                pdf=True, 
                                                recalc=True, 
                                                plot_only_good=True, 
                                                plot_error=True, 
                                                auto_x_range=False,
                                                subtraction_order=4,
                                                pressure_grad_range=[-200,0],
                                                thomson_time_window=10e-3, 
                                                normalized_velocity=True, 
                                                elm_duration=0.1e-3, 
                                                gradient_type='max', 
                                                averaging=averaging,
                                                dependence_error_threshold=0.3)
