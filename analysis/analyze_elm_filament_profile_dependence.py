#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 01:32:23 2022

@author: mlampert
"""

import copy
import os
import pickle

from flap_nstx.analysis import plot_elm_rotation_vs_gradient, plot_elm_rotation_vs_max_gradient
from flap_nstx.analysis import plot_elm_rotation_vs_gradient_before_vs_after, plot_elm_rotation_vs_ahmed_fitting

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

#Flap imports
import flap
import flap_nstx
flap_nstx.register()
import flap_mdsplus
import pandas

#Setting up FLAP
flap_mdsplus.register('NSTX_MDSPlus')    
thisdir = os.path.dirname(os.path.realpath(__file__))
fn = os.path.join(thisdir,"../flap_nstx.cfg")
flap.config.read(file_name=fn) 

def analyze_elm_filament_profile_dependence(plot_level=4,
                                            plot_pearson=False,
                                            pdf=True,
                                            plot=False,
                                            thomson_time_window=3e-3,
                                            plot_full_correlation=False,
                                            
                                            elm_window=400e-6,
                                            elm_duration=100e-6,
                                            correlation_threshold=0.6,
                                            ):
    if plot:
        plt.rc('font', family='serif', serif='Helvetica')
        labelsize=8
        linewidth=0.4
        major_ticksize=2
        plt.rc('text', usetex=False)
        plt.rcParams['pdf.fonttype'] = 42
        plt.rcParams['ps.fonttype'] = 42
        
        plt.rcParams['lines.linewidth'] = linewidth
        plt.rcParams['axes.linewidth'] = linewidth
        plt.rcParams['axes.labelsize'] = labelsize
        plt.rcParams['axes.titlesize'] = labelsize
        
        plt.rcParams['xtick.labelsize'] = labelsize
        plt.rcParams['xtick.major.size'] = major_ticksize
        plt.rcParams['xtick.major.width'] = linewidth
        plt.rcParams['xtick.minor.width'] = linewidth/2
        plt.rcParams['xtick.minor.size'] = major_ticksize/2
        
        plt.rcParams['ytick.labelsize'] = labelsize
        plt.rcParams['ytick.major.width'] = linewidth
        plt.rcParams['ytick.major.size'] = major_ticksize
        plt.rcParams['ytick.minor.width'] = linewidth/2
        plt.rcParams['ytick.minor.size'] = major_ticksize/2
        plt.rcParams['legend.fontsize'] = labelsize
        
    
    wd=flap.config.get_all_section('Module NSTX_GPI')['Working directory']
    if plot_level == 1:
        res=plot_elm_rotation_vs_gradient(maximum_gradient=True, 
                                          gradient_at_separatrix=False, 
                                          thomson_time_window=thomson_time_window, 
                                          plot=plot, 
                                          pdf=pdf, 
                                          recalc=True, 
                                          plot_preliminary=True,
                                          return_results=True)
        gradient, gradient_error, gpi_results_avg, gpi_results_max = res
        plot_level_str='basic'
        
    if plot_level == 2:
        res=plot_elm_rotation_vs_max_gradient(maximum_gradient=True, 
                                              gradient_at_separatrix=False, 
                                              thomson_time_window=thomson_time_window, 
                                              plot=plot, 
                                              pdf=pdf, 
                                              recalc=True, 
                                              plot_both=False, 
                                              thomson_before=True,
                                              return_results=True)
        
        gradient, gradient_thom_cor, gpi_results_avg, gpi_results_max=res
        plot_level_str='max_grad'
        
    if plot_level == 3:
        result_dict=plot_elm_rotation_vs_gradient_before_vs_after(thomson_time_window=thomson_time_window, 
                                                                  plot=plot, 
                                                                  pdf=pdf, 
                                                                  recalc=True,
                                                                  averaging='before_after', 
                                                                  parameter='global', 
                                                                  plot_thomson_profiles=True, 
                                                                  auto_x_range=False, 
                                                                  auto_y_range=False,
                                                                  return_results=True)
        
        gradient=result_dict['gradient_before']
        gradient_error=result_dict['gradient_before_error']
        gpi_results_avg=result_dict['gpi_results_avg_before']
        gpi_results_max=result_dict['gpi_results_max_before']
        plot_level_str='before_after'
        
    if plot_level == 4:
        res=plot_elm_rotation_vs_ahmed_fitting(parameter='value_at_max_grad_simple',#['grad_glob','ped_height','value_at_max_grad', 
                                                                    # "ped_width", max_grad_position, max_grad]
                                               averaging='full', #['before', 'after', 'full', 'elm']
                                               elm_window=400e-6,
                                               elm_duration=100e-6,
                                               plot=plot, 
                                               pdf=pdf, 
                                               plot_linear_fit=True, 
                                               plot_only_good=False, 
                                               dependence_error_threshold=1.2, 
                                               temperature_grad_range=[0,10], 
                                               pressure_grad_range=[0,40], 
                                               density_grad_range=[0,10], 
                                               auto_x_range=False, 
                                               auto_y_range=False,
                                               return_results=True)
        
        gradient, gradient_error, gpi_results_avg, gpi_results_max = res
                
                
    if plot_pearson:
        if pdf:
            pdf_pages=PdfPages(wd+'/plots/filament_profile_dependence_'+plot_level_str+'_pearson.pdf')
        correlation_matrix_avg=np.zeros([len(gradient.keys()),
                                         len(gpi_results_avg.keys())])
        correlation_matrix_max=copy.deepcopy(correlation_matrix_avg)
        key_profile=list(gradient.keys())
        key_gpi=list(gpi_results_avg.keys())
        
        for i_prof in range(len(key_profile)):
            for j_gpi in range(len(key_gpi)):
                
                if len(gpi_results_max[key_gpi[j_gpi]].shape) == 2:
                    ind_nan_b=np.logical_not(np.isnan(gpi_results_max[key_gpi[j_gpi]][:,0]))

                else:
                    ind_nan_b=np.logical_not(np.isnan(gpi_results_max[key_gpi[j_gpi]]))
                    
                ind_nan_a=np.logical_not(np.isnan(gradient[key_profile[i_prof]]))
                print(gradient[key_profile[i_prof]].shape, gpi_results_max[key_gpi[j_gpi]].shape)
                ind_nan=np.logical_and(ind_nan_a,ind_nan_b)
                if 'Velocity ccf FLAP' == key_gpi[j_gpi]:
                    corr_b_max=gpi_results_max[key_gpi[j_gpi]][:,0][ind_nan]
                    corr_b_avg=gpi_results_avg[key_gpi[j_gpi]][:,0][ind_nan]

                else:
                    corr_b_max=gpi_results_max[key_gpi[j_gpi]][ind_nan]
                    corr_b_avg=gpi_results_avg[key_gpi[j_gpi]][ind_nan]
                corr_b_max -= np.mean(corr_b_max)
                corr_b_avg -= np.mean(corr_b_avg)
                
                corr_a=gradient[key_profile[i_prof]][ind_nan]
                corr_a -= np.mean(corr_a)
                
                correlation_matrix_avg[i_prof,j_gpi]=np.sum(corr_a*corr_b_avg)/np.sqrt(np.sum(corr_a**2))/np.sqrt(np.sum(corr_b_avg**2))
                correlation_matrix_max[i_prof,j_gpi]=np.sum(corr_a*corr_b_max)/np.sqrt(np.sum(corr_a**2))/np.sqrt(np.sum(corr_b_max**2))
                    
        data = correlation_matrix_avg
        fig,ax=plt.subplots(figsize=(8.5/2.54,8.5/np.sqrt(2)/2.54))
        cs=plt.matshow(data.transpose(),fignum=0, cmap='seismic')
        plt.title('Maximum gradient vs. parameter average correlation')
        plt.xticks(ticks=np.arange(3), labels=key_profile, rotation='vertical')
        
        plt.yticks(ticks=np.arange(4), labels=['v_rad', 'omega', 'omega log', 'exp. velocity'])
        plt.colorbar()
        plt.show()
        plt.tight_layout(pad=0.1)
        if pdf:
            pdf_pages.savefig()
            
        fig,ax=plt.subplots(figsize=(8.5/2.54,8.5/np.sqrt(2)/2.54))
        data = correlation_matrix_max
        cs=plt.matshow(data.transpose(), fignum=0, cmap='seismic')
        plt.title('Maximum gradient vs. parameter maximum correlation')
        plt.xticks(ticks=np.arange(3), labels=key_profile, rotation='vertical')
        
        plt.yticks(ticks=np.arange(4), labels=['v_rad', 'omega', 'omega log', 'exp. velocity'])
        plt.colorbar()
        plt.show()
        plt.tight_layout(pad=0.1)
        
        if pdf:
            pdf_pages.savefig()
            pdf_pages.close()