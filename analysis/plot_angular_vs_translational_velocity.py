#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 13:43:48 2021

@author: mlampert
"""
import os
import copy

import flap
import flap_nstx
flap_nstx.register()
import flap_mdsplus
flap_mdsplus.register('NSTX_MDSPlus')

thisdir = os.path.dirname(os.path.realpath(__file__))
fn = os.path.join(thisdir,"flap_nstx.cfg")
flap.config.read(file_name=fn)

from flap_nstx.analysis import plot_nstx_gpi_angular_velocity_distribution, plot_nstx_gpi_velocity_distribution
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
import matplotlib.cm as cm

publication=True
if publication:
    #figsize=(8.5/2.54, 
    #         8.5/2.54/1.618*1.1)
    figsize=(17/2.54,10/2.54)
    plt.rc('font', family='serif', serif='Helvetica')
    labelsize=6
    linewidth=0.5
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
else:
    figsize=None
    
def plot_angular_vs_translational_velocity(window_average=500e-6,
                                                tau_range=[-500e-6,500e-6],
                                                sampling_time=2.5e-6,
                                                pdf=False,
                                                plot=True,
                                                return_results=False,
                                                return_error=False,
                                                plot_variance=True,
                                                plot_error=False,
                                                normalized_velocity=True,
                                                normalized_structure=True,
                                                subtraction_order=4,
                                                opacity=0.2,
                                                correlation_threshold=0.6,
                                                plot_max_only=False,
                                                plot_for_publication=False,
                                                gpi_plane_calculation=True,
                                                plot_scatter=True,
                                                elm_time_base='frame similarity',
                                                n_hist=50,
                                                min_max_range=False,
                                                nocalc=False,
                                                general_plot=True,
                                                plot_for_velocity=False,
                                                plot_for_structure=False,
                                                plot_for_dependence=False,):
    wd=flap.config.get_all_section('Module NSTX_GPI')['Working directory']
    
    time_vec, y_vector_rot, y_vector_avg_rot = plot_nstx_gpi_angular_velocity_distribution(return_results=True)
    
    time_vec, y_vector_tra, y_vector_avg_tra = plot_nstx_gpi_velocity_distribution(return_results=True)
    
    tau_ind=np.where(np.logical_and(time_vec >= tau_range[0]*1e3, time_vec <= tau_range[1]*1e3))
    y_vector_avg_tra[0]['Data']=y_vector_avg_tra[0]['Data'][tau_ind]  #vrad
    y_vector_avg_tra[1]['Data']=y_vector_avg_tra[1]['Data'][tau_ind]  #vpol
    y_vector_avg_tra[4]['Data']=y_vector_avg_tra[4]['Data'][tau_ind]  #dpol
    y_vector_avg_tra[5]['Data']=y_vector_avg_tra[5]['Data'][tau_ind]  #drad
    y_vector_avg_tra[8]['Data']=y_vector_avg_tra[8]['Data'][tau_ind]  #r-r_sep
    
    y_vector_avg_rot[6]['Data']=y_vector_avg_rot[6]['Data'][tau_ind] # ang vel skim
    y_vector_avg_rot[7]['Data']=y_vector_avg_rot[7]['Data'][tau_ind] # ang vel flap
    y_vector_avg_rot[8]['Data']=y_vector_avg_rot[8]['Data'][tau_ind] # exp vel skim
    y_vector_avg_rot[9]['Data']=y_vector_avg_rot[9]['Data'][tau_ind] # exp vel flap
    
    pdf_object=PdfPages(wd+'/plots/parameter_dependence_based_on_medians_angular.pdf')
    gs=GridSpec(2,2)
    plt.figure()   
    ax,fig=plt.subplots(figsize=(8.5/2.54,8.5/2.54))
    
    colors = iter(cm.gist_ncar(np.linspace(0, 1, len(y_vector_avg_tra[0]['Data']))))
    
    plt.subplot(gs[0,0])
    plt.plot(y_vector_avg_tra[1]['Data'],
             y_vector_avg_tra[0]['Data'],
             lw='0.2')
    for ind_a in range(len(y_vector_avg_tra[0]['Data'])):
        color=copy.deepcopy(next(colors))
        plt.scatter(y_vector_avg_tra[1]['Data'][ind_a], 
                    y_vector_avg_tra[0]['Data'][ind_a], 
                    color=color,
                    s=1)
    plt.xlabel('v pol [km/s]')
    plt.ylabel('v rad [km/s]')
    plt.title('vrad vs. vpol')
    
    colors = iter(cm.gist_ncar(np.linspace(0, 1, len(y_vector_avg_tra[0]['Data']))))
    plt.subplot(gs[0,1])
    plt.plot(y_vector_avg_tra[5]['Data'],
             y_vector_avg_tra[4]['Data'],
             lw='0.2')
    for ind_a in range(len(y_vector_avg_tra[4]['Data'])):
        color=copy.deepcopy(next(colors))
        plt.scatter(y_vector_avg_tra[5]['Data'][ind_a], 
                    y_vector_avg_tra[4]['Data'][ind_a], 
                    color=color,
                    s=1)
    plt.xlabel('d rad [mm]')
    plt.ylabel('d pol [mm]')
    plt.title('drad vs. dpol')
    
    colors = iter(cm.gist_ncar(np.linspace(0, 1, len(y_vector_avg_tra[0]['Data']))))
    plt.subplot(gs[1,0])
    plt.plot(y_vector_avg_tra[8]['Data'],
             y_vector_avg_tra[0]['Data'],
             lw='0.2')
    for ind_a in range(len(y_vector_avg_tra[8]['Data'])):
        color=copy.deepcopy(next(colors))
        plt.scatter(y_vector_avg_tra[8]['Data'][ind_a], 
                    y_vector_avg_tra[0]['Data'][ind_a], 
                    color=color,
                    s=1)
    
    plt.xlabel('r-r_sep [mm]')
    plt.ylabel('vrad [km/s]')
    plt.title('r-r_sep vs. vrad')
    
    colors = iter(cm.gist_ncar(np.linspace(0, 1, len(y_vector_avg_tra[0]['Data']))))
    plt.subplot(gs[1,1])
    plt.plot(y_vector_avg_tra[8]['Data'],
             y_vector_avg_tra[1]['Data'],
             lw='0.2')
    for ind_a in range(len(y_vector_avg_tra[8]['Data'])):
        color=copy.deepcopy(next(colors))
        plt.scatter(y_vector_avg_tra[8]['Data'][ind_a], 
                    y_vector_avg_tra[1]['Data'][ind_a], 
                    color=color,
                    s=1)
    plt.xlabel('r-r_sep [mm]')
    plt.ylabel('vpol [km/s]')
    plt.title('r-r_sep vs. vpol')           

    pdf_object.savefig()
    
    
    #r-rsep vs angular results
    
    gs=GridSpec(2,2)
    
    colors = iter(cm.gist_ncar(np.linspace(0, 1, len(y_vector_avg_tra[0]['Data']))))
    plt.subplot(gs[0,0])
    plt.plot(y_vector_avg_tra[8]['Data'],
             y_vector_avg_rot[6]['Data'],
             lw='0.2')
    for ind_a in range(len(y_vector_avg_tra[8]['Data'])):
        color=copy.deepcopy(next(colors))
        plt.scatter(y_vector_avg_tra[8]['Data'][ind_a], 
                    y_vector_avg_rot[6]['Data'][ind_a], 
                    color=color,
                    s=1)
    plt.xlabel('r-r_sep [mm]')
    plt.ylabel('omega [1/s]')
    plt.title('r-r_sep vs. omega skim')  

    colors = iter(cm.gist_ncar(np.linspace(0, 1, len(y_vector_avg_tra[0]['Data']))))
    plt.subplot(gs[0,1])
    plt.plot(y_vector_avg_tra[8]['Data'],
             y_vector_avg_rot[7]['Data'],
             lw='0.2')
    for ind_a in range(len(y_vector_avg_tra[8]['Data'])):
        color=copy.deepcopy(next(colors))
        plt.scatter(y_vector_avg_tra[8]['Data'][ind_a], 
                    y_vector_avg_rot[7]['Data'][ind_a], 
                    color=color,
                    s=1)
    plt.xlabel('r-r_sep [mm]')
    plt.ylabel('omega [1/s]')
    plt.title('r-r_sep vs. omega FLAP')  

    colors = iter(cm.gist_ncar(np.linspace(0, 1, len(y_vector_avg_tra[0]['Data']))))
    plt.subplot(gs[1,0])
    plt.plot(y_vector_avg_tra[8]['Data'],
             y_vector_avg_rot[8]['Data'],
             lw='0.2')
    for ind_a in range(len(y_vector_avg_tra[8]['Data'])):
        color=copy.deepcopy(next(colors))
        plt.scatter(y_vector_avg_tra[8]['Data'][ind_a], 
                    y_vector_avg_rot[8]['Data'][ind_a], 
                    color=color,
                    s=1)
    plt.xlabel('r-r_sep [mm]')
    plt.ylabel('exp vel [1/s]')
    plt.title('r-r_sep vs. exp vel skim')  

    colors = iter(cm.gist_ncar(np.linspace(0, 1, len(y_vector_avg_tra[0]['Data']))))
    plt.subplot(gs[1,1])
    plt.plot(y_vector_avg_tra[8]['Data'],
             y_vector_avg_rot[9]['Data'],
             lw='0.2')
    for ind_a in range(len(y_vector_avg_tra[8]['Data'])):
        color=copy.deepcopy(next(colors))
        plt.scatter(y_vector_avg_tra[8]['Data'][ind_a], 
                    y_vector_avg_rot[9]['Data'][ind_a], 
                    color=color,
                    s=1)
    plt.xlabel('r-r_sep [mm]')
    plt.ylabel('exp vel [1/s]')
    plt.title('r-r_sep vs. exp vel FLAP')      
    
    pdf_object.savefig()
    
    #v_rad vs angular results
    
    gs=GridSpec(2,2)
    
    colors = iter(cm.gist_ncar(np.linspace(0, 1, len(y_vector_avg_tra[0]['Data']))))
    plt.subplot(gs[0,0])
    plt.plot(y_vector_avg_tra[0]['Data'],
             y_vector_avg_rot[6]['Data'],
             lw='0.2')
    for ind_a in range(len(y_vector_avg_tra[0]['Data'])):
        color=copy.deepcopy(next(colors))
        plt.scatter(y_vector_avg_tra[0]['Data'][ind_a], 
                    y_vector_avg_rot[6]['Data'][ind_a], 
                    color=color,
                    s=1)
    plt.xlabel('vrad [km/s]')
    plt.ylabel('omega [1/s]')
    plt.title('vrad vs. ang vel skim')   

    colors = iter(cm.gist_ncar(np.linspace(0, 1, len(y_vector_avg_tra[0]['Data']))))
    plt.subplot(gs[0,1])
    plt.plot(y_vector_avg_tra[0]['Data'],
             y_vector_avg_rot[7]['Data'],
             lw='0.2')
    for ind_a in range(len(y_vector_avg_tra[0]['Data'])):
        color=copy.deepcopy(next(colors))
        plt.scatter(y_vector_avg_tra[0]['Data'][ind_a], 
                    y_vector_avg_rot[7]['Data'][ind_a], 
                    color=color,
                    s=1)
    plt.xlabel('vrad [km/s]')
    plt.ylabel('omega [1/s]')
    plt.title('vrad vs. ang vel FLAP')   

    colors = iter(cm.gist_ncar(np.linspace(0, 1, len(y_vector_avg_tra[0]['Data']))))
    plt.subplot(gs[1,0])
    plt.plot(y_vector_avg_tra[0]['Data'],
             y_vector_avg_rot[8]['Data'],
             lw='0.2')
    for ind_a in range(len(y_vector_avg_tra[0]['Data'])):
        color=copy.deepcopy(next(colors))
        plt.scatter(y_vector_avg_tra[0]['Data'][ind_a], 
                    y_vector_avg_rot[8]['Data'][ind_a], 
                    color=color,
                    s=1)
    plt.xlabel('vrad [km/s]')
    plt.ylabel('exp vel [1/s]')
    plt.title('vrad vs. exp vel sim')   

    colors = iter(cm.gist_ncar(np.linspace(0, 1, len(y_vector_avg_tra[0]['Data']))))
    plt.subplot(gs[1,1])
    plt.plot(y_vector_avg_tra[0]['Data'],
             y_vector_avg_rot[9]['Data'],
             lw='0.2')
    for ind_a in range(len(y_vector_avg_tra[0]['Data'])):
        color=copy.deepcopy(next(colors))
        plt.scatter(y_vector_avg_tra[0]['Data'][ind_a], 
                    y_vector_avg_rot[9]['Data'][ind_a], 
                    color=color,
                    s=1)
    plt.xlabel('vrad [km/s]')
    plt.ylabel('exp vel [1/s]')
    plt.title('vrad vs. exp vel FLAP')          
    pdf_object.savefig()
    
    pdf_object.close()