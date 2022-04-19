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
fn = os.path.join(thisdir,"../flap_nstx.cfg")
flap.config.read(file_name=fn)

from flap_nstx.gpi import plot_nstx_gpi_angular_velocity_distribution, plot_nstx_gpi_velocity_distribution

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
    labelsize=12
    linewidth=0.5
    major_ticksize=4
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
                                           plot_for_dependence=False,
                                           plot_for_pop_paper=False,
                                           plot_log_omega=False,
                                           figure_filename=None,
                                           ):
    
    wd=flap.config.get_all_section('Module NSTX_GPI')['Working directory']
    
    time_vec, y_vector_rot = plot_nstx_gpi_angular_velocity_distribution(plot_for_publication=True,
                                                                        window_average=window_average,
                                                                        subtraction_order=subtraction_order,
                                                                        correlation_threshold=correlation_threshold,
                                                                        pdf=False,
                                                                        plot=False,
                                                                        return_results=True,
                                                                        plot_all_time_traces=False,
                                                                        tau_range=[-1e-3,1e-3],
                                                                        )
    
    time_vec, y_vector_tra = plot_nstx_gpi_velocity_distribution(return_results=True,
                                                                 plot=False,
                                                                 pdf=False)
    
    tau_ind=np.where(np.logical_and(time_vec >= tau_range[0]*1e3, time_vec <= tau_range[1]*1e3))
    
    if not plot_for_pop_paper or figure_filename is None:
        pdf_object=PdfPages(wd+'/plots/parameter_dependence_based_on_medians_angular.pdf')
    else:
        pdf_object=PdfPages(figure_filename)
        
    if plot_log_omega:
        log_str=' log'
    else:
        log_str=''
    if not plot_for_pop_paper:
        gs=GridSpec(2,2)
        plt.figure()   
        ax,fig=plt.subplots(figsize=(8.5/2.54,8.5/2.54))
        colors = iter(cm.gist_ncar(np.linspace(0, 1, len(y_vector_tra['Velocity ccf radial']['median'][tau_ind]))))
        
        plt.subplot(gs[0,0])
        plt.plot(y_vector_tra['Velocity ccf poloidal']['median'][tau_ind],
                 y_vector_tra['Velocity ccf radial']['median'][tau_ind],
                 lw='0.2')
        for ind_a in range(len(y_vector_tra['Velocity ccf radial']['median'][tau_ind])):
            color=copy.deepcopy(next(colors))
            plt.scatter(y_vector_tra['Velocity ccf poloidal']['median'][tau_ind][ind_a], 
                        y_vector_tra['Velocity ccf radial']['median'][tau_ind][ind_a], 
                        color=color,
                        s=1)
        plt.xlabel('$v_{pol}$ [km/s]')
        plt.ylabel('$v_{rad}$ [km/s]')
        plt.title('$v_{rad}$ vs. $v_{pol}$')
        
        colors = iter(cm.gist_ncar(np.linspace(0, 1, len(y_vector_tra['Velocity ccf radial']['median'][tau_ind]))))
        plt.subplot(gs[0,1])
        plt.plot(y_vector_tra['Size max radial']['median'][tau_ind],
                 y_vector_tra['Size max poloidal']['median'][tau_ind],
                 lw='0.2')
        for ind_a in range(len(y_vector_tra['Velocity ccf radial']['median'][tau_ind])):
            color=copy.deepcopy(next(colors))
            plt.scatter(y_vector_tra['Size max radial']['median'][tau_ind][ind_a], 
                        y_vector_tra['Size max poloidal']['median'][tau_ind][ind_a], 
                        color=color,
                        s=1)
        plt.xlabel('$d_{rad}$ [mm]')
        plt.ylabel('$d_{pol}$ [mm]')
        plt.title('$d_{rad} vs. $d_{pol}$')
        
        # colors = iter(cm.gist_ncar(np.linspace(0, 1, len(y_vector_tra['Velocity ccf radial']['median']))))
        # plt.subplot(gs[1,0])
        # plt.plot(y_vector_tra['Distance']['median'],
        #          y_vector_tra['Velocity ccf radial']['median'],
        #          lw='0.2')
        # for ind_a in range(len(y_vector_tra['Distance']['median'])):
        #     color=copy.deepcopy(next(colors))
        #     plt.scatter(y_vector_tra['Distance']['median'][ind_a], 
        #                 y_vector_tra['Velocity ccf radial']['median'][ind_a], 
        #                 color=color,
        #                 s=1)
        
        # plt.xlabel('r-r_sep [mm]')
        # plt.ylabel('vrad [km/s]')
        # plt.title('r-r_sep vs. vrad')
        colors = iter(cm.gist_ncar(np.linspace(0, 1, len(y_vector_tra['Velocity ccf radial']['median'][tau_ind]))))
        plt.subplot(gs[1,0])
        plt.plot(y_vector_tra['Distance']['median'][tau_ind],
                 y_vector_tra['Velocity ccf radial']['median'][tau_ind],
                 lw='0.2')
        for ind_a in range(len(y_vector_tra['Distance']['median'][tau_ind])):
            color=copy.deepcopy(next(colors))
            plt.scatter(y_vector_tra['Distance']['median'][tau_ind][ind_a], 
                        y_vector_tra['Velocity ccf radial']['median'][tau_ind][ind_a], 
                        color=color,
                        s=1)
        
        plt.xlabel('$r-r_{sep}$ [mm]')
        plt.ylabel('$v_{rad}$ [km/s]')
        plt.title('$r-r_{sep}$ vs. $v_{rad}$')
        
        colors = iter(cm.gist_ncar(np.linspace(0, 1, len(y_vector_tra['Velocity ccf poloidal']['median'][tau_ind]))))
        plt.subplot(gs[1,1])
        plt.plot(y_vector_tra['Distance']['median'][tau_ind],
                 y_vector_tra['Velocity ccf poloidal']['median'][tau_ind],
                 lw='0.2')
        for ind_a in range(len(y_vector_tra['Distance']['median'][tau_ind])):
            color=copy.deepcopy(next(colors))
            plt.scatter(y_vector_tra['Distance']['median'][tau_ind][ind_a], 
                        y_vector_tra['Velocity ccf poloidal']['median'][tau_ind][ind_a], 
                        color=color,
                        s=1)
        plt.xlabel('$r-r_{sep}$ [mm]')
        plt.ylabel('$v_{pol}$ [km/s]')
        plt.title('$r-r_{sep}$ vs. $v_{pol}$')           
    
        pdf_object.savefig()
        
        
        #r-rsep vs angular results
        
        gs=GridSpec(2,2)
        
        colors = iter(cm.gist_ncar(np.linspace(0, 1, len(y_vector_tra['Velocity ccf radial']['median'][tau_ind]))))
        plt.subplot(gs[0,0])
        plt.plot(y_vector_tra['Distance']['median'][tau_ind],
                 y_vector_rot['Angular velocity ccf skim']['median'][tau_ind],
                 lw='0.2')
        for ind_a in range(len(y_vector_tra['Distance']['median'][tau_ind])):
            color=copy.deepcopy(next(colors))
            plt.scatter(y_vector_tra['Distance']['median'][tau_ind][ind_a], 
                        y_vector_rot['Angular velocity ccf skim']['median'][tau_ind][ind_a], 
                        color=color,
                        s=1)
        plt.xlabel('$r-r_{sep}$ [mm]')
        plt.ylabel('$\omega$ [rad/s]')
        plt.title('$r-r_{sep}$ vs. $\omega_{skim}$')  
    
        colors = iter(cm.gist_ncar(np.linspace(0, 1, len(y_vector_tra['Velocity ccf radial']['median'][tau_ind]))))
        plt.subplot(gs[0,1])
        plt.plot(y_vector_tra['Distance']['median'][tau_ind],
                 y_vector_rot['Angular velocity ccf FLAP']['median'][tau_ind],
                 lw='0.2')
        for ind_a in range(len(y_vector_tra['Distance']['median'][tau_ind])):
            color=copy.deepcopy(next(colors))
            plt.scatter(y_vector_tra['Distance']['median'][tau_ind][ind_a], 
                        y_vector_rot['Angular velocity ccf FLAP']['median'][tau_ind][ind_a], 
                        color=color,
                        s=1)
        plt.xlabel('$r-r_{sep}$ [mm]')
        plt.ylabel('$\omega$ [rad/s]')
        plt.title('$r-r_{sep}$ vs. $\omega_{FLAP}$')  
    
        colors = iter(cm.gist_ncar(np.linspace(0, 1, len(y_vector_tra['Velocity ccf radial']['median'][tau_ind]))))
        plt.subplot(gs[1,0])
        plt.plot(y_vector_tra['Distance']['median'][tau_ind],
                 y_vector_rot['Expansion velocity ccf skim']['median'][tau_ind],
                 lw='0.2')
        for ind_a in range(len(y_vector_tra['Distance']['median'][tau_ind])):
            color=copy.deepcopy(next(colors))
            plt.scatter(y_vector_tra['Distance']['median'][tau_ind][ind_a], 
                        y_vector_rot['Expansion velocity ccf skim']['median'][tau_ind][ind_a], 
                        color=color,
                        s=1)
        plt.xlabel('$r-r_{sep}$ [mm]')
        plt.ylabel('$f_{s}$')
        plt.title('$r-r_{sep}$ vs. $f_{s}$')  
    
        colors = iter(cm.gist_ncar(np.linspace(0, 1, len(y_vector_tra['Velocity ccf radial']['median'][tau_ind]))))
        plt.subplot(gs[1,1])
        plt.plot(y_vector_tra['Distance']['median'][tau_ind],
                 y_vector_rot['Expansion velocity ccf FLAP']['median'][tau_ind],
                 lw='0.2')
        for ind_a in range(len(y_vector_tra['Distance']['median'][tau_ind])):
            color=copy.deepcopy(next(colors))
            plt.scatter(y_vector_tra['Distance']['median'][tau_ind][ind_a], 
                        y_vector_rot['Expansion velocity ccf FLAP']['median'][tau_ind][ind_a], 
                        color=color,
                        s=1)
        plt.xlabel('$r-r_{sep}$ [mm]')
        plt.ylabel('$f_{s}$')
        plt.title('$r-r_{sep}$ vs. $f_{s,FLAP}$')      
        
        pdf_object.savefig()
        
        #v_rad vs angular results
        
        gs=GridSpec(2,2)
        
        colors = iter(cm.gist_ncar(np.linspace(0, 1, len(y_vector_tra['Velocity ccf radial']['median'][tau_ind]))))
        plt.subplot(gs[0,0])
        plt.plot(y_vector_tra['Velocity ccf radial']['median'][tau_ind],
                 y_vector_rot['Angular velocity ccf skim']['median'][tau_ind],
                 lw='0.2')
        for ind_a in range(len(y_vector_tra['Velocity ccf radial']['median'][tau_ind])):
            color=copy.deepcopy(next(colors))
            plt.scatter(y_vector_tra['Velocity ccf radial']['median'][tau_ind][ind_a], 
                        y_vector_rot['Angular velocity ccf skim']['median'][tau_ind][ind_a], 
                        color=color,
                        s=1)
        plt.xlabel('$v_{rad}$ [km/s]')
        plt.ylabel('$\omega$ [1/s]')
        plt.title('$v_{rad}$ vs. $\omega_{skim}$')   
    
        colors = iter(cm.gist_ncar(np.linspace(0, 1, len(y_vector_tra['Velocity ccf radial']['median'][tau_ind]))))
        plt.subplot(gs[0,1])
        plt.plot(y_vector_tra['Velocity ccf radial']['median'][tau_ind],
                 y_vector_rot['Angular velocity ccf FLAP']['median'][tau_ind],
                 lw='0.2')
        for ind_a in range(len(y_vector_tra['Velocity ccf radial']['median'][tau_ind])):
            color=copy.deepcopy(next(colors))
            plt.scatter(y_vector_tra['Velocity ccf radial']['median'][tau_ind][ind_a], 
                        y_vector_rot['Angular velocity ccf FLAP']['median'][tau_ind][ind_a], 
                        color=color,
                        s=1)
        plt.xlabel('$v_{rad}$ [km/s]')
        plt.ylabel('$\omega$ [rad/s]')
        plt.title('$v_{rad}$ vs. $\omega$ FLAP')   
    
        colors = iter(cm.gist_ncar(np.linspace(0, 1, len(y_vector_tra['Velocity ccf poloidal']['median'][tau_ind]))))
        plt.subplot(gs[1,0])
        plt.plot(y_vector_tra['Velocity ccf poloidal']['median'][tau_ind],
                 y_vector_rot['Angular velocity ccf skim']['median'][tau_ind],
                 lw='0.2')
        for ind_a in range(len(y_vector_tra['Velocity ccf poloidal']['median'][tau_ind])):
            color=copy.deepcopy(next(colors))
            plt.scatter(y_vector_tra['Velocity ccf poloidal']['median'][tau_ind][ind_a], 
                        y_vector_rot['Angular velocity ccf skim']['median'][tau_ind][ind_a], 
                        color=color,
                        s=1)
        plt.xlabel('$v_{pol}$ [km/s]')
        plt.ylabel('$\omega$ [rad/s]')
        plt.title('$v_{pol}$ vs. $\omega_{skim}$')   
    
        colors = iter(cm.gist_ncar(np.linspace(0, 1, len(y_vector_tra['Velocity ccf poloidal']['median'][tau_ind]))))
        plt.subplot(gs[1,1])
        plt.plot(y_vector_tra['Velocity ccf poloidal']['median'][tau_ind],
                 y_vector_rot['Angular velocity ccf FLAP']['median'][tau_ind],
                 lw='0.2')
        for ind_a in range(len(y_vector_tra['Velocity ccf poloidal']['median'][tau_ind])):
            color=copy.deepcopy(next(colors))
            plt.scatter(y_vector_tra['Velocity ccf poloidal']['median'][tau_ind][ind_a], 
                        y_vector_rot['Angular velocity ccf FLAP']['median'][tau_ind][ind_a], 
                        color=color,
                        s=1)
        plt.xlabel('$v_{pol}$ [km/s]')
        plt.ylabel('$\omega$ [rad/s]')
        plt.title('$v_{pol}$ vs. ang vel FLAP')          
        pdf_object.savefig()
        
        
        
    else:

        #gs=GridSpec(2,2)
        plt.figure()   
        fig,axs=plt.subplots(2,2,figsize=(17/2.54,14/2.54))
        colors = iter(cm.gist_ncar(np.linspace(0, 1, len(y_vector_tra['Velocity ccf radial']['median'][tau_ind]))))
        x_text=-0.2
        ax=axs[0,0]
        #plt.subplot(gs[0,0])
        ax.plot(y_vector_tra['Velocity ccf radial']['median'][tau_ind],
                y_vector_rot['Angular velocity ccf FLAP log']['median'][tau_ind],
                lw='0.2')
        for ind_a in range(len(y_vector_tra['Velocity ccf radial']['median'][tau_ind])):
            color=copy.deepcopy(next(colors))
            ax.scatter(y_vector_tra['Velocity ccf radial']['median'][tau_ind][ind_a], 
                        y_vector_rot['Angular velocity ccf FLAP log']['median'][tau_ind][ind_a], 
                        color=color,
                        s=4)
        ax.set_xlabel('$v_{rad}$ [km/s]')
        ax.set_ylabel('$\omega$ [krad/s]')
        ax.set_title('Radial vs. angular velocity')
        ax.text(x_text, 1.05, '(a)', transform=ax.transAxes, size=9)
        
        colors = iter(cm.gist_ncar(np.linspace(0, 1, len(y_vector_tra['Velocity ccf radial']['median'][tau_ind]))))
        #plt.subplot(gs[0,1])
        ax=axs[0,1]
        ax.plot(y_vector_tra['Velocity ccf radial']['median'][tau_ind],
                y_vector_rot['Expansion velocity ccf FLAP']['median'][tau_ind],
                lw='0.5')
        for ind_a in range(len(y_vector_tra['Velocity ccf radial']['median'][tau_ind])):
            color=copy.deepcopy(next(colors))
            ax.scatter(y_vector_tra['Velocity ccf radial']['median'][tau_ind][ind_a], 
                        y_vector_rot['Expansion velocity ccf FLAP']['median'][tau_ind][ind_a], 
                        color=color,
                        s=4)
        ax.set_xlabel('$v_{rad}$ [km/s]')
        ax.set_ylabel('$f_{S}$')
        ax.set_title('Radial velocity vs. scaling')
        ax.text(x_text, 1.05, '(b)', transform=ax.transAxes, size=9)
        colors = iter(cm.gist_ncar(np.linspace(0, 1, len(y_vector_tra['Velocity ccf radial']['median'][tau_ind]))))
        
        #plt.subplot(gs[1,0])
        ax=axs[1,0]
        ax.plot(y_vector_tra['Distance']['median'][tau_ind],
                 y_vector_rot['Angular velocity ccf FLAP log']['median'][tau_ind],
                 lw='0.5')
        for ind_a in range(len(y_vector_tra['Velocity ccf radial']['median'][tau_ind])):
            color=copy.deepcopy(next(colors))
            ax.scatter(y_vector_tra['Distance']['median'][tau_ind][ind_a], 
                        y_vector_rot['Angular velocity ccf FLAP log']['median'][tau_ind][ind_a], 
                        color=color,
                        s=4)
        ax.set_xlabel('r-$r_{sep}$ [mm]')
        ax.set_ylabel('$\omega$ [krad/s]')
        ax.set_title('Distance vs. angular velocity')
        ax.text(x_text, 1.05, '(c)', transform=ax.transAxes, size=9)
        
        colors = iter(cm.gist_ncar(np.linspace(0, 1, len(y_vector_tra['Velocity ccf radial']['median'][tau_ind]))))
        #plt.subplot(gs[1,1])
        ax=axs[1,1]
        ax.plot(y_vector_tra['Distance']['median'][tau_ind],
                y_vector_rot['Expansion velocity ccf FLAP']['median'][tau_ind],
                lw='0.5')
        for ind_a in range(len(y_vector_tra['Velocity ccf radial']['median'][tau_ind])):
            color=copy.deepcopy(next(colors))
            ax.scatter(y_vector_tra['Distance']['median'][tau_ind][ind_a], 
                        y_vector_rot['Expansion velocity ccf FLAP']['median'][tau_ind][ind_a], 
                        color=color,
                        s=4)
        ax.set_xlabel('r-$r_{sep}$ [mm]')
        ax.set_ylabel('$f_{S}$')
        ax.set_title('Distance vs. scaling')
        ax.text(x_text, 1.05, '(d)', transform=ax.transAxes, size=9)
        plt.tight_layout()
        pdf_object.savefig()
    pdf_object.close()