#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 24 23:42:48 2022

@author: mlampert
"""
import pandas
import copy

import os
import flap
import flap_nstx
import numpy as np
import scipy

from flap_nstx.gpi import nstx_gpi_velocity_analysis_spatio_temporal_displacement
from flap_nstx.analysis import read_gpi_results, read_thomson_results
from flap_nstx.tools import calculate_corr_acceptance_levels

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MaxNLocator

thisdir = os.path.dirname(os.path.realpath(__file__))
fn = os.path.join(thisdir,"../flap_nstx.cfg")
flap.config.read(file_name=fn)
flap_nstx.register()
wd=flap.config.get_all_section('Module NSTX_GPI')['Working directory']

def analyze_sheath_induced_rotation(pdf=False,
                                    plot=False,
                                    plot_trends=False,
                                    plot_error=False,
                                    plot_correlation_matrix=False,
                                    plot_correlation_evolution=False,
                                    plot_predictive_power_score=False,
                                    plot_gpi_correlation=False,
                                    corr_threshold=0.2,
                                    corr_thres_multiplier=1,
                                    
                                    thomson_time_window=5e-3,
                                    flux_range=[0.65,1.1],
                                    elm_window=400e-6,
                                    elm_duration=100e-6,
                                    recalc_gpi=False,
                                    recalc_thomson=False,
                                    threshold_corr=False,
                                    skip_uninteresting=False,
                                    transformation=None,
                                    transformation_power=None,
                                    throw_outliers=False,
                                    return_results=False,
                                    n_sigma=2,
                                    percentile_acceptance_level=99,
                                    plot_avg_only=False,
                                    plot_lines_with_value=False,
                                    plot_for_paper=False,
                                    std_thres_outlier=3,
                                    ):
    if pdf:
        pdf_pages=PdfPages(wd+'/plots/sheath_induced_rotation_model.pdf')
        
    if pdf and not plot:
        import matplotlib
        matplotlib.use('agg')
        
    gpi_results=read_gpi_results(elm_window=elm_window,
                                 elm_duration=elm_duration,
                                 correlation_threshold=0.7,
                                 transformation=transformation, #['log','power','exp','diff',]
                                 transformation_power=transformation_power,
                                 recalc_gpi=recalc_gpi,)
    
    thomson_results=read_thomson_results(thomson_time_window=thomson_time_window,
                                         flux_range=flux_range,
                                         recalc_thomson=recalc_thomson,
                                         transformation=transformation, #['log','power','exp','diff',]
                                         transformation_power=transformation_power)
    
    data_dict={'data':[],
               'error':[],
               'derived':{},
               'coord':{},
               'label':'',
               'unit':'',
               'comment':'',
               }
    
    model_angular_velocity={'data':{},
                            'derived':{},
                            'coord':{},
                            'shot':[],
                            'elm_time':[],
                            'code':"flap_nstx.analysis.analyze_sheath_induced_rotation",
                            }
    
    key='Model angular velocity'
    model_angular_velocity['data'][key]=copy.deepcopy(data_dict)
    model_angular_velocity['data'][key]['label']='$\omega_{sheath,max}$'
    model_angular_velocity['data'][key]['unit']='rad/s'
    
    key='Experimental angular velocity'
    model_angular_velocity['data'][key]=copy.deepcopy(data_dict)
    model_angular_velocity['data'][key]['label']='$\omega_{exp,max}$'
    model_angular_velocity['data'][key]['unit']='rad/s'

    model_angular_velocity['shot']=[]
    model_angular_velocity['elm_time']=[]
    
    database_file='/Users/mlampert/work/NSTX_workspace/db/ELM_findings_mlampert_velocity_good.csv'
    db=pandas.read_csv(database_file, index_col=0)
    elm_index=list(db.index)
    
    #omega=v_rad/r_f=Er x B/B^2/r_f=-grad_phi x B/B^2/r_f=(T_core-3)/r_f * B/B^2/r_f=(T_core-3)/B/r_f^2
    for index_elm in range(len(elm_index)):
        #preprocess velocity results, tackle with np.nan and outliers
        shot=int(db.loc[elm_index[index_elm]]['Shot'])
        #define ELM time for all the cases
        elm_time=db.loc[elm_index[index_elm]]['ELM time']/1e3
        flap.delete_data_object('*')
        T_core=thomson_results['Mate']['data']['Temperature']['value_at_max_grad_pres_mtanh']['value'][index_elm]*1e3 - 3#ev
        #Geometrical average around the ELM time
        data=(np.sqrt(gpi_results['data']['Size max radial']['data'][index_elm,:]*gpi_results['data']['Size max poloidal']['data'][index_elm,:])/2.)[160-5:160+5]
        try:
            r_f=np.max(data[~np.isnan(data)])
        except:
            r_f=np.nan
        
        R_pos=gpi_results['data']['Position max radial']['data'][index_elm,160]
        #try:
        if True:
            # if (gpi_results['data']['Size max radial']['data'][index_elm,160] != 0. and 
            #     ~np.isnan(gpi_results['data']['Size max radial']['data'][index_elm,160])):
            R_sep=flap.get_data('NSTX_MDSPlus',
                                name='\EFIT02::\RBDRY',
                                exp_id=shot,
                                object_name='BZZ0').slice_data(slicing={'Time':elm_time}).data.max()
            b_pol=flap.get_data('NSTX_MDSPlus',
                                name='\EFIT02::\BZZ0',
                                exp_id=shot,
                                object_name='BZZ0').slice_data(slicing={'Time':elm_time, 
                                                                        'Device R':R_sep}).data
            b_tor=flap.get_data('NSTX_MDSPlus',
                                name='\EFIT02::\BTZ0',
                                exp_id=shot,
                                object_name='BTZ0').slice_data(slicing={'Time':elm_time, 
                                                                        'Device R':R_sep}).data
            b_rad=flap.get_data('NSTX_MDSPlus',
                                name='\EFIT02::\BRZ0',
                                exp_id=shot,
                                object_name='BRZ0').slice_data(slicing={'Time':elm_time, 
                                                                        'Device R':R_sep}).data
            B=np.sqrt(b_pol**2+b_tor**2+b_rad**2)
            # else:
            #     B=np.nan
            #     b_pol=np.nan
            #     b_tor=np.nan
            #     b_rad=np.nan                                                 
        # except:
        #     B=np.nan
        #     b_pol=np.nan
        #     b_tor=np.nan
        #     b_rad=np.nan
        print(elm_time, T_core, b_pol,b_tor, B, r_f)
        omega_model=3*T_core*np.sqrt(b_pol**2+b_tor**2)/B**2/r_f**2
        model_angular_velocity['data']['Model angular velocity']['data'].append(omega_model)
        
        omega_exp=gpi_results['data']['Angular velocity ccf FLAP log']['data'][index_elm,160]
        model_angular_velocity['data']['Experimental angular velocity']['data'].append(omega_exp)
        
    model_angular_velocity['data']['Experimental angular velocity']['data']=np.asarray(model_angular_velocity['data']['Experimental angular velocity']['data'])
    model_angular_velocity['data']['Model angular velocity']['data']=np.asarray(model_angular_velocity['data']['Model angular velocity']['data'])
    print(model_angular_velocity['data']['Experimental angular velocity']['data'])
    print(model_angular_velocity['data']['Model angular velocity']['data'])
    if pdf or plot:

        plot_x_vs_y=[['Experimental angular velocity','Model angular velocity']]
        for ind_plot in range(len(plot_x_vs_y)):
            fig,ax=plt.subplots(figsize=(8.5/2.54,8.5/2.54/np.sqrt(2)))
            xdata=model_angular_velocity['data'][plot_x_vs_y[ind_plot][0]]['data']/1e3
            ydata=model_angular_velocity['data'][plot_x_vs_y[ind_plot][1]]['data']/1e3

            ind_not_nan=np.logical_and(~np.isnan(xdata),
                                       ~np.isnan(ydata))
            
            xdata=xdata[ind_not_nan]
            
            ydata=ydata[ind_not_nan]
            
            xdata_std=np.sqrt(np.var(xdata))
            ydata_std=np.sqrt(np.var(ydata))
            
            ind_keep=np.where(np.logical_and(np.abs(xdata-np.mean(xdata))<xdata_std*std_thres_outlier,
                                             np.abs(ydata-np.mean(ydata))<ydata_std*std_thres_outlier))
            color='tab:blue'
            ax.scatter(xdata[ind_keep],
                       ydata[ind_keep],
                       marker='o',
                       color=color,
                       s=2)
            ax.set_title('Sheath induced rotation model')
            ax.set_xlabel('$\omega_{exp,t_{ELM}}$ [krad/s]')
            ax.set_ylabel('$\omega_{sheath,t_{ELM}}$ [krad/s]')
            plt.tight_layout(pad=0.1)
            if pdf:
                pdf_pages.savefig()
        if pdf:
            pdf_pages.close()
            
        if not plot:
            matplotlib.use('qt5agg')