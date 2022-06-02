#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 12:39:01 2022

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

def calculate_shear_layer_vpol(elm_time_range=200e-6,                           #Time range the shear layer calculated in (actually [t_elm-t-25us,t_elm-25us])
                               elm_time_adjust=25e-6,                           #Leave this time out from the averaging for the profile estimation
                               shear_calc_t_elm_range=[-5e-3,0e-6],             #Available as of 03/22/2022: [-200e-6,0] and [-500e-6,200e-6]
                               shear_avg_t_elm_range=[-5e-3,-200e-6],           #Average time range for the shear layer calculations
                                                                                #   should be in between shear_calc_t_elm_range+{+fbin*samplingtime,-fbin*sampling_time}   
                               sg_filter_order=11,                              #Scholasky-Golay filter order
                               fbin=10,                                         #+-binning average for the poloidal velocity calculation
                               sampling_time=2.5e-6,

                               plot_shear_profile=False,                        #Plot each shear layer profile and v'pol
                               shot_to_plot=None,                               #Plot the rotation profile of the shot or shot list
                               
                               
                               nocalc=True,                                     #Recalculate the velocity results
                               verbose=False,
                               test=False,
                               return_results=True,
                               save_data_for_publication=False,
                               ):
    
    spatial_resolution=0.00375  #m/pix
    
    database_file='/Users/mlampert/work/NSTX_workspace/db/ELM_findings_mlampert_velocity_good.csv'
    db=pandas.read_csv(database_file, index_col=0)
    elm_index=list(db.index)
    elm=0.
    if save_data_for_publication:
        raise NotImplementedError('Do your job already!!!')
        
    if plot_shear_profile:
        pdf_pages=PdfPages(wd+'/plots/edge_shear_profiles.pdf')
        import matplotlib
        matplotlib.use('agg')
        
    data_dict={'data':[],
               'error':[],
               'unit':'',
               'label':'',
               }
        
    quantity_dict={'v_pol':copy.deepcopy(data_dict),
                   'v_pol_prime':copy.deepcopy(data_dict),
                   'v_pol_prime_prime':copy.deepcopy(data_dict),
                   
                   'v_pol_smooth':copy.deepcopy(data_dict),
                   'v_pol_prime_smooth':copy.deepcopy(data_dict),
                   'v_pol_prime_prime_smooth':copy.deepcopy(data_dict),
                   }
    
    coordinate_dict={'name':None,
                     'label':None,
                     'values':None,
                     'unit':None,
                     'index':None,
                     }
    
    shear_data={'data':copy.deepcopy(quantity_dict),
                'derived':{},
                'coord':{},
                'shot':[],
                'elm_time':[],
                'code':'flap_nstx.analysis.calculate_shear_layer_vpol'
                }
    
    shear_data['data']['v_pol']['unit']='m/s'
    shear_data['data']['v_pol']['label']='$v_{pol}$'
    
    shear_data['data']['v_pol_prime']['unit']='rad/s'
    shear_data['data']['v_pol_prime']['label']='$v\'_{pol}$'
    
    shear_data['data']['v_pol_prime_prime']['unit']='$m^{-1}s^{-1}$'
    shear_data['data']['v_pol_prime_prime']['label']='$v\'\'_{pol}$'
    
    
    shear_data['data']['v_pol_smooth']['unit']='m/s'
    shear_data['data']['v_pol_smooth']['label']='$v_{pol,SG}$'
    
    shear_data['data']['v_pol_prime_smooth']['unit']='rad/s'
    shear_data['data']['v_pol_prime_smooth']['label']='$v\'_{pol,SG}$'
    
    shear_data['data']['v_pol_prime_prime_smooth']['unit']='$m^{-1}s^{-1}$'
    shear_data['data']['v_pol_prime_prime_smooth']['label']='$v\'\'_{pol,SG}$'
    
    shear_data['derived']['Separatrix radius']={}
    shear_data['derived']['Separatrix radius']['data']=[]
    shear_data['derived']['Separatrix radius']['unit']='m'
    shear_data['derived']['Separatrix radius']['label']='$R_{sep}$'
    
    shear_data['coord']['r']=copy.deepcopy(coordinate_dict)
    
    shear_data['coord']['r']['name']='Radial'
    shear_data['coord']['r']['label']='R'
    shear_data['coord']['r']['unit']='m'
    shear_data['coord']['r']['index']=1
    
    # shear_data['coord']['psin']=copy.deepcopy(coordinate_dict)
    
    # shear_data['coord']['psin']['name']='Normalized flux'
    # shear_data['coord']['psin']['label']='$\Psi_{n}$'
    # shear_data['coord']['psin']['unit']=''
    # shear_data['coord']['psin']['index']=1
    
    
    for index_elm in range(len(elm_index)):
        #preprocess velocity results, tackle with np.nan and outliers
        shot=int(db.loc[elm_index[index_elm]]['Shot'])
        #define ELM time for all the cases
        elm_time=db.loc[elm_index[index_elm]]['ELM time']/1e3
        
        flap.delete_data_object('*')
        if verbose: print('Calculating '+str(shot)+ ' at '+str(elm_time*1e3)+'ms')
        elm=elm+1
        time_range=list(elm_time+np.asarray(shear_calc_t_elm_range))
        result=nstx_gpi_velocity_analysis_spatio_temporal_displacement(exp_id=shot, 
                                                                       time_range=time_range, 
                                                                       x_search=5,
                                                                       y_search=5,
                                                                       x_range=[5,49], 
                                                                       y_range=[35,45], 
                                                                       plot=False, 
                                                                       pdf=False, 
                                                                       nocalc=nocalc,
                                                                       return_results=True,
                                                                       fbin=fbin,
                                                                       )
        
        # if n_timewin is None:
        #     n_timewin=int(elm_time_range/sampling_time)-2*fbin-int(elm_time_adjust/sampling_time)
        # shear_calc_t_elm_range+[fbin]
        ind_range=np.where(np.logical_and(result['Time'] > elm_time+shear_avg_t_elm_range[0],
                                          result['Time'] < elm_time+shear_avg_t_elm_range[1]))
        
        vpol_rad=np.mean(result['Poloidal velocity'][:,:,ind_range[0]],axis=(1,2))  #Weird bug, but [0] fixes it.
        vpol_rad_error=np.sqrt(np.var(result['Poloidal velocity'][:,:,ind_range[0]],axis=(1,2)))
        if test:
            print('vpol_rad_error/vpol_rad',
                  vpol_rad_error/vpol_rad)
            
        vpol_prime=np.gradient(vpol_rad)/spatial_resolution
        vpol_prime_error=list((vpol_rad_error[0:-2]+vpol_rad_error[2:])/2)
        vpol_prime_error.insert(0,vpol_rad_error[0])
        vpol_prime_error.append(vpol_rad_error[-1])
        vpol_prime_error=np.asarray(vpol_prime_error)
        vpol_prime_error /= spatial_resolution
        if test:
            print('vpol_prime_error/vpol_prime',
                  vpol_prime_error/vpol_prime)
            
        vpol_prime_prime=np.gradient(vpol_prime)/spatial_resolution
        vpol_prime_prime_error=list((vpol_prime_error[0:-2]+vpol_prime_error[2:])/2)
        vpol_prime_prime_error.insert(0,vpol_prime_error[0])
        vpol_prime_prime_error.append(vpol_prime_error[-1])
        vpol_prime_prime_error=np.asarray(vpol_prime_prime_error)
        vpol_prime_prime_error /= spatial_resolution
        if test:
            print('vpol_prime_prime_error/vpol_prime_prime',
                  vpol_prime_prime_error/vpol_prime_prime)
        if index_elm == 0:
            shear_data['coord']['r']['values']=result['Image x']*spatial_resolution+1.402   #1.402 is the spatial offset of the innermost pixel of the GPI
            
        vpol_rad_smooth = scipy.signal.savgol_filter(vpol_rad, sg_filter_order, sg_filter_order//2)
        vpol_rad_smooth_error = scipy.signal.savgol_filter(vpol_rad_error, sg_filter_order, sg_filter_order//2)
        
        vpol_prime_smooth = scipy.signal.savgol_filter(vpol_prime, sg_filter_order, sg_filter_order//2)
        vpol_prime_smooth_error = scipy.signal.savgol_filter(vpol_prime_error, sg_filter_order, sg_filter_order//2)
        
        vpol_prime_prime_smooth = scipy.signal.savgol_filter(vpol_prime_prime, sg_filter_order, sg_filter_order//2)
        vpol_prime_prime_smooth_error = scipy.signal.savgol_filter(vpol_prime_prime_error, sg_filter_order, sg_filter_order//2)
        
        shear_data['data']['v_pol']['data'].append(vpol_rad)
        shear_data['data']['v_pol']['error'].append(vpol_rad_error)
        
        shear_data['data']['v_pol_prime']['data'].append(vpol_prime)
        shear_data['data']['v_pol_prime']['error'].append(vpol_prime_error)
        
        shear_data['data']['v_pol_prime_prime']['data'].append(vpol_prime_prime)
        shear_data['data']['v_pol_prime_prime']['error'].append(vpol_prime_prime_error)
        
        shear_data['data']['v_pol_smooth']['data'].append(vpol_rad_smooth)
        shear_data['data']['v_pol_smooth']['error'].append(vpol_rad_smooth_error)
        
        shear_data['data']['v_pol_prime_smooth']['data'].append(vpol_prime_smooth)
        shear_data['data']['v_pol_prime_smooth']['error'].append(vpol_prime_smooth_error)
        
        shear_data['data']['v_pol_prime_prime_smooth']['data'].append(vpol_prime_prime_smooth)
        shear_data['data']['v_pol_prime_prime_smooth']['error'].append(vpol_prime_prime_smooth_error)
        
        shear_data['shot'].append(shot)
        shear_data['elm_time'].append(elm_time)
        try:
            R_sep=flap.get_data('NSTX_MDSPlus',
                                name='\EFIT02::\RBDRY',
                                exp_id=shot,
                                object_name='SEP R OBJ').slice_data(slicing={'Time':elm_time}).data
        except:
            R_sep=np.nan
            
        shear_data['derived']['Separatrix radius']['data'].append(np.max(R_sep))
        
        if plot_shear_profile:
            if (shot_to_plot is None or 
                shot in np.asarray(shot_to_plot)):
                
                fig,ax=plt.subplots(figsize=(8.5/2.54,8.5/np.sqrt(2)/2.54))
                ax.plot(shear_data['coord']['r']['values'],
                        vpol_rad_smooth/1e3,
                        label='$v_{pol}$')
                ax.set_xlabel('R [m]')
                #ax.set_ylabel('$\partial v_{pol}/\partial x$ (1/s)')
                ax.set_ylabel('$v_{pol}$ [km/s]')
                ax.set_title('$v_{pol}$ vs. R for #'+str(shot)+' @ '+str(int(elm_time*1e3))+'ms')
                ax.xaxis.set_major_locator(MaxNLocator(5)) 
                ax.yaxis.set_major_locator(MaxNLocator(5))
                
                ax2=ax.twinx()
                ax2.set_ylabel('$\partial v_{pol}/\partial x$ [$10^{3}$/s]')
    
                ax2.plot(shear_data['coord']['r']['values'],
                          vpol_prime_smooth/1e3,
                          color='orange',
                          label='$v\'_{pol}$')
                ax2.set_ylim(-np.max(np.abs(vpol_prime_smooth/1e3)),
                             np.max(np.abs(vpol_prime_smooth/1e3)))
                ax2.yaxis.set_major_locator(MaxNLocator(5))
                

                plt.axvline(np.max(R_sep),
                            color='green')
                
                plt.tight_layout(pad=0.1)
                pdf_pages.savefig()
            
    for key in shear_data['data'].keys():
        shear_data['data'][key]['data']=np.asarray(shear_data['data'][key]['data'])
        shear_data['data'][key]['error']=np.asarray(shear_data['data'][key]['error'])
        
    if plot_shear_profile:
        pdf_pages.close()
        matplotlib.use('qt5agg')
        
    if return_results:
        return shear_data

    
def analyze_shear_filament_dependence(elm_window=400e-6,
                                      elm_duration=100e-6,
                                      recalc=False
                                      ):
    
    pdf_pages=PdfPages(wd+'/plots/shear_max_vs_maximum_angular_velocity.pdf')
    
    gpi_results=read_gpi_results(elm_window=elm_window,
                                          elm_duration=elm_duration,
                                          correlation_threshold=0.7,
                                          transformation=None, #['log','power','exp','diff',]
                                          transformation_power=None,
                                          recalc_gpi=recalc,)
    angular_velocity=gpi_results['data']['Angular velocity ccf FLAP log']['data']
    
    thomson_results=read_thomson_results(thomson_time_window=20e-3,
                                         flux_range=[0.65,1.1],
                                         recalc_thomson=recalc)
    
    shear_results=calculate_shear_layer_vpol(nocalc=not recalc,
                                             n_timewin=60,
                                             sg_filter_order=21)
    max_vpol_shear=[]
    for elm_ind in range(len(thomson_results['Mate']['elm_time'])):
        # max_position=thomson_results['Mate']['data']['Pressure']['position_r']['value'][elm_ind]
        # #ind_search_max=np.where(shear_results['coord']['r']['values'] > max_position)
        try:
            max_vpol_shear.append(np.min(shear_results['data']['v_pol_prime_smooth']['data'][elm_ind,:]))
        except:
            max_vpol_shear.append(np.nan)
    max_vpol_shear=np.asarray(max_vpol_shear)
    
    max_angular_velocity=[]
    for ind in range(angular_velocity.shape[0]):
        ind_notnan=np.logical_not(np.isnan(angular_velocity[ind,:]))
        max_ang_vel=np.max(angular_velocity[ind,:][ind_notnan])
        max_angular_velocity.append(max_ang_vel)
        
    max_angular_velocity=np.asarray(max_angular_velocity)
    
    ind_not_nan=np.where(np.logical_not(np.logical_or(np.isnan(max_vpol_shear),np.isnan(max_angular_velocity))))
    
    max_angular_velocity=max_angular_velocity[ind_not_nan]
    max_vpol_shear=max_vpol_shear[ind_not_nan]
    
    fig,ax=plt.subplots(figsize=(8.5/2.54,8.5/2.54/np.sqrt(2)))
    
    ax.scatter(max_vpol_shear,
               max_angular_velocity/1e3,
               s=2)
    
    ax.set_title('Velocity shear vs. maximum $\omega$')
    ax.set_xlabel(shear_results['data']['v_pol_prime_smooth']['label']+" ["+shear_results['data']['v_pol_prime_smooth']['unit']+"]")
    ax.set_ylabel(gpi_results['data']['Angular velocity ccf FLAP log']['label']+'$_{max}$ ['+gpi_results['data']['Angular velocity ccf FLAP log']['unit']+']')
    plt.tight_layout(pad=0.1)
    pdf_pages.savefig()
    pdf_pages.close()
    
    a=max_angular_velocity-max_angular_velocity.mean()
    b=max_vpol_shear-max_vpol_shear.mean()
    print('Correlation between vpol_max and omega_max:', np.sum(a*b)/np.sqrt((np.sum(a**2)*np.sum(b**2))))
    
def calculate_shear_induced_angular_velocity(elm_window=400e-6,
                                             elm_duration=100e-6,
                                             std_thres_multiplier=1.0,
                                             std_thres_outlier=3,
                                             
                                             filament_lifetime_threshold=5*2.5e-6,      #Minimum time the filament needs to be considered to be a valid event
                                             sampling_time=2.5e-6,                      #Sampling time DAAA
                                             time_range_thres=[-50e-6,100e-6],          #The filament's movement is analyzed in this time range
                                             shear_avg_t_elm_range=[-5e-3,-200e-6],     #The time range where the shear layer results are averaged
                                             
                                             nocalc=True,
                                             return_results=False,
                                             
                                             plot=True,
                                             plot_error=False,
                                             plot_for_publication=False,
                                             plot_model_time_series=False,
                                             
                                             pdf=True,
                                             verbose=False,
                                             
                                             test_angular_velocity=False,
                                             test=False,
                                             plot_median_shear_layer=False,
                                             ):
    """
    Calculation of the time dependent and independent shear induced filament
    rotation model for the ELM filament rotation study.
    
    Angular acceleration is generally deprecated, because it is a second order
    derivative utilizing a third order derivative which introduces increadibly
    large errors.
    
    Space (time) dependent angular velocity:
        omega(t) = (v_pol'+t*v_pol''*v_rad)/(1+(v_pol'*t)**2)
    
    Space independent angular velocity:
        omega(t) = v_pol'/(1+(v_pol'*t)**2)

    Parameters
    ----------
    elm_window : float
        DESCRIPTION. The default is 400e-6.
    elm_duration : float
        DESCRIPTION. The default is 100e-6.
    nocalc : boolean
        DESCRIPTION. The default is True.
    plot : boolean
        DESCRIPTION. The default is True.
    verbose : boolean
        DESCRIPTION. The default is False.

    Returns
    -------
    None. Plots the model and experimental angular velocity results.

    """
    if pdf:
        pdf_pages=PdfPages(wd+'/plots/shear_max_vs_maximum_angular_velocity.pdf')
    if pdf and plot_model_time_series:
        pdf_time_series=PdfPages(wd+'/plots/model_vs_experimental_shear_time_series.pdf')
        
    if pdf and not plot:
        import matplotlib
        matplotlib.use('agg')
        
    gpi_results=read_gpi_results(elm_window=elm_window,
                                 elm_duration=elm_duration,
                                 correlation_threshold=0.7,
                                 transformation=None, #['log','power','exp','diff',]
                                 transformation_power=None,
                                 recalc_gpi=not nocalc,
                                 )

    shear_results=calculate_shear_layer_vpol(nocalc=nocalc,
                                             shear_avg_t_elm_range=shear_avg_t_elm_range,
                                             sg_filter_order=21,
                                             test=test)
    
    database_file='/Users/mlampert/work/NSTX_workspace/db/ELM_findings_mlampert_velocity_good.csv'
    db=pandas.read_csv(database_file, index_col=0)
    elm_index=list(db.index)
    
    if test_angular_velocity:
        pdf_pages_ang_vel_test=PdfPages(wd+'/plots/angular_acceleration_fitting_test.pdf')
    
    if pdf and plot:
        pdf_pages=PdfPages(wd+'/plots/shear_induced_rotation_model.pdf')
        
    if pdf and not plot:
        import matplotlib
        matplotlib.use('agg')
        
    coord_dict={'values':None,
                'label':None,
                'unit':None,
                'index':None}
    
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
                            'code':"flap_nstx.analysis.calculate_shear_induced_angular_velocity",
                            }
    
    key='Angular velocity time dep model'
    model_angular_velocity['data'][key]=copy.deepcopy(data_dict)
    model_angular_velocity['data'][key]['label']='$\omega_{model,t}$'
    model_angular_velocity['data'][key]['unit']='rad/s'
    
    key='Angular velocity time indep model'
    model_angular_velocity['data'][key]=copy.deepcopy(data_dict)
    model_angular_velocity['data'][key]['label']='$\omega_{model}$'
    model_angular_velocity['data'][key]['unit']='rad/s'
    
    key='Shearing rate time dep avg'
    model_angular_velocity['data'][key]=copy.deepcopy(data_dict)
    model_angular_velocity['data'][key]['label']='$v^{,}_{pol,t,avg}$'
    model_angular_velocity['data'][key]['unit']='rad/s'
    
    key='Shearing rate time indep avg'
    model_angular_velocity['data'][key]=copy.deepcopy(data_dict)
    model_angular_velocity['data'][key]['label']='$v\'_{pol,avg}$'
    model_angular_velocity['data'][key]['unit']='rad/s'
    
    key='Shearing rate time dep max neg'
    model_angular_velocity['data'][key]=copy.deepcopy(data_dict)
    model_angular_velocity['data'][key]['label']='$v\'_{pol,t,min}$'
    model_angular_velocity['data'][key]['unit']='rad/s'
    
    key='Shearing rate time dep max pos'
    model_angular_velocity['data'][key]=copy.deepcopy(data_dict)
    model_angular_velocity['data'][key]['label']='$v\'_{pol,t,max}$'
    model_angular_velocity['data'][key]['unit']='rad/s'
    
    key='Shearing rate time indep max'
    model_angular_velocity['data'][key]=copy.deepcopy(data_dict)
    model_angular_velocity['data'][key]['label']='$v\'_{pol,max}$'
    model_angular_velocity['data'][key]['unit']='rad/s'    
    
    key='Angular velocity experimental max neg'
    model_angular_velocity['data'][key]=copy.deepcopy(data_dict)
    model_angular_velocity['data'][key]['label']='$\omega_{exp,min}$'
    model_angular_velocity['data'][key]['unit']='rad/s'
    
    key='Angular velocity experimental max pos'
    model_angular_velocity['data'][key]=copy.deepcopy(data_dict)
    model_angular_velocity['data'][key]['label']='$\omega_{exp,max}$'
    model_angular_velocity['data'][key]['unit']='rad/s'
    
    key='Angular velocity experimental max neg change t0'
    model_angular_velocity['data'][key]=copy.deepcopy(data_dict)
    model_angular_velocity['data'][key]['label']='$\Delta_{t_{0}} \omega_{exp,min}$'
    model_angular_velocity['data'][key]['unit']='rad/s'
    
    key='Angular velocity experimental max pos change t0'
    model_angular_velocity['data'][key]=copy.deepcopy(data_dict)
    model_angular_velocity['data'][key]['label']='$\Delta_{t_{0}} \omega_{exp,max}$'
    model_angular_velocity['data'][key]['unit']='rad/s'
    
    key='Angular velocity experimental avg'
    model_angular_velocity['data'][key]=copy.deepcopy(data_dict)
    model_angular_velocity['data'][key]['label']='$\omega_{exp,avg}$'
    model_angular_velocity['data'][key]['unit']='rad/s'
    
    key='Angular velocity experimental avg change t0'
    model_angular_velocity['data'][key]=copy.deepcopy(data_dict)
    model_angular_velocity['data'][key]['label']='$\Delta_{t_{0}} \omega_{exp,avg}$'
    model_angular_velocity['data'][key]['unit']='rad/s'
    
    key='Form factor time dep max'
    model_angular_velocity['data'][key]=copy.deepcopy(data_dict)
    model_angular_velocity['data'][key]['label']='$f_{vpol,t,max}$'
    model_angular_velocity['data'][key]['unit']=''
    
    key='Form factor time dep avg'
    model_angular_velocity['data'][key]=copy.deepcopy(data_dict)
    model_angular_velocity['data'][key]['label']='$f_{vpol,t,avg}$'
    model_angular_velocity['data'][key]['unit']=''
    
    key='Form factor time indep max'
    model_angular_velocity['data'][key]=copy.deepcopy(data_dict)
    model_angular_velocity['data'][key]['label']='$f_{vpol,max}$'
    model_angular_velocity['data'][key]['unit']=''
    
    key='Form factor time indep avg'
    model_angular_velocity['data'][key]=copy.deepcopy(data_dict)
    model_angular_velocity['data'][key]['label']='$f_{vpol,avg}$'
    model_angular_velocity['data'][key]['unit']=''
    
    key='First term over second term'
    model_angular_velocity['data'][key]=copy.deepcopy(data_dict)
    model_angular_velocity['data'][key]['label']='$\omega_{1}/\omega_{2}$'
    model_angular_velocity['data'][key]['unit']=''
    
    key='Average time series exp'
    model_angular_velocity['data'][key]=copy.deepcopy(data_dict)
    model_angular_velocity['data'][key]['label']='$\omega_{exp,median}$'
    model_angular_velocity['data'][key]['unit']='rad/s'
    
    key='Average time series model'
    model_angular_velocity['data'][key]=copy.deepcopy(data_dict)
    model_angular_velocity['data'][key]['label']='$\omega_{mod,median}$'
    model_angular_velocity['data'][key]['unit']='rad/s'
    
    key='Full time series model'
    model_angular_velocity['data'][key]=copy.deepcopy(data_dict)
    model_angular_velocity['data'][key]['label']='$\omega_{mod,full}$'
    model_angular_velocity['data'][key]['unit']='rad/s'
    
    key='Filament time range'
    model_angular_velocity['derived'][key]=copy.deepcopy(data_dict)
    model_angular_velocity['derived'][key]['label']='Time'
    model_angular_velocity['derived'][key]['unit']='s'
    
    key_derived='duration'
    model_angular_velocity['derived'][key]['derived'][key_derived]=copy.deepcopy(data_dict)
    model_angular_velocity['derived'][key]['derived'][key_derived]['label']='$\Delta t$'
    model_angular_velocity['derived'][key]['derived'][key_derived]['unit']='s'
    
    key='Filament radial range'
    model_angular_velocity['derived'][key]=copy.deepcopy(data_dict)
    model_angular_velocity['derived'][key]['label']='$\Delta r t$'
    model_angular_velocity['derived'][key]['unit']='m'
    
    key='Filament total rotation angle'
    model_angular_velocity['derived'][key]=copy.deepcopy(data_dict)
    model_angular_velocity['derived'][key]['label']='$\\Delta \\theta t$'
    model_angular_velocity['derived'][key]['unit']='deg'
    
    key='t'
    model_angular_velocity['coord'][key]=copy.deepcopy(coord_dict)
    model_angular_velocity['coord'][key]['unit']='s'
    model_angular_velocity['coord'][key]['label']='Time'
    model_angular_velocity['coord'][key]['index']=1
    
    exclude_keys_from_nans=['Average time series exp',
                            'Average time series model',
                            ]
    
    time_vec=gpi_results['coord']['time']['values']
    
    for index_elm in range(len(elm_index)):
        #preprocess velocity results, tackle with np.nan and outliers
        shot=int(db.loc[elm_index[index_elm]]['Shot'])
        #define ELM time for all the cases
        elm_time=db.loc[elm_index[index_elm]]['ELM time']/1000.
        model_angular_velocity['shot'].append(shot)
        model_angular_velocity['elm_time'].append(elm_time)
        flap.delete_data_object('*')
        if verbose: print('Calculating '+str(shot)+ ' at '+str(elm_time*1e3)+'ms')
        
        """
        Filament lifetime calculation
        """
        
        
        #Calculate the lifeline of the ELM filament
        #A threshold is calculated from the one standard deviation of the poloidal position
        position_gradient=np.diff(gpi_results['data']['Position max poloidal']['data'][index_elm,:])        
        ind_not_nan=np.logical_not(np.isnan(position_gradient))
        stddev=np.sqrt(np.var(position_gradient[ind_not_nan]))
        threshold=np.mean(np.abs(position_gradient[ind_not_nan]))+std_thres_multiplier*stddev
        ind=np.where(np.abs(position_gradient)>threshold)
        
        #Last index before and first index after the t_elm index in the middle, these two should be indicative of the filament's path
        ind1=np.asarray(ind)[(np.where(np.asarray(ind) < gpi_results['coord']['time']['values'].shape[0]//2))][-1]+1 
        ind2=np.asarray(ind)[(np.where(np.asarray(ind) > gpi_results['coord']['time']['values'].shape[0]//2))][0] 
        #Limiting the filament lifetime to a thresholded range
        ind_thres=np.asarray(np.asarray(time_range_thres)/sampling_time+gpi_results['coord']['time']['values'].shape[0]//2,dtype=int)
        
        if ind1 < ind_thres[0]: ind1=ind_thres[0]
        if ind2 > ind_thres[1]: ind2=ind_thres[1]
        #Filament radial path
        filament_path_radial=gpi_results['data']['Position max radial']['data'][index_elm,ind1:ind2+1]
        #Filament lifetime range
        time_range=[gpi_results['coord']['time']['values'][ind1],
                    gpi_results['coord']['time']['values'][ind2]]
        time_duration=time_range[1]-time_range[0]
        
        if time_duration > filament_lifetime_threshold:

            model_angular_velocity['derived']['Filament time range']['data'].append([time_range])
            model_angular_velocity['derived']['Filament time range']['error'].append(sampling_time)
            model_angular_velocity['derived']['Filament time range']['derived']['duration']['data'].append(time_duration)
            model_angular_velocity['derived']['Filament time range']['derived']['duration']['error'].append(2*sampling_time)
            model_angular_velocity['derived']['Filament radial range']['data'].append(np.asarray([min(filament_path_radial),
                                                                                                  max(filament_path_radial)]))
                                                                                      
            """
            TIME INDEPENDENT CALCULATION:    omega(t) = v_pol'/(1+(v_pol'*t)**2)
            """
            
            ind_not_nan_v_pol_prime = ~np.isnan(shear_results['data']['v_pol_prime_smooth']['data'][index_elm,:])
            ind_not_nan_v_pol_prime_prime = ~np.isnan(shear_results['data']['v_pol_prime_prime_smooth']['data'][index_elm,:])
            
            v_pol_prime=np.interp(gpi_results['data']['Position max radial']['data'][index_elm,ind1:ind2+1],
                                  shear_results['coord']['r']['values'][ind_not_nan_v_pol_prime],
                                  shear_results['data']['v_pol_prime_smooth']['data'][index_elm,:][ind_not_nan_v_pol_prime])
            v_pol_prime_error=np.interp(gpi_results['data']['Position max radial']['data'][index_elm,ind1:ind2+1],
                                        shear_results['coord']['r']['values'][ind_not_nan_v_pol_prime],
                                        shear_results['data']['v_pol_prime_smooth']['error'][index_elm,:][ind_not_nan_v_pol_prime])
    
            ind_not_nan = ~np.isnan(v_pol_prime)
            v_pol_prime = v_pol_prime[ind_not_nan]
            v_pol_prime_error=v_pol_prime_error[ind_not_nan]
            shear_avg=np.mean(v_pol_prime) #Avg shear rate in the range where the filament's path is
            # try:
            #     maxind=np.argmax(np.abs(v_pol_prime)) #Avg shear rate in the range where the filament's path is
            #     shear_avg=v_pol_prime[maxind]
            # except:
            #     shear_avg=np.nan
            time_duration_error=sampling_time  #Conservative estimate, one frame error 
            shear_avg_error=np.sqrt(np.var(v_pol_prime))
            
            omega_model = shear_avg/(1+(shear_avg*time_duration)**2)
            omega_model_error=(np.abs((1+(shear_avg*time_duration)**2)+shear_avg*(1+2*shear_avg*time_duration**2)/(1+(shear_avg*time_duration)**2)**2)*shear_avg_error+
                               np.abs(shear_avg/(1+(shear_avg*time_duration)**2)**2*2*time_duration*shear_avg**2)*time_duration_error)
            
            model_angular_velocity['data']['Angular velocity time indep model']['data'].append(omega_model)
            model_angular_velocity['data']['Angular velocity time indep model']['error'].append(omega_model_error)
            
            model_angular_velocity['data']['Shearing rate time indep avg']['data'].append(shear_avg)
            model_angular_velocity['data']['Shearing rate time indep avg']['error'].append(shear_avg_error)
            try:
                ind_max=np.argmax(np.abs(v_pol_prime))
                model_angular_velocity['data']['Shearing rate time indep max']['data'].append(v_pol_prime[ind_max])
                model_angular_velocity['data']['Shearing rate time indep max']['error'].append(v_pol_prime_error[ind_max])
            except:
                model_angular_velocity['data']['Shearing rate time indep max']['data'].append(np.nan)
                model_angular_velocity['data']['Shearing rate time indep max']['error'].append(np.nan)
            
            
            """
            TIME (SPACE) DEPENDENT CALCULATION:     
                Small angles: omega(t) = (v_pol'+t*v_pol''*v_rad)
                                          
                Large angles: omega(t) = (v_pol'+t*v_pol''*v_rad)/(1+(v_pol'*t)**2)
            """
            
            nwin_path=len(filament_path_radial)
            omega_model=np.zeros(nwin_path)
            omega_model_error=np.zeros(nwin_path)
            
            shear_tdep_model=np.zeros(nwin_path)
            shear_tdep_model_error=np.zeros(nwin_path)
            
            shear_tdep_model_1st_term=np.zeros(nwin_path)
            shear_tdep_model_2nd_term=np.zeros(nwin_path)
            
            
            for ind_time in range(ind1,ind2+1):
                time_prop = (ind_time-ind1) * sampling_time
                time_prop_error=2.1e-6/2 # Should be asymmetric, corresponds to +- half the exposition rate
                
                v_rad=gpi_results['data']['Velocity ccf FLAP radial']['data'][index_elm,ind_time]
                v_rad_error=0.00375/2.5e-6 #Pixel size / sampling time
                
                #v_rad=np.gradient(gpi_results['data']['Position max radial']['data'][index_elm,:])[ind_time]/sampling_time
                #a_rad=np.gradient(np.gradient(gpi_results['data']['Position max radial']['data'][index_elm,:]))[ind_time]/sampling_time**2
                
                #Interpolating the shearing rate for the filament position
                v_pol_prime=np.interp(gpi_results['data']['Position max radial']['data'][index_elm,ind_time],
                                      shear_results['coord']['r']['values'][ind_not_nan_v_pol_prime],
                                      shear_results['data']['v_pol_prime_smooth']['data'][index_elm,:][ind_not_nan_v_pol_prime])
                
                v_pol_prime_error=np.interp(gpi_results['data']['Position max radial']['data'][index_elm,ind_time],
                                            shear_results['coord']['r']['values'][ind_not_nan_v_pol_prime],
                                            shear_results['data']['v_pol_prime_smooth']['error'][index_elm,:][ind_not_nan_v_pol_prime])
                
                #Interpolating the shearing rate for the filament position
                v_pol_prime_prime=np.interp(gpi_results['data']['Position max radial']['data'][index_elm,ind_time],
                                            shear_results['coord']['r']['values'][ind_not_nan_v_pol_prime_prime],
                                            shear_results['data']['v_pol_prime_prime_smooth']['data'][index_elm,:][ind_not_nan_v_pol_prime_prime])
                
                v_pol_prime_prime_error=np.interp(gpi_results['data']['Position max radial']['data'][index_elm,ind_time],
                                                  shear_results['coord']['r']['values'][ind_not_nan_v_pol_prime_prime],
                                                  shear_results['data']['v_pol_prime_prime_smooth']['error'][index_elm,:][ind_not_nan_v_pol_prime_prime])

                
                if test:
                    print('v_pol_prime_delta_error',v_pol_prime_error/v_pol_prime)
                    print('v_rad_delta_error',v_rad_error/v_rad)
                    print('v_pol_prime_prime_delta_error',v_pol_prime_prime_error/v_pol_prime_prime)
                    print('time_prop_delta_error',time_prop_error/time_prop)
                    print('\n')
                    
                #Calculating the time dependent model
                f=(v_pol_prime + 0*time_prop*v_pol_prime_prime*v_rad)
                
                delta_f=(v_pol_prime_error+
                         np.abs(v_pol_prime_prime*v_rad)*time_prop_error+
                         np.abs(1+time_prop*v_rad)*v_pol_prime_prime_error+
                         np.abs(time_prop*v_pol_prime_prime)*v_rad_error)
                
                g=(1+(v_pol_prime*time_prop)**2)
                delta_g=np.abs(2*v_pol_prime*time_prop**2)*v_pol_prime_error+np.abs(2*v_pol_prime**2*time_prop)*time_prop_error
                omega_model[ind_time-ind1]=f/g
                omega_model_error[ind_time-ind1]=np.abs(delta_f/g)+np.abs(f/g**2)*delta_g
                
                #The time dependent shear is dphi/dt=d(vpol't)/dt=v_pol_prime=t*v_pol_prime_prime*v_rad)
                shear_tdep_model[ind_time-ind1]=v_pol_prime
                shear_tdep_model_1st_term[ind_time-ind1]=v_pol_prime
                shear_tdep_model_2nd_term[ind_time-ind1]=time_prop*v_pol_prime_prime*v_rad
                
                shear_tdep_model_error[ind_time-ind1]=delta_f
                
            if test:
                print(omega_model_error/omega_model)
                
            
            #Should be checked whether the angular valocity should be integrated here
            #model_angular_velocity['data']['Angular velocity time dep model']['data'].append(np.sum(omega_model))
            ind_not_nan = ~np.isnan(omega_model)
            try:
                ind_max=np.argmin((omega_model[ind_not_nan]))
                model_angular_velocity['data']['Angular velocity time dep model']['data'].append(omega_model[ind_not_nan][ind_max])
                model_angular_velocity['data']['Angular velocity time dep model']['error'].append(omega_model_error[ind_not_nan][ind_max])
            except:
                model_angular_velocity['data']['Angular velocity time dep model']['data'].append(np.nan)
                model_angular_velocity['data']['Angular velocity time dep model']['error'].append(np.nan)
            
            #try:
            if True:
                first_average=np.mean(shear_tdep_model_1st_term[ind_not_nan])
                second_average=np.mean(shear_tdep_model_2nd_term[ind_not_nan])
                model_angular_velocity['data']['First term over second term']['data'].append(first_average/second_average)
            #except:
            #    model_angular_velocity['data']['First term over second term']['data'].append(np.nan)
            try:
                ind_max=np.argmin((shear_tdep_model[ind_not_nan]))
                model_angular_velocity['data']['Shearing rate time dep max neg']['data'].append(shear_tdep_model[ind_not_nan][ind_max])
                model_angular_velocity['data']['Shearing rate time dep max neg']['error'].append(shear_tdep_model_error[ind_not_nan][ind_max])
            except:
                model_angular_velocity['data']['Shearing rate time dep max neg']['data'].append(np.nan)
                model_angular_velocity['data']['Shearing rate time dep max neg']['error'].append(np.nan)
            
            try:
                ind_max=np.argmax((shear_tdep_model[ind_not_nan]))
                model_angular_velocity['data']['Shearing rate time dep max pos']['data'].append(shear_tdep_model[ind_not_nan][ind_max])
                model_angular_velocity['data']['Shearing rate time dep max pos']['error'].append(shear_tdep_model_error[ind_not_nan][ind_max])
            except:
                model_angular_velocity['data']['Shearing rate time dep max pos']['data'].append(np.nan)
                model_angular_velocity['data']['Shearing rate time dep max pos']['error'].append(np.nan)    
            
            if plot_model_time_series:
                time_vec=gpi_results['coord']['time']['values'][ind1:ind2+1]
                omega_exp=gpi_results['data']['Angular velocity ccf FLAP log']['data'][index_elm,ind1:ind2+1]
                fig,ax=plt.subplots(figsize=(8.5/2.54,8.5/2.54/np.sqrt(2)))
                
                ax.plot(time_vec*1e6,shear_tdep_model/1e3, label='$\omega_{shear}$')
                ax.plot(time_vec*1e6,omega_model/1e3, label='$\omega_{model}$')
                ax.plot(time_vec*1e6,omega_exp/1e3, label='$\omega_{exp}$')
                
                ax.set_xlabel('$t-t_{ELM}$ [${\mu s}$]')
                ax.set_ylabel('$\omega [krad/s]$')
                ax.set_title('Angular velocities')
                ax.legend()
                plt.tight_layout(pad=0.1)
                pdf_time_series.savefig()
                ind_not_nan=np.where(np.logical_and(~np.isnan(omega_exp),~np.isnan(shear_tdep_model)))
                
                shear_tdep_model_full_time=np.zeros(len(gpi_results['coord']['time']['values']))
                shear_tdep_model_full_time[ind1:ind2+1]=shear_tdep_model
                model_angular_velocity['data']['Average time series model']['data'].append(list(shear_tdep_model_full_time))

                data_ang_vel=np.zeros(len(gpi_results['coord']['time']['values']))
                data_ang_vel[ind1:ind2+1]=gpi_results['data']['Angular velocity ccf FLAP log']['data'][index_elm,ind1:ind2+1]
                model_angular_velocity['data']['Average time series exp']['data'].append(list(data_ang_vel))
                
            try:
                model_angular_velocity['data']['Shearing rate time dep avg']['data'].append(np.mean(shear_tdep_model[ind_not_nan]))
                model_angular_velocity['data']['Shearing rate time dep avg']['error'].append(np.mean(shear_tdep_model_error[ind_not_nan])/np.sqrt(np.sum(ind_not_nan)))
            except:
                model_angular_velocity['data']['Shearing rate time dep avg']['data'].append(np.nan)
                model_angular_velocity['data']['Shearing rate time dep avg']['error'].append(np.nan)

            omega_exp=gpi_results['data']['Angular velocity ccf FLAP log']['data'][index_elm,ind1:ind2+1]
            ind_not_nan_omega_exp = ~np.isnan(omega_exp)
            
            try:
                ind_max=np.argmax(omega_exp[ind_not_nan_omega_exp])
                omega_max=omega_exp[ind_not_nan_omega_exp][ind_max]
                
                ind_min=np.argmin(omega_exp[ind_not_nan_omega_exp])
                omega_min=omega_exp[ind_not_nan_omega_exp][ind_min]
                
                omega_avg=np.mean(omega_exp[ind_not_nan_omega_exp])
                omega_t0 = np.mean(omega_exp[ind_not_nan_omega_exp][0:1])
            except:
                omega_max=np.nan
                omega_min=np.nan
                omega_avg=np.nan
                omega_t0=np.nan
            
            model_angular_velocity['data']['Angular velocity experimental max neg']['data'].append(omega_min)
            model_angular_velocity['data']['Angular velocity experimental max neg']['error'].append(np.pi/180.*sampling_time)    
            
            model_angular_velocity['data']['Angular velocity experimental max pos']['data'].append(omega_max)
            model_angular_velocity['data']['Angular velocity experimental max pos']['error'].append(np.pi/180.*sampling_time)
            
            model_angular_velocity['data']['Angular velocity experimental max neg change t0']['data'].append(omega_min-omega_t0)
            model_angular_velocity['data']['Angular velocity experimental max neg change t0']['error'].append(np.pi/180./sampling_time)

            model_angular_velocity['data']['Angular velocity experimental max pos change t0']['data'].append(omega_max-omega_t0)
            model_angular_velocity['data']['Angular velocity experimental max pos change t0']['error'].append(np.pi/180./sampling_time)
            
            model_angular_velocity['data']['Angular velocity experimental avg']['data'].append(omega_avg)
            model_angular_velocity['data']['Angular velocity experimental avg']['error'].append(np.pi/180./sampling_time)
            
            model_angular_velocity['data']['Angular velocity experimental avg change t0']['data'].append(omega_avg-omega_t0)
            model_angular_velocity['data']['Angular velocity experimental avg change t0']['error'].append(np.pi/180./sampling_time)

            
            try:
                model_angular_velocity['derived']['Filament rotation angle']['data'].append(np.sum(omega_exp[ind_not_nan_omega_exp]*sampling_time))
                model_angular_velocity['derived']['Filament rotation angle']['error'].append(np.pi/180./sampling_time*np.sqrt(np.sum(ind_not_nan_omega_exp)))
            except:
                model_angular_velocity['derived']['Filament total rotation angle']['data'].append(np.nan)
                model_angular_velocity['derived']['Filament total rotation angle']['error'].append(np.nan)
                
            for key1 in ['dep','indep']:
                for key2 in ['avg','max']:
                    try:
                        form_factor=(model_angular_velocity['data']['Angular velocity experimental '+key2]['data'][-1]/
                                     model_angular_velocity['data']['Shearing rate time '+key1+' '+key2]['data'][-1])
                        model_angular_velocity['data']['Form factor time '+key1+' '+key2]['data'].append(form_factor)
                        
                        form_factor_error_1=np.abs(form_factor*model_angular_velocity['data']['Angular velocity experimental '+key2]['error'][-1]/model_angular_velocity['data']['Angular velocity experimental '+key2]['data'][-1])
                        form_factor_error_2=np.abs(form_factor*model_angular_velocity['data']['Shearing rate time '+key1+' '+key2]['error'][-1]/model_angular_velocity['data']['Shearing rate time '+key1+' '+key2]['data'][-1])
                        model_angular_velocity['data']['Form factor time '+key1+' '+key2]['error'].append(form_factor_error_1+form_factor_error_2)
                    except:
                        model_angular_velocity['data']['Form factor time '+key1+' '+key2]['data'].append(np.nan)
                        model_angular_velocity['data']['Form factor time '+key1+' '+key2]['error'].append(np.nan)

        else:   #If the filament lifetime is too short (default:<12.5us (5frames))
            for key in model_angular_velocity['data'].keys():
                if key not in exclude_keys_from_nans:
                    model_angular_velocity['data'][key]['data'].append(np.nan)
                    model_angular_velocity['data'][key]['error'].append(np.nan)
                
            model_angular_velocity['derived']['Filament time range']['data'].append(np.nan)
            model_angular_velocity['derived']['Filament time range']['derived']['duration']['data'].append(np.nan)
            
            model_angular_velocity['derived']['Filament radial range']['data'].append([np.nan,np.nan])
            model_angular_velocity['derived']['Filament radial range']['error'].append([np.nan,np.nan])
            
            model_angular_velocity['derived']['Filament total rotation angle']['data'].append(np.nan)
            model_angular_velocity['derived']['Filament total rotation angle']['error'].append(np.nan)     
            
            
            
    #Transforming everything to numpy arrays (numpy cannot append to the input array)
    for key in model_angular_velocity['data'].keys():
        model_angular_velocity['data'][key]['data']=np.asarray(model_angular_velocity['data'][key]['data'])
        model_angular_velocity['data'][key]['error']=np.asarray(model_angular_velocity['data'][key]['error'])
    
    model_angular_velocity['derived']['Filament time range']['data']=np.asarray(model_angular_velocity['derived']['Filament time range']['data'])
    model_angular_velocity['derived']['Filament time range']['derived']['duration']['data']=np.asarray(model_angular_velocity['derived']['Filament time range']['derived']['duration']['data'])
    model_angular_velocity['derived']['Filament time range']['derived']['duration']['error']=np.asarray(model_angular_velocity['derived']['Filament time range']['derived']['duration']['error'])    
    
    
    if return_results:
        return model_angular_velocity
    
    if pdf or plot:

        plot_x_vs_y=[['Angular velocity experimental max','Form factor time indep max'],
                     ['Angular velocity experimental max','Form factor time dep max'],
                     ['Angular velocity experimental avg','Form factor time indep avg'],
                     ['Angular velocity experimental avg','Form factor time dep avg'],
                     # ['Angular velocity experimental','Angular velocity time indep model'],
                     # ['Angular velocity experimental','Angular velocity time dep model'],
                     
                     ['Angular velocity experimental max change t0','Form factor time indep avg'],
                     ['Angular velocity experimental max change t0','Form factor time dep avg'],
                     
                     ['Angular velocity experimental avg change t0','Form factor time indep avg'],
                     ['Angular velocity experimental avg change t0','Form factor time dep avg'],
                     ]
        
        if plot_for_publication:
            plot_x_vs_y=[['Angular velocity experimental avg change t0','Shearing rate time dep avg'],
                         ['Angular velocity experimental avg change t0','Angular velocity time dep model'],
                         
                         ['Angular velocity experimental max neg change t0','Shearing rate time dep max neg'],
                         ['Angular velocity experimental max pos change t0','Shearing rate time dep max pos'],
                         ]
        
        for ind_plot in range(len(plot_x_vs_y)):
            fig,ax=plt.subplots(figsize=(8.5/2.54,8.5/2.54/np.sqrt(2)))
            xdata=model_angular_velocity['data'][plot_x_vs_y[ind_plot][0]]['data']/1e3
            ydata=model_angular_velocity['data'][plot_x_vs_y[ind_plot][1]]['data']/1e3
            
            x_err=model_angular_velocity['data'][plot_x_vs_y[ind_plot][0]]['error']/1e3
            y_err=model_angular_velocity['data'][plot_x_vs_y[ind_plot][1]]['error']/1e3
            ind_not_nan=np.logical_and(~np.isnan(xdata),
                                       ~np.isnan(ydata))
            
            xdata=xdata[ind_not_nan]
            x_err=x_err[ind_not_nan]
            
            ydata=ydata[ind_not_nan]
            y_err=y_err[ind_not_nan]
            
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
            # ax.set_yscale("symlog")
            # ax.yaxis.set_major_locator(MaxNLocator(11))
            # ax.set_yticks(ticks=[-10,-5,-1,-0.5,-0.1,0,0.1,0.5,1,5,10])
            #ax.set_ylim([-2,2])
            """
            Linear fitting. Not quite indicative of the correlation.
            """
            # coeff=np.polyfit(xdata[ind_keep],ydata[ind_keep],1)
            # ax.plot(xdata[ind_keep],
            #         coeff[1]*xdata[ind_keep]+coeff[0],
            #         color='red')
            """
            Correlation calculation between the experimental and the model.
            """
            x_new=xdata[ind_keep]-np.mean(xdata[ind_keep])
            y_new=ydata[ind_keep]-np.mean(ydata[ind_keep])
            
            corr_coeff=np.sum(x_new*y_new)/(np.sqrt(np.sum(x_new**2)*np.sum(y_new**2)))
            corr_accept=calculate_corr_acceptance_levels()

            ind=int(len(xdata[ind_keep]))
            sign_level=corr_accept['avg'][ind]+corr_accept['stddev'][ind]*2
            
            print('\n'+plot_x_vs_y[ind_plot][0]+' vs. '+plot_x_vs_y[ind_plot][1]+' corr. coeff.: '+str(corr_coeff)+' sign. level:'+str(sign_level)+'\n')
            
            """
            Errorbar plotting
            """
            if plot_error:
                ax.errorbar(xdata[ind_keep],
                            ydata[ind_keep],
                            xerr=x_err[ind_keep],
                            yerr=y_err[ind_keep],
                            marker='o',
                            ls='',
                            color=color,
                            )
                
            ax.set_title(plot_x_vs_y[ind_plot][0]+' vs.\n'+plot_x_vs_y[ind_plot][1])
            ax.set_xlabel(model_angular_velocity['data'][plot_x_vs_y[ind_plot][0]]['label']+' [k'+model_angular_velocity['data'][plot_x_vs_y[ind_plot][0]]['unit']+']')
            if model_angular_velocity['data'][plot_x_vs_y[ind_plot][1]]['unit'] =='':
                ax.set_ylabel(model_angular_velocity['data'][plot_x_vs_y[ind_plot][1]]['label'])
            else:
                ax.set_ylabel(model_angular_velocity['data'][plot_x_vs_y[ind_plot][1]]['label']+' [k'+model_angular_velocity['data'][plot_x_vs_y[ind_plot][1]]['unit']+']')    
            
            plt.tight_layout(pad=0.1)
            plt.show()
            
            if pdf:
                pdf_pages.savefig()     
        
        if test_angular_velocity and pdf:
            pdf_pages_ang_vel_test.close()
            
        if pdf:
            pdf_pages.close()
            
        if plot_model_time_series and pdf:
            pdf_time_series.close()
            
        if not plot:
            matplotlib.use('qt5agg')
    
    if plot_median_shear_layer:
         shear=model_angular_velocity
         
         median_sep_r=np.median(shear['derived']['Separatrix radius']['data'])
         
         
         median_shearing_rate=np.percentile(shear['data']['v_pol_prime']['data'], 50,axis=0)/1e3
         ind_max=np.argmax(median_shearing_rate)
         p=np.polyfit(shear['coord']['r']['values'][0:ind_max+1],
                      median_shearing_rate[0:ind_max+1],
                      1)
         
         pdf_pages=PdfPages(wd+'/plots/fig_median_shear_layer.pdf')
         fig,ax=plt.subplots(figsize=(8.5/2.54,8.5/2.54/np.sqrt(2)))
         
         ax.plot(shear['coord']['r']['values'],
                 np.percentile(shear['data']['v_pol_prime']['data'], 50,axis=0)/1e3)
         ax.plot(shear['coord']['r']['values'],
                 np.percentile(shear['data']['v_pol_prime']['data'], 10,axis=0)/1e3,
                 color='red')
         ax.plot(shear['coord']['r']['values'],
                 np.percentile(shear['data']['v_pol_prime']['data'], 90,axis=0)/1e3,
                 color='red')
         x_coord_linear=shear['coord']['r']['values'][0:ind_max+1]
         ax.plot(x_coord_linear,
                 p[0]*x_coord_linear+p[1],
                 color='purple')
         
         
         ax.axvline(median_sep_r, 
                    color='green')
         
         ax.set_xlabel('R [m]')
         ax.set_ylabel('$v^{,}_{pol} [s^{-3}]$')
         ax.set_title('Median shearing rate profile')
         plt.tight_layout(pad=0.1)
         pdf_pages.savefig()
         pdf_pages.close()


def analyze_shear_distribution(pdf=False,
                               plot=False,
                               return_results=False,
                               elm_window=400e-6,
                               elm_duration=100e-6,
                               nocalc=False,
                               test=False,
                               shear_avg_t_elm_range=[5e-3,0e-3],
                               n_hist=50,
                               ):
    if pdf:
        pdf_pages=PdfPages(wd+'/plots/shear_model_distribution.pdf')
        
    if pdf and not plot:
        import matplotlib
        matplotlib.use('agg')
    database_file='/Users/mlampert/work/NSTX_workspace/db/ELM_findings_mlampert_velocity_good.csv'
    db=pandas.read_csv(database_file, index_col=0)
    elm_index=list(db.index)
    
    if pdf and plot:
        pdf_pages=PdfPages(wd+'/plots/shear_induced_rotation_model.pdf')
        
    if pdf and not plot:
        import matplotlib
        matplotlib.use('agg')
    
        
    gpi_results=read_gpi_results(elm_window=500e-6,
                                 elm_duration=100e-6,
                                 correlation_threshold=0.7,
                                 transformation=None, #['log','power','exp','diff',]
                                 transformation_power=None,
                                 recalc_gpi=not nocalc,
                                 )

    shear_results=calculate_shear_layer_vpol(nocalc=nocalc,
                                             shear_avg_t_elm_range=[-5e-3,-200e-6],
                                             sg_filter_order=21,
                                             test=test)
    
            
    coord_dict={'values':None,
                'label':None,
                'unit':None,
                'index':None}
    
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
                            'code':"flap_nstx.analysis.calculate_shear_induced_angular_velocity",
                            }
    
    key='Model angular velocity time series'
    model_angular_velocity['data'][key]=copy.deepcopy(data_dict)
    model_angular_velocity['data'][key]['label']='$\omega_{time,series}$'
    model_angular_velocity['data'][key]['unit']='rad/s'
    
    key='10th percentile'
    model_angular_velocity['derived'][key]=copy.deepcopy(data_dict)
    model_angular_velocity['derived'][key]['label']='$\omega_{model}$ 10th percentile'
    model_angular_velocity['derived'][key]['unit']='rad/s'
    
    key='90th percentile'
    model_angular_velocity['derived'][key]=copy.deepcopy(data_dict)
    model_angular_velocity['derived'][key]['label']='$\omega_{model}$ 90th percentile'
    model_angular_velocity['derived'][key]['unit']='rad/s'
    
    key='median'
    model_angular_velocity['derived'][key]=copy.deepcopy(data_dict)
    model_angular_velocity['derived'][key]['label']='$\omega_{model}$ median'
    model_angular_velocity['derived'][key]['unit']='rad/s'
    
    key='bins'
    model_angular_velocity['derived'][key]=copy.deepcopy(data_dict)
    model_angular_velocity['derived'][key]['label']='$\omega_{model}$'
    model_angular_velocity['derived'][key]['unit']='rad/s'
    
    key='histogram'
    model_angular_velocity['derived'][key]=copy.deepcopy(data_dict)
    model_angular_velocity['derived'][key]['label']='PDF'
    model_angular_velocity['derived'][key]['unit']=''
    
    model_angular_velocity['coord']['time']=copy.deepcopy(coord_dict)
    model_angular_velocity['coord']['time']['label']='Time'
    model_angular_velocity['coord']['time']['unit']='$\mu s$'
    
    model_angular_velocity['shot']=[]
    model_angular_velocity['elm_time']=[]
    model_angular_velocity['coord']['time']['values']=gpi_results['coord']['time']['values']
    
    time_vec=model_angular_velocity['coord']['time']['values']
    model_angular_velocity['coord']['time']['data']=time_vec
    for index_elm in range(len(elm_index)):
        #preprocess velocity results, tackle with np.nan and outliers
        shot=int(db.loc[elm_index[index_elm]]['Shot'])
        #define ELM time for all the cases
        elm_time=db.loc[elm_index[index_elm]]['ELM time']/1000.
        model_angular_velocity['shot'].append(shot)
        model_angular_velocity['elm_time'].append(elm_time)
        flap.delete_data_object('*')
        
        """
        Calculation for the entire time range
        """
        
        ind_not_nan_v_pol_prime = ~np.isnan(shear_results['data']['v_pol_prime_smooth']['data'][index_elm,:])
        omega_full_series=np.zeros(time_vec.shape)
        try:
            omega_full_series=np.interp(gpi_results['data']['Position max radial']['data'][index_elm,:],
                                        shear_results['coord']['r']['values'][ind_not_nan_v_pol_prime],
                                        shear_results['data']['v_pol_prime_smooth']['data'][index_elm,:][ind_not_nan_v_pol_prime])
        except:
            omega_full_series[:]=np.nan
            
        model_angular_velocity['data']['Model angular velocity time series']['data'].append(omega_full_series)
    model_angular_velocity['data']['Model angular velocity time series']['data']=np.asarray(model_angular_velocity['data']['Model angular velocity time series']['data'])
    
    
    result_histogram=np.zeros([len(time_vec),n_hist])
    result_bins=np.zeros(n_hist+1)
    percentiles=np.zeros([len(time_vec),3])
    valid_data_points=np.zeros(len(time_vec))
    
    for ind_time in range(len(time_vec)):
        data=model_angular_velocity['data']['Model angular velocity time series']['data'][:,ind_time]/1e3
        nonind_nan=~np.isnan(data)
        
        (result_histogram[ind_time,:],result_bins[:])=np.histogram(data[nonind_nan],bins=n_hist)
        result_histogram[ind_time,:]/=np.sum(result_histogram[ind_time,:])
        valid_data_points[ind_time]=np.sum(nonind_nan)
        
        model_angular_velocity['derived']['10th percentile']['data'].append(np.percentile(data[nonind_nan],10))
        model_angular_velocity['derived']['90th percentile']['data'].append(np.percentile(data[nonind_nan],90))
        model_angular_velocity['derived']['median']['data'].append(np.percentile(data[nonind_nan],50))
    
    model_angular_velocity['derived']['median']['data']=np.asarray(model_angular_velocity['derived']['median']['data'])
    model_angular_velocity['derived']['10th percentile']['data']=np.asarray(model_angular_velocity['derived']['10th percentile']['data'])
    model_angular_velocity['derived']['90th percentile']['data']=np.asarray(model_angular_velocity['derived']['90th percentile']['data'])
    
    model_angular_velocity['derived']['bins']['data']=(result_bins[0:-1]+result_bins[1:])/2
    model_angular_velocity['derived']['histogram']['data']=result_histogram
    
    if plot:    
        fig,ax=plt.subplots(figsize=(8.5/2.54,8.5/2.54/np.sqrt(2)))
                            
        im=ax.contourf(time_vec*1e6,
                       (result_bins[0:-1]+result_bins[1:])/2,
                       result_histogram.transpose(),
                       levels=n_hist,
                       )
        
        ax.plot(time_vec*1e6,
                model_angular_velocity['derived']['Median']['data'].append(np.percentile(data[nonind_nan],50)),
                color='red')
        
        ax.plot(time_vec*1e6,
                model_angular_velocity['derived']['10th percentile']['data'],
                color='white',
                lw=0.1)
        
        ax.plot(time_vec*1e6,
                model_angular_velocity['derived']['90th percentile']['data'],
                color='white',
                lw=0.1)
        
        ax.set_ylim(model_angular_velocity['derived']['10th percentile']['data'],
                    model_angular_velocity['derived']['90th percentile']['data'])
        
        ax.set_title('Omega model distribution')
        ax.set_xlabel('Time [$\mu$s]')
        ax.set_ylabel('$\omega_{model,t} [krad/s]$')
        ax.xaxis.set_major_locator(MaxNLocator(5)) 
        ax.yaxis.set_major_locator(MaxNLocator(5)) 
        plt.tight_layout(pad=0.1)
        if pdf:
            pdf_pages.savefig()
        fig,ax=plt.subplots(figsize=(8.5/2.54,8.5/2.54/np.sqrt(2)))
        
        ax.plot(time_vec*1e6, valid_data_points)
        ax.set_title('Number of valid data points vs. time')
        ax.set_xlabel('$t-t_{ELM} [\mu s]$')
        ax.set_ylabel('N')
        plt.tight_layout(pad=0.1)
        
        if pdf:
            pdf_pages.savefig()
            
        if pdf:
            pdf_pages.close()
        
    if return_results:
        return model_angular_velocity