#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 17:48:40 2020

@author: mlampert
"""

import os
import copy
import csv

import pandas

import numpy as np
import pickle

import flap
import flap_nstx

from flap_nstx.analysis import calculate_nstx_gpi_frame_by_frame_velocity, calculate_nstx_gpi_smooth_velocity
from flap_nstx.analysis import flap_nstx_thomson_data, get_nstx_thomson_gradient, get_fit_nstx_thomson_profiles

thisdir = os.path.dirname(os.path.realpath(__file__))
fn = os.path.join(thisdir,'../flap_nstx.cfg')
flap.config.read(file_name=fn)
flap_nstx.register()

def generate_database_for_jim(elm_window=500e-6,
                              elm_duration=100e-6,
                              averaging='before_after',               #The type of averaging for the _avg results ['before_after', 'full', 'elm']
                              gradient_type='max',                    #Type of the gradient calculation ['max', 'local', 'global']
                              scale_length=False,                     #Calculate the grad/abs instead of the gradient
                              recalc=False,                           #Recalculate the results and do not load from the pickle file
                              plot=False,                             #Plot the results with matplotlib
                              plot_error=False,
                              pdf=False,                              #Save the results into a PDF
                              thomson_time_window=2e-3,               #Time window of the Thomson data compared to the ELM time
                              correlation_threshold=0.6,
                              
                              spline_thomson=False,                   #Calculate the results from the spline fit Thomson data
                              auto_x_range=True,
                              auto_y_range=True,
                              pressure_grad_range=None,               #Plot range for the pressure gradient
                              density_grad_range=None,                #Plot range for the density gradient
                              temperature_grad_range=None,            #Plot range for the temperature gradient (no outliers, no range)
                              
                              thomson_frequency=60.,                  #Frequency of the Thomson scattering in Hz    
                              normalized_structure=True,
                              normalized_velocity=True,
                              subtraction_order=4,
                              plot_thomson_profiles=False,
                              plot_only_good=False,                   #Plot only those results, which satisfy the dependence_error_threshold condition.
                              dependence_error_threshold=0.5,         #Line fitting error dependence relative error threshold. Results under this value are plotted into a text file.
                              inverse_fit=False,
                              plot_linear_fit=True,
                              test=False,
                              window_average=500e-6,
                              sampling_time=2.5e-6,
                              ahmed_database=True,
                              ):

    
    nwin=int(window_average/sampling_time)
    database_single={'n_e':0.,
                     'T_e':0.,
                     'B_tor':0.,
                     'B_pol':0.,
                     'B_rad':0.,}
    
    notnan_db_single=copy.deepcopy(database_single)
    
    average_results={'Velocity ccf':np.zeros([2*nwin,2]),
                     'Velocity str avg':np.zeros([2*nwin,2]),
                     'Velocity str max':np.zeros([2*nwin,2]),
                     'Size avg':np.zeros([2*nwin,2]),
                     'Size max':np.zeros([2*nwin,2]),
                     'Position avg':np.zeros([2*nwin,2]),
                     'Position max':np.zeros([2*nwin,2]),                   
                     'Centroid avg':np.zeros([2*nwin,2]),
                     'Centroid max':np.zeros([2*nwin,2]),                          
                     'COG avg':np.zeros([2*nwin,2]),
                     'COG max':np.zeros([2*nwin,2]),
                     'Area avg':np.zeros([2*nwin]),
                     'Area max':np.zeros([2*nwin]),
                     'Elongation avg':np.zeros([2*nwin]),
                     'Elongation max':np.zeros([2*nwin]),                          
                     'Angle avg':np.zeros([2*nwin]),
                     'Angle max':np.zeros([2*nwin]),                          
                     'Str number':np.zeros([2*nwin]),
                     }
        
    notnan_counter=copy.deepcopy(average_results)
    
    if gradient_type not in ['max', 'local', 'global']:
        raise ValueError('Gradient_type should be one of the following: max, local, global')
    
    #GPI spatial_coefficients
    coeff_r=np.asarray([3.7183594,-0.77821046,1402.8097])/1000. #The coordinates are in meters, the coefficients are in mm
    
    flap.delete_data_object('*')
    wd=flap.config.get_all_section('Module NSTX_GPI')['Working directory']
    
    if spline_thomson:
        result_filename=wd+'/processed_data/'+'elm_profile_dependence_spline'
    else:
        result_filename=wd+'/processed_data/'+'elm_profile_dependence'
        
    if scale_length:
        result_filename+='_scale'
    result_filename+='_'+gradient_type+'_grad'
    result_filename+='_'+averaging+'_avg'
    result_filename+='_'+str(thomson_time_window*1000)+'ms_both'
    
        
    if normalized_structure:
        result_filename+='_ns'
    if normalized_velocity:
        result_filename+='_nv'
    result_filename+='_so'+str(subtraction_order)

    #Load and process the ELM database    
    database_file='/Users/mlampert/work/NSTX_workspace/db/ELM_findings_mlampert_velocity_good.csv'
    db=pandas.read_csv(database_file, index_col=0)
    elm_index=list(db.index)
    
    coeff_r=np.asarray([3.7183594,-0.77821046,1402.8097])/1000. #The coordinates are in meters, the coefficients are in mm
    coeff_z=np.asarray([0.18090118,3.0657776,70.544312])/1000.  #The coordinates are in meters, the coefficients are in mm
    coeff_r_new=3./800.
    coeff_z_new=3./800.
    det=coeff_r[0]*coeff_z[1]-coeff_z[0]*coeff_r[1]
    elm_number=0.
    
    if ahmed_database:
        
        db_ahmed=[]
        with open('/Users/mlampert/work/NSTX_workspace/WORK_MATE/Profile_fitsfur_Mate') as tsvfile:
            reader = csv.reader(tsvfile, delimiter='\t')
            for row in reader:
                line=[]
                for data in row:
                    try:
                        line.append(float(data.strip(' ')))
                    except:
                        pass
                db_ahmed.append(line)
            
        db_ahmed=np.asarray(db_ahmed).transpose()
        db_dict={'shot':db_ahmed[0,:],
                 'time1':db_ahmed[1,:],
                 'time2':db_ahmed[2,:],
                 'T_e_ped':db_ahmed[3,:],
                 'n_e_ped':db_ahmed[4,:],
                 'p_e_ped':db_ahmed[5,:],
                 'T_e_max_grad':db_ahmed[6,:],
                 'n_e_max_grad':db_ahmed[7,:],
                 'p_e_max_grad':db_ahmed[8,:],
                 'T_e_width':db_ahmed[9,:],
                 'n_e_width':db_ahmed[10,:],
                 'p_e_width':db_ahmed[11,:],
                 }
    
    for index_elm in range(len(elm_index)):
        elm_time=db.loc[elm_index[index_elm]]['ELM time']/1000.
        shot=int(db.loc[elm_index[index_elm]]['Shot'])
        
        if normalized_velocity:
            if normalized_structure:
                str_add='_ns'
            else:
                str_add=''
            filename=flap_nstx.analysis.filename(exp_id=shot,
                                                 working_directory=wd+'/processed_data',
                                                 time_range=[elm_time-2e-3,elm_time+2e-3],
                                                 comment='ccf_velocity_pfit_o'+str(subtraction_order)+'_fst_0.0'+str_add+'_nv',
                                                 extension='pickle')
        else:
            filename=wd+'/processed_data/'+db.loc[elm_index[index_elm]]['Filename']+'.pickle'
        #grad.slice_data(slicing=time_slicing)
        status=db.loc[elm_index[index_elm]]['OK/NOT OK']
        
        if status != 'NO':
            velocity_results=pickle.load(open(filename, 'rb'))
            det=coeff_r[0]*coeff_z[1]-coeff_z[0]*coeff_r[1]
                
            for key in ['Velocity ccf','Velocity str max','Velocity str avg','Size max','Size avg']:
                orig=copy.deepcopy(velocity_results[key])
                velocity_results[key][:,0]=coeff_r_new/det*(coeff_z[1]*orig[:,0]-coeff_r[1]*orig[:,1])
                velocity_results[key][:,1]=coeff_z_new/det*(-coeff_z[0]*orig[:,0]+coeff_r[0]*orig[:,1])
                    
            velocity_results['Velocity ccf'][np.where(velocity_results['Correlation max'] < correlation_threshold),:]=[np.nan,np.nan]
            time=velocity_results['Time']
            elm_time_interval_ind=np.where(np.logical_and(time >= elm_time-window_average,
                                                          time <= elm_time+window_average))
            
            elm_time=(time[elm_time_interval_ind])[np.argmin(velocity_results['Frame similarity'][elm_time_interval_ind])]
            elm_time_ind=np.argmin(np.abs(time-elm_time))
            for key in average_results.keys():
                if len(average_results[key].shape) == 1:
                    ind_nan=np.isnan(velocity_results[key][elm_time_ind-nwin:elm_time_ind+nwin])
                    notnan_counter[key]+=np.logical_not(ind_nan)
                    (velocity_results[key][elm_time_ind-nwin:elm_time_ind+nwin])[ind_nan]=0.
                    average_results[key]+=velocity_results[key][elm_time_ind-nwin:elm_time_ind+nwin]
                else:
                    ind_nan_rad=np.isnan(velocity_results[key][elm_time_ind-nwin:elm_time_ind+nwin,0])
                    ind_nan_pol=np.isnan(velocity_results[key][elm_time_ind-nwin:elm_time_ind+nwin,1])
                    
                    notnan_counter[key][:,0]+=np.logical_not(ind_nan_rad)
                    notnan_counter[key][:,1]+=np.logical_not(ind_nan_pol)
                    
                    (velocity_results[key][elm_time_ind-nwin:elm_time_ind+nwin,0])[ind_nan_rad]=0.
                    (velocity_results[key][elm_time_ind-nwin:elm_time_ind+nwin,1])[ind_nan_pol]=0.
                        
                    average_results[key][:,0]+=velocity_results[key][elm_time_ind-nwin:elm_time_ind+nwin,0]
                    average_results[key][:,1]+=velocity_results[key][elm_time_ind-nwin:elm_time_ind+nwin,1]

            if not ahmed_database:
                try:
                    n_e_param=get_fit_nstx_thomson_profiles(exp_id=shot,                                      #Shot number
                                                            pressure=False,                                   #Return the pressure profile paramenters
                                                            temperature=False,                                #Return the temperature profile parameters
                                                            density=True,                                    #Return the density profile parameters
                                                            flux_coordinates=True,                           #Calculate the results in flux coordinates
                                                            flux_range=[0.5,1.4],                                #The normalaized flux coordinates range for returning the results
                                                            test=False,
                                                            output_name=None,
                                                            return_parameters=True,
                                                            plot_time=None,
                                                            pdf_object=None,
                                                            )
                    n_e_alltime=n_e_param['Value at max']
                    time_thomson=n_e_param['Time']
                    elm_ind=np.argmin(np.abs(time_thomson-elm_time))
                    n_e=n_e_alltime[elm_ind]
                    print('n_e', n_e)
                    if n_e != 0.:
                        database_single['n_e']+=n_e
                        notnan_db_single['n_e']+=1
                except:
                    pass
                try:
                    T_e_param=get_fit_nstx_thomson_profiles(exp_id=shot,                                      #Shot number
                                                            pressure=False,                                   #Return the pressure profile paramenters
                                                            temperature=True,                                #Return the temperature profile parameters
                                                            density=False,                                    #Return the density profile parameters
                                                            flux_coordinates=True,                           #Calculate the results in flux coordinates
                                                            flux_range=[0.5,1.4],                                #The normalaized flux coordinates range for returning the results
                                                            test=False,
                                                            output_name=None,
                                                            return_parameters=True,
                                                            plot_time=None,
                                                            pdf_object=None,
                                                            )
                    T_e_alltime=T_e_param['Value at max']
                    time_thomson=T_e_param['Time']
                    elm_ind=np.argmin(np.abs(time_thomson-elm_time))
                    T_e=T_e_alltime[elm_ind]
    #                print('T_e',T_e)
                    if T_e != 0 and T_e < 2.:
                        database_single['T_e']+=T_e
                        notnan_db_single['T_e']+=1
                except:
                    pass
            else:
                db_ind=np.where(np.logical_and(db_dict['shot'] == shot, np.logical_and(db_dict['time1'] < elm_time*1e3, db_dict['time2'] > elm_time*1e3)))

                if db_ind[0] != []:
                    if db_ind[0].shape !=1:
                        T_e=db_dict['T_e_max_grad'][db_ind][0]
                        n_e=db_dict['n_e_max_grad'][db_ind][0]
                    else:
                        T_e=db_dict['T_e_max_grad'][db_ind]
                        n_e=db_dict['n_e_max_grad'][db_ind]
                    print(shot, elm_time, T_e, n_e, )
                    if T_e != 0 and T_e < 2.:
                        database_single['T_e']+=T_e
                        notnan_db_single['T_e']+=1
                    if n_e != 0.:
                        database_single['n_e']+=n_e
                        notnan_db_single['n_e']+=1
            try:
                if velocity_results['Position max'][elm_time_ind,0] != 0.:
                    b_pol=flap.get_data('NSTX_MDSPlus',
                                        name='\EFIT02::\BZZ0',
                                        exp_id=shot,
                                        object_name='BZZ0').slice_data(slicing={'Time':elm_time, 
                                                                                'Device R':velocity_results['Position max'][elm_time_ind,0]}).data
                    database_single['B_pol']+=b_pol
                    notnan_db_single['B_pol']+=1
            except:
                pass
            try:
                if velocity_results['Position max'][elm_time_ind,0] != 0.:
                    b_tor=flap.get_data('NSTX_MDSPlus',
                                        name='\EFIT02::\BTZ0',
                                        exp_id=shot,
                                        object_name='BTZ0').slice_data(slicing={'Time':elm_time, 
                                                                                'Device R':velocity_results['Position max'][elm_time_ind,0]}).data
                    database_single['B_tor']+=b_tor
                    
                    notnan_db_single['B_tor']+=1
            except:
                pass
            try:
                if velocity_results['Position max'][elm_time_ind,0] != 0.:
                    b_rad=flap.get_data('NSTX_MDSPlus',
                                        name='\EFIT02::\BRZ0',
                                        exp_id=shot,
                                        object_name='BRZ0').slice_data(slicing={'Time':elm_time, 
                                                                                'Device R':velocity_results['Position max'][elm_time_ind,0]}).data
                    database_single['B_rad']+=b_rad
                    notnan_db_single['B_rad']+=1
            except:
                pass

    for key in average_results.keys():
        notnan_counter[key][np.where(notnan_counter[key] == 0)] = 1.
        if 'ccf' in key:
            if len(average_results[key].shape) == 1:
                average_results[key]=average_results[key]/(notnan_counter[key])
            else:
                average_results[key][:,0]=average_results[key][:,0]/(notnan_counter[key][:,0])
                average_results[key][:,1]=average_results[key][:,1]/(notnan_counter[key][:,1])
        else:
            if len(average_results[key].shape) == 1:
                average_results[key]=average_results[key]/(notnan_counter[key])
            else:
                average_results[key][:,0]=average_results[key][:,0]/(notnan_counter[key][:,0])
                average_results[key][:,1]=average_results[key][:,1]/(notnan_counter[key][:,1])
    try:
        for key in database_single:
            database_single[key]/=notnan_db_single[key]
    except:
        print(key, database_single[key], notnan_db_single[key])
        
    for key in average_results:
        if len(average_results[key].shape) == 1:
            print(key, average_results[key][nwin])
        else:
            print(key, average_results[key][nwin,:]) 
    for key in database_single:
        print(key, database_single[key])    
    
    return average_results
        
        