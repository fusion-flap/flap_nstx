#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 14:01:47 2020

@author: mlampert
"""

import os
import pandas
import copy

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
import matplotlib.cm as cm

import pickle
import numpy as np
from scipy.signal import correlate

#Flap imports
import flap
import flap_nstx
flap_nstx.register()
import flap_mdsplus

#Setting up FLAP
flap_mdsplus.register('NSTX_MDSPlus')    
thisdir = os.path.dirname(os.path.realpath(__file__))
fn = os.path.join(thisdir,"flap_nstx.cfg")
flap.config.read(file_name=fn) 

def calculate_nstx_gpi_correlation_matrix(window_average=0.5e-3,
                                          elm_burst_window=False,
                                          sampling_time=2.5e-6,
                                          plot=False,
                                          normalized_structure=True,
                                          normalized_velocity=True,
                                          subtraction_order=4,
                                          calculate_average=True,
                                          gpi_plane_calculation=True):
    
    database_file='/Users/mlampert/work/NSTX_workspace/db/ELM_findings_mlampert_velocity_good.csv'
    db=pandas.read_csv(database_file, index_col=0)
    elm_index=list(db.index)
    if calculate_average:
        pearson_keys=['Velocity ccf',       #0,1
                      'Velocity str avg',   #2,3
                      'Size avg',           #4,5
                      'Position avg',       #6,7
                      'Area avg',           #8
                      'Elongation avg',     #9
                      'Angle avg']          #10
    else:
        pearson_keys=['Velocity ccf',       #0,1
                      'Velocity str max',   #2,3
                      'Size max',           #4,5
                      'Position max',       #6,7
                      'Area max',           #8
                      'Elongation max',     #9
                      'Angle max']          #10
    nelm=0.
    pearson_matrix=np.zeros([11,11,2]) #[velocity ccf, velocity str svg]
    nwin=int(window_average/sampling_time)

    for index_elm in range(len(elm_index)):
        #preprocess velocity results, tackle with np.nan and outliers
        shot=int(db.loc[elm_index[index_elm]]['Shot'])
        #define ELM time for all the cases
        elm_time=db.loc[elm_index[index_elm]]['ELM time']/1000.
        wd=flap.config.get_all_section('Module NSTX_GPI')['Working directory']
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

        status=db.loc[elm_index[index_elm]]['OK/NOT OK']
        
        if status != 'NO':
            velocity_results=pickle.load(open(filename, 'rb'))
            time=velocity_results['Time']
            
            if gpi_plane_calculation:
                coeff_r=np.asarray([3.7183594,-0.77821046,1402.8097])/1000. #The coordinates are in meters, the coefficients are in mm
                coeff_z=np.asarray([0.18090118,3.0657776,70.544312])/1000.  #The coordinates are in meters, the coefficients are in mm
                coeff_r_new=3./800.
                coeff_z_new=3./800.
                det=coeff_r[0]*coeff_z[1]-coeff_z[0]*coeff_r[1]
                
                for key in ['Velocity ccf','Velocity str max','Velocity str avg','Size max','Size avg']:
                    orig=copy.deepcopy(velocity_results[key])
                    velocity_results[key][:,0]=coeff_r_new/det*(coeff_z[1]*orig[:,0]-coeff_r[1]*orig[:,1])
                    velocity_results[key][:,1]=coeff_z_new/det*(-coeff_z[0]*orig[:,0]+coeff_r[0]*orig[:,1])
                
                velocity_results['Elongation max'][:]=(velocity_results['Size max'][:,0]-velocity_results['Size max'][:,1])/(velocity_results['Size max'][:,0]+velocity_results['Size max'][:,1])
                velocity_results['Elongation avg'][:]=(velocity_results['Size avg'][:,0]-velocity_results['Size avg'][:,1])/(velocity_results['Size avg'][:,0]+velocity_results['Size avg'][:,1])
            if elm_burst_window:
                elm_time_interval_ind=np.where(np.logical_and(time >= elm_time,
                                                              time <= elm_time+window_average))
            else:
                elm_time_interval_ind=np.where(np.logical_and(time >= elm_time-window_average,
                                                              time <= elm_time+window_average))
            elm_time=(time[elm_time_interval_ind])[np.argmin(velocity_results['Frame similarity'][elm_time_interval_ind])]
            elm_time_ind=np.argmin(np.abs(time-elm_time))
            ind=slice(elm_time_ind-nwin,elm_time_ind+nwin)
            for ind_first in range(11):
                for ind_second in range(11):
                    if ind_first <= 7:
                        a=velocity_results[pearson_keys[ind_first//2]][ind,np.mod(ind_first,2)]
                        ind_nan=np.isnan(a)
                        a[ind_nan]=0.
                    else:
                        a=velocity_results[pearson_keys[ind_first-4]][ind]
                    if ind_second <= 7:
                        b=velocity_results[pearson_keys[ind_second//2]][ind,np.mod(ind_second,2)]
                    else:
                        b=velocity_results[pearson_keys[ind_second-4]][ind]
                    ind_nan_a=np.isnan(a)
                    ind_nan_b=np.isnan(b)
                    a[ind_nan_a]=0.
                    a[ind_nan_b]=0.
                    b[ind_nan_a]=0.
                    b[ind_nan_b]=0.
                    a-=np.mean(a)
                    b-=np.mean(b)
                    cur_pear=np.sum(a*b)/(np.sqrt(np.sum(a**2)*np.sum(b**2)))
                    if cur_pear == np.nan:
                        cur_pear=0.
                        
                    pearson_matrix[ind_first,ind_second,0]+=cur_pear
            nelm+=1
            
    pearson_matrix[:,:,0]/=nelm
    
    for index_elm in range(len(elm_index)):
        #preprocess velocity results, tackle with np.nan and outliers
        shot=int(db.loc[elm_index[index_elm]]['Shot'])
        #define ELM time for all the cases
        elm_time=db.loc[elm_index[index_elm]]['ELM time']/1000.
        wd=flap.config.get_all_section('Module NSTX_GPI')['Working directory']
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

        status=db.loc[elm_index[index_elm]]['OK/NOT OK']
        
        if status != 'NO':
            velocity_results=pickle.load(open(filename, 'rb'))
            time=velocity_results['Time']
            
            if gpi_plane_calculation:
                coeff_r=np.asarray([3.7183594,-0.77821046,1402.8097])/1000. #The coordinates are in meters, the coefficients are in mm
                coeff_z=np.asarray([0.18090118,3.0657776,70.544312])/1000.  #The coordinates are in meters, the coefficients are in mm
                coeff_r_new=3./800.
                coeff_z_new=3./800.
                det=coeff_r[0]*coeff_z[1]-coeff_z[0]*coeff_r[1]
                
                for key in ['Velocity ccf','Velocity str max','Velocity str avg','Size max','Size avg']:
                    orig=copy.deepcopy(velocity_results[key])
                    velocity_results[key][:,0]=coeff_r_new/det*(coeff_z[1]*orig[:,0]-coeff_r[1]*orig[:,1])
                    velocity_results[key][:,1]=coeff_z_new/det*(-coeff_z[0]*orig[:,0]+coeff_r[0]*orig[:,1])
                
                velocity_results['Elongation max'][:]=(velocity_results['Size max'][:,0]-velocity_results['Size max'][:,1])/(velocity_results['Size max'][:,0]+velocity_results['Size max'][:,1])
                velocity_results['Elongation avg'][:]=(velocity_results['Size avg'][:,0]-velocity_results['Size avg'][:,1])/(velocity_results['Size avg'][:,0]+velocity_results['Size avg'][:,1])
                
            elm_time_interval_ind=np.where(np.logical_and(time >= elm_time-window_average,
                                                          time <= elm_time+window_average))
            elm_time=(time[elm_time_interval_ind])[np.argmin(velocity_results['Frame similarity'][elm_time_interval_ind])]
            elm_time_ind=np.argmin(np.abs(time-elm_time))
            ind=slice(elm_time_ind-nwin,elm_time_ind+nwin)
            for ind_first in range(11):
                for ind_second in range(11):
                    if ind_first <= 7:
                        a=velocity_results[pearson_keys[ind_first//2]][ind,np.mod(ind_first,2)]
                    else:
                        a=velocity_results[pearson_keys[ind_first-4]][ind]
                    if ind_second <= 7:
                        b=velocity_results[pearson_keys[ind_second//2]][ind,np.mod(ind_second,2)]
                    else:
                        b=velocity_results[pearson_keys[ind_second-4]][ind]
                    ind_nan_a=np.isnan(a)
                    ind_nan_b=np.isnan(b)
                    a[ind_nan_a]=0.
                    a[ind_nan_b]=0.
                    b[ind_nan_a]=0.
                    b[ind_nan_b]=0.
                    a-=np.mean(a)
                    b-=np.mean(b)
                    cur_pear=np.sum(a*b)/(np.sqrt(np.sum(a**2)*np.sum(b**2)))
                    if cur_pear == np.nan:
                        cur_pear=0.
                        
                    #pearson_matrix[ind_first,ind_second,1]+=(pearson_matrix[ind_first,ind_second,0]-cur_pear)**2            
                    pearson_matrix[ind_first,ind_second,1]+=(pearson_matrix[ind_first,ind_second,0]-cur_pear)**2                        
                        
    pearson_matrix[:,:,1]=np.sqrt(pearson_matrix[:,:,1]/(nelm-1))
    
    data = pearson_matrix[:,:,0]
    if plot:
        data[10,10]=-1   
        cs=plt.matshow(data, cmap='seismic')
        plt.xticks(ticks=np.arange(11), labels=['Velocity ccf R',       #0,1
                                                'Velocity ccf z',       #0,1
                                                'Velocity str avg R',   #2,3
                                                'Velocity str avg z',   #2,3
                                                'Size avg R',           #4,5
                                                'Size avg z',           #4,5
                                                'Position avg R',       #6,7
                                                'Position avg z',       #6,7
                                                'Area avg',           #8
                                                'Elongation avg',     #9
                                                'Angle avg'], rotation='vertical')
        plt.yticks(ticks=np.arange(11), labels=['Velocity ccf R',       #0,1
                                                'Velocity ccf z',       #0,1
                                                'Velocity str avg R',   #2,3
                                                'Velocity str avg z',   #2,3
                                                'Size avg R',           #4,5
                                                'Size avg z',           #4,5
                                                'Position avg R',       #6,7
                                                'Position avg z',       #6,7
                                                'Area avg',           #8
                                                'Elongation avg',     #9
                                                'Angle avg'])
        plt.colorbar()
        plt.show()
    return pearson_matrix

def calculate_nstx_gpi_average_correlation_matrix(window_average=0.5e-3,
                                                  elm_burst_window=False,
                                                  sampling_time=2.5e-6,
                                                  plot=False,
                                                  normalized_structure=True,
                                                  normalized_velocity=True,
                                                  subtraction_order=4,
                                                  calculate_average=False,
                                                  gpi_plane_calculation=True,
                                                  calculate_absolute=True):
    
    if calculate_average:
        pearson_keys=['Velocity ccf',       #0,1
                      'Velocity str avg',   #2,3
                      'Size avg',           #4,5
                      'Position avg',       #6,7
                      'Area avg',           #8
                      'Elongation avg',     #9
                      'Angle avg',
                      'Separatrix dist avg']          #10
    else:
        pearson_keys=['Velocity ccf',       #0,1
                      'Velocity str max',   #2,3
                      'Size max',           #4,5
                      'Position max',       #6,7
                      'Area max',           #8
                      'Elongation max',     #9
                      'Angle max',
                      'Separatrix dist avg']          #10
        
    pearson_matrix=np.zeros([12,12]) #[velocity ccf, velocity str svg]
    
    average_velocity_results=calculate_avg_velocity_results(window_average=500e-6,
                                                             sampling_time=2.5e-6,
                                                             pdf=False,
                                                             plot=False,
                                                             return_results=True,
                                                             plot_error=False,
                                                             normalized_velocity=True,
                                                             normalized_structure=True,
                                                             subtraction_order=4,
                                                             opacity=0.2,
                                                             correlation_threshold=0.6,
                                                             plot_max_only=False,
                                                             plot_for_publication=False,
                                                             gpi_plane_calculation=False,
                                                             plot_scatter=False,
                                                             )
    
    if gpi_plane_calculation:
        coeff_r=np.asarray([3.7183594,-0.77821046,1402.8097])/1000. #The coordinates are in meters, the coefficients are in mm
        coeff_z=np.asarray([0.18090118,3.0657776,70.544312])/1000.  #The coordinates are in meters, the coefficients are in mm
        coeff_r_new=3./800.
        coeff_z_new=3./800.
        det=coeff_r[0]*coeff_z[1]-coeff_z[0]*coeff_r[1]
        
        for key in ['Velocity ccf','Velocity str max','Velocity str avg','Size max','Size avg']:
            orig=copy.deepcopy(average_velocity_results[key])
            average_velocity_results[key][:,0]=coeff_r_new/det*(coeff_z[1]*orig[:,0]-coeff_r[1]*orig[:,1])
            average_velocity_results[key][:,1]=coeff_z_new/det*(-coeff_z[0]*orig[:,0]+coeff_r[0]*orig[:,1])
            if calculate_absolute:
                average_velocity_results[key]=np.abs(average_velocity_results[key])
        
        average_velocity_results['Elongation max'][:]=(average_velocity_results['Size max'][:,0]-average_velocity_results['Size max'][:,1])/(average_velocity_results['Size max'][:,0]+average_velocity_results['Size max'][:,1])
        average_velocity_results['Elongation avg'][:]=(average_velocity_results['Size avg'][:,0]-average_velocity_results['Size avg'][:,1])/(average_velocity_results['Size avg'][:,0]+average_velocity_results['Size avg'][:,1])
    average_velocity_results['Tau']/=1e3
    if elm_burst_window:
        ind=np.where(np.logical_and(average_velocity_results['Tau'] >= 0,
                                    average_velocity_results['Tau'] < window_average))
    else:
        ind=np.where(np.logical_and(average_velocity_results['Tau'] >= -window_average,
                                    average_velocity_results['Tau'] < window_average))
    for ind_first in range(12):
        for ind_second in range(12):
            if ind_first <= 7:
                a=average_velocity_results[pearson_keys[ind_first//2]][ind,np.mod(ind_first,2)]
                print(pearson_keys[ind_first//2])
            else:
                a=average_velocity_results[pearson_keys[ind_first-4]][ind]
                print(pearson_keys[ind_first-4])
            if ind_second <= 7:
                b=average_velocity_results[pearson_keys[ind_second//2]][ind,np.mod(ind_second,2)]
                print(pearson_keys[ind_second//2])
            else:
                b=average_velocity_results[pearson_keys[ind_second-4]][ind]
                print(pearson_keys[ind_second-4])
            a-=np.mean(a)
            b-=np.mean(b)
            pearson_matrix[ind_first,ind_second]=np.sum(a*b)/(np.sqrt(np.sum(a**2)*np.sum(b**2)))
            if pearson_matrix[ind_first,ind_second] == np.nan:
                pearson_matrix[ind_first,ind_second]=0.         
    
    data = pearson_matrix
    if plot:
        data[11,11]=-1   
        cs=plt.matshow(data, cmap='seismic')
        if calculate_average:
            title='avg'
        else:
            title='max'
        plt.xticks(ticks=np.arange(12), labels=['Velocity ccf R',       #0,1
                                                'Velocity ccf z',       #0,1
                                                'Velocity str '+title+' R',   #2,3
                                                'Velocity str '+title+' z',   #2,3
                                                'Size '+title+' R',           #4,5
                                                'Size '+title+' z',           #4,5
                                                'Position '+title+' R',       #6,7
                                                'Position '+title+' z',       #6,7
                                                'Area '+title+'',           #8
                                                'Elongation '+title+'',     #9
                                                'Angle '+title+'',
                                                'Separatrix dist '+title], rotation='vertical',
                                                )
            
        plt.yticks(ticks=np.arange(12), labels=['Velocity ccf R',       #0,1
                                                'Velocity ccf z',       #0,1
                                                'Velocity str '+title+' R',   #2,3
                                                'Velocity str '+title+' z',   #2,3
                                                'Size '+title+' R',           #4,5
                                                'Size '+title+' z',           #4,5
                                                'Position '+title+' R',       #6,7
                                                'Position '+title+' z',       #6,7
                                                'Area '+title+'',           #8
                                                'Elongation '+title+'',     #9
                                                'Angle '+title+'',
                                                'Separatrix dist '+title])
        plt.colorbar()
        plt.show()
    return pearson_matrix

    
def plot_all_parameters_vs_all_other(window_average=0.5e-3,
                                     sampling_time=2.5e-6,
                                     plot=True,
                                     normalized_structure=True,
                                     normalized_velocity=True,
                                     subtraction_order=4,
                                     calculate_average=True,
                                     gpi_plane_calculation=True,):
    
    database_file='/Users/mlampert/work/NSTX_workspace/db/ELM_findings_mlampert_velocity_good.csv'
    db=pandas.read_csv(database_file, index_col=0)
    gs=GridSpec(6,6)
    elm_index=list(db.index)
    if calculate_average:
        pearson_keys=['Velocity ccf',       #0,1
                      'Velocity str avg',   #2,3
                      'Size avg',           #4,5
                      'Position avg',       #6,7
                      'GPI Dalpha',           #8
                      'Elongation avg',     #9
                      'Angle avg',
                      'Separatrix dist avg', #10
                      ]          
    else:
        pearson_keys=['Velocity ccf',       #0,1
                      'Velocity str max',   #2,3
                      'Size max',           #4,5
                      'Position max',       #6,7
                      'GPI Dalpha',           #8
                      'Elongation max',     #9
                      'Angle max',
                      'Separatrix dist max', #10
                      ]
        
    plot_inds=np.asarray([0,1,4,5,8,11])
    coeff_r=np.asarray([3.7183594,-0.77821046,1402.8097])/1000. #The coordinates are in meters, the coefficients are in mm
    coeff_z=np.asarray([0.18090118,3.0657776,70.544312])/1000.  #The coordinates are in meters, the coefficients are in mm
    coeff_r_new=3./800.
    coeff_z_new=3./800.
    nwin=int(window_average/sampling_time)
    if plot:
        plt.figure()
    for index_elm in range(len(elm_index)):
        #preprocess velocity results, tackle with np.nan and outliers
        shot=int(db.loc[elm_index[index_elm]]['Shot'])
        #define ELM time for all the cases
        elm_time=db.loc[elm_index[index_elm]]['ELM time']/1000.
        wd=flap.config.get_all_section('Module NSTX_GPI')['Working directory']
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

        status=db.loc[elm_index[index_elm]]['OK/NOT OK']
        
        if status != 'NO':
            velocity_results=pickle.load(open(filename, 'rb'))
            
            velocity_results['Separatrix dist avg']=np.zeros(velocity_results['Position avg'].shape[0])
            velocity_results['Separatrix dist max']=np.zeros(velocity_results['Position max'].shape[0])
            velocity_results['GPI Dalpha']=velocity_results['GPI Dalpha'][0]
            
            
            R_sep=flap.get_data('NSTX_MDSPlus',
                                name='\EFIT02::\RBDRY',
                                exp_id=shot,
                                object_name='SEP R OBJ').slice_data(slicing={'Time':elm_time}).data
            z_sep=flap.get_data('NSTX_MDSPlus',
                                name='\EFIT02::\ZBDRY',
                                exp_id=shot,
                                object_name='SEP Z OBJ').slice_data(slicing={'Time':elm_time}).data
                                
            sep_GPI_ind=np.where(np.logical_and(R_sep > coeff_r[2],
                                                      np.logical_and(z_sep > coeff_z[2],
                                                                     z_sep < coeff_z[2]+79*coeff_z[0]+64*coeff_z[1])))
            try:
                sep_GPI_ind=np.asarray(sep_GPI_ind[0])
                sep_GPI_ind=np.insert(sep_GPI_ind,0,sep_GPI_ind[0]-1)
                sep_GPI_ind=np.insert(sep_GPI_ind,len(sep_GPI_ind),sep_GPI_ind[-1]+1)            
                z_sep_GPI=z_sep[(sep_GPI_ind)]
                R_sep_GPI=R_sep[sep_GPI_ind]
                for key in ['max','avg']:
                    for ind_time in range(len(velocity_results['Position '+key][:,0])):
                        ind_z_min=np.argmin(np.abs(z_sep_GPI-velocity_results['Position '+key][ind_time,1]))
                        if z_sep_GPI[ind_z_min] >= velocity_results['Position '+key][ind_time,1]:
                            ind1=ind_z_min
                            ind2=ind_z_min+1
                        else:
                            ind1=ind_z_min-1
                            ind2=ind_z_min
                            
                        velocity_results['Separatrix dist '+key][ind_time]=velocity_results['Position '+key][ind_time,0]-((velocity_results['Position '+key][ind_time,1]-z_sep_GPI[ind2])/(z_sep_GPI[ind1]-z_sep_GPI[ind2])*(R_sep_GPI[ind1]-R_sep_GPI[ind2])+R_sep_GPI[ind2])
            except:
                pass
            
            if gpi_plane_calculation:
                det=coeff_r[0]*coeff_z[1]-coeff_z[0]*coeff_r[1]
                
                for key in ['Velocity ccf','Velocity str max','Velocity str avg','Size max','Size avg']:
                    orig=copy.deepcopy(velocity_results[key])
                    velocity_results[key][:,0]=coeff_r_new/det*(coeff_z[1]*orig[:,0]-coeff_r[1]*orig[:,1])
                    velocity_results[key][:,1]=coeff_z_new/det*(-coeff_z[0]*orig[:,0]+coeff_r[0]*orig[:,1])
                
                velocity_results['Elongation max'][:]=(velocity_results['Size max'][:,0]-velocity_results['Size max'][:,1])/(velocity_results['Size max'][:,0]+velocity_results['Size max'][:,1])
                velocity_results['Elongation avg'][:]=(velocity_results['Size avg'][:,0]-velocity_results['Size avg'][:,1])/(velocity_results['Size avg'][:,0]+velocity_results['Size avg'][:,1])
            
            time=velocity_results['Time']
            elm_time_interval_ind=np.where(np.logical_and(time >= elm_time-window_average,
                                                          time <= elm_time+window_average))
            elm_time=(time[elm_time_interval_ind])[np.argmin(velocity_results['Frame similarity'][elm_time_interval_ind])]
            elm_time_ind=np.argmin(np.abs(time-elm_time))
            ind=slice(elm_time_ind-nwin,elm_time_ind+nwin)
            for ind_first in range(12):
                for ind_second in range(12):
                    if ind_first <= 7:
                        a=velocity_results[pearson_keys[ind_first//2]][ind,np.mod(ind_first,2)]
                    else:
                        a=velocity_results[pearson_keys[ind_first-4]][ind]
                    if ind_second <= 7:
                        b=velocity_results[pearson_keys[ind_second//2]][ind,np.mod(ind_second,2)]
                    else:
                        b=velocity_results[pearson_keys[ind_second-4]][ind]
                    if (ind_first in plot_inds and 
                        ind_second in plot_inds):
                        ind1=int(np.where(plot_inds == ind_first)[0])
                        ind2=int(np.where(plot_inds == ind_second)[0])
                        plt.subplot(gs[ind2,ind1])
                        try:
                            plt.scatter(a,b,color='tab:blue')
                        except:
                            print(ind_first,ind_second)
                            if ind_first == 8:
                                print(a)
                        if ind_first <= 7:
                            label_first=pearson_keys[ind_first//2]
                            label_second=['radial', 'poloidal'][np.mod(ind_first,2)]
                            xlabel=label_first+' '+label_second
                        else:
                            xlabel=pearson_keys[ind_first-4]
                        if ind_second <= 7:
                            label_first=pearson_keys[ind_second//2]
                            label_second=['radial', 'poloidal'][np.mod(ind_second,2)]
                            ylabel=label_first+' '+label_second
                        else:
                            ylabel=pearson_keys[ind_second-4]
                            
                        plt.xlabel(xlabel)
                        plt.ylabel(ylabel)
    plt.tight_layout()
    
    
def plot_all_parameters_vs_all_other_average(window_average=0.5e-3,
                                             sampling_time=2.5e-6,
                                             plot=True,
                                             normalized_structure=True,
                                             normalized_velocity=True,
                                             subtraction_order=4,
                                             calculate_average=False,
                                             gpi_plane_calculation=True,
                                             elm_burst_window=False,
                                             pdf=True,
                                             symbol_size=1,
                                             plot_error=False):
    
    
    if calculate_average:
        pearson_keys=['Velocity ccf',       #0,1
                      'Velocity str avg',   #2,3
                      'Size avg',           #4,5
                      'Position avg',       #6,7
                      'Area avg',           #8
                      'Elongation avg',     #9
                      'Angle avg',
                      'Separatrix dist avg']          #10
    else:
        pearson_keys=['Velocity ccf',       #0,1
                      'Velocity str max',   #2,3
                      'Size max',           #4,5
                      'Position max',       #6,7
                      'Area max',           #8
                      'Elongation max',     #9
                      'Angle max',
                      'Separatrix dist max']          #10
        
    plot_inds=np.asarray([0,1,4,5,11])
    gs=GridSpec(len(plot_inds),len(plot_inds))
    results=calculate_avg_velocity_results(window_average=500e-6,
                                           sampling_time=2.5e-6,
                                           pdf=False,
                                           plot=False,
                                           return_results=not plot_error,
                                           return_error=plot_error,
                                           plot_error=False,
                                           normalized_velocity=True,
                                           normalized_structure=True,
                                           subtraction_order=subtraction_order,
                                           opacity=0.2,
                                           correlation_threshold=0.6,
                                           plot_max_only=False,
                                           plot_for_publication=False,
                                           gpi_plane_calculation=gpi_plane_calculation,
                                           plot_scatter=False,
                                           )
    if not plot_error:
        velocity_results=results
    else:
        velocity_results,error_results=results
        
    
    velocity_results['Tau']/=1e3
    if elm_burst_window:
        ind=np.where(np.logical_and(velocity_results['Tau'] >= 0,
                                    velocity_results['Tau'] < window_average))
    else:
        ind=np.where(np.logical_and(velocity_results['Tau'] >= -window_average,
                                    velocity_results['Tau'] < window_average))
    for ind_first in range(12):
        for ind_second in range(12):
            if ind_first <= 7:
                a=velocity_results[pearson_keys[ind_first//2]][ind,np.mod(ind_first,2)]
                a=a[0,:]
                a_err=error_results[pearson_keys[ind_first//2]][ind,np.mod(ind_first,2)]
                a_err=a_err[0,:]
            else:
                a=velocity_results[pearson_keys[ind_first-4]][ind]                
                a_err=error_results[pearson_keys[ind_first-4]][ind]  
            if ind_second <= 7:
                b=velocity_results[pearson_keys[ind_second//2]][ind,np.mod(ind_second,2)]
                b=b[0,:]
                b_err=error_results[pearson_keys[ind_second//2]][ind,np.mod(ind_second,2)]
                b_err=b_err[0,:]
            else:
                b=velocity_results[pearson_keys[ind_second-4]][ind]
                b_err=error_results[pearson_keys[ind_second-4]][ind]
            if (ind_first in plot_inds and 
                ind_second in plot_inds):
                ind1=int(np.where(plot_inds == ind_first)[0])
                ind2=int(np.where(plot_inds == ind_second)[0])
                colors = iter(cm.gist_ncar(np.linspace(0, 1, len(a))))
                if ind_first != ind_second:
                    plt.subplot(gs[ind2,ind1])
                    plt.plot(a,b,lw='0.2')
                    for ind_a in range(len(a)):
                        color=copy.deepcopy(next(colors))
                        plt.scatter(a[ind_a], 
                                    b[ind_a], 
                                    color=color,
                                    s=symbol_size)
                        if plot_error:
                            plt.errorbar(a[ind_a],
                                         b[ind_a], 
                                         xerr=a_err[ind_a],
                                         yerr=b_err[ind_a],
                                         color=color,
                                         lw=0.2)
                    #plt.scatter(a,b,color='tab:blue')
                    if ind_first <= 7:
                        label_first=pearson_keys[ind_first//2]
                        label_second=['radial', 'poloidal'][np.mod(ind_first,2)]
                        xlabel=label_first+' '+label_second
                    else:
                        xlabel=pearson_keys[ind_first-4]
                    if ind_second <= 7:
                        label_first=pearson_keys[ind_second//2]
                        label_second=['radial', 'poloidal'][np.mod(ind_second,2)]
                        ylabel=label_first+' '+label_second
                    else:
                        ylabel=pearson_keys[ind_second-4]
                        
                    plt.xlabel(xlabel)
                    plt.ylabel(ylabel)
                    plt.xlim([min(a)-abs(max(a)-min(a))*0.1,max(a)+abs(min(a)-max(a))*0.1])
                    plt.ylim([min(b)-abs(max(b)-min(b))*0.1,max(b)+abs(min(b)-max(b))*0.1])
                else:
                    plt.subplot(gs[ind2,ind1])
                    plt.plot(velocity_results['Tau'][ind]*1e3, b, lw=0.2)
                    for ind_a in range(len(a)):
                        plt.scatter(velocity_results['Tau'][ind][ind_a]*1e3, 
                                    b[ind_a], 
                                    color=next(colors),
                                    s=symbol_size,
                                    )
                    plt.xlim([-window_average*1e3,window_average*1e3])
                    plt.ylim([min(b)-abs(max(b)-min(b))*0.1,max(b)+abs(min(b)-max(b))*0.1])
                    plt.xlabel('Tau [ms]')
                    if ind_second <= 7:
                        label_first=pearson_keys[ind_second//2]
                        label_second=['radial', 'poloidal'][np.mod(ind_second,2)]
                        ylabel=label_first+' '+label_second
                    else:
                        ylabel=pearson_keys[ind_second-4]
                    plt.ylabel(ylabel)
                    
    #plt.tight_layout()