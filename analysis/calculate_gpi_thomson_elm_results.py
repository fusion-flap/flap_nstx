#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 11:36:24 2020

@author: mlampert
"""
import os
import copy

import pandas
import time

import numpy as np
import pickle

import flap
import flap_nstx

from flap_nstx.analysis import calculate_nstx_gpi_avg_frame_velocity, calculate_nstx_gpi_smooth_velocity
from flap_nstx.analysis import flap_nstx_thomson_data, get_nstx_thomson_gradient

from matplotlib.backends.backend_pdf import PdfPages
thisdir = os.path.dirname(os.path.realpath(__file__))
fn = os.path.join(thisdir,"flap_nstx.cfg")
flap.config.read(file_name=fn)
flap_nstx.register()

def get_all_thomson_data_for_elms():
    database_file='/Users/mlampert/work/NSTX_workspace/ELM_findings_mlampert_velocity_good.csv'
    db=pandas.read_csv(database_file, index_col=0)
    elm_index=list(db.index)
    failed_shot=[]
    n_fail_shots=0
    previous_shot=0.
    for index_elm in range(len(elm_index)):
        #preprocess velocity results, tackle with np.nan and outliers
        shot=int(db.loc[elm_index[index_elm]]['Shot'])
        if shot == previous_shot:
            continue
        previous_shot=shot
        start_time=time.time()
        try:
            a=flap_nstx_thomson_data(shot, force_mdsplus=True)
        except:
            failed_shot.append({'Shot':shot})
            n_fail_shots+=1.
        print(failed_shot,n_fail_shots)
        finish_time=time.time()
        rem_time=(finish_time-start_time)*(len(elm_index)-index_elm+1)
        print('Remaining time from the calculation:'+str(rem_time/3600.)+'hours.')

def get_elms_with_thomson_profile(before=False,
                                  after=False,
                                  entire=False,
                                  time_window=2e-3):
    if before+after+entire != 1:
        raise ValueError('Set one of the input variables.')
        
    database_file='/Users/mlampert/work/NSTX_workspace/ELM_findings_mlampert_velocity_good.csv'
    thomson_dir='/Users/mlampert/work/NSTX_workspace/thomson_data/'
    db=pandas.read_csv(database_file, index_col=0)
    elm_index=list(db.index)
    elms_with_thomson=[]
    for index_elm in range(len(elm_index)):
        shot=int(db.loc[elm_index[index_elm]]['Shot'])
        elm_time=db.loc[elm_index[index_elm]]['ELM time']/1000.
        thomson=pickle.load(open(thomson_dir+'nstx_mdsplus_thomson_'+str(shot)+'.pickle','rb'))
        if before:
            start_time=elm_time-time_window
            end_time=elm_time
        elif after:
            start_time=elm_time
            end_time=elm_time+time_window
        elif entire:
            start_time=elm_time-time_window
            end_time=elm_time+time_window
            
        ind=(np.logical_and(thomson['ts_times'] > start_time,
                            thomson['ts_times'] < end_time))
        if np.sum(ind) > 0:
            if before:
                thomson_time=thomson['ts_times'][np.where(thomson['ts_times'] < elm_time)]
            if after:
                thomson_time=thomson['ts_times'][np.where(thomson['ts_times'] > elm_time)]
            if entire:
                thomson_time=thomson['ts_times'][ind]
            thomson_time=thomson_time[np.argmin(np.abs(thomson_time-elm_time))]
            elms_with_thomson.append({'shot':shot, 
                                      'elm_time':elm_time,
                                      'thomson_time':thomson_time,
                                      'index_elm':index_elm})
    return elms_with_thomson

def plot_elm_properties_vs_gradient(elm_duration=100e-6,
                                    recalc=False,
                                    plot=False,
                                    pdf=False,
                                    plot_preliminary=False,
                                    thomson_time_window=2e-3,
                                    spline_thomson=False,
                                    gradient_at_separatrix=True,
                                    maximum_gradient=False):
    flap.delete_data_object('*')
    wd=flap.config.get_all_section('Module NSTX_GPI')['Working directory']
    if spline_thomson:
        result_filename=wd+'/'+'elm_profile_dependence_spline'
    else:
        result_filename=wd+'/'+'elm_profile_dependence'
    if gradient_at_separatrix:
        result_filename+='_sep_grad'
    if maximum_gradient:
        result_filename+='_max_grad'
    scaling_db_file=result_filename+'.pickle'
    
    if not os.path.exists(scaling_db_file) or recalc:
        elms_with_thomson=get_elms_with_thomson_profile(before=True,
                                                        time_window=thomson_time_window)
        database_file='/Users/mlampert/work/NSTX_workspace/ELM_findings_mlampert_velocity_good.csv'
        db=pandas.read_csv(database_file, index_col=0)
        elm_index=list(db.index)
        
        gradient={'Pressure':[],
                  'Density':[],
                  'Temperature':[]}
        gradient_error=copy.deepcopy(gradient)
        
        nwin=int(elm_duration/2.5e-6)
        gpi_results_avg={'Velocity ccf':[],
                         'Velocity str avg':[],
                         'Velocity str max':[],
                         'Frame similarity':[],
                         'Correlation max':[],
                         'Size avg':[],
                         'Size max':[],
                         'Position avg':[],
                         'Position max':[],
                         'Centroid avg':[],
                         'Centroid max':[],                          
                         'COG avg':[],
                         'COG max':[],
                         'Area avg':[],
                         'Area max':[],
                         'Elongation avg':[],
                         'Elongation max':[],                          
                         'Angle avg':[],
                         'Angle max':[],                          
                         'Str number':[],
                         }
        gpi_results_max=copy.deepcopy(gpi_results_avg)
        for elm in elms_with_thomson:
            time_slicing={'Time':elm['thomson_time']}
            r_bdry=flap.get_data('NSTX_MDSPlus',
                                 name='\EFIT01::\RBDRY',
                                 exp_id=elm['shot'],
                                 object_name='SEP X OBJ'
                                 )
        
            z_bdry=flap.get_data('NSTX_MDSPlus',
                                 name='\EFIT01::\ZBDRY',
                                 exp_id=elm['shot'],
                                 object_name='SEP Y OBJ'
                                 )
            r_bdry=r_bdry.slice_data(slicing=time_slicing)
            z_bdry=z_bdry.slice_data(slicing=time_slicing)
            ind_ok=np.where(r_bdry.data > np.mean(r_bdry.data))                     #These two lines calculate the R position of the z=0 midplane
            r_at_separatrix = r_bdry.data[ind_ok][np.argmin(np.abs(z_bdry.data[ind_ok]))]
            if gradient_at_separatrix:
                gradient['Pressure'].append(get_nstx_thomson_gradient(exp_id=elm['shot'],pressure=True,r_pos=r_at_separatrix,spline_data=spline_thomson).slice_data(slicing=time_slicing).data)
                gradient['Density'].append(get_nstx_thomson_gradient(exp_id=elm['shot'],density=True,r_pos=r_at_separatrix,spline_data=spline_thomson).slice_data(slicing=time_slicing).data)
                gradient['Temperature'].append(get_nstx_thomson_gradient(exp_id=elm['shot'],temperature=True,r_pos=r_at_separatrix,spline_data=spline_thomson).slice_data(slicing=time_slicing).data)
            elif maximum_gradient:
                thomson_rad_coord=get_nstx_thomson_gradient(exp_id=elm['shot'],pressure=True,spline_data=spline_thomson).slice_data(slicing=time_slicing).coordinate('Device R')[0]
                rad_coord_ind=np.where(np.logical_and(thomson_rad_coord <= r_at_separatrix, 
                                                      thomson_rad_coord >= r_at_separatrix-0.1)) #arbitraty temporarily
                gradient['Pressure'].append(np.max(np.abs(get_nstx_thomson_gradient(exp_id=elm['shot'],pressure=True,spline_data=spline_thomson).slice_data(slicing=time_slicing).data[rad_coord_ind])))
                gradient['Density'].append(np.max(np.abs(get_nstx_thomson_gradient(exp_id=elm['shot'],density=True,spline_data=spline_thomson).slice_data(slicing=time_slicing).data[rad_coord_ind])))
                gradient['Temperature'].append(np.max(np.abs(get_nstx_thomson_gradient(exp_id=elm['shot'],temperature=True,spline_data=spline_thomson).slice_data(slicing=time_slicing).data[rad_coord_ind])))
            
            if not spline_thomson:
                gradient_error['Pressure'].append(get_nstx_thomson_gradient(exp_id=elm['shot'],pressure=True,r_pos=r_at_separatrix).slice_data(slicing=time_slicing).error)
                gradient_error['Density'].append(get_nstx_thomson_gradient(exp_id=elm['shot'],density=True,r_pos=r_at_separatrix).slice_data(slicing=time_slicing).error)
                gradient_error['Temperature'].append(get_nstx_thomson_gradient(exp_id=elm['shot'],temperature=True,r_pos=r_at_separatrix).slice_data(slicing=time_slicing).error)
            #grad.slice_data(slicing=time_slicing)
            filename=wd+'/'+db.loc[elm_index[elm['index_elm']]]['Filename']+'.pickle'
            status=db.loc[elm_index[elm['index_elm']]]['OK/NOT OK']
    
            if status != 'NO':
                elm_time=db.loc[elm_index[elm['index_elm']]]['ELM time']/1000.
                velocity_results=pickle.load(open(filename, 'rb'))
                for key in gpi_results_avg:
                    ind_nan=np.isnan(velocity_results[key])
                    velocity_results[key][ind_nan]=0.
                time=velocity_results['Time']
                elm_time_interval_ind=np.where(np.logical_and(time >= elm_time-0.5e-3,
                                                              time <= elm_time+0.5e-3))
                elm_time=(time[elm_time_interval_ind])[np.argmin(velocity_results['Frame similarity'][elm_time_interval_ind])]
                elm_time_ind=int(np.argmin(np.abs(time-elm_time)))
                for key in gpi_results_avg:
                    gpi_results_avg[key].append(np.mean(velocity_results[key][elm_time_ind:elm_time_ind+nwin],axis=0))
                    gpi_results_max[key].append(np.max(np.abs(velocity_results[key][elm_time_ind:elm_time_ind+nwin]),axis=0))
                    
        for variable in [gradient, gradient_error, gpi_results_avg, gpi_results_max]:
            for key in variable:
                variable[key]=np.asarray(variable[key])
                
        pickle.dump((gradient, gradient_error, gpi_results_avg, gpi_results_max), open(scaling_db_file,'wb'))
    else:
        gradient, gradient_error, gpi_results_avg, gpi_results_max = pickle.load(open(scaling_db_file,'rb'))
        
    if plot:
        import matplotlib
        matplotlib.use('QT5Agg')
        import matplotlib.pyplot as plt
    else:
        import matplotlib
        matplotlib.use('agg')
        import matplotlib.pyplot as plt
    if plot_preliminary:
        y_variables=[gpi_results_avg,gpi_results_max]
        title_addon=['(temporal avg)','(range max)']
        for var_ind in range(len(y_variables)):
            if pdf:
                pdf_pages=PdfPages(result_filename+'_'+title_addon[var_ind].replace('(','').replace(')','').replace(' ','_')+'.pdf')
            for key_gpi in y_variables[var_ind]:
                for key_grad in gradient:
                    if len(y_variables[var_ind][key_gpi].shape) == 2:
                        radvert=['radial', 'vertical']
                        for i in range(2):
                            fig, ax = plt.subplots()
                            if spline_thomson:
                                ax.scatter(gradient[key_grad],y_variables[var_ind][key_gpi][:,i])
                            else:
                                ax.errorbar(gradient[key_grad],
                                            y_variables[var_ind][key_gpi][:,i],
                                            xerr=gradient_error[key_grad],
                                            marker='o',
                                            ls='none')
                                ax.set_xlim(min_val-2*np.abs(max_val-min_val),#-2*np.sqrt(np.var(gradient_error[key_grad])),
                                        max_val+2*np.abs(max_val-min_val))#+2*np.sqrt(np.var(gradient_error[key_grad])))
                            ax.set_xlabel(key_grad+' gradient')
                            ax.set_ylabel(key_gpi+' '+radvert[i])
                            ax.set_title(key_grad+' gradient'+' vs. '+key_gpi+' '+radvert[i]+' '+title_addon[var_ind])
                            min_val=np.min(gradient[key_grad])
                            max_val=np.max(gradient[key_grad])
                            fig.tight_layout()
                            if pdf:
                                pdf_pages.savefig()
                    else:
                        fig, ax = plt.subplots()
                        if spline_thomson:
                            ax.scatter(gradient[key_grad],y_variables[var_ind][key_gpi])
                        else:
                            ax.errorbar(gradient[key_grad],
                                        y_variables[var_ind][key_gpi],
                                        xerr=gradient_error[key_grad],
                                        marker='o',
                                        ls='none')
                            ax.set_xlim(min_val-2*np.abs(max_val-min_val),#-2*np.sqrt(np.var(gradient_error[key_grad])),
                                    max_val+2*np.abs(max_val-min_val))#+2*np.sqrt(np.var(gradient_error[key_grad])))
                        ax.set_xlabel(key_grad+' gradient')
                        ax.set_ylabel(key_gpi)
                        ax.set_title(key_grad+' gradient'+' vs. '+key_gpi+' '+title_addon[var_ind])
                        min_val=np.min(gradient[key_grad])
                        max_val=np.max(gradient[key_grad])
                        fig.tight_layout()
                        if pdf:
                            pdf_pages.savefig()
        
            if pdf:
                pdf_pages.close()
    if not plot:
        plt.close('all')