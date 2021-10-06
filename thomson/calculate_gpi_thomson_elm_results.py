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

from flap_nstx.thomson import get_nstx_thomson_gradient, get_fit_nstx_thomson_profiles
from flap_nstx.publications import read_ahmed_fit_parameters

from matplotlib.backends.backend_pdf import PdfPages
thisdir = os.path.dirname(os.path.realpath(__file__))
fn = os.path.join(thisdir,'../flap_nstx.cfg')
flap.config.read(file_name=fn)
flap_nstx.register()

def get_all_thomson_data_for_elms():
    database_file='/Users/mlampert/work/NSTX_workspace/db/ELM_findings_mlampert_velocity_good.csv'
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
            flap.get_data('NSTX_THOMSON', exp_id=shot, options={'force_mdsplus':True})
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
        
    database_file='/Users/mlampert/work/NSTX_workspace/db/ELM_findings_mlampert_velocity_good.csv'
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
        database_file='/Users/mlampert/work/NSTX_workspace/db/ELM_findings_mlampert_velocity_good.csv'
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
                                 name='EFIT01::RBDRY',
                                 exp_id=elm['shot'],
                                 object_name='SEP X OBJ'
                                 )
        
            z_bdry=flap.get_data('NSTX_MDSPlus',
                                 name='EFIT01::ZBDRY',
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
                                min_val=np.min(gradient[key_grad])
                                max_val=np.max(gradient[key_grad])
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
                            fig.tight_layout()
                            if pdf:
                                pdf_pages.savefig()
                    else:
                        fig, ax = plt.subplots()
                        if spline_thomson:
                            ax.scatter(gradient[key_grad],y_variables[var_ind][key_gpi])
                        else:
                            min_val=np.min(gradient[key_grad])
                            max_val=np.max(gradient[key_grad])
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
                        fig.tight_layout()
                        if pdf:
                            pdf_pages.savefig()
        
            if pdf:
                pdf_pages.close()
    if not plot:
        plt.close('all')
        
        
def plot_elm_properties_vs_max_gradient(elm_duration=100e-6,
                                        recalc=False,                           #Recalculate the results and do not load from the pickle file
                                        plot=False,                             #Plot the results with matplotlib
                                        pdf=False,                              #Save the results into a PDF
                                        thomson_time_window=2e-3,               #Time window of the Thomson data compared to the ELM time
                                        thomson_before=False,                    #Find the shots with valid thomson results with thomson_time_window before the ELM
                                        thomson_after=False,                    #Find the shots with valid thomson results with thomson_time_window after the ELM
                                        spline_thomson=False,                   #Calculate the results from the spline fit Thomson data
                                        pressure_grad_range=[0,100],            #Plot range for the pressure gradient
                                        density_grad_range=[0,2e22],            #Plot range for the density gradient
                                        temperature_grad_range=None,            #Plot range for the temperature gradient (no outliers, no range)
                                        thomson_frequency=60.,                  #Frequency of the Thomson scattering in Hz    
                                        overplot=True,                    #Overplot the after times
                                        plot_both=True,                         #Plot both before and after results on the same plot
                                        ):
    if thomson_before+thomson_after != 1:
        raise ValueError('Either thomson_before or thomson_after should be set.')
        
    flap.delete_data_object('*')
    wd=flap.config.get_all_section('Module NSTX_GPI')['Working directory']
    if spline_thomson:
        result_filename=wd+'/'+'elm_profile_dependence_spline'
    else:
        result_filename=wd+'/'+'elm_profile_dependence'
    result_filename+='_'+str(thomson_time_window*1000)+'ms'
    
    if thomson_before:
        result_filename+='_before'
    if thomson_after:
        result_filename+='_after'
    scaling_db_file=result_filename+'.pickle'
    
    if not os.path.exists(scaling_db_file) or recalc:
        if thomson_before:
            elms_with_thomson=get_elms_with_thomson_profile(before=True,
                                                            time_window=thomson_time_window)
        if thomson_after:
            elms_with_thomson=get_elms_with_thomson_profile(after=True,
                                                            time_window=thomson_time_window)
            
        database_file='/Users/mlampert/work/NSTX_workspace/db/ELM_findings_mlampert_velocity_good.csv'
        db=pandas.read_csv(database_file, index_col=0)
        elm_index=list(db.index)
        
        gradient={'Pressure':[],
                  'Density':[],
                  'Temperature':[]}
        gradient_thom_cor={'Pressure':[],
                           'Density':[],
                           'Temperature':[]}
        
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

            #grad.slice_data(slicing=time_slicing)
            filename=wd+'/processed_data/'+db.loc[elm_index[elm['index_elm']]]['Filename']+'.pickle'
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
                if thomson_before:
                    time_slicing={'Time':elm['thomson_time']}
                    time_slicing_thom_cor={'Time':elm['thomson_time']+1/thomson_frequency}
                if thomson_after:
                    time_slicing={'Time':elm['thomson_time']}
                    time_slicing_thom_cor={'Time':elm['thomson_time']-1/thomson_frequency}
                    
                #BEFORE SLICING
                graddata=get_fit_nstx_thomson_profiles(exp_id=elm['shot'],
                                                       pressure=True,
                                                       spline_data=spline_thomson, 
                                                       flux_coordinates=True, 
                                                       flux_range=[0.5,1.4])
                gradient['Pressure'].append(graddata.slice_data(slicing=time_slicing).data)
                gradient_thom_cor['Pressure'].append(graddata.slice_data(slicing=time_slicing_thom_cor).data)
                
                graddata=get_fit_nstx_thomson_profiles(exp_id=elm['shot'],
                                                       density=True,
                                                       spline_data=spline_thomson, 
                                                       flux_coordinates=True, 
                                                       flux_range=[0.5,1.4])
                gradient['Density'].append(graddata.slice_data(slicing=time_slicing).data)
                gradient_thom_cor['Density'].append(graddata.slice_data(slicing=time_slicing_thom_cor).data)
                
                graddata=get_fit_nstx_thomson_profiles(exp_id=elm['shot'],
                                                   temperature=True,
                                                   spline_data=spline_thomson, 
                                                   flux_coordinates=True, 
                                                   flux_range=[0.5,1.4])
                gradient['Temperature'].append(graddata.slice_data(slicing=time_slicing).data)
                gradient_thom_cor['Temperature'].append(graddata.slice_data(slicing=time_slicing_thom_cor).data)                

                
        for variable in [gradient, gradient_thom_cor, gpi_results_avg, gpi_results_max]:
            for key in variable:
                variable[key]=np.asarray(variable[key])
                
        pickle.dump((gradient, gradient_thom_cor, gpi_results_avg, gpi_results_max), open(scaling_db_file,'wb'))
    else:
        gradient, gradient_thom_cor, gpi_results_avg, gpi_results_max = pickle.load(open(scaling_db_file,'rb'))
        
    if plot:
        import matplotlib
        matplotlib.use('QT5Agg')
        import matplotlib.pyplot as plt
    else:
        import matplotlib
        matplotlib.use('agg')
        import matplotlib.pyplot as plt
        
    from numpy import logical_and as AND
    
    non_null_ind=AND(AND(gradient['Temperature'] != 0.,
                         AND(gradient['Pressure'] != 0.,
                             gradient['Density'] != 0.)),
                     AND(gradient_thom_cor['Temperature'] != 0.,
                         AND(gradient_thom_cor['Pressure'] != 0.,
                             gradient_thom_cor['Density'] != 0.)))
    del AND
    
    thomson_freq_str=f"{1/thomson_frequency*1000-thomson_time_window*1000:.2f}"    
    if thomson_before:
        title_redblue='\n (blue: [-'+str(thomson_time_window*1000)+',0]ms before ELM)'
        if overplot:
            title_redblue+=',\n(red: [0,+'+thomson_freq_str+']ms after ELM)'
            
    if thomson_after:
        title_redblue='\n (blue: [0,'+str(thomson_time_window*1000)+']ms after ELM)'
        if overplot:
            title_redblue+=',\n(red: [-'+thomson_freq_str+',0]ms before ELM)'
            
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

                        ax.scatter(gradient[key_grad][non_null_ind],
                                   y_variables[var_ind][key_gpi][:,i][non_null_ind],
                                   marker='o')
                                       
                        if overplot:
                            ax.scatter(gradient_thom_cor[key_grad][non_null_ind],
                                   y_variables[var_ind][key_gpi][:,i][non_null_ind],
                                   marker='o', color='red')
                            
                        if key_grad == 'Pressure':
                            dimension='[kPa/m]'
                        elif key_grad == 'Temperature':
                            dimension='[keV/m]'
                        elif key_grad == 'Density':
                            dimension='[1/m3/m]'
                        else:
                            dimension=''
                            
                        if pressure_grad_range is not None and key_grad == 'Pressure':
                            ax.set_xlim(pressure_grad_range[0],pressure_grad_range[1])
                        elif temperature_grad_range is not None and key_grad == 'Temperature':
                            ax.set_xlim(temperature_grad_range[0],temperature_grad_range[1])
                        elif density_grad_range is not None and key_grad == 'Density':
                            ax.set_xlim(density_grad_range[0],density_grad_range[1])
                            
                        if 'Velocity' in key_gpi:
                            dimension_gpi='[m/s]'
                        if 'Size' in key_gpi:
                            dimension_gpi='[m]'
                            
                        ax.set_xlabel(key_grad+' gradient '+dimension)
                        ax.set_ylabel(key_gpi+' '+radvert[i]+' '+dimension_gpi)
                        ax.set_title(key_grad+' gradient'+' vs. '+key_gpi+' '+radvert[i]+' '+title_addon[var_ind]+title_redblue)
                        fig.tight_layout()
                        if pdf:
                            pdf_pages.savefig()
                else:
                    fig, ax = plt.subplots()
                    ax.scatter(gradient[key_grad][non_null_ind],
                               y_variables[var_ind][key_gpi][non_null_ind])
                    if overplot:
                        ax.scatter(gradient_thom_cor[key_grad][non_null_ind],
                                   y_variables[var_ind][key_gpi][non_null_ind],
                                   marker='o', color='red')
                    if key_grad == 'Pressure' and pressure_grad_range is not None:
                        ax.set_xlim(pressure_grad_range[0],
                                    pressure_grad_range[1])
                        dimension='[kPa/m]'
                    if key_grad == 'Temperature' and temperature_grad_range is not None:
                        ax.set_xlim(temperature_grad_range[0],
                                    temperature_grad_range[1])
                        dimension='[keV/m]'
                    if key_grad == 'Density' and density_grad_range is not None:
                        ax.set_xlim(density_grad_range[0],
                                    density_grad_range[1])
                        dimension='[1/m3/m]'   
                        
                    if 'Velocity' in key_gpi:
                        dimension_gpi='[m/s]'
                    if 'Size' in key_gpi:
                        dimension_gpi='[m]'
                            
                    ax.set_xlabel(key_grad+' gradient '+dimension)
                    ax.set_ylabel(key_gpi+' '+dimension_gpi)
                    ax.set_title(key_grad+' gradient'+' vs. '+key_gpi+' '+title_addon[var_ind]+title_redblue)
                    fig.tight_layout()
                    if pdf:
                        pdf_pages.savefig()
    
        if pdf:
            pdf_pages.close()
    if not plot:
        plt.close('all')
        
def plot_elm_properties_vs_gradient_before_vs_after(elm_window=500e-6,
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
                                                    normalized_velocity=False,
                                                    subtraction_order=1,
                                                    plot_thomson_profiles=False,
                                                    plot_only_good=False,                   #Plot only those results, which satisfy the dependence_error_threshold condition.
                                                    dependence_error_threshold=0.5,         #Line fitting error dependence relative error threshold. Results under this value are plotted into a text file.
                                                    inverse_fit=False,
                                                    plot_linear_fit=True,
                                                    test=False,
                                                    ):
    if averaging not in ['before_after', 'full', 'elm']:
        raise ValueError('Averaging should be one of the following: before_after, full, elm')
    
    if gradient_type not in ['max', 'local', 'global']:
        raise ValueError('Gradient_type should be one of the following: max, local, global')
    
    #GPI spatial_coefficients
    coeff_r=np.asarray([3.7183594,-0.77821046,1402.8097])/1000. #The coordinates are in meters, the coefficients are in mm
    
    gpi_radial_range=[coeff_r[0]*0+coeff_r[1]*80+coeff_r[2],
                      coeff_r[0]*64+coeff_r[1]*0+coeff_r[2]]
    
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

    scaling_db_file_before=result_filename+'_before.pickle'
    scaling_db_file_after=result_filename+'_after.pickle'
    if test:
        import matplotlib.pyplot as plt
        plt.figure()
        
        
    def tanh_fit_function(r, b_height, b_sol, b_pos, b_width):
        return (b_height-b_sol)/2*(np.tanh((b_pos-r)/(2*b_width))+1)+b_sol
    def linear_fit_function(x,a,b):
        return a*x+b
        
    if not os.path.exists(scaling_db_file_before) or not os.path.exists(scaling_db_file_after) or recalc:
        
        #Get the ELM events with thomson before and after the ELM times
        elms_with_thomson_before=get_elms_with_thomson_profile(before=True,
                                                               time_window=thomson_time_window)
        
        elms_with_thomson_after=get_elms_with_thomson_profile(after=True,
                                                              time_window=thomson_time_window)
        #Load and process the ELM database    
        database_file='/Users/mlampert/work/NSTX_workspace/db/ELM_findings_mlampert_velocity_good.csv'
        db=pandas.read_csv(database_file, index_col=0)
        elm_index=list(db.index)
        
        if plot_thomson_profiles:
            profiles_pdf=PdfPages(wd+'/plots/all_fit_profiles_from_the_db.pdf')
            import matplotlib.pyplot as plt
            plt.figure()    
        else:
            profiles_pdf=None
            
        for ind_before_after in range(2):  #Doing the before and after calculation
            #Dividing the before and after results
            if ind_before_after == 0:
                elms_with_thomson = elms_with_thomson_before
                scaling_db_file = scaling_db_file_before
            else:
                elms_with_thomson = elms_with_thomson_after
                scaling_db_file = scaling_db_file_after
            
            #Defining the variables for the calculation
            gradient={'Pressure':[],
                      'Density':[],
                      'Temperature':[]}
            gradient_error={'Pressure':[],
                            'Density':[],
                            'Temperature':[]}
            gpi_results_avg={'Velocity ccf':[],
                             'Velocity str avg':[],
                             'Velocity str max':[],
                             'Size avg':[],
                             'Size max':[],
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
                
                elm_time=db.loc[elm_index[elm['index_elm']]]['ELM time']/1000.
                shot=int(db.loc[elm_index[elm['index_elm']]]['Shot'])
                
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
                    filename=wd+'/processed_data/'+db.loc[elm_index[elm['index_elm']]]['Filename']+'.pickle'
                #grad.slice_data(slicing=time_slicing)
                status=db.loc[elm_index[elm['index_elm']]]['OK/NOT OK']
                
                density_profile_status=db.loc[elm_index[elm['index_elm']]]['Thomson dens']
                pressure_profile_status=db.loc[elm_index[elm['index_elm']]]['Thomson press']
                temperature_profile_status=db.loc[elm_index[elm['index_elm']]]['Thomson temp']
                if (status != 'NO' and 
                    density_profile_status != 'NO' and
                    temperature_profile_status != 'NO' and
                    pressure_profile_status != 'NO'):
                    
                    velocity_results=pickle.load(open(filename, 'rb'))
                    velocity_results['Velocity ccf'][np.where(velocity_results['Correlation max'] < correlation_threshold),:]=[np.nan,np.nan]
                    for key in gpi_results_avg:
                        ind_nan=np.isnan(velocity_results[key])
                        velocity_results[key][ind_nan]=0.
                    time=velocity_results['Time']
                    
                    elm_time_interval_ind=np.where(np.logical_and(time >= elm_time-elm_duration,
                                                                  time <= elm_time+elm_duration))
                    nwin=int(elm_window/2.5e-6)
                    n_elm=int(elm_duration/2.5e-6)
                    elm_time=(time[elm_time_interval_ind])[np.argmin(velocity_results['Frame similarity'][elm_time_interval_ind])]
                    elm_time_ind=int(np.argmin(np.abs(time-elm_time)))
                    
                    for key in gpi_results_avg:
                        if averaging == 'before_after':
                            #This separates the before and after times with the before and after Thomson results.
                            if ind_before_after == 0:
                                gpi_results_avg[key].append(np.mean(velocity_results[key][elm_time_ind-nwin:elm_time_ind],axis=0))
                            elif ind_before_after == 1:
                                gpi_results_avg[key].append(np.mean(velocity_results[key][elm_time_ind+n_elm:elm_time_ind+nwin],axis=0))
                        elif averaging == 'full':
                            gpi_results_avg[key].append(np.mean(velocity_results[key][elm_time_ind+n_elm:elm_time_ind+nwin],axis=0))                            
                            
                        elif averaging == 'elm':
                            gpi_results_avg[key].append(velocity_results[key][elm_time_ind])
                            
                        if len(velocity_results[key][elm_time_ind-nwin:elm_time_ind+nwin].shape) == 2:
                            max_ind_0=np.argmax(np.abs(velocity_results[key][elm_time_ind-nwin:elm_time_ind+nwin,0]),axis=0)
                            max_ind_1=np.argmax(np.abs(velocity_results[key][elm_time_ind-nwin:elm_time_ind+nwin,1]),axis=0)
                            gpi_results_max[key].append([velocity_results[key][elm_time_ind-nwin:elm_time_ind+nwin,0][max_ind_0],
                                                         velocity_results[key][elm_time_ind-nwin:elm_time_ind+nwin,1][max_ind_1]])
                        else:
                            max_ind=np.argmax(np.abs(velocity_results[key][elm_time_ind-nwin:elm_time_ind+nwin]),axis=0)
                            gpi_results_max[key].append(velocity_results[key][elm_time_ind-nwin:elm_time_ind+nwin][max_ind])

                    if plot_thomson_profiles:
                        plot_time=elm['thomson_time']
                    else:
                        plot_time=None
                        
                    temp_para=get_fit_nstx_thomson_profiles(exp_id=elm['shot'],
                                                            temperature=True,
                                                            spline_data=spline_thomson, 
                                                            flux_coordinates=True, 
                                                            flux_range=[0.5,1.4],
                                                            plot_time=plot_time,
                                                            pdf_object=profiles_pdf,
                                                            return_parameters=True)
                    
                    pres_para=get_fit_nstx_thomson_profiles(exp_id=elm['shot'],
                                                            pressure=True,
                                                            spline_data=spline_thomson, 
                                                            flux_coordinates=True, 
                                                            flux_range=[0.5,1.4],
                                                            plot_time=plot_time,
                                                            pdf_object=profiles_pdf,
                                                            return_parameters=True)
                    
                    dens_para=get_fit_nstx_thomson_profiles(exp_id=elm['shot'],
                                                            density=True,
                                                            spline_data=spline_thomson, 
                                                            flux_coordinates=True, 
                                                            flux_range=[0.5,1.4],
                                                            plot_time=plot_time,
                                                            pdf_object=profiles_pdf,
                                                            return_parameters=True)                        
                    index_time=np.argmin(np.abs(pres_para['Time']-elm['thomson_time']))
                    grad_keys=['Temperature','Density','Pressure']
                    grad_para=[temp_para, dens_para,  pres_para]
                        
                    for index_meas in range(3):
                        a=grad_para[index_meas]['Height'][index_time]
                        b=grad_para[index_meas]['SOL offset'][index_time]
                        c=grad_para[index_meas]['Position'][index_time]
                        d=grad_para[index_meas]['Width'][index_time]
                        
                        a_err=grad_para[index_meas]['Error']['Height'][index_time]
                        b_err=grad_para[index_meas]['Error']['SOL offset'][index_time]
                        c_err=grad_para[index_meas]['Error']['Position'][index_time]
                        d_err=grad_para[index_meas]['Error']['Width'][index_time]
                        
                        if gradient_type == 'max':
                            #Get the profile gradients as a flap object (the profiles are not gotten here)
                            
                            max_gradient=grad_para[index_meas]['Max gradient'][index_time]
                            max_gradient_error=grad_para[index_meas]['Error']['Max gradient'][index_time]
                            if scale_length:
                                local_value=(a+b)/2
                                local_value_error=np.abs(a_err/2)+np.abs(b_err/2)
                                gradient[grad_keys[index_meas]].append(max_gradient/local_value)
                                gradient_error[grad_keys[index_meas]].append(np.abs(max_gradient_error/local_value)+
                                                                             np.abs(max_gradient/local_value**2)*local_value_error)
                            else:
                                gradient[grad_keys[index_meas]].append(max_gradient)
                                gradient_error[grad_keys[index_meas]].append(max_gradient_error)
                            
                 
                        elif gradient_type == 'local':
                            
                            ind_thomson_in_gpi=np.where(np.logical_and(grad_para[index_meas]['Device R'][:,index_time] > gpi_radial_range[0],
                                                                       grad_para[index_meas]['Device R'][:,index_time] < gpi_radial_range[1]))
                            psi_norm_ind_gpi=grad_para[index_meas]['Flux r'][ind_thomson_in_gpi,index_time][0]

                            if (c/(2*d) > psi_norm_ind_gpi[0] and
                                c/(2*d) < psi_norm_ind_gpi[-1]):
                                x=0.
                            else:
                                psi=psi_norm_ind_gpi[0]
                                x=2*(c-psi)/d
                            local_gradient=-2/(d)*(a-b)/2*(1-np.tanh(x)**2)
                            local_gradient_error=((a_err+b_err)*(np.abs(1/d)*(1-np.tanh(x)**2)))
                            if scale_length:
                                local_value=(a-b)/2*(np.tanh((c-x)/(2*d))+1)+b
                                local_value_error=((a_err)/2*np.abs((np.tanh((c-x)/(2*d))+1))+          #a_error and b_error only
                                                   b_err*(np.abs(1/2*(np.tanh((c-x)/(2*d))+1)+1))
#                                                       c_err*np.abs((a-b)/d*(1-np.tanh((c-x)/(2*d))**2))
#                                                       d_err*np.abs((a-b)/2*(c-x)/(2*d**2)*(1-np.tanh((c-x)/(2*d))))
                                                   )
                                gradient[grad_keys[index_meas]].append(local_gradient/local_value)
                                gradient_error[grad_keys[index_meas]].append(np.abs(1/local_value)*local_gradient_error+
                                                                             np.abs(local_gradient/local_value**2)*local_value_error)
                            else:
                                gradient[grad_keys[index_meas]].append(local_gradient)
                                gradient_error[grad_keys[index_meas]].append(local_gradient_error)
                            
            for variable in [gradient, gradient_error, gpi_results_avg, gpi_results_max]:
                for key in variable:
                    variable[key]=np.asarray(variable[key])
                    
            pickle.dump((gradient, gradient_error, gpi_results_avg, gpi_results_max), open(scaling_db_file,'wb'))
            if ind_before_after == 0:
                gradient_before=copy.deepcopy(gradient)
                gradient_before_error=copy.deepcopy(gradient_error)
                gpi_results_avg_before=copy.deepcopy(gpi_results_avg)
                gpi_results_max_before=copy.deepcopy(gpi_results_max)
            if ind_before_after == 1:
                gradient_after=copy.deepcopy(gradient)
                gradient_after_error=copy.deepcopy(gradient_error)
                gpi_results_avg_after=copy.deepcopy(gpi_results_avg)
                gpi_results_max_after=copy.deepcopy(gpi_results_max)             
    else:
        gradient_before, gradient_before_error, gpi_results_avg_before, gpi_results_max_before = pickle.load(open(scaling_db_file_before,'rb'))
        gradient_after, gradient_after_error, gpi_results_avg_after, gpi_results_max_after = pickle.load(open(scaling_db_file_after,'rb'))
    
    if plot_thomson_profiles:
        profiles_pdf.savefig()
        profiles_pdf.close()
        
    if plot:
        import matplotlib
        matplotlib.use('QT5Agg')
        import matplotlib.pyplot as plt
    else:
        import matplotlib
        matplotlib.use('agg')
        import matplotlib.pyplot as plt
    
    from numpy import logical_and as AND

    non_zero_ind_before=AND(gradient_before['Temperature'] != 0.,
                            AND(gradient_before['Pressure'] != 0.,
                                gradient_before['Density'] != 0.))
    non_zero_ind_after=AND(gradient_after['Temperature'] != 0.,
                            AND(gradient_after['Pressure'] != 0.,
                                gradient_after['Density'] != 0.))
        
    del AND

    title_redblue='\n (blue: [-'+str(thomson_time_window*1000)+',0]ms before ELM)\n (red: [0,'+str(thomson_time_window*1000)+']ms after ELM)'

    y_variables_before=[gpi_results_avg_before,gpi_results_max_before]
    y_variables_after=[gpi_results_avg_after,gpi_results_max_after]
    
    title_addon=['(temporal avg)','(range max)']
    radvert=['radial', 'vertical']

    
    x_range={}
    if auto_x_range:
        for key_grad in ['Temperature','Pressure','Density']:
            
            ind_nan_before=np.logical_not(np.isnan(gradient_before[key_grad]))
            ind_nan_after=np.logical_not(np.isnan(gradient_after[key_grad]))
            
            x_range[key_grad]=[np.min([np.min(gradient_before[key_grad][ind_nan_before]),
                                       np.min(gradient_after[key_grad][ind_nan_after])]),
                               np.max([np.max(gradient_before[key_grad][ind_nan_before]),
                                       np.max(gradient_after[key_grad][ind_nan_after])])]

    
    if plot_error:
        result_filename+='_error'
    for var_ind in range(len(y_variables_after)):
        if pdf:
            filename=result_filename.replace('processed_data', 'plots')+'_'+title_addon[var_ind].replace('(','').replace(')','').replace(' ','_')
            pdf_pages=PdfPages(filename+'.pdf')
            file=open(filename+'_linear_dependence.txt', 'wt')
        for key_gpi in y_variables_after[var_ind].keys():
            for key_grad in gradient_after.keys():
                for i in range(2):
                    if len(y_variables_after[var_ind][key_gpi].shape) == 2:
                        fig, ax = plt.subplots()
                        y_var_before=y_variables_before[var_ind][key_gpi][:,i][non_zero_ind_before]
                        non_zero_ind_y_before=np.where(y_var_before != 0.)
                        
                        y_var_after=y_variables_after[var_ind][key_gpi][:,i][non_zero_ind_after]
                        non_zero_ind_y_after=np.where(y_var_after != 0.)
                        gpi_key_str_addon=' '+radvert[i]
                    elif len(y_variables_after[var_ind][key_gpi].shape) == 1:
                        if i == 1:
                            continue
                        fig, ax = plt.subplots()
                        y_var_before=y_variables_before[var_ind][key_gpi][non_zero_ind_before]
                        non_zero_ind_y_before=np.where(y_var_before != 0.)
                        y_var_after=y_variables_after[var_ind][key_gpi][non_zero_ind_after]
                        non_zero_ind_y_after=np.where(y_var_after != 0.)
                        gpi_key_str_addon=' '

                    for before_after_str in ['Before', 'After']:
                        if before_after_str == 'Before':
                            y_var=y_var_before
                            non_zero_ind_y=non_zero_ind_y_before
                            non_zero_ind=non_zero_ind_before
                            gradient=gradient_before
                            gradient_error=gradient_before_error
                            color='tab:blue'
                            fit_color='blue'
                        if before_after_str == 'After':
                            y_var=y_var_after
                            non_zero_ind_y=non_zero_ind_y_after
                            non_zero_ind=non_zero_ind_after
                            gradient=gradient_after
                            gradient_error=gradient_after_error
                            color='red'
                            fit_color='black'

                        ind_nan=np.logical_not(np.isnan(gradient[key_grad][non_zero_ind][non_zero_ind_y]))
                        try:
                            #Zero error results are neglected
                            ind_zero_error=np.where(gradient_error[key_grad][non_zero_ind][non_zero_ind_y][ind_nan] != 0)
                            sigma=(gradient_error[key_grad][non_zero_ind][non_zero_ind_y][ind_nan][ind_zero_error] /
                                   np.sum(gradient_error[key_grad][non_zero_ind][non_zero_ind_y][ind_nan][ind_zero_error]))
                            
                            x_adjusted_error=1/sigma
                            
                            val,cov=np.polyfit(gradient[key_grad][non_zero_ind][non_zero_ind_y][ind_nan][ind_zero_error],
                                               y_var[non_zero_ind_y][ind_nan][ind_zero_error], 
                                               1, 
                                               w=x_adjusted_error,
                                               cov=True)
                        except:
                            val=[np.nan, np.nan]
                            cov=np.asarray([[np.nan,np.nan],
                                            [np.nan,np.nan]])
        
                        a=val[0]
                        b=val[1]
                        delta_a=np.sqrt(cov[0,0])
                        delta_b=np.sqrt(cov[1,1])
                        
                        good_plot=True
                        if (np.abs(delta_a/a) < dependence_error_threshold and 
                            np.abs(delta_b/b) < dependence_error_threshold):
                            file.write(before_after_str+' result:\n')
                            file.write(key_gpi+' '+gpi_key_str_addon+' = '+f"{b:.4f}"+' +- '+f"{delta_b:.4f}"+' + ('+f"{a:.4f}"+'+-'+f"{delta_a:.4f}"+') * '+key_grad+' gradient'+'\n')
                            file.write('Relative error: delta_b/b: '+f"{np.abs(delta_b/b*100):.6f}"+'% , delta_a/a: '+f"{np.abs(delta_a/a*100):.6f}"+'%\n\n')
                        elif plot_only_good:
                            good_plot=False
                            
                            
                        if good_plot:
                            ax.scatter(gradient[key_grad][non_zero_ind][non_zero_ind_y],
                                       y_var[non_zero_ind_y],
                                       marker='o',
                                       color=color)
                     
                            if plot_error:
                                ax.errorbar(gradient[key_grad][non_zero_ind][non_zero_ind_y][ind_nan],
                                            y_var[non_zero_ind_y][ind_nan],
                                            xerr=gradient_error[key_grad][non_zero_ind][non_zero_ind_y][ind_nan],
                                            marker='o', 
                                            color=color,
                                            ls='')
                                
                            if plot_linear_fit:
                                ax.plot(gradient[key_grad][non_zero_ind][non_zero_ind_y][ind_nan],
                                        a*gradient[key_grad][non_zero_ind][non_zero_ind_y][ind_nan]+b, 
                                        color=fit_color)
                                ind_sorted=np.argsort(gradient[key_grad][non_zero_ind][non_zero_ind_y][ind_nan])
                                ax.fill_between(gradient[key_grad][non_zero_ind][non_zero_ind_y][ind_nan][ind_sorted],
                                                (a-delta_a)*gradient[key_grad][non_zero_ind][non_zero_ind_y][ind_nan][ind_sorted]+(b-delta_b),
                                                (a+delta_a)*gradient[key_grad][non_zero_ind][non_zero_ind_y][ind_nan][ind_sorted]+(b+delta_b),
                                                color=color,
                                                alpha=0.3)
                                ax.fill_between(gradient[key_grad][non_zero_ind][non_zero_ind_y][ind_nan][ind_sorted],
                                                (a-delta_a)*gradient[key_grad][non_zero_ind][non_zero_ind_y][ind_nan][ind_sorted]+(b+delta_b),
                                                (a+delta_a)*gradient[key_grad][non_zero_ind][non_zero_ind_y][ind_nan][ind_sorted]+(b-delta_b),
                                                color=color,
                                                alpha=0.3)
                        
                    if key_grad == 'Pressure':
                        dimension='[kPa/m]'
                    elif key_grad == 'Temperature':
                        dimension='[keV/m]'
                    elif key_grad == 'Density':
                        dimension='[1/m3/m]'
                    else:
                        dimension=''
                        
                    if not auto_x_range:
                        if pressure_grad_range is not None and key_grad == 'Pressure':
                            ax.set_xlim(pressure_grad_range[0],pressure_grad_range[1])
                        elif temperature_grad_range is not None and key_grad == 'Temperature':
                            ax.set_xlim(temperature_grad_range[0],temperature_grad_range[1])
                        elif density_grad_range is not None and key_grad == 'Density':
                            ax.set_xlim(density_grad_range[0],density_grad_range[1])
                    else:
                        ax.set_xlim(x_range[key_grad][0],
                                    x_range[key_grad][1])
                    if auto_y_range:
                        ax.set_ylim([np.min([np.min(y_var_before[non_zero_ind_y_before]),
                                             np.min(y_var_after[non_zero_ind_y_after])]),
                                     np.max([np.max(y_var_before[non_zero_ind_y_before]),
                                             np.max(y_var_after[non_zero_ind_y_after])])])
                        
                    if 'Velocity' in key_gpi:
                        dimension_gpi='[m/s]'
                    if 'Size' in key_gpi:
                        dimension_gpi='[m]'
                        
                    ax.set_xlabel(key_grad+' gradient '+dimension)
                    ax.set_ylabel(key_gpi+' '+title_addon[var_ind]+' '+dimension_gpi)
                    ax.set_title(key_grad+' gradient'+' vs. '+key_gpi+gpi_key_str_addon+' '+title_addon[var_ind]+title_redblue)
                    fig.tight_layout()
                    if pdf:
                        pdf_pages.savefig()       
                        
        if pdf:
            pdf_pages.close()
    if not plot:
        plt.close('all')           
    file.close()
    
def plot_elm_parameters_vs_ahmed_fitting(averaging='before',
                                         parameter='grad_glob',
                                         normalized_structure=True,
                                         normalized_velocity=True,
                                         subtraction_order=4,
                                         test=False,
                                         recalc=True,
                                         elm_window=500e-6,
                                         elm_duration=100e-6,
                                         correlation_threshold=0.6,
                                         plot=False,
                                         auto_x_range=True,
                                         auto_y_range=True,
                                         plot_error=True,
                                         pdf=True,
                                         dependence_error_threshold=0.5,
                                         plot_only_good=False,
                                         plot_linear_fit=False,
                                         pressure_grad_range=None,               #Plot range for the pressure gradient
                                         density_grad_range=None,                #Plot range for the density gradient
                                         temperature_grad_range=None,            #Plot range for the temperature gradient (no outliers, no range)
                                         ):
    
    
    
    if averaging not in ['before', 'after', 'full', 'elm']:
        raise ValueError('Averaging should be one of the following: before_after, full, elm')
    if parameter not in ['grad_glob','ped_height','value_at_max_grad']:
        raise ValueError('Parameter should be one of the following: grad_glob,ped_height,value_at_max_grad')
        #GPI spatial_coefficients
    coeff_r=np.asarray([3.7183594,-0.77821046,1402.8097])/1000. #The coordinates are in meters, the coefficients are in mm
    
    coeff_r=np.asarray([3.7183594,-0.77821046,1402.8097])/1000. #The coordinates are in meters, the coefficients are in mm
    coeff_z=np.asarray([0.18090118,3.0657776,70.544312])/1000.  #The coordinates are in meters, the coefficients are in mm
    coeff_r_new=3./800.
    coeff_z_new=3./800.
    
    
    flap.delete_data_object('*')
    wd=flap.config.get_all_section('Module NSTX_GPI')['Working directory']
    
    result_filename=wd+'/processed_data/'+'elm_profile_dependence'
        
    result_filename+='_'+averaging+'_avg'
    
        
    if normalized_structure:
        result_filename+='_ns'
    if normalized_velocity:
        result_filename+='_nv'
    result_filename+='_so'+str(subtraction_order)

    scaling_db_file=result_filename+'.pickle'

    if test:
        import matplotlib.pyplot as plt
        plt.figure()
    db_dict=read_ahmed_fit_parameters()
    
    if not os.path.exists(scaling_db_file) or recalc:
        
        #Load and process the ELM database    
        database_file='/Users/mlampert/work/NSTX_workspace/db/ELM_findings_mlampert_velocity_good.csv'
        db=pandas.read_csv(database_file, index_col=0)
        elm_index=list(db.index)

        
        #Defining the variables for the calculation
        gradient={'Pressure':[],
                  'Density':[],
                  'Temperature':[]}
        gradient_error={'Pressure':[],
                        'Density':[],
                        'Temperature':[]}
        gpi_results_avg={'Velocity ccf':[],
                         'Velocity str avg':[],
                         'Velocity str max':[],
                         'Size avg':[],
                         'Size max':[],
                         'Area avg':[],
                         'Area max':[],
                         'Elongation avg':[],
                         'Elongation max':[],                          
                         'Angle avg':[],
                         'Angle max':[],                          
                         'Str number':[],
                         }
        
        gpi_results_max=copy.deepcopy(gpi_results_avg)
        index_correction=0
        for elm_ind in elm_index:
            
            elm_time=db.loc[elm_ind]['ELM time']/1000.
            shot=int(db.loc[elm_ind]['Shot'])
            
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
                filename=wd+'/processed_data/'+db.loc[elm_ind]['Filename']+'.pickle'
            #grad.slice_data(slicing=time_slicing)
            status=db.loc[elm_ind]['OK/NOT OK']
            
            if status != 'NO':
                
                velocity_results=pickle.load(open(filename, 'rb'))
                
                det=coeff_r[0]*coeff_z[1]-coeff_z[0]*coeff_r[1]
                
                for key in ['Velocity ccf','Velocity str max','Velocity str avg','Size max','Size avg']:
                    orig=copy.deepcopy(velocity_results[key])
                    velocity_results[key][:,0]=coeff_r_new/det*(coeff_z[1]*orig[:,0]-coeff_r[1]*orig[:,1])
                    velocity_results[key][:,1]=coeff_z_new/det*(-coeff_z[0]*orig[:,0]+coeff_r[0]*orig[:,1])
                    
                velocity_results['Elongation max'][:]=(velocity_results['Size max'][:,0]-velocity_results['Size max'][:,1])/(velocity_results['Size max'][:,0]+velocity_results['Size max'][:,1])
                velocity_results['Elongation avg'][:]=(velocity_results['Size avg'][:,0]-velocity_results['Size avg'][:,1])/(velocity_results['Size avg'][:,0]+velocity_results['Size avg'][:,1])
                
                velocity_results['Velocity ccf'][np.where(velocity_results['Correlation max'] < correlation_threshold),:]=[np.nan,np.nan]
                for key in gpi_results_avg:
                    ind_nan=np.isnan(velocity_results[key])
                    velocity_results[key][ind_nan]=0.
                time=velocity_results['Time']
                
                elm_time_interval_ind=np.where(np.logical_and(time >= elm_time-elm_duration,
                                                              time <= elm_time+elm_duration))
                nwin=int(elm_window/2.5e-6)     #Length of the window for the calculation before and after the ELM
                n_elm=int(elm_duration/2.5e-6)  #Length of the ELM burst approx. 100us
                
                elm_time=(time[elm_time_interval_ind])[np.argmin(velocity_results['Frame similarity'][elm_time_interval_ind])]
                elm_time_ind=int(np.argmin(np.abs(time-elm_time)))
                
                for key in gpi_results_avg:
                    if averaging == 'before':
                        #This separates the before and after times with the before and after Thomson results.
                         gpi_results_avg[key].append(np.mean(velocity_results[key][elm_time_ind-nwin:elm_time_ind],axis=0))
                    elif averaging == 'after':
                         gpi_results_avg[key].append(np.mean(velocity_results[key][elm_time_ind+n_elm:elm_time_ind+nwin],axis=0))
                        
                    elif averaging == 'full':
                        gpi_results_avg[key].append(np.mean(velocity_results[key][elm_time_ind-nwin:elm_time_ind+nwin],axis=0))                            
                        
                    elif averaging == 'elm':
                        gpi_results_avg[key].append(np.mean(velocity_results[key][elm_time_ind:elm_time_ind+n_elm], axis=0))
                        
                    elif averaging == 'no':
                        gpi_results_avg[key].append(velocity_results[key][elm_time_ind])
                    
                    if len(velocity_results[key][elm_time_ind-nwin:elm_time_ind+nwin].shape) == 2:
                        
                        max_ind_0=np.argmax(np.abs(velocity_results[key][elm_time_ind-nwin:elm_time_ind+nwin,0]),axis=0)
                        max_ind_1=np.argmax(np.abs(velocity_results[key][elm_time_ind-nwin:elm_time_ind+nwin,1]),axis=0)
                        gpi_results_max[key].append([velocity_results[key][elm_time_ind-nwin:elm_time_ind+nwin,0][max_ind_0],
                                                     velocity_results[key][elm_time_ind-nwin:elm_time_ind+nwin,1][max_ind_1]])
                    else:
                        max_ind=np.argmax(np.abs(velocity_results[key][elm_time_ind-nwin:elm_time_ind+nwin]),axis=0)
                        gpi_results_max[key].append(velocity_results[key][elm_time_ind-nwin:elm_time_ind+nwin][max_ind])
                grad_keys=['Temperature',
                           'Density',
                           'Pressure']
                if np.sum(np.where(db_dict['shot'] == shot)) != 0:
                    
                    while db_dict['shot'][elm_ind-index_correction] != shot:
                        index_correction+=1
                        
                    db_ind=elm_ind-index_correction
                        
                    for key in grad_keys:
                        gradient[key].append(db_dict[key][parameter][db_ind])
                        
                        gradient_error[key].append(db_dict[key][parameter][db_ind]*0.)
                else:
                    for key in grad_keys:
                        gradient[key].append(db_dict[key][parameter][0]*np.nan)
                    
                        gradient_error[key].append(db_dict[key][parameter][0]*np.nan)
                        
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
    
    from numpy import logical_and as AND
    non_zero_ind=AND(gradient['Temperature'] != 0.,
                     AND(gradient['Pressure'] != 0.,
                         gradient['Density'] != 0.))
        
    del AND

    title_redblue='\n (blue: [-'+str(20)+',0]ms before ELM)\n (red: [0,'+str(20)+']ms after ELM)'

    y_variables=[gpi_results_avg,
                 gpi_results_max]
    
    title_addon=['(temporal avg)',
                 '(range max)']
    radvert=['radial', 'vertical']

    
    x_range={}
    if auto_x_range:
        for key_grad in ['Temperature','Pressure','Density']:
            
            ind_nan=np.logical_not(np.isnan(gradient[key_grad]))
            
            x_range[key_grad]=[np.min(gradient[key_grad][ind_nan]),
                               np.max(gradient[key_grad][ind_nan])]
                                       
            
    
    if plot_error:
        result_filename+='_error'
    for var_ind in range(len(y_variables)):
        if pdf:
            filename=result_filename.replace('processed_data', 'plots')+'_'+title_addon[var_ind].replace('(','').replace(')','').replace(' ','_')
            pdf_pages=PdfPages(filename+'.pdf')
            file=open(filename+'_linear_dependence.txt', 'wt')
        for key_gpi in y_variables[var_ind].keys():
            for key_grad in gradient.keys():
                for i in range(2):
                    if len(y_variables[var_ind][key_gpi].shape) == 2:
                        fig, ax = plt.subplots()
                        y_var=y_variables[var_ind][key_gpi][:,i][non_zero_ind]
                        non_zero_ind_y=np.where(y_var != 0.)
                        
                        gpi_key_str_addon=' '+radvert[i]
                    elif len(y_variables[var_ind][key_gpi].shape) == 1:
                        if i == 1:
                            continue
                        fig, ax = plt.subplots()
                        y_var=y_variables[var_ind][key_gpi][non_zero_ind]
                        non_zero_ind_y=np.where(y_var != 0.)

                        gpi_key_str_addon=' '

                    color='tab:blue'
                    fit_color='blue'

                    ind_nan=np.logical_not(np.isnan(gradient[key_grad][non_zero_ind][non_zero_ind_y]))
                    try:
                        #Zero error results are neglected
                        ind_zero_error=np.where(gradient_error[key_grad][non_zero_ind][non_zero_ind_y][ind_nan] != 0)
                        sigma=(gradient_error[key_grad][non_zero_ind][non_zero_ind_y][ind_nan][ind_zero_error] /
                               np.sum(gradient_error[key_grad][non_zero_ind][non_zero_ind_y][ind_nan][ind_zero_error]))
                        
                        x_adjusted_error=1/sigma
                        
                        val,cov=np.polyfit(gradient[key_grad][non_zero_ind][non_zero_ind_y][ind_nan][ind_zero_error],
                                           y_var[non_zero_ind_y][ind_nan][ind_zero_error], 
                                           1, 
                                           w=x_adjusted_error,
                                           cov=True)
                    except:
                        val=[np.nan, np.nan]
                        cov=np.asarray([[np.nan,np.nan],
                                        [np.nan,np.nan]])
    
                    a=val[0]
                    b=val[1]
                    delta_a=np.sqrt(cov[0,0])
                    delta_b=np.sqrt(cov[1,1])
                    
                    good_plot=True
                    if (np.abs(delta_a/a) < dependence_error_threshold and 
                        np.abs(delta_b/b) < dependence_error_threshold):
                        file.write(averaging+' result:\n')
                        file.write(key_gpi+' '+gpi_key_str_addon+' = '+f"{b:.4f}"+' +- '+f"{delta_b:.4f}"+' + ('+f"{a:.4f}"+'+-'+f"{delta_a:.4f}"+') * '+key_grad+' gradient'+'\n')
                        file.write('Relative error: delta_b/b: '+f"{np.abs(delta_b/b*100):.6f}"+'% , delta_a/a: '+f"{np.abs(delta_a/a*100):.6f}"+'%\n\n')
                    elif plot_only_good:
                        good_plot=False
                        
                        
                    if good_plot:
                        ax.scatter(gradient[key_grad][non_zero_ind][non_zero_ind_y],
                                   y_var[non_zero_ind_y],
                                   marker='o',
                                   color=color)
                 
                        if plot_error:
                            ax.errorbar(gradient[key_grad][non_zero_ind][non_zero_ind_y][ind_nan],
                                        y_var[non_zero_ind_y][ind_nan],
                                        xerr=gradient_error[key_grad][non_zero_ind][non_zero_ind_y][ind_nan],
                                        marker='o', 
                                        color=color,
                                        ls='')
                            
                        if plot_linear_fit:
                            ax.plot(gradient[key_grad][non_zero_ind][non_zero_ind_y][ind_nan],
                                    a*gradient[key_grad][non_zero_ind][non_zero_ind_y][ind_nan]+b, 
                                    color=fit_color)
                            ind_sorted=np.argsort(gradient[key_grad][non_zero_ind][non_zero_ind_y][ind_nan])
                            ax.fill_between(gradient[key_grad][non_zero_ind][non_zero_ind_y][ind_nan][ind_sorted],
                                            (a-delta_a)*gradient[key_grad][non_zero_ind][non_zero_ind_y][ind_nan][ind_sorted]+(b-delta_b),
                                            (a+delta_a)*gradient[key_grad][non_zero_ind][non_zero_ind_y][ind_nan][ind_sorted]+(b+delta_b),
                                            color=color,
                                            alpha=0.3)
                            ax.fill_between(gradient[key_grad][non_zero_ind][non_zero_ind_y][ind_nan][ind_sorted],
                                            (a-delta_a)*gradient[key_grad][non_zero_ind][non_zero_ind_y][ind_nan][ind_sorted]+(b+delta_b),
                                            (a+delta_a)*gradient[key_grad][non_zero_ind][non_zero_ind_y][ind_nan][ind_sorted]+(b-delta_b),
                                            color=color,
                                            alpha=0.3)
                        
                    if key_grad == 'Pressure':
                        dimension='[kPa/m]'
                    elif key_grad == 'Temperature':
                        dimension='[keV/m]'
                    elif key_grad == 'Density':
                        dimension='[1/m3/m]'
                    else:
                        dimension=''
                        
                    if not auto_x_range:
                        if pressure_grad_range is not None and key_grad == 'Pressure':
                            ax.set_xlim(pressure_grad_range[0],pressure_grad_range[1])
                        elif temperature_grad_range is not None and key_grad == 'Temperature':
                            ax.set_xlim(temperature_grad_range[0],temperature_grad_range[1])
                        elif density_grad_range is not None and key_grad == 'Density':
                            ax.set_xlim(density_grad_range[0],density_grad_range[1])
                    else:
                        ax.set_xlim(x_range[key_grad][0],
                                    x_range[key_grad][1])
                    if auto_y_range:
                        ax.set_ylim([np.min(y_var[non_zero_ind_y]),
                                     np.max(y_var[non_zero_ind_y])])
                                         
                        
                    if 'Velocity' in key_gpi:
                        dimension_gpi='[m/s]'
                    if 'Size' in key_gpi:
                        dimension_gpi='[m]'
                        
                    ax.set_xlabel(key_grad+' gradient '+dimension)
                    ax.set_ylabel(key_gpi+' '+title_addon[var_ind]+' '+dimension_gpi)
                    ax.set_title(key_grad+' gradient'+' vs. '+key_gpi+gpi_key_str_addon+' '+title_addon[var_ind]+title_redblue)
                    fig.tight_layout()
                    if pdf:
                        pdf_pages.savefig()       
                        
        if pdf:
            pdf_pages.close()
    if not plot:
        plt.close('all')           
    file.close()
