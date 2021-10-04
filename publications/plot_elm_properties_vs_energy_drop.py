#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 15:11:05 2020

@author: mlampert
"""

import os
import copy

import pandas

import numpy as np
import pickle

import flap
import flap_nstx

thisdir = os.path.dirname(os.path.realpath(__file__))
fn = os.path.join(thisdir,'../flap_nstx.cfg')
flap.config.read(file_name=fn)
flap_nstx.register()

from flap_nstx.gpi import calculate_nstx_gpi_avg_frame_velocity, calculate_nstx_gpi_smooth_velocity
from flap_nstx.thomson import flap_nstx_thomson_data, get_nstx_thomson_gradient, get_fit_nstx_thomson_profiles

from matplotlib.backends.backend_pdf import PdfPages

def get_nstx_efit_energy_data(exp_id=None,
                              efit_tree='EFIT02'):
    
    """
    \WB	    \EFIT01::TOP.RESULTS.AEQDSK:WB	    Poloidal magnetic field stored energy
    \WBDOT	\EFIT01::TOP.RESULTS.AEQDSK:WBDOT	time derivative of poloidal magnetic energy
    \WDIA	\EFIT01::TOP.RESULTS.AEQDSK:WDIA	    diamagnetic energy
    \WMHD	\EFIT01::TOP.RESULTS.AEQDSK:WMHD	    total plasma energy
    \WPDOT	\EFIT01::TOP.RESULTS.AEQDSK:WPDOT	time derivative of plasma stored energy
    """
    
    if exp_id is None:
        raise ValueError('exp_id (shotnumber) needs to be set for gathering the efit energy data! Returning...')
    
    data={}
    
    d=flap.get_data('NSTX_MDSPlus',exp_id=exp_id,name='\\'+efit_tree+'::\WB',object_name='WB')
    data['Time']=d.coordinate('Time')[0]
    data['Poloidal']=d.data
    
    d=flap.get_data('NSTX_MDSPlus',exp_id=exp_id,name='\\'+efit_tree+'::\WDIA',object_name='WDIA')
    data['Diamagnetic']=d.data
    
    d=flap.get_data('NSTX_MDSPlus',exp_id=exp_id,name='\\'+efit_tree+'::\WMHD',object_name='WMHD')
    data['Total']=d.data
    
    return data

def calculate_elm_properties_vs_energy_drop(elm_window=500e-6,
                                            elm_duration=100e-6,
                                            after_time_threshold=2e-3,
                                            averaging='before_after',               #The type of averaging for the _avg results ['before_after', 'full', 'elm']

                                            recalc=False,                           #Recalculate the results and do not load from the pickle file
                                            plot=False,                             #Plot the results with matplotlib
                                            plot_error=False,
                                            pdf=False,                              #Save the results into a PDF

                                            normalized_structure=True,
                                            normalized_velocity=True,
                                            subtraction_order=1,
                                            
                                            dependence_error_threshold=0.5,         #Line fitting error dependence relative error threshold. Results under this value are plotted into a text file.
                                            test=False,
                                            plot_linear_fit=True,
                                            plot_energy=False,
                                            plot_only_good=False,
                                            ):
    
    if averaging not in ['before_after', 'full', 'elm']:
        raise ValueError('Averaging should be one of the following: before_after, full, elm')
    
    flap.delete_data_object('*')
    wd=flap.config.get_all_section('Module NSTX_GPI')['Working directory']
    result_filename=wd+'/processed_data/'+'elm_energy_dependence_'+averaging+'_avg'    
        
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
    if plot_energy:
        import matplotlib.pyplot as plt
        plt.figure()
        energy_pdf=PdfPages(wd+'/plots/all_energy.pdf')
    if not os.path.exists(scaling_db_file_before) or not os.path.exists(scaling_db_file_after) or recalc:

        #Load and process the ELM database    
        database_file='/Users/mlampert/work/NSTX_workspace/db/ELM_findings_mlampert_velocity_good.csv'
        db=pandas.read_csv(database_file, index_col=0)
        elm_index=list(db.index)
        
        #Defining the variables for the calculation
        for ind_before_after in range(2):
            energy_results={'Total':[],
                            'Diamagnetic':[],
                            'Poloidal':[]}
    
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
            
            if ind_before_after == 0:
                scaling_db_file = scaling_db_file_before
            else:
                scaling_db_file = scaling_db_file_after
                
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
                                                         comment='ccf_velocity_pfit_o'+str(subtraction_order)+'_ct_0.6_fst_0.0'+str_add+'_nv',
                                                         extension='pickle')
                else:
                    filename=wd+'/processed_data/'+db.loc[elm_index[index_elm]]['Filename']+'.pickle'
                #grad.slice_data(slicing=time_slicing)
                status=db.loc[elm_index[index_elm]]['OK/NOT OK']
                if status != 'NO':
                    velocity_results=pickle.load(open(filename, 'rb'))
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
                    
                    energy_data=get_nstx_efit_energy_data(exp_id=shot)
                    if plot_energy:
                        plt.cla()
                        for key in ['Poloidal', 'Diamagnetic', 'Total']:
                            plt.plot(energy_data['Time'], energy_data[key], label=key)
                        plt.plot([elm_time,elm_time],[0,1e6])
                        plt.title('Energy vs time for ' + str(shot)+' @ '+f"{elm_time:.6f}")
                        plt.legend(loc='upper right', shadow=True)
                        plt.xlabel('Time')
                        plt.ylabel('Energy [J]')
                        plt.xlim(elm_time-50e-3,elm_time+50e-3)
                        energy_pdf.savefig()
                    index_time=np.argmin(np.abs(energy_data['Time']-elm_time))
                    
                    if energy_data['Time'][index_time] <= elm_time and ind_before_after == 1:
                        index_time = index_time + 1
                    if energy_data['Time'][index_time] > elm_time and ind_before_after == 0:
                        index_time = index_time - 1
                    energy_status='OK'
                    if (ind_before_after == 0 and (energy_data['Time'][index_time+1]-elm_time > after_time_threshold) or
                        ind_before_after == 1 and (energy_data['Time'][index_time]-elm_time > after_time_threshold)
                        ):
                        energy_status='NO'
                    
                    if (energy_status != 'NO'):
                        for keys in energy_results.keys():
                            print(keys)
                            energy_results[keys].append(energy_data[keys][index_time])
                            
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
                            
                        
            for variable in [energy_results, gpi_results_avg, gpi_results_max]:
                for key in variable:
                    variable[key]=np.asarray(variable[key])
                
            pickle.dump((energy_results, gpi_results_avg, gpi_results_max), open(scaling_db_file,'wb'))
            if ind_before_after == 0:
                energy_results_before=copy.deepcopy(energy_results)
                gpi_results_avg_before=copy.deepcopy(gpi_results_avg)
                gpi_results_max_before=copy.deepcopy(gpi_results_max)
            if ind_before_after == 1:
                energy_results_after=copy.deepcopy(energy_results)
                gpi_results_avg_after=copy.deepcopy(gpi_results_avg)
                gpi_results_max_after=copy.deepcopy(gpi_results_max)
    else:
        energy_results_before, gpi_results_avg_before, gpi_results_max_before = pickle.load(open(scaling_db_file_before,'rb'))
        energy_results_after, gpi_results_avg_after, gpi_results_max_after = pickle.load(open(scaling_db_file_after,'rb'))
    if plot_energy:
        energy_pdf.close()
    if plot:
        import matplotlib
        matplotlib.use('QT5Agg')
        import matplotlib.pyplot as plt
    else:
        import matplotlib
        matplotlib.use('agg')
        import matplotlib.pyplot as plt
    
    from numpy import logical_and as AND
    
    non_zero_ind_before=AND(energy_results_before['Poloidal'] != 0.,
                            AND(energy_results_before['Diamagnetic'] != 0.,
                                energy_results_before['Total'] != 0.))
    non_zero_ind_after=AND(energy_results_after['Poloidal'] != 0.,
                            AND(energy_results_after['Diamagnetic'] != 0.,
                                energy_results_after['Total'] != 0.))
    
    energy_results_before['Poloidal change']=energy_results_after['Poloidal']-energy_results_before['Poloidal']
    energy_results_after['Poloidal change']=energy_results_after['Poloidal']-energy_results_before['Poloidal']
    energy_results_before['Diamagnetic change']=energy_results_after['Diamagnetic']-energy_results_before['Diamagnetic']
    energy_results_after['Diamagnetic change']=energy_results_after['Diamagnetic']-energy_results_before['Diamagnetic']
    energy_results_before['Total change']=energy_results_after['Total']-energy_results_before['Total']
    energy_results_after['Total change']=energy_results_after['Total']-energy_results_before['Total']
    
    energy_results_before['Poloidal relative change']=energy_results_after['Poloidal']/energy_results_before['Poloidal']-1
    energy_results_after['Poloidal relative change']=energy_results_after['Poloidal']/energy_results_before['Poloidal']-1
    energy_results_before['Diamagnetic relative change']=energy_results_after['Diamagnetic']/energy_results_before['Diamagnetic']-1
    energy_results_after['Diamagnetic relative change']=energy_results_after['Diamagnetic']/energy_results_before['Diamagnetic']-1
    energy_results_before['Total relative change']=energy_results_after['Total']/energy_results_before['Total']-1
    energy_results_after['Total relative change']=energy_results_after['Total']/energy_results_before['Total']-1
    
    energy_results_before['Poloidal relative change']=(energy_results_after['Poloidal']-energy_results_before['Poloidal'])/energy_results_before['Total']
    energy_results_after['Poloidal relative change']=(energy_results_after['Poloidal']-energy_results_before['Poloidal'])/energy_results_before['Total']
    energy_results_before['Diamagnetic relative change']=(energy_results_after['Diamagnetic']-energy_results_before['Diamagnetic'])/energy_results_before['Total']
    energy_results_after['Diamagnetic relative change']=(energy_results_after['Diamagnetic']-energy_results_before['Diamagnetic'])/energy_results_before['Total']
    energy_results_before['Total relative change']=(energy_results_after['Total']-energy_results_before['Total'])/energy_results_before['Total']
    energy_results_after['Total relative change']=(energy_results_after['Total']-energy_results_before['Total'])/energy_results_before['Total']
    
    del AND
    
    y_variables_before=[gpi_results_avg_before,gpi_results_max_before]
    y_variables_after=[gpi_results_avg_after,gpi_results_max_after]
    
    title_addon=['(temporal avg)','(range max)']
    radvert=['radial', 'vertical']
    
    for var_ind in range(len(y_variables_after)):
        if pdf:
            filename=result_filename.replace('processed_data', 'plots')+'_'+title_addon[var_ind].replace('(','').replace(')','').replace(' ','_')
            pdf_pages=PdfPages(filename+'.pdf')
            file=open(filename+'_linear_dependence.txt', 'wt')
        for key_gpi in y_variables_after[var_ind].keys():
            for key_grad in energy_results_after.keys():
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
                            energy_results=energy_results_before
                            color='tab:blue'
                            fit_color='blue'
                        if before_after_str == 'After':
                            y_var=y_var_after
                            non_zero_ind_y=non_zero_ind_y_after
                            non_zero_ind=non_zero_ind_after
                            energy_results=energy_results_after
                            color='red'
                            fit_color='black'

                        ind_nan=np.logical_not(np.isnan(energy_results[key_grad][non_zero_ind][non_zero_ind_y]))
                        try:
                           
                            val,cov=np.polyfit(energy_results[key_grad][non_zero_ind][non_zero_ind_y][ind_nan],
                                               y_var[non_zero_ind_y][ind_nan], 
                                               1, 
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
                            file.write(key_gpi+' '+gpi_key_str_addon+' = '+f"{b:.4f}"+' +- '+f"{delta_b:.4f}"+' + ('+f"{a:.4f}"+'+-'+f"{delta_a:.4f}"+') * '+key_grad+' energy_results'+'\n')
                            file.write('Relative error: delta_b/b: '+f"{np.abs(delta_b/b*100):.6f}"+'% , delta_a/a: '+f"{np.abs(delta_a/a*100):.6f}"+'%\n\n')
                        elif plot_only_good:
                            good_plot=False
                            
                        if good_plot:
                            ax.scatter(energy_results[key_grad][non_zero_ind][non_zero_ind_y],
                                       y_var[non_zero_ind_y],
                                       marker='o',
                                       color=color)
                                
                            if plot_linear_fit:
                                ax.plot(energy_results[key_grad][non_zero_ind][non_zero_ind_y][ind_nan],
                                        a*energy_results[key_grad][non_zero_ind][non_zero_ind_y][ind_nan]+b, 
                                        color=fit_color)
                                ind_sorted=np.argsort(energy_results[key_grad][non_zero_ind][non_zero_ind_y][ind_nan])
                                ax.fill_between(energy_results[key_grad][non_zero_ind][non_zero_ind_y][ind_nan][ind_sorted],
                                                (a-delta_a)*energy_results[key_grad][non_zero_ind][non_zero_ind_y][ind_nan][ind_sorted]+(b-delta_b),
                                                (a+delta_a)*energy_results[key_grad][non_zero_ind][non_zero_ind_y][ind_nan][ind_sorted]+(b+delta_b),
                                                color=color,
                                                alpha=0.3)
                                ax.fill_between(energy_results[key_grad][non_zero_ind][non_zero_ind_y][ind_nan][ind_sorted],
                                                (a-delta_a)*energy_results[key_grad][non_zero_ind][non_zero_ind_y][ind_nan][ind_sorted]+(b+delta_b),
                                                (a+delta_a)*energy_results[key_grad][non_zero_ind][non_zero_ind_y][ind_nan][ind_sorted]+(b-delta_b),
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
                        
                    if 'Velocity' in key_gpi:
                        dimension_gpi='[m/s]'
                    if 'Size' in key_gpi:
                        dimension_gpi='[m]'
                        
                    ax.set_xlabel(key_grad+' energy results '+dimension)
                    ax.set_ylabel(key_gpi+' '+title_addon[var_ind]+' '+dimension_gpi)
                    ax.set_title(key_grad+' energy results'+' vs. '+key_gpi+gpi_key_str_addon+' '+title_addon[var_ind])
                    fig.tight_layout()
                    if pdf:
                        pdf_pages.savefig()       
                        
        if pdf:
            pdf_pages.close()
    if not plot:
        plt.close('all')           
    file.close()
    