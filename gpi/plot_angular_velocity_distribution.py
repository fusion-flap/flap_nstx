#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 20:46:46 2021

@author: mlampert
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 23:34:59 2020

@author: mlampert
"""
    
import os
import pandas
import copy

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pickle
import numpy as np

#Flap imports
import flap
import flap_nstx
flap_nstx.register()
import flap_mdsplus

#Setting up FLAP
flap_mdsplus.register('NSTX_MDSPlus')    
thisdir = os.path.dirname(os.path.realpath(__file__))
fn = os.path.join(thisdir,"../flap_nstx.cfg")
flap.config.read(file_name=fn) 

#Plot settings for publications
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
#TODO            

    #These are for a different analysis and a different method
    #define pre ELM time
    #define ELM burst time
    #define the post ELM time based on the ELM burst time
    #Calculate the average, maximum and the variance of the results in those time ranges
    #Calculate the averaged velocity trace around the ELM time
    #Calculate the correlation coefficients between the +-tau us time range around the ELM time
    #Classify the ELMs based on the correlation coefficents

def plot_nstx_gpi_angular_velocity_distribution(window_average=500e-6,
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
                                                ):
    
    if elm_time_base not in ['frame similarity', 'radial velocity']:
        raise ValueError('elm_time_base should be either "frame similarity" or "radial velocity"')
    wd=flap.config.get_all_section('Module NSTX_GPI')['Working directory']
    all_results_file=wd+'/processed_data/all_angular_velocity_results_file.pickle'
    database_file=wd+'/db/ELM_findings_mlampert_velocity_good.csv'

    db=pandas.read_csv(database_file, index_col=0)
    elm_index=list(db.index)
    n_elm=len(elm_index)
    
    nwin=int(window_average/sampling_time)
    time_vec=(np.arange(2*nwin)*sampling_time-window_average)*1e3
    
    all_results={'Velocity ccf FLAP':np.zeros([2*nwin,n_elm,2]),         
                 'Velocity ccf':np.zeros([2*nwin,n_elm,2]),
                 'Velocity ccf skim':np.zeros([2*nwin,n_elm,2]),
                 
                 'Angular velocity ccf':np.zeros([2*nwin,n_elm]),
                 'Angular velocity ccf FLAP':np.zeros([2*nwin,n_elm]),               
                 
                 'Expansion velocity ccf FLAP':np.zeros([2*nwin,n_elm]),
                 'Expansion velocity ccf':np.zeros([2*nwin,n_elm]), 
                 }
    
    hist_range_dict={'Velocity ccf FLAP':{'Radial':[-5e3,15e3],
                                          'Poloidal':[-25e3,5e3]},
                     'Velocity ccf':{'Radial':[-5e3,15e3],
                                     'Poloidal':[-25e3,5e3]},
                     'Velocity ccf skim':{'Radial':[-5e3,15e3],
                                          'Poloidal':[-25e3,5e3]},
                                 
                     'Angular velocity ccf FLAP':[-100e3,50e3],
                     'Angular velocity ccf':[-100e3,50e3],
                     'Expansion velocity ccf FLAP':[-0.2,0.2],
                     'Expansion velocity ccf':[-0.2,0.2],
                     }
    
    result_histograms={'Velocity ccf FLAP':np.zeros([2*nwin,n_hist,2]),
                       'Angular velocity ccf FLAP':np.zeros([2*nwin,n_hist]),
                       'Expansion velocity ccf FLAP':np.zeros([2*nwin,n_hist]),
                          
                       'Velocity ccf':np.zeros([2*nwin,n_hist,2]),
                       'Velocity ccf skim':np.zeros([2*nwin,n_hist,2]),
                       'Angular velocity ccf':np.zeros([2*nwin,n_hist]),
                       'Expansion velocity ccf':np.zeros([2*nwin,n_hist]), 
                       }
    
    average_results={'Velocity ccf FLAP':np.zeros([2*nwin,2]),
                       'Angular velocity ccf FLAP':np.zeros([2*nwin]),
                       'Expansion velocity ccf FLAP':np.zeros([2*nwin]),
                          
                       'Velocity ccf':np.zeros([2*nwin,2]),
                       'Velocity ccf skim':np.zeros([2*nwin,2]),
                       'Angular velocity ccf':np.zeros([2*nwin]),
                       'Expansion velocity ccf':np.zeros([2*nwin]), 
                       }
    
    moment_results={'average':copy.deepcopy(average_results),
                    'median':copy.deepcopy(average_results),
                    '10percentile':copy.deepcopy(average_results),
                    '90percentile':copy.deepcopy(average_results),
                    }
    
    result_bins={'Velocity ccf FLAP':np.zeros([n_hist+1,2]),
                 'Angular velocity ccf FLAP':np.zeros([n_hist+1]),
                 'Expansion velocity ccf FLAP':np.zeros([n_hist+1]),
                   
                 'Velocity ccf':np.zeros([n_hist+1,2]),
                 'Velocity ccf skim':np.zeros([n_hist+1,2]),
                 'Angular velocity ccf':np.zeros([n_hist+1]),
                 'Expansion velocity ccf':np.zeros([n_hist+1]), 
                 }
    
    if not nocalc:
        for index_elm in range(n_elm):
            #preprocess velocity results, tackle with np.nan and outliers
            shot=int(db.loc[elm_index[index_elm]]['Shot'])
            #define ELM time for all the cases
            elm_time=db.loc[elm_index[index_elm]]['ELM time']/1000.
            if normalized_velocity:
                filename=flap_nstx.tools.filename(exp_id=shot,
                                                  working_directory=wd+'/processed_data',
                                                  time_range=[elm_time-2e-3,elm_time+2e-3],
                                                  comment='ccf_ang_velocity_pfit_o'+str(subtraction_order)+'_fst_0.0',
                                                  extension='pickle')
                
            else:
                filename=wd+'/processed_data/'+db.loc[elm_index[index_elm]]['Filename']+'.pickle'
            status=db.loc[elm_index[index_elm]]['OK/NOT OK']
            
            if status != 'NO':
                velocity_results=pickle.load(open(filename, 'rb'))

                corr_thres_index=np.where(velocity_results['Correlation max'] < correlation_threshold)
                for key in velocity_results.keys():
                    if key in ['Velocity ccf','Velocity ccf FLAP','Velocity ccf skim']:
                        velocity_results[key][corr_thres_index,:]=[np.nan,np.nan]
                    elif key in ['Angular velocity ccf',
                                 'Angular velocity ccf FLAP',              
                                 'Expansion velocity ccf FLAP',
                                 'Expansion velocity ccf']:
                        velocity_results[key][corr_thres_index]=np.nan
                
                time=velocity_results['Time']
                elm_time_interval_ind=np.where(np.logical_and(time >= elm_time-window_average,
                                                              time <= elm_time+window_average))
                if elm_time_base == 'frame similarity':
                    elm_time=(time[elm_time_interval_ind])[np.argmin(velocity_results['Frame similarity'][elm_time_interval_ind])]
                elm_time_ind=np.argmin(np.abs(time-elm_time))
                
                for key in all_results:
                    if all_results[key].shape[1]==2:
                        all_results[key][:,index_elm,:]=velocity_results[key][elm_time_ind-nwin:elm_time_ind+nwin,:]
                    else:
                        all_results[key][:,index_elm]=velocity_results[key][elm_time_ind-nwin:elm_time_ind+nwin]
                        
        pickle.dump(all_results, open(all_results_file, 'wb'))
    else:
        all_results=pickle.load(open(all_results_file, 'rb'))
        
    for key in result_histograms:
        if len(result_histograms[key].shape)==3:
            if not min_max_range:
                hist_range_rad=hist_range_dict[key]['Radial']
                hist_range_pol=hist_range_dict[key]['Poloidal']
            else:
                hist_range=hist_range_dict[key]
                all_results_temp=all_results[key][:,:,0]
                not_nanind=np.logical_not(np.isnan(all_results_temp))
    
                hist_range_rad=[min(all_results_temp[not_nanind]),
                                max(all_results_temp[not_nanind])]
                all_results_temp=all_results[key][:,:,1]
                not_nanind=np.logical_not(np.isnan(all_results_temp))
    
                hist_range_pol=[min(all_results_temp[not_nanind]),
                                max(all_results_temp[not_nanind])]
            for i_time in range(2*nwin):
                nonind_nan_rad = np.logical_not(np.isnan(all_results[key][i_time,:,0]))
                nonind_nan_pol = np.logical_not(np.isnan(all_results[key][i_time,:,1]))

                (result_histograms[key][i_time,:,0],result_bins[key][:,0])=np.histogram(all_results[key][i_time,nonind_nan_rad,0],bins=n_hist,range=hist_range_rad)
                (result_histograms[key][i_time,:,1],result_bins[key][:,1])=np.histogram(all_results[key][i_time,nonind_nan_pol,1],bins=n_hist,range=hist_range_pol)
                result_histograms[key][i_time,:,0]/=np.sum(result_histograms[key][i_time,:,0])
                result_histograms[key][i_time,:,1]/=np.sum(result_histograms[key][i_time,:,1])
                
                moment_results['average'][key][i_time,0]=np.mean(all_results[key][i_time,nonind_nan_rad,0])
                moment_results['average'][key][i_time,1]=np.mean(all_results[key][i_time,nonind_nan_pol,1])
                
                moment_results['median'][key][i_time,0]=np.median(all_results[key][i_time,nonind_nan_rad,0])
                moment_results['median'][key][i_time,1]=np.median(all_results[key][i_time,nonind_nan_rad,1])
                
                moment_results['10percentile'][key][i_time,0]=np.percentile(all_results[key][i_time,nonind_nan_rad,0],10)
                moment_results['10percentile'][key][i_time,1]=np.percentile(all_results[key][i_time,nonind_nan_rad,1],10)
                
                moment_results['90percentile'][key][i_time,0]=np.percentile(all_results[key][i_time,nonind_nan_rad,0],90)
                moment_results['90percentile'][key][i_time,1]=np.percentile(all_results[key][i_time,nonind_nan_rad,1],90)
                
        else:
            not_nanind=np.logical_not(np.isnan(all_results[key]))
            if min_max_range:
                hist_range=[min(all_results[key][not_nanind]),
                            max(all_results[key][not_nanind])]
            else:
                hist_range=hist_range_dict[key]

            n_hist_cur=n_hist
            for i_time in range(2*nwin):
                nonind_nan=np.logical_not(np.isnan(all_results[key][i_time,:]))
                (result_histograms[key][i_time,:],result_bins[key])=np.histogram(all_results[key][i_time,nonind_nan],
                                                                                 bins=n_hist_cur,
                                                                                 range=hist_range)
                result_histograms[key][i_time,:]/=np.sum(result_histograms[key][i_time,:])
                
                moment_results['average'][key][i_time]=np.mean(all_results[key][i_time,nonind_nan])
                moment_results['median'][key][i_time]=np.median(all_results[key][i_time,nonind_nan])                
                moment_results['10percentile'][key][i_time]=np.percentile(all_results[key][i_time,nonind_nan],10)                
                moment_results['90percentile'][key][i_time]=np.percentile(all_results[key][i_time,nonind_nan],90)

    y_vector=[{'Title':'Radial velocity ccf FLAP',
               'Data':result_histograms['Velocity ccf FLAP'][:,:,0],
               'Bins':(result_bins['Velocity ccf FLAP'][0:-1,0]+result_bins['Velocity ccf FLAP'][1:,0])/2e3,
               'Bar width':(result_bins['Velocity ccf FLAP'][1,0]-result_bins['Velocity ccf FLAP'][0,0])/1e3,
               'ylabel':'v_rad FLAP[m/s]',
               },
    
              {'Title':'Poloidal velocity ccf FLAP',
               'Data':result_histograms['Velocity ccf FLAP'][:,:,1],
               'Bins':(result_bins['Velocity ccf FLAP'][0:-1,1]+result_bins['Velocity ccf FLAP'][1:,1])/2e3,
               'Bar width':(result_bins['Velocity ccf FLAP'][1,1]-result_bins['Velocity ccf FLAP'][0,1])/1e3,
               'ylabel':'v_pol FLAP [m/s]',
               },
              
              {'Title':'Radial velocity ccf',
               'Data':result_histograms['Velocity ccf'][:,:,0],
               'Bins':(result_bins['Velocity ccf'][0:-1,0]+result_bins['Velocity ccf'][1:,0])/2e3,
               'Bar width':(result_bins['Velocity ccf'][1,0]-result_bins['Velocity ccf'][0,0])/1e3,
               'ylabel':'v_rad [m/s]',
               },
    
              {'Title':'Poloidal velocity ccf',
               'Data':result_histograms['Velocity ccf'][:,:,1],
               'Bins':(result_bins['Velocity ccf'][0:-1,1]+result_bins['Velocity ccf'][1:,1])/2e3,
               'Bar width':(result_bins['Velocity ccf'][1,1]-result_bins['Velocity ccf'][0,1])/1e3,
               'ylabel':'v_pol [m/s]',
               },
              
              {'Title':'Radial velocity ccf skim',
               'Data':result_histograms['Velocity ccf skim'][:,:,0],
               'Bins':(result_bins['Velocity ccf skim'][0:-1,0]+result_bins['Velocity ccf skim'][1:,0])/2e3,
               'Bar width':(result_bins['Velocity ccf skim'][1,0]-result_bins['Velocity ccf skim'][0,0])/1e3,
               'ylabel':'v_rad skim [m/s]',
               },
    
              {'Title':'Poloidal velocity ccf skim',
               'Data':result_histograms['Velocity ccf skim'][:,:,1],
               'Bins':(result_bins['Velocity ccf skim'][0:-1,1]+result_bins['Velocity ccf skim'][1:,1])/2e3,
               'Bar width':(result_bins['Velocity ccf skim'][1,1]-result_bins['Velocity ccf skim'][0,1])/1e3,
               'ylabel':'v_pol skim[m/s]',
               },            
              
              {'Title':'Angular velocity ccf',
               'Data':result_histograms['Angular velocity ccf'],
               'Bins':(result_bins['Angular velocity ccf'][0:-1]+result_bins['Angular velocity ccf'][1:])/2,
               'Bar width':(result_bins['Angular velocity ccf'][1]-result_bins['Angular velocity ccf'][0]),
               'ylabel':'Angular velocity ccf',
               },
                            
               {'Title':'Angular velocity ccf FLAP',
               'Data':result_histograms['Angular velocity ccf FLAP'],
               'Bins':(result_bins['Angular velocity ccf FLAP'][0:-1]+result_bins['Angular velocity ccf FLAP'][1:])/2,
               'Bar width':(result_bins['Angular velocity ccf FLAP'][1]-result_bins['Angular velocity ccf FLAP'][0]),
               'ylabel':'Angular velocity ccf FLAP',
               },
                                          
               {'Title':'Expansion velocity ccf',
               'Data':result_histograms['Expansion velocity ccf'],
               'Bins':(result_bins['Expansion velocity ccf'][0:-1]+result_bins['Expansion velocity ccf'][1:])/2,
               'Bar width':(result_bins['Expansion velocity ccf'][1]-result_bins['Expansion velocity ccf'][0]),
               'ylabel':'Expansion velocity ccf',
               },
               {'Title':'Expansion velocity ccf FLAP',
               'Data':result_histograms['Expansion velocity ccf FLAP'],
               'Bins':(result_bins['Expansion velocity ccf FLAP'][0:-1]+result_bins['Expansion velocity ccf FLAP'][1:])/2,
               'Bar width':(result_bins['Expansion velocity ccf FLAP'][1]-result_bins['Expansion velocity ccf FLAP'][0]),
               'ylabel':'Expansion velocity ccf FLAP',
               },
              ]

    y_vector_avg=[{'Title':'Radial velocity ccf FLAP',
                   'Data':moment_results['median']['Velocity ccf FLAP'][:,0]/1e3,
                   '10th':moment_results['10percentile']['Velocity ccf FLAP'][:,0]/1e3,
                   '90th':moment_results['90percentile']['Velocity ccf FLAP'][:,0]/1e3,
                   'ylabel':'v_rad FLAP[m/s]',
                   },
        
                  {'Title':'Poloidal velocity ccf FLAP',
                   'Data':moment_results['median']['Velocity ccf FLAP'][:,1]/1e3,
                   '10th':moment_results['10percentile']['Velocity ccf FLAP'][:,1]/1e3,
                   '90th':moment_results['90percentile']['Velocity ccf FLAP'][:,1]/1e3,
                   'ylabel':'v_pol FLAP[m/s]',
                   },
                  
                  {'Title':'Radial velocity ccf',
                   'Data':moment_results['median']['Velocity ccf'][:,0]/1e3,
                   '10th':moment_results['10percentile']['Velocity ccf'][:,0]/1e3,
                   '90th':moment_results['90percentile']['Velocity ccf'][:,0]/1e3,
                   'ylabel':'v_rad ccf[m/s]',
                   },
        
                  {'Title':'Poloidal velocity ccf',
                   'Data':moment_results['median']['Velocity ccf'][:,1]/1e3,
                   '10th':moment_results['10percentile']['Velocity ccf'][:,1]/1e3,
                   '90th':moment_results['90percentile']['Velocity ccf'][:,1]/1e3,
                   'ylabel':'v_pol ccf[m/s]',
                   },
                  
                  {'Title':'Radial velocity ccf skim',
                   'Data':moment_results['median']['Velocity ccf skim'][:,0]/1e3,
                   '10th':moment_results['10percentile']['Velocity ccf skim'][:,0]/1e3,
                   '90th':moment_results['90percentile']['Velocity ccf skim'][:,0]/1e3,
                   'ylabel':'v_rad skim [m/s]',
                   },
        
                  {'Title':'Poloidal velocity ccf skim',
                   'Data':moment_results['median']['Velocity ccf skim'][:,1]/1e3,
                   '10th':moment_results['10percentile']['Velocity ccf skim'][:,1]/1e3,
                   '90th':moment_results['90percentile']['Velocity ccf skim'][:,1]/1e3,
                   'ylabel':'v_pol skim[m/s]',
                   },
                  
                  {'Title':'Angular velocity ccf',
                   'Data':moment_results['median']['Angular velocity ccf'],
                   '10th':moment_results['10percentile']['Angular velocity ccf'],
                   '90th':moment_results['90percentile']['Angular velocity ccf'],
                   'ylabel':'Angular velocity ccf',
                   },
                  {'Title':'Angular velocity ccf FLAP',
                   'Data':moment_results['median']['Angular velocity ccf FLAP'],
                   '10th':moment_results['10percentile']['Angular velocity ccf FLAP'],
                   '90th':moment_results['90percentile']['Angular velocity ccf FLAP'],
                   'ylabel':'Angular velocity ccf FLAP',
                   },

                  {'Title':'Expansion velocity ccf',
                   'Data':moment_results['median']['Expansion velocity ccf'],
                   '10th':moment_results['10percentile']['Expansion velocity ccf'],
                   '90th':moment_results['90percentile']['Expansion velocity ccf'],
                   'ylabel':'Expansion velocity ccf',
                   },   
                  {'Title':'Expansion velocity ccf FLAP',
                   'Data':moment_results['median']['Expansion velocity ccf FLAP'],
                   '10th':moment_results['10percentile']['Expansion velocity ccf FLAP'],
                   '90th':moment_results['90percentile']['Expansion velocity ccf FLAP'],
                   'ylabel':'Expansion velocity ccf FLAP',
                   },                  
                  ]

    pdf_object=PdfPages(wd+'/plots/all_angular_velocity_results_histograms.pdf')
    
    for i in range(len(y_vector)):
        plt.figure()
        
        plt.contourf(time_vec,
                    y_vector[i]['Bins'],
                    y_vector[i]['Data'].transpose(),
                    levels=n_hist,
                    )
        plt.plot(time_vec,
                 y_vector_avg[i]['Data'],
                 color='red')
        plt.plot(time_vec,
                 y_vector_avg[i]['10th'],
                 color='magenta')
        plt.plot(time_vec,
                 y_vector_avg[i]['90th'],
                 color='magenta')
        plt.title('Relative frequency of '+y_vector[i]['ylabel'])
        plt.xlabel('Time [ms]')
        plt.ylabel(y_vector[i]['ylabel'])
        plt.colorbar()
        pdf_object.savefig()
        
    for i in range(len(y_vector)):
        plt.figure()
#        plt.plot(y_vector[i]['Bins'], np.mean(y_vector[i]['Data'][nwin-2:nwin+3,:], axis=0))
        plt.bar(y_vector[i]['Bins'],y_vector[i]['Data'][nwin,:], width=y_vector[i]['Bar width'])
        plt.xlabel(y_vector[i]['ylabel'])
        plt.ylabel('f(x)')
        plt.title('Probablity distibution of '+y_vector[i]['ylabel'])
        pdf_object.savefig()
    
        
    pdf_object.close()
    
    if return_results:
        return time_vec, y_vector, y_vector_avg
    