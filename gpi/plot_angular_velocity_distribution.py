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
from matplotlib.ticker import MaxNLocator

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
                                                tau_range=[-2e-3,2e-3],
                                                sampling_time=2.5e-6,

                                                return_results=False,
                                                return_error=False,

                                                normalized_velocity=True,
                                                normalized_structure=True,
                                                
                                                subtraction_order=4,
                                                opacity=0.2,
                                                correlation_threshold=0.6,
                                                plot_max_only=False,
                                                plot_for_publication=False,
                                                gpi_plane_calculation=True,
                                                
                                                elm_time_base='frame similarity',
                                                n_hist=50,
                                                min_max_range=False,
                                                
                                                nocalc=False,
                                                pdf=False,
                                                plot=True,
                                                
                                                plot_for_velocity=False,
                                                plot_for_structure=False,
                                                plot_for_dependence=False,
                                                plot_scatter=True,
                                                plot_variance=True,
                                                plot_error=False,
                                                plot_all_time_traces=False,     #plots all angular velocity time traces in a single plot
                                                
                                                pdf_filename=None,
                                                figure_size=8.5,
                                                
                                                ):

    wd=flap.config.get_all_section('Module NSTX_GPI')['Working directory']
    #Plot settings for publications

    if plot_for_publication:
        #figsize=(8.5/2.54, 
        #         8.5/2.54/1.618*1.1)
        figsize=(figure_size/2.54,figure_size/np.sqrt(2)/2.54)
        plt.rc('font', family='serif', serif='Helvetica')
        if figure_size > 8.5:
            labelsize=12
        else:
            labelsize=8
        linewidth=0.4
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
        
    if elm_time_base not in ['frame similarity', 'radial velocity']:
        raise ValueError('elm_time_base should be either "frame similarity" or "radial velocity"')
    all_results_file=wd+'/processed_data/all_angular_velocity_results_file.pickle'
    database_file=wd+'/db/ELM_findings_mlampert_velocity_good.csv'

    db=pandas.read_csv(database_file, index_col=0)
    elm_index=list(db.index)
    n_elm=len(elm_index)
    
    nwin=int(window_average/sampling_time)
    time_vec=(np.arange(2*nwin)*sampling_time-window_average)*1e3
    
    all_results={'Velocity ccf FLAP':np.zeros([2*nwin,n_elm,2]),         
                 #'Velocity ccf':np.zeros([2*nwin,n_elm,2]),
                 'Velocity ccf skim':np.zeros([2*nwin,n_elm,2]),
                 
                 
                 'Angular velocity ccf FLAP':np.zeros([2*nwin,n_elm]),               
                 'Angular velocity ccf FLAP log':np.zeros([2*nwin,n_elm]),   
                 'Expansion velocity ccf FLAP':np.zeros([2*nwin,n_elm]),

                 'Angular velocity ccf':np.zeros([2*nwin,n_elm]),
                 'Angular velocity ccf log':np.zeros([2*nwin,n_elm]),
                 'Expansion velocity ccf':np.zeros([2*nwin,n_elm]), 
                 }
    
    hist_range_dict={'Velocity ccf FLAP':{'Radial':[-5e3,15e3],
                                          'Poloidal':[-25e3,5e3]},
                     'Velocity ccf skim':{'Radial':[-5e3,15e3],
                                          'Poloidal':[-25e3,5e3]},
                                 
                     'Angular velocity ccf FLAP':[-100e3,50e3],
                     'Angular velocity ccf FLAP log':[-100e3,50e3],
                     'Angular velocity ccf':[-100e3,50e3],
                     'Angular velocity ccf log':[-100e3,50e3],
                     'Expansion velocity ccf FLAP':[-0.2,0.2],
                     'Expansion velocity ccf':[-0.2,0.2],
                     }
    
    result_histograms={'Velocity ccf FLAP':np.zeros([2*nwin,n_hist,2]),
                       'Angular velocity ccf FLAP':np.zeros([2*nwin,n_hist]),
                       'Angular velocity ccf FLAP log':np.zeros([2*nwin,n_hist]),
                       'Expansion velocity ccf FLAP':np.zeros([2*nwin,n_hist]),
                          
                       #'Velocity ccf':np.zeros([2*nwin,n_hist,2]),
                       'Velocity ccf skim':np.zeros([2*nwin,n_hist,2]),
                       'Angular velocity ccf':np.zeros([2*nwin,n_hist]),
                       'Angular velocity ccf log':np.zeros([2*nwin,n_hist]),
                       'Expansion velocity ccf':np.zeros([2*nwin,n_hist]), 
                       }
    
    average_results={'Velocity ccf FLAP':np.zeros([2*nwin,2]),
                     'Angular velocity ccf FLAP':np.zeros([2*nwin]),
                     'Angular velocity ccf FLAP log':np.zeros([2*nwin]),
                     'Expansion velocity ccf FLAP':np.zeros([2*nwin]),
                          
                     # 'Velocity ccf':np.zeros([2*nwin,2]),
                     'Velocity ccf skim':np.zeros([2*nwin,2]),
                     'Angular velocity ccf':np.zeros([2*nwin]),
                     'Angular velocity ccf log':np.zeros([2*nwin]),
                     'Expansion velocity ccf':np.zeros([2*nwin]), 
                     }
    
    moment_results={'average':copy.deepcopy(average_results),
                    'median':copy.deepcopy(average_results),
                    '10percentile':copy.deepcopy(average_results),
                    '90percentile':copy.deepcopy(average_results),
                    }
    
    result_bins={'Velocity ccf FLAP':np.zeros([n_hist+1,2]),
                 'Angular velocity ccf FLAP':np.zeros([n_hist+1]),
                 'Angular velocity ccf FLAP log':np.zeros([n_hist+1]),
                 'Expansion velocity ccf FLAP':np.zeros([n_hist+1]),
                   
                 # 'Velocity ccf':np.zeros([n_hist+1,2]),
                 'Velocity ccf skim':np.zeros([n_hist+1,2]),
                 'Angular velocity ccf':np.zeros([n_hist+1]),
                 'Angular velocity ccf log':np.zeros([n_hist+1]),
                 'Expansion velocity ccf':np.zeros([n_hist+1]),
                 }
    
    if not nocalc:
        for index_elm in range(n_elm):
            #preprocess velocity results, tackle with np.nan and outliers
            shot=int(db.loc[elm_index[index_elm]]['Shot'])
            #define ELM time for all the cases
            elm_time=db.loc[elm_index[index_elm]]['Time prec']
            if normalized_velocity:
                filename=flap_nstx.tools.filename(exp_id=shot,
                                                  working_directory=wd+'/processed_data',
                                                  time_range=[elm_time+tau_range[0],elm_time+tau_range[1]],
                                                  comment='ccf_ang_velocity_pfit_o'+str(subtraction_order)+'_fst_0.0',
                                                  extension='pickle')
                
            else:
                filename=wd+'/processed_data/'+db.loc[elm_index[index_elm]]['Filename']+'.pickle'
            status=db.loc[elm_index[index_elm]]['OK/NOT OK']
            
            if status != 'NO':
                velocity_results=pickle.load(open(filename, 'rb'))

                corr_thres_index=np.where(velocity_results['Correlation max'] < correlation_threshold)
               # corr_thres_index=np.where(velocity_results['Correlation max polar FLAP'] < correlation_threshold)
               
                for key in velocity_results.keys():
                    if key in ['Velocity ccf',
                               'Velocity ccf FLAP',
                               'Velocity ccf skim']:
                        velocity_results[key][corr_thres_index,:]=[np.nan,np.nan]
                    elif key in ['Angular velocity ccf',
                                 'Angular velocity ccf log',
                                 'Angular velocity ccf FLAP',
                                 'Angular velocity ccf FLAP log',
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
                        try:
                            all_results[key][:,index_elm,:]=velocity_results[key][elm_time_ind-nwin:elm_time_ind+nwin,:]
                        except:
                            all_results[key][:,index_elm,:]=np.nan
                    else:
                        try:
                            all_results[key][:,index_elm]=velocity_results[key][elm_time_ind-nwin:elm_time_ind+nwin]
                        except:
                            all_results[key][:,index_elm]=np.nan
        pickle.dump(all_results, open(all_results_file, 'wb'))
    else:
        all_results=pickle.load(open(all_results_file, 'rb'))
        
    if plot_all_time_traces:
        pdf_pages_allsig=PdfPages(wd+'/plots/all_signal_one_plots.pdf')
        for key in all_results.keys():
            fig,ax=plt.subplots(figsize=(figure_size/2.54, figure_size/2.54/np.sqrt(2)))
            for ind_elm in range(len(all_results['Angular velocity ccf'][0,:])):
                ax.set_title(key+' all signal')
                ax.set_xlabel('$t-t_{ELM}$ [$\mu$s]')
                
                if 'Velocity' in key:
                    ax.plot(time_vec*1e3, all_results[key][:,ind_elm,0]/1e3)
                    ax.set_title(key+' radial all signal ')
                    ax.set_ylabel('$v_{rad}$ [km/s]')
                if 'Expansion' in key:
                    ax.plot(time_vec*1e3, all_results[key][:,ind_elm])
                    ax.set_title(key+' all signal ')
                    ax.set_ylabel('exp. vel. [a.u.]')
                if 'Angular' in key:
                    #ax.plot(time_vec*1e3,all_results[key][:,ind_elm]/1e3)
                    x=time_vec*1e3
                    y=all_results[key][:,ind_elm]*2.5e-6
                    nans, p= np.isnan(y), lambda z: z.nonzero()[0]
                    try:
                        y[nans]= np.interp(x[nans], x[~nans], y[~nans])
                    except:
                        pass
                    ax.plot(x, np.concatenate([np.flip(np.cumsum(np.flip(y[0:y.shape[0]//2]))),np.cumsum(y[y.shape[0]//2:])]))
                    ax.set_ylim(-np.pi,np.pi)
                    #ax.set_ylabel('$\omega$ [krad/s]')
                    ax.set_ylabel('angle [rad]')
            plt.tight_layout(pad=0.1)
            pdf_pages_allsig.savefig()
        pdf_pages_allsig.close()
            
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

    
    y_vector={'Velocity ccf FLAP radial':{'data':result_histograms['Velocity ccf FLAP'][:,:,0],
                                          'bins':(result_bins['Velocity ccf FLAP'][0:-1,0]+result_bins['Velocity ccf FLAP'][1:,0])/2e3,
                                          'bar width':(result_bins['Velocity ccf FLAP'][1,0]-result_bins['Velocity ccf FLAP'][0,0])/1e3,
                                          'median':moment_results['median']['Velocity ccf FLAP'][:,0]/1e3,
                                          '10th':moment_results['10percentile']['Velocity ccf FLAP'][:,0]/1e3,
                                          '90th':moment_results['90percentile']['Velocity ccf FLAP'][:,0]/1e3,
                                          'ylabel':'$v_{rad}$',
                                          'unit':'km/s',
                                          },
    
              'Velocity ccf FLAP poloidal':{'data':result_histograms['Velocity ccf FLAP'][:,:,1],
                                            'bins':(result_bins['Velocity ccf FLAP'][0:-1,1]+result_bins['Velocity ccf FLAP'][1:,1])/2e3,
                                            'bar width':(result_bins['Velocity ccf FLAP'][1,1]-result_bins['Velocity ccf FLAP'][0,1])/1e3,
                                            'median':moment_results['median']['Velocity ccf FLAP'][:,1]/1e3,
                                            '10th':moment_results['10percentile']['Velocity ccf FLAP'][:,1]/1e3,
                                            '90th':moment_results['90percentile']['Velocity ccf FLAP'][:,1]/1e3,
                                            'ylabel':'$v_{pol}$',
                                            'unit':'km/s',
                                            },
              
              'Velocity ccf skim radial':{'data':result_histograms['Velocity ccf skim'][:,:,0],
                                          'bins':(result_bins['Velocity ccf skim'][0:-1,0]+result_bins['Velocity ccf skim'][1:,0])/2e3,
                                          'bar width':(result_bins['Velocity ccf skim'][1,0]-result_bins['Velocity ccf skim'][0,0])/1e3,
                                          'median':moment_results['median']['Velocity ccf skim'][:,0]/1e3,
                                          '10th':moment_results['10percentile']['Velocity ccf skim'][:,0]/1e3,
                                          '90th':moment_results['90percentile']['Velocity ccf skim'][:,0]/1e3,
                                          'ylabel':'$v_{rad}$',
                                          'unit':'km/s',
                                          },
    
              'Velocity ccf skim poloidal':{'data':result_histograms['Velocity ccf skim'][:,:,1],
                                            'bins':(result_bins['Velocity ccf skim'][0:-1,1]+result_bins['Velocity ccf skim'][1:,1])/2e3,
                                            'bar width':(result_bins['Velocity ccf skim'][1,1]-result_bins['Velocity ccf skim'][0,1])/1e3,
                                            'median':moment_results['median']['Velocity ccf skim'][:,1]/1e3,
                                            '10th':moment_results['10percentile']['Velocity ccf skim'][:,1]/1e3,
                                            '90th':moment_results['90percentile']['Velocity ccf skim'][:,1]/1e3,
                                            'ylabel':'$v_{pol}$',
                                            'unit':'km/s',
                                            },   
              
              'Angular velocity ccf skim':{'data':result_histograms['Angular velocity ccf'],
                                           'bins':(result_bins['Angular velocity ccf'][0:-1]+result_bins['Angular velocity ccf'][1:])/2e3,
                                           'bar width':(result_bins['Angular velocity ccf'][1]-result_bins['Angular velocity ccf'][0])/1e3,
                                           'median':moment_results['median']['Angular velocity ccf']/1e3,
                                           '10th':moment_results['10percentile']['Angular velocity ccf']/1e3,
                                           '90th':moment_results['90percentile']['Angular velocity ccf']/1e3,
                                           'ylabel':'$\omega$',
                                           'unit':'krad/s',
                                           },
              
              'Angular velocity ccf skim log':{'data':result_histograms['Angular velocity ccf log'],
                                               'bins':(result_bins['Angular velocity ccf log'][0:-1]+result_bins['Angular velocity ccf log'][1:])/2e3,
                                               'bar width':(result_bins['Angular velocity ccf log'][1]-result_bins['Angular velocity ccf log'][0])/1e3,
                                               'median':moment_results['median']['Angular velocity ccf log']/1e3,
                                               '10th':moment_results['10percentile']['Angular velocity ccf log']/1e3,
                                               '90th':moment_results['90percentile']['Angular velocity ccf log']/1e3,
                                               'ylabel':'$\omega$',
                                               'unit':'krad/s',
                                               },
                            
              'Angular velocity ccf FLAP':{'data':result_histograms['Angular velocity ccf FLAP'],
                                           'bins':(result_bins['Angular velocity ccf FLAP'][0:-1]+result_bins['Angular velocity ccf FLAP'][1:])/2e3,
                                           'bar width':(result_bins['Angular velocity ccf FLAP'][1]-result_bins['Angular velocity ccf FLAP'][0])/1e3,
                                           'median':moment_results['median']['Angular velocity ccf FLAP']/1e3,
                                           '10th':moment_results['10percentile']['Angular velocity ccf FLAP']/1e3,
                                           '90th':moment_results['90percentile']['Angular velocity ccf FLAP']/1e3,
                                           'ylabel':'$\omega$',
                                           'unit':'krad/s',
                                           },
              
              'Angular velocity ccf FLAP log':{'data':result_histograms['Angular velocity ccf FLAP log'],
                                               'bins':(result_bins['Angular velocity ccf FLAP log'][0:-1]+result_bins['Angular velocity ccf FLAP log'][1:])/2e3,
                                               'bar width':(result_bins['Angular velocity ccf FLAP log'][1]-result_bins['Angular velocity ccf FLAP log'][0])/1e3,
                                               'median':moment_results['median']['Angular velocity ccf FLAP log']/1e3,
                                               '10th':moment_results['10percentile']['Angular velocity ccf FLAP log']/1e3,
                                               '90th':moment_results['90percentile']['Angular velocity ccf FLAP log']/1e3,
                                               'ylabel':'$\omega$',
                                               'unit':'krad/s',
                                               },
                                          
               'Expansion velocity ccf skim':{'data':result_histograms['Expansion velocity ccf'],
                                              'bins':(result_bins['Expansion velocity ccf'][0:-1]+result_bins['Expansion velocity ccf'][1:])/2,
                                              'bar width':(result_bins['Expansion velocity ccf'][1]-result_bins['Expansion velocity ccf'][0]),
                                              'median':moment_results['median']['Expansion velocity ccf'],
                                              '10th':moment_results['10percentile']['Expansion velocity ccf'],
                                              '90th':moment_results['90percentile']['Expansion velocity ccf'],
                                              'ylabel':'$f_{S}$',
                                              'unit':'',
                                              },
               
               'Expansion velocity ccf FLAP':{'data':result_histograms['Expansion velocity ccf FLAP'],
                                              'bins':(result_bins['Expansion velocity ccf FLAP'][0:-1]+result_bins['Expansion velocity ccf FLAP'][1:])/2,
                                              'bar width':(result_bins['Expansion velocity ccf FLAP'][1]-result_bins['Expansion velocity ccf FLAP'][0]),
                                              'median':moment_results['median']['Expansion velocity ccf FLAP'],
                                              '10th':moment_results['10percentile']['Expansion velocity ccf FLAP'],
                                              '90th':moment_results['90percentile']['Expansion velocity ccf FLAP'],
                                              'ylabel':'$f_{S}$',
                                              'unit':'',
                                              },
              }

    if pdf or plot:
        if pdf:
            if pdf_filename is None:
                pdf_filename=wd+'/plots/all_angular_velocity_results_histograms.pdf'
            pdf_object=PdfPages(pdf_filename)
            
        if not plot:
            import matplotlib
            matplotlib.use('agg')
            
        def fmt(x, pos):
            a = '{:3.2f}'.format(x)
            return a    
            
        for i in range(len(y_vector)):
            
            plt.figure()
            fig,ax=plt.subplots(figsize=figsize)
            im=ax.contourf(time_vec*1e3,
                           y_vector[i]['Bins'],
                           y_vector[i]['Data'].transpose(),
                           levels=n_hist,
                           )
            ax.plot(time_vec*1e3,
                     y_vector[i]['median'],
                     color='red')
            ax.plot(time_vec*1e3,
                     y_vector[i]['10th'],
                     color='white',
                     lw=linewidth/2)
            ax.plot(time_vec*1e3,
                     y_vector[i]['90th'],
                     color='white',
                     lw=linewidth/2)
            ax.set_ylim(np.min(y_vector[i]['10th']),np.max(y_vector[i]['90th']))
            ax.set_title(y_vector[i]['Title'].replace(" FLAP", ""))
            ax.set_xlabel('Time [$\mu$s]')
            ax.set_ylabel(y_vector[i]['ylabel']+' '+y_vector[i]['unit'])
            ax.xaxis.set_major_locator(MaxNLocator(5)) 
            ax.yaxis.set_major_locator(MaxNLocator(5)) 
            
            import matplotlib.ticker as ticker
            cbar=fig.colorbar(im, format=ticker.FuncFormatter(fmt))
            cbar.ax.tick_params(labelsize=6)
            plt.tight_layout(pad=0.1)
            if pdf:
                pdf_object.savefig()
            
        for i in range(len(y_vector)):
            plt.figure()
            fig,ax=plt.subplots(figsize=figsize)
            ax.bar(y_vector[i]['Bins'],y_vector[i]['Data'][nwin,:], width=y_vector[i]['Bar width'])
            plt.axvline(x=y_vector[i]['median'][nwin], ymin=0.0,ymax=1.0, color='red')
            plt.axvline(x=y_vector[i]['10th'][nwin], ymin=0.0,ymax=1.0, color='magenta')
            plt.axvline(x=y_vector[i]['90th'][nwin], ymin=0.0,ymax=1.0, color='magenta')
            ax.set_xlabel(y_vector[i]['ylabel']+' '+y_vector[i]['unit'])
            ax.set_ylabel('f(x)')
            ax.set_title(y_vector[i]['Title'].replace(" FLAP", ""))
            ax.xaxis.set_major_locator(MaxNLocator(5)) 
            ax.yaxis.set_major_locator(MaxNLocator(5)) 
            plt.tight_layout(pad=0.1)
            
            if pdf:
                pdf_object.savefig()
        
        if pdf:
            pdf_object.close()
            
        if not plot:
            matplotlib.use('qt5agg')
        
    if return_results:
        return time_vec, y_vector
    