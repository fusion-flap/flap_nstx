#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 16:43:09 2020

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
fn = os.path.join(thisdir,"flap_nstx.cfg")
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

def calculate_avg_velocity_results(window_average=500e-6,
                                   sampling_time=2.5e-6,
                                   pdf=False,
                                   plot=True,
                                   return_results=False,
                                   plot_error=True
                                   ):

    database_file='/Users/mlampert/work/NSTX_workspace/ELM_findings_mlampert_velocity_good.csv'
    db=pandas.read_csv(database_file, index_col=0)
    elm_index=list(db.index)
    
    nwin=int(window_average/sampling_time)
    average_results={'Velocity ccf':np.zeros([2*nwin,2]),
                     'Velocity str avg':np.zeros([2*nwin,2]),
                     'Velocity str max':np.zeros([2*nwin,2]),
                     'Frame similarity':np.zeros([2*nwin]),
                     'Correlation max':np.zeros([2*nwin]),
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
    
    variance_results=copy.deepcopy(average_results)
    notnan_counter_ccf=np.zeros([2*nwin])
    notnan_counter_str=np.zeros([2*nwin])
    for index_elm in range(len(elm_index)):
        #preprocess velocity results, tackle with np.nan and outliers
        shot=int(db.loc[elm_index[index_elm]]['Shot'])
        #define ELM time for all the cases
        elm_time=db.loc[elm_index[index_elm]]['ELM time']/1000.
        wd=flap.config.get_all_section('Module NSTX_GPI')['Working directory']
        filename=wd+'/'+db.loc[elm_index[index_elm]]['Filename']+'.pickle'
        status=db.loc[elm_index[index_elm]]['OK/NOT OK']

        if status != 'NO':
            velocity_results=pickle.load(open(filename, 'rb'))
            time=velocity_results['Time']
            elm_time_interval_ind=np.where(np.logical_and(time >= elm_time-window_average,
                                                          time <= elm_time+window_average))
            elm_time=(time[elm_time_interval_ind])[np.argmin(velocity_results['Frame similarity'][elm_time_interval_ind])]
            elm_time_ind=np.argmin(np.abs(time-elm_time))
            #current_elm_time=velocity_results['Time'][elm_time_ind[index_elm]]
            try:
            
                notnan_counter_ccf+=np.logical_not(np.isnan(velocity_results['Velocity ccf'][elm_time_ind-nwin:elm_time_ind+nwin,0]))
                notnan_counter_str+=np.logical_not(np.isnan(velocity_results['Velocity str avg'][elm_time_ind-nwin:elm_time_ind+nwin,0]))
                for key in average_results.keys():
                    ind_nan=np.isnan(velocity_results[key][elm_time_ind-nwin:elm_time_ind+nwin])
                    (velocity_results[key][elm_time_ind-nwin:elm_time_ind+nwin])[ind_nan]=0.
                    average_results[key]+=velocity_results[key][elm_time_ind-nwin:elm_time_ind+nwin]
                    
            except:
                print('Failed to add shot '+str(shot)+' @ '+str(elm_time)+' into the results.')
    
    for key in average_results.keys():
        if not 'ccf' in key:
            if len(average_results[key].shape) == 1:
                average_results[key]=average_results[key]/(notnan_counter_str-1)
            else:
                average_results[key][:,0]=average_results[key][:,0]/(notnan_counter_str-1)
                average_results[key][:,1]=average_results[key][:,1]/(notnan_counter_str-1)
        else:
            if len(average_results[key].shape) == 1:
                average_results[key]=average_results[key]/(notnan_counter_ccf-1)
            else:
                average_results[key][:,0]=average_results[key][:,0]/(notnan_counter_ccf-1)
                average_results[key][:,1]=average_results[key][:,1]/(notnan_counter_ccf-1)

    for index_elm in range(len(elm_index)):
        #preprocess velocity results, tackle with np.nan and outliers
        shot=int(db.loc[elm_index[index_elm]]['Shot'])
        #define ELM time for all the cases
        elm_time=db.loc[elm_index[index_elm]]['ELM time']/1000.
        filename=wd+'/'+db.loc[elm_index[index_elm]]['Filename']+'.pickle'
        status=db.loc[elm_index[index_elm]]['OK/NOT OK']

        if status != 'NO':
            velocity_results=pickle.load(open(filename, 'rb'))
            time=velocity_results['Time']
            elm_time_interval_ind=np.where(np.logical_and(time >= elm_time-window_average,
                                                          time <= elm_time+window_average))
            elm_time=(time[elm_time_interval_ind])[np.argmin(velocity_results['Frame similarity'][elm_time_interval_ind])]
            elm_time_ind=np.argmin(np.abs(time-elm_time))
            #current_elm_time=velocity_results['Time'][elm_time_ind[index_elm]]
            try:
                for key in average_results.keys():
                    ind_nan=np.isnan(velocity_results[key][elm_time_ind-nwin:elm_time_ind+nwin])
                    (velocity_results[key][elm_time_ind-nwin:elm_time_ind+nwin])[ind_nan]=0.
                    variance_results[key]+=(velocity_results[key][elm_time_ind-nwin:elm_time_ind+nwin]-average_results[key])**2
            except:
                pass
    for key in variance_results.keys():
        if not 'ccf' in key:
            if len(variance_results[key].shape) == 1:
                variance_results[key]=np.sqrt(variance_results[key]/(notnan_counter_str-2))
            else:
                variance_results[key][:,0]=np.sqrt(variance_results[key][:,0]/(notnan_counter_str-2))
                variance_results[key][:,1]=np.sqrt(variance_results[key][:,1]/(notnan_counter_str-2))
        else:
            if len(average_results[key].shape) == 1:
                variance_results[key]=np.sqrt(variance_results[key]/(notnan_counter_ccf-2))
            else:
                variance_results[key][:,0]=np.sqrt(variance_results[key][:,0]/(notnan_counter_ccf-2))
                variance_results[key][:,1]=np.sqrt(variance_results[key][:,1]/(notnan_counter_ccf-2))

    average_results['Tau']=(np.arange(2*nwin)*sampling_time-window_average)*1e3 #Let the results be in ms
    variance_results['Tau']=(np.arange(2*nwin)*sampling_time-window_average)*1e3 #Let the results be in ms
        #This is a bit unusual here, but necessary due to the structure size calculation based on the contours which are not plot
    if pdf or plot:

        plot_average_velocity_results(average_results=average_results,
                                      variance_results=variance_results,
                                      plot_error=plot_error,
                                      plot=plot,
                                      pdf=pdf)
    if return_results:
        return average_results
    
def plot_kmeans_clustered_elms(ncluster=4,
                               plot=True,
                               plot_error=True,
                               pdf=True,
                               base='Size max',
                               radial=False,
                               poloidal=False):
    
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    
    window_average=500e-6
    sampling_time=2.5e-6

    database_file='/Users/mlampert/work/NSTX_workspace/ELM_findings_mlampert_velocity_good.csv'
    db=pandas.read_csv(database_file, index_col=0)
    elm_index=list(db.index)
    
    nwin=int(window_average/sampling_time)
    if radial and poloidal:
        raise ValueError('Radial and poloidal cannot be set at the same time.')
    if radial or poloidal:
        if radial:
            index=0
        if poloidal:
            index=1
    
    database_file='/Users/mlampert/work/NSTX_workspace/ELM_findings_mlampert_velocity_good.csv'
    db=pandas.read_csv(database_file, index_col=0)
    elm_index=list(db.index)
    to_be_clustered=[]
    elm_labels={'ELM index':[],
                'Shot':[],
                'ELM time':[],
                'kmeans label':[]}
    for index_elm in range(len(elm_index)):
        #preprocess velocity results, tackle with np.nan and outliers
        shot=int(db.loc[elm_index[index_elm]]['Shot'])
        #define ELM time for all the cases
        elm_time=db.loc[elm_index[index_elm]]['ELM time']/1000.
        wd=flap.config.get_all_section('Module NSTX_GPI')['Working directory']
        filename=wd+'/'+db.loc[elm_index[index_elm]]['Filename']+'.pickle'
        status=db.loc[elm_index[index_elm]]['OK/NOT OK']
        
        if status != 'NO':
            velocity_results=pickle.load(open(filename, 'rb'))
            time=velocity_results['Time']
            elm_time_interval_ind=np.where(np.logical_and(time >= elm_time-window_average,
                                                          time <= elm_time+window_average))
            elm_time=(time[elm_time_interval_ind])[np.argmin(velocity_results['Frame similarity'][elm_time_interval_ind])]
            elm_time_ind=np.argmin(np.abs(time-elm_time))
            #current_elm_time=velocity_results['Time'][elm_time_ind[index_elm]]
            if base in ['Size','Position', 'Position', 'Centroid', 'COG', 'Velocity str']:
                base+=' max'
                cluster_vector=velocity_results[base][elm_time_ind-nwin:elm_time_ind+nwin,index]
                ind_nan=np.isnan(velocity_results[base][elm_time_ind-nwin:elm_time_ind+nwin,index])
                velocity_results[base][elm_time_ind-nwin:elm_time_ind+nwin][ind_nan]=0.
                to_be_clustered.append(cluster_vector)
            elif base == 'Velocity ccf':
                cluster_vector=velocity_results[base][elm_time_ind-nwin:elm_time_ind+nwin,index]
                ind_nan=np.isnan(velocity_results[base][elm_time_ind-nwin:elm_time_ind+nwin,index])
                velocity_results[base][elm_time_ind-nwin:elm_time_ind+nwin][ind_nan]=0.
                to_be_clustered.append(cluster_vector)
            else:
                base+=' max'                
                cluster_vector=velocity_results[base][elm_time_ind-nwin:elm_time_ind+nwin]
                ind_nan=np.isnan(velocity_results[base][elm_time_ind-nwin:elm_time_ind+nwin])
                velocity_results[base][elm_time_ind-nwin:elm_time_ind+nwin][ind_nan]=0.
                to_be_clustered.append(cluster_vector)
            elm_labels['ELM index'].append(elm_index[index_elm])
            elm_labels['Shot'].append(shot)
            elm_labels['ELM time'].append(elm_time)
    to_be_clustered=np.asarray(to_be_clustered)
    kmeans = KMeans(n_clusters=ncluster, random_state=0).fit(to_be_clustered)
    klabels=kmeans.labels_

    elm_labels['kmeans label']=klabels

    all_average_results=[]
    all_variance_results=[]
    plot_labels=[]
    
    for label in range(ncluster):
        
        average_results={'Velocity ccf':np.zeros([2*nwin,2]),
                         'Velocity str avg':np.zeros([2*nwin,2]),
                         'Velocity str max':np.zeros([2*nwin,2]),
                         'Frame similarity':np.zeros([2*nwin]),
                         'Correlation max':np.zeros([2*nwin]),
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
        
        label_boolean=np.where(klabels == label)
        variance_results=copy.deepcopy(average_results)
        notnan_counter_ccf=np.zeros([2*nwin])
        notnan_counter_str=np.zeros([2*nwin])
        elm_index=np.asarray(elm_index)
        elm_counter=0.
        print('The number of events in the cluster is: '+str(len(elm_index[label_boolean])))
        if len(elm_index[label_boolean]) == 1:
            print('A cluster with one event is not plotted.')
            continue
        for index_elm in range(len(elm_index[label_boolean])):
            #preprocess velocity results, tackle with np.nan and outliers
            shot=int(db.loc[elm_index[label_boolean][index_elm]]['Shot'])
            #define ELM time for all the cases
            elm_time=db.loc[elm_index[label_boolean][index_elm]]['ELM time']/1000.
            wd=flap.config.get_all_section('Module NSTX_GPI')['Working directory']
            filename=wd+'/'+db.loc[elm_index[label_boolean][index_elm]]['Filename']+'.pickle'
            status=db.loc[elm_index[label_boolean][index_elm]]['OK/NOT OK']
    
            if status != 'NO':
                velocity_results=pickle.load(open(filename, 'rb'))
                time=velocity_results['Time']
                elm_time_interval_ind=np.where(np.logical_and(time >= elm_time-window_average,
                                                              time <= elm_time+window_average))
                elm_time=(time[elm_time_interval_ind])[np.argmin(velocity_results['Frame similarity'][elm_time_interval_ind])]
                elm_time_ind=np.argmin(np.abs(time-elm_time))
                #current_elm_time=velocity_results['Time'][elm_time_ind[index_elm]]
                try:
                    notnan_counter_ccf+=np.logical_not(np.isnan(velocity_results['Velocity ccf'][elm_time_ind-nwin:elm_time_ind+nwin,0]))
                    notnan_counter_str+=np.logical_not(np.isnan(velocity_results['Velocity str avg'][elm_time_ind-nwin:elm_time_ind+nwin,0]))
                    for key in average_results.keys():
                        if key not in ['Frame similarity','Correlation max']:
                            ind_nan=np.isnan(velocity_results[key][elm_time_ind-nwin:elm_time_ind+nwin])
                            (velocity_results[key][elm_time_ind-nwin:elm_time_ind+nwin])[ind_nan]=0.
                        average_results[key]+=velocity_results[key][elm_time_ind-nwin:elm_time_ind+nwin]
                    elm_counter+=1
                except:
                    print('Failed to add shot '+str(shot)+' @ '+str(elm_time)+' into the results.')
        
        for key in average_results.keys():
            if key in ['Frame similarity', 'Correlation max']:
                average_results[key]/=elm_counter
            elif not 'ccf' in key:
                if len(average_results[key].shape) == 1:
                    average_results[key]=average_results[key]/(notnan_counter_str-1)
                else:
                    average_results[key][:,0]=average_results[key][:,0]/(notnan_counter_str)
                    average_results[key][:,1]=average_results[key][:,1]/(notnan_counter_str)
            else:
                if len(average_results[key].shape) == 1:
                    average_results[key]=average_results[key]/(notnan_counter_ccf-1)
                else:
                    average_results[key][:,0]=average_results[key][:,0]/(notnan_counter_ccf)
                    average_results[key][:,1]=average_results[key][:,1]/(notnan_counter_ccf)
            
        for index_elm in range(len(elm_index[label_boolean])):
            #preprocess velocity results, tackle with np.nan and outliers
            shot=int(db.loc[elm_index[label_boolean][index_elm]]['Shot'])
            #define ELM time for all the cases
            elm_time=db.loc[elm_index[label_boolean][index_elm]]['ELM time']/1000.
            filename=wd+'/'+db.loc[elm_index[label_boolean][index_elm]]['Filename']+'.pickle'
            status=db.loc[elm_index[label_boolean][index_elm]]['OK/NOT OK']
    
            if status != 'NO':
                velocity_results=pickle.load(open(filename, 'rb'))
                time=velocity_results['Time']
                elm_time_interval_ind=np.where(np.logical_and(time >= elm_time-window_average,
                                                              time <= elm_time+window_average))
                elm_time=(time[elm_time_interval_ind])[np.argmin(velocity_results['Frame similarity'][elm_time_interval_ind])]
                elm_time_ind=np.argmin(np.abs(time-elm_time))
                #current_elm_time=velocity_results['Time'][elm_time_ind[index_elm]]
                try:
                    for key in average_results.keys():
                        if key not in ['Frame similarity','Correlation max']:
                            ind_nan=np.isnan(velocity_results[key][elm_time_ind-nwin:elm_time_ind+nwin])
                            (velocity_results[key][elm_time_ind-nwin:elm_time_ind+nwin])[ind_nan]=0.
                        variance_results[key]+=(velocity_results[key][elm_time_ind-nwin:elm_time_ind+nwin]-average_results[key])**2
                except:
                    pass
                
        for key in variance_results.keys():
            if not 'ccf' in key:
                if len(variance_results[key].shape) == 1:
                    variance_results[key]=np.sqrt(variance_results[key]/(notnan_counter_str-1))
                else:
                    variance_results[key][:,0]=np.sqrt(variance_results[key][:,0]/(notnan_counter_str-1))
                    variance_results[key][:,1]=np.sqrt(variance_results[key][:,1]/(notnan_counter_str-1))
            else:
                if len(average_results[key].shape) == 1:
                    variance_results[key]=np.sqrt(variance_results[key]/(notnan_counter_ccf-1))
                else:
                    variance_results[key][:,0]=np.sqrt(variance_results[key][:,0]/(notnan_counter_ccf-1))
                    variance_results[key][:,1]=np.sqrt(variance_results[key][:,1]/(notnan_counter_ccf-1))
    
        average_results['Tau']=(np.arange(2*nwin)*sampling_time-window_average)*1e3 #Let the results be in ms
        variance_results['Tau']=(np.arange(2*nwin)*sampling_time-window_average)*1e3 #Let the results be in ms
        
        all_average_results.append(average_results)
        all_variance_results.append(variance_results)
        plot_labels.append(label)
    ylimits={}
        
    for index_labels in range(len(all_average_results)):
        for key in average_results.keys():
            if index_labels == 0 and len(all_average_results[index_labels][key].shape) == 1:
                ylimits[key]=np.asarray([(all_average_results[index_labels][key]-all_variance_results[index_labels][key]).min(),
                                         (all_average_results[index_labels][key]+all_variance_results[index_labels][key]).max()])
            elif index_labels == 0 and len(all_average_results[index_labels][key].shape) == 2:
                ylimits[key]=np.asarray([np.asarray([(all_average_results[index_labels][key][:,0]-all_variance_results[index_labels][key][:,0]).min(),
                                                     (all_average_results[index_labels][key][:,0]+all_variance_results[index_labels][key][:,0]).max()]),
                                         np.asarray([(all_average_results[index_labels][key][:,1]-all_variance_results[index_labels][key][:,1]).min(),
                                                     (all_average_results[index_labels][key][:,1]+all_variance_results[index_labels][key][:,1]).max()])])
            elif index_labels != 0 and len(all_average_results[index_labels][key].shape) == 1:
                ylimits[key]=np.asarray([np.asarray([ylimits[key][0],
                                                    (all_average_results[index_labels][key]-all_variance_results[index_labels][key]).min()]).min(),
                                         np.asarray([ylimits[key][1],
                                                    (all_average_results[index_labels][key]+all_variance_results[index_labels][key]).max()]).max()])
            elif index_labels != 0 and len(all_average_results[index_labels][key].shape) == 2:
                ylimits[key]=np.asarray([np.asarray([np.asarray([ylimits[key][0,0],
                                                                 (all_average_results[index_labels][key][:,0]-all_variance_results[index_labels][key][:,0]).min()]).min(),
                                                     np.asarray([ylimits[key][0,1],
                                                                 (all_average_results[index_labels][key][:,0]+all_variance_results[index_labels][key][:,0]).max()]).max()]),
                                        [np.asarray([ylimits[key][1,0],
                                                     (all_average_results[index_labels][key][:,1]-all_variance_results[index_labels][key][:,1]).min()]).min(),
                                         np.asarray([ylimits[key][1,1],
                                                     (all_average_results[index_labels][key][:,1]+all_variance_results[index_labels][key][:,1]).max()]).max()]])
    
    for ind in range(len(all_average_results)):
        plot_average_velocity_results(average_results=all_average_results[ind],
                                      variance_results=all_variance_results[ind],
                                      ylimits=ylimits,
                                      plot_error=plot_error,
                                      plot=plot,
                                      pdf=pdf,
                                      pdf_filename='NSTX_GPI_ALL_ELM_AVERAGE_RESULT_kmeans_label_'+str(plot_labels[ind])+'_'+base.replace(' ','_')+'_'+str(index))
        
def plot_average_velocity_results(average_results=None,
                                  variance_results=None,
                                  plot_error=True,
                                  pdf=True,
                                  plot=True,
                                  pdf_filename='NSTX_GPI_ALL_ELM_AVERAGE_RESULTS',
                                  ylimits=None):
    
    tau_range=[min(average_results['Tau']),max(average_results['Tau'])]
    if plot:
        import matplotlib
        matplotlib.use('QT5Agg')
        import matplotlib.pyplot as plt
    else:
        import matplotlib
        matplotlib.use('agg')
        import matplotlib.pyplot as plt
     
    plot_index=np.logical_not(np.isnan(average_results['Velocity ccf'][:,0]))
    plot_index_structure=np.logical_not(np.isnan(average_results['Elongation avg']))
    #Plotting the radial velocity
    if pdf:
        wd=flap.config.get_all_section('Module NSTX_GPI')['Working directory']
        pdf_filename=wd+'/'+pdf_filename
        if plot_error:
            pdf_filename+='_with_error'
        pdf_pages=PdfPages(pdf_filename+'.pdf')
        
    fig, ax = plt.subplots()
    
    ax.plot(average_results['Tau'][plot_index], 
             average_results['Velocity ccf'][plot_index,0])
    ax.scatter(average_results['Tau'][plot_index], 
                average_results['Velocity ccf'][plot_index,0], 
                s=5, 
                marker='o')
    if plot_error:
        x=average_results['Tau'][plot_index]
        y=average_results['Velocity ccf'][plot_index,0]
        dy=variance_results['Velocity ccf'][plot_index,0]
        ax.fill_between(x,y-dy,y+dy,
                        color='gray',
                        alpha=0.2)
    ax.plot(average_results['Tau'][plot_index], 
            average_results['Velocity str avg'][plot_index,0], 
            linewidth=0.5,
            color='red')
    ax.plot(average_results['Tau'][plot_index], 
             average_results['Velocity str max'][plot_index,0], 
             linewidth=0.5,
             color='green')

    ax.set_xlabel('Tau [ms]')
    ax.set_ylabel('v_rad[m/s]')
    ax.set_title('Radial velocity of '+'the average results.')
    ax.set_xlim(tau_range)
    if ylimits is not None:
        ax.set_ylim(ylimits['Velocity ccf'][0,:])
    fig.tight_layout()
    if pdf:
        pdf_pages.savefig()
    
    #Plotting the poloidal velocity
    fig, ax = plt.subplots()
    ax.plot(average_results['Tau'][plot_index], 
             average_results['Velocity ccf'][plot_index,1]) 
    ax.scatter(average_results['Tau'][plot_index], 
                average_results['Velocity ccf'][plot_index,1], 
                s=5, 
                marker='o')
    if plot_error:
        x=average_results['Tau'][plot_index]
        y=average_results['Velocity ccf'][plot_index,1]
        dy=variance_results['Velocity ccf'][plot_index,1]
        ax.fill_between(x,y-dy,y+dy,
                        color='gray',
                        alpha=0.2)
    ax.plot(average_results['Tau'][plot_index],
            average_results['Velocity str avg'][plot_index,1], 
            linewidth=0.5,
            color='red')
    ax.plot(average_results['Tau'][plot_index],
            average_results['Velocity str max'][plot_index,1], 
            linewidth=0.5,
            color='green')
    ax.set_xlabel('Tau [ms]')
    ax.set_ylabel('v_pol[m/s]')
    ax.set_title('Poloidal velocity of '+'the average results.')
    ax.set_xlim(tau_range)
    if ylimits is not None:
        ax.set_ylim(ylimits['Velocity ccf'][1,:])
    fig.tight_layout()
    if pdf:
        pdf_pages.savefig()
        
    #Plotting the correlations
    fig, ax = plt.subplots()
    #Frame similarity
    ax.plot(average_results['Tau'][plot_index_structure], 
             average_results['Frame similarity'][plot_index_structure],
             color='red') 
    ax.scatter(average_results['Tau'][plot_index_structure], 
               average_results['Frame similarity'][plot_index_structure], 
               s=5, 
               marker='o', 
               color='red')
    if plot_error:
        x=average_results['Tau'][plot_index_structure]
        y=average_results['Frame similarity'][plot_index_structure]
        dy=variance_results['Frame similarity'][plot_index_structure]
        ax.fill_between(x,y-dy,y+dy,
                        color='gray',
                        alpha=0.2)
    #Maximum correlation
    ax.plot(average_results['Tau'][plot_index_structure], 
             average_results['Correlation max'][plot_index_structure]) 
    
    ax.scatter(average_results['Tau'][plot_index_structure], 
                average_results['Correlation max'][plot_index_structure], 
                s=5, 
                marker='o')
    ax.set_xlabel('Tau [ms]')
    ax.set_ylabel('Correlation')
    ax.set_title('Maximum correlation (blue) and frame similarity (red) \n of the average results.')
    ax.set_xlim(tau_range)
    if ylimits is not None:
        ax.set_ylim(ylimits['Frame similarity'])
    fig.tight_layout()
    if pdf:
        pdf_pages.savefig()        
  
    #Plotting the radial size
    fig, ax = plt.subplots()
    #Maximum
    ax.plot(average_results['Tau'][plot_index_structure], 
             average_results['Size max'][:,0][plot_index_structure],
             color='red') 
    ax.scatter(average_results['Tau'][plot_index_structure], 
               average_results['Size max'][:,0][plot_index_structure], 
               s=5, 
               marker='o', 
               color='red')
    if plot_error:
        x=average_results['Tau'][plot_index_structure]
        y=average_results['Size max'][:,0][plot_index_structure]
        dy=variance_results['Size max'][:,0][plot_index_structure]
        ax.fill_between(x,y-dy,y+dy,
                        color='gray',
                        alpha=0.2)
    #Average
    ax.plot(average_results['Tau'][plot_index_structure], 
             average_results['Size avg'][:,0][plot_index_structure]) 
    
    ax.scatter(average_results['Tau'][plot_index_structure], 
                average_results['Size avg'][:,0][plot_index_structure], 
                s=5, 
                marker='o')
    ax.set_xlabel('Tau [ms]')
    ax.set_ylabel('Radial size [m]')
    ax.set_title('Average (blue) and maximum (red) radial\n size of structures of '+'the average results.')
    ax.set_xlim(tau_range)
    if ylimits is not None:
        ax.set_ylim(ylimits['Size max'][0,:])
    fig.tight_layout()
    if pdf:
        pdf_pages.savefig()
        
    #Plotting the poloidal size
    fig, ax = plt.subplots()
    #Maximum
    ax.plot(average_results['Tau'][plot_index_structure], 
             average_results['Size max'][:,1][plot_index_structure], 
             color='red') 
    ax.scatter(average_results['Tau'][plot_index_structure], 
                average_results['Size max'][:,1][plot_index_structure], 
                s=5, 
                marker='o', 
                color='red')
    if plot_error:
        x=average_results['Tau'][plot_index_structure]
        y=average_results['Size max'][:,1][plot_index_structure]
        dy=variance_results['Size max'][:,1][plot_index_structure]
        ax.fill_between(x,y-dy,y+dy,
                        color='gray',
                        alpha=0.2)
    #Average
    ax.plot(average_results['Tau'][plot_index_structure], 
             average_results['Size avg'][:,1][plot_index_structure]) 
    ax.scatter(average_results['Tau'][plot_index_structure], 
                average_results['Size avg'][:,1][plot_index_structure], 
                s=5, 
                marker='o')
    ax.set_xlabel('Tau [ms]')
    ax.set_ylabel('Poloidal size [m]')
    ax.set_title('Average (blue) and maximum (red) poloidal\n size of structures of '+'the average results.')    
    ax.set_xlim(tau_range)
    if ylimits is not None:
        ax.set_ylim(ylimits['Size max'][1,:])
    fig.tight_layout()
    if pdf:
        pdf_pages.savefig()    
        
    #Plotting the radial position of the fit ellipse
    fig, ax = plt.subplots()
    #Maximum
    ax.plot(average_results['Tau'][plot_index_structure], 
             average_results['Position max'][:,0][plot_index_structure], 
             color='red') 
    ax.scatter(average_results['Tau'][plot_index_structure], 
                average_results['Position max'][:,0][plot_index_structure], 
                s=5, 
                marker='o', 
                color='red')
    if plot_error:
        x=average_results['Tau'][plot_index_structure]
        y=average_results['Position max'][:,0][plot_index_structure]
        dy=variance_results['Position max'][:,0][plot_index_structure]
        ax.fill_between(x,y-dy,y+dy,
                        color='gray',
                        alpha=0.2)
    #Average
    ax.plot(average_results['Tau'][plot_index_structure], 
             average_results['Position avg'][:,0][plot_index_structure]) 
    ax.scatter(average_results['Tau'][plot_index_structure], 
                average_results['Position avg'][:,0][plot_index_structure], 
                s=5, 
                marker='o')
    ax.set_xlabel('Tau [ms]')
    ax.set_ylabel('Radial position [m]')            
    ax.set_title('Average (blue) and maximum (red) radial\n position of structures of '+'the average results.')   
    ax.set_xlim(tau_range)
    if ylimits is not None:
        ax.set_ylim(ylimits['Position max'][0,:])
    fig.tight_layout()
    if pdf:
        pdf_pages.savefig()                  

    #Plotting the radial centroid of the half path
    fig, ax = plt.subplots()
    #Maximum
    ax.plot(average_results['Tau'][plot_index_structure], 
             average_results['Centroid max'][plot_index_structure,0], 
             color='red') 
    ax.scatter(average_results['Tau'][plot_index_structure], 
                average_results['Centroid max'][plot_index_structure,0], 
                s=5, 
                marker='o', 
                color='red')
    if plot_error:
        x=average_results['Tau'][plot_index_structure]
        y=average_results['Centroid max'][:,0][plot_index_structure]
        dy=variance_results['Centroid max'][:,0][plot_index_structure]
        ax.fill_between(x,y-dy,y+dy,
                        color='gray',
                        alpha=0.2)
    #Average
    ax.plot(average_results['Tau'][plot_index_structure], 
             average_results['Centroid avg'][plot_index_structure,0]) 
    ax.scatter(average_results['Tau'][plot_index_structure], 
                average_results['Centroid avg'][plot_index_structure,0], 
                s=5, 
                marker='o')
    ax.set_xlabel('Tau [ms]')
    ax.set_ylabel('Radial centroid [m]')
    ax.set_title('Average (blue) and maximum (red) radial\n centroid of structures of '+'the average results.')   
    ax.set_xlim(tau_range)
    if ylimits is not None:
        ax.set_ylim(ylimits['Centroid max'][0,:])
    fig.tight_layout()
    if pdf:
        pdf_pages.savefig() 

    #Plotting the radial COG of the structure
    fig, ax = plt.subplots()
    #Maximum
    ax.plot(average_results['Tau'][plot_index_structure], 
             average_results['COG max'][plot_index_structure,0], 
             color='red') 
    ax.scatter(average_results['Tau'][plot_index_structure], 
                average_results['COG max'][plot_index_structure,0], 
                s=5, 
                marker='o', 
                color='red')      
    if plot_error:
        x=average_results['Tau'][plot_index_structure]
        y=average_results['COG max'][:,0][plot_index_structure]
        dy=variance_results['COG max'][:,0][plot_index_structure]
        ax.fill_between(x,y-dy,y+dy,
                        color='gray',
                        alpha=0.2)
    #Average
    ax.plot(average_results['Tau'][plot_index_structure], 
             average_results['COG avg'][plot_index_structure,0]) 
    ax.scatter(average_results['Tau'][plot_index_structure], 
                average_results['COG avg'][plot_index_structure,0], 
                s=5, 
                marker='o')
    ax.set_xlabel('Tau [ms]')
    ax.set_ylabel('Radial COG [m]')
    ax.set_title('Average (blue) and maximum (red) radial\n center of gravity of structures of '+'the average results.')   
    ax.set_xlim(tau_range)
    if ylimits is not None:
        ax.set_ylim(ylimits['COG max'][0,:])    
    fig.tight_layout()
    if pdf:
        pdf_pages.savefig()      
        
    #Plotting the poloidal position of the fit ellipse
    fig, ax = plt.subplots()
    #Maximum
    ax.plot(average_results['Tau'][plot_index_structure], 
             average_results['Position max'][:,1][plot_index_structure], 
             color='red')
    ax.scatter(average_results['Tau'][plot_index_structure], 
                average_results['Position max'][:,1][plot_index_structure], 
                s=5, 
                marker='o', 
                color='red')
    if plot_error:
        x=average_results['Tau'][plot_index_structure]
        y=average_results['Position max'][:,1][plot_index_structure]
        dy=variance_results['Position max'][:,1][plot_index_structure]
        ax.fill_between(x,y-dy,y+dy,
                        color='gray',
                        alpha=0.2)
    #Average
    ax.plot(average_results['Tau'][plot_index_structure], 
             average_results['Position avg'][:,1][plot_index_structure]) 
    ax.scatter(average_results['Tau'][plot_index_structure], 
                average_results['Position avg'][:,1][plot_index_structure], 
                s=5, 
                marker='o')     
    ax.set_xlabel('Tau [ms]')
    ax.set_ylabel('Poloidal position [m]')
    ax.set_title('Average (blue) and maximum (red) poloidal\n position of structures of '+'the average results.')   
    ax.set_xlim(tau_range)
    if ylimits is not None:
        ax.set_ylim(ylimits['Position max'][1,:])
    fig.tight_layout()
    if pdf:
        pdf_pages.savefig()                               
        
    #Plotting the poloidal centroid of the half path
    fig, ax = plt.subplots()
    #Maximum
    ax.plot(average_results['Tau'][plot_index_structure], 
             average_results['Centroid max'][plot_index_structure,1], 
             color='red') 
    ax.scatter(average_results['Tau'][plot_index_structure], 
                average_results['Centroid max'][plot_index_structure,1], 
                s=5, 
                marker='o', 
                color='red')
    if plot_error:
        x=average_results['Tau'][plot_index_structure]
        y=average_results['Centroid max'][:,1][plot_index_structure]
        dy=variance_results['Centroid max'][:,1][plot_index_structure]
        ax.fill_between(x,y-dy,y+dy,
                        color='gray',
                        alpha=0.2)
    #Average            
    ax.plot(average_results['Tau'][plot_index_structure], 
             average_results['Centroid avg'][plot_index_structure,1]) 
    ax.scatter(average_results['Tau'][plot_index_structure], 
                average_results['Centroid avg'][plot_index_structure,1], 
                s=5, 
                marker='o')
    ax.set_xlabel('Tau [ms]')
    ax.set_ylabel('Poloidal centroid [m]')
    ax.set_title('Average (blue) and maximum (red) poloidal\n centroid of structures of '+'the average results.')   
    ax.set_xlim(tau_range)
    if ylimits is not None:
        ax.set_ylim(ylimits['Centroid max'][1,:])    
    fig.tight_layout()
    if pdf:
        pdf_pages.savefig()          
        
    #Plotting the poloidal COG of the structure
    fig, ax = plt.subplots()
    #Maximum
    ax.plot(average_results['Tau'][plot_index_structure], 
             average_results['COG max'][plot_index_structure,1], 
             color='red') 
    ax.scatter(average_results['Tau'][plot_index_structure], 
                average_results['COG max'][plot_index_structure,1], 
                s=5, 
                marker='o', 
                color='red')
    if plot_error:
        x=average_results['Tau'][plot_index_structure]
        y=average_results['COG max'][:,1][plot_index_structure]
        dy=variance_results['COG max'][:,1][plot_index_structure]
        ax.fill_between(x,y-dy,y+dy,
                        color='gray',
                        alpha=0.2)        
    #Average            
    ax.plot(average_results['Tau'][plot_index_structure], 
             average_results['COG avg'][plot_index_structure,1]) 
    ax.scatter(average_results['Tau'][plot_index_structure], 
                average_results['COG avg'][plot_index_structure,1], 
                s=5, 
                marker='o')
    ax.set_xlabel('Tau [ms]')
    ax.set_ylabel('Poloidal COG [m]')
    ax.set_title('Average (blue) and maximum (red) radial\n center of gravity of structures of '+'the average results.')   
    ax.set_xlim(tau_range)
    if ylimits is not None:
        ax.set_ylim(ylimits['COG max'][1,:])        
    fig.tight_layout()
    if pdf:
        pdf_pages.savefig()
    
    #Plotting the elongation
    fig, ax = plt.subplots()
    #Maximum
    ax.plot(average_results['Tau'][plot_index_structure], 
             average_results['Elongation max'][plot_index_structure], 
             color='red') 
    ax.scatter(average_results['Tau'][plot_index_structure], 
                average_results['Elongation max'][plot_index_structure], 
                s=5, 
                marker='o', 
                color='red')
    if plot_error:
        x=average_results['Tau'][plot_index_structure]
        y=average_results['Elongation max'][plot_index_structure]
        dy=variance_results['Elongation max'][plot_index_structure]
        ax.fill_between(x,y-dy,y+dy,
                        color='gray',
                        alpha=0.2)        
    #Average
    ax.plot(average_results['Tau'][plot_index_structure], 
             average_results['Elongation avg'][plot_index_structure]) 
    ax.scatter(average_results['Tau'][plot_index_structure], 
                average_results['Elongation avg'][plot_index_structure], 
                s=5, 
                marker='o')     
    ax.set_xlabel('Tau [ms]')
    ax.set_ylabel('Elongation')
    ax.set_title('Average (blue) and maximum (red) elongation\n of structures of '+'the average results.')   
    ax.set_xlim(tau_range)
    if ylimits is not None:
        ax.set_ylim(ylimits['Elongation max'])     
    fig.tight_layout()
    if pdf:
        pdf_pages.savefig()                
    
    #Plotting the angle
    fig, ax = plt.subplots()
    #Maximum
    ax.plot(average_results['Tau'][plot_index_structure], 
             average_results['Angle max'][plot_index_structure], 
             color='red') 
    ax.scatter(average_results['Tau'][plot_index_structure], 
                average_results['Angle max'][plot_index_structure], 
                s=5, 
                marker='o', 
                color='red')
    if plot_error:
        x=average_results['Tau'][plot_index_structure]
        y=average_results['Angle max'][plot_index_structure]
        dy=variance_results['Angle max'][plot_index_structure]
        ax.fill_between(x,y-dy,y+dy,
                        color='gray',
                        alpha=0.2)
    #Average            
    ax.plot(average_results['Tau'][plot_index_structure], 
             average_results['Angle avg'][plot_index_structure]) 
    ax.scatter(average_results['Tau'][plot_index_structure], 
                average_results['Angle avg'][plot_index_structure], 
                s=5, 
                marker='o') 
    ax.set_xlabel('Tau [ms]')
    ax.set_ylabel('Angle [rad]')
    ax.set_title('Average (blue) and maximum (red) angle\n of structures of '+'the average results.')   
    ax.set_xlim(tau_range)
    if ylimits is not None:
        ax.set_ylim(ylimits['Angle max'])     
    fig.tight_layout()
    if pdf:
        pdf_pages.savefig()
        
    #Plotting the area
    fig, ax = plt.subplots()
    #Maximum
    ax.plot(average_results['Tau'][plot_index_structure], 
             average_results['Area max'][plot_index_structure], 
                color='red') 
    ax.scatter(average_results['Tau'][plot_index_structure], 
                average_results['Area max'][plot_index_structure], 
                s=5, 
                marker='o', 
                color='red')
    if plot_error:
        x=average_results['Tau'][plot_index_structure]
        y=average_results['Area max'][plot_index_structure]
        dy=variance_results['Area max'][plot_index_structure]
        ax.fill_between(x,y-dy,y+dy,
                        color='gray',
                        alpha=0.2)        
    #Average
    ax.plot(average_results['Tau'][plot_index_structure], 
             average_results['Area avg'][plot_index_structure]) 
    ax.scatter(average_results['Tau'][plot_index_structure], 
                average_results['Area avg'][plot_index_structure], 
                s=5, 
                marker='o')  
    ax.set_xlabel('Tau [ms]')
    ax.set_ylabel('Area [m2]')
    ax.set_title('Average (blue) and maximum (red) area\n of structures of '+'the average results.')   
    ax.set_xlim(tau_range)
    if ylimits is not None:
        ax.set_ylim(ylimits['Area max'])  
    fig.tight_layout()
    if pdf:
        pdf_pages.savefig()
        
    #Plotting the number of structures 
    fig, ax = plt.subplots()
    ax.plot(average_results['Tau'][:], 
             average_results['Str number'][:]) 
    ax.scatter(average_results['Tau'][:], 
                average_results['Str number'][:], 
                s=5, 
                marker='o', 
                color='red')
    if plot_error:
        x=average_results['Tau']
        y=average_results['Str number']
        dy=variance_results['Str number']
        ax.fill_between(x,y-dy,y+dy,
                        color='gray',
                        alpha=0.2)           
    ax.set_xlabel('Tau [ms]')
    ax.set_ylabel('Str number')
    ax.set_title('Number of structures vs. time of '+'the average results.')
    ax.set_xlim(tau_range)
    if ylimits is not None:
        ax.set_ylim(ylimits['Str number']) 
    fig.tight_layout()
    if pdf:
        pdf_pages.savefig()
    
    if pdf:
       pdf_pages.close()
    if not plot:
        plt.close('all')
        