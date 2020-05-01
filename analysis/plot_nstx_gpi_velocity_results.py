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

#Plot settings for publications
publication=False

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
                                   plot_error=True,
                                   normalized_velocity=False,
                                   normalized_structure=True,
                                   subtraction_order=1,
                                   opacity=0.2,
                                   correlation_threshold=0.6
                                   ):

    database_file='/Users/mlampert/work/NSTX_workspace/db/ELM_findings_mlampert_velocity_good.csv'
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
                     'GPI Dalpha':np.zeros([2*nwin]),
                     }   
    
    notnan_counter=copy.deepcopy(average_results)
    variance_results=copy.deepcopy(average_results)
    elm_counter=0.
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
            if ('GPI Dalpha' in velocity_results.keys() and index_elm == 0):
                average_results['GPI Dalpha']=np.zeros([2*nwin])
                variance_results['GPI Dalpha']=np.zeros([2*nwin])
            velocity_results['Velocity ccf'][np.where(velocity_results['Correlation max'] < correlation_threshold),:]=[np.nan,np.nan]
            time=velocity_results['Time']
            elm_time_interval_ind=np.where(np.logical_and(time >= elm_time-window_average,
                                                          time <= elm_time+window_average))
            elm_time=(time[elm_time_interval_ind])[np.argmin(velocity_results['Frame similarity'][elm_time_interval_ind])]
            elm_time_ind=np.argmin(np.abs(time-elm_time))
            try:
                for key in average_results.keys():
                    if len(average_results[key].shape) == 1:
                        ind_nan=np.isnan(velocity_results[key][elm_time_ind-nwin:elm_time_ind+nwin])
                        notnan_counter[key]+=np.logical_not(ind_nan)
                        if len(ind_nan) > 0 and key not in ['Frame similarity','Correlation max', 'GPI Dalpha']:
                            (velocity_results[key][elm_time_ind-nwin:elm_time_ind+nwin])[ind_nan]=0.
                        average_results[key]+=velocity_results[key][elm_time_ind-nwin:elm_time_ind+nwin]
                    else:
                        ind_nan_rad=np.isnan(velocity_results[key][elm_time_ind-nwin:elm_time_ind+nwin,0])
                        ind_nan_pol=np.isnan(velocity_results[key][elm_time_ind-nwin:elm_time_ind+nwin,1])
                        notnan_counter[key][:,0]+=np.logical_not(ind_nan_rad)
                        notnan_counter[key][:,1]+=np.logical_not(ind_nan_pol)
                        if len(ind_nan_rad) > 0 and len(ind_nan_pol) > 0:
                            (velocity_results[key][elm_time_ind-nwin:elm_time_ind+nwin,0])[ind_nan_rad]=0.
                            (velocity_results[key][elm_time_ind-nwin:elm_time_ind+nwin,1])[ind_nan_pol]=0.
                        average_results[key][:,0]+=velocity_results[key][elm_time_ind-nwin:elm_time_ind+nwin,0]
                        average_results[key][:,1]+=velocity_results[key][elm_time_ind-nwin:elm_time_ind+nwin,1]
            except:
                pass
            elm_counter+=1
    
    for key in average_results.keys():
        notnan_counter[key][np.where(notnan_counter[key] == 0)] = 1.
        if key in ['Frame similarity', 'Correlation max', 'GPI dalpha']:
            average_results[key]=average_results[key]/elm_counter
        elif not 'ccf' in key:
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

    for index_elm in range(len(elm_index)):
        #preprocess velocity results, tackle with np.nan and outliers
        shot=int(db.loc[elm_index[index_elm]]['Shot'])
        #define ELM time for all the cases
        elm_time=db.loc[elm_index[index_elm]]['ELM time']/1000.
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
            velocity_results['Velocity ccf'][np.where(velocity_results['Correlation max'] < correlation_threshold),:]=[np.nan,np.nan]
            time=velocity_results['Time']
            elm_time_interval_ind=np.where(np.logical_and(time >= elm_time-window_average,
                                                          time <= elm_time+window_average))
            elm_time=(time[elm_time_interval_ind])[np.argmin(velocity_results['Frame similarity'][elm_time_interval_ind])]
            elm_time_ind=np.argmin(np.abs(time-elm_time))
            #current_elm_time=velocity_results['Time'][elm_time_ind[index_elm]]
            try:
                for key in average_results.keys():
                    if len(average_results[key].shape) == 1:
                        ind_nan=np.isnan(velocity_results[key][elm_time_ind-nwin:elm_time_ind+nwin])
                        if len(ind_nan) > 0 and key not in ['Frame similarity','Correlation max', 'GPI Dalpha']:
                            (velocity_results[key][elm_time_ind-nwin:elm_time_ind+nwin])[ind_nan]=0.
                    else:
                        ind_nan_rad=np.isnan(velocity_results[key][elm_time_ind-nwin:elm_time_ind+nwin,0])
                        ind_nan_pol=np.isnan(velocity_results[key][elm_time_ind-nwin:elm_time_ind+nwin,1])
                        if len(ind_nan_rad) > 0 and len(ind_nan_pol) > 0:
                            (velocity_results[key][elm_time_ind-nwin:elm_time_ind+nwin,0])[ind_nan_rad]=0.
                            (velocity_results[key][elm_time_ind-nwin:elm_time_ind+nwin,1])[ind_nan_pol]=0.
                    variance_results[key]+=(velocity_results[key][elm_time_ind-nwin:elm_time_ind+nwin]-average_results[key])**2
            except:
                pass
    for key in variance_results.keys():
        if key in ['Frame similarity', 'Correlation max', 'GPI dalpha']:
            variance_results[key]=np.sqrt(variance_results[key]/elm_counter)
        elif not 'ccf' in key:
            if len(variance_results[key].shape) == 1:
                variance_results[key]=np.sqrt(variance_results[key]/(notnan_counter[key])) #SQRT FROM HERE
            else:
                variance_results[key][:,0]=np.sqrt(variance_results[key][:,0]/(notnan_counter[key][:,0]))
                variance_results[key][:,1]=np.sqrt(variance_results[key][:,1]/(notnan_counter[key][:,1]))
        else:
            if len(average_results[key].shape) == 1:
                variance_results[key]=np.sqrt(variance_results[key]/(notnan_counter[key]))
            else:
                variance_results[key][:,0]=np.sqrt(variance_results[key][:,0]/(notnan_counter[key][:,0]))
                variance_results[key][:,1]=np.sqrt(variance_results[key][:,1]/(notnan_counter[key][:,1])) #UNTIL HERE

    average_results['Tau']=(np.arange(2*nwin)*sampling_time-window_average)*1e3 #Let the results be in ms
    variance_results['Tau']=(np.arange(2*nwin)*sampling_time-window_average)*1e3 #Let the results be in ms
        #This is a bit unusual here, but necessary due to the structure size calculation based on the contours which are not plot
    
    if pdf or plot:
        plot_average_velocity_results(average_results=average_results,
                                      variance_results=variance_results,
                                      plot_error=plot_error,
                                      plot=plot,
                                      pdf=pdf,
                                      pdf_filename='NSTX_GPI_ALL_ELM_AVERAGE_RESULTS_'+str(correlation_threshold),
                                      normalized_velocity=normalized_velocity,
                                      opacity=opacity)
    if return_results:
        return average_results
    
def plot_kmeans_clustered_elms(ncluster=4,
                               plot=True,
                               plot_error=True,
                               pdf=True,
                               base='Size max',
                               radial=False,
                               poloidal=False,
                               normalized_velocity=False,
                               normalized_structure=True,
                               subtraction_order=1,
                               opacity=0.2,
                               kmeans_from='scikit',
                               kmeans_distance='correlation',
                               corr_threshold=0.7,
                               window_average=500e-6,
                               correlation_threshold=0.6):
    
    from sklearn.cluster import KMeans
    if window_average is None:
        window_average=500e-6
    sampling_time=2.5e-6
   
    nwin=int(window_average/sampling_time)
    if radial and poloidal:
        raise ValueError('Radial and poloidal cannot be set at the same time.')
    index=None
    if radial or poloidal:
        if radial:
            index=0
        if poloidal:
            index=1
    
    database_file='/Users/mlampert/work/NSTX_workspace/db/ELM_findings_mlampert_velocity_good.csv'
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
            velocity_results['Velocity ccf'][np.where(velocity_results['Correlation max'] < correlation_threshold),:]=[np.nan,np.nan]
            time=velocity_results['Time']
            elm_time_interval_ind=np.where(np.logical_and(time >= elm_time-window_average,
                                                          time <= elm_time+window_average))
            elm_time=(time[elm_time_interval_ind])[np.argmin(velocity_results['Frame similarity'][elm_time_interval_ind])]
            elm_time_ind=np.argmin(np.abs(time-elm_time))
            #current_elm_time=velocity_results['Time'][elm_time_ind[index_elm]]
            if base in ['Size', 'Position', 'Position', 'Centroid', 'COG', 'Velocity str']:
                base_full=base+' max'
                ind_nan=np.isnan(velocity_results[base_full][elm_time_ind-nwin:elm_time_ind+nwin,index])
                velocity_results[base_full][elm_time_ind-nwin:elm_time_ind+nwin][ind_nan]=0.
                cluster_vector=velocity_results[base_full][elm_time_ind-nwin:elm_time_ind+nwin,index]
                to_be_clustered.append(cluster_vector)
            elif base == 'Velocity ccf':
                base_full=base
                ind_nan=np.isnan(velocity_results[base_full][elm_time_ind-nwin:elm_time_ind+nwin,index])
                velocity_results[base_full][elm_time_ind-nwin:elm_time_ind+nwin][ind_nan]=0.
                cluster_vector=velocity_results[base_full][elm_time_ind-nwin:elm_time_ind+nwin,index]
                to_be_clustered.append(cluster_vector)
            else:
                if base != 'GPI Dalpha' and base != 'Str number':
                    base_full=base+' max'
                else:
                    base_full=base
                ind_nan=np.isnan(velocity_results[base_full][elm_time_ind-nwin:elm_time_ind+nwin])
                velocity_results[base_full][elm_time_ind-nwin:elm_time_ind+nwin][ind_nan]=0.
                cluster_vector=velocity_results[base_full][elm_time_ind-nwin:elm_time_ind+nwin]
                to_be_clustered.append(cluster_vector)
            elm_labels['ELM index'].append(elm_index[index_elm])
            elm_labels['Shot'].append(shot)
            elm_labels['ELM time'].append(elm_time)
    to_be_clustered=np.asarray(to_be_clustered)
    if kmeans_from == 'scikit':
        kmeans = KMeans(n_clusters=ncluster, random_state=0).fit(to_be_clustered)
        klabels=kmeans.labels_
    elif kmeans_from == 'flap':
        centres, klabels, dist = flap_nstx.analysis.kmeanssample(to_be_clustered, 
                                                                 ncluster, 
                                                                 nsample=len(elm_labels['Shot']),
                                                                 delta=.001, 
                                                                 maxiter=20, 
                                                                 metric=kmeans_distance, 
                                                                 verbose=0 )
        nlabel=[]
        [nlabel.append(len(np.where(klabels == i)[0])) for i in range(ncluster)]
            
#        print(centres, klabels, dist)
#        plt.figure()
#        plt.scatter(np.arange(len(klabels)),klabels)
#        plt.pause(0.001)
#        return np.sort(np.asarray(nlabel))
    elif kmeans_from == 'mlampert':
        clusters=[]
        n_elm=to_be_clustered.shape[0]
        dist_matrix=np.zeros([n_elm,n_elm])
        for i in range(n_elm):
            for j in range(n_elm):
                i_in=False
                j_in=False
                if kmeans_distance == 'correlation':
                    x=to_be_clustered[i,:]-np.mean(to_be_clustered[i,:])
                    y=to_be_clustered[j,:]-np.mean(to_be_clustered[j,:])
                    dist_matrix[i,j]=np.sum(x*y)/np.sqrt(np.sum(x**2)*np.sum(y**2))
                if kmeans_distance == 'cosine':
                    dist_matrix[i,j]=np.sum(to_be_clustered[i,:]*to_be_clustered[j,:])/np.sqrt(np.sum(to_be_clustered[i,:]**2)*np.sum(to_be_clustered[j,:]**2))
                if dist_matrix[i,j] > corr_threshold:
                    for i_cluster in range(len(clusters)):
                        if i in clusters[i_cluster]: 
                            i_in=i_cluster
                        if j in clusters[i_cluster]:
                            j_in=i_cluster
                    if i_in == False and j_in == False:
                        clusters.append([i])
                        if i != j:
                            clusters[-1].append(j)
                    if i_in != False and j_in == False:
                        clusters[i_in].append(j)
                    if j_in != False and i_in == False:
                        clusters[j_in].append(i)
        klabels=np.zeros(n_elm)
        for i in range(len(clusters)):
            for j in range(len(clusters[i])):
                klabels[clusters[i][j]]=i-1
                pass
        ncluster=int(np.max(klabels))+1

    elm_labels['kmeans label']=klabels

    all_average_results=[]
    all_variance_results=[]
    plot_labels=[]
    label_number=[]
    
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
        if 'GPI Dalpha' in velocity_results.keys():
            average_results['GPI Dalpha']=np.zeros([2*nwin])    
        notnan_counter=copy.deepcopy(average_results)
        label_boolean=np.where(klabels == label)
        variance_results=copy.deepcopy(average_results)
        elm_index=np.asarray(elm_index)
        elm_counter=0.
        print('The number of events in the cluster is: '+str(len(elm_index[label_boolean])))
        if len(elm_index[label_boolean]) < 3:
            print('A cluster with two or less events is not plotted.')
            continue
        for index_elm in range(len(elm_index[label_boolean])):
            #preprocess velocity results, tackle with np.nan and outliers
            shot=int(db.loc[elm_index[label_boolean][index_elm]]['Shot'])
            #define ELM time for all the cases
            elm_time=db.loc[elm_index[label_boolean][index_elm]]['ELM time']/1000.
            if normalized_velocity:
                if normalized_structure:
                    str_add='_ns'
                else:
                    str_add=''
                filename=flap_nstx.analysis.filename(exp_id=shot,
                                                     working_directory=wd+'/processed_data',
                                                     time_range=[elm_time-2e-3,elm_time+2e-3],
                                                     comment='ccf_velocity_pfit_o'+str(subtraction_order)+'fst_0.0'+str_add+'_nv',
                                                     extension='pickle')
            else:
                filename=wd+'/processed_data/'+db.loc[elm_index[index_elm]]['Filename']+'.pickle'
            status=db.loc[elm_index[label_boolean][index_elm]]['OK/NOT OK']
    
            if status != 'NO':
                velocity_results=pickle.load(open(filename, 'rb'))
                velocity_results['Velocity ccf'][np.where(velocity_results['Correlation max'] < correlation_threshold),:]=[np.nan,np.nan]
                time=velocity_results['Time']
                elm_time_interval_ind=np.where(np.logical_and(time >= elm_time-window_average,
                                                              time <= elm_time+window_average))
                elm_time=(time[elm_time_interval_ind])[np.argmin(velocity_results['Frame similarity'][elm_time_interval_ind])]
                elm_time_ind=np.argmin(np.abs(time-elm_time))
#                try:
                if True:
                    for key in average_results.keys():
                        if len(average_results[key].shape) == 1:
                            ind_nan=np.isnan(velocity_results[key][elm_time_ind-nwin:elm_time_ind+nwin])
                            notnan_counter[key]+=np.logical_not(ind_nan)
                            if len(ind_nan) > 0 and key not in ['Frame similarity','Correlation max', 'GPI Dalpha']:
                                (velocity_results[key][elm_time_ind-nwin:elm_time_ind+nwin])[ind_nan]=0.
                            average_results[key]+=velocity_results[key][elm_time_ind-nwin:elm_time_ind+nwin]
                        else:
                            ind_nan_rad=np.isnan(velocity_results[key][elm_time_ind-nwin:elm_time_ind+nwin,0])
                            ind_nan_pol=np.isnan(velocity_results[key][elm_time_ind-nwin:elm_time_ind+nwin,1])
                            notnan_counter[key][:,0]+=np.logical_not(ind_nan_rad)
                            notnan_counter[key][:,1]+=np.logical_not(ind_nan_pol)
                            if len(ind_nan_rad) > 0 and len(ind_nan_pol) > 0:
                                (velocity_results[key][elm_time_ind-nwin:elm_time_ind+nwin,0])[ind_nan_rad]=0.
                                (velocity_results[key][elm_time_ind-nwin:elm_time_ind+nwin,1])[ind_nan_pol]=0.
                            average_results[key][:,0]+=velocity_results[key][elm_time_ind-nwin:elm_time_ind+nwin,0]
                            average_results[key][:,1]+=velocity_results[key][elm_time_ind-nwin:elm_time_ind+nwin,1]
                    elm_counter+=1
#                except:
#                    print('Failed to add shot '+str(shot)+' @ '+str(elm_time)+' into the results.')
        
        for key in average_results.keys():
            notnan_counter[key][np.where(notnan_counter[key] == 0)] = 1.
            if key in ['Frame similarity', 'Correlation max', 'GPI dalpha']:
                average_results[key]/=elm_counter
            elif not 'ccf' in key:
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

            
        for index_elm in range(len(elm_index[label_boolean])):
            #preprocess velocity results, tackle with np.nan and outliers
            shot=int(db.loc[elm_index[label_boolean][index_elm]]['Shot'])
            #define ELM time for all the cases
            elm_time=db.loc[elm_index[label_boolean][index_elm]]['ELM time']/1000.
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
            status=db.loc[elm_index[label_boolean][index_elm]]['OK/NOT OK']
    
            if status != 'NO':
                velocity_results=pickle.load(open(filename, 'rb'))
                velocity_results['Velocity ccf'][np.where(velocity_results['Correlation max'] < correlation_threshold),:]=[np.nan,np.nan]
                time=velocity_results['Time']
                elm_time_interval_ind=np.where(np.logical_and(time >= elm_time-window_average,
                                                              time <= elm_time+window_average))
                elm_time=(time[elm_time_interval_ind])[np.argmin(velocity_results['Frame similarity'][elm_time_interval_ind])]
                elm_time_ind=np.argmin(np.abs(time-elm_time))
                #current_elm_time=velocity_results['Time'][elm_time_ind[index_elm]]
                for key in average_results.keys():
                    if len(average_results[key].shape) == 1:
                        ind_nan=np.isnan(velocity_results[key][elm_time_ind-nwin:elm_time_ind+nwin])
                        if len(ind_nan) > 0 and key not in ['Frame similarity','Correlation max', 'GPI Dalpha']:
                            (velocity_results[key][elm_time_ind-nwin:elm_time_ind+nwin])[ind_nan]=0.
                    else:
                        ind_nan_rad=np.isnan(velocity_results[key][elm_time_ind-nwin:elm_time_ind+nwin,0])
                        ind_nan_pol=np.isnan(velocity_results[key][elm_time_ind-nwin:elm_time_ind+nwin,1])
                        if len(ind_nan_rad) > 0 and len(ind_nan_pol) > 0:
                            (velocity_results[key][elm_time_ind-nwin:elm_time_ind+nwin,0])[ind_nan_rad]=0.
                            (velocity_results[key][elm_time_ind-nwin:elm_time_ind+nwin,1])[ind_nan_pol]=0.
                    variance_results[key]+=(velocity_results[key][elm_time_ind-nwin:elm_time_ind+nwin]-average_results[key])**2
                    
        for key in variance_results.keys():
            if key in ['Frame similarity', 'Correlation max', 'GPI dalpha']:
                variance_results[key]=np.sqrt(variance_results[key]/elm_counter)
            elif not 'ccf' in key:
                if len(variance_results[key].shape) == 1:
                    variance_results[key]=np.sqrt(variance_results[key]/(notnan_counter[key])) #SQRT FROM HERE
                else:
                    variance_results[key][:,0]=np.sqrt(variance_results[key][:,0]/(notnan_counter[key][:,0]))
                    variance_results[key][:,1]=np.sqrt(variance_results[key][:,1]/(notnan_counter[key][:,1]))
            else:
                if len(average_results[key].shape) == 1:
                    variance_results[key]=np.sqrt(variance_results[key]/(notnan_counter[key]))
                else:
                    variance_results[key][:,0]=np.sqrt(variance_results[key][:,0]/(notnan_counter[key][:,0]))
                    variance_results[key][:,1]=np.sqrt(variance_results[key][:,1]/(notnan_counter[key][:,1])) #UNTIL HERE
    
        average_results['Tau']=(np.arange(2*nwin)*sampling_time-window_average)*1e3 #Let the results be in ms
        variance_results['Tau']=(np.arange(2*nwin)*sampling_time-window_average)*1e3 #Let the results be in ms
        
        all_average_results.append(average_results)
        all_variance_results.append(variance_results)
        plot_labels.append(label)
        label_number.append(len(elm_index[label_boolean]))
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
        string=''
        if index is not None:
            if index == 0:
                string='radial'
            else: 
                string='poloidal'
        string+='_ct_'+str(correlation_threshold)
        pdf_filename='NSTX_GPI_ALL_ELM_AVERAGE_RESULT_kmeans_nc_'+str(ncluster)+'_label_'+str(plot_labels[ind])+'_evnum_'+str(label_number[ind])+'_'+base.replace(' ','_')+'_'+string
        plot_average_velocity_results(average_results=all_average_results[ind],
                                      variance_results=all_variance_results[ind],
                                      ylimits=ylimits,
                                      plot_error=plot_error,
                                      plot=plot,
                                      pdf=pdf,
                                      pdf_filename=pdf_filename,
                                      opacity=opacity)




def plot_average_velocity_results(average_results=None,
                                  variance_results=None,
                                  plot_error=True,
                                  pdf=True,
                                  plot=True,
                                  pdf_filename='NSTX_GPI_ALL_ELM_AVERAGE_RESULTS',
                                  ylimits=None,
                                  normalized_velocity=False,
                                  opacity=0.2):
    
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


    if pdf:
        wd=flap.config.get_all_section('Module NSTX_GPI')['Working directory']
        pdf_filename=wd+'/plots/'+pdf_filename
        if plot_error:
            pdf_filename+='_with_error'
        if normalized_velocity:
            pdf_filename+='_norm_vel'
        pdf_pages=PdfPages(pdf_filename+'.pdf')
    #Plotting the radial velocity from CCF        
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
                        alpha=opacity)
    ax.plot(average_results['Tau'][plot_index_structure], 
            average_results['Velocity str avg'][plot_index_structure,0], 
            linewidth=0.3,
            color='green')
    ax.plot(average_results['Tau'][plot_index_structure], 
             average_results['Velocity str max'][plot_index_structure,0], 
             linewidth=0.3,
             color='red')

    ax.set_xlabel('Tau [ms]')
    ax.set_ylabel('v_rad[m/s]')
    ax.set_title('Radial velocity of the average results. \n (blue: ccf, green: str avg, red: str max)')
    ax.set_xlim(tau_range)
    if ylimits is not None:
        ax.set_ylim(ylimits['Velocity ccf'][0,:])
    fig.tight_layout()
    if pdf:
        pdf_pages.savefig()
        
    #Plotting the radial velocity from the structures.
    fig, ax = plt.subplots()
    ax.plot(average_results['Tau'][plot_index_structure],
            average_results['Velocity str max'][plot_index_structure,0],
            color='red')
    ax.scatter(average_results['Tau'][plot_index_structure], 
                average_results['Velocity str max'][plot_index_structure,0], 
                s=5, 
                marker='o',
                color='red')
    if plot_error:
        x=average_results['Tau'][plot_index_structure]
        y=average_results['Velocity str max'][plot_index_structure,0]
        dy=variance_results['Velocity str max'][plot_index_structure,0]
        ax.fill_between(x,y-dy,y+dy,
                        color='gray',
                        alpha=opacity)
    ax.plot(average_results['Tau'][plot_index_structure],
            average_results['Velocity str avg'][plot_index_structure,0])
    ax.scatter(average_results['Tau'][plot_index_structure],
               average_results['Velocity str avg'][plot_index_structure,0], 
               s=5, 
               marker='o')
    
    ax.set_xlabel('Tau [ms]')
    ax.set_ylabel('v_rad[m/s]')
    ax.set_title('Radial velocity of the average (blue) and \n maximum (red) structures.')
    ax.set_xlim(tau_range)
    if ylimits is not None:
        ax.set_ylim(ylimits['Velocity str max'][1,:])
    fig.tight_layout()
    if pdf:
        pdf_pages.savefig()
        
    #Plotting the poloidal velocity from CCF
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
                        alpha=opacity)
    ax.plot(average_results['Tau'][plot_index_structure], 
            average_results['Velocity str avg'][plot_index_structure,1], 
            linewidth=0.3,
            color='green')
    ax.plot(average_results['Tau'][plot_index_structure], 
             average_results['Velocity str max'][plot_index_structure,1], 
             linewidth=0.3,
             color='red')        
    
    ax.set_xlabel('Tau [ms]')
    ax.set_ylabel('v_pol[m/s]')
    ax.set_title('Poloidal velocity of the average results. \n (blue: ccf, green: str avg, red: str max)')
    ax.set_xlim(tau_range)
    if ylimits is not None:
        ax.set_ylim(ylimits['Velocity ccf'][1,:])
    fig.tight_layout()
    if pdf:
        pdf_pages.savefig()

    #Plotting the poloidal velocity from the structures.
    fig, ax = plt.subplots()
    ax.plot(average_results['Tau'][plot_index_structure],
            average_results['Velocity str max'][plot_index_structure,1],
            color='red')
    ax.scatter(average_results['Tau'][plot_index_structure], 
                average_results['Velocity str max'][plot_index_structure,1], 
                s=5, 
                marker='o',
                color='red')

    if plot_error:
        x=average_results['Tau'][plot_index_structure]
        y=average_results['Velocity str max'][plot_index_structure,1]
        dy=variance_results['Velocity str max'][plot_index_structure,1]
        ax.fill_between(x,y-dy,y+dy,
                        color='gray',
                        alpha=opacity)
    ax.plot(average_results['Tau'][plot_index_structure],
            average_results['Velocity str avg'][plot_index_structure,1])    
    ax.scatter(average_results['Tau'][plot_index_structure],
               average_results['Velocity str avg'][plot_index_structure,1], 
               s=5, 
               marker='o')
    ax.set_xlabel('Tau [ms]')
    ax.set_ylabel('v_pol[m/s]')
    ax.set_title('Poloidal velocity of the average (blue) and \n maximum (red) structures.')
    ax.set_xlim(tau_range)
    if ylimits is not None:
        ax.set_ylim(ylimits['Velocity str max'][1,:])
    fig.tight_layout()
    if pdf:
        pdf_pages.savefig()
    
    #Plotting both radial and poloidal velocity from CCF        
    fig, ax = plt.subplots()
    
    ax.plot(average_results['Tau'][plot_index], 
             average_results['Velocity ccf'][plot_index,0])
    ax.scatter(average_results['Tau'][plot_index], 
                average_results['Velocity ccf'][plot_index,0], 
                s=5, 
                marker='o')
    ax.plot(average_results['Tau'][plot_index], 
             average_results['Velocity ccf'][plot_index,1],
             color='red')
    ax.scatter(average_results['Tau'][plot_index], 
               average_results['Velocity ccf'][plot_index,1],
               s=5, 
               marker='o',
               color='red')
    ax.set_xlabel('Tau [ms]')
    ax.set_ylabel('Velocity [m/s]')
    ax.set_title('Poloidal velocity (red) and radial velocity (blue)\n of the average results.')
    ax.set_xlim(tau_range)
    ax.grid()
    if ylimits is not None:
        ax.set_ylim(average_results['Velocity ccf'][plot_index,0].min(),
                    average_results['Velocity ccf'][plot_index,1].max())
    fig.tight_layout()
    if pdf:
        pdf_pages.savefig()
        
    #Plotting both radial and poloidal velocity from CCF        
    fig, ax = plt.subplots()
    
    ax.plot(average_results['Tau'][plot_index], 
            average_results['Velocity ccf'][plot_index,0]*average_results['Velocity ccf'][plot_index,1])
    ax.scatter(average_results['Tau'][plot_index], 
               average_results['Velocity ccf'][plot_index,0]*average_results['Velocity ccf'][plot_index,1], 
               s=5, 
               marker='o')

    ax.set_xlabel('Tau [ms]')
    ax.set_ylabel('Velocity^2 [m^2/s^2]')
    ax.set_title('Poloidal velocity*radial velocity of the average results.')
    ax.set_xlim(tau_range)

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
                        alpha=opacity)
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
        
    #Average signal in GPI
    if 'GPI Dalpha' in average_results.keys():
        fig, ax = plt.subplots()
        ax.plot(average_results['Tau'], 
                average_results['GPI Dalpha']) 
        
        ax.scatter(average_results['Tau'], 
                   average_results['GPI Dalpha'], 
                   s=5, 
                   marker='o')
        if plot_error:
            x=average_results['Tau']
            y=average_results['GPI Dalpha']
            dy=variance_results['GPI Dalpha']
            ax.fill_between(x,y-dy,y+dy,
                            color='gray',
                            alpha=opacity)
        ax.set_xlabel('Tau [ms]')
        ax.set_ylabel('GPI D alpha ')
        ax.set_title('Average GPI D alpha signal')
        ax.set_xlim(tau_range)
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
                        alpha=opacity)
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
                        alpha=opacity)
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
                        alpha=opacity)
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
                        alpha=opacity)
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
                        alpha=opacity)
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
                        alpha=opacity)
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
                        alpha=opacity)
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
                        alpha=opacity)        
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
                        alpha=opacity)        
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
                        alpha=opacity)
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
                        alpha=opacity)        
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
                marker='o')
    if plot_error:
        x=average_results['Tau']
        y=average_results['Str number']
        dy=variance_results['Str number']
        ax.fill_between(x,y-dy,y+dy,
                        color='gray',
                        alpha=opacity)           
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

def calculate_avg_tde_velocity_results():
    
    from statistics import harmonic_mean 
    
    database_file='/Users/mlampert/work/NSTX_workspace/ELM_findings_mlampert_velocity_good.csv'
    db=pandas.read_csv(database_file, index_col=0)
    elm_index=list(db.index)
    average_velocity=np.zeros([21,2])
    tau=(np.arange(21)/10.-1)*100e-6
    elm=0.
    for index_elm in range(len(elm_index)):
        #preprocess velocity results, tackle with np.nan and outliers
        shot=int(db.loc[elm_index[index_elm]]['Shot'])
        #define ELM time for all the cases
        elm_time=db.loc[elm_index[index_elm]]['ELM time']/1000.
        wd=flap.config.get_all_section('Module NSTX_GPI')['Working directory']
        status=db.loc[elm_index[index_elm]]['OK/NOT OK']
        elm_time_range=[elm_time-2e-3,elm_time+2e-3]
        if status != 'NO':
            filename_pol=wd+'/ELM_RESULTS_TDE_4ms/'+flap_nstx.analysis.filename(exp_id=shot, 
                                                                                time_range=elm_time_range,
                                                                                purpose='TDE poloidal velocity',
                                                                                extension='pickle')
            filename_rad=wd+'/ELM_RESULTS_TDE_4ms/'+flap_nstx.analysis.filename(exp_id=shot, 
                                                                                time_range=elm_time_range,
                                                                                purpose='TDE radial velocity',
                                                                                extension='pickle')
            data_pol=flap.load(filename_pol)
            data_rad=flap.load(filename_rad)
            #data_pol=np.mean(data_pol.data, axis=0)
            #data_rad=np.mean(data_rad.data, axis=0)
            print((np.argmax(np.abs(data_pol.data), axis=0)))
            #data_pol=[data_pol.data[np.argmax(np.abs(data_pol.data[:,i])),i] for i in range(len(data_pol.data[0,:]))]
            #data_rad=[data_rad.data[np.argmax(np.abs(data_rad.data[:,i])),i] for i in range(len(data_rad.data[0,:]))]
            data_pol=[harmonic_mean(data_pol.data[:,i]) for i in range(len(data_pol.data[0,:]))]
            data_rad=[harmonic_mean(data_rad.data[:,i]) for i in range(len(data_rad.data[0,:]))]
            ind_min=np.argmax(data_rad)
            ind_elm_range=slice(ind_min-10,ind_min+11)
            
            print(ind_elm_range)
            try:
                average_velocity[:,0]+=data_rad[ind_elm_range]
                average_velocity[:,1]+=data_pol[ind_elm_range]
                elm+=1
            except:
                continue
        average_velocity=average_velocity/elm
        
    plt.figure()
    plt.plot(tau, average_velocity[:,0])
    plt.show()
    plt.figure()
    plt.plot(tau, average_velocity[:,1])
    plt.show()
    
def calculate_avg_sz_velocity_results(window_average=500e-6,
                                      sampling_time=2.5e-6,
                                      pdf=False,
                                      plot=True,
                                      return_results=False,
                                      plot_error=True,
                                      normalized_velocity=False,
                                      normalized_structure=True,
                                      subtraction_order=1,
                                      opacity=0.2,
                                      correlation_threshold=0.6,
                                      plot_commulative_error=False,
                                      nocalc=False
                                      ):
    
    wd=flap.config.get_all_section('Module NSTX_GPI')['Working directory']
    database_file=wd+'/db/ELM_findings_mlampert_velocity_good.csv'
    db=pandas.read_csv(database_file, index_col=0)
    elm_index=list(db.index)
    
    nwin=int(window_average/sampling_time)
    
    elm_number=0.
    
    pickle_filename_all=wd+'/processed_data/all_sz_results.pickle'
    
    if not os.path.exists(pickle_filename_all) or nocalc is False:
        average_results={'Radial velocity':np.zeros([2*nwin]),
                         'Poloidal velocity':np.zeros([2*nwin]),
                         'Maximum correlation p':np.zeros([2*nwin]),
                         'Maximum correlation n':np.zeros([2*nwin]),
                         }
        
        own_average_results={'Radial velocity':np.zeros([2*nwin]),
                             'Poloidal velocity':np.zeros([2*nwin])}
        own_variance_results=copy.deepcopy(own_average_results)
        notnan_counter=copy.deepcopy(own_average_results)
        variance_results=copy.deepcopy(average_results)
        variance_results_commulative=copy.deepcopy(average_results)
        
        for index_elm in range(len(elm_index)):
            #preprocess velocity results, tackle with np.nan and outliers
            shot=int(db.loc[elm_index[index_elm]]['Shot'])
            #define ELM time for all the cases
            elm_time=db.loc[elm_index[index_elm]]['ELM time']/1000.
            status=db.loc[elm_index[index_elm]]['OK/NOT OK']
        
            if status != 'NO':
                flap.delete_data_object('*')
        
                print('Calculating '+str(shot)+ ' at '+str(elm_time))
                
        
                comment=''
                pickle_filename_sz=flap_nstx.analysis.filename(exp_id=shot,
                                                        working_directory=wd+'/processed_data',
                                                        time_range=[elm_time-1e-3,elm_time+1e-3],
                                                        purpose='sz velocity',
                                                        comment=comment,
                                                        extension='pickle')
                
                pickle_filename_ml=flap_nstx.analysis.filename(exp_id=shot,
                                                        working_directory=wd+'/processed_data',
                                                        time_range=[elm_time-2e-3,elm_time+2e-3],
                                                        comment='ccf_velocity_pfit_o1_fst_0.0_ns_nv',
                                                        extension='pickle')
    
                try:
                #if True:
                    results_sz=pickle.load(open(pickle_filename_sz, 'rb'))
                    velocity_results=pickle.load(open(pickle_filename_ml, 'rb'))
                    velocity_results['Velocity ccf'][np.where(velocity_results['Correlation max'] < correlation_threshold),:]=[np.nan,np.nan]
                    time_ml=velocity_results['Time']
                    elm_time_interval_ind=np.where(np.logical_and(time_ml >= elm_time-window_average,
                                                                  time_ml <= elm_time+window_average))
                    elm_time=(time_ml[elm_time_interval_ind])[np.argmin(velocity_results['Frame similarity'][elm_time_interval_ind])]
                    time_sz=results_sz['Time']
                    elm_time_ind_sz=np.argmin(np.abs(time_sz-elm_time))
                    elm_time_ind_ml=np.argmin(np.abs(time_ml-elm_time))
                    for key in average_results.keys():
                        average_results[key] += np.mean(results_sz[key][:,:,elm_time_ind_sz-nwin:elm_time_ind_sz+nwin], axis=(0,1))
                        variance_results_commulative[key] += np.var(results_sz[key][:,:,elm_time_ind_sz-nwin:elm_time_ind_sz+nwin], axis=(0,1))
                    
                    ind_nan_rad=np.isnan(velocity_results['Velocity ccf'][elm_time_ind_ml-nwin:elm_time_ind_ml+nwin,0])
                    ind_nan_pol=np.isnan(velocity_results['Velocity ccf'][elm_time_ind_ml-nwin:elm_time_ind_ml+nwin,1])
                    notnan_counter['Radial velocity']+=np.logical_not(ind_nan_rad)
                    notnan_counter['Poloidal velocity']+=np.logical_not(ind_nan_pol)
                    
                    if len(ind_nan_rad) > 0 and len(ind_nan_pol) > 0:
                        (velocity_results['Velocity ccf'][elm_time_ind_ml-nwin:elm_time_ind_ml+nwin,0])[ind_nan_rad]=0.
                        (velocity_results['Velocity ccf'][elm_time_ind_ml-nwin:elm_time_ind_ml+nwin,1])[ind_nan_pol]=0.
                    own_average_results['Radial velocity']+=velocity_results['Velocity ccf'][elm_time_ind_ml-nwin:elm_time_ind_ml+nwin,0]
                    own_average_results['Poloidal velocity']+=velocity_results['Velocity ccf'][elm_time_ind_ml-nwin:elm_time_ind_ml+nwin,1]
                    elm_number+=1
                except:
                    pass

        for key in average_results.keys():
            average_results[key]/=elm_number
            variance_results_commulative[key]=np.sqrt(variance_results_commulative[key])
        
        for key in own_average_results.keys():
            own_average_results[key] = own_average_results[key]/notnan_counter[key]
            
        for index_elm in range(len(elm_index)):
            #preprocess velocity results, tackle with np.nan and outliers
            shot=int(db.loc[elm_index[index_elm]]['Shot'])
            #define ELM time for all the cases
            elm_time=db.loc[elm_index[index_elm]]['ELM time']/1000.
            status=db.loc[elm_index[index_elm]]['OK/NOT OK']
        
            if status != 'NO':
                flap.delete_data_object('*')
        
                print('Calculating '+str(shot)+ ' at '+str(elm_time))
                elm_number+=1
        
                wd=flap.config.get_all_section('Module NSTX_GPI')['Working directory']
                comment=''
                pickle_filename_sz=flap_nstx.analysis.filename(exp_id=shot,
                                                        working_directory=wd+'/processed_data',
                                                        time_range=[elm_time-1e-3,elm_time+1e-3],
                                                        purpose='sz velocity',
                                                        comment=comment,
                                                        extension='pickle')
                pickle_filename_ml=flap_nstx.analysis.filename(exp_id=shot,
                                                        working_directory=wd+'/processed_data',
                                                        time_range=[elm_time-2e-3,elm_time+2e-3],
                                                        comment='ccf_velocity_pfit_o1_fst_0.0_ns_nv',
                                                        extension='pickle')
                try:            
                    results_sz=pickle.load(open(pickle_filename_sz, 'rb'))
                    velocity_results=pickle.load(open(pickle_filename_ml, 'rb'))
                    velocity_results['Velocity ccf'][np.where(velocity_results['Correlation max'] < correlation_threshold),:]=[np.nan,np.nan]
                    
                    time_ml=velocity_results['Time']
                    elm_time_interval_ind=np.where(np.logical_and(time_ml >= elm_time-window_average,
                                                                  time_ml <= elm_time+window_average))
                    
                    elm_time=(time_ml[elm_time_interval_ind])[np.argmin(velocity_results['Frame similarity'][elm_time_interval_ind])]
                    time_sz=results_sz['Time']
                    elm_time_ind_sz=np.argmin(np.abs(time_sz-elm_time))
                    elm_time_ind_ml=np.argmin(np.abs(time_ml-elm_time))
        
                    for key in average_results.keys():
                        variance_results[key] += (average_results[key]-np.mean(results_sz[key][:,:,elm_time_ind_sz-nwin:elm_time_ind_sz+nwin], axis=(0,1)))**2
                        
                    ind_nan_rad=np.isnan(velocity_results['Velocity ccf'][elm_time_ind_ml-nwin:elm_time_ind_ml+nwin,0])
                    ind_nan_pol=np.isnan(velocity_results['Velocity ccf'][elm_time_ind_ml-nwin:elm_time_ind_ml+nwin,1])
                    notnan_counter['Radial velocity']+=np.logical_not(ind_nan_rad)
                    notnan_counter['Poloidal velocity']+=np.logical_not(ind_nan_pol)
                    
                    if len(ind_nan_rad) > 0 and len(ind_nan_pol) > 0:
                        (velocity_results['Velocity ccf'][elm_time_ind_ml-nwin:elm_time_ind_ml+nwin,0])[ind_nan_rad]=0.
                        (velocity_results['Velocity ccf'][elm_time_ind_ml-nwin:elm_time_ind_ml+nwin,1])[ind_nan_pol]=0.
                        
                    own_variance_results['Radial velocity'] += (own_average_results['Radial velocity']-velocity_results['Velocity ccf'][elm_time_ind_ml-nwin:elm_time_ind_ml+nwin,0])**2
                    own_variance_results['Poloidal velocity'] += (own_average_results['Poloidal velocity']-velocity_results['Velocity ccf'][elm_time_ind_ml-nwin:elm_time_ind_ml+nwin,1])**2
    
                    elm_number+=1
                except:
                    pass
                
        for key in average_results.keys():
            variance_results[key]=np.sqrt(variance_results[key]/elm_number)
            
        for key in own_variance_results.keys():
            own_variance_results[key]=np.sqrt(own_variance_results[key]/(notnan_counter[key]))
        average_results['Tau']=(np.arange(2*nwin)*sampling_time-window_average)*1e3 #Let the results be in ms
        variance_results['Tau']=(np.arange(2*nwin)*sampling_time-window_average)*1e3 #Let the results be in m
        pickle.dump((average_results, variance_results, own_average_results, own_variance_results), open(pickle_filename_all, 'wb'))
    else:
        average_results, variance_results, own_average_results, own_variance_results=pickle.load(open(pickle_filename_all,'rb'))
    if pdf:
        pdf_filename=wd+'/plots/all_elm_results_from_sz_code'
        
        if plot_commulative_error:
            pdf_filename+='_comm_error'
        if plot_error:
            pdf_filename+='_with_error'
        pdf_object=PdfPages(pdf_filename+'.pdf')
    
    fig, ax = plt.subplots()
    ax.plot(average_results['Tau'], 
            average_results['Radial velocity']) 
    ax.scatter(average_results['Tau'], 
                average_results['Radial velocity'], 
                s=5, 
                marker='o')
    
    ax.plot(average_results['Tau'], 
            own_average_results['Radial velocity'], 
            color='red') 
    ax.scatter(average_results['Tau'], 
               own_average_results['Radial velocity'], 
               s=5, 
               marker='o',
               color='red')
    
    if plot_error:
        x=average_results['Tau']
        y=own_average_results['Radial velocity']
        dy=own_variance_results['Radial velocity']
        ax.fill_between(x,y-dy,y+dy,
                        color='red',
                        alpha=opacity)
        
        x=average_results['Tau']
        y=average_results['Radial velocity']
        dy=variance_results['Radial velocity']
        ax.fill_between(x,y-dy,y+dy,
                        color='tab:blue',
                        alpha=opacity)

    if plot_commulative_error:
        x=average_results['Tau']
        y=average_results['Radial velocity']
        dy=variance_results_commulative['Radial velocity']
        ax.fill_between(x,y-dy,y+dy,
                        color='gray',
                        alpha=opacity) 
    ax.set_xlabel('Tau [ms]')
    ax.set_ylabel('Radial velocity [m/s]')
    ax.set_title('Average radial velocity')

    fig.tight_layout()
    if pdf:
        pdf_object.savefig()
        
    fig, ax = plt.subplots()
    ax.plot(average_results['Tau'], 
            average_results['Poloidal velocity']) 
    ax.scatter(average_results['Tau'], 
                average_results['Poloidal velocity'], 
                s=5, 
                marker='o')
    ax.plot(average_results['Tau'], 
            own_average_results['Poloidal velocity'], 
            color='red') 
    ax.scatter(average_results['Tau'], 
               own_average_results['Poloidal velocity'], 
               s=5, 
               marker='o',
               color='red')
    if plot_error:
        x=average_results['Tau']
        y=average_results['Poloidal velocity']
        dy=variance_results['Poloidal velocity']
        ax.fill_between(x,y-dy,y+dy,
                        color='tab:blue',
                        alpha=opacity)   
        
        x=average_results['Tau']
        y=own_average_results['Poloidal velocity']
        dy=own_variance_results['Poloidal velocity']
        ax.fill_between(x,y-dy,y+dy,
                        color='red',
                        alpha=opacity)
        
    if plot_commulative_error:
        x=average_results['Tau']
        y=average_results['Poloidal velocity']
        dy=variance_results_commulative['Poloidal velocity']
        ax.fill_between(x,y-dy,y+dy,
                        color='gray',
                        alpha=opacity) 
    ax.set_xlabel('Tau [ms]')
    ax.set_ylabel('Poloidal velocity [m/s]')
    ax.set_title('Average poloidal velocity')

    fig.tight_layout()
    if pdf:
        pdf_object.savefig()
        
    fig, ax = plt.subplots()
    ax.plot(average_results['Tau'], 
            average_results['Maximum correlation n']) 
    ax.scatter(average_results['Tau'], 
                average_results['Maximum correlation n'], 
                s=5, 
                marker='o')
    ax.plot(average_results['Tau'], 
            average_results['Maximum correlation p']) 
    ax.scatter(average_results['Tau'], 
                average_results['Maximum correlation p'], 
                s=5, 
                marker='o')
    if plot_error:
        x=average_results['Tau']
        y=average_results['Maximum correlation n']
        dy=variance_results['Maximum correlation n']
        ax.fill_between(x,y-dy,y+dy,
                        color='tab:blue',
                        alpha=opacity)
        
        x=average_results['Tau']
        y=average_results['Maximum correlation p']
        dy=variance_results['Maximum correlation p']
        ax.fill_between(x,y-dy,y+dy,
                        color='red',
                        alpha=opacity)   
        
    ax.set_xlabel('Tau [ms]')
    ax.set_ylabel('Poloidal velocity [m/s]')
    ax.set_title('Average poloidal velocity')

    fig.tight_layout()
    if pdf:
        pdf_object.savefig()
        pdf_object.close()
    
    print(average_results['Maximum correlation p'])