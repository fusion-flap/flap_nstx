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
from matplotlib.gridspec import GridSpec
import pickle
import numpy as np
import matplotlib.cm as cm

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

def plot_nstx_gpi_velocity_distribution(window_average=500e-6,
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
                                        structure_finding_method='contour',
                                        interpolation=False,
                                        ):
    
    if elm_time_base not in ['frame similarity', 'radial velocity']:
        raise ValueError('elm_time_base should be either "frame similarity" or "radial velocity"')
    wd=flap.config.get_all_section('Module NSTX_GPI')['Working directory']
    all_results_file=wd+'/processed_data/all_results_file.pickle'
    database_file=wd+'/db/ELM_findings_mlampert_velocity_good.csv'

    db=pandas.read_csv(database_file, index_col=0)
    elm_index=list(db.index)
    n_elm=len(elm_index)
    
    nwin=int(window_average/sampling_time)
    time_vec=(np.arange(2*nwin)*sampling_time-window_average)*1e3
    
    all_results={'Velocity ccf':np.zeros([2*nwin,n_elm,2]),
                 'Velocity str max':np.zeros([2*nwin,n_elm,2]),
                 'Acceleration ccf':np.zeros([2*nwin,n_elm,2]),
                 'Frame similarity':np.zeros([2*nwin,n_elm]),
                 'Correlation max':np.zeros([2*nwin,n_elm]),   
                 'Size max':np.zeros([2*nwin,n_elm,2]),
                 'Position max':np.zeros([2*nwin,n_elm,2]),  
                 'Separatrix dist max':np.zeros([2*nwin,n_elm]),                            
                 'Centroid max':np.zeros([2*nwin,n_elm,2]),                            
                 'COG max':np.zeros([2*nwin,n_elm,2]),
                 'Area max':np.zeros([2*nwin,n_elm]),
                 'Elongation max':np.zeros([2*nwin,n_elm]),     
                 'Angle max':np.zeros([2*nwin,n_elm]),
                 'Str number':np.zeros([2*nwin,n_elm]),
                 'GPI Dalpha':np.zeros([2*nwin,n_elm]),
                 }
        
    hist_range_dict={'Velocity ccf':{'Radial':[-5e3,15e3],
                                     'Poloidal':[-25e3,5e3]},
                     'Velocity str max':{'Radial':[-10e3,20e3],
                                         'Poloidal':[-20e3,10e3]},
                                         
                     'Size max':{'Radial':[0,0.1],
                                 'Poloidal':[0,0.1]},
                                 
                     'Separatrix dist max':[-0.05,0.150],
                     'Elongation max':[-0.5,0.5],    
                     'Str number':[-0.5,10.5],
                     }
                     
    n_hist_str=int(hist_range_dict['Str number'][1]-hist_range_dict['Str number'][0])*4
    
    result_histograms={'Velocity ccf':np.zeros([2*nwin,n_hist,2]),
                       'Velocity str max':np.zeros([2*nwin,n_hist,2]),
                       'Size max':np.zeros([2*nwin,n_hist,2]),
                       'Separatrix dist max':np.zeros([2*nwin,n_hist]),                            
                       'Elongation max':np.zeros([2*nwin,n_hist]),     
                       'Str number':np.zeros([2*nwin,n_hist_str]),
                       }
    
    average_results={'Velocity ccf':np.zeros([2*nwin,2]),
                     'Velocity str max':np.zeros([2*nwin,2]),
                     'Size max':np.zeros([2*nwin,2]),
                     'Separatrix dist max':np.zeros([2*nwin]),                            
                     'Elongation max':np.zeros([2*nwin]),     
                     'Str number':np.zeros([2*nwin]),
                     }
    
    moment_results={'average':copy.deepcopy(average_results),
                    'median':copy.deepcopy(average_results),
                    '10percentile':copy.deepcopy(average_results),
                    '90percentile':copy.deepcopy(average_results),
                    }
    
    result_bins={'Velocity ccf':np.zeros([n_hist+1,2]),
                  'Velocity str max':np.zeros([n_hist+1,2]), 
                  'Size max':np.zeros([n_hist+1,2]),
                  'Separatrix dist max':np.zeros([n_hist+1]),
                  'Elongation max':np.zeros([n_hist+1]),    
                  'Str number':np.zeros([n_hist_str+1]),
                  }
    
    coeff_r=np.asarray([3.7183594,-0.77821046,1402.8097])/1000. #The coordinates are in meters, the coefficients are in mm
    coeff_z=np.asarray([0.18090118,3.0657776,70.544312])/1000.  #The coordinates are in meters, the coefficients are in mm
    coeff_r_new=3./800.
    coeff_z_new=3./800.
    
    if not nocalc:
        for index_elm in range(n_elm):
            #preprocess velocity results, tackle with np.nan and outliers
            shot=int(db.loc[elm_index[index_elm]]['Shot'])
            #define ELM time for all the cases
            elm_time=db.loc[elm_index[index_elm]]['ELM time']/1000.
            if structure_finding_method == 'contour':
                str_add_find_method=''
            elif structure_finding_method == 'watershed' and interpolation == False:
                str_add_find_method='_nointer_watershed'
            else:
                str_add_find_method=''
            if normalized_velocity:
                if normalized_structure:
                    str_add='_ns'
                else:
                    str_add=''
                filename=flap_nstx.tools.filename(exp_id=shot,
                                                  working_directory=wd+'/processed_data',
                                                  time_range=[elm_time-2e-3,elm_time+2e-3],
                                                  comment='ccf_velocity_pfit_o'+str(subtraction_order)+'_fst_0.0'+str_add+'_nv'+str_add_find_method,
                                                  extension='pickle')
            else:
                filename=wd+'/processed_data/'+db.loc[elm_index[index_elm]]['Filename']+'.pickle'
            status=db.loc[elm_index[index_elm]]['OK/NOT OK']
            
            if status != 'NO':
                velocity_results=pickle.load(open(filename, 'rb'))
                velocity_results['GPI Dalpha']=velocity_results['GPI Dalpha'][0]
                
                velocity_results['Separatrix dist avg']=np.zeros(velocity_results['Position avg'].shape[0])
                velocity_results['Separatrix dist max']=np.zeros(velocity_results['Position max'].shape[0])
                
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
                    GPI_z_vert=coeff_z[0]*np.arange(80)/80*64+coeff_z[1]*np.arange(80)+coeff_z[2]
                    R_sep_GPI_interp=np.interp(GPI_z_vert,np.flip(z_sep_GPI),np.flip(R_sep_GPI))
                    z_sep_GPI_interp=GPI_z_vert
                    for key in ['max','avg']:
                        for ind_time in range(len(velocity_results['Position '+key][:,0])):
                            velocity_results['Separatrix dist '+key][ind_time]=np.min(np.sqrt((velocity_results['Position '+key][ind_time,0]-R_sep_GPI_interp)**2 + 
                                                                                              (velocity_results['Position '+key][ind_time,1]-z_sep_GPI_interp)**2))
                            ind_z_min=np.argmin(np.abs(z_sep_GPI-velocity_results['Position '+key][ind_time,1]))
                            if z_sep_GPI[ind_z_min] >= velocity_results['Position '+key][ind_time,1]:
                                ind1=ind_z_min
                                ind2=ind_z_min+1
                            else:
                                ind1=ind_z_min-1
                                ind2=ind_z_min
                                
                            radial_distance=velocity_results['Position '+key][ind_time,0]-((velocity_results['Position '+key][ind_time,1]-z_sep_GPI[ind2])/(z_sep_GPI[ind1]-z_sep_GPI[ind2])*(R_sep_GPI[ind1]-R_sep_GPI[ind2])+R_sep_GPI[ind2])
                            if radial_distance < 0:
                                velocity_results['Separatrix dist '+key][ind_time]*=-1
                except:
                    pass
    
                if gpi_plane_calculation:
                    det=coeff_r[0]*coeff_z[1]-coeff_z[0]*coeff_r[1]
                    
                    for key in ['Velocity ccf','Velocity str max','Velocity str avg','Size max','Size avg']:
                        orig=copy.deepcopy(velocity_results[key])
                        velocity_results[key][:,0]=coeff_r_new/det*(coeff_z[1]*orig[:,0]-coeff_r[1]*orig[:,1])
                        velocity_results[key][:,1]=coeff_z_new/det*(-coeff_z[0]*orig[:,0]+coeff_r[0]*orig[:,1])
                        
                    velocity_results['Elongation max'][:]=-(velocity_results['Size max'][:,0]-velocity_results['Size max'][:,1])/(velocity_results['Size max'][:,0]+velocity_results['Size max'][:,1])
                
                velocity_results['Acceleration ccf']=copy.deepcopy(velocity_results['Velocity ccf'])
                velocity_results['Acceleration ccf'][1:,0]=(velocity_results['Velocity ccf'][1:,0]-velocity_results['Velocity ccf'][0:-1,0])/(2*sampling_time)
                velocity_results['Acceleration ccf'][1:,1]=(velocity_results['Velocity ccf'][1:,1]-velocity_results['Velocity ccf'][0:-1,1])/(2*sampling_time)
                    
                corr_thres_index=np.where(velocity_results['Correlation max'] < correlation_threshold)

                velocity_results['Velocity ccf'][corr_thres_index,:]=[np.nan,np.nan]
                velocity_results['Size max'][corr_thres_index,:]=[np.nan,np.nan]
                velocity_results['Elongation max'][corr_thres_index]=np.nan
                
                velocity_results['GPI Dalpha'][corr_thres_index]=np.nan
                velocity_results['Correlation max'][corr_thres_index]=np.nan
                velocity_results['Separatrix dist max'][corr_thres_index]=np.nan
                velocity_results['Str number'][corr_thres_index]=np.nan
                
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
            if key == 'Str number':
                n_hist_cur=n_hist_str
            else:
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

    y_vector=[{'Title':'Radial velocity ccf',
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
               
               {'Title':'Radial velocity str',
               'Data':result_histograms['Velocity str max'][:,:,0],
               'Bins':(result_bins['Velocity str max'][0:-1,0]+result_bins['Velocity str max'][1:,0])/2e3,
               'Bar width':(result_bins['Velocity str max'][1,0]-result_bins['Velocity str max'][0,0])/1e3,
               'ylabel':'v_rad str [m/s]',
               },
                              
              {'Title':'Poloidal velocity str',
               'Data':result_histograms['Velocity str max'][:,:,1],
               'Bins':(result_bins['Velocity str max'][0:-1,1]+result_bins['Velocity str max'][1:,1])/2e3,
               'Bar width':(result_bins['Velocity str max'][1,1]-result_bins['Velocity str max'][0,1])/1e3,
               'ylabel':'v_pol str [m/s]',
               },
              
              {'Title':'Size max radial',
               'Data':result_histograms['Size max'][:,:,0],
               'Bins':(result_bins['Size max'][0:-1,0]+result_bins['Size max'][1:,0])/2*1e3,
               'Bar width':(result_bins['Size max'][1,0]-result_bins['Size max'][0,0])*1e3,
               'ylabel':'Radial size [mm]',
               },
               
              {'Title':'Size max poloidal',
               'Data':result_histograms['Size max'][:,:,1],
               'Bins':(result_bins['Size max'][0:-1,1]+result_bins['Size max'][1:,1])/2*1e3,
               'Bar width':(result_bins['Size max'][1,1]-result_bins['Size max'][0,1])*1e3,
               'ylabel':'Poloidal size [mm]',
               },               
              
              {'Title':'Elongation max',
               'Data':result_histograms['Elongation max'],
               'Bins':(result_bins['Elongation max'][0:-1]+result_bins['Elongation max'][1:])/2,
               'Bar width':(result_bins['Elongation max'][1]-result_bins['Elongation max'][0]),
               'ylabel':'Elongation [a.u.]',
               },
               
               {'Title':'Str number',
               'Data':result_histograms['Str number'],
               'Bins':(result_bins['Str number'][0:-1]+result_bins['Str number'][1:])/2,
               'Bar width':1,
               'ylabel':'Str number',
               },
                
               {'Title':'Distance',
                'Data':result_histograms['Separatrix dist max'],
                'Bins':(result_bins['Separatrix dist max'][0:-1]+result_bins['Separatrix dist max'][1:])/2*1e3,
                'Bar width':(result_bins['Separatrix dist max'][1]-result_bins['Separatrix dist max'][0])*1e3,
                'ylabel':'Distance [mm]',
               },
              ]

    y_vector_avg=[{'Title':'Radial velocity ccf',
                   'Data':moment_results['median']['Velocity ccf'][:,0]/1e3,
                   '10th':moment_results['10percentile']['Velocity ccf'][:,0]/1e3,
                   '90th':moment_results['90percentile']['Velocity ccf'][:,0]/1e3,
                   'ylabel':'v_rad [m/s]',
                   },
        
                  {'Title':'Poloidal velocity ccf',
                   'Data':moment_results['median']['Velocity ccf'][:,1]/1e3,
                   '10th':moment_results['10percentile']['Velocity ccf'][:,1]/1e3,
                   '90th':moment_results['90percentile']['Velocity ccf'][:,1]/1e3,
                   'ylabel':'v_pol [m/s]',
                   },
                   
                   {'Title':'Radial velocity str',
                    'Data':moment_results['median']['Velocity str max'][:,0]/1e3,
                    '10th':moment_results['10percentile']['Velocity str max'][:,0]/1e3,
                    '90th':moment_results['90percentile']['Velocity str max'][:,0]/1e3,
                    'ylabel':'v_rad str [m/s]',
                   },
                                  
                  {'Title':'Poloidal velocity str',
                   'Data':moment_results['median']['Velocity str max'][:,1]/1e3,
                   '10th':moment_results['10percentile']['Velocity str max'][:,1]/1e3,
                   '90th':moment_results['90percentile']['Velocity str max'][:,1]/1e3,
                   'ylabel':'v_pol str [m/s]',
                   },
                  
                  {'Title':'Size max radial',
                   'Data':moment_results['median']['Size max'][:,0]*1e3,
                   '10th':moment_results['10percentile']['Size max'][:,0]*1e3,
                   '90th':moment_results['90percentile']['Size max'][:,0]*1e3,
                   'ylabel':'Radial size [mm]',
                   },
                   
                  {'Title':'Size max poloidal',
                   'Data':moment_results['median']['Size max'][:,1]*1e3,
                   '10th':moment_results['10percentile']['Size max'][:,1]*1e3,
                   '90th':moment_results['90percentile']['Size max'][:,1]*1e3,
                   'ylabel':'Poloidal size [mm]',
                   },               
                  
                  {'Title':'Elongation max',
                   'Data':moment_results['median']['Elongation max'],
                   '10th':moment_results['10percentile']['Elongation max'],
                   '90th':moment_results['90percentile']['Elongation max'],
                   'ylabel':'Elongation [a.u.]',
                   },
                  {'Title':'Str number',
                   'Data':moment_results['median']['Str number'],
                   '10th':moment_results['10percentile']['Str number'],
                   '90th':moment_results['90percentile']['Str number'],
                   'ylabel':'Str number',
                   },
                  {'Title':'Separatrix dist.',
                   'Data':moment_results['median']['Separatrix dist max']*1e3,
                   '10th':moment_results['10percentile']['Separatrix dist max']*1e3,
                   '90th':moment_results['90percentile']['Separatrix dist max']*1e3,
                   'ylabel':'Dist. [mm]'} 
                  ]
    if general_plot:
        pdf_object=PdfPages(wd+'/plots/all_velocity_results_histograms.pdf')
        
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
        
    else:
        if plot_for_structure:
            pdf_object=PdfPages(wd+'/plots/all_velocity_results_histograms_publication.pdf')
            gs=GridSpec(5,2)
            plt.figure()   
            ax,fig=plt.subplots(figsize=(8.5/2.54,8))
            index_to_plot=[4,5,6,7,8]
            for i in range(5):
                plt.subplot(gs[i,0])
                plt.contourf(time_vec,
                            y_vector[index_to_plot[i]]['Bins'],
                            y_vector[index_to_plot[i]]['Data'].transpose(),
                            levels=n_hist,
                            )
                plt.plot(time_vec,
                         y_vector_avg[index_to_plot[i]]['Data'],
                         color='red')
                plt.plot(time_vec,
                         y_vector_avg[index_to_plot[i]]['10th'],
                         color='magenta')
                plt.plot(time_vec,
                         y_vector_avg[index_to_plot[i]]['90th'],
                         color='magenta')
                plt.title('Relative frequency of '+y_vector[index_to_plot[i]]['ylabel'])
                plt.xlabel('Time [ms]')
                plt.ylabel(y_vector[index_to_plot[i]]['ylabel'])
                plt.colorbar()
                
                plt.subplot(gs[i,1])
                plt.bar(y_vector[index_to_plot[i]]['Bins'],
                       y_vector[index_to_plot[i]]['Data'][nwin,:], 
                       width=y_vector[index_to_plot[i]]['Bar width'])
                
                plt.axvline(x=y_vector_avg[index_to_plot[i]]['Data'][nwin], ymin=0.0,ymax=1.0, color='red')
                plt.axvline(x=y_vector_avg[index_to_plot[i]]['10th'][nwin], ymin=0.0,ymax=1.0, color='magenta')
                plt.axvline(x=y_vector_avg[index_to_plot[i]]['90th'][nwin], ymin=0.0,ymax=1.0, color='magenta')
                
                plt.xlabel(y_vector[index_to_plot[i]]['ylabel'])
                plt.ylabel('f(x)')
                plt.title('Probablity distibution of '+y_vector[index_to_plot[i]]['ylabel'])
    #        fig.tight_layout()
            pdf_object.savefig()
            pdf_object.close()
        elif plot_for_velocity:
            pdf_object=PdfPages(wd+'/plots/all_velocity_results_histograms_publication_velocity.pdf')
            gs=GridSpec(2,2)
            plt.figure()   
            ax,fig=plt.subplots(figsize=(17/2.54,12/2.54))
            index_to_plot=[0,1]
            print(time_vec[nwin])
            for i in range(2):
                plt.subplot(gs[0,i])
                plt.contourf(time_vec,
                            y_vector[index_to_plot[i]]['Bins'],
                            y_vector[index_to_plot[i]]['Data'].transpose(),
                            levels=n_hist,
                            )
                plt.plot(time_vec,
                         y_vector_avg[index_to_plot[i]]['Data'],
                         color='red')
                plt.plot(time_vec,
                         y_vector_avg[index_to_plot[i]]['10th'],
                         color='magenta')
                plt.plot(time_vec,
                         y_vector_avg[index_to_plot[i]]['90th'],
                         color='magenta')
                
                plt.title('Relative frequency of '+y_vector[index_to_plot[i]]['ylabel'])
                plt.xlabel('Time [ms]')
                plt.ylabel(y_vector[index_to_plot[i]]['ylabel'])
                plt.colorbar()
                
                plt.subplot(gs[1,i])
                plt.bar(y_vector[index_to_plot[i]]['Bins'],
                       y_vector[index_to_plot[i]]['Data'][nwin,:], 
                       width=y_vector[index_to_plot[i]]['Bar width'])
                plt.axvline(x=y_vector_avg[index_to_plot[i]]['Data'][nwin], ymin=0.0,ymax=1.0, color='red')
                plt.axvline(x=y_vector_avg[index_to_plot[i]]['10th'][nwin], ymin=0.0,ymax=1.0, color='magenta')
                plt.axvline(x=y_vector_avg[index_to_plot[i]]['90th'][nwin], ymin=0.0,ymax=1.0, color='magenta')
                plt.xlabel(y_vector[index_to_plot[i]]['ylabel'])
                plt.ylabel('f(x)')
                plt.title('Probablity distibution of '+y_vector[index_to_plot[i]]['ylabel'])
    #        fig.tight_layout()
            pdf_object.savefig()
            pdf_object.close()
        elif plot_for_dependence:
            tau_ind=np.where(np.logical_and(time_vec >= tau_range[0]*1e3, time_vec <= tau_range[1]*1e3))
            y_vector_avg[0]['Data']=y_vector_avg[0]['Data'][tau_ind]
            y_vector_avg[1]['Data']=y_vector_avg[1]['Data'][tau_ind]
            y_vector_avg[4]['Data']=y_vector_avg[4]['Data'][tau_ind]
            y_vector_avg[5]['Data']=y_vector_avg[5]['Data'][tau_ind]
            y_vector_avg[8]['Data']=y_vector_avg[8]['Data'][tau_ind]
            
            pdf_object=PdfPages(wd+'/plots/parameter_dependence_based_on_medians.pdf')
            gs=GridSpec(2,2)
            plt.figure()   
            ax,fig=plt.subplots(figsize=(8.5/2.54,8.5/2.54))
            
            colors = iter(cm.gist_ncar(np.linspace(0, 1, len(y_vector_avg[0]['Data']))))
            
            plt.subplot(gs[0,0])
            plt.plot(y_vector_avg[1]['Data'],
                     y_vector_avg[0]['Data'],
                     lw='0.2')
            for ind_a in range(len(y_vector_avg[0]['Data'])):
                color=copy.deepcopy(next(colors))
                plt.scatter(y_vector_avg[1]['Data'][ind_a], 
                            y_vector_avg[0]['Data'][ind_a], 
                            color=color,
                            s=1)
            plt.xlabel('v pol [km/s]')
            plt.ylabel('v rad [km/s]')
            
            plt.title('vrad vs. vpol')
            
            colors = iter(cm.gist_ncar(np.linspace(0, 1, len(y_vector_avg[0]['Data']))))
            plt.subplot(gs[0,1])
            plt.plot(y_vector_avg[5]['Data'],
                     y_vector_avg[4]['Data'],
                     lw='0.2')
            for ind_a in range(len(y_vector_avg[4]['Data'])):
                color=copy.deepcopy(next(colors))
                plt.scatter(y_vector_avg[5]['Data'][ind_a], 
                            y_vector_avg[4]['Data'][ind_a], 
                            color=color,
                            s=1)
            plt.xlabel('d rad [mm]')
            plt.ylabel('d pol [mm]')
            plt.title('drad vs. dpol')
            
            colors = iter(cm.gist_ncar(np.linspace(0, 1, len(y_vector_avg[0]['Data']))))
            plt.subplot(gs[1,0])
            plt.plot(y_vector_avg[8]['Data'],
                     y_vector_avg[0]['Data'],
                     lw='0.2')
            for ind_a in range(len(y_vector_avg[8]['Data'])):
                color=copy.deepcopy(next(colors))
                plt.scatter(y_vector_avg[8]['Data'][ind_a], 
                            y_vector_avg[0]['Data'][ind_a], 
                            color=color,
                            s=1)
            
            plt.xlabel('r-r_sep [mm]')
            plt.ylabel('vrad [km/s]')
            plt.title('r-r_sep vs. vrad')
            
            colors = iter(cm.gist_ncar(np.linspace(0, 1, len(y_vector_avg[0]['Data']))))
            plt.subplot(gs[1,1])
            plt.plot(y_vector_avg[8]['Data'],
                     y_vector_avg[1]['Data'],
                     lw='0.2')
            for ind_a in range(len(y_vector_avg[8]['Data'])):
                color=copy.deepcopy(next(colors))
                plt.scatter(y_vector_avg[8]['Data'][ind_a], 
                            y_vector_avg[1]['Data'][ind_a], 
                            color=color,
                            s=1)
            plt.xlabel('r-r_sep [mm]')
            plt.ylabel('vpol [km/s]')
            plt.title('r-r_sep vs. vpol')           

            pdf_object.savefig()
            pdf_object.close()
            
    if return_results:
        return time_vec, y_vector, y_vector_avg
            
            
#####
##### 2 B DEVELOPED
#####

def plot_nstx_gpi_watershed_distribution(window_average=500e-6,
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
                                        structure_finding_method='watershed',
                                        interpolation=False,
                                        ):
    
    if elm_time_base not in ['frame similarity', 'radial velocity']:
        raise ValueError('elm_time_base should be either "frame similarity" or "radial velocity"')
    wd=flap.config.get_all_section('Module NSTX_GPI')['Working directory']
    all_results_file=wd+'/processed_data/all_results_file_watershed.pickle'
    database_file=wd+'/db/ELM_findings_mlampert_velocity_good.csv'

    db=pandas.read_csv(database_file, index_col=0)
    elm_index=list(db.index)
    n_elm=len(elm_index)
    
    nwin=int(window_average/sampling_time)
    time_vec=(np.arange(2*nwin)*sampling_time-window_average)*1e3
    
    all_results={'Velocity ccf':np.zeros([2*nwin,n_elm,2]),
                 'Velocity str max':np.zeros([2*nwin,n_elm,2]),
                 'Acceleration ccf':np.zeros([2*nwin,n_elm,2]),
                 'Frame similarity':np.zeros([2*nwin,n_elm]),
                 'Correlation max':np.zeros([2*nwin,n_elm]),   
                 'Size max':np.zeros([2*nwin,n_elm,2]),
                 'Position max':np.zeros([2*nwin,n_elm,2]),  
                 'Separatrix dist max':np.zeros([2*nwin,n_elm]),                            
                 'Centroid max':np.zeros([2*nwin,n_elm,2]),                            
                 'COG max':np.zeros([2*nwin,n_elm,2]),
                 'Area max':np.zeros([2*nwin,n_elm]),
                 'Elongation max':np.zeros([2*nwin,n_elm]),     
                 'Angle max':np.zeros([2*nwin,n_elm]),
                 'Str number':np.zeros([2*nwin,n_elm]),
                 'GPI Dalpha':np.zeros([2*nwin,n_elm]),
                 #watershed related results
                 'Angle ALI max':np.zeros([2*nwin,n_elm]),
                 'Roundness max':np.zeros([2*nwin,n_elm]),
                 'Solidity max':np.zeros([2*nwin,n_elm]),
                 'Convexity max':np.zeros([2*nwin,n_elm]),
                 'Total curvature max':np.zeros([2*nwin,n_elm]),
                 'Total bending energy max':np.zeros([2*nwin,n_elm]),
                 }
        
    hist_range_dict={'Velocity ccf':{'Radial':[-5e3,15e3],
                                     'Poloidal':[-25e3,5e3]},
                     'Velocity str max':{'Radial':[-10e3,20e3],
                                         'Poloidal':[-20e3,10e3]},
                                         
                     'Size max':{'Radial':[0,0.1],
                                 'Poloidal':[0,0.1]},
                                 
                     'Separatrix dist max':[-0.05,0.150],
                     'Elongation max':[-0.5,0.5],    
                     'Str number':[-0.5,10.5],
                     'Angle ALI max':[-2*np.pi,2*np.pi],
                     'Roundness max':[-1,1],
                     'Solidity max':[-1,1],
                     'Convexity max':[-1,1],
                     'Total curvature max':[-10,10],
                     'Total bending energy max':[-100,100],
                     }
                     
    n_hist_str=int(hist_range_dict['Str number'][1]-hist_range_dict['Str number'][0])*4
    
    result_histograms={'Velocity ccf':np.zeros([2*nwin,n_hist,2]),
                       'Velocity str max':np.zeros([2*nwin,n_hist,2]),
                       'Size max':np.zeros([2*nwin,n_hist,2]),
                       'Separatrix dist max':np.zeros([2*nwin,n_hist]),                            
                       'Elongation max':np.zeros([2*nwin,n_hist]),     
                       'Str number':np.zeros([2*nwin,n_hist_str]),
                       }
    
    average_results={'Velocity ccf':np.zeros([2*nwin,2]),
                     'Velocity str max':np.zeros([2*nwin,2]),
                     'Size max':np.zeros([2*nwin,2]),
                     'Separatrix dist max':np.zeros([2*nwin]),                            
                     'Elongation max':np.zeros([2*nwin]),     
                     'Str number':np.zeros([2*nwin]),
                     }
    
    moment_results={'average':copy.deepcopy(average_results),
                    'median':copy.deepcopy(average_results),
                    '10percentile':copy.deepcopy(average_results),
                    '90percentile':copy.deepcopy(average_results),
                    }
    
    result_bins={'Velocity ccf':np.zeros([n_hist+1,2]),
                  'Velocity str max':np.zeros([n_hist+1,2]), 
                  'Size max':np.zeros([n_hist+1,2]),
                  'Separatrix dist max':np.zeros([n_hist+1]),
                  'Elongation max':np.zeros([n_hist+1]),    
                  'Str number':np.zeros([n_hist_str+1]),
                  }
    
    coeff_r=np.asarray([3.7183594,-0.77821046,1402.8097])/1000. #The coordinates are in meters, the coefficients are in mm
    coeff_z=np.asarray([0.18090118,3.0657776,70.544312])/1000.  #The coordinates are in meters, the coefficients are in mm
    coeff_r_new=3./800.
    coeff_z_new=3./800.
    
    if not nocalc:
        for index_elm in range(n_elm):
            #preprocess velocity results, tackle with np.nan and outliers
            shot=int(db.loc[elm_index[index_elm]]['Shot'])
            #define ELM time for all the cases
            elm_time=db.loc[elm_index[index_elm]]['ELM time']/1000.
            if structure_finding_method == 'contour':
                str_add_find_method=''
            elif structure_finding_method == 'watershed' and interpolation == False:
                str_add_find_method='_nointer_watershed'
            else:
                str_add_find_method=''
            if normalized_velocity:
                if normalized_structure:
                    str_add='_ns'
                else:
                    str_add=''
                filename=flap_nstx.tools.filename(exp_id=shot,
                                                     working_directory=wd+'/processed_data',
                                                     time_range=[elm_time-2e-3,elm_time+2e-3],
                                                     comment='ccf_velocity_pfit_o'+str(subtraction_order)+'_fst_0.0'+str_add+'_nv'+str_add_find_method,
                                                     extension='pickle')
            else:
                filename=wd+'/processed_data/'+db.loc[elm_index[index_elm]]['Filename']+'.pickle'
            status=db.loc[elm_index[index_elm]]['OK/NOT OK']
            
            if status != 'NO':
                velocity_results=pickle.load(open(filename, 'rb'))
                velocity_results['GPI Dalpha']=velocity_results['GPI Dalpha'][0]
                
                velocity_results['Separatrix dist avg']=np.zeros(velocity_results['Position avg'].shape[0])
                velocity_results['Separatrix dist max']=np.zeros(velocity_results['Position max'].shape[0])
                
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
                    GPI_z_vert=coeff_z[0]*np.arange(80)/80*64+coeff_z[1]*np.arange(80)+coeff_z[2]
                    R_sep_GPI_interp=np.interp(GPI_z_vert,np.flip(z_sep_GPI),np.flip(R_sep_GPI))
                    z_sep_GPI_interp=GPI_z_vert
                    for key in ['max','avg']:
                        for ind_time in range(len(velocity_results['Position '+key][:,0])):
                            velocity_results['Separatrix dist '+key][ind_time]=np.min(np.sqrt((velocity_results['Position '+key][ind_time,0]-R_sep_GPI_interp)**2 + 
                                                                                              (velocity_results['Position '+key][ind_time,1]-z_sep_GPI_interp)**2))
                            ind_z_min=np.argmin(np.abs(z_sep_GPI-velocity_results['Position '+key][ind_time,1]))
                            if z_sep_GPI[ind_z_min] >= velocity_results['Position '+key][ind_time,1]:
                                ind1=ind_z_min
                                ind2=ind_z_min+1
                            else:
                                ind1=ind_z_min-1
                                ind2=ind_z_min
                                
                            radial_distance=velocity_results['Position '+key][ind_time,0]-((velocity_results['Position '+key][ind_time,1]-z_sep_GPI[ind2])/(z_sep_GPI[ind1]-z_sep_GPI[ind2])*(R_sep_GPI[ind1]-R_sep_GPI[ind2])+R_sep_GPI[ind2])
                            if radial_distance < 0:
                                velocity_results['Separatrix dist '+key][ind_time]*=-1
                except:
                    pass
    
                if gpi_plane_calculation:
                    det=coeff_r[0]*coeff_z[1]-coeff_z[0]*coeff_r[1]
                    
                    for key in ['Velocity ccf','Velocity str max','Velocity str avg','Size max','Size avg']:
                        orig=copy.deepcopy(velocity_results[key])
                        velocity_results[key][:,0]=coeff_r_new/det*(coeff_z[1]*orig[:,0]-coeff_r[1]*orig[:,1])
                        velocity_results[key][:,1]=coeff_z_new/det*(-coeff_z[0]*orig[:,0]+coeff_r[0]*orig[:,1])
                        
                    velocity_results['Elongation max'][:]=-(velocity_results['Size max'][:,0]-velocity_results['Size max'][:,1])/(velocity_results['Size max'][:,0]+velocity_results['Size max'][:,1])
                
                velocity_results['Acceleration ccf']=copy.deepcopy(velocity_results['Velocity ccf'])
                velocity_results['Acceleration ccf'][1:,0]=(velocity_results['Velocity ccf'][1:,0]-velocity_results['Velocity ccf'][0:-1,0])/(2*sampling_time)
                velocity_results['Acceleration ccf'][1:,1]=(velocity_results['Velocity ccf'][1:,1]-velocity_results['Velocity ccf'][0:-1,1])/(2*sampling_time)
                    
                corr_thres_index=np.where(velocity_results['Correlation max'] < correlation_threshold)

                velocity_results['Velocity ccf'][corr_thres_index,:]=[np.nan,np.nan]
                velocity_results['Size max'][corr_thres_index,:]=[np.nan,np.nan]
                velocity_results['Elongation max'][corr_thres_index]=np.nan
                
                velocity_results['GPI Dalpha'][corr_thres_index]=np.nan
                velocity_results['Correlation max'][corr_thres_index]=np.nan
                velocity_results['Separatrix dist max'][corr_thres_index]=np.nan
                velocity_results['Str number'][corr_thres_index]=np.nan
                
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
            if key == 'Str number':
                n_hist_cur=n_hist_str
            else:
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

    y_vector=[{'Title':'Radial velocity ccf',
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
               
               {'Title':'Radial velocity str',
               'Data':result_histograms['Velocity str max'][:,:,0],
               'Bins':(result_bins['Velocity str max'][0:-1,0]+result_bins['Velocity str max'][1:,0])/2e3,
               'Bar width':(result_bins['Velocity str max'][1,0]-result_bins['Velocity str max'][0,0])/1e3,
               'ylabel':'v_rad str [m/s]',
               },
                              
              {'Title':'Poloidal velocity str',
               'Data':result_histograms['Velocity str max'][:,:,1],
               'Bins':(result_bins['Velocity str max'][0:-1,1]+result_bins['Velocity str max'][1:,1])/2e3,
               'Bar width':(result_bins['Velocity str max'][1,1]-result_bins['Velocity str max'][0,1])/1e3,
               'ylabel':'v_pol str [m/s]',
               },
              
              {'Title':'Size max radial',
               'Data':result_histograms['Size max'][:,:,0],
               'Bins':(result_bins['Size max'][0:-1,0]+result_bins['Size max'][1:,0])/2*1e3,
               'Bar width':(result_bins['Size max'][1,0]-result_bins['Size max'][0,0])*1e3,
               'ylabel':'Radial size [mm]',
               },
               
              {'Title':'Size max poloidal',
               'Data':result_histograms['Size max'][:,:,1],
               'Bins':(result_bins['Size max'][0:-1,1]+result_bins['Size max'][1:,1])/2*1e3,
               'Bar width':(result_bins['Size max'][1,1]-result_bins['Size max'][0,1])*1e3,
               'ylabel':'Poloidal size [mm]',
               },               
              
              {'Title':'Elongation max',
               'Data':result_histograms['Elongation max'],
               'Bins':(result_bins['Elongation max'][0:-1]+result_bins['Elongation max'][1:])/2,
               'Bar width':(result_bins['Elongation max'][1]-result_bins['Elongation max'][0]),
               'ylabel':'Elongation [a.u.]',
               },
               
               {'Title':'Str number',
               'Data':result_histograms['Str number'],
               'Bins':(result_bins['Str number'][0:-1]+result_bins['Str number'][1:])/2,
               'Bar width':1,
               'ylabel':'Str number',
               },
                
               {'Title':'Distance',
                'Data':result_histograms['Separatrix dist max'],
                'Bins':(result_bins['Separatrix dist max'][0:-1]+result_bins['Separatrix dist max'][1:])/2*1e3,
                'Bar width':(result_bins['Separatrix dist max'][1]-result_bins['Separatrix dist max'][0])*1e3,
                'ylabel':'Distance [mm]',
               },
              ]

    y_vector_avg=[{'Title':'Radial velocity ccf',
                   'Data':moment_results['median']['Velocity ccf'][:,0]/1e3,
                   '10th':moment_results['10percentile']['Velocity ccf'][:,0]/1e3,
                   '90th':moment_results['90percentile']['Velocity ccf'][:,0]/1e3,
                   'ylabel':'v_rad [m/s]',
                   },
        
                  {'Title':'Poloidal velocity ccf',
                   'Data':moment_results['median']['Velocity ccf'][:,1]/1e3,
                   '10th':moment_results['10percentile']['Velocity ccf'][:,1]/1e3,
                   '90th':moment_results['90percentile']['Velocity ccf'][:,1]/1e3,
                   'ylabel':'v_pol [m/s]',
                   },
                   
                   {'Title':'Radial velocity str',
                    'Data':moment_results['median']['Velocity str max'][:,0]/1e3,
                    '10th':moment_results['10percentile']['Velocity str max'][:,0]/1e3,
                    '90th':moment_results['90percentile']['Velocity str max'][:,0]/1e3,
                    'ylabel':'v_rad str [m/s]',
                   },
                                  
                  {'Title':'Poloidal velocity str',
                   'Data':moment_results['median']['Velocity str max'][:,1]/1e3,
                   '10th':moment_results['10percentile']['Velocity str max'][:,1]/1e3,
                   '90th':moment_results['90percentile']['Velocity str max'][:,1]/1e3,
                   'ylabel':'v_pol str [m/s]',
                   },
                  
                  {'Title':'Size max radial',
                   'Data':moment_results['median']['Size max'][:,0]*1e3,
                   '10th':moment_results['10percentile']['Size max'][:,0]*1e3,
                   '90th':moment_results['90percentile']['Size max'][:,0]*1e3,
                   'ylabel':'Radial size [mm]',
                   },
                   
                  {'Title':'Size max poloidal',
                   'Data':moment_results['median']['Size max'][:,1]*1e3,
                   '10th':moment_results['10percentile']['Size max'][:,1]*1e3,
                   '90th':moment_results['90percentile']['Size max'][:,1]*1e3,
                   'ylabel':'Poloidal size [mm]',
                   },               
                  
                  {'Title':'Elongation max',
                   'Data':moment_results['median']['Elongation max'],
                   '10th':moment_results['10percentile']['Elongation max'],
                   '90th':moment_results['90percentile']['Elongation max'],
                   'ylabel':'Elongation [a.u.]',
                   },
                  {'Title':'Str number',
                   'Data':moment_results['median']['Str number'],
                   '10th':moment_results['10percentile']['Str number'],
                   '90th':moment_results['90percentile']['Str number'],
                   'ylabel':'Str number',
                   },
                  {'Title':'Separatrix dist.',
                   'Data':moment_results['median']['Separatrix dist max']*1e3,
                   '10th':moment_results['10percentile']['Separatrix dist max']*1e3,
                   '90th':moment_results['90percentile']['Separatrix dist max']*1e3,
                   'ylabel':'Dist. [mm]'} 
                  ]
    if general_plot:
        pdf_object=PdfPages(wd+'/plots/all_velocity_results_histograms.pdf')
        
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
        
    else:
        if plot_for_structure:
            pdf_object=PdfPages(wd+'/plots/all_velocity_results_histograms_publication.pdf')
            gs=GridSpec(5,2)
            plt.figure()   
            ax,fig=plt.subplots(figsize=(8.5/2.54,8))
            index_to_plot=[4,5,6,7,8]
            for i in range(5):
                plt.subplot(gs[i,0])
                plt.contourf(time_vec,
                            y_vector[index_to_plot[i]]['Bins'],
                            y_vector[index_to_plot[i]]['Data'].transpose(),
                            levels=n_hist,
                            )
                plt.plot(time_vec,
                         y_vector_avg[index_to_plot[i]]['Data'],
                         color='red')
                plt.plot(time_vec,
                         y_vector_avg[index_to_plot[i]]['10th'],
                         color='magenta')
                plt.plot(time_vec,
                         y_vector_avg[index_to_plot[i]]['90th'],
                         color='magenta')
                plt.title('Relative frequency of '+y_vector[index_to_plot[i]]['ylabel'])
                plt.xlabel('Time [ms]')
                plt.ylabel(y_vector[index_to_plot[i]]['ylabel'])
                plt.colorbar()
                
                plt.subplot(gs[i,1])
                plt.bar(y_vector[index_to_plot[i]]['Bins'],
                       y_vector[index_to_plot[i]]['Data'][nwin,:], 
                       width=y_vector[index_to_plot[i]]['Bar width'])
                
                plt.axvline(x=y_vector_avg[index_to_plot[i]]['Data'][nwin], ymin=0.0,ymax=1.0, color='red')
                plt.axvline(x=y_vector_avg[index_to_plot[i]]['10th'][nwin], ymin=0.0,ymax=1.0, color='magenta')
                plt.axvline(x=y_vector_avg[index_to_plot[i]]['90th'][nwin], ymin=0.0,ymax=1.0, color='magenta')
                
                plt.xlabel(y_vector[index_to_plot[i]]['ylabel'])
                plt.ylabel('f(x)')
                plt.title('Probablity distibution of '+y_vector[index_to_plot[i]]['ylabel'])
    #        fig.tight_layout()
            pdf_object.savefig()
            pdf_object.close()
        elif plot_for_velocity:
            pdf_object=PdfPages(wd+'/plots/all_velocity_results_histograms_publication_velocity.pdf')
            gs=GridSpec(2,2)
            plt.figure()   
            ax,fig=plt.subplots(figsize=(17/2.54,12/2.54))
            index_to_plot=[0,1]
            print(time_vec[nwin])
            for i in range(2):
                plt.subplot(gs[0,i])
                plt.contourf(time_vec,
                            y_vector[index_to_plot[i]]['Bins'],
                            y_vector[index_to_plot[i]]['Data'].transpose(),
                            levels=n_hist,
                            )
                plt.plot(time_vec,
                         y_vector_avg[index_to_plot[i]]['Data'],
                         color='red')
                plt.plot(time_vec,
                         y_vector_avg[index_to_plot[i]]['10th'],
                         color='magenta')
                plt.plot(time_vec,
                         y_vector_avg[index_to_plot[i]]['90th'],
                         color='magenta')
                
                plt.title('Relative frequency of '+y_vector[index_to_plot[i]]['ylabel'])
                plt.xlabel('Time [ms]')
                plt.ylabel(y_vector[index_to_plot[i]]['ylabel'])
                plt.colorbar()
                
                plt.subplot(gs[1,i])
                plt.bar(y_vector[index_to_plot[i]]['Bins'],
                       y_vector[index_to_plot[i]]['Data'][nwin,:], 
                       width=y_vector[index_to_plot[i]]['Bar width'])
                plt.axvline(x=y_vector_avg[index_to_plot[i]]['Data'][nwin], ymin=0.0,ymax=1.0, color='red')
                plt.axvline(x=y_vector_avg[index_to_plot[i]]['10th'][nwin], ymin=0.0,ymax=1.0, color='magenta')
                plt.axvline(x=y_vector_avg[index_to_plot[i]]['90th'][nwin], ymin=0.0,ymax=1.0, color='magenta')
                plt.xlabel(y_vector[index_to_plot[i]]['ylabel'])
                plt.ylabel('f(x)')
                plt.title('Probablity distibution of '+y_vector[index_to_plot[i]]['ylabel'])
    #        fig.tight_layout()
            pdf_object.savefig()
            pdf_object.close()
        elif plot_for_dependence:
            tau_ind=np.where(np.logical_and(time_vec >= tau_range[0]*1e3, time_vec <= tau_range[1]*1e3))
            y_vector_avg[0]['Data']=y_vector_avg[0]['Data'][tau_ind]
            y_vector_avg[1]['Data']=y_vector_avg[1]['Data'][tau_ind]
            y_vector_avg[4]['Data']=y_vector_avg[4]['Data'][tau_ind]
            y_vector_avg[5]['Data']=y_vector_avg[5]['Data'][tau_ind]
            y_vector_avg[8]['Data']=y_vector_avg[8]['Data'][tau_ind]
            
            pdf_object=PdfPages(wd+'/plots/parameter_dependence_based_on_medians.pdf')
            gs=GridSpec(2,2)
            plt.figure()   
            ax,fig=plt.subplots(figsize=(8.5/2.54,8.5/2.54))
            
            colors = iter(cm.gist_ncar(np.linspace(0, 1, len(y_vector_avg[0]['Data']))))
            
            plt.subplot(gs[0,0])
            plt.plot(y_vector_avg[1]['Data'],
                     y_vector_avg[0]['Data'],
                     lw='0.2')
            for ind_a in range(len(y_vector_avg[0]['Data'])):
                color=copy.deepcopy(next(colors))
                plt.scatter(y_vector_avg[1]['Data'][ind_a], 
                            y_vector_avg[0]['Data'][ind_a], 
                            color=color,
                            s=1)
            plt.xlabel('v pol [km/s]')
            plt.ylabel('v rad [km/s]')
            
            plt.title('vrad vs. vpol')
            
            colors = iter(cm.gist_ncar(np.linspace(0, 1, len(y_vector_avg[0]['Data']))))
            plt.subplot(gs[0,1])
            plt.plot(y_vector_avg[5]['Data'],
                     y_vector_avg[4]['Data'],
                     lw='0.2')
            for ind_a in range(len(y_vector_avg[4]['Data'])):
                color=copy.deepcopy(next(colors))
                plt.scatter(y_vector_avg[5]['Data'][ind_a], 
                            y_vector_avg[4]['Data'][ind_a], 
                            color=color,
                            s=1)
            plt.xlabel('d rad [mm]')
            plt.ylabel('d pol [mm]')
            plt.title('drad vs. dpol')
            
            colors = iter(cm.gist_ncar(np.linspace(0, 1, len(y_vector_avg[0]['Data']))))
            plt.subplot(gs[1,0])
            plt.plot(y_vector_avg[8]['Data'],
                     y_vector_avg[0]['Data'],
                     lw='0.2')
            for ind_a in range(len(y_vector_avg[8]['Data'])):
                color=copy.deepcopy(next(colors))
                plt.scatter(y_vector_avg[8]['Data'][ind_a], 
                            y_vector_avg[0]['Data'][ind_a], 
                            color=color,
                            s=1)
            
            plt.xlabel('r-r_sep [mm]')
            plt.ylabel('vrad [km/s]')
            plt.title('r-r_sep vs. vrad')
            
            colors = iter(cm.gist_ncar(np.linspace(0, 1, len(y_vector_avg[0]['Data']))))
            plt.subplot(gs[1,1])
            plt.plot(y_vector_avg[8]['Data'],
                     y_vector_avg[1]['Data'],
                     lw='0.2')
            for ind_a in range(len(y_vector_avg[8]['Data'])):
                color=copy.deepcopy(next(colors))
                plt.scatter(y_vector_avg[8]['Data'][ind_a], 
                            y_vector_avg[1]['Data'][ind_a], 
                            color=color,
                            s=1)
            plt.xlabel('r-r_sep [mm]')
            plt.ylabel('vpol [km/s]')
            plt.title('r-r_sep vs. vpol')           

            pdf_object.savefig()
            pdf_object.close()
            
    