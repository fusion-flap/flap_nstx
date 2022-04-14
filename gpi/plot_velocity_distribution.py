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
from matplotlib.ticker import MaxNLocator

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

                                        return_results=False,
                                        return_error=False,
                                        normalized_velocity=True,
                                        normalized_structure=True,
                                        
                                        subtraction_order=4,
                                        correlation_threshold=0.6,
                                        gpi_plane_calculation=True,
                                        
                                        elm_time_base='frame similarity',
                                        n_hist=50,
                                        min_max_range=False,
                                        
                                        nocalc=False,
                                        
                                        plot=True,
                                        plot_max_only=False,
                                        plot_for_publication=False,
                                        general_plot=True,
                                        plot_for_velocity=False,
                                        plot_for_structure=False,
                                        plot_for_dependence=False,
                                        plot_scatter=True,
                                        plot_variance=True,
                                        plot_error=False,
                                        opacity=0.2,
                                        
                                        pdf=False,
                                        pdf_filename=None,
                                        
                                        structure_finding_method='contour',
                                        interpolation=False,
                                        figure_size=8.5,
                                        
                                        ):
    #Plot settings for publications

    if plot_for_publication:
        figsize=(figure_size/2.54,figure_size/np.sqrt(2)/2.54)
        plt.rc('font', family='serif', serif='Helvetica')
        if figure_size >8.5:
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
    
    
    y_vector={'Velocity ccf radial':{'data':result_histograms['Velocity ccf'][:,:,0],
                                     'bins':(result_bins['Velocity ccf'][0:-1,0]+result_bins['Velocity ccf'][1:,0])/2e3,
                                     'bar width':(result_bins['Velocity ccf'][1,0]-result_bins['Velocity ccf'][0,0])/1e3,
                                     'median':moment_results['median']['Velocity ccf'][:,0]/1e3,
                                     '10th':moment_results['10percentile']['Velocity ccf'][:,0]/1e3,
                                     '90th':moment_results['90percentile']['Velocity ccf'][:,0]/1e3,
                                     'ylabel':'$v_{rad}$',
                                     'unit':'km/s',
                                     },   
    
              'Velocity ccf poloidal':{'data':result_histograms['Velocity ccf'][:,:,1],
                                       'bins':(result_bins['Velocity ccf'][0:-1,1]+result_bins['Velocity ccf'][1:,1])/2e3,
                                       'bar width':(result_bins['Velocity ccf'][1,1]-result_bins['Velocity ccf'][0,1])/1e3,
                                       'median':moment_results['median']['Velocity ccf'][:,1]/1e3,
                                       '10th':moment_results['10percentile']['Velocity ccf'][:,1]/1e3,
                                       '90th':moment_results['90percentile']['Velocity ccf'][:,1]/1e3,
                                       'ylabel':'$v_{pol}$',
                                       'unit':'km/s',
                                       },
               
               'Velocity str radial':{'data':result_histograms['Velocity str max'][:,:,0],
                                      'bins':(result_bins['Velocity str max'][0:-1,0]+result_bins['Velocity str max'][1:,0])/2e3,
                                      'bar width':(result_bins['Velocity str max'][1,0]-result_bins['Velocity str max'][0,0])/1e3,
                                      'median':moment_results['median']['Velocity str max'][:,0]/1e3,
                                      '10th':moment_results['10percentile']['Velocity str max'][:,0]/1e3,
                                      '90th':moment_results['90percentile']['Velocity str max'][:,0]/1e3,
                                      'ylabel':'$v_{rad,str}',
                                      'unit':'km/s'
                                      },
                              
              'Velocity str poloidal':{'data':result_histograms['Velocity str max'][:,:,1],
                                       'bins':(result_bins['Velocity str max'][0:-1,1]+result_bins['Velocity str max'][1:,1])/2e3,
                                       'bar width':(result_bins['Velocity str max'][1,1]-result_bins['Velocity str max'][0,1])/1e3,
                                       'median':moment_results['median']['Velocity str max'][:,1]/1e3,
                                       '10th':moment_results['10percentile']['Velocity str max'][:,1]/1e3,
                                       '90th':moment_results['90percentile']['Velocity str max'][:,1]/1e3,
                                       'ylabel':'$v_{pol,str}$',
                                       'unit':'km/s',
                                       },
              
              'Size max radial':{'data':result_histograms['Size max'][:,:,0],
                                 'bins':(result_bins['Size max'][0:-1,0]+result_bins['Size max'][1:,0])/2*1e3,
                                 'bar width':(result_bins['Size max'][1,0]-result_bins['Size max'][0,0])*1e3,
                                 'median':moment_results['median']['Size max'][:,0]*1e3,
                                 '10th':moment_results['10percentile']['Size max'][:,0]*1e3,
                                 '90th':moment_results['90percentile']['Size max'][:,0]*1e3,
                                 'ylabel':'$d_{rad}$',
                                 'unit':'mm',
                                 },
               
              'Size max poloidal':{'data':result_histograms['Size max'][:,:,1],
                                   'bins':(result_bins['Size max'][0:-1,1]+result_bins['Size max'][1:,1])/2*1e3,
                                   'bar width':(result_bins['Size max'][1,1]-result_bins['Size max'][0,1])*1e3,
                                   'median':moment_results['median']['Size max'][:,1]*1e3,
                                   '10th':moment_results['10percentile']['Size max'][:,1]*1e3,
                                   '90th':moment_results['90percentile']['Size max'][:,1]*1e3,
                                   'ylabel':'$d_{pol}$',
                                   'unit':'mm',
                                   },               
              
              'Elongation max':{'data':result_histograms['Elongation max'],
                                'bins':(result_bins['Elongation max'][0:-1]+result_bins['Elongation max'][1:])/2,
                                'bar width':(result_bins['Elongation max'][1]-result_bins['Elongation max'][0]),
                                'median':moment_results['median']['Elongation max'],
                                '10th':moment_results['10percentile']['Elongation max'],
                                '90th':moment_results['90percentile']['Elongation max'],
                                'ylabel':'Elongation',
                                'unit':'',
                                },
               
              'Str number':{'data':result_histograms['Str number'],
                            'bins':(result_bins['Str number'][0:-1]+result_bins['Str number'][1:])/2,
                            'bar width':1,
                            'median':moment_results['median']['Str number'],
                            '10th':moment_results['10percentile']['Str number'],
                            '90th':moment_results['90percentile']['Str number'],
                            'ylabel':'N',
                            'unit':'',
                            },
                
               'Distance':{'data':result_histograms['Separatrix dist max'],
                           'bins':(result_bins['Separatrix dist max'][0:-1]+result_bins['Separatrix dist max'][1:])/2*1e3,
                           'bar width':(result_bins['Separatrix dist max'][1]-result_bins['Separatrix dist max'][0])*1e3,
                           'median':moment_results['median']['Separatrix dist max']*1e3,
                           '10th':moment_results['10percentile']['Separatrix dist max']*1e3,
                           '90th':moment_results['90percentile']['Separatrix dist max']*1e3,
                           'ylabel':'$r-r_{sep}$',
                           'unit':'mm',
                           },
              }
    
    if general_plot:
        if pdf:
            if pdf_filename is None:
                pdf_filename=wd+'/plots/all_velocity_results_histograms.pdf'
            pdf_object=PdfPages(pdf_filename)
            
        if not plot:
            import matplotlib
            matplotlib.use('agg')
            
        def fmt(x, pos):
            a = '{:3.2f}'.format(x)
            return a
            
        for key in y_vector.keys():
            plt.figure()
            fig,ax=plt.subplots(figsize=figsize)
            im=ax.contourf(time_vec*1e3,
                           y_vector[key]['bins'],
                           y_vector[key]['data'].transpose(),
                           levels=n_hist,
                           )
            ax.plot(time_vec*1e3,
                     y_vector[key]['median'],
                     color='red')
            ax.plot(time_vec*1e3,
                     y_vector[key]['10th'],
                     color='white',
                     lw=0.2)
            ax.plot(time_vec*1e3,
                     y_vector[key]['90th'],
                     color='white',
                     lw=0.2)
            
            ax.set_title('Relative frequency of '+y_vector[key]['ylabel'])
            ax.set_xlabel('$t-t_{ELM}$ [$\mu$s]')
            ax.set_ylabel(y_vector[key]['ylabel']+' '+y_vector[key]['unit'])
            ax.xaxis.set_major_locator(MaxNLocator(5)) 
            ax.yaxis.set_major_locator(MaxNLocator(5)) 
            ax.set_xticks(ticks=[-time_vec[0]*1e3,
                                 -time_vec[0]*0.5e3,
                                 0,
                                 time_vec[0]*0.5e3,
                                 time_vec[0]*1e3])
            
            import matplotlib.ticker as ticker
            cbar=fig.colorbar(im, format=ticker.FuncFormatter(fmt))
            cbar.ax.tick_params(labelsize=6)
            
            plt.tight_layout(pad=0.1)
            if pdf:
                pdf_object.savefig()
            
        for key in y_vector.keys():
            plt.figure()
            fig,ax=plt.subplots(figsize=figsize)
    #        plt.plot(y_vector[i]['Bins'], np.mean(y_vector[i]['Data'][nwin-2:nwin+3,:], axis=0))
            ax.bar(y_vector[key]['bins'],y_vector[key]['data'][nwin,:], width=y_vector[key]['bar width'])
            ax.set_xlabel(y_vector[key]['ylabel']+' '+y_vector[key]['unit'])
            ax.set_ylabel('f(x)')
            ax.set_title('PDF of '+y_vector[key]['ylabel'])
            ax.xaxis.set_major_locator(MaxNLocator(5)) 
            ax.yaxis.set_major_locator(MaxNLocator(5)) 
            ax.axvline(x=y_vector[key]['median'][nwin], ymin=0.0,ymax=1.0, color='red')
            ax.axvline(x=y_vector[key]['10th'][nwin], ymin=0.0,ymax=1.0, color='magenta')
            ax.axvline(x=y_vector[key]['90th'][nwin], ymin=0.0,ymax=1.0, color='magenta')
            plt.tight_layout(pad=0.1)
            if pdf:
                pdf_object.savefig()
        if not plot:
            matplotlib.use('qt5agg')
        if pdf:
            pdf_object.close()
        
    else:
        if plot_for_structure:
            pdf_object=PdfPages(wd+'/plots/all_velocity_results_histograms_publication.pdf')
            gs=GridSpec(5,2)
            plt.figure()   
            ax,fig=plt.subplots(figsize=(8.5/2.54,8))
            keys_to_plot=['Size max radial',
                          'Size max poloidal',
                          'Elongation',
                          'Str number',
                          'Distance']
            
            for i in range(len(keys_to_plot)):
                plt.subplot(gs[i,0])
                plt.contourf(time_vec,
                            y_vector[keys_to_plot[i]]['bins'],
                            y_vector[keys_to_plot[i]]['data'].transpose(),
                            levels=n_hist,
                            )
                plt.plot(time_vec,
                         y_vector[keys_to_plot[i]]['median'],
                         color='red')
                plt.plot(time_vec,
                         y_vector[keys_to_plot[i]]['10th'],
                         color='magenta')
                plt.plot(time_vec,
                         y_vector[keys_to_plot[i]]['90th'],
                         color='magenta')
                plt.title('Relative frequency of '+y_vector[keys_to_plot[i]]['ylabel'])
                plt.xlabel('Time [ms]')
                plt.ylabel(y_vector[keys_to_plot[i]]['ylabel'])
                plt.colorbar()
                
                plt.subplot(gs[i,1])
                plt.bar(y_vector[keys_to_plot[i]]['bins'],
                       y_vector[keys_to_plot[i]]['data'][nwin,:], 
                       width=y_vector[keys_to_plot[i]]['bar width'])
                
                plt.axvline(x=y_vector[keys_to_plot[i]]['data'][nwin], ymin=0.0,ymax=1.0, color='red')
                plt.axvline(x=y_vector[keys_to_plot[i]]['10th'][nwin], ymin=0.0,ymax=1.0, color='magenta')
                plt.axvline(x=y_vector[keys_to_plot[i]]['90th'][nwin], ymin=0.0,ymax=1.0, color='magenta')
                
                plt.xlabel(y_vector[keys_to_plot[i]]['ylabel'])
                plt.ylabel('f(x)')
                plt.title('Probablity distibution of '+y_vector[keys_to_plot[i]]['ylabel'])
    #        fig.tight_layout()
            pdf_object.savefig()
            pdf_object.close()
            
        elif plot_for_velocity:
            pdf_object=PdfPages(wd+'/plots/all_velocity_results_histograms_publication_velocity.pdf')
            gs=GridSpec(2,2)
            plt.figure()   
            ax,fig=plt.subplots(figsize=(17/2.54,12/2.54))
            keys_to_plot=['Velocity ccf radial','Velocity ccf poloidal']

            for i in range(2):
                plt.subplot(gs[0,i])
                plt.contourf(time_vec,
                            y_vector[keys_to_plot[i]]['bins'],
                            y_vector[keys_to_plot[i]]['data'].transpose(),
                            levels=n_hist,
                            )
                plt.plot(time_vec,
                         y_vector[keys_to_plot[i]]['median'],
                         color='red')
                plt.plot(time_vec,
                         y_vector[keys_to_plot[i]]['10th'],
                         color='magenta')
                plt.plot(time_vec,
                         y_vector[keys_to_plot[i]]['90th'],
                         color='magenta')
                
                plt.title('Relative frequency of '+y_vector[keys_to_plot[i]]['ylabel'])
                plt.xlabel('Time [ms]')
                plt.ylabel(y_vector[keys_to_plot[i]]['ylabel'])
                plt.colorbar()
                
                plt.subplot(gs[1,i])
                plt.bar(y_vector[keys_to_plot[i]]['bins'],
                       y_vector[keys_to_plot[i]]['data'][nwin,:], 
                       width=y_vector[keys_to_plot[i]]['bar width'])
                plt.axvline(x=y_vector[keys_to_plot[i]]['data'][nwin], ymin=0.0,ymax=1.0, color='red')
                plt.axvline(x=y_vector[keys_to_plot[i]]['10th'][nwin], ymin=0.0,ymax=1.0, color='magenta')
                plt.axvline(x=y_vector[keys_to_plot[i]]['90th'][nwin], ymin=0.0,ymax=1.0, color='magenta')
                plt.xlabel(y_vector[keys_to_plot[i]]['ylabel'])
                plt.ylabel('f(x)')
                plt.title('Probablity distibution of '+y_vector[keys_to_plot[i]]['ylabel'])
    #        fig.tight_layout()
            pdf_object.savefig()
            pdf_object.close()
        elif plot_for_dependence:
            tau_ind=np.where(np.logical_and(time_vec >= tau_range[0]*1e3, time_vec <= tau_range[1]*1e3))
            
            pdf_object=PdfPages(wd+'/plots/parameter_dependence_based_on_medians.pdf')
            
            fig,axs=plt.subplots(2,2,figsize=(8.5/2.54,8.5/2.54))
            
            
            ax=axs[0,0]
            colors = iter(cm.gist_ncar(np.linspace(0, 1, len(y_vector['Velocity ccf radial']['data'][tau_ind]))))
            ax.set_plot(y_vector['Velocity ccf poloidal']['median'][tau_ind],
                     y_vector['Velocity ccf radial']['median'][tau_ind],
                     lw='0.2')
            for ind_a in range(len(y_vector['Velocity ccf radial']['median'][tau_ind])):
                color=copy.deepcopy(next(colors))
                ax.set_scatter(y_vector['Velocity ccf poloidal']['median'][tau_ind][ind_a], 
                            y_vector['Velocity ccf radial']['median'][tau_ind][ind_a], 
                            color=color,
                            s=1)
            ax.set_xlabel('$v_pol$ [km/s]')
            ax.set_ylabel('$v_rad$ [km/s]')
            
            ax.set_title('$v_{rad}$ vs. $v_{pol}$')
            
            
            ax=axs[0,1]
            colors = iter(cm.gist_ncar(np.linspace(0, 1, len(y_vector['Velocity ccf radial']['data'][tau_ind]))))
            ax.plot(y_vector['Size max radial']['median'][tau_ind],
                     y_vector['Size max poloidal']['median'][tau_ind],
                     lw='0.2')
            for ind_a in range(len(y_vector['Size max poloidal']['median'][tau_ind])):
                color=copy.deepcopy(next(colors))
                ax.scatter(y_vector['Size max radial']['median'][tau_ind][ind_a],
                            y_vector['Size max poloidal']['median'][tau_ind][ind_a],
                            color=color,
                            s=1)
            ax.set_xlabel('$d_{rad}$ [mm]')
            ax.set_ylabel('$d_{pol}$ [mm]')
            ax.set_title('$d_{rad}$ vs. $d_{pol}$')

            ax=axs[1,0]
            colors = iter(cm.gist_ncar(np.linspace(0, 1, len(y_vector['Velocity ccf radial']['data'][tau_ind]))))
            ax.plot(y_vector['Distance']['median'][tau_ind],
                     y_vector['Velocity ccf radial']['median'][tau_ind],
                     lw='0.2')
            for ind_a in range(len(y_vector['Velocity ccf radial']['median'][tau_ind])):
                color=copy.deepcopy(next(colors))
                ax.scatter(y_vector['Distance']['median'][tau_ind][ind_a], 
                            y_vector['Velocity ccf radial']['median'][tau_ind][ind_a], 
                            color=color,
                            s=1)
            
            ax.set_xlabel('$r-r_{sep}$ [mm]')
            ax.set_ylabel('$v_{rad}$ [km/s]')
            ax.set_title('$r-r_{sep}$ vs. $v_{rad}$')
            
            ax=axs[1,1]
            colors = iter(cm.gist_ncar(np.linspace(0, 1, len(y_vector[0]['Data']))))
            ax.subplot(gs[1,1])
            ax.plot(y_vector['Distance']['median'][tau_ind],
                     y_vector['Velocity ccf poloidal']['median'][tau_ind],
                     lw='0.2')
            for ind_a in range(len(y_vector['Distance']['median'][tau_ind])):
                color=copy.deepcopy(next(colors))
                ax.scatter(y_vector['Distance']['median'][tau_ind][ind_a], 
                            y_vector['Velocity ccf poloidal']['median'][tau_ind][ind_a], 
                            color=color,
                            s=1)
            ax.set_xlabel('$r-r_{sep}$ [mm]')
            ax.set_ylabel('$v_{pol}$ [km/s]')
            ax.set_title('$r-r_{sep}$ vs. $v_{pol}$')
            
            plt.tight_layout(pad=0.1)
            pdf_object.savefig()
            pdf_object.close()

    if return_results:
        return time_vec, y_vector