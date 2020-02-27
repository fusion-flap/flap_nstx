#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 16:43:09 2020

@author: mlampert
"""
import os
import pandas
import time
import copy

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.lines as mlines
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
            elm_time_ind=np.argmin(velocity_results['Frame similarity'])
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
        wd=flap.config.get_all_section('Module NSTX_GPI')['Working directory']
        filename=wd+'/'+db.loc[elm_index[index_elm]]['Filename']+'.pickle'
        status=db.loc[elm_index[index_elm]]['OK/NOT OK']
        
        if status != 'NO':
            velocity_results=pickle.load(open(filename, 'rb'))
            elm_time_ind=np.argmin(velocity_results['Frame similarity'])
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
    tau_range=[min(average_results['Tau']),max(average_results['Tau'])]
    
    if plot or pdf:
        #This is a bit unusual here, but necessary due to the structure size calculation based on the contours which are not plot
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
            pdf_filename=wd+'/NSTX_GPI_ALL_ELM_AVERAGE_RESULTS'
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
            ax.errorbar(average_results['Tau'][plot_index], 
                        average_results['Velocity ccf'][plot_index,0], 
                        yerr=variance_results['Velocity ccf'][plot_index,0]
                        )
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
        ax.set_xlim(tau_range)
        ax.set_title('Radial velocity of '+'the average results.')
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
            ax.errorbar(average_results['Tau'][plot_index], 
                        average_results['Velocity ccf'][plot_index,1], 
                        yerr=variance_results['Velocity ccf'][plot_index,1]
                        )
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
            ax.errorbar(average_results['Tau'][plot_index], 
                        average_results['Size max'][plot_index,0], 
                        yerr=variance_results['Size max'][plot_index,0],
                        color='red'
                        )
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
            ax.errorbar(average_results['Tau'][plot_index], 
                        average_results['Size max'][plot_index,1], 
                        yerr=variance_results['Size max'][plot_index,1],
                        color='red'
                        )
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
        ax.set_xlabel('Tau [ms]')
        ax.set_ylabel('Str number')
        ax.set_title('Number of structures vs. time of '+'the average results.')
        ax.set_xlim(tau_range)
        fig.tight_layout()
        if pdf:
            pdf_pages.savefig()
        
                                
        if pdf:
           pdf_pages.close()
            
    if return_results:
        return average_results