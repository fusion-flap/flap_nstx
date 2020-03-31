#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 16:37:41 2020

@author: mlampert
"""
#Core modules
import os
import copy

import flap
import flap_nstx
flap_nstx.register()
import flap_mdsplus
flap_mdsplus.register('NSTX_MDSPlus')
from flap_nstx.analysis import nstx_gpi_one_frame_structure_finder

thisdir = os.path.dirname(os.path.realpath(__file__))
fn = os.path.join(thisdir,"flap_nstx.cfg")
flap.config.read(file_name=fn)

#Scientific modules
import matplotlib.style as pltstyle
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import numpy as np
import scipy
import pickle

#Plot settings for publications
publication=False

if publication:

    plt.rcParams['lines.linewidth'] = 4
    plt.rcParams['axes.linewidth'] = 4
    plt.rcParams['axes.labelsize'] = 28  # 28 for paper 35 for others
    plt.rcParams['axes.titlesize'] = 28
    plt.rcParams['xtick.labelsize'] = 30
    plt.rcParams['xtick.major.size'] = 10
    plt.rcParams['xtick.major.width'] = 4
    plt.rcParams['xtick.major.size'] = 6
    plt.rcParams['xtick.minor.width'] = 2
    plt.rcParams['xtick.minor.size'] = 4
    plt.rcParams['ytick.labelsize'] = 30
    plt.rcParams['ytick.major.width'] = 4
    plt.rcParams['ytick.major.size'] = 6
    plt.rcParams['ytick.minor.width'] = 2
    plt.rcParams['ytick.minor.size'] = 4
    plt.rcParams['legend.fontsize'] = 28

else:
    pltstyle.use('default')
    
def nstx_gpi_velocity_analysis_spatio_temporal_displacement(exp_id=None,                          #Shot number
                                                            time_range=None,                      #The time range for the calculation
                                                            data_object=None,                     #Input data object if available from outside (e.g. generated sythetic signal)
                                                            x_range=[0,63],                       #X range for the calculation
                                                            y_range=[0,79],                       #Y range for the calculation
                                                            x_search=10,
                                                            y_search=10,
                                                            
                                                            #Normalizer inputs
                                                            normalize='roundtrip',                #The normalizer function is split into two parts: before ELM and after ELM part. The ELM period is taken as continously  morphing average.
                                                            normalize_f_kernel='Elliptic',        #The kernel for filtering the gas cloud
                                                            normalize_f_high=1e3,                 #High pass frequency for the normalizer data
                                                            fbin=10,
                                                        
                                                                                                  #Plot options:
                                                            plot=True,                            #Plot the results
                                                            pdf=False,                            #Print the results into a PDF
                                                            plot_gas=True,                        #Plot the gas cloud parameters on top of the other results from the structure size calculation
                                                            plot_error=False,                     #Plot the errorbars of the velocity calculation based on the line fitting and its RMS error
                                                            error_window=4.,                      #Plot the average signal with the error bars calculated from the normalized variance.
                                                          
                                                            #File input/output options
                                                            filename=None,                        #Filename for restoring data
                                                            save_results=True,                    #Save the results into a .pickle file to filename+.pickle
                                                            nocalc=True,                          #Restore the results from the .pickle file from filename+.pickle
                                                            
                                                            #Output options:
                                                            return_results=False,                 #Return the results if set.
                                                            cache_data=True,                      #Cache the data or try to open is from cache
                                                            
                                                            #Test options
                                                            test=False,                           #Test the results
                                                            test_structures=False,                #Test the structure size calculation
                                                            test_gas_cloud=False,                 #Test the gas cloud property determination
                                                            test_histogram=False,                 #Plot the poloidal velocity histogram
                                                            ):
        
    #Constants for the calculation
    #Using the spatial calibration to find the actual velocities.
    coeff_r=np.asarray([3.7183594,-0.77821046,1402.8097])/1000. #The coordinates are in meters, the coefficients are in mm
    coeff_z=np.asarray([0.18090118,3.0657776,70.544312])/1000.  #The coordinates are in meters, the coefficients are in mm
    
    #Input error handling
    if exp_id is None and data_object is None:
        raise ValueError('Either exp_id or data_object needs to be set for the calculation.')
    if data_object is None:
        if time_range is None and filename is None:
            raise ValueError('It takes too much time to calculate the entire shot, please set a time_range.')
        else:    
            if type(time_range) is not list and filename is None:
                raise TypeError('time_range is not a list.')
            if filename is None and len(time_range) != 2:
                raise ValueError('time_range should be a list of two elements.')
    
    if data_object is not None and type(data_object) == str:
        if exp_id is None:
            exp_id='*'
        d=flap.get_data_object(data_object,exp_id=exp_id)
        time_range=[d.coordinate('Time')[0][0,0,0],
                    d.coordinate('Time')[0][-1,0,0]]
        exp_id=d.exp_id
        flap.add_data_object(d, 'GPI_SLICED_FULL')
               
    if filename is None:
        wd=flap.config.get_all_section('Module NSTX_GPI')['Working directory']
        filename=flap_nstx.analysis.filename(exp_id=exp_id,
                                             working_directory=wd,
                                             time_range=time_range,
                                             purpose='sz velocity',
                                             comment='')
        
    pickle_filename=filename+'.pickle'
    if os.path.exists(pickle_filename) and nocalc:
        try:
            pickle.load(open(pickle_filename, 'rb'))
        except:
            print('The pickle file cannot be loaded. Recalculating the results.')
            nocalc=False
    elif nocalc:
        print('The pickle file cannot be loaded. Recalculating the results.')
        nocalc=False

    if not nocalc:
        slicing={'Time':flap.Intervals(time_range[0],time_range[1])}
    #Read data
    if data_object is None:
        print("\n------- Reading NSTX GPI data --------")
        if cache_data:
            try:
                d=flap.get_data_object('GPI',exp_id=exp_id)
            except:
                print('Data is not cached, it needs to be read.')
                d=flap.get_data('NSTX_GPI',exp_id=exp_id,
                                name='',
                                object_name='GPI')
        else:
            d=flap.get_data('NSTX_GPI',exp_id=exp_id,
                            name='',
                            object_name='GPI')
      
        d=flap.slice_data('GPI',exp_id=exp_id, 
                          slicing=slicing,
                          output_name='GPI_SLICED_FULL')
        d.data=np.asarray(d.data, dtype='float32')
        
    count=d.data.shape[0]
    vpol_p     = np.zeros([64,80,count])    #  poloidal velocity in km/sec vs. pixel
    vrad_p     = np.zeros([64,80,count])    #  radial velocity vs. pixel
    vpol_n     = np.zeros([64,80,count])    #  poloidal velocity in km/sec vs. pixel
    vrad_n     = np.zeros([64,80,count])    #  radial velocity vs. pixel  
    vpol = np.zeros([64,80,count])
    vrad = np.zeros([64,80,count])
    cmax_n = np.zeros([64,80,count])
    cmax_p = np.zeros([64,80,count])
    
    sample_time=d.coordinate('Time')[0][1,0,0]-d.coordinate('Time')[0][0,0,0]
    ccorr_n=np.zeros([64,80,64,80])
    ccorr_p=np.zeros([64,80,64,80])
    for t0 in range(fbin+1,count-fbin-1):
        
        acorr_pix_n=np.sqrt(np.sum(d.data[t0-fbin-1:t0+fbin-1,:,:]**2,axis=0))
        acorr_pix_p=np.sqrt(np.sum(d.data[t0-fbin+1:t0+fbin+1,:,:]**2,axis=0))
        acorr_pix_ref=np.sqrt(np.sum(d.data[t0-fbin:t0+fbin,:,:]**2,axis=0))
        print((t0-fbin-1)/(count-2*(fbin-1))*100.)
        for i0 in range(x_range[0],x_range[1]):
            for j0 in range(y_range[0],y_range[1]):
                frame_ref=d.data[t0-fbin:t0+fbin,i0,j0]
                
                for i1 in range(x_range[0]-x_search,x_range[1]+x_search+1):
                    for j1 in range(y_range[0]-y_search,y_range[1]+y_search+1):
                        frame_n=d.data[t0-fbin-1:t0+fbin-1,i1,j1]
                        frame_p=d.data[t0-fbin+1:t0+fbin+1,i1,j1]
                        ccorr_n[i0,j0,i1,j1]=np.sum(frame_ref*frame_n)
                        ccorr_p[i0,j0,i1,j1]=np.sum(frame_ref*frame_p)
                        
        for i0 in range(x_range[0],x_range[1]+1):
            for j0 in range(y_range[0],y_range[1]+1):
                vcorr_p=np.zeros([64,80])
                vcorr_n=np.zeros([64,80])
                for itest in range(i0-x_search,i0+x_search+1):
                    for jtest in range(j0-y_search,j0+y_search+1):                       
                        vcorr_p[itest,jtest]=ccorr_p[i0,j0,itest,jtest]/(acorr_pix_ref[i0,j0]*acorr_pix_p[itest,jtest])
                        vcorr_n[itest,jtest]=ccorr_n[i0,j0,itest,jtest]/(acorr_pix_ref[i0,j0]*acorr_pix_n[itest,jtest])
                        
                index_p=np.unravel_index(np.argmax(vcorr_p),shape=vcorr_p.shape)
                index_n=np.unravel_index(np.argmax(vcorr_n),shape=vcorr_n.shape)
                
                cmax_p[i0,j0,t0]=vcorr_p[index_p]
                cmax_n[i0,j0,t0]=vcorr_n[index_n]
                
                delta_index_p=np.asarray(index_p)-np.asarray([i0,j0])
                delta_index_n=np.asarray(index_n)-np.asarray([i0,j0])
                                           
                vpol_p[i0,j0,t0]=(coeff_z[0]*delta_index_p[0]+
                                  coeff_z[1]*delta_index_p[1])/sample_time                                        
                vpol_n[i0,j0,t0]=(coeff_z[0]*delta_index_n[0]+
                                  coeff_z[1]*delta_index_n[1])/sample_time
                      
                vrad_p[i0,j0,t0]=(coeff_r[0]*delta_index_p[0]+
                                  coeff_r[1]*delta_index_p[1])/sample_time                                        
                vrad_n[i0,j0,t0]=(coeff_r[0]*delta_index_n[0]+
                                  coeff_r[1]*delta_index_n[1])/sample_time                       
    
                
                vpol_tot = (vpol_p - vpol_n)/2.	 	# Average p and n correlations
                vrad_tot = (vrad_p - vrad_n)/2.     # This is non causal
                
    for t0 in range(int(fbin/2),count-int(fbin/2)):                                       #This is non causal
        vpol[:,:,t0] = np.mean(vpol_tot[:,:,t0-int(fbin/2):t0+int(fbin/2)], axis=2)
        vrad[:,:,t0] = np.mean(vrad_tot[:,:,t0-int(fbin/2):t0+int(fbin/2)], axis=2)
        
    if plot:
        plt.figure()
        plt.errorbar(d.coordinate('Time')[0][:,0,0],
                     np.mean(vrad[x_range[0]:x_range[1],y_range[0]:y_range[1],:], axis=(0,1)),
                     np.sqrt(np.var(vrad[x_range[0]:x_range[1],y_range[0]:y_range[1],:], axis=(0,1))))
        plt.title('Radial velocity vs time')
        plt.xlabel('Time [s]')
        plt.ylabel('Radial velocity [m/s]')
        
        plt.figure()
        plt.errorbar(d.coordinate('Time')[0][:,0,0],
                     np.mean(vpol[x_range[0]:x_range[1],y_range[0]:y_range[1],:], axis=(0,1)),
                     np.sqrt(np.var(vpol[x_range[0]:x_range[1],y_range[0]:y_range[1],:], axis=(0,1))))
        plt.title('Poloidal velocity vs time')
        plt.xlabel('Time [s]')
        plt.ylabel('Poloidal velocity [m/s]')
        plt.pause(0.001)
        
    return_dict={'Time':d.coordinate('Time')[0][:,0,0],
                 'Radial velocity':vrad,
                 'Poloidal velocity':vpol,
                 'Maximum correlation p':cmax_p,
                 'Maximum correlation n':cmax_n}
    if return_results:
        return return_dict