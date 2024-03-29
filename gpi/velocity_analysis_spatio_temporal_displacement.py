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
from flap_nstx.gpi import nstx_gpi_contour_structure_finder

thisdir = os.path.dirname(os.path.realpath(__file__))
fn = os.path.join(thisdir,"../flap_nstx.cfg")
flap.config.read(file_name=fn)

#Scientific modules
import matplotlib.style as pltstyle
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import numpy as np
import scipy
import pickle
   
def nstx_gpi_velocity_analysis_spatio_temporal_displacement(exp_id=None,                          #Shot number
                                                            time_range=None,                      #The time range for the calculation
                                                            data_object=None,                     #Input data object if available from outside (e.g. generated sythetic signal)
                                                            x_range=[10,53],                       #X range for the calculation
                                                            y_range=[10,69],                       #Y range for the calculation
                                                            x_search=10,
                                                            y_search=10,
                                                            
                                                            fbin=10,
                                                            plot=True,                            #Plot the results
                                                            pdf=False,                            #Print the results into a PDF
                                                            plot_error=False,                     #Plot the errorbars of the velocity calculation based on the line fitting and its RMS error
                                                          
                                                            #File input/output options
                                                            filename=None,                        #Filename for restoring data
                                                            nocalc=True,                          #Restore the results from the .pickle file from filename+.pickle
                                                            return_results=False,
                                                            verbose=False,
                                                            ):
    
    #THE RESULT IS NON CAUSAL, BUT THERE IS NO PHASE SHIFT IN IT. CARE SHOULD BE TAKEN
    #WHEN THE RESULT IS USED FOR CROSS-CORRELATION ANALYSIS. FALSE CORRELATION VALUES
    #CAN RESULT FROM THE CALCULATION.    
    
    
    #Constants for the calculation
    #Using the spatial calibration to find the actual velocities.
    coeff_r=np.asarray([3.75,0,1402.8097])/1000. #The coordinates are in meters, the coefficients are in mm
    coeff_z=np.asarray([0,3.75,70.544312])/1000.  #The coordinates are in meters, the coefficients are in mm
    
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
        comment=''
        filename=flap_nstx.tools.filename(exp_id=exp_id,
                                          working_directory=wd+'/processed_data/',
                                          time_range=time_range,
                                          purpose='sz velocity',
                                          comment=comment)
        
    pickle_filename=filename+'.pickle'
    if not os.path.exists(pickle_filename) and nocalc:
        print('The pickle file cannot be loaded. Recalculating the results.')
        nocalc=False

    if nocalc is False:
        slicing={'Time':flap.Intervals(time_range[0],
                                       time_range[1])}
        #Read data
        if data_object is None:
            print("\n------- Reading NSTX GPI data --------")
            d=flap.get_data('NSTX_GPI',exp_id=exp_id,
                            name='',
                            object_name='GPI')
          
            d=flap.slice_data('GPI',exp_id=exp_id, 
                              slicing=slicing,
                              output_name='GPI_SLICED_FULL')
            d.data=np.asarray(d.data, dtype='float32')
            
        count=d.data.shape[0]
        vpol_p     = np.zeros([x_range[1]-x_range[0]+1,y_range[1]-y_range[0]+1,count])    #  poloidal velocity in km/sec vs. pixel
        vrad_p     = np.zeros([x_range[1]-x_range[0]+1,y_range[1]-y_range[0]+1,count])    #  radial velocity vs. pixel
        vpol_n     = np.zeros([x_range[1]-x_range[0]+1,y_range[1]-y_range[0]+1,count])    #  poloidal velocity in km/sec vs. pixel
        vrad_n     = np.zeros([x_range[1]-x_range[0]+1,y_range[1]-y_range[0]+1,count])    #  radial velocity vs. pixel  
        
        vpol = np.zeros([x_range[1]-x_range[0]+1,y_range[1]-y_range[0]+1,count])
        vrad = np.zeros([x_range[1]-x_range[0]+1,y_range[1]-y_range[0]+1,count])
        cmax_n = np.zeros([x_range[1]-x_range[0]+1,y_range[1]-y_range[0]+1,count])
        cmax_p = np.zeros([x_range[1]-x_range[0]+1,y_range[1]-y_range[0]+1,count])
        
        sample_time=d.coordinate('Time')[0][1,0,0]-d.coordinate('Time')[0][0,0,0]
        
        ccorr_n=np.zeros([x_range[1]-x_range[0]+1,
                          y_range[1]-y_range[0]+1,
                          2*x_search+1,
                          2*y_search+1])
            
        ccorr_p=np.zeros([x_range[1]-x_range[0]+1,
                          y_range[1]-y_range[0]+1,
                          2*x_search+1,
                          2*y_search+1])
        #Moving average frames should be created first for the entire frame so it could be used during the cross-correlation
        #moving_average_data = np.convolve(d.data, np.ones(fbin*2), 'valid') / fbin*2
        #offset_subtracted_data=
        for t0 in range(fbin+1,count-fbin-1):
            #Zero lag Autocorrelation calculation for the reference, +sample_time, -sample_time data
            n_data=d.data[t0-fbin-1:t0+fbin-1,
                          x_range[0]-x_search:x_range[1]+x_search+1,
                          y_range[0]-y_search:y_range[1]+y_search+1]
            acorr_pix_n=np.sqrt(np.sum((n_data-np.mean(n_data, axis=0))**2,axis=0))
            
            p_data=d.data[t0-fbin+1:t0+fbin+1,
                          x_range[0]-x_search:x_range[1]+x_search+1,
                          y_range[0]-y_search:y_range[1]+y_search+1]
            acorr_pix_p=np.sqrt(np.sum((p_data-np.mean(p_data, axis=0))**2,axis=0))
            
            ref_data=d.data[t0-fbin:t0+fbin,
                            x_range[0]:x_range[1]+1,
                            y_range[0]:y_range[1]+1]
            acorr_pix_ref=np.sqrt(np.sum((ref_data-np.mean(ref_data, axis=0))**2,axis=0))
            
            #print((t0-fbin-1)/(count-2*(fbin-1))*100.)
            #Zero lag Crosscovariance calculation for the positive and negative sample time signal
            for i0 in range(x_range[1]-x_range[0]+1):
                for j0 in range(y_range[1]-y_range[0]+1):
                    
                    frame_ref=d.data[t0-fbin:t0+fbin,
                                     i0+x_range[0],
                                     j0+y_range[0]]                             #oN=pointer*nt*nx*ny*nxs*nys
                    frame_ref=frame_ref-np.mean(frame_ref)                      #oN=2*2fbin*nt*nx*ny

                    frame_n=d.data[t0-fbin-1:t0+fbin-1,
                                   i0+x_range[0]-x_search:2*x_search+1+i0+x_range[0]-x_search,
                                   j0+y_range[0]-y_search:2*y_search+1+j0+y_range[0]-y_search]           #oN=pointer*nt*nx*ny*nxs*nys
                    frame_n=frame_n-np.mean(frame_n,axis=0)[None,:,:]                    #oN=2*2fbin*nt*nx*ny*nxs*nys
                    
                    frame_p=d.data[t0-fbin+1:t0+fbin+1,
                                   i0+x_range[0]-x_search:2*x_search+1+i0+x_range[0]-x_search,
                                   j0+y_range[0]-y_search:2*y_search+1+j0+y_range[0]-y_search]           #oN=pointer*nt*nx*ny*nxs*nys
                    frame_p=frame_p-np.mean(frame_p,axis=0)[None,:,:]                    #oN=2*2fbin*nt*nx*ny*nxs*nys #mean should come from the mean array, it is calculated multiple times
                    
                    ccorr_n[i0,j0,:,:]=np.sum(frame_ref[:,None,None]*frame_n, axis=0)      #oN=2*fbin*fbin*nt*nx*ny*nxs*nys    These could be generated with rolling
                    ccorr_p[i0,j0,:,:]=np.sum(frame_ref[:,None,None]*frame_p, axis=0)      #oN=2*fbin*fbin*nt*nx*ny*nxs*nys    correlation calculation getting rid of one fbin order
            
            #Calculating the actual cross-correlation coefficients       
            for i0 in range(x_range[1]-x_range[0]+1):
                for j0 in range(y_range[1]-y_range[0]+1):

                    vcorr_p=ccorr_p[i0,j0,:,:]/(acorr_pix_ref[i0,j0]*acorr_pix_p[i0:i0+2*x_search+1,j0:j0+2*y_search+1]) #2*nt*nx*ny*nxs*nys         This could be morphed to
                    vcorr_n=ccorr_n[i0,j0,:,:]/(acorr_pix_ref[i0,j0]*acorr_pix_n[i0:i0+2*x_search+1,j0:j0+2*y_search+1]) #2*nt*nx*ny*nxs*nys         matrix calculations.
                            
                    #Calculating the displacement in pixel coordinates
                    index_p=np.unravel_index(np.argmax(vcorr_p),shape=vcorr_p.shape)
                    index_n=np.unravel_index(np.argmax(vcorr_n),shape=vcorr_n.shape)
                    
                    cmax_p[i0,j0,t0]=vcorr_p[index_p]
                    cmax_n[i0,j0,t0]=vcorr_n[index_n]
                    
                    #Transforming the coordinates into spatial coordinates
                    delta_index_p=np.asarray(index_p)-np.asarray([x_search,y_search])
                    delta_index_n=np.asarray(index_n)-np.asarray([x_search,y_search])
                        
                    vpol_p[i0,j0,t0]=(coeff_z[0]*delta_index_p[0]+
                                      coeff_z[1]*delta_index_p[1])/sample_time                                        
                    vpol_n[i0,j0,t0]=(coeff_z[0]*delta_index_n[0]+
                                      coeff_z[1]*delta_index_n[1])/sample_time
                    vrad_p[i0,j0,t0]=(coeff_r[0]*delta_index_p[0]+
                                      coeff_r[1]*delta_index_p[1])/sample_time                                        
                    vrad_n[i0,j0,t0]=(coeff_r[0]*delta_index_n[0]+
                                      coeff_r[1]*delta_index_n[1])/sample_time                       
            #Calculating the average between the positive and negative shifted pixels
            vpol_tot = (vpol_p - vpol_n)/2.	 	# Average p and n correlations
            vrad_tot = (vrad_p - vrad_n)/2.     # This is non causal
            #print((t0-fbin)/(count-2*(fbin+1))*100,'% done from the calculation for shot '+str(exp_id))
        
        #Averaging in an fbin long time window
        for t0 in range(int(fbin/2),count-int(fbin/2)):
            vpol[:,:,t0] = np.mean(vpol_tot[:,:,t0-int(fbin/2):t0+int(fbin/2)], axis=2)
            vrad[:,:,t0] = np.mean(vrad_tot[:,:,t0-int(fbin/2):t0+int(fbin/2)], axis=2)
        
        time=d.coordinate('Time')[0][:,0,0][fbin:-fbin]
        vpol=vpol[:,:,fbin:-fbin]
        vrad=vrad[:,:,fbin:-fbin]
        vpol_tot=vpol_tot[:,:,fbin:-fbin]
        vrad_tot=vrad_tot[:,:,fbin:-fbin]
        

        results={'Time':time,
                 'Image x':np.arange(x_range[1]-x_range[0]+1)+x_range[0],
                 'Image y':np.arange(y_range[1]-y_range[0]+1)+y_range[0],
                 'Radial velocity smooth':vrad,
                 'Poloidal velocity smooth':vpol,
                 'Radial velocity':vrad_tot,
                 'Poloidal velocity':vpol_tot,
                 'Maximum correlation p':cmax_p,
                 'Maximum correlation n':cmax_n,
                 'X range':x_range,
                 'Y range':y_range,
                 'fbin':fbin}
        
        pickle.dump(results, open(pickle_filename, 'wb'))
    else:
        results=pickle.load(open(pickle_filename, 'rb'))
        
        if (results['X range'] != x_range or
            results['Y range'] != y_range or
            results['fbin'] != fbin):
            raise ValueError('The requested results are not in the saved file. Run the code with nocalc=False')
            
        if verbose: print('Data loaded from pickle file.')
    
    if pdf:
        pdf=PdfPages(filename.replace('processed_data', 'plots')+'.pdf')
        
    if plot:
        plt.figure()
        plt.errorbar(results['Time'],
                     np.mean(results['Radial velocity'], axis=(0,1)),
                     np.sqrt(np.var(results['Radial velocity'], axis=(0,1))))
        plt.title('Radial velocity vs time')
        plt.xlabel('Time [s]')
        plt.ylabel('Radial velocity [m/s]')
        if pdf:
            pdf.savefig()
        
        plt.figure()
        plt.errorbar(results['Time'],
                     np.mean(results['Poloidal velocity'], axis=(0,1)),
                     np.sqrt(np.var(results['Poloidal velocity'], axis=(0,1))))
        
        plt.title('Poloidal velocity vs time')
        plt.xlabel('Time [s]')
        plt.ylabel('Poloidal velocity [m/s]')
        plt.pause(0.001)
        if pdf:
            pdf.savefig()
            
        plt.figure()
        plt.errorbar(results['Time'],
                     np.mean(results['Maximum correlation p'], axis=(0,1)),
                     np.sqrt(np.var(results['Maximum correlation p'], axis=(0,1))))
        
        plt.title('Maximum correlation p vs time')
        plt.xlabel('Time [s]')
        plt.ylabel('Maximum correlation p')
        plt.pause(0.001)
        if pdf:
            pdf.savefig()
            pdf.close()
            
    if return_results:
        return results