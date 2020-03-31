#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 11:49:13 2020

@author: mlampert
"""
import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import flap
import flap_nstx
from flap_nstx.analysis import calculate_nstx_gpi_avg_frame_velocity

thisdir = os.path.dirname(os.path.realpath(__file__))
fn = os.path.join(thisdir,"flap_nstx.cfg")
flap.config.read(file_name=fn)
flap_nstx.register()



def calculate_results_poloidal_radial(exp_id=None,
                                      time_range=None,
                                      x_coordinates=np.arange(5)*10+10,
                                      y_coordinates=np.arange(7)*10+10,
                                      subtraction_order=1,
                                      x_window=10,
                                      y_window=10,
                                      cache_data=True,
                                      recalc=False,
                                      device_coordinates=False,
                                      nocalc=True,
                                      pdf=True,
                                      pdf_filename='spatial_ccf_velocity_vs_rad.pdf'):

    wd=flap.config.get_all_section('Module NSTX_GPI')['Working directory']
    pickle_filename=flap_nstx.analysis.filename(exp_id=exp_id,
                                                working_directory=wd,
                                                time_range=time_range,
                                                purpose='ccf velocity',
                                                extension='.pickle')
    
    result=None
    if nocalc and not recalc:
        if os.path.exists(pickle_filename):
            result=pickle.load(open(pickle_filename, 'rb'))
         else:
            nocalc=False
    #if result is not None:
    #    if (result['X coordinates'].all() != x_coordinates.all()  or
    #        result['Y coordinates'].all() != y_coordinates).all() :
    #        raise ValueError('The x coordinates in the loaded pickle file are not the same as the reequested one from in inputs. Run the code with the recalc=True option.')

    if not nocalc or recalc:
        result={'X coordinates':x_coordinates,
                'Y coordinates':y_coordinates,
                'Average velocity':np.zeros([len(x_coordinates),len(y_coordinates),2]),
                'Average velocity max':np.zeros([len(x_coordinates),len(y_coordinates),2]),
                'Average correlation':np.zeros([len(x_coordinates),len(y_coordinates)])
                }
        
        for ind_x_coord in range(len(x_coordinates)):
            for ind_y_coord in range(len(y_coordinates)):
                x_coord=x_coordinates[ind_x_coord]
                y_coord=y_coordinates[ind_y_coord]
                try:
                    results=calculate_nstx_gpi_avg_frame_velocity(exp_id=exp_id, 
                                                                  time_range=time_range, 
                                                                  x_range=[x_coord-x_window,x_coord+x_window],
                                                                  y_range=[y_coord-y_window,y_coord+y_window],
                                                                  skip_structure_calculation=True,
                                                                  plot=False,
                                                                  subtraction_order_for_velocity=subtraction_order,
                                                                  correlation_threshold=0.6,
                                                                  pdf=False, 
                                                                  nlevel=51, 
                                                                  nocalc=False, 
                                                                  filter_level=3, 
                                                                  normalize_for_size=False,
                                                                  normalize_for_velocity=True,
                                                                  threshold_coeff=1.,
                                                                  normalize_f_high=1e3, 
                                                                  normalize='roundtrip', 
                                                                  velocity_base='cog', 
                                                                  return_results=True, 
                                                                  plot_gas=True)
                except:
                    print('Calculation for x=['+str(x_coord-x_window)+','+str(x_coord+x_window)+'] y=['+str(y_coord-y_window)+','+str(x_coord+y_window)+'] failed.')
                
                n_max=len(results['Velocity ccf'][:,0])-20
                
                for ind in range(n_max):
                    cur_vel_rad=results['Velocity ccf'][ind:ind+20,0]
                    cur_vel_pol=results['Velocity ccf'][ind:ind+20,1]
                    
                    
                    ind_nan_pol=np.logical_not(np.isnan(cur_vel_pol))
                    ind_nan_rad=np.logical_not(np.isnan(cur_vel_rad))
                    
                    
                    if np.sum(ind_nan_rad) != 0:
                        argmax_rad=np.argmax(np.abs(cur_vel_rad[ind_nan_rad]))
                    if np.sum(ind_nan_pol) != 0:
                        argmax_pol=np.argmax(np.abs(cur_vel_pol[ind_nan_pol]))
                        
                        result['Average velocity max'][ind_x_coord,ind_y_coord,0] += cur_vel_rad[ind_nan_rad][argmax_rad]/n_max
                        result['Average velocity max'][ind_x_coord,ind_y_coord,1] += cur_vel_pol[ind_nan_pol][argmax_pol]/n_max
#                
                ind_nan=np.logical_not(np.isnan(results['Velocity ccf'][:,0]))    
                
                result['Average velocity'][ind_x_coord,ind_y_coord,0]=np.mean(results['Velocity ccf'][ind_nan,0])
                result['Average velocity'][ind_x_coord,ind_y_coord,1]=np.mean(results['Velocity ccf'][ind_nan,1])
                result['Average correlation'][ind_x_coord,ind_y_coord]=np.mean(results['Correlation max'])
        pickle.dump(result,open(pickle_filename,'wb'))
        
    if pdf:
        pdf_pages=PdfPages(wd+'/'+pdf_filename)
    
    if device_coordinates:
        coeff_r=np.asarray([3.7183594,-0.77821046,1402.8097])/1000. #The coordinates are in meters
        coeff_z=np.asarray([0.18090118,3.0657776,70.544312])/1000. #The coordinates are in meters        
        
        x_coordinates=result['X coordinates']*coeff_r[0]+np.mean(results['Y coordinates'])*coeff_r[1]+coeff_r[2]
        
        plt.figure()
        plt.plot(x_coordinates, np.mean(result['Average velocity'][:,:,0], axis=1))
        plt.errorbar(result['X coordinates'], 
                     np.mean(result['Average velocity'][:,:,0], axis=1),
                     np.sqrt(np.var(result['Average velocity'][:,:,0], axis=1)))
        plt.xlabel('R [m]')
        plt.ylabel('Radial velocity [m/s]')
        plt.title('Radial velocity vs. radius')
        if pdf:
            pdf_pages.savefig()
            
        plt.figure()
        plt.plot(result['X coordinates'], np.mean(result['Average velocity'][:,:,1], axis=1))
        plt.errorbar(x_coordinates, 
                     np.mean(result['Average velocity'][:,:,1], axis=1),
                     np.sqrt(np.var(result['Average velocity'][:,:,1], axis=1)))
        plt.xlabel('R [m]')
        plt.ylabel('Poloidal velocity [m/s]')
        plt.title('Poloidal velocity vs. radius')
        if pdf:
            pdf_pages.savefig()        
        
        plt.figure()
        plt.plot(result['X coordinates'], np.mean(result['Average velocity max'][:,:,0], axis=1))
        plt.errorbar(result['X coordinates'], 
                     np.mean(result['Average velocity max'][:,:,0], axis=1),
                     np.sqrt(np.var(result['Average velocity max'][:,:,0], axis=1)))
        plt.xlabel('R [m]')
        plt.ylabel('Radial velocity max [m/s]')
        plt.title('Radial velocity max vs. radius')
        if pdf:
            pdf_pages.savefig()
            
        plt.figure()
        plt.plot(result['X coordinates'], np.mean(result['Average velocity max'][:,:,1], axis=1))
        plt.errorbar(result['X coordinates'], 
                     np.mean(result['Average velocity max'][:,:,1], axis=1),
                     np.sqrt(np.var(result['Average velocity max'][:,:,1], axis=1)))
        plt.xlabel('R [m]')
        plt.ylabel('Poloidal velocity max [m/s]')
        plt.title('Poloidal velocity max vs. radius')
        if pdf:
            pdf_pages.savefig()   
        
        plt.figure()
        plt.plot(result['X coordinates'], np.mean(result['Average correlation'], axis=1))
        plt.errorbar(result['X coordinates'], 
                     np.mean(result['Average correlation'], axis=1),
                     np.sqrt(np.var(result['Average correlation'], axis=1)))
        plt.xlabel('R [m]')
        plt.ylabel('Correlation')
        plt.title('Correlation vs. radius')
        if pdf:
            pdf_pages.savefig()
            pdf_pages.close()
    else:
        plt.figure()
        plt.plot(result['X coordinates'], np.mean(result['Average velocity'][:,:,0], axis=1))
        plt.errorbar(result['X coordinates'], 
                     np.mean(result['Average velocity'][:,:,0], axis=1),
                     np.sqrt(np.var(result['Average velocity'][:,:,0], axis=1)))
        plt.xlabel('Image x [pix]')
        plt.ylabel('Radial velocity [m/s]')
        plt.title('Radial velocity vs. radius')
        if pdf:
            pdf_pages.savefig()
            
        plt.figure()
        plt.plot(result['X coordinates'], np.mean(result['Average velocity'][:,:,1], axis=1))
        plt.errorbar(result['X coordinates'], 
                     np.mean(result['Average velocity'][:,:,1], axis=1),
                     np.sqrt(np.var(result['Average velocity'][:,:,1], axis=1)))
        plt.xlabel('Image x [pix]')
        plt.ylabel('Poloidal velocity [m/s]')
        plt.title('Poloidal velocity vs. radius')
        if pdf:
            pdf_pages.savefig()
            
        plt.figure()
        plt.plot(result['X coordinates'], np.mean(result['Average correlation'], axis=1))
        plt.errorbar(result['X coordinates'], 
                     np.mean(result['Average correlation'], axis=1),
                     np.sqrt(np.var(result['Average correlation'], axis=1)))
        plt.xlabel('Image x [pix]')
        plt.ylabel('Correlation')
        plt.title('Correlation vs. radius')
        if pdf:
            pdf_pages.savefig()

        plt.figure()
        plt.plot(result['X coordinates'], np.mean(result['Average velocity max'][:,:,0], axis=1))
        plt.errorbar(result['X coordinates'], 
                     np.mean(result['Average velocity max'][:,:,0], axis=1),
                     np.sqrt(np.var(result['Average velocity max'][:,:,0], axis=1)))
        plt.xlabel('Image x [pix]')
        plt.ylabel('Radial velocity max [m/s]')
        plt.title('Radial velocity max vs. radius')
        if pdf:
            pdf_pages.savefig()
            
        plt.figure()
        plt.plot(result['X coordinates'], np.mean(result['Average velocity max'][:,:,1], axis=1))
        plt.errorbar(result['X coordinates'], 
                     np.mean(result['Average velocity max'][:,:,1], axis=1),
                     np.sqrt(np.var(result['Average velocity max'][:,:,1], axis=1)))
        plt.xlabel('Image x [pix]')
        plt.ylabel('Poloidal velocity max [m/s]')
        plt.title('Poloidal velocity max vs. radius')
        if pdf:
            pdf_pages.savefig()            
            
        plt.figure()
        plt.plot(result['X coordinates'], np.mean(result['Average correlation'], axis=1))
        plt.errorbar(result['X coordinates'], 
                     np.mean(result['Average correlation'], axis=1),
                     np.sqrt(np.var(result['Average correlation'], axis=1)))
        plt.xlabel('Image x [pix]')
        plt.ylabel('Correlation')
        plt.title('Correlation vs. radius')
        if pdf:
            pdf_pages.savefig()
            pdf_pages.close()
    print(result['Average velocity max'])