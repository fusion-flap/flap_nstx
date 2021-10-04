#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 00:19:15 2021

@author: mlampert
"""

import os

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import numpy as np
import pickle

import flap
import flap_nstx
from flap_nstx.analysis import generate_displaced_gaussian, generate_displaced_random_noise, calculate_nstx_gpi_frame_by_frame_velocity
from flap_nstx.analysis import show_nstx_gpi_video_frames
flap_nstx.register()

thisdir = os.path.dirname(os.path.realpath(__file__))
fn = os.path.join(thisdir,"../flap_nstx.cfg")
flap.config.read(file_name=fn)

styled=True
if styled:
    plt.rc('font', family='serif', serif='Helvetica')
    labelsize=9.
    linewidth=0.5
    major_ticksize=2.
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
    import matplotlib.style as pltstyle
    pltstyle.use('default')


def test_spatial_displacement_estimation(gaussian=False,                        #Test the inaccuracy with Gaussian-structures
                                         gaussian_frame_size=False,             #Test the frame size vs. inaccuracy
                                         gaussian_frame_vs_structure_size=False,#Test the frame size and structure size vs. inaccuracy
                                         random=False,                          #Test the inaccuracy with random displaced frames
                                         subtraction_order=1,
                                         interpolation='parabola',              #Parabola or bicubic
                                         #Gaussian analysis settings
                                         n_pol=80,                              #Used for both Gaussian,frame and random, number of poloidal displacements
                                         n_rad=64,                              #Used for both Gaussian and random, number of radial displacements
                                         res_split=4,                           #Gaussian: pixel displacement are split with this, sub-pixel result, Random: number of displacements is n_pol/res_split
                                         n_size=13,                             #Number of structure sizes in Gaussian and the mixed calculation
                                         size_mult=3,                           #Pixel size multiplier, equals the optical resolution of GPI
                                         
                                         #Random analysis settings
                                         n_rand=100,                            #Each calculation is done with n_rand times
                                         random_poloidal_range=[1,20],
                                         random_radial_range=[1,16],
                                         random_step=1,
                                         
                                         #Frame size analysis
                                         frame_size_range=[8,200],              #Frame size range for the analysis of frame size limitation
                                         frame_size_step=8,
                                         structure_size=None,
                                         relative_structure_size=0.2,          #Set it to None if structure size is set
                                         
                                         #Frame size vs structure size analysis
                                         structure_size_range=[0.05, 1.0],
                                         relative_frame_size=0.2,
                                         
                                         #General settings
                                         nocalc=False,
                                         plot=True,
                                         pdf=False,
                                         publication=False,
                                         plot_sample_gaussian=False,
                                         plot_sample_random=False,
                                         plot_example_event=False,
                                         plot_example_event_frames=False,
                                         save_data_into_txt=False,
                                         ):
    
    wd=flap.config.get_all_section('Module NSTX_GPI')['Working directory']

    """
    ANALYSIS OF THE GAUSSIAN DISPLACED STRUCTURES
    """   
      
    if gaussian:
        pickle_file=wd+'/processed_data/gaussian_results.pickle'  
        if not nocalc:
            pol_disp_vec=np.arange(n_pol)/res_split
            rad_disp_vec=np.arange(n_rad)/res_split
            size_vec=(np.arange(n_size)+1)*size_mult
            
            result_vec_poloidal=np.zeros([n_pol,n_size])
            result_vec_radial=np.zeros([n_rad,n_size])
            
            for i_pol in range(n_pol):
                for j_size in range(n_size):
                    generate_displaced_gaussian(displacement=[0,pol_disp_vec[i_pol]], 
                                                  rotation_frequency=0., 
                                                  size=[size_vec[j_size],size_vec[j_size]], 
                                                  size_velocity=[0,0], 
                                                  output_name='gaussian', 
                                                  n_frames=3)
                    
                    result=calculate_nstx_gpi_frame_by_frame_velocity(exp_id=0, 
                                                            data_object='gaussian', 
                                                            normalize=None, 
                                                            normalize_for_velocity=False,
                                                            normalize_for_size=False,
                                                            interpolation=interpolation,
                                                            skip_structure_calculation=True, 
                                                            plot=False, 
                                                            nocalc=False, 
                                                            return_results=True,
                                                            subtraction_order_for_velocity=subtraction_order)
                    result_vec_poloidal[i_pol,j_size]=result['Velocity ccf'][0,1]/(3750/2.5)
                    
            plt.figure()
            plt.contourf(pol_disp_vec, size_vec, result_vec_poloidal.transpose(),)
                    
            for i_rad in range(n_rad):
                for j_size in range(n_size):
                    generate_displaced_gaussian(displacement=[rad_disp_vec[i_rad],0], 
                                                  rotation_frequency=0., 
                                                  size=[size_vec[j_size],size_vec[j_size]], 
                                                  size_velocity=[0,0], 
                                                  output_name='gaussian', 
                                                  n_frames=3)
                    
                    result=calculate_nstx_gpi_frame_by_frame_velocity(exp_id=0, 
                                                            data_object='gaussian', 
                                                            normalize=None, 
                                                            normalize_for_size=False, 
                                                            normalize_for_velocity=False,
                                                            skip_structure_calculation=True, 
                                                            interpolation=interpolation,
                                                            plot=False, 
                                                            nocalc=False, 
                                                            return_results=True,
                                                            subtraction_order_for_velocity=subtraction_order)
                    
                    result_vec_radial[i_rad,j_size]=result['Velocity ccf'][0,0]/(3750/2.5)
            pickle.dump([result_vec_poloidal,pol_disp_vec, result_vec_radial, rad_disp_vec, size_vec], open(pickle_file, 'wb'))
        else:
            result_vec_poloidal, pol_disp_vec, result_vec_radial, rad_disp_vec, size_vec=pickle.load(open(pickle_file, 'rb'))        
        if pdf:
            pdf=PdfPages(wd+'/plots/synthetic_rad_pol_size_dependence_gauss_'+interpolation+'.pdf')
        if plot:
            plt.figure()
            plt.contourf(pol_disp_vec[1:], 
                         size_vec[1:], 
                         result_vec_poloidal[1:,1:].transpose()/pol_disp_vec[None,1:]-1,
                         levels=51,
                         cmap='jet')
            plt.colorbar()
            plt.xlabel('Poloidal displacement [pix]')
            plt.ylabel('Size [pix]')
            plt.title('Inaccuracy of poloidal velocity estimation')
            pdf.savefig()
            
            plt.figure()
            plt.contourf(rad_disp_vec[1:],
                         size_vec[1:], 
                         result_vec_radial[1:,1:].transpose()/rad_disp_vec[None,1:]-1, 
                         levels=51,
                         cmap='jet')
            plt.colorbar()
            plt.xlabel('Radial displacement [pix]')
            plt.ylabel('Size [pix]')
            plt.title('Inaccuracy of radial velocity estimation')
            pdf.savefig()

            plt.figure()
            for i in range(1,n_size):
                plt.plot(pol_disp_vec[1:], result_vec_poloidal[1:,i]/pol_disp_vec[1:]-1, label=str(size_vec[i]))
            plt.xlabel('Poloidal displacement [pix]')
            plt.ylabel('Inaccuracy [pix]')
            plt.title('Inaccuracy of poloidal velocity estimation')    
            plt.legend()
            pdf.savefig()
            
            plt.figure()
            for i in range(1,n_size):
                plt.plot(rad_disp_vec[1:], result_vec_radial[1:,i]/rad_disp_vec[1:]-1, label=str(size_vec[i]))
            plt.xlabel('Radial displacement [pix]')
            plt.ylabel('Inaccuracy [pix]')
            plt.title('Inaccuracy of radial velocity estimation')
            plt.legend()
            pdf.savefig()
            
            pdf.close()
            
            if save_data_into_txt:
                data=result_vec_radial[1:,1:]
                filename=wd+'/processed_data/figure_poloidal_uncertainty.txt'
                file1=open(filename, 'w+')
                file1.write('#Poloidal displacement vector in pixels\n')
                for i in range(1, len(pol_disp_vec)):
                    file1.write(str(pol_disp_vec[i])+'\t')
                file1.write('\n#Size vector in pixels\n')
                for i in range(1, len(size_vec)):
                    file1.write(str(size_vec[i])+'\t')
                file1.write('\n#Relative uncertainty of the velocity estimation\n')
                for i in range(1,len(data[0,:])):
                    string=''
                    for j in range(1,len(data[:,0])):
                        string+=str(data[j,i])+'\t'
                    string+='\n'
                    file1.write(string)
                file1.close()
                
    """
    **************************************************
    ANALYSIS OF THE FRAME SIZE DISPLACEMENT ESTIMATION
    **************************************************
    """         
          
    if gaussian_frame_size:
        pickle_file=wd+'/processed_data/gaussian_frame_size_results_'+interpolation+'.pickle'  
        if not nocalc:
            rel_disp_vec=np.arange(n_pol)/n_pol
            
            frame_size_vec=np.arange(frame_size_range[0],frame_size_range[1],step=frame_size_step)

            result_vec=np.zeros([n_pol,len(frame_size_vec)])
            
            for i_disp in range(n_pol):
                for j_frame_size in range(len(frame_size_vec)):
                    if relative_structure_size is not None:
                        structure_size=frame_size_vec[j_frame_size]*relative_structure_size
                    generate_displaced_gaussian(displacement=[0,rel_disp_vec[i_disp]*frame_size_vec[j_frame_size]], 
                                                r0=[frame_size_vec[j_frame_size]/2,
                                                    frame_size_vec[j_frame_size]/2],
                                                frame_size=[frame_size_vec[j_frame_size],
                                                            frame_size_vec[j_frame_size]],
                                                size=[structure_size,structure_size], 
                                                size_velocity=[0,0], 
                                                rotation_frequency=0.,
                                                output_name='gaussian', 
                                                n_frames=3)
                    try:
                        result=calculate_nstx_gpi_frame_by_frame_velocity(exp_id=0,
                                                                     x_range=[0,frame_size_vec[j_frame_size]-1],
                                                                     y_range=[0,frame_size_vec[j_frame_size]-1],
                                                                     data_object='gaussian', 
                                                                     normalize=None, 
                                                                     normalize_for_size=False,
                                                                     normalize_for_velocity=False,
                                                                     interpolation=interpolation,
                                                                     skip_structure_calculation=True, 
                                                                     plot=False, 
                                                                     nocalc=False, 
                                                                     return_results=True,
                                                                     subtraction_order_for_velocity=subtraction_order)
                        result_vec[i_disp,j_frame_size]=result['Velocity ccf'][0,1]/(3750/2.5)
                    except:
                        print('Failed FS:'+str(frame_size_vec[j_frame_size])+' D:'+str(rel_disp_vec[i_disp]))
                        result_vec[i_disp,j_frame_size]=np.nan

            pickle.dump([result_vec, rel_disp_vec, frame_size_vec], open(pickle_file, 'wb'))
        else:
            result_vec, rel_disp_vec, frame_size_vec=pickle.load(open(pickle_file, 'rb'))        

        if pdf:
            pdf=PdfPages(wd+'/plots/synthetic_frame_size_dependence_gauss_'+interpolation+'.pdf')
        if plot:
            plt.figure()
            plt.contourf(rel_disp_vec[1:], 
                         frame_size_vec[1:], 
                         result_vec[1:,1:].transpose()/(rel_disp_vec[None,1:]*frame_size_vec[1:,None])-1,
                         levels=51,
                         cmap='jet')
            plt.colorbar()
            plt.xlabel('Displacement [pix]')
            plt.ylabel('Frame size')
            plt.title('Inaccuracy of displacement estimation')
            pdf.savefig()
            
            plt.figure()
            for i in range(len(frame_size_vec)):
                plt.plot(rel_disp_vec[1:], 
                         result_vec[1:,i]/(rel_disp_vec[1:]*frame_size_vec[i])-1, 
                         label=str(frame_size_vec[i]))
            plt.xlabel('Radial displacement [pix]')
            plt.ylabel('Relative inaccuracy')
            plt.title('Inaccuracy of displacement estimation')
            plt.legend()
            pdf.savefig()
            
            pdf.close()
            
            if save_data_into_txt:
                data=result_vec_radial[1:,1:]
                filename=wd+'/processed_data/figure_5a.txt'
                file1=open(filename, 'w+')
                file1.write('#Poloidal displacement vector in pixels\n')
                for i in range(1, len(rel_disp_vec[1:])):
                    file1.write(str(rel_disp_vec[i])+'\t')
                file1.write('\n#Size vector in pixels\n')
                for i in range(1, len(size_vec)):
                    file1.write(str(size_vec[i])+'\t')
                file1.write('\n#Relative uncertainty of the velocity estimation\n')
                for i in range(1,len(data[0,:])):
                    string=''
                    for j in range(1,len(data[:,0])):
                        string+=str(data[j,i])+'\t'
                    string+='\n'
                    file1.write(string)
                file1.close()
    
    """
    *******************************************************
    FRAME SIZE VS STRUCTURE SIZE WITH CONSTANT DISPLACEMENT
    *******************************************************
    """
                
    if gaussian_frame_vs_structure_size:
        pickle_file=wd+'/processed_data/gaussian_frame_vs_str_size_results_'+interpolation+'.pickle'  
        if not nocalc:
            
            frame_size_vec=np.arange(frame_size_range[0],frame_size_range[1],step=frame_size_step)

            result_vec=np.zeros([n_size,len(frame_size_vec)])
            relative_structure_size_vec=np.arange(structure_size_range[0], structure_size_range[1], step=(structure_size_range[1]-structure_size_range[0])/n_size)
            for i_size in range(n_size):
                for j_frame_size in range(len(frame_size_vec)):
                    structure_size=frame_size_vec[j_frame_size]*relative_structure_size_vec[i_size]
                    generate_displaced_gaussian(displacement=[0,frame_size_vec[j_frame_size]*relative_frame_size], 
                                                r0=[frame_size_vec[j_frame_size]/2,
                                                    frame_size_vec[j_frame_size]/2],
                                                frame_size=[frame_size_vec[j_frame_size],
                                                            frame_size_vec[j_frame_size]],
                                                size=[structure_size,structure_size], 
                                                size_velocity=[0,0], 
                                                rotation_frequency=0.,
                                                output_name='gaussian',
                                                n_frames=3)
                    try:
                        result=calculate_nstx_gpi_frame_by_frame_velocity(exp_id=0,
                                                                     x_range=[0,frame_size_vec[j_frame_size]-1],
                                                                     y_range=[0,frame_size_vec[j_frame_size]-1],
                                                                     data_object='gaussian', 
                                                                     normalize=None, 
                                                                     normalize_for_size=False, 
                                                                     interpolation=interpolation,
                                                                     skip_structure_calculation=True, 
                                                                     parabola_fit=True,
                                                                     fitting_range=5,
                                                                     plot=False, 
                                                                     nocalc=False, 
                                                                     return_results=True,
                                                                     subtraction_order_for_velocity=subtraction_order)
                        result_vec[i_size,j_frame_size]=result['Velocity ccf'][0,1]/(3750/2.5)
                    except:
                        print('Failed FS:'+str(frame_size_vec[j_frame_size])+' SS:'+str(structure_size))
                        result_vec[i_size,j_frame_size]=np.nan

            pickle.dump([result_vec, relative_structure_size_vec, frame_size_vec], open(pickle_file, 'wb'))
        else:
            result_vec, relative_structure_size_vec, frame_size_vec=pickle.load(open(pickle_file, 'rb'))        
        if pdf:
            pdf=PdfPages(wd+'/plots/synthetic_frame_vs_str_size_dependence_gauss_'+interpolation+'.pdf')
        if plot:
            displacement=frame_size_vec*relative_frame_size
            plt.figure()
            plt.contourf(relative_structure_size_vec[1:-1], 
                         frame_size_vec[1:], 
                         result_vec[1:,1:].transpose()/displacement[1:,None]-1,
                         levels=51,
                         cmap='jet')
            plt.colorbar()
            plt.xlabel('Relative structure size')
            plt.ylabel('Frame size')
            plt.title('Inaccuracy of displacement estimation')
            pdf.savefig()
            
            plt.figure()
            for i in range(len(frame_size_vec)):
                plt.plot(relative_structure_size_vec[1:-1], 
                         result_vec[1:,i]/displacement[i]-1, 
                         label=str(frame_size_vec[i]))
            plt.xlabel('Relative structure size')
            plt.ylabel('Relative inaccuracy')
            plt.title('Inaccuracy of displacement estimation')
            plt.legend()
            pdf.savefig()
            
            pdf.close()
            
            if save_data_into_txt:
                data=result_vec_radial[1:,1:]
                filename=wd+'/processed_data/figure_5b.txt'
                file1=open(filename, 'w+')
                file1.write('#Poloidal displacement vector in pixels\n')
                for i in range(1, len(pol_disp_vec)):
                    file1.write(str(pol_disp_vec[i])+'\t')
                file1.write('\n#Relative size vector in pixels\n')
                for i in range(1, len(relative_structure_size_vec)):
                    file1.write(str(relative_structure_size_vec[i])+'\t')
                file1.write('\n#Relative uncertainty of the velocity estimation\n')
                for i in range(1,len(data[0,:])):
                    string=''
                    for j in range(1,len(data[:,0])):
                        string+=str(data[j,i])+'\t'
                    string+='\n'
                    file1.write(string)
                file1.close()                 
                
    """
    ANALYSIS OF RANDOM FRAMES DISPLACED
    """                
                
    if random:
        pickle_file=wd+'/processed_data/random_results.pickle' 
        if not nocalc:
            pol_disp_vec=np.arange(random_poloidal_range[0], random_poloidal_range[1]+1, step=random_step)
            rad_disp_vec=np.arange(random_radial_range[0], random_radial_range[1]+1, step=random_step)
            
            result_vec_poloidal=np.zeros([len(pol_disp_vec),n_rand])
            result_vec_radial=np.zeros([len(rad_disp_vec),n_rand])
            for i_pol in range(len(pol_disp_vec)):
                for j_rand in range(n_rand):
                    generate_displaced_random_noise(exp_id=0,
                                                    displacement=[0,pol_disp_vec[i_pol]],
                                                    frame_size=[64,80],
                                                    sampling_time=2.5e-6,
                                                    circular=False,
                                                    amplitude_range=[0,4095],
                                                    output_name='random',
                                                    test=False,
                                                    n_frame=3
                                                    )
                    
                    result=calculate_nstx_gpi_frame_by_frame_velocity(exp_id=0, 
                                                                      data_object='random', 
                                                                      normalize=None, 
                                                                      normalize_for_size=False, 
                                                                      normalize_for_velocity=False,
                                                                      skip_structure_calculation=True,
                                                                      plot=False, 
                                                                      nocalc=False, 
                                                                      return_results=True, 
                                                                      subtraction_order_for_velocity=1)
                    
                    result_vec_poloidal[i_pol,j_rand]=result['Velocity ccf'][0,1]/(3750/2.5)

            for i_rad in range(len(rad_disp_vec)):
                for j_rand in range(n_rand):
                    generate_displaced_random_noise(exp_id=0,
                                                    displacement=[rad_disp_vec[i_rad],0],
                                                    frame_size=[64,80],
                                                    sampling_time=2.5e-6,
                                                    circular=False,
                                                    amplitude_range=[0,4095],
                                                    output_name='random',
                                                    test=False,
                                                    n_frame=3
                                                    )
                    result=calculate_nstx_gpi_frame_by_frame_velocity(exp_id=0, 
                                                          data_object='random', 
                                                          normalize=None, 
                                                          normalize_for_size=False, 
                                                          skip_structure_calculation=True, 
                                                          plot=False, 
                                                          nocalc=False, 
                                                          return_results=True, 
                                                          subtraction_order_for_velocity=1)
                    result_vec_radial[i_rad,j_rand]=result['Velocity ccf'][0,0]/(3750/2.5)
            print(result_vec_radial)
            pickle.dump([result_vec_poloidal,pol_disp_vec, result_vec_radial, rad_disp_vec], open(pickle_file, 'wb'))
        else:
            result_vec_poloidal, pol_disp_vec, result_vec_radial, rad_disp_vec=pickle.load(open(pickle_file, 'rb'))        
        
        if pdf:
            pdf=PdfPages(wd+'/plots/synthetic_rad_pol_size_dependence_random.pdf')
        if plot:
            plt.figure()
            plt.plot(pol_disp_vec[:], np.mean(result_vec_poloidal[:,:],axis=1)/pol_disp_vec[:]-1)
            plt.fill_between(pol_disp_vec[:],
                             np.mean(result_vec_poloidal[:,:],axis=1)/pol_disp_vec[:]-1-np.sqrt(np.var(result_vec_poloidal[:,:]/pol_disp_vec[:,None]-1, axis=1)),
                             np.mean(result_vec_poloidal[:,:],axis=1)/pol_disp_vec[:]-1+np.sqrt(np.var(result_vec_poloidal[:,:]/pol_disp_vec[:,None]-1, axis=1)),
                             alpha=0.3)
            plt.xlim([1,20])
            plt.xticks(np.arange(1, 20, 2.0))
            plt.xlabel('Poloidal displacement [pix]')
            plt.ylabel('Relative uncertainty')
            plt.title('Uncertainty of poloidal velocity estimation')
            pdf.savefig()
            
            plt.figure()
            plt.plot(rad_disp_vec[:], np.mean(result_vec_radial[:,:],axis=1)/rad_disp_vec[:]-1)
            plt.fill_between(rad_disp_vec[:],
                             np.mean(result_vec_radial[:,:],axis=1)/rad_disp_vec[:]-1-np.sqrt(np.var(result_vec_radial[:,:]/rad_disp_vec[:,None]-1, axis=1)),
                             np.mean(result_vec_radial[:,:],axis=1)/rad_disp_vec[:]-1+np.sqrt(np.var(result_vec_radial[:,:]/rad_disp_vec[:,None]-1, axis=1)),
                             alpha=0.3)
            plt.xlim([1,16])
            plt.xticks(np.arange(1, 16, 2.0))
            plt.xlabel('Radial displacement [pix]')
            plt.ylabel('Relative uncertainty')
            plt.title('Uncertainty of radial velocity estimation')
            pdf.savefig()
            pdf.close()
            
        if save_data_into_txt:
            filename=wd+'/processed_data/figure_random_radial_uncertainty.txt'
            file1=open(filename, 'w+')
            file1.write('Displacement \t Uncertainty \t Variance \n')
            for i in range(1, len(rad_disp_vec)):
                file1.write(str(rad_disp_vec[:])+'\t'+
                            str(np.mean(result_vec_radial[:,:],axis=1)/rad_disp_vec[:]-1)+'\t'+
                            str(np.sqrt(np.var(result_vec_radial[:,:]/rad_disp_vec[:,None]-1, axis=1))))

            file1.close()
            
            filename=wd+'/processed_data/figure_random_poloidal_uncertainty.txt'
            file1=open(filename, 'w+')
            file1.write('Displacement \t Uncertainty \t Variance \n')
            for i in range(1, len(pol_disp_vec)):
                file1.write(str(pol_disp_vec[:])+'\t'+
                            str(np.mean(result_vec_poloidal[:,:],axis=1)/pol_disp_vec[:]-1)+'\t'+
                            str(np.sqrt(np.var(result_vec_poloidal[:,:]/pol_disp_vec[:,None]-1, axis=1))))

            file1.close()
    
    if plot_sample_gaussian:
        if pdf:
            pdf=PdfPages(wd+'/plots/synthetic_gaussian_sample_and_ccf.pdf')
        generate_displaced_gaussian(displacement=[0,10], 
                            r0=[30,40],
                            frame_size=[64,80],
                            size=[15,15], 
                            size_velocity=[0,0], 
                            rotation_frequency=0.,
                            output_name='gaussian',
                            n_frames=3)
        result=calculate_nstx_gpi_frame_by_frame_velocity(exp_id=0, 
                                      data_object='gaussian', 
                                      normalize=None, 
                                      normalize_for_size=False, 
                                      normalize_for_velocity=False,
                                      skip_structure_calculation=True, 
                                      plot=False, 
                                      nocalc=False, 
                                      return_results=True, 
                                      subtraction_order_for_velocity=1)
        plt.figure()
        flap.plot('gaussian', plot_type='contour', slicing={'Sample':0}, axes=['Image x', 'Image y'], options={'Equal axes':True})
        pdf.savefig()
        plt.figure()
        flap.plot('gaussian', plot_type='contour', slicing={'Sample':1}, axes=['Image x', 'Image y'], options={'Equal axes':True})
        pdf.savefig()
        plt.figure()
        flap.plot('GPI_FRAME_12_CCF', plot_type='contour', slicing={'Sample':0}, axes=['Image x lag', 'Image y lag'], options={'Equal axes':True})
        pdf.savefig()
        pdf.close()
        
    if plot_sample_random:
        if pdf:
            pdf=PdfPages(wd+'/plots/synthetic_random_sample_and_ccf.pdf')
        generate_displaced_random_noise(exp_id=0,
                            displacement=[20,0],
                            frame_size=[64,80],
                            sampling_time=2.5e-6,
                            circular=False,
                            amplitude_range=[0,4095],
                            output_name='random',
                            test=False,
                            n_frame=3
                            )
        result=calculate_nstx_gpi_frame_by_frame_velocity(exp_id=0, 
                                                          data_object='random', 
                                                          normalize=None, 
                                                          normalize_for_size=False, 
                                                          normalize_for_velocity=False,
                                                          skip_structure_calculation=True, 
                                                          plot=False, 
                                                          nocalc=False, 
                                                          return_results=True, 
                                                          subtraction_order_for_velocity=1)
        plt.figure()
        flap.plot('random', plot_type='image', slicing={'Sample':0}, axes=['Image x', 'Image y'], options={'Equal axes':True})
        if pdf:
            pdf.savefig()
        plt.figure()
        flap.plot('random', plot_type='image', slicing={'Sample':1}, axes=['Image x', 'Image y'], options={'Equal axes':True})
        if pdf:
            pdf.savefig()
        plt.figure()
        flap.plot('GPI_FRAME_12_CCF', plot_type='image', slicing={'Sample':0}, axes=['Image x lag', 'Image y lag'], options={'Equal axes':True})
        if pdf:
            pdf.savefig()
            pdf.close()
        
        
        
    if plot_example_event:
        calculate_nstx_gpi_frame_by_frame_velocity(exp_id=141319,
                                      time_range=[0.552,0.553],  
                                                            normalize='roundtrip', 
                                                            normalize_for_size=True, 
                                                            skip_structure_calculation=False, 
                                                            plot=True, 
                                                            pdf=True,
                                                            nocalc=True,
                                                            plot_scatter=False,
                                                            plot_for_publication=True,
                                                             
                                                            return_results=True,
                                                            subtraction_order_for_velocity=4)
    if plot_example_event_frames:
        plt.figure()
        ax,fig=plt.subplots(figsize=(3.35*2,5.5))
        pdf=PdfPages(wd+'/plots/fig__1413190.552500_9_frame.pdf')
        show_nstx_gpi_video_frames(exp_id=141319, 
                                   start_time=0.552500-6*2.5e-6,
                                   n_frame=9,
                                   logz=False,
                                   z_range=[0,3900],
                                   plot_filtered=False, 
                                   normalize=False,
                                   cache_data=False, 
                                   plot_flux=False, 
                                   plot_separatrix=True, 
                                   flux_coordinates=False,
                                   device_coordinates=True,
                                   new_plot=False,
                                   save_pdf=True,
                                   colormap='gist_ncar',
                                   save_for_paraview=False,
                                   colorbar_visibility=False
                                   )
        pdf.savefig()
        pdf.close()