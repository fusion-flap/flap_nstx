#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 11:51:52 2021

@author: mlampert
"""

import os
import time

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import numpy as np
import pickle

import flap
import flap_nstx
from flap_nstx.gpi import generate_displaced_gaussian, generate_displaced_random_noise, calculate_nstx_gpi_angular_velocity
from flap_nstx.gpi import show_nstx_gpi_video_frames
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


def test_angular_displacement_estimation(gaussian=False,                        #Test the inaccuracy with Gaussian-structures
                                         gaussian_frame_size=False,             #Test the frame size vs. inaccuracy
                                         gaussian_frame_vs_structure_size=False,#Test the frame size and structure size vs. inaccuracy
                                         scaling_factor_analysis=False,
                                         
                                         calc_flap=True,
                                         warp_polar_order=1,
                                         subtraction_order=2,
                                         zero_padding_scale=1,
                                         calculate_half_fft=False,
                                         interpolation='parabola',              #Parabola or bicubic
                                         #Gaussian analysis settings
                                         n_angle=90,                            #Used for both Gaussian,frame and random, number of poloidal displacements
                                         angle_range=[0.,90.],
                                         n_scale=40,
                                         scale_range=[0.01,0.2],
                                         res_split=4,                           #Gaussian: pixel displacement are split with this, sub-pixel result, Random: number of displacements is n_pol/res_split
                                         n_size=13,                             #Number of structure sizes in Gaussian and the mixed calculation
                                         size_mult=3,                           #Pixel size multiplier, equals the optical resolution of GPI
                                         
                                         #Random analysis settings
                                         n_rand=100,                            #Each calculation is done with n_rand times
                                         random_poloidal_range=[1,20],
                                         random_radial_range=[1,16],
                                         random_step=1,
                                         
                                         #Frame size analysis
                                         frame_size_range=[8,200],              #Frame size range for the analysis of frame size dependence
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
                                         gaussian_blur=False,
                                         publication=False,
                                         plot_sample_gaussian=False,
                                         plot_sample_random=False,
                                         plot_example_event=False,
                                         plot_example_event_frames=False,
                                         save_data_into_txt=False,
                                         hann_window=False,
                                         scale_correction_factor=1.0
                                         ):
    
    wd=flap.config.get_all_section('Module NSTX_GPI')['Working directory']

    """
    ANALYSIS OF THE GAUSSIAN DISPLACED STRUCTURES
    """   
    
    if gaussian:
        pickle_file=wd+'/processed_data/gaussian_angle_results.pickle'  
        if not nocalc:
            angle_rot_vec=np.arange(n_angle)/res_split

            size_vec=(np.arange(n_size)+1)*size_mult
            
            result_vec_angle=np.zeros([n_angle,n_size])
            sampling_time=2.5e-6
            for i_angle in range(n_angle):
                for j_size in range(n_size):
                    start_time=time.time()
                    generate_displaced_gaussian(displacement=[0,0], #[0,pol_disp_vec[i_pol]], 
                                                angle_per_frame=angle_rot_vec[i_angle], 
                                                size=[size_vec[j_size],size_vec[j_size]*2], 
                                                size_velocity=[0,0],
                                                sampling_time=sampling_time,
                                                output_name='gaussian', 
                                                n_frames=3)
                    
                    result=calculate_nstx_gpi_angular_velocity(exp_id=0, 
                                                               data_object='gaussian', 
                                                               normalize_for_velocity=False,
                                                               zero_padding_scale=zero_padding_scale,
                                                               sigma_low=3,
                                                               plot=False,
                                                               pdf=False,
                                                               gaussian_blur=True,
                                                               nocalc=False, 
                                                               return_results=True,
                                                               log_polar_data_shape=(360,320),
                                                               subtraction_order_for_velocity=subtraction_order)

                    result_vec_angle[i_angle,j_size]=result['Angle difference FLAP']
                    
                finish_time=time.time()
                rem_time=(finish_time-start_time)*(n_angle-i_angle)
                print('Remaining time from the calculation: '+str(int(np.mod(rem_time,60)))+'min.')
            plt.figure()
            plt.contourf(angle_rot_vec, size_vec, result_vec_angle.transpose(),)

            pickle.dump([result_vec_angle, angle_rot_vec, size_vec], open(pickle_file, 'wb'))
        else:
            result_vec_angle, angle_rot_vec, size_vec = pickle.load(open(pickle_file, 'rb'))        
            

        if pdf:
            pdf_pages=PdfPages(wd+'/plots/synthetic_angle_estimate_size_dep_gauss_'+interpolation+'.pdf')

        if plot:
            plt.figure()
            plt.contourf(angle_rot_vec[1:], 
                         size_vec[1:], 
                         result_vec_angle[1:,1:].transpose()/angle_rot_vec[None,1:]-1,
                         levels=51,
                         cmap='jet')
            plt.colorbar()
            plt.xlabel('Angle rotation [deg]')
            plt.ylabel('Size [pix]')
            plt.title('Inaccuracy of angle rotation estimation')
            if pdf:
                pdf_pages.savefig()


            plt.figure()
            for i in range(1,n_size):
                plt.plot(angle_rot_vec[1:], result_vec_angle[1:,i]/angle_rot_vec[1:]-1, label=str(size_vec[i]))
            plt.xlabel('Angle rotation [deg]')
            plt.ylabel('Inaccuracy [pix]')
            plt.title('Inaccuracy of angle estimation')    
            plt.legend()
            
            if pdf:
                pdf_pages.savefig()
                pdf_pages.close()
            
            if save_data_into_txt:
                data=result_vec_angle[1:,1:]
                filename=wd+'/processed_data/figure_angular_uncertainty.txt'
                file1=open(filename, 'w+')
                file1.write('#Angle rotation vector in pixels\n')
                for i in range(1, len(angle_rot_vec)):
                    file1.write(str(angle_rot_vec[i])+'\t')
                file1.write('\n#Size vector in pixels\n')
                for i in range(1, len(size_vec)):
                    file1.write(str(size_vec[i])+'\t')
                file1.write('\n#Relative uncertainty of the angular rotation estimation\n')
                for i in range(1,len(data[0,:])):
                    string=''
                    for j in range(1,len(data[:,0])):
                        string+=str(data[j,i])+'\t'
                    string+='\n'
                    file1.write(string)
                file1.close()
                
    if scaling_factor_analysis:
        pickle_file=wd+'/processed_data/gaussian_scaling_factor_results.pickle'  
        if not nocalc:
            

            scaling_factor_vec=np.arange(n_scale)/n_scale*(scale_range[1]-scale_range[0])+scale_range[0]
            angle_rot_vec=np.arange(n_angle)/n_angle*(angle_range[1]-angle_range[0])+angle_range[0]
            result_vec_angle=np.zeros([n_angle,n_scale])
            result_vec_scale=np.zeros([n_angle,n_scale])
            sampling_time=2.5e-6
            for i_angle in range(n_angle):
                for j_scale in range(n_scale):
                    start_time=time.time()
                    generate_displaced_gaussian(displacement=[0,0], #[0,pol_disp_vec[i_pol]], 
                                                angle_per_frame=angle_rot_vec[i_angle], 
                                                size=[10,15], 
                                                size_velocity=[scaling_factor_vec[j_scale],
                                                               scaling_factor_vec[j_scale]],
                                                sampling_time=sampling_time,
                                                output_name='gaussian', 
                                                #use_image_instead=True,
                                                n_frames=3)
                    
                    result=calculate_nstx_gpi_angular_velocity(exp_id=0, 
                                                               data_object='gaussian', 
                                                               normalize_for_velocity=False,
                                                               plot=False,
                                                               pdf=False,
                                                               upsample_factor=16,
                                                               fitting_range=5,
                                                               zero_padding_scale=zero_padding_scale,
                                                               gaussian_blur=gaussian_blur,
                                                               sigma_low=1,
                                                               sigma_high=None,
                                                               nocalc=False,
                                                               warp_polar_order=warp_polar_order,
                                                               return_results=True,
                                                               hann_window=hann_window,
                                                               #log_polar_data_shape=(360,320),
                                                               subtraction_order_for_velocity=subtraction_order,
                                                               calculate_half_fft=calculate_half_fft)
                    if calc_flap:
                        print(scaling_factor_vec[j_scale],
                              result['Expansion velocity ccf FLAP'],
                              scaling_factor_vec[j_scale]/result['Expansion velocity ccf FLAP'])
                        print(angle_rot_vec[i_angle],result['Angle difference FLAP log'], 
                              angle_rot_vec[i_angle]/result['Angle difference FLAP log'])
                        
                        result_vec_scale[i_angle,j_scale]=result['Expansion velocity ccf FLAP']
                        result_vec_angle[i_angle,j_scale]=result['Angle difference FLAP log']
                    else:
                        print(scaling_factor_vec[j_scale],
                              result['Expansion velocity ccf'],
                              scaling_factor_vec[j_scale]/result['Expansion velocity ccf'])
                        print(angle_rot_vec[i_angle],result['Angle difference log'], 
                              angle_rot_vec[i_angle]/result['Angle difference log'])
                        
                        result_vec_scale[i_angle,j_scale]=result['Expansion velocity ccf']
                        result_vec_angle[i_angle,j_scale]=result['Angle difference log']
                    #raise ValueError('asdf')
                finish_time=time.time()
                rem_time=(finish_time-start_time)*(n_angle-i_angle)
                print('Remaining time from the calculation: '+str(int(np.mod(rem_time,60)))+'min.')
            # plt.figure()
            # plt.contourf(angle_rot_vec, 
            #              scaling_factor_vec, 
            #              result_vec_scale.transpose(),
            #              levels=51)

            pickle.dump([result_vec_scale, angle_rot_vec, scaling_factor_vec], open(pickle_file, 'wb'))
        else:
            result_vec_scale, angle_rot_vec, scaling_factor_vec = pickle.load(open(pickle_file, 'rb'))        
            

        if pdf:
            pdf_pages=PdfPages(wd+'/plots/synthetic_angle_estimate_scale_dep_gauss_'+interpolation+'.pdf')

        if plot:
            plt.figure()
            plt.contourf(angle_rot_vec[1:], 
                         scaling_factor_vec[1:], 
                         (scaling_factor_vec[1:,None])/(result_vec_scale[1:,1:].transpose()),
                         levels=51,
                         cmap='jet')
            plt.colorbar()
            plt.xlabel('Angle rotation [deg]')
            plt.ylabel('Scaling factor')
            plt.title('Inaccuracy of scale rotation estimation')
            if pdf:
                pdf_pages.savefig()


            plt.figure()
            for i in range(1,n_angle):
                plt.plot(scaling_factor_vec[1:], 
                         (scaling_factor_vec[1:])/(result_vec_scale[i,1:].transpose()), 
                         label=str(angle_rot_vec[i]))
                
            plt.xlabel('Scaling factor setting')
            plt.ylabel('est/set ratio')
            plt.title('Inaccuracy of scale estimation')    
            plt.legend()
            if pdf:
                pdf_pages.savefig()


            plt.figure()
            for i in range(1,n_scale):
                plt.plot(angle_rot_vec[1:], 
                         result_vec_angle[1:,i]/angle_rot_vec[1:], 
                         label=str(scaling_factor_vec[i]))
            plt.xlabel('Angle rotation [deg]')
            plt.ylabel('est/set ratio')
            plt.title('Inaccuracy of angle estimation')    
            plt.legend()
            
            if pdf:
                pdf_pages.savefig()
                pdf_pages.close()
            
            if save_data_into_txt:
                data=result_vec_angle[1:,1:]
                filename=wd+'/processed_data/figure_angular_uncertainty.txt'
                file1=open(filename, 'w+')
                file1.write('#Angle rotation vector in pixels\n')
                for i in range(1, len(angle_rot_vec)):
                    file1.write(str(angle_rot_vec[i])+'\t')
                file1.write('\n#Size vector in pixels\n')
                for i in range(1, len(size_vec)):
                    file1.write(str(size_vec[i])+'\t')
                file1.write('\n#Relative uncertainty of the angular rotation estimation\n')
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

            angle_rot_vec=np.arange(n_angle)/res_split
            
            frame_size_vec=np.arange(frame_size_range[0],frame_size_range[1],step=frame_size_step)

            result_vec_angle=np.zeros([n_angle,len(frame_size_vec)])
            
            for i_angle in range(n_angle):
                for j_frame_size in range(len(frame_size_vec)):
                    if relative_structure_size is not None:
                        structure_size=frame_size_vec[j_frame_size] * relative_structure_size
                    
                    generate_displaced_gaussian(displacement=[0,0], #[0,pol_disp_vec[i_pol]], 
                                                r0=[frame_size_vec[j_frame_size]/2,
                                                    frame_size_vec[j_frame_size]/2],
                                                frame_size=[frame_size_vec[j_frame_size],
                                                            frame_size_vec[j_frame_size]],
                                                angle_per_frame=angle_rot_vec[i_angle], 
                                                size=[structure_size,structure_size*2], 
                                                size_velocity=[0,0],
                                                sampling_time=2.5e-6,
                                                output_name='gaussian', 
                                                n_frames=3)
                    try:
                    #if True:
                        result=calculate_nstx_gpi_angular_velocity(exp_id=0, 
                                                                   data_object='gaussian', 
                                                                   normalize_for_velocity=False,
                                                                   plot=False, 
                                                                   nocalc=False, 
                                                                   return_results=True,
                                                                   subtraction_order_for_velocity=subtraction_order)
    
                        result_vec_angle[i_angle,j_frame_size]=result['Angle difference FLAP']
                        print('Succeeded FS:'+str(frame_size_vec[j_frame_size])+' A:'+str(angle_rot_vec[i_angle]))
                    except:
                        #print('Failed FS:'+str(frame_size_vec[j_frame_size])+' D:'+str(angle_rot_vec[i_angle]))
                        result_vec_angle[i_angle,j_frame_size]=np.nan

            pickle.dump([result_vec_angle, angle_rot_vec, frame_size_vec], open(pickle_file, 'wb'))
        else:
            result_vec_angle, angle_rot_vec, frame_size_vec=pickle.load(open(pickle_file, 'rb'))        

        if pdf:
            pdf_pages=PdfPages(wd+'/plots/synthetic_frame_size_dependence_gauss_'+interpolation+'.pdf')
        if plot:
            plt.figure()
            plt.contourf(angle_rot_vec[1:], 
                         frame_size_vec[1:], 
                         result_vec_angle[1:,1:].transpose()/(angle_rot_vec[None,1:])-1,
                         levels=51,
                         cmap='jet')
            
            plt.colorbar()
            plt.xlabel('Angle rotation [deg]')
            plt.ylabel('Frame size')
            plt.title('Inaccuracy of rotation estimation')
            if pdf:
                pdf_pages.savefig()
            
            plt.figure()
            for i in range(len(frame_size_vec)):
                plt.plot(angle_rot_vec[1:], 
                         result_vec_angle[1:,i].transpose()/(angle_rot_vec[1:])-1,
                         label=str(frame_size_vec[i]))
            plt.xlabel('Angle rotation [deg]')
            plt.ylabel('Relative inaccuracy')
            plt.title('Inaccuracy of rotation estimation')
            plt.legend()
            if pdf:
                pdf_pages.savefig()
            
            plt.figure()
            plt.contourf(angle_rot_vec[1:], 
                         frame_size_vec[1:], 
                         result_vec_angle[1:,1:].transpose()-angle_rot_vec[None,1:],
                         levels=51,
                         cmap='jet')
            
            plt.colorbar()
            plt.xlabel('Angle rotation [deg]')
            plt.ylabel('Frame size [pix]')
            plt.title('Estimated rotation angle error vs. parameters')
            if pdf:
                pdf_pages.savefig()
            
            plt.figure()
            for i in range(len(frame_size_vec)):
                plt.plot(angle_rot_vec[1:], 
                         result_vec_angle[1:,i]-angle_rot_vec[1:],
                         label=str(frame_size_vec[i]))
            plt.xlabel('Angle rotation [deg]')
            plt.ylabel('Estimated rotation angle error [deg]')
            plt.title('Set vs. estimated rotation angle')
            plt.legend()
            
            if pdf:
                pdf_pages.savefig()
                pdf_pages.close()
            
            if save_data_into_txt:
                data=result_vec_angle[1:,1:]
                filename=wd+'/processed_data/figure_5a.txt'
                file1=open(filename, 'w+')
                file1.write('#Angle rotation vector in pixels\n')
                for i in range(1, len(angle_rot_vec[1:])):
                    file1.write(str(angle_rot_vec[i])+'\t')
                file1.write('\n#Size vector in pixels\n')
                for i in range(1, len(size_vec)):
                    file1.write(str(size_vec[i])+'\t')
                file1.write('\n#Relative uncertainty of the anglular rotation estimation\n')
                for i in range(1,len(data[0,:])):
                    string=''
                    for j in range(1,len(data[:,0])):
                        string+=str(data[j,i])+'\t'
                    string+='\n'
                    file1.write(string)
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
        result=calculate_nstx_gpi_angular_velocity(exp_id=0, 
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

        
    if plot_example_event:
        calculate_nstx_gpi_angular_velocity(exp_id=141319,
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