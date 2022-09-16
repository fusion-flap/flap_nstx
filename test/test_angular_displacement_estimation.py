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

from flap_nstx.gpi import generate_displaced_gaussian
from flap_nstx.gpi import analyze_gpi_structures, calculate_nstx_gpi_angular_velocity
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


def test_angular_displacement_estimation(size_angle=False,                        #Test the inaccuracy with Gaussian-structures
                                         scaling_factor_angle=False,
                                         frame_size_angle=False,
                                         elongation_angle=False,

                                         method='ccf',                          #ccf, contour or watershed
                                         angle_method='angle',                  #angle or ALI (angle of least incidence)

                                         scaling_factor_analysis=False,

                                         calc_flap=True,
                                         warp_polar_order=1,
                                         subtraction_order=2,
                                         zero_padding_scale=1,
                                         calculate_half_fft=False,
                                         interpolation='parabola',              #Parabola or bicubic
                                         gaussian_blur=False,
                                         hann_window=False,

                                         #Gaussian analysis settings
                                         n_angle=90,                            #Used for both Gaussian,frame and random, number of poloidal displacements
                                         angle_range=[-90.,90.],

                                         n_scale=40,
                                         scale_range=[0.01,0.2],

                                         n_elongation=10,
                                         elongation_range=[0.1,1.0],

                                         n_size=12,
                                         size_range=[4,40],

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

                                         publication=False,                     #Plot in publication quality

                                         plot_sample_gaussian=False,
                                         plot_sample_random=False,

                                         plot_example_event=False,
                                         plot_example_event_frames=False,

                                         save_data_into_txt=False,
                                         save_data_filename=None,
                                         return_results=False,
                                         ):

    wd=flap.config.get_all_section('Module NSTX_GPI')['Working directory']
    if pdf and not plot:
        import matplotlib
        matplotlib.use('agg')
    else:
        import matplotlib
        matplotlib.use('qt5agg')
    if size_angle:
        comment_str='_size_angle'
    elif frame_size_angle:
        comment_str='_frame_size_angle'
    elif scaling_factor_angle:
        comment_str='_scaling_factor_angle'
    elif elongation_angle:
        comment_str='_elongation_angle'

    general_filename_appendix='angular_test'+comment_str+'_'+interpolation+'_'+method+'_'+angle_method
    pickle_filename=wd+'/processed_data/'+general_filename_appendix+'.pickle'
    pdf_filename=wd+'/plots/'+general_filename_appendix+'.pdf'

    if save_data_filename is None:
        text_filename=wd+'/processed_data/'+general_filename_appendix+'_data_accessibility.txt'
    else:
        text_filename=save_data_filename


    angle_rot_vec=np.arange(n_angle)/(n_angle-1)*(angle_range[1]-angle_range[0])+angle_range[0]
    elongation_vec=np.arange(n_elongation)/(n_elongation-1)*(elongation_range[1]-elongation_range[0])+elongation_range[0]
    scaling_factor_vec=np.arange(n_scale)/(n_scale-1)*(scale_range[1]-scale_range[0])+scale_range[0]
    size_vec=np.arange(n_size)/(n_size-1)*(size_range[1]-size_range[0])+size_range[0]
    frame_size_vec=np.arange(frame_size_range[0],frame_size_range[1],step=frame_size_step)

    sampling_time=2.5e-6
    """
    ANALYSIS OF THE GAUSSIAN DISPLACED STRUCTURES
    """

    if size_angle:
        if not nocalc:
            result_vec_angle=np.zeros([n_angle,n_size])
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
                    if method == 'ccf':
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
                    else:
                        result=analyze_gpi_structures(data_object='gaussian',
                                                      nocalc=nocalc,
                                                      str_finding_method=method,
                                                      structure_pixel_calc=False,
                                                      test_structures=False,
                                                      return_results=True,
                                                      plot=False,
                                                      pdf=False,
                                                      ellipse_method='linalg',
                                                      fit_shape='ellipse',
                                                      prev_str_weighting='max_intensity')

                        if angle_method == 'angle':
                            try:
                                result_vec_angle[i_angle,j_size]=(result['data']['Angle']['max'][1]-result['data']['Angle']['max'][0])/np.pi*180
                            except:
                                print(result['data']['Angle']['max'])
                                raise ValueError('Wrong angle')

                        elif angle_method == 'ALI':
                            result_vec_angle[i_angle,j_size]=(result['data']['Angle of least inertia']['max'][1]-
                                                              result['data']['Angle of least inertia']['max'][0])/np.pi*180.
                    if result_vec_angle[i_angle,j_size] > 90:
                        result_vec_angle[i_angle,j_size]-=180

                    if result_vec_angle[i_angle,j_size] < -90:
                        result_vec_angle[i_angle,j_size]+=180

                finish_time=time.time()
                rem_time=(finish_time-start_time)*(n_angle-i_angle)
                print('Remaining time from the calculation: '+str(int(np.mod(rem_time,60)))+'min.')
            if plot:
                plt.figure()
                plt.contourf(angle_rot_vec, size_vec, result_vec_angle.transpose(),)

            pickle.dump([result_vec_angle, angle_rot_vec, size_vec], open(pickle_filename, 'wb'))
        else:
            result_vec_angle, angle_rot_vec, size_vec = pickle.load(open(pickle_filename, 'rb'))

        if plot or pdf:
            if pdf:
                pdf_pages=PdfPages(pdf_filename)

            plt.figure()
            plt.contourf(angle_rot_vec[:],
                         size_vec[:],
                         result_vec_angle[:,:].transpose()-angle_rot_vec[None,:],
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
                plt.plot(angle_rot_vec, result_vec_angle[:,i]-angle_rot_vec, label=str(size_vec[i]))
            plt.xlabel('Angle rotation [deg]')
            plt.ylabel('Inaccuracy [pix]')
            plt.title('Inaccuracy of angle estimation')
            plt.legend()

            if pdf:
                pdf_pages.savefig()
                pdf_pages.close()

            if save_data_into_txt:
                data=result_vec_angle[1:,1:]

                file1=open(text_filename, 'w+')
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

        if return_results:
            return {'angle_rot_vec':angle_rot_vec,
                    'result_vec_angle':result_vec_angle,
                    'size_vec':size_vec,
                    'data':data}

    if scaling_factor_angle:
        if not nocalc:
            result_vec_angle=np.zeros([n_angle,n_scale])
            result_vec_scale=np.zeros([n_angle,n_scale])
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
                    if method == 'ccf':
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
                    else:
                        result=analyze_gpi_structures(data_object='gaussian',
                                                      nocalc=nocalc,
                                                      str_finding_method=method,
                                                      structure_pixel_calc=True,
                                                      test_structures=True,
                                                      return_results=True,
                                                      plot=False,
                                                      pdf=False,
                                                      ellipse_method='linalg',
                                                      fit_shape='ellipse',
                                                      prev_str_weighting='max_intensity')
                        if angle_method == 'angle':
                            result_vec_angle[i_angle,j_scale]=(result['data']['Angle']['max'][1]-result['data']['Angle']['max'][0])/np.pi*180
                        elif angle_method == 'ALI':
                            result_vec_angle[i_angle,j_scale]=result['data']['Angle of least inertia']['max']

                    if calc_flap and method =='ccf' :
                        print(scaling_factor_vec[j_scale],
                              result['Expansion velocity ccf FLAP'],
                              scaling_factor_vec[j_scale]/result['Expansion velocity ccf FLAP'])
                        print(angle_rot_vec[i_angle],result['Angle difference FLAP log'],
                              angle_rot_vec[i_angle]/result['Angle difference FLAP log'])

                        result_vec_scale[i_angle,j_scale]=result['Expansion velocity ccf FLAP']
                        result_vec_angle[i_angle,j_scale]=result['Angle difference FLAP log']
                    elif method == 'ccf':
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

            pickle.dump([result_vec_scale, result_vec_angle, angle_rot_vec, scaling_factor_vec], open(pickle_filename, 'wb'))
        else:
            result_vec_scale, result_vec_angle, angle_rot_vec, scaling_factor_vec = pickle.load(open(pickle_filename, 'rb'))

        if plot or pdf:
            if pdf:
                pdf_pages=PdfPages(pdf_filename)

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

                file1=open(text_filename, 'w+')
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

    if return_results:
        return {'angle_rot_vec':angle_rot_vec,
                'result_vec_angle':result_vec_angle,
                'scaling_factor_vec':scaling_factor_vec,
                'result_vec_scale':result_vec_scale}
    """
    **************************************************
    ANALYSIS OF THE FRAME SIZE DISPLACEMENT ESTIMATION
    **************************************************
    """

    if frame_size_angle:
        if not nocalc:
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


                    #try:
                    if True:
                        if method == 'ccf':
                            result=calculate_nstx_gpi_angular_velocity(exp_id=0,
                                                                       data_object='gaussian',
                                                                       normalize_for_velocity=False,
                                                                       plot=False,
                                                                       nocalc=False,
                                                                       return_results=True,
                                                                       subtraction_order_for_velocity=subtraction_order)
                            result_vec_angle[i_angle,j_frame_size]=result['Angle difference FLAP']
                        else:
                            result=analyze_gpi_structures(data_object='gaussian',
                                                          nocalc=nocalc,
                                                          subtraction_order=subtraction_order,
                                                          str_finding_method=method,
                                                          test_structures=False,
                                                          return_results=True,
                                                          plot=False,
                                                          pdf=False,
                                                          structure_pixel_calc=False,
                                                          ellipse_method='linalg',
                                                          fit_shape='ellipse',
                                                          prev_str_weighting='max_intensity')
                            if angle_method == 'angle':
                                result_vec_angle[i_angle,j_frame_size]=(result['data']['Angle']['max'][1]-
                                                                        result['data']['Angle']['max'][0])/np.pi*180

                            elif angle_method == 'ALI':
                                result_vec_angle[i_angle,j_frame_size]=(result['data']['Angle of least inertia']['max'][1]-
                                                                        result['data']['Angle of least inertia']['max'][0])/np.pi*180


                        print('Succeeded FS:'+str(frame_size_vec[j_frame_size])+' A:'+str(angle_rot_vec[i_angle]))
                    #except:
                    #    #print('Failed FS:'+str(frame_size_vec[j_frame_size])+' D:'+str(angle_rot_vec[i_angle]))
                    #    result_vec_angle[i_angle,j_frame_size]=np.nan

            pickle.dump([result_vec_angle, angle_rot_vec, frame_size_vec], open(pickle_filename, 'wb'))
        else:
            result_vec_angle, angle_rot_vec, frame_size_vec=pickle.load(open(pickle_filename, 'rb'))


        if plot or pdf:
            if pdf:
                pdf_pages=PdfPages(pdf_filename)
            plt.figure()
            plt.contourf(angle_rot_vec,
                         frame_size_vec,
                         result_vec_angle.transpose()/(angle_rot_vec[None,:])-1,
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
            plt.contourf(angle_rot_vec,
                         frame_size_vec,
                         result_vec_angle.transpose()-angle_rot_vec[None,:],
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
                plt.plot(angle_rot_vec,
                         result_vec_angle[:,i]-angle_rot_vec,
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

                file1=open(text_filename, 'w+')
                file1.write('#Angle rotation vector in pixels\n')
                for i in range(len(angle_rot_vec[1:])):
                    file1.write(str(angle_rot_vec[i])+'\t')
                file1.write('\n#Size vector in pixels\n')
                for i in range(len(size_vec)):
                    file1.write(str(size_vec[i])+'\t')
                file1.write('\n#Relative uncertainty of the anglular rotation estimation\n')
                for i in range(len(data[0,:])):
                    string=''
                    for j in range(len(data[:,0])):
                        string+=str(data[j,i])+'\t'
                    string+='\n'
                    file1.write(string)
                file1.close()
        if return_results:
            return {'angle_rot_vec':angle_rot_vec,
                    'frame_size_vec':frame_size_vec,
                    'result_vec_angle':result_vec_angle}
    """
    **************************************************
    ANALYSIS OF THE ELONGATION VS ANGLE
    **************************************************
    """
    if elongation_angle:
        if not nocalc:
            frame_size=80
            result_vec_angle=np.zeros([n_angle,n_elongation])

            for i_angle in range(n_angle):
                for j_elong in range(len(elongation_vec)):
                    if relative_structure_size is not None:
                        structure_size=frame_size * relative_structure_size

                    generate_displaced_gaussian(displacement=[0,0], #[0,pol_disp_vec[i_pol]],
                                                r0=[32,32],
                                                frame_size=[frame_size,frame_size],
                                                angle_per_frame=angle_rot_vec[i_angle],
                                                size=[structure_size,
                                                      structure_size*elongation_vec[j_elong]],
                                                size_velocity=[0,0],
                                                sampling_time=2.5e-6,
                                                output_name='gaussian',
                                                n_frames=3)
                    #try:
                    if True:
                        if method == 'ccf':
                            result=calculate_nstx_gpi_angular_velocity(exp_id=0,
                                                                       data_object='gaussian',
                                                                       normalize_for_velocity=False,
                                                                       plot=False,
                                                                       nocalc=False,
                                                                       return_results=True,
                                                                       subtraction_order_for_velocity=subtraction_order)
                            result_vec_angle[i_angle,j_elong]=result['Angle difference FLAP']
                        else:
                            result=analyze_gpi_structures(data_object='gaussian',
                                                          nocalc=nocalc,
                                                          subtraction_order=subtraction_order,
                                                          str_finding_method=method,
                                                          test_structures=False,
                                                          return_results=True,
                                                          plot=False,
                                                          pdf=False,
                                                          structure_pixel_calc=False,
                                                          ellipse_method='linalg',
                                                          fit_shape='ellipse',
                                                          prev_str_weighting='max_intensity')
                            if angle_method == 'angle':
                                result_vec_angle[i_angle,j_elong]=(result['data']['Angle']['max'][1]-result['data']['Angle']['max'][0])/np.pi*180

                            elif angle_method == 'ALI':
                                result_vec_angle[i_angle,j_elong]=(result['data']['Angle of least inertia']['max'][1]-
                                                                   result['data']['Angle of least inertia']['max'][0])/np.pi*180


                        print('Succeeded FS:'+str(elongation_vec[j_elong])+' A:'+str(angle_rot_vec[i_angle]))
                    #except:
                    #    #print('Failed FS:'+str(frame_size_vec[j_frame_size])+' D:'+str(angle_rot_vec[i_angle]))
                    #    result_vec_angle[i_angle,j_frame_size]=np.nan

            pickle.dump([result_vec_angle, angle_rot_vec, elongation_vec], open(pickle_filename, 'wb'))
        else:
            result_vec_angle, angle_rot_vec, elongation_vec=pickle.load(open(pickle_filename, 'rb'))


        if plot or pdf:
            if pdf:
                pdf_pages=PdfPages(pdf_filename)
            plt.figure()
            plt.contourf(angle_rot_vec,
                         elongation_vec,
                         result_vec_angle.transpose()-angle_rot_vec[None,:],
                         levels=51,
                         cmap='jet')

            plt.colorbar()
            plt.xlabel('Angle rotation [deg]')
            plt.ylabel('Elongation')
            plt.title('Inaccuracy of rotation estimation')
            if pdf:
                pdf_pages.savefig()

            plt.figure()
            for i in range(len(elongation_vec)):
                plt.plot(angle_rot_vec[:],
                         result_vec_angle[:,i].transpose()-angle_rot_vec[:],
                         label=str(elongation_vec[i]))
            plt.xlabel('Angle rotation [deg]')
            plt.ylabel('Relative inaccuracy')
            plt.title('Inaccuracy of rotation estimation')
            plt.legend()

            if pdf:
                pdf_pages.savefig()
                pdf_pages.close()

            if save_data_into_txt:
                data=result_vec_angle

                file1=open(text_filename, 'w+')
                file1.write('#Angle rotation vector in pixels\n')
                for i in range(0, len(angle_rot_vec[1:])):
                    file1.write(str(angle_rot_vec[i])+'\t')
                file1.write('\n#Elongation vector\n')
                for i in range(0, len(elongation_vec)):
                    file1.write(str(elongation_vec[i])+'\t')
                file1.write('\n#Relative uncertainty of the anglular rotation estimation\n')
                for i in range(0,len(data[0,:])):
                    string=''
                    for j in range(1,len(data[:,0])):
                        string+=str(data[j,i])+'\t'
                    string+='\n'
                    file1.write(string)
                file1.close()
        if return_results:
            return {'angle_rot_vec':angle_rot_vec,
                    'elongation_vec':elongation_vec,
                    'result_vec_angle':result_vec_angle}

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

    if pdf and not plot:
        matplotlib.use('qt5agg')