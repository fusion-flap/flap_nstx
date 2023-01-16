#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 19:22:30 2022

@author: mlampert
"""

class Hell(Exception):pass

import os
import copy


import flap
import flap_nstx

thisdir = os.path.dirname(os.path.realpath(__file__))
fn = os.path.join(thisdir,"../flap_nstx.cfg")
flap.config.read(file_name=fn)
flap_nstx.register()

from flap_nstx.gpi import calculate_nstx_gpi_angular_velocity, show_nstx_gpi_video_frames
from flap_nstx.test import test_angular_displacement_estimation

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MaxNLocator

import numpy as np
from skimage.filters import window, difference_of_gaussians

wd=flap.config.get_all_section('Module NSTX_GPI')['Working directory']
fig_dir='/publication_figures/rsi_2022'

def plot_results_for_rsi_2022_paper(plot_figure=2,
                                    save_data_into_txt=False,
                                    plot_all=False,
                                    nocalc=False):

    if plot_all:
        for i in range(15):
            plot_results_for_rsi_2022_paper(plot_figure=i,
                                            save_data_into_txt=save_data_into_txt)
        return


    #Settings for fig 2,3,4,13,14

    exp_id_elm=141319
    time_range_elm=[0.552,0.553]
    elm_time=0.552500
    sample_to_plot_elm=[20874,20875]
    #Settings for fig 9,10 (blobs)
    exp_id_blob=141307
    time_range_blob=[0.484198-500e-6,
                     0.484198+500e-6]

    sigma_low=1
    sigma_high=None
    correlation_threshold=0.7

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

    """
    Structure rotation sketch
    """
    if plot_figure == 1:
        filename='fig1_example_rotation'
        pdf=PdfPages(wd+fig_dir+'/'+filename+'.pdf')
        d_raw=flap.get_data('NSTX_GPI',
                      exp_id=exp_id_elm,
                      name='',
                      object_name='GPI')
        fig,axs=plt.subplots(1,2,
                             figsize=(8.5/2.54,
                                      8.5/2.54/np.sqrt(2)))
        frame1=d_raw.slice_data(slicing={'Sample':sample_to_plot_elm[0]}).data
        frame2=d_raw.slice_data(slicing={'Sample':sample_to_plot_elm[1]}).data

        ax=axs[0]
        ax.contourf(np.arange(frame1.shape[0]),
                    np.arange(frame1.shape[1]),
                    frame1.T,
                    levels=51)
        ax.set_title('Raw frame #1')
        ax.set_xlabel('x [pix]')
        ax.set_ylabel('y [pix]')
        ax.set_aspect('equal')

        ax=axs[1]
        ax.contourf(np.arange(frame2.shape[0]),
                    np.arange(frame2.shape[1]),
                    frame2.T,
                    levels=51)
        ax.set_title('Raw frame #2')
        ax.set_xlabel('x [pix]')
        ax.set_ylabel('y [pix]')
        ax.set_aspect('equal')
        plt.tight_layout(pad=0.1)

        pdf.savefig()
        pdf.close()

        if save_data_into_txt:
            filename=wd+fig_dir+'/data_accessibility/'+filename+'.txt'
            file1=open(filename, 'w+')
            file1.write('\nFrame #1\n\n')
            for i in range(len(frame1[0,:])):
                string=''
                for j in range(len(frame1[:,0])):
                    string+=str(frame1[j,i])+'\t'
                string+='\n'
                file1.write(string)

            file1.write('\nFrame #2\n\n')
            for i in range(len(frame2[0,:])):
                string=''
                for j in range(len(frame2[:,0])):
                    string+=str(frame2[j,i])+'\t'
                string+='\n'
                file1.write(string)

            file1.close()

    if plot_figure == 2:
        print('This is the flowchart figure, no need to plot.')

    """
    PRE-PROCESSING FIGURE
    """
    if plot_figure == 3:
        filename='fig3_pre_processing'

        d_raw=flap.get_data('NSTX_GPI',
                      exp_id=exp_id_elm,
                      name='',
                      object_name='GPI')

        flap.slice_data('GPI', slicing={'Time':flap.Intervals(time_range_elm[0],time_range_elm[1])}, output_name='GPI_SLICED_FULL')

        slicing={'Time':flap.Intervals(time_range_elm[0],
                                       time_range_elm[1])}
        slicing_for_filtering=copy.deepcopy(slicing)
        slicing_for_filtering['Time']=flap.Intervals(time_range_elm[0]-1/1e3*10,
                                                     time_range_elm[1]+1/1e3*10)
        flap.slice_data('GPI',
                        exp_id=exp_id_elm,
                        slicing=slicing_for_filtering,
                        output_name='GPI_SLICED_FOR_FILTERING')

        slicing_time_only={'Time':flap.Intervals(time_range_elm[0],
                                                 time_range_elm[1])}

        normalized_data=flap_nstx.gpi.normalize_gpi('GPI_SLICED_FOR_FILTERING',
                                                    exp_id=exp_id_elm,
                                                    slicing_time=slicing_time_only,
                                                    normalize='roundtrip',
                                                    normalize_f_high=1e3,
                                                    normalizer_object_name='GPI_LPF_INTERVAL',
                                                    output_name='GPI_NORMALIZED',
                                                    return_object='data divide')

        detrended=flap_nstx.tools.detrend_multidim('GPI_NORMALIZED',
                                                   exp_id=exp_id_elm,
                                                   order=1,
                                                   coordinates=['Image x', 'Image y'],
                                                   output_name='GPI_DETREND_VEL')

        d=copy.deepcopy(flap.get_data_object('GPI_NORMALIZED'))

        d.data=d.data-detrended.data
        flap.add_data_object(d,'GPI_TREND')

        signals=['GPI_SLICED_FULL',
                 'GPI_NORMALIZED',
                 'GPI_DETREND_VEL',
                 'GPI_GAUSS_BLUR'
                 ]
        labels=['Fig. 3 (a)',
                'Fig. 3 (b)',
                'Fig. 3 (c)',
                'Fig. 3 (d)']

        titles=['Raw frame',
                'Normalized',
                'Trend subtract',
                'Gaussian blur',
                ]

        pdf=PdfPages(wd+fig_dir+'/'+filename+'.pdf')
        filename=wd+fig_dir+'/data_accessibility/'+filename+'.txt'
        file1=open(filename, 'w+')
        data_gauss_blur=copy.deepcopy(d)

        data_gauss_blur=d_raw.slice_data(slicing={'Sample':sample_to_plot_elm[0]})
        data_gauss_blur.data = difference_of_gaussians(data_gauss_blur.data, sigma_low, sigma_high)

        flap.add_data_object(data_gauss_blur,'GPI_GAUSS_BLUR')
        plt.figure()
        fig,axs=plt.subplots(2,2,
                             figsize=(8.5/2.54,
                                      8.5/2.54*80./64.))

        for index_grid_x in range(2):
            for index_grid_y in range(2):
                ax=axs[index_grid_x,index_grid_y]
                ind=index_grid_x*2+index_grid_y

                data=flap.get_data_object(signals[ind]).slice_data(slicing={'Sample':sample_to_plot_elm[0]}).data

                ax.contourf(np.arange(0,64),
                            np.arange(0,80),
                            data.T)
                ax.set_title(titles[ind])
                ax.set_xlabel('x [pix]')
                ax.set_ylabel('y [pix]')
                ax.set_aspect('equal')

                if save_data_into_txt:
                    file1.write('\n#'+signals[ind]+' data ('+labels[ind]+')\n\n')
                    for i in range(len(data[0,:])):
                        string=''
                        for j in range(len(data[:,0])):
                            string+=str(data[j,i])+'\t'
                        string+='\n'
                        file1.write(string)

        plt.tight_layout(pad=0.1)
        pdf.savefig()
        pdf.close()
        file1.close()


    """
    EXAMPLE LOG-POLAR TRANSFORMATION
    """
    if plot_figure == 4:
        filename='fig4_logpolar_transform'

        result=calculate_nstx_gpi_angular_velocity(exp_id=exp_id_elm,
                                                   time_range=[time_range_elm[0],
                                                               time_range_elm[1]],
                                                   normalize='roundtrip',
                                                   normalize_for_velocity=True,
                                                   plot=False,
                                                   pdf=True,
                                                   pdf_filename=wd+fig_dir+'/'+filename+'.pdf',
                                                   nocalc=nocalc,
                                                   plot_scatter=False,
                                                   plot_for_publication=True,
                                                   gaussian_blur=True,
                                                   calculate_half_fft=False,
                                                   test_into_pdf=True,
                                                   return_results=True,
                                                   plot_sample_frames=False,
                                                   sigma_low=sigma_low,sigma_high=sigma_high,
                                                   subtraction_order_for_velocity=2,
                                                   sample_to_plot=sample_to_plot_elm,

                                                   save_data_for_publication=True,
                                                   data_filename=wd+fig_dir+'/data_accessibility/'+filename+'.txt')

    """
    EXAMPLE CCCF FUNCTION
    """
    if plot_figure == 5:
        filename='fig5_example_cccf'
        result=calculate_nstx_gpi_angular_velocity(exp_id=exp_id_elm,
                                                   time_range=[time_range_elm[0],
                                                               time_range_elm[1]],
                                                   normalize='roundtrip',
                                                   normalize_for_velocity=True,
                                                   plot=False,
                                                   pdf=True,
                                                   pdf_filename=wd+fig_dir+'/'+filename+'.pdf',
                                                   nocalc=nocalc,
                                                   plot_scatter=False,
                                                   plot_for_publication=False,
                                                   gaussian_blur=True,
                                                   calculate_half_fft=False,
                                                   test_into_pdf=False,
                                                   return_results=True,
                                                   plot_ccf=True,
                                                   sigma_low=sigma_low,sigma_high=sigma_high,
                                                   subtraction_order_for_velocity=2,
                                                   sample_to_plot=[sample_to_plot_elm[1]],
                                                   save_data_for_publication=True,
                                                   data_filename=wd+fig_dir+'/data_accessibility/'+filename+'.txt')

    if plot_figure == 6:
        filename='fig6_correlation_threshold'
        #calculate_nstx_gpi_avg_frame_velocity(exp_id=141319,
        #                              time_range=[0.552,0.553],
        results=calculate_nstx_gpi_angular_velocity(exp_id=exp_id_elm,
                                                             time_range=time_range_elm,
                                                             normalize='roundtrip',
                                                             normalize_for_velocity=True,
                                                             plot=False,
                                                             pdf=False,
                                                             pdf_filename=wd+fig_dir+'/'+filename+'.pdf',
                                                             nocalc=nocalc,
                                                             plot_scatter=False,
                                                             plot_for_publication=False,
                                                             correlation_threshold=0.0,
                                                             return_results=True,
                                                             subtraction_order_for_velocity=2,
                                                             gaussian_blur=True,
                                                             sigma_low=sigma_low,sigma_high=sigma_high,
                                                             )

        corr_thres=np.arange(11)/10
        for i_corr in range(11):
            results=calculate_nstx_gpi_angular_velocity(exp_id=exp_id_elm,
                                                        time_range=time_range_elm,
                                                        normalize='roundtrip',
                                                        normalize_for_velocity=True,
                                                        plot=False,
                                                        pdf=False,
                                                        nocalc=True,
                                                        plot_scatter=False,
                                                        plot_for_publication=False,
                                                        correlation_threshold=corr_thres[i_corr],
                                                        return_results=True,
                                                        subtraction_order_for_velocity=2,
                                                        gaussian_blur=True,
                                                        sigma_low=sigma_low,sigma_high=sigma_high,
                                                        )

            time=results['Time']
            if i_corr==0:
                ang_vel=np.zeros([len(results['Angular velocity ccf FLAP log']),11])
            ang_vel[:,i_corr]=results['Angular velocity ccf FLAP log']



        pdf=PdfPages(wd+fig_dir+'/'+filename+'.pdf')

        plt.figure()
        fig,ax=plt.subplots(figsize=(8.5/2.54,2))
        for i in range(0,7,1):
            plt.plot(time, ang_vel[:,(10-i)]/1e3+i*100, label=str((10-i)/10))

        plt.title('Angular velocity vs. \n correlation threshold')
        plt.xlabel('Time [s]')
        plt.ylabel('$\omega[krad/s]$')
        plt.legend()
        plt.tight_layout(pad=0.1)
        pdf.savefig()
        pdf.close()

        if save_data_into_txt:
            filename=wd+fig_dir+'/data_accessibility/'+filename+'.txt'
            file1=open(filename, 'w+')
            string='Time [s]'
            for i in range(1,7,1):
                string+='\t rho='+str((10-i)/10)
            string+='\n'
            file1.write(string)
            for j in range(len(time)):
                string=str(time[j])
                for i in range(1,7,1):
                    string+='\t'+str(ang_vel[j,(10-i)]/1e3-i*10)
                string+='\n'
                file1.write(string)
            file1.close()

    """
    ROTATING GAUSSIAN STRUCTURE
    """
    if plot_figure == 7:
        filename='fig7_rotating_gaussian'

        test_angular_displacement_estimation(plot_sample_gaussian=True,
                                             pdf=True,
                                             pdf_filename=wd+fig_dir+'/'+filename+'.pdf',
                                             save_data_into_txt=save_data_into_txt,
                                             save_data_filename=wd+fig_dir+'/data_accessibility/'+filename+'.txt')

    """
    TESTING THE METHOD EXPANSION FRACTION ESTIMATION
    """
    if plot_figure == 8:
        filename='fig8_rot_estimation_frame_size_vs_angle'

        test_angular_displacement_estimation(frame_size_angle=True,

                                             method='ccf',
                                             angle_method='angle',

                                             n_angle=35,
                                             angle_range=[-85,85],

                                             n_size=9,
                                             size_range=[6,33],

                                             n_scale=10,
                                             scale_range=[0.01,0.19],

                                             n_elongation=10,
                                             elongation_range=[0.1,1.],

                                             frame_size_range=[16,200],
                                             frame_size_step=16,

                                             nocalc=nocalc,
                                             pdf=True,
                                             pdf_filename=wd+fig_dir+'/'+filename+'.pdf',
                                             plot=False,
                                             sigma_low=sigma_low,sigma_high=sigma_high,

                                             save_data_into_txt=save_data_into_txt,
                                             save_data_filename=wd+fig_dir+'/data_accessibility/'+filename+'.txt'
                                             )

    """
    TESTING THE METHOD ANGULAR DISPLACEMENT ESTIMATION
    """
    if plot_figure == 9:
        filename='fig9_rot_estimation_angle_vs_elongation'
        test_angular_displacement_estimation(elongation_angle=True,

                                              method='ccf',
                                              angle_method='angle',

                                              n_angle=38,
                                              angle_range=[-85,85],

                                              n_size=9,
                                              size_range=[6,33],

                                              n_scale=10,
                                              scale_range=[0.01,0.19],

                                              n_elongation=21,
                                              elongation_range=[1.,3.],

                                              frame_size_range=[8,200],
                                              frame_size_step=8,

                                              nocalc=nocalc,
                                              pdf=True,
                                              pdf_filename=wd+fig_dir+'/'+filename+'.pdf',
                                              plot=False,
                                              sigma_low=sigma_low,sigma_high=sigma_high,
                                              save_data_into_txt=save_data_into_txt,
                                              save_data_filename=wd+fig_dir+'/data_accessibility/'+filename+'.txt'
                                              )

    if plot_figure == 10:
        filename='fig10_rot_estimation_angle_vs_noise'
        n_noise=20
        n_angle=38
        results=test_angular_displacement_estimation(noise_angle=True,

                                                     method='ccf',
                                                     angle_method='angle',

                                                     n_angle=n_angle,
                                                     angle_range=[-85.,85.],

                                                     n_noise=n_noise,
                                                     noise_range=[0.0,2.0],
                                                     noise_repeat=25,

                                                     nocalc=nocalc,
                                                     pdf=False,
                                                     plot=False,
                                                     sigma_low=sigma_low,
                                                     sigma_high=sigma_high,
                                                     save_data_into_txt=False,
                                                     return_results=True
                                                     )
        angle_rot_vec=results['angle_rot_vec']
        noise_vec=results['noise_vec']
        result_vec_angle=results['result_vec_angle']

        pdf_pages=PdfPages(wd+fig_dir+'/'+filename+'.pdf',)

        fig, axs = plt.subplots(2,1,figsize=(8.5/2.54,8.5/2.54*2))
        ax=axs[0]
        fig1=ax.contourf(angle_rot_vec[:],
                         noise_vec[:],
                         np.abs(np.mean(result_vec_angle,axis=2).transpose()/angle_rot_vec[None,:]-1),
                         levels=51,
                         cmap='jet')
        plt.colorbar(fig1, ax=ax)
        ax.set_xlabel('Angle of rotation [deg]')
        ax.set_ylabel('Relative noise level')
        ax.set_title('Average inaccuracy of \n noisy angle rotation estimation')
        ax.text(-0.24, 1.07, '(a)', transform=ax.transAxes, size=9)

        ax=axs[1]
        fig2=ax.contourf(angle_rot_vec[:],
                         noise_vec[:],
                         np.sqrt(np.var(result_vec_angle/angle_rot_vec[:,None,None]-1, axis=2)).T,
                         levels=51,
                         cmap='jet')
        plt.colorbar(fig2,ax=ax)
        ax.set_xlabel('Angle of rotation [deg]')
        ax.set_ylabel('Relative noise level')
        ax.set_title('Standard deviation of inaccuracy of \n noisy angle rotation estimation')
        ax.text(-0.24, 1.07, '(b)', transform=ax.transAxes, size=9)
        plt.tight_layout(pad=0.5)

        pdf_pages.savefig()
        pdf_pages.close()

        if save_data_into_txt:
            data=np.abs(np.mean(result_vec_angle,axis=2).transpose()/angle_rot_vec[None,:]-1)
            stddev=np.sqrt(np.var(result_vec_angle/angle_rot_vec[:,None,None]-1, axis=2)).T

            file1=open(wd+fig_dir+'/data_accessibility/'+filename+'.txt', 'w+')
            file1.write('#Angle of rotation (deg) \n')
            for i in range(0, len(angle_rot_vec)):
                file1.write(str(angle_rot_vec[i])+'\t')
            file1.write('\n#Relative noise level\n')
            for i in range(0, len(noise_vec)):
                file1.write(str(noise_vec[i])+'\t')
            file1.write('\n#Mean relative uncertainty of the angular rotation estimation\n')
            for i in range(0,len(data[0,:])):
                string=''
                for j in range(0,len(data[:,0])):
                    string+=str(data[j,i])+'\t'
                string+='\n'
                file1.write(string)
            file1.write('\n#Stddev relative uncertainty of the angular rotation estimation\n')
            for i in range(0,len(stddev[0,:])):
                string=''
                for j in range(0,len(stddev[:,0])):
                    string+=str(stddev[j,i])+'\t'
                string+='\n'
                file1.write(string)
            file1.close()

    """
    EXAMPLE ROTATING BLOBS
    """
    if plot_figure == 11:
        filename='fig11_blob_rotation_example'
        plt.figure()
        ax,fig=plt.subplots(figsize=(8.5/2.54,8.5/2.54))
        show_nstx_gpi_video_frames(exp_id=exp_id_blob,
                                   time_range=time_range_blob,
                                   start_time=0.484198-2.5e-6*8,
                                   n_frame=9,
                                   logz=False,
                                   z_range=[0,6],
                                   plot_filtered=False,
                                   normalize=True,
                                   cache_data=False,
                                   plot_flux=False,
                                   plot_separatrix=True,
                                   flux_coordinates=False,
                                   device_coordinates=True,
                                   new_plot=False,

                                   pdf=True,
                                   pdf_filename=wd+fig_dir+'/'+filename+'.pdf',

                                   colormap='gist_ncar',
                                   save_for_paraview=False,
                                   colorbar_visibility=False,
                                   save_data_for_publication=True,
                                   data_filename=wd+fig_dir+'/data_accessibility/'+filename+'.txt'
                                   )

    """
    RESULTS FOR ROTATING BLOBS
    """
    if plot_figure == 12:
        correlation_threshold=0.0
        filename='fig12_blob_rotation_estimate'

        frame_properties=calculate_nstx_gpi_angular_velocity(exp_id=exp_id_blob,
                                                             time_range=time_range_blob,
                                                             normalize='roundtrip',
                                                             normalize_for_velocity=True,
                                                             plot=False,
                                                             pdf=False,
                                                             nocalc=nocalc,
                                                             plot_scatter=False,
                                                             plot_for_publication=False,
                                                             correlation_threshold=0.0,
                                                             return_results=True,
                                                             subtraction_order_for_velocity=None,
                                                             gaussian_blur=True,
                                                             sigma_low=sigma_low,sigma_high=sigma_high,
                                                             )

        nan_ind=np.where(frame_properties['Correlation max'] < correlation_threshold)

        frame_properties['Velocity ccf FLAP'][nan_ind,0] = np.nan
        frame_properties['Velocity ccf FLAP'][nan_ind,1] = np.nan
        frame_properties['Angular velocity ccf FLAP log'][nan_ind]=np.nan
        frame_properties['Expansion velocity ccf FLAP'][nan_ind]=np.nan


        plot_index=np.logical_and(np.logical_not(np.isnan(frame_properties['Velocity ccf FLAP'][:,0])),
                                  np.logical_and(frame_properties['Time'] >= time_range_blob[0],
                                                 frame_properties['Time'] <= time_range_blob[1]))

        #Plotting the radial velocity

        pdf_filename=wd+fig_dir+'/'+filename+'.pdf'
        pdf_pages=PdfPages(pdf_filename)

        figsize=(8.5/2.54,8.5/2.54/np.sqrt(2))

        fig, ax = plt.subplots(1,1, figsize=figsize)
        #fig, axs = plt.subplots(2,1, figsize=figsize)

        """
        ANGULAR AND EXPANSION VELOCITY PLOTTING FROM FLAP
        """

        #ax=axs[0]
        ax.plot(frame_properties['Time'][plot_index]*1e3,
                frame_properties['Angular velocity ccf FLAP log'][plot_index]/1e3,
                label='Angular velocity ccf FLAP log',
                color='tab:blue')
        ax.set_title('Angular velocity for shot #141307')
        #ax.text(-0.5, 1.2, '(a)', transform=ax.transAxes, size=9)
        # ax.xaxis.set_major_locator(MaxNLocator(5))
        # ax.yaxis.set_major_locator(MaxNLocator(5))
        #ax.set_xticks(ticks=[-500,-250,0,250,500])

        ax.set_xlabel('t [ms]')
        ax.set_ylabel('$\omega$ [krad/s]')
        #ax.set_xlim([-500,500])

        x1,x2=ax.get_xlim()
        y1,y2=ax.get_ylim()

        # ax=axs[1]

        # ax.plot(frame_properties['Time'][plot_index],
        #         frame_properties['Expansion velocity ccf FLAP'][plot_index],
        #         label='Expansion velocity ccf FLAP',
        #         color='tab:blue')
        # ax.set_title('Scaling factor')
        # ax.text(-0.5, 1.2, '(b)', transform=ax.transAxes, size=9)
        # ax.set_xlabel('$t-t_{ELM}$ $[\mu s]$')
        # ax.set_ylabel('$f_{S}$')
        # #ax.set_xlim([-500,500])

        # #ax.xaxis.set_major_locator(MaxNLocator(5))
        # #ax.yaxis.set_major_locator(MaxNLocator(5))
        # #ax.set_xticks(ticks=[-500,-250,0,250,500])

        # x1,x2=ax.get_xlim()
        # y1,y2=ax.get_ylim()
        # ax.set_title('Scaling factor')

        plt.tight_layout(pad=0.01)
        pdf_pages.savefig()
        pdf_pages.close()

        if save_data_into_txt:
            filename=wd+fig_dir+'/data_accessibility/'+filename+'.txt'
            file1=open(filename, 'w+')
            time=frame_properties['Time']
            angular_velocity=frame_properties['Angular velocity ccf FLAP']
            expansion_velocity=frame_properties['Expansion velocity ccf FLAP']

            file1.write('#Time (ms)\n')
            for i in range(1, len(time)):
                file1.write(str(time[i])+'\t')

            file1.write('\n#Angular velocity (rad/s)\n')
            for i in range(1, len(angular_velocity)):
                file1.write(str(angular_velocity[i])+'\t')

            file1.write('\n#Expansion velocity (1/s)\n')
            for i in range(1, len(expansion_velocity)):
                file1.write(str(expansion_velocity[i])+'\t')
            file1.close()

    """
    EXAMPLE ROTATING ELM FILAMENTS
    """
    if plot_figure == 13:
        filename='fig13_elm_filament_example'
        plt.figure()
        ax,fig=plt.subplots(figsize=(8.5/2.54,8.5/2.54))
        pdf=PdfPages(wd+fig_dir+'/'+filename+'.pdf')
        show_nstx_gpi_video_frames(exp_id=exp_id_elm,
                                   time_range=time_range_elm,
                                   start_time=elm_time-5*2.5e-6,
                                   n_frame=9,
                                   logz=False,
                                   z_range=[0,20],
                                   plot_filtered=False,
                                   normalize=True,
                                   cache_data=False,
                                   plot_flux=False,
                                   plot_separatrix=True,
                                   flux_coordinates=False,
                                   device_coordinates=True,
                                   new_plot=False,
                                   save_pdf=True,
                                   colormap='gist_ncar',
                                   save_for_paraview=False,
                                   colorbar_visibility=False,
                                   save_data_for_publication=True,
                                   data_filename=wd+fig_dir+'/data_accessibility/'+filename+'.txt'
                                   )

        pdf.savefig()
        pdf.close()

    """
    RESULTS FOR ROTATING FILAMENTS
    """
    if plot_figure == 14:
        filename='fig14_elm_filament_estimation'

        frame_properties=calculate_nstx_gpi_angular_velocity(exp_id=exp_id_elm,
                                                             time_range=time_range_elm,
                                                             normalize='roundtrip',
                                                             normalize_for_velocity=True,
                                                             plot=False,
                                                             pdf=False,
                                                             pdf_filename=wd+fig_dir+'/'+filename+'.pdf',
                                                             nocalc=nocalc,
                                                             plot_scatter=False,
                                                             plot_for_publication=False,
                                                             correlation_threshold=0.0,
                                                             return_results=True,
                                                             subtraction_order_for_velocity=2,
                                                             gaussian_blur=True,
                                                             sigma_low=sigma_low,sigma_high=sigma_high,
                                                             )


        nan_ind=np.where(frame_properties['Correlation max'] < correlation_threshold)

        frame_properties['Velocity ccf FLAP'][nan_ind,0] = np.nan
        frame_properties['Velocity ccf FLAP'][nan_ind,1] = np.nan
        frame_properties['Angular velocity ccf FLAP log'][nan_ind]=np.nan
        frame_properties['Expansion velocity ccf FLAP'][nan_ind]=np.nan


        plot_index=np.logical_and(np.logical_not(np.isnan(frame_properties['Velocity ccf FLAP'][:,0])),
                                  np.logical_and(frame_properties['Time'] >= time_range_elm[0],
                                                 frame_properties['Time'] <= time_range_elm[1]))

        #Plotting the radial velocity

        pdf_filename=wd+fig_dir+'/'+filename+'.pdf'
        pdf_pages=PdfPages(pdf_filename)

        figsize=(8.5/2.54,8.5/2.54/np.sqrt(2))

        frame_properties['Time']=np.arange(frame_properties['Time'].shape[0])*2.5-frame_properties['Time'].shape[0]//2*2.5

        fig, ax = plt.subplots(1,1, figsize=figsize)
        # fig, axs = plt.subplots(2,1, figsize=figsize)

        """
        ANGULAR AND EXPANSION VELOCITY PLOTTING FROM FLAP
        """

        # ax=axs[0]
        ax.plot(frame_properties['Time'][plot_index],
                frame_properties['Angular velocity ccf FLAP log'][plot_index]/1e3,
                label='Angular velocity ccf FLAP log',
                color='tab:blue')
        ax.set_title('Angular velocity for shot #141319; $t_{ELM}$=552.497ms')
        #ax.text(-0.5, 1.2, '(a)', transform=ax.transAxes, size=9)
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.set_xticks(ticks=[-500,-250,0,250,500])

        ax.set_xlabel('$t-t_{ELM}$ $[\mu s]$')
        ax.set_ylabel('$\omega$ [krad/s]')
        ax.set_xlim([-500,500])

        x1,x2=ax.get_xlim()
        y1,y2=ax.get_ylim()


        # ax=axs[1]

        # ax.plot(frame_properties['Time'][plot_index],
        #         frame_properties['Expansion velocity ccf FLAP'][plot_index],
        #         label='Expansion velocity ccf FLAP',
        #         color='tab:blue')
        # ax.set_title('Scaling factor')
        # ax.text(-0.5, 1.2, '(b)', transform=ax.transAxes, size=9)
        # ax.set_xlabel('$t-t_{ELM}$ $[\mu s]$')
        # ax.set_ylabel('$f_{S}$')
        # ax.set_xlim([-500,500])

        # ax.xaxis.set_major_locator(MaxNLocator(5))
        # ax.yaxis.set_major_locator(MaxNLocator(5))
        # ax.set_xticks(ticks=[-500,-250,0,250,500])

        # x1,x2=ax.get_xlim()
        # y1,y2=ax.get_ylim()
        # ax.set_title('Scaling factor')

        plt.tight_layout(pad=0.1)
        pdf_pages.savefig()
        pdf_pages.close()

        if save_data_into_txt:
            filename=wd+fig_dir+'/data_accessibility/'+filename+'.txt'
            file1=open(filename, 'w+')
            time=frame_properties['Time']
            angular_velocity=frame_properties['Angular velocity ccf FLAP']
            expansion_velocity=frame_properties['Expansion velocity ccf FLAP']

            file1.write('#Time (ms)\n')
            for i in range(1, len(time)):
                file1.write(str(time[i])+'\t')

            file1.write('\n#Angular velocity (rad/s)\n')
            for i in range(1, len(angular_velocity)):
                file1.write(str(angular_velocity[i])+'\t')

            file1.write('\n#Expansion velocity (1/s)\n')
            for i in range(1, len(expansion_velocity)):
                file1.write(str(expansion_velocity[i])+'\t')
            file1.close()

    if plot_figure == 15:
        filename='fig15_method_comparison'

        pdf_filename=wd+fig_dir+'/'+filename+'.pdf'
        pdf_pages=PdfPages(pdf_filename)
        if nocalc:
            nocalc_all=[True,True,True]
        else:
            nocalc_all=[False,False,False]
        from flap_nstx.analysis import compare_angular_velocity_methods
        results_elm=compare_angular_velocity_methods(exp_id=exp_id_elm,
                                                     time_range=time_range_elm,
                                                     plot=False,
                                                     return_results=True,
                                                     nocalc=nocalc_all,
                                                     )
        nan_ind=np.where(results_elm['cccf']['data']['Correlation max']['raw'] < correlation_threshold)
        results_elm['cccf']['data']['Angular velocity ccf FLAP log']['raw'][nan_ind]=np.nan


        plot_index_elm=np.logical_and(np.logical_not(np.isnan(results_elm['cccf']['data']['Angular velocity ccf FLAP log']['raw'])),
                                      np.logical_and(results_elm['cccf']['time'] >= time_range_elm[0],
                                                      results_elm['cccf']['time'] <= time_range_elm[1]))

        results_blob=compare_angular_velocity_methods(exp_id=exp_id_blob,
                                                      time_range=time_range_blob,
                                                      plot=False,
                                                      return_results=True,
                                                      nocalc=nocalc_all,
                                                      )

        nan_ind=np.where(results_blob['cccf']['data']['Correlation max']['raw'] < correlation_threshold)
        results_blob['cccf']['data']['Angular velocity ccf FLAP log']['raw'][nan_ind]=np.nan


        plot_index_blob=np.logical_and(np.logical_not(np.isnan(results_blob['cccf']['data']['Angular velocity ccf FLAP log']['raw'])),
                                      np.logical_and(results_blob['cccf']['time'] >= time_range_blob[0],
                                                      results_blob['cccf']['time'] <= time_range_blob[1]))
        """
        ANGULAR VELOCITY PLOTTING FOR ELMS
        """

        figsize=(8.5/2.54,
                 8.5/2.54*np.sqrt(2))
        fig, axs = plt.subplots(2,1, figsize=figsize)

        ax=axs[0]
        ax.plot(results_blob['cccf']['time'][plot_index_blob]*1e3,
                results_blob['cccf']['data']['Angular velocity ccf log']['raw'][plot_index_blob]/1e3,
                label='CCCF',
                lw=0.5,
                )
        ax.plot(results_blob['contour']['Time']*1e3,
                results_blob['contour']['derived']['Angular velocity angle']['avg']/1e3,
                label='contour',
                lw=0.2,
                )

        ax.plot(results_blob['watershed']['Time']*1e3,
                results_blob['watershed']['derived']['Angular velocity angle']['avg']/1e3,
                label='watershed',
                lw=0.2,
                )
        ax.plot(results_blob['cccf']['time'][plot_index_blob]*1e3,
                results_blob['cccf']['data']['Angular velocity ccf log']['raw'][plot_index_blob]/1e3,
                lw=0.5,
                color='tab:blue'
                )
        ax.set_ylim([-100,100])
        ax.set_title('Angular velocity for shot #141307')
        ax.text(-0.22, 1.05, '(a)', transform=ax.transAxes, size=9)
        ax.legend(fontsize=7, loc='upper left')

        ax.set_xlabel('t [ms]')
        ax.set_ylabel('$\omega$ [krad/s]')

        ax=axs[1]

        ax.plot(results_elm['cccf']['time'][plot_index_elm]*1e3,
                results_elm['cccf']['data']['Angular velocity ccf log']['raw'][plot_index_elm]/1e3,
                label='CCCF',
                lw=0.5,
                )
        ax.plot(results_elm['contour']['Time']*1e3,
                results_elm['contour']['derived']['Angular velocity angle']['avg']/1e3,
                label='contour',
                lw=0.2,
                )
        ax.plot(results_elm['watershed']['Time']*1e3,
                results_elm['watershed']['derived']['Angular velocity angle']['avg']/1e3,
                label='watershed',
                lw=0.2,
                )
        ax.plot(results_elm['cccf']['time'][plot_index_elm]*1e3,
                results_elm['cccf']['data']['Angular velocity ccf log']['raw'][plot_index_elm]/1e3,
                color='tab:blue',
                lw=0.5,
                )
        ax.set_title('Angular velocity for shot #141319')
        ax.text(-0.22, 1.05, '(b)', transform=ax.transAxes, size=9)

        ax.legend(fontsize=7, loc='upper left')
        ax.set_xlabel('t [ms]')
        ax.set_ylabel('$\omega$ [krad/s]')
        ax.set_ylim([-300,300])
        plt.tight_layout(pad=0.5)
        pdf_pages.savefig()
        pdf_pages.close()

        if save_data_into_txt:
            for ind_res in ['elm','blob']:
                if ind_res == 'blob':
                    result=results_blob
                    filename_cur=wd+fig_dir+'/data_accessibility/'+filename+'(a).txt'
                    plot_index=plot_index_blob
                elif ind_res == 'elm':
                    result=results_elm
                    filename_cur=wd+fig_dir+'/data_accessibility/'+filename+'(b).txt'
                    plot_index=plot_index_elm

                with open(filename_cur, 'w+') as file1:
                    time1=result['cccf']['time'][plot_index]*1e3
                    data1=result['cccf']['data']['Angular velocity ccf log']['raw'][plot_index]/1e3

                    time2=result['contour']['Time']*1e3
                    data2=result['contour']['derived']['Angular velocity angle']['avg']/1e3

                    time3=result['watershed']['Time']*1e3
                    data3=result['watershed']['derived']['Angular velocity angle']['avg']/1e3

                    file1.write('#Time (ms)\n')
                    for i in range(0, len(time1)):
                        file1.write(str(time1[i])+'\t')
                    file1.write('\n#Angular velocity CCCF (rad/s)\n')
                    for i in range(0, len(data1)):
                        file1.write(str(data1[i])+'\t')

                    file1.write('\n\n#Time (ms)\n')
                    for i in range(0, len(time2)):
                        file1.write(str(time2[i])+'\t')
                    file1.write('\n#Angular velocity contour (rad/s)\n')
                    for i in range(0, len(data2)):
                        file1.write(str(data2[i])+'\t')

                    file1.write('\n\n#Time (ms)\n')
                    for i in range(0, len(time3)):
                        file1.write(str(time3[i])+'\t')
                    file1.write('\n#Angular velocity watershed (rad/s)\n')
                    for i in range(0, len(data3)):
                        file1.write(str(data3[i])+'\t')

                    file1.close()
