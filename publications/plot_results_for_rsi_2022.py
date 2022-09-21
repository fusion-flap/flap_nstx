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

wd=flap.config.get_all_section('Module NSTX_GPI')['Working directory']
fig_dir='/publication_figures/rsi_2022'

def plot_results_for_rsi_2022_paper(plot_figure=2,
                                    save_data_into_txt=False,
                                    plot_all=False,
                                    nocalc=False):

    if plot_all:
        plot_figure=-1
        for i in range(15):
            plot_results_for_rsi_2022_paper(plot_figure=i,
                                            save_data_into_txt=save_data_into_txt)


    #Settings for fig 2,3,4,13,14

    exp_id=139950
    elm_time=0.384182467

    time_range=[elm_time-500e-6,elm_time+500e-6]
    #sample_elm_time=flap.get_data_object_ref('GPI').slice_data(slicing={'Time':0.501579}).coordinate('Sample')[0][0,0]
    sample_to_plot=[33475,33476] #Sample at ELM time
    #Settings for fig 9,10
    exp_id_blob=141307
    time_range_blob=[0.484198-500e-6,
                     0.484198+500e-6]



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
        print('This is the structure rotation sketch')
        return

    """
    PRE-PROCESSING FIGURE
    """
    if plot_figure == 2:
        raise Hell('Gaussian blur and normalization need to be implemented.')

        filename='fig2_pre_processing'

        flap.get_data('NSTX_GPI',
                      exp_id=exp_id,
                      name='',
                      object_name='GPI')
        flap.slice_data('GPI', slicing={'Time':flap.Intervals(time_range[0],time_range[1])}, output_name='GPI_SLICED_FULL')
        data_object_name='GPI_SLICED_DENORM_CCF_VEL'
        detrended=flap_nstx.analysis.detrend_multidim(data_object_name,
                                                      exp_id=exp_id,
                                                      order=1,
                                                      coordinates=['Image x', 'Image y'],
                                                      output_name='GPI_DETREND_VEL')

        d=copy.deepcopy(flap.get_data_object(data_object_name))

        d.data=d.data-detrended.data
        flap.add_data_object(d,'GPI_TREND')

        signals=[data_object_name,
                 'GPI_TREND',
                 'GPI_DETREND_VEL',
                 'GPI_GAUSS_BLUR']

        titles=['Raw frame',
                'Normalized',
                'Trend subtract',
                'Gaussian blur',
                ]

        pdf=PdfPages(wd+fig_dir+'/'+filename+'.pdf')
        filename=wd+fig_dir+'/data_accessibility/'+filename+'.txt'
        file1=open(filename, 'w+')

        plt.figure()
        fig,axs=plt.subplots(2,2,
                             figsize=(8.5/2.54,
                                      8.5/2.54))

        for index_grid_x in range(2):
            for index_grid_y in range(2):
                ax=axs[index_grid_x,index_grid_y]
                ind=index_grid_x*2+index_grid_y

                data=flap.get_data_object(signals[index_grid_x]).slice_data(slicing={'Sample':sample_to_plot[0]}).data

                ax.contourf(np.arange(0,64),
                            np.arange(0,80),
                            data.T)
                ax.set_title(titles[ind])
                ax.set_xlabel('x [pix]')
                ax.set_xlabel('y [pix]')

                if save_data_into_txt:
                    file1.write('\n#'+signals[index_grid_x]+' data\n\n')
                    for i in range(len(data[0,:])):
                        string=''
                        for j in range(len(data[:,0])):
                            string+=str(data[j,i])+'\t'
                        string+='\n'
                        file1.write(string)
        pdf.savefig()
        pdf.close()
        file1.close()


    """
    EXAMPLE LOG-POLAR TRANSFORMATION
    """
    if plot_figure == 3:
        filename='fig3_logpolar_transform'

        result=calculate_nstx_gpi_angular_velocity(exp_id=exp_id,
                                                   time_range=[time_range[0],
                                                               time_range[1]],
                                                   normalize='roundtrip',
                                                   normalize_for_velocity=True,
                                                   plot=False,
                                                   pdf=True,
                                                   pdf_filename=wd+fig_dir+'/'+filename+'.pdf',
                                                   nocalc=False,
                                                   plot_scatter=False,
                                                   plot_for_publication=True,
                                                   gaussian_blur=True,
                                                   calculate_half_fft=False,
                                                   test_into_pdf=True,
                                                   return_results=True,

                                                   subtraction_order_for_velocity=2,
                                                   sample_to_plot=sample_to_plot,
                                                   save_data_for_publication=True,
                                                   data_filename=wd+fig_dir+'/data_accessibility/'+filename+'.txt')

    """
    EXAMPLE CCCF FUNCTION
    """
    if plot_figure == 4:
        raise Hell('revise plot')
        filename='fig4_example_cccf'

        result=calculate_nstx_gpi_angular_velocity(exp_id=exp_id,
                                                   time_range=[time_range[0],
                                                               time_range[1]],
                                                   normalize='roundtrip',
                                                   normalize_for_velocity=True,
                                                   plot=False,
                                                   pdf=True,
                                                   pdf_filename=wd+fig_dir+'/'+filename+'.pdf',
                                                   nocalc=False,
                                                   plot_scatter=False,
                                                   plot_for_publication=False,
                                                   gaussian_blur=True,
                                                   calculate_half_fft=False,
                                                   test_into_pdf=False,
                                                   return_results=True,
                                                   plot_ccf=True,
                                                   subtraction_order_for_velocity=2,
                                                   sample_to_plot=[sample_to_plot[1]],
                                                   save_data_for_publication=True,
                                                   data_filename=wd+fig_dir+'/data_accessibility/'+filename+'.txt')

    """
    EXAMPLE FLOWCHART FIGURE
    """
    if plot_figure == 5:
        print('This is the flowchart figure.')
        return

    """
    ROTATING GAUSSIAN STRUCTURE
    """
    if plot_figure == 6:
        raise Hell('Revise plot')
        filename='fig6_rotating_gaussian'

        test_angular_displacement_estimation(plot_sample_gaussian=True,
                                             pdf=True,
                                             pdf_filename=wd+fig_dir+'/'+filename+'.pdf',
                                             save_data_into_txt=save_data_into_txt)

        if save_data_into_txt:
            filename=wd+fig_dir+'/data_accessibility/'+filename+'.txt'
            file1=open(filename, 'w+')

            data=flap.get_data_object('gaussian').slice_data(slicing={'Sample':0}).data
            file1.write("#Gaussian structure frame #1\n\n")
            for i in range(len(data[0,:])):
                string=''
                for j in range(len(data[:,0])):
                    string+=str(data[j,i])+'\t'
                string+='\n'
                file1.write(string)

            data=flap.get_data_object('gaussian').slice_data(slicing={'Sample':1}).data
            file1.write("\n#Gaussian structure frame #2\n\n")
            for i in range(len(data[0,:])):
                string=''
                for j in range(len(data[:,0])):
                    string+=str(data[j,i])+'\t'
                string+='\n'
                file1.write(string)
            file1.close()

    """
    TESTING THE METHOD ANGULAR DISPLACEMENT ESTIMATION
    """
    if plot_figure == 7:
        filename='fig7_rot_estimation_size_vs_elongation'
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

                                             save_data_into_txt=save_data_into_txt,
                                             save_data_filename=wd+fig_dir+'/data_accessibility/'+filename+'.txt'
                                             )

    """
    TESTING THE METHOD EXPANSION FRACTION ESTIMATION
    """
    if plot_figure == 8:
        filename='fig8_exp_estimation_size_vs_elongation'

        test_angular_displacement_estimation(frame_size_angle=True,

                                             method='ccf',
                                             angle_method='angle',

                                             n_angle=38,
                                             angle_range=[-85,85],

                                             n_size=9,
                                             size_range=[6,33],

                                             n_scale=10,
                                             scale_range=[0.01,0.19],

                                             n_elongation=10,
                                             elongation_range=[0.1,1.],

                                             frame_size_range=[16,200],
                                             frame_size_step=16,

                                             nocalc=False,
                                             pdf=True,
                                             pdf_filename=wd+fig_dir+'/'+filename+'.pdf',
                                             plot=False,

                                             save_data_into_txt=save_data_into_txt,
                                             save_data_filename=wd+fig_dir+'/data_accessibility/'+filename+'.txt'
                                             )

    """
    EXAMPLE ROTATING BLOBS
    """
    if plot_figure == 9:
        filename='fig9_blob_rotation_example'
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
    if plot_figure == 10:
        filename='fig10_blob_rotation_estimate'

        frame_properties=calculate_nstx_gpi_angular_velocity(exp_id=exp_id_blob,
                                                             time_range=time_range_blob,
                                                             normalize='roundtrip',
                                                             normalize_for_velocity=True,
                                                             plot=False,
                                                             pdf=False,
                                                             nocalc=nocalc,
                                                             plot_scatter=False,
                                                             plot_for_publication=False,
                                                             correlation_threshold=0.7,
                                                             return_results=True,
                                                             subtraction_order_for_velocity=2,
                                                             gaussian_blur=True,
                                                             )

        plot_index=np.logical_and(np.logical_not(np.isnan(frame_properties['Velocity ccf FLAP'][:,0])),
                                  np.logical_and(frame_properties['Time'] >= time_range_blob[0],
                                                 frame_properties['Time'] <= time_range_blob[1]))


        nan_ind=np.where(frame_properties['Correlation max'] < 0.7)

        frame_properties['Velocity ccf FLAP'][nan_ind,0] = np.nan
        frame_properties['Velocity ccf FLAP'][nan_ind,1] = np.nan
        frame_properties['Angular velocity ccf FLAP log'][nan_ind]=np.nan
        frame_properties['Expansion velocity ccf FLAP'][nan_ind]=np.nan

        #Plotting the radial velocity

        pdf_filename=wd+fig_dir+'/'+filename+'.pdf'
        pdf_pages=PdfPages(pdf_filename)

        figsize=(8.5/2.54,
                 8.5/2.54)


        """
        RADIAL AND POLOIDAL PLOTTING FROM FLAP
        """

        fig, axs = plt.subplots(2,1, figsize=figsize)


        """
        ANGULAR AND EXPANSION VELOCITY PLOTTING FROM FLAP
        """

        ax=axs[0]
        ax.plot(frame_properties['Time'][plot_index],
                frame_properties['Angular velocity ccf FLAP log'][plot_index]/1e3,
                label='Angular velocity ccf FLAP log',
                color='tab:blue')
        ax.set_title('Angular velocity FLAP')
        ax.text(-0.5, 1.2, '(a)', transform=ax.transAxes, size=9)
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.yaxis.set_major_locator(MaxNLocator(5))
        #ax.set_xticks(ticks=[-500,-250,0,250,500])

        ax.set_xlabel('$t-t_{ELM}$ $[\mu s]$')
        ax.set_ylabel('$\omega$ [krad/s]')
        #ax.set_xlim([-500,500])

        x1,x2=ax.get_xlim()
        y1,y2=ax.get_ylim()



        """
        ANGULAR AND EXPANSION VELOCITY PLOTTING FROM FLAP
        """

        ax=axs[1]

        ax.plot(frame_properties['Time'][plot_index],
                frame_properties['Expansion velocity ccf FLAP'][plot_index],
                label='Expansion velocity ccf FLAP',
                color='tab:blue')
        ax.set_title('Scaling factor')
        ax.text(-0.5, 1.2, '(b)', transform=ax.transAxes, size=9)
        ax.set_xlabel('$t-t_{ELM}$ $[\mu s]$')
        ax.set_ylabel('$f_{S}$')
        #ax.set_xlim([-500,500])

        #ax.xaxis.set_major_locator(MaxNLocator(5))
        #ax.yaxis.set_major_locator(MaxNLocator(5))
        #ax.set_xticks(ticks=[-500,-250,0,250,500])

        x1,x2=ax.get_xlim()
        y1,y2=ax.get_ylim()
        ax.set_title('Scaling factor')

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
    if plot_figure == 11:
        filename='fig11_elm_filament_example'
        plt.figure()
        ax,fig=plt.subplots(figsize=(8.5/2.54,8.5/2.54))
        pdf=PdfPages(wd+fig_dir+'/'+filename+'.pdf')
        show_nstx_gpi_video_frames(exp_id=exp_id,
                                   time_range=time_range,
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
    if plot_figure == 12:
        filename='fig12_elm_filament_estimation'

        frame_properties=calculate_nstx_gpi_angular_velocity(exp_id=exp_id,
                                                             time_range=time_range,
                                                             normalize='roundtrip',
                                                             normalize_for_velocity=True,
                                                             plot=False,
                                                             pdf=False,
                                                             pdf_filename=wd+fig_dir+'/'+filename+'.pdf',
                                                             nocalc=True,
                                                             plot_scatter=False,
                                                             plot_for_publication=False,
                                                             correlation_threshold=0.7,
                                                             return_results=True,
                                                             subtraction_order_for_velocity=2,
                                                             gaussian_blur=True,
                                                             )

        plot_index=np.logical_and(np.logical_not(np.isnan(frame_properties['Velocity ccf FLAP'][:,0])),
                                  np.logical_and(frame_properties['Time'] >= time_range[0],
                                                 frame_properties['Time'] <= time_range[1]))


        nan_ind=np.where(frame_properties['Correlation max'] < 0.7)

        frame_properties['Velocity ccf FLAP'][nan_ind,0] = np.nan
        frame_properties['Velocity ccf FLAP'][nan_ind,1] = np.nan
        frame_properties['Angular velocity ccf FLAP log'][nan_ind]=np.nan
        frame_properties['Expansion velocity ccf FLAP'][nan_ind]=np.nan

        #Plotting the radial velocity

        pdf_filename=wd+fig_dir+'/'+filename+'.pdf'
        pdf_pages=PdfPages(pdf_filename)

        figsize=(8.5/2.54,
                 8.5/2.54)

        """
        RADIAL AND POLOIDAL PLOTTING FROM FLAP
        """
        frame_properties['Time']=np.arange(frame_properties['Time'].shape[0])*2.5-frame_properties['Time'].shape[0]//2*2.5

        fig, axs = plt.subplots(2,1, figsize=figsize)

        """
        ANGULAR AND EXPANSION VELOCITY PLOTTING FROM FLAP
        """

        ax=axs[0]
        ax.plot(frame_properties['Time'][plot_index],
                frame_properties['Angular velocity ccf FLAP log'][plot_index]/1e3,
                label='Angular velocity ccf FLAP log',
                color='tab:blue')
        ax.set_title('Angular velocity FLAP')
        ax.text(-0.5, 1.2, '(a)', transform=ax.transAxes, size=9)
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.set_xticks(ticks=[-500,-250,0,250,500])

        ax.set_xlabel('$t-t_{ELM}$ $[\mu s]$')
        ax.set_ylabel('$\omega$ [krad/s]')
        ax.set_xlim([-500,500])

        x1,x2=ax.get_xlim()
        y1,y2=ax.get_ylim()



        """
        ANGULAR AND EXPANSION VELOCITY PLOTTING FROM FLAP
        """

        ax=axs[1]

        ax.plot(frame_properties['Time'][plot_index],
                frame_properties['Expansion velocity ccf FLAP'][plot_index],
                label='Expansion velocity ccf FLAP',
                color='tab:blue')
        ax.set_title('Scaling factor')
        ax.text(-0.5, 1.2, '(b)', transform=ax.transAxes, size=9)
        ax.set_xlabel('$t-t_{ELM}$ $[\mu s]$')
        ax.set_ylabel('$f_{S}$')
        ax.set_xlim([-500,500])

        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.set_xticks(ticks=[-500,-250,0,250,500])

        x1,x2=ax.get_xlim()
        y1,y2=ax.get_ylim()
        ax.set_title('Scaling factor')

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
