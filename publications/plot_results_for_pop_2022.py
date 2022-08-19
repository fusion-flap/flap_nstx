#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 11:49:01 2022

@author: mlampert
"""

import os
import copy
# import pickle


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
# from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator
import matplotlib.cm as cm

from flap_nstx.gpi import calculate_nstx_gpi_frame_by_frame_velocity
from flap_nstx.gpi import show_nstx_gpi_video_frames
from flap_nstx.gpi import calculate_nstx_gpi_angular_velocity
from flap_nstx.gpi import plot_nstx_gpi_angular_velocity_distribution, plot_nstx_gpi_velocity_distribution
from flap_nstx.analysis import plot_angular_vs_translational_velocity, plot_gpi_profile_dependence_ultimate
from flap_nstx.analysis import calculate_shear_induced_angular_velocity, calculate_shear_layer_vpol, analyze_shear_distribution
from flap_nstx.tools import calculate_corr_acceptance_levels

import flap
import flap_nstx
wd=flap.config.get_all_section('Module NSTX_GPI')['Working directory']
fig_dir='/publication_figures/pop_2022'


thisdir = os.path.dirname(os.path.realpath(__file__))
fn = os.path.join(thisdir,"../flap_nstx.cfg")
flap.config.read(file_name=fn)
flap_nstx.register()
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

class NoobError(Exception):
    pass

def plot_results_for_pop_2022(plot_figure=2,
                              plot_all=False,
                              save_data_into_txt=False,
                              gaussian_blur=True,
                              subtraction_order=2,
                              flap_or_skim='FLAP',
                              plot_for_analysis=False,
                              ):

    if plot_all:
        plot_figure=-1
        for i in range(12):
            plot_results_for_pop_2022(plot_figure=i,
                                      save_data_into_txt=save_data_into_txt)


    """
    ELM figure
    """
    if plot_figure == 1:
        print('This is the ELM figure, no need for calculation.')

    """
    GPI figure
    """
    if plot_figure == 2:
        print('This is the GPI figure, no need for calculation.')

    """
    Example frame pairs with observable rotation in between them
    """

    if plot_figure == 3:
        result=calculate_nstx_gpi_angular_velocity(exp_id=141319,
                                                   time_range=[0.5524,0.5526],
                                                   normalize='roundtrip',
                                                   normalize_for_velocity=True,
                                                   plot=False,
                                                   pdf=True,
                                                   nocalc=False,
                                                   plot_scatter=False,
                                                   plot_for_publication=True,
                                                   gaussian_blur=gaussian_blur,
                                                   calculate_half_fft=False,
                                                   test_into_pdf=True,
                                                   return_results=True,

                                                   subtraction_order_for_velocity=subtraction_order,
                                                   sample_to_plot=[20873,20874],
                                                   save_data_for_publication=True,
                                                   data_filename=wd+fig_dir+'/data_accessibility/fig_34')

    if plot_figure == 4:
        result=calculate_nstx_gpi_angular_velocity(exp_id=141319,
                                                   time_range=[0.5524,0.5526],
                                                   normalize='roundtrip',
                                                   normalize_for_velocity=True,
                                                   plot=False,
                                                   pdf=True,
                                                   nocalc=False,
                                                   plot_scatter=False,
                                                   plot_for_publication=False,
                                                   gaussian_blur=gaussian_blur,
                                                   calculate_half_fft=False,
                                                   test_into_pdf=False,
                                                   return_results=True,
                                                   plot_ccf=True,
                                                   subtraction_order_for_velocity=subtraction_order,
                                                   sample_to_plot=[20874],
                                                   save_data_for_publication=True,
                                                   data_filename=wd+fig_dir+'/data_accessibility/fig_34')
    """
    Flowchart of the Fourier Mellin based method
    """
    if plot_figure == 5:
        print('This is the flowchart figure, no need to plot.')


    """
    Example set of subsequent frames clearly exhibiting rotation
    """
    if plot_figure == 6:
        plt.figure()
        ax,fig=plt.subplots(figsize=(3.35*2,5.5))
        pdf=PdfPages(wd+fig_dir+'/fig_1413190.552500_9_frame.pdf')

        str_fitting=calculate_nstx_gpi_frame_by_frame_velocity(exp_id=141319,
                                                      time_range=[0.552,0.553],
                                                      plot=False,
                                                      subtraction_order_for_velocity=4,
                                                      skip_structure_calculation=False,
                                                      remove_interlaced_structures=False,
                                                      correlation_threshold=0.7,
                                                      pdf=False,
                                                      nlevel=51,
                                                      nocalc=True,
                                                      filter_level=5,
                                                      normalize_for_size=True,
                                                      normalize_for_velocity=True,
                                                      threshold_coeff=1.,
                                                      normalize_f_high=1e3,
                                                      normalize='roundtrip',
                                                      velocity_base='cog',
                                                      return_results=True,
                                                      plot_gas=False,
                                                      structure_pixel_calc=False,
                                                      structure_pdf_save=False,
                                                      test_structures=False,
                                                      save_data_for_publication=False,)

        ind_start=np.argmin(np.abs(str_fitting['Time']-(0.552500-6*2.5e-6)))
        overplot_points=np.zeros([3,3,2])

        for ind_y in range(3):
            for ind_x in range(3):
                overplot_points[ind_x,ind_y,:]=str_fitting['COG max'][ind_start+ind_y*3+ind_x,:]
                print(str_fitting['Time'][ind_start+ind_y*3+ind_x])
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
                                        colorbar_visibility=False,
                                        save_data_for_publication=True,
                                        data_filename=wd+fig_dir+'/data_accessibility/fig_1413190.552500_9_frame.txt',
                                        overplot_points=overplot_points
                                        )

        pdf.savefig()
        pdf.close()

    """
    Filament rotation estimation for the single shot
    """
    if plot_figure == 7:

        plt.rc('font', family='serif', serif='Helvetica')
        labelsize=7.
        linewidth=0.4
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

        time_range=[0.552,0.553]
        correlation_threshold=0.7
        frame_properties=calculate_nstx_gpi_angular_velocity(exp_id=141319,
                                                             time_range=time_range,
                                                             normalize='roundtrip',
                                                             normalize_for_velocity=True,
                                                             plot=False,
                                                             pdf=False,
                                                             nocalc=True,
                                                             plot_scatter=False,
                                                             plot_for_publication=False,
                                                             correlation_threshold=correlation_threshold,
                                                             return_results=True,
                                                             subtraction_order_for_velocity=2,
                                                             gaussian_blur=True,
                                                             )

        str_fitting=calculate_nstx_gpi_frame_by_frame_velocity(exp_id=141319,
                                                      time_range=time_range,
                                                      plot=False,
                                                      subtraction_order_for_velocity=4,
                                                      skip_structure_calculation=False,
                                                      remove_interlaced_structures=False,
                                                      correlation_threshold=correlation_threshold,
                                                      pdf=False,
                                                      nlevel=51,
                                                      nocalc=True,
                                                      filter_level=5,
                                                      normalize_for_size=False,
                                                      normalize_for_velocity=True,
                                                      threshold_coeff=1.,
                                                      normalize_f_high=1e3,
                                                      normalize='roundtrip',
                                                      velocity_base='cog',
                                                      return_results=True,
                                                      plot_gas=False,
                                                      structure_pixel_calc=False,
                                                      structure_pdf_save=False,
                                                      test_structures=False,
                                                      save_data_for_publication=False,
                                                      )

        plot_index=np.logical_and(np.logical_not(np.isnan(frame_properties['Velocity ccf FLAP'][:,0])),
                                  np.logical_and(frame_properties['Time'] >= time_range[0],
                                                 frame_properties['Time'] <= time_range[1]))


        nan_ind=np.where(frame_properties['Correlation max'] < correlation_threshold)

        frame_properties['Velocity ccf FLAP'][nan_ind,0] = np.nan
        frame_properties['Velocity ccf FLAP'][nan_ind,1] = np.nan
        frame_properties['Angular velocity ccf FLAP log'][nan_ind]=np.nan
        frame_properties['Expansion velocity ccf FLAP'][nan_ind]=np.nan

        #Plotting the radial velocity

        filename=flap_nstx.tools.filename(exp_id=141319,
                                          working_directory=wd+'/plots',
                                          time_range=time_range,
                                          purpose='ccf ang velocity')
        pdf_filename=filename+'.pdf'
        pdf_pages=PdfPages(pdf_filename)

        figsize=(8.5/2.54,
                 8.5/2.54)
        plt.rc('font', family='serif', serif='Helvetica')
        labelsize=9
        linewidth=0.5
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
        import matplotlib
        if plot_for_analysis:
            matplotlib.use('qt5agg')
        else:
            matplotlib.use('agg')

        """
        RADIAL AND POLOIDAL PLOTTING FROM FLAP
        """

        fig, axs = plt.subplots(2,2, figsize=figsize)

        ax=axs[1,0]
        frame_properties['Time']=np.arange(frame_properties['Time'].shape[0])*2.5-frame_properties['Time'].shape[0]//2*2.5

        ax.plot(frame_properties['Time'][plot_index],
                 frame_properties['Velocity ccf FLAP'][plot_index,0]/1e3,
                 label='Velocity ccf FLAP')
        ax.set_title('Radial velocity')
        ax.text(-0.5, 1.2, '(c)', transform=ax.transAxes, size=9)
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.set_xticks(ticks=[-500,-250,0,250,500])
        ax.set_xlabel('$t-t_{ELM}$ $[\mu s]$')
        ax.set_ylabel('$v_{rad}$ [km/s]')
        ax.set_xlim([-500,500])

        x1,x2=ax.get_xlim()
        y1,y2=ax.get_ylim()
        # ax.set_aspect((x2-x1)/(y2-y1)/1.618)


        ax=axs[1,1]

        ax.plot(frame_properties['Time'][plot_index],
                str_fitting['Separatrix dist max'][plot_index],
                label='Velocity ccf FLAP')
        ax.set_title('Separatrix distance')


        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.set_xticks(ticks=[-500,-250,0,250,500])
        ax.set_xlabel('$t-t_{ELM}$ $[\mu s]$')
        ax.text(-0.5, 1.2, '(d)', transform=ax.transAxes, size=9)
        ax.set_ylabel('$r-r_{sep}$ [mm]')
        ax.set_xlim([-500,500])

        x1,x2=ax.get_xlim()
        y1,y2=ax.get_ylim()

        """
        ANGULAR AND EXPANSION VELOCITY PLOTTING FROM SKIMAGE
        """

        ax=axs[0,0]
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
        ANGULAR AND EXPANSION VELOCITY PLOTTING FROM SKIMAGE
        """

        ax=axs[0,1]

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
            filename=wd+fig_dir+'/data_accessibility/figure_7abcd.txt'
            file1=open(filename, 'w+')
            time=result['Time']
            radial_velocity=result['Velocity ccf FLAP'][:,0]
            poloidal_velocity=result['Velocity ccf FLAP'][:,1]
            angular_velocity=result['Angular velocity ccf FLAP']
            expansion_velocity=result['Expansion velocity ccf FLAP']

            file1.write('#Time (ms)\n')
            for i in range(1, len(time)):
                file1.write(str(time[i])+'\t')

            file1.write('\n#Radial velocity (m/s)\n')
            for i in range(1, len(radial_velocity)):
                file1.write(str(radial_velocity[i])+'\t')

            file1.write('\n#Poloidal velocity (m/s)\n')
            for i in range(1, len(poloidal_velocity)):
                file1.write(str(poloidal_velocity[i])+'\t')

            file1.write('\n#Angular velocity (rad/s)\n')
            for i in range(1, len(angular_velocity)):
                file1.write(str(angular_velocity[i])+'\t')

            file1.write('\n#Expansion velocity (1/s)\n')
            for i in range(1, len(expansion_velocity)):
                file1.write(str(expansion_velocity[i])+'\t')
            file1.close()

    """
    Evolution of the rotation and expansion distribution functions
    """
    if plot_figure == 8:
        time_vec, y_vector = plot_nstx_gpi_angular_velocity_distribution(plot_for_publication=True,
                                                                         window_average=500e-6,
                                                                         subtraction_order=subtraction_order,
                                                                         correlation_threshold=0.7,
                                                                         pdf=False,
                                                                         plot=False,
                                                                         return_results=True,
                                                                         plot_all_time_traces=False,
                                                                         tau_range=[-1e-3,1e-3])
        figsize=(8.5/2.54,8.5/2.54)
        plt.rc('font', family='serif', serif='Helvetica')
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

        pdf_object=PdfPages(wd+fig_dir+'/fig_ang_velocity_distribution_log.pdf')


        import matplotlib
        if plot_for_analysis:
            matplotlib.use('qt5agg')
        else:
            matplotlib.use('agg')

        def fmt(x, pos):
            a = '{:3.2f}'.format(x)
            return a

        plt.figure()
        fig,((ax1,ax2),(ax3,ax4))=plt.subplots(2,2,figsize=figsize)
        for key in ['Angular velocity ccf '+flap_or_skim+' log',
                    'Expansion velocity ccf '+flap_or_skim+'']:

            if key == 'Angular velocity ccf '+flap_or_skim+' log':
                ax=ax1
                corner_text='(a)'
            else:
                ax=ax3
                corner_text='(c)'
            im=ax.contourf(time_vec*1e3,
                           y_vector[key]['bins'],
                           y_vector[key]['data'].transpose(),
                           levels=50,
                           )
            ax.plot(time_vec*1e3,
                     y_vector[key]['median'],
                     color='red',
                     lw=linewidth)
            ax.plot(time_vec*1e3,
                     y_vector[key]['10th'],
                     color='white',
                     lw=linewidth/2)
            ax.plot(time_vec*1e3,
                     y_vector[key]['90th'],
                     color='white',
                     lw=linewidth/2)
            ax.text(-0.5, 1.2, corner_text, transform=ax.transAxes, size=9)
            ax.set_title('Relative frequency of '+y_vector[key]['ylabel'])
            ax.set_xlabel('$t-t_{ELM}$ [$\mu$s]')
            ax.set_ylabel(y_vector[key]['ylabel']+' ['+y_vector[key]['unit']+']')
            ax.xaxis.set_major_locator(MaxNLocator(5))
            ax.yaxis.set_major_locator(MaxNLocator(5))
            ax.set_ylim(np.min(y_vector[key]['10th']),
                        np.max(y_vector[key]['90th']))
            ax.set_xticks(ticks=[-500,-250,0,250,500])

            import matplotlib.ticker as ticker
            cbar=fig.colorbar(im, format=ticker.FuncFormatter(fmt), ax=ax)
            cbar.ax.tick_params(labelsize=6)

            if key == 'Angular velocity ccf '+flap_or_skim+' log':
                ax=ax2
                corner_text='(b)'
            else:
                ax=ax4
                corner_text='(d)'
            # nwin=y_vector[key]['Data'].shape[0]//2
    #        plt.plot(y_vector[key]['Bins'], np.mean(y_vector[key]['Data'][nwin-2:nwin+3,:], axis=0))
            ax.plot(time_vec*1e3,
                     y_vector[key]['median'],
                     color='red',
                     lw=linewidth)

            ax.set_title('Median '+y_vector[key]['ylabel']+' evolution')
            ax.set_xlabel('$t-t_{ELM}$ [$\mu$s]')
            ax.set_ylabel(y_vector[key]['ylabel']+' ['+y_vector[key]['unit']+']')
            #ax.set_xlim([-500,500])
            ax.set_xlim([-200,200])
            # ax.xaxis.set_major_locator(MaxNLocator(5))
            ax.yaxis.set_major_locator(MaxNLocator(5))
            ax.xaxis.set_major_locator(MaxNLocator(5))
            #ax.set_xticks(ticks=[-500,-250,0,250,500])
            ax.set_xticks(ticks=[-200,-100,0,100,200])
            ax.text(-0.5, 1.2, corner_text, transform=ax.transAxes, size=9)
        plt.tight_layout(pad=0.1)
        pdf_object.savefig()
        pdf_object.close()
        if plot_for_analysis:
            matplotlib.use('qt5agg')
        else:
            matplotlib.use('agg')

        if save_data_into_txt:
            for ind in [5,7]:
                if ind == 5:
                    filename=wd+fig_dir+'/data_accessibility/figure_8ab.txt'
                if ind == 7:
                    filename=wd+fig_dir+'/data_accessibility/figure_8cd.txt'

                file1=open(filename, 'w+')

                file1.write('#Time (ms)\n')
                for i in range(1, len(time_vec)):
                    file1.write(str(time_vec[i])+'\t')

                file1.write('\n#Median\n')
                for i in range(1, len(y_vector[key]['median'])):
                    file1.write(str(y_vector[key]['median'])+'\t')

                file1.write('\n#10th perc.\n')
                for i in range(1, len(y_vector[key]['10th'])):
                    file1.write(str(y_vector[key]['10th'])+'\t')

                file1.write('\n#90th perc.\n')
                for i in range(1, len(y_vector[key]['90th'])):
                    file1.write(str(y_vector[key]['90th'])+'\t')

                file1.write('\n#Distribution bins\n')
                for i in range(1, len(y_vector[key]['bins'])):
                    file1.write(str(y_vector[key]['bins'])+'\t')
                file1.write('\n#Distribution\n')

                for i in range(len(y_vector[key]['data'][0,:])):
                    string=''
                    for j in range(len(y_vector[key]['data'][:,0])):
                        string+=str(y_vector[key]['data'][j,i])+'\t'
                    string+='\n'
                    file1.write(string)
                file1.close()

    """
    vrad,r-r_sep distribution plotting
    """

    if plot_figure == 9:
        time_vec, y_vector = plot_nstx_gpi_velocity_distribution(plot_for_publication=False,
                                                                 correlation_threshold=0.7,
                                                                 pdf=False,
                                                                 plot=False,
                                                                 return_results=True,
                                                                 figure_size=4.25)
        figsize=(8.5/2.54,8.5/2.54)
        plt.rc('font', family='serif', serif='Helvetica')
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

        pdf_object=PdfPages(wd+fig_dir+'/fig_trans_velocity_distribution.pdf')


        import matplotlib
        if plot_for_analysis:
            matplotlib.use('qt5agg')
        else:
            matplotlib.use('agg')
        def fmt(x, pos):
            a = '{:3.2f}'.format(x)
            return a
        plt.figure()
        fig,((ax1,ax2),(ax3,ax4))=plt.subplots(2,2,figsize=figsize)
        for key in ['Velocity ccf radial','Distance']:
            if key == 'Velocity ccf radial':
                ax=ax1
                corner_text='(a)'
            else:
                ax=ax3
                corner_text='(c)'
            im=ax.contourf(time_vec*1e3,
                           y_vector[key]['bins'],
                           y_vector[key]['data'].transpose(),
                           levels=51,
                           )
            ax.plot(time_vec*1e3,
                     y_vector[key]['median'],
                     color='red',
                     lw=linewidth)
            ax.plot(time_vec*1e3,
                     y_vector[key]['10th'],
                     color='white',
                     lw=linewidth/2)
            ax.plot(time_vec*1e3,
                     y_vector[key]['90th'],
                     color='white',
                     lw=linewidth/2)

            ax.set_title('Relative frequency of '+y_vector[key]['ylabel'])
            ax.set_xlabel('$t-t_{ELM}$ [$\mu$s]')
            ax.set_ylabel(y_vector[key]['ylabel']+' ['+y_vector[key]['unit']+']')
            ax.set_ylim(np.min(y_vector[key]['10th']),
                        np.max(y_vector[key]['90th']))
            ax.xaxis.set_major_locator(MaxNLocator(5))
            ax.yaxis.set_major_locator(MaxNLocator(5))
            ax.set_xticks(ticks=[-500,-250,0,250,500])
            ax.text(-0.5, 1.2, corner_text, transform=ax.transAxes, size=9)
            import matplotlib.ticker as ticker
            cbar=fig.colorbar(im, format=ticker.FuncFormatter(fmt), ax=ax)
            cbar.ax.tick_params(labelsize=6)


        for key in ['Velocity ccf radial','Distance']:
            if key == 'Velocity ccf radial':
                ax=ax2
                corner_text='(b)'
            else:
                ax=ax4
                corner_text='(d)'
            # nwin=y_vector[i]['Data'].shape[0]//2
    #        plt.plot(y_vector[i]['Bins'], np.mean(y_vector[i]['Data'][nwin-2:nwin+3,:], axis=0))
            ax.plot(time_vec*1e3,
                     y_vector[key]['median'],
                     color='red',
                     lw=linewidth)

            ax.set_title('Median '+y_vector[key]['ylabel']+' evolution')
            ax.set_xlabel('$t-t_{ELM}$ [$\mu$s]')
            ax.set_ylabel(y_vector[key]['ylabel']+' ['+y_vector[key]['unit']+']')
            ax.set_xlim([-500,500])
            ax.xaxis.set_major_locator(MaxNLocator(5))
            ax.set_xticks(ticks=[-500,-250,0,250,500])
            ax.yaxis.set_major_locator(MaxNLocator(5))
            ax.text(-0.5, 1.2, corner_text, transform=ax.transAxes, size=9)
        plt.tight_layout(pad=0.1)
        pdf_object.savefig()
        pdf_object.close()
        matplotlib.use('qt5agg')

        if save_data_into_txt:
            for key in ['Velocity ccf radial','Distance']:
                if key == 'Velocity ccf radial':
                    filename=wd+fig_dir+'/data_accessibility/figure_9ab.txt'
                if key == 'Distance':
                    filename=wd+fig_dir+'/data_accessibility/figure_9cd.txt'
                file1=open(filename, 'w+')

                file1.write('#Time (ms)\n')
                for i in range(1, len(time_vec)):
                    file1.write(str(time_vec[i])+'\t')

                file1.write('\n#Median\n')
                for i in range(1, len(y_vector[key]['median'])):
                    file1.write(str(y_vector[key]['median'])+'\t')

                file1.write('\n#10th perc.\n')
                for i in range(1, len(y_vector[key]['10th'])):
                    file1.write(str(y_vector[key]['10th'])+'\t')

                file1.write('\n#90th perc.\n')
                for i in range(1, len(y_vector[key]['90th'])):
                    file1.write(str(y_vector[key]['90th'])+'\t')

                file1.write('\n#Distribution bins\n')
                for i in range(1, len(y_vector[key]['bins'])):
                    file1.write(str(y_vector[key]['bins'])+'\t')
                file1.write('\n#Distribution\n')

                for i in range(len(y_vector[key]['data'][0,:])):
                    string=''
                    for j in range(len(y_vector[key]['data'][:,0])):
                        string+=str(y_vector[key]['data'][j,i])+'\t'
                    string+='\n'
                    file1.write(string)
                file1.close()

    """
    Dependence on the r-r_sep and v_rad vs. rotation
    """
    if plot_figure == 10:
        plot_angular_vs_translational_velocity(tau_range=[-200e-6,200e-6],
                                               subtraction_order=2,
                                               plot_for_pop_paper=True,
                                               plot_log_omega=True,
                                               correlation_threshold=0.7,
                                               figure_filename=wd+fig_dir+'/fig_parameter_dependence.pdf')

        #No need for data accessibility txt file because the data are in the previous txt files.
    """
    Dependence on plasma parameters
    """
    if plot_figure == 11:
        plot_gpi_profile_dependence_ultimate(plot_correlation_matrix=True,
                                             pdf=True,
                                             plot=True,
                                             recalc_gpi=False,
                                             recalc_thomson=False,
                                             thomson_time_window=5e-3,
                                             skip_uninteresting=True,
                                             elm_window=200e-6,
                                             plot_error=False,
                                             throw_outliers=True,
                                             flux_range=[0.65,1.1],
                                             threshold_corr=True,
                                             corr_thres_multiplier=1.,
                                             n_sigma=2)

    if plot_figure == 12:
        calculate_shear_layer_vpol(nocalc=True,
                                   shear_avg_t_elm_range=[-5e-3,-200e-6],
                                   sg_filter_order=21,
                                   test=False,
                                   plot_shear_profile=True,
                                   shot_to_plot=141319,
                                   save_data_for_publication=False,
                                   return_results=False)


    """
    Shear induced filament rotation sketch
    """
    if plot_figure == 13:
        raise ValueError('This is the sketch of the filament rotation mechanism')


    """
    Form factors vs. experimental observations
    """
    if plot_figure == 14:
        model_angular_velocity=calculate_shear_induced_angular_velocity(shear_avg_t_elm_range=[-5e-3,-200e-6],
                                                                        plot_error=False,
                                                                        test=False,
                                                                        std_thres_outlier=2.,
                                                                        return_results=True,
                                                                        plot_for_publication=False)


        res = analyze_shear_distribution(nocalc=True, return_results=True, pdf=True, n_hist=50)

        figsize=(8.5/2.54,8.5/2.54)
        plt.rc('font', family='serif', serif='Helvetica')
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


        time_vec, y_vector_rot = plot_nstx_gpi_angular_velocity_distribution(plot_for_publication=True,
                                                                             window_average=500e-6,
                                                                             subtraction_order=2,
                                                                             correlation_threshold=0.7,
                                                                             pdf=False,
                                                                             plot=False,
                                                                             return_results=True,
                                                                             plot_all_time_traces=False,
                                                                             tau_range=[-1e-3,1e-3],
                                                                             )

        pdf_object=PdfPages(wd+fig_dir+'/fig_model_ang_vel_distribution.pdf')


        import matplotlib
        if plot_for_analysis:
            matplotlib.use('qt5agg')
        else:
            matplotlib.use('agg')

        def fmt(x, pos):
            a = '{:3.2f}'.format(x)
            return a
        plt.figure()
        fig,ax_all=plt.subplots(2,2,figsize=figsize)

        ax=ax_all[0,0]
        time_vec=res['coord']['time']['data']*1e6

        im=ax.contourf(time_vec,
                    res['derived']['bins']['data'],
                    res['derived']['histogram']['data'].transpose())

        ax.plot(time_vec,
                 res['derived']['median']['data'],
                 color='red',
                 lw=linewidth)
        ax.plot(time_vec,
                 res['derived']['10th percentile']['data'],
                 color='white',
                 lw=linewidth/2)
        ax.plot(time_vec,
                res['derived']['90th percentile']['data'],
                color='white',
                lw=linewidth/2)

        ax.set_title('Relative frequency of $\omega_{model}$')
        ax.set_xlabel('$t-t_{ELM}$ [$\mu$s]')
        ax.set_ylabel('$\omega_{model} [krad/s]$')
        ax.set_ylim(np.min(res['derived']['10th percentile']['data']),
                    np.max(res['derived']['90th percentile']['data']))
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.set_xticks(ticks=[-500,-250,0,250,500])
        ax.text(-0.5, 1.2, '(a)', transform=ax.transAxes, size=9)
        import matplotlib.ticker as ticker
        cbar=fig.colorbar(im, format=ticker.FuncFormatter(fmt), ax=ax)
        cbar.ax.tick_params(labelsize=6)


        ax=ax_all[0,1]

        ax.plot(time_vec,
                res['derived']['median']['data'],
                color='red',
                lw=linewidth)

        ax.set_title('Median $\omega_{model}$ evolution')
        ax.set_xlabel('$t-t_{ELM}$ [$\mu$s]')
        ax.set_ylabel('$\omega_{model}$ [krad/s]')
        ax.set_xlim([-500,500])
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.set_xticks(ticks=[-500,-250,0,250,500])
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.text(-0.5, 1.2, '(b)', transform=ax.transAxes, size=9)
        plt.tight_layout(pad=0.1)

        plot_x_vs_y=[['Angular velocity experimental max neg','Shearing rate time dep max neg'],
                     ['Angular velocity experimental max pos','Shearing rate time dep max pos'],
                     ]

        for ind_plot in range(len(plot_x_vs_y)):
            if ind_plot == 0:
                ax=ax_all[1,0]
            else:
                ax=ax_all[1,1]
            xdata=model_angular_velocity['data'][plot_x_vs_y[ind_plot][0]]['data']/1e3
            ydata=model_angular_velocity['data'][plot_x_vs_y[ind_plot][1]]['data']/1e3

            ind_not_nan=np.logical_and(~np.isnan(xdata),
                                       ~np.isnan(ydata))

            xdata=xdata[ind_not_nan]
            ydata=ydata[ind_not_nan]

            xdata_std=np.sqrt(np.var(xdata))
            ydata_std=np.sqrt(np.var(ydata))

            ind_keep=np.where(np.logical_and(np.abs(xdata-np.mean(xdata))<xdata_std*3,
                                             np.abs(ydata-np.mean(ydata))<ydata_std*3))
            color='tab:blue'
            ax.scatter(xdata[ind_keep],
                       ydata[ind_keep],
                       marker='o',
                       color=color,
                       s=2)

            ax.set_title(plot_x_vs_y[ind_plot][0]+' vs.\n'+plot_x_vs_y[ind_plot][1])
            ax.set_xlabel(model_angular_velocity['data'][plot_x_vs_y[ind_plot][0]]['label']+' [k'+model_angular_velocity['data'][plot_x_vs_y[ind_plot][0]]['unit']+']')
            if model_angular_velocity['data'][plot_x_vs_y[ind_plot][1]]['unit'] =='':
                ax.set_ylabel(model_angular_velocity['data'][plot_x_vs_y[ind_plot][1]]['label'])
            else:
                ax.set_ylabel(model_angular_velocity['data'][plot_x_vs_y[ind_plot][1]]['label']+' [k'+model_angular_velocity['data'][plot_x_vs_y[ind_plot][1]]['unit']+']')

        plt.tight_layout(pad=0.1)

        pdf_object.savefig()
        pdf_object.close()
        matplotlib.use('qt5agg')

        if save_data_into_txt:
            filename=wd+fig_dir+'/data_accessibility/figure_16ab.txt'
            file1=open(filename, 'w+')

            file1.write('#Time (ms)\n')
            for i in range(1, len(time_vec)):
                file1.write(str(time_vec[i])+'\t')

            file1.write('\n#Median\n')
            for i in range(1, len(res['derived']['median']['data'])):
                file1.write(str(res['derived']['median']['data'][i])+'\t')

            file1.write('\n#10th perc.\n')
            for i in range(1, len(res['derived']['10th percentile']['data'])):
                file1.write(str(res['derived']['10th percentile']['data'][i])+'\t')

            file1.write('\n#90th perc.\n')
            for i in range(1, len(res['derived']['90th percentile']['data'])):
                file1.write(str(res['derived']['90th percentile']['data'][i])+'\t')

            file1.write('\n#Distribution bins\n')
            for i in range(1, len(res['derived']['bins']['data'])):
                file1.write(str(res['derived']['bins']['data'][i])+'\t')

            file1.write('\n#Distribution\n')
            for i in range(len(res['derived']['histogram']['data'][0,:])):
                string=''
                for j in range(len(res['derived']['histogram']['data'][:,0])):
                    string+=str(res['derived']['histogram']['data'][j,i])+'\t'
                string+='\n'
                file1.write(string)
            file1.close()