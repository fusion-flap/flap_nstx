#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  6 14:35:54 2020

@author: mlampert
"""
import os
import copy
import pickle

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec

from flap_nstx.analysis import calculate_nstx_gpi_frame_by_frame_velocity, flap_nstx_thomson_data
from flap_nstx.analysis import calculate_radial_acceleration_diagram, plot_nstx_gpi_velocity_distribution
from flap_nstx.analysis import nstx_gpi_generate_synthetic_data, test_spatial_displacement_estimation, show_nstx_gpi_video_frames

import flap
import flap_nstx

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
        
        
def plot_results_for_rsi_2021_paper(plot_figure=1, 
                                    save_data_into_txt=False):
    
    
    wd=flap.config.get_all_section('Module NSTX_GPI')['Working directory']
    if plot_figure == 1 or plot_figure == 2:
        pickle_file=wd+'/processed_data/2021_rsi_fig12.pickle'
        try:
            d1,d2,d3,d4=pickle.load(open(pickle_file,'rb'))
            flap.add_data_object(d1, 'GPI_SLICED_FULL')
            flap.add_data_object(d2, 'GPI_GAS_CLOUD')
            flap.add_data_object(d3, 'GPI_SLICED_DENORM_CCF_VEL')
            flap.add_data_object(d4, 'GPI_CCF_F_BY_F')
        except:
            calculate_nstx_gpi_frame_by_frame_velocity(exp_id=141319, 
                                                  time_range=[0.552497-500e-6,0.552497+500e-6], 
                                                  plot=False,
                                                  subtraction_order_for_velocity=4,
                                                  skip_structure_calculation=False,
                                                  correlation_threshold=0.,
                                                  pdf=False, 
                                                  nlevel=51, 
                                                  nocalc=False, 
                                                  filter_level=3, 
                                                  normalize_for_size=True,
                                                  normalize_for_velocity=True,
                                                  threshold_coeff=1.,
                                                  normalize_f_high=1e3, 
                                                  
                                                  normalize='roundtrip', 
                                                  velocity_base='cog', 
                                                  return_results=False, 
                                                  plot_gas=True)
            
            pickle.dump((flap.get_data_object('GPI_SLICED_FULL'),
                         flap.get_data_object('GPI_GAS_CLOUD'),
                         flap.get_data_object('GPI_SLICED_DENORM_CCF_VEL'),
                         flap.get_data_object('GPI_CCF_F_BY_F')), open(pickle_file, 'wb'))
            
    if plot_figure == 1:
        flap.get_data('NSTX_GPI',
                      exp_id=141319,
                      name='',
                      object_name='GPI')
        flap.slice_data('GPI', slicing={'Time':flap.Intervals(0.552497-500e-6,0.552497+500e-6)}, output_name='GPI_SLICED_FULL')
        data_object_name='GPI_SLICED_DENORM_CCF_VEL'
        detrended=flap_nstx.analysis.detrend_multidim(data_object_name,
                                                      exp_id=141319,
                                                      order=1, 
                                                      coordinates=['Image x', 'Image y'], 
                                                      output_name='GPI_DETREND_VEL')
        
        d=copy.deepcopy(flap.get_data_object(data_object_name))
        
        d.data=d.data-detrended.data
        flap.add_data_object(d,'GPI_TREND')
        
        signals=[data_object_name,
                 'GPI_TREND',
                 'GPI_DETREND_VEL']
        
        pdf=PdfPages(wd+'/plots/2021_rsi/figure_1_trend_subtraction.pdf')
        temp_vec=[flap.slice_data(signals[i],slicing={'Sample':20876}).data for i in range(3)]
        z_range=[np.min(temp_vec),
                 np.max(temp_vec)]
        gs=GridSpec(1,3)
        plt.figure()                
        ax,fig=plt.subplots(figsize=(8.5/2.54,2))
        for index_grid_x in range(3):
            plt.subplot(gs[0,index_grid_x])
            visibility=[True,True]
            if index_grid_x != 0:
                visibility[1]=False

            flap.plot(signals[index_grid_x], 
                      plot_type='contour', 
                      slicing={'Sample':20876},
                      axes=['Image x', 'Image y'],
                      options={'Z range':z_range,
                               'Interpolation': 'Closest value',
                               'Clear':False,
                               'Equal axes':True,
                               'Axes visibility':visibility,
                               'Colorbar':True,
                               },
                       plot_options={'levels':51},
                           )
            if save_data_into_txt:
                data=flap.get_data_object(signals[index_grid_x]).slice_data(slicing={'Sample':20876}).data
                filename=wd+'/data_accessibility/2021_rsi/figure_1_1_'+signals[index_grid_x]+'.txt'
                file1=open(filename, 'w+')
                for i in range(len(data[0,:])):
                    string=''
                    for j in range(len(data[:,0])):
                        string+=str(data[j,i])+'\t'
                    string+='\n'
                    file1.write(string)
                    
        # for index_grid_x in range(3):
        #     plt.subplot(gs[1,index_grid_x])
        #     visibility=[True,True]
        #     flap.plot(signals[index_grid_x], 
        #               plot_type='xy', 
        #               slicing={'Time':0.3249560, 'Image y':40},
        #               axes=['Image x'],
        #               options={'Interpolation': 'Closest value',
        #                        'Clear':False,
        #                        'Axes visibility':visibility,
        #                        }
        #                    )
        #     if index_grid_x == 0:
        #         print(np.sum(flap.slice_data(signals[index_grid_x], slicing={'Time':0.3249560}).data))
                
        #     if save_data_into_txt:
        #         data=flap.get_data_object(signals[index_grid_x]).slice_data(slicing={'Time':0.3249560}).data
        #         filename=wd+'/data_accessibility/2021_rsi/figure_1_2_'+signals[index_grid_x]+'.txt'
        #         file1=open(filename, 'w+')
        #         for i in range(len(data[0,:])):
        #             string=''
        #             for j in range(len(data[:,0])):
        #                 string+=str(data[j,i])+'\t'
        #             string+='\n'
        #             file1.write(string)
        pdf.savefig()
        pdf.close()
        
    if plot_figure == 2:
        pdf=PdfPages(wd+'/plots/2021_rsi/figure_frame_by_frame.pdf')
        gs=GridSpec(1,3)
        plt.figure()                
        ax,fig=plt.subplots(figsize=(8.5/2.54,2))
        plt.subplot(gs[0])
        flap.plot('GPI_SLICED_FULL', 
                  plot_type='contour', 
                  slicing={'Sample':20876}, 
                  axes=['Image x', 'Image y'],
                  options={'Z range':[0,4096],
                           'Interpolation': 'Closest value',
                           'Clear':False,
                           'Equal axes':True,
                           'Axes visibility':[True,True],
                           'Colorbar':True,
                           },
                   plot_options={'levels':51},
                       )
        plt.title("552.497ms")
        if save_data_into_txt:
            data=flap.get_data_object('GPI_SLICED_FULL').slice_data(slicing={'Sample':20876}).data
            filename=wd+'/data_accessibility/2021_rsi/figure_2a.txt'
            file1=open(filename, 'w+')
            for i in range(len(data[0,:])):
                string=''
                for j in range(len(data[:,0])):
                    string+=str(data[j,i])+'\t'
                string+='\n'
                file1.write(string)
            file1.close()
        plt.subplot(gs[1])
        flap.plot('GPI_SLICED_FULL', 
                   plot_type='contour', 
                   slicing={'Sample':20877}, 
                   axes=['Image x', 'Image y'],
                   options={'Z range':[0,4096],
                            'Interpolation': 'Closest value',
                            'Clear':False,
                            'Equal axes':True,
                            'Axes visibility':[True,False],
                            'Colorbar':True,
                            },
                    plot_options={'levels':51},
                    )
        plt.title("552.500ms")
        if save_data_into_txt:
            data=flap.get_data_object('GPI_SLICED_FULL').slice_data(slicing={'Sample':20877}).data
            filename=wd+'/data_accessibility/2021_rsi/figure_2b.txt'
            file1=open(filename, 'w+')
            for i in range(len(data[0,:])):
                string=''
                for j in range(len(data[:,0])):
                    string+=str(data[j,i])+'\t'
                string+='\n'
                file1.write(string)
            file1.close()
            
        plt.subplot(gs[2])
        flap.plot('GPI_CCF_F_BY_F', 
                  plot_type='contour', 
                  slicing={'Sample':20877,
                           'Image x':flap.Intervals(-32,32), #, 'Image x':flap.Intervals(-10,10),'Image y':flap.Intervals(-10,10)}, 
                           'Image y':flap.Intervals(-40,40),},
                  axes=['Image x', 'Image y'],
                  options={
                          #'Z range':[0,2048],
                           'Interpolation': 'Closest value',
                           'Clear':False,
                           'Equal axes':True,
                           'Axes visibility':[True,True],
                           #'Colormap':colormap,
                           'Colorbar':True,
                           #'Overplot options':oplot_options,
                           },
                   plot_options={'levels':51},
                       )
        plt.title("CCF")
        
        if save_data_into_txt:
            data=flap.get_data_object('GPI_CCF_F_BY_F').slice_data(slicing={'Sample':20877}).data
            filename=wd+'/data_accessibility/2021_rsi/figure_2c.txt'
            file1=open(filename, 'w+')
            for i in range(len(data[0,:])):
                string=''
                for j in range(len(data[:,0])):
                    string+=str(data[j,i])+'\t'
                string+='\n'
                file1.write(string)
            file1.close()
        
        
        pdf.savefig()
        pdf.close()
    
    if plot_figure == 3:
        time_range=[0.34725,0.34775]
        #calculate_nstx_gpi_avg_frame_velocity(exp_id=141319,
        #                              time_range=[0.552,0.553],  
        calculate_nstx_gpi_frame_by_frame_velocity(exp_id=139901,
                                                   time_range=time_range,  
                                                   normalize='roundtrip', 
                                                   normalize_for_size=True, 
                                                   skip_structure_calculation=True, 
                                                   plot=False, 
                                                   pdf=False,
                                                   nocalc=True,
                                                   plot_scatter=False,
                                                   plot_for_publication=True,
                                                   correlation_threshold=0.0,
                                                   return_results=True,
                                                   subtraction_order_for_velocity=4)
        corr_thres=np.arange(11)/10
        for i_corr in range(11):
            results=calculate_nstx_gpi_frame_by_frame_velocity(exp_id=139901,
                                                               time_range=time_range,  
                                                               normalize='roundtrip', 
                                                               normalize_for_size=True, 
                                                               skip_structure_calculation=False, 
                                                               plot=False, 
                                                               pdf=False,
                                                               nocalc=True,
                                                               plot_scatter=False,
                                                               plot_for_publication=True,
                                                               correlation_threshold=corr_thres[i_corr],
                                                               return_results=True,
                                                               subtraction_order_for_velocity=4)
            print(results['Velocity ccf'][:,0].shape)
            time=results['Time']
            if i_corr==0:
                pol_vel=np.zeros([len(results['Velocity ccf'][:,0]),11])
                rad_vel=np.zeros([len(results['Velocity ccf'][:,0]),11])
            pol_vel[:,i_corr]=results['Velocity ccf'][:,0]
            rad_vel[:,i_corr]=results['Velocity ccf'][:,1]
        
        pdf=PdfPages(wd+'/plots/2021_rsi/figure_vel_vs_corr_thres.pdf')
        
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
        plt.figure()                
        ax,fig=plt.subplots(figsize=(8.5/2.54,2))
        for i in range(0,7,1):
            plt.plot(time, pol_vel[:,(10-i)]/1e3-i*10, label=str((10-i)/10))
            
        plt.title('Poloidal velocity vs. \n correlation threshold')
        plt.xlabel('Time [s]')
        plt.ylabel('vpol [m/s]')
        #plt.legend()
        
        pdf.savefig()
        pdf.close()
        
        if save_data_into_txt:
            filename=wd+'/data_accessibility/2021_rsi/figure_3.txt'
            file1=open(filename, 'w+')
            string='Time [s]'
            for i in range(1,7,1):
                string+='\t rho='+str((10-i)/10)
            string+='\n'
            file1.write(string)
            for j in range(len(time)):
                string=str(time[j])
                for i in range(1,7,1):
                    string+='\t'+str(pol_vel[j,(10-i)]/1e3-i*10)
                string+='\n'
                file1.write(string)
            file1.close()
        
    
    if plot_figure == 4:
         test_spatial_displacement_estimation(plot_sample_gaussian=True, 
                                             pdf=True,
                                             save_data_into_txt=save_data_into_txt)
         
         if save_data_into_txt:
            data=flap.get_data_object('gaussian').slice_data(slicing={'Sample':0}).data
            filename=wd+'/data_accessibility/2021_rsi/figure_4a.txt'
            file1=open(filename, 'w+')
            for i in range(len(data[0,:])):
                string=''
                for j in range(len(data[:,0])):
                    string+=str(data[j,i])+'\t'
                string+='\n'
                file1.write(string)
            file1.close()
            
            data=flap.get_data_object('gaussian').slice_data(slicing={'Sample':1}).data
            filename=wd+'/data_accessibility/2021_rsi/figure_4b.txt'
            file1=open(filename, 'w+')
            for i in range(len(data[0,:])):
                string=''
                for j in range(len(data[:,0])):
                    string+=str(data[j,i])+'\t'
                string+='\n'
                file1.write(string)
            file1.close()    
            
            data=flap.get_data_object('GPI_FRAME_12_CCF').slice_data(slicing={'Sample':0}).data
            filename=wd+'/data_accessibility/2021_rsi/figure_4c.txt'
            file1=open(filename, 'w+')
            for i in range(len(data[0,:])):
                string=''
                for j in range(len(data[:,0])):
                    string+=str(data[j,i])+'\t'
                string+='\n'
                file1.write(string)
            file1.close()  
    
    if plot_figure == 5:
        
        test_spatial_displacement_estimation(gaussian_frame_vs_structure_size=True, 
                                             gaussian_frame_size=True, 
                                             gaussian=True, 
                                             interpolation='parabola', 
                                             pdf=True, 
                                             nocalc=True, 
                                             frame_size_range=[8,200], 
                                             frame_size_step=8,
                                             save_data_into_txt=save_data_into_txt)
    if plot_figure == 6:
        test_spatial_displacement_estimation(plot_sample_random=True, 
                                             pdf=True,
                                             save_data_into_txt=save_data_into_txt)
        
        if save_data_into_txt:
            data=flap.get_data_object('random').slice_data(slicing={'Sample':0}).data
            filename=wd+'/data_accessibility/2021_rsi/figure_6a.txt'
            file1=open(filename, 'w+')
            for i in range(len(data[0,:])):
                string=''
                for j in range(len(data[:,0])):
                    string+=str(data[j,i])+'\t'
                string+='\n'
                file1.write(string)
            file1.close()
            
            data=flap.get_data_object('random').slice_data(slicing={'Sample':1}).data
            filename=wd+'/data_accessibility/2021_rsi/figure_6b.txt'
            file1=open(filename, 'w+')
            for i in range(len(data[0,:])):
                string=''
                for j in range(len(data[:,0])):
                    string+=str(data[j,i])+'\t'
                string+='\n'
                file1.write(string)
            file1.close()    
            
            data=flap.get_data_object('GPI_FRAME_12_CCF').slice_data(slicing={'Sample':0}).data
            filename=wd+'/data_accessibility/2021_rsi/figure_6c.txt'
            file1=open(filename, 'w+')
            for i in range(len(data[0,:])):
                string=''
                for j in range(len(data[:,0])):
                    string+=str(data[j,i])+'\t'
                string+='\n'
                file1.write(string)
            file1.close()  
    
        
    if plot_figure == 7:
        test_spatial_displacement_estimation(random=True, 
                                             pdf=True, 
                                             nocalc=False,
                                             save_data_into_txt=save_data_into_txt)
        
    if plot_figure == 8:
        plt.figure()
        ax,fig=plt.subplots(figsize=(3.35*2,5.5))
        pdf=PdfPages(wd+'/plots/2021_rsi/figure_141319_0.552497_9_frame.pdf')
        
        show_nstx_gpi_video_frames(exp_id=141319, 
                                   start_time=0.552497-5*2.5e-6,
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
                                   colorbar_visibility=True,
                                   save_data_for_publication=save_data_into_txt
                                   )
        pdf.savefig()
        pdf.close()
    
    if plot_figure == 9:
        results=calculate_nstx_gpi_frame_by_frame_velocity(exp_id=141319,
                                                           time_range=[0.552,0.553],  
                                                           normalize='roundtrip', 
                                                           normalize_for_size=True, 
                                                           normalize_for_velocity=False,
                                                           skip_structure_calculation=False, 
                                                           plot=True, 
                                                           pdf=True,
                                                           nocalc=False,
                                                           plot_scatter=False,
                                                           plot_for_publication=True,
                                                           remove_interlaced_structures=True,
                                                           return_results=True,
                                                           subtraction_order_for_velocity=4)

        if save_data_into_txt:
            time=results['Time']
            filename=wd+'/data_accessibility/2021_rsi/figure_9.txt'
            file1=open(filename, 'w+')
            string='Time [s] \t v_rad_ccf \t v_rad_str \t v_pol_ccf \t v_pol_str \n'
            file1.write(string)
            for i in range(len(time)):
                string=str(time[i])+'\t'+\
                        str(results['Velocity ccf'][i,0])+'\t'+\
                        str(results['Velocity str max'][i,0])+'\t'+\
                        str(results['Velocity ccf'][i,1])+'\t'+\
                        str(results['Velocity str max'][i,1])+'\n'
                file1.write(string)
            file1.close()