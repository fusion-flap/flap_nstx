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
fn = os.path.join(thisdir,"flap_nstx.cfg")
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


def plot_results_for_pop_2020_paper(plot_figure=2, 
                                    save_data_into_txt=False):

    wd=flap.config.get_all_section('Module NSTX_GPI')['Working directory']

    
    from flap_nstx.analysis import show_nstx_gpi_video_frames

    if plot_figure == 2:
        #shot_number=139901                                                     #Original ones for the paper
        #time_range=[0.25,0.4]
        
        shot_number=141319
        time_range=[0.5,0.6]
        
        gs=GridSpec(5,2)
        ax,fig=plt.subplots(figsize=(8.5/2.54,6))
        pdf=PdfPages(wd+'/plots/figure_3_'+str(shot_number)+'_basic_plots.pdf')
        
        plt.subplot(gs[0,0])
        d=flap.get_data('NSTX_MDSPlus',
                      name='\WF::\DALPHA',
                      exp_id=shot_number,
                      object_name='DALPHA')
        d.plot(options={'Axes visibility':[False,True]})
        if save_data_into_txt:
            time=d.coordinate('Time')[0]
            data=d.data
            filename=wd+'/2020_pop_data_accessibility/figure_1a.txt'
            file1=open(filename, 'w+')
            for i in range(len(time)):
                file1.write(str(time[i])+'\t'+str(data[i])+'\n')
            file1.close()
            
        plt.xlim([0,1.2])
        plt.subplot(gs[1,0])
        d=flap.get_data('NSTX_GPI',
                      name='',
                      exp_id=shot_number,
                      object_name='GPI').slice_data(summing={'Image x':'Mean', 'Image y':'Mean'})
        d.plot(options={'Axes visibility':[False,True]})
        plt.xlim([0,1.2])

        if save_data_into_txt:
            time=d.coordinate('Time')[0]
            data=d.data
            filename=wd+'/2020_pop_data_accessibility/figure_1_b.txt'
            file1=open(filename, 'w+')
            for i in range(len(time)):
                file1.write(str(time[i])+'\t'+str(data[i])+'\n')
            file1.close()
                    

        plt.subplot(gs[2,0])
        d=flap.get_data('NSTX_MDSPlus',
                      name='IP',
                      exp_id=shot_number,
                      object_name='IP')
        d.plot(options={'Axes visibility':[False,True]})
        plt.xlim([0,1.2])
        
        if save_data_into_txt:
            time=d.coordinate('Time')[0]
            data=d.data
            filename=wd+'/2020_pop_data_accessibility/figure_1c.txt'
            file1=open(filename, 'w+')
            for i in range(len(time)):
                file1.write(str(time[i])+'\t'+str(data[i])+'\n')
            file1.close()
        
        plt.subplot(gs[3,0])
        d=flap_nstx_thomson_data(exp_id=shot_number, density=True, output_name='DENSITY')

        LID=np.trapz(d.data[:,:], d.coordinate('Device R')[0][:,:], axis=0)/(np.max(d.coordinate('Device R')[0][:,:],axis=0)-np.min(d.coordinate('Device R')[0][:,:],axis=0))
        plt.plot(d.coordinate('Time')[0][0,:],LID)
        plt.title('Line averaged density')
        plt.xlabel('Time [s]')
        plt.ylabel('n_e [m^-3]')
        plt.xlim([0,1.2])
        ax=plt.gca()
        ax.get_xaxis().set_visible(False)
        
        if save_data_into_txt:
            time=d.coordinate('Time')[0][0,:]
            data=LID
            filename=wd+'/2020_pop_data_accessibility/figure_1d.txt'
            file1=open(filename, 'w+')
            for i in range(len(time)):
                file1.write(str(time[i])+'\t'+str(data[i])+'\n')
            file1.close()
        
        plt.subplot(gs[4,0])
        magnetics=flap.get_data('NSTX_MDSPlus',
                                name='\OPS_PC::\\BDOT_L1DMIVVHF5_RAW',
                                exp_id=shot_number,
                                object_name='MIRNOV')
        
        magnetics.coordinates.append(copy.deepcopy(flap.Coordinate(name='Time equi',
                                           unit='s',
                                           mode=flap.CoordinateMode(equidistant=True),
                                           shape = [],
                                           start=magnetics.coordinate('Time')[0][0],
                                           step=magnetics.coordinate('Time')[0][1]-magnetics.coordinate('Time')[0][0],
                                           dimension_list=[0])))
        
        d=magnetics.filter_data(coordinate='Time equi',
                              options={'Type':'Bandpass',
                                       'f_low':100e3,
                                       'f_high':500e3,
                                       'Design':'Elliptic'})
        d.plot()
        plt.xlim([0,1.2])
        
        if save_data_into_txt:
            time=d.coordinate('Time')[0]
            data=d.data
            filename=wd+'/2020_pop_data_accessibility/figure_1e.txt'
            file1=open(filename, 'w+')
            for i in range(len(time)):
                file1.write(str(time[i])+'\t'+str(data[i])+'\n')
            file1.close()
        
        plt.subplot(gs[0,1])
        flap.get_data('NSTX_MDSPlus',
                      name='\WF::\DALPHA',
                      exp_id=shot_number,
                      object_name='DALPHA').plot(options={'Axes visibility':[False,True]})
        plt.xlim(time_range)
        plt.ylim([0,3])
        
        plt.subplot(gs[1,1])
        flap.get_data('NSTX_GPI',
                      name='',
                      exp_id=shot_number,
                      object_name='GPI').slice_data(summing={'Image x':'Mean', 'Image y':'Mean'}).plot(options={'Axes visibility':[False,True]})
        plt.xlim(time_range)
        
        plt.subplot(gs[2,1])
        flap.get_data('NSTX_MDSPlus',
                      name='IP',
                      exp_id=shot_number,
                      object_name='IP').plot(options={'Axes visibility':[False,True]})
        plt.xlim(time_range)
        
        plt.subplot(gs[3,1])
        d=flap_nstx_thomson_data(exp_id=shot_number, density=True, output_name='DENSITY')

        LID=np.trapz(d.data[:,:], d.coordinate('Device R')[0][:,:], axis=0)/(np.max(d.coordinate('Device R')[0][:,:],axis=0)-np.min(d.coordinate('Device R')[0][:,:],axis=0))
        plt.plot(d.coordinate('Time')[0][0,:],LID)
        plt.title('Line integrated density')
        plt.xlabel('Time [s]')
        plt.ylabel('n_e [m^-3]')
        plt.xlim(time_range)
        ax=plt.gca()
        ax.get_xaxis().set_visible(False)
        
        plt.subplot(gs[4,1])
        magnetics=flap.get_data('NSTX_MDSPlus',
                                name='\OPS_PC::\\BDOT_L1DMIVVHF5_RAW',
                                exp_id=shot_number,
                                object_name='MIRNOV')
        
        magnetics.coordinates.append(copy.deepcopy(flap.Coordinate(name='Time equi',
                                           unit='s',
                                           mode=flap.CoordinateMode(equidistant=True),
                                           shape = [],
                                           start=magnetics.coordinate('Time')[0][0],
                                           step=magnetics.coordinate('Time')[0][1]-magnetics.coordinate('Time')[0][0],
                                           dimension_list=[0])))
        
        magnetics.filter_data(coordinate='Time equi',
                              options={'Type':'Bandpass',
                                       'f_low':100e3,
                                       'f_high':500e3,
                                       'Design':'Elliptic'}).plot(slicing={'Time':flap.Intervals(0.25,0.4)})
    
        plt.xlim(time_range)
        ax=plt.gca()
        pdf.savefig()
        pdf.close()
        
            
        
    if plot_figure == 3:
        plt.figure()
        ax,fig=plt.subplots(figsize=(3.35*2,5.5))
        pdf=PdfPages(wd+'/plots/figure_5_139901_0.3249158_30_frame.pdf')
        show_nstx_gpi_video_frames(exp_id=139901, 
                                   start_time=0.3249158,
                                   n_frame=30,
                                   logz=False,
                                   z_range=[0,3900],
                                   plot_filtered=False, 
                                   normalize=False,
                                   cache_data=False, 
                                   plot_flux=False, 
                                   plot_separatrix=False, 
                                   flux_coordinates=False,
                                   device_coordinates=False,
                                   new_plot=False,
                                   save_pdf=True,
                                   colormap='gist_ncar',
                                   save_for_paraview=False,
                                   colorbar_visibility=False,
                                   save_data_for_publication=save_data_into_txt
                                   )
        pdf.savefig()
        pdf.close()

    if plot_figure == 4 or plot_figure == 5 or plot_figure == 6:
        try:
            d1,d2,d3,d4=pickle.load(open(wd+'/processed_data/fig_6_8_flap_object.pickle','rb'))
            flap.add_data_object(d1, 'GPI_SLICED_FULL')
            flap.add_data_object(d2, 'GPI_GAS_CLOUD')
            flap.add_data_object(d3, 'GPI_SLICED_DENORM_CCF_VEL')
            flap.add_data_object(d4, 'GPI_CCF_F_BY_F')
        except:
            calculate_nstx_gpi_frame_by_frame_velocity(exp_id=139901, 
                                                  time_range=[0.325-1e-3,0.325+1e-3], 
                                                  plot=False,
                                                  subtraction_order_for_velocity=1,
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
                         flap.get_data_object('GPI_CCF_F_BY_F')), open(wd+'/processed_data/fig_6_8_flap_object.pickle','wb'))
    if plot_figure == 4:
        pdf=PdfPages(wd+'/plots/figure_6_normalization.pdf')
        times=[0.3245,0.3249560,0.3255]
        signals=['GPI_SLICED_FULL',
                 'GPI_GAS_CLOUD',
                 'GPI_SLICED_DENORM_CCF_VEL']
        gs=GridSpec(3,3)
        plt.figure()                
        ax,fig=plt.subplots(figsize=(3.35,4))
        titles=['Raw frame', 'Gas cloud', 'Normalized']
        for index_grid_x in range(3):
            for index_grid_y in range(3):
                plt.subplot(gs[index_grid_x,index_grid_y])
                visibility=[True,True]
                if index_grid_x != 3-1:
                    visibility[0]=False
                if index_grid_y != 0:
                    visibility[1]=False
                z_range=None
                flap.plot(signals[index_grid_x], 
                          plot_type='contour', 
                          slicing={'Time':times[index_grid_y]}, 
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
                    if index_grid_y==1:
                        data=flap.get_data_object(signals[index_grid_x]).slice_data(slicing={'Time':times[index_grid_y]}).data
                        filename=wd+'/2020_pop_data_accessibility/figure_4_'+titles[index_grid_x]+'.txt'
                        file1=open(filename, 'w+')
                        for i in range(len(data[0,:])):
                            string=''
                            for j in range(len(data[:,0])):
                                string+=str(data[j,i])+'\t'
                            string+='\n'
                            file1.write(string)
                        file1.close()
                if index_grid_x == 0:
                    #ax=plt.gca()
                    plt.title(f"{times[index_grid_y]*1e3:.3f}"+' '+titles[index_grid_x])
                else:
                    plt.title(titles[index_grid_x])
    
        pdf.savefig()
        pdf.close()
    

    if plot_figure == 5:
        flap.get_data('NSTX_GPI',exp_id=139901,
                      name='',
                      object_name='GPI')
        flap.slice_data('GPI', slicing={'Time':flap.Intervals(0.3245,0.3255)}, output_name='GPI_SLICED_FULL')
        data_object_name='GPI_SLICED_DENORM_CCF_VEL'
        detrended=flap_nstx.analysis.detrend_multidim(data_object_name,
                                                      exp_id=139901,
                                                      order=4, 
                                                      coordinates=['Image x', 'Image y'], 
                                                      output_name='GPI_DETREND_VEL')
        
        d=copy.deepcopy(flap.get_data_object(data_object_name))
        
        d.data=d.data-detrended.data
        flap.add_data_object(d,'GPI_TREND')
        
        signals=[data_object_name,
                 'GPI_TREND',
                 'GPI_DETREND_VEL']
        
        pdf=PdfPages(wd+'/plots/figure_7_trend_subtraction.pdf')

        gs=GridSpec(1,3)
        plt.figure()                
        ax,fig=plt.subplots(figsize=(8.5/2.54,2))
        for index_grid_x in range(3):
            plt.subplot(gs[index_grid_x])
            visibility=[True,True]
            if index_grid_x != 0:
                visibility[1]=False
            z_range=[0,10]
            flap.plot(signals[index_grid_x], 
                      plot_type='contour', 
                      slicing={'Time':0.3249560},
                      axes=['Image x', 'Image y'],
                      options={'Interpolation': 'Closest value',
                               'Clear':False,
                               'Equal axes':True,
                               'Axes visibility':visibility,
                               'Colorbar':True,
                               },
                       plot_options={'levels':51},
                           )
            if save_data_into_txt:
                data=flap.get_data_object(signals[index_grid_x]).slice_data(slicing={'Time':0.3249560}).data
                filename=wd+'/2020_pop_data_accessibility/figure_5_'+signals[index_grid_x]+'.txt'
                file1=open(filename, 'w+')
                for i in range(len(data[0,:])):
                    string=''
                    for j in range(len(data[:,0])):
                        string+=str(data[j,i])+'\t'
                    string+='\n'
                    file1.write(string)
        pdf.savefig()
        pdf.close()
        

    if plot_figure == 6:
        pdf=PdfPages(wd+'/plots/figure_8_CCF_frame_by_frame.pdf')
        gs=GridSpec(1,3)
        plt.figure()                
        ax,fig=plt.subplots(figsize=(8.5/2.54,2))
        plt.subplot(gs[0])
        flap.plot('GPI_SLICED_FULL', 
                  plot_type='contour', 
                  slicing={'Sample':29806}, 
                  axes=['Image x', 'Image y'],
                  options={
                          'Z range':[0,4096],
                           'Interpolation': 'Closest value',
                           'Clear':False,
                           'Equal axes':True,
                           'Axes visibility':[True,True],
                           'Colorbar':True,
                           },
                   plot_options={'levels':51},
                       )
        plt.title("324.956ms")
        if save_data_into_txt:
            data=flap.get_data_object('GPI_SLICED_FULL').slice_data(slicing={'Sample':29806}).data
            filename=wd+'/2020_pop_data_accessibility/figure_6a.txt'
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
                   slicing={'Sample':29807}, 
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
        plt.title("324.959ms")
        if save_data_into_txt:
            data=flap.get_data_object('GPI_SLICED_FULL').slice_data(slicing={'Sample':29807}).data
            filename=wd+'/2020_pop_data_accessibility/figure_6b.txt'
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
                  slicing={'Sample':29807}, #, 'Image x':flap.Intervals(-10,10),'Image y':flap.Intervals(-10,10)}, 
                  
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
            data=flap.get_data_object('GPI_CCF_F_BY_F').slice_data(slicing={'Sample':29807}).data
            filename=wd+'/2020_pop_data_accessibility/figure_6c.txt'
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
        
    if plot_figure == 7:
        calculate_nstx_gpi_frame_by_frame_velocity(exp_id=139901, 
                                              time_range=[0.32495,0.325], 
                                              plot=False,
                                              subtraction_order_for_velocity=4,
                                              skip_structure_calculation=False,
                                              remove_interlaced_structures=False,
                                              correlation_threshold=0.5,
                                              pdf=True, 
                                              nlevel=51, 
                                              nocalc=False, 
                                              filter_level=5, 
                                              normalize_for_size=True,
                                              normalize_for_velocity=True,
                                              threshold_coeff=1.,
                                              normalize_f_high=1e3, 
                                              normalize='roundtrip', 
                                              velocity_base='cog', 
                                              return_results=False, 
                                              plot_gas=True,
                                              structure_pixel_calc=True,
                                              structure_pdf_save=True,
                                              test_structures=True,
                                              save_data_for_publication=True,
                                              )


    if plot_figure == 8:
        calculate_nstx_gpi_frame_by_frame_velocity(exp_id=139901, 
                                              time_range=[0.325-2e-3,0.325+2e-3], 
                                              plot_time_range=[0.325-0.5e-3,0.325+0.5e-3],
                                              plot=True,
                                              subtraction_order_for_velocity=4,
                                              skip_structure_calculation=False,
                                              correlation_threshold=0.6,
                                              pdf=True, 
                                              nlevel=51, 
                                              nocalc=True, 
                                              gpi_plane_calculation=True,
                                              filter_level=5, 
                                              normalize_for_size=True,
                                              normalize_for_velocity=True,
                                              threshold_coeff=1.,
                                              normalize_f_high=1e3, 
                                              normalize='roundtrip', 
                                              velocity_base='cog', 
                                              return_results=False, 
                                              plot_gas=True,
                                              plot_for_publication=True,
                                              plot_scatter=False,
                                              overplot_average=False,
                                              overplot_str_vel=False,
                                              save_data_for_publication=True,
                                              )

    if plot_figure == 9:
        plot_nstx_gpi_velocity_distribution(n_hist=50, 
                                            correlation_threshold=0.6, 
                                            nocalc=True, 
                                            general_plot=False, 
                                            plot_for_velocity=True)
        
    if plot_figure == 10:
        plot_nstx_gpi_velocity_distribution(n_hist=50, 
                                            correlation_threshold=0.6, 
                                            nocalc=True, 
                                            general_plot=False,
                                            plot_for_structure=True)
    
    if plot_figure == 11:
        plot_nstx_gpi_velocity_distribution(n_hist=50, 
                                            correlation_threshold=0.6, 
                                            nocalc=True, 
                                            general_plot=False, 
                                            plot_for_dependence=True) 
    
    if plot_figure == 12:
        calculate_radial_acceleration_diagram(elm_window=500e-6,
                                              calculate_dependence=True,
                                              elm_duration=100e-6,
                                              correlation_threshold=0.6,                                              
                                              elm_time_base='frame similarity',     #'radial acceleration', 'radial velocity', 'frame similarity'
                                              acceleration_base='linefit',           #numdev or linefit
                                              calculate_thick_wire=False,
                                              delta_b_threshold=1,                                              
                                              plot=False,
                                              plot_velocity=False,                                              
                                              auto_x_range=True,
                                              auto_y_range=True,
                                              plot_error=True,
                                              plot_clear_peak=False,                                              
                                              recalc=True,
                                              )
    if plot_figure == 13:
        calculate_radial_acceleration_diagram(elm_window=500e-6,
                                              calculate_acceleration=True,
                                              elm_duration=100e-6,
                                              correlation_threshold=0.6,                                              
                                              elm_time_base='frame similarity',     #'radial acceleration', 'radial velocity', 'frame similarity'
                                              acceleration_base='linefit',           #numdev or linefit
                                              calculate_thick_wire=False,
                                              delta_b_threshold=1,                                              
                                              plot=False,
                                              plot_velocity=False,                                              
                                              auto_x_range=True,
                                              auto_y_range=True,
                                              plot_error=True,
                                              plot_clear_peak=False,                                              
                                              recalc=True,
                                              )
    if plot_figure == 14:
        calculate_radial_acceleration_diagram(elm_window=500e-6,
                                              calculate_ion_drift_velocity=True,
                                              elm_duration=100e-6,
                                              correlation_threshold=0.6,                                              
                                              elm_time_base='frame similarity',     #'radial acceleration', 'radial velocity', 'frame similarity'
                                              acceleration_base='linefit',           #numdev or linefit
                                              calculate_thick_wire=False,
                                              delta_b_threshold=1,                                              
                                              plot=False,
                                              plot_velocity=False,                                              
                                              auto_x_range=True,
                                              auto_y_range=True,
                                              plot_error=True,
                                              plot_clear_peak=False,                                              
                                              recalc=True,
                                              )
            
    if plot_figure == 15:
        nstx_gpi_generate_synthetic_data(exp_id=1, 
                                         time=0.0001, 
                                         amplitude=1.0, 
                                         output_name='test', 
                                         poloidal_velocity=3e3, 
                                         radial_velocity=0., 
                                         poloidal_size=0.10,
                                         radial_size=0.05, 
                                         waveform_divider=1, 
                                         sinusoidal=True)
        d=flap.get_data_object('test', exp_id=1)
        d.data=d.data-np.mean(d.data,axis=0)
        calculate_nstx_gpi_frame_by_frame_velocity(data_object='test',
                                              exp_id=1,
                                              time_range=[0.000000,0.00005],
                                              plot=False,
                                              subtraction_order_for_velocity=1,
                                              skip_structure_calculation=False,
                                              correlation_threshold=0.5,
                                              pdf=True, 
                                              nlevel=51, 
                                              nocalc=False, 
                                              filter_level=5, 
                                              normalize_for_size=False,
                                              normalize_for_velocity=False,
                                              threshold_coeff=1.,
                                              normalize_f_high=1e3, 
                                              normalize=None, 
                                              velocity_base='cog', 
                                              return_results=False, 
                                              plot_gas=False,
                                              structure_pixel_calc=True,
                                              structure_pdf_save=True,
                                              test_structures=True
                                              )
        
        
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