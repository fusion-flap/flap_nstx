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

from flap_nstx.analysis import calculate_nstx_gpi_avg_frame_velocity, calculate_nstx_gpi_smooth_velocity, flap_nstx_thomson_data
from flap_nstx.analysis import nstx_gpi_velocity_analysis_spatio_temporal_displacement, plot_all_parameters_vs_all_other

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

plot=[False, #Figure 0
      False, #Figure 1
      False, #Figure 2
      True, #Figure 3
      False, #Figure 4
      False, #Figure 5
      False, #Figure 6
      False, #Figure 7
      False, #Figure 8
      False, #Figure 9
      False, #Figure 10
      False, #Figure 11
      False, #Figure 12
      False, #Figure 13
      False, #Figure 14
      False, #Figure 15
      False, #Figure 16
      ]
def plot_results_for_paper():
    pearson=False
    wd=flap.config.get_all_section('Module NSTX_GPI')['Working directory']
    #Figure 1
    '''NO CODE IS NEEDED'''
    #Figure 2
    '''NO CODE IS NEEDED'''
    
    #Figure 3
    from flap_nstx.analysis import show_nstx_gpi_video_frames
    #fig, ax = plt.subplots(figsize=(6.5,5))
    if plot[3]:
        gs=GridSpec(5,2)
        ax,fig=plt.subplots(figsize=(8.5/2.54,6))
        pdf=PdfPages(wd+'/plots/figure_3_139901_basic_plots.pdf')
        
        plt.subplot(gs[0,0])
        flap.get_data('NSTX_MDSPlus',
                      name='\WF::\DALPHA',
                      exp_id=139901,
                      object_name='DALPHA').plot(options={'Axes visibility':[False,True]})
        plt.xlim([0,1.2])
        plt.subplot(gs[1,0])
        flap.get_data('NSTX_GPI',
                      name='',
                      exp_id=139901,
                      object_name='GPI').slice_data(summing={'Image x':'Mean', 'Image y':'Mean'}).plot(options={'Axes visibility':[False,True]})
        plt.xlim([0,1.2])
        
        plt.xlim([0,1.2])
        plt.subplot(gs[2,0])
        flap.get_data('NSTX_MDSPlus',
                      name='IP',
                      exp_id=139901,
                      object_name='IP').plot(options={'Axes visibility':[False,True]})
        plt.xlim([0,1.2])
        plt.subplot(gs[3,0])
        d=flap_nstx_thomson_data(exp_id=139901, density=True, output_name='DENSITY')
        #dR = d.coordinate('Device R')[0][:,:]-np.insert(d.coordinate('Device R')[0][0:-1,:],0,0,axis=0)
        #LID=np.sum(d.data*dR,axis=0)/np.sum(dR)
        LID=np.trapz(d.data[:,:], d.coordinate('Device R')[0][:,:], axis=0)/(np.max(d.coordinate('Device R')[0][:,:],axis=0)-np.min(d.coordinate('Device R')[0][:,:],axis=0))
        plt.plot(d.coordinate('Time')[0][0,:],LID)
        plt.title('Line integrated density')
        plt.xlabel('Time [s]')
        plt.ylabel('n_e [m^-2]')
        plt.xlim([0,1.2])
        ax=plt.gca()
        ax.get_xaxis().set_visible(False)
        
        plt.subplot(gs[4,0])
        magnetics=flap.get_data('NSTX_MDSPlus',
                                name='\OPS_PC::\\BDOT_L1DMIVVHF5_RAW',
                                exp_id=139901,
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
                                       'Design':'Elliptic'}).plot()
        plt.xlim([0,1.2])
        
        plt.subplot(gs[0,1])
        flap.get_data('NSTX_MDSPlus',
                      name='\WF::\DALPHA',
                      exp_id=139901,
                      object_name='DALPHA').plot(options={'Axes visibility':[False,True]})
        plt.xlim([0.25,0.4])
        plt.ylim([0,3])
        plt.subplot(gs[1,1])
        flap.get_data('NSTX_GPI',
                      name='',
                      exp_id=139901,
                      object_name='GPI').slice_data(summing={'Image x':'Mean', 'Image y':'Mean'}).plot(options={'Axes visibility':[False,True]})
        plt.xlim([0.25,0.4])
        
        plt.subplot(gs[2,1])
        flap.get_data('NSTX_MDSPlus',
                      name='IP',
                      exp_id=139901,
                      object_name='IP').plot(options={'Axes visibility':[False,True]})
        plt.xlim([0.25,0.4])
        
        plt.subplot(gs[3,1])
        d=flap_nstx_thomson_data(exp_id=139901, density=True, output_name='DENSITY')
#        dR = d.coordinate('Device R')[0][:,:]-np.insert(d.coordinate('Device R')[0][0:-1,:],0,0,axis=0)
#        LID=np.sum(d.data*dR,axis=0)
        LID=np.trapz(d.data[:,:], d.coordinate('Device R')[0][:,:], axis=0)/(np.max(d.coordinate('Device R')[0][:,:],axis=0)-np.min(d.coordinate('Device R')[0][:,:],axis=0))
        plt.plot(d.coordinate('Time')[0][0,:],LID)
        plt.title('Line integrated density')
        plt.xlabel('Time [s]')
        plt.ylabel('n_e [m^-3]')
        plt.xlim([0.25,0.4])
        ax=plt.gca()
        ax.get_xaxis().set_visible(False)
#        ax.get_yaxis().set_visible(False)
        
        plt.subplot(gs[4,1])
        magnetics=flap.get_data('NSTX_MDSPlus',
                                name='\OPS_PC::\\BDOT_L1DMIVVHF5_RAW',
                                exp_id=139901,
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
    
        plt.xlim([0.25,0.4])
        ax=plt.gca()
#        ax.get_yaxis().set_visible(False)
        pdf.savefig()
        pdf.close()
        
    if plot[4]:
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
                                   plot_separatrix=True, 
                                   flux_coordinates=False,
                                   device_coordinates=True,
                                   new_plot=False,
                                   save_pdf=True,
                                   colormap='gist_ncar',
                                   save_for_paraview=False,
                                   colorbar_visibility=True
                                   )
        pdf.savefig()
        pdf.close()
    #Figure 5
    if plot[5] or plot[6] or plot[7]:
        try:
            d1,d2,d3,d4=pickle.load(open(wd+'/processed_data/fig_6_8_flap_object.pickle','rb'))
            flap.add_data_object(d1, 'GPI_SLICED_FULL')
            flap.add_data_object(d2, 'GPI_GAS_CLOUD')
            flap.add_data_object(d3, 'GPI_SLICED_DENORM_CCF_VEL')
            flap.add_data_object(d4, 'GPI_CCF_F_BY_F')
        except:
            calculate_nstx_gpi_avg_frame_velocity(exp_id=139901, 
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
    if plot[5]:
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
#                if index_grid_x == 0:
#                    z_range=[0,4096]
#                elif index_grid_x == 1:
#                    z_range=[0,400]
#                elif index_grid_x == 2:
#                    z_range=[0,40]
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
                                   #'Colormap':'gist_ncar',
                                   'Colorbar':True,
                                   #'Overplot options':oplot_options,
                                   },
                           plot_options={'levels':51},
                           )
                if index_grid_x == 0:
                    #ax=plt.gca()
                    plt.title(f"{times[index_grid_y]*1e3:.3f}"+' '+titles[index_grid_x])
                else:
                    plt.title(titles[index_grid_x])
    
        pdf.savefig()
        pdf.close()
    
    #Figure 6
    if plot[6]:
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
            colorbar=False
            flap.plot(signals[index_grid_x], 
                      plot_type='contour', 
                      slicing={'Time':0.3249560},
                      #slicing={'Sample':29808},
                      axes=['Image x', 'Image y'],
                      options={'Interpolation': 'Closest value',
                               'Clear':False,
                               'Equal axes':True,
                               'Axes visibility':visibility,
                               #'Colormap':colormap,
                               'Colorbar':True,
                               #'Overplot options':oplot_options,
                               },
                       plot_options={'levels':51},
                           )
        #fig.tight_layout()
        pdf.savefig()
        pdf.close()
        
    #Figure 7
    if plot[7]:
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
#                           'Colormap':'gist_ncar',
                           'Colorbar':True,
                           #'Overplot options':oplot_options,
                           },
                   plot_options={'levels':51},
                       )
        plt.title("324.956ms")
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
#                            'Colormap':'gist_ncar',
                            },
                    plot_options={'levels':51},
                    )
        plt.title("324.959ms")
        plt.subplot(gs[2])
        flap.plot('GPI_CCF_F_BY_F', 
                  plot_type='contour', 
                  slicing={'Sample':29807, 'Image x':flap.Intervals(-10,10),'Image y':flap.Intervals(-10,10)}, 
                  
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
        pdf.savefig()
        pdf.close()
        
    #Figure 8
    if plot[8]:
    #2x2 frames with the found structures during an ELM burst
        calculate_nstx_gpi_avg_frame_velocity(exp_id=139901, 
                                              time_range=[0.32495,0.325], 
                                              plot=False,
                                              subtraction_order_for_velocity=4,
                                              skip_structure_calculation=False,
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
                                              test_structures=True
                                              )
        #Post processing done with illustrator
        
    #Figure 9
    if plot[9]:
    #2x3
    #Synthetic GPI signal
    #Postprocessing done with illustrator
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
        calculate_nstx_gpi_avg_frame_velocity(data_object='test',
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
    #Figure 10
    if plot[10]:
    #Single shot results
        calculate_nstx_gpi_avg_frame_velocity(exp_id=139901, 
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
                                              overplot_str_vel=False)
    #2x3
    #Done with Illustrator
    #Figure 12
    if plot[11]:
    #Conditional averaged results
#        calculate_avg_velocity_results(pdf=True, 
#                                       plot=True, 
#                                       plot_max_only=True,
#                                       plot_for_publication=True,
#                                       normalized_velocity=True, 
#                                       subtraction_order=4, 
#                                       normalized_structure=True, 
#                                       opacity=0.5, 
#                                       correlation_threshold=0.6,
#                                       gpi_plane_calculation=True,
#                                       plot_scatter=False)
        plot_nstx_gpi_velocity_distribution(n_hist=50, correlation_threshold=0.6, nocalc=True, general_plot=False, plot_for_velocity=True)
        plot_nstx_gpi_velocity_distribution(n_hist=50, correlation_threshold=0.6, nocalc=True, general_plot=False, plot_for_structure=True)
    
    #Post processing done with Illustrator
    
    #Figure 11
    if plot[12]:
        if not pearson:
            pdf=PdfPages(wd+'/plots/figure_13_dependence.pdf')
            plt.figure()
            plt.subplots(figsize=(17/2.54,17/2.54/1.618))
            plot_all_parameters_vs_all_other_average(window_average=0.2e-3, symbol_size=0.3, plot_error=True)
            pdf.savefig()
            pdf.close() 
            
        else:
            pdf=PdfPages(wd+'/plots/figure_13_pearson_matrix.pdf')
            pearson=calculate_nstx_gpi_correlation_matrix(calculate_average=False,
                                                          gpi_plane_calculation=True,
                                                          window_average=0.050e-3,
                                                          elm_burst_window=True)
            data=pearson[:,:,0]
            variance=pearson[:,:,1]
            data[10,10]=-1
            plt.figure()
            plt.subplots(figsize=(8.5/2.54,8.5/2.54/1.618))
            plt.matshow(data, cmap='seismic')
            plt.xticks(ticks=np.arange(11), labels=['Velocity ccf R',       #0,1
                                                    'Velocity ccf z',       #0,1
                                                    'Velocity str max R',   #2,3
                                                    'Velocity str max z',   #2,3
                                                    'Size max R',           #4,5
                                                    'Size max z',           #4,5
                                                    'Position max R',       #6,7
                                                    'Position max z',       #6,7
                                                    'Area max',           #8
                                                    'Elongation max',     #9
                                                    'Angle max'], rotation='vertical')
            plt.yticks(ticks=np.arange(11), labels=['Velocity ccf R',       #0,1
                                                    'Velocity ccf z',       #0,1
                                                    'Velocity str max R',   #2,3
                                                    'Velocity str max z',   #2,3
                                                    'Size max R',           #4,5
                                                    'Size max z',           #4,5
                                                    'Position max R',       #6,7
                                                    'Position max z',       #6,7
                                                    'Area max',           #8
                                                    'Elongation max',     #9
                                                    'Angle max'])
            plt.colorbar()
            plt.show()
            pdf.savefig()
            plt.figure()
            plt.subplots(figsize=(8.5/2.54,8.5/2.54/1.618))
            variance[10,10]=-1
            variance[9,9]=1
            plt.matshow(variance, cmap='seismic')
            #plt.matshow(data, cmap='gist_ncar')
            plt.xticks(ticks=np.arange(11), labels=['Velocity ccf R',       #0,1
                                                    'Velocity ccf z',       #0,1
                                                    'Velocity str max R',   #2,3
                                                    'Velocity str max z',   #2,3
                                                    'Size max R',           #4,5
                                                    'Size max z',           #4,5
                                                    'Position max R',       #6,7
                                                    'Position max z',       #6,7
                                                    'Area max',           #8
                                                    'Elongation max',     #9
                                                    'Angle max'], rotation='vertical')
            plt.yticks(ticks=np.arange(11), labels=['Velocity ccf R',       #0,1
                                                    'Velocity ccf z',       #0,1
                                                    'Velocity str max R',   #2,3
                                                    'Velocity str max z',   #2,3
                                                    'Size max R',           #4,5
                                                    'Size max z',           #4,5
                                                    'Position max R',       #6,7
                                                    'Position max z',       #6,7
                                                    'Area max',           #8
                                                    'Elongation max',     #9
                                                    'Angle max'])
            plt.colorbar()
            plt.show()
            pdf.savefig()
            pdf.close()
            
    #Pierson matrix single plot
    #Figure 12
