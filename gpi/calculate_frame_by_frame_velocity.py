#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 21 23:17:45 2021

@author: mlampert
"""
  
#Core modules
import os
import copy
import cv2

import flap
import flap_nstx
flap_nstx.register('NSTX_GPI')
from flap_nstx.gpi import nstx_gpi_contour_structure_finder, nstx_gpi_watershed_structure_finder, normalize_gpi
from flap_nstx.tools import detrend_multidim

import flap_mdsplus
flap_mdsplus.register('NSTX_MDSPlus')

thisdir = os.path.dirname(os.path.realpath(__file__))
fn = os.path.join(thisdir,"../flap_nstx.cfg")
flap.config.read(file_name=fn)

#Scientific modules
import matplotlib.style as pltstyle
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import numpy as np
import scipy
import pickle
#Plot settings for publications
publication=False

if publication:

    plt.rcParams['lines.linewidth'] = 4
    plt.rcParams['axes.linewidth'] = 4
    plt.rcParams['axes.labelsize'] = 28
    plt.rcParams['axes.titlesize'] = 28
    plt.rcParams['xtick.labelsize'] = 30
    plt.rcParams['xtick.major.size'] = 10
    plt.rcParams['xtick.major.width'] = 4
    plt.rcParams['xtick.major.size'] = 6
    plt.rcParams['xtick.minor.width'] = 2
    plt.rcParams['xtick.minor.size'] = 4
    plt.rcParams['ytick.labelsize'] = 30
    plt.rcParams['ytick.major.width'] = 4
    plt.rcParams['ytick.major.size'] = 6
    plt.rcParams['ytick.minor.width'] = 2
    plt.rcParams['ytick.minor.size'] = 4
    plt.rcParams['legend.fontsize'] = 28

else:
    pltstyle.use('default')


def calculate_nstx_gpi_frame_by_frame_velocity(exp_id=None,                          #Shot number
                                               time_range=None,                      #The time range for the calculation
                                               data_object=None,                     #Input data object if available from outside (e.g. generated sythetic signal)
                                               x_range=None,                       #X range for the calculation
                                               y_range=None,                       #Y range for the calculation
                                                
                                               #Normalizer inputs
                                               normalize='roundtrip',                #Normalization options, 
                                                                                        #None: no normalization
                                                                                        #'roundtrip': zero phase LPF IIR filter
                                                                                        #'halved': different normalzation for before and after the ELM
                                                                                        #'simple': simple low-pass filtered normalization
                                               normalize_f_kernel='Elliptic',        #The kernel for filtering the gas cloud
                                               normalize_f_high=1e3,                 #High pass frequency for the normalizer data
                                               
                                               #Inputs for velocity pre processing
                                               normalize_for_velocity=True,          #Normalize the signal for velocity processing
                                               subtraction_order_for_velocity=1,     #Order of the 2D polynomial for background subtraction
                                               
                                               #Inputs for velocity processing
                                               flap_ccf=True,                        #Calculate the cross-correlation functions with flap instead of CCF
                                               correlation_threshold=0.6,            #Threshold for the maximum of the cross-correlation between the two frames (calculated with the actual maximum, not the fit value)
                                               frame_similarity_threshold=0.0,       #Similarity threshold between the two subsequent frames (CCF at zero lag) DEPRECATED when an ELM is present
                                               correct_acf_peak=True,
                                               valid_frame_thres=5,                  #The number of consequtive frames to consider the calculation valid.
                                               velocity_threshold=None,              #Velocity threshold for the calculation. Abova values are considered np.nan.
                                               parabola_fit=True,                    #Fit a parabola on top of the cross-correlation function (CCF) peak
                                               interpolation='parabola',             #Can be 'parabola', 'bicubic', and whatever is implemented in the future
                                               bicubic_upsample=4,                   #Interpolation upsample for bicubic interpolation
                                               fitting_range=5,                      #Fitting range of the peak of CCF 
                                               
                                               #Input for size pre-processing
                                               skip_structure_calculation=False,     #Self explanatory
                                               str_finding_method='contour',         # Contour or watershed based structure finding
                                               normalize_for_size=True,              #Normalize the signal for the size calculation
                                               subtraction_order_for_size=None,      #Polynomial subtraction order
                                               remove_interlaced_structures=True,    #Merge the found structures which contain each other
                                               
                                               #Inputs for size processing
                                               nlevel=51,                            #Number of contour levels for the structure size and velocity calculation.
                                               filter_level=3,                       #Number of embedded paths to be identified as an individual structure
                                               global_levels=False,                  #Set for having structure identification based on a global intensity level.
                                               levels=None,                          #Levels of the contours for the entire dataset. If None, it equals data.min(),data.max() divided to nlevel intervals.
                                               threshold_method='variance',          #variance or background for the size calculation
                                               threshold_coeff=1.0,                  #Variance multiplier threshold for size determination
                                               threshold_bg_range={'x':[54,65],      #For the background subtraction, ROI where the bg intensity is calculated
                                                                   'y':[0,79]},
                                               threshold_bg_multiplier=2.,           #Background multiplier for the thresholding
                                               weighting='intensity',                #Weighting of the results based on the 'number' of structures, the 'intensity' of the structures or the 'area' of the structures (options are in '')
                                               maxing='intensity',                   #Return the properties of structures which have the largest "area" or "intensity"
                                               velocity_base='cog',                  #Base of calculation of the advanced velocity, available options: 
                                                                                     #Center of gravity: 'cog'; Geometrical center: 'centroid'; Ellipse center: 'center'
                                               #Plot options:
                                               plot=True,                            #Plot the results
                                               pdf=False,                            #Print the results into a PDF
                                               plot_gas=False,  #NOT WORKING, NEEDS TO BE CHECKED                      #Plot the gas cloud parameters on top of the other results from the structure size calculation
                                               plot_error=False,                     #Plot the errorbars of the velocity calculation based on the line fitting and its RMS error
                                               error_window=4.,                      #Plot the average signal with the error bars calculated from the normalized variance.
                                               overplot_str_vel=True,                #Overplot the velocity calculated from the structure onto the one from CCF
                                               overplot_average=True,
                                               plot_scatter=True,
                                               structure_video_save=False,           #Save the video of the overplot ellipses
                                               structure_pdf_save=False,             #Save the struture finding algorithm's plot output into a PDF (can create very large PDF's, the number of pages equals the number of frames)
                                               structure_pixel_calc=False,           #Calculate and plot the structure sizes in pixels
                                               plot_time_range=None,                 #Plot the results in a different time range than the data is read from
                                               plot_for_publication=False,           #Modify the plot sizes to single column sizes and golden ratio axis ratios
                                               
                                               #File input/output options
                                               filename=None,                        #Filename for restoring data
                                               save_results=True,                    #Save the results into a .pickle file to filename+.pickle
                                               nocalc=True,                          #Restore the results from the .pickle file from filename+.pickle
                                               
                                               #Output options:
                                               return_results=False,                 #Return the results if set.
                                               return_pixel_displacement=False,
                                               cache_data=True,                      #Cache the data or try to open is from cache
                                               
                                               #Test options
                                               test=False,                           #Test the results
                                               test_structures=False,                #Test the structure size calculation
                                               test_gas_cloud=False,                 #Test the gas cloud property determination
                                               test_histogram=False,                 #Plot the poloidal velocity histogram
                                               save_data_for_publication=False,
                                               verbose=False,
                                               ):

    """
    Calculate frame by frame average frame velocity of the NSTX GPI signal. The
    code takes subsequent frames, calculates the 2D correlation function between
    the two and finds the maximum. Based on the pixel shift and the sampling
    time of the signal, the radial and poloidal velocity is calculated.
    The code assumes that the structures present in the subsequent frames are
    propagating with the same velocity. If there are multiple structures
    propagating in e.g. different direction or with different velocities, their
    effects are averaged over.
    """
    
    #Constants for the calculation
    #Using the spatial calibration to find the actual velocities.
    coeff_r=np.asarray([3.75, 0,    1402.8097])/1000. #The coordinates are in meters, the coefficients are in mm
    coeff_z=np.asarray([0,    3.75, 70.544312])/1000.  #The coordinates are in meters, the coefficients are in mm
    
    # Originally used coordinates for reference. (Vertical, radial geometrical coordinates)
    # coeff_r=np.asarray([3.7183594,-0.77821046,1402.8097])/1000. #The coordinates are in meters, the coefficients are in mm
    # coeff_z=np.asarray([0.18090118,3.0657776,70.544312])/1000.  #The coordinates are in meters, the coefficients are in mm

    #Input error handling
    if exp_id is None and data_object is None:
        raise ValueError('Either exp_id or data_object needs to be set for the calculation.')
        
    if data_object is None:
        if time_range is None and filename is None:
            raise ValueError('It takes too much time to calculate the entire shot, please set a time_range.')
        else:    
            if type(time_range) is not list and filename is None:
                raise TypeError('time_range is not a list.')
            if filename is None and len(time_range) != 2:
                raise ValueError('time_range should be a list of two elements.')
        
    if weighting not in ['number', 'area', 'intensity']:
        raise ValueError("Weighting can only be by the 'number', 'area' or 'intensity' of the structures.")
    if maxing not in ['area', 'intensity']:
        raise ValueError("Maxing can only be by the 'area' or 'intensity' of the structures.")
    if velocity_base not in ['cog', 'center', 'centroid']:
        raise ValueError("The base of the velocity can only be 'cog', 'center', 'centroid'")
        
    if correlation_threshold is not None and not flap_ccf:
        print('Correlation threshold is not taken into account if not calculated with FLAP.')
        
    if frame_similarity_threshold is not None and not flap_ccf:
        print('Correlation threshold is not taken into account if not calculated with FLAP.')
    
    if data_object is not None and type(data_object) == str:
        if exp_id is None:
            exp_id='*'
        d=flap.get_data_object(data_object,exp_id=exp_id)
        time_range=[d.coordinate('Time')[0][0,0,0],
                    d.coordinate('Time')[0][-1,0,0]]
        exp_id=d.exp_id
        flap.add_data_object(d, 'GPI_SLICED_FULL')
        if x_range is None or y_range is None:
            x_range=[0, d.data.shape[1]-1]
            y_range=[0, d.data.shape[2]-1]
            
        slicing={'Time':flap.Intervals(time_range[0],time_range[1]),
                 'Image x':flap.Intervals(x_range[0],x_range[1]),
                 'Image y':flap.Intervals(y_range[0],y_range[1])}
        
    wd=flap.config.get_all_section('Module NSTX_GPI')['Working directory']
    
    """
    SETTING UP THE FILENAME FOR DATA SAVING
    """
    
    if filename is None:
        
        if parabola_fit:
            comment='pfit_o'+str(subtraction_order_for_velocity)+\
                    '_fst_'+str(frame_similarity_threshold)
        else:
            comment='max_o'+str(subtraction_order_for_velocity)+\
                    '_fst_'+str(frame_similarity_threshold)
                    
        if normalize_for_size:
            comment+='_ns'
        if normalize_for_velocity:
            comment+='_nv'
        if remove_interlaced_structures:
            comment+='_nointer'
        comment+='_'+str_finding_method
        
        filename=flap_nstx.tools.filename(exp_id=exp_id,
                                             working_directory=wd+'/processed_data',
                                             time_range=time_range,
                                             purpose='ccf velocity',
                                             comment=comment)
        filename_was_none=True
    else:
        filename_was_none=False
        
    pickle_filename=filename+'.pickle'
    if os.path.exists(pickle_filename) and nocalc:
        try:
            pickle.load(open(pickle_filename, 'rb'))
        except:
            print('The pickle file cannot be loaded. Recalculating the results.')
            nocalc=False
    elif nocalc:
        print(pickle_filename)
        print('The pickle file cannot be loaded. Recalculating the results.')
        nocalc=False

    if not test and not test_structures and not test_gas_cloud and not structure_pdf_save:
        import matplotlib
        matplotlib.use('agg')
        
    import matplotlib.pyplot as plt
    
    if structure_pdf_save:
        filename=flap_nstx.tools.filename(exp_id=exp_id,
                                             working_directory=wd+'/plots',
                                             time_range=time_range,
                                             purpose='found structures',
                                             comment=comment,
                                             extension='pdf')
        pdf_structures=PdfPages(filename)
        
    if not nocalc:
        """
        READING THE DATA
        """
        #Read data
        if data_object is None:
            print("\n------- Reading NSTX GPI data --------")
            if cache_data:
                try:
                    d=flap.get_data_object('GPI',exp_id=exp_id)
                except:
                    print('Data is not cached, it needs to be read.')
                    d=flap.get_data('NSTX_GPI',exp_id=exp_id,
                                    name='',
                                    object_name='GPI')
            else:
                d=flap.get_data('NSTX_GPI',exp_id=exp_id,
                                name='',
                                object_name='GPI')
            if x_range is None or y_range is None:
                    x_range=[0, d.data.shape[1]-1]
                    y_range=[0, d.data.shape[2]-1]
                    
            if (fitting_range*2+1 > np.abs(x_range[1]-x_range[0]) or 
                fitting_range*2+1 > np.abs(y_range[1]-y_range[0])):
                raise ValueError('The fitting range for the parabola is too large for the given coordinate range.')
                
            slicing={'Time':flap.Intervals(time_range[0],time_range[1]),
                     'Image x':flap.Intervals(x_range[0],x_range[1]),
                     'Image y':flap.Intervals(y_range[0],y_range[1])}
            d=flap.slice_data('GPI',exp_id=exp_id, 
                              slicing=slicing,
                              output_name='GPI_SLICED_FULL')
            
        object_name_ccf_velocity='GPI_SLICED_FULL'
        object_name_str_size='GPI_SLICED_FULL'
        object_name_str_vel='GPI_SLICED_FULL'
        
        """
        NORMALIZATION PROCESS
        """
        
        normalizer_object_name='GPI_LPF_INTERVAL'

        slicing_for_filtering=copy.deepcopy(slicing)
        slicing_for_filtering['Time']=flap.Intervals(time_range[0]-1/normalize_f_high*10,
                                                     time_range[1]+1/normalize_f_high*10)
        slicing_time_only={'Time':flap.Intervals(time_range[0],
                                                 time_range[1])}
        if data_object is None:
            flap.slice_data('GPI',
                            exp_id=exp_id,
                            slicing=slicing_for_filtering,
                            output_name='GPI_SLICED_FOR_FILTERING')
            

        if normalize_for_size is False and normalize_for_velocity is False:
            normalize = None
            
        coefficient=normalize_gpi('GPI_SLICED_FOR_FILTERING',
                                  exp_id=exp_id,
                                  slicing_time=slicing_time_only,
                                  normalize=normalize,
                                  normalize_f_high=normalize_f_high,
                                  normalize_f_kernel=normalize_f_kernel,
                                  normalizer_object_name=normalizer_object_name,
                                  output_name='GPI_GAS_CLOUD')
            
        #Global gas levels

        if normalize is not None:
            
            gas_min=flap.get_data_object_ref('GPI_GAS_CLOUD', exp_id=exp_id).data.min()
            gas_max=flap.get_data_object_ref('GPI_GAS_CLOUD', exp_id=exp_id).data.max()
            gas_levels=np.arange(nlevel)/(nlevel-1)*(gas_max-gas_min)+gas_min            
            
            if normalize_for_velocity:
                data_obj=flap.get_data_object(object_name_ccf_velocity, exp_id=exp_id)
                data_obj.data = data_obj.data/coefficient
                flap.add_data_object(data_obj, 'GPI_SLICED_DENORM_CCF_VEL')
                object_name_ccf_velocity='GPI_SLICED_DENORM_CCF_VEL'
                
                data_obj=flap.get_data_object(object_name_str_vel, exp_id=exp_id)
                data_obj.data = data_obj.data/coefficient
                flap.add_data_object(data_obj, 'GPI_SLICED_DENORM_STR_VEL')
                object_name_str_vel='GPI_SLICED_DENORM_STR_VEL'            
                
            if normalize_for_size:
                data_obj=flap.get_data_object(object_name_str_size, exp_id=exp_id)
                if str_finding_method == 'contour':
                    data_obj.data = data_obj.data/coefficient
                elif str_finding_method == 'watershed':
                    #data_obj.data = data_obj.data-coefficient
                    data_obj.data = data_obj.data/coefficient
                flap.add_data_object(data_obj, 'GPI_SLICED_DENORM_STR_SIZE')
                object_name_str_size='GPI_SLICED_DENORM_STR_SIZE'
            
        #Subtract trend from data
        if subtraction_order_for_velocity is not None:
            if verbose: print("*** Subtracting the trend of the data ***")
            d=detrend_multidim(object_name_ccf_velocity,
                               exp_id=exp_id,
                               order=subtraction_order_for_velocity, 
                               coordinates=['Image x', 'Image y'], 
                               output_name='GPI_DETREND_CCF_VEL')
            object_name_ccf_velocity='GPI_DETREND_CCF_VEL'
            
            d=detrend_multidim(object_name_str_vel,
                               exp_id=exp_id,
                               order=subtraction_order_for_velocity, 
                               coordinates=['Image x', 'Image y'], 
                               output_name='GPI_DETREND_STR_VEL')
            object_name_str_vel='GPI_DETREND_STR_VEL'
            
        if subtraction_order_for_size is not None:
            if verbose: print("*** Subtracting the trend of the data ***")
            d=detrend_multidim(object_name_str_size,
                               exp_id=exp_id,
                               order=subtraction_order_for_size, 
                               coordinates=['Image x', 'Image y'], 
                               output_name='GPI_DETREND_STR_SIZE')
            object_name_str_size='GPI_DETREND_STR_SIZE'
            
        if interpolation != 'parabola':
            parabola_fit=False
            if interpolation == 'bicubic':
                d=flap.get_data_object_ref(object_name_ccf_velocity, exp_id=exp_id)
                d.data=scipy.ndimage.zoom(d.data, [1,bicubic_upsample,bicubic_upsample], order=3)
                d.shape=d.data.shape
                for i in range(len(d.coordinates)):
                    # if (d.coordinates[i].unit.name == 'Image x' or
                    #     d.coordinates[i].unit.name == 'Image y'):
                    #     d.coordinates[i].step = 1./bicubic_upsample
                    if (d.coordinates[i].unit.name == 'Device z' or
                        d.coordinates[i].unit.name == 'Device R'):
                        d.coordinates[i].values=scipy.ndimage.zoom(d.coordinates[i].values,
                                                                         bicubic_upsample, 
                                                                         order=1)
                        d.coordinates[i].shape=d.data.shape[1:]
                x_range=[x_range[0]*bicubic_upsample,(x_range[1]+1)*bicubic_upsample-1]
                y_range=[y_range[0]*bicubic_upsample,(y_range[1]+1)*bicubic_upsample-1]
                
                flap.delete_data_object(object_name_ccf_velocity, exp_id=exp_id)
                flap.add_data_object(d, object_name=object_name_ccf_velocity)
            
        if global_levels:
            if levels is None:
                d=flap.get_data_object_ref(object_name_str_size)
                min_data=d.data.min()
                max_data=d.data.max()
                levels=np.arange(nlevel)/(nlevel-1)*(max_data-min_data)+min_data

        thres_obj_str_size=flap.slice_data(object_name_str_size,
                                           exp_id=exp_id,
                                           summing={'Image x':'Mean',
                                                    'Image y':'Mean'},
                                                    output_name='GPI_SLICED_TIMETRACE')
        intensity_thres_level_str_size=np.sqrt(np.var(thres_obj_str_size.data))*threshold_coeff+np.mean(thres_obj_str_size.data)
        
        thres_obj_str_vel=flap.slice_data(object_name_str_size,
                                           exp_id=exp_id,
                                           summing={'Image x':'Mean',
                                                    'Image y':'Mean'},
                                                    output_name='GPI_SLICED_TIMETRACE')
        if threshold_method == 'variance':
            intensity_thres_level_str_vel=np.sqrt(np.var(thres_obj_str_vel.data))*threshold_coeff+np.mean(thres_obj_str_vel.data)
        elif threshold_method == 'background_average':
            intensity_thres_level_str_vel=threshold_bg_multiplier*np.mean(flap.slice_data(object_name_str_size, 
                                                                                  slicing={'Image x':flap.Intervals(threshold_bg_range['x'][0],
                                                                                                                    threshold_bg_range['x'][1]),
                                                                                           'Image y':flap.Intervals(threshold_bg_range['y'][0],
                                                                                                                    threshold_bg_range['y'][1])}).data)
        """
            VARIABLE DEFINITION
        """
        #Calculate correlation between subsequent frames in the data
        #Setting the variables for the calculation
        time_dim=d.get_coordinate_object('Time').dimension_list[0]
        n_frames=d.data.shape[time_dim]
        time=d.coordinate('Time')[0][:,0,0]
        sample_time=time[1]-time[0]
        sample_0=flap.get_data_object_ref(object_name_ccf_velocity).coordinate('Sample')[0][0,0,0]
        dalpha=flap.get_data_object_ref('GPI_SLICED_FULL').slice_data(summing={'Image x':'Mean', 'Image y':'Mean'}).data,
        
        
        frame_properties={'Shot':exp_id,
                          'Time':time[1:-1],
                          
                          #NON PROCESSED FRAME PARAMETERS
                          'GPI Dalpha':dalpha}
        
        #GAS CLOUD PARAMETERS       
        normal_keys=['Correlation max','Frame similarity', 
                     'GC area', 'GC angle', 'GC elongation', 
                     'Str number']
        vector_keys=['Frame COG','GC size', 'GC centroid', 'GC position', 'GC COG', 'Velocity ccf']
        
        for key in normal_keys:
            frame_properties[key]=np.zeros(len(time)-2)
        for key in vector_keys:
            frame_properties[key]=np.zeros([len(time)-2,2])
            
        str_normal_keys=['Area', 'Elongation', 'Angle', 'Angle ALI', 
                         'Roundness', 'Solidity', 'Convexity', 
                         'Total curvature', 'Total bending energy' ]
        
        str_vector_keys=['Velocity str', 'Size', 'Position', 'Centroid', 'COG']
        
        for avgmax_key in [' avg',' max']:
            for key in str_vector_keys:
                frame_properties[key+avgmax_key]=np.zeros([len(time)-2,2])
            for key in str_normal_keys:
                frame_properties[key+avgmax_key]=np.zeros([len(time)-2])
        
        ccf_data=np.zeros([n_frames-1,
                           (x_range[1]-x_range[0])*2+1,
                           (y_range[1]-y_range[0])*2+1])
        coord=[]
        coord.append(copy.deepcopy(flap.Coordinate(name='Sample',
                                                   unit='n.a.',
                                                   mode=flap.CoordinateMode(equidistant=True),
                                                   start=sample_0,
                                                   step=1,
                                                   dimension_list=[0]
                                                   )))
        
        coord.append(copy.deepcopy(flap.Coordinate(name='Time',
                                                   unit='s',
                                                   mode=flap.CoordinateMode(equidistant=True),
                                                   start=flap.get_data_object_ref(object_name_ccf_velocity).coordinate('Time')[0][0,0,0],
                                                   step=sample_time,
                                                   dimension_list=[0]
                                                   )))
        coord.append(copy.deepcopy(flap.Coordinate(name='Image x',
                                                   unit='Pixel',
                                                   mode=flap.CoordinateMode(equidistant=True),
                                                   start=-x_range[1],
                                                   step=1,
                                                   shape=[],
                                                   dimension_list=[1]
                                                   )))
        coord.append(copy.deepcopy(flap.Coordinate(name='Image y',
                                                   unit='Pixel',
                                                   mode=flap.CoordinateMode(equidistant=True),
                                                   start=-y_range[1],
                                                   step=1,
                                                   shape=[],
                                                   dimension_list=[2]
                                                   )))
        ccf_object = flap.DataObject(data_array=ccf_data,
                                     data_unit=flap.Unit(name='Signal',unit='Digit'),
                                     coordinates=coord,
                                     exp_id=exp_id,
                                     data_title='',
                                     data_source="NSTX_GPI")
        
        flap.add_data_object(ccf_object, 'GPI_CCF_F_BY_F')
        #Velocity is calculated between the two subsequent frames, the velocity time is the second frame's time
        delta_index=np.zeros(2)       
       
        #Inicializing for frame handling
        frame2=None
        frame2_vel=None
        frame2_size=None
        structures2_size=None
        structures2_vel=None
        invalid_correlation_frame_counter=0.
        
        if test or test_structures or test_gas_cloud or structure_pdf_save:
            my_dpi=80
            plt.figure(figsize=(800/my_dpi, 600/my_dpi), dpi=my_dpi)
            
        for i_frames in range(0,n_frames-2):
            """
            STRUCTURE VELOCITY CALCULATION BASED ON CCF CALCULATION
            """
            try:            
                print(str(i_frames/(n_frames-3)*100.)+"% done from the calculation.")
            except:
                pass
            
            slicing_frame1={'Sample':sample_0+i_frames}
            slicing_frame2={'Sample':sample_0+i_frames+1}
            if frame2 is None:
                frame1=flap.slice_data(object_name_ccf_velocity,
                                       exp_id=exp_id,
                                       slicing=slicing_frame1, 
                                       output_name='GPI_FRAME_1')
                #frame1.add_coordinate()
            else:
                frame1=copy.deepcopy(frame2)
                flap.add_data_object(frame1,'GPI_FRAME_1')
                
            frame2=flap.slice_data(object_name_ccf_velocity, 
                                   exp_id=exp_id,
                                   slicing=slicing_frame2,
                                   output_name='GPI_FRAME_2')

            frame1.data=np.asarray(frame1.data, dtype='float64')
            frame2.data=np.asarray(frame2.data, dtype='float64')                                
            
            if flap_ccf:
                flap.ccf('GPI_FRAME_2', 'GPI_FRAME_1', 
                         coordinate=['Image x', 'Image y'], 
                         options={'Resolution':1, 
                                  'Range':[[-(x_range[1]-x_range[0]),(x_range[1]-x_range[0])],
                                           [-(y_range[1]-y_range[0]),(y_range[1]-y_range[0])]], 
                                  'Trend removal':None, 
                                  'Normalize':True, 
                                  'Correct ACF peak':correct_acf_peak,
                                  'Interval_n': 1}, 
                         output_name='GPI_FRAME_12_CCF')
    
                ccf_object.data[i_frames,:,:]=flap.get_data_object_ref('GPI_FRAME_12_CCF').data
                
                if test:
                    print('Maximum correlation:'+str(ccf_object.data[i_frames,:,:].max()))
            else: #DEPRECATED (only to be used when the FLAP 2D correlation is not working for some reason.)
                frame1=np.asarray(d.data[i_frames,:,:],dtype='float64')
                frame2=np.asarray(d.data[i_frames+1,:,:],dtype='float64')
                ccf_object.data[i_frames,:,:]=scipy.signal.correlate2d(frame2,frame1, mode='full')
                
            
            max_index=np.asarray(np.unravel_index(ccf_object.data[i_frames,:,:].argmax(), ccf_object.data[i_frames,:,:].shape))
            
            #Fit a 2D polinomial on top of the peak
            if parabola_fit:
                area_max_index=tuple([slice(max_index[0]-fitting_range,
                                            max_index[0]+fitting_range+1),
                                      slice(max_index[1]-fitting_range,
                                            max_index[1]+fitting_range+1)])
                #Finding the peak analytically
                try:
                    coeff=flap_nstx.tools.polyfit_2D(values=ccf_object.data[i_frames,:,:][area_max_index],order=2)
                    index=[0,0]
                    index[0]=(2*coeff[2]*coeff[3]-coeff[1]*coeff[4])/(coeff[4]**2-4*coeff[2]*coeff[5])
                    index[1]=(-2*coeff[5]*index[0]-coeff[3])/coeff[4]                
                except:
                     index=[fitting_range,fitting_range]
                if (index[0] < 0 or 
                    index[0] > 2*fitting_range or 
                    index[1] < 0 or 
                    index[1] > 2*fitting_range):
                    
                    index=[fitting_range,fitting_range]
                #if ccf_object.data[i_frames,:,:].max() > correlation_threshold and flap_ccf:              
                
                delta_index=[index[0]+max_index[0]-fitting_range-ccf_object.data[i_frames,:,:].shape[0]//2,
                             index[1]+max_index[1]-fitting_range-ccf_object.data[i_frames,:,:].shape[1]//2]
                if test:
                    plt.contourf(ccf_object.data[i_frames,:,:].T, levels=np.arange(0,51)/25-1)
                    plt.scatter(index[0]+max_index[0]-fitting_range,
                                index[1]+max_index[1]-fitting_range)

            else: #if not parabola_fit:

                
                delta_index=[max_index[0]-ccf_object.data[i_frames,:,:].shape[0]//2,
                             max_index[1]-ccf_object.data[i_frames,:,:].shape[1]//2]
            
            frame_properties['Correlation max'][i_frames]=ccf_object.data[i_frames,:,:].max()
            frame_properties['Frame similarity'][i_frames]=ccf_object.data[i_frames,:,:][tuple(np.asarray(ccf_object.data[i_frames,:,:].shape)[:]//2)]
                
            if ccf_object.data[i_frames,:,:].max() < correlation_threshold and flap_ccf:
                invalid_correlation_frame_counter+=1
            else:
                invalid_correlation_frame_counter=0.
                
            if interpolation == 'bicubic':
                delta_index=[delta_index[0]/bicubic_upsample,
                             delta_index[1]/bicubic_upsample]
                
            if return_pixel_displacement:
                frame_properties['Velocity ccf'][i_frames,0]=delta_index[0]
                frame_properties['Velocity ccf'][i_frames,1]=delta_index[1]
            else:
                #Calculating the radial and poloidal velocity from the correlation map.
                frame_properties['Velocity ccf'][i_frames,0]=(coeff_r[0]*delta_index[0]+
                                                              coeff_r[1]*delta_index[1])/sample_time
                frame_properties['Velocity ccf'][i_frames,1]=(coeff_z[0]*delta_index[0]+
                                                              coeff_z[1]*delta_index[1])/sample_time       
            if not skip_structure_calculation:        
                """
                STRUCTURE SIZE CALCULATION AND MANIPULATION BASED ON STRUCTURE FINDING
                """
                
                """
                GAS CLOUD CALCULATION
                """
                if (normalize_for_size or normalize_for_velocity) and plot_gas:
                    flap.slice_data('GPI_GAS_CLOUD',
                                    exp_id=exp_id,
                                    slicing=slicing_frame2,
                                    output_name='GPI_GAS_CLOUD_SLICED')
                    gas_cloud_structure=nstx_gpi_contour_structure_finder(data_object='GPI_GAS_CLOUD_SLICED',
                                                                          exp_id=exp_id,
                                                                          filter_level=filter_level,
                                                                          nlevel=nlevel,
                                                                          levels=gas_levels,
                                                                          spatial=not structure_pixel_calc,
                                                                          pixel=structure_pixel_calc,
                                                                          remove_interlaced_structures=remove_interlaced_structures,
                                                                          test_result=test_gas_cloud)
                    n_gas_structures=len(gas_cloud_structure)
                    for gas_structure in gas_cloud_structure:
                        frame_properties['GC size'][i_frames,:]+=gas_structure['Size'][:]/n_gas_structures
                        frame_properties['GC centroid'][i_frames,:]+=gas_structure['Centroid'][:]/n_gas_structures
                        frame_properties['GC position'][i_frames,:]+=gas_structure['Center'][:]/n_gas_structures
                        frame_properties['GC COG'][i_frames,:]+=gas_structure['Center of gravity'][:]/n_gas_structures
                        frame_properties['GC area'][i_frames]+=gas_structure['Area']/n_gas_structures
                        frame_properties['GC elongation'][i_frames]+=gas_structure['Elongation']/n_gas_structures
                        frame_properties['GC angle'][i_frames]+=gas_structure['Angle']/n_gas_structures
                        
                if invalid_correlation_frame_counter < valid_frame_thres:        
                        
                    if frame2_vel is None:
                        frame1_vel=flap.slice_data(object_name_str_vel,
                                                    exp_id=exp_id,
                                                    slicing=slicing_frame1, 
                                                    output_name='GPI_FRAME_1_STR_VEL')
                    else:
                        frame1_vel=copy.deepcopy(frame2_vel)
                        
                    frame2_vel=flap.slice_data(object_name_str_vel, 
                                                exp_id=exp_id,
                                                slicing=slicing_frame2,
                                                output_name='GPI_FRAME_2_STR_VEL')
                    
                    frame1_vel.data=np.asarray(frame1_vel.data, dtype='float64')
                    frame2_vel.data=np.asarray(frame2_vel.data, dtype='float64')
                    
                    
                    if frame2_size is None:
                        frame1_size=flap.slice_data(object_name_str_size, 
                                                    exp_id=exp_id,
                                                    slicing=slicing_frame1,
                                                    output_name='GPI_FRAME_1_STR_SIZE')
                    else:
                        frame1_size=copy.deepcopy(frame2_size)
                        
                    frame2_size=flap.slice_data(object_name_str_size, 
                                                exp_id=exp_id,
                                                slicing=slicing_frame2,
                                                output_name='GPI_FRAME_2_STR_SIZE')
                    
                    frame1_size.data=np.asarray(frame1_size.data, dtype='float64')
                    frame2_size.data=np.asarray(frame2_size.data, dtype='float64')
                    
                    if str_finding_method == 'contour':
                        contour_mutual_settings={                             "exp_id":exp_id,
                                                                              "filter_level":filter_level,
                                                                              "nlevel":nlevel,
                                                                              "levels":levels,
                                                                              "spatial":not structure_pixel_calc,
                                                                              "remove_interlaced_structures":remove_interlaced_structures,
                                                                              "pixel":structure_pixel_calc}
                        if structures2_vel is None:
                            structures1_vel=nstx_gpi_contour_structure_finder(data_object='GPI_FRAME_1_STR_VEL',
                                                                              threshold_level=intensity_thres_level_str_vel,
                                                                              **contour_mutual_settings)
                        else:
                            structures1_vel=copy.deepcopy(structures2_vel)
                            
                        if structure_video_save or structure_pdf_save:
                            plt.cla()
                            test_structures=True
    
                        structures2_vel=nstx_gpi_contour_structure_finder(data_object='GPI_FRAME_2_STR_VEL',
                                                                          threshold_level=intensity_thres_level_str_vel,
                                                                          **contour_mutual_settings)
                        if structures2_size is None:
                            structures1_size=nstx_gpi_contour_structure_finder(data_object='GPI_FRAME_1_STR_SIZE',
                                                                               threshold_level=intensity_thres_level_str_size,
                                                                               test_result=test_structures,
                                                                               save_data_for_publication=save_data_for_publication,
                                                                               **contour_mutual_settings)
                        else:
                            structures1_size=copy.deepcopy(structures2_size)
                        structures2_size=nstx_gpi_contour_structure_finder(data_object='GPI_FRAME_2_STR_SIZE',
                                                                           threshold_level=intensity_thres_level_str_size,
                                                                           test_result=test_structures,
                                                                           save_data_for_publication=save_data_for_publication,
                                                                           **contour_mutual_settings)
                        
                    elif str_finding_method == 'watershed':# or str_finding_method == 'randomwalker':
                        
                        watershed_mutual_settings={                             "exp_id":exp_id,                             #Shot number (if data_object is not used)
                                                                                "spatial":not structure_pixel_calc,                          #Calculate the results in real spatial coordinates
                                                                                "pixel":structure_pixel_calc,                            #Calculate the results in pixel coordinates
                                                                                "mfilter_range":5,                        #Range of the median filter
                                                                                "threshold_method":'otsu',
                                                                                "test":False,                             #Test the contours and the structures before any kind of processing
                                                                                "nlevel":51,
                                                                                #"try_random_walker": str_finding_method == 'randomwalker',
                            }
                        if structures2_vel is None:
                            structures1_vel=nstx_gpi_watershed_structure_finder(data_object='GPI_FRAME_1_STR_VEL',
                                                                                threshold_level=intensity_thres_level_str_vel,                                   
                                                                                save_data_for_publication=save_data_for_publication,
                                                                                **watershed_mutual_settings)
                        else:
                            structures1_vel=copy.deepcopy(structures2_vel)
                            
                        if structure_video_save or structure_pdf_save:
                            plt.cla()
                            test_structures=True
    
                        structures2_vel=nstx_gpi_watershed_structure_finder(data_object='GPI_FRAME_2_STR_VEL',                  
                                                                            threshold_level=intensity_thres_level_str_vel,      
                                                                            save_data_for_publication=save_data_for_publication,
                                                                            **watershed_mutual_settings
                                                                            )
                        if structures2_size is None:
                            structures1_size=nstx_gpi_watershed_structure_finder(data_object='GPI_FRAME_1_STR_SIZE',            
                                                                                 threshold_level=intensity_thres_level_str_size,
                                                                                 test_result=test_structures,
                                                                                 **watershed_mutual_settings
                                                                                 )
                        else:
                            structures1_size=copy.deepcopy(structures2_size)
                            
                        structures2_size=nstx_gpi_watershed_structure_finder(data_object='GPI_FRAME_2_STR_SIZE',                       #Name of the FLAP.data_object
                                                                             threshold_level=intensity_thres_level_str_size,                   #Threshold level over which it is considered to be a structure
                                                                             test_result=test_structures,
                                                                             **watershed_mutual_settings
                                                                             )

                    if not structure_video_save:
                        plt.pause(0.001)
                        if structure_pdf_save:
                            plt.show()
                            pdf_structures.savefig()
                    else:
                        test_structures=False
                        fig = plt.gcf()
                        plt.title(str(exp_id)+' @ '+"{:.3f}".format(time[i_frames]*1e3)+'ms')
                        fig.canvas.draw()
                        # Get the RGBA buffer from the figure
                        w,h = fig.canvas.get_width_height()
                        try:
                            buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                            if buf.shape[0] == h*2 * w*2 * 3:
                                buf.shape = ( h*2, w*2, 3 )
                            else:
                                buf.shape = ( h, w, 3 )
                            buf = cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR)
                            try:
                                video
                            except NameError:
                                height = buf.shape[0]
                                width = buf.shape[1]
                                video_codec_code='mp4v'
                                comment=str_finding_method+'_bgthr_'+str(threshold_bg_multiplier)
                                filename=flap_nstx.tools.filename(exp_id=exp_id,
                                                                  working_directory=wd+'/plots',
                                                                  time_range=time_range,
                                                                  purpose='fit structures',
                                                                  comment=comment)
                                filename=wd+'/plots/NSTX_GPI_'+str(exp_id)+'_'+"{:.3f}".format(time[0]*1e3)+'_fit_structures_'+str_finding_method+'.mp4'
                                video = cv2.VideoWriter(filename,  
                                                        cv2.VideoWriter_fourcc(*video_codec_code), 
                                                        float(24), 
                                                        (width,height),
                                                        isColor=True)
                            video.write(buf)
                        except:
                            print('Video frame cannot be saved. Passing...')
                        
                    if structures1_vel is not None and len(structures1_vel) != 0:
                        valid_structure1_vel=True
                    else:
                        valid_structure1_vel=False            
                    if structures2_vel is not None and len(structures2_vel) != 0:
                        valid_structure2_vel=True
                    else:
                        valid_structure2_vel=False
                    if structures1_size is not None and len(structures1_size) != 0:
                        valid_structure1_size=True
                    else:
                        valid_structure1_size=False                        
                    if structures2_size is not None and len(structures2_size) != 0:
                        valid_structure2_size=True
                    else:
                        valid_structure2_size=False

                else:
                    valid_structure1_vel=False
                    valid_structure2_vel=False
                    valid_structure2_size=False
                    if verbose: print('Invalid consecutive number of frames: '+str(invalid_correlation_frame_counter))
                        
                """Structure size calculation based on the contours"""    
                #Crude average size calculation
                if valid_structure2_size:
                    #Calculating the average properties of the structures present in one frame
                    n_str2=len(structures2_size)
                    areas=np.zeros(len(structures2_size))
                    intensities=np.zeros(len(structures2_size))
                    for i_str2 in range(n_str2):
                        #Average size calculation based on the number of structures
                        areas[i_str2]=structures2_size[i_str2]['Area']
                        intensities[i_str2]=structures2_size[i_str2]['Intensity']
                        
                    #Calculating the averages based on the input setting
                    if weighting == 'number':
                        weight=1./n_str2
                    elif weighting == 'intensity':
                        weight=intensities/np.sum(intensities)
                    elif weighting == 'area':
                        weight=areas/np.sum(areas)
                    if verbose: print(n_str2, weight, i_frames)
                    for i_str2 in range(n_str2):   
                        #Quantities from Ellipse fitting
                        frame_properties['Size avg'][i_frames,:]+=structures2_size[i_str2]['Size']*weight[i_str2]
                        frame_properties['Angle avg'][i_frames]+=structures2_size[i_str2]['Angle']*weight[i_str2]
                        frame_properties['Elongation avg'][i_frames]+=structures2_size[i_str2]['Elongation']*weight[i_str2]
                        frame_properties['Position avg'][i_frames,:]+=structures2_size[i_str2]['Center']*weight[i_str2]
                        #Quantities from polygons
                        frame_properties['Area avg'][i_frames]+=structures2_size[i_str2]['Area']*weight[i_str2]
                        frame_properties['Centroid avg'][i_frames,:]+=structures2_size[i_str2]['Centroid']*weight[i_str2]
                        frame_properties['COG avg'][i_frames,:]+=structures2_size[i_str2]['Center of gravity']*weight[i_str2]
                        try:
                            frame_properties['Angle ALI avg'][i_frames]+=structures2_size[i_str2]['Polygon'].principal_axes_angle*weight[i_str2]
                            frame_properties['Roundness avg'][i_frames]+=structures2_size[i_str2]['Polygon'].roundness*weight[i_str2]
                            frame_properties['Solidity avg'][i_frames]+=structures2_size[i_str2]['Polygon'].solidity*weight[i_str2]
                            frame_properties['Convexity avg'][i_frames]+=structures2_size[i_str2]['Polygon'].convexity*weight[i_str2]
                            frame_properties['Total curvature avg'][i_frames]+=structures2_size[i_str2]['Polygon'].total_curvature*weight[i_str2]
                            frame_properties['Total bending energy avg'][i_frames]+=structures2_size[i_str2]['Polygon'].total_bending_energy*weight[i_str2]
                        except:
                            pass
                    
                    #The number of structures in a frame
                    frame_properties['Str number'][i_frames]=n_str2
#                    frame_properties['Structures'][i_frames]=structures2_size
                    #Calculating the properties of the structure having the maximum area or intensity
                    if maxing == 'area':
                        ind_max=np.argmax(areas)
                    elif maxing == 'intensity':
                        ind_max=np.argmax(intensities)
                        
                    #Properties of the max structure:
                    frame_properties['Size max'][i_frames,:]=structures2_size[ind_max]['Size']
                    frame_properties['Area max'][i_frames]=structures2_size[ind_max]['Area']
                    frame_properties['Angle max'][i_frames]=structures2_size[ind_max]['Angle']
                    frame_properties['Elongation max'][i_frames]=structures2_size[ind_max]['Elongation']
                    frame_properties['Position max'][i_frames,:]=structures2_size[ind_max]['Center']
                    frame_properties['Centroid max'][i_frames,:]=structures2_size[ind_max]['Centroid']
                    frame_properties['COG max'][i_frames,:]=structures2_size[ind_max]['Center of gravity']
                    try:
                        frame_properties['Angle ALI max'][i_frames]=structures2_size[ind_max]['Polygon'].principal_axes_angle
                        frame_properties['Roundness max'][i_frames]=structures2_size[ind_max]['Polygon'].roundness
                        frame_properties['Solidity max'][i_frames]=structures2_size[ind_max]['Polygon'].solidity
                        frame_properties['Convexity max'][i_frames]=structures2_size[ind_max]['Polygon'].convexity
                        frame_properties['Total curvature max'][i_frames]=structures2_size[ind_max]['Polygon'].total_curvature
                        frame_properties['Total bending energy max'][i_frames]=structures2_size[ind_max]['Polygon'].total_bending_energy
                    except:
                        pass
                        
                    #The center of gravity for the entire frame
                    x_coord=frame2_size.coordinate('Device R')[0]
                    y_coord=frame2_size.coordinate('Device z')[0]
                    frame_properties['Frame COG'][i_frames,:]=np.asarray([np.sum(x_coord*frame2_size.data)/np.sum(frame2_size.data),
                                                                          np.sum(y_coord*frame2_size.data)/np.sum(frame2_size.data)])
                else:
                    #Setting np.nan if no structure is available
                    
                    vector_keys=['Size', 'Position', 'COG', 'Centroid']
                    non_vector_keys=['Area', 'Angle', 'Elongation', 'Angle ALI', 'Roundness',
                                     'Solidity', 'Convexity', 'Total curvature',
                                     'Total bending energy']
                    for avg_max in [' avg', ' max']:
                        for key in vector_keys:
                            key_full=key+avg_max
                            frame_properties[key_full][i_frames,:]=[np.nan,np.nan]
                        for key in non_vector_keys:
                            key_full=key+avg_max
                            frame_properties[key_full][i_frames]=np.nan
                    
                    frame_properties['Str number'][i_frames]=0.
                    frame_properties['Frame COG'][i_frames,:]=np.asarray([np.nan,np.nan])
                   # frame_properties['Structures'][i_frames]=None
                   
                """
                Velocity calculation based on the contours
                """                                                                
                                
                if valid_structure1_vel and valid_structure2_vel:          
                    #if multiple structures merge into one, then previous position is their average
                    #if one structure is split up into to, then the previous position is the old's position
                    #Current velocity is the position change and sampling time's ratio
                    n_str1=len(structures1_vel)
                    n_str2=len(structures2_vel)
                    
                    prev_str_number=np.zeros([n_str2,n_str1])
                    prev_str_intensity=np.zeros([n_str2,n_str1])
                    prev_str_area=np.zeros([n_str2,n_str1])
                    prev_str_pos=np.zeros([n_str2,n_str1,2])
                    
                    current_str_area=np.zeros(n_str2)
                    current_str_intensity=np.zeros(n_str2)
                    
                    current_str_vel=np.zeros([n_str2,2])
                    if velocity_base == 'cog':
                        vel_base_key='Center of gravity'
                    elif velocity_base == 'center':
                        vel_base_key='Center'
                    elif velocity_base == 'centroid':
                        vel_base_key='Centroid'
                    elif velocity_base == 'frame cog':
                        vel_base_key='Centroid'                             #Just error handling, not actually used for the results
    
                    for i_str2 in range(n_str2):
                        for i_str1 in range(n_str1):
                            if structures2_vel[i_str2]['Half path'].intersects_path(structures1_vel[i_str1]['Half path']):
                                prev_str_number[i_str2,i_str1]=1.
                                prev_str_intensity[i_str2,i_str1]=structures1_vel[i_str1]['Intensity']
                                prev_str_area[i_str2,i_str1]=structures1_vel[i_str1]['Area']
                                prev_str_pos[i_str2,i_str1,:]=structures1_vel[i_str1][vel_base_key]
                                
                    for i_str2 in range(n_str2):            
                        if np.sum(prev_str_number[i_str2,:]) > 0:
    
                            current_str_intensity[i_str2]=structures2_vel[i_str2]['Intensity']
                            current_str_area[i_str2]=structures2_vel[i_str2]['Area']
                            
                            if weighting == 'number':
                                weight=1./np.sum(prev_str_number[i_str2,:])
                            elif weighting == 'intensity':
                                weight=prev_str_intensity[i_str2,:]/np.sum(prev_str_intensity[i_str2,:])
                            elif weighting == 'area':
                                weight=prev_str_area[i_str2,:]/np.sum(prev_str_area[i_str2,:])
                                
                            prev_str_pos_avg=np.asarray([np.sum(prev_str_pos[i_str2,:,0]*weight),
                                                         np.sum(prev_str_pos[i_str2,:,1]*weight)])
                            current_str_pos=structures2_vel[i_str2][vel_base_key]                        
                            current_str_vel[i_str2,:]=(current_str_pos-prev_str_pos_avg)/sample_time
    
                            
                    #Criterion for validity of the velocity
                    #If the structures are not overlapping, then the velocity cannot be valid.
                    ind_valid=np.where(np.sum(prev_str_number,axis=1) > 0)
                    
                    if np.sum(prev_str_number) != 0:
                        if weighting == 'number':
                            weight=1./n_str2
                        elif weighting == 'intensity':
                            weight=current_str_intensity/np.sum(current_str_intensity)
                        elif weighting == 'area':
                            weight=current_str_area/np.sum(current_str_area)
                            
                        #Calculating the average based on the number of valid moving structures
                        frame_properties['Velocity str avg'][i_frames,0]=np.sum(current_str_vel[ind_valid,0]*weight[ind_valid])
                        frame_properties['Velocity str avg'][i_frames,1]=np.sum(current_str_vel[ind_valid,1]*weight[ind_valid])
    
                    #Calculating the properties of the structure having the maximum area or intensity
                        if maxing == 'area':
                            ind_max=np.argmax(current_str_area[ind_valid])
                        elif maxing == 'intensity':
                            ind_max=np.argmax(current_str_intensity[ind_valid])
    
                        frame_properties['Velocity str max'][i_frames,:]=current_str_vel[ind_max,:]
                        
                        if abs(np.mean(current_str_vel[:,0])) > 10e3:
                            print('Structure velocity over 10km/s')
                            print('Current structure velocity: '+str(current_str_vel))
                            print('Position difference: '+str(structures2_vel[i_str2]['Center']-prev_str_pos[i_str2,:]))
                            
                        if velocity_threshold is not None:
                            if (abs(frame_properties['Velocity str max'][i_frames,0]) > velocity_threshold or
                                abs(frame_properties['Velocity str max'][i_frames,1]) > velocity_threshold):
                                print('Velocity validity threshold reached. Setting np.nan as velocity.')
                                frame_properties['Velocity str max'][i_frames,:]=[np.nan,np.nan]
                    else:
                        frame_properties['Velocity str avg'][i_frames,:]=[np.nan,np.nan]
                        frame_properties['Velocity str max'][i_frames,:]=[np.nan,np.nan]
                else:
                    frame_properties['Velocity str avg'][i_frames,:]=[np.nan,np.nan]
                    frame_properties['Velocity str max'][i_frames,:]=[np.nan,np.nan]
                
        #Wrapper for separatrix distance
        frame_properties=calculate_separatrix_distance(frame_properties=frame_properties,
                                                       skip_structure_calculation=skip_structure_calculation,
                                                       exp_id=exp_id,
                                                       coeff_r=coeff_r,
                                                       coeff_z=coeff_z,
                                                       )
            
        if (structure_video_save):
            cv2.destroyAllWindows()
            video.release()  
            del video
        if structure_pdf_save:
            pdf_structures.close()
        #Saving results into a pickle file
        
        pickle.dump(frame_properties,open(pickle_filename, 'wb'))
        if test:
            plt.close()
    else:
        print('--- Loading data from the pickle file ---')
        frame_properties=pickle.load(open(pickle_filename, 'rb'))

    """
    PLOTTING THE RESULTS
    """
        
    if not filename_was_none and not time_range is None:
        sample_time=frame_properties['Time'][1]-frame_properties['Time'][0]
        if time_range[0] < frame_properties['Time'][0]-sample_time or time_range[1] > frame_properties['Time'][-1]+sample_time:
            raise ValueError('Please run the calculation again with the timerange. The pickle file doesn\'t have the desired range')
    if time_range is None:
        time_range=[frame_properties['Time'][0],frame_properties['Time'][-1]]
    
    frame_properties=calculate_separatrix_distance(frame_properties=frame_properties,
                                                   skip_structure_calculation=skip_structure_calculation,
                                                   exp_id=exp_id,
                                                   coeff_r=coeff_r,
                                                   coeff_z=coeff_z,
                                                   )
    
    nan_ind=np.where(frame_properties['Correlation max'] < correlation_threshold)
    frame_properties['Velocity ccf'][nan_ind,0] = np.nan
    frame_properties['Velocity ccf'][nan_ind,1] = np.nan
    

    #Plotting the results
    if plot or pdf:
        #This is a bit unusual here, but necessary due to the structure size calculation based on the contours which are not plot
        if plot:
            import matplotlib
            matplotlib.use('QT5Agg')
            import matplotlib.pyplot as plt
        else:
            import matplotlib
            matplotlib.use('agg')
            import matplotlib.pyplot as plt     
        
        if plot_time_range is not None:
            if plot_time_range[0] < time_range[0] or plot_time_range[1] > time_range[1]:
                raise ValueError('The plot time range is not in the interval of the original time range.')
            time_range=plot_time_range
            
        plot_index=np.logical_and(np.logical_not(np.isnan(frame_properties['Velocity ccf'][:,0])),
                                  np.logical_and(frame_properties['Time'] >= time_range[0],
                                                 frame_properties['Time'] <= time_range[1]))
        
        plot_index_structure=np.logical_and(np.logical_not(np.isnan(frame_properties['Elongation avg'])),
                                            np.logical_and(frame_properties['Time'] >= time_range[0],
                                                           frame_properties['Time'] <= time_range[1]))

        #Plotting the radial velocity
        if pdf:
            wd=flap.config.get_all_section('Module NSTX_GPI')['Working directory']
            filename=flap_nstx.tools.filename(exp_id=exp_id,
                                                 working_directory=wd+'/plots',
                                                 time_range=time_range,
                                                 purpose='ccf velocity',
                                                 comment=comment+'_ct_'+str(correlation_threshold))
            pdf_filename=filename+'.pdf'
            pdf_pages=PdfPages(pdf_filename)
            
        if plot_for_publication:
            figsize=(8.5/2.54, 
                     8.5/2.54/1.618*1.1)
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
        else:
            figsize=None
        
        plot_vector=[{'x':frame_properties['Time'][plot_index],
                      'y':frame_properties['Velocity ccf'][plot_index,0],
                      'color':'tab:blue',
                      'overplot':'structure',
                      'overplot_x':frame_properties['Time'][plot_index],
                      'overplot_y':frame_properties['Velocity str max'][plot_index,0],
                      'overplot_color':'red',
                      'linewidth':0.3,
                      'overplot_gas':False,
                      'xlabel':'Time [s]',
                      'ylabel':'v_rad[m/s]',
                      'title':'Radial velocity of '+str(exp_id),
                      'pub_title':'Rad vel ccf',},
                     
                     {'x':frame_properties['Time'][plot_index],
                      'y':frame_properties['Velocity ccf'][plot_index,1],
                      'color':'tab:blue',
                      'overplot':'structure',
                      'overplot_x':frame_properties['Time'][plot_index],
                      'overplot_y':frame_properties['Velocity str max'][plot_index,1],
                      'overplot_color':'red',
                      'linewidth':0.3,
                      'overplot_gas':False,
                      'xlabel':'Time [s]',
                      'ylabel':'v_pol[m/s]',
                      'title':'Poloidal velocity of '+str(exp_id),
                      'pub_title':'Pol vel ccf',},
                     
                     {'x':frame_properties['Time'][plot_index_structure],
                      'y':frame_properties['Velocity str max'][plot_index_structure,1],
                      'color':'red',
                      'overplot':'average',
                      'overplot_x':frame_properties['Time'][plot_index_structure],
                      'overplot_y':frame_properties['Velocity str avg'][plot_index_structure,1],
                      'overplot_color':'tab:blue',
                      'linewidth':1.0,
                      'overplot_gas':False,
                      'xlabel':'Time [s]',
                      'ylabel':'v_pol[m/s]',
                      'title':'Poloidal str velocity of '+str(exp_id),
                      'pub_title':'Pol vel str',},
                     
                     {'x':frame_properties['Time'],
                      'y':frame_properties['Frame similarity'],
                      'color':'tab:blue',
                      'overplot':'yes',
                      'overplot_x':frame_properties['Time'][plot_index],
                      'overplot_y':frame_properties['Frame similarity'][plot_index],
                      'overplot_color':'red',
                      'linewidth':1.0,
                      'overplot_gas':False,
                      'xlabel':'Time [s]',
                      'ylabel':'Correlation coefficient',
                      'title':'Maximum correlation (red) and \nframe similarity (blue) of '+str(exp_id),
                      'pub_title':'Max corr',},
                     
                     {'x':frame_properties['Time'],
                      'y':frame_properties['GPI Dalpha'][0][1:-1],
                      'color':'tab:blue',
                      'overplot':False,
                      'overplot_gas':False,
                      'xlabel':'Time [s]',
                      'ylabel':'Average GPI signal',
                      'title':'Average GPI signal of '+str(exp_id),
                      'pub_title':'Avg gpi',},
                     
                     {'x':frame_properties['Time'][plot_index_structure],
                      'y':frame_properties['Size max'][:,0][plot_index_structure],
                      'color':'red',
                      'overplot':'average',
                      'overplot_x':frame_properties['Time'][plot_index_structure],
                      'overplot_y':frame_properties['Size avg'][:,0][plot_index_structure],
                      'overplot_color':'tab:blue',
                      'linewidth':1.0,
                      'overplot_gas':plot_gas,
                      'overplot_gas_x':frame_properties['Time'],                          
                      'overplot_gas_y':frame_properties['GC size'][:,0],
                      'xlabel':'Time [s]',
                      'ylabel':'Radial size [m]',
                      'title':'Average (blue) and '+maxing+' maximum (red) radial\n size of structures of '+str(exp_id),
                      'pub_title':'Rad size',},
                     
                     {'x':frame_properties['Time'][plot_index_structure],
                      'y':frame_properties['Size max'][:,1][plot_index_structure],
                      'color':'red',
                      'overplot':'average',
                      'overplot_x':frame_properties['Time'][plot_index_structure],
                      'overplot_y':frame_properties['Size avg'][:,1][plot_index_structure],
                      'overplot_color':'tab:blue',
                      'linewidth':1.0,
                      'overplot_gas':plot_gas,
                      'overplot_gas_x':frame_properties['Time'],                          
                      'overplot_gas_y':frame_properties['GC size'][:,1],
                      'xlabel':'Time [s]',
                      'ylabel':'Poloidal size [m]',
                      'title':'Average (blue) and '+maxing+' maximum (red) poloidal\n size of structures of '+str(exp_id),
                      'pub_title':'Pol size',},
                     
                     {'x':frame_properties['Time'][plot_index_structure],
                      'y':frame_properties['Position max'][:,0][plot_index_structure],
                      'color':'red',
                      'overplot':'average',
                      'overplot_x':frame_properties['Time'][plot_index_structure],
                      'overplot_y':frame_properties['Position avg'][:,0][plot_index_structure],
                      'overplot_color':'tab:blue',
                      'linewidth':1.0,
                      'overplot_gas':plot_gas,
                      'overplot_gas_x':frame_properties['Time'],                          
                      'overplot_gas_y':frame_properties['GC position'][:,0],
                      'xlabel':'Time [s]',
                      'ylabel':'Radial position [m]',
                      'title':'Average (blue) and '+maxing+' maximum (red) radial\n position of structures of '+str(exp_id),
                      'pub_title':'Rad pos',},
                     
                     {'x':frame_properties['Time'][plot_index_structure],
                      'y':frame_properties['Centroid max'][:,0][plot_index_structure],
                      'color':'red',
                      'overplot':'average',
                      'overplot_x':frame_properties['Time'][plot_index_structure],
                      'overplot_y':frame_properties['Centroid avg'][:,0][plot_index_structure],
                      'overplot_color':'tab:blue',
                      'linewidth':1.0,
                      'overplot_gas':plot_gas,
                      'overplot_gas_x':frame_properties['Time'],                          
                      'overplot_gas_y':frame_properties['GC centroid'][:,0],
                      'xlabel':'Time [s]',
                      'ylabel':'Radial centroid [m]',
                      'title':'Average (blue) and '+maxing+' maximum (red) radial\n centroid of structures of '+str(exp_id),
                      'pub_title':'Rad centroid',},
                     
                     {'x':frame_properties['Time'][plot_index_structure],
                      'y':frame_properties['COG max'][:,0][plot_index_structure],
                      'color':'red',
                      'overplot':'average',
                      'overplot_x':frame_properties['Time'][plot_index_structure],
                      'overplot_y':frame_properties['COG avg'][:,0][plot_index_structure],
                      'overplot_color':'tab:blue',
                      'linewidth':1.0,
                      'overplot_gas':plot_gas,
                      'overplot_gas_x':frame_properties['Time'],                          
                      'overplot_gas_y':frame_properties['GC COG'][:,0],
                      'xlabel':'Time [s]',
                      'ylabel':'Radial COG [m]',
                      'title':'Average (blue) and '+maxing+' maximum (red) radial\n center of gravity of structures of '+str(exp_id),
                      'pub_title':'Rad COG',},
                     
                     {'x':frame_properties['Time'][plot_index_structure],
                      'y':frame_properties['Position max'][:,1][plot_index_structure],
                      'color':'red',
                      'overplot':'average',
                      'overplot_x':frame_properties['Time'][plot_index_structure],
                      'overplot_y':frame_properties['Position avg'][:,1][plot_index_structure],
                      'overplot_color':'tab:blue',
                      'linewidth':1.0,
                      'overplot_gas':plot_gas,
                      'overplot_gas_x':frame_properties['Time'],                          
                      'overplot_gas_y':frame_properties['GC position'][:,1],
                      'xlabel':'Time [s]',
                      'ylabel':'Poloidal position [m]',
                      'title':'Average (blue) and '+maxing+' maximum (red) poloidal\n position of structures of '+str(exp_id),
                      'pub_title':'Pol pos',},
                     
                     {'x':frame_properties['Time'][plot_index_structure],
                      'y':frame_properties['Centroid max'][:,1][plot_index_structure],
                      'color':'red',
                      'overplot':'average',
                      'overplot_x':frame_properties['Time'][plot_index_structure],
                      'overplot_y':frame_properties['Centroid avg'][:,1][plot_index_structure],
                      'overplot_color':'tab:blue',
                      'linewidth':1.0,
                      'overplot_gas':plot_gas,
                      'overplot_gas_x':frame_properties['Time'],                          
                      'overplot_gas_y':frame_properties['GC centroid'][:,1],
                      'xlabel':'Time [s]',
                      'ylabel':'Poloidal centroid [m]',
                      'title':'Average (blue) and '+maxing+' maximum (red) poloidal\n centroid of structures of '+str(exp_id),
                      'pub_title':'Pol centroid',},                   
                     
                     {'x':frame_properties['Time'][plot_index_structure],
                      'y':frame_properties['COG max'][:,1][plot_index_structure],
                      'color':'red',
                      'overplot':'average',
                      'overplot_x':frame_properties['Time'][plot_index_structure],
                      'overplot_y':frame_properties['COG avg'][:,1][plot_index_structure],
                      'overplot_color':'tab:blue',
                      'linewidth':1.0,
                      'overplot_gas':plot_gas,
                      'overplot_gas_x':frame_properties['Time'],                          
                      'overplot_gas_y':frame_properties['GC COG'][:,1],
                      'xlabel':'Time [s]',
                      'ylabel':'Poloidal COG [m]',
                      'title':'Average (blue) and '+maxing+' maximum (red) poloidal\n center of gravity of structures of '+str(exp_id),
                      'pub_title':'Pol COG',},
                     
                     {'x':frame_properties['Time'][plot_index_structure],
                      'y':frame_properties['Separatrix dist max'][plot_index_structure],
                      'color':'red',
                      'overplot':'average',
                      'overplot_x':frame_properties['Time'][plot_index_structure],
                      'overplot_y':frame_properties['Separatrix dist avg'][plot_index_structure],
                      'overplot_color':'tab:blue',
                      'linewidth':1.0,
                      'overplot_gas':False,
                      'xlabel':'Time [s]',
                      'ylabel':'Separatrix dist [m]',
                      'title':'Average (blue) and '+maxing+' maximum (red) separatrix \n distance of structures of '+str(exp_id),
                      'pub_title':'Sep dist',},
                     
                     {'x':frame_properties['Time'][plot_index_structure],
                      'y':frame_properties['Elongation max'][plot_index_structure],
                      'color':'red',
                      'overplot':'average',
                      'overplot_x':frame_properties['Time'][plot_index_structure],
                      'overplot_y':frame_properties['Elongation avg'][plot_index_structure],
                      'overplot_color':'tab:blue',
                      'linewidth':1.0,
                      'overplot_gas':plot_gas,
                      'overplot_gas_x':frame_properties['Time'],
                      'overplot_gas_y':frame_properties['GC elongation'],
                      'xlabel':'Time [s]',
                      'ylabel':'Elongation',
                      'title':'Average (blue) and '+maxing+' maximum (red) elongation\n of structures of '+str(exp_id),
                      'pub_title':'Elongation',},        
                     
                     {'x':frame_properties['Time'][plot_index_structure],
                      'y':np.unwrap(frame_properties['Angle max'][plot_index_structure])-np.pi,
                      'color':'red',
                      'overplot':'average',
                      'overplot_x':frame_properties['Time'][plot_index_structure],
                      'overplot_y':np.unwrap(frame_properties['Angle avg'][plot_index_structure])-np.pi,
                      'overplot_color':'tab:blue',
                      'linewidth':1.0,
                      'overplot_gas':plot_gas,
                      'overplot_gas_x':frame_properties['Time'],
                      'overplot_gas_y':frame_properties['GC angle'],
                      'xlabel':'Time [s]',
                      'ylabel':'Angle [rad]',
                      'title':'Average (blue) and '+maxing+' maximum (red) angle\n of structures of '+str(exp_id),
                      'pub_title':'Angle',},                       

                     
                     {'x':frame_properties['Time'][plot_index_structure],
                      'y':frame_properties['Area max'][plot_index_structure],
                      'color':'red',
                      'overplot':'average',
                      'overplot_x':frame_properties['Time'][plot_index_structure],
                      'overplot_y':frame_properties['Area avg'][plot_index_structure],
                      'overplot_color':'tab:blue',
                      'linewidth':1.0,
                      'overplot_gas':plot_gas,
                      'overplot_gas_x':frame_properties['Time'],
                      'overplot_gas_y':frame_properties['GC area'],
                      'xlabel':'Time [s]',
                      'ylabel':'Area [m2]',
                      'title':'Average (blue) and '+maxing+' maximum (red) area\n of structures of '+str(exp_id),
                      'pub_title':'Area',}, 
                     
                     {'x':frame_properties['Time'][plot_index_structure],
                      'y':np.unwrap(frame_properties['Angle ALI max'][plot_index_structure], discont=np.pi/2)-np.pi,
                      'color':'red',
                      'overplot':'average',
                      'overplot_x':frame_properties['Time'][plot_index_structure],
                      'overplot_y':np.unwrap(frame_properties['Angle ALI avg'][plot_index_structure], discont=np.pi/2)-np.pi,
                      'overplot_color':'tab:blue',
                      'linewidth':1.0,
                      'overplot_gas':False,
                      'xlabel':'Time [s]',
                      'ylabel':'Angle',
                      'title':'Average (blue) and '+maxing+' maximum (red) angle ALI\n of structures of '+str(exp_id),
                      'pub_title':'Angle ALI',},    
                     
                     {'x':frame_properties['Time'][plot_index_structure],
                      'y':frame_properties['Roundness max'][plot_index_structure],
                      'color':'red',
                      'overplot':'average',
                      'overplot_x':frame_properties['Time'][plot_index_structure],
                      'overplot_y':frame_properties['Roundness avg'][plot_index_structure],
                      'overplot_color':'tab:blue',
                      'linewidth':1.0,
                      'overplot_gas':False,
                      'xlabel':'Time [s]',
                      'ylabel':'Roundness',
                      'title':'Average (blue) and '+maxing+' maximum (red) roundness\n of structures of '+str(exp_id),
                      'pub_title':'Roundness',},  
                     
                     {'x':frame_properties['Time'][plot_index_structure],
                      'y':frame_properties['Solidity max'][plot_index_structure],
                      'color':'red',
                      'overplot':'average',
                      'overplot_x':frame_properties['Time'][plot_index_structure],
                      'overplot_y':frame_properties['Solidity avg'][plot_index_structure],
                      'overplot_color':'tab:blue',
                      'linewidth':1.0,
                      'overplot_gas':False,
                      'xlabel':'Time [s]',
                      'ylabel':'Solidity',
                      'title':'Average (blue) and '+maxing+' maximum (red) solidity\n of structures of '+str(exp_id),
                      'pub_title':'Solidity',},


                     {'x':frame_properties['Time'][plot_index_structure],
                      'y':frame_properties['Convexity max'][plot_index_structure],
                      'color':'red',
                      'overplot':'average',
                      'overplot_x':frame_properties['Time'][plot_index_structure],
                      'overplot_y':frame_properties['Convexity avg'][plot_index_structure],
                      'overplot_color':'tab:blue',
                      'linewidth':1.0,
                      'overplot_gas':False,
                      'xlabel':'Time [s]',
                      'ylabel':'Convexity',
                      'title':'Average (blue) and '+maxing+' maximum (red) convexity\n of structures of '+str(exp_id),
                      'pub_title':'Convexity',},                     
                     

                     {'x':frame_properties['Time'][plot_index_structure],
                      'y':frame_properties['Total curvature max'][plot_index_structure],
                      'color':'red',
                      'overplot':'average',
                      'overplot_x':frame_properties['Time'][plot_index_structure],
                      'overplot_y':frame_properties['Total curvature avg'][plot_index_structure],
                      'overplot_color':'tab:blue',
                      'linewidth':1.0,
                      'overplot_gas':False,
                      'xlabel':'Time [s]',
                      'ylabel':'Curvature',
                      'title':'Average (blue) and '+maxing+' maximum (red) total curvature\n of structures of '+str(exp_id),
                      'pub_title':'Total curvature',},    

                     {'x':frame_properties['Time'][plot_index_structure],
                      'y':frame_properties['Total bending energy max'][plot_index_structure],
                      'color':'red',
                      'overplot':'average',
                      'overplot_x':frame_properties['Time'][plot_index_structure],
                      'overplot_y':frame_properties['Total bending energy avg'][plot_index_structure],
                      'overplot_color':'tab:blue',
                      'linewidth':1.0,
                      'overplot_gas':False,
                      'xlabel':'Time [s]',
                      'ylabel':'Bending energy',
                      'title':'Average (blue) and '+maxing+' maximum (red) total bending energy\n of structures of '+str(exp_id),
                      'pub_title':'Total bending energy',},  
                     
                     {'x':frame_properties['Time'],
                      'y':frame_properties['Str number'],
                      'color':'tab:blue',
                      'overplot':'no',
                      'overplot_gas':False,
                      'xlabel':'Time [s]',
                      'ylabel':'Str number',
                      'title':'Number of structures vs. time of '+str(exp_id),
                      'pub_title':'Str num',},    
                     
                     {'x':frame_properties['Time'][plot_index_structure],
                      'y':frame_properties['Frame COG'][plot_index_structure,0],
                      'color':'red',
                      'overplot':'no',
                      'overplot_gas':False,
                      'xlabel':'Time [s]',
                      'ylabel':'Radial frame COG [m]',
                      'title':'Radial center of gravity of the frame vs. time of '+str(exp_id),
                      'pub_title':'Rad frame COG',},     
                     
                     {'x':frame_properties['Time'][plot_index_structure],
                      'y':frame_properties['Frame COG'][plot_index_structure,1],
                      'color':'red',
                      'overplot':'no',
                      'overplot_gas':False,
                      'xlabel':'Time [s]',
                      'ylabel':'Poloidal frame COG [m]',
                      'title':'Poloidal center of gravity of the frame vs. time of '+str(exp_id),
                      'pub_title':'Pol frame COG',},  
                     ]    
            
        for i in range(len(plot_vector)):
            fig, ax = plt.subplots(figsize=figsize)
            ax.plot(plot_vector[i]['x'], 
                    plot_vector[i]['y'],
                    color=plot_vector[i]['color'])
            if plot_scatter:
                ax.scatter(plot_vector[i]['x'], 
                            plot_vector[i]['y'], 
                            s=5, 
                            marker='o',
                            color=plot_vector[i]['color'])
            if ((plot_vector[i]['overplot']=='structure' and overplot_str_vel) or
                (plot_vector[i]['overplot']=='average' and overplot_average) or
                (plot_vector[i]['overplot']=='yes')):
                ax.plot(plot_vector[i]['overplot_x'],
                        plot_vector[i]['overplot_y'], 
                        linewidth=plot_vector[i]['linewidth'],
                        color=plot_vector[i]['overplot_color'],)
                
            if plot_vector[i]['overplot_gas']:
                ax.plot(plot_vector[i]['overplot_gas_x'], 
                        plot_vector[i]['overplot_gas_y'],
                        color='orange')   
                
            ax.set_xlabel(plot_vector[i]['xlabel'])
            ax.set_ylabel(plot_vector[i]['ylabel'])
            ax.set_xlim(time_range)
            ax.set_title(plot_vector[i]['title'])
            if plot_for_publication:
                x1,x2=ax.get_xlim()
                y1,y2=ax.get_ylim()
                ax.set_aspect((x2-x1)/(y2-y1)/1.618)
                ax.set_title(plot_vector[i]['pub_title'])
            fig.tight_layout()
            if pdf:
                pdf_pages.savefig()
        if pdf:
           pdf_pages.close()
      
                
        if test_histogram:
            #Plotting the velocity histogram if test is set.
            hist,bin_edge=np.histogram(frame_properties['Velocity'][:,1], bins=(np.arange(100)/100.-0.5)*24000.)
            bin_edge=(bin_edge[:99]+bin_edge[1:])/2
            plt.figure()
            plt.plot(bin_edge,hist)
            plt.title('Poloidal velocity histogram')
            plt.xlabel('Poloidal velocity [m/s]')
            plt.ylabel('Number of points')
            
    if plot_for_publication:
        import matplotlib.style as pltstyle
        pltstyle.use('default')
    if return_results:
        return frame_properties
    
def calculate_separatrix_distance(frame_properties=None,
                                  skip_structure_calculation=None,
                                  exp_id=None,
                                  coeff_r=None,
                                  coeff_z=None,
                                  ):
            
    #Calculating the distance from the separatrix
    frame_properties['Separatrix dist avg']=np.zeros(frame_properties['Position avg'].shape[0])
    frame_properties['Separatrix dist max']=np.zeros(frame_properties['Position max'].shape[0])
    
    if not skip_structure_calculation:
        try:
            elm_time=(frame_properties['Time'][-1]+frame_properties['Time'][0])/2
            
            R_sep=flap.get_data('NSTX_MDSPlus',
                                name='\EFIT02::\RBDRY',
                                exp_id=exp_id,
                                object_name='SEP R OBJ').slice_data(slicing={'Time':elm_time}).data
            z_sep=flap.get_data('NSTX_MDSPlus',
                                name='\EFIT02::\ZBDRY',
                                exp_id=exp_id,
                                object_name='SEP Z OBJ').slice_data(slicing={'Time':elm_time}).data
            sep_GPI_ind=np.where(np.logical_and(R_sep > coeff_r[2],
                                                      np.logical_and(z_sep > coeff_z[2],
                                                                     z_sep < coeff_z[2]+79*coeff_z[0]+64*coeff_z[1])))
            sep_GPI_ind=np.asarray(sep_GPI_ind[0])
            sep_GPI_ind=np.insert(sep_GPI_ind,0,sep_GPI_ind[0]-1)
            sep_GPI_ind=np.insert(sep_GPI_ind,len(sep_GPI_ind),sep_GPI_ind[-1]+1)            
            z_sep_GPI=z_sep[(sep_GPI_ind)]
            R_sep_GPI=R_sep[sep_GPI_ind]
            GPI_z_vert=coeff_z[0]*np.arange(80)/80*64+coeff_z[1]*np.arange(80)+coeff_z[2]
            R_sep_GPI_interp=np.interp(GPI_z_vert,np.flip(z_sep_GPI),np.flip(R_sep_GPI))
            z_sep_GPI_interp=GPI_z_vert
            
            for key in ['max','avg']:
                for ind_time in range(len(frame_properties['Position '+key][:,0])):
                    frame_properties['Separatrix dist '+key][ind_time]=np.min(np.sqrt((frame_properties['Position '+key][ind_time,0]-R_sep_GPI_interp)**2 + 
                                                                              (frame_properties['Position '+key][ind_time,1]-z_sep_GPI_interp)**2))
                    ind_z_min=np.argmin(np.abs(z_sep_GPI-frame_properties['Position '+key][ind_time,1]))
                    if z_sep_GPI[ind_z_min] >= frame_properties['Position '+key][ind_time,1]:
                        ind1=ind_z_min
                        ind2=ind_z_min+1
                    else:
                        ind1=ind_z_min-1
                        ind2=ind_z_min
                        
                    radial_distance=frame_properties['Position '+key][ind_time,0]-((frame_properties['Position '+key][ind_time,1]-z_sep_GPI[ind2])/(z_sep_GPI[ind1]-z_sep_GPI[ind2])*(R_sep_GPI[ind1]-R_sep_GPI[ind2])+R_sep_GPI[ind2])
                    if radial_distance < 0:
                        frame_properties['Separatrix dist '+key][ind_time]*=-1
        except:
            print('Separatrix distance calculation failed.')
    return frame_properties