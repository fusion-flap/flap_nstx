#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 12:11:18 2022

@author: mlampert

@Derived from calculate_frame_by_frame_velocity.py
"""

#Core modules
import os
import copy
import cv2

import warnings
warnings.filterwarnings("ignore")

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


def analyze_gpi_structures(exp_id=None,                          #Shot number
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

                           #Input for size pre-processing
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
                           prev_str_weighting='intensity',       #weighting for the differential quantities like angular and linear velocity
                           str_size_lower_thres=0.00375*4,        #Structures having sizes under this value are filtered out from the results. (Default is 4 pixels for both radial, poloidal)
                           elongation_threshold=0.1,             #Structures having major/minor_axis-1 lower than this value are set to angle=np.nan
                           remove_orphans=True,                  #Structures which
                           calculate_rough_diff_velocities=False,
                            #Plot options:
                           plot=True,                            #Plot the results
                           pdf=False,                            #Print the results into a PDF
                           plot_gas=False,#NOT WORKING, NEEDS TO BE CHECKED                      #Plot the gas cloud parameters on top of the other results from the structure size calculation
                           plot_error=False,                     #Plot the errorbars of the velocity calculation based on the line fitting and its RMS error
                           error_window=4.,                      #Plot the average signal with the error bars calculated from the normalized variance.

                           overplot_average=True,
                           plot_scatter=False,
                           structure_video_save=False,           #Save the video of the overplot ellipses
                           structure_pdf_save=False,             #Save the struture finding algorithm's plot output into a PDF (can create very large PDF's, the number of pages equals the number of frames)
                           structure_pixel_calc=False,           #Calculate and plot the structure sizes in pixels
                           plot_time_range=None,                 #Plot the results in a different time range than the data is read from
                           plot_for_publication=False,           #Modify the plot sizes to single column sizes and golden ratio axis ratios
                           plot_vertical_line_at=None,
                           plot_str_by_str=False,

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
                           test_histogram=False,                 #Plot the poloidal velocity histogram
                           save_data_for_publication=False,
                           verbose=False,
                           ellipse_method='linalg',
                           fit_shape='ellipse'
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
        comment=''
        if normalize_for_size:
            comment+='_ns'

        if remove_interlaced_structures:
            comment+='_nointer'
        comment+='_'+str_finding_method

        filename=flap_nstx.tools.filename(exp_id=exp_id,
                                             working_directory=wd+'/processed_data',
                                             time_range=time_range,
                                             purpose='structure char',
                                             comment=comment)
        filename_was_none=True
    else:
        filename_was_none=False

    fit_shape=fit_shape.capitalize()

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

    if ((not test and not test_structures) or
        (not test and not plot and structure_pdf_save and test_structures)):
        import matplotlib
        matplotlib.use('agg')
    #    import matplotlib.pyplot as plt

    if not nocalc:
        if structure_pdf_save:
            filename=flap_nstx.tools.filename(exp_id=exp_id,
                                                 working_directory=wd+'/plots',
                                                 time_range=time_range,
                                                 purpose='found structures',
                                                 comment=comment,
                                                 extension='pdf')
            pdf_structures=PdfPages(filename)
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

            slicing={'Time':flap.Intervals(time_range[0],time_range[1]),
                     'Image x':flap.Intervals(x_range[0],x_range[1]),
                     'Image y':flap.Intervals(y_range[0],y_range[1])}
            d=flap.slice_data('GPI',exp_id=exp_id,
                              slicing=slicing,
                              output_name='GPI_SLICED_FULL')

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

        coefficient=normalize_gpi('GPI_SLICED_FOR_FILTERING',
                                  exp_id=exp_id,
                                  slicing_time=slicing_time_only,
                                  normalize=normalize,
                                  normalize_f_high=normalize_f_high,
                                  normalize_f_kernel=normalize_f_kernel,
                                  normalizer_object_name=normalizer_object_name,
                                  output_name='GPI_GAS_CLOUD')

        if normalize_for_size:
            data_obj=flap.get_data_object('GPI_SLICED_FULL', exp_id=exp_id)
            if str_finding_method == 'contour':
                data_obj.data = data_obj.data/coefficient
            elif str_finding_method == 'watershed':
                #data_obj.data = data_obj.data-coefficient
                data_obj.data = data_obj.data/coefficient
            flap.add_data_object(data_obj, 'GPI_SLICED_DENORM_STR_SIZE')
            object_name_str_size='GPI_SLICED_DENORM_STR_SIZE'

        if subtraction_order_for_size is not None:
            if verbose: print("*** Subtracting the trend of the data ***")
            d=detrend_multidim(object_name_str_size,
                               exp_id=exp_id,
                               order=subtraction_order_for_size,
                               coordinates=['Image x', 'Image y'],
                               output_name='GPI_DETREND_STR_SIZE')
            object_name_str_size='GPI_DETREND_STR_SIZE'

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
        """
            VARIABLE DEFINITION
        """
        #Calculate correlation between subsequent frames in the data
        #Setting the variables for the calculation
        time_dim=d.get_coordinate_object('Time').dimension_list[0]
        n_frames=d.data.shape[time_dim]
        time=d.coordinate('Time')[0][:,0,0]
        sample_time=time[1]-time[0]
        sample_0=flap.get_data_object_ref('GPI_SLICED_FULL',
                                          exp_id=exp_id).coordinate('Sample')[0][0,0,0]

        data_dict={'max':np.zeros([len(time)]),
                   'avg':np.zeros([len(time)]),
                   'stddev':np.zeros([len(time)]),
                   'raw':np.zeros([len(time)]),

                   'unit':None,
                   'label':None,
                   }

        frame_properties={'shot':exp_id,
                          'Time':time,
                          'data':{},
                          'derived':{},
                          'structures':[],
                          }

        """
        Frame characterizing parameters
        """

        key='Angle'
        frame_properties['data'][key]=copy.deepcopy(data_dict)
        frame_properties['data'][key]['label']='$\phi$'
        frame_properties['data'][key]['unit']='deg'

        key='Angle of least inertia'
        frame_properties['data'][key]=copy.deepcopy(data_dict)
        frame_properties['data'][key]['label']='$\phi_{ALI}$'
        frame_properties['data'][key]['unit']='deg'

        key='Area'
        frame_properties['data'][key]=copy.deepcopy(data_dict)
        frame_properties['data'][key]['label']='Area'
        frame_properties['data'][key]['unit']='$m^2$'

        key='Axes length minor'
        frame_properties['data'][key]=copy.deepcopy(data_dict)
        frame_properties['data'][key]['label']='a'
        frame_properties['data'][key]['unit']='$mm$'

        key='Axes length major'
        frame_properties['data'][key]=copy.deepcopy(data_dict)
        frame_properties['data'][key]['label']='a'
        frame_properties['data'][key]['unit']='$mm$'

        key='Center of gravity radial'
        frame_properties['data'][key]=copy.deepcopy(data_dict)
        frame_properties['data'][key]['label']='$COG_{rad}$'
        frame_properties['data'][key]['unit']='m'

        key='Center of gravity poloidal'
        frame_properties['data'][key]=copy.deepcopy(data_dict)
        frame_properties['data'][key]['label']='$COG_{pol}$'
        frame_properties['data'][key]['unit']='m'

        key='Centroid radial'
        frame_properties['data'][key]=copy.deepcopy(data_dict)
        frame_properties['data'][key]['label']='Centr. rad.'
        frame_properties['data'][key]['unit']='m'

        key='Centroid poloidal'
        frame_properties['data'][key]=copy.deepcopy(data_dict)
        frame_properties['data'][key]['label']='Centr. pol.'
        frame_properties['data'][key]['unit']='m'

        key='Convexity'
        frame_properties['data'][key]=copy.deepcopy(data_dict)
        frame_properties['data'][key]['label']='Convexity'
        frame_properties['data'][key]['unit']=''

        key='Elongation'
        frame_properties['data'][key]=copy.deepcopy(data_dict)
        frame_properties['data'][key]['label']='Elong.'
        frame_properties['data'][key]['unit']=''

        key='Frame COG radial'
        frame_properties['data'][key]=copy.deepcopy(data_dict)
        frame_properties['data'][key]['label']='$COG_{frame,rad}$'
        frame_properties['data'][key]['unit']='m'

        key='Frame COG poloidal'
        frame_properties['data'][key]=copy.deepcopy(data_dict)
        frame_properties['data'][key]['label']='$COG_{frame,rad}$'
        frame_properties['data'][key]['unit']='m'

        key='Position radial'
        frame_properties['data'][key]=copy.deepcopy(data_dict)
        frame_properties['data'][key]['label']='R'
        frame_properties['data'][key]['unit']='m'

        key='Position poloidal'
        frame_properties['data'][key]=copy.deepcopy(data_dict)
        frame_properties['data'][key]['label']='z'
        frame_properties['data'][key]['unit']='m'

        key='Roundness'
        frame_properties['data'][key]=copy.deepcopy(data_dict)
        frame_properties['data'][key]['label']='Round.'
        frame_properties['data'][key]['unit']=''

        key='Separatrix dist'
        frame_properties['data'][key]=copy.deepcopy(data_dict)
        frame_properties['data'][key]['label']='$r-r_{sep}$'
        frame_properties['data'][key]['unit']='m'

        key='Size radial'
        frame_properties['data'][key]=copy.deepcopy(data_dict)
        frame_properties['data'][key]['label']='$d_{rad}$'
        frame_properties['data'][key]['unit']='m'

        key='Size poloidal'
        frame_properties['data'][key]=copy.deepcopy(data_dict)
        frame_properties['data'][key]['label']='$d_{pol}$'
        frame_properties['data'][key]['unit']='m'

        key='Solidity'
        frame_properties['data'][key]=copy.deepcopy(data_dict)
        frame_properties['data'][key]['label']='Solidity'
        frame_properties['data'][key]['unit']=''

        key='Str number'
        frame_properties['data'][key]=copy.deepcopy(data_dict)
        frame_properties['data'][key]['label']='N'
        frame_properties['data'][key]['unit']=''

        key='Total bending energy'
        frame_properties['data'][key]=copy.deepcopy(data_dict)
        frame_properties['data'][key]['label']='$E_{bend}$'
        frame_properties['data'][key]['unit']=''

        key='Total curvature'
        frame_properties['data'][key]=copy.deepcopy(data_dict)
        frame_properties['data'][key]['label']='$\kappa_{tot}$'
        frame_properties['data'][key]['unit']=''

        """
        Differential parameters
        """

        differential_keys=['Velocity radial COG',
                           'Velocity poloidal COG',
                           'Velocity radial centroid',
                           'Velocity poloidal centroid',
                           'Velocity radial position',
                           'Velocity poloidal position',

                           'Expansion fraction area',
                           'Expansion fraction axes',
                           'Angular velocity angle',
                           'Angular velocity ALI',
                           ]

        key='Angular velocity angle'
        frame_properties['derived'][key]=copy.deepcopy(data_dict)
        frame_properties['derived'][key]['label']=''
        frame_properties['derived'][key]['unit']=''

        key='Angular velocity ALI'
        frame_properties['derived'][key]=copy.deepcopy(data_dict)
        frame_properties['derived'][key]['label']=''
        frame_properties['derived'][key]['unit']=''

        key='Expansion fraction area'
        frame_properties['derived'][key]=copy.deepcopy(data_dict)
        frame_properties['derived'][key]['label']=''
        frame_properties['derived'][key]['unit']=''

        key='Expansion fraction axes'
        frame_properties['derived'][key]=copy.deepcopy(data_dict)
        frame_properties['derived'][key]['label']=''
        frame_properties['derived'][key]['unit']=''

        key='Velocity radial position'
        frame_properties['derived'][key]=copy.deepcopy(data_dict)
        frame_properties['derived'][key]['label']=''
        frame_properties['derived'][key]['unit']='m/s'

        key='Velocity poloidal position'
        frame_properties['derived'][key]=copy.deepcopy(data_dict)
        frame_properties['derived'][key]['label']=''
        frame_properties['derived'][key]['unit']='m/s'

        key='Velocity radial COG'
        frame_properties['derived'][key]=copy.deepcopy(data_dict)
        frame_properties['derived'][key]['label']=''
        frame_properties['derived'][key]['unit']='m/s'

        key='Velocity poloidal COG'
        frame_properties['derived'][key]=copy.deepcopy(data_dict)
        frame_properties['derived'][key]['label']=''
        frame_properties['derived'][key]['unit']='m/s'

        key='Velocity radial centroid'
        frame_properties['derived'][key]=copy.deepcopy(data_dict)
        frame_properties['derived'][key]['label']=''
        frame_properties['derived'][key]['unit']='m/s'

        key='Velocity poloidal centroid'
        frame_properties['derived'][key]=copy.deepcopy(data_dict)
        frame_properties['derived'][key]['label']=''
        frame_properties['derived'][key]['unit']='m/s'

        #Inicializing for frame handling
        frame=None
        structures_dict=None
        invalid_correlation_frame_counter=0.

        if test or test_structures or structure_pdf_save:
            my_dpi=80
            plt.figure(figsize=(800/my_dpi, 600/my_dpi), dpi=my_dpi)

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

        for i_frames in range(0,n_frames):
            """
            STRUCTURE VELOCITY CALCULATION BASED ON CCF CALCULATION
            """

            print(str(int(i_frames/(n_frames-1)*100.))+"% done from the calculation.")

            slicing_frame={'Sample':sample_0+i_frames}

            frame=flap.slice_data('GPI_SLICED_FULL',
                                  exp_id=exp_id,
                                  slicing=slicing_frame,
                                  output_name='GPI_FRAME')

            frame.data=np.asarray(frame.data, dtype='float64')

            """
            STRUCTURE SIZE CALCULATION AND MANIPULATION BASED ON STRUCTURE FINDING
            """

            if normalize:
                frame=flap.slice_data(object_name_str_size,
                                      exp_id=exp_id,
                                      slicing=slicing_frame,
                                      output_name='GPI_FRAME')

                frame.data=np.asarray(frame.data, dtype='float64')

                if structure_video_save or structure_pdf_save:
                    plt.cla()
                    test_structures=True

                if str_finding_method == 'contour':
                    structures_dict=nstx_gpi_contour_structure_finder(data_object='GPI_FRAME',
                                                                      threshold_level=intensity_thres_level_str_size,
                                                                      test_result=test_structures,
                                                                      save_data_for_publication=save_data_for_publication,
                                                                      exp_id=exp_id,
                                                                      filter_level=filter_level,
                                                                      nlevel=nlevel,
                                                                      levels=levels,
                                                                      spatial=not structure_pixel_calc,
                                                                      remove_interlaced_structures=remove_interlaced_structures,
                                                                      pixel=structure_pixel_calc,
                                                                      ellipse_method=ellipse_method,
                                                                      str_size_lower_thres=str_size_lower_thres,
                                                                      elongation_threshold=elongation_threshold,
                                                                      skip_gaussian=True,
                                                                      )

                elif str_finding_method == 'watershed':# or str_finding_method == 'randomwalker':
                    structures_dict=nstx_gpi_watershed_structure_finder(data_object='GPI_FRAME',
                                                                        threshold_level=intensity_thres_level_str_size,
                                                                        test_result=test_structures,
                                                                        exp_id=exp_id,                             #Shot number (if data_object is not used)
                                                                        spatial=not structure_pixel_calc,                          #Calculate the results in real spatial coordinates
                                                                        pixel=structure_pixel_calc,                            #Calculate the results in pixel coordinates
                                                                        mfilter_range=5,                        #Range of the median filter
                                                                        threshold_method='otsu',
                                                                        test=False,                             #Test the contours and the structures before any kind of processing
                                                                        nlevel=51,
                                                                        ignore_side_structure=False,
                                                                        ignore_large_structure=True,
                                                                        str_size_lower_thres=str_size_lower_thres,
                                                                        elongation_threshold=elongation_threshold,
                                                                        skip_gaussian=True,
                                                                        #try_random_walker= str_finding_method == 'randomwalker',
                                                                        )
                frame_properties['structures'].append(structures_dict)

                if not structure_video_save:
                    #plt.pause(0.001)
                    if structure_pdf_save:
                        plt.title(str(exp_id)+' @ '+"{:.3f}".format(time[i_frames]*1e3)+'ms')
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

                if structures_dict is not None and len(structures_dict) != 0:
                    valid_structure_size=True
                else:
                    valid_structure_size=False

            else:
                valid_structure_size=False
                if verbose: print('Invalid consecutive number of frames: '+str(invalid_correlation_frame_counter))

            """
            Structure size calculation based on the contours
            """

            #Crude average size calculation
            if valid_structure_size:
                #Calculating the average properties of the structures present in one frame
                n_str=len(structures_dict)
                areas=np.zeros(len(structures_dict))
                intensities=np.zeros(len(structures_dict))

                for i_str in range(n_str):
                    #Average size calculation based on the number of structures
                    areas[i_str]=structures_dict[i_str]['Area']
                    intensities[i_str]=structures_dict[i_str]['Intensity']

                #Calculating the averages based on the input setting
                if weighting == 'number':
                    weight=np.zeros(n_str)
                    weight[:]=1./n_str
                elif weighting == 'intensity':
                    weight=intensities/np.sum(intensities)
                elif weighting == 'area':
                    weight=areas/np.sum(areas)

                for i_str in range(n_str):
                    #Quantities from Ellipse fitting
                    fit_obj_cur=structures_dict[i_str][fit_shape]
                    frame_properties['data']['Size radial']['avg'][i_frames]+=fit_obj_cur.size[0]*weight[i_str]
                    frame_properties['data']['Size poloidal']['avg'][i_frames]+=fit_obj_cur.size[1]*weight[i_str]
                    frame_properties['data']['Position radial']['avg'][i_frames]+=fit_obj_cur.center[0]*weight[i_str]
                    frame_properties['data']['Position poloidal']['avg'][i_frames]+=fit_obj_cur.center[1]*weight[i_str]
                    frame_properties['data']['Angle']['avg'][i_frames]+=fit_obj_cur.angle*weight[i_str]
                    frame_properties['data']['Elongation']['avg'][i_frames]+=fit_obj_cur.elongation*weight[i_str]
                    frame_properties['data']['Axes length minor']['avg'][i_frames]+=np.min(fit_obj_cur.axes_length)*weight[i_str]
                    frame_properties['data']['Axes length major']['avg'][i_frames]+=np.max(fit_obj_cur.axes_length)*weight[i_str]

                    #Quantities from polygons
                    polygon_cur=structures_dict[i_str]['Polygon']
                    frame_properties['data']['Centroid radial']['avg'][i_frames]+=polygon_cur.centroid[0]*weight[i_str]
                    frame_properties['data']['Centroid poloidal']['avg'][i_frames]+=polygon_cur.centroid[1]*weight[i_str]
                    frame_properties['data']['Center of gravity radial']['avg'][i_frames]+=polygon_cur.center_of_gravity[0]*weight[i_str]
                    frame_properties['data']['Center of gravity poloidal']['avg'][i_frames]+=polygon_cur.center_of_gravity[1]*weight[i_str]
                    frame_properties['data']['Area']['avg'][i_frames]+=polygon_cur.area*weight[i_str]
                    frame_properties['data']['Angle of least inertia']['avg'][i_frames]+=polygon_cur.principal_axes_angle*weight[i_str]
                    frame_properties['data']['Roundness']['avg'][i_frames]+=polygon_cur.roundness*weight[i_str]
                    frame_properties['data']['Solidity']['avg'][i_frames]+=polygon_cur.solidity*weight[i_str]
                    frame_properties['data']['Convexity']['avg'][i_frames]+=polygon_cur.convexity*weight[i_str]
                    frame_properties['data']['Total curvature']['avg'][i_frames]+=polygon_cur.total_curvature*weight[i_str]
                    frame_properties['data']['Total bending energy']['avg'][i_frames]+=polygon_cur.total_bending_energy*weight[i_str]

                #Calculating the properties of the structure having the maximum area or intensity
                if maxing == 'area':
                    ind_max=np.argmax(areas)
                elif maxing == 'intensity':
                    ind_max=np.argmax(intensities)

                #Properties of the max structure:
                fit_obj_cur=structures_dict[ind_max][fit_shape]
                frame_properties['data']['Size radial']['max'][i_frames]=fit_obj_cur.size[0]
                frame_properties['data']['Size poloidal']['max'][i_frames]=fit_obj_cur.size[1]
                frame_properties['data']['Position radial']['max'][i_frames]=fit_obj_cur.center[0]
                frame_properties['data']['Position poloidal']['max'][i_frames]=fit_obj_cur.center[1]
                frame_properties['data']['Angle']['max'][i_frames]=fit_obj_cur.angle
                frame_properties['data']['Elongation']['max'][i_frames]=fit_obj_cur.elongation
                frame_properties['data']['Axes length minor']['max'][i_frames]=np.min(fit_obj_cur.axes_length)
                frame_properties['data']['Axes length major']['max'][i_frames]=np.max(fit_obj_cur.axes_length)

                polygon_cur=structures_dict[ind_max]['Polygon']
                frame_properties['data']['Centroid radial']['max'][i_frames]=polygon_cur.centroid[0]
                frame_properties['data']['Centroid poloidal']['max'][i_frames]=polygon_cur.centroid[1]
                frame_properties['data']['Center of gravity radial']['max'][i_frames]=polygon_cur.center_of_gravity[0]
                frame_properties['data']['Center of gravity poloidal']['max'][i_frames]=polygon_cur.center_of_gravity[1]
                frame_properties['data']['Area']['max'][i_frames]=polygon_cur.area
                frame_properties['data']['Angle of least inertia']['max'][i_frames]=polygon_cur.principal_axes_angle
                frame_properties['data']['Roundness']['max'][i_frames]=polygon_cur.roundness
                frame_properties['data']['Solidity']['max'][i_frames]=polygon_cur.solidity
                frame_properties['data']['Convexity']['max'][i_frames]=polygon_cur.convexity
                frame_properties['data']['Total curvature']['max'][i_frames]=polygon_cur.total_curvature
                frame_properties['data']['Total bending energy']['max'][i_frames]=polygon_cur.total_bending_energy

                #The center of gravity for the entire frame
                frame_properties['data']['Frame COG radial']['max'][i_frames]=np.sum(frame.coordinate('Device R')[0]*frame.data)/np.sum(frame.data)
                frame_properties['data']['Frame COG radial']['avg'][i_frames]=frame_properties['data']['Frame COG radial']['max'][i_frames]
                frame_properties['data']['Frame COG poloidal']['max'][i_frames]=np.sum(frame.coordinate('Device z')[0]*frame.data)/np.sum(frame.data)
                frame_properties['data']['Frame COG poloidal']['avg'][i_frames]=frame_properties['data']['Frame COG poloidal']['max'][i_frames]

                #The number of structures in a frame
                frame_properties['data']['Str number']['max'][i_frames]=n_str
                frame_properties['data']['Str number']['avg'][i_frames]=n_str

                #Calculate the distance from the separatrix
                try:
                # if True:
                    for key in ['max','avg']:
                            frame_properties['data']['Separatrix dist'][key][i_frames]=np.min(np.sqrt((frame_properties['data']['Position radial'][key][i_frames]-R_sep_GPI_interp)**2 +
                                                                                                      (frame_properties['data']['Position poloidal'][key][i_frames]-z_sep_GPI_interp)**2))
                            ind_z_min=np.argmin(np.abs(z_sep_GPI-frame_properties['data']['Position poloidal'][key][i_frames]))
                            if z_sep_GPI[ind_z_min] >= frame_properties['data']['Position poloidal'][key][i_frames]:
                                ind1=ind_z_min
                                ind2=ind_z_min+1
                            else:
                                ind1=ind_z_min-1
                                ind2=ind_z_min

                            radial_distance=frame_properties['data']['Position radial'][key][i_frames]- \
                                ((frame_properties['data']['Position poloidal'][key][i_frames]-z_sep_GPI[ind2])/ \
                                 (z_sep_GPI[ind1]-z_sep_GPI[ind2])*(R_sep_GPI[ind1]-R_sep_GPI[ind2])+R_sep_GPI[ind2])
                            if radial_distance < 0:
                                frame_properties['data']['Separatrix dist'][key][i_frames]*=-1
                except:
                    frame_properties['data']['Separatrix dist'][key][i_frames]=np.nan
            else:
                #Setting np.nan if no structure is available
                for key in frame_properties['data'].keys():
                    frame_properties['data'][key]['avg'][i_frames]=np.nan
                    frame_properties['data'][key]['max'][i_frames]=np.nan
                    frame_properties['data'][key]['stddev'][i_frames]=np.nan

                frame_properties['data']['Str number']['max'][i_frames]=0.
                frame_properties['data']['Frame COG radial']['max'][i_frames]=np.nan
                frame_properties['data']['Frame COG poloidal']['max'][i_frames]=np.nan
               # frame_properties['Structures'][i_frames]=None

        if structure_pdf_save:
            pdf_structures.close()
        #Saving results into a pickle file

        pickle.dump(frame_properties,open(pickle_filename, 'wb'))
        if test:
            plt.close()
    else:
        print('--- Loading data from the pickle file ---')
        frame_properties=pickle.load(open(pickle_filename, 'rb'))

        #labels= 'label,born,died'

    highest_label=0
    n_frames=len(frame_properties['structures'])
    differential_keys=['Velocity radial COG',
                       'Velocity poloidal COG',
                       'Velocity radial centroid',
                       'Velocity poloidal centroid',
                       'Velocity radial position',
                       'Velocity poloidal position',

                       'Expansion fraction area',
                       'Expansion fraction axes',
                       'Angular velocity angle',
                       'Angular velocity ALI',
                       ]

    sample_time=frame_properties['Time'][1]-frame_properties['Time'][0]
    for i_frames in range(1,n_frames):

        structures_1=frame_properties['structures'][i_frames-1]
        structures_2=frame_properties['structures'][i_frames]

        if structures_1 is not None and len(structures_1) != 0:
            valid_structure_1=True
        else:
            valid_structure_1=False

        if structures_2 is not None and len(structures_2) != 0:
            valid_structure_2=True
        else:
            valid_structure_2=False

        if valid_structure_1 and valid_structure_2:
            for j_str1 in range(len(structures_1)):
                if structures_1[j_str1]['Label'] is None:
                    structures_1[j_str1]['Label']=highest_label+1
                    structures_1[j_str1]['Born']=True
                    structures_1[j_str1]['Died']=False
                    highest_label += 1

            n_str1_overlap=np.zeros(len(structures_1))
            #Calculating the averages based on the input setting
            n_str1=len(structures_1)
            n_str2=len(structures_2)
            str_overlap_matrix=np.zeros([n_str1,n_str2])
            for j_str2 in range(n_str2):
                for j_str1 in range(n_str1):
                    if structures_2[j_str2]['Half path'].intersects_path(structures_1[j_str1]['Half path']):
                        str_overlap_matrix[j_str1,j_str2]=1
            """
            example str_overlap_matrix = |0,0,0,1| lives
                                         |1,0,1,1| splits into three
                                         |0,0,1,0| lives
                                         |0,0,0,0| dies
                                          ^ split into
                                            ^ is born
                                              ^ merges into
                                                ^ merges into

            """
            #print('merging')
            for j_str2 in range(n_str2):
                merging_indices=np.squeeze(str_overlap_matrix[:,j_str2])

                #No overlap between the new structrure and the old ones
                if np.sum(merging_indices) == 0:
                    structures_2[j_str2]['Label'] = highest_label+1
                    highest_label += 1
                    print('HL: ',highest_label)
                    structures_2[j_str2]['Born'] = True


                #There is one overlap between the current and the previous
                elif np.sum(merging_indices) == 1:
                    ind_str1=np.where(merging_indices == 1)
                    #One and only one overlap
                    if np.sum(str_overlap_matrix[ind_str1[0],:]) == 1:
                        structures_2[j_str2]['Label'] = structures_1[int(ind_str1[0])]['Label']
                        structures_2[j_str2]=calculate_differential_keys(structure_2=structures_2[j_str2],
                                                                          structure_1=structures_1[int(ind_str1[0])],
                                                                          sample_time=sample_time,
                                                                          fit_shape=fit_shape)

                    #If splitting is happening, that's handled later.
                    else:
                        pass
                #Previous structures merge
                elif np.sum(merging_indices) > 1:
                    ind_merge=np.where(merging_indices == 1)
                    #There is merging, but there is no splitting


                    if np.sum(str_overlap_matrix[ind_merge,:]) == np.sum(merging_indices):

                        ind_high=ind_merge[0][0]
                        for ind_str1 in ind_merge[0]:
                            if structures_1[int(ind_high)]['Intensity'] < structures_1[int(ind_str1)]['Intensity']:
                                ind_high=ind_str1
                        structures_2[j_str2]['Label']=structures_1[int(ind_high)]['Label']
                        structures_2[j_str2]=calculate_differential_keys(structure_2=structures_2[j_str2],
                                                                          structure_1=structures_1[int(ind_high)],
                                                                          sample_time=sample_time,
                                                                          fit_shape=fit_shape)
                        for ind_str1 in ind_merge[0]:
                            structures_2[j_str2]['Parent'].append(structures_1[int(ind_str1)]['Label'])
                            structures_1[int(ind_str1)]['Merges']=True
                            structures_1[int(ind_str1)]['Child'].append(structures_2[j_str2]['Label'])

                    else:
                        #This is a weird situation where merges and splits occur at the same time
                        #Should be handled correctly, possibly assigning new labels to everything
                        #and not track anything. The merging/splitting labels should be assigned
                        #that needs further modification of the structures_segmentation.py

                        ind_high1=ind_merge[0][0]
                        for ind_str1 in ind_merge[0]:
                            if structures_1[int(ind_high1)]['Intensity'] < structures_1[int(ind_str1)]['Intensity']:
                                ind_high1=ind_str1

                        ind_high2=0
                        for ind_str1 in ind_merge[0]:
                            for ind_str2 in range(len(str_overlap_matrix[int(ind_str1),:])):
                                if (str_overlap_matrix[int(ind_str1),ind_str2] == 1 and
                                    structures_2[int(ind_high2)]['Intensity'] < structures_2[int(ind_str2)]['Intensity']):
                                    ind_high2=ind_str2

                        for ind_str1 in ind_merge[0]:
                            if np.sum(str_overlap_matrix[ind_str1,:]) == 1:
                                structures_1[int(ind_str1)]['Splits']=False
                            else:
                                structures_1[int(ind_str1)]['Splits']=True
                                ind_split=np.where(str_overlap_matrix[ind_str1,:] == 1)
                                for ind_str2 in ind_split[0]:
                                    if int(ind_str2) != int(ind_high2):
                                        structures_2[int(ind_str2)]['Label']=highest_label+1
                                        highest_label += 1
                                        structures_2[int(ind_str2)]['Parent']=structures_1[int(ind_high1)]['Label']
                                    else:
                                        structures_2[int(ind_high2)]['Label']=structures_1[int(ind_high1)]['Label']
                                        structures_2[int(ind_high2)]=calculate_differential_keys(structure_2=structures_2[int(ind_high2)],
                                                                                                  structure_1=structures_1[int(ind_high1)],
                                                                                                  sample_time=sample_time,
                                                                                                  fit_shape=fit_shape)

                                        structures_2[int(ind_high2)]['Parent']=structures_1[int(ind_high1)]['Label']
                                    structures_1[int(ind_str1)]['Child'].append(structures_2[ind_str2]['Label'])
                            structures_1[int(ind_str1)]['Merges']=True


                        print('Splitting and merging is occurring at the same time at frame #'+str(i_frames)+', t='+str(frame_properties['time'][i_frames]*1e3)+'ms')
                        if test:
                            print(str_overlap_matrix[ind_merge,:], merging_indices)

                            print('str1 label ',[structures_1[i]['Label'] for i in range(len(structures_1))])
                            print('str2 label ',[structures_2[i]['Label'] for i in range(len(structures_2))])
                            print('str1 child ',[structures_1[i]['Child'] for i in range(len(structures_1))])
                            print('str2 parent ',[structures_2[i]['Parent'] for i in range(len(structures_2))])
                            print('str1 splits ',[structures_1[i]['Splits'] for i in range(len(structures_1))])
                            print('str1 merges ',[structures_1[i]['Merges'] for i in range(len(structures_1))])

                            print(str_overlap_matrix)

            #print('splitting')
            for j_str1 in range(n_str1):
                splitting_indices=np.squeeze(str_overlap_matrix[j_str1,:])

                #Structure dies
                if np.sum(splitting_indices) == 0:
                    structures_1[j_str1]['Died'] = True

                #There is one and only one overlap, taken care of previously, here for completeness
                elif np.sum(splitting_indices) == 1:
                    pass

                #Previous structures are splitting into more new structures
                elif np.sum(splitting_indices) > 1:
                    ind_split=np.where(splitting_indices == 1)
                    if np.sum(str_overlap_matrix[:,ind_split]) == np.sum(splitting_indices):
                        ind_high=ind_split[0][0]
                        for ind_str2 in ind_split[0]:
                            if structures_2[int(ind_high)]['Intensity'] < structures_2[int(ind_str2)]['Intensity']:
                                ind_high=ind_str2
                        for ind_str2 in ind_split[0]:
                            if structures_2[int(ind_str2)]['Label'] is not None:
                                print([structures_1[i]['Label'] for i in range(len(structures_1))])
                                print([structures_2[i]['Label'] for i in range(len(structures_2))])
                                print('str_overlap_matrix')
                                print(str_overlap_matrix)
                                print('splitting_indices',splitting_indices)

                                print('ind_split', ind_split)
                                print("2", structures_2[ind_str2]['Label'])
                                raise ValueError('The code needs to be reworked because str2 should not have a label and it is a serious exception.')

                            if ind_str2 == ind_high:
                                structures_2[ind_str2]['Label']=structures_1[j_str1]['Label']
                                structures_2[ind_str2]=calculate_differential_keys(structure_2=structures_2[ind_str2],
                                                                                    structure_1=structures_1[j_str1],
                                                                                    sample_time=sample_time,
                                                                                    fit_shape=fit_shape)
                            else:
                                structures_2[ind_str2]['Label']=highest_label+1
                                highest_label += 1
                                print('HL: ',highest_label)
                            structures_2[ind_str2]['Parent'].append(structures_1[j_str1]['Label'])
                            structures_1[j_str1]['Child'].append(structures_2[ind_str2]['Label'])
                        structures_1[j_str1]['Splits']=True
                    else:
                        #This was handled in the previous case at the end of merging.
                        pass
            frame_properties['structures'][i_frames-1]=structures_1.copy()
            frame_properties['structures'][i_frames]=structures_2.copy()

            if calculate_rough_diff_velocities:
                for j_str2 in range(n_str2):
                    prev_str_weight=[]
                    for new_key in differential_keys:
                        structures_2[j_str2][new_key]=[]

                    #Check the new frame if it has overlap with the old frame
                    for j_str1 in range(n_str1):
                        if structures_2[j_str2]['Half path'].intersects_path(structures_1[j_str1]['Half path']):
                            if prev_str_weighting == 'number':
                                prev_str_weight.append(1.)
                            elif prev_str_weighting == 'intensity':
                                prev_str_weight.append(structures_1[j_str1]['Intensity'])
                            elif prev_str_weighting == 'area':
                                prev_str_weight.append(structures_1[j_str1]['Area'])
                            elif prev_str_weighting == 'max_intensity':
                                if np.argmax(intensities) == j_str1:
                                    prev_str_weight.append(1.)
                                else:
                                    prev_str_weight.append(0.)

                            structures_2[j_str2]['Velocity radial COG'].append((structures_2[j_str2]['Polygon'].center_of_gravity[0]-
                                                                               structures_1[j_str1]['Polygon'].center_of_gravity[0])/sample_time)
                            structures_2[j_str2]['Velocity poloidal COG'].append((structures_2[j_str2]['Polygon'].center_of_gravity[1]-
                                                                                 structures_1[j_str1]['Polygon'].center_of_gravity[1])/sample_time)

                            structures_2[j_str2]['Velocity radial centroid'].append((structures_2[j_str2]['Polygon'].centroid[0]-
                                                                                    structures_1[j_str1]['Polygon'].centroid[0])/sample_time)
                            structures_2[j_str2]['Velocity poloidal centroid'].append((structures_2[j_str2]['Polygon'].centroid[1]-
                                                                                      structures_1[j_str1]['Polygon'].centroid[1])/sample_time)

                            structures_2[j_str2]['Velocity radial position'].append((structures_2[j_str2][fit_shape].center[0]-
                                                                                    structures_1[j_str1][fit_shape].center[0])/sample_time)
                            structures_2[j_str2]['Velocity poloidal position'].append((structures_2[j_str2][fit_shape].center[1]-
                                                                                      structures_1[j_str1][fit_shape].center[1])/sample_time)

                            structures_2[j_str2]['Expansion fraction area'].append(np.sqrt(structures_2[j_str2]['Polygon'].area/
                                                                                           structures_1[j_str1]['Polygon'].area))
                            structures_2[j_str2]['Expansion fraction axes'].append(np.sqrt(structures_2[j_str2][fit_shape].axes_length[0]/
                                                                                           structures_1[j_str1][fit_shape].axes_length[0]*
                                                                                           structures_2[j_str2][fit_shape].axes_length[1]/
                                                                                           structures_1[j_str1][fit_shape].axes_length[1]))

                            structures_2[j_str2]['Angular velocity angle'].append((structures_2[j_str2][fit_shape].angle-
                                                                                  structures_1[j_str1][fit_shape].angle)/sample_time)
                            structures_2[j_str2]['Angular velocity ALI'].append((structures_2[j_str2]['Polygon'].principal_axes_angle-
                                                                                structures_1[j_str1]['Polygon'].principal_axes_angle)/sample_time)

                            n_str1_overlap[j_str1]+=1.

                    prev_str_weight=np.asarray(prev_str_weight)
                    prev_str_weight /= np.sum(prev_str_weight)

                    #structures_2[j_str2]['Label']=np.mean(structures_2[j_str2]['Label'])
                    for key in differential_keys:
                        structures_2[j_str2][key] = np.sum(np.asarray(structures_2[j_str2][key])*prev_str_weight)

            """
            Frame property filling up
            """
            n_str2=len(structures_2)
            areas=np.zeros(n_str2)
            intensities=np.zeros(n_str2)

            for j_str2 in range(n_str2):
                #Average size calculation based on the number of structures
                areas[j_str2]=structures_2[j_str2]['Area']
                intensities[j_str2]=structures_2[j_str2]['Intensity']

            areas /= np.sum(areas)
            intensities /= np.sum(intensities)
            #Calculating the averages based on the input setting
            if weighting == 'number':
                weight=np.zeros(n_str2)
                weight[:]=1./n_str2
            elif weighting == 'intensity':
                weight=intensities/np.sum(intensities)
            elif weighting == 'area':
                weight=areas/np.sum(areas)

            if maxing == 'area':
                ind_max=np.argmax(areas)
            elif maxing == 'intensity':
                ind_max=np.argmax(intensities)

            for key in differential_keys:
                for j_str2 in range(len(structures_2)):
                    try:
                        frame_properties['derived'][key]['avg'][i_frames] += structures_2[j_str2][key]*weight[j_str2]
                    except:
                        pass
                try:
                    frame_properties['derived'][key]['max'][i_frames]=structures_2[ind_max][key]
                except:
                    frame_properties['derived'][key]['max'][i_frames]=np.nan

        else:
            for key in differential_keys:
                frame_properties['derived'][key]['avg'][i_frames]=np.nan
                frame_properties['derived'][key]['max'][i_frames]=np.nan

    if remove_orphans:
        for i_frames in range(1,n_frames-1):
            try:
                prev_structures=frame_properties['structures'][i_frames-1]
                prev_labels=[prev_structures[i_str_prev]['Label'] for i_str_prev in range(len(prev_structures))]
            except:
                prev_labels=[]
            try:
                curr_structures=frame_properties['structures'][i_frames]
                curr_labels=[curr_structures[i_str_curr]['Label'] for i_str_curr in range(len(curr_structures))]
            except:
                curr_labels=[]
            try:
                next_structures=frame_properties['structures'][i_frames+1]
                next_labels=[next_structures[i_str_next]['Label'] for i_str_next in range(len(next_structures))]
            except:
                next_labels=[]


            n_curr_labels=len(curr_labels)
            for ind_label in range(n_curr_labels-1,-1,-1):
                if (curr_labels[ind_label] not in prev_labels and
                    curr_labels[ind_label] not in next_labels):
                    frame_properties['structures'][i_frames].pop(ind_label)


    if (structure_video_save):
        cv2.destroyAllWindows()
        video.release()
        del video

    """
    PLOTTING THE RESULTS
    """

    if not filename_was_none and not time_range is None:
        sample_time=frame_properties['Time'][1]-frame_properties['Time'][0]
        if time_range[0] < frame_properties['Time'][0]-sample_time or time_range[1] > frame_properties['Time'][-1]+sample_time:
            raise ValueError('Please run the calculation again with the timerange. The pickle file doesn\'t have the desired range')
    if time_range is None:
        time_range=[frame_properties['Time'][0],frame_properties['Time'][-1]]

    #Plotting the results
    if plot or pdf:
        #This is a bit unusual here, but necessary due to the structure size calculation based on the contours which are not plot
        if plot:
            import matplotlib
            matplotlib.use('QT5Agg')
            #import matplotlib.pyplot as plt
        else:
            import matplotlib
            matplotlib.use('agg')
           # import matplotlib.pyplot as plt

        if plot_time_range is not None:
            if plot_time_range[0] < time_range[0] or plot_time_range[1] > time_range[1]:
                raise ValueError('The plot time range is not in the interval of the original time range.')
            time_range=plot_time_range

        plot_index_structure=np.logical_and(np.logical_not(np.isnan(frame_properties['data']['Elongation']['avg'])),
                                            np.logical_and(frame_properties['Time'] >= time_range[0],
                                                           frame_properties['Time'] <= time_range[1]))

        #Plotting the radial velocity
        if pdf:
            wd=flap.config.get_all_section('Module NSTX_GPI')['Working directory']
            if plot_str_by_str:
                comment+='_sbs'
            filename=flap_nstx.tools.filename(exp_id=exp_id,
                                              working_directory=wd+'/plots',
                                              time_range=time_range,
                                              purpose='ccf velocity',
                                              comment=comment)

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

        keys=list(frame_properties['data'].keys())

        if not plot_str_by_str:
            for i in range(len(keys)):
                fig, ax = plt.subplots(figsize=figsize)
                if plot_vertical_line_at is not None:
                    ax.axvline(x=plot_vertical_line_at,
                             color='red')
                ax.plot(frame_properties['Time'][plot_index_structure],
                        frame_properties['data'][keys[i]]['max'][plot_index_structure],
                        color='tab:blue')
                if plot_scatter:
                    ax.scatter(frame_properties['Time'][plot_index_structure],
                               frame_properties['data'][keys[i]]['max'][plot_index_structure],
                                s=5,
                                marker='o',
                                color='tab:blue')
                if overplot_average:
                    ax.plot(frame_properties['time'][plot_index_structure],
                            frame_properties['data'][keys[i]]['avg'][plot_index_structure],
                            linewidth=0.5,
                            color='red',)

                ax.set_xlabel('Time $[\mu s]$')
                ax.set_ylabel(frame_properties['data'][keys[i]]['label']+' '+'['+frame_properties['data'][keys[i]]['unit']+']')
                ax.set_xlim(time_range)
                ax.set_title(str(keys[i])+ ' vs. time')
                if plot_for_publication:
                    x1,x2=ax.get_xlim()
                    y1,y2=ax.get_ylim()
                    ax.set_aspect((x2-x1)/(y2-y1)/1.618)
                fig.tight_layout(pad=0.1)

                if pdf:
                    pdf_pages.savefig()

            keys=list(frame_properties['derived'].keys())

            for i in range(len(keys)):
                fig, ax = plt.subplots(figsize=figsize)
                ax.plot(frame_properties['Time'][plot_index_structure],
                        frame_properties['derived'][keys[i]]['max'][plot_index_structure],
                        color='tab:blue')

                if plot_scatter:
                    ax.scatter(frame_properties['Time'][plot_index_structure],
                               frame_properties['derived'][keys[i]]['max'][plot_index_structure],
                                s=5,
                                marker='o',
                                color='tab:blue')

                if overplot_average:
                    ax.plot(frame_properties['Time'][plot_index_structure],
                            frame_properties['derived'][keys[i]]['avg'][plot_index_structure],
                            linewidth=0.5,
                            color='red',)

                ax.set_xlabel('Time $[\mu s]$')
                ax.set_ylabel(frame_properties['derived'][keys[i]]['label']+' '+'['+frame_properties['derived'][keys[i]]['unit']+']')
                ax.set_xlim(time_range)
                ax.set_title(str(keys[i])+ ' vs. time')

                if plot_for_publication:
                    x1,x2=ax.get_xlim()
                    y1,y2=ax.get_ylim()
                    ax.set_aspect((x2-x1)/(y2-y1)/1.618)
                fig.tight_layout(pad=0.1)

                if pdf:
                    pdf_pages.savefig()
        else:

            frame_properties['structures'][i_frames]
            max_str_label=0
            for i_frames in range(len(frame_properties['structures'])):
                if frame_properties['structures'][i_frames] is not None:
                    for j_str in range(len(frame_properties['structures'][i_frames])):
                        current_label=frame_properties['structures'][i_frames][j_str]['Label']
                        if current_label is not None and current_label > max_str_label:
                            max_str_label=current_label
            struct_by_struct=[]
            for ind in range(max_str_label):
                struct_by_struct.append({'Time':[],}.copy())

            analyzed_keys=['Centroid radial', 'Centroid poloidal',
                           'Position radial', 'Position poloidal',
                           'Position radial', 'Position poloidal',
                           'Area',
                           'Center of gravity radial', 'Center of gravity poloidal',

                           'Axes length minor','Axes length major',
                           'Size radial', 'Size poloidal',

                           'Angle', 'Elongation',
                           'Velocity radial COG', 'Velocity poloidal COG', 'Velocity radial centroid',
                           'Velocity poloidal centroid', 'Velocity radial position',
                           'Velocity poloidal position', 'Expansion fraction area',
                           'Expansion fraction axes', 'Angular velocity angle', 'Angular velocity ALI'
                           ]

            for i_frames in range(len(frame_properties['structures'])):
                if frame_properties['structures'][i_frames] is not None:
                    for j_str in range(len(frame_properties['structures'][i_frames])):
                        if (frame_properties['structures'][i_frames][j_str]['Label'] is not None and
                            frame_properties['structures'][i_frames][j_str]['Label'] != []):

                            ind_structure=int(frame_properties['structures'][i_frames][j_str]['Label'])-1
                            print(ind_structure)
                            for key_str in frame_properties['structures'][i_frames][j_str].keys():
                                if key_str in analyzed_keys:
                                    if key_str not in struct_by_struct[ind_structure].keys():
                                        struct_by_struct[ind_structure][key_str]=[]
                                    struct_by_struct[ind_structure][key_str].append(frame_properties['structures'][i_frames][j_str][key_str])

                            struct_by_struct[ind_structure]['Time'].append(frame_properties['Time'][i_frames])

            #return struct_by_struct

            for key in analyzed_keys:
                fig, ax = plt.subplots(figsize=figsize)

                if key not in ['Velocity radial COG', 'Velocity poloidal COG', 'Velocity radial centroid',
                               'Velocity poloidal centroid', 'Velocity radial position',
                               'Velocity poloidal position', 'Expansion fraction area',
                               'Expansion fraction axes', 'Angular velocity angle', 'Angular velocity ALI']:
                    for ind_str in range(len(struct_by_struct)):
                        try:
                            if struct_by_struct[ind_str]['Time'] !=[]:
                                ax.plot(struct_by_struct[ind_str]['Time'],
                                        struct_by_struct[ind_str][key],
                                        label=str(ind_str))

                                if plot_scatter:
                                    ax.scatter(struct_by_struct[ind_str]['Time'],
                                               struct_by_struct[ind_str][key],
                                               label=str(ind_str),
                                               s=5,
                                               marker='o')
                        except:
                            pass
                    ax.set_ylabel(frame_properties['data'][key]['label']+' '+'['+frame_properties['data'][key]['unit']+']')
                else:
                    for ind_str in range(len(struct_by_struct)):
                        try:
                            if struct_by_struct[ind_str]['Time'] !=[]:
                                ax.plot(struct_by_struct[ind_str]['Time'][1:],
                                        struct_by_struct[ind_str][key],
                                        label=str(ind_str))

                                if plot_scatter:
                                    ax.scatter(struct_by_struct[ind_str]['Time'][1:],
                                               struct_by_struct[ind_str][key],
                                               label=str(ind_str),
                                               s=5,
                                               marker='o')
                        except:
                            pass
                    ax.set_ylabel(frame_properties['derived'][key]['label']+' '+'['+frame_properties['derived'][key]['unit']+']')

                ax.set_xlabel('Time $[\mu s]$')
                ax.set_xlim(time_range)
                ax.set_title(str(key)+ ' vs. time')

                if plot_for_publication:
                    x1,x2=ax.get_xlim()
                    y1,y2=ax.get_ylim()
                    ax.set_aspect((x2-x1)/(y2-y1)/1.618)
                fig.tight_layout(pad=0.1)

                if pdf:
                    pdf_pages.savefig()
                plt.cla()

        if pdf:
           pdf_pages.close()
        if plot_for_publication:
            import matplotlib.style as pltstyle
            pltstyle.use('default')

    if return_results:
        return frame_properties

#Wrapper function for calculating differential key results.
def calculate_differential_keys(structure_2=None,
                                structure_1=None,
                                sample_time=None,
                                fit_shape='Ellipse',
                                ):

    structure_2['Velocity radial COG'] = (structure_2['Polygon'].center_of_gravity[0]-
                                          structure_1['Polygon'].center_of_gravity[0])/sample_time
    structure_2['Velocity poloidal COG'] = (structure_2['Polygon'].center_of_gravity[1]-
                                            structure_1['Polygon'].center_of_gravity[1])/sample_time
    structure_2['Velocity radial centroid']=(structure_2['Polygon'].centroid[0]-
                                             structure_1['Polygon'].centroid[0])/sample_time
    structure_2['Velocity poloidal centroid']=(structure_2['Polygon'].centroid[1]-
                                               structure_1['Polygon'].centroid[1])/sample_time
    structure_2['Velocity radial position']=(structure_2[fit_shape].center[0]-
                                             structure_1[fit_shape].center[0])/sample_time
    structure_2['Velocity poloidal position']=(structure_2[fit_shape].center[1]-
                                               structure_1[fit_shape].center[1])/sample_time
    structure_2['Expansion fraction area']=np.sqrt(structure_2['Polygon'].area/
                                                   structure_1['Polygon'].area)
    structure_2['Expansion fraction axes']=np.sqrt(structure_2[fit_shape].axes_length[0]/
                                                   structure_1[fit_shape].axes_length[0]*
                                                   structure_2[fit_shape].axes_length[1]/
                                                   structure_1[fit_shape].axes_length[1])
    structure_2['Angular velocity angle']=(structure_2[fit_shape].angle-
                                           structure_1[fit_shape].angle)/sample_time
    structure_2['Angular velocity ALI']=(structure_2['Polygon'].principal_axes_angle-
                                         structure_1['Polygon'].principal_axes_angle)/sample_time
    return structure_2