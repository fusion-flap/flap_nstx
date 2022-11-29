#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 12:55:24 2021

@author: mlampert
"""
#Core modules
import os
import copy
import pickle

import flap
import flap_nstx

flap_nstx.register()
from flap_nstx.tools import phase_cross_correlation_mod_ml

import flap_mdsplus
flap_mdsplus.register('NSTX_MDSPlus')

thisdir = os.path.dirname(os.path.realpath(__file__))
fn = os.path.join(thisdir,"../flap_nstx.cfg")
flap.config.read(file_name=fn)

#Scientific modules

import matplotlib.pyplot as plt

from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MaxNLocator

import numpy as np

from skimage.filters import window, difference_of_gaussians
from skimage.registration import phase_cross_correlation
from skimage.transform import warp_polar, rotate, rescale
#from skimage.util import img_as_float                                      #It's questionable if this has to be used or not

global act_plot_id, gca_invalid

def calculate_nstx_gpi_angular_velocity(exp_id=None,                                #Shot number
                                        time_range=None,                            #The time range for the calculation
                                        data_object=None,                           #Input data object if available from outside (e.g. generated sythetic signal)
                                        x_range=None,                               #X range for the calculation
                                        y_range=None,                               #Y range for the calculation

                                        #Normalizer inputs
                                        normalize_f_kernel='Elliptic',              #The kernel for filtering the gas cloud
                                        normalize_f_high=1e3,                       #High pass frequency for the normalizer data

                                        normalize='roundtrip',                      #Inputs for velocity pre processing
                                        normalize_for_velocity=True,                #Normalize the signal for velocity processing
                                        subtraction_order_for_velocity=2,           #Order of the 2D polynomial for background subtraction
                                        nlevel=51,

                                        #Inputs for velocity processing
                                        correlation_threshold=0.6,                  #Threshold for the maximum of the cross-correlation between the two frames (calculated with the actual maximum, not the fit value)
                                        frame_similarity_threshold=0.0,             #Similarity threshold between the two subsequent frames (CCF at zero lag) DEPRECATED when an ELM is present
                                        valid_frame_thres=5,                        #The number of consequtive frames to consider the calculation valid.
                                        fitting_range=5,                            #Fitting range of the peak of CCF

                                        #Options for angular velocity processing
                                        zero_padding_scale=1,
                                        upsample_factor=16,
                                        gaussian_blur=True,
                                        sigma_low=1,                                #Gaussian blurring bandpass highpass sigma
                                        sigma_high=None,                            #Gaussian blurring bandpass lowpass sigma, should be higher than sigma_low
                                        hann_window=False,                          #Calculate results with a Hann shaped window
                                        warp_polar_order=1,                         #Order of the log-polar transformation's interpolation
                                        log_polar_data_shape=None,
                                        calculate_half_fft=True,
                                        calculate_retransformation=False,           #Retransform the image from the rotation and calculate the velocity with that (produces noisy output)

                                        #Plot options:
                                        plot=True,                                  #Plot the results
                                        pdf=False,                                  #Print the results into a PDF
                                        pdf_filename=None,
                                        plot_gas=True,                              #Plot the gas cloud parameters on top of the other results from the structure size calculation
                                        plot_error=False,                           #Plot the errorbars of the velocity calculation based on the line fitting and its RMS error
                                        error_window=4.,                            #Plot the average signal with the error bars calculated from the normalized variance.
                                        plot_scatter=True,
                                        plot_time_range=None,                       #Plot the results in a different time range than the data is read from
                                        plot_for_publication=False,                 #Modify the plot sizes to single column sizes and golden ratio axis ratios
                                        plot_ccf=False,
                                        plot_skimage=False,                         #Splot scikit-image results instead of FLAP (only when plot_for_publication)

                                        sample_to_plot=None,                        #The sample numbers which should be plot for the paper. Should be a two element list.
                                        plot_sample_frames=True,
                                        save_data_for_publication=False,
                                        data_filename=None,

                                        #File input/output options
                                        filename=None,                              #Filename for restoring data
                                        save_results=True,                          #Save the results into a .pickle file to filename+.pickle
                                        nocalc=True,                                #Restore the results from the .pickle file from filename+.pickle

                                        #Output options:
                                        return_results=False,                       #Return the results if set.
                                        new_scheme_return=False,
                                        cache_data=True,                            #Cache the data or try to open is from cache

                                        #Test options
                                        test=False,                                 #Test the results
                                        test_into_pdf=False,
                                        test_mod_skimage=False,
                                        ):                                          #Sad face right there :D


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
    coeff_r_new=3./800.
    coeff_z_new=3./800.

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

    wd=flap.config.get_all_section('Module NSTX_GPI')['Working directory']

    if data_object is not None and type(data_object) == str:
        if exp_id is None:
            exp_id='*'
        d=flap.get_data_object(data_object, exp_id=exp_id)
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

    comment='pfit_o'+str(subtraction_order_for_velocity)+\
            '_fst_'+str(frame_similarity_threshold)


    if filename is None:
        filename=flap_nstx.tools.filename(exp_id=exp_id,
                                          working_directory=wd+'/processed_data',
                                          time_range=time_range,
                                          purpose='ccf ang velocity',
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

    if test_into_pdf:
        import matplotlib
        matplotlib.use('agg')
        if pdf_filename is None:
            filename=flap_nstx.tools.filename(exp_id=exp_id,
                                              working_directory=wd+'/plots',
                                              time_range=time_range,
                                              purpose='ccf ang velocity testing',
                                              comment=comment+'_ct_'+str(correlation_threshold))
            pdf_filename_test=filename+'.pdf'
            pdf_test=PdfPages(pdf_filename_test)
        else:
            pdf_test=PdfPages(pdf_filename)
    else:
        pdf_test=None

    if plot_ccf:
        if pdf:
            if pdf_filename is None:
                pdf_filename=flap_nstx.tools.filename(exp_id=exp_id,
                                                      working_directory=wd+'/plots',
                                                      time_range=time_range,
                                                      purpose='ang vel ccf',
                                                      comment=comment+'_ct_'+str(correlation_threshold),
                                                      extension='pdf')
            pdf_object_ccf=PdfPages(pdf_filename)
    import matplotlib.pyplot as plt

    if not nocalc:
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

            if (fitting_range*2+1 > np.abs(x_range[1]-x_range[0]) or
                fitting_range*2+1 > np.abs(y_range[1]-y_range[0])):
                raise ValueError('The fitting range for the parabola is too large for the given coordinate range.')
            d=flap.slice_data('GPI',exp_id=exp_id,
                              slicing=slicing,
                              output_name='GPI_SLICED_FULL')

        object_name_ccf_velocity='GPI_SLICED_FULL'

        #Normalize data for size calculation
        print("**** Calculating the gas cloud ****")

        normalizer_object_name='GPI_LPF_INTERVAL'

        slicing_for_filtering=copy.deepcopy(slicing)
        slicing_for_filtering['Time']=flap.Intervals(time_range[0]-1/normalize_f_high*10,
                                                     time_range[1]+1/normalize_f_high*10)


        if data_object is None:
            flap.slice_data('GPI',
                            exp_id=exp_id,
                            slicing=slicing_for_filtering,
                            output_name='GPI_SLICED_FOR_FILTERING')

            # #Roundtrip normalization
            if normalize_for_velocity:
                norm_obj=flap.filter_data('GPI_SLICED_FOR_FILTERING',
                                          exp_id=exp_id,
                                          coordinate='Time',
                                          options={'Type':'Lowpass',
                                                    'f_high':normalize_f_high,
                                                    'Design':normalize_f_kernel},
                                          output_name=normalizer_object_name)

                norm_obj.data=np.flip(norm_obj.data,axis=0)
                norm_obj=flap.filter_data(normalizer_object_name,
                                          exp_id=exp_id,
                                          coordinate='Time',
                                          options={'Type':'Lowpass',
                                                    'f_high':normalize_f_high,
                                                    'Design':normalize_f_kernel},
                                          output_name=normalizer_object_name)

                norm_obj.data=np.flip(norm_obj.data,axis=0)
                coefficient=flap.slice_data(normalizer_object_name,
                                            exp_id=exp_id,
                                            slicing=slicing,
                                            output_name='GPI_GAS_CLOUD').data

                data_obj=flap.get_data_object(object_name_ccf_velocity,exp_id=exp_id)
                data_obj.data = data_obj.data/coefficient
                flap.add_data_object(data_obj, 'GPI_SLICED_DENORM_CCF_VEL')
                object_name_ccf_velocity='GPI_SLICED_DENORM_CCF_VEL'
        else:
            pass


        #Subtract trend from data
        if subtraction_order_for_velocity is not None:
            print("*** Subtracting the trend of the data ***")
            d=flap_nstx.tools.detrend_multidim(object_name_ccf_velocity,
                                               exp_id=exp_id,
                                               order=subtraction_order_for_velocity,
                                               coordinates=['Image x', 'Image y'],
                                               output_name='GPI_DETREND_VEL')
            object_name_ccf_velocity='GPI_DETREND_VEL'
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
                          'GPI Dalpha':dalpha,
                          'Correlation max':np.zeros(len(time)-2),

                          'Frame similarity':np.zeros(len(time)-2),

                          'Velocity ccf FLAP':np.zeros([len(time)-2,2]),
                          'Correlation max polar FLAP':np.zeros(len(time)-2),
                          'Correlation max polar FLAP log':np.zeros(len(time)-2),
                          'Angular velocity ccf FLAP':np.zeros([len(time)-2]),
                          'Angular velocity ccf FLAP log':np.zeros([len(time)-2]),
                          'Angle difference FLAP':np.zeros([len(time)-2]),
                          'Angle difference FLAP log':np.zeros([len(time)-2]),
                          'Expansion velocity ccf FLAP':np.zeros([len(time)-2]),

                          'Velocity ccf skim':np.zeros([len(time)-2,2]),
                          'Correlation max polar':np.zeros(len(time)-2),
                          'Correlation max polar log':np.zeros(len(time)-2),
                          'Angular velocity ccf':np.zeros([len(time)-2]),
                          'Angular velocity ccf log':np.zeros([len(time)-2]),
                          'Angle difference':np.zeros([len(time)-2]),
                          'Angle difference log':np.zeros([len(time)-2]),
                          'Expansion velocity ccf':np.zeros([len(time)-2]),
                         }

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
        if log_polar_data_shape is None:
            log_polar_data_shape=(360,max(d.data.shape))

        ccf_data_polar=np.zeros(log_polar_data_shape)

        coord_polar=[]

        coord_polar.append(copy.deepcopy(flap.Coordinate(name='Angle',
                                                         unit='Pixel',
                                                         mode=flap.CoordinateMode(equidistant=True),
                                                         start=0,
                                                         step=1,
                                                         shape=[],
                                                         dimension_list=[0]
                                                         )))

        coord_polar.append(copy.deepcopy(flap.Coordinate(name='Radius',
                                                         unit='Pixel',
                                                         mode=flap.CoordinateMode(equidistant=True),
                                                         start=0,
                                                         step=1,
                                                         shape=[],
                                                         dimension_list=[1]
                                                         )))

        frame1_polar_fft_object = flap.DataObject(data_array=ccf_data_polar,
                                                  data_unit=flap.Unit(name='Signal',unit='Digit'),
                                                  coordinates=coord_polar,
                                                  exp_id=exp_id,
                                                  data_title='',
                                                  data_source="NSTX_GPI")

        frame2_polar_fft_object = copy.deepcopy(frame1_polar_fft_object)

        flap.add_data_object(frame1_polar_fft_object, 'GPI_FRAME_1_FFT_POLAR')
        flap.add_data_object(frame2_polar_fft_object, 'GPI_FRAME_2_FFT_POLAR')

        #Velocity is calculated between the two subsequent frames, the velocity time is the second frame's time
        delta_index_flap=np.zeros(2)

        #Inicializing for frame handling
        frame2=None
        invalid_correlation_frame_counter=0.

        frame2_fft_polar_log=0.     #These are only defined for error handling.
        frame2_fft_polar=0.

        if test:
            plt.figure()

        if test_into_pdf:
            import matplotlib
            matplotlib.use('agg')
        if sample_to_plot is not None:
            #gs=GridSpec(3, len(sample_to_plot))
            if plot_sample_frames:
                fig, axs = plt.subplots(3, len(sample_to_plot), figsize=(8.5/2.54, 13.7/2.54))
            else:
                fig, axs = plt.subplots(2, len(sample_to_plot), figsize=(8.5/2.54, 8.5/2.54))


        else:
            axs=None

        for i_frames in range(0,n_frames-2):
            """
            STRUCTURE VELOCITY CALCULATION BASED ON FLAP CCF CALCULATION
            """

            slicing_frame1={'Sample':sample_0+i_frames}
            slicing_frame2={'Sample':sample_0+i_frames+1}
            if frame2 is None:
                frame1=flap.slice_data(object_name_ccf_velocity,
                                       exp_id=exp_id,
                                       slicing=slicing_frame1,
                                       output_name='GPI_FRAME_1')
                frame1.add_coordinate()
            else:
                frame1=copy.deepcopy(frame2)
                flap.add_data_object(frame1,'GPI_FRAME_1')

            frame2=flap.slice_data(object_name_ccf_velocity,
                                   exp_id=exp_id,
                                   slicing=slicing_frame2,
                                   output_name='GPI_FRAME_2')

            sample_number=frame2.coordinate('Sample')[0][0,0]
            frame1.data=np.asarray(frame1.data, dtype='float64')
            frame2.data=np.asarray(frame2.data, dtype='float64')

            flap.ccf('GPI_FRAME_2', 'GPI_FRAME_1',
                     coordinate=['Image x', 'Image y'],
                     options={'Resolution':1,
                              'Range':[[-(x_range[1]-x_range[0]),(x_range[1]-x_range[0])],
                                       [-(y_range[1]-y_range[0]),(y_range[1]-y_range[0])]],
                              'Trend removal':None,
                              'Normalize':True,
                              'Interval_n': 1},
                     output_name='GPI_FRAME_12_CCF')


            ccf_object.data[i_frames,:,:]=flap.get_data_object_ref('GPI_FRAME_12_CCF').data

            if test:
                print('Maximum correlation:'+str(ccf_object.data[i_frames,:,:].max()))


            max_index=np.asarray(np.unravel_index(ccf_object.data[i_frames,:,:].argmax(), ccf_object.data[i_frames,:,:].shape))

            #Fit a 2D polinomial on top of the peak

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

            delta_index_flap=[index[0]+max_index[0]-fitting_range-ccf_object.data[i_frames,:,:].shape[0]//2,
                              index[1]+max_index[1]-fitting_range-ccf_object.data[i_frames,:,:].shape[1]//2]

            if test:
                plt.contourf(ccf_object.data[i_frames,:,:].T, levels=np.arange(0,51)/25-1)
                plt.scatter(index[0]+max_index[0]-fitting_range,
                            index[1]+max_index[1]-fitting_range)

            frame_properties['Correlation max'][i_frames]=ccf_object.data[i_frames,:,:].max()

            frame_properties['Frame similarity'][i_frames]=ccf_object.data[i_frames,:,:][tuple(np.asarray(ccf_object.data[i_frames,:,:].shape)[:]//2)]

            if ccf_object.data[i_frames,:,:].max() < correlation_threshold:
                invalid_correlation_frame_counter+=1
            else:
                invalid_correlation_frame_counter=0.
            #Calculating the radial and poloidal velocity from the correlation map.
            frame_properties['Velocity ccf FLAP'][i_frames,0]=coeff_r_new*delta_index_flap[0]/sample_time
            frame_properties['Velocity ccf FLAP'][i_frames,1]=coeff_z_new*delta_index_flap[1]/sample_time

            """
            STRUCTURE ANGULAR VELOCITY CALCULATION BASED ON FLAP CCF
            """

            radius=min([frame1.data.shape[0],frame1.data.shape[1]])/2
            if gaussian_blur:
                if i_frames == 0 :
                    frame1_filtered = difference_of_gaussians(frame1.data, sigma_low, sigma_high)
                frame2_filtered = difference_of_gaussians(frame2.data, sigma_low, sigma_high)
                flap.add_data_object(frame2,'GPI_FRAME_2_FILTERED')

            else:
                if i_frames == 0 :
                    frame1_filtered=frame1.data
                frame2_filtered=frame2.data
                flap.add_data_object(frame2,'GPI_FRAME_2_FILTERED')
            #The calculations only need to be done for the first frame in the first stop
            #In the following steps the first calculation is the same as the second in the
            #previous step

            shape=frame1_filtered.shape
            if i_frames == 0 :
                if hann_window:
                    frame1_filtered *= window('hann', frame1_filtered.shape)
                frame1_zero_padded=np.zeros(np.asarray(shape)*(2*zero_padding_scale+1))
                frame1_zero_padded[shape[0]*zero_padding_scale:shape[0]*(zero_padding_scale+1),
                                   shape[1]*zero_padding_scale:shape[1]*(zero_padding_scale+1)]=frame1_filtered

            if hann_window:
                frame2_filtered *= window('hann', frame2_filtered.shape)
            frame2_zero_padded=np.zeros(np.asarray(shape)*(2*zero_padding_scale+1))
            frame2_zero_padded[shape[0]*zero_padding_scale:shape[0]*(zero_padding_scale+1),
                               shape[1]*zero_padding_scale:shape[1]*(zero_padding_scale+1)]=frame2_filtered


            if i_frames == 0 :
                frame1_fft=np.absolute( np.fft.fftshift( np.fft.fftn(frame1_zero_padded, axes=[0,1])))
            frame2_fft=np.absolute( np.fft.fftshift( np.fft.fftn(frame2_zero_padded, axes=[0,1])))

            if i_frames == 0 :
                frame1_fft_polar_log = warp_polar(frame1_fft,
                                                  scaling='log',
                                                  radius=radius,
                                                  order=warp_polar_order,
                                                  output_shape=log_polar_data_shape,)
            else:
                frame1_fft_polar_log = copy.deepcopy(frame2_fft_polar_log)
            frame2_fft_polar_log = warp_polar(frame2_fft,
                                              scaling='log',
                                              radius=radius,
                                              order=warp_polar_order,
                                              output_shape=log_polar_data_shape,
                                              )

            if i_frames == 0 :
                frame1_fft_polar = warp_polar(frame1_fft,
                                              radius=radius,
                                              order=warp_polar_order)
            else:
                frame1_fft_polar = copy.deepcopy(frame2_fft_polar)
            frame2_fft_polar = warp_polar(frame2_fft,
                                          radius=radius,
                                          order=warp_polar_order)

            """
            CALCULATION OF THE CCF WITH FLAP
            """

            for i_log_or_not in range(2):
                if i_log_or_not == 0:
                    frame1_polar_fft_object.data=frame1_fft_polar - np.mean(frame1_fft_polar)
                    frame2_polar_fft_object.data=frame2_fft_polar - np.mean(frame2_fft_polar)
                else:
                    frame1_polar_fft_object.data=frame1_fft_polar_log - np.mean(frame1_fft_polar_log)
                    frame2_polar_fft_object.data=frame2_fft_polar_log - np.mean(frame2_fft_polar_log)

                ccf_object_polar=flap.ccf('GPI_FRAME_1_FFT_POLAR',
                                          'GPI_FRAME_2_FFT_POLAR',
                                          coordinate=['Angle', 'Radius'],
                                          options={'Resolution':1,
                                                   'Range':[[-359,359],
                                                             [-radius+1,radius-1]],
                                                   'Trend removal':None,
                                                   'Normalize':True,
                                                   'Interval_n': 1},
                                                   output_name='GPI_FRAME_12_CCF_POLAR')
                if i_log_or_not == 0:
                    frame_properties['Correlation max polar FLAP'][i_frames]=np.max(flap.get_data_object_ref('GPI_FRAME_12_CCF_POLAR').data)
                else:
                    frame_properties['Correlation max polar FLAP log'][i_frames]=np.max(flap.get_data_object_ref('GPI_FRAME_12_CCF_POLAR').data)

                max_index=np.asarray(np.unravel_index(ccf_object_polar.data.argmax(),
                                                      ccf_object_polar.data.shape))

                #Crude way to get rid of a bunch of code from the main part, sry, not sry
                plot_angular_velocity_calc_test(test=test,
                                                test_into_pdf=test_into_pdf,
                                                pdf_test=pdf_test,
                                                sample_number=sample_number,
                                                sample_to_plot=sample_to_plot,
                                                plot_sample_frames=plot_sample_frames,
                                                i_log_or_not=i_log_or_not,
                                                axs=axs,
                                                frame2_fft_polar_log=frame2_fft_polar_log,
                                                frame2_fft=frame2_fft,
                                                save_data_for_publication=save_data_for_publication,
                                                data_filename=data_filename,
                                                time=time,
                                                i_frames=i_frames)

                if (plot_ccf and
                    sample_to_plot is not None and
                    not plot_skimage
                    and i_log_or_not):

                    ccf=flap.get_data_object_ref('GPI_FRAME_12_CCF_POLAR').data
                    xdata=flap.get_data_object_ref('GPI_FRAME_12_CCF_POLAR').coordinate('Radius lag')[0][0,:]
                    ydata=flap.get_data_object_ref('GPI_FRAME_12_CCF_POLAR').coordinate('Angle lag')[0][:,0]

                    plot_angular_velocity_ccf(sample_to_plot=sample_to_plot,
                                              xdata=xdata,
                                              ydata=ydata,
                                              ccf=ccf,
                                              pdf=pdf,
                                              title='FLAP CCF',
                                              pdf_object=pdf_object_ccf,
                                              plot_for_publication=plot_for_publication,
                                              save_data_for_publication=save_data_for_publication,
                                              data_filename=data_filename,
                                              time=time,
                                              sample_number=sample_number,
                                              i_frames=i_frames)

                #Fit a 2D polinomial on top of the peak

                area_max_index=tuple([slice(max_index[0]-fitting_range,
                                            max_index[0]+fitting_range+1),
                                      slice(max_index[1]-fitting_range,
                                            max_index[1]+fitting_range+1)])
                #Finding the peak analytically
                try:
                    coeff=flap_nstx.tools.polyfit_2D(values=ccf_object_polar.data[area_max_index],order=2)
                    index=[0,0]
                    index[0]=(2*coeff[2]*coeff[3]-coeff[1]*coeff[4])/(coeff[4]**2-4*coeff[2]*coeff[5])
                    index[1]=(-2*coeff[5]*index[0]-coeff[3])/coeff[4]
                except:
                    index=[fitting_range,fitting_range]
                delta_index_flap=[index[0]+max_index[0]-fitting_range-ccf_object_polar.data.shape[0]//2,
                                  index[1]+max_index[1]-fitting_range-ccf_object_polar.data.shape[1]//2]
                if test:
                    plt.scatter(delta_index_flap[0],delta_index_flap[1], color='red')
                    #plt.pause(0.5)
                if i_log_or_not == 0:
                    frame_properties['Angular velocity ccf FLAP'][i_frames]=delta_index_flap[0]/180.*np.pi/sample_time #dphi/dt
                    frame_properties['Angle difference FLAP'][i_frames]=delta_index_flap[0]
                else:
                    klog = frame2_fft_polar_log.shape[1] / np.log(radius)
                    shift_scale_log = (np.exp(delta_index_flap[1] / klog))
                    frame_properties['Expansion velocity ccf FLAP'][i_frames]=(shift_scale_log-1.)

                    frame_properties['Angular velocity ccf FLAP log'][i_frames]=delta_index_flap[0]/180.*np.pi/sample_time #dphi/dt
                    frame_properties['Angle difference FLAP log'][i_frames]=delta_index_flap[0]

            """
            CALCULATION BASED ON SKIMAGE PHASE CROSS CORRELATION (PRODUCES MORE ACCURATE RESULTS THAN FLAP)
            """
            if calculate_half_fft:
                frame1_fft_input=frame1_fft_polar[:frame1_fft_polar.shape[0] // 2, :]
                frame2_fft_input=frame2_fft_polar[:frame2_fft_polar.shape[0] // 2, :]
            else:
                frame1_fft_input=frame1_fft_polar
                frame2_fft_input=frame2_fft_polar

            shape=frame1_fft_input.shape
            frame1_input_zero_padded=np.zeros(np.asarray(shape)*(2*zero_padding_scale+1))
            frame1_input_zero_padded[shape[0]*zero_padding_scale:shape[0]*(zero_padding_scale+1),
                                     shape[1]*zero_padding_scale:shape[1]*(zero_padding_scale+1)]=frame1_fft_input
            shape=frame2_fft_input.shape
            frame2_input_zero_padded=np.zeros(np.asarray(shape)*(2*zero_padding_scale+1))
            frame2_input_zero_padded[shape[0]*zero_padding_scale:shape[0]*(zero_padding_scale+1),
                                     shape[1]*zero_padding_scale:shape[1]*(zero_padding_scale+1)]=frame2_fft_input

            shift_rot, error, phasediff, cross_corr_max, cross_corr_lin = phase_cross_correlation_mod_ml(frame1_input_zero_padded,
                                                                                                         frame2_input_zero_padded,
                                                                                                         upsample_factor=upsample_factor,
                                                                                                         polyfit_2D=flap_nstx.tools.polyfit_2D)
            shiftr, shiftc = shift_rot[:2]

            shape_vec=frame1_fft_polar_log.shape

            if calculate_half_fft:
                frame1_fft_log_input=frame1_fft_polar_log[:frame1_fft_polar_log.shape[0] // 2, :]
                frame2_fft_log_input=frame2_fft_polar_log[:frame2_fft_polar_log.shape[0] // 2, :]
            else:
                frame1_fft_log_input=frame1_fft_polar_log
                frame2_fft_log_input=frame2_fft_polar_log

            # shape=frame1_fft_log_input.shape
            # frame1_log_input_zero_padded=np.zeros(np.asarray(shape)*(2*zero_padding_scale+1))
            # frame1_log_input_zero_padded[shape[0]*zero_padding_scale:shape[0]*(zero_padding_scale+1),
            #                          shape[1]*zero_padding_scale:shape[1]*(zero_padding_scale+1)]=frame1_fft_log_input

            # shape=frame2_fft_log_input.shape
            # frame2_log_input_zero_padded=np.zeros(np.asarray(shape)*(2*zero_padding_scale+1))
            # frame2_log_input_zero_padded[shape[0]*zero_padding_scale:shape[0]*(zero_padding_scale+1),
            #                              shape[1]*zero_padding_scale:shape[1]*(zero_padding_scale+1)]=frame2_fft_log_input
            frame1_log_input_zero_padded=frame1_fft_log_input
            frame2_log_input_zero_padded=frame2_fft_log_input

            pc_output = phase_cross_correlation_mod_ml(frame1_log_input_zero_padded,
                                                       frame2_log_input_zero_padded,
                                                       upsample_factor=upsample_factor,
                                                       polyfit_2D=flap_nstx.tools.polyfit_2D)

            (shift_rot_log, error_log, phasediff, cross_corr_max_log, cross_corr_log)=pc_output

            if plot_ccf and plot_skimage:
                xdata=np.arange(cross_corr_log.shape[0])-cross_corr_log.shape[0]//2
                ydata=np.arange(cross_corr_log.shape[1])-cross_corr_log.shape[1]//2

                plot_angular_velocity_ccf(sample_to_plot=sample_to_plot,
                                          sample_number=sample_number,
                                          xdata=ydata,
                                          ydata=xdata,
                                          ccf=np.fft.fftshift(cross_corr_log),

                                          pdf=pdf,
                                          title='skim CCF',
                                          pdf_object=pdf_object_ccf,
                                          plot_for_publication=plot_for_publication,
                                          save_data_for_publication=save_data_for_publication,
                                          data_filename=data_filename,
                                          time=time,
                                          i_frames=i_frames)

            shiftr_log, shiftc_log = shift_rot_log[:2]

            # Calculate scale factor from translation
            klog = shape_vec[1] / np.log(radius)
            shift_scale_log = (np.exp(shiftc_log / klog))

            shiftr=shiftr*360/shape_vec[0]
            shiftr_log=shiftr_log*360/shape_vec[0]


            frame_properties['Correlation max polar'][i_frames]=np.abs(cross_corr_max)
            frame_properties['Correlation max polar log'][i_frames]=np.abs(cross_corr_max_log)

            frame_properties['Angular velocity ccf'][i_frames]=shiftr/180.*np.pi/sample_time #dphi/dt
            frame_properties['Angular velocity ccf log'][i_frames]=shiftr_log/180.*np.pi/sample_time #dphi/dt

            frame_properties['Angle difference'][i_frames]=shiftr
            frame_properties['Angle difference log'][i_frames]=shiftr_log

            frame_properties['Expansion velocity ccf'][i_frames]=shift_scale_log-1


            if calculate_retransformation:
                frame1_retransformed_scaled=rescale(rotate(frame2.data,-shiftr),1/shift_scale_log)
                frame1_retransformed=np.zeros(frame1.data.shape)

                x_addon=0
                y_addon=0

                ret_shape=frame1_retransformed.shape
                scaled_shape=frame1_retransformed_scaled.shape

                if (ret_shape[0]-scaled_shape[0])//2-ret_shape[0]+(ret_shape[0]-scaled_shape[0])//2 != -scaled_shape[0]:
                    x_addon=1

                if (ret_shape[1]-scaled_shape[1])//2-ret_shape[1]+(ret_shape[1]-scaled_shape[1])//2 != -scaled_shape[1]:
                    y_addon=1

                if shift_scale_log > 1:
                    frame1_retransformed[(ret_shape[0]-scaled_shape[0])//2+x_addon:ret_shape[0]-(ret_shape[0]-scaled_shape[0])//2,
                                         (ret_shape[1]-scaled_shape[1])//2+y_addon:ret_shape[1]-(ret_shape[1]-scaled_shape[1])//2]=frame1_retransformed_scaled
                else:
                    frame1_retransformed = frame1_retransformed_scaled[(scaled_shape[0]-ret_shape[0])//2+x_addon:scaled_shape[0]-(scaled_shape[0]-ret_shape[0])//2,
                                                                       (scaled_shape[1]-ret_shape[1])//2+y_addon:scaled_shape[1]-(scaled_shape[1]-ret_shape[1])//2]

                shift_tra, error, phasediff, cross_corr = phase_cross_correlation(frame1.data,
                                                                                  frame1_retransformed,
                                                                                  upsample_factor=upsample_factor)

                frame_properties['Velocity ccf'][i_frames,0] = -1 * coeff_r_new * shift_tra[0] / sample_time
                frame_properties['Velocity ccf'][i_frames,1] = -1 * coeff_z_new * shift_tra[1] / sample_time

            shift_tra, error, phasediff = phase_cross_correlation(frame1.data,
                                                                  frame2.data,
                                                                  upsample_factor=upsample_factor)

            frame_properties['Velocity ccf skim'][i_frames,0] = -1 * coeff_r_new * shift_tra[0] / sample_time
            frame_properties['Velocity ccf skim'][i_frames,1] = -1 * coeff_z_new * shift_tra[1] / sample_time

        pickle.dump(frame_properties,open(pickle_filename, 'wb'))
        if test:
            plt.close()
        if test_into_pdf:
            plt.tight_layout(pad=0.1)
            pdf_test.savefig()
        if plot_ccf and pdf:
            pdf_object_ccf.close()
    else:
        print('--- Loading data from the pickle file ---')
        frame_properties=pickle.load(open(pickle_filename, 'rb'))


    if not filename_was_none and not time_range is None:
        sample_time=frame_properties['Time'][1]-frame_properties['Time'][0]
        if time_range[0] < frame_properties['Time'][0]-sample_time or time_range[1] > frame_properties['Time'][-1]+sample_time:
            raise ValueError('Please run the calculation again with the timerange. The pickle file doesn\'t have the desired range')
    if time_range is None:
        time_range=[frame_properties['Time'][0],frame_properties['Time'][-1]]

    nan_ind=np.where(frame_properties['Correlation max'] < correlation_threshold)
    frame_properties['Velocity ccf FLAP'][nan_ind,:] = [np.nan,np.nan]
    frame_properties['Angular velocity ccf'][nan_ind] = np.nan
    frame_properties['Angular velocity ccf log'][nan_ind] = np.nan
    frame_properties['Angular velocity ccf FLAP'][nan_ind] = np.nan
    frame_properties['Angular velocity ccf FLAP log'][nan_ind] = np.nan
    frame_properties['Expansion velocity ccf'][nan_ind] = np.nan
    frame_properties['Expansion velocity ccf FLAP'][nan_ind] = np.nan

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

        plot_index=np.logical_and(np.logical_not(np.isnan(frame_properties['Velocity ccf FLAP'][:,0])),
                                  np.logical_and(frame_properties['Time'] >= time_range[0],
                                                 frame_properties['Time'] <= time_range[1]))

        #Plotting the radial velocity
        if pdf:
            wd=flap.config.get_all_section('Module NSTX_GPI')['Working directory']
            filename=flap_nstx.tools.filename(exp_id=exp_id,
                                                 working_directory=wd+'/plots',
                                                 time_range=time_range,
                                                 purpose='ccf ang velocity',
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

        """
        RADIAL AND POLOIDAL PLOTTING FROM FLAP
        """

        fig, axs = plt.subplots(2,2, figsize=figsize)
        ax=axs[1,0]
        frame_properties['Time']=np.arange(frame_properties['Time'].shape[0])*2.5-frame_properties['Time'].shape[0]//2*2.5

        if calculate_retransformation:
            ax.plot(frame_properties['Time'][plot_index],
                    frame_properties['Velocity ccf'][plot_index,0]/1e3,
                    label='Velocity ccf recon')
        if plot_for_publication:
            if plot_skimage:
                ax.plot(frame_properties['Time'][plot_index],
                         frame_properties['Velocity ccf skim'][plot_index,0]/1e3,
                         label='Velocity ccf skim')
                ax.set_title('Radial velocity skimage of '+str(exp_id)+'')
            else:
                ax.plot(frame_properties['Time'][plot_index],
                         frame_properties['Velocity ccf FLAP'][plot_index,0]/1e3,
                         label='Velocity ccf FLAP')
                ax.set_title('Radial velocity FLAP of '+str(exp_id))
        else:
            ax.plot(frame_properties['Time'][plot_index],
                     frame_properties['Velocity ccf skim'][plot_index,0]/1e3,
                     label='skim')
            ax.plot(frame_properties['Time'][plot_index],
                     frame_properties['Velocity ccf FLAP'][plot_index,0]/1e3,
                     label='FLAP')
            ax.set_title('Radial velocity FLAP and skimage')
            ax.legend()
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.set_xticks(ticks=[-500,-250,0,250,500])
        ax.set_xlabel('$t-t_{ELM}$ $[\mu s]$')
        ax.set_ylabel('$v_{rad}$ [km/s]')
        ax.set_xlim([-500,500])

        if plot_for_publication:
            x1,x2=ax.get_xlim()
            y1,y2=ax.get_ylim()
            ax.set_aspect((x2-x1)/(y2-y1)/1.618)
            ax.set_title('Radial velocity')


        ax=axs[1,1]

        if calculate_retransformation:
            ax.plot(frame_properties['Time'][plot_index],
                     frame_properties['Velocity ccf'][plot_index,1]/1e3,
                     label='Velocity ccf recon')
        if plot_for_publication:

            if plot_skimage:
                ax.plot(frame_properties['Time'][plot_index],
                         frame_properties['Velocity ccf skim'][plot_index,1]/1e3,
                         label='Velocity ccf skim')
                ax.set_title('Poloidal velocity skimage of '+str(exp_id)+'')
            else:
                # ax.plot(frame_properties['Time'][plot_index],
                #          frame_properties['Velocity ccf FLAP'][plot_index,1]/1e3,
                #          label='Velocity ccf FLAP')
                # ax.set_title('Poloidal velocity FLAP of '+str(exp_id))
                ax.plot(frame_properties['Time'][plot_index],
                          frame_properties['Velocity ccf FLAP'][plot_index,1]/1e3,
                          label='Velocity ccf FLAP')
                ax.set_title('Poloidal velocity FLAP of '+str(exp_id))
        else:
            ax.plot(frame_properties['Time'][plot_index],
                     frame_properties['Velocity ccf skim'][plot_index,1]/1e3,
                     label='skim')
            ax.plot(frame_properties['Time'][plot_index],
                     frame_properties['Velocity ccf FLAP'][plot_index,1]/1e3,
                     label='FLAP')
            ax.set_title('Poloidal velocity FLAP and skimage')
            ax.legend()

        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.set_xticks(ticks=[-500,-250,0,250,500])
        ax.set_xlabel('$t-t_{ELM}$ $[\mu s]$')
        #ax.set_ylabel('$v_{pol}$ [km/s]')
        ax.set_ylabel('$r-r_{sep}$ [mm]')
        ax.set_xlim([-500,500])

        if plot_for_publication:
            x1,x2=ax.get_xlim()
            y1,y2=ax.get_ylim()
            ax.set_aspect((x2-x1)/(y2-y1)/1.618)
            ax.set_title('Poloidal velocity')

        """
        ANGULAR AND EXPANSION VELOCITY PLOTTING FROM SKIMAGE
        """

        ax=axs[0,0]
        if plot_for_publication:

            if plot_skimage:
                ax.plot(frame_properties['Time'][plot_index],
                         frame_properties['Angular velocity ccf log'][plot_index]/1e3,
                         label='Angular velocity ccf log',
                         color='tab:blue'
                         )
                ax.set_title('Angular velocity skim')
            else:
                ax.plot(frame_properties['Time'][plot_index],
                         frame_properties['Angular velocity ccf FLAP'][plot_index]/1e3,
                         label='Angular velocity ccf FLAP',
                         color='tab:blue')
                ax.set_title('Angular velocity FLAP')
        else:
            ax.plot(frame_properties['Time'][plot_index],
                      frame_properties['Angular velocity ccf'][plot_index]/1e3,
                      label='skim',
                      color='orange')
            ax.plot(frame_properties['Time'][plot_index],
                     frame_properties['Angular velocity ccf log'][plot_index]/1e3,
                     label='skim log',
                     color='tab:blue'
                     )
            ax.plot(frame_properties['Time'][plot_index],
                      frame_properties['Angular velocity ccf FLAP'][plot_index]/1e3,
                      label='FLAP',
                      color='red')
            ax.plot(frame_properties['Time'][plot_index],
                     frame_properties['Angular velocity ccf FLAP log'][plot_index]/1e3,
                     label='FLAP log',
                     color='green')
            ax.set_title('$\omega$')
            ax.legend()

        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.set_xticks(ticks=[-500,-250,0,250,500])

        ax.set_xlabel('$t-t_{ELM}$ $[\mu s]$')
        ax.set_ylabel('$\omega$ [krad/s]')
        ax.set_xlim([-500,500])

        if plot_for_publication:
            x1,x2=ax.get_xlim()
            y1,y2=ax.get_ylim()
            ax.set_aspect((x2-x1)/(y2-y1)/1.618)
            if plot_skimage:
                ax.set_title('Ang vel CCF skimage')


        """
        ANGULAR AND EXPANSION VELOCITY PLOTTING FROM SKIMAGE
        """

        ax=axs[0,1]
        if plot_for_publication:
            if plot_skimage:
                ax.plot(frame_properties['Time'][plot_index],
                         frame_properties['Expansion velocity ccf'][plot_index],
                         label='Expansion velocity ccf ',
                         )
                ax.set_title('Scaling of '+str(exp_id)+' with skimage')
            else:
                ax.plot(frame_properties['Time'][plot_index],
                        frame_properties['Expansion velocity ccf FLAP'][plot_index],
                        label='Expansion velocity ccf FLAP',
                        color='tab:blue')
                ax.set_title('Scaling of '+str(exp_id)+' with FLAP')
        else:
            ax.plot(frame_properties['Time'][plot_index],
                     frame_properties['Expansion velocity ccf'][plot_index],
                     label='skim',
                     )
            ax.plot(frame_properties['Time'][plot_index],
                    frame_properties['Expansion velocity ccf FLAP'][plot_index],
                    label='FLAP')
            ax.set_title('Scaling')
            ax.legend()

        ax.set_xlabel('$t-t_{ELM}$ $[\mu s]$')
        ax.set_ylabel('Scaling')
        ax.set_xlim([-500,500])

        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.set_xticks(ticks=[-500,-250,0,250,500])
        if plot_for_publication:
            x1,x2=ax.get_xlim()
            y1,y2=ax.get_ylim()
            ax.set_aspect((x2-x1)/(y2-y1)/1.618)
            ax.set_title('Scaling factor')
        plt.tight_layout(pad=0.1)
        if pdf:
            pdf_pages.savefig()

    if pdf:
        pdf_pages.close()

    if test_into_pdf:
        matplotlib.use('qt5agg')
        pdf_test.close()

    if new_scheme_return:
        time=frame_properties['Time']
        del frame_properties['Shot']
        del frame_properties['Time']
        del frame_properties['GPI Dalpha']

        frame_properties_old=copy.deepcopy(frame_properties)
        del frame_properties

        data_dict={'raw':np.zeros([len(time)]),

                   'unit':None,
                   'label':None,
                   }

        frame_properties={'shot':exp_id,
                          'time':time,
                          'data':{},
                          'derived':{},
                          'structures':[],
                          }

        for key in list(frame_properties_old.keys()):

            key_avgmax='raw'
            key_new=key

            if len(frame_properties_old[key].shape) == 1:
                try:
                    frame_properties['data'][key_new][key_avgmax]
                except:
                    frame_properties['data'][key_new]=copy.deepcopy(data_dict)
                frame_properties['data'][key_new][key_avgmax]=frame_properties_old[key]
            else:

                try:
                    frame_properties['data'][key_new+' radial'][key_avgmax]
                except:
                    frame_properties['data'][key_new+' radial']=copy.deepcopy(data_dict)
                frame_properties['data'][key_new+' radial'][key_avgmax]=frame_properties_old[key][:,0]
                try:
                    frame_properties['data'][key_new+' poloidal'][key_avgmax]
                except:
                    frame_properties['data'][key_new+' poloidal']=copy.deepcopy(data_dict)
                frame_properties['data'][key_new+' poloidal'][key_avgmax]=frame_properties_old[key][:,1]

    if return_results:
        return frame_properties


def plot_angular_velocity_calc_test(test=False,
                                    test_into_pdf=False,
                                    pdf_test=None,
                                    sample_number=None,
                                    sample_to_plot=None,
                                    plot_sample_frames=True,
                                    i_log_or_not=0,
                                    axs=None,
                                    frame2_fft_polar_log=None,
                                    frame2_fft=None,
                                    save_data_for_publication=False,
                                    data_filename=None,
                                    time=None,
                                    i_frames=None):
    if test and not test_into_pdf:
        plt.cla()

        flap.plot('GPI_FRAME_12_CCF_POLAR',
                  plot_type='contour',
                  axes=['Angle lag', 'Radius lag'])

        plt.pause(0.1)
        plt.show()
        #flap.plot('GPI_FRAME_2_FFT_POLAR', plot_type='contour', axes=['Angle', 'Radius'])

    if test_into_pdf and sample_number in sample_to_plot:
        ind_sample_to_plot=int(np.where(np.asarray(sample_to_plot) == sample_number)[0])

        x_text=0.0
        y_text=1.06

        if plot_sample_frames:
            data=flap.get_data_object_ref('GPI_FRAME_2_FILTERED').data
            ax=axs[0,ind_sample_to_plot]
            ax.contourf(np.arange(data.shape[0]),
                        np.arange(data.shape[1]),
                        data.transpose(),
                        levels=51)
            ax.set_title('Frame #'+str(ind_sample_to_plot+1))
            ax.set_xlabel('x [pix]')
            ax.set_ylabel('y [pix]')
            ax.set_aspect('equal')

            if sample_number == sample_to_plot[0]:
                corner_str='(a)'
            else:
                corner_str='(b)'

            ax.text(x_text, y_text, corner_str, transform=ax.transAxes, size=9)

            # ax.xaxis.set_major_locator(MaxNLocator(5))
            # ax.yaxis.set_major_locator(MaxNLocator(5))

            if save_data_for_publication and data_filename is not None:
                file1=open(data_filename+'_'+str(time[i_frames]*1e3)+'_frame.txt', 'w+')
                data=flap.get_data_object_ref('GPI_FRAME_2').data
                for i in range(len(data[0,:])):
                    string=''
                    for j in range(len(data[:,0])):
                        string+=str(data[j,i])+'\t'
                    string+='\n'
                    file1.write(string)
                file1.close()
            elif save_data_for_publication and data_filename is None:
                print('No data_filename was given. Data is not saved into txt.')
        if plot_sample_frames:
            row_ind=1
        else:
            row_ind=0
        ax=axs[row_ind,ind_sample_to_plot]
        ax.contourf(np.arange(frame2_fft.shape[0])-frame2_fft.shape[0]//2,
                    np.arange(frame2_fft.shape[1])-frame2_fft.shape[1]//2,
                    frame2_fft.transpose(), levels=51)

        ax.set_xlabel('$x_{FFT}$ [pix]')
        ax.set_ylabel('$y_{FFT}$ [pix]')
        ax.set_xlim([-32,32])
        ax.set_ylim([-40,40])
        ax.set_title('FFT magnitude #'+str(ind_sample_to_plot+1))
        ax.set_aspect('equal')

        if sample_number == sample_to_plot[0]:
            corner_str='(c)'
        else:
            corner_str='(d)'

        ax.text(x_text, y_text, corner_str, transform=ax.transAxes, size=9)
        # ax.xaxis.set_major_locator(MaxNLocator(5))
        # ax.yaxis.set_major_locator(MaxNLocator(5))

        if save_data_for_publication and data_filename is not None:
            file1=open(data_filename+'_'+str(time[i_frames]*1e3)+'_fft_linear.txt', 'w+')
            data=frame2_fft
            xdata=np.arange(frame2_fft.shape[0])-frame2_fft.shape[0]//2
            ydata=np.arange(frame2_fft.shape[1])-frame2_fft.shape[1]//2

            file1.write('#x lag (pix)\n')
            for i in range(len(xdata)):
                file1.write(str(xdata[i])+'\t')

            file1.write('\n#y lag (pix)\n')
            for i in range(len(ydata)):
                file1.write(str(ydata[i])+'\t')

            file1.write('\n#FFT linear\n')
            for i in range(len(data[0,:])):
                string=''
                for j in range(len(data[:,0])):
                    string+=str(data[j,i])+'\t'
                string+='\n'
                file1.write(string)
            file1.close()

        elif save_data_for_publication and data_filename is None:
            print('No data_filename was given. Data is not saved into txt.')
        if plot_sample_frames:
            row_ind=2
        else:
            row_ind=1
        ax=axs[row_ind,ind_sample_to_plot]
        #plt.subplot(gs[2,ind_sample_to_plot])
        xdata=np.arange(frame2_fft_polar_log.shape[0])
        ydata=np.arange(frame2_fft_polar_log.shape[1])
        #ydata=(np.exp(np.arange(frame2_fft_polar_log.shape[1]) / radius / np.log(radius)))

        data=frame2_fft_polar_log
        ax.contourf(ydata,
                     xdata,
                     data, levels=51)

        ax.set_ylabel('$\\theta$ [deg]')
        ax.set_xlabel('r [pix]')

        ax.set_title('FFT magn. log-pol #'+str(ind_sample_to_plot+1))
        if sample_number == sample_to_plot[0]:
            corner_str='(e)'
        else:
            corner_str='(f)'
        ax.text(x_text, y_text, corner_str, transform=ax.transAxes, size=9)
        # ax.xaxis.set_major_locator(MaxNLocator(5))
        # ax.yaxis.set_major_locator(MaxNLocator(5))

        if save_data_for_publication and data_filename is not None:
            file1=open(data_filename+'_'+str(time[i_frames]*1e3)+'_fft_logpol.txt', 'w+')

            file1.write('#phi (deg)\n')
            for i in range(len(xdata)):
                file1.write(str(xdata[i])+'\t')

            file1.write('\n#r (pix)\n')
            for i in range(len(ydata)):
                file1.write(str(ydata[i])+'\t')

            file1.write('\n#FFT logpol\n')
            for i in range(len(data[0,:])):
                string=''
                for j in range(len(data[:,0])):
                    string+=str(data[j,i])+'\t'
                string+='\n'
                file1.write(string)
            file1.close()

        elif save_data_for_publication and data_filename is None:
            print('No data_filename was given. Data is not saved into txt.')

def plot_angular_velocity_ccf(sample_to_plot=None,
                              sample_number=0,
                              xdata=None,
                              ydata=None,
                              time=None,
                              i_frames=None,
                              ccf=None,

                              levels=51,

                              pdf=False,
                              title='',
                              pdf_object=None,
                              plot_for_publication=False,
                              save_data_for_publication=True,
                              data_filename=None):

    if sample_number in sample_to_plot:
        fig, ax=plt.subplots(figsize=(8.5/2.54, 8.5/2.54))
        im=ax.contourf(xdata,
                     ydata,
                     ccf, #No transpose is needed, the transposed ccf is plot from the first place
                     levels=levels)

        ax.set_title(title+' t='+str(time[i_frames]*1e3)+'ms')
        ax.set_xlabel('r lag [pix]')
        ax.set_ylabel('$\\theta$ lag [deg]')

        axins = ax.inset_axes([0.6, 0.6, 0.4, 0.4])
        axins.contourf(xdata,
                       ydata,
                       ccf, #No transpose is needed, the transposed ccf is plot from the first place
                       levels=levels)

        axins.set_xlim(-5,5)
        axins.set_ylim(-15,5)

        ax.indicate_inset_zoom(axins, edgecolor="black")
        fig.colorbar(im)
        plt.tight_layout()

        if pdf:
            pdf_object.savefig()
        plt.cla()

        if save_data_for_publication and data_filename is not None:
            data=ccf
            file1=open(data_filename+'_'+str(time[i_frames]*1e3)+'_2D_logpol_CCCF.txt', 'w+')

            file1.write('#phi lag (deg)\n')
            for i in range(len(xdata)):
                file1.write(str(xdata[i])+'\t')

            file1.write('\n#r lag (pix)\n')
            for i in range(len(ydata)):
                file1.write(str(ydata[i])+'\t')

            file1.write('\n#2D CCCF data\n')
            for i in range(len(data[0,:])):
                string=''
                for j in range(len(data[:,0])):
                    string+=str(data[j,i])+'\t'
                string+='\n'
                file1.write(string)
            file1.close()

        elif save_data_for_publication and data_filename is None:
            print('No data_filename was given. Data is not saved into txt.')