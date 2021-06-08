#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 28 23:34:52 2021

@author: mlampert
"""
  
#Core modules
import os
import copy

import flap
import flap_nstx
flap_nstx.register()
import flap_mdsplus
flap_mdsplus.register('NSTX_MDSPlus')

thisdir = os.path.dirname(os.path.realpath(__file__))
fn = os.path.join(thisdir,"flap_nstx.cfg")
flap.config.read(file_name=fn)

#Scientific modules
import matplotlib.style as pltstyle
import matplotlib.pyplot as plt

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
    
    
def calculate_sde_velocity(data_object,                     #String or flap.DataObject
                           time_range=None,                      #The time range for the calculation
                           
                           x_range=None,                       #X range for the calculation
                           y_range=None,                       #Y range for the calculation
                           spatial_calibration=None,            #Spatial calibration coefficients for the pixels. Format: {'X':[c1,c2,c3,c4],'Y':[c5,c6,c7,c8]},
                                                                #X=c1*x+c2*y+c3*xy+c4; Y=c5*x+c6*y+c7*xy+c8 where x,y are pixel coordinates. If None, coefficients are determined from the data.
                           #Normalizer inputs
                           normalize='roundtrip',                #Normalization options, 
                                                                    #None: no normalization
                                                                    #'roundtrip': zero phase LPF IIR filter
                                                                    #'halved': different normalzation for before and after the ELM
                                                                    #'simple': simple low-pass filtered normalization
                           normalize_type='division',
                           normalize_f_kernel='Elliptic',        #The kernel for filtering the gas cloud
                           normalize_f_high=1e3,                 #High pass frequency for the normalizer data
                           
                           subtraction_order=1,     #Order of the 2D polynomial for background subtraction
                           
                           #Inputs for velocity processing
                           correlation_threshold=0.6,            #Threshold for the maximum of the cross-correlation between the two frames (calculated with the actual maximum, not the fit value)
                           frame_similarity_threshold=0.0,       #Similarity threshold between the two subsequent frames (CCF at zero lag) DEPRECATED when an ELM is present
                           correct_acf_peak=True,
                           valid_frame_thres=5,                  #The number of consequtive frames to consider the calculation valid.
                           velocity_threshold=None,              #Velocity threshold for the calculation. Abova values are considered np.nan.
                           parabola_fit=True,                    #Fit a parabola on top of the cross-correlation function (CCF) peak
                           interpolation='parabola',             #Can be 'parabola', 'bicubic', and whatever is implemented in the future
                           bicubic_upsample=4,                   #Interpolation upsample for bicubic interpolation
                           fitting_range=5,                      #Fitting range of the peak of CCF 

                                                                 #Center of gravity: 'cog'; Geometrical center: 'centroid'; Ellipse center: 'center'
                           
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
                           test_histogram=False,                 #Plot the poloidal velocity histogram
                           save_data_for_publication=False,
                           ):
    
    """
    Calculate frame by frame average frame velocity. The
    code takes subsequent frames, calculates the 2D correlation function between
    the two and finds the maximum. Based on the pixel shift and the sampling
    time of the signal, the radial and poloidal velocity is calculated.
    The code assumes that the structures present in the subsequent frames are
    propagating with the same velocity. If there are multiple structures
    propagating in e.g. different direction or with different velocities, their
    effects are averaged over.
    """
    
    
    #Input error handling
    if data_object is None:
        raise ValueError('Either exp_id or data_object needs to be set for the calculation.')

    if type(time_range) is not list and filename is None:
        raise TypeError('time_range is not a list.')
    if filename is None and len(time_range) != 2:
        raise ValueError('time_range should be a list of two elements.')
              
    if correlation_threshold is not None:
        print('Correlation threshold is not taken into account if not calculated with FLAP.')
        
    if frame_similarity_threshold is not None:
        print('Correlation threshold is not taken into account if not calculated with FLAP.')

    if parabola_fit:
        comment='pfit_o'+str(subtraction_order)+\
                '_fst_'+str(frame_similarity_threshold)
    else:
        comment='max_o'+str(subtraction_order)+\
                '_fst_'+str(frame_similarity_threshold)
        
    pickle_filename=filename+'_'+comment+'.pickle'
    
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
    
    if not nocalc:
                
        if type(data_object) is str:
            input_obj_name=data_object[:]
            d=flap.get_data_object_ref(data_object)
        else:
            try:
                input_obj_name='DATA'
                d=copy.deepcopy(data_object)
                flap.add_data_object(d, input_obj_name)
            except:
                raise ValueError('Input object needs to be either string or flap.DataObject')
        if time_range is None:
            time_range=[d.coordinate('Time')[0][0,0,0],
                        d.coordinate('Time')[0][-1,0,0]]
        exp_id=d.exp_id
        
        if x_range is None or y_range is None:
            x_range=[d.coordinate('Image x')[0][0,0,0], 
                     d.coordinate('Image x')[0][0,-1,0]]
            y_range=[d.coordinate('Image y')[0][0,0,0], 
                     d.coordinate('Image y')[0][0,0,-1]]
        
        if not return_pixel_displacement and spatial_calibration is None:
            try:
                R=d.coordinate('Device R')[0][0,:,:]
                z=d.coordinate('Device z')[0][0,:,:]
            except:
                raise ValueError('The given object doesn\'t have "Device R" and "Device z" coordinates')
            
            spatial_vector=np.asarray([R[0,0],z[0,0],R[0,1],z[0,1],R[1,0],z[1,0],R[1,1],z[1,1]])
            pixel_matrix=[]
            for x in [0,1]:
                for y in [0,1]:
                    pixel_matrix.append([x,y,x*y,1,0,0,0,0])
                    pixel_matrix.append([0,0,0,0,x,y,x*y,1])
            pixel_matrix=np.asarray(pixel_matrix)
            
            spatial_calibration=np.linalg.inv(pixel_matrix) @ spatial_vector

        if (fitting_range*2+1 > np.abs(x_range[1]-x_range[0]) or 
            fitting_range*2+1 > np.abs(y_range[1]-y_range[0])):
            print(x_range, y_range, fitting_range)
            raise ValueError('The fitting range for the parabola is too large for the given coordinate range.')
                            
        slicing={'Time':flap.Intervals(time_range[0],time_range[1]),
                 'Image x':flap.Intervals(x_range[0],x_range[1]),
                 'Image y':flap.Intervals(y_range[0],y_range[1])}

        d=flap.slice_data(input_obj_name,exp_id=exp_id, 
                          slicing=slicing,
                          output_name='SLICED_FULL')
            
        object_name_ccf_velocity='SLICED_FULL'
        
        #Normalize data for size calculation

        normalizer_object_name='LPF_INTERVAL'

        slicing_for_filtering=copy.deepcopy(slicing)
        slicing_for_filtering['Time']=flap.Intervals(time_range[0]-1/normalize_f_high*10,
                                                     time_range[1]+1/normalize_f_high*10)
        slicing_time_only={'Time':flap.Intervals(time_range[0],
                                                 time_range[1])}

        flap.slice_data(input_obj_name,
                        exp_id=exp_id,
                        slicing=slicing_for_filtering,
                        output_name='SLICED_FOR_FILTERING')            
        
        if normalize == 'simple':
            flap.filter_data('SLICED_FOR_FILTERING',
                             exp_id=exp_id,
                             coordinate='Time',
                             options={'Type':'Lowpass',
                                      'f_high':normalize_f_high,
                                      'Design':normalize_f_kernel},
                             output_name=normalizer_object_name)
            coefficient=flap.slice_data(normalizer_object_name,
                                        exp_id=exp_id,
                                        slicing=slicing_time_only,
                                        output_name='background').data
            
        elif normalize == 'roundtrip':
            norm_obj=flap.filter_data('SLICED_FOR_FILTERING',
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
                                        slicing=slicing_time_only,
                                        output_name='background').data
            
        elif normalize == 'halved':
            data_obj=flap.get_data_object_ref(object_name_ccf_velocity).slice_data(summing={'Image x':'Mean', 'Image y':'Mean'})
            ind_peak=np.argmax(data_obj.data)
            data_obj_reverse=copy.deepcopy(flap.get_data_object('SLICED_FOR_FILTERING'))
            data_obj_reverse.data=np.flip(data_obj_reverse.data, axis=0)
            flap.add_data_object(data_obj_reverse,'SLICED_FOR_FILTERING_REV')
            
            normalizer_object_name_reverse='LPF_INTERVAL_REV'
            flap.filter_data('SLICED_FOR_FILTERING',
                             exp_id=exp_id,
                             coordinate='Time',
                             options={'Type':'Lowpass',
                                      'f_high':normalize_f_high,
                                      'Design':normalize_f_kernel},
                             output_name=normalizer_object_name)
            coefficient1_sliced=flap.slice_data(normalizer_object_name,
                                         exp_id=exp_id,
                                         slicing=slicing_time_only)
            
            coefficient2=flap.filter_data('SLICED_FOR_FILTERING_REV',
                                         exp_id=exp_id,
                                         coordinate='Time',
                                         options={'Type':'Lowpass',
                                                  'f_high':normalize_f_high,
                                                  'Design':normalize_f_kernel},
                                         output_name=normalizer_object_name_reverse)
            
            coefficient2.data=np.flip(coefficient2.data, axis=0)
            coefficient2_sliced=flap.slice_data(normalizer_object_name_reverse,
                                                exp_id=exp_id,
                                                slicing=slicing_time_only)
            
            coeff1_first_half=coefficient1_sliced.data[:ind_peak-4,:,:]
            coeff2_second_half=coefficient2_sliced.data[ind_peak-4:,:,:]
            coefficient=np.append(coeff1_first_half,coeff2_second_half, axis=0)
            coefficient_dataobject=copy.deepcopy(coefficient1_sliced)
            coefficient_dataobject.data=coefficient
            flap.add_data_object(coefficient_dataobject, 'background')
            
            
        if normalize is not None:
            data_obj=flap.get_data_object(object_name_ccf_velocity)
            
            if normalize_type == 'division':            
                data_obj.data = data_obj.data/coefficient
            elif normalize_type == 'subtraction':
                data_obj.data = data_obj.data-coefficient
            else:
                raise ValueError('normalize_type is either division or subtraction. Returning...')
                
            flap.add_data_object(data_obj, 'SLICED_DENORM_CCF_VEL')
            object_name_ccf_velocity='SLICED_DENORM_CCF_VEL'
            
            
        #Subtract trend from data
        if subtraction_order is not None:
            print("*** Subtracting the trend of the data ***")
            d=flap_nstx.analysis.detrend_multidim(object_name_ccf_velocity,
                                                  exp_id=exp_id,
                                                  order=subtraction_order, 
                                                  coordinates=['Image x', 'Image y'], 
                                                  output_name='DETREND_VEL')
            object_name_ccf_velocity='DETREND_VEL'
            
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
        dalpha=flap.get_data_object_ref('SLICED_FULL').slice_data(summing={'Image x':'Mean', 'Image y':'Mean'}).data,
        
        frame_properties={'Shot':exp_id,
                          'Time':time[1:-1],
                          'Mean':dalpha,
                          'Correlation max':np.zeros(len(time)-2),
                          'Frame similarity':np.zeros(len(time)-2),
                          'Velocity ccf':np.zeros([len(time)-2,2]),
                          'Displacement':np.zeros([len(time)-2,2]),
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
        
        flap.add_data_object(ccf_object, 'CCF_F_BY_F')
        #Velocity is calculated between the two subsequent frames, the velocity time is the second frame's time
        delta_index=np.zeros(2)       
       
        #Inicializing for frame handling
        frame2=None
        invalid_correlation_frame_counter=0.
        
        if test:
            plt.figure()
            
        for i_frames in range(0,n_frames-2):
            """
            STRUCTURE VELOCITY CALCULATION BASED ON CCF CALCULATION
            """
            
            print(str(i_frames/(n_frames-3)*100.)+"% done from the calculation.")
            
            slicing_frame1={'Sample':sample_0+i_frames}
            slicing_frame2={'Sample':sample_0+i_frames+1}
            if frame2 is None:
                frame1=flap.slice_data(object_name_ccf_velocity,
                                       exp_id=exp_id,
                                       slicing=slicing_frame1, 
                                       output_name='FRAME_1')
                #frame1.add_coordinate()
            else:
                frame1=copy.deepcopy(frame2)
                flap.add_data_object(frame1,'FRAME_1')
                
            frame2=flap.slice_data(object_name_ccf_velocity, 
                                   exp_id=exp_id,
                                   slicing=slicing_frame2,
                                   output_name='FRAME_2')

            frame1.data=np.asarray(frame1.data, dtype='float64')
            frame2.data=np.asarray(frame2.data, dtype='float64')                                
            

            flap.ccf('FRAME_2', 'FRAME_1', 
                     coordinate=['Image x', 'Image y'], 
                     options={'Resolution':1, 
                              'Range':[[-(x_range[1]-x_range[0]),(x_range[1]-x_range[0])],
                                       [-(y_range[1]-y_range[0]),(y_range[1]-y_range[0])]], 
                              'Trend removal':None, 
                              'Normalize':True, 
                              'Correct ACF peak':correct_acf_peak,
                              'Interval_n': 1}, 
                     output_name='FRAME_12_CCF')

            ccf_object.data[i_frames,:,:]=flap.get_data_object_ref('FRAME_12_CCF').data
            
            if test:
                print('Maximum correlation:'+str(ccf_object.data[i_frames,:,:].max()))
            
            max_index=np.asarray(np.unravel_index(ccf_object.data[i_frames,:,:].argmax(), ccf_object.data[i_frames,:,:].shape))
            
            #Fit a 2D polinomial on top of the peak
            if parabola_fit:
                area_max_index=tuple([slice(max_index[0]-fitting_range,
                                            max_index[0]+fitting_range+1),
                                      slice(max_index[1]-fitting_range,
                                            max_index[1]+fitting_range+1)])
                #Finding the peak analytically
                try:
                    coeff=flap_nstx.analysis.polyfit_2D(values=ccf_object.data[i_frames,:,:][area_max_index],order=2)
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
                
            if ccf_object.data[i_frames,:,:].max() < correlation_threshold:
                invalid_correlation_frame_counter+=1
            else:
                invalid_correlation_frame_counter=0.
                
            if interpolation == 'bicubic':
                delta_index=[delta_index[0]/bicubic_upsample,
                             delta_index[1]/bicubic_upsample]
                

            frame_properties['Displacement'][i_frames,0]=(spatial_calibration[0]*delta_index[0]+
                                                          spatial_calibration[1]*delta_index[1]+
                                                          spatial_calibration[2]*delta_index[0]*delta_index[1])
            frame_properties['Displacement'][i_frames,1]=(spatial_calibration[4]*delta_index[0]+
                                                          spatial_calibration[5]*delta_index[1]+
                                                          spatial_calibration[6]*delta_index[0]*delta_index[1])
            
            #Calculating the radial and poloidal velocity from the correlation map.
            frame_properties['Velocity ccf'][i_frames,0]=frame_properties['Displacement'][i_frames,0]/sample_time
            
            frame_properties['Velocity ccf'][i_frames,1]=frame_properties['Displacement'][i_frames,1]/sample_time   
        #Saving results into a pickle file
        
        pickle.dump(frame_properties,open(pickle_filename, 'wb'))
        if test:
            plt.close()
    else:
        print('--- Loading data from the pickle file ---')
        frame_properties=pickle.load(open(pickle_filename, 'rb'))
        

    # sample_time=frame_properties['Time'][1]-frame_properties['Time'][0]
    # if time_range[0] < frame_properties['Time'][0]-sample_time or time_range[1] > frame_properties['Time'][-1]+sample_time:
    #     raise ValueError('Please run the calculation again with the timerange. The pickle file doesn\'t have the desired range')
        
    if time_range is None:
        time_range=[frame_properties['Time'][0],frame_properties['Time'][-1]]
            
    nan_ind=np.where(frame_properties['Correlation max'] < correlation_threshold)
    frame_properties['Velocity ccf'][nan_ind,0] = np.nan
    frame_properties['Velocity ccf'][nan_ind,1] = np.nan
    return frame_properties

def calculate_sde_velocity_distribution(data_object,                            #String or flap.DataObject
                                        time_range=None,                        #The time range for the calculation
                           
                                        x_range=None,                           #X range for the calculation
                                        y_range=None,                           #Y range for the calculation
                                        
                                        x_res=None,
                                        y_res=None,
                                        
                                        x_step=None,
                                        y_step=None,
                                        
                                        spatial_calibration=None,               #Spatial calibration coefficients for the pixels. Format: {'X':[c1,c2,c3,c4],'Y':[c5,c6,c7,c8]},):
                                                                                #Normalizer inputs
                                        normalize='roundtrip',                  #Normalization options, 
                                                                                #None: no normalization
                                                                                #'roundtrip': zero phase LPF IIR filter
                                                                                #'halved': different normalzation for before and after the ELM
                                                                                #'simple': simple low-pass filtered normalization
                                        normalize_type='division',
                                        normalize_f_kernel='Elliptic',          #The kernel for filtering the gas cloud
                                        normalize_f_high=1e3,                   #High pass frequency for the normalizer data
                                        
                                        subtraction_order=1,                    #Order of the 2D polynomial for background subtraction
                                        
                                        #Inputs for velocity processing
                                        correlation_threshold=0.6,              #Threshold for the maximum of the cross-correlation between the two frames (calculated with the actual maximum, not the fit value)
                                        frame_similarity_threshold=0.0,         #Similarity threshold between the two subsequent frames (CCF at zero lag) DEPRECATED when an ELM is present
                                        correct_acf_peak=True,
                                        valid_frame_thres=5,                    #The number of consequtive frames to consider the calculation valid.
                                        velocity_threshold=None,                #Velocity threshold for the calculation. Abova values are considered np.nan.
                                        parabola_fit=True,                      #Fit a parabola on top of the cross-correlation function (CCF) peak
                                        interpolation='parabola',               #Can be 'parabola', 'bicubic', and whatever is implemented in the future
                                        bicubic_upsample=4,                     #Interpolation upsample for bicubic interpolation
                                        fitting_range=5,                        #Fitting range of the peak of CCF 
             
                                                                                #Center of gravity: 'cog'; Geometrical center: 'centroid'; Ellipse center: 'center'
                                        
                                        #File input/output options
                                        filename=None,                          #Filename for restoring data
                                        save_results=True,                      #Save the results into a .pickle file to filename+.pickle
                                        nocalc=True,                            #Restore the results from the .pickle file from filename+.pickle
                                        
                                        #Output options:
                                        return_results=False,                   #Return the results if set.
                                        return_displacement=False,
                                        return_data_object=False,
                                        cache_data=True,                        #Cache the data or try to open is from cache
                                        
                                        #Test options
                                        test=False,                             #Test the results
                                        test_histogram=False,                   #Plot the poloidal velocity histogram
                                        save_data_for_publication=False,
                                        ):
    
    #Input error handling
    if data_object is None:
        raise ValueError('Either exp_id or data_object needs to be set for the calculation.')

    if type(time_range) is not list and filename is None:
        raise TypeError('time_range is not a list.')
    if filename is None and len(time_range) != 2:
        raise ValueError('time_range should be a list of two elements.')

    if parabola_fit:
        comment='pfit_o'+str(subtraction_order)+\
                '_fst_'+str(frame_similarity_threshold)
    else:
        comment='max_o'+str(subtraction_order)+\
                '_fst_'+str(frame_similarity_threshold)
        
    pickle_filename=filename+'_'+comment+'.pickle'
    
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
    
    if not nocalc:
                
        if type(data_object) is str:
            input_obj_name=data_object[:]
            d=flap.get_data_object_ref(data_object)
        else:
            try:
                input_obj_name='DATA'
                d=data_object
                flap.add_data_object(d, input_obj_name)
            except:
                raise ValueError('Input object needs to be either string or flap.DataObject')
        if time_range is None:
            time_range=[d.coordinate('Time')[0][0,0,0],
                        d.coordinate('Time')[0][-1,0,0]]
        exp_id=d.exp_id
        
        if x_range is None or y_range is None:
            x_range=[0, d.data.shape[1]-1]
            y_range=[0, d.data.shape[2]-1]
        
        if spatial_calibration is None:
            try:
                R=d.coordinate('Device R')[0][0,:,:]
                z=d.coordinate('Device z')[0][0,:,:]
            except:
                raise ValueError('The given object doesn\'t have "Device R" and "Device z" coordinates')
            
            spatial_vector=np.asarray([R[0,0],z[0,0],R[0,1],z[0,1],R[1,0],z[1,0],R[1,1],z[1,1]])
            pixel_matrix=[]
            for x in [0,1]:
                for y in [0,1]:
                    pixel_matrix.append([x,y,x*y,1,0,0,0,0])
                    pixel_matrix.append([0,0,0,0,x,y,x*y,1])
            pixel_matrix=np.asarray(pixel_matrix)
            
            spatial_calibration=np.linalg.inv(pixel_matrix) @ spatial_vector
                    
        if (fitting_range*2+1 > np.abs(x_range[1]-x_range[0]) or 
            fitting_range*2+1 > np.abs(y_range[1]-y_range[0])):
            raise ValueError('The fitting range for the parabola is too large for the given coordinate range.')
            
        if x_res is None or x_res is None:    
            n_slicing_x=1
        else:
            n_slicing_x=int((x_range[1]-x_range[0]-x_res)/x_step+1)    
            
        if y_res is None or y_step is None:
            n_slicing_y=1
        else:
            n_slicing_y=int((y_range[1]-y_range[0]-y_res)/y_step+1)    
        
        
        d_time_sliced=flap.slice_data('DATA',
                                      slicing={'Time':flap.Intervals(time_range[0],time_range[1])},
                                      output_name='DATA_SLICED')
        
        time=d_time_sliced.coordinate('Time')[0][:,0,0]
        
        result_dict={'Shot':exp_id,
                     'Time':time[1:-1],
                     'Correlation max':np.zeros([len(time)-2, n_slicing_x, n_slicing_y]),
                     'Frame similarity':np.zeros([len(time)-2, n_slicing_x, n_slicing_y]),
                     'Velocity ccf':np.zeros([len(time)-2, n_slicing_x, n_slicing_y, 2]),
                     'Displacement':np.zeros([len(time)-2, n_slicing_x, n_slicing_y, 2]),
                     'Image x':np.zeros(n_slicing_x),
                     'Image y':np.zeros(n_slicing_y),
                     'Device R':np.zeros(n_slicing_x),
                     'Device z':np.zeros(n_slicing_y),
                     }

        for i_slicing_x in range(n_slicing_x):
            for j_slicing_y in range(n_slicing_y):
                slicing={}
                if x_res is None:
                    slicing['Image x']=flap.Intervals(x_range[0],
                                                      x_range[1])
                else:
                    slicing['Image x']=flap.Intervals(x_range[0]+i_slicing_x*x_step,
                                                      x_range[0]+i_slicing_x*x_step+x_res)
                if y_res is None:
                    slicing['Image y']=flap.Intervals(y_range[0],
                                                      y_range[1])
                else:
                    slicing['Image y']=flap.Intervals(y_range[0]+j_slicing_y*y_step,
                                                      y_range[0]+j_slicing_y*y_step+y_res)

                data_full_sliced=flap.slice_data('DATA_SLICED',slicing=slicing, output_name='DATA_SLICED_FULL')
                
                result_window=calculate_sde_velocity('DATA_SLICED_FULL',                     #String or flap.DataObject
                                                   #Normalizer inputs
                                                   normalize=normalize,                #Normalization options, 
                                                                                            #None: no normalization
                                                                                            #'roundtrip': zero phase LPF IIR filter
                                                                                            #'halved': different normalzation for before and after the ELM
                                                                                            #'simple': simple low-pass filtered normalization
                                                   normalize_type=normalize_type,
                                                   normalize_f_kernel=normalize_f_kernel,        #The kernel for filtering the gas cloud
                                                   normalize_f_high=normalize_f_high,                 #High pass frequency for the normalizer data
                                                   
                                                   subtraction_order=subtraction_order,     #Order of the 2D polynomial for background subtraction
                                                   
                                                   #Inputs for velocity processing
                                                   correlation_threshold=correlation_threshold,            #Threshold for the maximum of the cross-correlation between the two frames (calculated with the actual maximum, not the fit value)
                                                   frame_similarity_threshold=frame_similarity_threshold,       #Similarity threshold between the two subsequent frames (CCF at zero lag) DEPRECATED when an ELM is present
                                                   correct_acf_peak=correct_acf_peak,
                                                   valid_frame_thres=valid_frame_thres,                  #The number of consequtive frames to consider the calculation valid.
                                                   velocity_threshold=velocity_threshold,              #Velocity threshold for the calculation. Abova values are considered np.nan.
                                                   parabola_fit=parabola_fit,                    #Fit a parabola on top of the cross-correlation function (CCF) peak
                                                   interpolation=interpolation,             #Can be 'parabola', 'bicubic', and whatever is implemented in the future
                                                   
                                                   #File input/output options
                                                   filename=filename,                        #Filename for restoring data
                                                   save_results=save_results,                    #Save the results into a .pickle file to filename+.pickle
                                                   nocalc=nocalc,                          #Restore the results from the .pickle file from filename+.pickle
                                                   
                                                   #Output options:
                                                   return_results=True,                  #Return the results if set.
                                                   cache_data=False,                      #Cache the data or try to open is from cache
                                                   
                                                   #Test options
                                                   test=test,                           #Test the results
                                                   test_histogram=test_histogram,                 #Plot the poloidal velocity histogram
                                                   save_data_for_publication=save_data_for_publication,)
                
                result_dict['Correlation max'][:,i_slicing_x,j_slicing_y] = result_window['Correlation max']
                result_dict['Frame similarity'][:,i_slicing_x,j_slicing_y] = result_window['Frame similarity']
                result_dict['Velocity ccf'][:,i_slicing_x,j_slicing_y,:] = result_window['Velocity ccf']
                result_dict['Displacement'][:,i_slicing_x,j_slicing_y,:] = result_window['Displacement']
                if x_res is None or x_step is None:
                    result_dict['Image x'][i_slicing_x]=(x_range[0]+x_range[1])/2
                else:
                    result_dict['Image x'][i_slicing_x]=x_range[0]+i_slicing_x*x_step+x_res/2.
                if y_res is None or y_step is None:
                    result_dict['Image y'][j_slicing_y]=(y_range[0]+y_range[1])/2
                else:
                    result_dict['Image y'][j_slicing_y]=y_range[0]+j_slicing_y*y_step+y_res/2.
                
                result_dict['Device R'][i_slicing_x]=np.mean(data_full_sliced.coordinate('Device R')[0])
                result_dict['Device z'][j_slicing_y]=np.mean(data_full_sliced.coordinate('Device z')[0])
                
    if return_data_object:
        if return_displacement:
            return_key='Displacement'
            unit='m'
        else:
            return_key='Velocity ccf'
            unit='m/s'
            
        coord=[None] * 6
        coord[0]=(copy.deepcopy(flap.Coordinate(name='Time',
                                                   unit='s',
                                                   mode=flap.CoordinateMode(equidistant=True),
                                                   start=result_dict['Time'][0],
                                                   step=result_dict['Time'][1]-result_dict['Time'][0],
                                                   #shape=time_arr.shape,
                                                   dimension_list=[0]
                                                   )))
        
        coord[1]=(copy.deepcopy(flap.Coordinate(name='Sample',
                                                   unit='n.a.',
                                                   mode=flap.CoordinateMode(equidistant=True),
                                                   start=0,
                                                   step=1,
                                                   dimension_list=[0]
                                                   )))
        if x_step is None:
            x_step=1.
            
        coord[2]=(copy.deepcopy(flap.Coordinate(name='Image x',
                                                   unit='Pixel',
                                                   mode=flap.CoordinateMode(equidistant=True),
                                                   start=result_dict['Image x'][0],
                                                   step=x_step,
                                                   shape=[],
                                                   dimension_list=[1]
                                                   )))
        if y_step is None:
            y_step=1.
            
        coord[3]=(copy.deepcopy(flap.Coordinate(name='Image y',
                                                   unit='Pixel',
                                                   mode=flap.CoordinateMode(equidistant=True),
                                                   start=result_dict['Image y'][0],
                                                   step=y_step,
                                                   shape=[],
                                                   dimension_list=[2]
                                                   )))
        if len(result_dict['Device R']) == 1:
            coord[4]=(copy.deepcopy(flap.Coordinate(name='Device R',
                                   unit='m',
                                   mode=flap.CoordinateMode(equidistant=True),
                                   start=result_dict['Device R'],
                                   step=1,
                                   shape=[],
                                   dimension_list=[1]
                                   )))
        else:
            coord[4]=(copy.deepcopy(flap.Coordinate(name='Device R',
                                    unit='m',
                                    mode=flap.CoordinateMode(equidistant=False),
                                    values=result_dict['Device R'],
                                    shape=result_dict['Device R'].shape,
                                    dimension_list=[1]
                                    )))
        
        if len(result_dict['Device z']) == 1:
            coord[5]=(copy.deepcopy(flap.Coordinate(name='Device z',
                                   unit='m',
                                   mode=flap.CoordinateMode(equidistant=True),
                                   start=result_dict['Device z'],
                                   step=1,
                                   shape=[],
                                   dimension_list=[2]
                                   )))
        else:    
            coord[5]=(copy.deepcopy(flap.Coordinate(name='Device z',
                                   unit='m',
                                   mode=flap.CoordinateMode(equidistant=False),
                                   values=result_dict['Device z'],
                                   shape=result_dict['Device z'].shape,
                                   dimension_list=[2]
                                   )))
            
        data_objects=[None] * 2
        if return_key == 'Displacement':
            data_title='SDE displacement'
        elif return_key == 'Velocity ccf':
            data_title='SDE velocity'
            
        for ind_data_obj in range(2):
           
            data_objects[ind_data_obj]=flap.DataObject(data_array=result_dict[return_key][:,:,:,ind_data_obj],
                                                       data_unit=flap.Unit(name=data_title,unit=unit),
                                                       coordinates=coord,
                                                       exp_id=exp_id,
                                                       data_title=data_title,
                                                       info={'Options':None},
                                                       data_source="NSTX_GPI")
            
        return data_objects
    else:
        return result_dict

def plot_sde_velocity_distribution(data_object,
                                   exp_id=None,
                                   result_objects=None):
    
    if type(data_object) is type(flap.DataObject):
        flap.add_data_object(data_object,'DATA_OBJECT_SDE')
        d=data_object
    else:
        d=flap.get_data_object_ref(data_object, exp_id=exp_id)
    
    oplot_options={'arrow':[{'Data object X':result_objects[0],
                             'Data object Y':result_objects[1],
                             'Plot':True,
                             'width':0.1
                             }]}
        
    d.plot(plot_type='animation', 
           axes=['Device R', 'Device z', 'Time'],
           options={'Wait':0.0, 
                    'Clear':False,
                    'Overplot options':oplot_options,
                    'Equal axes': True,
                    'Plot units':{'Time':'s',
                                  'Device R':'m',
                                  'Device z':'m'}
                    }
           )

    