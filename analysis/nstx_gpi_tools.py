#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 13:37:37 2019

@author: mlampert
"""
#Core imports
import os
import copy
#Importing and setting up the FLAP environment
import flap
import flap_nstx
flap_nstx.register()

thisdir = os.path.dirname(os.path.realpath(__file__))
fn = os.path.join(thisdir,"flap_nstx.cfg")
flap.config.read(file_name=fn)

import matplotlib.pyplot as plt
import numpy as np

#from scipy.signal import find_peaks    #This is not working as reliably as the CWT
from scipy.signal import find_peaks_cwt
#Scientific library imports    

def calculate_nstx_gpi_norm_coeff(exp_id=None,              # Experiment ID
                                  f_high=1e2,               # Low pass filter frequency in Hz
                                  design='Chebyshev II',    # IIR filter design (from scipy)
                                  test=False,               # Testing input
                                  filter_data=True,         # IIR LPF the data
                                  time_range=None,          # Timer range for the averaging in ms [t1,t2]
                                  calc_around_max=False,    # Calculate the average around the maximum of the GPI signal
                                  time_window=50.,          # The time window for the calc_around_max calculation
                                  cache_data=True,          #
                                  verbose=False,
                                  output_name='GPI_NORMALIZER',
                                  #add_flux_r=False,
                                  ):
    
    #This function calculates the GPI normalizer image with which all the GPI
    #images should be divided. Returns a flap data object. The inputs are 
    #expained next to the inputs.

    normalizer_options={'LP freq':f_high,
                        'Filter design': design,
                        'Filter data':filter_data,
                        'Time range':time_range,
                        'Calc around max': calc_around_max,
                        'Time window': time_window,
                        #'Flux R': add_flux_r,
                        }

    if not cache_data:
        flap.delete_data_object(output_name,
                                'GPI_*_FILTERED_*',
                                'GPI_MEAN')

    if calc_around_max and time_range is not None:
        print('Both calc_around_max and time_range cannot be set.')
        print('Setting calc_around_max to False')
        calc_around_max=False

    #Get the data from the cine file
    if exp_id is not None:
        try:
            flap.get_data_object_ref(exp_id=exp_id,object_name='GPI')
        except:
            if verbose or test:
                print('Data is not cached, it needs to be read.')
                print("\n------- Reading NSTX GPI data --------")
            flap.get_data('NSTX_GPI',exp_id=exp_id,name='',object_name='GPI')
        object_name='GPI'
        #if add_flux_r:
        #    flap.add_coordinate(object_name, exp_id=exp_id, coordinates='Flux r')
    else:
        raise ValueError('The experiment ID needs to be set.')
        
    if time_range is not None:
        sliced_object_name=object_name+'_'+str(time_range[0])+'_'+str(time_range[1])
        try:
            flap.get_data_object_ref(exp_id=exp_id,object_name=sliced_object_name)
        except:
            flap.slice_data(object_name,exp_id=exp_id,
                            slicing={'Time':flap.Intervals(time_range[0],time_range[1])}, 
                            output_name=object_name+'_'+str(time_range[0])+'_'+str(time_range[1])
                            )
        object_name=sliced_object_name
        
        
    #Highpass filter the data to get rid of the spikes
    if filter_data:
        filtered_data_object_name=object_name+'_FILTERED_LP_'+str(f_high)+'_'+design.replace(' ','')
        try:
            flap.get_data_object_ref(exp_id=exp_id,object_name=filtered_data_object_name)
        except:
            if verbose or test:
                print('Filtered data is not cached, it needs to be filtered.')
                print("\n------- Filtering NSTX GPI data --------")
            flap.filter_data(object_name,exp_id=exp_id,
                             coordinate='Time',
                             options={'Type':'Lowpass',
                                      'f_high':f_high, 
                                      'Design':design},
                                      output_name=filtered_data_object_name)
        object_name=filtered_data_object_name
        
    if calc_around_max and time_range is None:
        #Calculate the average image for a time window around the maximum signal
        d=flap.slice_data(object_name,
                          summing={'Image x':'Mean','Image y':'Mean'},
                          output_name='GPI_MEAN')

        max_time_index=np.argmax(d.data)
        max_time=d.coordinate('Time')[0][max_time_index]
        flap.slice_data(object_name,exp_id=exp_id,
                        slicing={'Time':flap.Intervals(max_time-time_window,max_time+time_window)},
                        summing={'Time':'Mean'},
                        output_name=output_name)
    else:
        #Calculate the average image for the entire shot
        d=flap.slice_data(object_name,exp_id=exp_id,
                        summing={'Time':'Mean'},
                        output_name=output_name)
    object_name=output_name
    
    d.info['Normalizer options']=normalizer_options
    
    if test:
        plt.figure()
#        if False:
#        if add_flux_r:
#            flap.plot(object_name, 
#                      axes=['Flux r', 'Device z'], 
#                      exp_id=exp_id, 
#                      plot_type='contour',
#                      plot_options={'levels':21}
#                      )
#        else:
        flap.plot(object_name, 
                  axes=['Device R', 'Device z'], 
                  exp_id=exp_id, 
                  plot_type='contour',
                  plot_options={'levels':21}
                  )
    return d

def calculate_nstx_gpi_reference(object_name=None,
                                 exp_id=None,
                                 time_range=None,
                                 reference_pixel=None,
                                 reference_area=None,
                                 reference_position=None,
                                 reference_flux=None,
                                 filter_low=None,
                                 filter_high=None,
                                 filter_design='Chebyshev II',
                                 output_name=None
                                 ):
    try:
        input_object=flap.get_data_object_ref(object_name, exp_id=exp_id)
    except:
        raise IOError('The given object_name doesn\'t exist in the FLAP storage.')
    if output_name is None:
        output_name=object_name+'_REF'
        
    if reference_pixel is None and reference_position is None and reference_flux is None:
        raise ValueError('There is no reference given. Please set reference_pixel or reference_position or reference_flux.')
        
    if filter_low is not None or filter_high is not None:
        if filter_low is not None and filter_high is None:
            filter_type='Highpass'
        if filter_low is None and filter_high is not None:
            filter_type='Lowpass'
        if filter_low is not None and filter_high is None:
            filter_type='Bandpass'
        
        flap.filter_data(object_name,exp_id=exp_id,
                         coordinate='Time',
                         options={'Type':filter_type,
                                  'f_low':filter_low,
                                  'f_high':filter_high, 
                                  'Design':filter_design},
                         output_name=object_name+'_FILTERED')
        object_name=object_name+'_FILTERED'
    slicing_dict={}
    
    if time_range is not None:
        slicing_dict['Time']=flap.Intervals(time_range[0],time_range[1])
        
    if reference_pixel is not None:
        #Single pixel correlation
        if reference_area is None:
            slicing_dict['Image x']=reference_pixel[0]
            slicing_dict['Image y']=reference_pixel[1]
            summing_dict=None
        else:
            if type(reference_area) is not list:
                reference_area=[reference_area,reference_area]
            #Handling the edges:
            if reference_pixel[0]-reference_area[0] < 0:
                reference_pixel[0]=reference_area[0]
            if reference_pixel[1]-reference_area[0] < 0:
                reference_pixel[1]=reference_area[0]
                
            if reference_pixel[0]+reference_area[1] > input_object.data.shape[1]:
                reference_pixel[0]=input_object.data.shape[1]-reference_area[1]
            if reference_pixel[1]+reference_area[1] > input_object.data.shape[2]:
                reference_pixel[1]=input_object.data.shape[2]-reference_area[1]
                
            slicing_dict['Image x']=flap.Intervals(reference_pixel[0]-reference_area[0],
                                                   reference_pixel[0]+reference_area[0])
            slicing_dict['Image y']=flap.Intervals(reference_pixel[1]-reference_area[1],
                                                   reference_pixel[1]+reference_area[1])
            summing_dict={'Image x':'Mean', 'Image y':'Mean'}

    if reference_position is not None:
        if reference_area is None:
            try:
                slicing_dict['Device R']=reference_position[0]
                slicing_dict['Device z']=reference_position[1]
                summing_dict=None
            except:
                raise ValueError('Reference position is outside the measurement range.')
        else:
            if type(reference_area) is not list:
                reference_area=[reference_area,reference_area]
            try:
            #Multiple pixel correlation (averaged)
                slicing_dict['Device R']=flap.Intervals(reference_position[0]-reference_area[0],
                                                        reference_position[0]+reference_area[0])
                slicing_dict['Device z']=flap.Intervals(reference_position[1]-reference_area[1],
                                                        reference_position[1]+reference_area[1])
                summing_dict={'Device R':'Mean', 'Device z':'Mean'}
            except:
                raise ValueError('Reference position is outside the measurement range.')
                
    if reference_flux is not None:
        if len(reference_flux) != 2:
            raise ValueError('The reference position needs to be a 2 element list (Psi,z).')
        if reference_area is None:
            try:
                slicing_dict['Flux r']=flap.Intervals(reference_flux[0])
                slicing_dict['Device z']=flap.Intervals(reference_flux[1])
                summing_dict=None
            except:
                raise ValueError('Reference position is outside the measurement range.')
        else:
            if len(reference_area) !=2:
                 raise ValueError('The reference area needs to be a 2 element list (Psi,z).')
            try:
            #Multiple pixel correlation (averaged)
                slicing_dict['Flux r']=flap.Intervals(reference_flux[0]-reference_area[0],
                                                      reference_flux[0]+reference_area[0])
                slicing_dict['Device z']=flap.Intervals(reference_flux[1]-reference_area[1],
                                                        reference_flux[1]+reference_area[1])
                summing_dict={'Flux r':'Mean', 'Device z':'Mean'}  
            except:
                raise ValueError('Reference position is outside the measurement range.')    
                
    reference_signal=flap.slice_data(object_name, exp_id=exp_id,
                                     slicing=slicing_dict,
                                     summing=summing_dict,
                                     output_name=output_name)
    return reference_signal
    
def find_filaments(data_object=None,      #FLAP data objectCould be set instead of exp_id and time_range 
                   exp_id=None,           #Shot number
                   time_range=None,       #Time range for the filament finding
                   frange=[0.1e3,100e3],  #Frequency range to pre-condition the data
                   normalize=False,       #Normalize (divide) the data with the time average in time_range
                   ref_pixel=[10,40],     #The pixel to find the peak in.                   
                   horizontal_sum=False,  #Sum up the pixels vertically in xrange
                   xrange=[0,32],         #Range for summing up the pixels
                   vertical_sum=False,    #Sum up all the pixels vertically in yrange
                   yrange=[10,70],        #Range for summing up the pixels
                   width_range=[1,30],    #The width range for the CWT peak finding algorithm
                   cache_data=False,      #Try to gather the cached data (exp_id, timerange input)
                   return_index=False,     #Return the peak times instaed of the peak indices
                   test=False):           #Plot the resulting data along with the peaks
    
    #Read signal
    if data_object is None:
        data_object='GPI_FILAMENTS'
        if time_range is None:
            print('The time range needs to set for the calculation.')
            return
        else:    
            if (type(time_range) is not list and len(time_range) != 2):
                raise TypeError('time_range needs to be a list with two elements.')
        if exp_id is not None:
            print("\n------- Reading NSTX GPI data --------")
            if cache_data:
                try:
                    d=flap.get_data_object_ref(exp_id=exp_id,object_name=data_object)
                except:
                    print('Data is not cached, it needs to be read.')
                    d=flap.get_data('NSTX_GPI',exp_id=exp_id,name='',object_name=data_object)
            else:
                d=flap.get_data('NSTX_GPI',exp_id=exp_id,name='',object_name=data_object)
        else:
            raise ValueError('The experiment ID needs to be set.')
        slicing={'Time':flap.Intervals(time_range[0],time_range[1]),
                 'Image x':ref_pixel[0],
                 'Image y':ref_pixel[1]}
        summing=None
        if normalize:
            normalizer=flap.slice_data(data_object,
                                       slicing={'Time':flap.Intervals(time_range[0],
                                                                      time_range[1])},
                                       summing={'Time':'Mean'}).data
            
            for i_x in range(d.data.shape[1]):
                for i_y in range(d.data.shape[2]):
                    d.data[:,i_x,i_y]=d.data[:,i_x,i_y]/normalizer[i_x,i_y]
    else:
        if normalize:
            normalizer=flap.slice_data(data_object,summing={'Time':'Mean'}).data
            d=flap.get_data_object_ref(data_object).data
            for i_x in range(d.data.shape[1]):
                for i_y in range(d.data.shape[2]):
                    d.data[:,i_x,i_y]=d.data[:,i_x,i_y]/normalizer[i_x,i_y]
        slicing={'Image x':ref_pixel[0],
                 'Image y':ref_pixel[1]}
        summing=None
    if vertical_sum or horizontal_sum:
        summing={}
        if horizontal_sum:
            slicing['Image x']=flap.Intervals(xrange[0],xrange[1])
            summing['Image x']='Mean'
        if vertical_sum:
            slicing['Image y']=flap.Intervals(yrange[0],yrange[1])
            summing['Image y']='Mean'
    flap.slice_data(data_object, 
                    slicing=slicing,
                    summing=summing,
                    output_name='GPI_SLICED_FILAMENTS')
    
    #Filter signal to HPF 100Hz    
    d=flap.filter_data('GPI_SLICED_FILAMENTS',
                       coordinate='Time',
                       options={'Type':'Bandpass',
                                'f_low':frange[0],
                                'f_high':frange[1],
                                'Design':'Chebyshev II'},
                       output_name='GPI_FILTERED_FILAMENTS')

    #ind=find_peaks(d.data, distance=25, threshold=threshold)[0]         #This method needs quite a lot of tinkering, it is deprecated
    ind=find_peaks_cwt(d.data, np.arange(width_range[0],width_range[1])) #This method is working quite well without any data preconditioning except the filtering
    try:
        flap.delete_data_object('GPI_FILAMENTS')
    except:
        pass
    flap.delete_data_object('GPI_SLICED_FILAMENTS')
    flap.delete_data_object('GPI_FILTERED_FILAMENTS')
    if test:
        plt.figure()
        plt.plot(d.coordinate('Time')[0],d.data)
        plt.scatter(d.coordinate('Time')[0][ind],d.data[ind], color='red')
        plt.xlabel('Time [s]')
        plt.ylabel('GPI signal [a.u.]')
        plt.show()
    if return_index:
        return ind
    else:
        return d.coordinate('Time')[0][ind]

def detrend_multidim(data_object=None,
                     exp_id=None,
                     coordinates=None,
                     order=None,
                     test=False,
                     return_trend=False,
                     output_name=None):
    if exp_id is not None:
        d=flap.get_data_object(data_object, exp_id=exp_id)
    else:
        d=flap.get_data_object(data_object)
        
    total_dim=len(d.data.shape)
    if total_dim > 4:
        raise TypeError('Dataset over 4 dimensions is not supported.')
    ndim=len(coordinates)    
    if ndim > 3:
        raise ValueError('Detrend is not supported over 3D.')
    if ndim == 2:
        coord_obj_1=d.get_coordinate_object(coordinates[0])
        coord_obj_2=d.get_coordinate_object(coordinates[1])
        dim1=coord_obj_1.dimension_list
        dim2=coord_obj_2.dimension_list
        dim=np.unique(np.append(dim1,dim2))
        
        #[1,j,j2,j3,i,ij,ij2,i2,i2j,i3]
        points=np.asarray([[i**k * j**l for k in range(order+1) for l in range(order-k+1)] for i in range(d.data.shape[dim[0]]) for j in range(d.data.shape[dim[1]])]) #The actual polynomial calculation
        if ndim == total_dim:
            values=np.reshape(d.data,d.data.shape[dim[0]]*d.data.shape[dim[1]]) #Reshapes to the same order as the points are aranged
            coeff=np.dot(np.dot(np.linalg.inv(np.dot(points.T,points)),points.T),values)#This performs the linear regression
            trend=np.dot(points,coeff)
            trend=np.reshape(trend,[d.data.shape[dim[0]],d.data.shape[dim[1]]])
            d.data=d.data-trend
            if test:
                import matplotlib.pyplot as plt
                plt.figure()
                plt.contourf(trend.T,levels=51)
                plt.figure()
                plt.contourf(d.data,levels=51)
                plt.figure()
                plt.contourf(d.data-trend,levels=51)
        else:
            alldim=np.arange(ndim)
            non_detrend_dim=np.where(np.logical_and(alldim != dim1,alldim != dim2))[0][0]
            n_fit=d.data.shape[non_detrend_dim]
            for i in range(n_fit):
                index=[slice(None)] * total_dim
                index[non_detrend_dim]=i
                values=np.reshape(d.data[tuple(index)],d.data.shape[dim[0]]*d.data.shape[dim[1]])
                coeff=np.dot(np.dot(np.linalg.inv(np.dot(points.T,points)),points.T),values)#This performs the linear regression
                trend=np.dot(points,coeff)
                trend=np.reshape(trend,[d.data.shape[dim[0]],d.data.shape[dim[1]]])
                d.data[tuple(index)]=d.data[tuple(index)]-trend
                
    if ndim == 3:
        coord_obj_1=d.get_coordinate_object(coordinates[0])
        coord_obj_2=d.get_coordinate_object(coordinates[1])
        coord_obj_3=d.get_coordinate_object(coordinates[2])
        dim1=coord_obj_1.dimension_list
        dim2=coord_obj_2.dimension_list
        dim3=coord_obj_3.dimension_list
        dim=np.unique(np.append(np.append(dim1,dim2),dim3))
        points=np.asarray([[i**l * j**m * k**n for l in range(order+1) for m in range(order-l+1) for n in range(order-l-m+1)] for i in range(d.data.shape[dim[0]]) for j in range(d.data.shape[dim[1]]) for k in range(d.data.shape[dim[2]])])
        if ndim == total_dim:
            values=np.reshape(d.data,d.data.shape[dim[0]]*d.data.shape[dim[1]]*d.data.shape[dim[2]])
            coeff=np.dot(np.dot(np.linalg.inv(np.dot(points.T,points)),points.T),values)#This performs the linear regression
            trend=np.dot(points,coeff)
            trend=np.reshape(trend,[d.data.shape[dim[0]],d.data.shape[dim[1]],d.data.shape[dim[2]]])
            d.data=d.data-trend
        else:
            alldim=np.arange(ndim)
            non_detrend_dim=np.where(np.logical_and(alldim != dim1,alldim != dim2))[0][0]
            n_fit=d.data.shape[non_detrend_dim]
            for i in range(n_fit):
                index=[slice(None)] * total_dim
                index[non_detrend_dim]=i
                values=np.reshape(d.data[tuple(index)],d.data.shape[dim[0]]*d.data.shape[dim[1]])
                coeff=np.dot(np.dot(np.linalg.inv(np.dot(points.T,points)),points.T),values)#This performs the linear regression
                trend=np.dot(points,coeff)
                trend=np.reshape(trend,[d.data.shape[dim[0]],d.data.shape[dim[1]],d.data.shape[dim[2]]])
                d.data[tuple(index)]=d.data[tuple(index)]-trend
    if output_name is not None:
        try:
            flap.add_data_object(d,output_name)
        except Exception as e:
            raise e
    if not return_trend:
        return d
    else:
        return trend
    
def filename(exp_id=None,
             time_range=None,
             purpose=None,
             frange=None,
             comment=None,
             extension=None):
    if exp_id is None:
        raise ValueError('The exp_id needs to be set for the filename.')
    filename='NSTX_GPI_'+str(exp_id)
    if time_range is None:
        filename+='_whole'
    elif len(time_range) == 2 and type(time_range) == list:
        filename+='_'+str(time_range[0])+'_'+str(time_range[1])
    else:
        raise ValueError('Time range should be a two element list.')
    if purpose is not None:
        if type(purpose) is str:
            filename+='_'+purpose.replace(' ','_')
        else:
            raise TypeError('Purpose should be a string.')
    if frange is not None:
        if len(frange) == 2 and type(frange) == list:
            filename+='_freq_'+str(frange[0])+'_'+str(frange[1])
        else:
            raise ValueError('Frequency range should be a two element list if not None.')
    if comment is not None:
        if type(comment) is str:
            filename+='_'+comment.replace(' ','_')
        else:
            raise TypeError('Purpose should be a string.')
    if extension is not None:
        if type(extension) is str:
            filename+='.'+extension
        else:
            raise TypeError('Extension should be a string.')
        
    return filename

def polyfit_2D(x=None,
               y=None,
               values=None,
               order=None,
               irregular=False,
               return_fit=False):
    
    if not irregular:
        if len(values.shape) != 2:
            raise ValueError('Values are not 2D')
        if x is not None and y is not None:
            if x.shape != values.shape or y.shape != values.shape:
                raise ValueError('There should be as many points as values and their shape should match.')
        if order is None:
            raise ValueError('The order is not set.')
        if (x is None and y is not None) or (x is not None and y is None):
            raise ValueError('Either both or neither x and y need to be set.')
        if x is None and y is None:
            polynom=np.asarray([[i**k * j**l for k in range(order+1) for l in range(order-k+1)] for i in range(values.shape[0]) for j in range(values.shape[1])]) #The actual polynomial calculation
        else:
            polynom=np.asarray([[x[i,j]**k * y[i,j]**l for k in range(order+1) for l in range(order-k+1)] for i in range(values.shape[0]) for j in range(values.shape[1])]) #The actual polynomial calculation
        original_shape=values.shape
        values=np.reshape(values, values.shape[0]*values.shape[1])
        if not return_fit:
            return np.dot(np.dot(np.linalg.inv(np.dot(polynom.T,polynom)),polynom.T),values) #This performs the linear regression
        else:
            return np.reshape(np.dot(polynom,np.dot(np.dot(np.linalg.inv(np.dot(polynom.T,polynom)),polynom.T),values)),original_shape)
    else:
        if x.shape != y.shape or x.shape != values.shape:
            raise ValueError('The points should be an [n,2] vector.')
        if len(x.shape) != 1 or len(y.shape) != 1 or len(values.shape) != 1:
            raise ValueError('x,y,values should be a 1D vector when irregular is set.')
        if order is None:
            raise ValueError('The order is not set.')
        polynom=np.asarray([[x[i]**k * y[i]**l for k in range(order+1) for l in range(order-k+1)] for i in range(values.shape[0])]) #The actual polynomial calculation
        if not return_fit:
            return np.dot(np.dot(np.linalg.inv(np.dot(polynom.T,polynom)),polynom.T),values) #This performs the linear regression
        else:
            return np.dot(polynom,np.dot(np.dot(np.linalg.inv(np.dot(polynom.T,polynom)),polynom.T),values))
        
def subtract_photon_peak_2D(autocorr=None,     #INPUT autocorrelation metrix
                            order=2,           #Order of the fitting
                            neglect_range=1,   #Range to be substituted by the fit 1=middle value, 2=+-1 area around middle etc.
                            fitting_range=2    #Range to be fit with the polynom
                            ):
    if autocorr is None:
        raise ValueError('No input is given.')
    index=[0] * 2
    middle_index=[0] * 2
    for i in range(2):
        index[i]=slice(autocorr.shape[i]//2-fitting_range,autocorr.shape[i]//2+fitting_range+1)
        if neglect_range == 1:
            middle_index[i]=autocorr.shape[i]//2
        else:
            middle_index[i]=slice(autocorr.shape[i]//2-(neglect_range-1),autocorr.shape[i]//2+(neglect_range-1)+1)
    x=np.zeros(autocorr.shape)
    y=np.zeros(autocorr.shape)
    for j in range(x.shape[1]):
        x[:,j]=np.arange(x.shape[0])
    for i in range(y.shape[0]):
        y[i,:]=np.arange(y.shape[1])
    _autocorr=copy.deepcopy(autocorr)
    _autocorr[tuple(middle_index)]=np.nan
    to_be_fit_index=np.logical_not(np.isnan(_autocorr[tuple(index)]))

    x_to_be_fit=(x[tuple(index)])[to_be_fit_index]
    y_to_be_fit=(y[tuple(index)])[to_be_fit_index]
    autocorr_to_be_fit=(_autocorr[tuple(index)])[to_be_fit_index]

    coeff=flap_nstx.analysis.polyfit_2D(x=x_to_be_fit,
                                        y=y_to_be_fit,
                                        values=autocorr_to_be_fit, 
                                        order=order,
                                        irregular=True)
    points=np.asarray([[x[k,l]**i * y[k,l]**j for k in range(x.shape[0]) for l in range(x.shape[1]) ] for i in range(order+1) for j in range(order-i+1)], dtype='float64')
    fit = np.dot(coeff,points)
    fit=np.reshape(fit, autocorr.shape)
    _autocorr[tuple(middle_index)]=fit[tuple(middle_index)]
    return _autocorr