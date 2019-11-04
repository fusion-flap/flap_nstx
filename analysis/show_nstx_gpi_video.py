# -*- coding: utf-8 -*-

import os

import flap
import flap_nstx
import flap_mdsplus
import matplotlib.pyplot as plt

flap_nstx.register()
flap_mdsplus.register('NSTX_MDSPlus')

def show_nstx_gpi_video(exp_id=None, time_range=None, plot_filtered=False, 
                        cache_data=False, plot_efit=False, flux_coordinates=False,
                        new_plot=True):
    
    if ((exp_id is None) and (time_range is None)):
        print('The correct way to call the code is the following:\n')
        print('show_nstx_gpi_video(exp_id=141918, time_range=[250.,260.], plot_filtered=True, cache_data=False, plot_efit=True, flux_coordinates=False)\n')
        print('INPUTs: \t\t Description: \t\t\t\t\t Type: \t\t\t Default values: \n')
        print('exp_id: \t\t The shot number. \t\t\t\t int \t\t\t Default: None')
        print('time_range: \t\t The time range. \t\t\t\t [float,float] \t\t Default: None')
        print('plot_filtered: \t\t High pass filter the data from 1e2. \t\t boolean \t\t Default: False')
        print('cache_data: \t\t Cache the FLAP data objects. \t\t\t boolean \t\t Default: False')
        print('plot_efit: \t\t Plot the efit profiles onto the video. \t boolean \t\t Default: False')
        print('flux_coordinates: \t Plot the data in flux coordinates. \t\t boolean \t\t Default: False')
        print('new_plot: \t\t Preserve the previous plot. \t\t boolean \t\t Default: True')
        return
    
    if time_range is None:
        print('time_range is None, the entire shot is plot.')
        slicing_range=None
    else:    
        if (type(time_range) is not list and len(time_range) != 2):
            raise TypeError('time_range needs to be a list with two elements.')
        time_range=[time_range[0]/1000., time_range[1]/1000.]    
        slicing_range={'Time':flap.Intervals(time_range[0],time_range[1])}
        
    if cache_data: #This needs to be enhanced to actually cache the data no matter what
        flap.delete_data_object('*')
    if exp_id is not None:
        print("\n------- Reading NSTX GPI data --------")
        if cache_data:
            try:
                d=flap.get_data_object_ref(exp_id=exp_id,object_name='GPI')
            except:
                print('Data is not cached, it needs to be read.')
        else:
            d=flap.get_data('NSTX_GPI',exp_id=exp_id,name='',object_name='GPI')
        object_name='GPI'
    else:
        raise ValueError('The experiment ID needs to be set.')
        
    if plot_filtered:
        print("**** Filtering GPI")
        object_name='GPI_FILTERED'
        d=flap.filter_data('GPI',output_name='GPI_FILTERED',coordinate='Time',
                           options={'Type':'Highpass','f_low':1e2/1e3,
                                    'Design':'Chebyshev II'}) #Data is in milliseconds
       
    if plot_efit:
        print('Gathering MDSPlus EFIT data.')
        flap.get_data('NSTX_MDSPlus',
                      name='\EFIT01::\RBDRY',
                      exp_id=exp_id,
                      object_name='SEP X OBJ'
                      )
    
        flap.get_data('NSTX_MDSPlus',
                      name='\EFIT01::\ZBDRY',
                      exp_id=exp_id,
                      object_name='SEP Y OBJ'
                      )
    #The NSTX limiter data (or the EFIT) is not correct and there are events
    #outside the limiter which is not feasible. Hence, the corresponding parts
    #are commented out.
    
    #    flap.get_data('NSTX_MDSPlus',
    #                  name='\EFIT01::\RLIM',
    #                  exp_id=exp_id,
    #                  object_name='LIM X OBJ'
    #                  )
    #
    #    flap.get_data('NSTX_MDSPlus',
    #                  name='\EFIT01::\ZLIM',
    #                  exp_id=exp_id,
    #                  object_name='LIM Y OBJ'
    #                  )
        
        flap.get_data('NSTX_MDSPlus',
                      name='\EFIT01::\PSIRZ',
                      exp_id=exp_id,
                      object_name='PSI RZ OBJ'
                      )
        
        flap.get_data('NSTX_MDSPlus',
                      name='\EFIT01::\GAPIN',
                      exp_id=exp_id,
                      object_name='GAPIN'
                      )
        flap.get_data('NSTX_MDSPlus',
                      name='\EFIT01::\SSIMAG',
                      exp_id=exp_id,
                      object_name='SSIMAG'
                      )
        flap.get_data('NSTX_MDSPlus',
                      name='\EFIT01::\SSIBRY',
                      exp_id=exp_id,
                      object_name='SSIBRY'
                      )
        
        efit_options={'Plot separatrix': True,
                      'Separatrix X': 'SEP X OBJ',
                      'Separatrix Y': 'SEP Y OBJ',
                      'Separatrix color': 'red',
    #                  'Plot limiter': True,
    #                  'Limiter X': 'LIM X OBJ',
    #                  'Limiter Y': 'LIM Y OBJ',
    #                  'Limiter color': 'white',
                      'Plot flux': True,
                      'Flux XY': 'PSI RZ OBJ'}
    else:
        efit_options=None
        
    if flux_coordinates:
        print("**** Addig Flux r coordinates")
        d.add_coordinate(coordinates='Flux r',exp_id=exp_id)
        print("**** Addig Flux theta coordinates")
        d.add_coordinate(coordinates='Flux theta',exp_id=exp_id)
        x_axis='Flux r'
        y_axis='Flux theta'
    else:
        x_axis='Device R'
        y_axis='Device z'
    if new_plot:
        plt.figure()
        
    print("**** Plotting GPI")
    flap.plot(object_name,plot_type='animation',
              exp_id=exp_id,
              slicing=slicing_range,
              axes=[x_axis,y_axis,'Time'],
              options={'Z range':[0,512],'Wait':0.0,'Clear':False,
                       'EFIT options':efit_options,
                       'Colormap':'jet'})
    
thisdir = os.path.dirname(os.path.realpath(__file__))
fn = os.path.join(thisdir,"flap_nstx.cfg")
flap.config.read(file_name=fn)            

#show_nstx_gpi_video(exp_id=141918, time_range=[250.,260.], plot_filtered=True, cache_data=False, plot_efit=True, flux_coordinates=False)