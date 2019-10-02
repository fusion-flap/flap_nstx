# -*- coding: utf-8 -*-
"""
Created on Fri May 10 18:50:28 2019

@author: Mate Lampert (mlampert@pppl.gov)
"""

import os

import flap
import flap_nstx
import flap_mdsplus
import matplotlib.pyplot as plt
import time

start_time=time.time()

flap_nstx.register()
flap_mdsplus.register('NSTX_MDSPlus')

def test_NSTX_GPI_data_animation(exp_id=141918):
    flap.delete_data_object('*')
    print("\n------- test data read with NSTX GPI data --------")

    d=flap.get_data('NSTX_GPI',exp_id=exp_id,name='',object_name='GPI')
    print("**** Storage contents")
    flap.list_data_objects()
    plt.close('all')
    print("**** Filtering GPI")
    #d_filter=flap.filter_data('GPI',output_name='GPI_filt',coordinate='Time',
    #                          options={'Type':'Highpass','f_low':1e2/1e3,'Design':'Chebyshev II'}) #Data is in milliseconds
    
    print('Gathering MDSPlus data')

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
    
    d=flap.get_data('NSTX_MDSPlus',
                  name='\EFIT01::\PSIRZ',
                  exp_id=exp_id,
                  object_name='PSI RZ OBJ'
                  )
    
    d=flap.get_data('NSTX_MDSPlus',
                  name='\EFIT01::\GAPIN',
                  exp_id=exp_id,
                  object_name='GAPIN'
                  )
    d=flap.get_data('NSTX_MDSPlus',
                  name='\EFIT01::\SSIMAG',
                  exp_id=exp_id,
                  object_name='SSIMAG'
                  )
    d=flap.get_data('NSTX_MDSPlus',
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
                  'Flux XY': 'PSI RZ OBJ',}
    
    print("**** Plotting filtered GPI")
    flap.plot('GPI',plot_type='animation',
              slicing={'Time':flap.Intervals(250.,260.)},
              axes=['Device R','Device z','Time'],
              options={'Z range':[0,512],'Wait':0.0,'Clear':False,
                       'EFIT options':efit_options})
    
    flap.list_data_objects()

def test_NSTX_GPI_norm_flux_coord(exp_id=141918):
    flap.delete_data_object('*')
    print("\n------- test data read with NSTX GPI data and plotting it as a function of normalized flux coordinates --------")

    d=flap.get_data('NSTX_GPI',exp_id=exp_id,name='',object_name='GPI')
    print("**** Storage contents")
    flap.list_data_objects()
    plt.close('all')
    print("**** Adding normalized flux coordinates.")
    print("--- %s seconds ---" % (time.time() - start_time))
    #d.add_coordinate(coordinates='Flux r',exp_id=exp_id)
    print("--- %s seconds ---" % (time.time() - start_time))
    d.add_coordinate(coordinates='Flux theta',exp_id=exp_id)
    flap.list_data_objects()
    print('Coordinate was read.')
    print("--- %s seconds ---" % (time.time() - start_time))
    flap.plot('GPI',plot_type='animation',\
                slicing={'Time':flap.Intervals(250.,260.)},\
                axes=['Flux R','Flux theta','Time'],\
                options={'Z range':[0,512],'Wait':0.0,'Clear':False})
    print("--- %s seconds ---" % (time.time() - start_time))
    
def test_flap_time(exp_id=140620):
    flap.delete_data_object('*')
    print("\n------- test data read with NSTX GPI data --------")

    d=flap.get_data('NSTX_GPI',exp_id=exp_id,name='',object_name='GPI')
    print("**** Storage contents")
    flap.list_data_objects()
    plt.close('all')
    #print("**** Filtering GPI")
    #d_filter=flap.filter_data('GPI',output_name='GPI_filt',coordinate='Time',
    #                          options={'Type':'Highpass','f_low':1e2/1e3,'Design':'Chebyshev II'}) #Data is in milliseconds

    print("**** Plotting filtered GPI")
    #flap.plot('GPI',plot_type='animation',
    #          slicing={'Time':flap.Intervals(550.,580.)},
    #          axes=['Device R','Device z','Time'],
    #          options={'Z range':[0,512],'Wait':0.0,'Clear':False})
    #
    print(d.time)
    
    flap.list_data_objects()
    
# Reading configuration file in the test directory
thisdir = os.path.dirname(os.path.realpath(__file__))
fn = os.path.join(thisdir,"test_nstx_gpi.cfg")
flap.config.read(file_name=fn)


#test_flap_time()
test_NSTX_GPI_data_animation()
#test_NSTX_GPI_norm_flux_coord()
