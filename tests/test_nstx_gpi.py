# -*- coding: utf-8 -*-
"""
Created on Fri May 10 18:50:28 2019

@author: Sandor Zoletnik  (zoletnik.sandor@wigner.mta.hu)
"""

import matplotlib.pyplot as plt
import os

import flap
import flap_nstx_gpi

flap_nstx_gpi.register()

def test_NSTX_GPI_data():
    flap.delete_data_object('*')
    print("\n------- test data read with NSTX GPI data --------")

    d=flap.get_data('NSTX_GPI',exp_id=142232,name='',object_name='GPI')
    print("**** Storage contents")
    flap.list_data_objects()
    plt.close('all')
    print("**** Plotting GPI")
    flap.plot('GPI',plot_type='animation',slicing={'Time':flap.Intervals(400.,410.)},axes=['Device R','Device z','Time'],options={'Z range':[0,512],'Wait':0.0,'Clear':False})
    #flap.plot('GPI',plot_type='animation',slicing={'Time':flap.Intervals(0.40,0.41)},axes=['Image y','Image x','Time'],options={'Z range':[0,512],'Wait':0.0,'Clear':False})
    
# Reading configuration file in the test directory
thisdir = os.path.dirname(os.path.realpath(__file__))
fn = os.path.join(thisdir,"test_nstx_gpi.cfg")
flap.config.read(file_name=fn)

test_NSTX_GPI_data()
