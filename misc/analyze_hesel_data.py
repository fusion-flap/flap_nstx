#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 16:23:01 2021

@author: mlampert
"""

from flap_nstx.analysis import calculate_tde_velocity,  calculate_sde_velocity, calculate_sde_velocity_distribution
#Core modules
import os
import copy
import h5py
import pickle

from matplotlib.backends.backend_pdf import PdfPages

import matplotlib.pyplot as plt

import flap
import flap_nstx
flap_nstx.register()
import flap_mdsplus
flap_mdsplus.register('NSTX_MDSPlus')

thisdir = os.path.dirname(os.path.realpath(__file__))
fn = os.path.join(thisdir,"flap_nstx.cfg")
flap.config.read(file_name=fn)
wd=flap.config.get_all_section('Module NSTX_GPI')['Working directory']

import numpy as np


def read_flap_hesel_data(output_name=None,
                         read_he_data=False,
                         read_electric_field=False,
                         radial=False,
                         poloidal=False,):
    if read_he_data:
        f=h5py.File('/Users/mlampert/work/NSTX_workspace/AUG_HESEL_files/n_He__nHESEL_AUG_00000.h5', 'r')
        data_field_key='n_He'
        name='He density'
        unit='m-3'
    elif read_electric_field:
        f=h5py.File('/Users/mlampert/work/NSTX_workspace/AUG_HESEL_files/phi__nHESEL_AUG_00000.h5', 'r')
        data_field_key='phi'
        name='Electric field'
        unit='m-3'
    else:
        f=h5py.File('/Users/mlampert/work/NSTX_workspace/AUG_HESEL_files/n_e__nHESEL_AUG_00000.h5', 'r')
        data_field_key='n_e'
        name='e- density'
        unit='m-3'
    
    x_coord=np.asarray(list(f['axes']['x_axis']))
    y_coord=np.asarray(list(f['axes']['x_axis']))
    t_coord=np.asarray(list(f['axes']['t_axis']))
    
    if not read_electric_field:
        data=np.asarray(list(f['fields'][data_field_key])) #t, x, y
    else:
        scalar_field=np.asarray(list(f['fields'][data_field_key]))
        if radial:
            index=0
            name='E_r'
            divider=x_coord[1]-x_coord[0]
            
        if poloidal:
            index=1
            name='E_p'
            divider=y_coord[1]-y_coord[0]
            
        data=np.asarray(np.gradient(scalar_field))[index,:,:,:]/divider
        
        unit='V/m'
        
    coord=[]
    coord.append(copy.deepcopy(flap.Coordinate(name='Time',
                               unit='s',
                               mode=flap.CoordinateMode(equidistant=True),
                               start=t_coord[0],
                               step=t_coord[1]-t_coord[0],
                               #shape=time_arr.shape,
                               dimension_list=[0]
                               )))
    
    coord.append(copy.deepcopy(flap.Coordinate(name='Sample',
                               unit='n.a.',
                               mode=flap.CoordinateMode(equidistant=True),
                               start=0,
                               step=1,
                               dimension_list=[0]
                               )))
    
    coord.append(copy.deepcopy(flap.Coordinate(name='Device R',
                               unit='m',
                               mode=flap.CoordinateMode(equidistant=True),
                               start=x_coord[0],
                               step=x_coord[1]-x_coord[0],
                               dimension_list=[1]
                               )))
    coord.append(copy.deepcopy(flap.Coordinate(name='Device z',
                               unit='m',
                               mode=flap.CoordinateMode(equidistant=True),
                               start=y_coord[0],
                               step=y_coord[1]-y_coord[0],
                               dimension_list=[2]
                               )))
    
    coord.append(copy.deepcopy(flap.Coordinate(name='Image x',
                               unit='pix',
                               mode=flap.CoordinateMode(equidistant=True),
                               start=0,
                               step=1,
                               dimension_list=[1]
                               )))
    coord.append(copy.deepcopy(flap.Coordinate(name='Image y',
                               unit='pix',
                               mode=flap.CoordinateMode(equidistant=True),
                               start=0,
                               step=1,
                               dimension_list=[2]
                               )))    
    
    data_unit = flap.Unit(name=name,
                          unit=unit)
    
    d = flap.DataObject(data_array=data,
                        error=None,
                        data_unit=data_unit,
                        coordinates=coord, 
                        exp_id=0,
                        data_title='AUG HESEL DATA')
    
    if output_name is not None:
        flap.add_data_object(d, output_name)
    return d

def analyze_hesel_data():
    try:
        d=flap.get_data_object_ref('HESEL_DATA')
    except:
        d=read_flap_hesel_data(output_name='HESEL_DATA')
    
    a=calculate_sde_velocity(d, time_range=[0,100e-6], filename='sde_code_trial', nocalc=False, correct_acf_peak=True, subtraction_order=4)
    
def analyze_hesel_data_2D_sde(time_range=[0,1e-3],
                              x_step=20,
                              x_res=40,
                              y_step=None,
                              y_res=None,
                              correct_acf_peak=False,
                              subtraction_order=1,
                              correlation_threshold=0.7,        
                              nocalc=False,
                              colormap=None,
                              video_framerate=24,
                              video_saving_only=False,
                              plot_video=False,
                              plot_time_series=False,
                              plot_time_contour=False):
    
    pickle_filename=wd+'/processed_data/hesel_velocity_analysis.pickle'
    try:
        d=flap.get_data_object_ref('HESEL_DATA')
    except:
        d=read_flap_hesel_data(output_name='HESEL_DATA')
        d_he=read_flap_hesel_data(output_name='HESEL_DATA', read_he_data=True)
        d.data=d.data*d_he.data
    if not nocalc:
        result=calculate_sde_velocity_distribution(d, 
                                                   time_range=time_range, 
                                                   x_step=x_step, x_res=x_res, y_step=y_step, y_res=y_res, 
                                                   filename=wd+'/processed_data/sde_code_trial', 
                                                   nocalc=False, 
                                                   correct_acf_peak=correct_acf_peak, 
                                                   subtraction_order=1, 
                                                   normalize=None,
                                                   correlation_threshold=correlation_threshold, 
                                                   return_data_object=True, 
                                                   return_displacement=True
                                                   )
        pickle.dump(result, open(pickle_filename, 'wb'))
    else:
        result=pickle.load(open(pickle_filename, 'rb'))
    
    flap.add_data_object(result[0], 'X_OBJ')
    flap.add_data_object(result[1], 'Y_OBJ')
    
    oplot_options={'arrow':{'DISP':{'Data object X':'X_OBJ',
                                    'Data object Y':'Y_OBJ',
                                    'Plot':True,
                                    'width':0.0001,
                                    'color':'red',
                                    }}}
    if plot_video:
        if video_saving_only:
            if video_saving_only:
                import matplotlib
                current_backend=matplotlib.get_backend()
                matplotlib.use('agg')
                waittime=0.
            else:
                waittime=1./24.
                waittime=0.
                
            d.plot(plot_type='anim-image',
                   axes=['Device R','Device z','Time'],
                   slicing={'Time':flap.Intervals(time_range[0],time_range[1])},
                   options={'Wait':0.0,'Clear':False,
                            'Overplot options':oplot_options,
                            'Colormap':colormap,
                            'Equal axes':True,
                            'Waittime':waittime,
                            'Video file':wd+'/plots/HESEL_DATA/HESEL_VIDEO_WITH_POLOIDAL_VELOCITY.mp4',
                            'Video format':'mp4',
                            'Video framerate':video_framerate,
                            'Prevent saturation':False,
                            })
            
            if video_saving_only:
                import matplotlib
                matplotlib.use(current_backend)
        else:
            d.plot(plot_type='anim-contour', 
                   slicing={'Time':flap.Intervals(time_range[0],time_range[1])},
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
    if plot_time_series:
        titles=['Radial vel.','Poloidal vel.']

        pdf=PdfPages(wd+'/plots/HESEL_DATA/velocity_analysis.pdf')
        for i in [0,1]:
            
            result[i].data_unit.name=titles[i]
            result[i].data_unit.unit='m/s'
            result[i].data = result[i].data/(result[i].coordinate('Time')[0][1,0,0]-result[0].coordinate('Time')[0][0,0,0])
            
            plt.figure()
            result[i].plot(axes=['Time', 'Device R'], plot_type='contour', slicing={'Image y':255.5})
            plt.title(titles[i])
            pdf.savefig()
            
            plt.figure()
            result[i].plot(axes=['Time'], plot_type='multi xy', slicing={'Image y':255.5})
            pdf.savefig()
            
        plt.figure()
        for j in range(10):
            d_new=flap.slice_data('X_OBJ',slicing={'Image y':255.5}).slice_data(slicing={'Sample':flap.Intervals(j*80,(j+1)*80)}, summing={'Sample':'Mean'})
            time=int(np.mean(d_new.coordinate('Time')[0])*1e6)
            d_new.plot(axes=['Device R'], plot_type='xy', plot_options={'label':'Time: '+str(time)+'us'})
            plt.legend()
            
        pdf.savefig()
        
        plt.figure()
        for j in range(10):
            d_new2=flap.slice_data('Y_OBJ',slicing={'Image y':255.5}).slice_data(slicing={'Sample':flap.Intervals(j*80,(j+1)*80)}, summing={'Sample':'Mean'})
            time=int(np.mean(d_new2.coordinate('Time')[0])*1e6)
            d_new2.plot(axes=['Device R'], plot_type='xy', plot_options={'label':'Time: '+str(time)+'us'})
            plt.legend()
        pdf.savefig()
        pdf.close()
        
def analyze_hesel_data_2D_tde():

    d=read_flap_hesel_data(output_name='HESEL_DATA')
    calculate_tde_velocity(d,
                           y_direction=True, 
                           save_data=True, 
                           time_res=2e-4, 
                           taures=1.2e-6, 
                           time_delay=False, 
                           nocalc=False, 
                           xrange=[200,512], 
                           yrange=[100,200], 
                           filter_data=True, 
                           f_high=200e3, 
                           correlation_threshold=0.0,
                           plot=False)
    pdf=PdfPages(wd+'/plots/HESEL_DATA/velocity_tde_analysis.pdf')
    plt.figure()
    flap.plot('DATA_POL_VELOCITY', 
          exp_id=0,
          plot_type='contour', 
          axes=['Time', 'Device R'],
          plot_options={'levels':51},
          slicing={'Time':flap.Intervals(0.3e-3, 1.2e-3)}, 
          options={'Colormap':'gist_ncar'})
    pdf.savefig()

    
    plt.figure()
    time_vec=flap.get_data_object_ref('DATA_POL_VELOCITY').coordinate('Time')[0][0,:]
    for j in range(len(time_vec)):
        d_new2=flap.slice_data('DATA_POL_VELOCITY',slicing={'Time':time_vec[j]})
        time=int(np.mean(d_new2.coordinate('Time')[0])*1e6)
        d_new2.plot(axes=['Device R'], plot_type='xy', plot_options={'label':'Time: '+str(time)+'us'})
    plt.legend()
    pdf.savefig()
    pdf.close()