# -*- coding: utf-8 -*-

"""
Created on Sat Jul 24 13:22:00 2019

@author: Lampert

This is the flap module for NSTX GPI diagnostic
Needs pims package installed
e.g.: conda install -c conda-forge pims

"""

import os
import numpy as np
import copy
import subprocess
import pims
from scipy import interpolate

import flap
import flap_mdsplus
#from .spatcal import *

if (flap.VERBOSE):
    print("Importing flap_nstx_gpi")
    
    
def nstx_gpi_get_data(exp_id=None, data_name=None, no_data=False, options=None, coordinates=None, data_source=None):
    #translate the input variables to the actual directory name on portal
    #copy the file from the portal to PC if it is not already there
    #interpret the .cin file
        # read the header
        # read the data
    if (exp_id is None):
        raise ValueError('exp_id should be set for NSTX GPI.')
    if (type(exp_id) is not int):
        raise TypeError("exp_id should be an integer.")

    default_options = {'Local datapath':'data',
                       'Datapath':None,
                       'Scaling':'Digit',
                       'Offset timerange': None,
                       'Calibration': False,
                       'Calib. path': 'cal',
                       'Calib. file': None,
                       'Phase' : None,
                       'State' : None,
                       'Start delay': 0,
                       'End delay': 0,
                       'Download only': False
                       }
    _options = flap.config.merge_options(default_options,options,data_source='NSTX_GPI')
    #folder decoder
    folder={
               '_0_': 'Phantom71-5040',
               '_1_': 'Phantom710-9206',
               '_2_': 'Phantom73-6747',
               '_3_': 'Phantom73-6663',
               '_4_': 'Phantom73-8032',
               '_5_': 'Phantom710-9205',
               '_6_': 'Miro4-9373'
            }
    data_title='NSTX GPI data'
    if (exp_id < 118929):
        year=2005
    if (exp_id >= 118929) and (exp_id < 122270):
        year=2006
    if (exp_id >= 122270) and (exp_id < 126511):
        year=2007
    if (exp_id >= 126511) and (exp_id < 131565):
        year=2008
    if (exp_id >= 131565) and (exp_id < 137110):
        year=2009
    if (exp_id >= 137110):
        year=2010
    
    if (year < 2006):
        cam='_0_'
    if (year == 2007 or year == 2008):
        cam='_1_'
    if (year == 2009):
        cam='_2_'
    if (year == 2010):
        cam='_5_'
    
    if (year < 2006):
        file_name='nstx'+str(exp_id)+'.cin'
    else:
        file_name='nstx'+cam+str(exp_id)+'.cin'
    file_folder=_options['Datapath']+'/'+folder[cam]+\
                '/'+str(year)+'/'
    remote_file_name=file_folder+file_name
    local_file_folder=_options['Local datapath']+'/'+str(exp_id)+'/'
    if not os.path.exists(_options['Local datapath']):
        raise SystemError("The local datapath cannot be found.")
        return
    
    if not (os.path.exists(local_file_folder+file_name)):
        if not (os.path.exists(local_file_folder)):
            try:
                os.mkdir(local_file_folder)
            except:
                raise SystemError("The folder cannot be created." 
                                  +"Dumping the file to scratch")
                local_file_folder=_options['Local datapath']+'/scratch'

        
        p = subprocess.Popen(["scp", _options['User']+"@"+_options['Server']+':'+
                              remote_file_name, local_file_folder])
        sts = os.waitpid(p.pid, 0)
        if not (os.path.exists(local_file_folder+file_name)):
            raise SystemError("The file couldn't be transferred to the local directory.")
    if (_options['Download only']):
            d = flap.DataObject(data_array=np.asarray([0,1]),
                        data_unit=None,
                        coordinates=None,
                        exp_id=exp_id,
                        data_title=data_title,
                        info={'Options':_options},
                        data_source="NSTX_GPI")
            return d
        
    images=pims.Cine(local_file_folder+file_name)

    data_arr=np.asarray(images[:], dtype=np.int16)
    data_unit = flap.Unit(name='Signal',unit='Digit')

    time_arr=np.asarray([i[0].timestamp() - images.trigger_time['datetime'].timestamp()-images.trigger_time['second_fraction']  for i in images.frame_time_stamps],dtype=np.float)\
                    +np.asarray([i[1]              for i in images.frame_time_stamps],dtype=np.float)

    coord = [None]*6
    
    
        #def get_time_to_trigger(self, i):
        #"""Get actual time (s) of frame i, relative to trigger."""
        #ti = 
        #ti = images.frame_time_stamps[i][0].timestamp() + images.frame_time_stamps[i][1]
        #tt= 
        #tt = images.trigger_time['datetime'].timestamp() + images.trigger_time['second_fraction']
        #return ti - tt
    
    coord[0]=(copy.deepcopy(flap.Coordinate(name='Time',
                                               unit='ms',
                                               mode=flap.CoordinateMode(equidistant=True),
                                               start=time_arr[0]*1000.,
                                               step=(time_arr[1]-time_arr[0])*1000.,
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
    coord[2]=(copy.deepcopy(flap.Coordinate(name='Image x',
                                               unit='Pixel',
                                               mode=flap.CoordinateMode(equidistant=True),
                                               start=0,
                                               step=1,
                                               shape=[],
                                               dimension_list=[1]
                                               )))
    coord[3]=(copy.deepcopy(flap.Coordinate(name='Image y',
                                               unit='Pixel',
                                               mode=flap.CoordinateMode(equidistant=True),
                                               start=0,
                                               step=1,
                                               shape=[],
                                               dimension_list=[2]
                                               )))
    
        #Get the spatial calibration for the GPI data
    #This spatial calibration is based on the rz_map.dat which used a linear
    #approximation for the transformation between pixel and spatial coordinates
    #This needs to be updated as soon as more information is available on the
    #calibration coordinates.
      
    coeff_r=np.asarray([3.7183594,-0.77821046,1402.8097])
    coeff_z=np.asarray([0.18090118,3.0657776,70.544312])
     
    coord[4]=(copy.deepcopy(flap.Coordinate(name='Device R',
                                               unit='mm',
                                               mode=flap.CoordinateMode(equidistant=True),
                                               start=coeff_r[2],
                                               step=[coeff_r[0],coeff_r[1]],
                                               dimension_list=[1,2]
                                               )))
    
    coord[5]=(copy.deepcopy(flap.Coordinate(name='Device z',
                                               unit='mm',
                                               mode=flap.CoordinateMode(equidistant=True),
                                               start=coeff_z[2],
                                               step=[coeff_z[0],coeff_z[1]],
                                               dimension_list=[1,2]
                                               )))
    
    _options["Trigger time"]=images.trigger_time['second_fraction']
    _options["FPS"]=images.frame_rate
    _options["X size"]=images.frame_shape[0]
    _options["Y size"]=images.frame_shape[1]
    _options["Bits"]=images.bitmapinfo_dict["bi_bit_count"]
    
    d = flap.DataObject(data_array=data_arr,
                        data_unit=data_unit,
                        coordinates=coord,
                        exp_id=exp_id,
                        data_title=data_title,
                        info={'Options':_options},
                        data_source="NSTX_GPI")
    return d

def add_coordinate(data_object,
                   coordinates,
                   exp_id=None,
                   options=None): 
    #This part of the code provides normalized flux coordinates for the GPI data
    if ('Device phi' in coordinates):
        try:
            gpi_time=data_object.coordinate('Time')[0][:,0,0]
            gpi_r_coord=data_object.coordinate('Device R')[0]
            gpi_z_coord=data_object.coordinate('Device z')[0]
        except:
            raise ValueError('R,z or t coordinates are missing.')
        try:
            psi_rz_obj=flap.get_data('NSTX_MDSPlus',
                                     name='\EFIT01::\PSIRZ',
                                     exp_id=data_object.exp_id,
                                     object_name='PSIRZ_FOR_COORD')
            psi_mag=flap.get_data('NSTX_MDSPlus',
                                     name='\EFIT01::\SSIMAG',
                                     exp_id=data_object.exp_id,
                                     object_name='SSIMAG_FOR COORD')
            psi_bdry=flap.get_data('NSTX_MDSPlus',
                                     name='\EFIT01::\SSIBRY',
                                     exp_id=data_object.exp_id,
                                     object_name='SSIBRY_FOR_COORD')
        except:
            raise ValueError("The PSIRZ MDSPlus node cannot be reached.")
        try:
            psi_values=psi_rz_obj.data
            psi_t_coord=psi_rz_obj.coordinate('Time')[0][:,0,0]*1000. #GPI data is in ms.
            psi_r_coord=psi_rz_obj.coordinate('Device R')[0]*1000 #GPI data is in mm.
            psi_z_coord=psi_rz_obj.coordinate('Device z')[0]*1000.#GPI data is in mm.
        except:
            raise ValueError("The flux data cannot be found.")
        #Do the interpolation
        psi_values_spat_interpol=np.zeros([psi_t_coord.shape[0],gpi_r_coord.shape[1],gpi_r_coord.shape[2]])
        try:
            for index_t in range(psi_t_coord.shape[0]):
                points=np.asarray([psi_r_coord[index_t,:,:].flatten(),psi_z_coord[index_t,:,:].flatten()]).transpose()
                values=((psi_values[index_t]-psi_mag.data[index_t])/(psi_bdry.data[index_t]-psi_mag.data[index_t])).flatten()
                psi_values_spat_interpol[index_t,:,:]=interpolate.griddata(points,values,(gpi_r_coord[0,:,:].transpose(),gpi_z_coord[0,:,:].transpose()),method='cubic').transpose()
            psi_values_total_interpol=np.zeros(data_object.data.shape)
            for index_r in range(gpi_r_coord.shape[1]):
                for index_z in range(gpi_r_coord.shape[2]):
                    psi_values_total_interpol[:,index_r,index_z]=np.interp(gpi_time,psi_t_coord,psi_values_spat_interpol[:,index_r,index_z])              
        except:
            raise ValueError("An error has occured during the interpolation.")
        new_coordinates=(copy.deepcopy(flap.Coordinate(name='Device phi',
                                       unit='',
                                       mode=flap.CoordinateMode(equidistant=False),
                                       values=psi_values_total_interpol,
                                       shape=psi_values_total_interpol.shape,
                                       dimension_list=[0,1,2]
                                       )))
        data_object.coordinates.append(new_coordinates)
        
        return data_object

def register():
    flap.register_data_source('NSTX_GPI', get_data_func=nstx_gpi_get_data, add_coord_func=add_coordinate)
