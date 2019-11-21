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

import matplotlib.pyplot as plt
from matplotlib import path as pltPath

import flap

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
        raise TypeError("exp_id should be an integer and not %s"%(type(exp_id)))

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
        os.waitpid(p.pid, 0)
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

    data_arr=np.flip(np.asarray(images[:], dtype=np.int16),2) #The original data is 80x64, this line converts it to 64x80
    
    data_unit = flap.Unit(name='Signal',unit='Digit')
    
    #time data cannot be directly extracted from a PIMS file as it is stored as ms timestamps 
    #and it is converted to datetime which is somehow not precise.
    #The following two codes are giving equivalent results within the framerate accuracy of the measurement.
    
    #trigger_time = datetime + second fraction of the first frame - datetime - second fraction of the trigger time
    #trigger_time=(images.frame_time_stamps[0][0].timestamp()+images.frame_time_stamps[0][1]-
    #              images.trigger_time['datetime'].timestamp()-images.trigger_time['second_fraction'])

    #The header dict contains the capture information along with the entire image number and the first_image_no (when the recording started)
    #The frame_rate corresponds with the one from IDL.
    trigger_time=images.header_dict['first_image_no']/images.frame_rate
    
    
    coord = [None]*6
    
    
    
        #def get_time_to_trigger(self, i):
        #"""Get actual time (s) of frame i, relative to trigger."""
        #ti = 
        #ti = images.frame_time_stamps[i][0].timestamp() + images.frame_time_stamps[i][1]
        #tt= 
        #tt = images.trigger_time['datetime'].timestamp() + images.trigger_time['second_fraction']
        #return ti - tt
    
    coord[0]=(copy.deepcopy(flap.Coordinate(name='Time',
                                               unit='s',
                                               mode=flap.CoordinateMode(equidistant=True),
                                               start=trigger_time,
                                               step=1/float(images.frame_rate),
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
      
    coeff_r=np.asarray([3.7183594,-0.77821046,1402.8097])/1000. #The coordinates are in meters
    coeff_z=np.asarray([0.18090118,3.0657776,70.544312])/1000. #The coordinates are in meters
     
    coord[4]=(copy.deepcopy(flap.Coordinate(name='Device R',
                                               unit='m',
                                               mode=flap.CoordinateMode(equidistant=True),
                                               start=coeff_r[2],
                                               step=[coeff_r[0],coeff_r[1]],
                                               dimension_list=[1,2]
                                               )))
    
    coord[5]=(copy.deepcopy(flap.Coordinate(name='Device z',
                                               unit='m',
                                               mode=flap.CoordinateMode(equidistant=True),
                                               start=coeff_z[2],
                                               step=[coeff_z[0],coeff_z[1]],
                                               dimension_list=[1,2]
                                               )))
    
    _options["Trigger time [s]"]=trigger_time
    _options["FPS"]=images.frame_rate
    _options["Sample time [s]"]=1/float(images.frame_rate)
    _options["Exposure time [s]"]=images.tagged_blocks['exposure_only'][0] 
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
    if ('Flux r' in coordinates):
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
                                     object_name='SSIMAG_FOR_COORD')
            psi_bdry=flap.get_data('NSTX_MDSPlus',
                                     name='\EFIT01::\SSIBRY',
                                     exp_id=data_object.exp_id,
                                     object_name='SSIBRY_FOR_COORD')
        except:
            raise ValueError("The PSIRZ MDSPlus node cannot be reached.")
        psi_values=psi_rz_obj.data
        psi_t_coord=psi_rz_obj.coordinate('Time')[0][:,0,0]
        psi_r_coord=psi_rz_obj.coordinate('Device R')[0]
        psi_z_coord=psi_rz_obj.coordinate('Device z')[0]
        #Do the interpolation
        psi_values_spat_interpol=np.zeros([psi_t_coord.shape[0],gpi_r_coord.shape[1],gpi_r_coord.shape[2]])
        try:
            for index_t in range(psi_t_coord.shape[0]):
                points=np.asarray([psi_r_coord[index_t,:,:].flatten(),psi_z_coord[index_t,:,:].flatten()]).transpose()
                values=((psi_values[index_t]-psi_mag.data[index_t])/(psi_bdry.data[index_t]-psi_mag.data[index_t])).flatten()
                values[np.isnan(values)]=0.
                psi_values_spat_interpol[index_t,:,:]=interpolate.griddata(points,values,(gpi_r_coord[0,:,:].transpose(),gpi_z_coord[0,:,:].transpose()),method='cubic').transpose()
            psi_values_total_interpol=np.zeros(data_object.data.shape)
            for index_r in range(gpi_r_coord.shape[1]):
                for index_z in range(gpi_r_coord.shape[2]):
                    psi_values_total_interpol[:,index_r,index_z]=np.interp(gpi_time,psi_t_coord,psi_values_spat_interpol[:,index_r,index_z])              
        except:
            raise ValueError("An error has occured during the interpolation.")
        psi_values_total_interpol[np.isnan(psi_values_total_interpol)]=0.
        new_coordinates=(copy.deepcopy(flap.Coordinate(name='Flux r',
                                       unit='',
                                       mode=flap.CoordinateMode(equidistant=False),
                                       values=psi_values_total_interpol,
                                       shape=psi_values_total_interpol.shape,
                                       dimension_list=[0,1,2]
                                       )))
        data_object.coordinates.append(new_coordinates)
        
    if ('Flux theta' in coordinates):
        print("ADDING FLUX THETA IS DEPRECATED AND MIGHT FAIL.")
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
            r_mag_axis=flap.get_data('NSTX_MDSPlus',
                                     name='\EFIT01::\RMAXIS',
                                     exp_id=data_object.exp_id,
                                     object_name='RMAXIS_FOR_COORD')
            z_mag_axis=flap.get_data('NSTX_MDSPlus',
                                     name='\EFIT01::\ZMAXIS',
                                     exp_id=data_object.exp_id,
                                     object_name='ZMAXIS_FOR_COORD')
            r_bdry_obj=flap.get_data('NSTX_MDSPlus',
                                     name='\EFIT01::\RBDRY',
                                     exp_id=exp_id,
                                     object_name='SEP X OBJ'
                                     )
            z_bdry_obj=flap.get_data('NSTX_MDSPlus',
                                     name='\EFIT01::\ZBDRY',
                                     exp_id=exp_id,
                                     object_name='SEP Y OBJ'
                                     )
        except:
            raise ValueError("The PSIRZ MDSPlus node cannot be reached.")
        try:
        #if True:
            psi_values=psi_rz_obj.data
            psi_t_coord=psi_rz_obj.coordinate('Time')[0][:,0,0]
            psi_r_coord=psi_rz_obj.coordinate('Device R')[0]
            psi_z_coord=psi_rz_obj.coordinate('Device z')[0]
            r_bdry=r_bdry_obj.data
            z_bdry=z_bdry_obj.data
            r_maxis=r_mag_axis.data
            z_maxis=z_mag_axis.data
        except:
            raise ValueError("The flux data cannot be found.")
        
        nlevel=101
        angle_values_spat_interpol=np.zeros([psi_t_coord.shape[0],gpi_r_coord.shape[1],gpi_r_coord.shape[2]])
        for index_t in range(psi_t_coord.shape[0]):
            poloidal_coord=np.zeros([0,3])
            #Get the contour plot paths of the constant psi surfaces
            psi_contour=plt.contour(psi_r_coord[index_t,:,:].transpose(),
                                    psi_z_coord[index_t,:,:].transpose(),
                                    psi_values[index_t,:,:], levels=nlevel)
            for index_collections in range(len(psi_contour.collections)): #Iterate through all the constant surfaces
                n_paths=len(psi_contour.collections[index_collections].get_paths()) #Get the actual paths, there might be more for one constant outside the separatrix.
                if n_paths > 0:
                    for index_paths in range(n_paths):
                        path=psi_contour.collections[index_collections].get_paths()[index_paths].vertices # Get the actual coordinates of the curve.   
                        #np.argmin(np.abs(path[:,1]))
                        #The following code calculates the angles. The full arclength is calculated along with the partial arclengths.
                        # The angle is then difined as the fraction of the arclength fraction and the entire arclength.
                        arclength=0.
                        current_path_angles=[]
                        arclength_0=np.sqrt((path[0,0]-path[-1,0])**2 +
                                            (path[0,1]-path[-1,1])**2)
                        for index_path_points in range(-1,len(path[:,0])-1):
                            arclength += np.sqrt((path[index_path_points+1,0]-path[index_path_points,0])**2 +
                                                 (path[index_path_points+1,1]-path[index_path_points,1])**2)
                            current_path_angles.append([path[index_path_points+1,0],
                                                        path[index_path_points+1,1],
                                                        arclength-arclength_0])
                        current_path_angles=np.asarray(current_path_angles)
                        min_ind=0
                        #The angles needs to be measured from the midplane. Hence a correction rotation subtracted from the angle.
                        for index_path_points in range(len(path[:,0])):
                            if ((np.abs(path[index_path_points,1]-z_maxis[index_t]) <
                                 np.abs(path[min_ind,1]-z_maxis[index_t]))   and 
                               (path[index_path_points,0] > r_maxis[index_t])):
                                min_ind=index_path_points
                        rotation=(current_path_angles[min_ind,2]/(arclength-arclength_0))*2.*np.pi
                        current_path_angles[:,2] = (current_path_angles[:,2]/(arclength-arclength_0))*2*np.pi
                        current_path_angles[:,2] = -1*(current_path_angles[:,2]-rotation)
                        #The angles are corrected to be between -Pi and +Pi
                        for i_angle in range(len(current_path_angles[:,2])):
                            if current_path_angles[i_angle,2] > np.pi:
                                current_path_angles[i_angle,2] -= 2*np.pi
                            if current_path_angles[i_angle,2] < -np.pi:
                                current_path_angles[i_angle,2] += 2*np.pi    
                        poloidal_coord=np.append(poloidal_coord,current_path_angles, axis=0)
                        
            points=poloidal_coord[:,(1,0)]
            values=poloidal_coord[:,2]                        
                        
            #No point of having coordinates outside the separatrix
            #The following needs to be done:
            #1. Calculate the angles at 90% of the separatrix
            boundary_fraction=0.90
            boundary_data=np.asarray([r_bdry[index_t],z_bdry[index_t]])
            points_at_fraction=np.asarray([(boundary_data[0,:]-r_maxis[index_t])*boundary_fraction+r_maxis[index_t],
                                     (boundary_data[1,:]-z_maxis[index_t])*boundary_fraction+z_maxis[index_t]])
            values_at_fraction=interpolate.griddata(points,values,(points_at_fraction[1,:],points_at_fraction[0,:]),method='cubic')
            values_at_fraction[np.isnan(values_at_fraction)]=0.
            #2. Redefine the poloidal coord vector to only have points inside the 95% of the separatrix
            bfrac_path=[]
            for index_bdry in range(len(points_at_fraction[0,:])):
                bfrac_path.append((points_at_fraction[0,index_bdry],points_at_fraction[1,index_bdry]))
            bfrac_path=pltPath.Path(bfrac_path)
            
            poloidal_coord_new=np.zeros([0,3])
            poloidal_coord_new=[[0,0,0]]
            for index_poloidal_coord in range(len(poloidal_coord[:,0])):
                if (bfrac_path.contains_point(poloidal_coord[index_poloidal_coord,(0,1)])):
                    poloidal_coord_new.append(poloidal_coord[index_poloidal_coord,:])

            poloidal_coord_new=np.asarray(poloidal_coord_new)[1:-1]
            if (len(poloidal_coord_new[:,0]) < 4):
                points=poloidal_coord[:,(1,0)]
                values=poloidal_coord[:,2]    
            else:      
            #3. Add points to poloidal coord vector with some resolution (close to the EFIT resolution)
                for index_expansion in range(21):
                    zoom=(index_expansion * 0.05) + 1.
                    expanded_points=np.asarray([(boundary_data[0,:]-r_maxis[index_t])*zoom+r_maxis[index_t],
                                                (boundary_data[1,:]-z_maxis[index_t])*zoom+z_maxis[index_t]])
                    
                    new_points=np.asarray([expanded_points[0,:],expanded_points[1,:],values_at_fraction])
                    poloidal_coord_new=np.append(poloidal_coord_new,new_points.transpose(), axis=0)
                points=poloidal_coord_new[:,(1,0)]
                values=poloidal_coord_new[:,2]
            #4. Do the interpolation with the redefined poloidal coordinates.
            #The angles are calculated for the GPI's coordinate frame.
            angle_values_spat_interpol[index_t,:,:]=interpolate.griddata(points,
                                                                         values,
                                                                         (gpi_r_coord[0,:,:].transpose(),gpi_z_coord[0,:,:].transpose()),
                                                                         method='cubic').transpose()
            if (options is not None and options['Debug']):
            #if True:
                plt.cla()
                plt.tricontourf(points[:,1],points[:,0],values, levels=51)
                #plt.colorbar()  
                plt.scatter(gpi_r_coord[0,:,:],gpi_z_coord[0,:,:])
                plt.scatter(r_bdry[index_t,:],z_bdry[index_t,:])
                plt.axis('equal')
                plt.show()
                plt.pause(1)
            
        #Temporal interpolation for the angle values
        angle_values_total_interpol=np.zeros(data_object.data.shape)
        for index_r in range(gpi_r_coord.shape[1]):
            for index_z in range(gpi_r_coord.shape[2]):
                angle_values_total_interpol[:,index_r,index_z]=np.interp(gpi_time,psi_t_coord,angle_values_spat_interpol[:,index_r,index_z])
        plt.cla()
        new_coordinates=(copy.deepcopy(flap.Coordinate(name='Flux theta',
                                       unit='rad',
                                       mode=flap.CoordinateMode(equidistant=False),
                                       values=angle_values_total_interpol,
                                       shape=angle_values_total_interpol.shape,
                                       dimension_list=[0,1,2]
                                       )))
        data_object.coordinates.append(new_coordinates)
        
    if ('Device theta' in coordinates):
        try:
            gpi_time=data_object.coordinate('Time')[0][:,0,0]
            gpi_r_coord=data_object.coordinate('Device R')[0]
            gpi_z_coord=data_object.coordinate('Device z')[0]
        except:
            raise ValueError('R,z or t coordinates are missing.')
        try:
            r_mag_axis=flap.get_data('NSTX_MDSPlus',
                                     name='\EFIT01::\RMAXIS',
                                     exp_id=data_object.exp_id,
                                     object_name='RMAXIS_FOR_COORD')
            z_mag_axis=flap.get_data('NSTX_MDSPlus',
                                     name='\EFIT01::\ZMAXIS',
                                     exp_id=data_object.exp_id,
                                     object_name='ZMAXIS_FOR_COORD')
        except:
            raise ValueError("The PSIRZ MDSPlus node cannot be reached.")
        try:
        #if True:
            t_maxis=r_mag_axis.coordinate('Time')[0]
            r_maxis=r_mag_axis.data
            z_maxis=z_mag_axis.data
        except:
            raise ValueError("The flux data cannot be found.")
        r_maxis_at_gpi_range=np.interp(gpi_time,t_maxis,r_maxis)
        z_maxis_at_gpi_range=np.interp(gpi_time,t_maxis,z_maxis)
        poloidal_angles=np.zeros(gpi_r_coord.shape)
        for index_t in range(gpi_time.shape[0]):
            y=gpi_z_coord[index_t,:,:]-z_maxis_at_gpi_range[index_t]
            x=gpi_r_coord[index_t,:,:]-r_maxis_at_gpi_range[index_t]
            poloidal_angles[index_t,:,:]=np.arctan(y/x)
        new_coordinates=(copy.deepcopy(flap.Coordinate(name='Device theta',
                               unit='rad',
                               mode=flap.CoordinateMode(equidistant=False),
                               values=poloidal_angles,
                               shape=poloidal_angles.shape,
                               dimension_list=[0,1,2]
                               )))
        data_object.coordinates.append(new_coordinates)
    return data_object

def register():
    flap.register_data_source('NSTX_GPI', get_data_func=nstx_gpi_get_data, add_coord_func=add_coordinate)
