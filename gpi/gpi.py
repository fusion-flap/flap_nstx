#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 14:28:29 2021

@author: mlampert
"""

#Core modules
import os
import copy

import flap
import flap_nstx
flap_nstx.register()
from flap_nstx.tools import Polygon

import flap_mdsplus
flap_mdsplus.register('NSTX_MDSPlus')

thisdir = os.path.dirname(os.path.realpath(__file__))
fn = os.path.join(thisdir,"../flap_nstx.cfg")
flap.config.read(file_name=fn)

class StructureSegmentation:
    def __init__(self):
        pass
        
    def _watershed_structure_finder(self):
        raise NotImplementedError('Not implemented...')
        
    def _contour_structure_finder(self):
        raise NotImplementedError('Not implemented...')
        
    @property
    def time():
        raise NotImplementedError('Not implemented...')
        
    @property
    def velocity(self):
        return self.__velocity
    
    @velocity.setter
    def velocity(self):
        raise NotImplementedError('Not implemented...')
        self.__velocity=None #_velocity
        
    @property
    def gaussian_size(self):
        return self.__gaussian_size
    
    @gaussian_size.setter
    def gaussian_size(self):
        self.__gaussian_size=None
        
    @property
    def angle(self):
        return self.__angle
    
    @angle.setter
    def angle(self):
        raise NotImplementedError('Not implemented...')
        self.__angle=None #_angle
    
    @property
    def elongation(self):
        return self.__elongation
    
    @elongation.setter
    def elongation(self):
        self.__elongation=None
    
    @property
    def angle_of_least_inertia(self):
        return self.__angle_of_least_inertia
    
    @angle_of_least_inertia.setter
    def angle_of_least_inertia(self):
        self.__angle_of_least_inertia=None
        
    @property
    def convexity(self):
        raise NotImplementedError('Not implemented...')
    
    @property
    def solidity(self):
        raise NotImplementedError('Not implemented...')
        
    @property
    def bending_energy(self):
        raise NotImplementedError('Not implemented...')    
    
    @property
    def curvature(self):
        raise NotImplementedError('Not implemented...')
    
    @property
    def structure_number(self):
        raise NotImplementedError('Not implemented...')

    @property
    def position(self):
        raise NotImplementedError('Not implemented...')
        
    @property
    def separatrix_distance(self):
        raise NotImplementedError('Not implemented...')
        
    @property
    def center_of_gravity(self):
        raise NotImplementedError('Not implemented...')
        
    @property
    def centroid(self):
        raise NotImplementedError('Not implemented...')
        
    @property
    def area(self):
        raise NotImplementedError('Not implemented...')

    @property
    def full_evolution(self):
        raise NotImplementedError('Not implemented...')
        
class SingleStructure(Polygon):
    def __init__():
        pass
    

class Velocity:
    def __init__(self):
        pass
    
class AngularVelocity:
    def __init__(self):
        pass

class Result:
    def __init__(self):
        pass

    def name(self):
        pass
    
    def unit(self):
        pass

        
class Gpi:
    def __init__(self,
                 exp_id=None,
                 time_range=None,
                 object_name='GPI',
                 pdf=False,
                 test=False):
        
        self.exp_id=exp_id
        self.time_range=time_range
        self.object_name=object_name
        self.pdf=pdf,
        self.test=test
        
    def load_data(self):
        self.data_object()
        
    @property
    def data_object(self):
        d = flap.get_data('NSTX GPI', 
                          exp_id=self.exp_id, 
                          name='', 
                          object_name=self.object_name)
        
        return d.slice_data(slicing={'Time':flap.Intervals(self.time_range[0],self.time_range[1])})
    
    @property
    def data(self):
        return self.data_object.data
    
    @property
    def time(self):
        try:
            return self.data_object.coordinate('Time')[0][:,0,0]
        except:
            return ValueError('Couldn\'t load time vector')
    
    def video(self):
        raise NotImplementedError('Not implemented...')
        
    @property
    def normalize(self,
                  normalization='roundtrip'):
        raise NotImplementedError('Not implemented...')
        
    @property
    def detrend(self, order=4):
        raise NotImplementedError('Not implemented...')
    
    @property
    def synthetic_data_object(self,
                              object_name=None):
        raise NotImplementedError('Not implemented...')
        
    def _calculate_sde_velocity(self):
        raise NotImplementedError('Not implemented...')
        
    def _calculate_tde_velocity(self):
        raise NotImplementedError('Not implemented...')

    def _calculate_stde_velocity(self):
        raise NotImplementedError('Not implemented...')
        
    def _calculate_angular_velocity(self):
        raise NotImplementedError('Not implemented...')
    
    def structure(self,
                  method='watershed'):
        normalized=self.normalize(input_object=self.data_object)
        normalized_detrended=self.detrend(input_object=normalized)
        
        return Structure(exp_id=self.exp_id,
                         input_object=normalized_detrended,
                         method=method)
    
    @property
    def sde_velocity(self):
        raise NotImplementedError('Not implemented...')
    
    @property    
    def tde_velocity(self):
        raise NotImplementedError('Not implemented...')
    
    @property
    def stde_velocity(self):
        raise NotImplementedError('Not implemented...')
        
    @property
    def angular_velocity(self):
        raise NotImplementedError('Not implemented...')
        