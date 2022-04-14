#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 00:48:01 2022

@author: mlampert
"""

import copy
import os
import pickle

#Flap imports
import flap
import flap_mdsplus
import flap_nstx
from flap_nstx.analysis import read_ahmed_fit_parameters,get_elms_with_thomson_profile
from flap_nstx.thomson import get_fit_nstx_thomson_profiles
from flap_nstx.tools import calculate_corr_acceptance_levels
flap_nstx.register()

#Setting up FLAP
flap_mdsplus.register('NSTX_MDSPlus')    
thisdir = os.path.dirname(os.path.realpath(__file__))
fn = os.path.join(thisdir,"../flap_nstx.cfg")
flap.config.read(file_name=fn) 
wd=flap.config.get_all_section('Module NSTX_GPI')['Working directory']


import numpy as np
import pandas
import scipy

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

plt.rc('font', family='serif', serif='Helvetica')
labelsize=8.
linewidth=0.5
major_ticksize=2.
minor_ticksize=0.
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
plt.rcParams['xtick.minor.size'] = minor_ticksize

plt.rcParams['ytick.labelsize'] = labelsize
plt.rcParams['ytick.major.width'] = linewidth
plt.rcParams['ytick.major.size'] = major_ticksize
plt.rcParams['ytick.minor.width'] = linewidth/2
plt.rcParams['ytick.minor.size'] = minor_ticksize
plt.rcParams['legend.fontsize'] = labelsize

def read_gpi_results(elm_window=400e-6,
                     elm_duration=100e-6,
                     correlation_threshold=0.6,
                     transformation=None, #['log','power','exp','diff',]
                     transformation_power=None,
                     recalc_gpi=False,
                     ):
    
    if transformation is not None:
        if transformation == 'power':
            str_add='_'+str(transformation_power)+'_power_trans'
        else:
            str_add='_'+transformation+'_trans'
    else:
        str_add=''
        
    gpi_rotation_results_db_file=wd+'/processed_data/gpi_rotation_profile_analysis'+str_add+'_elmwin_'+str(int(elm_window*1e6))+'us.pickle'
        
    if recalc_gpi or not os.path.exists(gpi_rotation_results_db_file):
        #Load and process the ELM database    
        database_file=wd+'/db/ELM_findings_mlampert_velocity_good.csv'
        db=pandas.read_csv(database_file, index_col=0)
        elm_index=list(db.index)
        
        
        dict_avg={'before':[],
                  'during':[],
                  'after':[],
                  'full':[],
                  }
        
        gpi_result_dict={'data':[],
                         'derived':{'avg':copy.deepcopy(dict_avg),
                                    'stddev':copy.deepcopy(dict_avg),
                                    'skewness':copy.deepcopy(dict_avg),
                                    'kurtosis':copy.deepcopy(dict_avg),
                                          
                                    'elm':[],
                                    'max':[],},
                         'label':'',
                         'unit':'',

                         }
                
        gpi_results={'data':{'Velocity ccf FLAP radial':copy.deepcopy(gpi_result_dict),
                             'Velocity ccf FLAP poloidal':copy.deepcopy(gpi_result_dict),
                             'Angular velocity ccf FLAP':copy.deepcopy(gpi_result_dict),
                             'Angular velocity ccf FLAP log':copy.deepcopy(gpi_result_dict),
                             'Expansion velocity ccf FLAP':copy.deepcopy(gpi_result_dict),
                             
                             'Acceleration ccf radial':copy.deepcopy(gpi_result_dict),
                             'Acceleration ccf poloidal':copy.deepcopy(gpi_result_dict),
                             'Size max radial':copy.deepcopy(gpi_result_dict),
                             'Size max poloidal':copy.deepcopy(gpi_result_dict),
                             'Position max radial':copy.deepcopy(gpi_result_dict),
                             'Position max poloidal':copy.deepcopy(gpi_result_dict),
                             'Separatrix dist max':copy.deepcopy(gpi_result_dict),
                             'Centroid max radial':copy.deepcopy(gpi_result_dict),
                             'Centroid max poloidal':copy.deepcopy(gpi_result_dict),
                             'COG max radial':copy.deepcopy(gpi_result_dict),
                             'COG max poloidal':copy.deepcopy(gpi_result_dict),
                             'Area max':copy.deepcopy(gpi_result_dict),
                             'Elongation max':copy.deepcopy(gpi_result_dict),
                             'Angle max':copy.deepcopy(gpi_result_dict),
                             'Str number':copy.deepcopy(gpi_result_dict),
                             },
                     'shot':[],
                     'elm_time':[],
                     'coord':{}
                     }
        
        
        rotation_keys=list(gpi_results['data'].keys())[0:5]
        translation_keys=list(gpi_results['data'].keys())[5:]
        
        """
        Filling up labels and units for GPI data
        """
        
        gpi_results['data']['Velocity ccf FLAP radial']['label']='$v_{rad}$'
        gpi_results['data']['Velocity ccf FLAP poloidal']['label']='$v_{pol}$'
        gpi_results['data']['Angular velocity ccf FLAP']['label']='$\omega$'
        gpi_results['data']['Angular velocity ccf FLAP log']['label']='$\omega$'
        gpi_results['data']['Expansion velocity ccf FLAP']['label']='exp. vel'
        
        gpi_results['data']['Acceleration ccf radial']['label']='$a_{rad}$'
        gpi_results['data']['Acceleration ccf radial']['label']='$a_{pol}$'
        gpi_results['data']['Size max radial']['label']='$d_{rad}$'
        gpi_results['data']['Size max poloidal']['label']='$d_{pol}$'
        gpi_results['data']['Position max radial']['label']='r'
        gpi_results['data']['Position max poloidal']['label']='z'
        gpi_results['data']['Separatrix dist max']['label']='$r-r_{sep}$'
        gpi_results['data']['Centroid max radial']['label']='$centroid_{rad}$'
        gpi_results['data']['Centroid max poloidal']['label']='$centroid_{pol}$'
        gpi_results['data']['COG max radial']['label']='$COG_{rad}$'
        gpi_results['data']['COG max poloidal']['label']='$COG_{pol}$'
        gpi_results['data']['Area max']['label']='area'
        gpi_results['data']['Elongation max']['label']='elong.'
        gpi_results['data']['Angle max']['label']='Angle'
        gpi_results['data']['Str number']['label']='Str. num.'
        
        
        gpi_results['data']['Velocity ccf FLAP radial']['unit']='km/s'
        gpi_results['data']['Velocity ccf FLAP poloidal']['unit']='km/s'
        gpi_results['data']['Angular velocity ccf FLAP']['unit']='krad/s'
        gpi_results['data']['Angular velocity ccf FLAP log']['unit']='krad/s'
        gpi_results['data']['Expansion velocity ccf FLAP']['unit']='[a.u.]'
        
        gpi_results['data']['Acceleration ccf radial']['unit']='$m/s^2'
        gpi_results['data']['Acceleration ccf radial']['unit']='$m/s^2'
        gpi_results['data']['Size max radial']['unit']='m'
        gpi_results['data']['Size max poloidal']['unit']='m'
        gpi_results['data']['Position max radial']['unit']='m'
        gpi_results['data']['Position max poloidal']['unit']='m'
        gpi_results['data']['Separatrix dist max']['unit']='m'
        gpi_results['data']['Centroid max radial']['unit']='m'
        gpi_results['data']['Centroid max poloidal']['unit']='m'
        gpi_results['data']['COG max radial']['unit']='m'
        gpi_results['data']['COG max poloidal']['unit']='m'
        gpi_results['data']['Area max']['unit']='$m^2$'
        gpi_results['data']['Elongation max']['unit']=''
        gpi_results['data']['Angle max']['unit']='deg'
        gpi_results['data']['Str number']['unit']=''
        
        coordinate_dict={'label':None,
                         'values':None,
                         'unit':None,
                         'index':None,
                         }
        
        gpi_results['coord']['time']=copy.deepcopy(coordinate_dict)
        gpi_results['coord']['time']['label']='$t-t_{ELM}$'
        gpi_results['coord']['time']['unit']='s'
        gpi_results['coord']['time']['index']=1
        gpi_results['coord']['time']['values']=((np.arange(2*elm_window/2.5e-6)-elm_window/2.5e-6)*2.5e-6)
        
        translation_results=pickle.load(open(wd+'/processed_data/all_results_file.pickle', 'rb'))
        trans_ind_shift=translation_results['Str number'].shape[0] // 2
        for elm_ind in elm_index:
            
            elm_time=db.loc[elm_ind]['ELM time']/1000.
            shot=int(db.loc[elm_ind]['Shot'])
            
            filename_rotation=flap_nstx.tools.filename(exp_id=shot,
                                                       working_directory=wd+'/processed_data',
                                                       time_range=[elm_time-600e-6,elm_time+600e-6],
                                                       purpose='ccf ang velocity',
                                                       comment='pfit_o4_fst_0.0')
            # str_add='_ns'
            # filename_translation=flap_nstx.analysis.filename(exp_id=shot,
            #                                                  working_directory=wd+'/processed_data',
            #                                                  time_range=[elm_time-2e-3,elm_time+2e-3],
            #                                                  comment='ccf_velocity_pfit_o4_fst_0.0_ns_nv',
            #                                                  extension='pickle')
            #grad.slice_data(slicing=time_slicing)
            status=db.loc[elm_ind]['OK/NOT OK']
            
            if status != 'NO':
                rotation_results=pickle.load(open(filename_rotation+'.pickle', 'rb'))

                corr_thres_ind=np.where(rotation_results['Correlation max'] < correlation_threshold)
                rotation_results['Velocity ccf FLAP'][corr_thres_ind,:]=[np.nan,np.nan]
                
                for key in ['Angular velocity ccf FLAP', 
                            'Angular velocity ccf FLAP log', 
                            'Expansion velocity ccf FLAP']:
                    rotation_results[key][corr_thres_ind]=np.nan
                
                coeff_r=np.asarray([3.7183594,-0.77821046,1402.8097])/1000. #The coordinates are in meters, the coefficients are in mm
                coeff_z=np.asarray([0.18090118,3.0657776,70.544312])/1000.  #The coordinates are in meters, the coefficients are in mm
                coeff_r_new=3./800.
                coeff_z_new=3./800.
                
                det=coeff_r[0]*coeff_z[1]-coeff_z[0]*coeff_r[1]
                
                rotation_results['Velocity ccf FLAP radial']=(coeff_r_new/det*(coeff_z[1]*rotation_results['Velocity ccf FLAP'][:,0]-
                                                                               coeff_r[1]*rotation_results['Velocity ccf FLAP'][:,1]))
                rotation_results['Velocity ccf FLAP poloidal']=(coeff_z_new/det*(-coeff_z[0]*rotation_results['Velocity ccf FLAP'][:,0]+
                                                                                 coeff_r[0]*rotation_results['Velocity ccf FLAP'][:,1]))
                
                for key in ['Acceleration ccf', 'Size max', 'Position max', 'Centroid max', 'COG max']:
                    translation_results[key+' radial']=translation_results[key][:,:,0]
                    translation_results[key+' poloidal']=translation_results[key][:,:,1]
                    
                rotation_results.pop('Velocity ccf FLAP')
                time=rotation_results['Time']
                
                elm_time_interval_ind=np.where(np.logical_and(time >= elm_time-elm_duration,
                                                              time <= elm_time+elm_duration))
                nwin=int(elm_window/2.5e-6)     #Length of the window for the calculation before and after the ELM
                n_elm=int(elm_duration/2.5e-6)  #Length of the ELM burst approx. 100us
                
                elm_time=(time[elm_time_interval_ind])[np.argmin(rotation_results['Frame similarity'][elm_time_interval_ind])]
                elm_time_ind=int(np.argmin(np.abs(time-elm_time)))
                    
                for key in gpi_results['data'].keys():
                    for key_avg in ['before', 'during', 'after', 'full']:
                        
                        if key_avg == 'before':   twin=[elm_time_ind-nwin,elm_time_ind]
                            
                        elif key_avg == 'during': twin=[elm_time_ind,elm_time_ind+n_elm]
                            
                        elif key_avg == 'after':  twin=[elm_time_ind+n_elm,elm_time_ind+nwin]
                            
                        elif key_avg == 'full':   twin=[elm_time_ind-nwin,elm_time_ind+nwin]
                        
                        if key in rotation_keys: 
                            ind_not_nan=np.logical_not(np.isnan(rotation_results[key][twin[0]:twin[1]]))
                            if transformation == 'diff':
                                data=np.gradient(rotation_results[key][twin[0]:twin[1]][ind_not_nan])
                            else:
                                data=rotation_results[key][twin[0]:twin[1]][ind_not_nan]
                            
                        elif key in translation_keys:
                            twin=[twin[0]-elm_time_ind+trans_ind_shift,
                                  twin[1]-elm_time_ind+trans_ind_shift]
                            ind_not_nan=np.logical_not(np.isnan(translation_results[key][twin[0]:twin[1],elm_ind]))
                            if transformation =='diff':
                                data=np.diff(translation_results[key][twin[0]:twin[1],elm_ind][ind_not_nan])
                            else:
                                data=translation_results[key][twin[0]:twin[1],elm_ind][ind_not_nan]
                        
                        for key_moment in ['avg', 'stddev', 'skewness', 'kurtosis']:
                            if key_moment == 'avg':
                                gpi_results['data'][key]['derived'][key_moment][key_avg].append(np.mean(data))
                                
                            elif key_moment == 'stddev':
                                gpi_results['data'][key]['derived'][key_moment][key_avg].append(np.sqrt(np.var(data)))
                                
                            elif key_moment == 'skewness':
                                gpi_results['data'][key]['derived'][key_moment][key_avg].append(scipy.stats.skew(data))
                                
                            elif key_moment == 'kurtosis':
                                gpi_results['data'][key]['derived'][key_moment][key_avg].append(scipy.stats.kurtosis(data))
                            
                    try:
                        if key in rotation_keys: 
                            data=rotation_results[key][elm_time_ind-n_elm:elm_time_ind+n_elm]
                        elif key in translation_keys:
                            data=translation_results[key][trans_ind_shift-n_elm:trans_ind_shift+n_elm,elm_ind]

                        ind_not_nan=np.logical_not(np.isnan(data))
                        max_ind=np.argmax(np.abs(data[ind_not_nan]))
                        gpi_results['data'][key]['derived']['max'].append(data[ind_not_nan][max_ind])
                    except:
                        gpi_results['data'][key]['derived']['max'].append(np.nan)
                        
                    if key in rotation_keys: 
                        gpi_results['data'][key]['derived']['elm'].append(rotation_results[key][elm_time_ind])
                        gpi_results['data'][key]['data'].append(rotation_results[key][elm_time_ind-nwin:elm_time_ind+nwin])
                    elif key in translation_keys:
                        gpi_results['data'][key]['derived']['elm'].append(translation_results[key][trans_ind_shift,elm_ind])
                        gpi_results['data'][key]['data'].append(translation_results[key][trans_ind_shift-nwin:trans_ind_shift+nwin,elm_ind])
                    else:
                        raise KeyError('Dumbass')

            gpi_results['shot'].append(shot)
            gpi_results['elm_time'].append(elm_time)
                    
        for key_param in gpi_results['data'].keys():
            for key_res in gpi_results['data'][key_param].keys():
                if key_res == 'data':
                    gpi_results['data'][key_param]['data']=np.asarray(gpi_results['data'][key_param]['data'])
                elif key_res == 'derived':
                    for key_avg in gpi_results['data'][key_param]['derived'].keys():
                        if type(gpi_results['data'][key_param]['derived'][key_avg]) is list:
                            gpi_results['data'][key_param]['derived'][key_avg]=np.asarray(gpi_results['data'][key_param]['derived'][key_avg])
                        elif type(gpi_results['data'][key_param]['derived'][key_avg]) is dict:
                            for key_range in gpi_results['data'][key_param]['derived'][key_avg].keys():
                                gpi_results['data'][key_param]['derived'][key_avg][key_range]=np.asarray(gpi_results['data'][key_param]['derived'][key_avg][key_range])
                            

        pickle.dump(gpi_results, open(gpi_rotation_results_db_file,'wb'))          
    else:
        gpi_results = pickle.load(open(gpi_rotation_results_db_file,'rb'))

    return gpi_results


def read_thomson_results(thomson_time_window=5e-3,
                         flux_range=[0.65,1.1],
                         recalc_thomson=False,
                         transformation=None, #['log','power','exp','diff',]
                         transformation_power=None
                         ): 
    
    """
    Thomson data handling
    """
    if transformation is not None:
        if transformation == 'power':
            str_add='_'+str(transformation_power)+'_power_trans'
        else:
            str_add='_'+transformation+'_trans'
    else:
        str_add=''
    thomson_results_db_file=wd+'/processed_data/thomson_profile_analysis_'+str(int(thomson_time_window*1e3))+'ms'+str_add+'.pickle'
    if recalc_thomson or not os.path.exists(thomson_results_db_file):
        database_file=wd+'/db/ELM_findings_mlampert_velocity_good.csv'
        db=pandas.read_csv(database_file, index_col=0)
        elm_index=list(db.index)
        
        value_error={'value':[],
                     'error':[],
                     'label':'',
                     'unit':'',
                     }
        
        profile_results={'max_gradient_mtanh':copy.deepcopy(value_error),
                         'max_gradient_tanh':copy.deepcopy(value_error),
                         'value_at_max_grad_mtanh':copy.deepcopy(value_error),
                         'value_at_max_grad_tanh':copy.deepcopy(value_error),
                         'global_gradient':copy.deepcopy(value_error),
                         'pedestal_height':copy.deepcopy(value_error),
                         'pedestal_width':copy.deepcopy(value_error),
                         'SOL_offset':copy.deepcopy(value_error),
                         'position_r':copy.deepcopy(value_error),
                         'position_psi':copy.deepcopy(value_error),
                         }
        
        thomson_results_solo={'data':{'Pressure':copy.deepcopy(profile_results),
                                      'Density':copy.deepcopy(profile_results),
                                      'Temperature':copy.deepcopy(profile_results)},
                              'shot':[],
                              'elm_time':[],
                              }
        thomson_results={'Ahmed':copy.deepcopy(thomson_results_solo),
                         'Mate':copy.deepcopy(thomson_results_solo),
                         'Comment':{'Ahmed':'',
                                    'Mate':'',
                                    }
                         }
        
        thomson_results['Comment']['Mate']='Results from Mate tanh and mtanh fitting. Number of Thomson points is defined.'
        thomson_results['Comment']['Ahmed']='Results from Ahmeds mtanh fitting, number of Thomson points is not defined, only the time_range is (20ms)'
        
        """
        Filling up labels and units for Thomson data
        """
        
        for key_prof in thomson_results['Mate'].keys():
            for key_solo in profile_results.keys():
                for key_user in ['Ahmed','Mate']:
                    if key_prof == 'Pressure':
                        if 'gradient' in key_solo:
                            thomson_results[key_user]['data'][key_prof][key_solo]['label']='$dp/d\psi$'
                            thomson_results[key_user]['data'][key_prof][key_solo]['unit']='kPa'
                        elif 'width' in key_solo:
                            thomson_results[key_user]['data'][key_prof][key_solo]['label']='psi'
                            thomson_results[key_user]['data'][key_prof][key_solo]['unit']=''
                        else:
                            thomson_results[key_user]['data'][key_prof][key_solo]['label']='$p_{e}$'
                            thomson_results[key_user]['data'][key_prof][key_solo]['unit']='kPa'
                            
                    if key_prof == 'Density':
                        if 'gradient' in key_solo:
                            thomson_results[key_user]['data'][key_prof][key_solo]['label']='$dn_{e}/d\psi$'
                            thomson_results[key_user]['data'][key_prof][key_solo]['unit']='$m^{-3}$'
                        else:
                            thomson_results[key_user]['data'][key_prof][key_solo]['label']='$n_{e}$ ML'
                            thomson_results[key_user]['data'][key_prof][key_solo]['unit']='$m^{-3}$'
                            
                    if key_prof == 'Temperature':
                        if 'gradient' in key_solo:
                            thomson_results[key_user]['data'][key_prof][key_solo]['label']='$dT_{e}/d\psi$'
                            thomson_results[key_user]['data'][key_prof][key_solo]['unit']='keV'
                        else:
                            thomson_results[key_user]['data'][key_prof][key_solo]['label']='$T_{e}$'
                            thomson_results[key_user]['data'][key_prof][key_solo]['unit']='keV'
                            
        """
        Filling up Ahmed's data
        """
        
        ahmed_db=read_ahmed_fit_parameters()
        index_correction=0
        
        for elm_ind in elm_index:
            
            elm_time=db.loc[elm_ind]['ELM time']/1000.
            shot=int(db.loc[elm_ind]['Shot'])
            
            key_correspondence={'max_gradient_mtanh':'max_grad',
                                'max_gradient_tanh':'max_grad_simple',
                                'value_at_max_grad_mtanh':'value_at_max_grad',
                                'value_at_max_grad_tanh':'value_at_max_grad_simple',
                                'global_gradient':'grad_glob',
                                'pedestal_height':'ped_height',
                                'pedestal_width':'ped_width',
                                'SOL_offset':'b',
                                'position_psi':'xo',
                                }
                
            if np.sum(np.where(ahmed_db['shot'] == shot)) != 0:
                
                while ahmed_db['shot'][elm_ind-index_correction] != shot:
                    index_correction+=1
                    
                db_ind=elm_ind-index_correction
                thomson_results['Ahmed']['shot'].append(ahmed_db['shot'][db_ind])
                thomson_results['Ahmed']['elm_time'].append(elm_time)
                for key in thomson_results['Ahmed']['data'].keys():
                    for key_param in key_correspondence.keys():
                        if transformation == 'log':
                            trans_value=np.log(np.abs(ahmed_db[key][key_correspondence[key_param]][db_ind]))
                        elif transformation == 'exp':
                            trans_value=np.exp(np.abs(ahmed_db[key][key_correspondence[key_param]][db_ind]))
                        elif transformation == 'power':
                            trans_value=np.power(np.abs(ahmed_db[key][key_correspondence[key_param]][db_ind]),transformation_power)
                        else:
                            trans_value=ahmed_db[key][key_correspondence[key_param]][db_ind]
                        thomson_results['Ahmed']['data'][key][key_param]['value'].append(trans_value)
                        thomson_results['Ahmed']['data'][key][key_param]['error'].append(0.)
            else:
                for key in thomson_results['Ahmed']['data'].keys():
                    for key_param in key_correspondence.keys():
                        thomson_results['Ahmed']['data'][key][key_param]['value'].append(np.nan)
                        thomson_results['Ahmed']['data'][key][key_param]['error'].append(np.nan)
        
        """
        Fitting my data
        """
        elms_with_thomson=get_elms_with_thomson_profile(before=True,
                                                        time_window=thomson_time_window,
                                                        reverse_db=True)
        
        bool_dict={'Temperature':[True,False,False],
                   'Density':[False,True,False],
                   'Pressure':[False,False,True]}
        
        key_correspondence={'max_gradient_mtanh':'Max gradient',
                            'max_gradient_tanh':'Max gradient',
                            'value_at_max_grad_mtanh':'Value at max',
                            'value_at_max_grad_tanh':'Value at max',
                            'global_gradient':'Global gradient',
                            'pedestal_height':'Height',
                            'pedestal_width':'Width',
                            'SOL_offset':'SOL offset',
                            'position_psi':'Position',
                            'position_r':'Position r',
                            }
        
        for elm_ind in elm_index:
            elm_time=db.loc[elm_ind]['ELM time']/1000.
            shot=int(db.loc[elm_ind]['Shot'])
            thomson_results['Mate']['shot'].append(shot)
            thomson_results['Mate']['elm_time'].append(elm_time)
            #Get the ELM events with thomson before and after the ELM times
            for key in thomson_results['Mate']['data'].keys():

                if (shot in elms_with_thomson['shot'] and 
                    elm_time in elms_with_thomson['elm_time']):
                    index_ts=elms_with_thomson['index_ts'][np.where(np.logical_and(elms_with_thomson['shot'] == shot,
                                                                                   elms_with_thomson['elm_time'] == elm_time))][0]

                    if True:
                        para_modtanh=get_fit_nstx_thomson_profiles(exp_id=shot,
                                                                   temperature=bool_dict[key][0],
                                                                   density=bool_dict[key][1],
                                                                   pressure=bool_dict[key][2],
                                                                   spline_data=False, 
                                                                   flux_coordinates=True, 
                                                                   flux_range=flux_range,
                                                                   modified_tanh=True,
                                                                   return_parameters=True)
                    
                        for key_param in key_correspondence.keys():
                            if 'mtanh' in key_param:
                                if transformation == 'log':
                                    trans_value=np.log(np.asarray(para_modtanh[key_correspondence[key_param]])[index_ts])
                                    trans_error=1/trans_value*np.asarray(para_modtanh['Error'][key_correspondence[key_param]])[index_ts]
                                elif transformation == 'exp':
                                    trans_value=np.exp(np.asarray(para_modtanh[key_correspondence[key_param]])[index_ts])
                                    trans_error=np.exp(trans_value)*np.asarray(para_modtanh['Error'][key_correspondence[key_param]])[index_ts]
                                elif transformation == 'power':
                                    trans_value=np.power(np.asarray(para_modtanh[key_correspondence[key_param]])[index_ts],transformation_power)
                                    trans_error=transformation_power*np.power(trans_value,transformation_power-1)*np.asarray(para_modtanh['Error'][key_correspondence[key_param]])[index_ts]
                                else:
                                    trans_value=np.asarray(para_modtanh[key_correspondence[key_param]])[index_ts]
                                    trans_error=np.asarray(para_modtanh['Error'][key_correspondence[key_param]])[index_ts]
                                if trans_value == 0:
                                    thomson_results['Mate']['data'][key][key_param]['value'].append(np.nan)
                                    thomson_results['Mate']['data'][key][key_param]['error'].append(np.nan)
                                else:
                                    thomson_results['Mate']['data'][key][key_param]['value'].append(trans_value)
                                    thomson_results['Mate']['data'][key][key_param]['error'].append(trans_error)

                               

                    if True:
                        para_tanh=get_fit_nstx_thomson_profiles(exp_id=shot,
                                                                   temperature=bool_dict[key][0],
                                                                   density=bool_dict[key][1],
                                                                   pressure=bool_dict[key][2],
                                                                   spline_data=False, 
                                                                   flux_coordinates=True, 
                                                                   flux_range=flux_range,
                                                                   modified_tanh=False,
                                                                   return_parameters=True)
                        for key_param in key_correspondence.keys():
                            if 'mtanh' not in key_param:
                                if transformation == 'log':
                                    trans_value=np.log(np.asarray(para_tanh[key_correspondence[key_param]])[index_ts])
                                    trans_error=1/trans_value*np.asarray(para_tanh['Error'][key_correspondence[key_param]])[index_ts]
                                elif transformation == 'exp':
                                    trans_value=np.exp(np.asarray(para_tanh[key_correspondence[key_param]])[index_ts])
                                    trans_error=np.exp(trans_value)*np.asarray(para_tanh['Error'][key_correspondence[key_param]])[index_ts]
                                elif transformation == 'power':
                                    trans_value=np.power(np.asarray(para_tanh[key_correspondence[key_param]])[index_ts],transformation_power)
                                    trans_error=transformation_power*np.power(trans_value,transformation_power-1)*np.asarray(para_tanh['Error'][key_correspondence[key_param]])[index_ts]
                                else:
                                    trans_value=np.asarray(para_tanh[key_correspondence[key_param]])[index_ts]
                                    trans_error=np.asarray(para_tanh['Error'][key_correspondence[key_param]])[index_ts]
                                    
                                if trans_value == 0:
                                    thomson_results['Mate']['data'][key][key_param]['value'].append(np.nan)
                                    thomson_results['Mate']['data'][key][key_param]['error'].append(np.nan)
                                else:
                                    thomson_results['Mate']['data'][key][key_param]['value'].append(trans_value)
                                    thomson_results['Mate']['data'][key][key_param]['error'].append(trans_error)
                                    
                else:
                    for key_param in key_correspondence.keys():
                        thomson_results['Mate']['data'][key][key_param]['value'].append(np.nan)
                        thomson_results['Mate']['data'][key][key_param]['error'].append(np.nan)
                    
        for key_user in ['Ahmed', 'Mate']:
            for key_prof in ['Density', 'Pressure', 'Temperature']:
                for key_param in thomson_results[key_user]['data'][key_prof].keys():
                    for key_valerr in ['value', 'error']:
                        thomson_results[key_user]['data'][key_prof][key_param][key_valerr]=np.asarray(thomson_results[key_user]['data'][key_prof][key_param][key_valerr])

            
        pickle.dump(thomson_results, open(thomson_results_db_file,'wb'))          
    else:
        thomson_results = pickle.load(open(thomson_results_db_file,'rb'))
    
    return thomson_results

def plot_gpi_profile_dependence_ultimate(pdf=False,
                                         plot=False,
                                         plot_trends=False,
                                         plot_error=False,
                                         plot_correlation_matrix=False,
                                         plot_correlation_evolution=False,
                                         plot_predictive_power_score=False,
                                         plot_gpi_correlation=False,
                                         corr_threshold=0.2,
                                         
                                         thomson_time_window=5e-3,
                                         flux_range=[0.65,1.1],
                                         elm_window=400e-6,
                                         elm_duration=100e-6,
                                         recalc_gpi=False,
                                         recalc_thomson=False,
                                         threshold_corr=False,
                                         skip_uninteresting=False,
                                         transformation=None,
                                         transformation_power=None,
                                         throw_outliers=False,
                                         return_results=False,
                                         ):
    
    gpi_results=read_gpi_results(elm_window=elm_window,
                                 elm_duration=elm_duration,
                                 correlation_threshold=0.6,
                                 transformation=transformation, #['log','power','exp','diff',]
                                 transformation_power=transformation_power,
                                 recalc_gpi=recalc_gpi,)
    
    thomson_results=read_thomson_results(thomson_time_window=thomson_time_window,
                                         flux_range=flux_range,
                                         recalc_thomson=recalc_thomson,
                                         transformation=transformation, #['log','power','exp','diff',]
                                         transformation_power=transformation_power)
                                            
    if skip_uninteresting:
        del gpi_results['data']['Angular velocity ccf FLAP log']
        del gpi_results['data']['Velocity ccf FLAP poloidal']
        for key in gpi_results['data'].keys():
            if not plot_trends:
                del gpi_results['data'][key]['derived']['stddev']
            del gpi_results['data'][key]['derived']['skewness']
            del gpi_results['data'][key]['derived']['kurtosis']
            for key2 in gpi_results['data'][key]['derived'].keys():
                try:
                    # del gpi_results[key][key2]['after']
                    del gpi_results['data'][key]['derived'][key2]['full']
                except:
                    pass
        for key_prof in ['Density', 'Pressure', 'Temperature']:
            del thomson_results['Mate']['data'][key_prof]['global_gradient']
            del thomson_results['Mate']['data'][key_prof]['value_at_max_grad_mtanh']
            del thomson_results['Mate']['data'][key_prof]['max_gradient_mtanh']
            
        momentum_keys=['avg']
        avg_keys=['before','during', 'after']
        users=['Mate']
    else:
        momentum_keys=['avg','stddev','skewness','kurtosis']
        avg_keys=['before','during','after','full']
        users=[
               #'Ahmed',
               'Mate',
               ]
    
    if not plot:
        import matplotlib
        matplotlib.use('agg')
        
    corr_accept=calculate_corr_acceptance_levels()
    std_multiplier=3.
        
    """
    Plot everything vs. everything
    """
    
    if plot_trends:

        if pdf:
            pdf_pages=PdfPages(wd+'/plots/gpi_vs_ts_trends_TS_'+str(int(thomson_time_window*1e3))+'ms_elmwin_'+str(int(elm_window*1e6))+'us.pdf')
            
        nwin=len(gpi_results[list(gpi_results['data'].keys())[0]]['data'][0,:])
        time_vec=(np.arange(nwin)-nwin//2)*2.5e-6
        color='tab:blue'
        
        for key_user in users:
            for key_prof in ['Density', 'Pressure', 'Temperature']:
                for key_ts_param in thomson_results[key_user]['data']['data'][key_prof].keys():
                    for key_gpi_param in gpi_results['data'].keys():
                        for key_gpi_res in gpi_results['data'][key_gpi_param]['derived'].keys():
                            if key_gpi_res in momentum_keys and key_gpi_res != 'stddev':
                                for key_range in avg_keys:
                                    signal_a=gpi_results['data'][key_gpi_param]['derived'][key_gpi_res][key_range]
                                    signal_b=thomson_results[key_user]['data'][key_prof][key_ts_param]['value']
                                    
                                    ind_not_nan=np.logical_not(np.logical_or(np.isnan(signal_a),np.isnan(signal_b)))
                                    
                                    signal_a=signal_a[ind_not_nan]
                                    signal_b=signal_b[ind_not_nan]
                                    if throw_outliers:
                                        avg_a=np.mean(signal_a)
                                        stddev_a=np.sqrt(np.var(signal_a))

                                        avg_b=np.mean(signal_b)
                                        stddev_b=np.sqrt(np.var(signal_b))
                                        
                                        ind_throw=np.where(np.logical_or((np.logical_or(signal_a < avg_a-std_multiplier*stddev_a, 
                                                                                        signal_a > avg_a+std_multiplier*stddev_a)),
                                                                  np.logical_or(signal_b < avg_b-std_multiplier*stddev_b, 
                                                                                        signal_b > avg_b+std_multiplier*stddev_b)))
                                                                  
                                        
                                        signal_a[ind_throw]=np.nan
                                        signal_b[ind_throw]=np.nan
                                    else:
                                        ind_throw=np.where(signal_a == signal_a)
                                        
                                    fix,ax=plt.subplots(figsize=(8.5/2.54,8.5/2.54/np.sqrt(2)))
                                    ax.scatter(signal_a, 
                                               signal_b,
                                               marker='o',
                                               color=color)
                                    
                                    if plot_error:
                                        ax.errorbar(signal_a,
                                                    signal_b,
                                                    xerr=gpi_results['data'][key_gpi_param]['derived']['stddev'][key_range][ind_not_nan],
                                                    yerr=thomson_results[key_user]['data'][key_prof][key_ts_param]['error'][ind_not_nan],
                                                    marker='o',
                                                    ls='',
                                                    color=color)
                                        
                                    ax.set_title(key_gpi_param+' '+key_gpi_res+' '+key_range+' vs.\n'+key_prof+' '+key_ts_param)
                                    ax.set_xlabel(key_gpi_param+' '+key_gpi_res+' '+key_range)
                                    ax.set_ylabel(thomson_results[key_user]['data'][key_prof][key_ts_param]['label']+' ['+
                                                  thomson_results[key_user]['data'][key_prof][key_ts_param]['unit']+']')
                                    
                                    plt.tight_layout(pad=0.1)
                                    pdf_pages.savefig()

                            elif key_gpi_res not in ['elm','max'] and key_gpi_res != 'stddev': 

                                signal_a=gpi_results['data'][key_gpi_param]['derived'][key_gpi_res]
                                signal_b=thomson_results[key_user]['data'][key_prof][key_ts_param]['value']
                                
                                ind_not_nan=np.logical_not(np.logical_or(np.isnan(signal_a),np.isnan(signal_b)))
                                
                                signal_a=signal_a[ind_not_nan]
                                signal_b=signal_b[ind_not_nan]
                                
                                if throw_outliers:
                                    avg_a=np.mean(signal_a)
                                    stddev_a=np.sqrt(np.var(signal_a))

                                    avg_b=np.mean(signal_b)
                                    stddev_b=np.sqrt(np.var(signal_b))
                                    ind_keep=np.where(np.logical_not(np.logical_or((np.logical_or(signal_a < avg_a-std_multiplier*stddev_a, 
                                                                                                   signal_a > avg_a+std_multiplier*stddev_a)),
                                                                                    np.logical_or(signal_b < avg_b-std_multiplier*stddev_b, 
                                                                                                  signal_b > avg_b+std_multiplier*stddev_b))))
                                                              
                                    
                                    signal_a=signal_a[ind_keep]
                                    signal_b=signal_b[ind_keep]

                                fix,ax=plt.subplots(figsize=(8.5/2.54,8.5/2.54/np.sqrt(2)))
                                ax.scatter(signal_a, 
                                           signal_b,
                                           marker='o',
                                           color=color)
                                if plot_error:
                                    ax.errorbar(signal_a,
                                                signal_b,
                                                xerr=gpi_results['data'][key_gpi_param]['derived']['stddev'][key_range][ind_not_nan],
                                                yerr=thomson_results[key_user]['data'][key_prof][key_ts_param]['error'][ind_not_nan],
                                                marker='o',
                                                ls='',
                                                color=color)
                                    
                                ax.set_title(key_gpi_param+' '+key_gpi_res+' '+key_range+' vs.\n'+key_prof+' '+key_ts_param)
                                ax.set_xlabel(key_gpi_param+' '+key_gpi_res+' '+key_range)
                                ax.set_ylabel(thomson_results[key_user]['data'][key_prof][key_ts_param]['label']+' ['+
                                              thomson_results[key_user]['data'][key_prof][key_ts_param]['unit']+']')
                                
                                plt.tight_layout(pad=0.1)
                                pdf_pages.savefig()
    """
    Calculate correlation matrices
    """
    if plot_correlation_matrix:
        if threshold_corr:
            str_add='_thresh'
        else:
            str_add=''
            
        if pdf:
            pdf_pages=PdfPages(wd+'/plots/pearson_matrix_ts_vs_gpi_TS_'+str(int(thomson_time_window*1e3))+'ms'+str_add+'.pdf')
            
        nwin=len(gpi_results['data'][list(gpi_results['data'].keys())[0]]['data'][0,:])
        time_vec=(np.arange(nwin)-nwin//2)*2.5e-6
        correlation_dict={}
        corr_thres_level_dict={}
        
        for key_user in users:
            correlation_dict[key_user]={}
            corr_thres_level_dict[key_user]={}
            for key_prof in ['Density', 'Pressure', 'Temperature']:
                correlation_dict[key_user][key_prof]={}
                corr_thres_level_dict[key_user][key_prof]={}
                for key_ts_param in thomson_results[key_user]['data'][key_prof].keys():
                    correlation_dict[key_user][key_prof][key_ts_param]={}
                    corr_thres_level_dict[key_user][key_prof][key_ts_param]={}
                    for key_gpi_param in gpi_results['data'].keys():
                        correlation_dict[key_user][key_prof][key_ts_param][key_gpi_param]={}
                        corr_thres_level_dict[key_user][key_prof][key_ts_param][key_gpi_param]={}
                        for key_gpi_res in gpi_results['data'][key_gpi_param]['derived'].keys():
                            if key_gpi_res in momentum_keys:
                                for key_range in avg_keys:
                                    print(key_gpi_res,key_range,key_gpi_param)
                                    signal_a=gpi_results['data'][key_gpi_param]['derived'][key_gpi_res][key_range]
                                    signal_b=thomson_results[key_user]['data'][key_prof][key_ts_param]['value']
                                    
                                    ind_not_nan=np.logical_not(np.logical_or(np.isnan(signal_a),np.isnan(signal_b)))
                                    
                                    signal_a=signal_a[ind_not_nan]
                                    signal_b=signal_b[ind_not_nan]
                                    
                                    if throw_outliers:
                                        avg_a=np.mean(signal_a)
                                        stddev_a=np.sqrt(np.var(signal_a))

                                        avg_b=np.mean(signal_b)
                                        stddev_b=np.sqrt(np.var(signal_b))
                                        ind_keep=np.where(np.logical_not(np.logical_or((np.logical_or(signal_a < avg_a-std_multiplier*stddev_a, 
                                                                                                       signal_a > avg_a+std_multiplier*stddev_a)),
                                                                                        np.logical_or(signal_b < avg_b-std_multiplier*stddev_b, 
                                                                                                      signal_b > avg_b+std_multiplier*stddev_b))))
                                                                  
                                        signal_a=signal_a[ind_keep]
                                        signal_b=signal_b[ind_keep]

                                    signal_a -= np.mean(signal_a)
                                    signal_b -= np.mean(signal_b)
                                    try:
                                        corr=np.sum(signal_a*signal_b)/np.sqrt((np.sum(signal_a**2))*(np.sum(signal_b**2)))
                                    except:
                                        print('Houston, we have a problem...')
                                    correlation_dict[key_user][key_prof][key_ts_param][key_gpi_param][key_gpi_res+' '+key_range]=corr
                                    corr_thres_level_dict[key_user][key_prof][key_ts_param][key_gpi_param][key_gpi_res+' '+key_range]=corr_accept['avg'][np.sum(ind_not_nan)]+corr_accept['stddev'][np.sum(ind_not_nan)]
                                    
                            elif key_gpi_res in ['elm','max']:
                                signal_a=gpi_results['data'][key_gpi_param]['derived'][key_gpi_res]
                                signal_b=thomson_results[key_user]['data'][key_prof][key_ts_param]['value']
                                
                                ind_not_nan=np.logical_not(np.logical_or(np.isnan(signal_a),np.isnan(signal_b)))
                                
                                signal_a=signal_a[ind_not_nan]
                                signal_b=signal_b[ind_not_nan]
                                
                                if throw_outliers:
                                    avg_a=np.mean(signal_a)
                                    stddev_a=np.sqrt(np.var(signal_a))

                                    avg_b=np.mean(signal_b)
                                    stddev_b=np.sqrt(np.var(signal_b))
                                    ind_keep=np.where(np.logical_not(np.logical_or((np.logical_or(signal_a < avg_a-std_multiplier*stddev_a, 
                                                                                                   signal_a > avg_a+std_multiplier*stddev_a)),
                                                                                    np.logical_or(signal_b < avg_b-std_multiplier*stddev_b, 
                                                                                                  signal_b > avg_b+std_multiplier*stddev_b))))
                                                              
                                    signal_a=signal_a[ind_keep]
                                    signal_b=signal_b[ind_keep]

                                signal_a -= np.mean(signal_a)
                                signal_b -= np.mean(signal_b)
                        
                                try:
                                    corr=np.sum(signal_a*signal_b)/np.sqrt((np.sum(signal_a**2))*(np.sum(signal_b**2)))
                                except:
                                    print('Houston, we have a problem...')
                                correlation_dict[key_user][key_prof][key_ts_param][key_gpi_param][key_gpi_res]=corr
                                corr_thres_level_dict[key_user][key_prof][key_ts_param][key_gpi_param][key_gpi_res]=corr_accept['avg'][np.sum(ind_not_nan)]+corr_accept['stddev'][np.sum(ind_not_nan)]
        
                                
        ts_label_num=(len(list(correlation_dict.keys()))*
                      len(list(correlation_dict[key_user].keys()))*
                      len(list(correlation_dict[key_user][key_prof].keys()))
                      )
        
        gpi_label_num=(len(list(correlation_dict[key_user][key_prof][key_ts_param].keys()))*
                       len(list(correlation_dict[key_user][key_prof][key_ts_param][key_gpi_param].keys()))
                      )
        correlation_matrix=np.zeros([ts_label_num,gpi_label_num])
        corr_thres_level_matrix=np.zeros([ts_label_num,gpi_label_num])
        ts_labels=[]
        
        i_ts=0
        for key_user in correlation_dict.keys():
            for key_prof in correlation_dict[key_user].keys():
                for key_ts_param in correlation_dict[key_user][key_prof].keys():
                    key_prof_new=key_prof.replace('Temperature','$T_e$')
                    key_prof_new=key_prof_new.replace('Density','$n_e$')
                    key_prof_new=key_prof_new.replace('Pressure','$p_{e}$')
                    
                    key_ts_param_new=key_ts_param.replace('max_gradient_tanh','$\\nabla_{max}$ '+key_prof_new)
                    key_ts_param_new=key_ts_param_new.replace('value_at_max_grad_tanh','at $\\nabla_{max}$ '+key_prof_new)
                    key_ts_param_new=key_ts_param_new.replace('pedestal_height','$h_{ped}$')
                    key_ts_param_new=key_ts_param_new.replace('pedestal_width','$w_{ped}$')
                    key_ts_param_new=key_ts_param_new.replace('SOL_offset',key_prof_new+'$_{,SOL}$')
                    
                    if ('nabla' in key_ts_param_new or 'SOL' in key_ts_param_new) and 'at' not in key_ts_param_new: key_prof_new=''
                    ts_labels.append(key_prof_new+' '+key_ts_param_new+' '+
                                     key_user.replace('Mate','').replace('Ahmed','AD'))
                    j_gpi=0
                    
                    gpi_labels=[]
                    for key_gpi_param in correlation_dict[key_user][key_prof][key_ts_param].keys():
                        for key_gpi_res in correlation_dict[key_user][key_prof][key_ts_param][key_gpi_param].keys():
                            correlation_matrix[i_ts,j_gpi]=correlation_dict[key_user][key_prof][key_ts_param][key_gpi_param][key_gpi_res]
                            corr_thres_level_matrix[i_ts,j_gpi]=corr_thres_level_dict[key_user][key_prof][key_ts_param][key_gpi_param][key_gpi_res]
                            j_gpi+=1
                            key_gpi_param_new=key_gpi_param.replace('Velocity ccf FLAP radial','$v_{rad}$')
                            key_gpi_param_new=key_gpi_param_new.replace('Velocity ccf FLAP poloidal','$v_{pol}$')
                            key_gpi_param_new=key_gpi_param_new.replace('Angular velocity ccf FLAP','$\omega$')
                            key_gpi_param_new=key_gpi_param_new.replace('Expansion velocity ccf FLAP','exp. vel.')
                            
                            gpi_labels.append(key_gpi_param_new+' '+key_gpi_res)
                    i_ts += 1

        
        if threshold_corr:
            for i_ts in range(ts_label_num):
                for j_gpi in range(gpi_label_num):
                    if abs(correlation_matrix[i_ts,j_gpi]) < corr_thres_level_matrix[i_ts,j_gpi]:
                        correlation_matrix[i_ts,j_gpi]=0.
                    
        plt.matshow(correlation_matrix, 
                    #fignum=fig, 
                    cmap='seismic',vmin=-1,vmax=1)

        plt.xticks(ticks=np.arange(correlation_matrix.shape[1]), labels=gpi_labels, rotation='vertical',
                                                )
            
        plt.yticks(np.arange(correlation_matrix.shape[0]), labels=ts_labels)
        plt.colorbar()
        ax = plt.gca()
        ax.set_title('Thomson profile parameter vs. GPI thresholded correlation')
        #plt.tight_layout(pad=0.1)
        ax.set_xticks(np.arange(0, gpi_label_num, 1))
        ax.set_yticks(np.arange(0, ts_label_num, 1))
        ax.set_xticks(np.arange(-.5, gpi_label_num, 1), minor=True)
        ax.set_yticks(np.arange(-.5, ts_label_num, 1), minor=True)

# Gridlines based on minor ticks
        ax.grid(which='minor', color='black', linestyle='-', linewidth=0.5)
        plt.show()
        if pdf:
            pdf_pages.savefig()
            
            
    if plot_predictive_power_score:
        import pandas as pd
        import ppscore as pps
        import seaborn as sns
        
        df = pd.DataFrame()
        
        corr_accept=calculate_corr_acceptance_levels()
        if threshold_corr:
            str_add='_thresh'
        else:
            str_add=''
            
        if pdf:
            pdf_pages=PdfPages(wd+'/plots/predictive_power_score_ts_vs_gpi_TS_'+str(int(thomson_time_window*1e3))+'ms'+str_add+'.pdf')
            
        nwin=len(gpi_results['data'][list(gpi_results['data'].keys())[0]]['data'][0,:])
        time_vec=(np.arange(nwin)-nwin//2)*2.5e-6
        correlation_dict={}
        corr_thres_level_dict={}
        column_list=[]
        for key_user in users:
            for key_prof in ['Density', 'Pressure', 'Temperature']:
                for key_ts_param in thomson_results[key_user]['data'][key_prof].keys():
                    df[key_prof+' '+key_ts_param]=thomson_results[key_user]['data'][key_prof][key_ts_param]['value']
                    column_list.append(key_prof+' '+key_ts_param)
        for key_gpi_param in gpi_results['data'].keys():
            for key_gpi_res in gpi_results['data'][key_gpi_param].keys():
                if key_gpi_res in momentum_keys:
                    for key_range in avg_keys:

                        df[key_gpi_param+' '+key_gpi_res+' '+key_range]=gpi_results['data'][key_gpi_param]['derived'][key_gpi_res][key_range]
                        
                elif key_gpi_res in ['elm','max']:
                    df[key_gpi_param+' '+key_gpi_res]=gpi_results['data'][key_gpi_param]['derived'][key_gpi_res]
        df.dropna(thresh=1)
        from scipy import stats
        df[(np.abs(stats.zscore(df)) < 3).all(axis=1)]
        
        matrix_df = pps.matrix(df)[['x', 'y', 'ppscore']].pivot(columns='x', index='y', values='ppscore')
        plt.figure()
        sns.heatmap(matrix_df, 
                    vmin=0, vmax=1, cmap="Blues", linewidths=0.5, 
                    #annot=True
                    )
        if pdf:
            pdf_pages.savefig()
                        
    """
    Plot correlation time series
    """
    
    if plot_correlation_evolution:
        import matplotlib
        matplotlib.use('agg')
        
        if pdf:
            pdf_pages=PdfPages(wd+'/plots/plot_correlation_evolution_advanced_TS_'+str(int(thomson_time_window*1e3))+'ms'+str_add+'.pdf')
        
        nwin=len(gpi_results['data'][list(gpi_results['data'].keys())[0]]['data'][0,:])
        time_vec=(np.arange(nwin)-nwin//2)*2.5e-6
        correlation_matrix={}

        for key_user in ['Ahmed', 
                         'Mate',
                         ]:
            for key in gpi_results['data'].keys():
                ndata=len(gpi_results['data'][key]['data'][0,:])
                correlation_matrix[key]={}
                correlation_matrix[key][key_user]={}
                
                for key_prof in ['Temperature','Density','Pressure']:
                    correlation_matrix[key][key_user][key_prof]={}
                    for key_param in thomson_results[key_user]['data'][key_prof].keys():
                        correlation_matrix[key][key_user][key_param]=np.zeros(ndata)
                        correlation_levels=np.zeros([ndata,2])
                        for ind_time in range(ndata):
                            signal_a=gpi_results['data'][key]['data'][:,ind_time]
                            signal_b=thomson_results[key_user]['data'][key_prof][key_param]['value']
                            ind_not_nan=np.logical_not(np.logical_or(np.isnan(signal_a),np.isnan(signal_b)))
                            
                            signal_a=signal_a[ind_not_nan]
                            signal_a -= np.mean(signal_a)
                            
                            signal_b=signal_b[ind_not_nan]
                            signal_b -= np.mean(signal_b)
                            
                            if throw_outliers:

                                avg_b=np.mean(signal_b)
                                stddev_b=np.sqrt(np.var(signal_b))
                                ind_keep=np.where(np.logical_not(np.logical_or((np.logical_or(signal_a < avg_a-std_multiplier*stddev_a, 
                                                                                               signal_a > avg_a+std_multiplier*stddev_a)),
                                                                                np.logical_or(signal_b < avg_b-std_multiplier*stddev_b, 
                                                                                              signal_b > avg_b+std_multiplier*stddev_b))))
                                                          
                                signal_a=signal_a[ind_keep]
                                signal_b=signal_b[ind_keep]

                            try:
                                corr=np.sum(signal_a*signal_b)/np.sqrt((np.sum(signal_a**2))*(np.sum(signal_b**2)))
                            except:
                                print('Houston, we have a problem...')
                            
                            correlation_matrix[key][key_user][key_param][ind_time]=corr
                            correlation_levels[ind_time,0]=corr_accept['avg'][np.sum(ind_not_nan)]
                            correlation_levels[ind_time,1]=corr_accept['stddev'][np.sum(ind_not_nan)]
                        
                        thres_plus_onesig=correlation_levels[:,0]+correlation_levels[:,1]    
                        
                        fig,ax=plt.subplots(figsize=(8.5/2.54,8.5/2.54/np.sqrt(2)))
                        ax.plot(time_vec/1e-6, correlation_matrix[key][key_user][key_param])
                        ax.plot(time_vec/1e-6, thres_plus_onesig, color='red')
                        ax.plot(time_vec/1e-6, -thres_plus_onesig, color='red')
                        ax.set_title(key.replace(' ccf FLAP','')+' vs. '+key_prof+' '+key_param+' '+key_user.replace('Mate','ML').replace('Ahmed','AD'))
                        ax.set_xlabel('Time [$\mu$s]')
                        ax.set_ylabel('Correlation')
                        plt.tight_layout(pad=0.1)
                        pdf_pages.savefig()
                        
        matplotlib.use('qt5agg')
        
    if plot_gpi_correlation:
        if not plot:
            matplotlib.use('agg')
        if pdf:
            pdf_pages=PdfPages(wd+'/plots/gpi_vs_ts_trends_TS_'+str(int(thomson_time_window*1e3))+'ms_elmwin_'+str(int(elm_window*1e6))+'us.pdf')
        corr_function_multi={}
        time_lag_max={}
        correlation_max={}
        time_lag_vec=((np.arange(161)-80)*2.5)
        for key1 in gpi_results['data'].keys():
            time_lag_max[key1]={}
            correlation_max[key1]={}
            corr_function_multi[key1]={}
            for key2 in gpi_results['data'].keys():
                time_lag_max[key1][key2]=[]
                correlation_max[key1][key2]=[]
                corr_function_multi[key1][key2]=[]
                fig,ax=plt.subplots(figsize=(8.5/2.54,8.5/2.54/np.sqrt(2)))
                for elm_ind in range(gpi_results['data'][key1]['data'].shape[0]):
                    signal1=gpi_results['data'][key1]['data'][elm_ind,:]
                    signal2=gpi_results['data'][key2]['data'][elm_ind,:]
                                        
                    ind_nan=(np.logical_or(np.isnan(signal1),np.isnan(signal2)))
                    signal1[ind_nan]=0
                    signal2[ind_nan]=0
                    
                    coord=[]
                    
                    coord.append(copy.deepcopy(flap.Coordinate(name='Sample',
                                                               unit='n.a.',
                                                               mode=flap.CoordinateMode(equidistant=True),
                                                               start=0,
                                                               step=1,
                                                               dimension_list=[0]
                                                               )))
                    
                    coord.append(copy.deepcopy(flap.Coordinate(name='Time',
                                                               unit='s',
                                                               mode=flap.CoordinateMode(equidistant=True),
                                                               start=-elm_window,
                                                               step=2.5e-6,
                                                               dimension_list=[0]
                                                               )))
                    
                    signal1_obj = flap.DataObject(data_array=signal1,
                                                 data_unit=flap.Unit(name='Signal',unit='Digit'),
                                                 coordinates=coord,
                                                 exp_id=elm_ind,
                                                 data_title='',
                                                 data_source="NSTX_GPI")
                    
                    signal2_obj = flap.DataObject(data_array=signal2,
                                                 data_unit=flap.Unit(name='Signal',unit='Digit'),
                                                 coordinates=coord,
                                                 exp_id=elm_ind,
                                                 data_title='',
                                                 data_source="NSTX_GPI")
                    
                    flap.add_data_object(signal1_obj, 'signal1')
                    flap.add_data_object(signal2_obj, 'signal2')

                    if True:
                        corr_function=flap.ccf('signal2', 'signal1', 
                                               coordinate=['Time'], 
                                               options={'Range':[-elm_window,elm_window], 
                                                        'Trend removal':None, 
                                                        'Normalize':True, 
                                                        'Correct ACF peak':False,
                                                        'Interval_n': 1}, 
                                 output_name='GPI_FRAME_12_CCF').data
                        flap.delete_data_object('signal1')
                        flap.delete_data_object('signal2')
                    else:   
                        corr_function=scipy.signal.correlate(signal1,signal2, mode='full', method='auto')
                        
                        auto_corr1=np.sqrt(np.sum(signal1**2))
                        auto_corr2=np.sqrt(np.sum(signal2**2))
                        
                        corr_function=corr_function/auto_corr1/auto_corr2
                        
                    if abs(max(corr_function)) > corr_threshold:
                        corr_function_multi[key1][key2].append(corr_function)
                        time_lag_max[key1][key2].append(time_lag_vec[np.argmax(np.abs(corr_function))])
                        correlation_max[key1][key2].append(corr_function[np.argmax(np.abs(corr_function))])
                #     ax.plot(time_lag_vec,corr_function)

                #     ax.set_title(key1.replace(' ccf FLAP','')+' vs. '+key2.replace(' ccf FLAP',''))
                #     ax.set_xlabel('Time lag [$\mu$s]')
                #     ax.set_ylabel('Correlation')
                #     plt.tight_layout(pad=0.1)
                # pdf_pages.savefig()
                
                corr_function_multi[key1][key2]=np.asarray(corr_function_multi[key1][key2])
                
                # if key1 == key2:
                #     return corr_function_multi
                #corr_function_multi[key1][key2][(np.isnan(corr_function_multi[key1][key2]))]=0.
                
                fig,ax=plt.subplots(figsize=(8.5/2.54,8.5/2.54/np.sqrt(2)))

                ax.plot(time_lag_vec, np.median(corr_function_multi[key1][key2],axis=0))
                ax.plot(time_lag_vec, np.percentile(corr_function_multi[key1][key2],10,axis=0), color='red')
                ax.plot(time_lag_vec, np.percentile(corr_function_multi[key1][key2],90,axis=0), color='red')
                # # print(np.mean(corr_function_multi, axis=0))
                # ax.errorbar(time_lag_vec,
                #             np.median(corr_function_multi,axis=0),
                #             yerr=[-,
                #                   np.percentile(corr_function_multi,90,axis=0)],
                #             marker='',
                #             ls='',
                #             color='tab:blue')

                ax.set_title(key1.replace(' ccf FLAP','')+' vs. '+key2.replace(' ccf FLAP',''))
                ax.set_xlabel('Time lag [$\mu$s]')
                ax.set_ylabel('Correlation')
                plt.tight_layout(pad=0.1)
                pdf_pages.savefig()
                
    if pdf:
        pdf_pages.close()
    if not plot:
        matplotlib.use('qt5agg')
        for key1 in time_lag_max.keys():
            for key2 in time_lag_max.keys():
                if key1 != key2:
                    print('time_lag_max',key1,key2,
                          np.median(time_lag_max[key1][key2]), 
                          np.mean(time_lag_max[key1][key2]),
                          np.percentile(time_lag_max[key1][key2],10),
                          np.percentile(time_lag_max[key1][key2],90))
                    print('corr',key1,key2,np.median(correlation_max[key1][key2]),
                          np.mean(correlation_max[key1][key2]),
                          np.percentile(correlation_max[key1][key2],10),
                          np.percentile(correlation_max[key1][key2],90))
    if return_results:
        return corr_function_multi, time_lag_max, correlation_max