#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 19:58:05 2020

@author: mlampert
"""

import os
import copy
import pickle

import pandas
import numpy as np

import matplotlib
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit, root

import flap
import flap_nstx

from flap_nstx.analysis import calculate_nstx_gpi_frame_by_frame_velocity, calculate_nstx_gpi_tde_velocity
from flap_nstx import flap_nstx_thomson_data, get_nstx_thomson_gradient, get_fit_nstx_thomson_profiles
from flap_nstx.publications import read_ahmed_fit_parameters, read_ahmed_edge_current, read_ahmed_matlab_file
from flap_nstx.analysis import thick_wire_estimation_numerical

thisdir = os.path.dirname(os.path.realpath(__file__))
fn = os.path.join(thisdir,'../flap_nstx.cfg')
flap.config.read(file_name=fn)
flap_nstx.register()

styled=True
if styled:
    plt.rc('font', family='serif', serif='Helvetica')
    labelsize=12.
    linewidth=0.5
    major_ticksize=6.
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
    plt.rcParams['xtick.minor.size'] = major_ticksize/2
    
    plt.rcParams['ytick.labelsize'] = labelsize
    plt.rcParams['ytick.major.width'] = linewidth
    plt.rcParams['ytick.major.size'] = major_ticksize
    plt.rcParams['ytick.minor.width'] = linewidth/2
    plt.rcParams['ytick.minor.size'] = major_ticksize/2
    plt.rcParams['legend.fontsize'] = labelsize
else:
    import matplotlib.style as pltstyle
    pltstyle.use('default')

def calculate_phase_diagram(averaging='before',
                            parameter='grad_glob',
                            normalized_structure=True,
                            normalized_velocity=True,
                            subtraction_order=4,
                            test=False,
                            recalc=True,
                            elm_window=500e-6,
                            elm_duration=100e-6,
                            correlation_threshold=0.6,
                            plot=False,
                            auto_x_range=True,
                            auto_y_range=True,
                            plot_error=True,
                            pdf=True,
                            dependence_error_threshold=0.5,
                            plot_only_good=False,
                            plot_linear_fit=False,
                            pressure_grad_range=None,               #Plot range for the pressure gradient
                            density_grad_range=None,                #Plot range for the density gradient
                            temperature_grad_range=None,            #Plot range for the temperature gradient (no outliers, no range)
                            ):
    
    
    coeff_r=np.asarray([3.7183594,-0.77821046,1402.8097])/1000. #The coordinates are in meters, the coefficients are in mm
    
    coeff_r=np.asarray([3.7183594,-0.77821046,1402.8097])/1000. #The coordinates are in meters, the coefficients are in mm
    coeff_z=np.asarray([0.18090118,3.0657776,70.544312])/1000.  #The coordinates are in meters, the coefficients are in mm
    coeff_r_new=3./800.
    coeff_z_new=3./800.
    
    
    flap.delete_data_object('*')
    wd=flap.config.get_all_section('Module NSTX_GPI')['Working directory']
    
    result_filename=wd+'/processed_data/'+'elm_profile_dependence'
        
    result_filename+='_'+averaging+'_avg'
    
        
    if normalized_structure:
        result_filename+='_ns'
    if normalized_velocity:
        result_filename+='_nv'
    result_filename+='_so'+str(subtraction_order)

    scaling_db_file=result_filename+'.pickle'
    db=read_ahmed_fit_parameters()
    
    X=[]
    Y=[]
    
    if not os.path.exists(scaling_db_file) or recalc:
        
        #Load and process the ELM database    
        database_file='/Users/mlampert/work/NSTX_workspace/db/ELM_findings_mlampert_velocity_good.csv'
        db=pandas.read_csv(database_file, index_col=0)
        elm_index=list(db.index)

        
        for elm_ind in elm_index:
            
            elm_time=db.loc[elm_ind]['ELM time']/1000.
            shot=int(db.loc[elm_ind]['Shot'])
            
            if normalized_velocity:
                if normalized_structure:
                    str_add='_ns'
                else:
                    str_add=''
                filename=flap_nstx.analysis.filename(exp_id=shot,
                                                     working_directory=wd+'/processed_data',
                                                     time_range=[elm_time-2e-3,elm_time+2e-3],
                                                     comment='ccf_velocity_pfit_o'+str(subtraction_order)+'_fst_0.0'+str_add+'_nv',
                                                     extension='pickle')
            else:
                filename=wd+'/processed_data/'+db.loc[elm_ind]['Filename']+'.pickle'
            #grad.slice_data(slicing=time_slicing)
            status=db.loc[elm_ind]['OK/NOT OK']
            
            if status != 'NO':
                
                velocity_results=pickle.load(open(filename, 'rb'))
                
                det=coeff_r[0]*coeff_z[1]-coeff_z[0]*coeff_r[1]
                
                for key in ['Velocity ccf','Velocity str max','Velocity str avg','Size max','Size avg']:
                    orig=copy.deepcopy(velocity_results[key])
                    velocity_results[key][:,0]=coeff_r_new/det*(coeff_z[1]*orig[:,0]-coeff_r[1]*orig[:,1])
                    velocity_results[key][:,1]=coeff_z_new/det*(-coeff_z[0]*orig[:,0]+coeff_r[0]*orig[:,1])
                    
                velocity_results['Elongation max'][:]=(velocity_results['Size max'][:,0]-velocity_results['Size max'][:,1])/(velocity_results['Size max'][:,0]+velocity_results['Size max'][:,1])
                velocity_results['Elongation avg'][:]=(velocity_results['Size avg'][:,0]-velocity_results['Size avg'][:,1])/(velocity_results['Size avg'][:,0]+velocity_results['Size avg'][:,1])
                
                velocity_results['Velocity ccf'][np.where(velocity_results['Correlation max'] < correlation_threshold),:]=[np.nan,np.nan]

                time=velocity_results['Time']
                
                elm_time_interval_ind=np.where(np.logical_and(time >= elm_time-elm_duration,
                                                              time <= elm_time+elm_duration))
                
                elm_time=(time[elm_time_interval_ind])[np.argmin(velocity_results['Frame similarity'][elm_time_interval_ind])]
                elm_time_ind=int(np.argmin(np.abs(time-elm_time)))
                
                try:
                    if velocity_results['Position max'][elm_time_ind,0] != 0.:
                        b_pol=flap.get_data('NSTX_MDSPlus',
                                            name='EFIT02::BZZ0',
                                            exp_id=shot,
                                            object_name='BZZ0').slice_data(slicing={'Time':elm_time, 
                                                                                    'Device R':velocity_results['Position max'][elm_time_ind,0]}).data
                except:
                    pass
                try:
                    if velocity_results['Position max'][elm_time_ind,0] != 0.:
                        b_tor=flap.get_data('NSTX_MDSPlus',
                                            name='EFIT02::BTZ0',
                                            exp_id=shot,
                                            object_name='BTZ0').slice_data(slicing={'Time':elm_time, 
                                                                                    'Device R':velocity_results['Position max'][elm_time_ind,0]}).data

                except:
                    pass
                try:
                    if velocity_results['Position max'][elm_time_ind,0] != 0.:
                        b_rad=flap.get_data('NSTX_MDSPlus',
                                            name='EFIT02::BRZ0',
                                            exp_id=shot,
                                            object_name='BRZ0').slice_data(slicing={'Time':elm_time, 
                                                                                    'Device R':velocity_results['Position max'][elm_time_ind,0]}).data
                except:
                    pass
                
                try:                
                    shot_inds=np.where(db['shot'] == shot)
                    ind_db=tuple(shot_inds[0][np.where(np.abs(db['time2'][shot_inds]/1000.-elm_time) == np.min(np.abs(db['time2'][shot_inds]/1000.-elm_time)))])
                    n_e=db['Density']['value_at_max_grad'][ind_db]*1e20 #Conversion to 1/m3
                    T_e=db['Temperature']['value_at_max_grad'][ind_db]*1e3*1.16e4 #Conversion to K
                    
                    k_x=2*np.pi/velocity_results['Size max'][elm_time_ind,0]
                    k_y=2*np.pi/velocity_results['Size max'][elm_time_ind,1]
                    R=velocity_results['Position max'][elm_time_ind,0]
                    L_N=velocity_results['Size max'][elm_time_ind,0]
                    m_e=9.1093835e-31
                    B=np.sqrt(b_pol**2+b_tor**2+b_rad**2)
                    q_e=1.6e-19
                    epsilon_0=8.854e-12
                    omega_pe=np.sqrt(n_e*q_e**2/m_e/epsilon_0)
                    v_e=velocity_results['Velocity ccf'][elm_time_ind,0]
                    
                    gamma=5/3.
                    Z=1.
                    k=1.38e-23                                                      #Boltzmann constant
                    m_i=2.014*1.66e-27                                               # Deuterium mass
                    c_s=np.sqrt(gamma*Z*k*(T_e)/m_i)
                    c=3e8
                    delta_e=c/omega_pe
                    
                    omega_A=B/np.sqrt(4*np.pi*1e-7*n_e*m_e)
                    omega_A_CGS=B/np.sqrt(4*np.pi*n_e*m_e)
                    omega_eta=v_e*(np.sqrt(k_x**2 + k_y**2)*delta_e)**2
                    
                    gamma_MHD=c_s**2/(R*L_N)
                    
                    X.append(omega_eta/omega_A_CGS)
                    Y.append(gamma_MHD**2/omega_A**2)
                except:
                    pass

    
    plt.figure()
    plt.scatter(np.abs(X),np.abs(Y))
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(min(X),max(X))
    plt.ylim(min(Y),max(Y))
    plt.title('Curvature vs. resistivity')
    plt.xlabel('omega_eta / omega_A')
    plt.ylabel('gamma_MHD^2 / omega_A^2')
    
def calculate_radial_acceleration_diagram(elm_window=500e-6,
                                          elm_duration=100e-6,
                                          correlation_threshold=0.6,
                                          
                                          elm_time_base='frame similarity',     #'radial acceleration', 'radial velocity', 'frame similarity'
                                          acceleration_base='numdev',           #numdev or linefit
                                          
                                          calculate_thick_wire=True,
                                          delta_b_threshold=1,
                                          
                                          plot=False,
                                          plot_velocity=False,
                                          
                                          auto_x_range=True,
                                          auto_y_range=True,
                                          plot_error=True,
                                          plot_clear_peak=False,
                                          calculate_acceleration=False,
                                          calculate_dependence=False,           #Calculate and plot dependence between the filament parameters and plasma parameters
                                          calculate_ion_drift_velocity=False,
                                          calculate_greenwald_fraction=False,
                                          calculate_collisionality=False,
                                          recalc=True,
                                          test=False,
                                          ):
        
    
    def linear_fit_function(x,a,b):
        return a*x+b
    def mtanh_function(x,a,b,c,h,x0):
        return (h+b)/2 + (h-b)/2*((1 - a*2*(x - x0)/c)*np.exp(-2*(x - x0)/c) - np.exp(2*(x-x0)/c))/(np.exp(2*(x - x0)/c) + np.exp(-2*(x - x0)/c))
    def mtanh_dx_function(x,a,b,c,h,x0):
        return ((h-b)*((4*a*(x-x0)+(-a-4)*c)*np.exp((4*(x-x0))/c)-a*c))/(c**2*(np.exp((4*(x-x0))/c)+1)**2)
    def mtanh_dxdx_function(x,a,b,c,h,x0):
        return -(8*(h-b)*np.exp((4*(x-x0))/c)*((2*a*x-2*a*x0+(-a-2)*c)*np.exp((4*(x-x0))/c)-2*a*x+2*a*x0+(2-a)*c))/(c**3*(np.exp((4*(x-x0))/c)+1)**3)
    
    if acceleration_base not in ['numdev','linefit']:
        raise ValueError('acceleration_base should be either "numdev" or "linefit"')
    
    
    coeff_r=np.asarray([3.7183594,-0.77821046,1402.8097])/1000. #The coordinates are in meters, the coefficients are in mm
    
    coeff_r=np.asarray([3.7183594,-0.77821046,1402.8097])/1000. #The coordinates are in meters, the coefficients are in mm
    coeff_z=np.asarray([0.18090118,3.0657776,70.544312])/1000.  #The coordinates are in meters, the coefficients are in mm
    coeff_r_new=3./800.
    coeff_z_new=3./800.
    
    sampling_time=2.5e-6
    
    gamma=5/3.
    Z=1.
    k_B=1.38e-23                                                      #Boltzmann constant
    
    mu0=4*np.pi*1e-7
    q_e=1.602e-19
    m_e=9.1e-31                                           # Deuterium mass
    m_i=2.014*1.66e-27   
    
    epsilon_0=8.85e-12
    
    flap.delete_data_object('*')
    wd=flap.config.get_all_section('Module NSTX_GPI')['Working directory']
    
    result_filename='radial_acceleration_analysis'
    
    if elm_time_base == 'frame similarity':
        result_filename+='_fs'
    elif elm_time_base == 'radial velocity':
        result_filename+='_rv'
    elif elm_time_base == 'radial acceleration':
        result_filename+='_ra'
    
    if calculate_thick_wire:
        result_filename+='_thick'
    
    result_filename+='_dblim_'+str(delta_b_threshold)
    
    db_nt=read_ahmed_fit_parameters()
    db_cur=read_ahmed_edge_current()
    db_data=read_ahmed_matlab_file()
    
    db_data_shotlist=[]
    for i_shotind in range(len(db_data)):
        db_data_shotlist.append(db_data[i_shotind]['shot'])
    db_data_shotlist=np.asarray(db_data_shotlist)
    
    db_data_timelist=[]
    for i_shotind in range(len(db_data)):
        db_data_timelist.append(db_data[i_shotind]['time2'])
    db_data_timelist=np.asarray(db_data_timelist)
    
    dependence_db={'Current':[],
                   'Pressure grad':[],
                   'Pressure grad own':[],
                   'Density grad':[],
                   'Density grad own':[],
                   'Temperature grad':[],
                   'Temperature grad own':[],
                   'Triangularity':[],
                   'Velocity ccf':[],
                   'Size max':[]}
    dependence_db_err=copy.deepcopy(dependence_db)
    
    ion_drift_velocity_db={'Drift vel':[],
                           'ExB vel':[],
                           'Error':[],
                           'Poloidal vel':[],
                           'Crossing psi':[],
                           }
    
    greenwald_limit_db={'nG':[],
                        'ne maxgrad':[],
                        'Greenwald fraction':[],
                        'Velocity ccf':[],
                        'Size max':[],
                        'Elongation max':[],
                        'Str number':[],}
    
    collisionality_db={'ei collision rate':[],
                       'Temperature':[],
                       'Collisionality':[],
                       'Velocity ccf':[],
                       'Size max':[],
                       'Elongation max':[],
                       'Str number':[],}
    
    a_curvature=[]
    a_curvature_error=[]
    
    a_thin_wire=[]
    a_thin_wire_error=[]
    
    a_measurement=[]
    a_measurement_error=[]
    
    append_index=0
    good_peak_indices=[]  
    lower_pol_vel=0.
    plt.figure()
    if not os.path.exists(wd+'/processed_data/'+result_filename+'.pickle') or recalc:
        if not recalc and not wd+'/processed_data/'+result_filename+'.pickle':
            print('Pickle file not found. Results will be recalculated!')
        if plot_velocity:
            matplotlib.use('agg')
            pdf_velocities=PdfPages(wd+'/plots/velocity_results_for_ELMs.pdf')    
            plt.figure()
        #Load and process the ELM database    
        database_file='/Users/mlampert/work/NSTX_workspace/db/ELM_findings_mlampert_velocity_good.csv'
        db=pandas.read_csv(database_file, index_col=0)
        elm_index=list(db.index)

        
        for elm_ind in elm_index:
            
            elm_time=db.loc[elm_ind]['ELM time']/1000.
            shot=int(db.loc[elm_ind]['Shot'])
            
            filename=flap_nstx.analysis.filename(exp_id=shot,
                                                 working_directory=wd+'/processed_data',
                                                 time_range=[elm_time-2e-3,elm_time+2e-3],
                                                 comment='ccf_velocity_pfit_o4_fst_0.0_ns_nv',
                                                 extension='pickle')
            #grad.slice_data(slicing=time_slicing)
            status=db.loc[elm_ind]['OK/NOT OK']
            radial_velocity_status=db.loc[elm_ind]['Radial velocity peak']
            radial_peak_status=db.loc[elm_ind]['Clear peak']
            
            if status != 'NO' and radial_velocity_status != 'No':
                
                velocity_results=pickle.load(open(filename, 'rb'))
                velocity_results['Separatrix dist avg']=np.zeros(velocity_results['Position avg'].shape[0])
                velocity_results['Separatrix dist max']=np.zeros(velocity_results['Position max'].shape[0])
                
                R_sep=flap.get_data('NSTX_MDSPlus',
                                    name='\EFIT01::\RBDRY',
                                    exp_id=shot,
                                    object_name='SEP R OBJ').slice_data(slicing={'Time':elm_time}).data
                z_sep=flap.get_data('NSTX_MDSPlus',
                                    name='\EFIT01::\ZBDRY',
                                    exp_id=shot,
                                    object_name='SEP Z OBJ').slice_data(slicing={'Time':elm_time}).data
                                    
                sep_GPI_ind=np.where(np.logical_and(R_sep > coeff_r[2],
                                                          np.logical_and(z_sep > coeff_z[2],
                                                                         z_sep < coeff_z[2]+79*coeff_z[0]+64*coeff_z[1])))
                try:
                    sep_GPI_ind=np.asarray(sep_GPI_ind[0])
                    sep_GPI_ind=np.insert(sep_GPI_ind,0,sep_GPI_ind[0]-1)
                    sep_GPI_ind=np.insert(sep_GPI_ind,len(sep_GPI_ind),sep_GPI_ind[-1]+1)            
                    z_sep_GPI=z_sep[(sep_GPI_ind)]
                    R_sep_GPI=R_sep[sep_GPI_ind]
                    GPI_z_vert=coeff_z[0]*np.arange(80)/80*64+coeff_z[1]*np.arange(80)+coeff_z[2]
                    R_sep_GPI_interp=np.interp(GPI_z_vert,np.flip(z_sep_GPI),np.flip(R_sep_GPI))
                    z_sep_GPI_interp=GPI_z_vert
                    for key in ['max','avg']:
                        for ind_time in range(len(velocity_results['Position '+key][:,0])):
                            velocity_results['Separatrix dist '+key][ind_time]=np.min(np.sqrt((velocity_results['Position '+key][ind_time,0]-R_sep_GPI_interp)**2 + 
                                                                                              (velocity_results['Position '+key][ind_time,1]-z_sep_GPI_interp)**2))
                            ind_z_min=np.argmin(np.abs(z_sep_GPI-velocity_results['Position '+key][ind_time,1]))
                            if z_sep_GPI[ind_z_min] >= velocity_results['Position '+key][ind_time,1]:
                                ind1=ind_z_min
                                ind2=ind_z_min+1
                            else:
                                ind1=ind_z_min-1
                                ind2=ind_z_min
                                
                            radial_distance=velocity_results['Position '+key][ind_time,0]-((velocity_results['Position '+key][ind_time,1]-z_sep_GPI[ind2])/(z_sep_GPI[ind1]-z_sep_GPI[ind2])*(R_sep_GPI[ind1]-R_sep_GPI[ind2])+R_sep_GPI[ind2])
                            if radial_distance < 0:
                                velocity_results['Separatrix dist '+key][ind_time]*=-1
                except:
                    pass
                
                det=coeff_r[0]*coeff_z[1]-coeff_z[0]*coeff_r[1]
                
                for key in ['Velocity ccf','Velocity str max','Velocity str avg','Size max','Size avg']:
                    orig=copy.deepcopy(velocity_results[key])
                    velocity_results[key][:,0]=coeff_r_new/det*(coeff_z[1]*orig[:,0]-coeff_r[1]*orig[:,1])
                    velocity_results[key][:,1]=coeff_z_new/det*(-coeff_z[0]*orig[:,0]+coeff_r[0]*orig[:,1])
                    
                velocity_results['Elongation max'][:]=(velocity_results['Size max'][:,0]-velocity_results['Size max'][:,1])/(velocity_results['Size max'][:,0]+velocity_results['Size max'][:,1])
                velocity_results['Elongation avg'][:]=(velocity_results['Size avg'][:,0]-velocity_results['Size avg'][:,1])/(velocity_results['Size avg'][:,0]+velocity_results['Size avg'][:,1])
                
                velocity_results['Velocity ccf'][np.where(velocity_results['Correlation max'] < correlation_threshold),:]=[np.nan,np.nan]
                
                #THIS NEEDS REVISION AS THE DATA IS TOO NOISY FOR DIFFERENTIAL CALCULATION
                
                velocity_results['Acceleration ccf']=copy.deepcopy(velocity_results['Velocity ccf'])
                velocity_results['Acceleration ccf'][1:,0]=(velocity_results['Velocity ccf'][1:,0]-velocity_results['Velocity ccf'][0:-1,0])/sampling_time
                velocity_results['Acceleration ccf'][1:,1]=(velocity_results['Velocity ccf'][1:,1]-velocity_results['Velocity ccf'][0:-1,1])/sampling_time
                time=velocity_results['Time']
                
                elm_time_interval_ind=np.where(np.logical_and(time >= elm_time-elm_duration,
                                                              time <= elm_time+elm_duration))
                elm_time=(time[elm_time_interval_ind])[np.argmin(velocity_results['Frame similarity'][elm_time_interval_ind])]                    
                elm_time_ind=int(np.argmin(np.abs(time-elm_time)))
                print(time[0], elm_time)
                
                if elm_time_base == 'radial velocity':
                    ind_notnan=np.logical_not(np.isnan(velocity_results['Velocity ccf'][elm_time_ind-40:elm_time_ind+40,0]))
                    elm_time=(time[elm_time_ind-40:elm_time_ind+40][ind_notnan])[np.argmax(velocity_results['Velocity ccf'][elm_time_ind-40:elm_time_ind+40,0][ind_notnan])]
                    elm_time_ind=int(np.argmin(np.abs(time-elm_time)))
                elif elm_time_base == 'radial acceleration':
                    ind_notnan=np.logical_not(np.isnan(velocity_results['Acceleration ccf'][elm_time_ind-40:elm_time_ind+40,0]))
                    elm_time=(time[elm_time_ind-40:elm_time_ind+40][ind_notnan])[np.argmax(velocity_results['Acceleration ccf'][elm_time_ind-40:elm_time_ind+40,0][ind_notnan])]
                    elm_time_ind=int(np.argmin(np.abs(time-elm_time)))
                else:
                    pass
                shot_inds=np.where(db_nt['shot'] == shot)
                ind_db=tuple(shot_inds[0][np.where(np.abs(db_nt['time2'][shot_inds]/1000.-elm_time) == np.min(np.abs(db_nt['time2'][shot_inds]/1000.-elm_time)))])
                
                shot_inds_2=np.where(db_data_shotlist == shot)
                ind_db_2=(shot_inds_2[0][np.where(np.abs(db_data_timelist[shot_inds_2]/1000.-elm_time) == np.min(np.abs(db_data_timelist[shot_inds_2]/1000.-elm_time)))])
                
                n_e=db_nt['Density']['value_at_max_grad'][ind_db]*1e20 #Conversion to 1/m3
                
                ind_error_ne=np.where(np.logical_and(db_data[ind_db_2[0]]['psi_n'] < 1.1,
                                                  db_data[ind_db_2[0]]['psi_n'] > 0.7))
                n_e_error=np.mean(db_data[ind_db_2[0]]['n_e_err_psi'][ind_error_ne])
                
                
                T_e=db_nt['Temperature']['value_at_max_grad'][ind_db]*1e3*1.16e4 #Conversion to K
                ind_error_te=np.where(np.logical_and(db_data[ind_db_2[0]]['psi_t'] < 1.1,
                                                     db_data[ind_db_2[0]]['psi_t'] > 0.7))
                T_e_error=np.mean(db_data[ind_db_2[0]]['t_e_err_psi'][ind_error_te])
                
                j_edge=db_cur['Current'][ind_db]*1e6
                j_edge_error=j_edge*0.10                                        #Suspected fitting error of the edge current.


                psi_n_e=db_data[ind_db_2[0]]['psi_n']
                dev_n_e=db_data[ind_db_2[0]]['dev_n']
                                
                a_param=db_nt['Density']['a'][ind_db]
                b_param=db_nt['Density']['b'][ind_db]
                c_param=db_nt['Density']['c'][ind_db]
                h_param=db_nt['Density']['h'][ind_db]
                x0_param=db_nt['Density']['xo'][ind_db]
                
                max_n_e_grad_psi=root(mtanh_dxdx_function, x0_param, args=(a_param,b_param,c_param,h_param,x0_param), method='hybr')                    

                sep_inner_dist_max_grad=np.interp(max_n_e_grad_psi.x, np.asarray(psi_n_e)[:,0], np.asarray(dev_n_e)[:,0])
                sep_inner_dist_max_grad=np.interp(x0_param, np.asarray(psi_n_e)[:,0], np.asarray(dev_n_e)[:,0])

#                plt.plot(dev_n_e[:,0], db_data[ind_db_2[0]]['n_e_dev'])
#                plt.pause(1.0)
                if sep_inner_dist_max_grad > 0.1:
                    sep_inner_dist_max_grad=np.nan
                
                n_i=n_e #Quasi neutrality
                n_i_error=n_e_error
                
                R=velocity_results['Position max'][elm_time_ind,0]
                R_error=3.75e-3
                
                c_s2=gamma*Z*k_B*(T_e)/m_i
                delta_b=np.mean(velocity_results['Size max'][elm_time_ind-4:elm_time_ind+1,0])
                delta_b_error=10e-3
                
                """
                HIJACKING INFO FOR DEPENDENCE CALCULATION
                """
                if calculate_dependence:
                    
                    dependence_db['Velocity ccf'].append(velocity_results['Velocity ccf'][elm_time_ind,:])
                    dependence_db_err['Velocity ccf'].append(np.asarray([3.75e-3/2.5e-6,3.75e-3/2.5e-6]))
                    
                    dependence_db['Size max'].append(velocity_results['Size max'][elm_time_ind,:])
                    dependence_db_err['Size max'].append([delta_b_error,delta_b_error])
                    
                    dependence_db['Current'].append(j_edge)
                    dependence_db_err['Current'].append(j_edge*0.1)
                    for key in ['Density','Temperature', 'Pressure']:
                        a_param=db_nt[key]['a'][ind_db]
                        b_param=db_nt[key]['b'][ind_db]
                        c_param=db_nt[key]['c'][ind_db]
                        h_param=db_nt[key]['h'][ind_db]
                        x0_param=db_nt[key]['xo'][ind_db]
                        if key== 'Density':
                            profile_bl=[True,False,False]
                        elif key == 'Temperature':
                            profile_bl=[False,True,False]
                        elif key == 'Pressure':
                            profile_bl=[False,False,True]
                        thomson_profiles=get_fit_nstx_thomson_profiles(exp_id=shot,
                                                                       density=profile_bl[0],
                                                                       temperature=profile_bl[1],
                                                                       pressure=profile_bl[2],
                                                                       flux_coordinates=True,
                                                                       return_parameters=True)
                        time_ind=np.argmin(np.abs(thomson_profiles['Time']-elm_time))
                        
                        dependence_db[key+' grad'].append(max(mtanh_dx_function(np.arange(0,1.4,0.01),a_param,b_param,c_param,h_param,x0_param)))
                        dependence_db_err[key+' grad'].append(thomson_profiles['Error']['Max gradient'][time_ind])  
                    
                """
                END OF HIJACKING
                """
                
                try:
                    if velocity_results['Position max'][elm_time_ind,0] != 0.:
                        b_pol=flap.get_data('NSTX_MDSPlus',
                                            name='\EFIT02::\BZZ0',
                                            exp_id=shot,
                                            object_name='BZZ0').slice_data(slicing={'Time':elm_time, 
                                                                                    'Device R':velocity_results['Position max'][elm_time_ind,0]}).data
                except:
                    pass
                try:
                    if velocity_results['Position max'][elm_time_ind,0] != 0.:
                        b_tor=flap.get_data('NSTX_MDSPlus',
                                            name='\EFIT02::\BTZ0',
                                            exp_id=shot,
                                            object_name='BTZ0').slice_data(slicing={'Time':elm_time, 
                                                                                    'Device R':velocity_results['Position max'][elm_time_ind,0]}).data

                except:
                    pass
                try:
                    if velocity_results['Position max'][elm_time_ind,0] != 0.:
                        b_rad=flap.get_data('NSTX_MDSPlus',
                                            name='\EFIT02::\BRZ0',
                                            exp_id=shot,
                                            object_name='BRZ0').slice_data(slicing={'Time':elm_time, 
                                                                                    'Device R':velocity_results['Position max'][elm_time_ind,0]}).data
                except:
                    pass
                
                B=np.sqrt(b_pol**2+b_tor**2+b_rad**2)
                
                omega_i=q_e*B/m_i

                """
                HIJACKING FOR ION DIAMAGNETIC DRIFT VELOCITY CALCULATION
                """
                if calculate_ion_drift_velocity:
                    
                    d=flap_nstx_thomson_data(exp_id=shot, pressure=True, add_flux_coordinates=True, output_name='pressure')
                    time_index=np.argmin(np.abs(d.coordinate('Time')[0][0,:]-elm_time))
                    dpsi_per_dr=((d.coordinate('Device R')[0][0:-1,0]-d.coordinate('Device R')[0][1:,0])/(d.coordinate('Flux r')[0][0:-1,time_index]-d.coordinate('Flux r')[0][1:,time_index]))[-10:]
                    
                    a_param=db_nt['Pressure']['a'][ind_db]
                    b_param=db_nt['Pressure']['b'][ind_db]
                    c_param=db_nt['Pressure']['c'][ind_db]
                    h_param=db_nt['Pressure']['h'][ind_db]
                    x0_param=db_nt['Pressure']['xo'][ind_db]
                    
                    psi_prof=d.coordinate('Flux r')[0][-10:,time_index]
                    grad_p=mtanh_dx_function(psi_prof,a_param,b_param,c_param,h_param,x0_param)*dpsi_per_dr
                    
                    a_param=db_nt['Density']['a'][ind_db]
                    b_param=db_nt['Density']['b'][ind_db]
                    c_param=db_nt['Density']['c'][ind_db]
                    h_param=db_nt['Density']['h'][ind_db]
                    x0_param=db_nt['Density']['xo'][ind_db]
                    
                    n_i_profile=mtanh_function(psi_prof,a_param,b_param,c_param,h_param,x0_param)*dpsi_per_dr*1e20
                    poloidal_velocity=velocity_results['Velocity ccf'][elm_time_ind,1]
                    drift_velocity=-grad_p /(q_e * n_i_profile * B)
                    
                    if -poloidal_velocity/1e3 < max(drift_velocity) and -poloidal_velocity > 0:
                        max_ind=np.argmax(drift_velocity)
                        drift_velocity_trunk=drift_velocity[max_ind:]
                        

                        sort_ind=np.argsort(drift_velocity_trunk)
                        psi_crossing=np.interp(-poloidal_velocity/1e3, drift_velocity_trunk[sort_ind], psi_prof[max_ind:][sort_ind])
#                        print(psi_prof[max_ind:], drift_velocity_trunk, psi_crossing, -poloidal_velocity/1e3)
#                        plt.plot(psi_prof[max_ind:], drift_velocity_trunk)
#                        plt.scatter(psi_crossing, -poloidal_velocity/1e3, color='red')
#                        plt.pause(0.3)
                    else:
                        psi_crossing=np.nan
                        
                    nanind=np.logical_not(np.isnan(velocity_results['Velocity ccf'][0:150,0]))
                    try:
                        exb_velocity=max((velocity_results['Velocity ccf'][0:150,0])[nanind])
                    except:
                        exb_velocity=0.
                    
                    ion_drift_velocity_db['Drift vel'].append(drift_velocity)
                    ion_drift_velocity_db['ExB vel'].append(exb_velocity)
                    ion_drift_velocity_db['Error'].append(0.)
                    ion_drift_velocity_db['Poloidal vel'].append(-poloidal_velocity)
                    ion_drift_velocity_db['Crossing psi'].append(psi_crossing)
                """
                HIJACKING GREENWALD FRACTION CALCULATION
                """
                if calculate_greenwald_fraction:
                    ip=flap.get_data('NSTX_MDSPlus',
                                     name='\EFIT02::\IPMEAS',
                                     exp_id=shot,
                                     object_name='IP').slice_data(slicing={'Time':elm_time}).data
                    a_minor=flap.get_data('NSTX_MDSPlus',
                                     name='\EFIT02::\AMINOR',
                                     exp_id=shot,
                                     object_name='IP').slice_data(slicing={'Time':elm_time}).data    
                    greenwald_fraction=n_e/(ip/(np.pi*a_minor)*1e14)
                    print('n_e/n_g=',greenwald_fraction)
                    greenwald_limit_db['nG']=ip/(np.pi*a_minor)*1e14
                    greenwald_limit_db['ne maxgrad']=n_e
                    greenwald_limit_db['Greenwald fraction'].append(greenwald_fraction)
                                        
                    greenwald_limit_db['Velocity ccf'].append(velocity_results['Velocity ccf'][elm_time_ind,:])
                    greenwald_limit_db['Size max'].append(velocity_results['Size max'][elm_time_ind,:])
                    greenwald_limit_db['Elongation max'].append(velocity_results['Elongation max'][elm_time_ind])
                    greenwald_limit_db['Str number'].append(velocity_results['Str number'][elm_time_ind])
                    
                """
                HIJACKING COLLISIONALITY CALCULATION
                """    
                if calculate_collisionality:

                    ln_LAMBDA=17
                    tau_ei= 12* np.pi**(1.5)/np.sqrt(2) * np.sqrt(m_e)*T_e**(1.5)*epsilon_0**2/(n_i*Z**2*q_e**4*np.log(ln_LAMBDA))
                    ei_collision_rate=1/tau_ei
                    
                    q95=flap.get_data('NSTX_MDSPlus',
                                     name='\EFIT02::\Q95',
                                     exp_id=shot,
                                     object_name='Q95').slice_data(slicing={'Time':elm_time}).data   
                                      
                    R_ped=flap.get_data('NSTX_MDSPlus',
                                     name='\EFIT02::\RMIDOUT',
                                     exp_id=shot,
                                     object_name='RMIDOUT').slice_data(slicing={'Time':elm_time}).data-0.02
                    
                    collisionality=ei_collision_rate/k_B*T_e/(q95*R_ped)
                    print('Coll=',collisionality)
                    collisionality_db['Temperature'].append(T_e)
                    collisionality_db['ei collision rate'].append(ei_collision_rate)
                    collisionality_db['Collisionality'].append(collisionality)
                    
                    collisionality_db['Velocity ccf'].append(velocity_results['Velocity ccf'][elm_time_ind,:])
                    collisionality_db['Size max'].append(velocity_results['Size max'][elm_time_ind,:])
                    collisionality_db['Elongation max'].append(velocity_results['Elongation max'][elm_time_ind])
                    collisionality_db['Str number'].append(velocity_results['Str number'][elm_time_ind])
                    
                """
                END OF HIJACKING
                """
                
                #print(np.sqrt(c_s2), velocity_results['Velocity ccf'][elm_time_ind,0]/np.sqrt(c_s2),velocity_results['Velocity ccf'][elm_time_ind,1]/np.sqrt(c_s2))
                
                #print(np.sqrt(c_s2)/omega_i, velocity_results['Size max'][elm_time_ind,0]/(np.sqrt(c_s2)/omega_i), velocity_results['Size max'][elm_time_ind,1]/(np.sqrt(c_s2)/omega_i))

                d=velocity_results['Separatrix dist max'][elm_time_ind]-sep_inner_dist_max_grad                
                d_error=18.75e-3 #EFIT + pixel + 5mm suspected max_grad_error
                
                if j_edge > 0 and n_i > 0 and d > 0 and d > delta_b_threshold*delta_b:
                    """
                    CURVATURE BASED CALCULATION
                    """
                    
                    a_curvature.append(c_s2/R)
                    a_curvature_error.append(c_s2*(T_e_error/T_e + R_error/R))
                    print(a_curvature_error[-1]/a_curvature[-1])
                    """
                    GPI BASED CALCULATION
                    """
                    
                    if acceleration_base == 'numdev':
                        a_measurement.append(velocity_results['Acceleration ccf'][elm_time_ind,0])
                        a_measurement_error.append(2*3.75e-3/6.25e-12)
                    elif acceleration_base == 'linefit':

                        x_data=velocity_results['Time'][elm_time_ind-4:elm_time_ind+1]
                        y_data=velocity_results['Velocity ccf'][elm_time_ind-4:elm_time_ind+1,0]
                        
                        y_data_error=copy.deepcopy(y_data)
                        y_data_error[:]=3.75e-3/2.5e-6
                        
                        notnan_ind=np.logical_not(np.isnan(y_data))
                        x_data=x_data[notnan_ind]
                        y_data=y_data[notnan_ind]
                        y_data_error=y_data_error[notnan_ind]
                        
                        p0=[np.mean((y_data[1:]-y_data[0:-1]))/sampling_time,
                            -velocity_results['Time'][elm_time_ind-4]*np.mean((y_data[1:]-y_data[0:-1]))/sampling_time]
                        popt, pcov = curve_fit(linear_fit_function, 
                                               x_data, 
                                               y_data, 
                                               sigma=y_data_error,
                                               p0=p0)
    
                        perr = np.sqrt(np.diag(pcov))
                        
                        a_measurement.append(popt[0])
                        a_measurement_error.append(perr[0])
                        
                    """
                    THIN/THICK WIRE BASED CALCULATION
                    """
                        
                    if calculate_thick_wire:
                        if d > 2*delta_b:
                            a_thin_wire.append(mu0*j_edge**2 * (np.pi*delta_b**2)/(2*np.pi*d*m_i*n_i))
                        else:
                        #if True:
                            print('Calculating thick wire estimate. Hang on! It takes a while.')
                            current_acceleration=thick_wire_estimation_numerical(j0=j_edge,
                                                                                 r=delta_b,
                                                                                 d=d,
                                                                                 n_mesh=30,
                                                                                 n_i=n_i,
                                                                                 acceleration=True,)
                            
                            a_thin_wire.append(copy.deepcopy(current_acceleration))
                    else:
                        a_thin_wire.append(mu0*j_edge**2 * (np.pi*delta_b**2)/(2*np.pi*d*m_i*n_i))
                        
                    j_error_term = mu0 * j_edge * delta_b**2 / (d * m_i * n_i) * j_edge_error
                    
                    delta_b_error_term= mu0 * j_edge**2 * delta_b / (d * m_i * n_i) * delta_b_error
                    
                    d_error_term= mu0 * j_edge**2 * delta_b**2 / (2 * d**2 * m_i * n_i) * d_error
                    
                    n_i_error_term= mu0 * j_edge**2 * delta_b**2 / (2 * d * m_i * n_i**2) * n_i_error
                    
                    a_thin_wire_error.append(copy.deepcopy(j_error_term) + 
                                             copy.deepcopy(delta_b_error_term) + 
                                             copy.deepcopy(d_error_term) +
                                             copy.deepcopy(n_i_error_term))
                    append_index+=1
                    if radial_peak_status == 'y':
                        good_peak_indices.append(append_index)

                            
                if plot_velocity:
                    tauwindow=100e-6
                    nwindow=int(tauwindow/sampling_time)
                    plt.figure()
                    plt.plot(velocity_results['Time'][elm_time_ind-nwindow:elm_time_ind+nwindow]*1e3,
                             velocity_results['Velocity ccf'][elm_time_ind-nwindow:elm_time_ind+nwindow,0])
                    plt.scatter(velocity_results['Time'][elm_time_ind]*1e3,
                                velocity_results['Velocity ccf'][elm_time_ind,0])
                    print(shot, 
                          elm_time, 
                          velocity_results['Time'][elm_time_ind]*1e3,
                          velocity_results['Velocity ccf'][elm_time_ind,0])
                    if acceleration_base == 'linefit':
                        plt.plot(x_data*1e3,
                                 popt[0]*x_data+popt[1])
                    plt.xlabel('Time [ms]')
                    plt.ylabel('Radial velocity [m/s]')
                    plt.title('Radial velocity of #'+str(shot)+' at '+str(elm_time))
                    pdf_velocities.savefig()
                    plt.close()
                    
                    plt.figure()
                    plt.plot(velocity_results['Time'][elm_time_ind-nwindow:elm_time_ind+nwindow]*1e3,
                             velocity_results['Acceleration ccf'][elm_time_ind-nwindow:elm_time_ind+nwindow,0])
                    plt.scatter(velocity_results['Time'][elm_time_ind]*1e3,
                                velocity_results['Acceleration ccf'][elm_time_ind,0])
                    if acceleration_base == 'linefit':
                        fit_acc=copy.deepcopy(x_data)
                        fit_acc[:]=popt[0]
                        plt.plot(x_data*1e3,
                                 fit_acc)
                    
                    plt.xlabel('Time [ms]')
                    plt.ylabel('Radial acceleration [m/s2]')
                    plt.title('Radial acceleration ccf of #'+str(shot)+' at '+str(elm_time))
                    pdf_velocities.savefig()
                    
                    plt.figure()
                    plt.plot(psi_n_e,mtanh_function(psi_n_e,a_param,b_param,c_param,h_param,x0_param))
                    plt.plot(psi_n_e,mtanh_dx_function(psi_n_e,a_param,b_param,c_param,h_param,x0_param), color='red')
                    plt.plot(psi_n_e,mtanh_dxdx_function(psi_n_e,a_param,b_param,c_param,h_param,x0_param), color='black')
                    plt.scatter(max_n_e_grad_psi.x,mtanh_dxdx_function(max_n_e_grad_psi.x,a_param,b_param,c_param,h_param,x0_param))
                    plt.xlabel('psi')
                    plt.ylabel('n_e,dxn_e,dxdxn_e')
                    pdf_velocities.savefig()
                    
                    plt.close()
                    
                    
        pickle.dump((a_measurement,a_measurement_error,
                     a_thin_wire,a_thin_wire_error,
                     a_curvature,a_curvature_error,
                     dependence_db,dependence_db_err,
                     ion_drift_velocity_db,
                     greenwald_limit_db,
                     good_peak_indices),open(wd+'/processed_data/'+result_filename+'.pickle','wb'))
    else:
        a_measurement,a_measurement_error,a_thin_wire,a_thin_wire_error,a_curvature,a_curvature_error,dependence_db,dependence_db_err,ion_drift_velocity_db,greenwald_limit_db,good_peak_indices=pickle.load(open(wd+'/processed_data/'+result_filename+'.pickle','rb'))
        print('Results are loaded.')
    
    pdfpages=PdfPages(wd+'/plots/'+result_filename+'.pdf')
    
    a_measurement=np.asarray(a_measurement)
    a_measurement_error=np.asarray(a_measurement_error)
    
    a_thin_wire=np.asarray(a_thin_wire)
    a_thin_wire_error=np.asarray(a_thin_wire_error)
    
    a_curvature=np.asarray(a_curvature)
    a_curvature_error=np.asarray(a_curvature_error)
    #good_peak_indices=tuple(good_peak_indices)
    
    """
    Linear plotting of a_thin_wire and a_curvature
    """
    if calculate_acceleration:
        plt.figure()
        plt.scatter(a_measurement, 
                    a_thin_wire, 
                    label='a_thin_wire')
        plt.errorbar(a_measurement, 
                     a_thin_wire, 
                     xerr=a_measurement_error,
                     yerr=a_thin_wire_error,
                     ls='none')
        plt.legend()
        plt.title('Measured acceleration vs. a_thin_wire')
        plt.xlabel('a_meas [m/s2]')
        plt.ylabel('a_thin_wire [m/s2]')
        pdfpages.savefig()
        
        plt.figure()
        plt.scatter(a_measurement, 
                    a_curvature, 
                    label='a_curv')
        plt.errorbar(a_measurement, 
                     a_curvature, 
                     xerr=a_measurement_error,
                     yerr=a_curvature_error,
                     ls='none')       
        plt.legend()
        plt.title('Measured acceleration vs. a_curv')
        plt.xlabel('a_meas [m/s2]')
        plt.ylabel('a_curv [m/s2]')
        pdfpages.savefig()
        
            
        """
        Linear plotting of a_thin_wire_peak and a_curvature_peak
        """
        if plot_clear_peak:
            plt.figure()
            plt.scatter(a_measurement[good_peak_indices], 
                        a_thin_wire[good_peak_indices], 
                        label='a_thin_peak')
            plt.errorbar(a_measurement[good_peak_indices], 
                         a_thin_wire[good_peak_indices], 
                         xerr=a_measurement_error[good_peak_indices],
                         yerr=a_thin_wire_error[good_peak_indices],
                         ls='none')
            plt.legend()
            plt.title('Measured acceleration vs. a_thin_peak')
            plt.xlabel('a_meas [m/s2]')
            plt.ylabel('a_thin_peak [m/s2]')
            pdfpages.savefig()
            
        if plot_clear_peak:
            plt.figure()
            plt.scatter(a_measurement[good_peak_indices], 
                        a_curvature[good_peak_indices], 
                        label='a_curv_peak')
            plt.errorbar(a_measurement[good_peak_indices], 
                         a_curvature[good_peak_indices], 
                         xerr=a_measurement_error[good_peak_indices],
                         yerr=a_curvature_error[good_peak_indices],
                         ls='none')       
            plt.legend()
            plt.title('Measured acceleration vs. a_curv_peak')
            plt.xlabel('a_meas [m/s2]')
            plt.ylabel('a_curv_peak [m/s2]')
            pdfpages.savefig()
        
        """
        Logarithmic plotting of a_thin_wire and a_curvature
        """
        
        plt.figure()
        plt.scatter(a_measurement, 
                    a_thin_wire, 
                    label='a_thin')
        plt.errorbar(a_measurement, 
                     a_thin_wire, 
                     xerr=a_measurement_error,
                     yerr=a_thin_wire_error,
                     ls='none')
        plt.legend()
        plt.title('Measured acceleration vs. a_thin_wire')
        plt.xlabel('a_meas [m/s2]')
        plt.ylabel('a_thin_wire [m/s2]')
        plt.xlim(3e7,2e9)
        #plt.ylim([1e7,1e12])
        plt.xscale('log')
        plt.yscale('log')
        pdfpages.savefig()
        
        plt.figure()
        plt.scatter(a_measurement, 
                    a_curvature, 
                    label='a_curv')
        plt.errorbar(a_measurement, 
                     a_curvature, 
                     xerr=a_measurement_error,
                     yerr=a_curvature_error,
                     ls='none')
        plt.legend()
        plt.title('Measured acceleration vs. a_curv')
        plt.xlabel('a_meas [m/s2]')
        plt.ylabel('a_curv [m/s2]')
        plt.xlim(3e7,2e9)
        #plt.ylim([1e7,1e12])
        plt.xscale('log')
        plt.yscale('log')
        pdfpages.savefig()
        
        """
        Logarithmic plotting of a_thin_wire_peak and a_curvature_peak
        """
        if plot_clear_peak:
            plt.figure()
            plt.scatter(a_measurement[good_peak_indices],
                        a_curvature[good_peak_indices], 
                        label='a_curv_peak', 
                        color='red')
            plt.errorbar(a_measurement[good_peak_indices], 
                         a_curvature[good_peak_indices], 
                         xerr=a_measurement_error[good_peak_indices],
                         yerr=a_curvature_error[good_peak_indices],
                         color='red',
                         ls='none')
                    
            plt.legend()
            plt.title('Measured acceleration vs. a_curv_peak')
            plt.xlabel('a_meas [m/s2]')
            plt.ylabel('a_curv [m/s2]')
            plt.xlim(3e7,2e9)
            #plt.ylim([1e7,1e12])
            plt.xscale('log')
            plt.yscale('log')
            pdfpages.savefig()
        
            plt.figure()
            plt.scatter(a_measurement[good_peak_indices],
                        a_thin_wire[good_peak_indices], 
                        label='a_thin_peak', 
                        color='black')
            plt.errorbar(a_measurement[good_peak_indices], 
                         a_thin_wire[good_peak_indices], 
                         xerr=a_measurement_error[good_peak_indices],
                         yerr=a_thin_wire_error[good_peak_indices],
                         color='black',
                         ls='none')
                
            plt.legend()
            plt.title('Measured acceleration vs. a_thin_wire')
            plt.xlabel('a_meas [m/s2]')
            plt.ylabel('a_thin_wire [m/s2]')
            plt.xlim(3e7,2e9)
            #plt.ylim([1e7,1e12])
            plt.xscale('log')
            plt.yscale('log')
            pdfpages.savefig()
        
        
        f_j=np.sqrt(a_measurement/a_thin_wire)
        f_j_error=(0.5 / np.sqrt(a_measurement*a_thin_wire) * a_measurement_error +
                   0.5 * np.sqrt(a_measurement/a_thin_wire**3) * a_thin_wire_error)
        
        f_b=a_measurement/a_curvature
        f_b_error=a_measurement_error/a_curvature+a_measurement/a_curvature**2*a_curvature_error
        
        """
        Linear plotting of f_J and f_b
        """
        plt.figure()
        plt.scatter(a_measurement, 
                    f_j, 
                    label='f_J')
        plt.errorbar(a_measurement, 
                     f_j, 
                     xerr=a_measurement_error,
                     yerr=f_j_error,
                     ls='none')
        plt.legend()
    #    plt.xlim(0,0.4e10)
        #plt.ylim([0.001,10])
        plt.title('Measured acceleration vs. f_J')
        plt.xlabel('a_meas [m/s2]')
        plt.ylabel('f_J')
        pdfpages.savefig()
        
        plt.figure()
        plt.scatter(a_measurement, 
                    f_b, 
                    label='f_b')
        plt.errorbar(a_measurement, 
                     f_b, 
                     xerr=a_measurement_error,
                     yerr=f_b_error,
                     ls='none')
        plt.legend()
    #    plt.xlim(0,0.4e10)
        #plt.ylim([0.001,10])
        plt.title('Measured acceleration vs. f_b')
        plt.xlabel('a_meas [m/s2]')
        plt.ylabel('f_b')
        pdfpages.savefig()
        
        """
        Linear plotting of f_J_peak and f_b_peak
        """
        if plot_clear_peak:
            plt.figure()
            plt.scatter(a_measurement[good_peak_indices],
                        f_j[good_peak_indices], 
                        label='f_J_peak', 
                        color='red')
            plt.errorbar(a_measurement[good_peak_indices], 
                         f_j[good_peak_indices], 
                         xerr=a_measurement_error[good_peak_indices],
                         yerr=f_j_error[good_peak_indices],
                         color='red',
                         ls='none')
            plt.legend()
        #    plt.xlim(0,0.4e10)
            #plt.ylim([0.001,10])
            plt.title('Measured acceleration vs. f_J_peak')
            plt.xlabel('a_meas [m/s2]')
            plt.ylabel('f_J_peak')
            pdfpages.savefig()
        if plot_clear_peak:
            plt.figure()
            plt.scatter(a_measurement[good_peak_indices],
                        f_b[good_peak_indices], 
                        label='f_b_peak', 
                        color='black')
            plt.errorbar(a_measurement[good_peak_indices], 
                         f_b[good_peak_indices], 
                         xerr=a_measurement_error[good_peak_indices],
                         yerr=f_b_error[good_peak_indices],
                         color='black',
                         ls='none')
            plt.legend()
        #    plt.xlim(0,0.4e10)
            #plt.ylim([0.001,10])
            plt.title('Measured acceleration vs. f_b_peak')
            plt.xlabel('a_meas [m/s2]')
            plt.ylabel('f_b_peak')
            pdfpages.savefig()
           
        """
        Logarithmic plotting of f_J and f_b
        """
        
        plt.figure()
        plt.scatter(a_measurement, 
                    f_j, 
                    label='f_J')
        plt.errorbar(a_measurement, 
                     f_j, 
                     xerr=a_measurement_error,
                     yerr=f_j_error,
                     ls='none')
        plt.legend()
    #    plt.xlim(0,0.4e10)
        #plt.ylim([0.001,10])
        plt.title('Measured acceleration vs. f_J')
        plt.xlabel('a_meas [m/s2]')
        plt.ylabel('f_J')
        plt.xscale('log')
        plt.yscale('log')
        pdfpages.savefig()
        
        plt.figure()
        plt.scatter(a_measurement, 
                    f_b, 
                    label='f_b')
        plt.errorbar(a_measurement, 
                     f_b, 
                     xerr=a_measurement_error,
                     yerr=f_b_error,
                     ls='none')
        plt.legend()
    #    plt.xlim(0,0.4e10)
        #plt.ylim([0.001,10])
        plt.title('Measured acceleration vs. f_b')
        plt.xlabel('a_meas [m/s2]')
        plt.ylabel('f_b')
        plt.xscale('log')
        plt.yscale('log')
        pdfpages.savefig()
        
        """
        Plotting of a_meas vs. a_curv + a_J
        """
    
        plt.figure()
        plt.scatter(a_measurement, 
                    a_thin_wire+a_curvature)
        plt.errorbar(a_measurement, 
                     a_thin_wire+a_curvature, 
                     xerr=a_measurement_error,
                     yerr=a_thin_wire_error+a_curvature_error,
                     ls='none')
        plt.plot(np.arange(13)*1e9,np.arange(13)*1e9, color='red')
    #    plt.xlim(0,0.4e10)
        #plt.ylim([0.001,10])
        plt.title('Measured acceleration vs. modelled')
        plt.xlabel('a_meas [m/s2]')
        plt.ylabel('a_thin + a_curv')
        plt.xscale('log')
        plt.yscale('log')
        pdfpages.savefig()
        
        
        """
        Logarithmic plotting of f_J_peak and f_b_peak
        """
        if plot_clear_peak:
            plt.figure()
            plt.scatter(a_measurement[good_peak_indices],
                        f_j[good_peak_indices], 
                        label='f_J_peak', 
                        color='red')
            plt.errorbar(a_measurement[good_peak_indices], 
                         f_j[good_peak_indices], 
                         xerr=a_measurement_error[good_peak_indices],
                         yerr=f_j_error[good_peak_indices],
                         color='red',
                         ls='none')
            plt.legend()
        #    plt.xlim(0,0.4e10)
            #plt.ylim([0.001,10])
            plt.title('Measured acceleration vs. f_J_peak')
            plt.xlabel('a_meas [m/s2]')
            plt.ylabel('f_J_peak')
            plt.xscale('log')
            plt.yscale('log')
            pdfpages.savefig()
            
            plt.figure()
            plt.scatter(a_measurement[good_peak_indices],
                        f_b[good_peak_indices], 
                        label='f_b_peak', 
                        color='black')
            plt.errorbar(a_measurement[good_peak_indices], 
                         f_b[good_peak_indices], 
                         xerr=a_measurement_error[good_peak_indices],
                         yerr=f_b_error[good_peak_indices],
                         color='black',
                         ls='none')
            plt.legend()
        #    plt.xlim(0,0.4e10)
            #plt.ylim([0.001,10])
            plt.title('Measured acceleration vs. f_b_peak')
            plt.xlabel('a_meas [m/s2]')
            plt.ylabel('f_b_peak')
            plt.xscale('log')
            plt.yscale('log')
            pdfpages.savefig()
        
        """
        Histograms of f_J and f_b
        """
        
        plt.figure()
    #    f_j=f_j[np.where(f_j < 4)]
        plt.hist(f_j, label='f_J', bins=61)
        plt.legend()
        plt.title('Histogram of f_J')
        plt.xlabel('f_J')
        plt.ylabel('N')
        pdfpages.savefig()
        
        f_b=f_b[np.where(f_b > 0)]
    #    f_b=f_b[np.where(f_b < 1)]
        plt.figure()
        plt.hist(f_b, label='f_b', bins=21)
        plt.legend()
        plt.title('Histogram of f_b')
        plt.xlabel('f_b')
        plt.ylabel('N')
        pdfpages.savefig()
        
        """
        Histograms of f_J and f_b without outliers
        """
        
        plt.figure()
        f_j=f_j[np.where(f_j < 1.7)]
        plt.hist(f_j, label='f_J', bins=15)
        plt.legend()
        plt.title('Histogram of f_J')
        plt.xlabel('f_J')
        plt.ylabel('N')
        pdfpages.savefig()
        
        f_b=f_b[np.where(f_b > 0)]
    #    f_b=f_b[np.where(f_b < 1)]
        plt.figure()
        plt.hist(f_b, label='f_b', bins=21)
        plt.legend()
        plt.title('Histogram of f_b')
        plt.xlabel('f_b')
        plt.ylabel('N')
        pdfpages.savefig()
        
        pdfpages.close()
        if plot_velocity:
            pdf_velocities.close()
            
    if calculate_dependence:
        pdfpages=PdfPages(wd+'/plots/'+result_filename+'_dependence.pdf')
        rad_pol_title=['radial', 'poloidal']
        for key_efit in ['Current',
                         'Pressure grad',
                         'Density grad',
                         'Temperature grad',]:
            for key_gpi in ['Velocity ccf','Size max']:
                for k in range(2):
                    plt.figure()
                    plt.scatter(dependence_db[key_efit], 
                                np.asarray(dependence_db[key_gpi])[:,k])
    #
    #                plt.errorbar(dependence_db[key_efit], 
    #                             np.asarray(dependence_db[key_gpi])[:,k],
    #                              xerr=dependence_db_err[key_efit],
    #                              yerr=np.asarray(dependence_db_err[key_gpi])[:,k],
    #                             ls='none')
                    plt.title(key_efit + ' vs. ' + key_gpi+ ' '+rad_pol_title[k])
                    plt.xlabel(key_efit)
                    plt.ylabel(key_gpi+ ' '+rad_pol_title[k])
                    
                    pdfpages.savefig()
        pdfpages.close()
    
    if calculate_ion_drift_velocity:
        pdfpages=PdfPages(wd+'/plots/'+result_filename+'_ion_drift.pdf')

        plt.figure()
        plt.scatter(np.asarray(ion_drift_velocity_db['Poloidal vel'])/1e3, 
                    np.max(np.asarray(ion_drift_velocity_db['Drift vel']), axis=1))
        plt.plot([0,20],[0,20], color='red')
        
        plt.title('Poloidal velocity vs. Ion drift velocity')
        plt.xlabel('v_pol [km/s]')
        plt.ylabel('v_drift [km/s]')
        pdfpages.savefig()

        print(np.abs(np.asarray(ion_drift_velocity_db['Poloidal vel'])).shape)

        plt.figure()
        x=np.asarray(ion_drift_velocity_db['Crossing psi'])
        not_nan_ind=np.logical_not(np.isnan(ion_drift_velocity_db['Crossing psi']))
        a,b=np.histogram(x[not_nan_ind], bins=10)
        plt.hist(x[not_nan_ind], bins=10)
        plt.plot([np.median(x[not_nan_ind]),np.median(x[not_nan_ind])],[0,100], color='red')
        plt.plot([np.percentile(x[not_nan_ind],10),np.percentile(x[not_nan_ind],10)],[0,100], color='magenta')
        plt.plot([np.percentile(x[not_nan_ind],90),np.percentile(x[not_nan_ind],90)],[0,100], color='magenta')                    
        plt.ylim([0,20])
        plt.title('Histogram of psi')
        plt.xlabel('Psi')
        plt.ylabel('N')
        pdfpages.savefig()
        pdfpages.close()
        
        print(np.mean(ion_drift_velocity_db['Drift vel']))
        print(np.mean(ion_drift_velocity_db['ExB vel']))
    if calculate_greenwald_fraction:
        pdfpages=PdfPages(wd+'/plots/'+result_filename+'_greenwald.pdf')
        rad_pol_title=['radial', 'poloidal']
        for key_gpi in ['Velocity ccf','Size max']:
            for k in range(2):
                plt.figure()
                plt.scatter(greenwald_limit_db['Greenwald fraction'], 
                            np.asarray(greenwald_limit_db[key_gpi])[:,k])
#
#                plt.errorbar(dependence_db[key_efit], 
#                             np.asarray(dependence_db[key_gpi])[:,k],
#                              xerr=dependence_db_err[key_efit],
#                              yerr=np.asarray(dependence_db_err[key_gpi])[:,k],
#                             ls='none')
                plt.title('Greenwald fraction' + ' vs. ' + key_gpi+ ' '+rad_pol_title[k])
                plt.xlabel('Greenwald fraction')
                plt.ylabel(key_gpi+ ' '+rad_pol_title[k])
                
                pdfpages.savefig()
        for key_gpi in ['Elongation max', 'Str number']:
            plt.figure()
            plt.scatter(greenwald_limit_db['Greenwald fraction'], 
                        np.asarray(greenwald_limit_db[key_gpi]))
            plt.title('Greenwald fraction' + ' vs. ' + key_gpi)
            plt.xlabel('Greenwald fraction')
            plt.ylabel(key_gpi+ ' '+rad_pol_title[k])
            
            pdfpages.savefig()
        pdfpages.close()
        
    if calculate_collisionality:
        pdfpages=PdfPages(wd+'/plots/'+result_filename+'_collisionality.pdf')
        rad_pol_title=['radial', 'poloidal']
        for key_gpi in ['Velocity ccf','Size max']:
            for k in range(2):
                plt.figure()
                plt.scatter(collisionality_db['Collisionality'], 
                            np.asarray(collisionality_db[key_gpi])[:,k])
#
#                plt.errorbar(dependence_db[key_efit], 
#                             np.asarray(dependence_db[key_gpi])[:,k],
#                              xerr=dependence_db_err[key_efit],
#                              yerr=np.asarray(dependence_db_err[key_gpi])[:,k],
#                             ls='none')
                plt.title('Collisionality' + ' vs. ' + key_gpi+ ' '+rad_pol_title[k])
                plt.xlabel('Collisionality')
                plt.ylabel(key_gpi+ ' '+rad_pol_title[k])
                
                pdfpages.savefig()
        for key_gpi in ['Elongation max', 'Str number']:
            plt.figure()
            plt.scatter(collisionality_db['Collisionality'], 
                        np.asarray(collisionality_db[key_gpi]))
            plt.title('Collisionality' + ' vs. ' + key_gpi)
            plt.xlabel('Collisionality')
            plt.ylabel(key_gpi+ ' '+rad_pol_title[k])
            
            pdfpages.savefig()
        pdfpages.close()