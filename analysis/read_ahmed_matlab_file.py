#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 13:18:26 2020

@author: mlampert
"""
import csv
import os

from scipy.io import loadmat
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

import flap
import flap_nstx
flap_nstx.register()
import flap_mdsplus

#Setting up FLAP
flap_mdsplus.register('NSTX_MDSPlus')    
thisdir = os.path.dirname(os.path.realpath(__file__))
fn = os.path.join(thisdir,"../flap_nstx.cfg")
flap.config.read(file_name=fn) 

def read_ahmed_matlab_file(plot=False):
    
    if plot:
        import matplotlib
        matplotlib.use('QT5Agg')
        import matplotlib.pyplot as plt
    else:
        import matplotlib
        matplotlib.use('agg')
        import matplotlib.pyplot as plt
    
    wd=flap.config.get_all_section('Module NSTX_GPI')['Working directory']
    
    pdf_filename='ahmed_all_plots.pdf'
    pdf_filename=wd+'/plots/'+pdf_filename
    pdf_pages=PdfPages(pdf_filename+'.pdf')
    
    db=loadmat('/Users/mlampert/work/NSTX_workspace/WORK_MATE/Fits_ELMfindingsmlampertvelocitygood.mat')
    db_out=[]
    '''[0][shot_index][0][elm_index][shotnum,t1,t2]'''
    for i_shot in range(db['data']['out'][0].shape[0]):
        for i_elm in range(db['data']['out'][0][i_shot][0].shape[0]):
            shot=db['data']['out'][0][i_shot][0][i_elm][0][0][0]
            time1=db['data']['out'][0][i_shot][0][i_elm][1][0][0]
            time2=db['data']['out'][0][i_shot][0][i_elm][2][0][0]
            data_psi=db['data']['out'][0][i_shot][0][i_elm][3][0][0][1][0][0]       #
            data_dev=db['data']['out'][0][i_shot][0][i_elm][3][0][0][0][0][0]       #
            
            psi_n=data_psi[0][0][0][0]
            n_e_psi=data_psi[0][0][0][1]
            n_e_err_psi=data_psi[0][0][0][2]
            
            psi_t=data_psi[1][0][0][0]
            t_e_psi=data_psi[1][0][0][1]
            t_e_err_psi=data_psi[1][0][0][2]
            
            psi_p=data_psi[2][0][0][0]
            p_e_psi=data_psi[2][0][0][1]
            p_e_err_psi=data_psi[2][0][0][2]
            
            dev_n=data_dev[0][0][0][0]
            n_e_dev=data_dev[0][0][0][1]
            n_e_err_dev=data_dev[0][0][0][2]
            
            dev_t=data_dev[1][0][0][0]
            t_e_dev=data_dev[1][0][0][1]
            t_e_err_dev=data_dev[1][0][0][2]
            
            dev_p=data_dev[2][0][0][0]
            p_e_dev=data_dev[2][0][0][1]
            p_e_err_dev=data_dev[2][0][0][2]
            
            if plot:
                plt.figure()
                plt.plot(psi_n,n_e_psi)
                plt.errorbar(psi_n,n_e_psi,yerr=n_e_err_psi)
                plt.xlabel('psi_n')
                plt.ylabel('n_e [1e20m-3]')
                plt.title('Density for '+str(shot)+' '+str(time1)+' '+str(time2))
                pdf_pages.savefig()
                
                plt.figure()
                plt.plot(psi_t,t_e_psi)
                plt.errorbar(psi_t,t_e_psi,yerr=t_e_err_psi)
                plt.xlabel('psi_n')
                plt.ylabel('t_e [keV]')
                plt.title('Temperature for '+str(shot)+' '+str(time1)+' '+str(time2))
                pdf_pages.savefig()
                
                plt.figure()
                plt.plot(psi_t,p_e_psi)
                plt.errorbar(psi_p,p_e_psi,yerr=p_e_err_psi)
                plt.xlabel('psi_n')
                plt.ylabel('p_e [kPa]')
                plt.title('Pressure for '+str(shot)+' '+str(time1)+' '+str(time2))
                pdf_pages.savefig()      
                
            db_out.append({'shot':shot,
                           'time1':time1,
                           'time2':time2,
                           
                           'psi_n':psi_n,
                           'n_e_psi':n_e_psi,
                           'n_e_err_psi':n_e_err_psi,
                           
                           'psi_t':psi_t,
                           't_e_psi':t_e_psi,
                           't_e_err_psi':t_e_err_psi,
                           
                           'psi_p':psi_p,
                           'p_e_psi':p_e_psi,
                           'p_e_err_psi':p_e_err_psi,
                           
                           'dev_n':dev_n,
                           'n_e_dev':n_e_dev,
                           'n_e_err_dev':n_e_err_dev,
                           
                           'dev_t':dev_t,
                           't_e_dev':t_e_dev,
                           't_e_err_dev':t_e_err_dev,
                           
                           'dev_p':dev_p,
                           'p_e_dev':p_e_dev,
                           'p_e_err_dev':p_e_err_dev,
                        })
    pdf_pages.close()
    if not plot:
        import matplotlib
        matplotlib.use('QT5Agg')
        import matplotlib.pyplot as plt
    
    return db_out    
        
        
def read_ahmed_fit_parameters():
    
    db_ahmed=[]
    with open('/Users/mlampert/work/NSTX_workspace/WORK_MATE/Profile_fitsfur_Mate') as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')
        for row in reader:
            line=[]
            for data in row:
                try:
                    line.append(float(data.strip(' ')))
                except:
                    pass
            db_ahmed.append(line)
        
    db_ahmed=np.asarray(db_ahmed).transpose()
    db_dict={'shot':db_ahmed[0,:],
             'time1':db_ahmed[1,:],
             'time2':db_ahmed[2,:],
             
             'Temperature':{'a':db_ahmed[3,:],
                            'b':db_ahmed[4,:],
                            'c':db_ahmed[5,:],
                            'h':db_ahmed[6,:],
                            'xo':db_ahmed[7,:],},
                    
             'Density':{'a':db_ahmed[8,:],
                        'b':db_ahmed[9,:],
                        'c':db_ahmed[10,:],
                        'h':db_ahmed[11,:],
                        'xo':db_ahmed[12,:],},
                   
             'Pressure':{'a':db_ahmed[13,:],
                         'b':db_ahmed[14,:],
                         'c':db_ahmed[15,:],
                         'h':db_ahmed[16,:],
                         'xo':db_ahmed[17,:],},
             }
    for key in ['Temperature','Density','Pressure']:
        db_dict[key]['grad_glob']=(db_dict[key]['h']-db_dict[key]['b'])/db_dict[key]['c']
        db_dict[key]['ped_height']=(db_dict[key]['h']-db_dict[key]['b'])
        db_dict[key]['value_at_max_grad']=(db_dict[key]['h']+db_dict[key]['b'])/2
        db_dict[key]['ped_width']=db_dict[key]['c']
    
    
    """
    The model equation for this fitting is the following:
    ans(x) = (h+b)/2 + (h-b)/2*((1 - a*2*(x - xo)/c).*exp(-2*(x - xo)/c) - exp(2*(x 
                    - xo)/c))./(exp(2*(x - xo)/c) + exp(-2*(x - xo)/c))
    """
    return db_dict

def read_ahmed_edge_current():
    
    db_ahmed=[]
    with open('/Users/mlampert/work/NSTX_workspace/WORK_MATE/Profile_fitsfur_Matewithcurrent') as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')
        for row in reader:
            line=[]
            for data in row:
                try:
                    line.append(float(data.strip(' ')))
                except:
                    pass
            db_ahmed.append(line)
        
    db_ahmed=np.asarray(db_ahmed).transpose()
    db_dict={'shot':db_ahmed[0,:],
             'time1':db_ahmed[1,:],
             'time2':db_ahmed[2,:],
             
             'Current':db_ahmed[9,:],
             }
    
    
    """
    The model equation for this fitting is the following:
    ans(x) = (h+b)/2 + (h-b)/2*((1 - a*2*(x - xo)/c).*exp(-2*(x - xo)/c) - exp(2*(x 
                    - xo)/c))./(exp(2*(x - xo)/c) + exp(-2*(x - xo)/c))
    """
    return db_dict