#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 17:38:34 2020

@author: mlampert
"""

import os
import copy

import numpy as np
import pickle
import pandas

import time as time_module

import flap
import flap_nstx

from flap_nstx.analysis import calculate_nstx_gpi_avg_frame_velocity, calculate_nstx_gpi_smooth_velocity
from flap_nstx.analysis import flap_nstx_thomson_data, get_nstx_thomson_gradient, get_fit_nstx_thomson_profiles

from matplotlib import ticker
from matplotlib.backends.backend_pdf import PdfPages

thisdir = os.path.dirname(os.path.realpath(__file__))
fn = os.path.join(thisdir,"flap_nstx.cfg")
flap.config.read(file_name=fn)
flap_nstx.register()

def calculate_magnetics_spectrogram(exp_id=None,
                                    time_range=None,
                                    channel=1,
                                    time_res=1e-3,
                                    freq_res=None,
                                    frange=None,
                                    recalc=False,
                                    plot=True,
                                    pdf=False,
                                    pdfobject=None,
                                    ):
    
    wd=flap.config.get_all_section('Module NSTX_GPI')['Working directory']
    filename=flap_nstx.analysis.filename(exp_id=exp_id,
                                         working_directory=wd+'/processed_data',
                                         time_range=time_range,
                                         comment='magnetic_spectrogram_hf_ch'+str(channel)+'_tr_'+str(time_res)+'_frange_'+str(frange[0])+'_'+str(frange[1]),
                                         extension='pickle')
    if not recalc and not os.path.exists(filename):
        print('File doesn\'t exist, needs to be calculated!')
        recalc=True
    if recalc or not os.path.exists(filename):
        if freq_res is None:
            freq_res=2/time_res
    
        magnetics=flap.get_data('NSTX_MDSPlus',
                           name='\OPS_PC::\\BDOT_L1DMIVVHF'+str(channel)+'_RAW',
                           exp_id=139901,
                           object_name='MIRNOV')
    
        magnetics.coordinates.append(copy.deepcopy(flap.Coordinate(name='Time equi',
                                                   unit='s',
                                                   mode=flap.CoordinateMode(equidistant=True),
                                                   shape = [],
                                                   start=magnetics.coordinate('Time')[0][0],
                                                   step=magnetics.coordinate('Time')[0][1]-magnetics.coordinate('Time')[0][0],
                                                   dimension_list=[0])))
        n_time=int((time_range[1]-time_range[0])/time_res)
        spectrum=[]
        for i in range(n_time-1):
            spectrum.append(flap.apsd('MIRNOV', 
                                      coordinate='Time equi', 
                                      intervals={'Time equi':flap.Intervals(time_range[0]+(i-0.5)*time_res,
                                                                            time_range[0]+(i+1.5)*time_res)},
                                      options={'Res':freq_res,
                                               'Range':frange,
                                               'Interval':1,
                                               'Trend':None,
                                               'Logarithmic':False,
                                               'Hanning':True},
                                      output_name='MIRNOV_TWIN_APSD').data)
        time=np.arange(n_time-1)*time_res+time_range[0]
        freq=flap.get_data_object_ref('MIRNOV_TWIN_APSD').coordinate('Frequency')[0]
        data=np.asarray(spectrum).T
        pickle.dump((time,freq,data), open(filename, 'wb'))
    else:
        time, freq, data = pickle.load(open(filename, 'rb'))
    if plot:
        import matplotlib
        matplotlib.use('QT5Agg')
        import matplotlib.pyplot as plt
    else:
        import matplotlib
        matplotlib.use('agg')
        import matplotlib.pyplot as plt
    if pdf:
        filename=flap_nstx.analysis.filename(exp_id=exp_id,
                                         working_directory=wd+'/plots',
                                         time_range=time_range,
                                         comment='magnetic_spectrogram_hf_ch'+str(channel)+'_tr_'+str(time_res)+'_frange_'+str(frange[0])+'_'+str(frange[1]),
                                         extension='pdf')
        spectrogram_pdf=PdfPages(filename)
    plt.figure()
    plt.contourf(time,
                 freq/1000.,
                 data, 
                 locator=ticker.LogLocator(),
                 cmap='jet',
                 levels=101)
    plt.title('BDOT_L1DMIVVHF'+str(channel)+' spectrogram for '+str(exp_id)+' with fres '+str(1/time_res/1000.)+'kHz')
    plt.xlabel('Time [s]')
    plt.ylabel('Frequency [kHz]')
    plt.pause(0.001)
    
    if pdf:
        spectrogram_pdf.savefig()
        spectrogram_pdf.close()
        
def calculate_elm_db_magnetics_spectrogram(channel=1, 
                                           recalc=False,
                                           pdf=True,
                                           plot=True):
    
    database_file='/Users/mlampert/work/NSTX_workspace/db/ELM_findings_mlampert_velocity_good.csv'
    db=pandas.read_csv(database_file, index_col=0)
    elm_index=list(db.index)
    shot_elm={'Shot':[],
              'ELM':[]}
    for index_elm in range(len(elm_index)):
        elm_time=db.loc[index_elm]['ELM time']/1000.
        shot=int(db.loc[index_elm]['Shot'])
        shot_elm['Shot'].append(shot)
        shot_elm['ELM'].append(elm_time)

    
    unique_shots=np.unique(shot_elm['Shot'])
    n_shot=unique_shots.shape[0]
    ind_step=0.
    
    wd=flap.config.get_all_section('Module NSTX_GPI')['Working directory']
    pdfobject=PdfPages(wd+'/plots/all_elm_spectrograms.pdf')
    
    for u_shot in unique_shots:
        start_time=time_module.time()
        ind=np.where(shot_elm['Shot'] == u_shot)[0]
            
        spec=calculate_magnetics_spectrogram(exp_id=u_shot, 
                                             time_range=[shot_elm['ELM'][int(ind[0])]-10e-3,shot_elm['ELM'][int(ind[-1])]+10e-3], 
                                             time_res=0.1e-3, 
                                             frange=[20e3,1000e3], 
                                             channel=channel, 
                                             recalc=recalc,
                                             pdf=pdf,
                                             plot=plot,
                                             pdfobject=pdfobject)
        pdfobject.savefig()
        spec=calculate_magnetics_spectrogram(exp_id=u_shot, 
                                             time_range=[shot_elm['ELM'][int(ind[0])]-10e-3,shot_elm['ELM'][int(ind[-1])]+10e-3], 
                                             time_res=1e-3, 
                                             frange=[2e3,1000e3], 
                                             channel=channel, 
                                             recalc=recalc,
                                             pdf=pdf,
                                             plot=plot,
                                             pdfobject=pdfobject)
        pdfobject.savefig()
        print('Remaining time is: '+str((time_module.time()-start_time)*(n_shot-1-ind_step)))
        ind_step+=1
    pdfobject.close()