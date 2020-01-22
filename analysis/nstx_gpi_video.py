#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 11:38:06 2019

@author: mlampert
"""

# -*- coding: utf-8 -*-
#Core imports
import os
#Flap imports
import flap
import flap_nstx
flap_nstx.register()
import flap_mdsplus
flap_mdsplus.register('NSTX_MDSPlus')
    
thisdir = os.path.dirname(os.path.realpath(__file__))
fn = os.path.join(thisdir,"flap_nstx.cfg")
flap.config.read(file_name=fn) 

#Scientific package imports
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

def show_nstx_gpi_video(exp_id=None, 
                        time_range=None,
                        #z_range=[0,512],
                        z_range=None,
                        logz=False,
                        plot_filtered=False,
                        normalize=None,            #options: 'Time dependent' or 'Time averaged'
                        normalizer_time_range=None,
                        cache_data=False, 
                        plot_flux=False,
                        plot_separatrix=False,
                        plot_limiter=False,
                        flux_coordinates=False,
                        device_coordinates=False,
                        new_plot=True,
                        save_video=False,
                        video_saving_only=False,
                        prevent_saturation=False,
                        colormap='gist_ncar',
                        ):                
    
    if ((exp_id is None) and (time_range is None)):
        print('The correct way to call the code is the following:\n')
        print('show_nstx_gpi_video(exp_id=141918, time_range=[250.,260.], plot_filtered=True, cache_data=False, plot_efit=True, flux_coordinates=False)\n')
        print('INPUTs: \t\t Description: \t\t\t\t\t Type: \t\t\t Default values: \n')
        print('exp_id: \t\t The shot number. \t\t\t\t int \t\t\t Default: None')
        print('time_range: \t\t The time range. \t\t\t\t [float,float] \t\t Default: None')
        print('plot_filtered: \t\t High pass filter the data from 1e2. \t\t boolean \t\t Default: False')
        print('cache_data: \t\t Cache the FLAP data objects. \t\t\t boolean \t\t Default: False')
        print('plot_efit: \t\t Plot the efit profiles onto the video. \t boolean \t\t Default: False')
        print('flux_coordinates: \t Plot the data in flux coordinates. \t\t boolean \t\t Default: False')
        print('new_plot: \t\t Preserve the previous plot. \t\t boolean \t\t Default: True')
        return
    
    if logz and z_range[0] == 0:
        print('Z range should not start with 0 when logarithmic Z axis is set. Forcing it to be 1 for now.')
        z_range[0]=1
    
    if time_range is None:
        print('time_range is None, the entire shot is plotted.')
        slicing_range=None
    else:    
        if (type(time_range) is not list and len(time_range) != 2):
            raise TypeError('time_range needs to be a list with two elements.')
        #time_range=[time_range[0]/1000., time_range[1]/1000.] 
        slicing_range={'Time':flap.Intervals(time_range[0],time_range[1])}
        
    if exp_id is not None:
        print("\n------- Reading NSTX GPI data --------")
        if cache_data:
            try:
                d=flap.get_data_object_ref(exp_id=exp_id,object_name='GPI')
            except:
                print('Data is not cached, it needs to be read.')
                d=flap.get_data('NSTX_GPI',exp_id=exp_id,name='',object_name='GPI')
        else:
            d=flap.get_data('NSTX_GPI',exp_id=exp_id,name='',object_name='GPI')
        object_name='GPI'
    else:
        raise ValueError('The experiment ID needs to be set.')
        
    if plot_filtered:
        print("**** Filtering GPI ****")
        object_name='GPI_FILTERED'
        d=flap.filter_data('GPI',output_name='GPI_FILTERED',coordinate='Time',
                           options={'Type':'Highpass','f_low':1e2,
                                    'Design':'Chebyshev II'}) #Data is in milliseconds
       
    if ((plot_flux or plot_separatrix) and not flux_coordinates):
        print('Gathering MDSPlus EFIT data.')
        oplot_options={}
        if plot_separatrix:
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
            oplot_options['path']={'separatrix':{'Data object X':'SEP X OBJ',
                                                 'Data object Y':'SEP Y OBJ',
                                                 'Plot':True,
                                                 'Color':'red'}}
            
        if plot_flux:
            d=flap.get_data('NSTX_MDSPlus',
                            name='\EFIT02::\PSIRZ',
                            exp_id=exp_id,
                            object_name='PSI RZ OBJ'
                            )
            oplot_options['contour']={'flux':{'Data object':'PSI RZ OBJ',
                                              'Plot':True,
                                              'Colormap':None,
                                              'nlevel':51}}
        #oplot_options['line']={'trial':{'Horizontal':[[0.200,'red'],[0.250,'blue']],
        #                                'Vertical':[[1.450,'red'],[1.500,'blue']],
        #                                'Plot':True
        #                               }}
            
    else:
        oplot_options=None
        
    if flux_coordinates:
        print("**** Adding Flux r coordinates")
        d.add_coordinate(coordinates='Flux r',exp_id=exp_id)
        x_axis='Flux r'
        y_axis='Device z'
        if plot_separatrix:
            oplot_options={}
            oplot_options['line']={'separatrix':{'Vertical':[[1.0,'red']],
                                                 'Plot':True}}
    elif device_coordinates:
        x_axis='Device R'
        y_axis='Device z'
    else:
        x_axis='Image x'
        y_axis='Image y'
    if new_plot:
        plt.figure()
        

    if normalize is not None:
        print("**** Normalizing GPI ****")
        if normalize in ['Time averaged','Time dependent']:
            if normalize == 'Time averaged':
                coefficient=flap_nstx.analysis.calculate_nstx_gpi_norm_coeff(exp_id=exp_id,              # Experiment ID
                                                          time_range=normalizer_time_range,
                                                          f_high=1e2,               # Low pass filter frequency in Hz
                                                          design='Chebyshev II',    # IIR filter design (from scipy)
                                                          filter_data=True,         # IIR LPF the data
                                                          cache_data=True,
                                                          )
            if normalize == 'Time dependent':
                coefficient=flap.filter_data('GPI',exp_id=exp_id,
                                             output_name='GPI_LPF',coordinate='Time',
                                             options={'Type':'Lowpass',
                                                      'f_high':1e2,
                                                      'Design':'Chebyshev II'}) #Data is in milliseconds
        
            d.data = d.data/coefficient.data #This should be checked to some extent, it works with smaller matrices
        else:
            raise ValueError('Normalize can either be "Time averaged" or "Time dependent".')
            
    if save_video:
        if time_range is not None:
            video_filename='NSTX_GPI_'+str(exp_id)+'_'+str(time_range[0])+'_'+str(time_range[1])+'.mp4'
        else:
            video_filename='NSTX_GPI_'+str(exp_id)+'_FULL.mp4'
    else:
        video_filename=None
    if video_saving_only:
        save_video=True
        
    if not save_video:
        flap.plot(object_name,plot_type='animation',
                  exp_id=exp_id,
                  slicing=slicing_range,
                  axes=[x_axis,y_axis,'Time'],
                  options={'Z range':z_range,'Wait':0.0,'Clear':False,
                           'Overplot options':oplot_options,
                           'Colormap':colormap,
                           'Log z':logz,
                           'Equal axes':True,
                           'Prevent saturation':prevent_saturation,
                           'Plot units':{'Time':'s',
                                         'Device R':'m',
                                         'Device z':'m'}
                           })
    else:
        if video_saving_only:
            import matplotlib
            current_backend=matplotlib.get_backend()
            matplotlib.use('agg')
            waittime=0.
        else:
            waittime=1./24.
            waittime=0.
        flap.plot(object_name,plot_type='anim-image',
                  exp_id=exp_id,
                  slicing=slicing_range,
                  axes=[x_axis,y_axis,'Time'],
                  options={'Z range':z_range,'Wait':0.0,'Clear':False,
                           'Overplot options':oplot_options,
                           'Colormap':colormap,
                           'Equal axes':True,
                           'Waittime':waittime,
                           'Video file':video_filename,
                           'Video format':'mp4',
                           'Prevent saturation':prevent_saturation,
                           })
        if video_saving_only:
            import matplotlib
            matplotlib.use(current_backend)

    
"""------------------------------------------------------------------------"""
    
    
def show_nstx_gpi_video_frames(exp_id=None, 
                           time_range=None,
                           n_frame=20,
                           logz=False,
                           z_range=[0,512],
                           plot_filtered=False, 
                           normalize=False,
                           cache_data=False, 
                           plot_flux=False, 
                           plot_separatrix=False, 
                           flux_coordinates=False,
                           device_coordinates=False,
                           new_plot=True,
                           save_pdf=False,
                           colormap='gist_ncar',
                           save_for_paraview=False,
                           ):
    
    if time_range is None:
        print('time_range is None, the entire shot is plotted.')
    else:    
        if (type(time_range) is not list and len(time_range) != 2):
            raise TypeError('time_range needs to be a list with two elements.')
        
    if not cache_data: #This needs to be enhanced to actually cache the data no matter what
        flap.delete_data_object('*')
    if exp_id is not None:
        print("\n------- Reading NSTX GPI data --------")
        if cache_data:
            try:
                d=flap.get_data_object_ref(exp_id=exp_id,object_name='GPI')
            except:
                print('Data is not cached, it needs to be read.')
                d=flap.get_data('NSTX_GPI',exp_id=exp_id,name='',object_name='GPI')
        else:
            d=flap.get_data('NSTX_GPI',exp_id=exp_id,name='',object_name='GPI')
        object_name='GPI'
    else:
        raise ValueError('The experiment ID needs to be set.')
    if plot_filtered:
        print("**** Filtering GPI")
        object_name='GPI_FILTERED'
        try:
            flap.get_data_object_ref(object_name, exp_id=exp_id)
        except:
            flap.filter_data('GPI',exp_id=exp_id,output_name='GPI_FILTERED',coordinate='Time',
                             options={'Type':'Highpass',
                                      'f_low':1e2,
                                      'Design':'Chebyshev II'}) #Data is in milliseconds
                
    if plot_flux or plot_separatrix:
        print('Gathering MDSPlus EFIT data.')
        oplot_options={}
        if plot_separatrix:
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
        if plot_flux:
            d=flap.get_data('NSTX_MDSPlus',
                            name='\EFIT01::\PSIRZ',
                            exp_id=exp_id,
                            object_name='PSI RZ OBJ'
                            )
    else:
        oplot_options=None
    if flux_coordinates:
        print("**** Adding Flux r coordinates")
        d.add_coordinate(coordinates='Flux r',exp_id=exp_id)
        x_axis='Flux r'
        y_axis='Device z'
    elif device_coordinates:
        x_axis='Device R'
        y_axis='Device z'
    else:
        x_axis='Image x'
        y_axis='Image y'        
        
    if n_frame == 30:
        ny=6
        nx=5
        gs=GridSpec(nx,ny)
        for index_grid_x in range(nx):
            for index_grid_y in range(ny):
                print(time_range[0]+(time_range[1]-time_range[0])/(n_frame-1)*(index_grid_x*ny+index_grid_y))
                plt.subplot(gs[index_grid_x,index_grid_y])
                time=time_range[0]+(time_range[1]-time_range[0])/(n_frame-1)*(index_grid_x*ny+index_grid_y)
                if plot_flux:
                    flap.slice_data('PSI RZ OBJ',slicing={'Time':time},output_name='PSI RZ SLICE',options={'Interpolation':'Linear'})
                    oplot_options['contour']={'flux':{'Data object':'PSI RZ SLICE',
                                                      'Plot':True,
                                                      'Colormap':None,
                                                      'nlevel':51}}
                    
                if plot_separatrix:
                    flap.slice_data('SEP X OBJ',slicing={'Time':time},output_name='SEP X SLICE',options={'Interpolation':'Linear'})
                    flap.slice_data('SEP Y OBJ',slicing={'Time':time},output_name='SEP Y SLICE',options={'Interpolation':'Linear'})
                    oplot_options['path']={'separatrix':{'Data object X':'SEP X SLICE',
                                                         'Data object Y':'SEP Y SLICE',
                                                         'Plot':True,
                                                         'Color':'red'}}
                visibility=[True,True]
                if index_grid_y != 0:
                    visibility[1]=False
                if index_grid_x != nx-1:
                    visibility[0]=False
                flap.plot(object_name,plot_type='contour',
                          exp_id=exp_id,
                          slicing={'Time':time},
                          axes=[x_axis,y_axis,'Time'],
                          options={'Z range':z_range,
                                   'Interpolation': 'Closest value',
                                   'Clear':False,
                                   'Equal axes':True,
                                   'Plot units':{'Device R':'mm',
                                                 'Device z':'mm'},
                                   'Axes visibility':visibility,
                                   'Colormap':colormap,
                                   'Overplot options':oplot_options,
                                   })
                
                
                plt.title(str(exp_id)+' @ '+f"{time*1000:.4f}"+'ms')
    if save_pdf:
        plt.savefig('NSTX_GPI_video_frames_'+str(exp_id)+'_'+str(time_range[0])+'_'+str(time_range[1])+'_nf_'+str(n_frame)+'.pdf')
    
def show_nstx_gpi_timetrace(exp_id=None,
                            cache_data=True,
                            plot_filtered=False,
                            time_range=None,
                            new_plot=False,
                            overplot=False,
                            scale=1.0,
                            save_pdf=False
                            ):
    plot_options={}
    if time_range is None:
        print('time_range is None, the entire shot is plot.')
        slicing_range=None
    else:    
        if (type(time_range) is not list and len(time_range) != 2):
            raise TypeError('time_range needs to be a list with two elements.')
        plot_options['X range']=time_range
        slicing_range={'Time':flap.Intervals(time_range[0],time_range[1])}
    if exp_id is not None:
        print("\n------- Reading NSTX GPI data --------")
        if cache_data:
            try:
                d=flap.get_data_object_ref(exp_id=exp_id,object_name='GPI')
            except:
                print('Data is not cached, it needs to be read.')
                d=flap.get_data('NSTX_GPI',exp_id=exp_id,name='',object_name='GPI')
        else:
            flap.get_data('NSTX_GPI',exp_id=exp_id,name='',object_name='GPI')
    else:
        raise ValueError('The experiment ID needs to be set.')
    flap.slice_data('GPI',
                    #slicing=slicing_range,
                    slicing=slicing_range,
                    summing={'Image x':'Mean','Image y':'Mean'},
                    output_name='GPI_MEAN')
    object_name='GPI_MEAN'
    
    if plot_filtered:
        print("**** Filtering GPI")
        object_name='GPI_MEAN_FILTERED'
        flap.filter_data('GPI_MEAN',output_name='GPI_MEAN_FILTERED',coordinate='Time',
                         options={'Type':'Highpass',
                                  'f_low':1e2,
                                  'Design':'Chebyshev II'}) #Data is in milliseconds
    if scale != 1.0:
        d=flap.get_data_object_ref(object_name, exp_id)
        d.data=d.data*scale
    if new_plot and not overplot:
        plt.figure()
    elif overplot:
        plot_options['Force axes']=True
    else:
        plt.cla()
    plot_options['All points']=True
    
    flap.plot(object_name,
              axes=['Time', '__Data__'],
              exp_id=exp_id,
              options=plot_options)
    if save_pdf:
        if time_range is not None:
            filename='NSTX_'+str(exp_id)+'_GPI_'+str(time_range[0])+'_'+str(time_range[1])+'_mean.pdf'
        else:
            filename='NSTX_'+str(exp_id)+'_GPI_mean.pdf'
        plt.savefig(filename)
        
def show_nstx_gpi_slice_traces(exp_id=None,
                               time_range=None,
                               x_slices=np.linspace(0,60,14),
                               y_slices=np.linspace(0,70,16),
                               x_summing=False,
                               y_summing=False,
                               z_range=[0,512],
                               zlog=False,
                               filename=None,
                               filter_data=False,
                               save_pdf=False,
                               pdf_saving_only=False):
    
    if pdf_saving_only:
        import matplotlib
        current_backend=matplotlib.get_backend()
        matplotlib.use('agg')
    import matplotlib.pyplot as plt
    if save_pdf:
        pdf=PdfPages(filename)
    plt.cla() 
    if filename is None:
        filename='NSTX_GPI_SLICE_'+str(exp_id)+'_'+str(time_range[0])+'_'+str(time_range[1])+'.pdf'
    
    flap.get_data('NSTX_GPI', exp_id=exp_id, name='', object_name='GPI')
        
    if filter_data:
        flap.filter_data('GPI',exp_id=exp_id,
                     coordinate='Time',
                     options={'Type':'Highpass',
                              'f_low':1e2,
                              'Design':'Chebyshev II'})
    if not x_summing:
        for i in range(len(x_slices)):
            plt.figure()
            flap.plot('GPI', plot_type='image', 
                      axes=['Time', 'Image y'], 
                      slicing={'Time':flap.Intervals(time_range[0],time_range[1]), 'Image x':x_slices[i]}, 
                      #plot_options={'levels':100}, 
                      options={'Z range':z_range,'Log z':zlog})
            plt.title('NSTX GPI '+str(exp_id)+' Image x = '+str(int(x_slices[i])))
            if save_pdf:
                pdf.savefig()
                plt.close()
    else:
        plt.figure()
        flap.plot('GPI', plot_type='image', 
                  axes=['Time', 'Image y'], 
                  slicing={'Time':flap.Intervals(time_range[0],time_range[1])}, 
                  summing={'Image x':'Mean'},
                  #plot_options={'levels':100}, 
                  options={'Z range':z_range,'Log z':zlog})
        plt.title('NSTX GPI '+str(exp_id)+' Mean x pixels')
        if save_pdf:
            pdf.savefig()
            plt.close()
    if not x_summing:    
        for j in range(len(y_slices)):
            if not y_summing:
                slicing={'Time':flap.Intervals(time_range[0],time_range[1]), 'Image y':y_slices[i]}
                y_summing_opt=None
            else:
                slicing={'Time':flap.Intervals(time_range[0],time_range[1])}
                y_summing_opt={'Image y':'Mean'}
            plt.figure()
            flap.plot('GPI', plot_type='image', 
                      axes=['Time', 'Image x'], 
                      slicing=slicing,
                      summing=y_summing_opt,
                      #plot_options={'levels':100}, 
                      options={'Z range':z_range,'Log z':zlog})
            plt.title('NSTX GPI '+str(exp_id)+' Image y = '+str(int(y_slices[j])))
            if save_pdf:
                pdf.savefig()
                plt.close()
    else:
        plt.figure()
        flap.plot('GPI', plot_type='image', 
                  axes=['Time', 'Image x'], 
                  slicing={'Time':flap.Intervals(time_range[0],time_range[1])}, 
                  summing={'Image y':'Mean'},
                  #plot_options={'levels':100}, 
                  options={'Z range':z_range,'Log z':zlog})
        plt.title('NSTX GPI '+str(exp_id)+' Mean y pixels')
        if save_pdf:
            pdf.savefig()
            plt.close()
    if save_pdf:
        pdf.close() 
    if pdf_saving_only:
        import matplotlib
        matplotlib.use(current_backend)           

#show_nstx_gpi_video(exp_id=141918, time_range=[250.,260.], plot_filtered=True, cache_data=False, plot_efit=True, flux_coordinates=False)