#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 11:38:06 2019

@author: mlampert
"""

# -*- coding: utf-8 -*-
#Core imports
import os
import copy


#Scientific package imports
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

#Flap imports
import flap
import flap_nstx
flap_nstx.register()
import flap_mdsplus

#Setting up FLAP
flap_mdsplus.register('NSTX_MDSPlus')
thisdir = os.path.dirname(os.path.realpath(__file__))
fn = os.path.join(thisdir,"../flap_nstx.cfg")
flap.config.read(file_name=fn)


def show_nstx_gpi_video(exp_id=None,                                            #Shot number
                        time_range=None,                                        #Time range to show the video in, if not set, the enire shot is shown
                        z_range=None,                                           #Range for the contour/color levels, if not set, min-max is divided
                        logz=False,                                             #Plot the image in a logarithmic coloring
                        plot_filtered=False,                                    #Plot a high pass (100Hz) filtered video
                        normalize=None,                                         #Normalize the video by dividing it with a processed GPI signal
                                                                                #    options: 'Time dependent' (LPF filtered) (recommended)
                                                                                #             'Time averaged' (LPF filtered and averaged for the time range)
                                                                                #             'Simple' (Averaged)
                        normalizer_time_range=None,                             #Time range for the time dependent normalization
                        subtract_background=False,                              #Subtract the background from the image (mean of the time series)
                        plot_flux=False,                                        #Plot the flux surfaces onto the video
                        plot_separatrix=False,                                  #Plot the separatrix onto the video
                        plot_limiter=False,                                     #Plot the limiter of NSTX from EFIT
                        flux_coordinates=False,                                 #Plot the signal as a function of magnetic coordinates
                        device_coordinates=False,                               #Plot the signal as a function of the device coordinates
                        new_plot=True,                                          #Plot the video into a new figure window
                        save_video=False,                                       #Save the video into an mp4 format
                        video_saving_only=False,                                #Saving only the video, not plotting it
                        video_framerate=24,
                        prevent_saturation=False,                               #Prevent saturation of the image by restarting the colormap
                        colormap='gist_ncar',                                   #Colormap for the plotting
                        cache_data=True,                                       #Try to load the data from the FLAP storage
                        ):

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

    if time_range is None:
        print('time_range is None, the entire shot is plotted.')
        slicing=None
    else:
        if (type(time_range) is not list and len(time_range) != 2):
            raise TypeError('time_range needs to be a list with two elements.')
        #time_range=[time_range[0]/1000., time_range[1]/1000.]
        slicing={'Time':flap.Intervals(time_range[0],time_range[1])}
        d=flap.slice_data(object_name,
                          exp_id=exp_id,
                          slicing=slicing,
                          output_name='GPI_SLICED')
        object_name='GPI_SLICED'

    if plot_filtered:
        print("**** Filtering GPI ****")

        d=flap.filter_data(object_name,
                           exp_id=exp_id,
                           output_name='GPI_FILTERED',coordinate='Time',
                           options={'Type':'Highpass',
                                    'f_low':1e2,
                                    'Design':'Chebyshev II'})
        object_name='GPI_FILTERED'

    if normalize is not None:
        print("**** Normalizing GPI ****")
        d=flap.get_data_object(object_name, exp_id=exp_id)
        if normalize in ['Time averaged','Time dependent', 'Simple']:
            if normalize == 'Time averaged':
                coefficient=flap_nstx.tools.calculate_nstx_gpi_norm_coeff(exp_id=exp_id,
                                                          time_range=normalizer_time_range,
                                                          f_high=1e2,
                                                          design='Chebyshev II',
                                                          filter_data=True,
                                                          cache_data=True,
                                                          )
            if normalize == 'Time dependent':
                coefficient=flap.filter_data('GPI',
                                             exp_id=exp_id,
                                             output_name='GPI_LPF',
                                             coordinate='Time',
                                             options={'Type':'Lowpass',
                                                      'f_high':1e2,
                                                      'Design':'Chebyshev II'})
                if slicing is not None:
                    coefficient=coefficient.slice_data(slicing=slicing)
            if normalize == 'Simple':
                coefficient=flap.slice_data(object_name,summing={'Time':'Mean'})

            data_obj=copy.deepcopy(d)
            data_obj.data = data_obj.data/coefficient.data
            flap.add_data_object(data_obj, 'GPI_DENORM')
            object_name='GPI_DENORM'
        else:
            raise ValueError('Normalize can either be "Time averaged","Time dependent" or "Simple".')

    if subtract_background: #DEPRECATED, DOESN'T DO MUCH HELP
        print('**** Subtracting background ****')
        d=flap.get_data_object_ref(object_name, exp_id=exp_id)
        background=flap.slice_data(object_name,
                                   exp_id=exp_id,
                                   summing={'Time':'Mean'})

        data_obj=copy.deepcopy(d)
        data_obj.data=data_obj.data/background.data

        flap.add_data_object(data_obj, 'GPI_BGSUB')
        object_name='GPI_BGSUB'

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

    if save_video:
        if time_range is not None:
            video_filename='NSTX_GPI_'+str(exp_id)+'_'+str(time_range[0])+'_'+str(time_range[1])+'.mp4'
        else:
            video_filename='NSTX_GPI_'+str(exp_id)+'_FULL.mp4'
    else:
        video_filename=None

    if video_saving_only:
        save_video=True

    if z_range is None:
        d=flap.get_data_object_ref(object_name, exp_id=exp_id)
        z_range=[d.data.min(),d.data.max()]

    if z_range[1] < 0:
        raise ValueError('All the values are negative, Logarithmic plotting is not allowed.')

    if logz and z_range[0] <= 0:
        print('Z range should not start with 0 when logarithmic Z axis is set. Forcing it to be 1 for now.')
        z_range[0]=1.

    if not save_video:
        flap.plot(object_name,plot_type='animation',
                  exp_id=exp_id,
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
                  axes=[x_axis,y_axis,'Time'],
                  options={'Z range':z_range,'Wait':0.0,'Clear':False,
                           'Overplot options':oplot_options,
                           'Colormap':colormap,
                           'Equal axes':True,
                           'Waittime':waittime,
                           'Video file':video_filename,
                           'Video format':'mp4',
                           'Video framerate':video_framerate,
                           'Prevent saturation':prevent_saturation,
                           })
        if video_saving_only:
            import matplotlib
            matplotlib.use(current_backend)


"""------------------------------------------------------------------------"""


def show_nstx_gpi_video_frames(exp_id=None,
                               time_range=None,
                               start_time=None,
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
                               colorbar_visibility=True,
                               save_data_for_publication=False,
                               data_filename=None,
                               overplot_points=None,

                               pdf=False,
                               pdf_filename=None,
                               ):

    if time_range is None and start_time is None:
        print('time_range is None, the entire shot is plotted.')

    if time_range is not None:
        if (type(time_range) is not list and len(time_range) != 2):
            raise TypeError('time_range needs to be a list with two elements.')

    if start_time is not None:
        if type(start_time) is not int and type(start_time) is not float:
            raise TypeError('start_time needs to be a number.')

    if not cache_data: #This needs to be enhanced to actually cache the data no matter what
        flap.delete_data_object('*')

    if plot_separatrix or plot_flux:
        print('Setting device_coordinates = True, plotting separatrix or flux is not available in image coordinates.')
        device_coordinates = True

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


    if time_range is None:
        time_range=[start_time,start_time+n_frame*2.5e-6]

    if pdf:
        save_pdf=True

    wd=flap.config.get_all_section('Module NSTX_GPI')['Working directory']

    if normalize:
        flap.slice_data(object_name,
                        slicing={'Time':flap.Intervals(time_range[0]-1/1e3*10,
                                                       time_range[1]+1/1e3*10)},
                        output_name='GPI_SLICED_FOR_FILTERING')

        norm_obj=flap.filter_data('GPI_SLICED_FOR_FILTERING',
                                  exp_id=exp_id,
                                  coordinate='Time',
                                  options={'Type':'Lowpass',
                                           'f_high':1e3,
                                           'Design':'Elliptic'},
                                  output_name='GAS_CLOUD')

        norm_obj.data=np.flip(norm_obj.data,axis=0)
        norm_obj=flap.filter_data('GAS_CLOUD',
                                  exp_id=exp_id,
                                  coordinate='Time',
                                  options={'Type':'Lowpass',
                                           'f_high':1e3,
                                           'Design':'Elliptic'},
                                 output_name='GAS_CLOUD')

        norm_obj.data=np.flip(norm_obj.data,axis=0)
        coefficient=flap.slice_data('GAS_CLOUD',
                                    exp_id=exp_id,
                                    slicing={'Time':flap.Intervals(time_range[0],time_range[1])},
                                    output_name='GPI_GAS_CLOUD').data

        data_obj=flap.slice_data('GPI',
                                 exp_id=exp_id,
                                 slicing={'Time':flap.Intervals(time_range[0],time_range[1])})
        data_obj.data = data_obj.data/coefficient
        flap.add_data_object(data_obj, 'GPI_SLICED_DENORM')
        object_name='GPI_SLICED_DENORM'

    if plot_filtered:
        print("**** Filtering GPI")
        object_name='GPI_FILTERED'
        try:
            flap.get_data_object_ref(object_name, exp_id=exp_id)
        except:
            flap.filter_data(object_name,
                             exp_id=exp_id,
                             coordinate='Time',
                             options={'Type':'Highpass',
                                      'f_low':1e2,
                                      'Design':'Chebyshev II'},
                             output_name='GPI_FILTERED') #Data is in milliseconds
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
        x_axis='Device R'
        y_axis='Device z'
    else:
        oplot_options=None
    if flux_coordinates:
        print("**** Adding Flux r coordinates")
        d.add_coordinate(coordinates='Flux r',exp_id=exp_id)
        x_axis='Flux r'
        y_axis='Device z'
        plot_units={'Flux r':'','Device z':'m'}
    elif device_coordinates:
        x_axis='Device R'
        y_axis='Device z'
        plot_units={'Device R':'m','Device z':'m'}
    else:
        x_axis='Image x'
        y_axis='Image y'
        plot_units=None
    if start_time is not None:
        start_sample_num=flap.slice_data(object_name,
                                         slicing={'Time':start_time}).coordinate('Sample')[0][0,0]
    if n_frame == 30:
        ny=6
        nx=5
    elif n_frame == 20:
        ny=5
        nx=4
    elif n_frame == 9:
        ny=3
        nx=3
    else:
        if len(n_frame) == 2:
            ny=n_frame[0]
            nx=n_frame[1]
        else:
            raise ValueError('The set n_frame doesn\'t have a corresponding setting. Please set it to n_frame=[nx,ny] ')

    gs=GridSpec(nx,ny)

    for index_grid_x in range(nx):
        for index_grid_y in range(ny):

            plt.subplot(gs[index_grid_x,index_grid_y])

            if start_time is not None:
                sample=start_sample_num+index_grid_x*ny+index_grid_y
                slicing={'Sample':sample}
            else:
                time=time_range[0]+(time_range[1]-time_range[0])/(n_frame-1)*(index_grid_x*ny+index_grid_y)
                slicing={'Time':time}
            d=flap.slice_data(object_name, slicing=slicing, output_name='GPI_SLICED')
            slicing={'Time':d.coordinate('Time')[0][0,0]}

            if plot_flux and device_coordinates:
                flap.slice_data('PSI RZ OBJ',slicing=slicing,
                                output_name='PSI RZ SLICE',
                                options={'Interpolation':'Linear'})
                oplot_options['contour']={'flux':{'Data object':'PSI RZ SLICE',
                                                  'Plot':True,
                                                  'Colormap':None,
                                                  'nlevel':51}}

            if plot_separatrix and device_coordinates:
                flap.slice_data('SEP X OBJ',slicing=slicing,output_name='SEP X SLICE',
                                options={'Interpolation':'Linear'})
                flap.slice_data('SEP Y OBJ',slicing=slicing,output_name='SEP Y SLICE',
                                options={'Interpolation':'Linear'})
                oplot_options['path']={'separatrix':{'Data object X':'SEP X SLICE',
                                                     'Data object Y':'SEP Y SLICE',
                                                     'Plot':True,
                                                     'Color':'red'}}
            visibility=[True,True]
            if index_grid_x != nx-1:
                visibility[0]=False
            if index_grid_y != 0:
                visibility[1]=False
            flap.plot('GPI_SLICED',
                      plot_type='contour',
                      exp_id=exp_id,
                      axes=[x_axis,y_axis,'Time'],
                      options={'Z range':z_range,
                               'Interpolation': 'Closest value',
                               'Clear':False,
                               'Equal axes':True,
                               'Plot units':plot_units,
                               'Axes visibility':visibility,
                               'Colormap':colormap,
                               'Colorbar':colorbar_visibility,
                               'Overplot options':oplot_options,
                               },
                       plot_options={'levels':51},
                       )
            if overplot_points is not None:
                plt.scatter(overplot_points[index_grid_x,index_grid_y,0],
                            overplot_points[index_grid_x,index_grid_y,1],
                            color='red',
                            marker='x',
                            )
            if save_data_for_publication:
                data=flap.get_data_object('GPI_SLICED').data

                if start_time is not None:
                    string_add=str(sample)
                else:
                    string_add=str(time)

                if data_filename is None:
                    filename=wd+'/data_accessibility/NSTX_GPI_video_frames_'+str(exp_id)+'_'+string_add+'.txt'
                else:
                    filename=data_filename+'_'+string_add+'.txt'

                with open(filename, 'w+') as file1:
                    file1.write()
                    for i in range(len(data[0,:])):
                        string=''
                        for j in range(len(data[:,0])):
                            string+=str(data[j,i])+'\t'
                        string+='\n'
                        file1.write(string)
                    file1.close()

            actual_time=d.coordinate('Time')[0][0,0]
            #plt.title(str(exp_id)+' @ '+f"{actual_time*1000:.4f}"+'ms')
            plt.title(f"{actual_time*1000:.3f}"+'ms')
    if save_pdf:
        if pdf_filename is None:
            if time_range is not None:
                plt.savefig(wd+'/plots/NSTX_GPI_video_frames_'+str(exp_id)+'_'+str(time_range[0])+'_'+str(time_range[1])+'_nf_'+str(n_frame)+'.pdf')
            else:
                plt.savefig(wd+'/plots/NSTX_GPI_video_frames_'+str(exp_id)+'_'+str(start_time)+'_nf_'+str(n_frame)+'.pdf')
        else:
            plt.savefig(pdf_filename)

def show_nstx_gpi_timetrace(exp_id=None,
                            plot_filtered=False,
                            time_range=None,
                            new_plot=False,
                            overplot=False,
                            scale=1.0,
                            save_pdf=False,
                            cache_data=True,
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