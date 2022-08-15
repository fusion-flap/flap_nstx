#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 23:12:05 2021

@author: mlampert
"""

#Core imports
import os
import copy
#Importing and setting up the FLAP environment
import flap
import flap_nstx
flap_nstx.register()

thisdir = os.path.dirname(os.path.realpath(__file__))
fn = os.path.join(thisdir,"../flap_nstx.cfg")
flap.config.read(file_name=fn)
#Scientific library imports

from flap_nstx.tools import Polygon, FitEllipse, FitGaussian

import cv2

import imutils

import matplotlib.pyplot as plt
#from matplotlib.patches import Ellipse

import numpy as np
#import sys
#np.set_printoptions(threshold=sys.maxsize)
import scipy

from skimage.feature import peak_local_max
from skimage.filters import threshold_otsu
from skimage.segmentation import watershed


def nstx_gpi_contour_structure_finder(data_object=None,                       #Name of the FLAP.data_object
                                      exp_id='*',                             #Shot number (if data_object is not used)
                                      time=None,                              #Time when the structures need to be evaluated (when exp_id is used)
                                      sample=None,                            #Sample number where the structures need to be evaluated (when exp_id is used)
                                      spatial=False,                          #Calculate the results in real spatial coordinates
                                      pixel=False,                            #Calculate the results in pixel coordinates
                                      mfilter_range=5,                        #Range of the median filter
                                      nlevel=80//5,                           #The number of contours to be used for the calculation (default:ysize/mfilter_range=80//5)
                                      levels=None,                            #Contour levels from an input and not from automatic calculation
                                      threshold_level=None,                   #Threshold level over which it is considered to be a structure
                                                                                    #if set, the value is subtracted from the data and contours are found after that.
                                                                                    #Negative values are substituted with 0.
                                      filter_struct=True,                     #Filter out the structures with less than filter_level number of contours
                                      filter_level=None,                      #The number of contours threshold for structures filtering (default:nlevel//4)
                                      remove_interlaced_structures=False,     #Filter out the structures which are interlaced. Only the largest structures is preserved, others are removed.
                                      test_result=False,                      #Test the result only (plot the contour and the found structures)
                                      test=False,                             #Test the contours and the structures before any kind of processing
                                      save_data_for_publication=False,
                                      ellipse_method='linalg',                #linalg or skimage
                                      fit_shape='ellipse',                    #ellipse or gaussian
                                      str_size_lower_thres=4*0.00375,         #lower threshold for structures
                                      elongation_threshold=0.1,               #Threshold for angle definition, the angle of circular structures cannot be determined.
                                      skip_gaussian=True,
                                      ):

    """
    The method calculates the radial and poloidal sizes of the structures
    present in one from of the GPI image. It gathers the isosurface contour
    coordinates and determines the structures based on certain criteria. In
    principle no user input is necessary, the code provides a robust solution.
    The sizes are determined by fitting an ellipse onto the contour at
    half-height. The code returns the following list:
        a[structure_index]={'Paths':     [list of the paths, type: matplotlib.path.Path],
                            'Half path': [path at the half level of the structure]
                            'Levels':    [levels of the paths, type: list],
                            'Center':    [center of the ellipse in px,py or R,z coordinates, type: numpy.ndarray of two elements],
                            'Size':      [size of the ellipse in x and y direction or R,z direction, type: ]numpy.ndarray of two elements,
                            'Angle':      [angle of the ellipse compared to horizontal in radians, type: numpy.float64],
                            'Area':      [area of the polygon at the half level],
                            ('Ellipse':  [the entire ellipse object, returned if test_result is True, type: flap_nstx.tools.FitEllipse])
                            }
    """

    """
    ----------------
    READING THE DATA
    ----------------
    """
    if type(data_object) is str:
        data_object=flap.get_data_object_ref(data_object, exp_id=exp_id)
        if len(data_object.data.shape) != 2:
            raise IOError('The inpud data_object is not 2D. The method only processes 2D data.')

    if skip_gaussian and fit_shape=='gaussian':
        raise ValueError('If the gaussian fitting is skipped, the fitting shape cannot be gaussian...')

    if data_object is None:
        if (exp_id is None) or ((time is None) and (sample is None)):
            raise IOError('exp_id and time needs to be set if data_object is not set.')
        try:
            data_object=flap.get_data_object_ref('GPI', exp_id=exp_id)
        except:
            print('---- Reading GPI data ----')
            data_object=flap.get_data('NSTX_GPI', exp_id=exp_id, name='', object_name='GPI')

        if (time is not None) and (sample is not None):
            raise IOError('Either time or sample can be set, not both.')

        if time is not None:
            data_object=data_object.slice_data(slicing={'Time':time})
        if sample is not None:
            data_object=data_object.slice_data(slicing={'Sample':sample})

    try:
        data_object.data
    except:
        raise IOError('The input data object should be a flap.DataObject')

    if len(data_object.data.shape) != 2:
        raise TypeError('The frame dataobject needs to be a 2D object without a time coordinate.')

    if pixel:
        x_coord_name='Image x'
        y_coord_name='Image y'
    if spatial:
        x_coord_name='Device R'
        y_coord_name='Device z'

    x_coord=data_object.coordinate(x_coord_name)[0]
    y_coord=data_object.coordinate(y_coord_name)[0]

    """
    ----------------
    READING THE DATA
    ----------------
    """

    data = scipy.ndimage.median_filter(data_object.data, mfilter_range)

    if test:
        plt.cla()

    if threshold_level is not None:
        if data.max() < threshold_level:
            print('The maximum of the signal doesn\'t reach the threshold level.')
            return None
        data_thres = data - threshold_level
        data_thres[np.where(data_thres < 0)] = 0.

    if levels is None:
        levels=np.arange(nlevel)/(nlevel-1)*(data_thres.max()-data_thres.min())+data_thres.min()
    else:
        nlevel=len(levels)

    try:
        structure_contours=plt.contourf(x_coord, y_coord, data_thres, levels=levels)
    except:
        plt.cla()
        plt.close()
        print('Failed to create the contours for the structures.')
        return None

    # if not test or test_result:
    #     plt.cla()
    structures=[]
    one_structure={'Paths':[None],
                   'Levels':[None],
                   'Born':False,
                   'Died':False,
                   'Splits':False,
                   'Merges':False,
                   'Label':None,
                   'Parent':[],
                   'Child':[],
                   }

    # structures.append({'Polygon':full_polygon,
    #                     'Vertices':full_polygon.vertices,
    #                     'Half path':Path(max_contour_looped,codes),
    #                     'X coord':full_polygon.x_data,
    #                     'Y coord':full_polygon.y_data,
    #                     'Data':full_polygon.data,
    #                     'Born':False,
    #                     'Died':False,
    #                     'Parent':None,
    #                     'Label':None,
    #                     })

    if test:
        print('Plotting levels')
    else:
        plt.close()
    #The following lines are the core of the code. It separates the structures
    #from each other and stores the in the structure list.

    """
    Steps of the algorithm:

        1st step: Take the paths at the highest level and store them. These
                  create the initial structures
        2nd step: Take the paths at the second highest level
            2.1 step: if either of the previous paths contain either of
                      the paths at this level, the corresponding
                      path is appended to the contained structure from the
                      previous step.
            2.2 step: if none of the previous structures contain the contour
                      at this level, a new structure is created.
        3rd step: Repeat the second step until it runs out of levels.
        4th step: Delete those structures from the list which doesn't have
                  enough paths to be called a structure.

    (Note: a path is a matplotlib path, a structure is a processed path)
    """
    for i_lev in range(len(structure_contours.collections)-1,-1,-1):
        cur_lev_paths=structure_contours.collections[i_lev].get_paths()
        n_paths_cur_lev=len(cur_lev_paths)

        if len(cur_lev_paths) > 0:
            if len(structures) == 0:
                for i_str in range(n_paths_cur_lev):
                    structures.append(copy.deepcopy(one_structure))
                    structures[i_str]['Paths'][0]=cur_lev_paths[i_str]
                    structures[i_str]['Levels'][0]=levels[i_lev]
            else:
                for i_cur in range(n_paths_cur_lev):
                    new_path=True
                    cur_path=cur_lev_paths[i_cur]
                    for j_prev in range(len(structures)):
                        if cur_path.contains_path(structures[j_prev]['Paths'][-1]):
                            structures[j_prev]['Paths'].append(cur_path)
                            structures[j_prev]['Levels'].append(levels[i_lev])
                            new_path=False
                    if new_path:
                        structures.append(copy.deepcopy(one_structure))
                        structures[-1]['Paths'][0]=cur_path
                        structures[-1]['Levels'][0]=levels[i_lev]
                    if test:
                        x=cur_lev_paths[i_cur].to_polygons()[0][:,0]
                        y=cur_lev_paths[i_cur].to_polygons()[0][:,1]
                        plt.plot(x,y)
                        plt.axis('equal')
                        plt.pause(0.001)

    #Cut the structures based on the filter level
    if filter_level is None:
        filter_level=nlevel//5

    if filter_struct:
        cut_structures=[]
        for i_str in range(len(structures)):
            if len(structures[i_str]['Levels']) > filter_level:
                cut_structures.append(structures[i_str])
    structures=cut_structures

    if test:
        print('Plotting structures')
        plt.cla()
        plt.set_aspect(1.0)
        for struct in structures:
            plt.contourf(x_coord, y_coord, data, levels=levels)
            for path in struct['Paths']:
                x=path.to_polygons()[0][:,0]
                y=path.to_polygons()[0][:,1]
                plt.plot(x,y)
            plt.pause(0.001)
            plt.cla()
            #plt.axis('equal')
            plt.set_aspect(1.0)
        plt.contourf(x_coord, y_coord, data, levels=levels)
        plt.colorbar()

    #Finding the contour at the half level for each structure and
    #calculating its properties
    if len(structures) > 1:
        #Finding the paths at FWHM
        paths_at_half=[]
        for i_str in range(len(structures)):
            half_level=(structures[i_str]['Levels'][-1]+structures[i_str]['Levels'][0])/2.
            ind_at_half=np.argmin(np.abs(structures[i_str]['Levels']-half_level))
            paths_at_half.append(structures[i_str]['Paths'][ind_at_half])

        #Process the structures which are embedded (cut the inner one)
        if remove_interlaced_structures:
            structures_to_be_removed=[]
            for ind_path1 in range(len(paths_at_half)):
                for ind_path2 in range(ind_path1,len(paths_at_half),1):
                    if ind_path1 != ind_path2:
                        if paths_at_half[ind_path2].contains_path(paths_at_half[ind_path1]):
                            structures_to_be_removed.append(ind_path1)
                        if paths_at_half[ind_path2] == paths_at_half[ind_path1]:
                            structures_to_be_removed.append(ind_path2)
            structures_to_be_removed=np.unique(structures_to_be_removed)
            cut_structures=[]
            for i_str in range(len(structures)):
                if i_str not in structures_to_be_removed:
                    cut_structures.append(structures[i_str])
            structures=cut_structures

    #Calculate the ellipse and its properties for the half level contours
    for i_str in range(len(structures)):

        str_levels=structures[i_str]['Levels']
        half_level=(str_levels[-1]+str_levels[0])/2.
        ind_at_half=np.argmin(np.abs(str_levels-half_level))
        n_path=len(structures[i_str]['Levels'])

        polygon_areas=np.zeros(n_path)
        polygon_centroids=np.zeros([n_path,2])
        polygon_intensities=np.zeros(n_path)

        for i_path in range(n_path):
            polygon=structures[i_str]['Paths'][i_path].to_polygons()
            if polygon != []:
                polygon=polygon[0]
                polygon_areas[i_path]=flap_nstx.tools.Polygon(polygon[:,0],polygon[:,1]).area
                polygon_centroids[i_path,:]=flap_nstx.tools.Polygon(polygon[:,0],polygon[:,1]).centroid
            if i_path == 0:
                polygon_intensities[i_path]=polygon_areas[i_path]*str_levels[i_path]
            else:
                polygon_intensities[i_path]=(polygon_areas[i_path]-polygon_areas[i_path-1])*str_levels[i_path]

        half_coords=structures[i_str]['Paths'][ind_at_half].to_polygons()[0]

        half_polygon=flap_nstx.tools.Polygon(half_coords[:,0],half_coords[:,1])
        x_data=[]
        y_data=[]
        int_data=[]
        coords_2d=[]
        data_inside_half=[]

        x_inds_of_half=list(np.where(np.logical_and(x_coord[:,0] >= np.min(half_coords[:,0]),
                                                    x_coord[:,0] <= np.max(half_coords[:,0])))[0])
        y_inds_of_half=list(np.where(np.logical_and(y_coord[0,:] >= np.min(half_coords[:,1]),
                                                    y_coord[0,:] <= np.max(half_coords[:,1])))[0])
        for i_coord_x in x_inds_of_half:
            for j_coord_y in y_inds_of_half:
                coords_2d.append([x_coord[:,0][i_coord_x],y_coord[0,:][j_coord_y]])
                data_inside_half.append(data_thres[i_coord_x,j_coord_y])

        coords_2d=np.asarray(coords_2d)
        data_inside_half=np.asarray(data_inside_half)
        ind_inside_half_path=structures[i_str]['Paths'][ind_at_half].contains_points(coords_2d)
        x_data=coords_2d[ind_inside_half_path,0]
        y_data=coords_2d[ind_inside_half_path,1]
        int_data=data_inside_half[ind_inside_half_path]
        """
        Polygon calculations
        """
        half_polygon=Polygon(x=half_coords[:,0],
                             y=half_coords[:,1],
                             x_data=np.asarray(x_data),
                             y_data=np.asarray(y_data),
                             data=np.asarray(int_data),
                             test=test)

        structures[i_str]['Half path']=structures[i_str]['Paths'][ind_at_half]

        structures[i_str]['Half level']=half_level
        structures[i_str]['Centroid']=half_polygon.centroid
        structures[i_str]['Area']=half_polygon.area
        structures[i_str]['Intensity']=half_polygon.intensity
        structures[i_str]['Center of gravity']=half_polygon.center_of_gravity

        structures[i_str]['Polygon']=half_polygon
        structures[i_str]['Vertices']=half_polygon.vertices
        structures[i_str]['X coord']=half_polygon.x_data
        structures[i_str]['Y coord']=half_polygon.y_data
        structures[i_str]['Data']=half_polygon.data

        ellipse=FitEllipse(x=half_coords[:,0],
                           y=half_coords[:,1],
                           method=ellipse_method)
        structures[i_str]['Ellipse']=ellipse
        if not skip_gaussian:
            gaussian=FitGaussian(x=x_data,
                                 y=y_data,
                                 data=np.asarray(int_data))
            structures[i_str]['Gaussian']=gaussian
        else:
            structures[i_str]['Gaussian']=False
        if fit_shape=='ellipse':
            structure=ellipse
        elif fit_shape=='gaussian':
            structure=gaussian

        structures[i_str]['Axes length']=structure.axes_length
        structures[i_str]['Center']=structure.center
        structures[i_str]['Size']=structure.size
        structures[i_str]['Angle']=structure.angle
        structures[i_str]['Elongation']=structure.elongation

        if structures[i_str]['Axes length'][1]/structures[i_str]['Axes length'][0] < elongation_threshold:
            structures[i_str]['Angle']=np.nan

    n_str=len(structures)
    for i_str in range(n_str):
        rev_ind=n_str-1-i_str
        if (structures[rev_ind]['Size'][0] < str_size_lower_thres or
            structures[rev_ind]['Size'][1] < str_size_lower_thres):
            structures.pop(rev_ind)

    fitted_structures=[]
    for i_str in range(len(structures)):
        if structures[i_str]['Size'] is not None:
            fitted_structures.append(structures[i_str])

    structures=fitted_structures
    if test_result:
        #fig,ax=plt.subplots(figsize=(8.5/2.54, 8.5/2.54/1.62))
        fig,ax=plt.subplots(figsize=(10,10))
        ax.set_aspect(1.0)

        plt.contourf(x_coord, y_coord, data, levels=nlevel)
        plt.colorbar()
        if len(structures) > 0:
            #Parametric reproduction of the Ellipse
            R=np.arange(0,2*np.pi,0.01)
            for i_structure in range(len(structures)):
                structure=structures[i_structure]
                if structure['Half path'] is not None and structure['Ellipse'] is not None:
                    phi=structure['Angle']
                    a,b=structure['Ellipse'].axes_length

                    x=structure['Half path'].to_polygons()[0][:,0]
                    y=structure['Half path'].to_polygons()[0][:,1]
                    xx = structure['Center'][0] + \
                         a*np.cos(R)*np.cos(phi) - \
                         b*np.sin(R)*np.sin(phi)
                    yy = structure['Center'][1] + \
                         a*np.cos(R)*np.sin(phi) + \
                         b*np.sin(R)*np.cos(phi)

                    # a,b=structure['Ellipse'].axes_length
                    # ell=Ellipse(structure['Center'][1],a*2.,b*2.,phi)
                    # ell_coord=ell.get_verts()
                    # x=ell_coord[:,0]
                    # y=ell_coord[:,1]

                    plt.plot(x,y)    #Plot the half path polygon
                    plt.plot(xx,yy)  #Plot the ellipse
                    plt.plot([structure['Center'][0]-structure['Axes length'][0]*np.cos(structure['Angle']),
                              structure['Center'][0]+structure['Axes length'][0]*np.cos(structure['Angle'])],
                             [structure['Center'][1]-structure['Axes length'][0]*np.sin(structure['Angle']),
                              structure['Center'][1]+structure['Axes length'][0]*np.sin(structure['Angle'])],
                             color='magenta')

                    plt.scatter(structure['Centroid'][0],
                                structure['Centroid'][1], color='yellow')
                    plt.scatter(structure['Center of gravity'][0],
                                structure['Center of gravity'][1], color='red')

                    if save_data_for_publication:
                        exp_id=data_object.exp_id
                        time=data_object.coordinate('Time')[0][0,0]
                        wd=flap.config.get_all_section('Module NSTX_GPI')['Working directory']
                        filename=wd+'/'+str(exp_id)+'_'+str(time)+'_half_path_no.'+str(i_structure)+'.txt'
                        file1=open(filename, 'w+')
                        for i in range(len(x)):
                            file1.write(str(x[i])+'\t'+str(y[i])+'\n')
                        file1.close()

                        filename=wd+'/'+str(exp_id)+'_'+str(time)+'_fit_ellipse_no.'+str(i_structure)+'.txt'
                        file1=open(filename, 'w+')
                        for i in range(len(xx)):
                            file1.write(str(xx[i])+'\t'+str(yy[i])+'\n')
                        file1.close()

            plt.xlabel('Image x')
            plt.ylabel('Image y')
            plt.title(str(exp_id)+' @ '+str(data_object.coordinate('Time')[0][0,0]))
            plt.show()
            plt.pause(0.001)

        if save_data_for_publication:
            exp_id=data_object.exp_id
            time=data_object.coordinate('Time')[0][0,0]
            wd=flap.config.get_all_section('Module NSTX_GPI')['Working directory']
            filename=wd+'/'+str(exp_id)+'_'+str(time)+'_raw_data.txt'
            file1=open(filename, 'w+')
            for i in range(len(data[0,:])):
                string=''
                for j in range(len(data[:,0])):
                    string+=str(data[j,i])+'\t'
                string+='\n'
                file1.write(string)
            file1.close()

    return structures

def nstx_gpi_watershed_structure_finder(data_object=None,                       #Name of the FLAP.data_object
                                        exp_id='*',                             #Shot number (if data_object is not used)
                                        time=None,                              #Time when the structures need to be evaluated (when exp_id is used)
                                        sample=None,                            #Sample number where the structures need to be evaluated (when exp_id is used)
                                        spatial=False,                          #Calculate the results in real spatial coordinates
                                        pixel=False,                            #Calculate the results in pixel coordinates
                                        mfilter_range=5,                        #Range of the median filter

                                        threshold_method='otsu',
                                        threshold_level=None,                   #Threshold level over which it is considered to be a structure
                                                                                #if set, the value is subtracted from the data and contours are found after that.
                                                                                #Negative values are substituted with 0.
                                        ignore_side_structure=True,
                                        ignore_large_structure=True,
                                        test_result=False,                      #Test the result only (plot the contour and the found structures)
                                        test=False,                             #Test the contours and the structures before any kind of processing
                                        nlevel=51,                              #Number of contour levels for plotting

                                        save_data_for_publication=False,
                                        plot_full=False,
                                        ellipse_method='linalg',                #linalg (direct eigenfunction calculation),
                                        fit_shape='ellipse',                    #ellipse or gaussian
                                        str_size_lower_thres=4*0.00375,
                                        elongation_threshold=0.1,
                                        skip_gaussian=False
                                        # try_random_walker=False,
                                        ):

    """
    The method calculates the radial and poloidal sizes of the structures
    present in one from of the GPI image. It gathers the isosurface contour
    coordinates and determines the structures based on certain criteria. In
    principle no user input is necessary, the code provides a robust solution.
    The sizes are determined by fitting an ellipse onto the contour at
    half-height. The code returns the following list:
        a[structure_index]={'Paths':     [list of the paths, type: matplotlib.path.Path],
                            'Half path': [path at the half level of the structure]
                            'Levels':    [levels of the paths, type: list],
                            'Center':    [center of the ellipse in px,py or R,z coordinates, type: numpy.ndarray of two elements],
                            'Size':      [size of the ellipse in x and y direction or R,z direction, type: ]numpy.ndarray of two elements,
                            'Angle':      [angle of the ellipse compared to horizontal in radians, type: numpy.float64],
                            'Area':      [area of the polygon at the half level],
                            ('Ellipse':  [the entire ellipse object, returned if test_result is True, type: flap_nstx.tools.FitEllipse])
                            }
    """


    """
    ----------------
    READING THE DATA
    ----------------
    """

    if skip_gaussian and fit_shape=='gaussian':
        raise ValueError('If the gaussian fitting is skipped, the fitting shape cannot be gaussian...')

    if type(data_object) is str:
        data_object=flap.get_data_object_ref(data_object, exp_id=exp_id)
        if len(data_object.data.shape) != 2:
            raise IOError('The inpud data_object is not 2D. The method only processes 2D data.')

    if data_object is None:
        if (exp_id is None) or ((time is None) and (sample is None)):
            raise IOError('exp_id and time needs to be set if data_object is not set.')
        try:
            data_object=flap.get_data_object_ref('GPI', exp_id=exp_id)
        except:
            print('---- Reading GPI data ----')
            data_object=flap.get_data('NSTX_GPI', exp_id=exp_id, name='', object_name='GPI')

        if (time is not None) and (sample is not None):
            raise IOError('Either time or sample can be set, not both.')

        if time is not None:
            data_object=data_object.slice_data(slicing={'Time':time})
        if sample is not None:
            data_object=data_object.slice_data(slicing={'Sample':sample})

    try:
        data_object.data
    except:
        raise IOError('The input data object should be a flap.DataObject')

    if len(data_object.data.shape) != 2:
        raise TypeError('The frame dataobject needs to be a 2D object without a time coordinate.')

    if pixel:
        x_coord_name='Image x'
        y_coord_name='Image y'
    if spatial:
        x_coord_name='Device R'
        y_coord_name='Device z'

    x_coord=data_object.coordinate(x_coord_name)[0]
    y_coord=data_object.coordinate(y_coord_name)[0]

    #Filtering
    data = scipy.ndimage.median_filter(data_object.data, mfilter_range)
    levels=np.arange(nlevel)/(nlevel-1)*(data.max()-data.min())+data.min()

    if test:
        plt.cla()

    #Thresholding
    if threshold_level is not None:
        if data.max() < threshold_level:
            print('The maximum of the signal doesn\'t reach the threshold level.')
            return None
        data_thresholded = data - threshold_level
        data_thresholded[np.where(data_thresholded < 0)] = 0.
    else:
        data_thresholded = data
    """
    ----------------------
    Finding the structures
    ----------------------
    """

    if threshold_method == 'otsu':      #Histogram based, puts threshold between the two largest peaks
         thresh = threshold_otsu(data_thresholded)

    binary = data_thresholded > thresh

    if test:
         plt.contourf(binary)
    binary = np.asarray(binary, dtype='uint8')

    #distance_transformed = ndimage.distance_transform_edt(data_thresholded) #THIS IS UNNECESSARY AS THE STRUCTURES DO NOT HAVE DISTINCT BORDERS

    localMax = peak_local_max(copy.deepcopy(data_thresholded),
                              indices=False,
                              min_distance=5,
                              labels=binary)

    markers = scipy.ndimage.label(localMax,
                                  structure=np.ones((3, 3)))[0]

    labels = watershed(-data_thresholded, markers, mask=binary)

    # if try_random_walker:
    #     labels = random_walker(data_thresholded, markers, beta=10, mode='bf')

    structures=[]

    for label in np.unique(labels):
        # if the label is zero, we are examining the 'background'
        # so simply ignore it
        if label == 0:
            continue
        # otherwise, allocate memory for the label region and draw
        # it on the mask
        mask = np.zeros(data.shape, dtype="uint8")
        mask[labels == label] = 255
        	# detect contours in the mask and grab the largest one
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,  cv2.CHAIN_APPROX_SIMPLE)

        cnts = imutils.grab_contours(cnts)
        max_contour = max(cnts, key=cv2.contourArea)

        if spatial:
            max_contour=np.asarray([x_coord[max_contour[:,0,1],max_contour[:,0,0]],
                                    y_coord[max_contour[:,0,1],max_contour[:,0,0]]])

        from matplotlib.path import Path

        if max_contour.shape[0] != 1:
            indices=np.where(labels == label)
            codes=[Path.MOVETO]
            for i_code in range(1,len(max_contour.transpose()[:,0])):
                codes.append(Path.CURVE4)
            codes.append(Path.CLOSEPOLY)

            max_contour_looped=np.zeros([max_contour.shape[1]+1,max_contour.shape[0]])
            max_contour_looped[0:-1,:]=max_contour.transpose()
            max_contour_looped[-1,:]=max_contour[:,0]
            vertices=copy.deepcopy(max_contour_looped)

            full_polygon=Polygon(x=vertices[:,0],
                                 y=vertices[:,1],
                                 x_data=x_coord[indices],
                                 y_data=y_coord[indices],
                                 data=data[indices],
                                 test=test)

            structures.append({'Polygon':full_polygon,
                               'Vertices':full_polygon.vertices,
                               'Half path':Path(max_contour_looped,codes),
                               'X coord':full_polygon.x_data,
                               'Y coord':full_polygon.y_data,
                               'Data':full_polygon.data,
                               'Born':False,
                               'Died':False,
                               'Splits':False,
                               'Merges':False,
                               'Label':None,
                               'Parent':[],
                               'Child':[],
                               })


    if not test:
        plt.close()

    if test:
        print('Plotting structures')
        plt.cla()
        plt.set_aspect(1.0)

        plt.contourf(x_coord, y_coord, data, levels=levels)
        plt.contourf(x_coord, y_coord, mask+1, alpha=0.5)
        for struct in structures:
            x=struct['X coord']
            y=struct['y coord']
            plt.plot(x,y)

        plt.pause(0.001)



    #Calculate the ellipse and its properties for the half level contours
    for i_str in range(len(structures)):

                # Create x and y indices
        x_str = structures[i_str]['X coord']
        y_str = structures[i_str]['Y coord']
        data_str=structures[i_str]['Data']
        if not skip_gaussian:
            gaussian=FitGaussian(x=x_str,
                                     y=y_str,
                                     data=data_str)
        else:
            gaussian=None
        ellipse=FitEllipse(x=structures[i_str]['Vertices'][:,0],
                               y=structures[i_str]['Vertices'][:,1],
                               method=ellipse_method)

        if fit_shape=='ellipse':
            structure=ellipse
        elif fit_shape=='gaussian':
            structure=gaussian

        size=structure.size
        center=structure.center
        if np.iscomplex(size[0]) or np.iscomplex(size[1]):
            print('Size is complex')
            structure.set_invalid()

        if ignore_large_structure:
            if (size[0] > x_coord.max()-x_coord.min() or
                size[1] > y_coord.max()-y_coord.min()):
                print('Size is larger than the frame size.')
                structure.set_invalid()

        if ignore_side_structure:
            if (np.sum(structures[i_str]['X coord'] == x_coord.min()) != 0 or
                np.sum(structures[i_str]['X coord'] == x_coord.max()) != 0 or
                np.sum(structures[i_str]['Y coord'] == y_coord.min()) != 0 or
                np.sum(structures[i_str]['Y coord'] == y_coord.max()) != 0 or

                center[0] < x_coord.min() or
                center[0] > x_coord.max() or
                center[1] < y_coord.min() or
                center[1] > y_coord.max()
                ):
                print('Structure is at the border of the frame.')

                structure.set_invalid()
        if not skip_gaussian:
            structures[i_str]['Gaussian']=gaussian
        else:
            structures[i_str]['Gaussian']=None

        structures[i_str]['Ellipse']=ellipse

        #Structure parameters
        structures[i_str]['Axes length']=structure.axes_length
        structures[i_str]['Angle']=structure.angle
        structures[i_str]['Center']=structure.center
        structures[i_str]['Size']=structure.size
        structures[i_str]['Elongation']=structure.elongation

        #Polygon parameters
        polygon=structures[i_str]['Polygon']
        structures[i_str]['Intensity']=polygon.intensity
        structures[i_str]['Area']=polygon.area
        structures[i_str]['Centroid']=polygon.centroid
        structures[i_str]['Center of gravity']=polygon.center_of_gravity

        if structures[i_str]['Axes length'][1]/structures[i_str]['Axes length'][0] < elongation_threshold:
            structures[i_str]['Angle']=np.nan

    fitted_structures=[]
    for i_str in range(len(structures)):
        if structures[i_str]['Size'] is not None:
            fitted_structures.append(structures[i_str])

    n_str=len(structures)
    for i_str in range(n_str):
        rev_ind=n_str-1-i_str
        if (structures[rev_ind]['Size'][0] < str_size_lower_thres or
            structures[rev_ind]['Size'][1] < str_size_lower_thres):
            structures.pop(rev_ind)

    if test_result:
        import sys
        from matplotlib.gridspec import GridSpec
        def on_press(event):
            print('press', event.key)
            sys.stdout.flush()
            if event.key == 'x':
                plt.show()
                plt.pause(0.001)
                return 'Close'
        if plot_full:
            gs=GridSpec(2,2)
            fig,ax=plt.subplots(figsize=(10,10))
            fig.canvas.mpl_connect('key_press_event', on_press)
            plt.cla()
            ax.set_aspect(1.0)

            plt.subplot(gs[0,0])
            plt.contourf(x_coord, y_coord, data, levels=levels)
            plt.title('data')

            plt.subplot(gs[0,1])
            plt.contourf(x_coord, y_coord, data_thresholded)
            plt.title('thresholded')

            plt.subplot(gs[1,0])
            plt.contourf(x_coord, y_coord, binary)
            plt.title('binary')

            plt.subplot(gs[1,1])
            plt.contourf(x_coord, y_coord, labels)
            plt.title('labels')

            for i_x in range(2):
                for j_y in range(2):
                    if len(structures) > 0:
                        #Parametric reproduction of the Ellipse
                        plt.subplot(gs[i_x,j_y])
                        plot_structures_for_testing(structures=structures,
                                                    save_data_for_publication=save_data_for_publication,
                                                    data_object=data_object)
            plt.xlabel('Image x')
            plt.ylabel('Image y')
            plt.xlim([x_coord.min(),x_coord.max()])
            plt.ylim([y_coord.min(),y_coord.max()])
            plt.show()
            plt.pause(0.01)
        else:
            fig,ax=plt.subplots(figsize=(10,10))
            plt.cla()
            ax.set_aspect(1.0)
            plt.contourf(x_coord, y_coord, data, levels=levels)
            plt.title('data')

            if len(structures) > 0:
                #Parametric reproduction of the Ellipse
                plot_structures_for_testing(structures=structures,
                                            save_data_for_publication=save_data_for_publication,
                                            data_object=data_object)
            plt.xlabel('Image x')
            plt.ylabel('Image y')
            plt.xlim([x_coord.min(),x_coord.max()])
            plt.ylim([y_coord.min(),y_coord.max()])
            #plt.title(str(exp_id)+' @ '+str(data_object.coordinate('Time')[0][0,0]))
            plt.show()
            plt.pause(0.01)



        if save_data_for_publication:
            exp_id=data_object.exp_id
            time=data_object.coordinate('Time')[0][0,0]
            wd=flap.config.get_all_section('Module NSTX_GPI')['Working directory']
            filename=wd+'/'+str(exp_id)+'_'+str(time)+'_raw_data.txt'
            file1=open(filename, 'w+')
            for i in range(len(data[0,:])):
                string=''
                for j in range(len(data[:,0])):
                    string+=str(data[j,i])+'\t'
                string+='\n'
                file1.write(string)
            file1.close()

    return structures

def plot_structures_for_testing(structures=None,
                                save_data_for_publication=False,
                                data_object=None
                                ):
    if structures is None or data_object is None:
        raise ValueError('structures and data_object need to be set.')

    R=np.arange(0,2*np.pi,0.01)
    for i_str in range(len(structures)):
        structure=structures[i_str]
        phi=structure['Angle']
        try:
            a=structures[i_str]['Axes length'][0]
            b=structures[i_str]['Axes length'][1]
            x=structures[i_str]['Vertices'][:,0]
            y=structures[i_str]['Vertices'][:,1]

            xx = structure['Center'][0] + \
                 a*np.cos(R)*np.cos(phi) - \
                 b*np.sin(R)*np.sin(phi)
            yy = structure['Center'][1] + \
                 a*np.cos(R)*np.sin(phi) + \
                 b*np.sin(R)*np.cos(phi)

            plt.plot(x,y, color='magenta')    #Plot the half path polygon
            plt.plot(xx,yy)  #Plot the ellipse
            plt.plot([structure['Center'][0]-structure['Axes length'][0]*np.cos(structure['Angle']),
                      structure['Center'][0]+structure['Axes length'][0]*np.cos(structure['Angle'])],
                     [structure['Center'][1]-structure['Axes length'][0]*np.sin(structure['Angle']),
                      structure['Center'][1]+structure['Axes length'][0]*np.sin(structure['Angle'])],
                     color='magenta')

            plt.scatter(structure['Centroid'][0],
                        structure['Centroid'][1], color='yellow')
            plt.scatter(structure['Center'][0],
                        structure['Center'][1], color='red')
        except:
            pass

        if save_data_for_publication:
            exp_id=data_object.exp_id
            time=data_object.coordinate('Time')[0][0,0]
            wd=flap.config.get_all_section('Module NSTX_GPI')['Working directory']
            filename=wd+'/'+str(exp_id)+'_'+str(time)+'_half_path_no.'+str(i_str)+'.txt'
            file1=open(filename, 'w+')
            for i in range(len(x)):
                file1.write(str(x[i])+'\t'+str(y[i])+'\n')
            file1.close()

            filename=wd+'/'+str(exp_id)+'_'+str(time)+'_fit_ellipse_no.'+str(i_str)+'.txt'
            file1=open(filename, 'w+')
            for i in range(len(xx)):
                file1.write(str(xx[i])+'\t'+str(yy[i])+'\n')
            file1.close()
        #plt.title(str(exp_id)+' @ '+str(data_object.coordinate('Time')[0][0,0]))