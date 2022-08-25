#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 02:12:04 2022

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
import scipy

from skimage.feature import peak_local_max
from skimage.filters import threshold_otsu
from skimage.segmentation import watershed


def identify_structures(#General inputs
                        data_object=None,                       #Name of the FLAP.data_object
                        exp_id='*',                             #Shot number (if data_object is not used)
                        time=None,                              #Time when the structures need to be evaluated (when exp_id is used)
                        sample=None,                            #Sample number where the structures need to be evaluated (when exp_id is used)
                        spatial=False,                          #Calculate the results in real spatial coordinates
                        pixel=False,                            #Calculate the results in pixel coordinates
                        mfilter_range=5,                        #Range of the median filter

                        ignore_side_structure=False,
                        ignore_large_structure=False,

                        ellipse_method='linalg',                #linalg or skimage
                        fit_shape='ellipse',                    #ellipse or gaussian
                        str_size_lower_thres=4*0.00375,         #lower threshold for structures
                        elongation_threshold=0.1,               #Threshold for angle definition, the angle of circular structures cannot be determined.

                        str_finding_method='contour',          #Contour or watershed for now

                        #Contour segmentation inputs
                        nlevel=51,                           #The number of contours to be used for the calculation (default:ysize/mfilter_range=80//5)
                        levels=None,                            #Contour levels from an input and not from automatic calculation
                        threshold_level=None,                   #Threshold level over which it is considered to be a structure
                                                                #if set, the value is subtracted from the data and contours are found after that.
                                                                #Negative values are substituted with 0.
                        filter_struct=True,                     #Filter out the structures with less than filter_level number of contours
                        filter_level=None,                      #The number of contours threshold for structures filtering (default:nlevel//4)
                        remove_interlaced_structures=False,     #Filter out the structures which are interlaced. Only the largest structures is preserved, others are removed.

                        #Watershed specific inputs
                        threshold_method='otsu',

                        #Plotting and testing
                        plot_full=False,
                        plot_result=False,                      #Test the result only (plot the contour and the found structures)
                        test=False,                             #Test the contours and the structures before any kind of processing
                        save_data_for_publication=False,
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
        x_unit_name='[pix]'

        y_coord_name='Image y'
        y_unit_name='[pix]'

    elif spatial:
        x_coord_name='Device R'
        x_unit_name=''
        y_coord_name='Device z'
        y_unit_name=''
    else:
        raise TypeError('Cannot do pixel and spatial calculation at the same time.')

    x_coord=data_object.coordinate(x_coord_name)[0]
    y_coord=data_object.coordinate(y_coord_name)[0]

    structures=[]

    one_structure={'Polygon':None,  #Calculated during segmentation
                   'Half path':None,

                   'Vertices':None, #Calculated after segmentation
                   'X coord':None,
                   'Y coord':None,
                   'Data':None,

                   'Born':False,    #Calculated during tracking in analyze_gpi_structures
                   'Died':False,
                   'Splits':False,
                   'Merges':False,
                   'Label':None,
                   'Parent':[],
                   'Child':[],
                   }

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
        data_thresholded = data - threshold_level
        data_thresholded[np.where(data_thresholded < 0)] = 0.
    else:
        data_thresholded = data

    if str_finding_method == 'contour':

        if levels is None:
            levels=np.arange(nlevel)/(nlevel-1)*(data_thresholded.max()-data_thresholded.min())+data_thresholded.min()
        else:
            nlevel=len(levels)

        try:
            structure_contours=plt.contourf(x_coord,
                                            y_coord,
                                            data_thresholded, levels=levels)
        except:
            plt.cla()
            plt.close()
            print('Failed to create the contours for the structures.')
            return None

        prelim_structures=[]
        pre_one_struct={'Paths':[None],
                        'Levels':[None]}

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
                if len(prelim_structures) == 0:
                    for i_str in range(n_paths_cur_lev):
                        prelim_structures.append(copy.deepcopy(pre_one_struct))
                        prelim_structures[i_str]['Paths'][0]=cur_lev_paths[i_str]
                        prelim_structures[i_str]['Levels'][0]=levels[i_lev]
                else:
                    for i_cur in range(n_paths_cur_lev):
                        new_path=True
                        cur_path=cur_lev_paths[i_cur]
                        for j_prev in range(len(prelim_structures)):
                            if cur_path.contains_path(prelim_structures[j_prev]['Paths'][-1]):
                                prelim_structures[j_prev]['Paths'].append(cur_path)
                                prelim_structures[j_prev]['Levels'].append(levels[i_lev])
                                new_path=False
                        if new_path:
                            prelim_structures.append(copy.deepcopy(pre_one_struct))
                            prelim_structures[-1]['Paths'][0]=cur_path
                            prelim_structures[-1]['Levels'][0]=levels[i_lev]
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
            for i_str in range(len(prelim_structures)):
                if len(prelim_structures[i_str]['Levels']) > filter_level:
                    cut_structures.append(prelim_structures[i_str])
        prelim_structures=cut_structures

        if test:
            print('Plotting structures')
            plt.cla()
            for struct in prelim_structures:
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
        if len(prelim_structures) > 1:
            #Finding the paths at FWHM
            paths_at_half=[]
            for i_str in range(len(prelim_structures)):
                half_level=(prelim_structures[i_str]['Levels'][-1]+prelim_structures[i_str]['Levels'][0])/2.
                ind_at_half=np.argmin(np.abs(prelim_structures[i_str]['Levels']-half_level))
                paths_at_half.append(prelim_structures[i_str]['Paths'][ind_at_half])

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
                for i_str in range(len(prelim_structures)):
                    if i_str not in structures_to_be_removed:
                        cut_structures.append(prelim_structures[i_str])
                prelim_structures=cut_structures
        if test: print('N after removing interlaced:',len(prelim_structures))
        for i_str in range(len(prelim_structures)):

            str_levels=prelim_structures[i_str]['Levels']
            half_level=(str_levels[-1]+str_levels[0])/2.
            ind_at_half=np.argmin(np.abs(str_levels-half_level))
            n_path=len(prelim_structures[i_str]['Levels'])

            polygon_areas=np.zeros(n_path)
            polygon_centroids=np.zeros([n_path,2])
            polygon_intensities=np.zeros(n_path)

            for i_path in range(n_path):
                polygon=prelim_structures[i_str]['Paths'][i_path].to_polygons()
                if polygon != []:
                    polygon=polygon[0]
                    polygon_areas[i_path]=flap_nstx.tools.Polygon(polygon[:,0],polygon[:,1]).area
                    polygon_centroids[i_path,:]=flap_nstx.tools.Polygon(polygon[:,0],polygon[:,1]).centroid
                if i_path == 0:
                    polygon_intensities[i_path]=polygon_areas[i_path]*str_levels[i_path]
                else:
                    polygon_intensities[i_path]=(polygon_areas[i_path]-polygon_areas[i_path-1])*str_levels[i_path]

            half_coords=prelim_structures[i_str]['Paths'][ind_at_half].to_polygons()[0]

            coords_2d=[]
            data_inside_half=[]

            x_inds_of_half=list(np.where(np.logical_and(x_coord[:,0] >= np.min(half_coords[:,0]),
                                                        x_coord[:,0] <= np.max(half_coords[:,0])))[0])
            y_inds_of_half=list(np.where(np.logical_and(y_coord[0,:] >= np.min(half_coords[:,1]),
                                                        y_coord[0,:] <= np.max(half_coords[:,1])))[0])
            for i_coord_x in x_inds_of_half:
                for j_coord_y in y_inds_of_half:
                    coords_2d.append([x_coord[:,0][i_coord_x],y_coord[0,:][j_coord_y]])
                    data_inside_half.append(data_thresholded[i_coord_x,j_coord_y])

            coords_2d=np.asarray(coords_2d)
            data_inside_half=np.asarray(data_inside_half)

            try:
                ind_inside_half_path=prelim_structures[i_str]['Paths'][ind_at_half].contains_points(coords_2d)
            except:
                prelim_structures[i_str]['Size']=None
                continue

            x_data=coords_2d[ind_inside_half_path,0]
            y_data=coords_2d[ind_inside_half_path,1]
            int_data=data_inside_half[ind_inside_half_path]

            half_polygon=Polygon(x=half_coords[:,0],
                                 y=half_coords[:,1],
                                 x_data=np.asarray(x_data),
                                 y_data=np.asarray(y_data),
                                 data=np.asarray(int_data),
                                 test=test)

            structures.append(copy.deepcopy(one_structure))
            structures[-1]['Half path']=prelim_structures[i_str]['Paths'][ind_at_half]
            structures[-1]['Polygon']=half_polygon

    if str_finding_method == 'watershed':

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

                structures.append(copy.deepcopy(one_structure))
                structures[-1]['Half path']=Path(max_contour_looped,codes)
                structures[-1]['Polygon']=full_polygon

    #Calculate the ellipse and its properties for the half level contours

    for i_str in range(len(structures)):

        polygon=structures[i_str]['Polygon']

        structures[i_str]['Vertices']=polygon.vertices
        structures[i_str]['X coord']=polygon.x_data
        structures[i_str]['Y coord']=polygon.y_data
        structures[i_str]['Data']=polygon.data

        structures[i_str]['Centroid']=polygon.centroid
        structures[i_str]['Centroid radial']=polygon.centroid[0]
        structures[i_str]['Centroid poloidal']=polygon.centroid[1]

        structures[i_str]['Area']=polygon.area
        structures[i_str]['Intensity']=polygon.intensity

        structures[i_str]['Center of gravity']=polygon.center_of_gravity
        structures[i_str]['Center of gravity radial']=polygon.center_of_gravity[0]
        structures[i_str]['Center of gravity poloidal']=polygon.center_of_gravity[1]

        if fit_shape=='ellipse':
            ellipse=FitEllipse(x=polygon.x,
                               y=polygon.y,
                               method=ellipse_method)

            structures[i_str]['Ellipse']=ellipse
            fit_struct=ellipse

        elif fit_shape=='gaussian':
            gaussian=FitGaussian(x=polygon.x_data,
                                 y=polygon.y_data,
                                 data=polygon.data)
            structures[i_str]['Gaussian']=gaussian
            fit_struct=gaussian

        structures[i_str]['Axes length']=fit_struct.axes_length
        structures[i_str]['Axes length minor']=fit_struct.axes_length[0]
        structures[i_str]['Axes length major']=fit_struct.axes_length[1]

        structures[i_str]['Center']=fit_struct.center
        structures[i_str]['Center radial']=fit_struct.center[0]
        structures[i_str]['Center poloidal']=fit_struct.center[1]

        structures[i_str]['Position']=fit_struct.center
        structures[i_str]['Position radial']=fit_struct.center[0]
        structures[i_str]['Position poloidal']=fit_struct.center[1]

        structures[i_str]['Size']=fit_struct.size
        structures[i_str]['Size radial']=fit_struct.size[0]
        structures[i_str]['Size poloidal']=fit_struct.size[1]

        structures[i_str]['Angle']=fit_struct.angle
        structures[i_str]['Elongation']=fit_struct.elongation

        if structures[i_str]['Axes length'][1]/structures[i_str]['Axes length'][0] < elongation_threshold:
            structures[i_str]['Angle']=np.nan

        size=fit_struct.size
        if np.iscomplex(size[0]) or np.iscomplex(size[1]):
            print('Size is complex')
            fit_struct.set_invalid()

        if ignore_large_structure:
            if (size[0] > x_coord.max()-x_coord.min() or
                size[1] > y_coord.max()-y_coord.min()):
                print('Size is larger than the frame size.')
                fit_struct.set_invalid()

        center=fit_struct.center
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

                fit_struct.set_invalid()
    if test: print('N before size thres:',len(structures))
    n_str=len(structures)
    for i_str in range(n_str):
        rev_ind=n_str-1-i_str
        if (str_size_lower_thres is not None and
            structures[rev_ind]['Size'] is not None):

            if (structures[rev_ind]['Size'][0] < str_size_lower_thres or
                structures[rev_ind]['Size'][1] < str_size_lower_thres):
                if test:
                    print('sx',structures[rev_ind]['Size'][0])
                    print('sy',structures[rev_ind]['Size'][1])
                    print('thres',str_size_lower_thres)
                structures.pop(rev_ind)

    if test: print('N after size thres:',len(structures))

    fitted_structures=[]
    for i_str in range(len(structures)):
        if structures[i_str]['Size'] is not None:
            fitted_structures.append(structures[i_str])

    structures=fitted_structures
    if test: print('N after fitting:',len(structures))
    if plot_result:
        fig,ax=plt.subplots(figsize=(8.5/2.54, 8.5/2.54))
        #fig,ax=plt.subplots(figsize=(10,10))

        plt.contourf(x_coord, y_coord, data, levels=nlevel)
        ax.set_aspect(1.0)
        plt.colorbar()

    elif plot_full and str_finding_method == 'watershed':

        plt.cla()
        fig,axes=plt.subplots(2,2,
                              figsize=(10,10))

        ax=axes[0,0]
        ax.contourf(x_coord,
                    y_coord,
                    data,
                    levels=nlevel)
        ax.set_title('data')

        ax=axes[0,1]
        ax.contourf(x_coord,
                    y_coord,
                    data_thresholded)
        ax.set_title('thresholded')

        ax=axes[1,0]
        ax.contourf(x_coord,
                    y_coord,
                    binary)
        ax.set_title('binary')

        ax=axes[1,1]
        ax.contourf(x_coord,
                    y_coord,
                    labels)
        ax.set_title('labels')

        for ax in axes:
            ax.set_aspect(1.0)
            ax.set_xlabel('Image x')
            ax.set_ylabel('Image y')
            ax.set_xlim([x_coord.min(),x_coord.max()])
            ax.set_ylim([y_coord.min(),y_coord.max()])

    elif plot_full and str_finding_method == 'contour':
        raise ValueError('plot_full cannot be set when contour segmentation is performed.')

    else:
        pass

    if len(structures) > 0:
        #Parametric reproduction of the Ellipse
        R=np.arange(0,2*np.pi,0.01)
        for i_str in range(len(structures)):
            if (structures[i_str]['Half path'] is not None and
                structures[i_str]['Ellipse'] is not None):

                phi=structures[i_str]['Angle']
                a,b=structures[i_str]['Axes length']

                x_polygon=structures[i_str]['Polygon'].x
                y_polygon=structures[i_str]['Polygon'].y

                x_ellipse = (structures[i_str]['Center'][0] +
                             a*np.cos(R)*np.cos(phi) -
                             b*np.sin(R)*np.sin(phi))
                y_ellipse = (structures[i_str]['Center'][1] +
                             a*np.cos(R)*np.sin(phi) +
                             b*np.sin(R)*np.cos(phi))

                if plot_result or plot_full: #This plots the structures and the fit ellipses one by one
                    def _plot_ellipses_centers(ax_cur,
                                               x_polygon,
                                               y_polygon,
                                               x_ellipse,
                                               y_ellipse,
                                               structure):
                        ax_cur.plot(x_polygon,
                                    y_polygon)    #Plot the half path polygon

                        ax_cur.plot(x_ellipse,
                                    y_ellipse)  #Plot the ellipse

                        ax_cur.plot([structure['Center'][0]-structure['Axes length'][0]*np.cos(structure['Angle']),
                                     structure['Center'][0]+structure['Axes length'][0]*np.cos(structure['Angle'])],
                                    [structure['Center'][1]-structure['Axes length'][0]*np.sin(structure['Angle']),
                                     structure['Center'][1]+structure['Axes length'][0]*np.sin(structure['Angle'])],
                                    color='magenta')

                        ax_cur.scatter(structure['Centroid'][0],
                                       structure['Centroid'][1],
                                       color='yellow')

                        ax_cur.scatter(structure['Center of gravity'][0],
                                       structure['Center of gravity'][1],
                                       color='red')

                    if plot_result:
                        _plot_ellipses_centers(ax, x_polygon, y_polygon, x_ellipse, y_ellipse, structures[i_str])
                    if plot_full:
                        for ax_cur in axes:
                            _plot_ellipses_centers(ax_cur, x_polygon, y_polygon, x_ellipse, y_ellipse, structures[i_str])


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
                    for i in range(len(x_ellipse)):
                        file1.write(str(x_ellipse[i])+'\t'+str(y_ellipse[i])+'\n')
                    file1.close()

            if plot_result:
                ax.set_xlabel(x_coord_name + ' '+ x_unit_name)
                ax.set_ylabel(x_coord_name + ' '+ y_unit_name)
                ax.set_title(str(exp_id)+' @ '+str(data_object.coordinate('Time')[0][0,0]))
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