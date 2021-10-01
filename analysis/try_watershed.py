#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 22:54:06 2021

@author: mlampert
"""

# import the necessary packages

from skimage.segmentation import watershed
from skimage.feature import peak_local_max

from scipy import ndimage
import numpy as np

import imutils
import cv2
import os


#Flap imports
import flap
import flap_nstx
flap_nstx.register()
import flap_mdsplus
from flap_nstx.analysis import nstx_gpi_watershed_structure_finder

#Setting up FLAP
flap_mdsplus.register('NSTX_MDSPlus')    
thisdir = os.path.dirname(os.path.realpath(__file__))
fn = os.path.join(thisdir,"flap_nstx.cfg")
flap.config.read(file_name=fn) 
wd=flap.config.get_all_section('Module NSTX_GPI')['Working directory']  

import matplotlib.pyplot as plt

def try_watershed():
    
        # construct the argument parse and parse the arguments

    # load the image and perform pyramid mean shift filtering
    # to aid the thresholding step
    image = cv2.imread(wd+'/mints2.jpg')
    # scale_percent = 60 # percent of original size
    
    # image = cv2.resize(image, (1920,1080), interpolation = cv2.INTER_AREA)
        
    
    shifted = cv2.pyrMeanShiftFiltering(image, 21, 51)
    cv2.imshow("Input", image)
    cv2.startWindowThread()
    # convert the mean shift image to grayscale, then apply
    # Otsu's thresholding
    gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255,
    	cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    cv2.imshow("Thresh", thresh)
    
        # compute the exact Euclidean distance from every binary
    # pixel to the nearest zero pixel, then find peaks in this
    # distance map
    D = ndimage.distance_transform_edt(thresh)
    localMax = peak_local_max(D, indices=False, min_distance=20,
    	labels=thresh)
    # perform a connected component analysis on the local peaks,
    # using 8-connectivity, then appy the Watershed algorithm
    markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
    labels = watershed(-D, markers, mask=thresh)
    print("[INFO] {} unique segments found".format(len(np.unique(labels)) - 1))
    
    # loop over the unique labels returned by the Watershed
    # algorithm
    for label in np.unique(labels):
        # if the label is zero, we are examining the 'background'
        # so simply ignore it
        if label == 0:
            continue
        # otherwise, allocate memory for the label region and draw
        # it on the mask
        mask = np.zeros(gray.shape, dtype="uint8")
        mask[labels == label] = 255
        	# detect contours in the mask and grab the largest one
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
    		cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)
            
        	# draw a circle enclosing the object
        ((x, y), r) = cv2.minEnclosingCircle(c)
            
        #cv2.circle(image, (int(x), int(y)), int(r), (255, 255, 0), 2)
        cv2.putText(image, "#{}".format(label), (int(x) - 10, int(y)),
    		cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            # show the output image
        for (i,c) in enumerate(cnts):
            cv2.drawContours(image, [c], -1, (0,255,0), 2)
    cv2.imshow("Output", image)
    
    cv2.waitKey(0)
    
    cv2.destroyAllWindows()
    #for i in range (1,5):
    #    cv2.waitKey(1)
    
    
def try_watershed_skimage():
    from scipy import ndimage as ndi
    import matplotlib.pyplot as plt
    
    from skimage.morphology import disk
    from skimage import data
    from skimage.filters import rank, threshold_otsu
    from skimage.util import img_as_ubyte
    
    image = cv2.imread(wd+'/mints2.jpg')
    image = cv2.pyrMeanShiftFiltering(image, 21, 51)
    image = np.mean(image, axis=2, dtype='uint8')
    # denoise image
    thresh = cv2.threshold(image, 0, 255,
     	cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    #thresh = threshold_otsu(image)

    binary = image > thresh
    binary = np.asarray(binary, dtype='uint8')
    D = ndimage.distance_transform_edt(binary)
    localMax = peak_local_max(D, indices=False, min_distance=5, labels=binary)
    markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
    
    # find continuous region (low gradient -
    # where less than 10 for this image) --> markers
    # disk(5) is used here to get a more smooth image
    # markers = rank.gradient(binary, disk(5)) < 10
    # markers = ndi.label(markers)[0]
    
    shifted = cv2.pyrMeanShiftFiltering(image, 21, 51)
    # convert the mean shift image to grayscale, then apply
    # Otsu's thresholding
    gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
    binary = cv2.threshold(gray, 0, 255,
    	cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    
    
        # compute the exact Euclidean distance from every binary
    # pixel to the nearest zero pixel, then find peaks in this
    # distance map
    D = ndimage.distance_transform_edt(binary)
    localMax = peak_local_max(D, indices=False, min_distance=20,
    	labels=binary)
    # perform a connected component analysis on the local peaks,
    # using 8-connectivity, then appy the Watershed algorithm
    markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
    labels = watershed(-D, markers, mask=binary)
    print("[INFO] {} unique segments found".format(len(np.unique(labels)) - 1))
    
    #print(binary.shape)
    # process the watershed
    #labels = watershed(-D, markers, mask=binary)
    
    for label in np.unique(labels):
        # if the label is zero, we are examining the 'background'
        # so simply ignore it
        if label == 0:
            continue
        # otherwise, allocate memory for the label region and draw
        # it on the mask
        mask = np.zeros(gray.shape, dtype="uint8")
        mask[labels == label] = 255
        	# detect contours in the mask and grab the largest one
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,  cv2.CHAIN_APPROX_SIMPLE)
        
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)
            
        	# draw a circle enclosing the object
        ((x, y), r) = cv2.minEnclosingCircle(c)
            
        #cv2.circle(image, (int(x), int(y)), int(r), (255, 255, 0), 2)
        cv2.putText(image, "#{}".format(label), (int(x) - 10, int(y)),
    		cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            # show the output image
        for (i,c) in enumerate(cnts):
            cv2.drawContours(image, [c], -1, (0,255,0), 2)
    
    # display results
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 8), sharex=True, sharey=True, subplot_kw={'adjustable':'box'})
    ax = axes.ravel()
    
    ax[0].imshow(image, cmap=plt.cm.get_cmap("gray"), interpolation='nearest')
    ax[0].set_title("Original")
    
    ax[1].imshow(binary, cmap=plt.cm.get_cmap("gist_ncar"), interpolation='nearest')
    ax[1].set_title("Local Gradient")
    
    ax[2].imshow(D, cmap=plt.cm.get_cmap("gist_ncar"), interpolation='nearest')
    ax[2].set_title("Markers")
    
    ax[3].imshow(image, cmap=plt.cm.get_cmap("gray"), interpolation='nearest')
    ax[3].imshow(labels, cmap=plt.cm.get_cmap("gist_ncar"), interpolation='nearest', alpha=.7)
    ax[3].set_title("Segmented")
    
    for a in ax:
        a.axis('off')
    
    fig.tight_layout()
    plt.show()
    
    
def try_watershed_gpi(exp_id=139901,
                      time_range=[0.3245,0.3255],
                      time_point=0.32496,
                      threshold_coeff=1.0,
                      nlevel=51,
                      test_structures=True,
                      video_save=False,
                      threshold_bg_range={'x':[54,65],      #For the background subtraction, ROI where the bg intensity is calculated
                                          'y':[0,79]},
                      threshold_bg_multiplier=1.):

    d=flap.get_data('NSTX_GPI',exp_id=exp_id,
                    name='',
                    object_name='GPI')

            
        
    slicing={'Time':flap.Intervals(time_range[0],time_range[1])}
    d=flap.slice_data('GPI',exp_id=exp_id, 
                      slicing=slicing,
                      output_name='GPI_SLICED_FULL')    

    object_name_str_size='GPI_SLICED_FULL'
    object_name_str_vel='GPI_SLICED_FULL'
    
    #Normalize data for size calculation
    slicing_for_filtering={'Time':flap.Intervals(time_range[0]-1/1e3*10,
                                                 time_range[1]+1/1e3*10)}
    flap.slice_data('GPI',
                    exp_id=exp_id,
                    slicing=slicing_for_filtering,
                    output_name='GPI_SLICED_FOR_FILTERING')


    normalizer_object_name='GPI_LPF_INTERVAL'

    norm_obj=flap.filter_data('GPI_SLICED_FOR_FILTERING',
                                 exp_id=exp_id,
                                 coordinate='Time',
                                 options={'Type':'Lowpass',
                                          'f_high':1e3,
                                          'Design':'Elliptic'},
                                 output_name=normalizer_object_name)
    
    norm_obj.data=np.flip(norm_obj.data,axis=0)
    norm_obj=flap.filter_data(normalizer_object_name,
                                 exp_id=exp_id,
                                 coordinate='Time',
                                 options={'Type':'Lowpass',
                                          'f_high':1e3,
                                          'Design':'Elliptic'},
                                 output_name=normalizer_object_name)
    
    norm_obj.data=np.flip(norm_obj.data,axis=0)                
    coefficient=flap.slice_data(normalizer_object_name,
                                exp_id=exp_id,
                                slicing=slicing,
                                output_name='GPI_GAS_CLOUD').data 
    
    data_obj=flap.get_data_object(object_name_str_size)
    data_obj.data = data_obj.data/coefficient
    flap.add_data_object(data_obj, 'GPI_SLICED_DENORM_STR_SIZE')
    object_name_str_size='GPI_SLICED_DENORM_STR_SIZE'
    
    
    thres_obj_str_size=flap.slice_data(object_name_str_size,
                                       summing={'Image x':'Mean',
                                                'Image y':'Mean'},
                                                output_name='GPI_SLICED_TIMETRACE')
    intensity_thres_level_str_size=np.sqrt(np.var(thres_obj_str_size.data))*threshold_coeff+np.mean(thres_obj_str_size.data)
    thres_obj_str_vel=flap.slice_data(object_name_str_size,
                                       summing={'Image x':'Mean',
                                                'Image y':'Mean'},
                                                output_name='GPI_SLICED_TIMETRACE')
    intensity_thres_level_str_vel=np.sqrt(np.var(thres_obj_str_vel.data))*threshold_coeff+np.mean(thres_obj_str_vel.data)
    
    intensity_thres_level_str_vel=threshold_bg_multiplier*np.mean(flap.slice_data(object_name_str_size, 
                                                                                  slicing={'Image x':flap.Intervals(threshold_bg_range['x'][0],
                                                                                                                    threshold_bg_range['x'][1]),
                                                                                           'Image y':flap.Intervals(threshold_bg_range['y'][0],
                                                                                                                    threshold_bg_range['y'][1])}).data)
    
    if video_save:
        import matplotlib
        current_backend=matplotlib.get_backend()
        matplotlib.use('agg')
        
    for tp in time_point:

        slicing_frame1={'Time':tp}    
        frame1_vel=flap.slice_data(object_name_str_size,
                                   exp_id=exp_id,
                                   slicing=slicing_frame1, 
                                   output_name='GPI_FRAME_1_SIZE')
        
        
        structures1_vel=nstx_gpi_watershed_structure_finder(data_object='GPI_FRAME_1_SIZE',                       #Name of the FLAP.data_object
                                                            exp_id=exp_id,                             #Shot number (if data_object is not used)
                                                            spatial=True,                          #Calculate the results in real spatial coordinates
                                                            pixel=False,                            #Calculate the results in pixel coordinates
                                                            mfilter_range=5,                        #Range of the median filter
                                                            
                                                            threshold_method='otsu',
                                                            threshold_level=intensity_thres_level_str_vel,                   #Threshold level over which it is considered to be a structure
                                                                                                #if set, the value is subtracted from the data and contours are found after that. 
                                                                                                        #Negative values are substituted with 0.
                                                                                                    
                                                            test_result=test_structures,                      #Test the result only (plot the contour and the found structures)
                                                            test=False,                             #Test the contours and the structures before any kind of processing
                                                            nlevel=51,                              #Number of contour levels for plotting
                                                                                                    
                                                            save_data_for_publication=False,)
        try:
            if structures1_vel == 'Close':
                break
        except:
            pass
        if video_save:
            fig = plt.gcf()
            fig.canvas.draw()
            # Get the RGBA buffer from the figure
            w,h = fig.canvas.get_width_height()
            buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            if buf.shape[0] == h*2 * w*2 * 3:
                buf.shape = ( h*2, w*2, 3 ) #ON THE MAC'S INTERNAL SCREEN, THIS COULD FAIL, NEEDS A MORE ELEGANT FIX
            else:
                buf.shape = ( h, w, 3 )
            buf = cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR)
            try:
                video
            except NameError:
                
                height = buf.shape[0]
                width = buf.shape[1]
                video_codec_code='mp4v'
                filename='NSTX_GPI_'+str(exp_id)+'fit_structures_watershed_trial.mp4'
                video = cv2.VideoWriter(filename,  
                                        cv2.VideoWriter_fourcc(*video_codec_code), 
                                        float(24), 
                                        (width,height),
                                        isColor=True)
            video.write(buf)
    if (video_save):
        matplotlib.use(current_backend)
        cv2.destroyAllWindows()
        video.release()  
        del video