#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 12:19:50 2020

@author: mlampert
"""

import numpy as np
from imageio import imread

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from skimage import data
from skimage.registration import phase_cross_correlation, optical_flow_tvl1
from skimage.transform import warp_polar, rotate, rescale, warp
from skimage.util import img_as_float

from skimage.color import rgb2gray
from skimage.filters import difference_of_gaussians
from scipy.fftpack import fft2, fftshift


styled=True
if styled:
    plt.rc('font', family='serif', serif='Helvetica')
    labelsize=12.
    linewidth=0.5
    major_ticksize=2.
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

"""

Most of the codes found here are reliant on the documentation presented here:
https://scikit-image.org/docs/stable/auto_examples/registration/plot_register_rotation.html

The horse picture and related troubleshooting is referenced from here:
    https://sthoduka.github.io/imreg_fmt/
    https://sthoduka.github.io/imreg_fmt/docs/fourier-mellin-transform/
    
"""
from flap_nstx.gpi import generate_displaced_gaussian
def try_rotation(angle=25.,
                 scale=1.3,
                 zero_padding=True,
                 zero_padding_scale=2,
                 plot_fft_polar=True,
                 ):
    pdf=PdfPages('/Users/mlampert/work/NSTX_workspace/plots/try_rotation_image_warp.pdf')
    #image = data.retina()
    image_path='/Users/mlampert/work/NSTX_workspace/horse.png'
    image = imread(image_path)[:,:,0]
    image = img_as_float(image)
    image=generate_displaced_gaussian(displacement=[0,0], #[0,pol_disp_vec[i_pol]], 
                                      angle_per_frame=5, 
                                      size=[15,10], 
                                      size_velocity=[0.1,
                                                      0.1],
                                      sampling_time=2.5e-6,
                                      output_name='gaussian', 
                                      n_frames=3).data[1,:,:]
    
    radius=min(image.shape[0],image.shape[1])/2
    
    #rotated_path='/Users/mlampert/work/NSTX_workspace/horse_rot_scale.png'
    #rotated = imread(rotated_path)[:,:,0]
    if scale > 1:
        rotated = rescale(rotate(image, angle), scale)[0:image.shape[0],0:image.shape[1]]
        rotated = img_as_float(rotated)
    else:
        rotated=np.zeros(image.shape)
        rotated_small = rescale(rotate(image, angle), scale)
        shape=np.asarray(image.shape)
        
        small_shape=np.asarray(rotated_small.shape)
        x_addon=0
        y_addon=0
        if (shape[0]-(shape[0]-small_shape[0])//2)-(shape[0]-small_shape[0])//2 != small_shape[0]:
            x_addon=1
        if (shape[1]-(shape[1]-small_shape[1])//2)-(shape[1]-small_shape[1])//2 != small_shape[1]:
            y_addon=1
        
        rotated[(shape[0]-small_shape[0])//2+x_addon:shape[0]-(shape[0]-small_shape[0])//2,
                (shape[1]-small_shape[1])//2+y_addon:shape[1]-(shape[1]-small_shape[1])//2]=rotated_small
                
        rotated = img_as_float(rotated)
    

    image = difference_of_gaussians(image, 3, None)
    rotated = difference_of_gaussians(rotated, 3, None)

    shape=np.asarray(image.shape)
    if zero_padding:
        image_zero_padded=np.zeros(np.asarray(shape)*(2*zero_padding_scale+1))
        image_zero_padded[shape[0]*zero_padding_scale:shape[0]*(zero_padding_scale+1),
                          shape[1]*zero_padding_scale:shape[1]*(zero_padding_scale+1)]=image
        
        
        rotated_zero_padded=np.zeros(np.asarray(shape)*(2*zero_padding_scale+1))
        rotated_zero_padded[shape[0]*zero_padding_scale:shape[0]*(zero_padding_scale+1),
                            shape[1]*zero_padding_scale:shape[1]*(zero_padding_scale+1)]=rotated
    else:
        image_zero_padded=image
        rotated_zero_padded=rotated
    
    image_fft_rebuilt=np.absolute( np.fft.fftshift( np.fft.fftn(image_zero_padded, axes=[0,1])))
    rotated_fft_rebuilt=np.absolute( np.fft.fftshift( np.fft.fftn(rotated_zero_padded, axes=[0,1])))
    
    
    
    print(image_fft_rebuilt.shape, image.shape)
    
    image_fft_polar_log = warp_polar(image_fft_rebuilt, 
                                 radius=radius,
                                 scaling='log',
                                 )

    rotated_fft_polar_log = warp_polar(rotated_fft_rebuilt, 
                                   radius=radius,
                                   scaling='log',
                                   )
    
    image_fft_polar = warp_polar(image_fft_rebuilt, 
                                 radius=radius,
                                 )

    rotated_fft_polar = warp_polar(rotated_fft_rebuilt, 
                                   radius=radius,
                                   )
    image_polar=warp_polar(image, 
                           radius=radius,
                           )
    rotated_polar=warp_polar(rotated, 
                             radius=radius,
                             )
    print(image_fft_polar.shape)
    fig, axes = plt.subplots(4, 2, figsize=(8, 8))
    ax = axes.ravel()
    ax[0].set_title("Original")
    ax[0].imshow(image)
    ax[0].set_xlabel('x [pix]')
    ax[0].set_ylabel('y [pix]')
    
    ax[1].set_title("Rotated")
    ax[1].imshow(rotated)
    ax[1].set_xlabel('x [pix]')
    ax[1].set_ylabel('y [pix]')
    
    ax[2].set_title("Original fft")
    ax[2].imshow(image_fft_rebuilt)
    ax[2].set_xlabel('lx [1/pix]')
    ax[2].set_ylabel('ly [1/pix]')
    
    ax[3].set_title("Rotated fft")
    ax[3].imshow(rotated_fft_rebuilt)
    ax[3].set_xlabel('lx [1/pix]')
    ax[3].set_ylabel('ly [1/pix]')
    if plot_fft_polar:
        ax[4].set_title("Polar-Transformed Original")
        ax[4].imshow(image_fft_polar)
        ax[4].set_xlabel('lx [1/pix]')
        ax[4].set_ylabel('ly [1/pix]')
    
        
        ax[5].set_title("Polar-Transformed Rotated")
        ax[5].imshow(rotated_fft_polar)
        ax[5].set_xlabel('lx_log [1/pix]')
        ax[5].set_ylabel('ly_polar [1/pix]')
    else:
        ax[4].set_title("Polar-Transformed Original")
        ax[4].imshow(image_polar)
        ax[4].set_xlabel('x_log [log(pix)]')
        ax[4].set_ylabel('y_polar [degree]')
    
        
        ax[5].set_title("Polar-Transformed Rotated")
        ax[5].imshow(rotated_polar)
        ax[5].set_xlabel('x_log [log(pix)]')
        ax[5].set_ylabel('y_polar [degree]')
    
    plt.show()
    
    shift_rot, error, phasediff, cross_correlation = phase_cross_correlation(image_fft_polar, 
                                                                             rotated_fft_polar,
                                                                             upsample_factor=20)
    shiftr, shiftc = shift_rot[:2]
    
    shift_rot_log, error_log, phasediff, cross_correlation = phase_cross_correlation(image_fft_polar_log, 
                                                                                     rotated_fft_polar_log,
                                                                                     upsample_factor=20)
    shiftr_log, shiftc_log = shift_rot_log[:2]
    
    # Calculate scale factor from translation
    klog = image_fft_polar_log.shape[1] / np.log(radius)
    shift_scale_log = (np.exp(shiftc_log / klog))
    
    image_retransformed_small=rescale(rotate(rotated,-shiftr),1/shift_scale_log)

    image_retransformed=np.zeros(image.shape)
    x_addon=0
    y_addon=0
    
    print(image_retransformed.shape)
    print(image_retransformed_small.shape)
    
    ret_shape=image_retransformed.shape
    small_shape=image_retransformed_small.shape
    
    if (ret_shape[0]-small_shape[0])//2-ret_shape[0]+(ret_shape[0]-small_shape[0])//2 != -small_shape[0]:
        x_addon=1
        
    if (ret_shape[1]-small_shape[1])//2-ret_shape[1]+(ret_shape[1]-small_shape[1])//2 != -small_shape[1]:
        y_addon=1
        
    if shift_scale_log > 1:        
        image_retransformed[(ret_shape[0]-small_shape[0])//2+x_addon:ret_shape[0]-(ret_shape[0]-small_shape[0])//2,
                            (ret_shape[1]-small_shape[1])//2+y_addon:ret_shape[1]-(ret_shape[1]-small_shape[1])//2]=image_retransformed_small
    else:
        image_retransformed = image_retransformed_small[(small_shape[0]-ret_shape[0])//2+x_addon:small_shape[0]-(small_shape[0]-ret_shape[0])//2,
                                                        (small_shape[1]-ret_shape[1])//2+y_addon:small_shape[1]-(small_shape[1]-ret_shape[1])//2]
    
    x_size=image.shape[0]
    y_size=image.shape[1]
    
    print(image.shape,image_retransformed.shape)

    shift_tra, error, phasediff, cross_correlation = phase_cross_correlation(image, 
                                                                             image_retransformed,
                                                                             upsample_factor=20)
    
    shift_tra=np.asarray(shift_tra,dtype='int16')
    print(shift_tra,error, phasediff)
    
    if shift_tra[0] >= 0 and shift_tra[1] >= 0:
        image_retransformed[np.abs(shift_tra[0]):,
                            np.abs(shift_tra[1]):]=image_retransformed[0:x_size-np.abs(shift_tra[0]),
                                                                       0:y_size-np.abs(shift_tra[1])]
    if shift_tra[0] < 0 and shift_tra[1] >= 0:
        image_retransformed[0:x_size-np.abs(shift_tra[0]),
                            np.abs(shift_tra[1]):]=image_retransformed[np.abs(shift_tra[0]):,
                                                                       0:y_size-np.abs(shift_tra[1])]
        
    if shift_tra[0] < 0 and shift_tra[1] < 0:
        image_retransformed[0:x_size-np.abs(shift_tra[0]),
                            0:y_size-np.abs(shift_tra[1])]=image_retransformed[np.abs(shift_tra[0]):,
                                                                               np.abs(shift_tra[1]):]
        
    if shift_tra[0] >= 0 and shift_tra[1] < 0:
        image_retransformed[np.abs(shift_tra[0]):,
                            0:y_size-np.abs(shift_tra[1])]=image_retransformed[0:x_size-np.abs(shift_tra[0]),
                                                                               np.abs(shift_tra[1]):]


                        
    ax[6].set_title("Retransformed Original")
    ax[6].imshow(image_retransformed)
    ax[6].set_xlabel('x [pix]')
    ax[6].set_ylabel('y [pix]')
    
    ax[7].set_title("Retransformed minus Original")
    ax[7].imshow(image_retransformed-image)
    ax[7].set_xlabel('x [pix]')
    ax[7].set_ylabel('y [pix]')
    fig.tight_layout()
    
    pdf.savefig()
    pdf.close()
    
    print(f"Expected value for cc rotation in degrees: {angle}")
    print(f"Recovered value for cc rotation: {shiftr}")
    print(f"Recovered value for cc rotation log: {shiftr_log}")
    print()
    print(f"Expected value for scaling difference: {scale}")
    print(f"Recovered value for scaling difference: {shift_scale_log}")
    print(f"Recovered value ratio: {(scale-1)/(shift_scale_log-1)}")
def try_scaling():
    
    """
    
    ORIGINAL SCALING AND ROTATING ALGORITHM ON SCIKIT-IMAGE
    https://scikit-image.org/docs/stable/auto_examples/registration/plot_register_rotation.html
    
    """

    angle = 24
    scale = 2.0
    shiftr = 30
    shiftc = 15
    
    image_path='/Users/mlampert/work/NSTX_workspace/horse.png'
    image = imread(image_path)
    
    image = rgb2gray(image)
    translated = image[shiftr:, shiftc:]
    
    rotated = rotate(translated, angle)
    rescaled = rescale(rotated, scale)
    sizer, sizec = image.shape
    rts_image = rescaled[:sizer, :sizec]
    
    # Now try working in frequency domain
    # First, band-pass filter both images
    wimage = difference_of_gaussians(image, 5, 20)
    rts_wimage = difference_of_gaussians(rts_image, 5, 20)
    
    # window images
    #wimage = image * window('hann', image.shape)
    #rts_wimage = rts_image * window('hann', image.shape)
    
    # work with shifted FFT magnitudes
    image_fs = np.abs(fftshift(fft2(wimage)))
    rts_fs = np.abs(fftshift(fft2(rts_wimage)))
    
    # Create log-polar transformed FFT mag images and register
    shape = image_fs.shape
    radius = shape[0] // 2  # only take lower frequencies
    warped_image_fs = warp_polar(image_fs, 
                                 radius=radius, 
                                 output_shape=shape,
                                 scaling='log', 
                                 order=0)
    
    warped_rts_fs = warp_polar(rts_fs, 
                               radius=radius, 
                               output_shape=shape,
                               scaling='log', 
                               order=0)
    
    warped_image_fs = warped_image_fs[:shape[0] // 2, :]  # only use half of FFT
    warped_rts_fs = warped_rts_fs[:shape[0] // 2, :]
    
    shifts, error, phasediff = phase_cross_correlation(warped_image_fs,
                                                       warped_rts_fs,
                                                       upsample_factor=10)
    
    # Use translation parameters to calculate rotation and scaling parameters
    shiftr, shiftc = shifts[:2]
    recovered_angle = (360 / shape[0]) * shiftr
    klog = shape[1] / np.log(radius)
    shift_scale = np.exp(shiftc / klog)
    
    fig, axes = plt.subplots(3, 2, figsize=(8, 8))
    ax = axes.ravel()
    
    ax[0].set_title("Original")
    ax[0].imshow(image)
    ax[1].set_title("Rotated")
    ax[1].imshow(rts_image)
    
    ax[2].set_title("Original Image FFT\n(magnitude; zoomed)")
    center = np.array(shape) // 2
    ax[2].imshow(image_fs[center[0] - radius:center[0] + radius,
                          center[1] - radius:center[1] + radius],
                 cmap='magma')
    
    ax[3].set_title("Modified Image FFT\n(magnitude; zoomed)")
    ax[3].imshow(rts_fs[center[0] - radius:center[0] + radius,
                        center[1] - radius:center[1] + radius],
                 cmap='magma')
    
    ax[4].set_title("Log-Polar-Transformed\nOriginal FFT")
    ax[4].imshow(warped_image_fs, cmap='magma')
    ax[5].set_title("Log-Polar-Transformed\nModified FFT")
    ax[5].imshow(warped_rts_fs, cmap='magma')
    
    fig.suptitle('Working in frequency domain can recover rotation and scaling')
    plt.show()
    
    print(f"Expected value for cc rotation in degrees: {angle}")
    print(f"Recovered value for cc rotation: {recovered_angle}")
    print()
    print(f"Expected value for scaling difference: {scale}")
    print(f"Recovered value for scaling difference: {shift_scale}")
    
    
def try_optical_flow():
    
    """
    OPTICAL FLOW CALCULATION FOR EACH PIXEL IN AN IMAGE:
        Source:
            https://scikit-image.org/docs/stable/api/skimage.registration.html#rda1bf5cbeff5-3
    """
    
    # --- Load the sequence
    image0, image1, disp = data.stereo_motorcycle()
    
    # --- Convert the images to gray level: color is not supported.
    image0 = rgb2gray(image0)
    image1 = rgb2gray(image1)
    
    # --- Compute the optical flow
    v, u = optical_flow_tvl1(image0, image1)
    print(v.shape,u.shape)
    # --- Use the estimated optical flow for registration
    
    nr, nc = image0.shape
    
    row_coords, col_coords = np.meshgrid(np.arange(nr), np.arange(nc),
                                         indexing='ij')
    
    image1_warp = warp(image1, 
                       np.array([row_coords + v, col_coords + u]),
                       mode='nearest')
    
    # build an RGB image with the unregistered sequence
    seq_im = np.zeros((nr, nc, 3))
    seq_im[..., 0] = image1
    seq_im[..., 1] = image0
    seq_im[..., 2] = image0
    
    # build an RGB image with the registered sequence
    reg_im = np.zeros((nr, nc, 3))
    reg_im[..., 0] = image1_warp
    reg_im[..., 1] = image0
    reg_im[..., 2] = image0
    
    # build an RGB image with the registered sequence
    target_im = np.zeros((nr, nc, 3))
    target_im[..., 0] = image0
    target_im[..., 1] = image0
    target_im[..., 2] = image0
    
    # --- Show the result
    
    fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(5, 10))
    
    ax0.imshow(seq_im)
    ax0.set_title("Unregistered sequence")
    ax0.set_axis_off()
    
    ax1.imshow(reg_im)
    ax1.set_title("Registered sequence")
    ax1.set_axis_off()
    
    ax2.imshow(target_im)
    ax2.set_title("Target")
    ax2.set_axis_off()
    
    fig.tight_layout()
    plt.show()
