#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 00:08:02 2022

@author: mlampert
"""
import numpy as np
from skimage._shared.fft import fftmodule as fft
from skimage.registration._masked_phase_cross_correlation import _masked_phase_cross_correlation
from skimage.registration._phase_cross_correlation import _upsampled_dft, _compute_error, _compute_phasediff

def phase_cross_correlation_mod_ml(reference_image, moving_image,
                                   upsample_factor=1, space="real",
                                   return_error=True, reference_mask=None,
                                   moving_mask=None, overlap_ratio=0.3,
                                   fitting_range=5, polyfit_2D=None):
    """Efficient subpixel image translation registration by cross-correlation.

    This code gives the same precision as the FFT upsampled cross-correlation
    in a fraction of the computation time and with reduced memory requirements.
    It obtains an initial estimate of the cross-correlation peak by an FFT and
    then refines the shift estimation by upsampling the DFT only in a small
    neighborhood of that estimate by means of a matrix-multiply DFT.

    Parameters
    ----------
    reference_image : array
        Reference image.
    moving_image : array
        Image to register. Must be same dimensionality as
        ``reference_image``.
    upsample_factor : int, optional
        Upsampling factor. Images will be registered to within
        ``1 / upsample_factor`` of a pixel. For example
        ``upsample_factor == 20`` means the images will be registered
        within 1/20th of a pixel. Default is 1 (no upsampling).
        Not used if any of ``reference_mask`` or ``moving_mask`` is not None.
    space : string, one of "real" or "fourier", optional
        Defines how the algorithm interprets input data. "real" means
        data will be FFT'd to compute the correlation, while "fourier"
        data will bypass FFT of input data. Case insensitive. Not
        used if any of ``reference_mask`` or ``moving_mask`` is not
        None.
    return_error : bool, optional
        Returns error and phase difference if on, otherwise only
        shifts are returned. Has noeffect if any of ``reference_mask`` or
        ``moving_mask`` is not None. In this case only shifts is returned.
    reference_mask : ndarray
        Boolean mask for ``reference_image``. The mask should evaluate
        to ``True`` (or 1) on valid pixels. ``reference_mask`` should
        have the same shape as ``reference_image``.
    moving_mask : ndarray or None, optional
        Boolean mask for ``moving_image``. The mask should evaluate to ``True``
        (or 1) on valid pixels. ``moving_mask`` should have the same shape
        as ``moving_image``. If ``None``, ``reference_mask`` will be used.
    overlap_ratio : float, optional
        Minimum allowed overlap ratio between images. The correlation for
        translations corresponding with an overlap ratio lower than this
        threshold will be ignored. A lower `overlap_ratio` leads to smaller
        maximum translation, while a higher `overlap_ratio` leads to greater
        robustness against spurious matches due to small overlap between
        masked images. Used only if one of ``reference_mask`` or
        ``moving_mask`` is None.

    Returns
    -------
    shifts : ndarray
        Shift vector (in pixels) required to register ``moving_image``
        with ``reference_image``. Axis ordering is consistent with
        numpy (e.g. Z, Y, X)
    error : float
        Translation invariant normalized RMS error between
        ``reference_image`` and ``moving_image``.
    phasediff : float
        Global phase difference between the two images (should be
        zero if images are non-negative).

    References
    ----------
    .. [1] Manuel Guizar-Sicairos, Samuel T. Thurman, and James R. Fienup,
           "Efficient subpixel image registration algorithms,"
           Optics Letters 33, 156-158 (2008). :DOI:`10.1364/OL.33.000156`
    .. [2] James R. Fienup, "Invariant error metrics for image reconstruction"
           Optics Letters 36, 8352-8357 (1997). :DOI:`10.1364/AO.36.008352`
    .. [3] Dirk Padfield. Masked Object Registration in the Fourier Domain.
           IEEE Transactions on Image Processing, vol. 21(5),
           pp. 2706-2718 (2012). :DOI:`10.1109/TIP.2011.2181402`
    .. [4] D. Padfield. "Masked FFT registration". In Proc. Computer Vision and
           Pattern Recognition, pp. 2918-2925 (2010).
           :DOI:`10.1109/CVPR.2010.5540032`

    """
    if (reference_mask is not None) or (moving_mask is not None):
        return _masked_phase_cross_correlation(reference_image, moving_image,
                                               reference_mask, moving_mask,
                                               overlap_ratio)

    # images must be the same shape
    if reference_image.shape != moving_image.shape:
        raise ValueError("images must be same shape")

    # assume complex data is already in Fourier space
    if space.lower() == 'fourier':
        src_freq = reference_image
        target_freq = moving_image
    # real data needs to be fft'd.
    elif space.lower() == 'real':
        src_freq = np.fft.fftn(reference_image)
        target_freq = np.fft.fftn(moving_image)
    else:
        raise ValueError('space argument must be "real" of "fourier"')

    # Whole-pixel shift - Compute cross-correlation by an IFFT
    shape = src_freq.shape
    image_product = src_freq * target_freq.conj()
    cross_correlation = np.fft.ifftn(image_product)

    # Locate maximum
    maxima = np.unravel_index(np.argmax(np.abs(cross_correlation)),
                              cross_correlation.shape)
    midpoints = np.array([np.fix(axis_size / 2) for axis_size in shape])

    shifts = np.stack(maxima).astype(np.float64)
    shifts[shifts > midpoints] -= np.array(shape)[shifts > midpoints]

    if upsample_factor == 1:
        if return_error:
            src_amp = np.sum(np.real(src_freq * src_freq.conj()))
            src_amp /= src_freq.size
            target_amp = np.sum(np.real(target_freq * target_freq.conj()))
            target_amp /= target_freq.size
            CCmax = cross_correlation[maxima]
    # If upsampling > 1, then refine estimate with matrix multiply DFT
    else:
        # Initial shift estimate in upsampled grid
        shifts = np.round(shifts * upsample_factor) / upsample_factor
        upsampled_region_size = np.ceil(upsample_factor * 5)
        # Center of output array at dftshift + 1
        dftshift = np.fix(upsampled_region_size / 2)
        upsample_factor = np.array(upsample_factor, dtype=np.float64)
        # Matrix multiply DFT around the current shift estimate
        sample_region_offset = dftshift - shifts*upsample_factor
        cross_correlation_max = _upsampled_dft(image_product.conj(),
                                               upsampled_region_size,
                                               upsample_factor,
                                               sample_region_offset).conj()
        # Locate maximum and map back to original pixel grid

        max_index=np.asarray(np.unravel_index(cross_correlation_max.argmax(),
                                              cross_correlation_max.shape))
        slice_vec=[0,0]
        for ind_s1s2 in [0,1]:
            if max_index[ind_s1s2] < fitting_range:
                slice_vec[ind_s1s2]=slice(0,max_index[ind_s1s2]+fitting_range+1)
            elif max_index[0] > cross_correlation_max.shape[ind_s1s2]-fitting_range:
                slice_vec[ind_s1s2]=slice(max_index[ind_s1s2]-fitting_range,cross_correlation_max.shape[ind_s1s2]+1)
            else:
                slice_vec[ind_s1s2]=slice(max_index[ind_s1s2]-fitting_range,max_index[ind_s1s2]+fitting_range+1)
        area_max_index=tuple(slice_vec)
        #Finding the peak analytically
        try:
            if polyfit_2D is None:
                raise IOError('polyfit_2D should be the function from flap_nstx.tools.polyfit_2D')
            coeff=polyfit_2D(values=cross_correlation_max[area_max_index],order=2)
            index=np.zeros(2)
            index[0]=(2*coeff[2]*coeff[3]-coeff[1]*coeff[4])/(coeff[4]**2-4*coeff[2]*coeff[5])
            index[1]=(-2*coeff[5]*index[0]-coeff[3])/coeff[4]
        except:
            index=np.asarray([fitting_range,fitting_range])

        max_index_new=max_index+index-fitting_range

        CCmax = cross_correlation_max[tuple(max_index)]
        cross_corr=np.abs(cross_correlation)
        shifts = shifts + (np.stack(max_index_new).astype(np.float64) - dftshift) / upsample_factor

        if return_error:
            src_amp = np.sum(np.real(src_freq * src_freq.conj()))
            target_amp = np.sum(np.real(target_freq * target_freq.conj()))

    # If its only one row or column the shift along that dimension has no
    # effect. We set to zero.
    for dim in range(src_freq.ndim):
        if shape[dim] == 1:
            shifts[dim] = 0

    if return_error:
        # Redirect user to masked_phase_cross_correlation if NaNs are observed
        if np.isnan(CCmax) or np.isnan(src_amp) or np.isnan(target_amp):
            raise ValueError(
                "NaN values found, please remove NaNs from your "
                "input data or use the `reference_mask`/`moving_mask` "
                "keywords, eg: "
                "phase_cross_correlation(reference_image, moving_image, "
                "reference_mask=~np.isnan(reference_image), "
                "moving_mask=~np.isnan(moving_image))")

        return shifts, _compute_error(CCmax, src_amp, target_amp),\
                _compute_phasediff(CCmax), CCmax, cross_corr
    else:
        return shifts
