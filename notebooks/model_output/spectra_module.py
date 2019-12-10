#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 12:03:15 2018

@author: seb
"""
import numpy as np
from numpy.fft import fftshift

def rfft_to_fft(image):
    '''
    Perform a RFFT on the image (2 or 3D) and return the absolute value in
    the same format as you would get with the fft (negative frequencies).
    This avoids ever having to have the full complex cube in memory.
    Inputs
    ------
    image : numpy.ndarray
        2 or 3D array.
    Outputs
    -------
    fft_abs : absolute value of the fft.
    '''

    ndim = len(image.shape)

    if ndim < 2 or ndim > 3:
        raise TypeError("Dimension of image must be 2D or 3D.")

    last_dim = image.shape[-1]

    fft_abs = np.abs(np.fft.rfftn(image))

    if ndim == 2:
        if last_dim % 2 == 0:
            fftstar_abs = fft_abs.copy()[:, -2:0:-1]
        else:
            fftstar_abs = fft_abs.copy()[:, -1:0:-1]

        fftstar_abs[1::, :] = fftstar_abs[:0:-1, :]

        return np.concatenate((fft_abs, fftstar_abs), axis=1)

def make_radial_arrays(shape, y_center=None, x_center=None):

    if y_center is None:
        y_center = np.floor(shape[0] / 2.).astype(int)
    else:
        y_center = int(y_center)

    if x_center is None:
        x_center = np.floor(shape[1] / 2.).astype(int)
    else:
        x_center = int(x_center)

    y = np.arange(-y_center, shape[0] - y_center)
    x = np.arange(-x_center, shape[1] - x_center)

    yy, xx = np.meshgrid(y, x, indexing='ij')

    return yy, xx

def make_radial_freq_arrays(shape):

    yfreqs = np.fft.fftshift(np.fft.fftfreq(shape[0]))
    xfreqs = np.fft.fftshift(np.fft.fftfreq(shape[1]))

    yy_freq, xx_freq = np.meshgrid(yfreqs, xfreqs, indexing='ij')

    return yy_freq[::-1], xx_freq[::-1]

def spec2d(U,logspacing=None,return_freqs=True,binsize=10):
    # compute 2d power spectra
    fft = fftshift(rfft_to_fft(U))
    psd2 = np.power(fft, 2.)
    
    
    yy, xx = make_radial_arrays(psd2.shape)
    dists = np.sqrt(yy**2 + xx**2)
    
    nbins = int(np.round(dists.max() / binsize) + 1)
    
    if return_freqs:
        max_bin = 0.8
    else:
        max_bin = dists.max()
    
    if return_freqs:
        min_bin = 1.0 / min(psd2.shape)
    else:
        min_bin = 0.5
            
    if return_freqs:
        yy_freq, xx_freq = make_radial_freq_arrays(psd2.shape)
        freqs_dist = np.sqrt(yy_freq**2 + xx_freq**2)
        zero_freq_val = freqs_dist[np.nonzero(freqs_dist)].min() / 2.
        freqs_dist[freqs_dist == 0] = zero_freq_val
        
    if logspacing:
        bins = np.logspace(np.log10(min_bin), np.log10(max_bin), nbins + 1)
    else:
        bins = np.linspace(min_bin, max_bin, nbins + 1)
    
    if return_freqs:
        dist_arr = freqs_dist
    else:
        dist_arr = dists
    
    from scipy.stats import binned_statistic
    ps1D, bin_edge, cts = binned_statistic(dist_arr.ravel(),
                                               psd2.ravel(),
                                               bins=bins,
                                               statistic=np.nanmean)
    bin_cents = (bin_edge[1:] + bin_edge[:-1]) / 2.
    return ps1D, bin_cents