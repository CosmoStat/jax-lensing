# -*- coding: utf-8 -*-
"""
Code adapted from lenspack (Austin Peel, Cosmostat) for jax compatibiliy.
https://github.com/austinpeel/lenspack/blob/master/lenspack/image/inversion.py
"""

import jax.numpy as np
import jax
from jax import random

def ks93(g1, g2):
    """Direct inversion of weak-lensing shear to convergence.
    This function is an implementation of the Kaiser & Squires (1993) mass
    mapping algorithm. Due to the mass sheet degeneracy, the convergence is
    recovered only up to an overall additive constant. It is chosen here to
    produce output maps of mean zero. The inversion is performed in Fourier
    space for speed.
    Parameters
    ----------
    g1, g2 : array_like
        2D input arrays corresponding to the first and second (i.e., real and
        imaginary) components of shear, binned spatially to a regular grid.
    Returns
    -------
    kE, kB : tuple of numpy arrays
        E-mode and B-mode maps of convergence.
    Raises
    ------
    AssertionError
        For input arrays of different sizes.
    See Also
    --------
    bin2d
        For binning a galaxy shear catalog.
    Examples
    --------
    >>> # (g1, g2) should in practice be measurements from a real galaxy survey
    >>> g1, g2 = 0.1 * np.random.randn(2, 32, 32) + 0.1 * np.ones((2, 32, 32))
    >>> kE, kB = ks93(g1, g2)
    >>> kE.shape
    (32, 32)
    >>> kE.mean()
    1.0842021724855044e-18
    """
    # Check consistency of input maps
    assert g1.shape == g2.shape

    # Compute Fourier space grids
    (nx, ny) = g1.shape
    k1, k2 = np.meshgrid(np.fft.fftfreq(ny), np.fft.fftfreq(nx))

    # Compute Fourier transforms of g1 and g2
    g1hat = np.fft.fft2(g1)
    g2hat = np.fft.fft2(g2)

    # Apply Fourier space inversion operator
    p1 = k1 * k1 - k2 * k2
    p2 = 2 * k1 * k2
    k2 = k1 * k1 + k2 * k2
    #k2[0, 0] = 1  # avoid division by 0
    k2 = jax.ops.index_update(k2, jax.ops.index[0, 0], 1.) # avoid division by 0
    kEhat = (p1 * g1hat + p2 * g2hat) / k2
    kBhat = -(p2 * g1hat - p1 * g2hat) / k2


    # Transform back to pixel space
    kE = np.fft.ifft2(kEhat).real
    kB = np.fft.ifft2(kBhat).real

    return kE, kB

def ks93inv(kE, kB):
    """Direct inversion of weak-lensing convergence to shear.
    This function provides the inverse of the Kaiser & Squires (1993) mass
    mapping algorithm, namely the shear is recovered from input E-mode and
    B-mode convergence maps.
    Parameters
    ----------
    kE, kB : array_like
        2D input arrays corresponding to the E-mode and B-mode (i.e., real and
        imaginary) components of convergence.
    Returns
    -------
    g1, g2 : tuple of numpy arrays
        Maps of the two components of shear.
    Raises
    ------
    AssertionError
        For input arrays of different sizes.
    See Also
    --------
    ks93
        For the forward operation (shear to convergence).
    """
    # Check consistency of input maps
    assert kE.shape == kB.shape

    # Compute Fourier space grids
    (nx, ny) = kE.shape
    k1, k2 = np.meshgrid(np.fft.fftfreq(ny), np.fft.fftfreq(nx))

    # Compute Fourier transforms of kE and kB
    kEhat = np.fft.fft2(kE)
    kBhat = np.fft.fft2(kB)

    # Apply Fourier space inversion operator
    p1 = k1 * k1 - k2 * k2
    p2 = 2 * k1 * k2
    k2 = k1 * k1 + k2 * k2
    #k2[0, 0] = 1  # avoid division by 0
    k2 = jax.ops.index_update(k2, jax.ops.index[0, 0], 1) # avoid division by 0
    g1hat = (p1 * kEhat - p2 * kBhat) / k2
    g2hat = (p2 * kEhat + p1 * kBhat) / k2

    # Transform back to pixel space
    g1 = np.fft.ifft2(g1hat).real
    g2 = np.fft.ifft2(g2hat).real

    return g1, g2

if __name__=='__main__':

    key = random.PRNGKey(0)

    # (g1, g2) should in practice be measurements from a real galaxy survey
    g1, g2 = 0.1 * random.normal(key, (2, 32, 32)) + 0.1 * np.ones((2, 32, 32))
    #g1, g2 = 0.1 * np.random.randn(2, 32, 32) + 0.1 * np.ones((2, 32, 32))
    kE, kB = ks93(g1, g2)
    print(np.mean(kE))

    # Computing shear from convergence
    gamma1, gamma2 = ks93inv(kE, kB)
    print('gamma1.shape', gamma1.shape)
