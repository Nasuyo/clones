#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 10:12:00 2019

@author: schroeder

Module that contains everything that has to do with spherical harmonics.
"""

# Imports ---------------------------------------------------------------------
import pyshtools as sh
import numpy as np
import xarray as xr
import json
import time
import scipy
from scipy.fftpack import rfft
from scipy.fftpack import fft
from clones import cfg
from clones.DDKfilter import DDKfilter

# -----------------------------------------------------------------------------
def sh2sh(f_lm, From, To):
    """Function to change unit of coefficients.
    
    Converter for spherical harmonic coefficients' form and unit.
    
    :param f_lm: coefficients
    :type f_lm: pyshtools.SHCoeffs
    :param From: input unit
    :type From: str
    :param To: output unit
    :type To: str
    :return: the transformed coefficients
    :rtype: pyshtools.SHCoeffs
    
    Possible units:
        'pot' ... dimensionless Stokes coeffs (e.g. GRACE L2)
            'U' ... geopotential [m^2/s^2]
            'N' ... geoid height [m]
            'GRACE' ... geoid height [m], but with lmax=120 and filtered
        'ff' ... fractional frequency [-]
        'h' .... elevation [m]
        'mass' ... dimensionless surface loading coeffs
            'sd' ... surface density [kg/m^2]
            'ewh' ... equivalent water height [m]
        'gravity'... [m/s^2]
    """
    
    if From == To:
        # print('Well, that was a hard one!')
        return f_lm
    
    # Constants
    lmax = f_lm.lmax  # maximum degree of the coefficients
    M = sh.constant.mass_wgs84.value  # mass of the Earth [m]
    GM = sh.constant.gm_wgs84.value  # gravitational constant times mass [m^2/s^2]
    R = sh.constant.r3_wgs84.value  # mean radius of the Earth [m]
    c = scipy.constants.c  # speed of light [m/s]
    rho_e = 3 * M / (4*np.pi * R**3)  # average density of the Earth [kg/m^3]
    rho_w = 1025  # average density of seawater [kg/m^3]
    
    # Load love numbers
    with open(cfg.PATHS['lln_path'] + '/lln.json') as f:
        lln = np.array(json.load(f))
    lln_k1 = lln[0:lmax+1,3] + 1
    lln_h = lln[0:lmax+1,1]
    
    if From in('pot', 'U', 'N'): 
        # First: Back to GRACE format: 'pot'
        if From == 'U':
            f_lm = f_lm * R / GM
        elif From == 'N':
             f_lm = f_lm / R
        elif From == 'ff':
            f_lm = f_lm * c**2 * R / GM
             
        if To in('pot', 'U', 'N', 'GRACE'):
        # Then: Into the desired format (unless it's 'pot')
            if To == 'U':
                f_lm = f_lm * GM / R
            elif To == 'N':
                f_lm = f_lm * R
            elif To == 'GRACE':
                f_lm = ddk(f_lm.pad(120), 3) * R
            # Done
            return f_lm
        elif To in('mass', 'sd', 'ewh'):
            # Then: Load love transformation
            lln_k1[1] = 0.02601378180 + 1  # from CM to CF
            pot2mass = (2*np.arange(0,lmax+1)+1) / lln_k1  # upweight high degrees
            f_lm = f_lm.to_array() * pot2mass.reshape(lmax+1,1)
            # And then: Into the desired format (unless it's 'mass')
            if To == 'sd':
                f_lm = f_lm * R * rho_e / 3
            elif To == 'ewh':
                f_lm = f_lm * R * rho_e / 3 / rho_w
            # Done
            return sh.SHCoeffs.from_array(f_lm)
        elif To == 'h':
            # Then: Load love transformation ~ pot -> mass -> def
            lln_k1[1] = 0.02601378180 + 1  # from CM to CF
            lln_h[1] = -0.2598639762  # from CM to CF
            pot2def = lln_h / lln_k1
            f_lm = sh.SHCoeffs.to_array(f_lm) * pot2def.reshape(lmax+1,1) * R
            # Done
            return sh.SHCoeffs.from_array(f_lm)
        elif To == 'ff':
            # Part that is changed by a potential change
            f_lm_pot = f_lm * GM / R / c**2
            # Part that is changed by a height change
            lln_k1[1] = 0.02601378180 + 1  # from CM to CF
            lln_h[1] = -0.2598639762  # from CM to CF
            pot2def = lln_h / lln_k1
            f_lm_h = sh.SHCoeffs.to_array(f_lm) * pot2def.reshape(lmax+1,1) * R
            f_lm_h = f_lm_h * GM / R**2 / c**2
            f_lm = sh.SHCoeffs.from_array(-f_lm_h) + f_lm_pot
            # Done
            return f_lm
        elif To == 'gravity':
            pot2grav = np.arange(lmax+1)+1
            f_lm_pot = f_lm.to_array() * pot2grav.reshape(lmax+1, 1)
            f_lm_pot = -f_lm_pot * GM / R**2
            lln_k1[1] = 0.02601378180 + 1  # from CM to CF
            lln_h[1] = -0.2598639762  # from CM to CF
            pot2def = lln_h / lln_k1
            f_lm_h = (sh.SHCoeffs.to_array(f_lm) * pot2def.reshape(lmax+1,1)
                      * R * (-2))
            f_lm_h = f_lm_h * GM / R**3
            return sh.SHCoeffs.from_array(-f_lm_h + f_lm_pot)
        else:
            print('Choose a proper output unit!')
        
    elif From in('mass', 'sd', 'ewh'):
        # First: Back to mass format: 'mass'
        if From == 'sd':
            f_lm = f_lm / R / rho_e * 3
        elif From == 'ewh':
            f_lm = f_lm / R / rho_e * 3 * rho_w
            
        if To in('mass', 'sd', 'ewh'):
        # Then: Into the desired format (unless it's 'mass')
            if To == 'sd':
                f_lm = f_lm * R * rho_e / 3
            elif To == 'ewh':
                f_lm = f_lm * R * rho_e / 3 / rho_w
            # Done
            return f_lm
        elif To in('pot', 'U', 'N', 'GRACE'):
        # Then: Load love transformation
            mass2pot = lln_k1 / (2*np.arange(0,lmax+1)+1)  # downweight high degrees
            f_lm = f_lm.to_array() * mass2pot.reshape(lmax+1,1)
            # And then: Into the desired format (unless it's 'pot')
            if To == 'U':
                f_lm = f_lm * GM / R
            elif To == 'N':
                f_lm = f_lm * R
            elif To == 'GRACE':
                f_lm = ddk(sh.SHCoeffs.from_array(f_lm).pad(120), 4) * R
                return f_lm
            # Done
            return sh.SHCoeffs.from_array(f_lm)
        elif To == 'h':
        # Then: Load love transformation ~ pot -> mass -> def
#            lln_h[1] = -0.2598639762  # from CM to CF
            mass2def = lln_h / (2*np.arange(0,lmax+1)+1)  # downweight high degrees
            f_lm = sh.SHCoeffs.to_array(f_lm) * mass2def.reshape(lmax+1,1) * R
            # Done
            return sh.SHCoeffs.from_array(f_lm)
        elif To == 'ff':
            # Part that is changed by a potential change
            mass2pot = lln_k1 / (2*np.arange(0,lmax+1)+1)  # downweight high degrees
            f_lm_pot = sh.SHCoeffs.to_array(f_lm) * mass2pot.reshape(lmax+1,1)
            f_lm_pot = f_lm_pot * GM / R / c**2
            # Part that is changed by a height change
            mass2def = lln_h / (2*np.arange(0,lmax+1)+1)  # downweight high degrees
            f_lm_h = (sh.SHCoeffs.to_array(f_lm) * mass2def.reshape(lmax+1,1)
                      * R)
            f_lm_h = f_lm_h * GM / R**2 / c**2
            # Done
            return sh.SHCoeffs.from_array(-f_lm_h + f_lm_pot)
        elif To == 'gravity':
            # Part that is changed by a potential change
            mass2pot = lln_k1 / (2*np.arange(lmax+1)+1)  # downweight high degrees
            f_lm_pot = sh.SHCoeffs.to_array(f_lm) * mass2pot.reshape(lmax+1, 1)
            pot2grav = np.arange(lmax+1)+1  # before this it was like for ff
            f_lm_pot = f_lm_pot * pot2grav.reshape(lmax+1, 1)
            f_lm_pot = -f_lm_pot * GM / R**2
            # Part that is changed by a height change
            mass2def = lln_h / (2*np.arange(0,lmax+1)+1)  # downweight high degrees
            f_lm_h = (sh.SHCoeffs.to_array(f_lm) * mass2def.reshape(lmax+1,1)
                      * R * (-2))
            f_lm_h = -f_lm_h * GM / R**3
            # Done
            return sh.SHCoeffs.from_array(-f_lm_h + f_lm_pot)
        else:
            print('Choose a proper output unit!')
       
    elif From == 'h':
        print('NOT YET IMPLEMENTED!')  # TODO
        return f_lm
    
    elif From == 'ff':
        print('NOT YET IMPLEMENTED!')  # TODO
        return f_lm
    
    elif From == 'gravity':
        print('NOT YET IMPLEMENTED!')  # TODO
        return f_lm
    
    elif From == 'GRACE':
        print('Not possible with a DDK filter')  # TODO
        return f_lm
    
    else:
        print('Choose a proper input unit!')
        return f_lm
    
def shcoeffs2netcdf(filename, coeffs, unit='', description=''):
    """Store a pyshtools.SHCoeffs instance into a netcdf file.
    
    :param filename: name of the saved file
    :type filename: str
    :param coeffs: the pyshtools.SHCoeffs instance
    :type coeffs: pyshtools.SHCoeffs
    :param unit: unit of the coeffs
    :type unit: str
    :param description: description of the data
    :type description: str
    """
    
    if not filename[-3:] == '.nc':
        filename += '.nc'
    
    ds = xr.Dataset()
    ds.attrs['creation_date'] = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    ds.coords['degree'] = ('degree', np.arange(coeffs.lmax+1))
    ds.coords['order'] = ('order', np.arange(coeffs.lmax+1))
    # c coeffs as lower triangular matrix
    c = coeffs.coeffs[0, :, :]
    # s coeffs as upper triangular matrix
    s = np.transpose(coeffs.coeffs[1, :, :])
    s = np.vstack([s[1:], s[0]])
    ds['coeffs'] = (('degree', 'order'), c + s)
    ds['coeffs'].attrs['description'] = description
    ds['coeffs'].attrs['unit'] = unit
    
    ds.to_netcdf(filename)
        
def shcoeffs_from_netcdf(filename):
    """Loads a pyshtools.SHCoeffs instance from a netcdf file.
    
    The coefficients need to be stored as a single 2D matrix with c as lower
    triangular matrix and s as upper triangular matrix as done with the function
    shcoeffs2netcdf.
    
    :param filename: the name of the netcdf file
    :type filename: str
    :rparam: the spherical harmonic coefficients
    :rtype: pyshtools.SHCoeffs
    """
    
    if not filename[-3:] == '.nc':
        filename += '.nc'
    
    ds = xr.open_dataset(filename)
    c = np.tril(ds.coeffs.data)
    s = np.triu(ds.coeffs.data, k=1)
    s = np.vstack([s[-1], s[:-1]])
    s = np.transpose(s)
    cs = np.array([c, s])
    coeffs = sh.SHCoeffs.from_array(cs)
    
    return coeffs

def disc_autocovariance(x):
    """Discrete autocovariance function after (9.7) in 'Zeitreihenanalyse'."""
    n = len(x)
    m = int(n/10)
    C = np.zeros(m+1)
    x = x - np.mean(x)
    for k in range(m+1):
        C[k] = x[:n-k].dot(x[k:n]) / (n-k-1)
    
    return C

def amplitude_spectrum(C, dt):
    """Amplitude spectrum = Scaled fourier transform of the discrete
    autocovariance function (power spectrum) after (9.52).
    """
    
    def hamming(j, m):
        return 0.54 + 0.46 * np.cos(j/m*np.pi)
    m = len(C) - 1
    P = np.zeros(m+1)
    j = np.arange(1, m)
    for k in range(m):
        P[k] = 4 * dt * (0.5 * (C[0]+(-1)**k*hamming(m, m)*C[m])
                         + hamming(j, m).dot(C[j]*np.cos(np.pi*k*j/m)))
    A = np.sqrt(np.abs(P) / m / dt)
    dv = 1 / 2 / m / dt  # delta ny
    vn = 1 / 2 / dt  # nyquist
    v = np.arange(0, vn, dv)
    
    return v, A

def time2freq(delta_t, signal):
    """Uses scipy's fftpack.fft to Fourier-transform the timeseries.
    
    :param delta_t: measurement sampling rate [s]
    :type delta_t: float
    :param signal: the signal
    :type signal: numpy.array of floats
    :rparam f: frequency vector [Hz]
    :rtype f: numpy.array of floats
    :rparam freq: absolute values of the amplitudes
    :rtype freq: numpy.array of floats
    """
    
    count = len(signal)
    Fs_nyquist = 1 / (2*delta_t)  # [Hz]
    anz_freq = int(np.floor(count/2)) + 1  # number of frequencies [Hz]
    f = Fs_nyquist * np.linspace(0, 1, anz_freq)  # frequency vector [Hz]
    freq = np.abs(fft(signal))[:anz_freq] / count * 2  # unit of the signal
    
    return f, freq

def sh_nm2i(n, m, nmax):
    """Returns an index for degree/order of sh coefficients.
    
    I = [0
         1, 4
         2, 5, 7
         3, 6, 8, 9]]
    """
    
    if m > n:
        raise ValueError('Order should not be larger than degree!')
    return round(m*(nmax+1) - (m*(m+1))/2 + n)

def sh_mat2vec(NM):
    """Converts a triangular matrix into a vector with the lower half.
    
    See sh_nm2i for the indices.
    """
    N = len(NM)-1
    v = np.zeros(sh_nm2i(N, N, N)+1)
    for n in range(N+1):
        for m in range(n+1):
            v[sh_nm2i(n, m, N)] = NM[n][m]
    return v

def sh_vec2mat(v, N):
    """Converts a vector to a triangular matrix.
    
    See sh_nm2i for the indices.
    """
    
    NM = np.zeros((N+1, N+1))
    for n in range(N+1):
        for m in range(n+1):
            NM[n][m] = v[sh_nm2i(n, m, N)]
    return NM

def ddk(coeffs, x):
    """Function for an easier call of the right filter."""
    
    if x == 1:
        DDK = DDKfilter(cfg.PATHS['lln_path'] + '/DDK/Wbd_2-120.a_1d14p_4');
        coeffs_filtered = DDK(coeffs)  # coeffs
    elif x == 2:
        DDK = DDKfilter(cfg.PATHS['lln_path'] + '/DDK/Wbd_2-120.a_1d13p_4');
        coeffs_filtered = DDK(coeffs)  # coeffs
    elif x == 3:
        DDK = DDKfilter(cfg.PATHS['lln_path'] + '/DDK/Wbd_2-120.a_1d12p_4');
        coeffs_filtered = DDK(coeffs)  # coeffs
    elif x == 4:
        DDK = DDKfilter(cfg.PATHS['lln_path'] + '/DDK/Wbd_2-120.a_5d11p_4');
        coeffs_filtered = DDK(coeffs)  # coeffs
    elif x == 5:
        DDK = DDKfilter(cfg.PATHS['lln_path'] + '/DDK/Wbd_2-120.a_1d11p_4');
        coeffs_filtered = DDK(coeffs)  # coeffs
    elif x == 6:
        DDK = DDKfilter(cfg.PATHS['lln_path'] + '/DDK/Wbd_2-120.a_5d10p_4');
        coeffs_filtered = DDK(coeffs)  # coeffs
    elif x == 7:
        DDK = DDKfilter(cfg.PATHS['lln_path'] + '/DDK/Wbd_2-120.a_1d10p_4');
        coeffs_filtered = DDK(coeffs)  # coeffs
    elif x == 8:
        DDK = DDKfilter(cfg.PATHS['lln_path'] + '/DDK/Wbd_2-120.a_5d9p_4');
        coeffs_filtered = DDK(coeffs)  # coeffs
    else:
        print('Number 1 to 8 please!')
        return coeffs
    return coeffs_filtered