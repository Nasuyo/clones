#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 10:12:00 2019

@author: schroeder

Module containing everything that has to do with spherical harmonics.
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
def sh2sh(f_lm, From, To, lmax=False):
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
            'ewhGRACE' ... equivalent water height [m], but with lmax=120 and
                           filtered
        'gravity'... [m/s^2]
    """
    
    # np.random.seed(7)
    if lmax:
        f_lm = f_lm.pad(lmax)
        
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
    cfg.configure()
    # Load love numbers
    with open(cfg.PATHS['lln_path'] + '/lln.json') as f:
        lln = np.array(json.load(f))
    lln_k1 = lln[0:lmax+1,3] + 1
    # lln_k1_atmo = lln[0:lmax+1,3] - 1  # because atm. mass lowers surface potential --> relic from a wrong belief that Voigt et al (2016) were kind of right...
    lln_h = lln[0:lmax+1,1]
    
    if From in('pot', 'U', 'N'): 
        # First: Back to GRACE format: 'pot'
        if From == 'U':
            f_lm = f_lm * R / GM
        elif From == 'N':
             f_lm = f_lm / R
        elif From == 'ff':  # here, the vertical uplift part is kinda missing..
            f_lm = f_lm * c**2 * R / GM
             
        if To in('pot', 'U', 'N', 'GRACE'):
            # if atmo:  # in case the masses are above the clock...
                # lln_k1[1] = 0.02601378180 + 1  # from CM to CF
                # lln_k1_atmo[1] = 0.02601378180 - 1  # from CM to CF
                
                # f_lm = f_lm.to_array() * lln_k1_atmo.reshape(lmax+1,1) \
                #     / lln_k1.reshape(lmax+1,1)
                # f_lm = sh.SHCoeffs.from_array(f_lm)
        # Then: Into the desired format (unless it's 'pot')
            if To == 'U':
                f_lm = f_lm * GM / R
            elif To == 'N':
                f_lm = f_lm * R
            elif To == 'GRACE':
                # f_lm = ddk(f_lm.pad(100) * R, 3)
                f_lm = f_lm.pad(66) * R
                # filename = '/home/schroeder/CLONETS/data/ITSG-2018_n120_2007mean_sigma.nc'
                # sigma = shcoeffs_from_netcdf(filename)
                # sigma = sh2sh(sigma, 'pot', 'N')
                # f_lm_witherr = np.random.normal(f_lm.coeffs, sigma.coeffs)
                # f_lm.coeffs = f_lm_witherr
            # Done
            return f_lm
        elif To in('mass', 'sd', 'ewh', 'ewhGRACE'):
            # Then: Load love transformation
            lln_k1[1] = 0.02601378180 + 1  # from CM to CF
            pot2mass = (2*np.arange(0,lmax+1)+1) / lln_k1  # upweigh high degrees
            if To == 'ewhGRACE':
                f_lm = ddk(f_lm.pad(120), 3)
                pot2mass = pot2mass[:121]
                lmax = 120
            f_lm = f_lm.to_array() * pot2mass.reshape(lmax+1,1)
            # And then: Into the desired format (unless it's 'mass')
            if To == 'sd':
                f_lm = f_lm * R * rho_e / 3
            elif To == 'ewh' or To == 'ewhGRACE':
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
            return sh.SHCoeffs.from_array(f_lm_h - f_lm_pot)
        else:
            print('Choose a proper output unit!')
        
    elif From in('mass', 'sd', 'ewh'):
        # First: Back to mass format: 'mass'
        if From == 'sd':
            f_lm = f_lm / R / rho_e * 3
        elif From == 'ewh':
            f_lm = f_lm / R / rho_e * 3 * rho_w
            
        if To in('mass', 'sd', 'ewh', 'ewhGRACE'):
        # Then: Into the desired format (unless it's 'mass')
            if To == 'sd':
                f_lm = f_lm * R * rho_e / 3
            elif To == 'ewh':
                f_lm = f_lm * R * rho_e / 3 / rho_w
            elif To == 'ewhGRACE':  # TODO: correct this so that it is filtered in potential format
                f_lm = ddk(f_lm.pad(120) * R * rho_e / 3 / rho_w, 3)
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
#TODO: this is probs wrong; filter first, then convert to ewh!
                f_lm = ddk(sh.SHCoeffs.from_array(f_lm).pad(120) * R, 3)
                # filename = '/home/schroeder/CLONETS/data/ITSG-2018_n120_2007mean_sigma.nc'
                # sigma = shcoeffs_from_netcdf(filename)
                # sigma = sh2sh(sigma, 'pot', 'N')
                # f_lm_witherr = np.random.normal(f_lm.coeffs, sigma.coeffs)
                # f_lm.coeffs = f_lm_witherr
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
            f_lm_h = f_lm_h * GM / R**3
            # Done
            return sh.SHCoeffs.from_array(f_lm_h - f_lm_pot)
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
    
    elif From == 'GRACE' or From == 'ewhGRACE':
        print('Not possible with a DDK filter')  # TODO
        return f_lm
    
    else:
        print('Choose a proper input unit!')
        return f_lm
    
def Sigma_xx_2points_from_formal(sigma, lon1, lat1, lon2, lat2, lmax=False, gaussian=False):
    """Computes 2-point Covariance matrix from GRACE formal errors."""
    
    if lmax:
        sigma = sigma.pad(lmax)
    row = np.append(np.arange(sigma.lmax), sigma.lmax)
    colat1 = 90 - lat1  # [°]
    colat2 = 90 - lat2  # [°]
    # Earth radius
    R = sh.constant.r3_wgs84.value  # [m]
    # Legendre functions
    leg1 = sh.legendre.legendre(sigma.lmax, np.cos(colat1*np.pi/180))
    leg2 = sh.legendre.legendre(sigma.lmax, np.cos(colat2*np.pi/180))
    # sin and cos of m*lambda
    sin1 = np.tile(np.sin(row*lon1*np.pi/180), (sigma.lmax+1, 1))
    cos1 = np.tile(np.cos(row*lon1*np.pi/180), (sigma.lmax+1, 1))
    sin2 = np.tile(np.sin(row*lon2*np.pi/180), (sigma.lmax+1, 1))
    cos2 = np.tile(np.cos(row*lon2*np.pi/180), (sigma.lmax+1, 1))
    # Gaussian weights
    if gaussian:
        W = np.transpose(np.tile(np.array(
            [w for w in W_l(sigma.lmax, gaussian)]), (sigma.lmax+1, 1)))
    else:
        W = np.ones((sigma.lmax+1, sigma.lmax+1))
    # Jacobian F
    ul = R * leg1 * W * sin1  # upper left
    ur = R * leg1 * W * cos1  # upper right
    ll = R * leg2 * W * sin2  # lower left
    lr = R * leg2 * W * cos2  # lower right
    # Sigma_ll
    sigma_s2 = sigma.coeffs[1, :, :]**2  # upper left
    sigma_c2 = sigma.coeffs[0, :, :]**2  # lower right
    # Sigma_xx
    Sigma_xx = np.zeros((2, 2))
    for i in range(sigma.lmax+1):
        for j in range(sigma.lmax+1):
            F_ij = np.array([[ul[i, j], ur[i, j]], [ll[i, j], lr[i, j]]])
            Sigma_ll_ij = np.array([[sigma_s2[i, j], 0], [0, sigma_c2[i, j]]])
            Sigma_xx_ij = np.matmul(np.matmul(F_ij, Sigma_ll_ij), F_ij.transpose())
            # if np.any(np.isnan(np.sqrt(Sigma_xx_ij))):
            #     print(i, j)
            Sigma_xx = Sigma_xx + Sigma_xx_ij
            # if i == 2 and j == 2:
                # print(F_ij)
                # print(Sigma_ll_ij)
                # print(Sigma_xx_ij)
    # Sigma_xx = np.abs(Sigma_xx)
    # print(Sigma_xx)
    # print('sigma_N0: ', np.sqrt(Sigma_xx[0, 0]), ' [m]')
    # print('sigma_N1: ', np.sqrt(Sigma_xx[1, 1]), ' [m]')
    # alright, first step done! now: sigma_deltaN^2
    F2 = np.array([-1, 1])
    sigma_deltaN2 = np.matmul(np.matmul(F2, Sigma_xx), F2.transpose())
    sigma_deltaN = np.sqrt(sigma_deltaN2)
    # print("sigma_deltaN^2: ", sigma_deltaN2)
    # print("sigma_deltaN: ", sigma_deltaN, ' [m]')
    
    return sigma_deltaN

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

def read_SHCoeffs_errors(path, headerlines, k=True):
    """Reads an ascii file with spherical harmonics.
    
    The columns have to be (key,) L, M, C, S, sigmaC, sigmaS. Returns a
    dictionary with pyshtools.SHCoeffs objects for the coefficients and their
    sigmas.
    
    :type path: str
    :param path: path to the ascii file
    :type headerlines: int
    :param headerlines: number of header lines
    :type k: bool
    :param k: Is the first column a key?
    :rtype: dict with pyshtools.SHCoeffs
    :rparam: coeffs, sigma are the keys. The SHCoeffs are the variables
    """
    
    with open(path) as f:
        for i in range(headerlines):
            print(f.readline(), end='')
        rows = []
        for line in f:
            row = line.split()
            if k:
                row = [float(i) for i in row[1:]]
            else:
                row = [float(i) for i in row]
            row[0:2] = [int(i) for i in row[0:2]]
            rows.append(row)
    N = int(np.sqrt(len(rows)*2)) - 1
    c = np.zeros((N+1, N+1))
    s = np.zeros((N+1, N+1))
    sigmaC = np.zeros((N+1, N+1))
    sigmaS = np.zeros((N+1, N+1))
    for row in rows[:-1]:
        c[row[0], row[1]] = row[2]
        s[row[0], row[1]] = row[3]
        sigmaC[row[0], row[1]] = row[4]
        sigmaS[row[0], row[1]] = row[5]   
#        print(row[0], row[1])
    coeffs = sh.SHCoeffs.from_array(np.transpose(np.dstack((c, s)),
                                                 (2, 0, 1)))
    sigma = sh.SHCoeffs.from_array(np.transpose(np.dstack((sigmaC, sigmaS)),
                                                 (2, 0, 1)))
    return {'coeffs': coeffs, 'sigma': sigma}

def read_SHCoeffs(path, headerlines, k=True, lastline=False, D=False):
    """Reads an ascii file with spherical harmonics.
    
    The columns have to be key, L, M, C, S. Returns pyshtools.SHCoeffs object.
    
    :type path: str
    :param path: path to the ascii file
    :type headerlines: int
    :param headerlines: number of header lines
    :rtype: pyshtools.SHCoeffs
    :rparam: The SHCoeffs, size [2, N+1, N+1]
    """
    
    with open(path) as f:
        for i in range(headerlines):
            print(f.readline(), end='')
        rows = []
        for line in f:
            row = line.split()
            if D:
                row[3:] = [s.replace('D', 'E') for s in row[3:]]
            if k:
                row = [float(i) for i in row[1:]]
            else:
                row = [float(i) for i in row]
            row[0:2] = [int(i) for i in row[0:2]]
            rows.append(row)
    if lastline:
        rows.pop()
    N = int(np.sqrt(len(rows)*2)) - 1
    c = np.zeros((N+1, N+1))
    s = np.zeros((N+1, N+1))
    # print(N)
    # print(np.shape(c))
    # print(len(rows))
    for row in rows:
        # print(row[0], row[1])
        c[row[0], row[1]] = row[2]
        s[row[0], row[1]] = row[3]
        # if row[0] == 96 and row[1] == 96:
        #     pass
    return sh.SHCoeffs.from_array(np.transpose(np.dstack((c, s)), (2, 0, 1)))

def disc_autocovariance(x):
    """Discrete autocovariance function after (9.7) in 'Zeitreihenanalyse'."""
    n = len(x)
    m = n# int(n/10)
    C = np.zeros(m+1)
    x = x - np.mean(x)
    for k in range(m+1):
        C[k] = x[:n-k].dot(x[k:n]) / (n-k-1)
    
    return C

# TODO:
# def autocovariance_fct2matrix(C):
#     """Returns the corresponding autocovariance matrix to an input function."""
#     n = len(C)
#     m = int(n/10)
#     K = C / C[0]  # Covariance to Correlation function
#     Q = np.zeros((m, m))
#     for i in range(m):
#         for j in range(i):
#             Q[i, ]
    
def autocovariance_fct2neff(C, m=False):
    """Returns effective measurement number from autocovariance function."""
    n = len(C)
    if m == False:
        m = int(n/10)
    K = C / C[0]  # Covariance to Correlation function
    temp = 0
    for k in range(1, m):
        temp += (n-k) / n * K[k]
    neff = n / (1 + 2 * temp)
    print(temp)

    return neff

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

def read_cov(filename, lmax, d0=False, d1=False):
    """Read in spherical harmonic coefficients' covariance matrix."""
    
    dim = (lmax+1)**2 - 4
    if d0:
        dim += 1 
    if d1:
        dim += 3
    
    with open(filename) as f:
        for i in range(2):
            print(f.readline(), end='')
        C = np.empty((dim, dim))
        C[:] = np.nan
        for i, line in enumerate(f):
            row = line.split()
            row = [float(i) for i in row]
            C[i, i:] = np.array(row)
            C[i:, i] = np.array(row)
    return C

def variances_from_cov(C, lmax, order='order', d0=False, d1=False):
    """Get the variances from the covariance matrix."""
    
    Cd = np.diag(C)
    var = sh.SHCoeffs.from_zeros(lmax)
    i = 0
    for order in range(lmax+1):
        for degree in range(order, lmax+1):
            if degree==0 and d0==False or degree==1 and d1==False:
                continue
            var.coeffs[0, degree, order] = Cd[i]
            i += 1
            if order > 0:
                var.coeffs[1, degree, order] = Cd[i]
                i += 1
    return var

def degree_variances_from_variances(var, To='pot'):
    """Converts spherical harmonic coefficients to degree variances.
    
    :param var: variance coefficients
    :type var: pyshtools.SHCoeffs
    :param To: output unit
    :type To: str
    """
    
    try:
        R = sh.constants.Earth.wgs84.r3.value  # mean radius of the Earth [m]
    except:
        R = sh.constant.r3_wgs84.value  # mean radius of the Earth [m]
    kappa = np.zeros(var.lmax+1)
    
    for l in range(var.lmax+1):
        for m in range(l+1):
            kappa[l] = kappa[l] + var.coeffs[0, l, m] + var.coeffs[1, l, m]
    if To == 'N':
        kappa = kappa * R**2  # [m^2]
    return kappa

def degree_amplitudes_from_degree_variances(kappa, From, To, cumulative=False):
    """Converts degree variances to (cumulative) degree amplitudes.
    
    :param kappa: degree variances
    :type kappa: np.array of floats
    :param From: unit of the variances
    :type From: str
    :param To: unit of the amplitudes
    :type To: str
    :param cumulative: cumulative degree variances or not
    :type cumulative: boolean
    """
    
    N = len(kappa)
    if To == 'ewh' and From != 'ewh':
        with open('lln.json') as f:
            lln = np.array(json.load(f))
        lln_k1 = lln[:, 3] + 1
        lln_k1[1] = 0.02601378180 + 1  # from CM to CF
        geoid2ewh = np.array([5513.59/3/1025 * (2*i+1) / lln_k1[i] for i in range(1025)])
    
    if cumulative:
        if From == 'N':
            if To == 'N':
                sigma = np.array([np.sqrt(np.sum(kappa[1:i+1]))
                                  for i in range(N)])
            if To == 'ewh':
                sigma = np.zeros(N)
                for i in range(N):
                    for j in range(i+1):
                        sigma[i] += geoid2ewh[j]**2 * kappa[j]
                sigma = np.sqrt(sigma)     
        else:  # TODO
            print('Not yet implemented!')
    else:
        if From == 'N':
            if To == 'N':
                sigma = np.sqrt(kappa)
            if To == 'ewh':
                sigma = geoid2ewh[:N] * np.sqrt(kappa)
        else:  # TODO
            print('Not yet implemented!')
    return sigma

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

def W_l(N, r):
    '''A generator for W from Wahr et al. (1998).'''
    
    R = sh.constant.r3_wgs84.value / 1000  # [km]
    b = np.log(2) / (1-np.cos(r/R))
    l = 0
    while l < N+1:
        if l == 0:
            W = 1 #/ 2 / np.pi
        elif l == 1:
            W_1 = W
            W = W * ((1+np.e**(-2*b))/(1-np.e**(-2*b)) - 1/b)
        else:
            W, W_1 = W * (-(2*(l-1) + 1)/b) + W_1, W
        l += 1
        yield W

def gauss_filter(f_lm, r=350):
    """Gaussian filter for spherical harmonic coefficients.
    
    A simple filter after Wahr et al. (1998).
    
    :type f_lm: pyshtools.SHCoeffs
    :param f_lm: The SHCoeffs, size [2, N+1, N+1]
    :type r: float, optional
    :param r: Filter radius in km
    :rtype: pyshtools.SHCoeffs
    :rparam: The filtered SHCoeffs, size [2, N+1, N+1]
    """
    
        # das hier macht zwar keinen generator, aber ist iwie falsch...
        # W = np.zeros(N)
        # W[0] = 1 / 2 / np.pi
        # W[1] = W[0] * ((1+np.e**(-2*b))/(1-np.e**(-2*b)) - 1/b)
        # for l in range(2, N):
        #     W[l] =  W[l-1] * (-(2*l + 1)/b) + W[l-2]
        # return W        
    
    W = np.array([w for w in W_l(f_lm.lmax, r)])
    f_lm = f_lm.coeffs * W.reshape(1, len(W), 1)
    return sh.SHCoeffs.from_array(f_lm)

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