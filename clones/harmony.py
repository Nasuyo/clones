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
import json
from clones import cfg

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
        'h' .... elevation [m]
        'mass' ... dimensionless surface loading coeffs
            'sd' ... surface density [kg/m^2]
            'ewh' ... equivalent water height [m]
        'gravity'... [m/s^2]
    """
    
    if From == To:
        print('Well, that was a hard one!')
        return f_lm
    
    # Constants
    N = f_lm.lmax  # maximum degree of the coefficients
    M = sh.constant.mass_wgs84.value  # mass of the Earth [m]
    GM = sh.constant.gm_wgs84.value  # gravitational constant times mass [m^2/s^2]
    R = sh.constant.r3_wgs84.value  # mean radius of the Earth [m]
    rho_e = 3 * M / (4*np.pi * R**3)  # average density of the Earth [kg/m^3]
    rho_w = 1025  # average density of seawater [kg/m^3]
    
    # Load love numbers
    with open(cfg.PATHS['lln_path'] + '/lln.json') as f:
        lln = np.array(json.load(f))
    lln_k1 = lln[0:N+1,3] + 1
    lln_h = lln[0:N+1,1]
    
    if From in('pot', 'U', 'N'): 
        # First: Back to GRACE format: 'pot'
        if From == 'U':
            f_lm = f_lm * R / GM
        elif From == 'N':
             f_lm = f_lm / R
             
        if To in('pot', 'U', 'N'):
        # Then: Into the desired format (unless it's 'pot')
            if To == 'U':
                f_lm = f_lm * GM / R
            elif To == 'N':
                 f_lm = f_lm * R
            # Done
            return f_lm
        elif To in('mass', 'sd', 'ewh'):
            # Then: Load love transformation
            lln_k1[1] = 0.02601378180 + 1  # from CM to CF
            pot2mass = (2*np.arange(0,N+1)+1) / lln_k1  # upweight high degrees
            f_lm = f_lm.to_array() * pot2mass.reshape(N+1,1)
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
            f_lm = sh.SHCoeffs.to_array(f_lm) * pot2def.reshape(N+1,1) * R
            # Done
            return sh.SHCoeffs.from_array(f_lm)
        elif To == 'gravity':
            print('NOT YET IMPLEMENTED!')  # TODO
            return f_lm
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
        elif To in('pot', 'U', 'N'):
        # Then: Load love transformation
            mass2pot = lln_k1 / (2*np.arange(0,N+1)+1)  # downweight high degrees
            f_lm = f_lm.to_array() * mass2pot.reshape(N+1,1)
            # And then: Into the desired format (unless it's 'pot')
            if To == 'U':
                f_lm = f_lm * GM / R
            elif To == 'N':
                f_lm = f_lm * R
            # Done
            return sh.SHCoeffs.from_array(f_lm)
        elif To == 'h':
        # Then: Load love transformation ~ pot -> mass -> def
#            lln_h[1] = -0.2598639762  # from CM to CF
            mass2def = lln_h / (2*np.arange(0,N+1)+1)  # downweight high degrees
            f_lm = sh.SHCoeffs.to_array(f_lm) * mass2def.reshape(N+1,1) * R
            # Done
            return sh.SHCoeffs.from_array(f_lm)
        elif To == 'gravity':
            print('NOT YET IMPLEMENTED!')  # TODO
            return f_lm
        else:
            print('Choose a proper output unit!')
       
    elif From == 'h':
        print('NOT YET IMPLEMENTED!')  # TODO
        return f_lm
    
    elif From == 'gravity':
        print('NOT YET IMPLEMENTED!')  # TODO
        return f_lm
    
    else:
        print('Choose a proper input unit!')
        return f_lm
    
