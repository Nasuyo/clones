#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 10:28:55 2019

@author: schroeder
"""

# Imports ---------------------------------------------------------------------
from clones import harmony, cfg, utils
import time
from time import gmtime, strftime
import datetime
import numpy as np
import scipy.constants
import pyshtools as sh
import colorednoise as cn
import astropy
import geopy
import geopy.distance
import os
import shutil
import xarray as xr
import json
import geojson
import contextily as ctx
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import salem
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pygmt
from shapely.geometry import Point, LineString, Polygon
import copy

# Classes and functions -------------------------------------------------------

# TODO: Tests
# TODO: knotenpunkte ohne uhr

class Clock():
    """An optical clock.
    
    A class for an optical clock entity with all its nessecary information such
    as location, stability, drift, gepotential and elevation timeseries, ...
    
    :param lat: latitude of the clock location
    :type lat: float
    :param lon: longitude of the clock location
    :type lon: float
    :param location: name of (usually) the location of the clock
    :type location: str
    :param country: Country of the location
    :type country: str
    :param path: directory of the clock
    :type path: str
    :param links: fibre links to other clocks
    :type links: list of Clocks
    """
    
    def __init__(self, location='', lat=0, lon=0, country=None, path=False):
        """Builds an optical clock."""
        
        self.location = location
        self.country = country
        self.lat = lat
        self.lon = lon
        self.links = []
        self.states = []
        
    def __repr__(self):
        """To be printed if instance is pwritten."""
        
        summary = '<clones.Clock>\n'
        summary += 'Location: ' + self.location + '\n'
        summary += 'Lat, Lon: (' + str(self.lon) + ', ' + str(self.lat) + ')\n'
        summary += 'Links to: ('
        for c, s in zip(self.links, self.states):
            summary += c.location + ' (' + str(s) + ')' + ', '
        return summary + ')\n \n'
    
    def link_to(self, b, s):
        """Link to another clock."""
        
        self.links.append(b)
        self.states.append(s)
        # TODO: add a pointer to or a Link itself to this (self) instance of clock?
        
    def h2ff(self, h):
        """Convert elevation change to fractional frequency change."""
        
        g = 9.81  # [m/s^2]
        c = astropy.constants.c.value  # [m/s] 
        ff = h * g / c**2  # [s/s]
        return ff
    
    def N2ff(self, N):
        """Convert geoid height change to fractional frequency change."""
        
        g = 9.81  # [m/s^2]
        c = astropy.constants.c.value  # [m/s] 
        ff = N * g / c**2  # [s/s]
        return ff
    
    def construct_noise(self, N):
        """Construct noise for the 3 scenarios.
        
        Constructs noise in the unitless fractional frequency. Creates a list
        with 3 numpy arrays, for 3 scenarios. Each array is the sum of 3
        different noises, clock white noise, gnss white noise, and gnss flicker
        noise.
        
        :param N: length of the time series
        :type N: int
        """
        
        sigma_clock = np.array([1e-18, 1e-19, 1e-20])
        sigma_gnss_white = np.array([1, 0.5, 0.2]) * 1e-3 * \
            scipy.constants.g / scipy.constants.c**2
        sigma_gnss_flicker = np.array([1, 0.5, 0.2]) * 1e-3 * \
            scipy.constants.g / scipy.constants.c**2
        noise_clock = [np.random.normal(0, s, N) for s in
                       sigma_clock]
        noise_gnss_white = [np.random.normal(0, s, N) for s in
                            sigma_gnss_white]
        noise_gnss_flicker = [cn.powerlaw_psd_gaussian(1, N) * s
                              for s in sigma_gnss_flicker]
        noise = [noise_clock[i] + noise_gnss_flicker[i] +
                 noise_gnss_white[i] for i in range(3)]
        
        return noise
    
    def _sh2timeseries(self, F_lm, t, kind, unit, unitTo=[]):
        """Expand spherical harmonics into timeseries at clock location.
        
        DEPRECATED!
        
        :param F_lm: spherical harmonic coefficients
        :type F_lm: list of pyshtools.SHCoeffs
        :param t: time vector of decimal years
        :type t: numpy.array
        :param kind: origin of the mass redistribution
        :type kind: str
        :param unit: the input unit of the coefficients
        :type unit: str
        
        Possible units:
            'pot' ... dimensionless Stokes coeffs (e.g. GRACE L2)
                'U' ... geopotential [m^2/s^2]
                'N' ... geoid height [m]
            'h' .... elevation [m]
            'mass' ... dimensionless surface loading coeffs
                'sd' ... surface density [kg/m^2]
                'ewh' ... equivalent water height [m]
            'gravity'... [m/s^2]
            
        Possible kinds:
            'I' ... Ice
            'H' ... Hydrology
            'A' ... Atmosphere
            'O' ... Ocean
            'GIA'.. Glacial Isostatic Adjustment
            'S' ... Solid Earth            
        """
        
        # TODO: store time of effect only in network or also in network?
        
        if not unitTo:  # convert the unit of the spherical harmonics?
            if (kind == 'I' and hasattr(self, 'I') or
                kind == 'H' and hasattr(self, 'H') or
                kind == 'A' and hasattr(self, 'A') or
                kind == 'O' and hasattr(self, 'O') or
                kind == 'GIA' and hasattr(self, 'GIA') or
                kind == 'S' and hasattr(self, 'S')):
                effect = []
                for f_lm in F_lm:
                    effect.append(f_lm.expand(lat=self.lat, lon=self.lon))
                try:
                    k = getattr(self, kind)
                    k[unit] = np.array(effect)
                except:
                    print('Choose a proper kind of effect!') 
#                if kind == 'I':
#                    self.I[unit] = np.array(effect)
#                elif kind == 'H':
#                    self.H[unit] = np.array(effect)
#                elif kind == 'A':
#                    self.A[unit] = np.array(effect)
#                elif kind == 'O':
#                    self.O[unit] = np.array(effect)
#                elif kind == 'GIA':
#                    self.GIA[unit] = np.array(effect)
#                elif kind == 'S':
#                    self.S[unit] = np.array(effect)
#                else:
#                    print('Choose a proper kind of effect!')
            else:  # kind not yet there!
                effect = []
                for f_lm in F_lm:
                    effect.append(f_lm.expand(lat=self.lat, lon=self.lon))
                effects = {}
                effects['t'] = t
                effects[unit] = np.array(effect)
                try:
                    setattr(self, kind, effects)
                except:
                    print('Choose a proper kind of effect!')
                    
        else:  # convert the unit of the spherical harmonics!
            if (kind == 'I' and hasattr(self, 'I') or
                kind == 'H' and hasattr(self, 'H') or
                kind == 'A' and hasattr(self, 'A') or
                kind == 'O' and hasattr(self, 'O') or
                kind == 'GIA' and hasattr(self, 'GIA') or
                kind == 'S' and hasattr(self, 'S')):
                for u in unitTo:
                    effect = []
                    for f_lm in F_lm:
                        f_lm_u = harmony.sh2sh(f_lm, unit, u)
                        effect.append(f_lm_u.expand(lat=self.lat, lon=self.lon))
                    try:
                        k = getattr(self, kind)
                        k[u] = np.array(effect)
                    except:
                        print('Choose a proper kind of effect!')                          
#                    if kind == 'I':
#                        self.I[u] = np.array(effect)
#                    elif kind == 'H':
#                        self.H[u] = np.array(effect)
#                    elif kind == 'A':
#                        self.A[u] = np.array(effect)
#                    elif kind == 'O':
#                        self.O[u] = np.array(effect)
#                    elif kind == 'GIA':
#                        self.GIA[u] = np.array(effect)
#                    elif kind == 'S':
#                        self.S[u] = np.array(effect)
#                    else:
#                        print('Choose a proper kind of effect!')  
            else:  # kind not yet there!
                effects = {}
                effects['t'] = t
                for u in unitTo:
                    effect = []
                    for f_lm in F_lm:
                        f_lm_u = harmony.sh2sh(f_lm, unit, u)
                        effect.append(f_lm_u.expand(lat=self.lat,
                                                    lon=self.lon))
                    effects[u] = np.array(effect)
                try:
                    setattr(self, kind, effects)
                except:
                    print('Choose a proper kind of effect!')
                    
    def sh2timeseries(self, T, esc, unitFrom, unitTo, t_ref=False,
                      reset=False, error=False, sigma=False, filt=False,
                      lmin=False, lmax=False, field=False):
        """Expands spherical harmonics at clock location.
        
        Takes the spherical harmonics from the data folder at a given time
        interval and expands them at the clock location.
        
        :param T: list of dates
        :type T: str or datetime.date(time)
        :param esc: earth system component
        :type esc: str
        :param unitFrom: unit of the input coefficients
        :type unitFrom: str
        :param unitTo: unit of the timeseries
        :type unitTo: str
        :param t_ref: reference time for the series
        :type t_ref: str or datetime.date(time)
        :param reset: shall the timeseries be calculated again
        :type reset: boolean, optional
        :param error: the clock error type: 'white' or 'allan'
        :type error: str
        :param sigma: uncertainty of the clock
        :type sigma: float
        :param filt: filter width for the data, has to be an odd number
        :type filt: integer, optional
        :return T_date: dates of the timeseries
        :rtype T_date: list of str
        :return series: timeseries
        :rtype series: list of floats
        
        Possible units:
            'pot' ... dimensionless Stokes coeffs (e.g. GRACE L2)
                'U' ... geopotential [m^2/s^2]
                'N' ... geoid height [m]
            'h' ... elevation [m]
            'ff' ... fractional frequency [-]
            'mass' ... dimensionless surface loading coeffs
                'sd' ... surface density [kg/m^2]
                'ewh' ... equivalent water height [m]
            'gravity'... [m/s^2]
            
        Possible earth system components:
            'I' ... Ice
            'H' ... Hydrology
            'A' ... Atmosphere
            'GIA'.. Glacial Isostatic Adjustment
        """
        
        esc_dict = {'I': 'oggm_',
                    'I_scandinavia': 'oggm_',
                    'H': 'clm_tws_',
                    'A': 'coeffs_',
                    'GRACE_ITSG2018': 'ITSG_Grace2018_n120_'}
        
        # TODO: Check whether the timeseries is already there
        
        path = cfg.PATHS['data_path'] + esc + '/'
        # make strings if time is given in datetime objects
        if not isinstance(t_ref, str):
            t_ref = datetime.datetime.strftime(t_ref, format='%Y_%m_%d')
        if not isinstance(T[0], str):
            T = [datetime.datetime.strftime(t, format='%Y_%m_%d') for t in T]
        # Check whether the data is available
        if not os.path.exists(path + esc_dict[esc] + t_ref + '.nc'):
            print(path + esc_dict[esc] + t_ref + '.nc', ' does not exist.')
            return
        for t in T:
            if not os.path.exists(path + esc_dict[esc] + t + '.nc'):
                print(path + esc_dict[esc] + t + '.nc', ' does not exist.')
                return
            
        series = []
        for t in T:
            f_lm = harmony.shcoeffs_from_netcdf(path + esc_dict[esc] + t)
            f_lm_ref = harmony.shcoeffs_from_netcdf(path + esc_dict[esc]
                                                    + t_ref)
            f_lm = harmony.sh2sh(f_lm, unitFrom, unitTo, lmax=lmax)
            f_lm_ref = harmony.sh2sh(f_lm_ref, unitFrom, unitTo, lmax=lmax)
            # if lmin:
            #     f_lm = f_lm - f_lm.pad(lmin).pad(f_lm.lmax)
            #     f_lm_ref = f_lm_ref - f_lm_ref.pad(lmin).pad(f_lm_ref.lmax)
            # if lmax:
            #     f_lm = f_lm.pad(lmax).pad(f_lm.lmax)
            #     f_lm_ref = f_lm_ref.pad(lmax).pad(f_lm_ref.lmax)
            f_lm = f_lm - f_lm_ref
            if field:
                t0 = time.time()
                grid = f_lm.expand()
                t0 = time.time()
                lat_limits = [44, 61]
                lon_limits = [0, 25]
                lat1 = np.ndarray.flatten(np.array(np.where(grid.lats() < lat_limits[1])))
                lat2 = np.ndarray.flatten(np.array(np.where(grid.lats() > lat_limits[0])))
                lat_ind = np.intersect1d(lat1, lat2)
                lon1 = np.ndarray.flatten(np.array(np.where(grid.lons() < lon_limits[1])))
                lon2 = np.ndarray.flatten(np.array(np.where(grid.lons() > lon_limits[0])))
                if lon_limits[0] >= 0:
                    lon_ind = np.intersect1d(lon2, lon1)
                else:
                    lon_ind = np.concatenate((lon2, lon1))
                true_eu = grid.data[lat_ind][:, lon_ind]
                true_eu_flat = np.ndarray.flatten(true_eu)
                lats_eu = grid.lats()[lat_ind]
                lons_eu = grid.lons()[lon_ind]
                grid_eu = np.meshgrid(lons_eu, lats_eu)
                lats_eu = grid_eu[1].flatten()
                lons_eu = grid_eu[0].flatten()
                series.append(true_eu_flat)
            else:
                series.append(f_lm.expand(lat=self.lat, lon=self.lon))
        try:
            T_date = [datetime.datetime.strptime(t, '%Y_%m_%d') for t in T]
        except:
            T_date = [datetime.datetime.strptime(t, '%Y_%m') for t in T]
            
        if field:
            series = np.array(series)
            return T_date, series, lats_eu, lons_eu
        
        if filt:
            series = utils.ma(np.array(series), filt)
        if sigma:
            noise = self.construct_noise(len(series))
            if filt:
                noise = [utils.ma(noi, filt) for noi in noise]
            return T_date, series, noise
        
        return T_date, series    
    
    def sh2timeseries_error(self, add, T, esc, unitFrom, unitTo, t_ref=False,
                            reset=False, error=False, sigma=False, filt=False,
                            lmin=False, lmax=False):
        """Expands spherical harmonics at clock location.
        
        Takes the spherical harmonics from the data folder at a given time
        interval, adds a given additional set of sh's, e.g. errors, and expands
        them at the clock location.
        
        :param T: list of dates
        :type T: str or datetime.date(time)
        :param esc: earth system component
        :type esc: str
        :param unitFrom: unit of the input coefficients
        :type unitFrom: str
        :param unitTo: unit of the timeseries
        :type unitTo: str
        :param t_ref: reference time for the series
        :type t_ref: str or datetime.date(time)
        :param reset: shall the timeseries be calculated again
        :type reset: boolean, optional
        :param error: the clock error type: 'white' or 'allan'
        :type error: str
        :param sigma: uncertainty of the clock
        :type sigma: float
        :param filt: filter width for the data, has to be an odd number
        :type filt: integer, optional
        :return T_date: dates of the timeseries
        :rtype T_date: list of str
        :return series: timeseries
        :rtype series: list of floats
        
        Possible units:
            'pot' ... dimensionless Stokes coeffs (e.g. GRACE L2)
                'U' ... geopotential [m^2/s^2]
                'N' ... geoid height [m]
            'h' ... elevation [m]
            'ff' ... fractional frequency [-]
            'mass' ... dimensionless surface loading coeffs
                'sd' ... surface density [kg/m^2]
                'ewh' ... equivalent water height [m]
            'gravity'... [m/s^2]
            
        Possible earth system components:
            'I' ... Ice
            'H' ... Hydrology
            'A' ... Atmosphere
            'GIA'.. Glacial Isostatic Adjustment
        """
        
        esc_dict = {'I': 'oggm_',
                    'I_scandinavia': 'oggm_',
                    'H': 'clm_tws_',
                    'A': 'coeffs_',
                    'GRACE_ITSG2018': 'ITSG_Grace2018_n120_'}
        
        # TODO: Check whether the timeseries is already there
        
        path = cfg.PATHS['data_path'] + esc + '/'
        # make strings if time is given in datetime objects
        if not isinstance(t_ref, str):
            t_ref = datetime.datetime.strftime(t_ref, format='%Y_%m_%d')
        if not isinstance(T[0], str):
            T = [datetime.datetime.strftime(t, format='%Y_%m_%d') for t in T]
        # Check whether the data is available
        if not os.path.exists(path + esc_dict[esc] + t_ref + '.nc'):
            print(path + esc_dict[esc] + t_ref + '.nc', ' does not exist.')
            return
        for t in T:
            if not os.path.exists(path + esc_dict[esc] + t + '.nc'):
                print(path + esc_dict[esc] + t + '.nc', ' does not exist.')
                return
            
        series = []
        series_upper = []
        series_lower = []
        for t in T:
            f_lm = harmony.shcoeffs_from_netcdf(path + esc_dict[esc] + t)
            f_lm_ref = harmony.shcoeffs_from_netcdf(path + esc_dict[esc]
                                                    + t_ref)
            f_lm = harmony.sh2sh(f_lm, unitFrom, unitTo)
            f_lm_upper = harmony.sh2sh(f_lm, unitFrom, unitTo) + add
            f_lm_lower = harmony.sh2sh(f_lm, unitFrom, unitTo) - add
            f_lm_ref = harmony.sh2sh(f_lm_ref, unitFrom, unitTo)
            f_lm_ref_upper = harmony.sh2sh(f_lm_ref, unitFrom, unitTo) + add
            f_lm_ref_lower = harmony.sh2sh(f_lm_ref, unitFrom, unitTo) - add
            f_lm = f_lm - f_lm_ref
            f_lm_upper = f_lm_upper - f_lm_ref_upper
            f_lm_lower = f_lm_lower - f_lm_ref_lower
            print(f_lm_ref_upper.expand(lat=self.lat, lon=self.lon) -
                  f_lm_ref_lower.expand(lat=self.lat, lon=self.lon))
            series.append(f_lm.expand(lat=self.lat, lon=self.lon))
            series_upper.append(f_lm_upper.expand(lat=self.lat, lon=self.lon))
            series_lower.append(f_lm_lower.expand(lat=self.lat, lon=self.lon))
        
        try:
            T_date = [datetime.datetime.strptime(t, '%Y_%m_%d') for t in T]
        except:
            T_date = [datetime.datetime.strptime(t, '%Y_%m') for t in T]
        
        if filt:
            series = utils.ma(np.array(series), filt)
            series_upper = utils.ma(np.array(series_upper), filt)
            series_lower = utils.ma(np.array(series_lower), filt)
        # if sigma:
        #     noise = self.construct_noise(len(series))
            # if filt:
            #     noise = [utils.ma(noi, filt) for noi in noise]
            return T_date, series, series_upper, series_lower#, noise
        
        return T_date, series, series_upper, series_lower
     
    def plotTimeseries(self, T, esc, unitFrom, unitTo, t_ref=False,
                       reset=False, error=False, sigma=False, save=False):
        """Plots time series at clock location.
        
        Uses sh2timeseries() and plots the resulting time series at clock
        location.
        
        :param T: list of dates
        :type T: str or datetime.date(time)
        :param esc: earth system component(s)
        :type esc: str or list of str
        :param unitFrom: unit of the input coefficients
        :type unitFrom: str
        :param unitTo: unit of the timeseries
        :type unitTo: str 
        :param t_ref: reference time for the series
        :type t_ref: str or datetime.date(time), optional
        :param reset: shall the timeseries be calculated again
        :type reset: boolean, optional
        :param error: the clock error type: 'white' or 'allan'
        :type error: str, optional
        :param sigma: uncertainty of the clock
        :type sigma: float, optional
        
        Possible units:
            'pot' ... dimensionless Stokes coeffs (e.g. GRACE L2)
                'U' ... geopotential [m^2/s^2]
                'N' ... geoid height [m]
            'h' ... elevation [m]
            'ff' ... fractional frequency [-]
            'mass' ... dimensionless surface loading coeffs
                'sd' ... surface density [kg/m^2]
                'ewh' ... equivalent water height [m]
            'gravity'... [m/s^2]
            
        Possible earth system components:
            'I' ... Ice
            'H' ... Hydrology
            'A' ... Atmosphere
            'GIA'.. Glacial Isostatic Adjustment
        """
        
        np.random.seed(7)
        unit_dict = {'U': 'gravitational potential [m$^2$/s$^2$]',
                    'N': 'Geoid height [mm]',
                    'h': 'Elevation [mm]',
                    'sd': 'Surface Density [kg/m$^2$]',
                    'ewh': 'Equivalent water height [m]',
                    'gravity': 'gravitational acceleration [m/s$^2$]',
                    'ff': 'Fractional frequency [-]'}
        plt.rcParams.update({'font.size': 13})  # set before making the figure!        
        fig, ax = plt.subplots()
        
        if sigma:
            if isinstance(esc, list):
                for e in esc:
                    T, data, noise = self.sh2timeseries(
                        T, e, unitFrom, unitTo, t_ref=t_ref, reset=reset,
                        error=error, sigma=sigma)
                    if unitTo in('N', 'h', 'GRACE'):
                        data = [i * 1e3 for i in data]
                        noise = [i * 1e3 for i in noise] # noise muss arrays sein in der liste
                    plt.plot(T, data, label=e, linewidth=0.5)
                    for i, (noi, sig) in enumerate(zip(noise[:-1], sigma[:-1])):
                        plt.plot(T, data+noi, ':', linewidth=1,
                                 label='noise scenario ' + str(i+1))
            else:
                T, data, noise = self.sh2timeseries(
                    T, esc, unitFrom, unitTo, t_ref=t_ref, reset=reset,
                    error=error, sigma=sigma)
                if unitTo in('N', 'h', 'GRACE'):
                    data = [i * 1e3 for i in data]
                    noise = [i * 1e3 for i in noise]
                p = plt.plot(T, data, label=esc, linewidth=1.5, zorder=0)
                if sigma==3:
                    for i, noi in enumerate(noise):
                        color = np.array(['tab:orange', 'tab:red', 'tab:brown'])[i]
                        plt.plot(T, data+noi, linewidth=0.5, color=color,
                                 label=esc + ' + noise scenario ' + str(i+1))
                else:
                    for i, noi in enumerate(noise[:-1]):
                        color = np.array(['tab:orange', 'tab:red'])[i]
                        plt.plot(T, data+noi, linewidth=0.5, color=color,
                                 label=esc + ' + noise scenario ' + str(i+1))
        else:
            if isinstance(esc, list):
                for e in esc:
                    T, data = self.sh2timeseries(T, e, unitFrom, unitTo,
                                                 t_ref=t_ref, reset=reset)
                    if unitTo in('N', 'h', 'GRACE'):
                        data = [i * 1e3 for i in data]
                    plt.plot(T, data, label=e)
            else:
                T, data = self.sh2timeseries(T, esc, unitFrom, unitTo,
                                             t_ref=t_ref, reset=reset)
                if unitTo in('N', 'h', 'GRACE'):
                    data = [i * 1e3 for i in data]
                plt.plot(T, data, label=esc)
        
        plt.ylabel(unit_dict[unitTo])
        plt.xticks(rotation=90)
        plt.title(self.location)
        plt.grid()
        plt.legend()
        plt.tight_layout()
        
        for i, noi in enumerate(noise):
            np.save(cfg.PATHS['data_path'] + 'ts_results/' + esc + '_' +
                    unitTo + '_' + self.location + '_noise' + str(i), data+noi)
        
        if save:
            path = (cfg.PATHS['fig_path'] + 'timeseries/' + esc + '_' + unitTo
                    + '_' + self.location + '.pdf')
            plt.savefig(path)
            #TODO: case for esc is list
    
    def plotTimeFrequencies(self, T, esc, unitFrom, unitTo, delta_t,
                            fmax=False, t_ref=False, reset=False, error=False,
                            sigma=False, filt=False, save=False):
        """Plots time series in frequency domain at clock location.
        
        Uses sh2timeseries() and plots the resulting time series in frequency
        domain at clock location.
        
        :param T: list of dates
        :type T: str or datetime.date(time)
        :param esc: earth system component(s)
        :type esc: str or list of str
        :param unitFrom: unit of the input coefficients
        :type unitFrom: str
        :param unitTo: unit of the timeseries
        :type unitTo: str 
        :param t_ref: reference time for the series
        :type t_ref: str or datetime.date(time), optional
        :param reset: shall the timeseries be calculated again
        :type reset: boolean, optional
        :param error: the clock error type: 'white' or 'allan'
        :type error: str, optional
        :param sigma: uncertainty of the clock
        :type sigma: float, optional
        :param delta_t: measurement sampling rate [s]
        :type delta_t: float
        :param fmax: maximum plottable frequency band
        :type fmax: integer, optional
        :param filt: filter width for the data, has to be an odd number
        :type filt: integer, optional
        
        Possible units:
            'pot' ... dimensionless Stokes coeffs (e.g. GRACE L2)
                'U' ... geopotential [m^2/s^2]
                'N' ... geoid height [m]
            'h' ... elevation [m]
            'ff' ... fractional frequency [-]
            'mass' ... dimensionless surface loading coeffs
                'sd' ... surface density [kg/m^2]
                'ewh' ... equivalent water height [m]
            'gravity'... [m/s^2]
            
        Possible earth system components:
            'I' ... Ice
            'H' ... Hydrology
            'A' ... Atmosphere
            'GIA'.. Glacial Isostatic Adjustment
        """
        
        np.random.seed(7)
        unit_dict = {'U': 'gravitational potential [m$^2$/s$^2$]',
                    'N': 'Geoid height [mm]',
                    'h': 'Elevation [mm]',
                    'sd': 'Surface Density [kg/m$^2$]',
                    'ewh': 'Equivalent water height [m]',
                    'gravity': 'gravitational acceleration [m/s$^2$]',
                    'ff': 'Fractional frequency [-]'}
        plt.rcParams.update({'font.size': 13})  # set before making the figure!        
        fig, ax = plt.subplots()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        if sigma:
            if type(esc) is list:  # DEPRECATED!
                for e in esc:
                    T, data, noise = self.sh2timeseries(
                        T, e, unitFrom, unitTo, t_ref=t_ref, reset=reset,
                        error=error, sigma=sigma, filt=filt)
                    f, freq = harmony.time2freq(delta_t, data)
                    # f, noisy_freq = harmony.time2freq(delta_t, data+noise)
                    noise_level = []
                    for noi in noise:
                        fn, noisy = harmony.time2freq(delta_t, noi)
                        noise_level.append(np.mean(noisy) * np.ones(len(f)))
                    if fmax:
                        f, freq, noise_level = (f[:fmax], freq[:fmax],
                                                [n[:fmax] for n
                                                 in noise_level])
                    # plt.plot(f*86400*365, noisy_freq, 'x', label=e+' + noise',
                    #          color=p[0].get_color())
                    for lvl, sig in zip(noise_level, sigma):
                        plt.plot(f*86400*365, lvl,
                                 label='noise level for $\sigma$='+str(sig))
                    p = plt.plot(f[1:]*86400*365, freq[1:], '.-', label=e)
            else:
                # T, data, noise = self.sh2timeseries(
                #     T, esc, unitFrom, unitTo, t_ref=t_ref, reset=reset,
                #     error=error, sigma=sigma, filt=filt)
                # f, freq = harmony.time2freq(delta_t, data)
                # # f, noisy_freq = harmony.time2freq(delta_t, data+noise)
                # noise_level = []
                # for noi in noise:
                #     fn, noisy = harmony.time2freq(delta_t, noi)
                #     noise_level.append(np.mean(noisy) * np.ones(len(f)))
                # if fmax:
                #     f, freq, noise_level = (f[:fmax], freq[:fmax],
                #                             [n[:fmax] for n in noise_level])
                # # plt.plot(f*86400*365, noisy_freq, 'x', label=e+' + noise',
                # #          color=p[0].get_color())
                # for lvl, sig in zip(noise_level, sigma):
                #     plt.plot(f*86400*365, lvl,
                #              label='noise level for $\sigma$='+str(sig))
                # p = plt.plot(f[1:]*86400*365, freq[1:], '.-', label=esc)
                T, data = self.sh2timeseries(
                    T, esc, unitFrom, unitTo, t_ref=t_ref, reset=reset,
                    filt=filt)
                f, freq = harmony.time2freq(delta_t, data)
                if fmax:
                    f, freq = f[:fmax], freq[:fmax]
                noise = np.load('/home/schroeder/CLONETS/data/noise.npy')
                plt.plot(f[1:]*86400*365, noise[0, :], '--', color='black',
                         label='noise levels')
                plt.plot(f[1:]*86400*365, noise[1, :], '--', color='black')
                plt.plot(f[1:]*86400*365, noise[2, :], '--', color='black')
                plt.plot(f[1:]*86400*365, freq[1:], '.-', label=esc)
                
                np.save(cfg.PATHS['data_path'] + 'ts_results/' + esc +
                        '_freq_' + self.location, freq[1:])
                np.save(cfg.PATHS['data_path'] + 'ts_results/x_freq', f[1:]*86400*365)
        else:
            if type(esc) is list:
                for e in esc:
                    T, data = self.sh2timeseries(
                        T, e, unitFrom, unitTo, t_ref=t_ref, reset=reset,
                        filt=filt)
                    f, freq = harmony.time2freq(delta_t, data)
                    if fmax:
                        f, freq = f[:fmax], freq[:fmax]
                    plt.plot(f[1:]*86400*365, freq[1:], '.-', label=e)
            else:
                T, data = self.sh2timeseries(
                    T, esc, unitFrom, unitTo, t_ref=t_ref, reset=reset,
                    filt=filt)
                f, freq = harmony.time2freq(delta_t, data)
                if fmax:
                    f, freq = f[:fmax], freq[:fmax]
                plt.plot(f[1:]*86400*365, freq[1:], '.-', label=esc)
        
        plt.title(self.location)
        plt.xlabel('Frequencies [1/yr]')
        plt.yscale('log')
        plt.xscale('log')
        plt.ylabel(unit_dict[unitTo])
        plt.grid()
        # plt.legend(loc='lower left')
        plt.legend()
        plt.tight_layout()
        
        if save:
            path = (cfg.PATHS['fig_path'] + 'timeseries/' + esc + '_' + unitTo
                    + '_' + self.location + '_spectral.pdf')
            plt.savefig(path)
            #TODO: case for esc is list
    
    def plotCorrelation(self, T, esc, unitFrom, unitTo, save=False,
                        trend=False):
        """Plots the correlation of all grid cells' time series.
        
        Computes the correlation of all grid cells' time series with respect
        to the clock's timeseries and plots it with pyGMT.
        
        :param T: list of dates
        :type T: str or datetime.date(time)
        :param esc: earth system component
        :type esc: str
        :param unitFrom: unit of the input coefficients
        :type unitFrom: str
        :param unitTo: unit of the timeseries
        :type unitTo: str
        :param save: shall the figure be saved
        :type save: boolean
        :return fig: the figure object
        :rtype fig: pygmt.Figure
        :return grid: the plottet data grid
        :rtype grid: pyshtools.SHGrid
        
        Possible units:
            'pot' ... dimensionless Stokes coeffs (e.g. GRACE L2)
                'U' ... geopotential [m^2/s^2]
                'N' ... geoid height [m]
                'GRACE' ... geoid height [m], but with lmax=120 and filtered
            'h' ... elevation [m]
            'ff' ... fractional frequency [-]
            'mass' ... dimensionless surface loading coeffs
                'sd' ... surface density [kg/m^2]
                'ewh' ... equivalent water height [m]
            'gravity'... [m/s^2]
            
        Possible earth system components:
            'I' ... Ice
            'H' ... Hydrology
            'A' ... Atmosphere
        """
        
        esc_dict = {'I': 'oggm_',
                    'H': 'clm_tws_',
                    'A': 'coeffs_'}
        cb_dict = {'U': '"RMS of gravitational potential [m@+2@+/s@+2@+]"',
                   'N': '"RMS of Geoid height [mm]"',
                   'h': '"RMS of Elevation [mm]"',
                   'sd': '"RMS of Surface Density [kg/m@+2@+]"',
                   'ewh': '"RMS of Equivalent water height [m]"',
                   'gravity': '"RMS of gravitational acceleration [m/s@+2@+]"',
                   'ff': '"RMS of Fractional frequency [-]"',
                   'GRACE': '"RMS of Geoid height [mm]"'}
        
        T_frac = np.array([utils.datetime2frac(t) for t in T])
        # make strings if time is given in datetime objects
        if not isinstance(T[0], str):
            T = [datetime.datetime.strftime(t, format='%Y_%m_%d') for t in T]
        
        path = cfg.PATHS['data_path'] + esc + '/'
        f_lm = harmony.shcoeffs_from_netcdf(path + esc_dict[esc] + T[0])
        grid = f_lm.pad(720).expand()
        y = np.arange(int(grid.nlat/10), int(grid.nlat/3))
        x_east = np.arange(int(grid.nlon/6))
        x_west = np.arange(int(grid.nlon/20*19), grid.nlon)
        # the lons and lats of the europe grid
        LONS = np.concatenate((grid.lons()[x_west], grid.lons()[x_east]))
        LATS = grid.lats()[y]
        europe = np.zeros((len(y), len(x_west)+len(x_east)))
        EUROPA = []
        clo_ts = []
        for t in T:
            f_lm = harmony.shcoeffs_from_netcdf(path + esc_dict[esc] + t)
            f_lm = harmony.sh2sh(f_lm, unitFrom, unitTo)
            clo_ts.append(f_lm.pad(720).expand(lat=self.lat, lon=self.lon))
            grid = f_lm.pad(720).expand()
            europe[:, :len(x_west)] = grid.data[y, x_west[0]:]
            europe[:, len(x_west):] = grid.data[y, :len(x_east)]
            EUROPA.append(copy.copy(europe))
            print(t)
        EUROPA = np.array(EUROPA)  # 365er liste mit ~500x400 arrays
        clo_ts = np.array(clo_ts)
        
        if trend == 'annual' or trend == 'semiannual':
            if trend == 'annual':
                model, A = utils.annual_trend(T_frac, clo_ts, semi=False)
            else:
                model, A = utils.annual_trend(T_frac, clo_ts)
            EUROPA_trend = A.dot(model)
            clo_ts = clo_ts - EUROPA_trend
        
        EUROPA_corr = np.zeros((np.shape(EUROPA)[1:]))
        distances = np.zeros((np.shape(EUROPA)[1:]))
        
        for i in range(np.shape(EUROPA)[1]):
            for j in range(np.shape(EUROPA)[2]):
                distances[i, j] = geopy.distance.geodesic(
                    (self.lat, self.lon), (LATS[i], LONS[j])).km
                if trend == 'annual' or trend == 'semiannual':
                    if trend == 'annual':
                        model, A = utils.annual_trend(T_frac, EUROPA[:, i, j],
                                                      semi=False)
                    else:
                        model, A = utils.annual_trend(T_frac, EUROPA[:, i, j])
                    EUROPA_trend = A.dot(model)
                    residual = EUROPA[:, i, j] - EUROPA_trend
                    EUROPA_corr[i, j] = np.corrcoef(clo_ts, residual)[0, 1]
                else:
                    EUROPA_corr[i, j] = np.corrcoef(clo_ts,
                                                    EUROPA[:, i, j])[0, 1]
                # EUROPA_corr[i, j] = distances[i, j]
                
        # grid.data = np.zeros((np.shape(grid.data)))
        data = np.zeros((np.shape(grid.data)))
        data[y, x_west[0]:] = EUROPA_corr[:, :len(x_west)]
        data[y, :len(x_east)] = EUROPA_corr[:, len(x_west):]
        grid.data = data
        
        x = grid.lons()
        y = grid.lats()
        # find out what the datalimits are within the shown region
        data_lim = np.concatenate((grid.to_array()[200:402, -81:],
                                   grid.to_array()[200:402, :242]), axis=1)
        datamax = np.max(data_lim)
        datamin = np.min(data_lim)
        print(datamin, datamax)
        
        da = xr.DataArray(data, coords=[y, x], dims=['lat', 'lon'])
        # save the dataarray as netcdf to work around the 360° plotting problem
        da.to_dataset(name='dataarray').to_netcdf(path + '../temp/pygmt.nc')
        
        fig = pygmt.Figure() 
        # pygmt.makecpt(cmap='viridis', series=[datamin, datamax], reverse=True)
        # fig.grdimage(path + '../temp/pygmt.nc', region=[-10, 30, 40, 65],
        #               projection="S10/90/6i", frame="ag")  # frame: a for the standard frame, g for the grid lines
        # fig.coast(region=[-10, 30, 40, 65], projection="S10/90/6i", frame="a",
        #           shorelines="1/0.1p,white", borders="1/0.1p,white")
        # fig.plot(self.lon, self.lat, style="c0.07i", color="white",
        #           pen="black")
        # fig.colorbar(frame='paf+l' + 'correlation')  # @+x@+ for ^x
        
        # if save:
        #     savepath = path + '../../fig/'
        #     savename = (os.path.join(savepath, esc, self.location + '_'
        #                               + esc_dict[esc] + T[0] + '-' + T[-1] + '_'
        #                               + unitTo + '_corr.pdf'))
        #     fig.savefig(savename)
        
        return fig, grid, distances, EUROPA_corr
        
class Link():
    """A fibre link between two clocks.
    
    :param a: clock A
    :type a: Clock
    :param b: clock B
    :type b: Clock
    :param status: status of the link
    :type status: int
    
    possible states:
        0 ... existing
        1 ... planned for extension phase 1
        2 ... planned for extension phase 2 (and so on)
        9 ... imaginable
    """
    
    def __init__(self, a, b, state=9):
        """Instanciates a fibre link between two clocks a and b."""
        
        self.state = state
        self.a = a
        self.b = b
        self.name = a.location + ' --- ' + b.location
        
    def __repr__(self):
        """To be printed if instance is written."""
        
        summary = '<clones.Link>\n'
        summary += 'From: ' + self.a.location + '\n'
        summary += 'To: ' + self.b.location + '\n'
        summary += 'State: ' + str(self.state) + '\n'
        summary += ('Distance: ' + str(np.round(self.length())) +
                    ' km\n')
        return summary
        
    def length(self):
        """Returns the length of the fibre link in km."""
        
        return geopy.distance.geodesic((self.a.lat, self.a.lon),
                                       (self.b.lat, self.b.lon)).km
    
    def analyze_degrees(self, T, esc, unitFrom, unitTo, lmax, t_ref=False):
        
        T_a, data_a = self.a.sh2timeseries(T, esc, unitFrom, unitTo,
                                           t_ref=t_ref)
        T_b, data_b = self.b.sh2timeseries(T, esc, unitFrom, unitTo,
                                           t_ref=t_ref)
        full_timeseries = np.array(data_b) - np.array(data_a)
        
        power = np.zeros(lmax)
        for i in range(lmax):
            T_a2, data_a2 = self.a.sh2timeseries(T, esc, unitFrom, unitTo,
                                                 t_ref=t_ref, lmin=i)
            T_b2, data_b2 = self.b.sh2timeseries(T, esc, unitFrom, unitTo,
                                                 t_ref=t_ref, lmin=i)
            trunc_timeseries = np.array(data_b2) - np.array(data_a2)
            power[i] = utils.rms(full_timeseries, trunc_timeseries)
            print(i)

        if unitTo in('N', 'h', 'GRACE'):
            power = power * 1e3
        
        return power
    
    def analyze_degrees2(self, T, esc, unitFrom, unitTo, lmax, t_ref=False):
        
        T_a, data_a = self.a.sh2timeseries(T, esc, unitFrom, unitTo,
                                           t_ref=t_ref)
        T_b, data_b = self.b.sh2timeseries(T, esc, unitFrom, unitTo,
                                           t_ref=t_ref)
        full_timeseries = np.array(data_b) - np.array(data_a)
        
        power = np.zeros(lmax)
        for i in range(lmax):
            T_a2, data_a2 = self.a.sh2timeseries(T, esc, unitFrom, unitTo,
                                                 t_ref=t_ref, lmin=i)
            T_b2, data_b2 = self.b.sh2timeseries(T, esc, unitFrom, unitTo,
                                                 t_ref=t_ref, lmin=i)
            trunc_timeseries = np.array(data_b2) - np.array(data_a2)
            power[i] = utils.rms(trunc_timeseries)
            print(i)

        if unitTo in('N', 'h', 'GRACE'):
            power = power * 1e3
        
        return power
    
    def analyze_degrees3(self, T, esc, unitFrom, unitTo, lmax, t_ref=False):
        
        T_a, data_a = self.a.sh2timeseries(T, esc, unitFrom, unitTo,
                                           t_ref=t_ref)
        T_b, data_b = self.b.sh2timeseries(T, esc, unitFrom, unitTo,
                                           t_ref=t_ref)
        full_timeseries = np.array(data_b) - np.array(data_a)
        
        power = np.zeros(lmax)
        for i in range(lmax):
            T_a2, data_a2 = self.a.sh2timeseries(T, esc, unitFrom, unitTo,
                                                 t_ref=t_ref, lmax=i)
            T_b2, data_b2 = self.b.sh2timeseries(T, esc, unitFrom, unitTo,
                                                 t_ref=t_ref, lmax=i)
            trunc_timeseries = np.array(data_b2) - np.array(data_a2)
            power[i] = utils.rms(full_timeseries, trunc_timeseries)
            print(i)

        if unitTo in('N', 'h', 'GRACE'):
            power = power * 1e3
        
        return power
            
    def plotTimeseries(self, T, esc, unitFrom, unitTo, t_ref=False,
                       reset=False, error=False, sigma=False, filt=False,
                       lmax=False, save=False):
        """Plots the differential time series.
        
        Uses sh2timeseries() for both clocks and plots the resulting
        differential time series.
        
        :param T: list of dates
        :type T: str or datetime.date(time)
        :param esc: earth system component(s)
        :type esc: str or list of str
        :param unitFrom: unit of the input coefficients
        :type unitFrom: str
        :param unitTo: unit of the timeseries
        :type unitTo: str 
        :param t_ref: reference time for the series
        :type t_ref: str or datetime.date(time)
        :param reset: shall the timeseries be calculated again
        :type reset: boolean, optional
        
        Possible units:
            'pot' ... dimensionless Stokes coeffs (e.g. GRACE L2)
                'U' ... geopotential [m^2/s^2]
                'N' ... geoid height [m]
            'h' ... elevation [m]
            'ff' ... fractional frequency [-]
            'mass' ... dimensionless surface loading coeffs
                'sd' ... surface density [kg/m^2]
                'ewh' ... equivalent water height [m]
            'gravity'... [m/s^2]
            
        Possible earth system components:
            'I' ... Ice
            'H' ... Hydrology
            'A' ... Atmosphere
            'GIA'.. Glacial Isostatic Adjustment
        """
        
        np.random.seed(7)
        unit_dict = {'U': 'gravitational potential [m$^2$/s$^2$]',
                    'N': 'Geoid height [mm]',
                    'h': 'Elevation [mm]',
                    'sd': 'Surface Density [kg/m$^2$]',
                    'ewh': 'Equivalent water height [m]',
                    'gravity': 'gravitational acceleration [m/s$^2$]',
                    'ff': 'Fractional frequency [-]'}
        plt.rcParams.update({'font.size': 13})  # set before making the figure!        
        fig, ax = plt.subplots()
        
        if sigma:
            if type(esc) is list:
                for e in esc:
                    T_a, data_a, noise_a = self.a.sh2timeseries(
                        T, e, unitFrom, unitTo, t_ref=t_ref, reset=reset,
                        filt=filt, error=error, sigma=sigma)
                    T_b, data_b, noise_b = self.b.sh2timeseries(
                        T, e, unitFrom, unitTo, t_ref=t_ref, reset=reset,
                        filt=filt, error=error, sigma=sigma)
                    data_a, data_b = np.array(data_a), np.array(data_b)
                    data = data_b - data_a
                    noise = [a + b for a, b in zip(noise_a, noise_b)]
                    if unitTo in('N', 'h', 'GRACE'):
                        data_a, data_b, noise = (
                            data_a * 1e3, data_b * 1e3, [noi * 1e3 for noi in 
                                                         noise])
                    for noi, sig in zip(noise, sigma):
                        plt.plot(T, data+noi, ':', linewidth=1,
                                 label='noise at $\sigma$='+str(sig))
                    plt.plot(T, data, label=e)
            else:
                T_a, data_a, noise_a = self.a.sh2timeseries(
                    T, esc, unitFrom, unitTo, t_ref=t_ref, reset=reset,
                    filt=filt, error=error, sigma=sigma)
                T_b, data_b, noise_b = self.b.sh2timeseries(
                    T, esc, unitFrom, unitTo, t_ref=t_ref, reset=reset,
                    filt=filt, error=error, sigma=sigma)
                data_a, data_b = np.array(data_a), np.array(data_b)
                data = data_b - data_a
                noise = [a + b for a, b in zip(noise_a, noise_b)]
                if unitTo in('N', 'h', 'GRACE'):
                    data_a, data_b, noise = (
                        data_a * 1e3, data_b * 1e3, [noi * 1e3 for noi in 
                                                     noise])
                p = plt.plot(T, data, label=esc, linewidth=1.5, zorder=0)
                if sigma==3:
                    for i, noi in enumerate(noise):
                        color = np.array(['tab:orange', 'tab:red', 'tab:brown'])[i]
                        plt.plot(T, data+noi, linewidth=0.5, color=color,
                                 label=esc + ' + noise scenario ' + str(i+1))
                else:
                    for i, noi in enumerate(noise[:-1]):
                        color = np.array(['tab:orange', 'tab:red'])[i]
                        plt.plot(T, data+noi, linewidth=0.5, color=color,
                                 label=esc + ' + noise scenario ' + str(i+1))
                    
        else:
            if type(esc) is list:
                for e in esc:
                    T_a, data_a = self.a.sh2timeseries(T, e, unitFrom, unitTo,
                                                       t_ref=t_ref, reset=reset,
                                                       filt=filt, lmax=lmax)
                    T_b, data_b = self.b.sh2timeseries(T, e, unitFrom, unitTo,
                                                       t_ref=t_ref, reset=reset,
                                                       filt=filt, lmax=lmax)
                    data_a, data_b = np.array(data_a), np.array(data_b)
                    if unitTo in('N', 'h', 'GRACE'):
                        data_a = data_a * 1e3
                        data_b = data_b * 1e3
                    plt.plot(T, data_b-data_a, label=e)
            else:
                T_a, data_a = self.a.sh2timeseries(T, esc, unitFrom, unitTo,
                                                   t_ref=t_ref, reset=reset,
                                                   filt=filt, lmax=lmax)
                T_b, data_b = self.b.sh2timeseries(T, esc, unitFrom, unitTo,
                                                   t_ref=t_ref, reset=reset,
                                                   filt=filt, lmax=lmax)
                data_a, data_b = np.array(data_a), np.array(data_b)
                if unitTo in('N', 'h', 'GRACE'):
                        data_a = data_a * 1e3
                        data_b = data_b * 1e3
                plt.plot(T, data_b-data_a, label=esc)
        
        plt.ylabel(unit_dict[unitTo])
        plt.xticks(rotation=90)
        plt.title(self.b.location + ' - ' + self.a.location)
        plt.grid()
        plt.legend()
        plt.tight_layout()
        
        np.save(cfg.PATHS['data_path'] + 'ts_results/' + esc + '_' + unitTo +
                '_' + self.b.location + '-' + self.a.location, data_b-data_a)
        for i, noi in enumerate(noise):
            np.save(cfg.PATHS['data_path'] + 'ts_results/' + esc + '_' +
                    unitTo + '_' + self.b.location + '-' + self.a.location +
                    '_noise' + str(i), data+noi)
            
        if save:
            path = (cfg.PATHS['fig_path'] + 'timeseries/' + esc + '_' + unitTo
                    + '_' + self.a.location 
                    + '_' + self.b.location + '.pdf')
            plt.savefig(path)
            #TODO: case for esc is list
        
        return data_b-data_a
        
    def plotTimeFrequencies(self, T, esc, unitFrom, unitTo, delta_t,
                            fmax=False, t_ref=False, reset=False, error=False,
                            sigma=False, filt=False, save=False):
        """Plots time series in frequency domain at clock location.
        
        Uses sh2timeseries() and plots the resulting time series in frequency
        domain at clock location.
        
        :param T: list of dates
        :type T: str or datetime.date(time)
        :param esc: earth system component(s)
        :type esc: str or list of str
        :param unitFrom: unit of the input coefficients
        :type unitFrom: str
        :param unitTo: unit of the timeseries
        :type unitTo: str 
        :param t_ref: reference time for the series
        :type t_ref: str or datetime.date(time), optional
        :param reset: shall the timeseries be calculated again
        :type reset: boolean, optional
        :param error: the clock error type: 'white' or 'allan'
        :type error: str, optional
        :param sigma: uncertainty of the clock
        :type sigma: float, optional
        :param delta_t: measurement sampling rate [s]
        :type delta_t: float
        :param fmax: maximum plottable frequency band
        :type fmax: integer, optional
        :param filt: filter width for the data, has to be an odd number
        :type filt: integer, optional
        
        Possible units:
            'pot' ... dimensionless Stokes coeffs (e.g. GRACE L2)
                'U' ... geopotential [m^2/s^2]
                'N' ... geoid height [m]
            'h' ... elevation [m]
            'ff' ... fractional frequency [-]
            'mass' ... dimensionless surface loading coeffs
                'sd' ... surface density [kg/m^2]
                'ewh' ... equivalent water height [m]
            'gravity'... [m/s^2]
            
        Possible earth system components:
            'I' ... Ice
            'H' ... Hydrology
            'A' ... Atmosphere
            'GIA'.. Glacial Isostatic Adjustment
        """

        np.random.seed(7)
        unit_dict = {'U': 'gravitational potential [m$^2$/s$^2$]',
                    'N': 'Geoid height [mm]',
                    'h': 'Elevation [mm]',
                    'sd': 'Surface Density [kg/m$^2$]',
                    'ewh': 'Equivalent water height [m]',
                    'gravity': 'gravitational acceleration [m/s$^2$]',
                    'ff': 'Fractional frequency [-]'}
        plt.rcParams.update({'font.size': 13})  # set before making the figure!        
        fig, ax = plt.subplots()
        # ax = plt.figure().gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        if sigma:
            if type(esc) is list:
                for e in esc:
                    T_a, data_a, noise_a = self.a.sh2timeseries(
                        T, e, unitFrom, unitTo, t_ref=t_ref, reset=reset,
                        filt=filt, error=error, sigma=sigma)
                    T_b, data_b, noise_b = self.b.sh2timeseries(
                        T, e, unitFrom, unitTo, t_ref=t_ref, reset=reset,
                        filt=filt, error=error, sigma=sigma)
                    data_a, data_b = np.array(data_a), np.array(data_b)
                    data = data_b - data_a
                    noise = np.array(noise_a) + np.array(noise_b)
                    f, freq = harmony.time2freq(delta_t, data)
                    # f, noisy_freq = harmony.time2freq(delta_t, data+noise)
                    noise_level = []
                    for noi in noise:
                        fn, noisy = harmony.time2freq(delta_t, noi)
                        noise_level.append(np.mean(noisy) * np.ones(len(f)))
                    if fmax:
                        f, freq, noise_level = (f[:fmax], freq[:fmax],
                                                [n[:fmax] for n
                                                 in noise_level])
                    # plt.plot(f*86400*365, noisy_freq, 'x', label=e+' + noise',
                    #          color=p[0].get_color())
                    # plt.plot(f*86400*365, noise_level, label='noise level for $\sigma$='+str(sigma),
                    #          color=p[0].get_color())
                    for lvl, sig in zip(noise_level, sigma):
                        plt.plot(f*86400*365, lvl,
                                 label='noise level for $\sigma$='+str(sig))
                    p = plt.plot(f[1:]*86400*365, freq[1:], '.-', label=e)
            else:
                T_a, data_a = self.a.sh2timeseries(
                    T, esc, unitFrom, unitTo, t_ref=t_ref, reset=reset,
                    filt=filt)
                T_b, data_b = self.b.sh2timeseries(
                    T, esc, unitFrom, unitTo, t_ref=t_ref, reset=reset,
                    filt=filt)
                data_a, data_b = np.array(data_a), np.array(data_b)
                data = data_b - data_a
                f, freq = harmony.time2freq(delta_t, data)
                # f, noisy_freq = harmony.time2freq(delta_t, data+noise)
                if fmax:
                    f, freq = (f[:fmax], freq[:fmax])
                noise = np.load('/home/schroeder/CLONETS/data/noise.npy')
                noise = noise * np.sqrt(2)
                plt.plot(f[1:]*86400*365, noise[0, :], '--', color='black',
                         label='noise levels')
                plt.plot(f[1:]*86400*365, noise[1, :], '--', color='black')
                plt.plot(f[1:]*86400*365, noise[2, :], '--', color='black')
                plt.plot(f[1:]*86400*365, freq[1:], '.-', label=esc)
                
                np.save(cfg.PATHS['data_path'] + 'ts_results/' + esc +
                        '_freq_' + self.b.location + '-' + self.a.location, freq[1:])
                
        else:
            if type(esc) is list:
                for e in esc:
                    T_a, data_a = self.a.sh2timeseries(
                        T, e, unitFrom, unitTo, t_ref=t_ref, reset=reset,
                        filt=filt)
                    T_b, data_b = self.b.sh2timeseries(
                        T, e, unitFrom, unitTo, t_ref=t_ref, reset=reset,
                        filt=filt)
                    data_a, data_b = np.array(data_a), np.array(data_b)
                    data = data_b - data_a
                    f, freq = harmony.time2freq(delta_t, data)
                    if fmax:
                        f, freq, (f[:fmax], freq[:fmax])
                    p = plt.plot(f[1:]*86400*365, freq[1:], '.-', label=e)
            else:
                T_a, data_a = self.a.sh2timeseries(
                    T, esc, unitFrom, unitTo, t_ref=t_ref, reset=reset,
                    filt=filt)
                T_b, data_b = self.b.sh2timeseries(
                    T, esc, unitFrom, unitTo, t_ref=t_ref, reset=reset,
                    filt=filt)
                data_a, data_b = np.array(data_a), np.array(data_b)
                data = data_b - data_a
                f, freq = harmony.time2freq(delta_t, data)
                if fmax:
                    f, freq = (f[:fmax], freq[:fmax])
                p = plt.plot(f[1:]*86400*365, freq[1:], '.-', label=esc)
        
        plt.title(self.b.location + ' - ' + self.a.location)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Frequencies [1/yr]')
        plt.ylabel(unit_dict[unitTo])
        plt.grid()
        # plt.legend()
        plt.legend(loc='lower left')
        plt.tight_layout()
        
        if save:
            path = (cfg.PATHS['fig_path'] + 'timeseries/' + esc + '_' + unitTo
                    + '_' + self.a.location
                    + '_' + self.b.location + '_spectral.pdf')
            plt.savefig(path)
            #TODO: case for esc is list

class Network():
    """A network of optical clocks.
    
    A class for the network of several optical clocks (class Clock). Consists
    of the clocks and the links (class Link) between the clocks and has helpful
    tools for visualisation.
    
    :param clocks: list of the clocks
    :type clocks: list of Clocks
    :param links: list of the links between the clocks
    :type links: list of Links
    """
    
    def __init__(self, name, reset=False):
        """Builds an optical clock network."""
        
        self.clocks = []
        self.links = []
        if name == 'clonets':
            braunschweig = Clock('Braunschweig', 52.2903, 10.4547, country='Germany')  # PTB
            wettzell = Clock('Wettzell', 49.1300, 12.900, country='Germany')  # observatory
            bonn = Clock('Bonn', 50.7250, 7.1000, country='Germany')  # uni bonn
            munich = Clock('Munich', 48.1600, 11.5000, country='Germany')  # tu munich
            potsdam = Clock('Potsdam', 52.3826, 13.0642, country='Germany')  # GFZ
            delft = Clock('Delft', 51.9900, 4.4000, country='Netherlands')  # VSL
            london = Clock('London', 51.4200, 0.3400, country='UK')  # NPL
            paris = Clock('Paris', 48.8300, 2.2200, country='France')  # OBSPARIS
            strasbourg = Clock('Strasbourg', 48.5841, 7.7633, country='France')  # uni strasbourg
            bern = Clock('Bern', 46.9300, 7.4500, country='Switzerland')  # METAS
            torino = Clock('Torino', 45.0000, 7.6300, country='Italy')  # INRIM
            vienna = Clock('Vienna', 48.2000, 16.3700, country='Austria')  # BEV
            ljubljana = Clock('Ljubljana', 46.0473, 14.4932, country='Slovenia')  # SIQ
            prague = Clock('Prague', 50.0700, 14.5300, country='Czech Republic')  # CMI
            torun = Clock('Torun', 53.0000, 18.6038, country='Poland')  # AMO (?)
            posen = Clock('Posen', 52.4071, 16.9538, country='Poland')  # PSNC
            warsaw = Clock('Warsaw', 52.2000, 21.0000, country='Poland')  # GUM
            helsinki = Clock('Helsinki', 60.1800, 24.8257, country='Finland')  # VTT MIKES
            gothenburg = Clock('Gothenburg', 57.6858, 11.9556, country='Sweden')  # RISE
            
            self.add_clock(braunschweig)
            self.add_clock(wettzell)
            self.add_clock(bonn)
            self.add_clock(munich)
            self.add_clock(potsdam)
            self.add_clock(delft)
            self.add_clock(london)
            self.add_clock(paris)
            self.add_clock(strasbourg)
            self.add_clock(bern)
            self.add_clock(torino)
            self.add_clock(vienna)
            self.add_clock(ljubljana)
            self.add_clock(prague)
            self.add_clock(torun)
            self.add_clock(warsaw)
            self.add_clock(posen)
            self.add_clock(helsinki)
            self.add_clock(gothenburg)
            
            # existing
            self.add_link(braunschweig, strasbourg, 0)
            self.add_link(braunschweig, munich, 0)
            self.add_link(paris, strasbourg, 0)
            self.add_link(paris, london, 0)
            self.add_link(warsaw, posen, 0)
            self.add_link(torun, posen, 0)
            self.add_link(prague, vienna, 0)
            # imaginable
            self.add_link(bonn, braunschweig, 9)
            self.add_link(bonn, strasbourg, 9)
            self.add_link(bonn, delft, 9)
            self.add_link(wettzell, munich, 9)
            self.add_link(wettzell, prague, 9)
            self.add_link(posen, prague, 9)
            # self.add_link(braunschweig, prague, 9)
            # extension 1
            self.add_link(gothenburg, helsinki, 1)
            self.add_link(munich, strasbourg, 1)
            self.add_link(braunschweig, potsdam, 1)
            self.add_link(braunschweig, delft, 1)
            self.add_link(braunschweig, gothenburg, 1)
            self.add_link(potsdam, posen, 1)
            self.add_link(strasbourg, bern, 1)
            self.add_link(bern, torino, 1)
            self.add_link(paris, torino, 1)
            self.add_link(munich, vienna, 1)
            self.add_link(vienna, warsaw, 1)
            # extension 2
            self.add_link(london, delft, 2)
            self.add_link(torino, ljubljana, 2)
            self.add_link(ljubljana, vienna, 2)
            self.add_link(warsaw, helsinki, 2)
            # reference Braunschweig
            self.add_link(braunschweig, bern, 5)
            self.add_link(braunschweig, helsinki, 5)
            self.add_link(braunschweig, london, 5)
        elif name == 'euref':
            filename = '/home/schroeder/CLONETS/data/EUREF_Permanent_GNSS_Network.csv'
            stations = utils.load_euref(filename)
            
            north = 65
            south = 40
            west = -10
            east = 30
            for station in stations:
                name = station[0]
                lat = station[1]
                lon = station[2]
                if lon > 180:
                    lon = lon - 360
                if lon > west and lon < east and lat < north and lat > south:
                    self.add_clock(Clock(name, lat, lon))
            
            braunschweig = Clock('Braunschweig', 52.2903, 10.4547, country='Germany')  # PTB
            wettzell = Clock('Wettzell', 49.1300, 12.900, country='Germany')  # observatory
            bonn = Clock('Bonn', 50.7250, 7.1000, country='Germany')  # uni bonn
            munich = Clock('Munich', 48.1600, 11.5000, country='Germany')  # tu munich
            potsdam = Clock('Potsdam', 52.3826, 13.0642, country='Germany')  # GFZ
            delft = Clock('Delft', 51.9900, 4.4000, country='Netherlands')  # VSL
            london = Clock('London', 51.4200, 0.3400, country='UK')  # NPL
            paris = Clock('Paris', 48.8300, 2.2200, country='France')  # OBSPARIS
            strasbourg = Clock('Strasbourg', 48.5841, 7.7633, country='France')  # uni strasbourg
            bern = Clock('Bern', 46.9300, 7.4500, country='Switzerland')  # METAS
            torino = Clock('Torino', 45.0000, 7.6300, country='Italy')  # INRIM
            vienna = Clock('Vienna', 48.2000, 16.3700, country='Austria')  # BEV
            ljubljana = Clock('Ljubljana', 46.0473, 14.4932, country='Slovenia')  # SIQ
            prague = Clock('Prague', 50.0700, 14.5300, country='Czech Republic')  # CMI
            torun = Clock('Torun', 53.0000, 18.6038, country='Poland')  # AMO (?)
            posen = Clock('Posen', 52.4071, 16.9538, country='Poland')  # PSNC
            warsaw = Clock('Warsaw', 52.2000, 21.0000, country='Poland')  # GUM
            helsinki = Clock('Helsinki', 60.1800, 24.8257, country='Finland')  # VTT MIKES
            gothenburg = Clock('Gothenburg', 57.6858, 11.9556, country='Sweden')  # RISE
            
            self.add_clock(braunschweig)
            self.add_clock(wettzell)
            self.add_clock(bonn)
            self.add_clock(munich)
            self.add_clock(potsdam)
            self.add_clock(delft)
            self.add_clock(london)
            self.add_clock(paris)
            self.add_clock(strasbourg)
            self.add_clock(bern)
            self.add_clock(torino)
            self.add_clock(vienna)
            self.add_clock(ljubljana)
            self.add_clock(prague)
            self.add_clock(torun)
            self.add_clock(warsaw)
            self.add_clock(posen)
            self.add_clock(helsinki)
            self.add_clock(gothenburg)
        elif name == 'psmsl' or name == 'psmsl_only':
            filename = '/home/schroeder/CLONETS/data/PSMSL_stations.json'
            stations = utils.load_psmsl(filename)
            
            for station in stations:
                n = station[0]
                lat = station[1]
                lon = station[2]
                self.add_clock(Clock(n, lat, lon))
            
            if name == 'psmsl':
                braunschweig = Clock('Braunschweig', 52.2903, 10.4547, country='Germany')  # PTB
                wettzell = Clock('Wettzell', 49.1300, 12.900, country='Germany')  # observatory
                bonn = Clock('Bonn', 50.7250, 7.1000, country='Germany')  # uni bonn
                munich = Clock('Munich', 48.1600, 11.5000, country='Germany')  # tu munich
                potsdam = Clock('Potsdam', 52.3826, 13.0642, country='Germany')  # GFZ
                delft = Clock('Delft', 51.9900, 4.4000, country='Netherlands')  # VSL
                london = Clock('London', 51.4200, 0.3400, country='UK')  # NPL
                paris = Clock('Paris', 48.8300, 2.2200, country='France')  # OBSPARIS
                strasbourg = Clock('Strasbourg', 48.5841, 7.7633, country='France')  # uni strasbourg
                bern = Clock('Bern', 46.9300, 7.4500, country='Switzerland')  # METAS
                torino = Clock('Torino', 45.0000, 7.6300, country='Italy')  # INRIM
                vienna = Clock('Vienna', 48.2000, 16.3700, country='Austria')  # BEV
                ljubljana = Clock('Ljubljana', 46.0473, 14.4932, country='Slovenia')  # SIQ
                prague = Clock('Prague', 50.0700, 14.5300, country='Czech Republic')  # CMI
                torun = Clock('Torun', 53.0000, 18.6038, country='Poland')  # AMO (?)
                posen = Clock('Posen', 52.4071, 16.9538, country='Poland')  # PSNC
                warsaw = Clock('Warsaw', 52.2000, 21.0000, country='Poland')  # GUM
                helsinki = Clock('Helsinki', 60.1800, 24.8257, country='Finland')  # VTT MIKES
                gothenburg = Clock('Gothenburg', 57.6858, 11.9556, country='Sweden')  # RISE
                
                self.add_clock(braunschweig)
                self.add_clock(wettzell)
                self.add_clock(bonn)
                self.add_clock(munich)
                self.add_clock(potsdam)
                self.add_clock(delft)
                self.add_clock(london)
                self.add_clock(paris)
                self.add_clock(strasbourg)
                self.add_clock(bern)
                self.add_clock(torino)
                self.add_clock(vienna)
                self.add_clock(ljubljana)
                self.add_clock(prague)
                self.add_clock(torun)
                self.add_clock(warsaw)
                self.add_clock(posen)
                self.add_clock(helsinki)
                self.add_clock(gothenburg)
        elif name == 'eike_sg':
            filename = '/home/schroeder/CLONETS/data/NGL_gnss_Stations.txt'
            stations = utils.load_ngl(filename)
            
            north = 65
            south = 40
            west = -10
            east = 30
            for station in stations:
                name = station[0]
                lat = station[1]
                lon = station[2]
                if lon > 180:
                    lon = lon - 360
                if lon > west and lon < east and lat < north and lat > south:
                    self.add_clock(Clock(name, lat, lon))
            
            braunschweig = Clock('Braunschweig', 52.2903, 10.4547, country='Germany')  # PTB
            wettzell = Clock('Wettzell', 49.1300, 12.900, country='Germany')  # observatory
            bonn = Clock('Bonn', 50.7250, 7.1000, country='Germany')  # uni bonn
            munich = Clock('Munich', 48.1600, 11.5000, country='Germany')  # tu munich
            potsdam = Clock('Potsdam', 52.3826, 13.0642, country='Germany')  # GFZ
            delft = Clock('Delft', 51.9900, 4.4000, country='Netherlands')  # VSL
            london = Clock('London', 51.4200, 0.3400, country='UK')  # NPL
            paris = Clock('Paris', 48.8300, 2.2200, country='France')  # OBSPARIS
            strasbourg = Clock('Strasbourg', 48.5841, 7.7633, country='France')  # uni strasbourg
            bern = Clock('Bern', 46.9300, 7.4500, country='Switzerland')  # METAS
            torino = Clock('Torino', 45.0000, 7.6300, country='Italy')  # INRIM
            vienna = Clock('Vienna', 48.2000, 16.3700, country='Austria')  # BEV
            ljubljana = Clock('Ljubljana', 46.0473, 14.4932, country='Slovenia')  # SIQ
            prague = Clock('Prague', 50.0700, 14.5300, country='Czech Republic')  # CMI
            torun = Clock('Torun', 53.0000, 18.6038, country='Poland')  # AMO (?)
            posen = Clock('Posen', 52.4071, 16.9538, country='Poland')  # PSNC
            warsaw = Clock('Warsaw', 52.2000, 21.0000, country='Poland')  # GUM
            helsinki = Clock('Helsinki', 60.1800, 24.8257, country='Finland')  # VTT MIKES
            gothenburg = Clock('Gothenburg', 57.6858, 11.9556, country='Sweden')  # RISE
            
            self.add_clock(braunschweig)
            self.add_clock(wettzell)
            self.add_clock(bonn)
            self.add_clock(munich)
            self.add_clock(potsdam)
            self.add_clock(delft)
            self.add_clock(london)
            self.add_clock(paris)
            self.add_clock(strasbourg)
            self.add_clock(bern)
            self.add_clock(torino)
            self.add_clock(vienna)
            self.add_clock(ljubljana)
            self.add_clock(prague)
            self.add_clock(torun)
            self.add_clock(warsaw)
            self.add_clock(posen)
            self.add_clock(helsinki)
            self.add_clock(gothenburg)
        else:
            print('Network name not found! Network empty!')
            
    def __repr__(self):
        """To be printed if instance is written."""
        
        summary = '<Network of optical clocks>\n'
        for c in self.clocks:
            summary += c.location + '\n'
        return summary + '\n'
            
    def search_clock(self, attr, value):
        """Search for a clock in the network.
        
        :param attr: the attribute via which is searched
        :type attr: str
        :param value: the value which is searched for or a range (if searched
        via lat or lon)
        :type value: str or tupel of 2 floats (if searched via lat or lon)
        :return: list of the fitting clocks
        :rtype: list of Clocks
        """
        
        clos = []
        if attr == 'location':
            for clo in self.clocks:
                if clo.location == value:
                    clos.append(clo)
        elif attr == 'country':
            for clo in self.clocks:
                if clo.country == value:
                    clos.append(clo)
        elif attr == 'lat':
            for clo in self.clocks:
                if clo.lat > value[0] and clo.lat < value[1]:
                    clos.append(clo)
        elif attr == 'lon':
            for clo in self.clocks:
                if clo.lon > value[0] and clo.lon < value[1]:
                    clos.append(clo)
        return clos
    
    def search_link(self, attr, value):
        """Search for a link in the network.
        
        :param attr: the attribute via which is searched
        :type attr: str
        :param value: the value which is searched for
        :type value: str or tupel of 2 floats (if searched via lat or lon)
        :return: list of the fitting links
        :rtype: list of Links
        
        Possible Attibutes:
            - location ... one location of the link
            - locations ... both locations of the link
            - distance ... tuple or list or numpy.array of min and max distance
                           in [km]
        """
        
        l = []
        if attr == 'location':
            for li in self.links:
                if li.a.location == value or li.b.location == value:
                    l.append(li)
        elif attr == 'locations':
            for li in self.links:
                if (li.a.location == value[0] and li.b.location == value[1] or
                    li.a.location == value[1] and li.b.location == value[0]):
                    l.append(li)
        elif attr == 'distance':
            for li in self.links:
                if li.length() > value[0] and li.length() < value[1]:
                    l.append(li)
                
        print(str(len(l)) + ' link(s) found.')
        return l

    def add_clock(self, clo):
        """Adds a clock to the network.
        
        :param clock: an optical clock
        :type clock: Clock
        :param links: links of the clock to other clocks in the network
        :type links: list of names or clocks, optional
        """
        
        if clo not in self.clocks:
            self.clocks.append(clo)
        else:
            print('Warning: Clock ' + clo + ' is already in the network!')
        
    def add_link(self, a, b, state=9):
        """Build a fibre link between two clocks a and b."""
        
        L = Link(a, b, state)
        self.links.append(L)
        a.link_to(b, state)
        b.link_to(a, state)
        
    def lats(self):
        """Returns an array of the latitudes of all clocks."""
        
        lats = [clo.lat for clo in self.clocks]
        return np.array(lats)
        
    def lons(self):
        """Returns an array of the longitudes of all clocks."""
        
        lons = [clo.lon for clo in self.clocks]
        return np.array(lons)
    
    def _sh2timeseries(self, F_lm, t, kind, unit, unitTo=[]):
        """Expand spherical harmonics into timeseries at each clock location.
        
        DEPRECATED!
        
        :param F_lm: spherical harmonic coefficients
        :type F_lm: list of pyshtools.SHCoeffs
        :param t: time vector of decimal years
        :type t: numpy.array
        :param kind: origin of the mass redistribution
        :type kind: str
        :param unit: the input unit of the coefficients
        :type unit: str
        
        Possible units:
            'pot' ... dimensionless Stokes coeffs (e.g. GRACE L2)
                'U' ... geopotential [m^2/s^2]
                'N' ... geoid height [m]
            'h' .... elevation [m]
            'mass' ... dimensionless surface loading coeffs
                'sd' ... surface density [kg/m^2]
                'ewh' ... equivalent water height [m]
            'gravity'... [m/s^2]
            
        Possible kinds:
            'I' ... Ice
            'H' ... Hydrology
            'A' ... Atmosphere
            'O' ... Ocean
            'GIA'.. Glacial Isostatic Adjustment
            'S' ... Solid Earth            
        """
        
        for clo in self.clocks:
            clo.sh2timeseries(F_lm, t, kind, unit, unitTo=unitTo)
            
    def h_N_2ff(self, effect_names):
        """Convert elevation and geoid change to ff change for all clocks."""
        
        for effect_name in effect_names:
            for clo in self.clocks:
                effect = getattr(clo, effect_name)
                effect['ff'] = clo.h2ff(effect['h']) + clo.N2ff(effect['N'])
            
    def to_file(self):
        """Writes the clock info of all network clocks into netcdf and json
        files."""
        
        for clo in self.clocks:
            clo.to_file()

    def plot(self, background='TerrainBackground'):
        """Plots the network on top of a basemap.
        
        Deprecated!"""
        
        print('WARNING: This plot-function is deprecated. Use plotNetwork()'
              + ' instead!')
       
        plt.rcParams.update({'font.size': 13})  # set before making the figure!        
        # Clocks = Points
        df = pd.DataFrame({'City': [], 'Country': [], 'Latitude': [],
                           'Longitude': []})
        for clo in self.clocks:
            df = df.append([{'City': clo.location, 'Country': clo.country,
                             'Latitude': clo.lat, 'Longitude': clo.lon}],
                           ignore_index=True)
        gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.Longitude,
                                                               df.Latitude))
        gdf.crs = {'init': 'epsg:4326', 'no_defs': True}
        gdf = gdf.to_crs(epsg=3857)
        
        # Links = LineStrings
        df_links = pd.DataFrame({'From': [], 'To': [], 'length': []})
        links = []
        for l in self.links:
            links.append(LineString([Point(l.a.lon, l.a.lat),
                                     Point(l.b.lon, l.b.lat)]))
            df_links = df_links.append([{'From': l.a.location,
                                         'To': l.b.location,
                                         'length': l.length()}])
        gdf_links = gpd.GeoDataFrame(df_links, geometry=links)
        gdf_links.crs = {'init': 'epsg:4326', 'no_defs': True}
        gdf_links = gdf_links.to_crs(epsg=3857)
        
        # Plot
        ax = gdf.plot(figsize=(10, 10), color='red', edgecolor='k')
        gdf_links.plot(ax=ax, color='red')
        for idx, row in gdf.iterrows():
#            plt.annotate(s=row['City'], xy=(row.geometry.x, row.geometry.y))
            plt.text(row.geometry.x+3e4, row.geometry.y, s=row['City'],
                     bbox={'facecolor': 'white', 'alpha':0.8, 'pad': 2,
                           'edgecolor':'none'})
        if background == 'TerrainBackground':
            ctx.add_basemap(ax, url=ctx.providers.Stamen.TerrainBackground) 
        elif background == 'toner':
            ctx.add_basemap(ax, url=ctx.providers.Stamen.TonerBackground)
        elif background == 'night':
            ctx.add_basemap(ax, url=ctx.providers.NASAGIBS.ViirsEarthAtNight2012)
        
        return gdf, gdf_links
    
    def plotNetwork(self, zoom='close', save=False):
        """."""
             
        if zoom == 'close':
            region = [0, 26, 44, 62]
            projection = 'S13/90/6i'
        elif zoom == 'far':
            region = [-20, 40, 30, 70]
            projection = 'S10/90/6i'
        else:
            region = [-10, 30, 40, 65]
            projection = 'S10/90/6i'
        relief = pygmt.datasets.load_earth_relief('30s')
        
        fig = pygmt.Figure()
        pygmt.makecpt(cmap='wiki-2.0', series=[-3172*0.45, 6000*0.45])
        fig.grdimage(relief, region=region,
                      projection=projection, frame="ag")
        fig.coast(region=region, projection=projection, frame="a",
                  shorelines="1/0.3p,black", borders="1/0.3p,black")
        fig.plot(self.lons(), self.lats(), style="c0.09i", color="white",
                  pen="black")
        s0 = False
        s1 = False
        s2 = False
        s9 = False
        for l in self.links:
            x = [l.a.lon, l.b.lon]
            y = [l.a.lat, l.b.lat]
            if l.state == 0:
                if s0:
                    fig.plot(x, y, pen="1p,red")
                else:
                    fig.plot(x, y, pen="1p,red", label='"Existing links"')
                    s0 = True
            elif l.state == 1:
                if s1:
                    fig.plot(x, y, pen="1p,blue,-")
                else:
                    fig.plot(x, y, pen="1p,blue,-", label='"Possible links"')
                    s1 = True
            elif l.state == 2:
                if s2:
                    fig.plot(x, y, pen="1p,blue,-")
                else:
                    fig.plot(x, y, pen="1p,blue,-")#, label='"Phase 2"')
                    s2 = True
            elif l.state in [1, 2, 9]:
                if s9:
                    fig.plot(x, y, pen="1p,blue,-")
                else:
                    fig.plot(x, y, pen="1p,blue,-")
                    s9 = True
            
        for clo in self.clocks:
            if clo.location == 'Potsdam':
                fig.text(x=clo.lon-0.3, y=clo.lat+0.6, text=clo.location, region=region,
                         projection=projection, font='12p,Helvetica,black',
                         justify='LT')
            elif clo.location == 'Helsinki':
                fig.text(x=clo.lon-3.3, y=clo.lat+1, text=clo.location, region=region,
                         projection=projection, font='12p,Helvetica,black',
                         justify='LT')
            elif clo.location == 'Posen':
                fig.text(x=clo.lon+0.3, y=clo.lat, text='Poznan', region=region,
                         projection=projection, font='12p,Helvetica,black',
                         justify='LT')
            else:
                fig.text(x=clo.lon+0.3, y=clo.lat, text=clo.location, region=region,
                         projection=projection, font='12p,Helvetica,black',
                         justify='LT')
        fig.legend(position='g2/59', box='+gwhite+p1p')
        
        if save:
            fig.savefig('/home/schroeder/CLONETS/fig/clonets_gmt.pdf')
        
        return fig
    
    def plotESC(self, esc, t, unitFrom, unitTo, t_ref=None, save=False,
                degreevariances=False, relief=False, lmin=False, lmax=False,
                world=False):
        """Plots the earth system component signal on a map.
        
        :param esc: earth system component
        :type esc: str
        :param t: point in time
        :type t: datetime.datetime or str
        :param unitFrom: unit in which the data is stored
        :type unitFrom: str
        :param unitTo: unit that is to be plotted
        :type unitTo: str
        :param t_ref: optional reference point in time
        :type t_ref: datetime.datetime or str (must match t)
        :return: the figure object
        :rtype: pygmt.Figure
        
        Possible units:
            'U' ... geopotential [m^2/s^2]
            'N' ... geoid height [m]
            'h' .... elevation [m]
            'sd' ... surface density [kg/m^2]
            'ewh' ... equivalent water height [m]
            'gravity'... [m/s^2]
            'ff' ... fractional frequency [-]
            'GRACE' ... geoid height [m], but with lmax=120 and filtered
            
        Possible esc's:
            'I' ... Ice
            'H' ... Hydrology
            'A' ... Atmosphere
            'GIA'.. Glacial Isostatic Adjustment
        """
        
        cb_dict = {'U': '"gravitational potential [m@+2@+/s@+2@+]"',
                     'N': '"Geoid height [mm]"',
                     'h': '"Elevation [mm]"',
                     'sd': '"Surface Density [kg/m@+2@+]"',
                     'ewh': '"Equivalent water height [m]"',
                     'ewhGRACE': '"Equivalent water height [m]"',
                     'gravity': '"gravitational acceleration [m/s@+2@+]"',
                     'ff': '"Fractional frequency [-]"',
                     'GRACE': '"Geoid height [mm]"'}
        esc_dict = {'I': 'oggm_',
                    'H': 'clm_tws_',
                    'H_snow': 'clm_tws_',
                    'A': 'coeffs_',
                    'O': 'O_',
                    'GRACE_ITSG2018': 'ITSG_Grace2018_n120_'}

        path = cfg.PATHS['data_path']
        savepath = path + '../fig/'
        
        if type(t) is not str:
            t = datetime.datetime.strftime(t, format='%Y_%m_%d')
            if t_ref:
                t_ref = datetime.datetime.strftime(t_ref, format='%Y_%m_%d')
        f_lm = harmony.shcoeffs_from_netcdf(
            os.path.join(path, esc, esc_dict[esc] + t + '.nc'))
        if t_ref:
            f_lm_ref = harmony.shcoeffs_from_netcdf(
                os.path.join(path, esc, esc_dict[esc] + t_ref + '.nc'))
            f_lm.coeffs = f_lm.coeffs - f_lm_ref.coeffs
        f_lm = harmony.sh2sh(f_lm, unitFrom, unitTo)
        if lmin or lmax:
            if lmin and not lmax:
                f_lm = f_lm.pad(720) - f_lm.pad(lmin).pad(720)
            elif lmax and not lmin:
                f_lm = f_lm.pad(lmax).pad(720)
            elif lmin and lmax:
                f_lm = f_lm.pad(lmax).pad(720) - f_lm.pad(lmin).pad(720)
        
        if degreevariances:
            return f_lm
        
        grid = f_lm.pad(720).expand()
        if unitTo in('N', 'h', 'GRACE'):
            grid.data = grid.data * 1e3
        
        if world:
            fig = self.plot_world_720(grid, path+esc+'/', '', esc, unitTo,
                                   cb_dict, 'esc', t)
        else:
            fig = self.plot_europe_720(grid, path+esc+'/', '', esc, unitTo,
                                       cb_dict, 'esc', t)
        
        if save:
            if save=='png':
                if t_ref:
                    savename = (os.path.join(savepath, esc, esc_dict[esc] + t + '-'
                                             + t_ref + '_' + unitTo + '.png'))
                else:
                    savename = (os.path.join(savepath, esc, esc_dict[esc] + t
                                             + '_' + unitTo + '.png'))
                if lmin:
                    savename = savename[:-4] + 'lmin' + str(lmin) + '.png'
                if lmax:
                    savename = savename[:-4] + 'lmax' + str(lmax) + '.png'
                if world:
                    savename = savename[:-4] + '_world.png'
            else:
                if t_ref:
                    savename = (os.path.join(savepath, esc, esc_dict[esc] + t + '-'
                                             + t_ref + '_' + unitTo + '.pdf'))
                else:
                    savename = (os.path.join(savepath, esc, esc_dict[esc] + t
                                             + '_' + unitTo + '.pdf'))
                if lmin:
                    savename = savename[:-4] + 'lmin' + str(lmin) + '.pdf'
                if lmax:
                    savename = savename[:-4] + 'lmax' + str(lmax) + '.pdf'
                if world:
                    savename = savename[:-4] + '_world.png'
            fig.savefig(savename)
        return fig, grid
    
    def plotESCatClocks(self, esc, t, unitFrom, unitTo, t_ref=None,
                        loc_ref=False, save=False, world=False):
        """Plots the earth system component signal on a map.
        
        :param esc: earth system component
        :type esc: str
        :param t: point in time
        :type t: datetime.datetime or str
        :param unitFrom: unit in which the data is stored
        :type unitFrom: str
        :param unitTo: unit that is to be plotted
        :type unitTo: str
        :param t_ref: optional reference point in time
        :type t_ref: datetime.datetime or str (must match t)
        :return: the figure object
        :rtype: pygmt.Figure
        
        Possible units:
            'U' ... geopotential [m^2/s^2]
            'N' ... geoid height [m]
            'h' .... elevation [m]
            'sd' ... surface density [kg/m^2]
            'ewh' ... equivalent water height [m]
            'gravity'... [m/s^2]
            'ff' ... fractional frequency [-]
            'GRACE' ... geoid height [m], but with lmax=120 and filtered
            
        Possible esc's:
            'I' ... Ice
            'H' ... Hydrology
            'A' ... Atmosphere
            'O' ... Ocean
            'GIA'.. Glacial Isostatic Adjustment
        """
        
        cb_dict = {'U': '"gravitational potential [m@+2@+/s@+2@+]"',
                     'N': '"Geoid height [mm]"',
                     'h': '"Elevation [mm]"',
                     'sd': '"Surface Density [kg/m@+2@+]"',
                     'ewh': '"Equivalent water height [m]"',
                     'ewhGRACE': '"Equivalent water height [m]"',
                     'gravity': '"gravitational acceleration [m/s@+2@+]"',
                     'ff': '"Fractional frequency [-]"',
                     'GRACE': '"Geoid height [mm]"'}
        esc_dict = {'I': 'oggm_',
                    'H': 'clm_tws_',
                    'A': 'coeffs_',
                    'O': 'O_',
                    'GRACE_ITSG2018': 'ITSG_Grace2018_n120_'}

        path = cfg.PATHS['data_path']
        savepath = path + '../fig/'
        
        if type(t) is not str:
            t = datetime.datetime.strftime(t, format='%Y_%m_%d')
            if t_ref:
                t_ref = datetime.datetime.strftime(t_ref, format='%Y_%m_%d')
        f_lm = harmony.shcoeffs_from_netcdf(
            os.path.join(path, esc, esc_dict[esc] + t + '.nc'))
        if t_ref:
            f_lm_ref = harmony.shcoeffs_from_netcdf(
                os.path.join(path, esc, esc_dict[esc] + t_ref + '.nc'))
            f_lm = f_lm - f_lm_ref
        
        f_lm = harmony.sh2sh(f_lm, unitFrom, unitTo)
        points = np.array(
            [f_lm.expand(lat=clo.lat, lon=clo.lon) for clo in self.clocks])
        
        if loc_ref:
            clo = self.search_clock('location', loc_ref)[0]
            point_ref = f_lm.expand(lat=clo.lat, lon=clo.lon)
            points = points - point_ref
            
        if unitTo in('N', 'h', 'GRACE'):
            points = points * 1e3
        datamax = 4#max(abs(points))
        print(points)
        print(min(points), max(points))
        
        data = {'data': points, 'lat': self.lats(), 'lon': self.lons()}
        df = pd.DataFrame(data)

        fig = pygmt.Figure()
        with pygmt.clib.Session() as session:
            if world:
                session.call_module('gmtset', 'FONT 15p')
            else:
                session.call_module('gmtset', 'FONT 20p')
        pygmt.makecpt(cmap='roma', series=[-datamax, datamax], reverse=False)
        # fig.coast(region=[-10, 30, 40, 65], projection="S10/90/6i", frame=["a", "WSne"],
        #           shorelines="1/0.1p,black", borders="1/0.1p,black",
        #           land='grey')
        if world:
            fig.coast(projection="R15c", region="d", frame=["a", "WSne"],
                      shorelines="1/0.1p,black", borders="1/0.1p,black",
                      land='grey')
        else:
            fig.coast(region=[-30, 30, 20, 75], projection="S0/90/6i", frame=["a", "WSne"],
                      shorelines="1/0.1p,black", borders="1/0.1p,black",
                      land='grey')
        # TODO: könnte colorbar zero nicht in der mitte haben... überprüfen!
        fig.plot(x=df.lon, y=df.lat, style='c0.11i', color=(df.data/4.05+1)/2,
                 cmap='roma')
        with pygmt.clib.Session() as session:
            if world:
                session.call_module('gmtset', 'FONT 18p')
            else:
                session.call_module('gmtset', 'FONT 24p')       
        fig.colorbar(position='JMR+w13c/0.5c', frame='paf+l' + cb_dict[unitTo])  # @+x@+ for ^x
        
        if save:
            if save=='png':
                if t_ref:
                    savename = (os.path.join(savepath, esc, esc_dict[esc] + t + '-'
                                             + t_ref + '_' + unitTo + '_clockwise.png'))
                else:
                    savename = (os.path.join(savepath, esc, esc_dict[esc] + t
                                             + '_' + unitTo + '_clockwise.png'))
                if world:
                    savename = savename[:-4] + '_world.png'
            else:
                if t_ref:
                    savename = (os.path.join(savepath, esc, esc_dict[esc] + t + '-'
                                             + t_ref + '_' + unitTo + '_clockwise.pdf'))
                else:
                    savename = (os.path.join(savepath, esc, esc_dict[esc] + t
                                             + '_' + unitTo + '_clockwise.pdf'))
                if world:
                    savename = savename[:-4] + '_world.png'
            fig.savefig(savename)

        return fig, points
    
    def plotRMS(self, T, esc, unitFrom, unitTo, reset=False, save=False,
                trend=None, lmin=False, lmax=False, hourly=False):
        """Plots the Root Mean Square on a map.
        
        Expands the spherical harmonics from the data folder for each grid
        point and each time step. Then computes the root mean square error for
        each of the grid point time series.

        :param T: list of dates
        :type T: str or datetime.date(time)
        :param esc: earth system component
        :type esc: str
        :param unitFrom: unit of the input coefficients
        :type unitFrom: str
        :param unitTo: unit of the timeseries
        :type unitTo: str
        :param reset: shall the timeseries be calculated again
        :type reset: boolean, optional
        :param save: shall the figure be saved
        :type save: boolean
        :param trend: shall a trend be subtracted
        :type trend: str or odd int, optional
        :return fig: the figure object
        :rtype fig: pygmt.Figure
        :return grid: the plottet data grid
        :rtype grid: pyshtools.SHGrid
        
        Possible trends:
            'linear' ... just the linear trend
            'trend' ... trend is plotted instead of the RMS
            as number: width of the moving average filter, thus the cutoff
                       period length; has to be an odd integer
        
        Possible units:
            'pot' ... dimensionless Stokes coeffs (e.g. GRACE L2)
                'U' ... geopotential [m^2/s^2]
                'N' ... geoid height [m]
                'GRACE' ... geoid height [m], but with lmax=120 and filtered
            'h' ... elevation [m]
            'ff' ... fractional frequency [-]
            'mass' ... dimensionless surface loading coeffs
                'sd' ... surface density [kg/m^2]
                'ewh' ... equivalent water height [m]
            'gravity'... [m/s^2]
            
        Possible earth system components:
            'I' ... Ice
            'H' ... Hydrology
            'A' ... Atmosphere
            'GIA'.. Glacial Isostatic Adjustment
        """
        
        esc_dict = {'I': 'oggm_',
                    'H': 'clm_tws_',
                    'A': 'coeffs_'}
        cb_dict = {'U': '"RMS of gravitational potential [m@+2@+/s@+2@+]"',
                   'N': '"Geoid height variability [mm]"',
                   'h': '"RMS of Elevation [mm]"',
                   'sd': '"RMS of Surface Density [kg/m@+2@+]"',
                   'ewh': '"Mass variability in EWH [m]"',
                   'gravity': '"RMS of gravitational acceleration [m/s@+2@+]"',
                   'ff': '"RMS of Fractional frequency [-]"',
                   'GRACE': '"RMS of Geoid height [mm]"'}
        if trend == 'trend':
            cb_dict = {'U': '"gravitational potential [m@+2@+/s@+2@+]"',
                         'N': '"Geoid height [mm]"',
                         'h': '"Elevation [mm]"',
                         'sd': '"Surface Density [kg/m@+2@+]"',
                         'ewh': '"Equivalent water height [m]"',
                         'gravity': '"gravitational acceleration [m/s@+2@+]"',
                         'ff': '"Fractional frequency [-]"',
                         'GRACE': '"Geoid height [mm]"'}
        
        if isinstance(T[0], str):
            T_frac = [datetime.datetime.strptime(t, '%Y_%m') for t in T]
        try:
            T_frac = np.array([utils.datetime2frac(t) for t in T_frac])
        except: 
            T_frac = np.array([utils.datetime2frac(t) for t in T])
        # make strings if time is given in datetime objects
        if not isinstance(T[0], str):
            if hourly:
                T = [datetime.datetime.strftime(t, format='%Y_%m_%d_%H')
                     for t in T]
            else:
                T = [datetime.datetime.strftime(t, format='%Y_%m_%d') for t in T]
        
        path = cfg.PATHS['data_path'] + esc + '/'
        f_lm = harmony.shcoeffs_from_netcdf(path + esc_dict[esc] + T[0])
        grid = f_lm.pad(720).expand()
        y = np.arange(int(grid.nlat/10), int(grid.nlat/3))
        x_east = np.arange(int(grid.nlon/6))
        x_west = np.arange(int(grid.nlon/20*19), grid.nlon)
        europe = np.zeros((len(y), len(x_west)+len(x_east)))
        EUROPA = []
        for t in T:
            f_lm = harmony.shcoeffs_from_netcdf(path + esc_dict[esc] + t)
            f_lm = harmony.sh2sh(f_lm, unitFrom, unitTo)
            if lmin or lmax:
                if lmin and not lmax:
                    f_lm = f_lm.pad(720) - f_lm.pad(lmin).pad(720)
                elif lmax and not lmin:
                    f_lm = f_lm.pad(lmax).pad(720)
                elif lmin and lmax:
                    f_lm = f_lm.pad(lmax).pad(720) - f_lm.pad(lmin).pad(720)
            grid = f_lm.pad(720).expand()
            europe[:, :len(x_west)] = grid.data[y, x_west[0]:]
            europe[:, len(x_west):] = grid.data[y, :len(x_east)]
            EUROPA.append(copy.copy(europe))
            print(t)
        EUROPA = np.array(EUROPA)  # 365er liste mit ~500x400 arrays
        
        EUROPA_rms = np.zeros((np.shape(EUROPA)[1:]))
        EUROPA_coef = np.zeros((np.shape(EUROPA)[1:]))
        for i in range(np.shape(EUROPA)[1]):
            for j in range(np.shape(EUROPA)[2]):
                if trend == 'trend':
                    t = np.arange(len(EUROPA[:, i, j]))
                    model = utils.trend(t, EUROPA[:, i, j])
                    # model = utils.annual_trend(T_frac, EUROPA[:, i, j])
                    EUROPA_rms[i, j] = model.coef_[0]# * 12
                    # EUROPA_rms[i, j] = model[1]
                elif trend == 'linear':
                    # t = np.arange(len(EUROPA[:, i, j]))
                    # model = utils.trend(t, EUROPA[:, i, j])
                    model, A = utils.annual_trend(T_frac, EUROPA[:, i, j])
                    # EUROPA_trend = (t * model.coef_[0] + model.intercept_)
                    EUROPA_trend = A[:, :2].dot(model[:2])
                    EUROPA_rms[i, j] = (utils.rms(EUROPA[:, i, j],
                                                  EUROPA_trend))
                elif trend == 'annual' or trend == 'semiannual':
                    if trend == 'annual':
                        model, A = utils.annual_trend(np.array([utils.datetime2frac(datetime.datetime.strptime(t, '%Y_%m_%d')) for t in T]), EUROPA[:, i, j],
                                                      semi=False)
                    else:
                        # model, A = utils.annual_trend(np.array([utils.datetime2frac(datetime.datetime.strptime(t, '%Y_%m_%d')) for t in T]), EUROPA[:, i, j])
                        model, A = utils.annual_trend(T_frac, EUROPA[:, i, j])
                    EUROPA_trend = A.dot(model)
                    EUROPA_rms[i, j] = (utils.rms(EUROPA[:, i, j],
                                                  EUROPA_trend))
                elif trend == 'mean':  # only experimental
                    EUROPA_rms[i, j] = np.mean(EUROPA[:, i, j])
                elif trend == 'slope':  # only experimental, deprecated
                    t = np.arange(len(EUROPA[:, i, j]))
                    model = utils.trend(t, EUROPA[:, i, j])
                    EUROPA_rms[i, j] = model.coef_[0] * len(EUROPA[:, i, j])
                elif isinstance(trend, int):
                    # filtered = utils.butter_highpass(EUROPA[:, i, j], trend)
                    filtered = (EUROPA[:, i, j]
                                - utils.ma(EUROPA[:, i, j], trend))
                    EUROPA_rms[i, j] = utils.rms(filtered)
                else:
                    EUROPA_rms[i, j] = utils.rms(EUROPA[:, i, j])
            print(i)
        
        # grid.data = np.zeros((np.shape(grid.data)))
        data = np.zeros((np.shape(grid.data)))
        data[y, x_west[0]:] = EUROPA_rms[:, :len(x_west)]
        data[y, :len(x_east)] = EUROPA_rms[:, len(x_west):]
        if unitTo in('N', 'h', 'GRACE'):
            data = data * 1e3
        grid.data = data
        
        fig = self.plot_europe_720(grid, path, trend, esc, unitTo, cb_dict,
                                   'rms', '')
        
        if save:
            savepath = path + '../../fig/'
            savename = (os.path.join(savepath, esc, esc_dict[esc] + T[0] + '-'
                                     + T[-1] + '_' + unitTo + '_RMS.pdf'))
            if trend == 'linear':
                savename = savename[:-4] + '_detrended.pdf'
            if trend == 'annual':
                savename = savename[:-4] + '_detrended_annual.pdf'
            if trend == 'semiannual':
                savename = savename[:-4] + '_detrended_semiannual.pdf'
            elif trend == 'trend':
                savename = savename[:-8] + '_trend.pdf'
            elif isinstance(trend, (int, float)):
                savename = savename[:-4] + '_filtered_'+ str(trend) + '.pdf'
            if lmin:
                savename = savename[:-4] + 'lmin' + str(lmin) + '.pdf'
            if lmax:
                savename = savename[:-4] + 'lmax' + str(lmax) + '.pdf'
            fig.savefig(savename)
        
        return fig, grid
    
    def plotRMS2(self, T, esc, unitFrom, unitTo, reset=False, save=False,
                trend=None):
        """Plots the Root Mean Square on a map.
        
        Expands the spherical harmonics from the data folder for each grid
        point and each time step. Then computes the root mean square error for
        each of the grid point time series.

        :param T: list of dates
        :type T: str or datetime.date(time)
        :param esc: earth system component
        :type esc: str
        :param unitFrom: unit of the input coefficients
        :type unitFrom: str
        :param unitTo: unit of the timeseries
        :type unitTo: str
        :param reset: shall the timeseries be calculated again
        :type reset: boolean, optional
        :param save: shall the figure be saved
        :type save: boolean
        :param trend: shall a trend be subtracted
        :type trend: str or odd int, optional
        :return fig: the figure object
        :rtype fig: pygmt.Figure
        :return grid: the plottet data grid
        :rtype grid: pyshtools.SHGrid
        
        Possible trends:
            'linear' ... just the linear trend
            'trend' ... trend is plotted instead of the RMS
            as number: width of the moving average filter, thus the cutoff
                       period length; has to be an odd integer
        
        Possible units:
            'pot' ... dimensionless Stokes coeffs (e.g. GRACE L2)
                'U' ... geopotential [m^2/s^2]
                'N' ... geoid height [m]
                'GRACE' ... geoid height [m], but with lmax=120 and filtered
            'h' ... elevation [m]
            'ff' ... fractional frequency [-]
            'mass' ... dimensionless surface loading coeffs
                'sd' ... surface density [kg/m^2]
                'ewh' ... equivalent water height [m]
            'gravity'... [m/s^2]
            
        Possible earth system components:
            'I' ... Ice
            'H' ... Hydrology
            'A' ... Atmosphere
            'GIA'.. Glacial Isostatic Adjustment
        """
        
        esc_dict = {'I': 'oggm_',
                    'H': 'clm_tws_',
                    'A': 'coeffs_'}
        cb_dict = {'U': '"RMS of gravitational potential [m@+2@+/s@+2@+]"',
                   'N': '"Geoid height variability [mm]"',
                   'h': '"RMS of Elevation [mm]"',
                   'sd': '"RMS of Surface Density [kg/m@+2@+]"',
                   'ewh': '"Mass variability in EWH [m]"',
                   'gravity': '"RMS of gravitational acceleration [m/s@+2@+]"',
                   'ff': '"RMS of Fractional frequency [-]"',
                   'GRACE': '"RMS of Geoid height [mm]"'}
        if trend == 'trend':
            cb_dict = {'U': '"gravitational potential [m@+2@+/s@+2@+]"',
                         'N': '"Geoid height [mm]"',
                         'h': '"Elevation [mm]"',
                         'sd': '"Surface Density [kg/m@+2@+]"',
                         'ewh': '"Equivalent water height [m]"',
                         'gravity': '"gravitational acceleration [m/s@+2@+]"',
                         'ff': '"Fractional frequency [-]"',
                         'GRACE': '"Geoid height [mm]"'}
        
        T_frac = np.array([utils.datetime2frac(t) for t in T])
        # make strings if time is given in datetime objects
        if not isinstance(T[0], str):
            T = [datetime.datetime.strftime(t, format='%Y_%m_%d') for t in T]
        
        path = cfg.PATHS['data_path'] + esc + '/'
        f_lm = harmony.shcoeffs_from_netcdf(path + esc_dict[esc] + T[0])
        grid = f_lm.pad(720).expand()
        y = np.arange(int(grid.nlat/10), int(grid.nlat/3))
        x_east = np.arange(int(grid.nlon/6))
        x_west = np.arange(int(grid.nlon/20*19), grid.nlon)
        europe = np.zeros((len(y), len(x_west)+len(x_east)))
        EUROPA = []
        for t in T:
            f_lm = harmony.shcoeffs_from_netcdf(path + esc_dict[esc] + t)
            f_lm = harmony.sh2sh(f_lm, unitFrom, unitTo)
            grid = f_lm.pad(720).expand()
            europe[:, :len(x_west)] = grid.data[y, x_west[0]:]
            europe[:, len(x_west):] = grid.data[y, :len(x_east)]
            EUROPA.append(copy.copy(europe))
            print(t)
        EUROPA = np.array(EUROPA)  # 365er liste mit ~500x400 arrays
        
        EUROPA_rms = np.zeros((np.shape(EUROPA)[1:]))
        EUROPA_coef = np.zeros((np.shape(EUROPA)[1:]))
        for i in range(np.shape(EUROPA)[1]):
            for j in range(np.shape(EUROPA)[2]):
                if trend == 'trend':
                    t = np.arange(len(EUROPA[:, i, j]))
                    model = utils.trend(t, EUROPA[:, i, j])
                    # model = utils.annual_trend(T_frac, EUROPA[:, i, j])
                    EUROPA_rms[i, j] = model.coef_[0]# * 12
                    # EUROPA_rms[i, j] = model[1]
                elif trend == 'linear':
                    # t = np.arange(len(EUROPA[:, i, j]))
                    # model = utils.trend(t, EUROPA[:, i, j])
                    model, A = utils.annual_trend(T_frac, EUROPA[:, i, j])
                    # EUROPA_trend = (t * model.coef_[0] + model.intercept_)
                    EUROPA_trend = A[:, :2].dot(model[:2])
                    EUROPA_rms[i, j] = (utils.rms(EUROPA[:, i, j],
                                                  EUROPA_trend))
                elif trend == 'annual' or trend == 'semiannual':
                    if trend == 'annual':
                        model, A = utils.annual_trend(T_frac, EUROPA[:, i, j],
                                                      semi=False)
                    else:
                        model, A = utils.annual_trend(T_frac, EUROPA[:, i, j])
                    EUROPA_trend = A.dot(model)
                    EUROPA_rms[i, j] = (utils.rms(EUROPA[:, i, j],
                                                  EUROPA_trend))
                elif trend == 'mean':  # only experimental
                    EUROPA_rms[i, j] = np.mean(EUROPA[:, i, j])
                elif trend == 'slope':  # only experimental, deprecated
                    t = np.arange(len(EUROPA[:, i, j]))
                    model = utils.trend(t, EUROPA[:, i, j])
                    EUROPA_rms[i, j] = model.coef_[0] * len(EUROPA[:, i, j])
                elif isinstance(trend, int):
                    # filtered = utils.butter_highpass(EUROPA[:, i, j], trend)
                    filtered = (EUROPA[:, i, j]
                                - utils.ma(EUROPA[:, i, j], trend))
                    EUROPA_rms[i, j] = utils.rms(filtered)
                else:
                    EUROPA_rms[i, j] = utils.rms(EUROPA[:, i, j])
            print(i)
        
        # grid.data = np.zeros((np.shape(grid.data)))
        data = np.zeros((np.shape(grid.data)))
        data[y, x_west[0]:] = EUROPA_rms[:, :len(x_west)]
        data[y, :len(x_east)] = EUROPA_rms[:, len(x_west):]
        if unitTo in('N', 'h', 'GRACE'):
            data = data * 1e3
        grid.data = data
        
        x = grid.lons()
        y = grid.lats()
        # find out what the datalimits are within the shown region
        data_lim = np.concatenate((grid.to_array()[200:402, -81:],
                                grid.to_array()[200:402, :242]), axis=1)
        datamax = np.max(abs(data_lim))
        datamin = np.min(abs(data_lim))
        
        da = xr.DataArray(data, coords=[y, x], dims=['lat', 'lon'])
        # save the dataarray as netcdf to work around the 360° plotting problem
        da.to_dataset(name='dataarray').to_netcdf(path + '../temp/pygmt.nc')
        fig = pygmt.Figure()
        if trend == 'trend':
            pygmt.makecpt(cmap='polar', series=[-datamax, datamax],
                          reverse=True)
        else:
            pygmt.makecpt(cmap='drywet', series=[datamin, datamax])
        fig.grdimage(path + '../temp/pygmt.nc', region=[-15, 40, 40, 65],
                     projection="M10i")  # frame: a for the standard frame, g for the grid lines
        if esc == 'H' or esc == 'I' and unitTo == 'ewh':
            fig.coast(region=[-15, 40, 40, 65], projection="M10i",
                      shorelines="1/0.1p,black",
                      borders="1/0.2p,black", water='white')
        else:
            fig.coast(region=[-15, 40, 40, 65], projection="M10i",
                      shorelines="1/0.1p,black",
                      borders="1/0.2p,black")
        nmis = [0, 5, 6, 7, 9, 10, 12, 11, 13, 15, 17, 18]
        nmi_links = []
        for i, i2 in zip(nmis[:-1], nmis[1:]):
            self.add_link(self.clocks[i], self.clocks[i2], 8)
        self.add_link(self.clocks[nmis[-1]], self.clocks[nmis[0]], 8)
        for l in self.links:
            x = [l.a.lon, l.b.lon]
            y = [l.a.lat, l.b.lat]
            if l.state == 8:
                fig.plot(x, y, pen="3p,red")
        fig.plot(self.lons()[nmis], self.lats()[nmis], style="c0.45i", color="white",
                 pen="black")
        if save:
            savepath = path + '../../fig/'
            savename = savepath + 'openlichkeitswork3.pdf'
            fig.savefig(savename)
            
        for l in self.links:
            x = [l.a.lon, l.b.lon]
            y = [l.a.lat, l.b.lat]
            if l.state == 8:
                fig.plot(x, y, pen="4p,red")
        fig.plot(self.lons()[nmis], self.lats()[nmis], style="c0.45i", color="white",
                 pen="black")
        if save:
            savename = savepath + 'openlichkeitswork4.pdf'
            fig.savefig(savename)
        
        return fig, grid
    
    def plotRMSatClocks(self, T, esc, unitFrom, unitTo, reset=False,
                        save=False, trend=None, sigma=False):
        """Plots the Root Mean Square on a map.
        
        Expands the spherical harmonics from the data folder for each grid
        point and each time step. Then computes the root mean square error for
        each of the grid point time series.

        :param T: list of dates
        :type T: str or datetime.date(time)
        :param esc: earth system component
        :type esc: str
        :param unitFrom: unit of the input coefficients
        :type unitFrom: str
        :param unitTo: unit of the timeseries
        :type unitTo: str
        :param reset: shall the timeseries be calculated again
        :type reset: boolean, optional
        :param save: shall the figure be saved
        :type save: boolean
        :param trend: shall a trend be subtracted
        :type trend: str or odd int, optional
        :return fig: the figure object
        :rtype fig: pygmt.Figure
        :return grid: the plottet data grid
        :rtype grid: pyshtools.SHGrid
        
        Possible trends:
            'linear' ... just the linear trend
            as number: width of the moving average filter, thus the cutoff
                       period length; has to be an odd integer
        
        Possible units:
            'pot' ... dimensionless Stokes coeffs (e.g. GRACE L2)
                'U' ... geopotential [m^2/s^2]
                'N' ... geoid height [m]
                'GRACE' ... geoid height [m], but with lmax=120 and filtered
            'h' ... elevation [m]
            'ff' ... fractional frequency [-]
            'mass' ... dimensionless surface loading coeffs
                'sd' ... surface density [kg/m^2]
                'ewh' ... equivalent water height [m]
            'gravity'... [m/s^2]
            
        Possible earth system components:
            'I' ... Ice
            'H' ... Hydrology
            'A' ... Atmosphere
            'GIA'.. Glacial Isostatic Adjustment
        """
        
        esc_dict = {'I': 'oggm_',
                    'H': 'clm_tws_',
                    'A': 'coeffs_'}
        cb_dict = {'U': '"RMS of gravitational potential [m@+2@+/s@+2@+]"',
                   'N': '"RMS of Geoid height [mm]"',
                   'h': '"RMS of Elevation [mm]"',
                   'sd': '"RMS of Surface Density [kg/m@+2@+]"',
                   'ewh': '"RMS of Equivalent water height [m]"',
                   'gravitational': '"RMS of gravitational acceleration [m/s@+2@+]"',
                   'ff': '"RMS of Fractional frequency [-]"',
                   'GRACE': '"RMS of Geoid height [mm]"'}
        
        T_frac = np.array([utils.datetime2frac(t) for t in T])
        # make strings if time is given in datetime objects
        if not isinstance(T[0], str):
            T = [datetime.datetime.strftime(t, format='%Y_%m_%d') for t in T]
        
        path = cfg.PATHS['data_path'] + esc + '/'
        EUROPA = []
        for t in T:
            f_lm = harmony.shcoeffs_from_netcdf(path + esc_dict[esc] + t)
            f_lm = harmony.sh2sh(f_lm, unitFrom, unitTo)
            points = np.array(
                [f_lm.expand(lat=clo.lat, lon=clo.lon) for clo in self.clocks])
            EUROPA.append(copy.copy(points))
            print(t)
        EUROPA = np.array(EUROPA)  # 365er-liste mit ~20er arrays
            
        EUROPA_rms = np.zeros((np.shape(EUROPA)[1]))
        # EUROPA_coef = np.zeros((np.shape(EUROPA)[1]))
        for i in range(np.shape(EUROPA)[1]):
            if trend == 'linear':
                model, A = utils.annual_trend(T_frac, EUROPA[:, i])
                EUROPA_trend = A[:, :2].dot(model[:2])
                EUROPA_rms[i] = (utils.rms(EUROPA[:, i], EUROPA_trend))
            elif trend == 'annual' or trend == 'semiannual':
                if trend == 'annual':
                    model, A = utils.annual_trend(T_frac, EUROPA[:, i],
                                                  semi=False)
                else:
                    model, A = utils.annual_trend(T_frac, EUROPA[:, i])
                EUROPA_trend = A.dot(model)
                EUROPA_rms[i] = (utils.rms(EUROPA[:, i], EUROPA_trend))
            elif isinstance(trend, int):
                filtered = EUROPA[:, i] - utils.ma(EUROPA[:, i], trend)
                EUROPA_rms[i] = utils.rms[filtered]
            else:
                EUROPA_rms[i] = utils.rms(EUROPA[:, i])
        
        if unitTo in('N', 'h', 'GRACE'):
            EUROPA_rms = EUROPA_rms * 1e3
        datamax = np.max(abs(EUROPA_rms))
        
        data = {'data': EUROPA_rms, 'lat': self.lats(), 'lon': self.lons()}
        df = pd.DataFrame(data)
        
        fig = pygmt.Figure()
        pygmt.makecpt(cmap='drywet', series=[0, datamax])
        fig.coast(region=[-10, 30, 40, 65], projection="S10/90/6i",
                  frame="a", shorelines="1/0.1p,black",
                  borders="1/0.1p,black")
        fig.plot(x=df.lon, y=df.lat, style='c0.1i', color=df.data/datamax,
                 cmap='drywet')
        # fig.colorbar(frame=['paf+lewh', 'y+l:m'])  # @+x@+ for ^x
        # fig.colorbar(position='JMR', frame='paf+l' + cb_dict[unitTo])  # @+x@+ for ^x
        fig.colorbar(frame='paf+l' + cb_dict[unitTo])  # @+x@+ for ^x

        if save:
            savepath = path + '../../fig/'
            savename = (os.path.join(savepath, esc, esc_dict[esc] + T[0] + '-'
                                     + T[-1] + '_' + unitTo
                                     + '_clockwise_RMS.pdf'))
            if trend == 'linear':
                savename = savename[:-4] + '_detrended.pdf'
            fig.savefig(savename)
        
        return fig, EUROPA_rms
            
    def _plotTimeseries(self, kind, unit):
        """Plots the timeseries of all clocks.
        
        DEPRECATED!
        
        Plots all clocks in one plot, takes a certain (or all summed up) signal
        and unit to plot.
        
        :param kind:
        :type kind: str
        :param unit:
        :type unit: str
        
        Possible units:
            'U' ... geopotential [m^2/s^2]
            'N' ... geoid height [m]
            'h' .... elevation [m]
            'sd' ... surface density [kg/m^2]
            'ewh' ... equivalent water height [m]
            'gravity'... [m/s^2]
            'ff' ... fractional frequency [-]
            
        Possible kinds:
            'I' ... Ice
            'H' ... Hydrology
            'A' ... Atmosphere
            'O' ... Ocean
            'GIA'.. Glacial Isostatic Adjustment
            'S' ... Solid Earth
        """
        #TODO: plot mean values?
        plt.rcParams.update({'font.size': 13})  # set before making the figure!
        fig, ax = plt.subplots(figsize=(10, 10))
        for clo in self.clocks:
            try:
                effect =  getattr(clo, kind)
                data = effect[unit]
                t = effect['t']
                ax.plot(t, data, label=clo.location)
            except:
                print('Clock in ' + clo.location + ' does not have information'
                      + ' from ' + kind + ' about ' + unit)
        ax.set_xlabel('time [y]')
        unit_dict = {'U': 'gravitational potential [m$^2$/s$^2$]',
                     'N': 'Geoid height [m]',
                     'h': 'Elevation [m]',
                     'sd': 'Surface Density [kg/m$^2$]',
                     'ewh': 'Equivalent water height [m]',
                     'gravitational': 'gravitational acceleration [m/s$^2$]',
                     'ff': 'Fractional frequency [-]'}
        ax.set_ylabel(unit_dict[unit])
        ax.grid()
        ax.legend()
        plt.tight_layout()
        
    def plotTimeseries(self, T, esc, unitFrom, unitTo, t_ref=False,
                       reset=False, loc=False, loc_ref=False, lmin=False,
                       lmax=False):
        """Plots time series at each clock location.
        
        Uses sh2timeseries() for all clocks and plots the resulting timeseries.
        
        :param T: list of dates
        :type T: str or datetime.date(time)
        :param esc: earth system component
        :type esc: str
        :param unitFrom: unit of the input coefficients
        :type unitFrom: str
        :param unitTo: unit of the timeseries
        :type unitTo: str 
        :param t_ref: reference time for the series
        :type t_ref: str or datetime.date(time)
        :param reset: shall the timeseries be calculated again
        :type reset: boolean, optional
        
        Possible units:
            'pot' ... dimensionless Stokes coeffs (e.g. GRACE L2)
                'U' ... geopotential [m^2/s^2]
                'N' ... geoid height [m]
            'h' ... elevation [m]
            'ff' ... fractional frequency [-]
            'mass' ... dimensionless surface loading coeffs
                'sd' ... surface density [kg/m^2]
                'ewh' ... equivalent water height [m]
            'gravity'... [m/s^2]
            
        Possible earth system components:
            'I' ... Ice
            'H' ... Hydrology
            'A' ... Atmosphere
            'GIA'.. Glacial Isostatic Adjustment
        """
        
        unit_dict = {'U': 'gravitational potential [m$^2$/s$^2$]',
                    'N': 'Geoid height [mm]',
                    'GRACE': 'Geoid height [mm]',
                    'h': 'Elevation [mm]',
                    'sd': 'Surface Density [kg/m$^2$]',
                    'ewh': 'Equivalent water height [m]',
                    'gravity': 'gravitational acceleration [m/s$^2$]',
                    'ff': 'Fractional frequency [-]'}
        plt.rcParams.update({'font.size': 13})  # set before making the figure!        
        fig, ax = plt.subplots()
        # next(ax._get_lines.prop_cycler)
        
        T_str = T.copy()
        if loc_ref:
            next(ax._get_lines.prop_cycler)
            clo = self.search_clock('location', loc_ref)[0]
            T_ref, data_ref = clo.sh2timeseries(
                T, esc, unitFrom, unitTo, t_ref=t_ref, reset=reset, lmin=lmin,
                lmax=lmax)
            data_ref = np.array(data_ref)
            if unitTo in('N', 'h', 'GRACE'):
                data_ref = data_ref * 1e3
                
        for number, clo in enumerate(self.clocks):
            if (loc and clo.location in loc) or loc==False:
                T, data = clo.sh2timeseries(
                    T_str, esc, unitFrom, unitTo, t_ref=t_ref, reset=reset,
                    lmin=lmin, lmax=lmax)
                data = np.array(data)
                if unitTo in('N', 'h', 'GRACE'):
                    data = data * 1e3
                # if number > 9 and number < 20:
                #     plt.plot(T, data, linestyle='--', label=clo.location)
                # elif number > 19 and number < 30:
                #     plt.plot(T, data, ':', label=clo.location)
                # elif number > 29 and number < 40:
                #     plt.plot(T, data, '-.', label=clo.location)
                if loc_ref:
                    data = data - data_ref
                if number == 0:
                    plt.plot(T, data)
                else:
                    plt.plot(T, data, label=clo.location)
                if loc_ref:
                    np.save(cfg.PATHS['data_path'] + 'ts_results/' + esc + '_' +
                            unitTo + '_' + clo.location + '-' + loc_ref, data)
                else:
                    np.save(cfg.PATHS['data_path'] + 'ts_results/' + esc + '_' +
                            unitTo + '_' + clo.location, data)
            else:
                number -= 1
        # clo = self.search_clock('location', 'Braunschweig')[0]
        # T, data = clo.sh2timeseries(T_str, esc, unitFrom, unitTo,
        #                             t_ref=t_ref)
        # data = np.array(data)
        # if unitTo in('N', 'h', 'GRACE'):
        #     data = data * 1e3
        # plt.plot(T, data, label=clo.location, linewidth=2.5, color='tab:blue')
        
        # plt.ylim([-0.42, 0.34])
        plt.ylabel(unit_dict[unitTo])
        plt.xticks(rotation=90)
        # plt.title(esc)
        plt.grid()
        plt.legend(loc=1)#bbox_to_anchor=(1., 1.))
        plt.tight_layout()
        
        
        path = (cfg.PATHS['fig_path'] + 'timeseries/partofnetwork.pdf')
        plt.savefig(path)
        
        return fig
    
    def plotTimeFrequencies(self, T, esc, unitFrom, unitTo, delta_t,
                            fmax=False,  t_ref=False, loc=False,
                            loc_ref=False, save=False, lmin=False, lmax=False):
        """Plots the spectral domain of the time series at clock location.
        
        Uses sh2timeseries() for all clocks and plots the resulting timeseries.
        
        :param T: list of dates
        :type T: str or datetime.date(time)
        :param esc: earth system component
        :type esc: str
        :param unitFrom: unit of the input coefficients
        :type unitFrom: str
        :param unitTo: unit of the timeseries
        :type unitTo: str 
        :param t_ref: reference time for the series
        :type t_ref: str or datetime.date(time)
        :param reset: shall the timeseries be calculated again
        :type reset: boolean, optional
        
        Possible units:
            'pot' ... dimensionless Stokes coeffs (e.g. GRACE L2)
                'U' ... geopotential [m^2/s^2]
                'N' ... geoid height [m]
            'h' ... elevation [m]
            'ff' ... fractional frequency [-]
            'mass' ... dimensionless surface loading coeffs
                'sd' ... surface density [kg/m^2]
                'ewh' ... equivalent water height [m]
            'gravity'... [m/s^2]
            
        Possible earth system components:
            'I' ... Ice
            'H' ... Hydrology
            'A' ... Atmosphere
            'GIA'.. Glacial Isostatic Adjustment
        """
        
        unit_dict = {'U': 'gravitational potential [m$^2$/s$^2$]',
                    'N': 'Geoid height [mm]',
                    'h': 'Elevation [mm]',
                    'sd': 'Surface Density [kg/m$^2$]',
                    'ewh': 'Equivalent water height [m]',
                    'gravity': 'gravitational acceleration [m/s$^2$]',
                    'ff': 'Fractional frequency [-]'}
        np.random.seed(7)
        plt.rcParams.update({'font.size': 13})  # set before making the figure!        
        fig, ax = plt.subplots()
        
        T_str = T.copy()
        if loc_ref:
            next(ax._get_lines.prop_cycler)
            clo = self.search_clock('location', loc_ref)[0]
            T_ref, data_ref = clo.sh2timeseries(T, esc, unitFrom, unitTo,
                                                t_ref=t_ref, lmin=lmin,
                                                lmax=lmax)
            data_ref = np.array(data_ref)
            if unitTo in('N', 'h', 'GRACE'):
                data_ref = data_ref * 1e3
        
        for number, clo in enumerate(self.clocks):
            if (loc and clo.location in loc) or loc==False:
                T, data = clo.sh2timeseries(T_str, esc, unitFrom, unitTo,
                                            t_ref=t_ref, lmin=lmin, lmax=lmax)
                data = np.array(data)
                if unitTo in('N', 'h', 'GRACE'):
                    data = data * 1e3
                if loc_ref:
                    data = data - data_ref
                f, freq = harmony.time2freq(delta_t, data)
                if fmax:
                    f, freq = (f[:fmax], freq[:fmax])
                if esc == 'I':
                    plt.plot(f[3:]*86400*365, freq[3:], '.-', label=clo.location)
                else:
                    plt.plot(f[1:]*86400*365, freq[1:], '.-', label=clo.location)
                
                if loc_ref:
                    np.save(cfg.PATHS['data_path'] + 'ts_results/' + esc + 
                            '_freq_' + clo.location + '-' + loc_ref, freq[1:])
                else:
                    np.save(cfg.PATHS['data_path'] + 'ts_results/' + esc +
                            '_freq_' + clo.location, freq[1:])
            else:
                number -= 1
                
        noise = np.load('/home/schroeder/CLONETS/data/noise.npy')
        noise = noise * np.sqrt(2)
        # plt.plot(f[1:]*86400*365, noise[0, :], '--', color='black',
        #          label='noise levels')
        # plt.plot(f[1:]*86400*365, noise[1, :], '--', color='black')
        # plt.plot(f[1:]*86400*365, noise[2, :], '--', color='black')
        plt.plot(np.arange(1, 183), noise[0, :], '--', color='black',
                 label='noise levels')
        plt.plot(np.arange(1, 183), noise[1, :], '--', color='black')
        plt.plot(np.arange(1, 183), noise[2, :], '--', color='black')
        
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Frequencies [1/yr]')
        plt.ylabel(unit_dict[unitTo])
        plt.grid()
        # plt.legend()
        plt.legend(loc='lower right')
        plt.tight_layout()
        
        if save:
            path = (cfg.PATHS['fig_path'] + 'timeseries/partofnetwork_'
                    + unitTo + '_spectral.pdf')
            plt.savefig(path)
        
        return fig

    def plotErrorTimeseries(self, esc, unitFrom, unitTo, loc, monthly=False,
                            save=False):
        
        unit_dict = {'U': 'gravitational potential [m$^2$/s$^2$]',
                    'N': 'Geoid height [mm]',
                    'h': 'Elevation [mm]',
                    'sd': 'Surface Density [kg/m$^2$]',
                    'ewh': 'Equivalent water height [m]',
                    'gravity': 'gravitational acceleration [m/s$^2$]',
                    'ff': 'Fractional frequency [-]'}
        t0 = datetime.date(2007, 1, 1)
        T = [t0 + datetime.timedelta(d) for d in range(0, 365)]
        t_ref = '2007'
        # clocks        
        clo0 = self.search_clock('location', 'Braunschweig')[0]
        clo1 = self.search_clock('location', loc)[0]
        
        t, data0 = clo0.sh2timeseries(T, esc, unitFrom, unitTo, t_ref=t_ref)
        t, data1 = clo1.sh2timeseries(T, esc, unitFrom, unitTo, t_ref=t_ref)
        data = (np.array(data1) - np.array(data0)) * 1e3
        
        # Sigma clocks
        sigma1 = 1e-18
        sigma2 = 1e-19
        sigma3 = 1e-20
        sigma1 = (sigma1 * scipy.constants.c**2 / sh.constant.gm_wgs84.value *
                 sh.constant.r3_wgs84.value**2 * 1e3 * np.sqrt(2))  # [mm]
        sigma2 = (sigma2 * scipy.constants.c**2 / sh.constant.gm_wgs84.value *
                 sh.constant.r3_wgs84.value**2 * 1e3 * np.sqrt(2))  # [mm]
        sigma3 = (sigma3 * scipy.constants.c**2 / sh.constant.gm_wgs84.value *
                 sh.constant.r3_wgs84.value**2 * 1e3 * np.sqrt(2))  # [mm]
        sigma1 = sigma1 + 2 * np.sqrt(2)
        sigma2 = sigma2 + 1 * np.sqrt(2)
        sigma3 = sigma3 + 0.4 * np.sqrt(2)
        
        # GRACE
        T = [t_ref + '_' + f'{d:02}' for d in range(1, 13)]
        files = ['/home/schroeder/CLONETS/data/ITSG-Grace2018_n120_2007-'
                  + f'{d:02}' + 'sigma.nc' for d in range(1, 13)]
        sigma_grace = [harmony.sh2sh(harmony.shcoeffs_from_netcdf(f), 'pot', 'pot')
                       for f in files]  # 12 sh sets with the errors ...
        # Error Propagation # TODO: this only works for 'pot' to 'N', so dont use this function with other units...
        sigma_xx1 = np.array([harmony.Sigma_xx_2points_from_formal(
            x, clo0.lon, clo0.lat, clo1.lon, clo1.lat, lmax=100, gaussian=350)
            for x in sigma_grace])  # [m]
        sigma_xx2 = np.array([harmony.Sigma_xx_2points_from_formal(
            x, clo0.lon, clo0.lat, clo1.lon, clo1.lat, lmax=80)   
            for x in sigma_grace])  # [m]
        sigma_xx3 = np.array([harmony.Sigma_xx_2points_from_formal(
            x, clo0.lon, clo0.lat, clo1.lon, clo1.lat, lmax=60)   
            for x in sigma_grace])  # [m]
        s_grace1 = 1e3 * sigma_xx1  # [mm]
        s_grace2 = 1e3 * sigma_xx2  # [mm]
        s_grace3 = 1e3 * sigma_xx3  # [mm]
        
        print(clo1.location, np.mean(s_grace1))
        print(clo1.location, np.mean(s_grace2))
        print(clo1.location, np.mean(s_grace3))
        # np.array(
        #     [np.sqrt(s.pad(120).expand(lat=clo0.lat, lon=clo0.lon)**2
        #       + s.pad(120).expand(lat=clo1.lat, lon=clo1.lon)**2)
        #       for s in sigma_grace]) * 1.1 / np.sqrt(2)  # ... expanded to the 2 clock locations
        
        # tm, grace0, grace0_upper, grace0_lower = clo0.sh2timeseries_error(
        #     sigma_grace[0], T, esc, unitFrom, 'GRACE',)  # simulated GRACE time series
        # tm, grace1, grace1_upper, grace1_lower = clo1.sh2timeseries_error(
        #     sigma_grace[0], T, esc, unitFrom, 'GRACE', t_ref=t_ref)  # simulated GRACE time series
        tm, grace0 = clo0.sh2timeseries(T, esc, unitFrom, unitTo, t_ref=t_ref, lmax=80)
        tm, grace1 = clo1.sh2timeseries(T, esc, unitFrom, unitTo, t_ref=t_ref, lmax=80)
        grace = 1e3 * (np.array(grace1) - np.array(grace0))  # difference between the two
        tm = [x + datetime.timedelta(14) for x in tm]
        
        # averaging to monthly clock data
        if monthly:
            datam = utils.daily2monthly(t, data)
            sigma1 = sigma1 - 2 * np.sqrt(2) + 2 * np.sqrt(2) / np.sqrt(4)
            sigma2 = sigma2 - 1 * np.sqrt(2) + 1 * np.sqrt(2) / np.sqrt(4)
            sigma3 = sigma3 - 0.4 * np.sqrt(2) + 0.4 * np.sqrt(2) / np.sqrt(4)
            t = tm
            data = datam
                
        plt.rcParams.update({'font.size': 13})  # set before making the figure!        
        fig, ax = plt.subplots()
        
        plt.plot(tm, grace, label='GRACE', color='tab:orange')
        # plt.plot(t, data+sigma1, '--', color='tab:blue', label='network error', linewidth=0.7)
        # plt.plot(t, data+sigma2, '--', color='tab:blue', label='network error', linewidth=0.7)
        # plt.plot(t, data+sigma3, '--', color='tab:blue', linewidth=0.7)
        # plt.plot(t, data-sigma1, '--', color='tab:blue', linewidth=0.7)
        # plt.plot(t, data-sigma2, '--', color='tab:blue', linewidth=0.7)
        # plt.plot(t, data-sigma3, '--', color='tab:blue', linewidth=0.7)
        plt.plot(t, data, label='Network', color='tab:blue')
        plt.fill_between(t, data+sigma1, data-sigma1, alpha=0.1, color='tab:blue')
        plt.fill_between(t, data+sigma2, data-sigma2, alpha=0.2, color='tab:blue')
        plt.fill_between(t, data+sigma3, data-sigma3, alpha=0.3, color='tab:blue',
                          label='Network error')
        plt.plot(tm, grace+s_grace1,  '--', color='tab:orange', label='GRACE error', linewidth=0.7)
        plt.plot(tm, grace+s_grace2,  '--', color='tab:orange', linewidth=0.7)
        plt.plot(tm, grace+s_grace3,  '--', color='tab:orange', linewidth=0.7)
        plt.plot(tm, grace-s_grace1,  '--', color='tab:orange', linewidth=0.7)
        plt.plot(tm, grace-s_grace2,  '--', color='tab:orange', linewidth=0.7)
        plt.plot(tm, grace-s_grace3,  '--', color='tab:orange', linewidth=0.7)
        # plt.fill_between(tm, grace+s_grace1, grace-s_grace1, alpha=0.1, color='tab:orange')
        # plt.fill_between(tm, grace+s_grace2, grace-s_grace2, alpha=0.2, color='tab:orange')
        # plt.fill_between(tm, grace+s_grace3, grace-s_grace3, alpha=0.3, color='tab:orange',
        #                   label='GRACE error')
       
        plt.ylim([-10, 10])
        plt.xlim([t[0] + datetime.timedelta(-35), t[-1] + datetime.timedelta(35)])
        plt.ylabel(unit_dict[unitTo])
        plt.xticks(rotation=90)
        plt.grid()
        plt.legend()
        plt.tight_layout()
        
        path = cfg.PATHS['data_path'] + '/'

        if save:
            savepath = path + '../fig/'
            if monthly:
                savename = (os.path.join(savepath, 'timeseries', esc + '_wsigma_'
                                         + loc + '_monthly.pdf'))
            else:    
                savename = (os.path.join(savepath, 'timeseries', esc + '_wsigma_'
                                         + loc + '.pdf'))
            plt.savefig(savename)
        
        return (fig, t, tm, data, grace, np.array([sigma1, sigma2, sigma3]),
                np.array([s_grace1, s_grace2, s_grace3]))
        
    def plot_world_720(self, grid, path, trend, esc, unitTo, cb_dict,
                        inp_func, t):
        
        x = grid.lons()
        y = grid.lats()
        datamax = np.max(abs(grid.data))
        print(datamax)
        datamax = 4  # WATCH OUT
        
        da = xr.DataArray(grid.data, coords=[y, x], dims=['lat', 'lon'])
        # save the dataarray as netcdf to work around the 360° plotting problem
        da.to_dataset(name='dataarray').to_netcdf(path + '../temp/pygmt.nc')
        with pygmt.clib.Session() as session:
            session.call_module('gmtset', 'FONT 15p')
        fig = pygmt.Figure()
        if trend == 'trend':
            pygmt.makecpt(cmap='polar', series=[-datamax, datamax],
                          reverse=True)
        elif inp_func == 'esc':
            pygmt.makecpt(cmap='roma', series=[-datamax, datamax])
        else:
            pygmt.makecpt(cmap='viridis', series=[0, datamax], reverse=True)
        fig.grdimage(path + '../temp/pygmt.nc', projection="R15c", region="d", 
                     frame=["ag", "WSne"])  # frame: a for the standard frame, g for the grid lines
        if (esc == 'H' or esc == 'I') and unitTo == 'ewh':
            fig.coast(shorelines="1/0.1p,black", projection="R15c", region="d", 
                      borders="1/0.1p,black", water='white')
        else:
            fig.coast(shorelines="1/0.1p,black", projection="R15c", region="d", 
                      borders="1/0.1p,black")
        fig.plot(self.lons(), self.lats(), style="c0.07i", color="white",
                 pen="black")
        with pygmt.clib.Session() as session:
            session.call_module('gmtset', 'FONT 18p')
        fig.colorbar(position='JMR+w10c/0.6c', frame='paf+l' + cb_dict[unitTo])  # @+x@+ for ^x
        
        return fig

        
    def plot_europe_720(self, grid, path, trend, esc, unitTo, cb_dict,
                        inp_func, t):
        
        x = grid.lons()
        y = grid.lats()
        # find out what the datalimits are within the shown region
        data_lim = np.concatenate((grid.to_array()[200:402, -81:],
                                grid.to_array()[200:402, :242]), axis=1)
        datamax = np.max(abs(data_lim))
        print(datamax)
        datamax = 4  # WATCH OUT
        
        da = xr.DataArray(grid.data, coords=[y, x], dims=['lat', 'lon'])
        # save the dataarray as netcdf to work around the 360° plotting problem
        da.to_dataset(name='dataarray').to_netcdf(path + '../temp/pygmt.nc')
        with pygmt.clib.Session() as session:
            session.call_module('gmtset', 'FONT 20p')
        fig = pygmt.Figure()
        if trend == 'trend':
            pygmt.makecpt(cmap='polar', series=[-datamax, datamax],
                          reverse=True)
        elif inp_func == 'esc':
            pygmt.makecpt(cmap='roma', series=[-datamax, datamax])
        else:
            pygmt.makecpt(cmap='viridis', series=[0, datamax], reverse=True)
        # fig.grdimage(path + '../temp/pygmt.nc', region=[-10, 30, 40, 65],
        #              projection="S10/90/6i", frame=["ag", "WSne"])  # frame: a for the standard frame, g for the grid lines
        fig.grdimage(path + '../temp/pygmt.nc', region=[-30, 30, 20, 75],
                     projection="S0/90/6i", frame=["ag", "WSne"])  # frame: a for the standard frame, g for the grid lines
        if (esc == 'H' or esc == 'I') and unitTo == 'ewh':
            fig.coast(region=[-10, 30, 40, 65], projection="S10/90/6i",
                      shorelines="1/0.1p,black",
                      borders="1/0.1p,black", water='white')
        else:
            fig.coast(region=[-30, 30, 20, 75], projection="S0/90/6i",
                      shorelines="1/0.1p,black",
                      borders="1/0.1p,black")
        fig.plot(self.lons(), self.lats(), style="c0.07i", color="white",
                 pen="black")
        with pygmt.clib.Session() as session:
            session.call_module('gmtset', 'FONT 24p')
        fig.colorbar(position='JMR+w13c/0.5c', frame='paf+l' + cb_dict[unitTo])  # @+x@+ for ^x
        
        return fig
