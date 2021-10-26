#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 14:40:03 2020

@author: schroeder

Module that contains some utility functions.
"""

# Imports ---------------------------------------------------------------------
import pyshtools as sh
import numpy as np
import pandas as pd
from numpy import sqrt, mean, square
import xarray as xr
import json
import geojson
import time
import scipy
from clones import cfg
from sklearn.linear_model import LinearRegression
from scipy import signal
from datetime import datetime as dt
from datetime import timedelta
import time

# -----------------------------------------------------------------------------

def trend(t, data):
    """Fits a linear trend.
    
    Parameters
    ----------
    t: np.array of floats
        time stamps
    data: np.array of floats
        time series
        
    Returns
    -------
    model: sklearn.linearLinearRegression
        linear regression model with slope 'coef_' and y-intercept 'intercept_'
    """
    
    try:
        model = LinearRegression().fit(t, data)
    except:
        model = LinearRegression().fit(t.reshape((-1,1)), data)
    return model

def annual_trend(t, data, semi=True):
    """Fits a linear trend plus a (semi-) annual signal.
    
    Parameters
    ----------
    t: np.array of floats
        time vector as year fraction
    data: np.array of floats
        time series
    semi: bool, optional
        set True if semiannual signal is to be estimated as well
    
    Returns
    -------
    x: np.array of floats [4x1] (semiannual: [6x1])
        [y-intercept, slope (per yr), cos and sin parameters]
    transpose(At): np.array of floats
        design matrix
    """
    
    if semi:
        At = [np.ones(len(data)), t, np.cos(2*np.pi*t), np.sin(2*np.pi*t),
              np.cos(4*np.pi*t), np.sin(4*np.pi*t)]
    else:
        At = [np.ones(len(data)), t, np.cos(2*np.pi*t), np.sin(2*np.pi*t)]
    At = np.array(At)  # A transposed
    n = At.dot(data)
    Ni = At.dot(np.transpose(At))  # inverse of N
    x = np.linalg.solve(Ni, n)  # [y-intercept, slope (per yr), cos and sin parameters]
    
    return x, np.transpose(At)    

def C_from_data(Y):
    """Computes Covariance Matrix from data matrix.
    
    Parameters
    ----------
    Y: np.array of floats
        Data matrix with [location x time]
    
    Returns
    -------
    C: np.array of floats
        Covariance matrix of the data
    """
    
    N_P = np.shape(Y)[1]  # number of locations
    C = Y.T @ Y / N_P
    return C

def rms(x, y=None):
    """Computes the root mean square.
    
    Has the option of computing the root mean square error/difference between
    two arrays. Be aware: If you have two time series, use x and y, do not just
    parse x-y! If you do the latter, it will ignore the constant offset between
    x and y and just compute the std of the difference time series.
    
    Parameters
    ----------
    x: np.array of floats
        time series
    y: np.array of floats, optional
        another time series
    
    Returns
    -------
    z: float
        The root mean square        
    """
    
    if y is not None:
        z = sqrt(mean(square(x-y)))
    else:
        z = sqrt(mean(square(x-mean(x))))
    return z

def butter_highpass(x, freq):
    """A highpass filter using scipy.signal's butter filter.    

    Only high frequencies pass :P
    
    Parameters
    ----------
    x: np.array of floats
        signal time series
    freq: float
        cutoff frequency
    
    Returns
    -------
    x_filtered: np.array of floats
        filtered signal
    """
    
    sos = signal.butter(10, freq, 'highpass', fs=len(x), output='sos')
    x_filtered = signal.sosfilt(sos, x)
    return x_filtered

def ma(x, width):
    """Moving Average filter.
    
    Parameters
    ----------
    x: np.array of floats
        signal time series
    width: odd int
        width of the moving average
        
    Returns
    -------
    x_filtered: np.array of floats
        filtered signal
    """
    
    x_filtered = np.zeros(len(x))
    
    half_width = int((width-1) / 2)
    f = np.ones(width) / width
    start = np.ones(half_width) * np.mean(x[:half_width])
    end = np.ones(half_width) * np.mean(x[-half_width:])
    x = np.concatenate((start, x, end))

    for i in range(len(x_filtered)):
        x_filtered[i] = np.sum(x[i:i+width] * f)
        
    return x_filtered

def daily2monthly(t, data):
    """Averages daily to monthly data.
    
    Parameters
    ----------
    t: list of datetime.date
        time stamps of input data
    data: np.array of floats
        data time series
    
    Returns
    -------
    datam: np.array of floats
        monthly data time series
    """
# TODO: Maybe add tm as return
    day = 0
    dayinmonth = 1
    datam = np.zeros(12)
    for month in range(1, 13):
        try:
            while t[day].month == month:
                datam[month-1] += data[day]
                day += 1
                dayinmonth += 1
        except:
            pass
        datam[month-1] = datam[month-1] / (dayinmonth-1)
        month += 1
        dayinmonth = 1
            
    return datam

def datetime2frac(date):
    """Convert a datetime.datetime object into a floating year.
    
    Accurate to a few microseconds.
    
    Parameters
    ----------
    date: datetime.datetime
        input time
    
    Returns
    -------
    float
        fractional year
    """
    
    def sinceEpoch(date): # returns seconds since epoch
        return time.mktime(date.timetuple())# + date.microsecond / 1E6
    s = sinceEpoch
    
    year = date.year
    startOfThisYear = dt(year=year, month=1, day=1)  # in datetime.datetime
    startOfNextYear = dt(year=year+1, month=1, day=1)  # in datetime.datetime
    
    secondsElapsed = round(s(date) - s(startOfThisYear), 6)  # [s]
    yearDuration = s(startOfNextYear) - s(startOfThisYear)  # [s]
    fraction = secondsElapsed/yearDuration  # [y]

    return date.year + fraction       

def frac2datetime(fyear):
    """Convert a floating year into a datetime.datetime object.
    
    Accurate to a few microseconds.
    
    Parameters
    ----------
    fyear: float
        fractional year
    
    Returns
    -------
    datetime.datetime
        converted time
    """
    
    def sinceEpoch(date): # returns seconds since epoch
        return time.mktime(date.timetuple())# + date.microsecond / 1E6
    s = sinceEpoch
    
    year = int(fyear)
    startOfThisYear = dt(year=year, month=1, day=1)  # in datetime.datetime
    startOfNextYear = dt(year=year+1, month=1, day=1)  # in datetime.datetime
    
    fraction = fyear - year  # in decimals years
    yearDuration = s(startOfNextYear) - s(startOfThisYear)  # [s]
    secondsElapsed = round(yearDuration * fraction, 6)  # [s]
    microseconds = int((secondsElapsed - np.floor(secondsElapsed)) *
                                1e6)  # [µs]
    
    return (dt(year=year, month=1, day=1, hour=0, minute=0,
              second=0, microsecond=microseconds) +
            timedelta(0, int(secondsElapsed)))  # in datetime.datetime

def load_ngl(filename):
    """Loads in the names and locations of NGL stations.
    
    Parameters
    ----------
    filename: str
        path + name of the NGL station location file
        
    Returns
    -------
    rows: list of str and floats
        [name, lat, lon]
    """
    
    with open(filename) as f:
        print(f.readline())
        rows = []
        for line in f:
            row = line.split()
            row[1:3] = [float(i) for i in row[1:3]]
            rows.append(row[0:3])
            
    return rows

def load_euref(filename):
    """Loads in the names and locations of EUREF stations.
    
    Parameters
    ----------
    filename: str
        path + name of the EUREF station location file
        
    Returns
    -------
    stations: list of str and floats
        [name, lat, lon]
    """
    
    df = pd.read_csv(filename)
    names = list(df['Name'])
    lats = list(df['Latitude'])
    lons = list(df['Longitude'])
    stations = list([[names[i], lats[i], lons[i]] for i in range(len(names))])
    
    return stations

def load_psmsl(filename):
    """Loads in the names and locations of PSMSL tide gauges.
    
    Parameters
    ----------
    filename: str
        path + name of the PSMSL station location file
        
    Returns
    -------
    stations: list of str and floats
        [name, lat, lon]
    """
    
    with open(filename) as f:
        gj = geojson.load(f)
    features = gj['features']
    names = [f.properties['name'] for f in features]
    lons = [f.geometry.coordinates[0] for f in features]
    lats = [f.geometry.coordinates[1] for f in features]
    stations = list([[names[i], lats[i], lons[i]] for i in range(len(names))])
    
    return stations
    