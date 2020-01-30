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
from numpy import sqrt, mean, square
import xarray as xr
import json
import time
import scipy
from clones import cfg
from sklearn.linear_model import LinearRegression

# -----------------------------------------------------------------------------

def trend(t, data):
    """Fits a linear trend."""
    try:
        model = LinearRegression().fit(t, data)
    except:
        model = LinearRegression().fit(t.reshape((-1,1)), data)
    return model
    

def rms(x, y=None):
    """Computes the root mean square.
    
    Has the option of computing the root mean square error/difference between
    two arrays. Be aware: If you have two time series, use x and y, do not just
    parse x-y!
    
    :type x: array of floats
    :param x: First array
    :type y: array of floats, optional
    :param y: Second array
    :rtype: float
    :rparam: The root mean square
    """
    
    if y is not None:
        z = sqrt(mean(square(x-y)))
    else:
        z = sqrt(mean(square(x-mean(x))))
    return z
