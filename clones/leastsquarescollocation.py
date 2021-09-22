#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 18:36:21 2019

@author: schroeder

- LSC
"""

import numpy as np
import scipy
import geopy
import geopy.distance
import matplotlib.pyplot as plt
import h5py
import astropy
import geopandas as gpd
import time

def sphericalDistance(Lat1, Lon1, Lat2, Lon2, spherical=False):
    """."""
    
    L1 = np.transpose(np.array([Lat1, Lon1]))
    L2 = np.transpose(np.array([Lat2, Lon2]))
    D = np.zeros((len(L1), len(L2)))
    # if the 2 input grids are the same, a triangular matrix is enough
    if np.all(Lat1 == Lat2) and np.all(Lon1 == Lon2):
        for i in range(len(L1)):
            for j in range(i+1, len(L2)):
                if spherical:
                    D[i, j] = geopy.distance.great_circle(tuple(L1[i]), tuple(L2[j])).km
                else:
                    D[i, j] = geopy.distance.distance(tuple(L1[i]), tuple(L2[j])).km
        D = D + np.transpose(D)
    else:
        for i, l1 in enumerate(L1):
            for j, l2 in enumerate(L2):
                if spherical:
                    D[i, j] = geopy.distance.great_circle(tuple(l1), tuple(l2)).km
                else:
                    D[i, j] = geopy.distance.distance(tuple(l1), tuple(l2)).km
    return D

def empiricalCovarianceFunction(Lat, Lon, data, D, classDistance):
    """Compute an empirical covariance function."""
    
    # intervals [km]
    interval = np.arange(0, np.max(D), classDistance)
    interval[-1] = np.max(D)
    # data squared matrix
    data2 = np.outer(data, data)
    # some intializations
    classDistanceMean = np.zeros(len(interval))  # mean distance for each class
    gamma = np.zeros(len(interval))  # variance/ covariance per class
    gamma[0] = np.mean(np.diag(data2))
    hist = np.zeros(len(interval))  # number of realizations per class
    hist[0] = len(data)
    # loop over the distance classes
    for i in range(1, len(interval)):
        index = np.logical_and(D > interval[i-1], D <= interval[i])  # index matrix
        hist[i] = np.sum(index)
        gamma[i] = np.mean(data2[index])
        classDistanceMean[i] = np.mean(D[index])
    delta = gamma / gamma[0]  # correlation function
    
    return gamma, delta, classDistanceMean, D

def covMatFromExponentialCovFct(A, C0, alpha):
    """Computes covariance matrix of A with parameters C0 and alpha."""
    
    return C0 * (A / alpha + 1) * np.exp(-A / alpha)

def interpLSC(Lat, Lon, data, Lat2, Lon2, D_grid, alpha, gamma=None):
    """Interpolate data to a grid."""
    
    D_between = sphericalDistance(Lat, Lon, Lat2, Lon2)
    D = sphericalDistance(Lat, Lon, Lat, Lon)
    Cov_ll = covMatFromExponentialCovFct(D, gamma[0], alpha)
    Cov_gl = covMatFromExponentialCovFct(D_between, gamma[0], alpha)
    Cov_gg = covMatFromExponentialCovFct(D_grid, gamma[0], alpha)
    # Remove mean for simple kriging
    rm_mean = np.mean(data)
    data = data - rm_mean
    # interpolate
    h = np.linalg.solve(Cov_ll, Cov_gl)
    grid_interp = np.matmul(np.transpose(h), data) + rm_mean
    # Covariance matrix
    Cov_grid_interp = (np.matmul(np.matmul(np.transpose(h), Cov_ll), h) -
                       2 * np.matmul(np.transpose(h), Cov_gl) + Cov_gg)
    
    return grid_interp, Cov_grid_interp

def my_scatter(data, x, y, figsize, fontsize=13, base=False, colorbar=True,
               colormap='plasma', colorbar_label=False, vmin=False, vmax=False,
               xlabel=False, ylabel=False, xlim=False, ylim=False, grid=False,
               title=False, coast=False, aspect=False):
    """A simple scatter plot."""
    #TODO: keyword arguments verwenden
    
    plt.rcParams.update({'font.size': fontsize})  # set before making the figure!
    
    fig, ax = plt.subplots(figsize=figsize)
    if coast:
        f = '/home/schroeder/Shapefiles/ne_10m_coastline/ne_10m_coastline_0_360.shp'
        coast = gpd.read_file(f)
        coast.plot(ax=ax, zorder=0)
        # plt.xlim([np.min(x)-1, np.max(x)+1])
        # plt.ylim([np.min(y)-1, np.max(y)+1])
    if vmin and vmax:
        Plot = ax.scatter(x=x, y=y, c=data, cmap=colormap, vmin=vmin,
                          vmax=vmax)
    elif vmin:
        Plot = ax.scatter(x=x, y=y, c=data, cmap=colormap, vmin=vmin)
    elif vmax:
        Plot = ax.scatter(x=x, y=y, c=data, cmap=colormap, vmax=vmax)
    else:
        Plot = ax.scatter(x=x, y=y, c=data, cmap=colormap)
    if colorbar:
        cbar = plt.colorbar(Plot)
        cbar.set_label(colorbar_label)        
    if base:
        if not xlim:
            xlim = ax.get_xlim()
        if not ylim:
            ylim = ax.get_ylim()
        base = gpd.read_file(base)
        base.plot(ax=ax, color='none', edgecolor='black', alpha=0.3)
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    if title:
        plt.title(title)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    if grid:
        plt.grid()
    if aspect:
        ax.set_aspect(aspect)
    