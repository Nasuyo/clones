#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 14:43:39 2021

@author: schroeder
"""

import numpy as np
import xarray as xr
import pandas as pd
import pyshtools as sh
import pygmt
import geojson
import pyshtools as sh
import scipy

GM = sh.constant.gm_wgs84.value  # gravitational constant times mass [m^2/s^2]
R = sh.constant.r3_wgs84.value  # mean radius of the Earth [m]
c = scipy.constants.c  # speed of light [m/s]


def plot_world(grid, path, esc, unitTo, cb_dict, rms=False, vmax=False):
    """Helper function for plotting on world map.
    
    Parameters
    ----------
    grid: sh.SHGrid
        grid to be plotted
    path: str
        path to the save location
    trend: str
        leads to different colormaps
    esc: str
        earth system component (Ice, Atmosphere, Hydrology, Ocean)
    unitTo: str
        unit of the plotted signal
    cb_dict: dictionary
        colorbar label dictionary
    inp_func: str, optional
        leads to a different colormap
        
    Returns
    -------
    fig: pygmt.Figure
        the plotted figure object
    """
    
    x = grid.lons()
    y = grid.lats()
    
    filename = path + 'PSMSL_stations.json'
    with open(filename) as f:
        gj = geojson.load(f)
    features = gj['features']
    names = [f.properties['name'] for f in features]
    x_stations = [f.geometry.coordinates[0] for f in features]  # lons
    y_stations = [f.geometry.coordinates[1] for f in features]  # lats
    
    if not vmax:
        vmax = np.max(abs(grid.data))
        print(vmax)
    
    da = xr.DataArray(grid.data, coords=[y, x], dims=['lat', 'lon'])
    # save the dataarray as netcdf to work around the 360° plotting problem
    da.to_dataset(name='dataarray').to_netcdf(path + 'temp/pygmt.nc')
    with pygmt.clib.Session() as session:
        session.call_module('gmtset', 'FONT 15p')
    fig = pygmt.Figure()
    if rms:
        pygmt.makecpt(cmap='viridis', series=[0, vmax], reverse=True)
    else:
        pygmt.makecpt(cmap='roma', series=[-vmax, vmax])
    fig.grdimage(path + 'temp/pygmt.nc', projection="R15c", region="d", 
                 frame=["ag", "WSne"])  # frame: a for the standard frame, g for the grid lines
    fig.coast(shorelines="1/0.1p,black", projection="R15c", region="d", 
              borders="1/0.1p,black")
    fig.plot(x_stations, y_stations, style="c0.07i", color="white",
             pen="black")
    with pygmt.clib.Session() as session:
        session.call_module('gmtset', 'FONT 18p')
    fig.colorbar(position='JMR+w10c/0.6c', frame='paf+l' + cb_dict[unitTo])  # @+x@+ for ^x
    
    return fig

def plot_world_clockwise(data, lats, lons, esc, unitTo, cb_dict, rms,
                         vmax=False):
    
    fig = pygmt.Figure()
    with pygmt.clib.Session() as session:
        session.call_module('gmtset', 'FONT 15p')
    if not vmax:
        vmax = np.max(abs(data))
    pygmt.makecpt(cmap='viridis', reverse=True, series=[0, vmax])
    fig.coast(projection="R15c", region="d", frame=["a", "WSne"],
              shorelines="1/0.1p,black", borders="1/0.1p,black",
              land='grey')
    fig.plot(x=lons, y=lats, style='c0.1i', color=1-data/vmax,
             cmap='viridis')
    with pygmt.clib.Session() as session:
        session.call_module('gmtset', 'FONT 18p')
    fig.colorbar(position='JMR+w10c/0.6c', frame='paf+l' + cb_dict[unitTo])  # @+x@+ for ^x

    return fig

def plot_eu(grid, path, esc, unitTo, cb_dict, rms=False, vmax=False):
    """Helper function for plotting on Europe map.
    
    Parameters
    ----------
    grid: sh.SHGrid
        grid to be plotted
    path: str
        path to the save location
    esc: str
        earth system component (Ice, Atmosphere, Hydrology, Ocean)
    unitTo: str
        unit of the plotted signal
    cb_dict: dictionary
        colorbar label dictionary
    rms: bool, optional
        leads to a different colormap
        
    Returns
    -------
    fig: pygmt.Figure
        the plotted figure object
    """
    
    x = grid.lons()
    y = grid.lats()
    
    filename = path + 'PSMSL_stations.json'
    with open(filename) as f:
        gj = geojson.load(f)
    features = gj['features']
    names = [f.properties['name'] for f in features]
    x_stations = [f.geometry.coordinates[0] for f in features]  # lons
    y_stations = [f.geometry.coordinates[1] for f in features]  # lats
    
    # find out what the datalimits are within the shown region: (#TODO: HARDCODED to region)
    if not vmax:
        data_lim = np.concatenate((grid.to_array()[200:402, -81:],
                                   grid.to_array()[200:402, :242]), axis=1)
        vmax = np.max(abs(data_lim))
        print(vmax)
    
    da = xr.DataArray(grid.data, coords=[y, x], dims=['lat', 'lon'])
    # save the dataarray as netcdf to work around the 360° plotting problem
    da.to_dataset(name='dataarray').to_netcdf(path + 'temp/pygmt.nc')
    with pygmt.clib.Session() as session:
        session.call_module('gmtset', 'FONT 20p')
    fig = pygmt.Figure()
    if rms:
        pygmt.makecpt(cmap='viridis', series=[0, vmax], reverse=True)
    else:
        pygmt.makecpt(cmap='roma', series=[-vmax, vmax])
    region = [-30, 30, 20, 70]
    fig.grdimage(path + 'temp/pygmt.nc', region=region,
                 projection="S0/90/6i", frame=["ag", "WSne"])  # frame: a for the standard frame, g for the grid lines
    fig.coast(region=region, projection="S0/90/6i",
              shorelines="1/0.1p,black",
              borders="1/0.1p,black")
    fig.plot(x_stations, y_stations, style="c0.07i", color="white",
             pen="black")
    with pygmt.clib.Session() as session:
        session.call_module('gmtset', 'FONT 24p')
    cbar_pos = 'g' + str(region[1]-2) + '/' + str(region[2]-10) + '+w13c/0.5c'
    fig.colorbar(position=cbar_pos, frame='paf+l' + cb_dict[unitTo])  # @+x@+ for ^x
    
    return fig

    
def plot_eu_clockwise(data, lats, lons, esc, unitTo, cb_dict, rms=False,
                      vmax=False):
    """Helper function for plotting on Europe map.
    
    Parameters
    ----------
    data: np.array
        clock data
    lats, lons: np.array
        latitudes and longitudes of the data points
    esc: str
        earth system component (Ice, Atmosphere, Hydrology, Ocean)
    unitTo: str
        unit of the plotted signal
    cb_dict: dictionary
        colorbar label dictionary
    rms: bool, optional
        leads to a different colormap
        
    Returns
    -------
    fig: pygmt.Figure
        the plotted figure object
    """
    
    fig = pygmt.Figure()
    with pygmt.clib.Session() as session:
        session.call_module('gmtset', 'FONT 20p')
    if not vmax:
        vmax = np.max(abs(data))
    pygmt.makecpt(cmap='viridis', reverse=True, series=[0, vmax])
    region = [-30, 30, 20, 70]
    fig.coast(region=region, projection="S0/90/6i",
              frame=["ag", "WSne"], shorelines="1/0.1p,black",
              borders="1/0.1p,black", land='grey')
    fig.plot(x=lons, y=lats, style='c0.1i', color=1-data/vmax,
             cmap='viridis')
    with pygmt.clib.Session() as session:
        session.call_module('gmtset', 'FONT 24p')     
    cbar_pos = 'g' + str(region[1]-2) + '/' + str(region[2]-10) + '+w13c/0.5c'
    fig.colorbar(position=cbar_pos, frame='paf+l' + cb_dict[unitTo])  # @+x@+ for ^x

    return fig

# Options ---------------------------------------------------------------------
T = '1995_01-2006_12'
esc = 'O'
unitTo = 'h'
region = 'na_atlantic'  # 'eu' or 'world'
vmax = 2
path = '../../data/'
rms = True
cb_dict = {'U': '"RMS of gravitational potential [m@+2@+/s@+2@+]"',
           'N': '"RMS of Geoid height [mm]"',
           'h': '"RMS of geometric height [mm]"',
           'sd': '"RMS of Surface Density [kg/m@+2@+]"',
           'ewh': '"RMS in Equivalent Water Height [m]"',
           'gravity': '"RMS of gravitational acceleration [m/s@+2@+]"',
           'ff': '"RMS of physical height [mm]"',
           'GRACE': '"RMS of Geoid height [mm]"'}

# Plot grid -------------------------------------------------------------------
# filename = path + '../fig/' + esc + '/' + esc + '_' + T + '_' + unitTo + '_RMS.nc'
# grid = sh.SHGrid.from_netcdf(filename)
# if unitTo == 'ff':
#     grid = grid * 1e3 / GM * R**2 * c**2 
# if region == 'world':
#     fig = plot_world(grid, path, esc, unitTo, cb_dict, rms, vmax=vmax)
#     fig.savefig(filename[:-3] + '_world.pdf')
# else:
#     fig = plot_eu(grid, path, esc, unitTo, cb_dict, rms, vmax=vmax)
#     fig.savefig(filename[:-3] + '_eu.pdf')
    
# Plot clockwise --------------------------------------------------------------
filename = path + '../fig/' + esc + '/' + esc + '_' + T + '_' + unitTo + '_clockwise_RMS_meanclock.pkl'
data = pd.read_pickle(filename)
clock_data = np.array(data.data)
if unitTo == 'ff':
    clock_data = clock_data * 1e3 / GM * R**2 * c**2 
lats = np.array(data.lat)
lons = np.array(data.lon)
if region == 'world':
    fig2 = plot_world_clockwise(clock_data, lats, lons, esc, unitTo, cb_dict, rms, vmax=vmax)
    fig2.savefig(filename[:-4] + '_world.pdf')
else:
    fig2 = plot_eu_clockwise(clock_data, lats, lons, esc, unitTo, cb_dict, rms, vmax=vmax)
    fig2.savefig(filename[:-4] + '_eu.pdf')
