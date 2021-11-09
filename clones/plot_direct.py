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


def plot_world(grid, path, esc, unitTo, cb_dict, rms=False, datamax=False):
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
    
    if not datamax:
        datamax = np.max(abs(grid.data))
        print(datamax)
    
    da = xr.DataArray(grid.data, coords=[y, x], dims=['lat', 'lon'])
    # save the dataarray as netcdf to work around the 360° plotting problem
    da.to_dataset(name='dataarray').to_netcdf(path + 'temp/pygmt.nc')
    with pygmt.clib.Session() as session:
        session.call_module('gmtset', 'FONT 15p')
    fig = pygmt.Figure()
    if rms:
        pygmt.makecpt(cmap='viridis', series=[0, datamax], reverse=True)
    else:
        pygmt.makecpt(cmap='roma', series=[-datamax, datamax])
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
                         datamax=False):
    
    fig = pygmt.Figure()
    with pygmt.clib.Session() as session:
        session.call_module('gmtset', 'FONT 15p')
    if not datamax:
        datamax = np.max(abs(data))
    pygmt.makecpt(cmap='viridis', reverse=True, series=[0, datamax])
    fig.coast(projection="R15c", region="d", frame=["a", "WSne"],
              shorelines="1/0.1p,black", borders="1/0.1p,black",
              land='grey')
    fig.plot(x=lons, y=lats, style='c0.1i', color=1-data/datamax,
             cmap='viridis')
    with pygmt.clib.Session() as session:
        session.call_module('gmtset', 'FONT 18p')
    fig.colorbar(position='JMR+w10c/0.6c', frame='paf+l' + cb_dict[unitTo])  # @+x@+ for ^x

    return fig

def plot_eu(grid, path, esc, unitTo, cb_dict, rms=False, datamax=False):
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
    if not datamax:
        data_lim = np.concatenate((grid.to_array()[200:402, -81:],
                                   grid.to_array()[200:402, :242]), axis=1)
        datamax = np.max(abs(data_lim))
        print(datamax)
    
    da = xr.DataArray(grid.data, coords=[y, x], dims=['lat', 'lon'])
    # save the dataarray as netcdf to work around the 360° plotting problem
    da.to_dataset(name='dataarray').to_netcdf(path + 'temp/pygmt.nc')
    with pygmt.clib.Session() as session:
        session.call_module('gmtset', 'FONT 20p')
    fig = pygmt.Figure()
    if rms:
        pygmt.makecpt(cmap='viridis', series=[0, datamax], reverse=True)
    else:
        pygmt.makecpt(cmap='roma', series=[-datamax, datamax])
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
                      datamax=False):
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
    if not datamax:
        datamax = np.max(abs(data))
    pygmt.makecpt(cmap='viridis', reverse=True, series=[0, datamax])
    region = [-30, 30, 20, 70]
    fig.coast(region=region, projection="S0/90/6i",
              frame=["ag", "WSne"], shorelines="1/0.1p,black",
              borders="1/0.1p,black", land='grey')
    fig.plot(x=lons, y=lats, style='c0.1i', color=1-data/datamax,
             cmap='viridis')
    with pygmt.clib.Session() as session:
        session.call_module('gmtset', 'FONT 24p')     
    cbar_pos = 'g' + str(region[1]-2) + '/' + str(region[2]-10) + '+w13c/0.5c'
    fig.colorbar(position=cbar_pos, frame='paf+l' + cb_dict[unitTo])  # @+x@+ for ^x

    return fig

# Options ---------------------------------------------------------------------
esc = 'AOHIS'
path = '../../data/'
unitTo = 'N'
cb_dict = {'U': '"RMS of gravitational potential [m@+2@+/s@+2@+]"',
           'N': '"RMS of Geoid height [mm]"',
           'h': '"RMS of Elevation [mm]"',
           'sd': '"RMS of Surface Density [kg/m@+2@+]"',
           'ewh': '"Mass variability in EWH [m]"',
           'gravity': '"RMS of gravitational acceleration [m/s@+2@+]"',
           'ff': '"RMS of Fractional frequency [-]"',
           'GRACE': '"RMS of Geoid height [mm]"'}
rms = True
datamax = 5

# Load grid -------------------------------------------------------------------
filename = path + '../fig/AOHIS/AOHIS_2006_01-2006_12_N_RMS.nc'
grid = sh.SHGrid.from_netcdf(filename)
# Load clockwise data ---------------------------------------------------------
filename = path + '../fig/AOHIS/AOHIS_2006_01-2006_12_N_clockwise_RMS.pkl'
data = pd.read_pickle(filename)
clock_data = np.array(data.data)
lats = np.array(data.lat)
lons = np.array(data.lon)
# Function calls --------------------------------------------------------------
fig = plot_eu(grid, path, esc, unitTo, cb_dict, rms, datamax=False)
fig = plot_world_clockwise(clock_data, lats, lons, esc, unitTo, cb_dict, rms,
                           datamax=False)
