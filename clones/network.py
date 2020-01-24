#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 10:28:55 2019

@author: schroeder
"""

# Imports ---------------------------------------------------------------------
from clones import harmony, cfg
from time import gmtime, strftime
import datetime
import numpy as np
import astropy
import geopy
import geopy.distance
import os
import shutil
import xarray as xr
import json
import contextily as ctx
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import salem
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pygmt
from shapely.geometry import Point, LineString, Polygon

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
        
        if path:
            self.from_file(path)
        else:
            self.location = location
            self.country = country
            self.lat = lat
            self.lon = lon
            self.links = []
        
    def __repr__(self):
        """To be printed if instance is written."""
        
        summary = '<clones.Clock>\n'
        summary += 'Location: ' + self.location + '\n'
        summary += 'Lat, Lon: (' + str(self.lon) + ', ' + str(self.lat) + ')\n'
        summary += 'Links to: ('
        for c in self.links:
            summary += c.location + ', '
        return summary + ')\n \n'
    
    def link_to(self, b):
        """Link to another clock."""
        
        self.links.append(b)
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
    
    def sh2timeseries(self, F_lm, t, kind, unit, unitTo=[]):
        """Expand spherical harmonics into timeseries at clock location.
        
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
                        effect.append(f_lm_u.expand(lat=self.lat, lon=self.lon))
                    effects[u] = np.array(effect)
                try:
                    setattr(self, kind, effects)
                except:
                    print('Choose a proper kind of effect!')
            
    def to_file(self):
        """Writes the clock into netcdf files and a json readme."""
        
        if self.path is None:
            print('Warning: Clock ' + self.location +
                  ' is not part of a network')
            return

        # the readme data
        clo = {}
        clo['location'] = self.location
        clo['lat'] = self.lat
        clo['lon'] = self.lon
        clo['links'] = [c.location for c in self.links]
        clo['country'] = self.country
        with open(self.path + 'readme.json', 'w+') as f:
            json.dump(clo, f)
        
        # loop for the different sources
        effect_names = []
        for effect_name in ('I', 'H', 'A', 'O', 'GIA', 'S'):
            if hasattr(self, effect_name):
                effect_names.append(effect_name)
        
        # dict for the different signal descriptions and units
        signal = {'N': ('Geoid height', 'm'),
                  'h': ('Elevation', 'm'),
                  'g': ('Gravity acceleration', 'm/s^2'),
                  'ff': ('Fractional Frequency', '-')}
        
        # write the different sources to netcdf
        for effect_name in effect_names:
            effect = getattr(self, effect_name)
            ds = xr.Dataset()
            ds.attrs['creation_date'] = strftime("%Y-%m-%d %H:%M:%S", gmtime())
            ds.coords['time'] = ('time', effect['t'])
            for key, value in effect.items():
                if not key == 't':
                    ds[key] = ('time', value)
                    ds[key].attrs['description'] = signal[key][0]
                    ds[key].attrs['unit'] = signal[key][1]
            ds.to_netcdf(self.path + effect_name + '.nc')
            
    def from_file(self, path):
        """Called when the clock is initialized from a folder."""
        
        with open(path + 'readme.json', 'r') as f:
            clo = json.load(f)
        
        self.location = clo['location']
        self.country = clo['country']
        self.lat = clo['lat']
        self.lon = clo['lon']
        self.links = clo['links']
        
        # loop for the different sources
        effect_names = []
        for effect_name in ('I', 'H', 'A', 'O', 'GIA', 'S'):
            if effect_name + '.nc' in os.listdir(path):
                effect_names.append(effect_name)
                
        for effect_name in effect_names:
            ds = xr.open_dataset(path + effect_name + '.nc')
            dct = {}
            try:
                dct['t'] = ds['time'].data
            except:
                print('Warning: Time not included in ' + effect_name)
            for var in ds.data_vars:
                try:
                    dct[var] = ds[var].data
                except:
                    print('Warning: ' + var + ' not included in ' + effect_name)
            # try:
            #     dct['h'] = ds['h'].data
            # except:
            #     print('Warning: Elevation not included in ' + effect_name)
            # try:
            #     dct['N'] = ds['N'].data
            # except:
            #     print('Warning: Geoid height not included in ' + effect_name)
            setattr(self, effect_name, dct)  # equal to: self.effect_name = dct, but effect_name can be a string

    def add_noise(self):
        """."""
        # TODO, make an allantools.Dataset from the timeseries?

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
        summary += ('Distance: ' + str(np.round(self.length())) +
                    ' km\n')
        return summary
        
    def length(self):
        """Returns the length of the fibre link in km."""
        
        return geopy.distance.geodesic((self.a.lat, self.a.lon),
                                       (self.b.lat, self.b.lon)).km

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
    
    def __init__(self, reset=False):
        """Builds an optical clock network."""
        
        if reset and os.path.isdir(cfg.PATHS['WORKING_DIR'] + 'clocks'):
            shutil.rmtree(cfg.PATHS['WORKING_DIR'] + 'clocks')
            self.clocks = []
            self.links = []
            if not os.path.exists('clocks'):
                os.makedirs('clocks')
        elif not reset and os.path.isdir(cfg.PATHS['WORKING_DIR'] + 'clocks'):
            self.from_file()
        else:
            self.clocks = []
            self.links = []
            os.makedirs(cfg.PATHS['WORKING_DIR'] + 'clocks')
                    
    def __repr__(self):
        """To be printed if instance is written."""
        
        summary = '<Network of optical clocks>\n'
        for c in self.clocks:
            summary += c.location + '\n'
        return summary + '\n'
            
    def from_file(self):
        """Initializes an already built optical clock network."""
        
        self.clocks = []
        self.links = []
        clos = next(os.walk(cfg.PATHS['WORKING_DIR'] + 'clocks'))[1]
        for clo in clos:
            clock = Clock(path=cfg.PATHS['WORKING_DIR'] + 'clocks/' + clo + '/')
            self.add_clock(clock)  
        
        already_linked = []
        for clo in self.clocks:  # clocks in network
            for li in clo.links:  # string links in particular clock
                for c in self.search_clock('location', li):  # clock with this string
                    if not c.location in already_linked:
                        self.add_link(c, clo)                    
            already_linked.append(clo.location)
            # remove the links which are stored as strings from json loading
            clo.links[:] = [li for li in clo.links if not isinstance(li, str)]           
        
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
            
        clo.path = 'clocks/' + clo.location + '/'
        if not os.path.exists(clo.path):
            os.mkdir(clo.path)
        
    def add_link(self, a, b, state=9):
        """Build a fibre link between two clocks a and b."""
        
        L = Link(a, b, state)
        self.links.append(L)
        a.link_to(b)
        b.link_to(a)
        
    def lats(self):
        """Returns an array of the latitudes of all clocks."""
        
        lats = [clo.lat for clo in self.clocks]
        return np.array(lats)
        
    def lons(self):
        """Returns an array of the longitudes of all clocks."""
        
        lons = [clo.lon for clo in self.clocks]
        return np.array(lons)
    
    def sh2timeseries(self, F_lm, t, kind, unit, unitTo=[]):
        """Expand spherical harmonics into timeseries at each clock location.
        
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
    
    def plotNetwork(self, zoom='close'):
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
        relief = pygmt.datasets.load_earth_relief('02m')
        
        fig = pygmt.Figure()
        pygmt.makecpt(cmap='wiki-2.0', series=[-3175*0.45, 6000*0.45])
        fig.grdimage(relief, region=region,
                      projection=projection, frame="ag")
        fig.coast(region=region, projection=projection, frame="a",
                  shorelines="0.3p,black", borders="1/0.5p,black")
        fig.plot(self.lons(), self.lats(), style="c0.09i", color="white",
                  pen="black")
        # fig.plot(CLONETS.lons(), CLONETS.lats(), style="c0.09i", color="red",
        #           pen="black")
        for clo in self.clocks:
        # for clo in CLONETS.clocks:
            if clo.location == 'Potsdam':
                fig.text(x=clo.lon-0.3, y=clo.lat+0.6, text=clo.location, region=region,
                         projection=projection, font='12p,Helvetica,black',
                         justify='LT')
            elif clo.location == 'Helsinki':
                fig.text(x=clo.lon-3.3, y=clo.lat+1, text=clo.location, region=region,
                         projection=projection, font='12p,Helvetica,black',
                         justify='LT')
            else:
                # fig.plot(clo.lon, clo.lat, style='l1+t"blah"', color='white')
                fig.text(x=clo.lon+0.3, y=clo.lat, text=clo.location, region=region,
                         projection=projection, font='12p,Helvetica,black',
                         justify='LT')
        # for l in self.links:
        # # for l in CLONETS.links:
        #     fig.plot(style='ql'+str(l.a.lon)+'/'+str(l.a.lat)+'/'+str(l.b.lon)+'/'+str(l.b.lat)+'+lblah')
            
                
        return fig
    
    def plotSignal(self, kind, t, unitFrom, unitTo, t_ref=None, save=False):
        """Plots the signal on a map.
        
        :param kind:
        :type kind: str
        :param t: point in time
        :type t: datetime.datetime or str
        :param unitFrom: unit in which the data is stored
        :type unitFrom: str
        :param unitTo: unit that is to be plotted
        :type unitTo: str
        :param t_ref: optional reference point in time
        :type t_ref: datetime.datetime or str (must match t)
        
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
        
        cb_dict = {'U': '"Gravity potential [m@+2@+/s@+2@+]"',
                     'N': '"Geoid height [m]"',
                     'h': '"Elevation [m]"',
                     'sd': '"Surface Density [kg/m@+2@+]"',
                     'ewh': '"Equivalent water height [m]"',
                     'gravity': '"Gravity acceleration [m/s@+2@+]"',
                     'ff': '"Fractional frequency [-]"'}
        signal_dict = {'I': 'oggm_',
                       'I_scandinavia': 'oggm_',
                       'H': 'clm_tws_',
                       'A': 'coeffs_'}
        unit_dict = {'U': 'Gravity potential [m$^2$/s$^2$]',
                   'N': 'Geoid height [m]',
                   'h': 'Elevation [m]',
                   'sd': 'Surface Density [kg/m$^2$]',
                   'ewh': 'Equivalent water height [m]',
                   'gravity': 'Gravity acceleration [m/s$^2$]',
                   'ff': 'Fractional frequency [-]'}
        path = cfg.PATHS['data_path']
        savepath = path + '../fig/'
        
        if type(t) is not str:
            t = datetime.datetime.strftime(t, format='%Y_%m_%d')
            if t_ref:
                t_ref = datetime.datetime.strftime(t_ref, format='%Y_%m_%d')
        f_lm = harmony.shcoeffs_from_netcdf(
            os.path.join(path, kind, signal_dict[kind] + t + '.nc'))
        if t_ref:
            f_lm_ref = harmony.shcoeffs_from_netcdf(
                os.path.join(path, kind, signal_dict[kind] + t_ref + '.nc'))
            f_lm = f_lm - f_lm_ref
        f_lm = harmony.sh2sh(f_lm, unitFrom, unitTo)
        
        grid = f_lm.pad(720).expand()
        data = grid.to_array()
        x = grid.lons()
        y = grid.lats()
        # find out what the datalimits are within the shown region
        data_lim = np.concatenate((grid.to_array()[200:402, -81:],
                                grid.to_array()[200:402, :242]), axis=1)
        datamax = np.max(abs(data_lim))
        
        da = xr.DataArray(data, coords=[y, x], dims=['lat', 'lon'])
        # save the dataarray as netcdf to work around the 360° plotting problem
        da.to_dataset(name='dataarray').to_netcdf(path + 'temp/pygmt.nc')
        fig = pygmt.Figure()
        pygmt.makecpt(cmap='polar', series=[-datamax, datamax], reverse=True)
        fig.grdimage(path + 'temp/pygmt.nc', region=[-10, 30, 40, 65],
                     projection="S10/90/6i", frame="ag")  # frame: a for the standard frame, g for the grid lines
        fig.coast(region=[-10, 30, 40, 65], projection="S10/90/6i", frame="a",
                  shorelines="0.1p,black", borders="1/0.1p,black")
        fig.plot(self.lons(), self.lats(), style="c0.07i", color="white",
                 pen="black")
        # fig.colorbar(frame=['paf+lewh', 'y+l:m'])  # @+x@+ for ^x
        fig.colorbar(frame='paf+l' + cb_dict[unitTo])  # @+x@+ for ^x
        
        if save:
            if t_ref:
                fig.savefig(os.path.join(savepath, kind, signal_dict[kind] + t
                                         + '-' + t_ref + '_' + unitTo + '.pdf'))    
            else:
                fig.savefig(os.path.join(savepath, kind, signal_dict[kind] + t
                                         + '.pdf'))
        return fig

    def plotTimeseries(self, kind, unit):
        """Plots the timeseries of all clocks.
        
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
                print('Clock in ' + clo.location + ' does not have information' +
                      ' from ' + kind + ' about ' + unit)
        ax.set_xlabel('time [y]')
        unit_dict = {'U': 'Gravity potential [m$^2$/s$^2$]',
                     'N': 'Geoid height [m]',
                     'h': 'Elevation [m]',
                     'sd': 'Surface Density [kg/m$^2$]',
                     'ewh': 'Equivalent water height [m]',
                     'gravity': 'Gravity acceleration [m/s$^2$]',
                     'ff': 'Fractional frequency [-]'}
        ax.set_ylabel(unit_dict[unit])
        ax.grid()
        ax.legend()
        plt.tight_layout()
        