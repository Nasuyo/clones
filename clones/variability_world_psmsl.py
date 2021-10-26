#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 13:05:44 2021

@author: schroeder
"""

# Imports ---------------------------------------------------------------------
from clones.network import Clock, Link, Network
from clones import cfg, harmony, utils
import numpy as np
import matplotlib.pyplot as plt
import pygmt
import pandas as pd
import datetime
import time

# Settings --------------------------------------------------------------------
cfg.configure()
esc = 'O'
unitFrom = 'pot'
unitTo = 'N'
cb_dict = {'U': '"gravitational potential [m@+2@+/s@+2@+]"',
            'N': '"Geoid height [mm]"',
            'h': '"Elevation [mm]"',
            'sd': '"Surface Density [kg/m@+2@+]"',
            'ewh': '"Equivalent water height [m]"',
            'ewhGRACE': '"Equivalent water height [m]"',
            'gravity': '"gravitational acceleration [m/s@+2@+]"',
            'ff': '"Fractional frequency [-]"',
            'GRACE': '"Geoid height [mm]"'}
# Monthly string timeseries ---------------------------------------------------
t_ref = '2006'
T = [t_ref + '_' + f'{d:02}' for d in range(1, 13)]

# Cluster regions  ------------------------------------------------------------
clusters = ['eu_atlantic', 'mediterranean', 'na_atlantic', 'na_pacific',
            'asia_pacific', 'oceania']

# Cluster mean clocks ---------------------------------------------------------
clocks = []
timeseries = []
for region in clusters:
    CLONETS = Network('psmsl_only', region=region)
    c_mean, c_mean_loc = CLONETS.mean_clock(T, 'O', 'pot', 'N')
    clocks.append(Clock(region, lat=c_mean_loc[0], lon=c_mean_loc[1]))
    timeseries.append(c_mean)
 
# World mean clock ------------------------------------------------------------
world_mean = np.mean(np.array(timeseries), axis=0)

# RMS -------------------------------------------------------------------------
RMS = np.array([utils.rms(timeseries[i], world_mean)
                for i in range(len(timeseries))])
if unitTo in('N', 'h', 'GRACE'):
    RMS = RMS * 1e3   
# Plot ------------------------------------------------------------------------
data = {'data': RMS, 'lat': np.array([c.lat for c in clocks]),
        'lon': np.array([c.lon for c in clocks])}
df = pd.DataFrame(data)

fig = pygmt.Figure()
with pygmt.clib.Session() as session:
    session.call_module('gmtset', 'FONT 15p')
pygmt.makecpt(cmap='viridis', series=[0, 1], reverse=True)
fig.coast(projection="R15c", region="d", frame=["a", "WSne"],
          shorelines="1/0.1p,black", borders="1/0.1p,black",
          land='grey')
fig.plot(x=df.lon, y=df.lat, style='c0.21i', color=1-df.data,
                 cmap='viridis')
with pygmt.clib.Session() as session:
    session.call_module('gmtset', 'FONT 18p')
fig.colorbar(position='JMR+w10c/0.6c', frame='paf+l' + cb_dict[unitTo])  # @+x@+ for ^x
fig.show()
path = (cfg.PATHS['fig_path'] + 'O/O_2006_01-2006_12_N_clockwise_RMS_world_mean.pdf')
fig.savefig(path)

for region in clusters:
    CLONETS = Network('psmsl_only', region=region)
    fig, data, c_mean = CLONETS.plotTimeseriesAndMean(T, 'O', 'pot', 'N',
                                                      save=False)
    fig.savefig(cfg.PATHS['fig_path'] + 'timeseries/' + region + '.pdf')
plt.rcParams.update({'font.size': 13})  # set before making the figure!        
fig, ax = plt.subplots()
T = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 
     'Oct', 'Nov', 'Dec', ]
plt.plot(T, world_mean*1e3, color='tab:blue', linewidth=2, zorder=5)
plt.plot(T, np.transpose(np.array(timeseries))*1e3, color='tab:orange', linewidth=0.5)
plt.ylabel('Geoid height [mm]')
plt.xticks(rotation=90)
# plt.title(self.region)
plt.grid()
plt.legend(['world mean clock', 'regional mean clocks'])#bbox_to_anchor=(1., 1.))
plt.tight_layout()

path = (cfg.PATHS['fig_path'] + 'timeseries/world_mean_clock.pdf')
plt.savefig(path)