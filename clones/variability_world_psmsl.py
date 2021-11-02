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
import geopandas as gpd
from shapely.geometry import mapping
import datetime
import time

# Settings --------------------------------------------------------------------
cfg.configure()
esc = 'AOHIS'
unitFrom = 'pot'
unitTo = 'N'
cb_dict = {'U': '"gravitational potential [m@+2@+/s@+2@+]"',
            'N': '"RMS of Geoid height [mm]"',
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
timeseries2 = []
for region in clusters:
    CLONETS = Network('psmsl_only', region=region)
    c_mean, c_mean_loc = CLONETS.mean_clock(T, esc, 'pot', 'N')
    c_mean2, c_mean_loc2 = CLONETS.mean_clock(T, 'O', 'pot', 'N')
    clocks.append(Clock(region, lat=c_mean_loc[0], lon=c_mean_loc[1]))
    timeseries.append(c_mean)
    timeseries2.append(c_mean2)

rms_monthly_w = np.array([utils.rms(np.array(timeseries)[:, i], c_mean[i]) for i in 
                        range(12)])
rms_monthly_w2 = np.array([utils.rms(np.array(timeseries2)[:, i], c_mean2[i]) for i in 
                        range(12)])
# World mean clock ------------------------------------------------------------
world_mean = np.mean(np.array(timeseries), axis=0)
world_mean2 = np.mean(np.array(timeseries2), axis=0)

# RMS -------------------------------------------------------------------------
RMS = np.array([utils.rms(timeseries[i], world_mean)
                for i in range(len(timeseries))])
if unitTo in('N', 'h', 'GRACE'):
    RMS = RMS * 1e3   
# Plot ------------------------------------------------------------------------
data = {'data': RMS, 'lat': np.array([c.lat for c in clocks]),
        'lon': np.array([c.lon for c in clocks])}
df = pd.DataFrame(data)

datamax = np.max(abs(data['data']))
fig = pygmt.Figure()
with pygmt.clib.Session() as session:
    session.call_module('gmtset', 'FONT 15p')
pygmt.makecpt(cmap='viridis', series=[0, datamax], reverse=True)
fig.coast(projection="R15c", region="d", frame=["a", "WSne"],
          shorelines="1/0.1p,black", borders="1/0.1p,black",
          land='grey')
fig.plot(x=df.lon, y=df.lat, style='c0.21i', color=1-df.data/datamax, cmap='viridis')

# for region in clusters:  #TODO
#     bla = gpd.read_file('../data/' + region + '.json')
#     blah = mapping(bla)
#     poly = np.array(blah['features'][0]['geometry']['coordinates'])[0,:,:]
#     fig.plot(data=poly, pen="1.0p", projection="R15c", straight_line=True)

with pygmt.clib.Session() as session:
    session.call_module('gmtset', 'FONT 18p')
fig.colorbar(position='JMR+w10c/0.6c', frame='paf+l' + cb_dict[unitTo])  # @+x@+ for ^x
fig.show()
path = (cfg.PATHS['fig_path'] + 'AOHIS/AOHIS_2006_01-2006_12_N_clockwise_RMS_world_mean.pdf')
fig.savefig(path)

RMS_monthly = []
for region in clusters:
    CLONETS = Network('psmsl_only', region=region)
    fig, data, c_mean = CLONETS.plotTimeseriesAndMean(
        T, esc, 'pot', 'N', save=False, vmin=-5, vmax=5, display_mean=False,
        two=True)
    fig2, data2, c_mean2 = CLONETS.plotTimeseriesAndMean(
        T, 'O', 'pot', 'N', save=False, vmin=-5, vmax=5, display_mean=False,
        two=True)
    # fig.savefig(cfg.PATHS['fig_path'] + 'timeseries/' + region + '_var_two.pdf')
    rms_monthly = np.array([utils.rms(data[i, :], c_mean[i]) for i in 
                            range(np.shape(data)[0])])
    rms_monthly2 = np.array([utils.rms(data2[i, :], c_mean2[i]) for i in 
                            range(np.shape(data2)[0])])
    fig, ax = plt.subplots()
    ax.bar(np.arange(1, 13)-0.2, rms_monthly, color='tab:orange', width=0.4, label='AOHIS')
    ax.bar(np.arange(1, 13)+0.2, rms_monthly2, color='tab:blue', width=0.4, label='Ocean only')
    plt.title(region)
    plt.ylabel('RMS of Geoid height [mm]')
    ax.set_xticks(np.arange(1, 13))
    ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug',
                        'Sep', 'Oct', 'Nov', 'Dec'])
    plt.xticks(rotation=90)
    plt.ylim([0, 4])
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(cfg.PATHS['fig_path'] + 'AOHIS/monthly_mean_' + region + '.pdf')
    RMS_monthly.append(rms_monthly)
## World mean clock plot ------------------------------------------------------    
# plt.rcParams.update({'font.size': 13})  # set before making the figure!        
# fig, ax = plt.subplots()
# T = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 
#       'Oct', 'Nov', 'Dec']
# # plt.plot(T, world_mean*1e3, color='tab:blue', linewidth=2, zorder=5)
# # plt.plot(T, np.transpose(np.array(timeseries))*1e3, color='tab:orange', linewidth=0.5)
# region_ts = np.array([timeseries[i]-world_mean for i in range(len(timeseries))])*1e3
# region_ts2 = np.array([timeseries2[i]-world_mean2 for i in range(len(timeseries))])*1e3
# lines_O = plt.plot(T, region_ts2.T, color='tab:blue', linewidth=0.5)
# lines_AOHIS = plt.plot(T, region_ts.T, color='tab:orange', linewidth=0.5)
# plt.ylabel('Geoid height [mm]')
# plt.xticks(rotation=90)
# plt.ylim([-5, 5])
# plt.title('World')
# plt.grid()
# plt.legend([lines_O[0], lines_AOHIS[0]], ['Ocean only', 'AOHIS'])
# plt.tight_layout()

# path = (cfg.PATHS['fig_path'] + 'timeseries/world_mean_clock_var_two.pdf')
# plt.savefig(path)

fig, ax = plt.subplots()
ax.bar(np.arange(1, 13)-0.2, rms_monthly_w*1e3, color='tab:orange', width=0.4, label='AOHIS')
ax.bar(np.arange(1, 13)+0.2, rms_monthly_w2*1e3, color='tab:blue', width=0.4, label='Ocean only')
plt.title('World')
plt.ylabel('RMS of Geoid height [mm]')
ax.set_xticks(np.arange(1, 13))
ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug',
                    'Sep', 'Oct', 'Nov', 'Dec'])
plt.xticks(rotation=90)
plt.ylim([0, 5])
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig(cfg.PATHS['fig_path'] + 'AOHIS/monthly_mean_world.pdf')