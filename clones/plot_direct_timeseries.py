#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 14:43:39 2021

@author: schroeder
"""

import numpy as np
import pyshtools as sh
import scipy
from clones import utils
from datetime import datetime
import matplotlib.pyplot as plt

GM = sh.constant.gm_wgs84.value  # gravitational constant times mass [m^2/s^2]
R = sh.constant.r3_wgs84.value  # mean radius of the Earth [m]
c = scipy.constants.c  # speed of light [m/s]

# Options ---------------------------------------------------------------------
T = []
for y in range(1995, 2007):
    for d in range(1, 13):
        T.append(str(y) + '_' + f'{d:02}')      # str
T = [datetime.strptime(t, '%Y_%m') for t in T]  # datetime
T = [utils.datetime2frac(t) for t in T]         # float
link = 'Bern-Braunschweig'
esc = 'O'
unitTo = 'ff'
vmax = 6
path = '../../data/'
cb_dict = {'U': '"RMS of gravitational potential [m@+2@+/s@+2@+]"',
           'N': '"RMS of Geoid height [mm]"',
           'h': '"RMS of geometric height [mm]"',
           'sd': '"RMS of Surface Density [kg/m@+2@+]"',
           'ewh': '"RMS in Equivalent Water Height [m]"',
           'gravity': '"RMS of gravitational acceleration [m/s@+2@+]"',
           'ff': 'RMS of physical height [mm]',
           'GRACE': '"RMS of Geoid height [mm]"'}
# Load and compute ------------------------------------------------------------
A = np.load(path + 'ts_results/A_' + unitTo + '_' + link + '.npy')
O = np.load(path + 'ts_results/O_' + unitTo + '_' + link + '.npy')
H = np.load(path + 'ts_results/H_' + unitTo + '_' + link + '.npy')
AOHIS = np.load(path + 'ts_results/AOHIS_' + unitTo + '_' + link + '.npy')
if unitTo == 'ff':
    A = A * 1e3 / GM * R**2 * c**2 
    O = O * 1e3 / GM * R**2 * c**2 
    H = H * 1e3 / GM * R**2 * c**2 
    AOHIS = AOHIS * 1e3 / GM * R**2 * c**2 
model = utils.trend(np.array(T), np.array(AOHIS))
AOHIS_trend = np.array([model.intercept_ + t * model.coef_ for t in T]).flatten()
AOHIS_detrended = AOHIS - AOHIS_trend
# Plot ------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 4))
plt.plot(T, A, label='A', color='tab:orange')
plt.plot(T, O, label='O', color='tab:blue')
plt.plot(T, H, label='H', color='tab:green')
# plt.plot(T, AOHIS_detrended, label='AOHIS')
# plt.plot(T, AOHIS, label='AOHIS')
# plt.plot(T, AOHIS_trend, label='AOHIS_trend')
plt.grid()
plt.ylim((-vmax, vmax))
plt.legend()
plt.ylabel(cb_dict[unitTo])
plt.savefig(path + '../fig/timeseries/' + link + '_' + unitTo + '_AOH.pdf')

fig, ax = plt.subplots(figsize=(8, 4))
# plt.plot(T, A, label='A', color='tab:orange')
# plt.plot(T, O, label='O', color='tab:blue')
# plt.plot(T, H, label='H', color='tab:green')
plt.plot(T, AOHIS_detrended, label='AOHIS', color='tab:red')
# plt.plot(T, AOHIS, label='AOHIS')
# plt.plot(T, AOHIS_trend, label='AOHIS_trend')
plt.grid()
plt.ylim((-vmax, vmax))
plt.legend()
plt.ylabel(cb_dict[unitTo])
plt.savefig(path + '../fig/timeseries/' + link + '_' + unitTo + '_AOHIS.pdf')