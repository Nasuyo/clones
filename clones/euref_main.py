#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 17:28:22 2019

@author: schroeder

This script is a first attempt to collect the functionality of optical clock
simulation. Within the project CLOck NETwork Services (CLONETS) it represents
the actual clock rate simulation based on geopotential and height differences:
CLOck NEtwork Simulation (CLONES)
"""

# -------------------------------- CLONES ----------------------------------- #

# Imports ---------------------------------------------------------------------
from clones.network import Clock, Link, Network
from clones import cfg, harmony, utils
import numpy as np
import pyshtools as sh
import matplotlib.pyplot as plt
import datetime
from scipy.fftpack import fft
import scipy.constants
import time

# Settings --------------------------------------------------------------------
cfg.configure()

# Clock initialisation --------------------------------------------------------
CLONETS = Network()
print(CLONETS)

# # Plot network with pyGMT -----------------------------------------------------
# fig = CLONETS.plotNetwork()
# fig.show()

# Define two clocks -----------------------------------------------------------
bonn = CLONETS.search_clock('location', 'Bonn')[0]
bern = CLONETS.search_clock('location', 'Bern')[0]
braunschweig = CLONETS.search_clock('location', 'Braunschweig')[0]
torino = CLONETS.search_clock('location', 'Torino')[0]
helsinki = CLONETS.search_clock('location', 'Helsinki')[0]
gothenburg = CLONETS.search_clock('location', 'Gothenburg')[0]
strasbourg = CLONETS.search_clock('location', 'Strasbourg')[0]
london = CLONETS.search_clock('location', 'London')[0]

# Daily datetime timeseries ---------------------------------------------------
t0 = datetime.date(2007, 1, 1)
T = [t0 + datetime.timedelta(d) for d in range(0, 365)]
t_ref = '2007'

# # Monthly string timeseries ---------------------------------------------------
# t_ref = '2007'
# T = [t_ref + '_' + f'{d:02}' for d in range(1, 13)]

# Plot clock timeseries -------------------------------------------------------
sigma = False#[1e-18, 1e-19, 1e-20]
unitTo = 'N'


# # Plot Root Mean Square -------------------------------------------------------
# fig, data = CLONETS.plotRMS(T, 'H', 'ewh', 'N', save=True, trend='', lmax=40)
# fig.show()

# # Plot Root Mean Square for öffentlichkeit-----------------------------------
# fig, data = CLONETS.plotRMS2(T, 'H', 'ewh', 'ewh', save=True, trend=31)
# fig.show()

# Plot Earth System Component -------------------------------------------------
unitTo = 'N'
esc = 'A'
lmax = 720
d1 = '2007_06'
d0 = '2007'
fig, data = CLONETS.plotESCatClocks(esc, d1, 'pot', unitTo, t_ref=d0,
                                    save=False)
# fig.savefig('/home/schroeder/CLONETS/fig/A/euref_2007_06-2007_N_pointwise_rel_to_Braunschweig.pdf')
# fig, data2 = CLONETS.plotESC(esc, d1, 'pot', unitTo, t_ref=d0, save=False)
# fig.savefig('/home/schroeder/CLONETS/fig/A/euref_2007_06-2007_N_pointwise.png')
fig.show()


fig, ax = plt.subplots(figsize=(1.5, 3))
plt.hist(data,  orientation='horizontal', bins=15)
plt.ylabel('Geoid height [mm]')
plt.xlabel('Number of stations')
plt.savefig('/home/schroeder/CLONETS/fig/histogram_inset_euref.pdf')

# fig = CLONETS.plot_europe_720(data2, cfg.PATHS['data_path'] + esc + '/',
#                               '', esc, unitTo, cb_dict, 'esc')
# fig.show()


