#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 09:12:47 2021

@author: schroeder
"""

import numpy as np
from clones.network import Clock, Link, Network
from clones import cfg, harmony, utils
from clones import leastsquarescollocation as lsc
from os import walk
import pyshtools as sh
import time
import global_land_mask
import matplotlib.pyplot as plt

## Options --------------------------------------------------------------------
save_figs = False
compute_D = True
add_clocks = False
np.random.seed(7)
## Constants ------------------------------------------------------------------
R = sh.constant.r3_wgs84.value
lmax = 60
classDistance = 300  # [km]
alpha = 300  # [km]

## Load the "true" values: the hydrological model -----------------------------
# hydrology
path = '../../data/H/'
# filenames = next(walk(path), (None, None, []))[2]  # [] if no file
filename_mean = 'clm_tws_2007.nc'
filename = 'clm_tws_2007_01_01.nc'
nest_2007 = sh.SHCoeffs.from_netcdf(path+filename_mean, lmax=lmax)
nest_2007 = harmony.sh2sh(nest_2007, 'ewh', 'N')
nest_20070101 = sh.SHCoeffs.from_netcdf(path+filename, lmax=lmax)
nest_20070101 = harmony.sh2sh(nest_20070101, 'ewh', 'N')
nest_20070101 = nest_20070101 - nest_2007

# atmosphere
path = '../../data/A/'
filename_mean = 'coeffs_2007.nc'
filename = 'coeffs_2007_01_01.nc'
atm_2007 = sh.SHCoeffs.from_netcdf(path+filename_mean, lmax=lmax)
atm_2007 = harmony.sh2sh(atm_2007, 'pot', 'N')
atm_20070101 = sh.SHCoeffs.from_netcdf(path+filename, lmax=lmax)
atm_20070101 = harmony.sh2sh(atm_20070101, 'pot', 'N')
atm_20070101 = atm_20070101 - atm_2007

# sum
nest_20070101 = nest_20070101 + atm_20070101

## Load the GRACE and Bender errors -------------------------------------------
# filename = '/home/schroeder/CLONETS/data/von_Basem_Covarianzmatrix/Full_Cov_Matrix_grace_deg100.txt'
# f_bender = '/home/schroeder/CLONETS/data/von_Basem_Covarianzmatrix/Full_Cov_Matrix_bender_deg100.txt'
# C = harmony.read_cov(filename, 100)
# C_bender = harmony.read_cov(f_bender, 100)
# var = harmony.variances_from_cov(C, lmax)
# var.to_netcdf('var.nc')
var = sh.SHCoeffs.from_netcdf('var.nc')
# var_bender = harmony.variances_from_cov(C_bender, 100)
GRACE_std = sh.SHCoeffs.from_array(np.sqrt(var.coeffs)) * R
# Bender_std = sh.SHCoeffs.from_array(np.sqrt(var_bender.coeffs)) * R

random_errors = np.random.randn(np.shape(GRACE_std.coeffs)[
                                1], np.shape(GRACE_std.coeffs)[2])
GRACE_w_errors = (sh.SHCoeffs.from_array(GRACE_std.coeffs * random_errors) +
                  nest_20070101.pad(lmax))
# nest_20070101.pad(100).plot_spectrum2d(vmin=1e-18, vmax=1e-6)
# GRACE_w_errors.plot_spectrum2d(vmin=1e-18, vmax=1e-6)
# nest_20070101.expand().plot(colorbar='bottom', cmap_limits=[-0.004, 0.01])
# GRACE_w_errors.expand().plot(colorbar='bottom', cmap_limits=[-0.004, 0.01])
# plt.show()

## simulate the clocks --------------------------------------------------------
cfg.configure()
CLONETS = Network('clonets')
clock_gain = []
GRACE_gain = []  # 0.65 mm
for bla in range(100):
    random_errors = np.random.randn(np.shape(GRACE_std.coeffs)[
                                1], np.shape(GRACE_std.coeffs)[2])
    GRACE_w_errors = (sh.SHCoeffs.from_array(GRACE_std.coeffs * random_errors) +
                      nest_20070101.pad(lmax))
    clock_lats = np.array([c.lat for c in CLONETS.clocks])
    clock_lons = np.array([c.lon for c in CLONETS.clocks])
    if add_clocks:
        rng = np.random.default_rng(seed=7)
        bla = rng.random(add_clocks) * 21 + 42
        clock_lats = np.concatenate((clock_lats, bla))
        rng = np.random.default_rng(seed=8)
        bla = rng.random(add_clocks) * 27
        # np.random.shuffle(bla)
        clock_lons = np.concatenate((clock_lons, bla))
    
    is_on_land = global_land_mask.globe.is_land(clock_lats, clock_lons)
    clock_lats = clock_lats[is_on_land]
    clock_lons = clock_lons[is_on_land]
    
    clock_geoid = [nest_20070101.expand(lat=clock_lats[i], lon=clock_lons[i])
                   for i in range(len(clock_lats))]
    factor = 0.915
    clocks_std = np.sqrt(np.sqrt((0.001*factor)**2 + (0.001*factor)**2)**2 +
                         0.0005**2)  # 1mm per clock and 0.5 mm for GNSS overall
    clocks_std = 0.0014  # [mm] in other words^^
    random_errors_for_clocks = np.random.randn(np.shape(clock_geoid)[0])
    clock_w_errors = np.array(clock_geoid) #+ random_errors_for_clocks * clocks_std
    # plt.plot(clock_geoid, '.', markersize=10, label='true')
    # plt.plot(clock_w_errors, '.', markersize=10, label='noisy')
    # plt.legend()
    # plt.grid()
    # plt.show()
    
    ## define area, distances, covariance matrix ----------------------------------
    # area
    lat_limits = [44, 61]
    lon_limits = [0, 25]
    nest_grid = nest_20070101.expand()
    lat1 = np.ndarray.flatten(np.array(np.where(nest_grid.lats() < lat_limits[1])))
    lat2 = np.ndarray.flatten(np.array(np.where(nest_grid.lats() > lat_limits[0])))
    lat_ind = np.intersect1d(lat1, lat2)
    lon1 = np.ndarray.flatten(np.array(np.where(nest_grid.lons() < lon_limits[1])))
    lon2 = np.ndarray.flatten(np.array(np.where(nest_grid.lons() > lon_limits[0])))
    if lon_limits[0] >= 0:
        lon_ind = np.intersect1d(lon2, lon1)
    else:
        lon_ind = np.concatenate((lon2, lon1))
    true_eu = nest_grid.data[lat_ind][:, lon_ind]
    true_eu_flat = np.ndarray.flatten(true_eu)
    lats_eu = nest_grid.lats()[lat_ind]
    lons_eu = nest_grid.lons()[lon_ind]
    grid_eu = np.meshgrid(lons_eu, lats_eu)
    lats_eu = grid_eu[1].flatten()
    lons_eu = grid_eu[0].flatten()
    # ocean mask: only keep points on land
    lons_eu[lons_eu > 180] -= 360
    is_on_land = global_land_mask.globe.is_land(lats_eu, lons_eu)
    lons_eu[lons_eu < 0] += 360
    lats_eu_land = lats_eu[is_on_land]
    lons_eu_land = lons_eu[is_on_land]
    true_eu_flat_land = true_eu_flat[is_on_land]
    
    # distances
    if compute_D:
        t0 = time.time()
        # verbosity = 1
        # D = lsc.sphericalDistance(lats_eu[np.arange(0, len(lats_eu), verbosity)],
        #                           lons_eu[np.arange(0, len(lons_eu), verbosity)],
        #                           lats_eu[np.arange(0, len(lats_eu), verbosity)],
        #                           lons_eu[np.arange(0, len(lons_eu), verbosity)])
        D = lsc.sphericalDistance(lats_eu_land, lons_eu_land,
                                  lats_eu_land, lons_eu_land)
        print(time.time() - t0)
        # covariance
        # gamma, delta, classDistanceMean, D = lsc.empiricalCovarianceFunction(
        #     lats_eu[np.arange(0, len(lats_eu), verbosity)],
        #     lons_eu[np.arange(0, len(lats_eu), verbosity)],
        #     true_eu_flat[np.arange(0, len(true_eu_flat), verbosity)], D, classDistance)
        gamma, delta, classDistanceMean, D = lsc.empiricalCovarianceFunction(
            lats_eu_land, lons_eu_land, true_eu_flat_land, D, classDistance)
        # np.save('D', D)
        # np.save('gamma', gamma)
        # np.save('delta', delta)
        # np.save('classDistanceMean', classDistanceMean)
    else:
        D = np.load('D.npy')
        gamma = np.load('gamma.npy')  # empirical Covariance function
        delta = np.load('delta.npy')  # empirical correlation function
        classDistanceMean = np.load('classDistanceMean.npy')
    
    dist = np.arange(0, classDistanceMean[-1] + classDistance)
    expFct = lsc.covMatFromExponentialCovFct(dist, gamma[0], alpha)
    plt.plot(classDistanceMean, gamma, 'o')
    plt.plot(dist, expFct, '.')
    plt.show()
    
    # GRACE flat
    GRACE_w_errors_flat_land = np.array([GRACE_w_errors.expand(lon=lons_eu_land[i],
                                                               lat=lats_eu_land[i])
                                         for i in range(len(lons_eu_land))])
    
    ## LSC ------------------------------------------------------------------------
    grid_interp, Cov_grid_interp = lsc.interpLSC(
        clock_lats, clock_lons, clock_w_errors, lats_eu_land, lons_eu_land, D,
        alpha, gamma)
    
    ## Plots ----------------------------------------------------------------------
    # LSC
    # lsc.my_scatter(np.concatenate((clock_w_errors*1e3, grid_interp*1e3)),
    #                 np.concatenate((clock_lons, lons_eu_land)),
    #                 np.concatenate((clock_lats, lats_eu_land)), (6, 6), grid=True,
    #                 vmin=-15, vmax=7,
    #                 colorbar_label='Geoid heights  [mm]', title='LSC projected',
    #                 coast=True, aspect=1.4, xlim=[0, 26], ylim=[43, 62])
    # plt.savefig('../../fig/LSC/LSC.png', dpi=300)
    # # LSC without clocks
    # lsc.my_scatter(grid_interp*1e3,
    #                 lons_eu_land,
    #                 lats_eu_land, (6, 6), grid=True, vmin=-15, vmax=7,
    #                 colorbar_label='Geoid heights  [mm]', title='LSC projected',
    #                 coast=True, aspect=1.4, xlim=[0, 26], ylim=[43, 62])
    # plt.savefig('../../fig/LSC/LSC_wo_clocks.png', dpi=300)
    # # # "True" values
    # lsc.my_scatter(true_eu_flat_land*1e3,
    #                 lons_eu_land,
    #                 lats_eu_land, (6, 6), grid=True, vmin=-15, vmax=7,
    #                 colorbar_label='Geoid heights  [mm]', title='True', coast=True,
    #                 aspect=1.4, xlim=[0, 26], ylim=[43, 62])
    # plt.savefig('../../fig/LSC/True.png', dpi=300)   
    # Difference
    # lsc.my_scatter(grid_interp*1e3-true_eu_flat_land*1e3,
    #                 lons_eu_land,
    #                 lats_eu_land, (6, 6), grid=True,
    #                 colorbar_label='Geoid heights [mm]', title='LSC - true',
    #                 coast=True, aspect=1.4, xlim=[0, 26], ylim=[43, 62])
    # plt.savefig('../../fig/LSC/LSC-True.png', dpi=300)
    # # True values incl. clocks
    # lsc.my_scatter(np.concatenate((clock_w_errors*1e3, true_eu_flat_land*1e3)),
    #                np.concatenate((clock_lons, lons_eu_land)),
    #                np.concatenate((clock_lats, lats_eu_land)), (6, 6), grid=True,
    #                colorbar_label='Geoid heights  [mm]', title='true with clocks',
    #                coast=True, aspect=1.4, xlim=[0, 26], ylim=[43, 62])
    # plt.savefig('../../fig/LSC/True2.png', dpi=300)
    # # GRACE
    lsc.my_scatter(GRACE_w_errors_flat_land*1e3,
                    lons_eu_land,
                    lats_eu_land, (6, 6), grid=True, vmin=-15, vmax=7,
                    colorbar_label='Geoid heights [mm]', title='GRACE', coast=True,
                    aspect=1.4, xlim=[0, 26], ylim=[43, 62])
    plt.savefig('../../fig/LSC/GRACE.png', dpi=300)
    # # Difference GRACE - true
    lsc.my_scatter(GRACE_w_errors_flat_land*1e3-true_eu_flat_land*1e3,
                    lons_eu_land,
                    lats_eu_land, (6, 6), grid=True,
                    colorbar_label='Geoid heights [mm]', title='GRACE - true',
                    coast=True, aspect=1.4, xlim=[0, 26], ylim=[43, 62])
    plt.savefig('../../fig/LSC/GRACE-True.png', dpi=300)
    
    # only clocks
    # lsc.my_scatter(clock_w_errors*1e3,
    #                 clock_lons,
    #                 clock_lats, (6, 6), grid=True, vmin=-15, vmax=7,
    #                 colorbar_label='Geoid heights  [mm]', title='Clocks only',
    #                 coast=True, aspect=1.4, xlim=[0, 26], ylim=[43, 62])
    # plt.savefig('../../fig/LSC/clocks.png', dpi=300)
    # # true and true clocks
    # lsc.my_scatter(np.concatenate((np.array(clock_geoid)*1e3, true_eu_flat_land*1e3)),
    #                np.concatenate((clock_lons, lons_eu_land)),
    #                np.concatenate((clock_lats, lats_eu_land)), (6, 6), grid=True,
    #                colorbar_label='Geoid heights  [mm]', title='true with true clocks',
    #                coast=True, aspect=1.4, xlim=[0, 26], ylim=[43, 62])
    ## RMSE -----------------------------------------------------------------------
    rms_grace = utils.rms(GRACE_w_errors_flat_land, true_eu_flat_land) * 1e3
    print('GRACE RMSE: ', str(rms_grace))
    rms_clocks = utils.rms(grid_interp, true_eu_flat_land) * 1e3
    print('Clocks RMSE: ', str(rms_clocks))

    clock_gain.append(rms_clocks)
    GRACE_gain.append(rms_grace)
    # TODO. Note: das sind natürlich keine SH's. "GRACE" ist also nicht besonders aussagekräftig so wie's gerade ist.
    # naja doch die unterschiede zum true sind auch hier aussagekräftig.
    # 1. abspeichern und in ppt
    # 2. RMSE berechnen
    # 3. RMSE bei verschiedenen Uhrendichten berechnen
    # 4. Ozean und Atmosphäre dazuholen
# plt.plot(np.arange(29, 219), clock_gain)
# plt.xlabel('#clocks')
# plt.ylabel('RMSE [mm]')
# plt.grid()
# plt.savefig('../../fig/LSC/alpha300.png', dpi=300)
