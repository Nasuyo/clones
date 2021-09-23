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
from clones import leastsquarescollocation as lsc
import numpy as np
import pyshtools as sh
import matplotlib.pyplot as plt
import datetime
from scipy.fftpack import fft
import scipy.constants
import time
import colorednoise as cn


# Settings --------------------------------------------------------------------
cfg.configure()

# Clock initialisation --------------------------------------------------------
CLONETS = Network('psmsl_only')
print(CLONETS)

# Plot network with pyGMT -----------------------------------------------------
# fig = CLONETS.plotNetwork(save=True)
# fig.show()

# Define two clocks -----------------------------------------------------------
# bonn = CLONETS.search_clock('location', 'Bonn')[0]
# bern = CLONETS.search_clock('location', 'Bern')[0]
# braunschweig = CLONETS.search_clock('location', 'Braunschweig')[0]
# torino = CLONETS.search_clock('location', 'Torino')[0]
# helsinki = CLONETS.search_clock('location', 'Helsinki')[0]
# gothenburg = CLONETS.search_clock('location', 'Gothenburg')[0]
# strasbourg = CLONETS.search_clock('location', 'Strasbourg')[0]
# london = CLONETS.search_clock('location', 'London')[0]

# # Hourly datetime timeseries --------------------------------------------------
# T = [datetime.datetime(2007, 1, d) for d in range(1, 32)]
# # T = [datetime.datetime(2007, 1, d, h) for d in range(1, 32) for h in range(24)]
# t_ref = '2007_01'

# Daily datetime timeseries ---------------------------------------------------
t0 = datetime.date(2007, 1, 1)
T = [t0 + datetime.timedelta(d) for d in range(0, 365)]
t_ref = '2007'

# # Monthly string timeseries ---------------------------------------------------
t_ref = '2006'
T = [t_ref + '_' + f'{d:02}' for d in range(1, 13)]

# # Monthly string timeseries 3 years -------------------------------------------
# months = ['%02d' % (i) for i in range(1, 13)] * 3
# months.insert(0, months.pop())
# months.insert(0, months.pop())
# months.insert(0, months.pop())
# years = 3 * ['2005'] + 12 * ['2006'] + 12 * ['2007'] + 9 * ['2008']
# T = [y + '_' + m for (y, m) in zip(years, months)]

# Plot clock timeseries -------------------------------------------------------
sigma = [1e-18, 1e-19, 1e-20]
unitTo = 'ff'
# bern.plotTimeseries(T, 'H', 'ewh', unitTo, t_ref=t_ref, sigma=sigma, save=True)
# bern.plotTimeFrequencies(T, 'H', 'ewh', unitTo, 86400, fmax=20, t_ref=t_ref,
#                           sigma=sigma,
#                           save=True)
# strasbourg.plotTimeseries(T, 'H', 'ewh', unitTo, t_ref=t_ref, sigma=sigma,
#                           save=True)
# strasbourg.plotTimeFrequencies(T, 'H', 'ewh', unitTo, 86400, fmax=20,
#                                 t_ref=t_ref, sigma=sigma,
#                                 save=True)
# torino.plotTimeseries(T, 'H', 'ewh', unitTo, t_ref=t_ref, sigma=sigma,
#                       save=True)
# torino.plotTimeFrequencies(T, 'H', 'ewh', unitTo, 86400, fmax=20, t_ref=t_ref,
#                             sigma=sigma,
#                             save=True)
# braunschweig.plotTimeseries(T, 'A', 'pot', 'ewh', t_ref=t_ref, sigma=sigma,
#                             save=False)
# braunschweig.plotTimeseries(T, 'A', 'pot', 'N', t_ref=t_ref, sigma=sigma,
#                             save=False)
# braunschweig.plotTimeseries(T, 'A', 'pot', 'h', t_ref=t_ref, sigma=sigma,
#                             save=False)
# braunschweig.plotTimeseries(T, 'A', 'pot', 'ff', t_ref=t_ref, sigma=sigma,
#                             save=True)
# braunschweig.plotTimeFrequencies(T, 'A', 'pot', unitTo, 86400,
#                                   t_ref=t_ref, sigma=sigma, save=False)
# braunschweig.plotTimeseries(T, 'A', 'pot', 'ff', t_ref=t_ref, sigma=2,
#                             save=False)
# braunschweig.plotTimeFrequencies(T, 'A', 'pot', unitTo, 86400,
#                                   t_ref=t_ref, sigma=sigma, save=True)
# gothenburg.plotTimeseries(T, 'H', 'ewh', unitTo, t_ref=t_ref, sigma=sigma,
#                           save=True)
# gothenburg.plotTimeFrequencies(T, 'H', 'ewh', unitTo, 86400, fmax=20,
#                                 t_ref=t_ref, sigma=sigma, save=True)
# helsinki.plotTimeseries(T, 'A', 'pot', 'ewh', t_ref=t_ref, sigma=sigma,
#                         save=False)
# helsinki.plotTimeFrequencies(T, 'H', 'ewh', unitTo, 86400, fmax=20,
#                               t_ref=t_ref, sigma=sigma, save=True)
# london.plotTimeseries(T, 'H', 'ewh', unitTo, t_ref=t_ref, sigma=sigma,
#                       save=True)
# london.plotTimeFrequencies(T, 'H', 'ewh', unitTo, 86400, fmax=20,
#                             t_ref=t_ref, sigma=sigma, save=True)

# T, signal = bern.sh2timeseries(T, 'H', 'ewh', unitTo, t_ref=t_ref)
# filt365 = signal - utils.ma(signal, 365)
# filt183 = signal - utils.ma(signal, 183)
# filt31 = signal - utils.ma(signal, 31)
# plt.plot(T, signal, label='signal')
# plt.plot(T, filt365, label='1 year filter')
# plt.plot(T, filt183, label='6 months filter')
# plt.plot(T, filt31, label='1 month filter')
# plt.grid()
# plt.legend()
# plt.show()

# # FFT of the clock signal -----------------------------------------------------
# delta_x = 86400  # [s]
# anz = len(T)
# len_x = delta_x * anz  # [s]
# Fs = 1 / delta_x  # [Hz]
# Fs_nyquist = 1 / (2*delta_x)  # [Hz]
# t = np.arange(anz) * delta_x  # Zeit Vektor [s]
# anz_freq = int(np.ceil(anz/2)) + 1  # Anzahl der Frequenzen [Hz]
# f = Fs_nyquist * np.linspace(0, 1, anz_freq)  # Frequenzvektor [Hz]

# freq = fft(signal)
# freq365 = fft(filt365)
# freq31 = fft(filt31)
# freq183 = fft(filt183)
# plt.plot(f[:20]*delta_x*365, 2*np.abs(freq[:20]), '.-', label='signal')
# plt.plot(f[:20]*delta_x*365, 2*np.abs(freq365[:20]), '.-', label='1 yr filter')
# plt.plot(f[:20]*delta_x*365, 2*np.abs(freq183[:20]), '.-', label='6 mo filter')
# plt.plot(f[:20]*delta_x*365, 2*np.abs(freq31[:20]), '.-', label='1 mo filter')
# plt.grid()
# plt.legend()
# plt.xlabel('Frequencies [1/yr]')
# plt.show()

# EOF -------------------------------------------------------------------------
# data matrix
# Y = [c.sh2timeseries(T, 'H', 'ewh', 'N', t_ref=t_ref, lmax=120)[1] for c in
#      CLONETS.clocks]
# Y = np.array(Y).T
# # Eigenvalue decomposition
# C = utils.C_from_data(Y)
# e_values, e_vectors = np.linalg.eig(C)  # the column [:,i] is the eigenvector corresponding to the eigenvalue [i]
# print('Variability of lambda 1: ' + str(e_values[0]/np.sum(e_values)) + ' %')
# fig, ax = plt.subplots(figsize=(5,4))
# plt.plot(e_values, '.-')
# from matplotlib.ticker import MaxNLocator
# ax.xaxis.set_major_locator(MaxNLocator(integer=True))
# plt.yscale('log')
# plt.title('Eigenvalues')
# plt.grid()
# plt.tight_layout()
# plt.savefig('../../fig/EOF/eigenvalues1.png', dpi=300)
# plt.show()
# # Principal Components
# D = Y @ e_vectors  # principal components in the columns
# PC1 = Y @ e_vectors[:, 0]  # example for the first principle component; = D[:, 0]
# # Reconstruction
# rec = D @ e_vectors.T  # reconstruction; = Y
# rec1 = np.outer(PC1, e_vectors[:, 0].reshape(1, 19))
# rec2 = np.outer(D[:, 1], e_vectors[:, 1].reshape(1, 19))
# plt.plot(T, Y[:, 0]*1e3, label='Original')
# plt.plot(T, rec1[:, 0]*1e3, label='Mode 1 rec.')
# plt.plot(T, rec2[:, 0]*1e3, label='Mode 2 rec.')
# plt.grid()
# plt.legend()
# plt.xticks(rotation=90)
# plt.ylabel('geoid height [mm]')
# plt.tight_layout()
# plt.savefig('../../fig/EOF/reconstruction1.png', dpi=300)
# plt.show()
# # Plot EOF 1
# lsc.my_scatter(e_vectors[:, 0], CLONETS.lons(), CLONETS.lats(), (6, 6),
#                grid=True, title='EOF 1', coast=True, aspect=1.4, xlim=[0, 26],
#                ylim=[43, 62], colorbar_label='', ylabel='Lat [°]', xlabel='Lon [°]')
# plt.tight_layout()
# plt.savefig('../../fig/EOF/EOF1.png', dpi=300)
# plt.show()
# plt.plot(PC1)
# plt.grid()
# plt.xlabel('days')
# plt.title('PC 1')
# plt.tight_layout()
# plt.savefig('../../fig/EOF/PC1.png', dpi=300)

# # EOF for GRACE
# _, Y_grace, lats, lons = CLONETS.clocks[0].sh2timeseries(
#     T, 'H', 'ewh', 'N', t_ref=t_ref, lmax=120, field=True)
# C_grace = utils.C_from_data(Y_grace)
# grace_values, grace_vectors = np.linalg.eig(C_grace)  # the column [:,i] is the eigenvector corresponding to the eigenvalue [i]
# print('Variability of lambda 1: ' + str(grace_values[0]/np.sum(grace_values)) +
#       ' %')
# plt.plot(grace_values, '.-')
# plt.yscale('log')
# plt.title('Eigenvalues')
# plt.show()
# # Principal Components
# D_grace = Y_grace @ grace_vectors  # principal components in the columns
# PC1_grace = Y_grace @ grace_vectors[:, 0]  # example for the first principle component; = D[:, 0]
# # Reconstruction
# rec_grace = D_grace @ grace_vectors.T  # reconstruction; = Y
# rec1_grace = np.outer(PC1_grace, grace_vectors[:, 0].reshape(1, len(grace_values)))
# rec2_grace = np.outer(D_grace[:, 1], grace_vectors[:, 1].reshape(1, len(grace_values)))
# plt.plot(T, Y_grace[:, 0], label='Original')
# plt.plot(T, rec1_grace[:, 0], label='mode 1')
# plt.plot(T, rec2_grace[:, 0], label='mode 2')
# plt.grid()
# plt.legend()
# plt.show()
# # Plot EOF 1
# lsc.my_scatter(grace_vectors[:, 0], lons, lats, (6, 6), grid=True, 
#                title='GRACE EOF 1', coast=True, aspect=1.4, xlim=[0, 26],
#                ylim=[43, 62])
# plt.show()
# plt.plot(PC1_grace, label='GRACE PC1')
# plt.plot(PC1, label='clocks PC1')
# plt.grid()
# plt.legend()
# plt.show()
# plt.plot(-PC1_grace/6, label='GRACE PC1 scaled')
# plt.plot(PC1, label='clocks PC1')
# plt.grid()
# plt.legend()
# plt.show()

# Conclusion from reconstruction: EOF1 explains about 98% of the variance.
# Expected result at this stage: Great similarity unless the first mode --
# representing the annual + semiannual signal -- is filtered out.
# next steps: 1. model clock errors and GRACE errors --> mainLSC file probably
# 2. try to separate northern and southern stations, maybe then ill
# get better results regarding the eigenvalues
# 3. "Naturally, this would only make sense, if the positions of the new data correspond to
# those that are being characterized by the EOFs.
# In regard to this master thesis GRACE data as well as NOAH data is indeed evaluated at
# the station’s positions."

# # Plot link timeseries --------------------------------------------------------
# braun_stras = CLONETS.search_link('locations',
#                                   ('Braunschweig', 'Strasbourg'))[0]
# braun_goth = CLONETS.search_link('locations',
#                                   ('Braunschweig', 'Gothenburg'))[0]
# braun_hel = CLONETS.search_link('locations', ('Braunschweig', 'Helsinki'))[0]
# braun_bern = CLONETS.search_link('locations', ('Braunschweig', 'Bern'))[0]
# braun_lond = CLONETS.search_link('locations', ('Braunschweig', 'London'))[0]

# braun_stras.plotTimeseries(T, 'H', 'ewh', unitTo, t_ref=t_ref, sigma=sigma,
#                             save=True)
# braun_stras.plotTimeFrequencies(T, 'H', 'ewh', unitTo, 86400, t_ref=t_ref,
#                                 sigma=sigma,
#                                 fmax=20, save=True)

# braun_goth.plotTimeseries(T, 'H', 'ewh', unitTo, t_ref=t_ref, sigma=sigma,
#                           save=True)
# braun_goth.plotTimeFrequencies(T, 'H', 'ewh', unitTo, 86400, t_ref=t_ref,
#                                 sigma=sigma,
#                                 fmax=20, save=True)

# braun_hel.plotTimeseries(T, 'H', 'ewh', unitTo, t_ref=t_ref, sigma=sigma,
#                           save=True)
# braun_hel.plotTimeFrequencies(T, 'H', 'ewh', unitTo, 86400, t_ref=t_ref,
#                               sigma=sigma,
#                               fmax=20, save=True)

# data = braun_bern.plotTimeseries(T, 'H', 'ewh', unitTo, t_ref=t_ref, sigma=sigma,
#                                  save=False)
# braun_bern.plotTimeFrequencies(T, 'H', 'ewh', unitTo, 86400, t_ref=t_ref,
#                                 sigma=sigma, save=False)
# braun_bern.plotTimeseries(T, 'A', 'pot', unitTo, t_ref=t_ref, sigma=sigma,
#                           save=False)
# braun_bern.plotTimeFrequencies(T, 'A', 'pot', unitTo, 86400, t_ref=t_ref,
#                                 sigma=sigma, save=False)

# braun_lond.plotTimeseries(T, 'H', 'ewh', unitTo, t_ref=t_ref, sigma=sigma,
#                           save=True)
# braun_lond.plotTimeFrequencies(T, 'H', 'ewh', unitTo, 86400, t_ref=t_ref,
#                                 sigma=sigma,
#                                 fmax=20, save=True)

# Find out n_eff --------------------------------------------------------------
# data = braun_bern.plotTimeseries(T, 'A', 'pot', unitTo, t_ref=t_ref, sigma=sigma,
#                                   save=False)
# L = 365

# N = 1000
# n_eff = np.zeros(3)
# for j in range(N):
#     sigma_gnss_white = np.array([1, 0.5, 0.2]) * 1e-3 * \
#         scipy.constants.g / scipy.constants.c**2
#     sigma_gnss_flicker = np.array([1, 0.5, 0.2]) * 1e-3 * \
#         scipy.constants.g / scipy.constants.c**2
#     noise_gnss_white = [np.random.normal(0, s, L) for s in
#                         sigma_gnss_white]
#     noise_gnss_flicker = [cn.powerlaw_psd_gaussian(1, L) * s
#                           for s in sigma_gnss_flicker]
#     noise = [noise_gnss_flicker[i] + noise_gnss_white[i] for i in range(3)]
#     # get the neff
#     for i in range(len(noise)):
#         C = harmony.disc_autocovariance(data + noise[i])
#         n_eff[i] += harmony.autocovariance_fct2neff(C[:31], m=31)
# n_eff = n_eff / N

# # Plot network timeseries -----------------------------------------------------
# loc= ['Bern', 'London', 'Warsaw', 'Gothenburg', 'Helsinki']
# loc_ref = 'Braunschweig'
# t_ref = '200510-200809'
# # t_ref = '2007'
# fig = CLONETS.plotTimeseries(T, 'I', 'ewh', 'ff', t_ref=t_ref, loc=loc,
#                               loc_ref=loc_ref)#, lmax=179)
# plt.show()

# tt, data_ref = braunschweig.sh2timeseries(T, 'H', 'ewh', unitTo, t_ref=t_ref)
# tt, data = helsinki.sh2timeseries(T, 'H', 'ewh', unitTo, t_ref=t_ref)
# data, data_Ref = 1e3 * np.array(data), 1e3 * np.array(data_ref)
# plt.plot(tt, data-data_ref)
# tt, data_ref = braunschweig.sh2timeseries(T, 'H', 'ewh', 'N', t_ref=t_ref, lmax=10)
# tt, data = helsinki.sh2timeseries(T, 'H', 'ewh', 'N', t_ref=t_ref, lmax=10)
# data, data_Ref = 1e3 * np.array(data), 1e3 * np.array(data_ref)
# plt.plot(tt, data-data_ref)
# T = [t_ref + '_' + f'{d:02}' for d in range(1, 13)]
# tt, data_ref = braunschweig.sh2timeseries(T, 'H', 'ewh', 'N', t_ref=t_ref)
# tt, data = helsinki.sh2timeseries(T, 'H', 'ewh', 'N', t_ref=t_ref)
# data, data_Ref = 1e3 * np.array(data), 1e3 * np.array(data_ref)
# plt.plot(tt, data-data_ref)


# # Plot network frequencies ----------------------------------------------------
# loc= ['Bern', 'London', 'Warsaw', 'Gothenburg', 'Helsinki']
# loc_ref = 'Braunschweig'
# fig = CLONETS.plotTimeFrequencies(T, 'I', 'ewh', 'ff', 86400*365/12, t_ref=t_ref,
#                                   loc=loc, loc_ref=loc_ref, save=False)#, lmax=179)

# Plot GRACE vs clocks timeseries ---------------------------------------------
# Assuming we know the elevation change better than the geoid change
# fig, t, tm, data, grace, sigma, s_grace = CLONETS.plotErrorTimeseries(
#     'A', 'pot', 'N', 'Bern', monthly=False, save=False)
# np.save('/home/schroeder/CLONETS/data/ts_results/Bern_tm.npy', tm)
# np.save('/home/schroeder/CLONETS/data/ts_results/Bern_data.npy', data)
# np.save('/home/schroeder/CLONETS/data/ts_results/Bern_grace.npy', grace)
# np.save('/home/schroeder/CLONETS/data/ts_results/Bern_sigma.npy', sigma)
# np.save('/home/schroeder/CLONETS/data/ts_results/Bern_s_grace.npy', s_grace)
# fig, t, tm, data, grace, sigma, s_grace = CLONETS.plotErrorTimeseries(
#     'A', 'pot', 'N', 'Bonn', monthly=False, save=False)
# np.save('/home/schroeder/CLONETS/data/ts_results/Bonn_tm.npy', tm)
# np.save('/home/schroeder/CLONETS/data/ts_results/Bonn_data.npy', data)
# np.save('/home/schroeder/CLONETS/data/ts_results/Bonn_grace.npy', grace)
# np.save('/home/schroeder/CLONETS/data/ts_results/Bonn_sigma.npy', sigma)
# np.save('/home/schroeder/CLONETS/data/ts_results/Bonn_s_grace.npy', s_grace)


# # Plot Root Mean Square -------------------------------------------------------
# fig, data = CLONETS.plotRMS(T, 'H', 'ewh', 'ewh', save=True, hourly=False,
#                             trend=31)
# fig.show()

# # Plot Root Mean Square for öffentlichkeit-----------------------------------
# fig, data = CLONETS.plotRMS2(T, 'H', 'ewh', 'ewh', save=True, trend=31)
# fig.show()

# Plot Earth System Component -------------------------------------------------
unitTo = 'N'
esc = 'O'
# lmax = 179
d1 = '2006_01'
d0 = '2006'
fig, data = CLONETS.plotESCatClocks(esc, d1, 'pot', unitTo, t_ref=d0,
                                    world=False, save=False)#,
                                    # loc_ref='Braunschweig')
# fig, data2 = CLONETS.plotESC(esc, d1, 'pot', unitTo, t_ref=d0, save=False,
#                               world=True)#, lmax=40)
fig.show()

# fig, ax = plt.subplots(figsize=(2, 3))
# plt.hist(data, orientation='horizontal', bins=10)

# Video
# T = [datetime.datetime.strftime(t, format='%Y_%m_%d') for t in T]
for d1 in T:
    fig, data = CLONETS.plotESC(esc, d1, 'pot', unitTo, t_ref=False, save='png',
                                        world=False)
    
    

# cb_dict = {'U': '"gravitational potential [m@+2@+/s@+2@+]"',
#             'N': '"Geoid height [mm]"',
#             'h': '"Elevation [mm]"',
#             'sd': '"Surface Density [kg/m@+2@+]"',
#             'ewh': '"Equivalent water height [m]"',
#             'gravity': '"gravitational acceleration [m/s@+2@+]"',
#             'ff': '"Fractional frequency [-]"',
#             'GRACE': '"Geoid height [mm]"'}
# fig = CLONETS.plot_europe_720(data2, cfg.PATHS['data_path'] + esc + '/',
#                               '', esc, unitTo, cb_dict, 'esc')
# fig.show()

# spec = sigma2.spectrum()
# plt.plot(np.arange(2, 50), 6371000*np.sqrt(spec[2:50]))
# plt.yscale('log')
# plt.grid()
# plt.savefig('/home/schroeder/CLONETS/fig/H/' + esc + '_' + str(lmax) + '_' + d1 + '.pdf')

# Plot Autocorrelation --------------------------------------------------------
# clocks = []
# for clo in CLONETS.clocks:
#     fig, data, dist, corr = clo.plotCorrelation(
#         T, 'A', 'pot', unitTo, save=False, trend='')
#     # fig.show()
#     plt.plot(dist.flatten(), corr.flatten(), '.')
    
#     number = np.zeros(100)
#     value = np.zeros(100)
#     for d, c in zip(dist.flatten(), corr.flatten()):
#         dist_class = int(d/100)
#         number[dist_class] += 1
#         value[dist_class] += c
#     value = value / number
#     plt.plot(np.arange(100)*100+50, value, label=clo.location)
#     clocks.append(value)
# plt.grid()
# plt.ylabel('Correlation')
# plt.xlabel('Distance [km]')
# plt.savefig('/home/schroeder/CLONETS/fig/A/Correlation.pdf')
# np.save('/home/schroeder/CLONETS/fig/A/Correlation.npy', np.array(clocks))

# clocks = np.load('/home/schroeder/CLONETS/fig/H/Correlation.npy')
# for i in [0, 6, 9, 15, 17, 18]:
#     plt.plot(np.arange(100)*100+50, clocks[i], label=CLONETS.clocks[i].location)
# plt.grid()
# plt.legend(loc='lower left')
# plt.ylabel('Correlation')
# plt.xlabel('Distance [km]')
# plt.xlim([0, 3000])
# plt.savefig('/home/schroeder/CLONETS/fig/H/Correlation_6_update.pdf')

# # Plot link timeseries for different degrees ----------------------------------
# l = CLONETS.search_link('locations', ('Braunschweig', 'Strasbourg'))[0]
# S = []
# for lmax in np.arange(5, 721, 5):
#     S.append(l.plotTimeseries(T, 'H', 'ewh', unitTo, t_ref=t_ref, lmax=lmax))
# S = np.array(S)
# np.save('/home/schroeder/CLONETS/data/Braunschweig_Strasbourg_different_lmax_ff.npy', S)

# S = np.load('/home/schroeder/CLONETS/data/Braunschweig_Strasbourg_different_lmax_N.npy')
# NUM_COLORS = 120
# cm = plt.get_cmap('plasma')  #'plasma' 'inferno' 'cool' 'ocean' 'coolwarm'
# fig, axes = plt.subplots(nrows=2, figsize=(6, 6))
# axes[0].set_prop_cycle(color=[cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)])
# for i in range(NUM_COLORS):
#     axes[0].plot(S[i, :])
# axes[0].grid()
# axes[0].set_ylabel('geoid height [mm]')

# blah = [utils.rms(S[i, :], S[-1, :]) for i in range(NUM_COLORS)]
# axes[1].plot(np.arange(NUM_COLORS)*6, blah, label='RMS to lmax=720')
# axes[1].grid()
# axes[1].set_ylabel('geoid height [mm]')
# axes[1].set_xlabel('$l_{max}$')
# axes[1].legend()
# plt.savefig('/home/schroeder/CLONETS/fig/H/Braunschweig_Strasbourg_different_lmax_N.pdf')

# power = l.analyze_degrees3(T, 'H', 'ewh', 'ewh', 719, t_ref=t_ref)
# np.save('/home/schroeder/CLONETS/data/power3.npy', power)

# # Save the network ------------------------------------------------------------
# CLONETS.to_file()

# Averaging -------------------------------------------------------------------

# path = '/home/schroeder/PyDeal/results/ERA5_ITG_DH_100_absolute/'

# Tstr = [datetime.datetime.strftime(t, format='%Y%m%d') for t in T]
# T_ = [datetime.datetime.strftime(t, format='%Y_%m_%d') for t in T]
# H = [f'{h:02}' for h in range(24)]

# for t, t_ in zip(Tstr, T_):
#     CS = sh.SHCoeffs.from_zeros(100)
#     for h in H:
#         cs = harmony.shcoeffs_from_netcdf(path+'coeffs_'+t+h+'.nc')
#         CS = CS + cs
#     CS = CS / 24
#     harmony.shcoeffs2netcdf(path+'coeffs_'+t_+'.nc', CS)

# # daily to monthly
# month = 11
# Tm = []
# [Tm.append(t) for t in T if t.month==month]
# CS = sh.SHCoeffs.from_zeros(100)
# for t in Tm:
#     cs = harmony.shcoeffs_from_netcdf(
#         path + 'coeffs_' + datetime.datetime.strftime(t, format='%Y_%m_%d'))
#     CS = CS + cs
# CS = CS / len(Tm)
# harmony.shcoeffs2netcdf(path + 'coeffs_'
#                         + datetime.datetime.strftime(t, format='%Y_%m'), CS)

# # monthly to annual
# CS = sh.SHCoeffs.from_zeros(100)
# for t in T:
#     cs = harmony.shcoeffs_from_netcdf(
#         path + 'coeffs_' + datetime.datetime.strftime(t, format='%Y_%m_%d'))
#     CS = CS + cs
# CS = CS / 365
# harmony.shcoeffs2netcdf(path + 'coeffs_2007.nc', CS)
    

# path = '/home/schroeder/CLONETS/data/H/'
# CS = sh.SHCoeffs.from_zeros(720)
# m = []
# mass = []
# month_length = 31
# for i in [f'{d:02}' for d in range(1, month_length+1)]:
#     CS = CS + harmony.shcoeffs_from_netcdf(path + 'clm_tws_2007_12_' + i)
#     # mass.append(harmony.shcoeffs_from_netcdf(path + 'oggm_2007_' + m[i] + '.nc').coeffs[0,0,0])
# CS = CS / month_length
# harmony.shcoeffs2netcdf(path + 'clm_tws_2007_12.nc', CS)

# path = '/home/schroeder/CLONETS/data/H/'
# CS = sh.SHCoeffs.from_zeros(720)
# t0 = datetime.date(2007, 1, 1)
# T = [t0 + datetime.timedelta(d) for d in range(365)]
# T = [datetime.datetime.strftime(t, format='%Y_%m_%d') for t in T]
# for t in T:
#     CS = CS + harmony.shcoeffs_from_netcdf(path + 'clm_tws_' + t)
# CS = CS / 365
# harmony.shcoeffs2netcdf(path + 'clm_tws_2007.nc', CS)

# F_lm = []
# for i in [f'{d:02}' for d in range(1, 31)]:
#     f_lm = harmony.shcoeffs_from_netcdf('/home/schroeder/CLONETS/data/H/clm_tws_2007_06_'
#                                         + i)
#     F_lm.append(f_lm)
# t = np.arange(2007 + 151/365, 2007 + 181/365, 1/365)

# path = '/home/schroeder/CLONETS/data/H_snow/'
# CS = sh.SHCoeffs.from_zeros(720)
# for t in T:
#     CS = CS + harmony.shcoeffs_from_netcdf(path + 'clm_tws_' + t)
# CS = CS / 12
# harmony.shcoeffs2netcdf(path + 'clm_tws_2007', CS)

# path = cfg.PATHS['data_path'] + 'A/'
# for d in range(1, 32):
#     for i in range(24):
#         blah = harmony.shcoeffs_from_netcdf(path+'coeffs_200701'+f'{d:02}'+f'{i:02}')
#         harmony.shcoeffs2netcdf(path+'coeffs_2007_01_'+f'{d:02}'+'_'+f'{i:02}', blah)

