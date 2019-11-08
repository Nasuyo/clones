#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 17:28:22 2019

@author: schroeder

This script is a first attempt to collect the functionality of optical clock
simulation. Within the project CLOck NETwork Services (CLONETS) it represents
the actual clock rate simulation based on geopotential and height differences:
CLOck NETwork Simulation (CLONES)
"""

# -------------------------------- CLONES ----------------------------------- #

# Imports ---------------------------------------------------------------------
import numpy as np
import geopy
from clones.network import Clock, Link, Network
# Clock initialisation --------------------------------------------------------


CLONETS = Network()
bonn = Clock('Bonn', 51., 7., country='Germany')
bern = Clock('Bern', 45., 6., country='Switzerland')
CLONETS.add_clock(bonn)
CLONETS.add_clock(bern)
CLONETS.add_link(bonn, bern)
print(CLONETS)


