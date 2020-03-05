#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 13:03:44 2019

@author: schroeder
dir of script: os.getcwd()
dir of fct: os.path.dirname(os.path.abspath(__file__))
"""

# Imports ---------------------------------------------------------------------
import configparser
import os

# Classes and functions -------------------------------------------------------

def configure():
    """Load the config file and set global variables."""

    global PATHS
    PATHS = {}
    PATHS['lln_path'] = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), '..', 'data')
        
    config = configparser.ConfigParser()
    config.read(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                             'config.cfg'))
        
    PATHS['WORKING_DIR'] = config['paths']['WORKING_DIR']
    PATHS['data_path'] = config['paths']['data_path']
    PATHS['fig_path'] = config['paths']['fig_path']
