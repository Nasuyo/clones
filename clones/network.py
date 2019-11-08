#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 10:28:55 2019

@author: schroeder
"""

# Imports ---------------------------------------------------------------------
import numpy as np
import geopy

# Classes and functions -------------------------------------------------------

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
    :param links: fibre links to other clocks
    :type links: list of Clocks
    """
    
    def __init__(self, location, lat, lon, country=None):
        """Builds an optical clock."""
        
        self.location = location
        self.country = country
        self.lat = lat
        self.lon = lon
        self.links = []
#        if links is not None:
#            self.links = [clo for clo in links]
        
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
        # TODO: maybe add a pointer to or a Link itself to this (self) instance of clock
        
class Link():
    """A fibre link between two clocks."""
    
    def __init__(self, a, b):
        """Instanciates a fibre link between two clocks a and b."""
        
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
        """Returns the length of the fibre link."""
        
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
    
    def __init__(self):
        """Builds an optical clock network."""
        
        self.clocks = []
        self.links = []
        
    def __repr__(self):
        """To be printed if instance is written."""
        
        summary = '<Network of optical clocks>\n'
        for c in self.clocks:
            summary += c.location + '\n'
        return summary + '\n'

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
        
    def add_link(self, a, b):
        """Build a fibre link between two clocks a and b."""
        
        L = Link(a, b)
        self.links.append(L)
        a.link_to(b)
        b.link_to(a)
        
        
    
# func:
#:return: [ReturnDescription]
#:rtype: [ReturnType]