#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 14:22:38 2021

@author: schroeder
"""
  
import folium    
                      
m = folium.Map(location=(0, 0), zoom_start=2)
folium.GeoJson('eu_atlantic.json', name='eu_atlantic').add_to(m)
folium.GeoJson('mediterranean.json', name='mediterranean', 
                style_function=lambda x: {'color': 'red'}).add_to(m)
folium.GeoJson('na_atlantic.json', name='na_atlantic', 
                style_function=lambda x: {'color': 'green'}).add_to(m)
folium.GeoJson('na_pacific.json', name='na_pacific', 
                style_function=lambda x: {'color': 'blue'}).add_to(m)
folium.GeoJson('asia_pacific.json', name='asia_pacific', 
                style_function=lambda x: {'color': 'orange'}).add_to(m)
folium.GeoJson('oceania.json', name='oceania', 
                style_function=lambda x: {'color': 'cyan'}).add_to(m)
m.save('clusters.html')

