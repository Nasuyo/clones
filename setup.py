#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 10:36:40 2019

@author: schroeder

Setup - Create a module for the CLOck NEtwork Simulator - CLONES
"""

import setuptools
from setuptools import find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

print(find_packages("clones"))
setuptools.setup(
    name="clones",
    author="Stefan Schroeder",
    author_email="schroeder@geod.uni-bonn.de",
    description="Python tools for optical clock simulation",
    long_description=long_description,
    packages=find_packages("./clones"),
    package_dir={"":"."},
    classifiers=["Programming Language :: Python :: 3",
        "Operating System :: POSIX :: Linux",
        "Topic :: Scientific/Engineering",
        "Development Status :: 1 - Planning"]
)
