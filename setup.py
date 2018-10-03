#!/usr/bin/env python
# encoding: utf-8

import setuptools
from setuptools import setup

with open("README.rst", "r", encoding = "utf8") as fh:
    long_description = fh.read()

setup(name='torch-salad',
      version='0.2.1-alpha',
      description='Semi-supervised Adaptive Learning Across Domains',
      long_description=long_description,
      long_description_content_type="text/x-rst",
      url='https://domainadaptation.org',
      author='Steffen Schneider',
      author_email='steffen.schneider@tum.de',
      packages=setuptools.find_packages(),
      classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)",
        "Operating System :: OS Independent",
        "Development Status :: 2 - Pre-Alpha",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
      ],
)
