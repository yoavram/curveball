#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of curveball.
# https://github.com/yoavram/curveball

# Licensed under the MIT license:
# http://www.opensource.org/licenses/MIT-license
# Copyright (c) 2015, Yoav Ram <yoav@yoavram.com>

from setuptools import setup, find_packages
import os
import versioneer

with open('README.md') as f:
    README = f.read()

setup(
    name='curveball',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description='Predicting competition results from growth curves',
    long_description=README,
    long_description_content_type='text/markdown',
    keywords='microbiology biomath evolution',
    author='Yoav Ram',
    author_email='yoav@yoavram.com',
    url='https://github.com/yoavram/curveball',
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: End Users/Desktop',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',                
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Operating System :: OS Independent',
    ],
    packages=find_packages(),
    include_package_data=True,
    package_data={
        'data': ['*'],
        'plate_templates': ['*'],
    },
    data_files=[('data', ['data/Tecan_280715.xlsx', 'data/Tecan_210115.xlsx', 'data/Tecan_210115.csv', 'data/20130211_dh.zip', 'data/plate_9_OD.mat', 'data/Sunrise_180515_0916.xlsx', 'data/BioTekSynergy.xlsx']),
                  ('plate_templates', ['plate_templates/checkerboard.csv', 'plate_templates/G-RG-R.csv']),
    ],
    install_requires=[
        # remember to use 'package-name>=x.y.z,<x.y+1.0' notation (this way you get bugfixes)
        'future>=0.18.0',
        'click>=7.0',
        'lxml>=4.7.0',
        'xlrd>=1.2.0,<2.0',
        'numpy>=1.20.0',
        'scipy>=1.6.0',
        'matplotlib>=3.2.0',
        'pandas>=1.3.0',
        'seaborn>=0.11.0',
        'scikit-learn>=0.24.0',
        'sympy>=1.9.0',
        'lmfit>=1.0.0',
        'webcolors>=1.7',
        'python-dateutil>=2.8.0'
    ],
    extras_require={
        'tests': [
            'pytest',
            'pytest-cov',
            'pillow'
        ],
        'docs': [
            'sphinx>=1.3.0',
            'numpydoc',
            'sphinx_rtd_theme'
        ]
    },
    entry_points={
        'console_scripts': [
            # add cli scripts here in this form:
            'curveball=curveball.scripts.cli:cli',
        ],
    },
)
