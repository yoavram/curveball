#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of curveball.
# https://github.com/yoavram/curveball

# Licensed under the MIT license:
# http://www.opensource.org/licenses/MIT-license
# Copyright (c) 2015, Yoav Ram <yoav@yoavram.com>

from setuptools import setup, find_packages
from curveball import __version__

tests_require = [
    'mock',
    'nose',
    'coverage',
    'yanc',
    'tox',
    'ipdb',
    'coveralls',
    'sphinx',
]

setup(
    name='curveball',
    version=__version__,
    description='model microbial growth curves',
    long_description='''Curveball uses ecological and evolutionary models to analyze microbial growth curves''',
    keywords='microbiology biomath evolution',
    author='Yoav Ram',
    author_email='yoav@yoavram.com',
    url='https://github.com/yoavram/curveball',
    license='MIT',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Operating System :: Unix',
        'Programming Language :: Python :: 2.7',
        #'Programming Language :: Python :: 3.4',
        'Operating System :: OS Independent',
    ],
    packages=find_packages(),
    include_package_data=True,
    data_files=[('data', ['data/Tecan_280715.xlsx']),
                  ('plate_templates', ['plate_templates/checkerboard.csv', 'plate_templates/G-RG-R.csv']),
    ],
    install_requires=[
        # add your dependencies here
        # remember to use 'package-name>=x.y.z,<x.y+1.0' notation (this way you get bugfixes)
        'Click',
    ],
    extras_require={
        'tests': tests_require,
    },
    entry_points={
        'console_scripts': [
            # add cli scripts here in this form:
            'curveball=curveball.scripts.runner:cli',
        ],
    },
)
