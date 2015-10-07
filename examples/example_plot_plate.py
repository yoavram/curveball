#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of curveball.
# https://github.com/yoavram/curveball

# Licensed under the MIT license:
# http://www.opensource.org/licenses/MIT-license
# Copyright (c) 2015, Yoav Ram <yoav@yoavram.com>import pandas as pd
import curveball

if __name__ == '__main__':
	plate = pd.read_csv('plate_templates/G-RG-R.csv')
	fig, ax = curveball.plots.plot_plate(plate)	
	fig.savefig('docs/_static/example_plot_plate.svg')