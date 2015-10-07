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
	df = curveball.ioutils.read_tecan_xlsx('data/Tecan_280715.xlsx', label='OD', plate=plate)
	green = curveball.models.fit_model(df[df.Strain == 'G'], PLOT=False, PRINT=False)[0]
	red = curveball.models.fit_model(df[df.Strain == 'R'], PLOT=False, PRINT=False)[0]
	t, y, fig, ax = curveball.competitions.compete(green, red, PLOT=True, colors=['green', 'red'])
	fig.savefig('docs/_static/example_competition.svg')