#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of curveball.
# https://github.com/yoavram/curveball

# Licensed under the MIT license:
# http://www.opensource.org/licenses/MIT-license
# Copyright (c) 2015, Yoav Ram <yoav@yoavram.com>import pandas as pd
import curveball
import pandas as pd
import warnings
warnings.simplefilter('ignore', FutureWarning)

if __name__ == '__main__':
	plate = pd.read_csv('plate_templates/G-RG-R.csv')
	df = curveball.ioutils.read_tecan_xlsx('data/Tecan_280715.xlsx', label='OD', plate=plate)
	models, fig, ax = curveball.models.fit_model(df[df.Strain == 'G'], PLOT=True, PRINT=False)
	fig.savefig('docs/_static/example_model_fitting.svg')
