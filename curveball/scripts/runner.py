#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of curveball.
# https://github.com/yoavram/curveball

# Licensed under the MIT license:
# http://www.opensource.org/licenses/MIT-license
# Copyright (c) 2015, Yoav Ram <yoav@yoavram.com>
import os.path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set_style("ticks")
import click
import curveball

@click.command()
@click.option('--folder', prompt=True, help='folder to process')
@click.option('--plate_folder', default='plate_templates', help='plate templates default folder')
@click.option('--plate_file', default='checkerboard.csv', help='plate templates csv file')
@click.option('--blank_strain', default='0', help='blank strain for background calibration')
@click.option('--ref_strain', default='1',  help='reference strain for competitions')
@click.option('--max_time', default=np.inf, help='omit data after max_time hours')
@click.option('-v/-V', '--verbose/--no-verbose', default=True)
def main(folder, plate_folder, plate_file, blank_strain, ref_strain, max_time, verbose):
	if verbose:		
		click.echo('=' * 40)
		click.echo('Curveball %s' % curveball.__version__)	
		click.echo('=' * 40)
		click.echo('- Processing %s.' % folder)
		plate_path = os.path.join(plate_folder, plate_file)
		click.echo('- Using plate template from %s.' % plate_path)
		click.echo('- Blank strain: %s; Reference strain: %s.' % (blank_strain, ref_strain))
		click.echo('- Omitting data after %.2f hours.' % max_time)
		click.echo('-' * 40)

if __name__ == '__main__':
    main()