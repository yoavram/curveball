#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of curveball.
# https://github.com/yoavram/curveball

# Licensed under the MIT license:
# http://www.opensource.org/licenses/MIT-license
# Copyright (c) 2015, Yoav Ram <yoav@yoavram.com>
import os.path
import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set_style("ticks")
import click
import curveball

PLOT = True
ERROR_COLOR = 'red'
file_extension_handlers = {'.mat': curveball.ioutils.read_tecan_mat}


def echo_error(message):
	click.secho("Error: %s" % message, fg=ERROR_COLOR)


def process_file(filepath, plate, blank_strain, ref_strain, max_time):
	click.echo('Filename: %s.' % click.format_filename(filepath))
	fn,ext = os.path.splitext(filepath)
	click.echo('Extension: %s.' % ext)
	handler = file_extension_handlers.get(ext)
	click.echo('Handler: %s.' % handler)
	if  handler == None:
		click.echo("No handler.")
		return

	try: 
		df = handler(filepath, plate=plate, max_time=max_time)
	except IOError as e:
		echo_error('Failed reading data file, %s' % e.message)
		return False

	if PLOT:
		wells_plot_fn = fn + '_wells.png'
		curveball.plots.plot_wells(df, output_filename=wells_plot_fn)
		click.echo("Wrote wells plot to %s" % wells_plot_fn)

		strains_plot_fn = fn + '_strains.png'
		curveball.plots.plot_strains(df, output_filename=strains_plot_fn)
		click.echo("Wrote strains plot to %s" % strains_plot_fn)

	for strain in df.Strain.unique():
		strain_df = df[df.Strain == strain]
		res = curveball.models.fit_model(strain_df, PLOT=PLOT, PRINT=True)
		if PLOT:
			models,fig,ax = res
			strain_plot_fn = fn + ('_strain_%s.png' % strain)
			fig.savefig(strain_plot_fn)
			click.echo("Wrote strain %s plot to %s" % (strain, strain_plot_fn))
		else:
			models = res

		# IM HERE!



	return True


def process_folder(folder, plate_path, blank_strain, ref_strain, max_time):
	try:
		plate = pd.read_csv(plate_path)
	except IOError as e:
		echo_error('Failed reading plate file, %s' % e.message)
		return False
	fig,ax = curveball.plots.plot_plate(plate)
	fig.show()
	click.confirm('Is this the plate you wanted?', default=False, abort=True, show_default=True)

	files = glob.glob(os.path.join(folder, '*'))
	if not files:
		echo_error("No files found in folder %s" % folder)
		return False

	for fn in files:
		filepath = os.path.join(folder, fn)
		process_file(filepath, plate, blank_strain, ref_strain, max_time)

	return True


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
		click.secho('=' * 40, fg='cyan')
		click.secho('Curveball %s' % curveball.__version__, fg='cyan')	
		click.secho('=' * 40, fg='cyan')
		click.echo('- Processing %s.' % click.format_filename(folder))
		plate_path = os.path.join(plate_folder, plate_file)
		click.echo('- Using plate template from %s.' % click.format_filename(plate_path))
		click.echo('- Blank strain: %s; Reference strain: %s.' % (blank_strain, ref_strain))
		click.echo('- Omitting data after %.2f hours.' % max_time)
		click.echo('-' * 40)

	process_folder(folder, plate_path, blank_strain, ref_strain, max_time)


if __name__ == '__main__':
    main()