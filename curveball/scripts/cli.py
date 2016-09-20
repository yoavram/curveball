#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of curveball.
# https://github.com/yoavram/curveball

# Licensed under the MIT license:
# http://www.opensource.org/licenses/MIT-license
# Copyright (c) 2015, Yoav Ram <yoav@yoavram.com>
from builtins import map
import sys
import os.path
import pkg_resources
import glob
import warnings
# catch some future warnings, mostly caused by matplotlib
warnings.simplefilter(action="ignore", category=FutureWarning)
import curveball
import numpy as np
import pandas as pd
import click
import xlrd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("ticks")


VERBOSE = False
PLOT = True
PROMPT = True
ERROR_COLOR = 'red'
INFO_COLOR = 'white'
file_extension_handlers = {
	'.mat': curveball.ioutils.read_tecan_mat, 
	'.xlsx': curveball.ioutils.read_tecan_xlsx,
	'.csv': curveball.ioutils.read_curveball_csv,
}


def echo_error(message):
	click.secho("Error: %s" % message, fg=ERROR_COLOR)


def echo_info(message):
	if VERBOSE:
		click.secho(message, fg=INFO_COLOR)


def ioerror_to_click_exception(io_error):
	raise click.FileError(io_error.filename, hint=io_error.message)


def to_dict(ctx, param, value):
    return dict(value)


def to_set(ctx, param, value):
    return set(value)


def get_filename(filepath):
	"""Get a file name out of a file path.

	Parameters
	----------
	filepath : str

	Returns
	-------
	str
		filename
	"""
	if filepath is None:
		return ''
	filename = os.path.split(filepath)[-1]
	if filename is None:
		return ''
	return filename


def find_plate_file(plate_folder, plate_file):
	"""Finds a plate file, either in the current working dir or in the package data resources.

	Parameters
	----------
	plate_file : str
		the filename of the plate file, may include absolute or relative path

	Returns
	-------
	str
		the full path of the plate file.
	"""
	plate_path = os.path.join(plate_folder, plate_file)
	if not os.path.exists(plate_path):
		# if plate path doesn't exist try to get it from package data
		plate_path = pkg_resources.resource_filename(plate_folder, plate_file)
	if not os.path.exists(plate_path):
		raise click.FileError(plate_path, hint="can't find file.")
	return plate_path


def load_plate(plate_path):
	"""Loads a plate template from a CSV file.

	Parameters
	----------
	plate_path : str
		full or relative path to the plate template file.

	Returns
	-------
	pandas.DataFrame
		the plate template in tidy data format (see :py:mod:`.ioutils`).

	See also
	--------
	find_plate_file
	"""	
	try:
		plate = pd.read_csv(plate_path)
	except IOError as e:
		ioerror_to_click_exception(e)
	except pd.parser.CParserError as e:
		raise click.FileError(plate_path, hint="parser error, probably not a CSV file, {0}".format(e.args[0]))
	return plate


def where(ctx, param, value):
	"""Prints the path where Curveball is installed and exits. 

	Parameters are generally ignored; 
	if `value` or `ctx.resilient_parsing` are not empty/:const:`False`/:const:`None`, 
	the function returns without doing anything.
	"""
	if not value or ctx.resilient_parsing:
		return
	path = curveball.__file__
	folder = os.path.split(path)[0]
	click.secho(click.format_filename(folder))
	ctx.exit()


@click.group()
@click.option('-v/-V', '--verbose/--no-verbose', default=False)
@click.option('-l/-L', '--plot/--no-plot', default=True)
@click.option('-p/-P', '--prompt/--no-prompt', default=False)
@click.option('--where', is_flag=True, default=False, is_eager=True, callback=where, help='prints the path where Curveball is installed')
@click.version_option(version=curveball.__version__, prog_name=curveball.__name__)
def cli(verbose, plot, prompt, where):
	"""Main entry point to curveball

	To get help for the parameters, run:

	>>> curveball --help
	"""
	global VERBOSE
	VERBOSE = verbose
	global PLOT
	PLOT = plot
	global PROMPT
	PROMPT = prompt
	if VERBOSE:
		click.secho('=' * 40, fg='cyan')
		click.secho('Curveball %s' % curveball.__version__, fg='cyan')
		click.secho('=' * 40, fg='cyan')


@click.option('--plate_folder', default='plate_templates', help='plate templates default folder', type=click.Path())
@click.option('--plate_file', default='checkerboard.csv', help='plate templates csv file')
@click.option('-o', '--output_file', default='-', help='output file path', type=click.File(mode='w', lazy=True))
@click.option('--list', is_flag=True, default=False, help='list plate templates in the default folder')
@click.option('--show', is_flag=True, default=False, help='display the plate template as an image')
@cli.command()
def plate(plate_folder, plate_file, output_file, list, show):
	"""Read and print a plate template from a plate template CSV file.

	To get help for the parameters, run:

	>>> curveball plate --help
	"""
	if list:
		files = pkg_resources.resource_listdir('plate_templates', '')
		files = [fn for fn in files if os.path.splitext(fn)[-1].lower() == '.csv']
		files = os.linesep.join(files)
		click.echo(files)
		return
	plate_path = find_plate_file(plate_folder, plate_file)
	plate = load_plate(plate_path)
	if show:
		fig, ax = curveball.plots.plot_plate(plate)
		if output_file.name == '-':
			plt.show()
		else:
			fig.savefig(output_file.name)
	else:
		plate.to_csv(output_file, index=False)
	if output_file.name != '-':
		echo_info("Wrote output to {0}".format(click.format_filename(output_file.name)))


@click.argument('path', type=click.Path(exists=True, readable=True))
@click.option('--plate_folder', default='plate_templates', help='plate templates default folder', type=click.Path())
@click.option('--plate_file', default='checkerboard.csv', help='plate templates csv file')
@click.option('-o', '--output_file', default='-', help='output csv file path', type=click.File(mode='w', lazy=True))
@click.option('--blank_strain', default='0', type=str, help='blank strain for background calibration')
@click.option('--ref_strain', default='1',  type=str, help='reference strain for competitions')
@click.option('--max_time', default=np.inf, help='omit data after max_time hours')
@click.option('--guess', type=(str, float), multiple=True, callback=to_dict, help='set the initial guess for a parameter')
@click.option('--param_min', type=(str, float), multiple=True, callback=to_dict, help='set the minimum allowed value for a parameter')
@click.option('--param_max', type=(str, float), multiple=True, callback=to_dict, help='set the maximum allowed value for a parameter')
@click.option('--param_fix', type=str, multiple=True, callback=to_set, help='fix a parameter to it\'s initial guess')
@click.option('--weights/--no-weights', default=False, help="use weights for the fitting procedure")
@click.option('--ci/--no-ci', default=False, help="find confidence intervals for lag and max growth rate")
@click.option('--nsamples', default=1000, help="number of bootstrap samples to use, only applicable when using --ci")
@cli.command()
def analyse(path, output_file, plate_folder, plate_file, blank_strain, ref_strain, max_time, guess, param_min, param_max, param_fix, weights, ci, nsamples):
	"""Analyse growth curves data using Curveball.

	To get help for the parameters, run:

	>>> curveball plate --help
	"""
	results = []
	plate_path = find_plate_file(plate_folder, plate_file)

	if VERBOSE:
		click.echo('- Processing %s' % click.format_filename(path))		
		click.echo('- Using plate template from %s' % click.format_filename(plate_path))
		click.echo('- Blank strain: %s; Reference strain: %s' % (blank_strain, ref_strain))
		click.echo('- Omitting data after %.2f hours' % max_time)
		click.echo('-' * 40)
	
	plate = load_plate(plate_path)
	plate.Strain = list(map(str, plate.Strain))
	plate_strains = plate.Strain.unique().tolist()	
	if PROMPT:
		fig,ax = curveball.plots.plot_plate(plate)
		fig.show()
		click.echo("Plate with %d strains: %s" % (len(plate_strains), ', '.join(plate_strains)))
		click.confirm('Is this the plate you wanted?', default=False, abort=True, show_default=True)
	if os.path.isdir(path):
		files = glob.glob(os.path.join(path, '*'))
		#files = [os.path.join(path, fn) for fn in files]
	else:
		files = glob.glob(path)
	
	files = [fn for fn in files if os.path.splitext(fn)[-1].lower() in file_extension_handlers.keys()]
	if not files:
		raise click.ClickException("No data files found in folder {0}".format(click.format_filename(path)))
	
	with click.progressbar(files, label='Processing files:', item_show_func=get_filename, color='green') as bar:
		for filepath in bar:
			file_results = _process_file(filepath, plate, blank_strain, ref_strain, max_time, guess, param_min, param_max, param_fix, weights, ci, nsamples)
			results.extend(file_results)
	
	output_table = pd.DataFrame(results)
	output_table.to_csv(output_file, index=False)
	if VERBOSE and output_file.name != '-':
		click.secho("Wrote output to %s" % output_file.name, fg='green')


def _process_file(filepath, plate, blank_strain, ref_strain, max_time, guess, param_min, param_max, param_fix, weights, ci, nsamples):
	"""Analyses a single growth curves file.

	See also
	--------
	analyse
	"""
	results = []	
	fn, ext = os.path.splitext(filepath)
	echo_info("\tHandler: {1}\n".format(filepath, ext))
	handler = file_extension_handlers.get(ext)
	if handler is None:
		echo_info("No handler found for file {0}".format(click.format_filename(filepath)))
		return results
	try: 
		if np.isfinite(max_time):			
			df = handler(filepath, plate=plate, max_time=max_time)
		else:
			df = handler(filepath, plate=plate)
	except IOError as e:
		ioerror_to_click_exception(e)
	except xlrd.biffh.XLRDError as e:
		raise click.FileError(filepath, hint="parser error, probably not a {1} file, {0}".format(e.args[0], ext))

	strains = plate.Strain.unique().tolist()

	if blank_strain is not None and blank_strain != 'none': 
		if blank_strain in strains:
			bg = df[(df.Strain == blank_strain) & (df.Time == df.Time.min())]
			bg = bg.OD.mean()
			df.OD -= bg
			df.loc[df.OD < 0, 'OD'] = 0
		else:
			echo_error("Warning! Blank strain '%s' doesn't exist" % blank_strain)

	if PLOT:
		wells_plot_fn = fn + '_wells.png'
		g = curveball.plots.plot_wells(df, output_filename=wells_plot_fn)
		echo_info("Wrote wells plot to %s" % click.format_filename(wells_plot_fn))

		strains_plot_fn = fn + '_strains.png'
		g = curveball.plots.plot_strains(df, output_filename=strains_plot_fn)
		echo_info("Wrote strains plot to %s" % click.format_filename(strains_plot_fn))
	
	if blank_strain in strains: 
		strains.remove(blank_strain)
	if ref_strain in strains:
		strains.remove(ref_strain)
		strains.insert(0, ref_strain)
	else:
		echo_error("Warning, reference strains '%s' doesn't exist!" % ref_strain)

	for strain in strains:
		strain_df = df[df.Strain == strain]
		_ = curveball.models.fit_model(strain_df, param_guess=guess, param_min=param_min, param_max=param_max, param_fix=param_fix, use_weights=weights, PLOT=PLOT, PRINT=VERBOSE)
		if PLOT:
			fit_results,fig,ax = _
			strain_plot_fn = fn + ('_strain_%s.png' % strain)
			fig.savefig(strain_plot_fn)
			echo_info("Wrote strain %s plot to %s" % (strain, click.format_filename(strain_plot_fn)))
		else:
			fit_results = _

		res = {}
		fit = fit_results[0]
		res['folder'] = os.path.dirname(filepath)
		res['filename'] = os.path.splitext(os.path.basename(fn))[0]
		res['strain'] = strain
		res['model'] = fit.model.name
		res['RSS'] = fit.chisqr
		res['RMSD'] = np.sqrt(res['RSS'] / fit.ndata)
		res['NRMSD'] = res['RMSD'] / (strain_df.OD.max() - strain_df.OD.min())
		res['CV(RMSD)'] = res['RMSD'] / (strain_df.OD.mean())
		res['bic'] = fit.bic
		res['aic'] = fit.aic
		res['weighted_bic'] = fit.weighted_bic
		res['weighted_aic'] = fit.weighted_aic
		params = fit.params
		res['y0'] = params['y0'].value
		res['K'] = params['K'].value
		res['r'] = params['r'].value
		res['nu'] = params['nu'].value if 'nu' in params else 1
		res['q0'] = params['q0'].value if 'q0' in params else 0
		res['v'] = params['v'].value if 'v' in params else 0
		res['has_lag'] = curveball.models.has_lag(fit_results)
		res['has_nu'] = curveball.models.has_nu(fit_results, PRINT=VERBOSE)
		res['max_growth_rate'] = curveball.models.find_max_growth(fit)[-1]
		res['min_doubling_time'] = curveball.models.find_min_doubling_time(fit)
		res['lag'] = curveball.models.find_lag(fit)
		if ci:
			param_samples = curveball.models.bootstrap_params(strain_df, fit, nsamples=nsamples)
			_, _, low, high = curveball.models.find_max_growth_ci(fit, param_samples)
			res['max_growth_rate_low'] = low
			res['max_growth_rate_high'] = high			
			low, high = curveball.models.find_lag_ci(fit, param_samples)
			res['lag_low'] = low
			res['lag_high'] = high			
			low, high = curveball.models.find_min_doubling_time_ci(fit, param_samples)
			res['min_doubling_time_low'] = low
			res['min_doubling_time_high'] = high
			low, high = curveball.models.find_K_ci(param_samples)
			res['K_low'] = low
			res['K_high'] = high

		if strain == ref_strain:
			ref_fit = fit
			res['w'] = 1
		elif ref_strain in strains:
			colors = plate[plate.Strain.isin([strain, ref_strain])].Color.unique()
			_ = curveball.competitions.compete(fit, ref_fit, hours=df.Time.max(), colors=colors, PLOT=PLOT)
			if PLOT:
				t,y,fig,ax = _
				competition_plot_fn = fn + ('_%s_vs_%s.png' % (strain, ref_strain))
				fig.savefig(competition_plot_fn)
				echo_info("Wrote competition %s vs %s plot to %s" % (strain, ref_strain, click.format_filename(strain_plot_fn)))
			else:
				t,y = _
			res['w'] = curveball.competitions.fitness_LTEE(y, assay_strain=0, ref_strain=1)
			# TODO CI for w
		results.append(res)
		plt.clf()
	return results


if __name__ == '__main__':
    cli()