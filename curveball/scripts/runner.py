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


VERBOSE = False
PLOT = True
PROMPT = True
ERROR_COLOR = 'red'
INFO_COLOR = 'white'
file_extension_handlers = {
							'.mat': curveball.ioutils.read_tecan_mat, 
							'.xlsx': curveball.ioutils.read_tecan_xlsx
						  }


def echo_error(message):
	click.secho("Error: %s" % message, fg=ERROR_COLOR)


def echo_info(message):
	if VERBOSE:
		click.secho(message, fg=INFO_COLOR)


def VERBOSE_version(ctx, param, value):
    if not value or ctx.resilient_parsing:
        return
    click.echo(curveball.__version__)
    ctx.exit()


@click.group()
@click.option('-v/-V', '--verbose/--no-verbose', default=False)
@click.option('-l/-L', '--plot/--no-plot', default=True)
@click.option('-p/-P', '--prompt/--no-prompt', default=False)
@click.option('--version', is_flag=True, callback=VERBOSE_version, expose_value=False, is_eager=True)
def cli(verbose, plot, prompt):
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


@click.argument('path', type=click.Path(exists=True, readable=True))
@click.option('--plate_folder', default='plate_templates', help='plate templates default folder', type=click.Path(exists=True, dir_okay=True, readable=True))
@click.option('--plate_file', default='checkerboard.csv', help='plate templates csv file')
@click.option('-o', '--output_file', default='-', help='output csv file path', type=click.File(mode='w', lazy=True))
@click.option('--blank_strain', default='0', help='blank strain for background calibration')
@click.option('--ref_strain', default='1',  help='reference strain for competitions')
@click.option('--max_time', default=np.inf, help='omit data after max_time hours')
@cli.command()
def analyse(path, output_file, plate_folder, plate_file, blank_strain, ref_strain, max_time):
	"""Analyse growth curves using Curveball.
	Outputs estimated growth traits and fitness of all strains in all files in folder PATH or matching the pattern PATH.
	"""
	results = []
	plate_path = os.path.join(plate_folder, plate_file)	

	if VERBOSE:
		click.echo('- Processing %s' % click.format_filename(path))		
		click.echo('- Using plate template from %s' % click.format_filename(plate_path))
		click.echo('- Blank strain: %s; Reference strain: %s' % (blank_strain, ref_strain))
		click.echo('- Omitting data after %.2f hours' % max_time)
		click.echo('-' * 40)

	try:
		plate = pd.read_csv(plate_path)
	except IOError as e:
		echo_error('Failed reading plate file, %s' % e.message)
		return results
	
	plate.Strain = map(unicode, plate.Strain)
	plate_strains = plate.Strain.unique().tolist()	
	if PROMPT:
		fig,ax = curveball.plots.plot_plate(plate)
		fig.show()
		click.echo("Plate with %d strains: %s" % (len(plate_strains), ', '.join(plate_strains)))
		click.confirm('Is this the plate you wanted?', default=False, abort=True, show_default=True)

	if os.path.isdir(path):
		files = glob.glob(os.path.join(path, '*'))
		files = map(lambda fn: os.path.join(path, fn), files)
	else:
		files = glob.glob(path)

	files = filter(lambda fn: os.path.splitext(fn)[-1].lower() in file_extension_handlers.keys(), files)
	if not files:
		echo_error("No files to analyze found in %s" % click.format_filename(path))
		return results

	with click.progressbar(files, label='Processing files:') as bar:
		for filepath in bar:
			file_results = process_file(filepath, plate, blank_strain, ref_strain, max_time)
			results.extend(file_results)
	
	output_table = pd.DataFrame(results)
	output_table.to_csv(output_file, index=False)
	click.secho("Wrote output to %s" % output_file.name, fg='green')


def process_file(filepath, plate, blank_strain, ref_strain, max_time):
	results = []	
	fn,ext = os.path.splitext(filepath)
	echo_info('Extension: %s' % ext)
	handler = file_extension_handlers.get(ext)
	echo_info('Handler: %s' % handler.__name__)
	if  handler == None:
		echo_info("No handler")
		return results
	try: 
		if np.isfinite(max_time):
			df = handler(filepath, plate=plate, max_time=max_time)
		else:
			df = handler(filepath, plate=plate)
	except IOError as e:
		echo_error('Failed reading data file, %s' % e.message)
		return results

	if not blank_strain == None:
		bg = df[(df.Strain == blank_strain) & (df.Time == df.Time.min())]		
		bg = bg.OD.mean()
		df.OD -= bg
		df.loc[df.OD < 0, 'OD'] = 0		

	if PLOT:
		wells_plot_fn = fn + '_wells.png'
		curveball.plots.plot_wells(df, output_filename=wells_plot_fn)
		echo_info("Wrote wells plot to %s" % click.format_filename(wells_plot_fn))

		strains_plot_fn = fn + '_strains.png'
		curveball.plots.plot_strains(df, output_filename=strains_plot_fn)
		echo_info("Wrote strains plot to %s" % click.format_filename(strains_plot_fn))

	strains = plate.Strain.unique().tolist()
	strains.remove(blank_strain)
	strains.remove(ref_strain)
	strains.insert(0, ref_strain)	

	with click.progressbar(strains, label='Fitting strain growth curves') as bar:
		for strain in bar:
			strain_df = df[df.Strain == strain]
			_ = curveball.models.fit_model(strain_df, PLOT=PLOT, PRINT=VERBOSE)
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
			res['bic'] = fit.bic
			res['aic'] = fit.aic
			res['benchmark'] = curveball.models.benchmark(fit_results)
			params = fit.params
			res['y0'] = params['y0'].value
			res['K'] = params['K'].value
			res['r'] = params['r'].value
			res['nu'] = params['nu'].value if 'nu' in params else 1
			res['q0'] = params['q0'].value if 'q0' in params else 0
			res['v'] = params['v'].value if 'v' in params else 0
			res['max_growth_rate'] = curveball.models.find_max_growth(fit, PLOT=False)[-1]
			res['lag'] = curveball.models.find_lag(fit, PLOT=False)
			res['has_lag'] = curveball.models.has_lag(fit_results)
			res['has_nu'] = curveball.models.has_nu(fit_results, PRINT=VERBOSE)			

			if strain == ref_strain:
				ref_fit = fit
				res['w'] = 1
			else:
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

			results.append(res)
	return results


if __name__ == '__main__':
    cli()