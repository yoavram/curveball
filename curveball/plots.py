#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of curveball.
# https://github.com/yoavram/curveball

# Licensed under the MIT license:
# http://www.opensource.org/licenses/MIT-license
# Copyright (c) 2015, Yoav Ram <yoav@yoavram.com>
from __future__ import division
from builtins import range
from past.utils import old_div
import copy
import numpy as np
import scipy.stats
import matplotlib as mpl
import matplotlib.pyplot as plt
try:
	from pandas.plotting import lag_plot # py3.6
except ImportError:
	from pandas.tools.plotting import lag_plot # py3.5
import seaborn as sns
sns.set_style("ticks")

from matplotlib.patches import RegularPolygon
from string import ascii_uppercase


def plot_wells(df, x='Time', y='OD', plot_func=plt.plot, output_filename=None):
	"""Plot a grid of plots, one for each well in the plate.

	The facetting is done by the ``Row`` and ``Col`` columns of `df`.
	The colors are given by the ``Color`` column, 
	the labels of the colors are given by the ``Strain`` column.
	If ``Strain`` is missing then the coloring is done by the ``Well`` column.

	Parameters
	----------
	df : pandas.DataFrame
		growth curve data, see :py:mod:`curveball.ioutils` for a detailed definition.
	x : str, optional
		name of column for x-axis, defaults to ``Time``.
	y : str, optional
		name of column for y-axis, defaults to ``OD``.
	plot_func : func, optional
		function to use for plotting, defaults to :py:func:`matplotlib.pyplot.plot`
	output_filename : str, optional 
		filename to save the resulting figure; if not given, figure is not saved.

	Returns
	-------
	seaborn.FacetGrid
		figure object.
	"""
	if 'Strain' in df:
		hue = 'Strain'
		palette = df.Color.unique() if 'Color' in df else sns.color_palette()
		hue_order = df.Strain.unique()
		palette[palette == '#ffffff'] = '#000000'
	else:
		hue = 'Well'
		palette = sns.color_palette()
		hue_order = df.Well
	height = len(df.Row.unique())
	width = len(df.Col.unique())
	g = sns.FacetGrid(df, hue=hue, col='Col', row='Row',
                      palette=palette, hue_order=hue_order,
                      sharex=True, sharey=True, height=1,
                      aspect=old_div(width,float(height)), despine=True,margin_titles=True)
	g.map(plot_func, x, y)
	g.fig.set_figwidth(width)
	g.fig.set_figheight(height)
	plt.locator_params(nbins=4) # 4 ticks is enough
	g.set_axis_labels('','') 	# remove facets axis labels
	g.fig.text(0.5, 0, x, size='x-large') # xlabel
	g.fig.text(-0.01, 0.5, y, size='x-large', rotation='vertical') # ylabel
	if output_filename:
		g.savefig(output_filename, bbox_inches='tight', pad_inches=1)
	return g


def plot_strains(data, x='Time', y='OD', plot_func=plt.plot, by=None, agg_func=np.mean, hue='Strain', color=None, output_filename=None, **kwargs):
	"""Aggregate by strain and plot the results on one figure with different color for each strain.

	The grouping of the data is done by the ``Strain`` and either ``Cycle Nr.`` or ``Time`` columns of `data`;
	the aggregation is done by the `agg_func`, which defaults to :py:func:`numpy.mean`.
	The colors are given by the ``Color`` column, the labels of the colors are given by the ``Strain`` column of `data`.

	Parameters
	----------
	data : pandas.DataFrame
		growth curve data, see :py:mod:`curveball.ioutils` for a detailed definition.
	x : str, optional
		name of column for x-axis, defaults to ``Time``.
	y : str, optional
		name of column for y-axis, defaults to ``OD``.
	plot_func : func, optional
		function to use for plotting, defaults to :py:func:`matplotlib.pyplot.plot`
	by : tuple of str, optional
		used for grouping the data, defaults to ``('Strain', 'Cycle Nr.')`` or ``('Strain', 'Time')``, whichever is available.
	plot_func : func, optional
		function to use for aggregating the data, defaults to :py:func:`numpy.mean`.
	color : seaborn color palette
		a seaborn color palette to use if there is no ``Color`` column; if not given, using the default palette.
	output_filename : str, optional 
		filename to save the resulting figure; if not given, figure is not saved.

	Returns
	-------
	seaborn.FacetGrid
		figure object.

	Raises
	------
	ValueError
		raised if `by` isn't set and `data` doesn't contain ``Strain`` and either ``Time`` or ``Cycle Nr.``.
	"""
	if 'Color' in data:
		palette = data.Color.unique()
		palette[palette == '#ffffff'] = '#000000'
	else:
		palette = color or sns.color_palette()
	if by is None:
		if 'Cycle Nr.' in data and 'Strain' in data:
			by = ['Strain', 'Cycle Nr.']
		elif 'Time' in data and 'Strain' in data:
			by = ['Strain', 'Time']
		else:
			raise ValueError("If by is not set then data must have column Strain and either Time or Cycle Nr.")
	grp = data.groupby(by=by)
	agg = grp.aggregate(agg_func).reset_index()
	g = sns.FacetGrid(agg, hue=hue, height=5, aspect=1.5, palette=palette, hue_order=data[hue].unique())
	g.map(plot_func, x, y);
	g.add_legend()
	if output_filename:
		g.savefig(output_filename, bbox_inches='tight', pad_inches=1)
	return g


def tsplot(data, x='Time', y='OD', ci_level=95, ax=None, color=None, output_filename=None, **kwargs):
	"""Time series plot of the data by strain (if applicable) or well.

	The grouping of the data is done by the value of `x` and ``Strain``, if such a column exists in `data`; 
	otherwise it is done by `x` and ``Well``.
	The aggregation is done by :py:func:`seaborn.lineplot` which calculates the mean with a confidence interval.
	The colors are given by the ``Color`` column, the labels of the colors are given by the ``Strain`` column; 
	if ``Strain`` and ``Color`` don't exist in `data` then
	the function will use a default palette and color the lines by well.

	Parameters
	----------
	data : pandas.DataFrame
		growth curve data, see :py:mod:`curveball.ioutils` for a detailed definition.
	x : str, optional
		name of column for x-axis, defaults to ``Time``.
	y : str, optional
		name of column for y-axis, defaults to ``OD``.
	ci_level : int, optional
		confidence interval width in precent (0-100), defaults to 95.
	ax : matplotlib.axes.Axes, optional
		plot into this axes, if not given create a new figure.
	color : seaborn color palette
		a seaborn color palette to use if there is no ``Color`` column; if not given, using the default palette.
	output_filename : str, optional 
		filename to save the resulting figure; if not given, figure is not saved.

	Returns
	-------
	matplotlib.axes.Axes
		axes object
	"""
	if 'Strain' in data:
		condition = 'Strain'		
	else:
		condition = 'Well'
	if 'Color' in data:
		palette = data['Color'].unique()
		palette[palette == '#ffffff'] = '#000000'
	else: 
		palette = color or sns.color_palette()

	g = sns.lineplot(data=data, x=x, hue=condition, y=y,
					err_style='band', ci=ci_level, palette=list(palette), ax=ax)
	sns.despine()
	if output_filename:
		g.figure.savefig(output_filename, bbox_inches='tight', pad_inches=1)
	return g


def plot_plate(data, edge_color='#888888', output_filename=None):
	"""Plot of the plate color mapping.

	The function will plot the color mapping in `data`:
	a grid with enough columns and rows for the ``Col`` and ``Row`` columns in `data`,
	where the color of each grid cell given by the ``Color`` column.

	Parameters
	----------
	data : pandas.DataFrame
		growth curve data, see :py:mod:`curveball.ioutils` for a detailed definition.
	edge_color : str
		color hex string for the grid edges.

	Returns
	-------
	fig : matplotlib.figure.Figure
		figure object
	ax : numpy.ndarray
		array of axis objects.
	"""
	plate = data.pivot('Row', 'Col', 'Color').values
	height, width = plate.shape
	fig = plt.figure(figsize=((width + 2.0) / 3.0, (height + 2.0) / 3.0))
	ax = fig.add_axes((0.05, 0.05, 0.9, 0.9),
	                            aspect='equal', frameon=False,
	                            xlim=(-0.05, width + 0.05),
	                            ylim=(-0.05, height + 0.05))
	for axis in (ax.xaxis, ax.yaxis):
	    axis.set_major_formatter(plt.NullFormatter())
	    axis.set_major_locator(plt.NullLocator())

	# Create the grid of squares
	squares = np.array([[RegularPolygon((i + 0.5, j + 0.5),
	                                         numVertices=4,
	                                         radius=0.5 * np.sqrt(2),
	                                         orientation=old_div(np.pi, 4),
	                                         ec=edge_color,
	                                         fc=plate[height-1-j,i])
	                          for j in range(height)]
	                         for i in range(width)])
	[ax.add_patch(sq) for sq in squares.flat]
	ax.set_xticks(np.arange(width) + 0.5)
	ax.set_xticklabels(np.arange(1, 1 + width))
	ax.set_yticks(np.arange(height) + 0.5)
	ax.set_yticklabels(ascii_uppercase[height-1::-1])
	ax.xaxis.tick_top()
	ax.yaxis.tick_left()
	ax.tick_params(length=0, width=0)
	if output_filename:
		fig.savefig(output_filename, bbox_inches='tight', pad_inches=1)
	return fig, ax


def plot_params_distribution(param_samples, color='k', cmap="viridis", alpha=None):
	"""Plots a distribution of model parameter samples generated with :py:func:`curveball.models.sample_params`.

	Parameters
	----------
	param_samples : pandas.DataFrame
		data frame of samples; each row is one sample, each column is one parameter.
	alpha : float
		transparency of plot markers, defaults to :math:`1/n^{1/4}` where *n* is number of rows in `param_samples`.

	Returns
	-------
	seaborn.Grid
		figure object
	"""
	nsamples = param_samples.shape[0]
	g = sns.PairGrid(param_samples)
	if alpha is None:
		alpha = 1.0 / np.power(nsamples, 1.0 / 4.0)
	g.map_upper(plt.scatter, alpha=alpha, color=color)
	g.map_lower(sns.kdeplot, cmap=cmap, legend=False, shade=True, shade_lowest=False)
	g.map_diag(plt.hist, facecolor=color) # https://github.com/mwaskom/seaborn/pull/788
	return g


def _plot_fitted_histogram(data, rv=scipy.stats.norm, color='k', label=None, alpha=0.5, ax=None):
	"""This is basically `sns.distplot(fit=rv)`.
	TODO: `low,high = np.percentile(x, 2.5), np.percentile(x, 97.5)`
	"""
	if ax is None:
		fig, ax = plt.subplots(1, 1)
	else:
		fig = ax.figure
	rv_params = rv.fit(data)
	rv_inst = rv(*rv_params)	
	nbins = min(100, len(data))
	n, bins, patches = ax.hist(data, bins=nbins, color=color, alpha=alpha, density=True)
	ax.plot(bins, rv_inst.pdf(bins), color='k', lw=2)
	ax.annotate(
		r'$\mu={:.2g}, \sigma={:.2g}$'.format(rv_inst.mean(), 
		rv_inst.std()), 
		xy=(bins[len(bins)//2], np.max(n)), 
		xycoords="data", 
		horizontalalignment='center', 
		fontsize=plt.rcParams['axes.labelsize']
	)
	return fig, ax


def plot_model_residuals(model_fit, rv=scipy.stats.norm, color='k'):
	"""Plot of the residuals of a model fit.

	The function will plot the residuals - the difference between data and model - for a given model fit.
	The left panel shows the residuals over time; the right panel shows the histogram of the residuals with a fitted distribution curve.

	Parameters
	----------
	model_fit : lmfit.ModelResult
		the result of a model fitting procedure.
	rv : scipy.stats.rv_continuous, optional
		:py:class:`scipy.stats.rv_continuous` random variable whose probability density function (pdf)
		will be fitted to the histogram. Defaults to the normal distribution (`scipy.stats.norm`).
	color : str, optional
		color string for the plot, defaults to `k` for black.

	Returns
	-------
	fig : matplotlib.figure.Figure
		figure object
	ax : numpy.ndarray
		array of axis objects.
	"""
	w, h= plt.rcParams['figure.figsize']
	fig,ax = plt.subplots(1, 2, figsize=(w * 2, h))

	model_fit.plot_residuals(ax=ax[0], data_kws={'color': color}) # removed, causes bug in lmfit: fit_kws={'color': color})
	ax[0].set_xlabel('Time (hr)')
	ax[0].set_ylabel('Residuals')
	ax[0].legend().set_visible(False)
	ax[0].set_title('')

	_plot_fitted_histogram(model_fit.residual, rv=rv, color=color, ax=ax[1])
	ax[1].set(xlabel='Residuals', ylabel='Frequency')

	fig.tight_layout()
	sns.despine()	
	return fig, ax


def plot_residuals(df, time='Time', value='OD', resid_func=lambda x: x - x.mean(), rv=scipy.stats.norm, 
	color='k', ax=None):
	"""Plot of the residuals of in the data.

	The function will plot the residuals - the difference between data and average at each time point.
	The left panel shows the residuals over time.
	The middle panel shows the histogram of the residuals with a fitted distribution (defaults to Gaussian).
	The right panel shows the regression between the standard deviation at time `t+1` and `t` to identify autocorrelation.


	Parameters
	----------
	df : pandas.DataFrame
		a data frame with columns ``Time`` and ``OD``.
	time : str, optional
		name of column over which to group and plot the residuals. Defaults to ``Time``.
	value : str, optional
		name of column in `df` of the value on which to compute the residuals. Defaults to ``OD``.
	resid_func : function, optional
		function to calculate residuals. Defaults to ``x - x.mean()``.
	rv : scipy.stats.rv_continuous, optional
		:py:class:`scipy.stats.rv_continuous` random variable whose probability density function (pdf)
		will be fitted to the histogram. Defaults the normal distribution (:py:class:`scipy.stats.norm`).
	color : str, optional
		color string for the plot, defaults to `k` for black.

	Returns
	-------
	fig : matplotlib.figure.Figure
		figure object
	ax : numpy.ndarray
		array of axis objects.
	"""
	w, h= plt.rcParams['figure.figsize']
	fig,ax = plt.subplots(1, 3, figsize=(w * 3, h))

	residuals = df.groupby(time)[value].transform(resid_func).values

	ax[0].plot(df[time], residuals, ls='', marker='o', color=color)
	ax[0].set(xlabel=time, ylabel='Residuals')	
	
	_plot_fitted_histogram(residuals, rv=rv, color=color, ax=ax[1])
	ax[1].set(xlabel='Residuals', ylabel='Frequency')
	
	sigmas = df.groupby(time)[value].std()	
	linreg = scipy.stats.linregress(sigmas.values[:-1], sigmas.values[1:])
	eq = r'$\sigma_{{t+1}} = {:.2g} + {:.2g} \sigma_{{t}}$'.format(linreg.intercept, linreg.slope)
	sigma_range = np.linspace(sigmas.min(), sigmas.max())
	ax[2].plot(sigma_range, sigma_range, color='k', ls='--', label=r'$\sigma_{t+1}=\sigma_{t}$')
	ax[2].plot(sigma_range, linreg.intercept + linreg.slope * sigma_range, color=color, label=eq)
	lag_plot(sigmas, c='k', ax=ax[2])
	ax[2].set(xlabel=r'$\sigma_{t}$', ylabel=r'$\sigma_{t+1}$')
	ax[2].legend(loc='upper left')

	fig.tight_layout()
	sns.despine()	
	return fig, ax


def plot_sample_fit(model_fit, param_samples, fit_kws=None, data_kws=None, sample_kws=None):
	"""Plot of sampled curve fits.

	The function will plot the main model fit and the sampled curve fits based on a table of sample parameters.


	Parameters
	----------
	model_fit : lmfit.ModelResult
		the result of a model fitting procedure.
	param_samples : pandas.DataFrame
		data frame of samples; each row is one sample, each column is one parameter.
	fit_kws, data_kws, sample_kws : dict
		dictionaries of plot directives for the fit, data, and sampled fit curves.

	Returns
	-------
	fig : matplotlib.figure.Figure
		figure object
	ax : numpy.ndarray
		array of axis objects.
	"""
	t = np.linspace(0, model_fit.userkws['t'].max())
	def f(params):
		return model_fit.model.eval(t=t, params=params)

	nsamples = param_samples.shape[0]
	_fit_kws = dict(linewidth=5)
	if fit_kws: _fit_kws.update(fit_kws)
	_data_kws = dict(marker='.')
	if data_kws: _data_kws.update(data_kws)
	_sample_kws = dict(linestyle='--', color='gray', alpha=1/np.sqrt(nsamples))
	if sample_kws: _sample_kws.update(sample_kws)

	ax = model_fit.plot_fit(init_kws={'ls': ''}, fit_kws=_fit_kws, data_kws=_data_kws)
	for i in range(nsamples):
		sample = param_samples.iloc[i, :]
		params = model_fit.params.copy()
		for k, v in params.items():
			if v.vary:
				params[k].set(value=sample[k])
		plt.plot(t, f(params), **_sample_kws)
	ax.legend().set_visible(False)
	ax.set(ylabel='OD', xlabel='Time', title='')

	sns.despine()
	return ax.figure, ax