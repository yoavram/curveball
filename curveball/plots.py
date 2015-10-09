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
import numpy as np
import matplotlib.pyplot as plt
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
                      sharex=True, sharey=True, size=1,
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


def plot_strains(df, x='Time', y='OD', plot_func=plt.plot, by=None, agg_func=np.mean, hue='Strain', output_filename=None):
	"""Aggregate by strain and plot the results on one figure with different color for each strain.

	The grouping of the data is done by the ``Strain`` and either ``Cycle Nr.`` or ``Time`` columns of `df`; 
	the aggregation is done by the `agg_func`, which defaults to :py:func:`numpy.mean`.
	The colors are given by the ``Color`` column, the labels of the colors are given by the ``Strain`` column of `df`.

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
	by : tuple of str, optional
		used for grouping the data, defaults to ``('Strain', 'Cycle Nr.')`` or ``('Strain', 'Time')``, whichever is available.
	plot_func : func, optional
		function to use for aggregating the data, defaults to :py:func:`numpy.mean`
	output_filename : str, optional 
		filename to save the resulting figure; if not given, figure is not saved.

	Returns
	-------
	seaborn.FacetGrid
		figure object.

	Raises
	------
	ValueError
		raised if `by` isn't set and `df` doesn't contain ``Strain`` and either ``Time`` or ``Cycle Nr.``.
	"""
	palette = df.Color.unique() if 'Color' in df else sns.color_palette()
	palette[palette == '#ffffff'] = '#000000'
	if by is None:
		if 'Cycle Nr.' in df and 'Strain' in df:
			by = ('Strain', 'Cycle Nr.')
		elif 'Time' in df and 'Strain' in df:
			by = ('Strain', 'Time')
		else:
			raise ValueError("If by is not set then df must have column Strain and either Time or Cycle Nr.")
		
	grp = df.groupby(by=by)
	agg = grp.aggregate(agg_func).reset_index()
	g = sns.FacetGrid(agg, hue=hue, size=5, aspect=1.5, palette=palette, hue_order=df[hue].unique())
	g.map(plot_func, x, y);
	g.add_legend()
	if output_filename:
		g.savefig(output_filename, bbox_inches='tight', pad_inches=1)
	return g


def tsplot(df, x='Time', y='OD', ci_level=95, output_filename=None):
	"""Time series plot of the data by strain (if applicable) or well.

	The grouping of the data is done by the value of `x` and ``Strain``, if such a column exists in `df`; 
	otherwise it is done by `x` and ``Well``.
	The aggregation is done by :py:func`seaborn.tsplot` which calculates the mean with a confidence interval.
	The colors are given by the ``Color`` column, the labels of the colors are given by the ``Strain`` column; 
	if ``Strain`` and ``Color`` don't exist in `df` then 
	the function will use a default palette and color the lines by well.

	Parameters
	----------
	df : pandas.DataFrame
		growth curve data, see :py:mod:`curveball.ioutils` for a detailed definition.
	x : str, optional
		name of column for x-axis, defaults to ``Time``.
	y : str, optional
		name of column for y-axis, defaults to ``OD``.
	ci_level : int, optional
		confidence interval width in precent (0-100), defaults to 95.	
	output_filename : str, optional 
		filename to save the resulting figure; if not given, figure is not saved.
	

	Returns
	-------
	matplotlib.axes.Axes
		figure object.
	"""
	if 'Strain' in df:
		condition = 'Strain'
		palette = df.Color.unique() if 'Color' in df else sns.color_palette()
		palette[palette == '#ffffff'] = '#000000'
	else:
		condition = 'Well'
		palette = sns.color_palette()
	g = sns.tsplot(df, time=x, unit='Well', condition=condition, value=y,
					err_style='ci_band', ci=ci_level, color=palette)
	sns.despine()
	if output_filename:
		g.figure.savefig(output_filename, bbox_inches='tight', pad_inches=1)
	return g


def plot_plate(df, edge_color='#888888', output_filename=None):
	"""Plot of the plate color mapping.

	The function will plot the color mapping in `df`: 
	a grid with enough columns and rows for the ``Col`` and ``Row`` columns in `df`, 
	where the color of each grid cell given by the ``Color`` column.

	Parameters
	----------
	df : pandas.DataFrame
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
	plate = df.pivot('Row', 'Col', 'Color').as_matrix()
	height, width = plate.shape
	fig = plt.figure(figsize=(old_div((width + 2), 3.), old_div((height + 2), 3.)))
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


def plot_params_distribution(param_samples, alpha=None):
	"""Plots a distribution of model parameter samples generated with :py:func:`curveball.models.sample_params`.

	Parameters
	----------
	param_samples : pandas.DataFrame
		data frame of samples; each row is one sample, each column is one parameter.
	alpha : float
		transparency of plot markers, defaults to :math:`1/n^{1/4}` where *n* is number of rows in `param_samples`.

	Returns
	-------
	sns.Grid
		figure object
	"""
	nsamples = param_samples.shape[0]
	g = sns.PairGrid(param_samples)
	if alpha is None:
		alpha = old_div(1,np.power(nsamples, old_div(1.,4)))
	g.map_upper(plt.scatter, alpha=alpha)
	g.map_lower(sns.kdeplot, cmap="Blues_d", legend=False)
	g.map_diag(plt.hist)
	return g