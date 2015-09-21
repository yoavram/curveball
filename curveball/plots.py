#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of curveball.
# https://github.com/yoavram/curveball

# Licensed under the MIT license:
# http://www.opensource.org/licenses/MIT-license
# Copyright (c) 2015, Yoav Ram <yoav@yoavram.com>
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("ticks")

from matplotlib.patches import RegularPolygon
from string import ascii_uppercase


def plot_wells(df, x='Time', y='OD', plot_func=plt.plot, output_filename=None):
	"""Plot a grid of plots, one for each well in the plate.

	The facetting is done by the `Row` and `Col` columns of `df`.
	The colors are given by the `Color` column, the labels of the colors are given by the `Strain` column.

	Args:
		- df: :py:class:`pandas.DataFrame`.
		- x: name of column for x-axis, string.
		- y: name of column for y-axis, string.
		- plot_func: a function to use for plotting, defaults to :py:func:`matplotlib.pyplot.plot`
		- output_filename: optional filename to save the resulting figure, string.

	Returns:
		g: :py:class:`seaborn.FacetGrid`
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
                      aspect=width/float(height), despine=True,margin_titles=True)
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

	The grouping of the data is done by the `Strain` and `Cycle Nr.` columns; the aggregation is done by the `agg_func`, which defaults to `mean`.
	The colors are given by the `Color` column, the labels of the colors are given by the `Strain` column.

	Args:
		- df: :py:class:`pandas.DataFrame`.
		- x: name of column for x-axis, string.
		- y: name of column for y-axis, string.
		- plot_func: a function to use for plotting, defaults to :py:func:`matplotlib.pyplot.plot`.
		- by: a :py:class:tuple to use for grouping the data, defaults to `('Strain', 'Cycle Nr.')` or `('Strain', 'Time')`, whichever is available.
		- agg_func: a function to use for aggregating the data, defaults to :py:func:`numpy.mean`.
		- output_filename: optional filename to save the resulting figure, string.

	Returns:
		g: :py:class:`seaborn.FacetGrid`
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
	"""tsplot plot of the data by strain (if applicable) or well.

	The grouping of the data is done by `x` and `Strain` if such a column exists; otherwise it is done by `x` and `Well`.
	The aggregation is done by `seaborn.tsplot` which calculates the mean with a confidence interval.
	The colors are given by the `Color` column, the labels of the colors are given by the `Strain` column; if `Strain` and `Color` don't exist the function will use a default palette and color the lines by well.

	Args:
		- df: :py:class:`pandas.DataFrame`.
		- x: name of column for x-axis, string.
		- y: name of column for y-axis, string.
		- ci_level: confidence interval width in precent (int; 0-100).
		- output_filename: optional filename to save the resulting figure, string.

	Returns:
		g: :py:class:`seaborn.FacetGrid`
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
		g.savefig(output_filename, bbox_inches='tight', pad_inches=1)
	return g


def plot_plate(df, edge_color='#888888'):
	"""Plot of the plate color mapping.

	The function will plot the color mapping in `df`: a grid with enough columns and rows for the `Col` and `Row` columns in `df`, where the color of each grid cell given by the `Color` column.

	Args:
		- df: :py:class:`pandas.DataFrame`.
		- edge_color: a color hex string for the grid edges.

	Returns:
		fig, ax : :py:class:`tuple`
			- fig: the :py:class:`matplotlib.figure.Figure` object
			- ax: an array of axis objects.
	"""
	plate = df.pivot('Row', 'Col', 'Color').as_matrix()
	height, width = plate.shape
	fig = plt.figure(figsize=((width + 2) / 3., (height + 2) / 3.))
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
	                                         orientation=np.pi / 4,
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
	return fig, ax


def plot_params_distribution(param_samples, alpha=None):
	nsamples = param_samples.shape[0]
	g = sns.PairGrid(param_samples)
	if alpha is None:
		alpha = 1/np.power(nsamples, 1./4)
	g.map_upper(plt.scatter, alpha=alpha)
	g.map_lower(sns.kdeplot, cmap="Blues_d", legend=False)
	g.map_diag(plt.hist)
	return g