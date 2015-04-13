#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of curveball.
# https://github.com/yoavram/curveball

# Licensed under the MIT license:
# http://www.opensource.org/licenses/MIT-license
# Copyright (c) 2015, Yoav Ram <yoavram+github@gmail.com>
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("ticks")

from matplotlib.patches import RegularPolygon
from string import ascii_uppercase


def plot_timeseries_grid(df, x, y, func=plt.plot, output_filename=None):
	if 'Strain' in df:
		hue = 'Strain'
		palette = df.Color.unique()
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
	g.map(func, x, y)
	g.fig.set_figwidth(width)
	g.fig.set_figheight(height)
	plt.locator_params(nbins=4) # 4 ticks is enough
	g.set_axis_labels('','') 	# remove facets axis labels
	g.fig.text(0.5, 0, x, size='x-large') # xlabel
	g.fig.text(-0.01, 0.5, y, size='x-large', rotation='vertical') # ylabel
	if output_filename:
		g.savefig(output_filename, bbox_inches='tight', pad_inches=1)
	return g


def plot_mean_timeseries(df, x='Time', y='OD', output_filename=None):
	palette = df.Color.unique()
	palette[palette == '#ffffff'] = '#000000'
	grp = df.groupby(by=('Strain', 'Cycle Nr.'))
	agg = grp.mean().reset_index()
	g = sns.FacetGrid(agg, hue='Strain', size=5, aspect=1.5, palette=palette, hue_order=df.Strain.unique())
	g.map(plt.plot, x, y);
	g.add_legend()
	if output_filename:
		g.savefig(output_filename, bbox_inches='tight', pad_inches=1)
	return g


def tsplot(df, x='Time', y='OD', alfa=0.05, output_filename=None):
	if 'Strain' in df:
		condition = 'Strain'
		palette = df.Color.unique()
		palette[palette == '#ffffff'] = '#000000'
	else:
		condition = 'Well'
		palette = sns.color_palette()
	g = sns.tsplot(df, time=x, unit='Well', condition=condition, value=y,
					err_style='ci_band', ci=100 * (1 - alfa), color=palette)
	sns.despine()
	if output_filename:
		g.savefig(output_filename, bbox_inches='tight', pad_inches=1)
	return g


def plot_plate(df, edge_color='#888888'):
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