#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of curveball.
# https://github.com/yoavram/curveball

# Licensed under the MIT license:
# http://www.opensource.org/licenses/MIT-license
# Copyright (c) 2015, Yoav Ram <yoav@yoavram.com>
from __future__ import print_function
from __future__ import division
from builtins import str
from builtins import range
import warnings
from distutils.version import LooseVersion

import requests
import click
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.linear_model
import sklearn.preprocessing
import sklearn.pipeline
import sklearn.grid_search
import webcolors


def smooth(x, y, PLOT=False, **kwargs):
    """Estimate a smoothing function.

    The function finds a polynomial function that fits to the data using a linear regression model
    with polynomial features and using a cross-validation grid search to find the best polynomial degree.

    Parameters
    ----------
    x : numpy.ndarray
        array of floats for the independent variable
    y : numpy.ndarray
        array of floats for the dependent variable
    PLOT : bool, optional
        if :const:`True`, plots a figure of the input and smoothed data, defaults to :const:`False`
    kwargs : optional
        extra keyword arguments passed to the underlying smoothing function.

    Returns
    -------
    f : function
        smooth function that corresponds to the data.
    fig : matplotlib.figure.Figure
        if the argument `PLOT` was :const:`True`, the generated figure.
    ax : matplotlib.axes.Axes
        if the argument `PLOT` was :const:`True`, the generated axis.
    """
    assert len(x) == len(y)
    x = np.array(x)
    y = np.array(y)

    # do a grid search to find the optimal polynomial degree
    model = sklearn.grid_search.GridSearchCV(
        # use a linear model with polynomial features
        sklearn.pipeline.Pipeline([
            ('poly', sklearn.preprocessing.PolynomialFeatures(degree=3)),
            ('linear', sklearn.linear_model.LinearRegression())
        ]),
        cv=kwargs.get('cv', min(5, len(x))),
        param_grid={
            'poly__degree': np.arange(
                kwargs.get('min_degree', 3),
                kwargs.get('max_degree', 14),
                2
            )
        }
    )
    x = x.reshape(-1, 1)
    model.fit(x, y)

    def predict(_x):
        return model.predict(np.array(_x).reshape(-1, 1))

    if PLOT:
        yhat = predict(x)
        fig, ax = plt.subplots(1, 1)
        ax.plot(x, yhat, 'k--')
        ax.plot(x, y, 'ko')
        ax.set(xlabel='x', ylabel='y')
        sns.despine()
        return predict, fig, ax

    return predict


def color_name_to_hex(name, default='#000000'):
    """Convert a color name to a hex values.

    Does not attempt to convert names that already start with '#'.

    Parameters
    ----------
    name : str
        color name to convert, i.e. "red"
    default : str
        the default color in case name is invalid, default value is "#000000".

    Returns
    -------
    str
        hex value corresponding to name
    """
    try:
        name = str(name)
        if name.startswith('#'):
            return name
        return webcolors.name_to_hex(name)
    except ValueError:
        return default


def check_version(repository='pypi', pkg='curveball', owner='yoavram'):
    from curveball import __version__ as cur_ver
    if repository == 'pypi':
        url = 'http://pypi.python.org/pypi/{pkg}/json'.format(pkg=pkg)
        cmd = 'python -m pip install --upgrade {pkg}'.format(pkg=pkg)
        version_extractor = lambda d: d['info']['version']
    elif repository == 'anaconda':
        url = 'http://api.anaconda.org/package/{owner}/{pkg}'.format(pkg=pkg, owner=owner)
        cmd = 'conda update -c {owner} {pkg}'.format(pkg=pkg, owner=owner)
        version_extractor = lambda d: d['latest_version']
    else:
        raise ValueError("Unknown repository: {}".format(repository))

    try:
        r = requests.get(url)
    except Exception as e:
        warnings.warn("Couldn't establish connection to {}: {}".format(repository, e))
        return
    if not r.ok:
        warnings.warn("Couldn't establish connection to {}: {}".format(repository, r.reason))
        return

    repo_ver = version_extractor(r.json())
    if LooseVersion(repo_ver) > LooseVersion(cur_ver):
            msg = "You are using {pkg} version {cur_ver}, however version {repo_ver} is available.\nYou should consider upgrading via the '{cmd}' command."
            msg = msg.format(cmd=cmd, pkg=pkg, cur_ver=cur_ver, repo_ver=repo_ver)
            click.secho(msg, fg='red')
