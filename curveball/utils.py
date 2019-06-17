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

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.linear_model
import sklearn.preprocessing
import sklearn.pipeline
try:
    from sklearn.model_selection import GridSearchCV
except ImportError:
    from sklearn.grid_search import GridSearchCV
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
    model = GridSearchCV(
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
    if kwargs.get('PRINT'):
        print(model)
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
