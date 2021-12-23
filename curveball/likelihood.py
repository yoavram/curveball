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
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set_style("ticks")
import curveball
import curveball.baranyi_roberts_model


def loglik(t, y, y_sig, f, penalty=None, **params):
    r"""Computes the log-likelihood of seeing the data given a model 
    assuming normal distributed observation/measurement errors.

    .. math::

        \log{L(y | \theta)} = -\frac{1}{2} \sum_i { \log{(2 \pi \sigma_{i}^{2})} + \frac{(y - f(t_i; \theta))^2}{\sigma_{i}^{2}} }

    which is the log-likelihood of seeing the data points :math:`t_i, y_i` 
    with measurement error :math:`\sigma_i`
    given the model function :math:`f`, the model parameters :math:`\theta`, 
    and that the measurement error at time :math:`t_i` has a normal distribution with mean 0.

    Parameters
    ----------
    t : np.ndarray
        one dimensional array of time
    y : np.ndarray
        one dimensional array of the means of the observations 
    y_sig : np.ndarray
        one dimensional array of standrad deviations of the observations
    f : callable
        a function the calculates the expected observations (`f(t)`) from `t` and any parameters in `params`
    penalty : callable
        a function that calculates a scalar penalty from the parameters in `params` to be substracted from the log-likelihood
    params : floats, optional
        model parameters

    Returns
    -------
    float
        the log-likelihood result
    """
    yhat = f(t, **params)
    val = -0.5 * np.sum(np.log(2 * np.pi * y_sig ** 2) + (y - yhat) ** 2 / y_sig ** 2)
    if penalty:
        val -= penalty(**params)
    return val
        

def ridge_regularization(lam, **center):
    r"""Create a penaly function that employs the ridge regularization method:

    .. math::

        P = \lambda ||\theta - \theta_0||_2

    where :math:`\lambda` is the regularization scale, 
    :math:`\theta` is the model parameters vector, 
    and :math:`\theta_0` is the model parameters guess vector.
    This is similar to using a multivariate Gaussian prior distribution on the model parameters 
    with the Gaussian centerd at :math:`\theta_0` and scaled by :math:`\lambda`.

    Parameters
    ----------
    lam : float
        the penalty factor or regularization scale
    center : floats, optional
        guesses of model parameters

    Returns
    -------
    callable
        the penalty function, accepts model parameters as float keyword arguments and returns a float penalty to the log-likelihood

    Examples
    --------
    >>> penalty = ridge_regularization(1, y=0.1, K=1, r=1)
    >>> loglik(t, y, y_sig, logistic, penalty=penalty, y0=0.12, K=0.98, r=1.1)
    """
    def _ridge_regularization(**params):
        return lam * np.linalg.norm([v - center.get(k, 0) for k, v in params.items() if np.isfinite(v)])
    return _ridge_regularization


def loglik_r_nu(r_range, nu_range, df, f=curveball.baranyi_roberts_model.baranyi_roberts_function, 
                penalty=None, **params):
    r"""Estimates the log-likelihood surface for :math:`r` and :math:`\nu` given data and a model function.

    Parameters
    ----------
    r_range, nu_range : numpy.ndarray
        vectors of floats of :math:`r` and :math:`\nu` values on which to compute the log-likelihood
    df : pandas.DataFrame
        data frame with `Time` and `OD` columns
    f : callable, optional
        model function, defaults to :py:func:`curveball.baranyi_roberts_model.baranyi_roberts_function`
    penalty : callable, optional
        if given, the result of `penalty` will be substracted from the log-likelihood for each parameter set
    params : floats
        values for the model model parameters used by `f`

    Returns
    -------
    np.ndarray
        two-dimensional array of log-likelihood calculations;
        value at index `i, j` will have the value for `r_range[i]` and `nu_range[j]`

    See also
    --------
    loglik
    loglik_r_q0
    """
    if not params:
        params = dict()
    t = df.Time.unique()
    y = df.groupby('Time')['OD'].mean().values
    y_sig = df.groupby('Time')['OD'].std().values
    
    output = np.empty((len(r_range), len(nu_range)))
    for i, r in enumerate(r_range):
        params['r'] = r
        for j, nu in enumerate(nu_range):
            params['nu'] = nu
            output[i,j] = loglik(t, y, y_sig, f, penalty, **params)
    
    return output


def loglik_r_q0(r_range, q0_range, df, f=curveball.baranyi_roberts_model.baranyi_roberts_function, 
                penalty=None, **params):
    r"""Estimates the log-likelihood surface for :math:`r` and :math:`\nu` given data and a model function.

    Parameters
    ----------
    r_range, q0_range : numpy.ndarray
        vectors of floats of :math:`r` and :math:`q_0` values on which to compute the log-likelihood
    df : pandas.DataFrame
        data frame with `Time` and `OD` columns
    f : callable, optional
        model function, defaults to :py:func:`curveball.baranyi_roberts_model.baranyi_roberts_function`
    penalty : callable, optional
        if given, the result of `penalty` will be substracted from the log-likelihood for each parameter set
    params : floats
        values for the model model parameters used by `f`

    Returns
    -------
    np.ndarray
        two-dimensional array of log-likelihood calculations; 
        value at index `i, j` will have the value for `r_range[i]` and `q0_range[j]`

    See also
    --------
    loglik
    loglik_r_nu
    """
    if not params:
        params = dict()
    t = df.Time.unique()
    y = df.groupby('Time')['OD'].mean().values
    y_sig = df.groupby('Time')['OD'].std().values
    
    output = np.empty((len(r_range), len(q0_range)))
    for i, r in enumerate(r_range):
        params['r'] = r
        for j, q0 in enumerate(q0_range):
            params['q0'] = q0
            output[i,j] = loglik(t, y, y_sig, f, penalty, **params)
    
    return output


def plot_loglik(Ls, xrange, yrange, xlabel=None, ylabel=None, columns=4, fig_title=None, normalize=True,
                ax_titles=None, cmap='viridis', colorbar=True, ax_width=4, ax_height=4, ax=None):
    r"""Plots one or more log-likelihood surfaces.

    Parameters
    ----------
    Ls : sequence of numpy.ndarray
        list or tuple of log-likelihood two-dimensional arrays; if one array is given it will be converted to a size 1 list
    xrange, yrange : np.ndarray
        values on x-axis and y-axis of the plot (rows and columns of `Ls`, respectively)
    xlabel, ylabel : str, optional
        strings for x and y labels
    columns : int, optional
        number of columns in case that `Ls` has more than one matrice
    fig_title : str, optional
        a title for the whole figure
    normalize : bool, optional
        if :py:const:`True`, all matrices will be plotted using a single color scale
    ax_titles : list or tuple of str, optional
        titles corresponding to the different matrices in `Ls`
    cmap : str. optional
        name of a matplotlib colormap (to see list, call :py:func:`matplotlib.pyplot.colormaps`), defaults to `viridis`
    colorbar : bool, optional
        if :py:const:`True` a colorbar will be added to the plot
    ax_width, ax_height : int
        width and height of each panel (one for each matrice in `Ls`)
    ax : matplotlib axes or numpy.ndarray of axes
        if given, will plot into `ax`, otherwise will create a new figure

    Returns
    -------
    fig : matplotlib.figure.Figure
        figure object
    ax : numpy.ndarray
        array of axis objects

    Examples
    --------
    >>> L = loglik_r_nu(rs, nus, df, y0=y0, K=K, q0=q0, v=v)
    >>> plot_loglik(L0, rs, nus, normalize=False, fig_title=fig_title, xlabel=r'$r$', ylabel=r'$\nu$', colorbar=False)
    """
    if not isinstance(Ls, (list, tuple)):
        Ls = [Ls]
    columns = min(columns, len(Ls))
    rows = int(np.ceil(len(Ls) / columns))
    if ax is None:
        fig, ax = plt.subplots(rows, columns, sharex=True, sharey=True, figsize=(ax_width*columns, ax_height*rows))
    else:
        fig = ax.figure
    if not hasattr(ax, '__iter__'):
        ax = np.array(ax, ndmin=2)
    if ax.ndim == 1:
        ax.resize((rows, columns))
    vmin = np.nanmin(Ls)
    vmax = np.nanmax(Ls)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    
    for i, L in enumerate(Ls):
        row = i // columns
        col = i % columns
        _ax = ax[row, col]
        
        im = _ax.imshow(L.T, cmap=cmap, aspect=1, origin='lower')
        if normalize:
            im.set_norm(norm)
    
        if colorbar:
            plt.colorbar(im, ax=_ax, label='Log Likelihood', fraction=0.03, pad=0.25, format='%1.e')

        _ax.xaxis.grid(color='k', ls='--', alpha=0.5)
        _ax.yaxis.grid(color='k', ls='--', alpha=0.5)

        xstep = len(xrange) // 5
        xticks = list(range(0, len(xrange), xstep)) + [len(xrange)-1]
        xticklabels = xrange[xticks]
        _ax.set_xticks(xticks)
        _ax.set_xticklabels(['{:.2g}'.format(x) for x in xticklabels], rotation='vertical')

        ystep = len(xrange) // 5
        yticks = list(range(0, len(yrange), ystep)) + [len(yrange)-1]
        yticklabels = yrange[yticks]
        _ax.set_yticks(yticks)
        _ax.set_yticklabels(['{:.2g}'.format(y) for y in yticklabels])

        imax, jmax = (L == np.nanmax(L)).nonzero()
        _ax.scatter(imax, jmax, marker='o', s=50, color='r')
        
        if row == rows-1 and xlabel:
            _ax.set_xlabel(xlabel)
        if col == 0 and ylabel:
            _ax.set_ylabel(ylabel)
        if ax_titles:
            _ax.set_title(ax_titles[i])

    if fig_title:
        fig.text(0.5, 1, fig_title, fontsize=24, horizontalalignment='center')     
    fig.tight_layout()
    sns.despine()
    return fig, ax


def plot_model_loglik(m, df, fig_title=None):
    r"""Plot the log-ikelihood surfaces for :math:`\nu` over :math:`r` and :math:`q_0` over :math:`r` for given data and model fitting result.

    Parameters
    ----------
    m : lmfit.model.ModelResult
        model for which to plot the log-likelihood surface
    df : pandas.DataFrame
        data frame with `Time` and `OD` columns used to fit the model
    fig_title : str
        title for the plot

    Returns
    -------
    fig : matplotlib.figure.Figure
        figure object
    ax : numpy.ndarray
        array of axis objects

    Examples
    --------
    >>> m = curveball.models.fit_model(df)
    >>> curveball.likelihood.plot_model_loglik(m, df)

    """
    K = m.best_values['K']
    y0 = m.best_values['y0']
    r = m.best_values['r']
    nu = m.best_values.get('nu', 1)
    q0 = m.best_values.get('q0', np.inf)
    v = m.best_values.get('v', np.inf)

    rs = np.logspace(-2, 4, 100)
    nus = np.logspace(-3, 2.5, 100)
    q0s = np.logspace(-3, 2.5, 100)

    ir = ( abs(rs - r).argmin() )
    inu = ( abs(nus - nu).argmin() )
    iq0 = ( abs(q0s - m.best_values.get('q0', np.inf)).argmin() )
    ir_in = ( abs(rs - m.init_values['r']).argmin() )
    inu_in = ( abs(nus - m.init_values.get('nu', 1)).argmin() )
    iq0_in = ( abs(q0s - m.init_values.get('q0', np.inf)).argmin() )

    L0 = loglik_r_nu(rs, nus, df, y0=y0, K=K, q0=q0, v=v)
    L1 = loglik_r_q0(rs, q0s, df, y0=y0, K=K, nu=nu, v=v)
        
    w, h= plt.rcParams['figure.figsize']
    fig, ax = plt.subplots(1, 2, figsize=(w * 2, h))

    plot_loglik(L0, rs, nus, normalize=False, fig_title=fig_title,
                xlabel=r'$r$', ylabel=r'$\nu$', colorbar=False, ax=ax[0])
    ax[0].scatter(ir, inu, marker='s', s=30, color='w', edgecolors='k')
    ax[0].scatter(ir_in, inu_in, marker='^', s=30, color='w', edgecolors='k')

    plot_loglik(L1, rs, q0s, normalize=False,
                xlabel=r'$r$', ylabel=r'$q_0$', colorbar=True, ax=ax[1])
    ax[1].scatter(ir, iq0, marker='s', s=30, color='w', edgecolors='k')
    ax[1].scatter(ir_in, iq0_in, marker='^', s=30, color='w', edgecolors='k')

    fig.tight_layout()
    sns.despine()
    return fig, ax