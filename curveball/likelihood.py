#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of curveball.
# https://github.com/yoavram/curveball

# Licensed under the MIT license:
# http://www.opensource.org/licenses/MIT-license
# Copyright (c) 2015, Yoav Ram <yoav@yoavram.com>
from __future__ import print_function
from __future__ import division
from builtins import filter
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
    yhat = f(t, **params)
    val = -0.5 * np.sum(np.log(2 * np.pi * y_sig ** 2) + (y - yhat) ** 2 / y_sig ** 2)
    if penalty:
        val -= penalty(**params)
    return val
        

def ridge_regularization(lam, **center):
    def _ridge_regularization(**params):
        return lam * np.linalg.norm([v - center.get(k, 0) for k, v in params.items() if np.isfinite(v)])
    return _ridge_regularization


def loglik_r_nu(r_range, nu_range, df, f=curveball.baranyi_roberts_model.baranyi_roberts_function, 
                penalty=None, **params):
    if not params:
        params = dict()
    t = df.Time.unique()
    y = df.groupby('Time').OD.mean().as_matrix()
    y_sig = df.groupby('Time').OD.std().as_matrix()
    
    output = np.empty((len(r_range), len(nu_range)))
    for i, r in enumerate(r_range):
        params['r'] = r
        for j, nu in enumerate(nu_range):
            params['nu'] = nu
            output[i,j] = loglik(t, y, y_sig, f, penalty, **params)
    
    return output


def loglik_r_q0(r_range, q0_range, df, f=curveball.baranyi_roberts_model.baranyi_roberts_function, 
                penalty=None, **params):
    if not params:
        params = dict()
    t = df.Time.unique()
    y = df.groupby('Time').OD.mean().as_matrix()
    y_sig = df.groupby('Time').OD.std().as_matrix()
    
    output = np.empty((len(r_range), len(q0_range)))
    for i, r in enumerate(r_range):
        params['r'] = r
        for j, q0 in enumerate(q0_range):
            params['q0'] = q0
            output[i,j] = loglik(t, y, y_sig, f, penalty, **params)
    
    return output


def plot_loglik(Ls, xrange, yrange, xlabel=None, ylabel=None, columns=4, fig_title=None, normalize=True,
                ax_titles=None, cmap='viridis', colorbar=True, ax_width=4, ax_height=4, ax=None):
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
        
        im = _ax.imshow(L.T, cmap=cmap, aspect=1, origin=(0,0))
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
