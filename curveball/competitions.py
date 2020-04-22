#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of curveball.
# https://github.com/yoavram/curveball

# Licensed under the MIT license:
# http://www.opensource.org/licenses/MIT-license
# Copyright (c) 2015, Yoav Ram <yoav@yoavram.com>
from __future__ import division
from builtins import range
import warnings
import copy
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import curve_fit
import pandas as pd
import lmfit
import curveball
import seaborn as sns
sns.set_style("ticks")


def _alfa(t, q0, v):
    if np.isinf(q0) or np.isinf(v):
        return 1.0
    return q0 / (q0 + np.exp(-v * t))

def double_baranyi_roberts_ode0(y, t, r, K, nu, q0, v):
    r"""A two species Baranyi-Roberts model [1]_. The function calculates the population growth rate at given time points.

    .. math::

        \frac{dN_i}{dt} = r_i \alpha_i(t) N_i \Big(1 - \Big(\frac{\sum_{j}{N_j}}{K_i}\Big)^{\nu_i}\Big)

    - :math:`N_i`: population size of strain *i*.
    - :math:`r_i`: initial per capita growth rate of strain *i*
    - :math:`K_i`: maximum population size of strain *i*
    - :math:`\nu_i`: curvature of the logsitic term of strain *i*
    - :math:`\alpha_i(t)= \frac{q_{0,i}}{q_{0,i} + e^{-v_i t}}`
    - :math:`q_{0,i}`: initial adjustment of strain *i* to current environment
    - :math:`v_i`: adjustment rate of strain *i*

    Parameters
    ----------
    y : float, float
        population size
    t : float
        time, usually in hours
    r : float, float
        initial per capita growth rate
    K : float, float
        maximum population size (:math:`K>0`)
    nu : float, float
        curvature of the logsitic term (:math:`\nu>0`)
    q0 : float, float
        initial adjustment to current environment (:math:`0<q_0<1`)
    v : float, float
        adjustment rate (:math:`v>0`)

    Returns
    -------
    float, float
        population growth rate.

    References
    ----------
    .. [1] Baranyi, J., Roberts, T. A., 1994. `A dynamic approach to predicting bacterial growth in food <www.ncbi.nlm.nih.gov/pubmed/7873331>`_. Int. J. Food Microbiol.
    """
    alfa = _alfa(t, q0[0], v[0]), _alfa(t, q0[1], v[1])
    dydt = alfa[0] * r[0] * y[0] * (1 - ((y[0] + y[1]) / K[0])**nu[0]), alfa[1] * r[1] * y[1] * (1 - ((y[0] + y[1]) / K[1])**nu[1])
    return dydt


def double_baranyi_roberts_ode1(y, t, r, K, nu, q0, v):
    r"""A two species Baranyi-Roberts model. The function calculates the population growth rate at given time points.

    .. math::

        \frac{dN_i}{dt} = r_i \alpha_i(t) N_i \Big(1 - \Big(\sum_{j}{\frac{N_j}{K_j}}\Big)^{\nu_i}\Big)

    See also
    --------
    curveball.competitions.double_baranyi_roberts_ode0
    """
    alfa = _alfa(t, q0[0], v[0]), _alfa(t, q0[1], v[1])
    dydt = alfa[0] * r[0] * y[0] * (1 - (y[0] / K[0] + y[1] / K[1])**nu[0]), alfa[1] * r[1] * y[1] * (1 - (y[0] / K[0] + y[1]/K[1])**nu[1])
    return dydt


def double_baranyi_roberts_ode2(y, t, r, K, nu, q0, v):
    r"""A two species Baranyi-Roberts model. The function calculates the population growth rate at given time points.

    .. math::

        \frac{dN_i}{dt} = r_i \alpha_i(t) N_i \Big(1 - \sum_{j}{\Big(\frac{N_j}{K_j}\Big)^{\nu_j}}\Big)

    See also
    --------
    curveball.competitions.double_baranyi_roberts_ode0
    """
    alfa = _alfa(t, q0[0], v[0]), _alfa(t, q0[1], v[1])
    dydt = alfa[0] * r[0] * y[0] * (1 - (y[0] / K[0])**nu[0] - (y[1] / K[1])**nu[1]), alfa[1] * r[1] * y[1] * (1 - (y[0] / K[0])**nu[0] - (y[1] / K[1])**nu[1])
    return dydt


def double_baranyi_roberts_ode3(y, t, r, K, nu, q0, v):
    r"""A two species Baranyi-Roberts model. The function calculates the population growth rate at given time points.

    .. math::

        \frac{dN_i}{dt} = r_i \alpha_i(t) N_i \Big( 1 - \Big(\frac{\sum_j{N_j}}{\bar{K}}\Big)^{\nu_i} \Big)

    See also
    --------
    curveball.competitions.double_baranyi_roberts_ode0
    """
    alfa = _alfa(t, q0[0], v[0]), _alfa(t, q0[1], v[1])
    Kmean = (K[0] * y[0] + K[1] * y[1]) / (y[0] + y[1])
    dydt = alfa[0] * r[0] * y[0] * (1 - ((y[0] + y[1]) / Kmean)**nu[0]), alfa[1] * r[1] * y[1] * (1 - ((y[0] + y[1]) / Kmean)**nu[1])
    return dydt

def double_baranyi_roberts_ode4(y, t, r, K, nu, q0, v):
    r"""A two species Baranyi-Roberts model. The function calculates the population growth rate at given time points.

    .. math::

        \frac{dN_i}{dt} = r_i \alpha_i(t) N_i

    See also
    --------
    curveball.competitions.double_baranyi_roberts_ode0
    """
    alfa = _alfa(t, q0[0], v[0]), _alfa(t, q0[1], v[1])
    dydt = alfa[0] * r[0] * y[0], alfa[1] * r[1] * y[1]
    return dydt


def double_baranyi_roberts_ode5(y, t, r, K, nu, q0, v):
    r"""A two species Baranyi-Roberts model. The function calculates the population growth rate at given time points.

    .. math::

        \frac{dN_i}{dt} = r_i \alpha_i(t) N_i \Big( 1 - \Big(\frac{N_i}{K_i}\Big)^{\nu_i}\Big)

    See also
    --------
    curveball.competitions.double_baranyi_roberts_ode0
    """
    alfa = _alfa(t, q0[0], v[0]), _alfa(t, q0[1], v[1])
    dydt = alfa[0] * r[0] * y[0] * (1 - (y[0]/K[0])**nu[0]), alfa[1] * r[1] * y[1] * (1 - (y[1]/K[1])**nu[1])
    return dydt


def double_baranyi_roberts_ode6(y, t, r, K, nu, q0, v):
    r"""A two species Baranyi-Roberts model. The function calculates the population growth rate at given time points.

    .. math::

        \frac{dN_i}{dt} = r_i \alpha_i(t) N_i \Big(1 - \Big(\frac{\sum_j{N_j}}{K_i}\Big)^{\bar{\nu}}\Big)

    See also
    --------
    curveball.competitions.double_baranyi_roberts_ode0
    """
    alfa = _alfa(t, q0[0], v[0]), _alfa(t, q0[1], v[1])
    numean = (nu[0]*y[0] + nu[1]*y[1]) / (y[0] + y[1])
    dydt = alfa[0] * r[0] * y[0] * (1 - ((y[0]+y[1])/K[0])**numean), alfa[1] * r[1] * y[1] * (1 - ((y[0]+y[1])/K[1])**numean)
    return dydt


def double_baranyi_roberts_ode7(y, t, r, K, nu, q0, v):
    r"""A two species Baranyi-Roberts model. The function calculates the population growth rate at given time points.

    .. math::

        \frac{dN_1}{dt} = 0 \;\;\; \frac{dN_2}{dt} = r_2 \alpha_2(t) N_2 \Big(1 - \Big(\frac{N_2}{K_2}\Big)^{\nu_1}\Big)

    See also
    --------
    curveball.competitions.double_baranyi_roberts_ode0
    """
    alfa = _alfa(t, q0[0], v[0]), _alfa(t, q0[1], v[1])
    numean = (nu[0]*y[0] + nu[1]*y[1]) / (y[0] + y[1])
    dydt = 0, alfa[1] * r[1] * y[1] * (1 - (y[1]/K[1])**nu[1])
    return dydt


def double_baranyi_roberts_ode8(y, t, r, K, nu, q0, v):
    r"""A two species Baranyi-Roberts model. The function calculates the population growth rate at given time points.

    .. math::

        \frac{dN_i}{dt} = r_i N_i \Big(1 - \Big(\frac{\sum_{j}{N_j}}{K_i}\Big)^{\nu_i}\Big)

    See also
    --------
    curveball.competitions.double_baranyi_roberts_ode0
    """
    dydt = r[0] * y[0] * (1 - ((y[0] + y[1]) / K[0])**nu[0]), r[1] * y[1] * (1 - ((y[0] + y[1]) / K[1])**nu[1])
    return dydt


def double_baranyi_roberts_gimenez_delgado_ode(y, t, r, K, nu, q0, v):
    r"""A two species Baranyi-Roberts model with competition model inspired by Gimenez and Delgado (2004).
    The function calculates the population growth rate at given time points.

    .. math::

        \frac{dN_i}{dt} = r_i \alpha_i(t) N_i \prod_j{\Big(1 - \Big(\frac{N_j}{K_j}\Big)^{\nu_j}\Big)}

    See also
    --------
    curveball.competitions.double_baranyi_roberts_ode0
    """
    alfa = _alfa(t, q0[0], v[0]), _alfa(t, q0[1], v[1])
    prod = (1 - (y[0]/K[0])**(nu[0])) * (1 - (y[1]/K[1])**(nu[1]))
    dydt = alfa[0] * r[0] * y[0] * prod, alfa[1] * r[1] * y[1] * prod
    return dydt


def compete(m1, m2, p0=(0.5, 0.5), y0=None, t=None, hours=24, num_of_points=100, 
            nsamples=1, lag_phase=True, ode=double_baranyi_roberts_ode1,
            params1=None, params2=None, ci=95, colors=None, ax=None, PLOT=False,
            sampler='covar', df1=None, df2=None):
    """Simulate competitions between two strains using growth parameters estimated
    by fitting growth models to growth curves data.

    Integrates a 2-dimensional ODE system given by `ode`
    with parameters extracted from :py:class:`lmfit.model.ModelResult` instances `m1` and `m2`.
    This implementation includes plotting (if required by `PLOT`);
    resampling from the distribution of model parameters (when `nsamples` > 1);
    changing the ODE system (by providing a different function in `ode`);
    and more.

    The function competes two strains/species;
    therefore it expects two :py:class:`lmfit.model.ModelResult` objects,
    two initial values in `y0`, etc.

    Parameters
    ----------
    m1, m2 : lmfit.model.ModelResult
        model fitting results of growth models defined in :py:mod:`curveball.models`.
    p0, y0 : tuple, optional
        `p0` is the initial relative frequencies; `y0` is the initial population sizes.
        if `y0` is given than ``y0[0]`` and ``y0[0]`` are the initial population size for `m1` and `m2`, respectively.
        if `y0` is not given, then it will be set to the average estimated `y0` in `m1` and `m2` multiplied by `p0`.
    t : numpy.ndarray or None, optional
        array of time points in which to calculate the population sizes.
    hours : int, optional
        if `t` is not given, determines how many hours should the competition proceed, defaults to 24.
    num_of_points : int, optional
        if `t` is not given, determines the number of time points to use, defaults to 100.
    nsamples : int, optional
        how many replicates of the competition should be simulated;
        if `nsamples` = 1, only one competition is simulated with the estimated parameters;
        otherwise `nsamples` competitions are simulated with parameters drawn from a parameter distribution
        determined by `sampler`. Defaults to 1.
    lag_phase : bool, optional
        if `True`, use lag phase as given by `m1` and `m2`. Otherwise, override the lag phase parameters to prevent a lag phase. Defaults to :const:`True`.
    ode : func, optional
        an ordinary differential systems system defined by a function that accepts ``y``, ``t``, and additional arguments, and returns the derivate of ``y`` with respect to ``t``. Defaults to :py:func:`.double_baranyi_roberts_ode0`.
    params1, params2 : dict, optional
        dictionaries of model parameter values; if given, overrides values from `m1` and `m2`.
    ci : float, optional
        confidence interval size, in (0, 100), only applicable when `PLOT` is :const:`True`, defaults to 95%.
    colors : sequence of str, optional
        if `PLOT` is :const:`True`, this sets the colors of the drawn lines. `colors[0]` will be used for `m1`; `colors[1]` for `m2`. If not provided, defaults to the current pallete.
    ax : matplotlib.axes.Axes, optional
        if `PLOT` is :const:`True`, an axes to plot into; if not provided, a new one is created.
    PLOT : bool, optional
        if :const:`True`, the function will plot the curves of *y* as a function of *t*. Defaults to :const:`False`.
    sampler : str, optional
        if ``covar``, the parameters will be sampled using the covariance matrix (:py:func:`curveball.models.sample_params`);
        if ``bootstrap`` the parameters will be sampled by resampling the growth curves (:py:func:`curveball.models.bootstrap_params`).
    df1, df2 : pandas.DataFrame, optional
        the data frames used to fit `m1` and `m2`, only used when `sampler` is ``bootstrap``.

    Returns
    -------
    t : numpy.ndarray
        1d (or 2d, if `nsamples`>1) array of time points, in hours.
    y: numpy.ndarray
        2d (or 3d, if `nsamples`>1) array of strain frequencies.
        First axis is time, second axis is strain, third axis (if applicable) is sample.
    fig : matplotlib.figure.Figure
        figure object.
    ax : numpy.ndarray
        array of :py:class:`matplotlib.axes.Axes` objects.

    Raises
    ------
    TypeError
        if `m1` or `m2` are not :py:class:`lmfit.model.ModelResult`.
    AssertionError
        if an intermediate calculation produced an invalid result (not guaranteed).

    Examples
    --------
    >>> import pandas as pd
    >>> import curveball
    >>> plate = pd.read_csv('plate_templates/G-RG-R.csv')
    >>> df = curveball.ioutils.read_tecan_xlsx('data/Tecan_280715.xlsx', label='OD', plate=plate)
    >>> green = curveball.models.fit_model(df[df.Strain == 'G'], PLOT=False, PRINT=False)[0]
    >>> red = curveball.models.fit_model(df[df.Strain == 'R'], PLOT=False, PRINT=False)[0]
    >>> t, y = curveball.competitions.compete(green, red, PLOT=False)

    Notes
    -----
    To debug, uncomment lines to return the ``infodict`` that is returned from :py:func:`scipy.integrate.odeint` is run with ``full_output=1``.
    """
    if not isinstance(m1, lmfit.model.ModelResult):
        raise TypeError("m1 must be %s, instead it is %s", lmfit.model.ModelResult, type(m1))
    if not isinstance(m2, lmfit.model.ModelResult):
        raise TypeError("m2 must be %s, instead it is %s", lmfit.model.ModelResult, type(m2))

    if t is None:
        t = np.linspace(0, hours, num_of_points)

    if nsamples > 1:
        # draw random params
        sampler = sampler.lower()
        if sampler == 'covar':
            m1_samples = curveball.models.sample_params(m1, nsamples, params=params1)
            m2_samples = curveball.models.sample_params(m2, nsamples, params=params2)
        elif sampler == 'bootstrap':
            if params1 or params2:
                warnings.warn("Bootstrap sampling doesn't support params1 and params2 arguments")
            if df1 is None or df2 is None:
                raise ValueError("Bootstrap sampling requires kwargs df1 and df2")
            m1_fixed = {pname for pname,p in m1.params.items() if not p.vary}
            m2_fixed = {pname for pname,p in m2.params.items() if not p.vary}
            # FIXME bootstrap_params has changed
            m1_samples = curveball.models.bootstrap_params(df1, m1, nsamples,
                                                           fit_kws={'param_fix': m1_fixed})
            m2_samples = curveball.models.bootstrap_params(df2, m2, nsamples,
                                                           fit_kws={'param_fix': m2_fixed})
        else:
            raise ValueError("Unknow sampler method: {0}".format(sampler))
        min_nsamples = min(len(m1_samples), len(m2_samples))
        if nsamples > min_nsamples:
            warnings.warn("{0} resamples lost".format(nsamples - min_nsamples))
            nsamples = min_nsamples
    else:
        nsamples = 1
        # override model result params with arguments params1 and params2
        if params1:
            _params = copy.copy(m1.best_values)
            _params.update(params1)
            params1 = _params
        else:
            params1 = m1.best_values
        if params2:
            _params = copy.copy(m2.best_values)
            _params.update(params2)
            params2 = _params
        else:
            params2 = m2.best_values
        # param samples contain the model fit estimated params
        m1_samples = pd.DataFrame([params1])
        m2_samples = pd.DataFrame([params2])
        assert len(m1_samples) == len(m2_samples)

    y = np.empty((len(t), 2, nsamples))
    #infodict = [None]*nsamples # DEBUG

    # simulate the ode for each param sample
    for i in range(nsamples):
        if y0 is None:
            p0 = p0[0] / (p0[0] + p0[1]), p0[1] / (p0[0] + p0[1])
            y0 = m1_samples.iloc[i]['y0'] * p0[0], m2_samples.iloc[i]['y0'] * p0[1]
        r = m1_samples.iloc[i]['r'], m2_samples.iloc[i]['r']
        K = m1_samples.iloc[i]['K'], m2_samples.iloc[i]['K']
        nu = m1_samples.iloc[i].get('nu', 1.0), m2_samples.iloc[i].get('nu', 1.0)
        if lag_phase:
            q0 = m1_samples.iloc[i].get('q0', np.inf), m2_samples.iloc[i].get('q0', np.inf)
            v = m1_samples.iloc[i].get('v', r[0]), m2_samples.iloc[i].get('v', r[1])
        else:
            q0 = np.inf, np.inf
            v = np.inf, np.inf

        args = (r, K, nu, q0, v)
        y[:,:,i] = odeint(ode, y0=y0, t=t, args=args)

        # DEBUG
        #_y_,info = odeint(double_baranyi_roberts_ode0, y0, t, args=args, full_output=1)
        #info['args'] = (y0,) + args
        #infodict[i] = info
        #if info['message'] == 'Integration successful.':
        #   y[:,:,i] = _y_

    if PLOT:
        if ax is None:
            fig,ax = plt.subplots(1, 1)
        else:
            fig = ax.get_figure()

        df = pd.DataFrame()
        for i in range(y.shape[1]):
            _df = pd.DataFrame(y[:,i,:])
            _df['Time'] = t
            _df = pd.melt(_df, id_vars='Time', var_name='Replicate', value_name='y')
            _df['Strain'] = i
            df = pd.concat((df, _df))

        if colors is not None:
            colors = {i:c for i,c in enumerate(colors)}
        sns.lineplot(data=df, x='Time', hue='Strain', y='y',
                        ci=ci, palette=colors, ax=ax)
        ax.set_xlabel('Time (hour)')
        ax.set_ylabel('OD')
        sns.despine()
        return t,y,fig,ax
    return t,y#,infodict


def selection_coefs_ts(t, y, ax=None, PLOT=False):
    r"""Calculate selection coefficient according to the following formula[2]_,
    where :math:`A(t), B(t)` are population densities of assay strain *A* and reference strain *B* at time *t*:

    .. math::

        s = \frac{d}{dt} \log{\frac{A(t)}{B(t)}}

    Parameters
    ----------
    t : numpy.ndarray
        array of time points, as produced by :py:func:`compete`
    y : numpy.ndarray
        array of population densities, as produced by :py:func:`compete`, where the first axis is time and the second axis is strain.
    ax : matplotlib.axes.Axes, optional
        if `PLOT` is :const:`True`, an axes to plot into; if not provided, a new one is created.
    PLOT : bool, optional
        if :const:`True`, the function will plot the curve of *s* as a function of *t*.

    Returns
    -------
    svals : numpy.ndarray
        the selection coefficients of the assay strain relative to the reference strain over time.
    fig : matplotlib.figure.Figure
        figure object.
    ax : numpy.ndarray of matplotlib.axes.Axes
        array of axes objects.

    Notes
    -----
    This formula assumes that the frequencies of the strains follow a logistic curve.
    Lag phases, interactions, etc. may cause this formula to become irrelevant.

    References
    ----------
    .. [12] Chevin, L-M. 2011. `On Measuring Selection in Experimental Evolution <http://dx.doi.org/10.1098/rsbl.2010.0580>`_. Biology Letters.
    """
    dt = np.ediff1d(t, to_end=1)
    dt[-1] = dt[-2]
    svals = np.gradient(np.log(y[:,0] / y[:,1]), dt)
    svals[np.isinf(svals)] = svals[np.isfinite(svals)].max()

    if PLOT:
        if ax is None:
            fig,ax = plt.subplots(1,1)
        else:
            fig = ax.get_figure()
        ax.plot(t, svals)
        ax.set_ylabel('Selection coefficient')
        ax.set_xlabel('Time (hour)')
        sns.despine()
        return svals, fig, ax

    return svals


def fitness_LTEE(y, ref_strain=0, assay_strain=1, t0=0, t1=-1, ci=0):
    r"""Calculate relative fitness according to the definition used in the *Long Term Evolutionary Experiment* (LTEE) [3]_,
    where :math:`A(t), B(t)` are population densities of assay strain *A* and reference strain *B* at time *t*:

    .. math::

        \omega = \frac{\log{(A(t) / A(0))}}{\log{(B(t) / B(0))}}

    If `ci` > 0, treats the third axis of `y` as replicates (from a parameteric boostrap) to calculate the confidence interval on the fitness.

    Parameters
    ----------
    y : numpy.ndarray
        array of population densities, as produced by :py:func:`compete`, where the first axis is time and the second axis is strain. A third axis is also applicable if `ci`>0.
    ref_strain : int, optional
        the index of the reference strain within `y`. This strain's fitness is set to 1 by definition. Defaults to 0 (first).
    assay_strain : int, optional
        the index of the assay strain within `y`. The result will be the fitness of this strain relative to the fitness of the reference strain. Defaults to 1 (second).
    t0 : int, optional
        the index of the time point from which to start the calculation of the relative fitness, defaults to 0 (first).
    t1 : int, optional
        the index of the time point at which to end the calculation of the relative fitness, defaults to -1 (last).
    ci : float between 0 and 1, optional
        if not zero, a confidence interval will be calculated using the third axis of `y` as replicates.

    Returns
    -------
    w : float
        the fitness of the assay strain relative to the reference strain.
    low : float
        if `ci` > 0 and `y.ndim` = 3, this is the low limit of the confidence interval of the fitness
    high : float
        if `ci` > 0 and `y.ndim` = 3, this is the higher limit of the confidence interval of the fitness

    Raises
    ------
    ValueError
        if confidence interval is requested (`ci` > 0) but there are no replicates (`y.ndim` != 3).

    Notes
    -----
    The result may depend on the choice of `t0` and `t1` as well as the strain designations (`ref_strain` and `assay_strain`).

    References
    ----------
    .. [3] Wiser, M. J. & Lenski, R. E. 2015 `A Comparison of Methods to Measure Fitness in Escherichia coli <http://dx.plos.org/10.1371/journal.pone.0126210>`_. PLoS One.
    """
    if ci == 0:
        y = y.reshape(y.shape + (1,))
    elif y.ndim != 3:
        raise ValueError()
    w = np.zeros(y.shape[2])
    for i in range(y.shape[2]):
        At0, Bt0 = y[t0, assay_strain, i], y[t0, ref_strain, i]
        At1, Bt1 = y[t1, assay_strain, i], y[t1, ref_strain, i]
        w[i] = np.log(At1 / At0) / np.log(Bt1 / Bt0)

    if ci == 0:
        return w.mean()
    else:
        margin = (1 - ci) * 50
        return w.mean(), np.percentile(w, margin), np.percentile(w, ci * 100 + margin)


def baranyi_roberts_gd(y, t, *args):
    r"""
    Fujikawa, Hiroshi, and Mohammad Z Sakha. 2014. Prediction of Microbial Growth in Mixed Culture with a Competition Model. Biocontrol Science 19 (2): 89-92.
    """
    y1, y2 = y
    K, r, nu, q0, v, a = args
    r1, r2 = r
    K1, K2 = K
    nu1, nu2 = nu
    q01, q02 = q0
    v1, v2 = v
    a1, a2 = a
    alfa1 = _alfa(t, q01, v1)
    alfa2 = _alfa(t, q02, v2)
    dy1dt = r1 * alfa1 * y1 * (1 - (y1/K1)**nu1) * (1 - (y2/K2)**(nu2*a2))
    dy2dt = r2 * alfa2 * y2 * (1 - (y1/K1)**(nu1*a1)) * (1 - (y2/K2)**nu2)
    return [dy1dt, dy2dt]


def baranyi_roberts_lv(y, t, *args):
    r"""
    Fujikawa, Hiroshi, and Mohammad Z Sakha. 2014. Prediction of Microbial Growth in Mixed Culture with a Competition Model. Biocontrol Science 19 (2): 89-92.
    """
    y1, y2 = y
    K, r, nu, q0, v, a = args
    r1, r2 = r
    K1, K2 = K
    nu1, nu2 = nu
    q01, q02 = q0
    v1, v2 = v
    a1, a2 = a
    alfa1 = _alfa(t, q01, v1)
    alfa2 = _alfa(t, q02, v2)
    Kmax = max(K1**nu1, K2**nu2)
    dy1dt = r1 * alfa1 * y1 * (1 - (y1/K1)**nu1) * (1 - (y1**a1 + y2**a2)/Kmax)
    dy2dt = r2 * alfa2 * y2 * (1 - (y2/K2)**nu2) * (1 - (y1**a1 + y2**a2)/Kmax)
    return [dy1dt, dy2dt]


def baranyi_roberts_yr2(y, t, *args):
    y1, y2 = y
    K, r, nu, q0, v, a = args
    r1, r2 = r
    K1, K2 = K
    nu1, nu2 = nu
    q01, q02 = q0
    v1, v2 = v
    a1, a2 = a
    alfa1 = _alfa(t, q01, v1)
    alfa2 = _alfa(t, q02, v2)
    dy1dt = r1 * alfa1 * y1 * (1 - (y1/K1)**nu1 - (a2 * y2/K2)**nu2)
    dy2dt = r2 * alfa2 * y2 * (1 - (a1 * y1/K1)**nu1 - (y2/K2)**nu2)
    return [dy1dt, dy2dt]


def baranyi_roberts_yr(y, t, *args):
    y1, y2 = y
    K, r, nu, q0, v, a = args
    r1, r2 = r
    K1, K2 = K
    nu1, nu2 = nu
    q01, q02 = q0
    v1, v2 = v
    a1, a2 = a
    alfa1 = _alfa(t, q01, v1)
    alfa2 = _alfa(t, q02, v2)
    dy1dt = r1 * alfa1 * y1 * (1 - (y1**nu1) / (K1**nu1) - a2 * (y2**nu2) / (K1**nu1))
    dy2dt = r2 * alfa2 * y2 * (1 - a1 * (y1**nu1) / (K2**nu2) - (y2**nu2) / (K2**nu2))
    return [dy1dt, dy2dt]

# obsolete - just call baranyi_roberts_yr(y, t, *args, (1, 1))
# def baranyi_roberts_yr_a1(y, t, *args):
#     return baranyi_roberts_yr(y, t, *args, (1, 1))


def fit_and_compete(m1, m2, df_mixed, y0=None, aguess=(1, 1), fixed=False,
                    ode=baranyi_roberts_yr, num_of_points=100, method='nelder',
                    value_var = 'OD', time_var = 'Time',
                    PLOT=False, colors=sns.color_palette('Set1', 3)):
    best_values1 = m1.best_values if hasattr(m1, 'best_values') else m1
    best_values2 = m2.best_values if hasattr(m2, 'best_values') else m2
    K = best_values1['K'], best_values2['K']
    r = best_values1['r'], best_values2['r']
    nu = best_values1.get('nu',1), best_values2.get('nu',1)
    q0 = best_values1.get('q0', np.inf), best_values2.get('q0', np.inf)
    v = best_values1.get('v', r[0]), best_values2.get('v', r[1])
    if y0 is None:
        y0 = best_values1['y0']/2, best_values2['y0']/2

    y_mixed = df_mixed.groupby(time_var)[value_var].mean().values
    y_mixed_std = df_mixed.groupby(time_var)[value_var].std().values
    t_mixed = np.unique(df_mixed[time_var])
    t = np.linspace(0, t_mixed.max(), num_of_points)

    if fixed:
        a = aguess
        a1, a2 = a
    else:
        def mixed_model(t, a1, a2):
            y = odeint(ode, y0, t, args=(K, r, nu, q0, v, (a1, a2)))
            return y.sum(axis=1)
        model = lmfit.Model(mixed_model)

        params = model.make_params(a1=aguess[0], a2=aguess[1])
        params['a1'].set(min=1e-1, max=10)
        params['a2'].set(min=1e-1, max=10)
        
        result = model.fit(data=y_mixed, t=t_mixed, params=params, weights=1.0 / y_mixed_std, 
            method=method, nan_policy='propagate')

        a = result.best_values['a1'], result.best_values['a2']
        a1, a2 = a

    y = odeint(ode, y0, t, args=(K, r, nu, q0, v, a))

    if PLOT:
        ysum = y.sum(axis=1)
        p1 = y[:, 0] / ysum
        p2 = y[:, 1] / ysum
        MRSE = ((odeint(ode, y0, t_mixed, args=(K, r, nu, q0, v, a)).sum(axis=1) - y_mixed)**2).mean()

        w, h = plt.rcParams['figure.figsize']
        fig, ax = plt.subplots(1, 2, figsize=(w * 2, h), sharex=True)


        ax[0].plot(t, y[:, 0], color=colors[0])
        ax[0].plot(t, y[:, 1], color=colors[1])
        ax[0].plot(t, ysum, color=colors[2])
        if not fixed:
            ax[0].plot(t_mixed, result.init_fit, ls='-.', color=colors[2])
        ax[0].errorbar(t_mixed, y_mixed, y_mixed_std, fmt='o', color=colors[2])

        ax[0].set(xlabel='Time (hour)', ylabel='OD',
                  xlim=(-0.5, t.max() + 0.5),
                  title="MRSE: {:.2g}".format(MRSE))

        ax[1].plot(t, p1, color=colors[0])
        ax[1].plot(t, p2, color=colors[1])

        ax[1].set(xlabel='Time (hour)', ylabel='Frequency',
                  xlim=(-0.5, t.max() + 0.5), ylim=(0, 1),
                  title="a1={:.2g}, a2={:.2g}".format(*a))

        sns.despine()
        fig.tight_layout()
        return t, y, a, fig, ax
    return t, y, a


def fit_and_compete_ci(param_samples1, param_samples2, df_mixed, y0=None, ci=0.95,
    colors=('g', 'r'), line_kws=None, ci_kws=None, ax=None, PLOT=False):
    if line_kws is None:
        line_kws = dict()
    if ci_kws is None:
        ci_kws = dict(color='gray', alpha=0.5)
    nsamples = param_samples1.shape[0]
    assert param_samples1.shape[0] == param_samples2.shape[0], "Parameters samples should have the same length"
    t = [None] * nsamples
    y = [None] * nsamples
    a = [None] * nsamples
    for i in range(nsamples):
        sample1 = param_samples1.iloc[i, :]
        sample2 = param_samples2.iloc[i, :]
        t[i], y[i], a[i] = curveball.competitions.fit_and_compete(sample1, sample2, df_mixed, y0=y0)

    t = np.array(t)
    y = np.array(y)
    a = np.array(a)
    ysum = y.sum(axis=2)
    f1 = y[:, :, 0] / ysum
    f2 = y[:, :, 1] / ysum
    f = np.array((f1, f2)).T

    margin = (1.0 - ci) * 50.0

    low_f = np.percentile(f, margin, axis=1)
    high_f = np.percentile(f, ci * 100.0 + margin, axis=1)
    avg_f = f.mean(axis=1)

    low_a = np.percentile(a, margin, axis=0)
    avg_a = a.mean(axis=0)
    high_a = np.percentile(a, ci * 100.0 + margin, axis=0)

    if PLOT:
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure
        ax.fill_between(t[0, :], low_f[:, 0], high_f[:, 0], **ci_kws)
        ax.plot(t[0,:], avg_f[:, 0], color=colors[0], **line_kws)

        ax.fill_between(t[0, :], low_f[:, 1], high_f[:, 1], **ci_kws)
        ax.plot(t[0,:], avg_f[:, 1], color=colors[1], **line_kws)

        ax.set(xlabel='Time', ylabel='Frequency')
        fig.tight_layout()
        sns.despine()
        return low_a, avg_a, high_a, low_f, avg_f, high_f, fig, ax

    return low_a, avg_a, high_a, low_f, avg_f, high_f