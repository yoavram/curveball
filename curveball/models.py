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
import sys
import numbers
from warnings import warn
import numpy as np
import matplotlib.pyplot as plt
import collections
try:
	from scipy.stats.distributions import chi2
	chisqprob = chi2.sf
except ImportError:
	from scipy.stats import chisqprob
from scipy.stats import linregress
from scipy.misc import derivative
import pandas as pd
import copy
import inspect
import lmfit
import sympy
import seaborn as sns
sns.set_style("ticks")
import curveball
import curveball.baranyi_roberts_model
from curveball.utils import smooth


def is_model(cls):
    """Returns :const:`True` if the input is a subclass of :py:class:`lmfit.model.Model`.

    Parameters
    ----------
    cls : class

    Returns
    -------
    bool
    """
    return inspect.isclass(cls) and issubclass(cls, lmfit.model.Model)


def get_models(module):
    """Finds and returns all models in `module`.

    Parameters
    ----------
    module : module
        the module in which to look for models

    Returns
    -------
    list
        list of subclasses of :py:class:`lmfit.model.Model` that can be used with :py:func:`curveball.models.fit_model`

    See also
    --------
    fit_model
    is_model
    curveball.baranyi_roberts_model
    """
    return [m[1] for m in inspect.getmembers(module, curveball.models.is_model)]


def bootstrap_params(df, model_result, nsamples, unit='Well', fit_kws=None):
    """Sample model parameters by fitting the model to resampled data.

    The data is bootstraped by drawing growth curves (sampling from the `unit` column in `df`) with replacement.

    Parameters
    ----------
    df : pandas.DataFrame
        the data to fit
    model_result : lmfit.model.ModelResult
        result of fitting a :py:class:`lmfit.model.Model` to data in ``df``
    nsamples : int
        number of samples to draw
    unit : str, optional
        the name of the column in `df` that identifies a resampling unit, defaults to ``Well``
    fit_kws : dict, optional
        dict of kwargs for `fit_model`
    Returns
    -------
    pandas.DataFrame
        data frame of samples; each row is a sample, each column is a parameter.

    Raises
    ------
    ValueError : if `model_result` isn't an instance of :py:class:`lmfit.model.ModelResult`
    ValueError : if `df` is empty

    See also
    --------
    sample_params
    """
    if not isinstance(model_result, lmfit.model.ModelResult):
        raise TypeError(
            "Input model_class must be a {0}, but it is {1}".format(
                lmfit.model.ModelResult.__name__, 
                model_result.__class__.__name__))
    if df.empty:
        raise ValueError("Input data frame df is empty")
        
    if fit_kws is None:
        fit_kws = {}
    if not 'param_fix' in fit_kws:
        fit_kws['param_fix'] = {pname for pname, param in model_result.params.items() if not param.vary}
    if not 'param_max' in fit_kws:
        fit_kws['param_max'] = {pname: param.max for pname, param in model_result.params.items()}
    if not 'param_max' in fit_kws:
        fit_kws['param_min'] = {pname: param.min for pname, param in model_result.params.items()}
    model_class = type(model_result.model)

    unique_units = pd.Series(df[unit].unique())
    grouped = df.groupby(unit)
    param_samples = [None] * nsamples

    for i in range(nsamples):
        sampled_units = unique_units.sample(frac=1, replace=True)
        _df = pd.concat(grouped.get_group(grp_id) for grp_id in sampled_units.values)
        assert (_df[unit].unique() == sampled_units.unique()).all()
        assert _df.shape == df.shape    
        model_fit = curveball.models.fit_model(_df, models=model_class, 
                                               PLOT=False, PRINT=False, 
                                               **fit_kws)[0]
        param_samples[i] = model_fit.best_values
    return pd.DataFrame(param_samples)


def sample_params(model_fit, nsamples, params=None, covar=None):
    """Random sample of parameter values from a truncated multivariate normal distribution defined by the 
    covariance matrix of the a model fitting result.

    Parameters
    ----------
    model_fit : lmfit.model.ModelResult
        the model fit that defines the sampled distribution
    nsamples : int
        number of samples to make
    params : dict, optional
        a dictionary of model parameter values; if given, overrides values from `model_fit`
    covar : numpy.ndarray, optional
        an array containing the parameters covariance matrix; if given, overrides values from `model_fit`

    Returns
    -------
    pandas.DataFrame
        data frame of samples; each row is a sample, each column is a parameter.

    See also
    --------
    bootstrap_params
    """
    if params is None:
        params = model_fit.params
    else:
        _params = model_fit.params.copy()
        for pname, pvalue in params.items():
            _params[pname].value = pvalue
        params = _params
    if covar is None:
        covar = model_fit.covar
    if covar is None:
        raise ValueError("Covariance matrix for {0} is invalid (None).".format(model_fit.model))
    if covar.ndim != 2:
        warn("Covariance matrix doesn't have 2 dimensions: \n{}".format(covar))
    w, h = covar.shape
    if w != h:
        warn("Covariance matrix is not square: \n{}".format(covar))
    names = [p.name for p in params.values() if p.vary]
    means = [p.value for p in params.values() if p.vary]
        
    param_samples = np.random.multivariate_normal(means, covar, nsamples)
    param_samples = pd.DataFrame(param_samples, columns=names)
    idx = np.zeros(nsamples) == 0
    for p in params.values():
        if not p.vary:
            param_samples[p.name] = p.value
        idx = idx & (param_samples[p.name] >= p.min) & (param_samples[p.name] <= p.max)
    if param_samples.shape[0] < nsamples:
        warn("Warning: truncated {0} parameter samples; please report at {1}, including the data and use case.".format(nsamples - param_samples.shape[0], "https://github.com/yoavram/curveball/issues"))
    param_samples = param_samples[idx]
    return param_samples
    


def noisify_normal_additive(data, std, rng=None):
    if not rng:
        rng = np.random
    return data +  rng.normal(0, std, data.shape)


def noisify_lognormal_multiplicative(data, std, random_seed=None):
    if random_seed is None:
        rng = np.random
    else:
        rng = np.random.RandomState(random_seed)
    return data * rng.lognormal(0, std, data.shape)


def randomize(t=12, y0=0.1, K=1.0, r=0.1, nu=1.0, q0=np.inf, v=np.inf, 
        func=curveball.baranyi_roberts_model.baranyi_roberts_function, reps=1, noise_std=0.02, 
        noise_func=noisify_lognormal_multiplicative, random_seed=None, as_df=True, 
        data_label='OD', time_label='Time', replicate_label='Well'):
    if isinstance(t, numbers.Number):
        t = np.linspace(0, t)
    y = func(t, y0, K, r, nu, q0, v)
    y.resize((len(t),))
    y = y.repeat(reps)
    y.resize((len(t), reps))
    well = np.arange(reps).repeat(len(t))#.resize(len(t), reps)
    if noise_std > 0:
        y = noise_func(y, noise_std, random_seed)
    y[y < 0] = 0.0
    y = y.flatten()
    t = t.repeat(reps)
    well = well.flatten()
    if as_df:
        return pd.DataFrame({data_label: y, time_label: t, replicate_label: well})
    else:
        return t, y


def lrtest(m0, m1, alfa=0.05):
    r"""Performs a likelihood ratio test on two nested models.

    For two models, one nested in the other 
    (meaning that the nested model estimated parameters are a subset of the nesting model), 
    the test statistic :math:`D` is:

    .. math::

        \Lambda = \Big( \Big(\frac{\sum{(X_i - \hat{X_i}(\theta_1))^2}}{\sum{(X_i - \hat{X_i}(\theta_0))^2}}\Big)^{n/2} \Big)

        D = -2 log \Lambda

        lim_{n \to \infty} D \sim \chi^2_{df=\Delta}


    where :math:`\Lambda` is the likelihood ratio, :math:`D` is the statistic, 
    :math:`X_{i}` are the data points, :math:`\hat{X_i}(\theta)` is the 
    model prediction with parameters :math:`\theta`, 
    :math:`\theta_i` is the parameters estimation for model :math:`i`, 
    :math:`n` is the number of data points, and :math:`\Delta` is the 
    difference in number of parameters between the models.

    The function compares between two :py:class:`lmfit.model.ModelResult` objects. 
    These are the results of fitting models to the same dataset 
    using the `lmfit <lmfit.github.io/lmfit-py>`_ package.

    The function compares between model fit `m0` and `m1` and assumes that `m0` is nested in `m1`, 
    meaning that the set of varying parameters of `m0` is a subset of the varying parameters of `m1`. 
    :py:attr:`lmfit.model.ModelResult.chisqr` is the sum of the square of the residuals of the fit. 
    :py:attr:`lmfit.model.ModelResult.ndata` is the number of data points. 
    :py:attr:`lmfit.model.ModelResult.nvarys` is the number of varying parameters.

    Parameters
    ----------
    m0, m1 : lmfit.model.ModelResult
        objects representing two model fitting results. `m0` is assumed to be nested in `m1`.
    alfa : float, optional
        test significance level, defaults to 0.05 = 5%.

    Returns
    -------
    prefer_m1 : bool
        should we prefer `m1` over `m0`
    pval : float
        the test p-value
    D : float
        the test statistic
    ddf : int 
        the number of degrees of freedom

    Notes
    ----------
    - `Generalised Likelihood Ratio Test Example <http://www.stat.sc.edu/~habing/courses/703/GLRTExample.pdf>`_.
    - `IPython notebook <http://nbviewer.ipython.org/github/yoavram/ipython-notebooks/blob/master/likelihood%20ratio%20test.ipynb>`_.
    """
    n0 = m0.ndata
    k0 = m0.nvarys
    chisqr0 = m0.chisqr
    assert chisqr0 > 0, chisqr0
    n1 = m1.ndata
    assert n0 == n1
    k1 = m1.nvarys
    chisqr1 = m1.chisqr
    assert chisqr1 > 0, chisqr1
    D = -n0 * (np.log(m1.chisqr) - np.log(m0.chisqr))
    assert D > 0, D
    ddf = k1 - k0
    assert ddf > 0, ddf
    pval = chisqprob(D, ddf)
    prefer_m1 = pval < alfa
    return prefer_m1, pval, D, ddf


def find_max_growth(model_fit, params=None, after_lag=True):
    r"""Estimates the maximum population and specific growth rates from the model fit.

    The function calculates the maximum population growth rate :math:`a=\max{\frac{dy}{dt}}` 
    as the derivative of the model curve and calculates its maximum. 
    It also calculates the maximum of the per capita growth rate :math:`\mu = \max{\frac{dy}{y \cdot dt}}`.
    The latter is more useful as a metric to compare different strains or treatments 
    as it does not depend on the population size/density.    

    For example, in the logistic model the population growth rate is a quadratic function of :math:`y` 
    so the maximum is realized when the 2nd derivative is zero:

    .. math::

        \frac{dy}{dt} = r y (1 - \frac{y}{K}) \Rightarrow

        \frac{d^2 y}{dt^2} = r - 2\frac{r}{K}y \Rightarrow

        \frac{d^2 y}{dt^2} = 0 \Rightarrow y = \frac{K}{2} \Rightarrow

        \max{\frac{dy}{dt}} = \frac{r K}{4}

    In contrast, the per capita growth rate a linear function of :math:`y` 
    and so its maximum is realized when :math:`y=y_0`:

    .. math::

        \frac{dy}{y \cdot dt} = r (1 - \frac{y}{K}) \Rightarrow

        \max{\frac{dy}{y \cdot dt}} = \frac{dy}{y \cdot dt}(y=y_0) = r (1 - \frac{y_0}{K})     

    Parameters
    ----------
    model_fit : lmfit.model.ModelResult
        the result of a model fitting procedure
    params : lmfit.parameter.Parameters, optional
        if provided, these parameters will override `model_fit`'s parameters
    after_lag : bool
        if true, only explore the time after the lag phase. Otherwise start at time zero. Defaults to :const:`True`.

    Returns
    -------
    t1 : float
        the time when the maximum population growth rate is achieved in the units of the `model_fit` ``Time`` variable.
    y1 : float
        the population size or density (OD) for which the maximum population growth rate is achieved.
    a : float
        the maximum population growth rate.
    t2 : float
        the time when the maximum per capita growth rate is achieved in the units of the `model_fit` `Time` variable.
    y2 : float
        the population size or density (OD) for which the maximum per capita growth rate is achieved.
    mu : float
        the the maximum specific (per capita) growth rate.

    See also
    --------
    find_max_growth_ci
    """
    if params is None:
        params = model_fit.params

    y0 = params['y0'].value
    K  = params['K'].value

    t0 = find_lag(model_fit) if after_lag else 0
    t0 = max(t0, 0)
    t1 = model_fit.userkws['t'].max()
    t = np.linspace(t0, t1)     
    def f(t): 
        return model_fit.model.eval(t=t, params=params)
    y = f(t)
    dfdt = derivative(f, t)

    a = dfdt.max()
    i = dfdt.argmax()
    t1 = t[i]
    y1 = y[i]

    dfdt_y = dfdt / y
    mu = dfdt_y.max()
    i = dfdt_y.argmax()
    t2 = t[i]
    y2 = y[i]
    
    return t1, y1, a, t2, y2, mu


def find_max_growth_ci(model_fit, param_samples, after_lag=True, ci=0.95):
    """Estimates a confidence interval for the maximum population/specific growth rates from the model fit.
    
    The maximum population/specific growth rate for each parameter sample is calculated.
    The confidence interval of the rate is the lower and higher percentiles such that 
    `ci` percent of the random rates are within the confidence interval.

    Parameters
    ----------
    model_fit : lmfit.model.ModelResult
        the result of a model fitting procedure
    param_samples : pandas.DataFrame
        parameter samples, generated using :function:`sample_params` or :function:`bootstrap_params`
    after_lag : bool
        if true, only explore the time after the lag phase. Otherwise start at time zero. Defaults to :const:`True`        
    ci : float, optional
        the fraction of lag durations that should be within the calculated limits. 0 < `ci` <, defaults to 0.95
    
    Returns
    -------
    low_a, high_a : float
        the lower and the higher boundaries of the confidence interval of the the maximum population growth rate in the units of the `model_fit` ``OD``/``Time`` (usually OD/hours).
    low_mu, high_mu : float
        the lower and the higher boundaries of the confidence interval of the the maximum specific growth rate in the units of the `model_fit` 1/``Time`` variable (usually 1/hours).


    See also
    --------
    find_max_growth
    """
    if not 0 <= ci <= 1:
        raise ValueError("ci must be between 0 and 1")
    nsamples = param_samples.shape[0]
    aa = np.zeros(nsamples)
    mumu = np.zeros(nsamples)        
    for i in range(param_samples.shape[0]):
        sample = param_samples.iloc[i,:]
        params = model_fit.params.copy()
        for k,v in params.items():
            if v.vary:
                params[k].set(value=sample[k])
        _, _, a, _, _, mu = find_max_growth(model_fit, params=params, after_lag=after_lag)
        aa[i] = a
        mumu[i] = mu

    margin = (1.0 - ci) * 50.0

    idx = np.isfinite(aa) & (aa >= 0)
    if not idx.all():
        warn("Warning: omitting {0} non-finite growth rate values".format(len(aa) - idx.sum()))
    aa = aa[idx]
    low_a = np.percentile(aa, margin)
    high_a = np.percentile(aa, ci * 100.0 + margin)
    assert high_a >= low_a, aa.tolist()

    idx = np.isfinite(mumu) & (mumu >= 0)
    if not idx.all():
        warn("Warning: omitting {0} non-finite growth rate values".format(len(mumu) - idx.sum()))
    mumu = mumu[idx]
    low_mu = np.percentile(mumu, margin)
    high_mu = np.percentile(mumu, ci * 100.0 + margin)
    assert high_mu >= low_mu, mumu.tolist()
    return low_a, high_a, low_mu, high_mu


def find_min_doubling_time(model_fit, params=None, PLOT=False):
    """Estimates the minimal doubling time from the model fit.

    The function evaluates a growth curves based on the model fit and supplied parameters (if any) 
    and calculates that time required to double the population density at each time point. 
    It then returns the minimal doubling time.

    Parameters
    ----------
    model_fit : lmfit.model.ModelResult
        the result of a model fitting procedure
    params : lmfit.parameter.Parameters, optional
        if provided, these parameters will override `model_fit`'s parameters
    PLOT : bool, optional
        if :const:`True`, the function will plot the Cook's distances of the wells and the threshold. # FIXME

    Returns
    -------
    min_dbl : float
        the minimal time it takes the population density to double.
    fig : matplotlib.figure.Figure
        if the argument `PLOT` was :const:`True`, the generated figure.
    ax : matplotlib.axes.Axes
        if the argument `PLOT` was :const:`True`, the generated axis.
    """
    if params is None:
        params = model_fit.params
    def f(t): 
        return model_fit.model.eval(t=t, params=params)

    t1 = model_fit.userkws['t'].max()
    t = np.linspace(0, t1, 1000)
    y = f(t)

    def find_point(point):
        return abs(y - point).argmin()

    imax = find_point(y.max() / 2) + 1
    doubling_times = np.zeros(imax)
    for i0, y0 in enumerate(y[:imax]):
        i1 = find_point(2 * y0)
        doubling_times[i0] = t[i1] - t[i0]
    min_dbl = doubling_times.min()

    if PLOT:
        fig, ax = plt.subplots(1, 1)
        ax.plot(t[:imax], doubling_times, '-')
        ax.axhline(min_dbl, ls='--', color='k')
        ax.set(xlabel='Time', ylabel='Doubling time')
        sns.despine()
        return min_dbl, fig, ax

    return min_dbl


def find_min_doubling_time_ci(model_fit, param_samples, ci=0.95):
    """Estimates a confidence interval for the minimal doubling time from the model fit.

    The minimal doubling time for each parameter sample is calculated.
    The confidence interval of the doubling time is the lower and higher percentiles such that 
    `ci` percent of the random lag durations are within the confidence interval.

    Parameters
    ----------
    model_fit : lmfit.model.ModelResult
        the result of a model fitting procedure
    param_samples : pandas.DataFrame
        parameter samples, generated using :function:`sample_params` or :function:`bootstrap_params`    
    ci : float, optional
        the fraction of doubling times that should be within the calculated limits. 0 < `ci` <, defaults to 0.95.
    
    Returns
    -------
    low, high : float
        the lower and the higher boundaries of the confidence interval of the minimal doubling times in the units of the `model_fit` ``Time`` variable (usually hours).

    See also
    --------
    find_min_doubling_time
    """
    if not 0 <= ci <= 1:
        raise ValueError("ci must be between 0 and 1")
    nsamples = param_samples.shape[0]
    dbls = np.zeros(nsamples)    
    for i in range(param_samples.shape[0]):
        sample = param_samples.iloc[i,:]
        params = model_fit.params.copy()
        for k, v in params.items():
            if v.vary:
                v.value = sample[k]
        dbls[i] = find_min_doubling_time(model_fit, params=params)

    margin = (1.0 - ci) * 50.0
    idx = np.isfinite(dbls) & (dbls >= 0)
    if not idx.all():
        warn("Warning: omitting {0} non-finite doubling time values".format(len(dbls) - idx.sum()))
    dbls = dbls[idx]
    low = np.percentile(dbls, margin)
    high = np.percentile(dbls, ci * 100.0 + margin)
    assert high >= low, dbls.tolist()
    return low, high


def find_K_ci(param_samples, ci=0.95):
    """Estimates a confidence interval for ``K``, the maximum population density from the model fit.

    The confidence interval of the doubling time is the lower and higher percentiles such that 
    `ci` percent of the max densities are within the confidence interval.

    Parameters
    ----------
    param_samples : pandas.DataFrame
        parameter samples, generated using :function:`sample_params` or :function:`bootstrap_params`    
    ci : float, optional
        the fraction of doubling times that should be within the calculated limits. 0 < `ci` <, defaults to 0.95.
    
    Returns
    -------
    low, high : float
        the lower and the higher boundaries of the confidence interval of the maximum population density.
    
    """
    if not 0 <= ci <= 1:
        raise ValueError("ci must be between 0 and 1")    
    Ks = param_samples['K']    
    
    margin = (1.0 - ci) * 50.0
    idx = np.isfinite(Ks) & (Ks >= 0)
    if not idx.all():
        warn("Warning: omitting {0} non-finite K values".format(len(Ks) - idx.sum()))
    Ks = Ks[idx]
    low = np.percentile(Ks, margin)
    high = np.percentile(Ks, ci * 100.0 + margin)
    assert high >= low, Ks.tolist()
    return low, high


def find_lag(model_fit, params=None):
    """Estimates the lag duration from the model fit.

    The function calculates the tangent line to the model curve at the point of maximum derivative (the inflection point). 
    The time when this line intersects with :math:`N_0` (the initial population size) 
    is labeled :math:`\lambda` and is called the lag duration time [fig2.2]_.

    Parameters
    ----------
    model_fit : lmfit.model.ModelResult
        the result of a model fitting procedure
    params : lmfit.parameter.Parameters, optional
        if provided, these parameters will override `model_fit`'s parameters

    Returns
    -------
    lam : float
        the lag phase duration in the units of the `model_fit` ``Time`` variable (usually hours).    

    References
    ----------
    .. [fig2.2] Fig. 2.2 pg. 19 in Baranyi, J., 2010. `Modelling and parameter estimation of bacterial growth with distributed lag time. <http://www2.sci.u-szeged.hu/fokozatok/PDF/Baranyi_Jozsef/Disszertacio.pdf>`_.

    See also
    --------
    find_lag_ci
    has_lag
    """
    if params is None:
        params = model_fit.params
    
    y0 = params['y0'].value
    K  = params['K'].value

    t = model_fit.userkws['t']
    t = np.linspace(t.min(), t.max())
    def f(t): 
        return model_fit.model.eval(t=t, params=params)
    y = f(t)    
    dfdt = derivative(f, t)
    idx = y > K / np.e
    if idx.sum() == 0:
        warn("All values are below K/e")
        return np.nan
    t = t[idx]
    y = y[idx]
    dfdt = dfdt[idx]

    a = dfdt.max()
    i = dfdt.argmax()
    t1 = t[i]
    y1 = y[i]
    b = y1 - a * t1
    lam = (y0 - b) / a
    return lam


def find_lag_ci(model_fit, param_samples, ci=0.95):
    """Estimates a confidence interval for the lag duration from the model fit.

    The lag duration for each parameter sample is calculated.
    The confidence interval of the lag is the lower and higher percentiles such that 
    `ci` percent of the random lag durations are within the confidence interval.

    Parameters
    ----------
    model_fit : lmfit.model.ModelResult
        the result of a model fitting procedure
    param_samples : pandas.DataFrame
        parameter samples, generated using :function:`sample_params` or :function:`bootstrap_params`    
    ci : float, optional
        the fraction of lag durations that should be within the calculated limits. 0 < `ci` <, defaults to 0.95.
    
    Returns
    -------
    low, high : float
        the lower and the higher boundaries of the confidence interval of the lag phase duration in the units of the `model_fit` ``Time`` variable (usually hours).

    See also
    --------
    find_lag
    has_lag    
    """    
    if not 0 <= ci <= 1:
        raise ValueError("ci must be between 0 and 1")
    nsamples = param_samples.shape[0]
    lags = np.zeros(nsamples)    
    for i in range(param_samples.shape[0]):
        sample = param_samples.iloc[i,:]
        params = model_fit.params.copy()
        for k,v in params.items():
            if v.vary:
                params[k].set(value=sample[k])
        lags[i] = find_lag(model_fit, params=params)

    margin = (1.0 - ci) * 50.0
    idx = np.isfinite(lags)
    if not idx.all():
        warn("Warning: omitting {0} non-finite lag values".format(len(lags) - idx.sum()))
    lags = lags[idx]
    idx = (lags >= 0)
    if not idx.all():
        warn("Warning: omitting {0} negative lag values".format(len(lags) - idx.sum()))
    if not idx.any(): # no legal lag values left
        return np.nan, np.nan, np.nan
    lags = lags[idx]
    low = np.percentile(lags, margin)
    high = np.percentile(lags, ci * 100.0 + margin)
    assert high >= low, lags.tolist()
    return low, high


def has_lag(model_fits, alfa=0.05, PRINT=False):
    r"""Checks if if the best fit has statisticaly significant lag phase :math:`\lambda > 0`.

    If the best fitted model doesn't has a lag phase to begin with, return :const:`False`. 
    This includes the logistic model and Richards model.

    Otherwise, a likelihood ratio test will be perfomed with nesting determined according to Figure 1. 
    The null hypothesis of the test is that :math:`\frac{1}{v} = 0` , 
    i.e. the adjustment rate :math:`v` is infinite and therefore there is no lag phase.

    The function will return :const:`True` if the null hypothesis is rejected, 
    otherwise it will return :const:`False`.

    Parameters
    ----------
    model_fits : sequence of lmfit.model.ModelResult
        the results of several model fitting procedures, ordered by their statistical preference. Generated by :py:func:`fit_model`.
    alfa : float, optional
        test significance level, defaults to 0.05 = 5%.
    PRINT : bool, optional
        if :const:`True`, the function will print the result of the underlying statistical test; defaults to :const:`False`.

    Returns
    -------
    bool
        the result of the hypothesis test. :const:`True` if the null hypothesis was rejected and the data suggest that there is a significant lag phase.

    Raises
    ------
    ValueError
        raised if the fittest of the :py:class:`lmfit.model.ModelResult` objects in `model_fits` is of an unknown model.
    """
    m1 = model_fits[0]
    if np.isposinf(m1.best_values.get('q0', np.inf)) and np.isposinf(m1.best_values.get('v', np.inf)):
        if PRINT:
            print('H1 model has no lag')
        return False
        
    try:
        m0_model_class = m1.model.nested_models['lag']
    except KeyError:
        raise ValueError("The best fit model {0} has no nested model for testing lag".format(m1.model.name))
    try:
        m0 = [m for m in model_fits if isinstance(m.model, m0_model_class)][0]
    except IndexError:
        raise ValueError("No {0} in model results.".format(m0_model_class.name))
    
    prefer_m1, pval, D, ddf = lrtest(m0, m1, alfa=alfa)
    if PRINT:
        print("Tested H0: %s vs. H1: %s; D=%.2g, ddf=%d, p-value=%.2g" % (m0.model.name, m1.model.name, D, ddf, pval))    
    return prefer_m1


def has_nu(model_fits, alfa=0.05, PRINT=False):
    r"""Checks if if the best fit has :math:`\nu \ne 1` and if so if that is statisticaly significant.

    If the best fitted model has :math:`\nu = 1` to begin with, return :const:`False`. This includes the logistic model.
    Otherwise, a likelihood ratio test will be perfomed with nesting determined according to Figure 1. 
    The null hypothesis of the test is that :math:`\nu = 1`; if it is rejected than the function will return :const:`True`.
    Otherwise it will return :const:`False`.

    Parameters
    ----------
    model_fits : list lmfit.model.ModelResult
        the results of several model fitting procedures, ordered by their statistical preference. Generated by :py:func:`fit_model`.
    alfa : float, optional
        test significance level, defaults to 0.05 = 5%.
    PRINT : bool, optional
        if :const:`True`, the function will print the result of the underlying statistical test; defaults to :const:`False`.

    Returns
    -------
    bool
        the result of the hypothesis test. :const:`True` if the null hypothesis was rejected and the data suggest that :math:`\nu` is significantly different from one.

    Raises
    ------
    ValueError
        raised if the fittest of the :py:class:`lmfit.model.ModelResult` objects in `model_fits` is of an unknown model.
    """
    m1 = model_fits[0]
    if m1.best_values.get('nu', 1.0) == 1.0:
        return False

    try:
        m0_model_class = m1.model.nested_models['nu']
    except KeyError:
        raise ValueError("The best fit model {} has no nested model for testing nu".format(m1.model.name))
    try:
        m0 = [m for m in model_fits if isinstance(m.model, m0_model_class)][0]
    except IndexError:
        raise ValueError("No {} in model results.".format(m0_model_class.name))

    prefer_m1, pval, D, ddf = lrtest(m0, m1, alfa=alfa)
    if PRINT:
        msg = "Tested H0: %s (nu=%.2g) vs. H1: %s (nu=%.2g); D=%.2g, ddf=%d, p-value=%.2g"  
        print(msg % (m0.model.name, m0.best_values.get('nu', 1), m1.model.name, m1.best_values.get('nu', 1), D, ddf, pval))
    return prefer_m1


def make_Dfun(model, params):
    expr, t, args = model.get_sympy_expr(params)
    partial_derivs = [None]*len(args)
    for i,x in enumerate(args):
        dydx = expr.diff(x)
        dydx = sympy.lambdify(args=(t,) + args, expr=dydx, modules="numpy")
        partial_derivs[i] = dydx        
    def Dfun(params, y, a, t):
        values = [ par.value for par in params.values() if par.vary ]
        res = np.array([dydx(t, *values) for dydx in partial_derivs])
        expected_shape = (len(values), len(t))
        if res.shape != expected_shape:
            raise TypeError("Dfun result shape for {0} is incorrect, expected {1} but it is {2}.".format(model.name, expected_shape, res.shape))
        return res
    return Dfun


def cooks_distance(df, model_fit, use_weights=True):
    """Calculates Cook's distance of each well given a specific model fit. 

    Cook's distance is an estimate of the influence of a data curve when performing model fitting; 
    it is used to find wells (growth curve replicates) that are suspicious as outliers.
    The higher the distance, the more suspicious the curve.

    Parameters
    ----------
    df : pandas.DataFrame
        growth curve data, see :py:mod:`curveball.ioutils` for a detailed definition.
    model_fit : lmfit.model.ModelResult
        result of model fitting procedure
    use_weights : bool, optional
        should the function use standard deviation across replicates as weights for the fitting procedure, defaults to :const:`True`.

    Returns
    -------
    dict
        a dictionary of Cook's distances: keys are wells (from the `Well` column in `df`), values are Cook's distances.

    Notes
    -----
    `Wikipedia <https://en.wikipedia.org/wiki/Cook's_distance>`_
    """
    p = model_fit.nvarys
    MSE = model_fit.chisqr / model_fit.ndata
    wells = df.Well.unique()
    D = {}
    
    for well in wells:    
        _df = df[df.Well != well]
        time = _df['Time'].values
        OD = _df['OD'].values
        weights =  calc_weights(_df) if use_weights else None
        model_fit_i = copy.deepcopy(model_fit)
        model_fit_i.fit(data=OD, t=time, weights=weights)
        D[well] = model_fit_i.chisqr / (p * MSE)
    return D


def find_outliers(df, model_fit, deviations=2, use_weights=True, ax=None, PLOT=False):
    """Find outlier wells in growth curve data.

    Uses the Cook's distance approach (`cooks_distance`); 
    values of Cook's distance that are `deviations` standard deviations **above** the mean
    are defined as outliers.

    Parameters
    ----------
    df : pandas.DataFrame
        growth curve data, see :py:mod:`curveball.ioutils` for a detailed definition.
    model_fit : lmfit.model.ModelResult
        result of model fitting procedure
    deviations : float, optional
        the number of standard deviations that defines an outlier, defaults to 2.
    use_weights : bool, optional
        should the function use standard deviation across replicates as weights for the fitting procedure, defaults to :const:`True`.
    ax : matplotlib.axes.Axes, optional
        an axes to plot into; if not provided, a new one is created.
    PLOT : bool, optional
        if :const:`True`, the function will plot the Cook's distances of the wells and the threshold.

    Returns
    -------
    outliers : list
        the labels of the outlier wells.
    fig : matplotlib.figure.Figure
        if the argument `PLOT` was :const:`True`, the generated figure.
    ax : matplotlib.axes.Axes
        if the argument `PLOT` was :const:`True`, the generated axis.
    """
    D = cooks_distance(df, model_fit, use_weights=use_weights)
    D = sorted(D.items()) # TODO SortedDict?
    distances = [x[1] for x in D]        
    dist_mean, dist_std = np.mean(distances), np.std(distances)
    outliers = [well for well,dist in D if dist > dist_mean + deviations * dist_std]
    if PLOT:
        if ax is None:
            fig,ax = plt.subplots(1,1)
        else:
            fig = ax.get_figure()
        wells = [x[0] for x in D]            
        ax.stem(distances, linefmt='k-', basefmt='')
        ax.axhline(y=dist_mean, ls='-', color='k')
        ax.axhline(y=dist_mean + deviations * dist_std, ls='--', color='k')
        ax.axhline(y=dist_mean - deviations * dist_std, ls='--', color='k')
        ax.set_xticks(range(len(wells)))
        ax.set_xlim(-0.5, len(wells) - 0.5)
        ax.set_xticklabels(wells, rotation=90)
        ax.set_xlabel('Well')
        ax.set_ylabel("Cook's distance")
        sns.despine()
        return outliers,fig,ax
    return outliers


def find_all_outliers(df, model_fit, deviations=2, max_outlier_fraction=0.1, use_weights=True, PLOT=False):    
    """Iteratively find outlier wells in growth curve data.

    At each iteration, calls :py:func:`find_outliers`.
    Iterations stop when no more outliers are found or when the fraction of wells defined as outliers
    is above `max_outlier_fraction`.
        
    Parameters
    ----------
    df : pandas.DataFrame
        growth curve data, see :py:mod:`curveball.ioutils` for a detailed definition.
    model_fit : lmfit.model.ModelResult
        result of model fitting procedure
    deviations : float, optional
        the number of standard deviations that defines an outlier, defaults to 2.
    max_outlier_fraction : float, optional
        maximum fraction of wells to define as outliers, defaults to 0.1 = 10%.
    use_weights : bool, optional
        should the function use standard deviation across replicates as weights for the fitting procedure, defaults to :const:`True`.
    ax : matplotlib.axes.Axes
        an axes to plot into; if not provided, a new one is created.
    PLOT : bool, optional
        if :const:`True`, the function will plot the Cook's distances of the wells and the threshold.

    Returns
    -------
    outliers : list
        a list of lists: the list nested at index *i* contains the labels of the outlier wells found at iteration *i*.
    fig : matplotlib.figure.Figure
        if the argument `PLOT` was :const:`True`, the generated figure.
    ax : matplotlib.axes.Axes
        if the argument `PLOT` was :const:`True`, the generated axis.
    """
    outliers = []
    num_wells = len(df.Well.unique())
    df = copy.deepcopy(df)
    if PLOT:
        fig = plt.figure()
        o, fig, ax = find_outliers(df, model_fit, deviations=deviations, use_weights=use_weights, ax=fig.add_subplot(), PLOT=PLOT)
    else:
        o = find_outliers(df, model_fit, deviations=deviations, use_weights=use_weights, PLOT=PLOT)
    outliers.append(o)
    while len(outliers[-1]) != 0 and len(sum(outliers, [])) <  max_outlier_fraction * num_wells:
        df = df[~df.Well.isin(outliers[-1])]
        assert df.shape[0] > 0, df.shape[0] 
        model_fit = fit_model(df, use_weights=use_weights, PLOT=False, PRINT=False)[0]
        if PLOT:
            o, fig, ax = find_outliers(df, model_fit, deviations=deviations, use_weights=use_weights, ax=fig.add_subplot(), PLOT=PLOT)            
        else:
            o = find_outliers(df, model_fit, deviations=deviations, use_weights=use_weights, PLOT=PLOT)
        outliers.append(o)
    if PLOT:
        return outliers[:-1],fig,ax
    return outliers[:-1]


def calc_weights(df, PLOT=False):
    """Calculate weights for the fittiing procedure based on the standard deviations at each time point.

    If there is more than one replicate, use the standard deviations as weight.
    Warn about NaN and infinite values.

    Parameters
    ----------
    df : pandas.DataFrame
        data frame with `Time` and `OD` columns
    PLOT : bool, optional
        if :const:`True`, plot the weights by time

    Returns
    -------
    weights : np.ndarray
        array of weights, calculated as the inverse of the standard deviations at each time point
    fig : matplotlib.figure.Figure
        if the argument `PLOT` was :const:`True`, the generated figure.
    ax : matplotlib.axes.Axes
        if the argument `PLOT` was :const:`True`, the generated axis.
    """
    deviations = df.groupby('Time')['OD'].transform(lambda x: np.repeat(x.std(), len(x))).values
    if np.isnan(deviations).any():
        warn("NaN in deviations, can't use weights")
        weights = None
    else:
        weights = 1.0 / deviations
        # if any weight is nan, raise error
        idx = np.isnan(weights)
        if idx.any():
            raise ValueError("NaN weights are illegal, indices: {0}".format(idx))
        # if any weight is infinite, change to the max
        idx = np.isinf(weights)
        if idx.all():
            warn("All weights are infinite, proceeding without weights)")
            weights = None
        elif idx.any():
            warn("Found infinite weight, changing to maximum ({0} occurences)".format(idx.sum()))
            weights[idx] = weights[~idx].max()
    if PLOT:
        fig, ax = plt.subplots(1, 1)
        ax.plot(df.Time, weights, 'o')
        ax.set_xlabel('Time')
        ax.set_ylabel('Weight')
        sns.despine()
        return weights, fig, ax
    return weights


def information_criteria_weights(results):
    r"""Calculate weighted AIC and BIC for model results.

    .. :math::

        w_i = \frac{e^{-\frac{x_i - \bar{x}}{2}}}{\sum_j{e^{-\frac{x_j - \bar{x}}{2}}}}

    where :math:`w_i` is the weighted measure for model result :math:`i` and
    :math:`x_i` is the AIC or BIC of model result :math:`i`.

    Parameters
    ----------
    results : sequence of lmfit.model.ModelResult
        use the :py:attr:`lmfit.attr.ModelResult.aic` and :py:attr:`lmfit.model.ModelResult.bic` attributes to add a `weighted_aic` and `weighted_bic` attribute.

    Notes
    -----
    - `Weighted AIC explained <http://stats.stackexchange.com/questions/35010/aic-bic-number-interpretation/44360#44360>`_.
    """
    aics = np.array([m.aic for m in results])
    bics = np.array([m.bic for m in results])
    weighted_aics = np.exp(-0.5 * (aics - aics.min()))
    weighted_aics /= weighted_aics.sum()
    weighted_bics = np.exp(-0.5 * (bics - bics.min()))
    weighted_bics /= weighted_bics.sum()
    for m, b, a in zip(results, weighted_bics, weighted_aics):
        m.weighted_aic = a
        m.weighted_bic = b


def nvarys(params):
    return len([p for p in params.values() if p.vary])


def fit_exponential_growth_phase(t, N, k=2):
    r"""Fits an exponential model to the exponential growth phase.

    Fits a polynomial p(t)~N, finds tmax the time of the maximum of the derivative dp/dt,
    and fits a linear function to log(N) around tmax.
    The resulting slope (a) and intercept (b) are the parameters of the exponential model:

    .. math::

        N(t) = N_0 e^{at}

        N_0 = e^b

    Arguments
    ---------
    t : np.ndarray
        time
    N : np.ndarray
        `N[i]` is the population size at time `t[i]`
    k : int
        number of points to take around tmax, defaults to 2 for a total of 5 points

    Returns
    -------
    slope : float
        slope of the linear regression, a
    intercept : float
        intercept of the linear regression, b
    """
    N_smooth = smooth(t, N)
    dNdt = derivative(N_smooth, t)
    imax = dNdt.argmax()
    idx = np.arange(imax - k, imax + k)
    slope, intercept = linregress(t[idx], np.log(N[idx]))[:2]
    return slope, intercept


def fit_model(df, param_guess=None, param_min=None, param_max=None, param_fix=None, 
              models=None, use_weights=False, use_Dfun=False, method='leastsq', ax=None, PLOT=True, PRINT=True):
    r"""Fit and select a growth model to growth curve data.

    This function fits several growth models to growth curve data (``OD`` as a function of ``Time``).

    Parameters
    ----------
    df : pandas.DataFrame
        growth curve data, see :py:mod:`curveball.ioutils` for a detailed definition.
    param_guess : dict, optional
        a dictionary of parameter guesses to use (key: :py:class:`str` of param name; value: :py:class:`float` of param guess).
    param_min : dict, optional
        a dictionary of parameter minimum bounds to use (key: :py:class:`str` of param name; value: :py:class:`float` of param min bound).
    param_max : dict, optional
        a dictionary of parameter maximum bounds to use (key: :py:class:`str` of param name; value: :py:class:`float` of param max bound).
    param_fix : set, optional
        a set of names (:py:class:`str`) of parameters to fix rather then vary, while fitting the models.
    use_weights : bool, optional
        should the function use the deviation across replicates as weights for the fitting procedure, defaults to :const:`False`.
    use_Dfun : bool, optional
        should the function calculate the partial derivatives of the model functions to be used in the fitting procedure, defaults to :const:`False`.
    models : one or more model classes, optional
        model classes (not instances) to use for fitting; defaults to all model classes in `curveball.baranyi_roberts_model`.
    method : str, optional
        the minimization method to use, defaults to `leastsq`, 
        can be anything accepted by :py:func:`lmfit.minimizer.Minimizer.minimize` or :py:func:`lmfit.minimizer.Minimizer.scalar_minimize`.
    ax : matplotlib.axes.Axes, optional
        an axes to plot into; if not provided, a new one is created.
    PLOT : bool, optional
        if :const:`True`, the function will plot the all model fitting results.
    PRINT : bool, optional
        if :const:`True`, the function will print the all model fitting results.

    Returns
    -------
    models : list of lmfit.model.ModelResult
        all model fitting results, sorted by increasing BIC (a measure of fit quality).
    fig : matplotlib.figure.Figure
        figure object.
    ax : numpy.ndarray
        array of :py:class:`matplotlib.axes.Axes` objects, one for each model result, with the same order as `models`.

    Raises
    ------
    TypeError
        if one of the input parameters is of the wrong type (not guaranteed).
    ValueError
        if the input is bad, for example, `df` is empty (not guaranteed).
    AssertionError
        if any of the intermediate calculated values are inconsistent (for example, ``y0<0``).

    Examples
    --------
    >>> import curveball
    >>> import pandas as pd
    >>> import matplotlib.pyplot as plt
    >>> plate = pd.read_csv('plate_templates/G-RG-R.csv')
    >>> df = curveball.ioutils.read_tecan_xlsx('data/Tecan_280715.xlsx', label='OD', plate=plate)
    >>> green_models = curveball.models.fit_model(df[df.Strain == 'G'], PLOT=True)
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input df must be a %s, but it is %s" % (pd.DataFrame.__name__, df.__class__.__name__))
    if df.shape[0] == 0:
        raise ValueError("No rows in input df")
    
    if param_guess is None:
        param_guess = dict()
    if param_fix is None:
        param_fix = set()

    df = df.sort_values(by=['Time', 'OD'])
    time = df['Time'].values
    OD = df['OD'].values
    weights =  calc_weights(df) if use_weights else None
    # TODO why should we use weights if we use the whole data set?
    ODerr = df.groupby('Time').OD.transform(lambda x: np.repeat(x.std(), len(x))).values
   
    if models is None:
        models = get_models(curveball.baranyi_roberts_model)
    elif is_model(models):
        models = [models]
    results = [None] * len(models)
    for i, model_class in enumerate(models):
        model = model_class()
        params = model.guess(data=OD, t=time, param_guess=param_guess, param_min=param_min, param_max=param_max, param_fix=param_fix)    
        fit_kws = {'Dfun': make_Dfun(model, params), "col_deriv":True} if use_Dfun else {}        
        model_result = model.fit(data=OD, t=time, params=params, weights=weights, fit_kws=fit_kws, method=method)
        results[i] = model_result

    # sort by increasing BIC
    information_criteria_weights(results)
    results.sort(key=lambda m: m.bic)

    if PRINT:
        print(results[0].fit_report(show_correl=False))
    if PLOT:        
        dy = df['OD'].max() / 50.0
        dx = df['Time'].max() / 25.0
        columns = min(3, len(results))
        rows = int(np.ceil(len(results) / columns))
        w = max(8, 4 * columns)
        h = max(6, 3*rows)
        fig, ax = plt.subplots(rows, columns, sharex=True, sharey=True, figsize=(w, h))
        if not hasattr(ax, '__iter__'):
            ax = np.array(ax, ndmin=2)
        for i,fit in enumerate(results):
            row = i // columns
            col = i % columns
            _ax = ax[row, col]
            vals = fit.best_values
            fit.plot_fit(ax=_ax, datafmt='.', data_kws={'alpha':0.3}, fit_kws={'lw': 4}, yerr=ODerr)
            _ax.axhline(y=vals.get('y0', 0), color='k', ls='--')
            _ax.axhline(y=vals.get('K', 0), color='k', ls='--')          
            title = '%s %dp\nBIC: %.3f\ny0=%.2f, K=%.2f, r=%.2g\n' + r'$\nu$=%.2g, $q_0$=%.2g, v=%.2g'
            title = title % (fit.model.name, fit.nvarys, fit.bic, vals.get('y0', np.nan), vals.get('K', np.nan), vals.get('r', np.nan), vals.get('nu', np.nan), vals.get('q0', np.nan), vals.get('v', np.nan))
            _ax.set_title(title)
            _ax.get_legend().set_visible(False)
            _ax.set_xlabel('')
            _ax.set_ylabel('')
            if col == 0:
                _ax.set_ylabel('OD')
            if row == rows - 1:
                _ax.set_xlabel('Time')
        _ax.set_xlim(0, 1.1 * df['Time'].max())
        _ax.set_ylim(0.9 * df['OD'].min(), 1.1 * df['OD'].max())
        sns.despine()
        fig.tight_layout()
        return results, fig, ax
    return results


if __name__ == '__main__':    
    # simulate 30 growth curves
    df = randomize(t=12, y0=0.12, K=0.56, r=0.8, nu=1.8, q0=0.2, v=0.8, reps=30, noise_std=0.04, random_seed=0)

    # fit models to growth curves
    results, fig, ax = fit_model(df, use_Dfun=True, PLOT=True, PRINT=True)    
    plt.savefig('test_models.png')
    plt.show()

