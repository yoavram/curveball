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

from scipy.stats import chisqprob
from lmfit import Model


def logistic_function(t, y0, r, K):
    r"""The logistic growth model is the standard growth model in ecology.

    .. math::
        \frac{dy}{dt} = r y \Big(1 - \frac{y}{K}\Big) \Rightarrow
        y(t) = \frac{K}{1 - \Big(1 - \frac{K}{y_0} \Big)e^{-r t}}


    - :math:`y_0`: initial population size
    - K: maximum population size
    - r: initial growth rate per capita

    See also: `Wikipedia <http://en.wikipedia.org/wiki/Logistic_function>`_
    """
    return richards_function(t ,y0, r, K, 1.)


def richards_function(t, y0, r, K, nu):
    r"""Richards growth model (or the generalized logistic model) in a generalization of the logistic model that allows the inflection point to be anywhere along the curve.

    .. math::

        \frac{dy}{dt} = r y \Big( 1 - \Big(\frac{y}{K}\Big)^{\nu} \Big) \Rightarrow

        y(t) = \frac{K}{\Big[1 - \Big(1 - \Big(\frac{K}{y_0}\Big)^{\nu}\Big) e^{-r \nu t}\Big]^{1/\nu}}

    - :math:`y_0`: initial population size
    - K: maximum population size
    - r: initial growth rate per capita
    - :math:`\nu`: curvature of the logsitic term

    See also: `Wikipedia <http://en.wikipedia.org/wiki/Generalised_logistic_function>`_
    """
    return K / ((1 - (1 - (K/y0)**nu) * np.exp(-r * nu * t))**(1./nu))


def baranyi_roberts_function(t, y0, r, K, nu, q0, v):
    r"""The Baranyi-Roberts growth model is an extension of the Richards model that adds time lag.

    .. math::

        \frac{dy}{dt} = r \alpha(t) y \Big( 1 - \Big(\frac{y}{K}\Big)^{\nu} \Big) \Rightarrow

        y(t) = \frac{K}{\Big[1 - \Big(1 - \Big(\frac{K}{y_0}\Big)^{\nu}\Big) e^{-r \nu A(t)}\Big]^{1/\nu}}

        A(t) = \int_0^t{\alpha(s)ds} = \int_0^t{\frac{q_0}{q_0 + e^{-v t}} ds} = t + \frac{1}{v} \log{\Big( \frac{e^{-v t} + q0}{1 + q0} \Big)}


    - :math:`y_0`: initial population size
    - K: maximum population size
    - r: initial growth rate per capita
    - :math:`\nu`: curvature of the logsitic term
    - :math:`q_0`: initial adjustment to current environment
    - v: adjustment rate

    See also: `Baranyi, J., Roberts, T. a., 1994. A dynamic approach to predicting bacterial growth in food. Int. J. Food Microbiol. 23, 277–294. <www.ncbi.nlm.nih.gov/pubmed/7873331>`_
    """
    At = t + (1./v) * np.log((np.exp(-v * t) + q0)/(1 + q0))
    return K / ((1 - (1 - (K/y0)**nu) * np.exp( -r * nu * At ))**(1./nu))


def lrtest(m0, m1, alfa=0.05):
    r"""Perform a likelihood ratio test on two nested models.

    For two models, one nested in the other (meaning that the nested model estimated parameters are a subset of the nesting model), the test statistic :math:`D` is:

    .. math::

        \Lambda = \Big( \Big(\frac{\sum{(X_i - \hat{X_i}(\theta_1))^2}}{\sum{(X_i - \hat{X_i}(\theta_0))^2}}\Big)^{n/2} \Big)

        D = -2 log \Lambda

        lim_{n \to \infty} D \sim \chi^2_{df=\Delta}


    where :math:`\Lambda` is the likelihood ratio, :math:`D` is the statistic, :math:`X_{i}` are the data points, :math:`\hat{X_i}(\theta)` is the model prediction with parameters :math:`\theta`, :math:`\theta_i` is the parameters estimation for model :math:`i`, $n$ is the number of data points and :math:`\Delta` is the difference in number of parameters between the models.

    The function compares between two :py:class:`lmfit.ModelFit` objects. These are the results of fitting models to the same data set using the `lmfit <lmfit.github.io/lmfit-py>`_ package

    The function compares between model fit `m0` and `m1` and assumes that `m0` is nested in `m1`, meaning that the set of varying parameters of `m0` is a subset of the varying parameters of `m1`. The property `chisqr` of the :py:class:`ModelFit` objects is the sum of the square of the residuals of the fit. `ndata` is the number of data points. `nvarys` is the number of varying parameters.

    Args:
        - m0, m1: :py:class:`lmfit.Model` objects representing two models. `m0` is nested in `m1`.
        - alfa: The test significance level (default: 5%).

    Returns:
        prefer_m1, pval, D, ddf: :py:class:`tuple`
            - prefer_m1: should we prefer `m1` over `m0`, :py:class:`bool`
            - pval: the test p-value, :py:class:`float`
            - D: the test statistic, :py:class:`float`
            - ddf: the number of degrees of freedom, :py:class:`int`

    See also: `Generalized Likelihood Ratio Test Example <http://www.stat.sc.edu/~habing/courses/703/GLRTExample.pdf>`_, `IPython notebook <http://nbviewer.ipython.org/github/yoavram/ipython-notebooks/blob/master/likelihood%20ratio%20test.ipynb>`_
    """
    n0 = m0.ndata
    k0 = m0.nvarys
    chisqr0 = m0.chisqr
    assert chisqr0 > 0
    n1 = m1.ndata
    assert n0 == n1
    k1 = m1.nvarys
    chisqr1 = m1.chisqr
    assert chisqr1 > 0
    Lambda = (m1.chisqr / m0.chisqr)**(n0 / 2.)
    D = -2 * np.log( Lambda )
    assert D > 0
    ddf = k1 - k0
    assert ddf > 0
    pval = chisqprob(D, ddf)
    prefer_m1 = pval < alfa
    return prefer_m1, pval, D, ddf


def fit_model(df, ax=None, PLOT=True, PRINT=True):
    r"""Fit a growth model to data.

    This function will attempt to fit a growth model to `OD~Time` taken from the `df` :py:class:`pandas.DataFrame`.
    The function is still being developed.
    """
    _df = df.groupby('Time')['OD'].agg([np.mean, np.std]).reset_index().rename(columns={'mean':'OD'})
    models = []

    # TODO: make MyModel, inherit from Model, use Model.guess
    Kguess  = _df.OD.max()
    y0guess = _df.OD.min()
    nuguess = 1.0
    _df['dODdTime'] = np.gradient(_df.OD, _df.Time)
    rguess  = 4 * _df.dODdTime[~np.isinf(_df.dODdTime)].max() / Kguess
    q0guess, vguess = 0.1, 1

    params = baranyi_roberts_model.make_params(y0=y0guess, K=Kguess, r=rguess, nu=nuguess, q0=q0guess, v=vguess)
    params['y0'].set(min=1-10)
    params['K'].set(min=1-10)
    params['r'].set(min=1-10)
    params['nu'].set(min=1-10)
    params['q0'].set(min=1e-10, max=1)
    params['v'].set(min=1e-10)

    # Baranyi-Roberts = Richards /w lag (6 params)
    result = baranyi_roberts_model.fit(data=_df.OD, t=_df.Time, params=params, weights=1./_df['std'])
    models.append(result)

    # Baranyi-Roberts /w nu=1 = Logistic /w lag (5 params)
    params['nu'].set(vary=False)
    result = baranyi_roberts_model.fit(data=_df.OD, t=_df.Time, params=params, weights=1./_df['std'])
    models.append(result)

    # Richards = Baranyi-Roberts /wout lag (4 params)
    params = richards_model.make_params(y0=y0guess, K=Kguess, r=rguess, nu=nuguess)
    params['y0'].set(min=1-10)
    params['K'].set(min=1-10)
    params['r'].set(min=1-10)
    params['nu'].set(min=1-10)
    result = richards_model.fit(data=_df.OD, t=_df.Time, params=params, weights=1./_df['std'])
    models.append(result)

    # Logistic = Richards /w nu=1 (3 params)
    params = logistic_model.make_params(y0=y0guess, K=Kguess, r=rguess)
    params['y0'].set(min=1-10)
    params['K'].set(min=1-10)
    params['r'].set(min=1-10)
    result = logistic_model.fit(data=_df.OD, t=_df.Time, params=params, weights=1./_df['std'])
    models.append(result)

    # sort by increasing bic
    models.sort(key=lambda m: m.bic)

    if PRINT:
        print models[0].fit_report()
        vals = models[0].best_values
        lam = np.log(1. + 1./vals['q0']) / vals['v']
        print "lambda:", lam
    if PLOT:
        dy = _df.OD.max()/50
        dx = _df.Time.max()/25
        fig, ax = plt.subplots(1, len(models), sharex=True, sharey=True, figsize=(16,6))
        for i,fit in enumerate(models):
            vals = fit.best_values
            fit.plot_fit(ax=ax[i], datafmt='.', fit_kws={'lw':4})
            ax[i].axhline(y=vals['y0'], color='k', ls='--')
            ax[i].axhline(y=vals['K'], color='k', ls='--')
            if 'q0' in vals:
                lam = np.log(1. + 1./vals['q0']) / vals['v']
            else:
                lam = 0
            ax[i].axvline(x=lam, color='k', ls='--')
            ax[i].text(x=lam + dx, y=_df.OD.min() - 3*dy, s=r'$\lambda=$%.2f' % lam)
            title = '%s %dp\nBIC: %d\ny0=%.2f, K=%.2f, r=%.2g\n' + r'$\nu$=%.2g, $q_0$=%.2g, v=%.2g'
            title = title % (fit.model.name, fit.nvarys, fit.bic, vals['y0'], vals['K'], vals['r'], vals.get('nu',0), vals.get('q0',0), vals.get('v',0))
            ax[i].set_title(title)
            ax[i].get_legend().set_visible(False)
            ax[i].set_xlim(0, 1.1 * _df.Time.max())
            ax[i].set_ylim(0, 1.1 * _df.OD.max())
            ax[i].set_xlabel('Time')
            ax[i].set_ylabel('')
        ax[0].set_ylabel('OD')
        sns.despine()
        fig.tight_layout()
        return models, fig, ax
    return models


logistic_model = Model(logistic_function)
richards_model = Model(richards_function)
baranyi_roberts_model = Model(baranyi_roberts_function)
