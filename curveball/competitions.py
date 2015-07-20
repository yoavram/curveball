#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of curveball.
# https://github.com/yoavram/curveball

# Licensed under the MIT license:
# http://www.opensource.org/licenses/MIT-license
# Copyright (c) 2015, Yoav Ram <yoavram+github@gmail.com>
import curveball
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import pandas as pd
import seaborn as sns
sns.set_style("ticks")


def double_logistic_ode(y, t, r, K):
    dydt = r[0] * y[0] * (1 - (y[0] / K[0] + y[1] / K[1])), r[1] * y[1] * (1 - (y[0] / K[0] + y[1] / K[1]))
    return dydt


# def richards_ode(y, t, r, K, nu):    
#     return r * y * (1 - (y/K)**nu)


# def double_richards_ode(y, t, r, K, nu):    
#     dydt = r[0] * y[0] * (1 - (y.sum() / K[0])**nu[0]), r[1] * y[1] * (1 - (y.sum() / K[1])**nu[1])
#     return dydt


def baranyi_roberts_ode(y, t, r, K, nu, q0, v):
    alfa = q0 / (q0 + np.exp(-v * t))
    return alfa * r * y * (1 - (y/K)**nu)


def double_baranyi_roberts_ode(y, t, r, K, nu, q0, v):
    alfa = q0[0] / (q0[0] + np.exp(-v[0] * t)), q0[1] / (q0[1] + np.exp(-v[1] * t))
    dydt = alfa[0] * r[0] * y[0] * (1 - (y[0] / K[0])**nu[0] - (y[1] / K[1])**nu[1]), alfa[1] * r[1] * y[1] * (1 - (y[0] / K[0])**nu[0] - (y[1] / K[1])**nu[1])
    return dydt


from scipy.integrate import odeint

def compete(m1, m2, y0=None, hours=24, nsamples=1, lag_phase=True, num_of_points=100, colors=None, ax=None, PLOT=False):
    t = np.linspace(0, hours, num_of_points)
    if y0 is None:
        y0 = np.array(m1.best_values['y0'], m2.best_values['y0'])
        y0 = np.mean(y0)/2, np.mean(y0)/2
    if nsamples > 1:
        m1_samples = curveball.models.sample_params(m1, nsamples)
        m2_samples = curveball.models.sample_params(m2, nsamples)
    else:
        nsamples = 1
        m1_samples = pd.DataFrame([m1.best_values])
        m2_samples = pd.DataFrame([m2.best_values])

    y = np.zeros((num_of_points, 2, nsamples))
    #infodict = [None]*nsamples # DEBUG
    
    for i in range(nsamples):
        r = max(m1_samples.iloc[i]['r'],1e-6), max(m2_samples.iloc[i]['r'],1e-6)
        K = max(m1_samples.iloc[i]['K'],y0[0]), max(m2_samples.iloc[i]['K'],y0[1])
        nu = max(m1_samples.iloc[i].get('nu', 1.0),1e-6), max(m2_samples.iloc[i].get('nu', 1.0),1e-6)
        q0 = 1.0,1.0
        v = 1e6,1e6
        if lag_phase:
            q0 = max(m1_samples.iloc[i].get('q0', 1.0), 1e-6), max(m2_samples.iloc[i].get('q0', 1.0), 1e-6)
            v = max(m1_samples.iloc[i].get('v', 1e6), 1e-6), max(m2_samples.iloc[i].get('v', 1e6), 1e-6)
        args = (r, K, nu, q0, v)
        
        y[:,:,i] = odeint(double_baranyi_roberts_ode, y0, t, args=args)

        # DEBUG
        #_y_,info = odeint(double_baranyi_roberts_ode, y0, t, args=args, full_output=1)        
        #if info['message'] != 'Integration successful.':
        #    info['args'] = (y0,) + args
        #    infodict[i] = info
        #else:
        #    y[:,:,i] = _y_
    
    if PLOT:
        if ax is None:
            fig,ax = plt.subplots(1, 1)
        else:
            fig = ax.get_figure()
        for i in range(nsamples):
            ax.plot(t, y[:,:,i], alpha=nsamples**(-0.2))
            if not colors is None:
                ax.get_lines()[-2].set_color(colors[0])
                ax.get_lines()[-1].set_color(colors[1])
        ax.plot(t, y.mean(axis=2), lw=5)
        if not colors is None:
        	ax.get_lines()[-2].set_color(colors[0])
        	ax.get_lines()[-1].set_color(colors[1])
        ax.set_xlabel('Time (hour)')
        ax.set_ylabel('OD')
        sns.despine()
        return t,y,fig,ax

    return t,y


def selection_coefs_ts(t, y, ax=None, PLOT=False):
	svals = np.gradient(np.log(y[:,0]/y[:,1]), t)
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
		return svals,fig,ax

	return svals


def fitness_LTEE(y, ref_strain=0, assay_strain=1, t0=0, t1=-1):
	r'''Calculate relative fitness according to the definition used in the Long Term Evolutionary Experiment (LTEE):

	.. math::

		\omega = \frac{\log{(A(t)/A(0))}}{\log{(B(t)/B(0))}}


	where :math:`A(t), B(t)` are population densities of assay strain :math:`A` and reference strain :math:`B` at time :math:`t`.

	Args:
        - y: :py:class:`numpy.ndarray` array of population densities, as produced by :py:func:`curveball.competitions.compete`, where the first axis is time and the second axis is strain.
        - ref_strain: :py:class:`int` of the index of the reference strain within `y`. This strain's fitness is set to 1 by definition.
        - assay_strain: :py:class:`int` of the index of the assay strain within `y`. The result will be the fitness of this strain relative to the fitness of the reference strain.
        - t0: :py:class:`int` the index of the time point from which to start the calculation of the relative fitness. 0 by default.
        - t1: :py:class:`int` the index of the time point at which to end the calculation of the relative fitness. -1 (last) by default.

    Returns:
        w: :py:class:`float`, the fitness of the assay strain relative to the reference strain. Note that this may depend on the choice of `t0` and `t1` as well as the reference strain.
    

	See also: `Wiser, M. J. & Lenski, R. E., 2015 A Comparison of Methods to Measure Fitness in *Escherichia coli*. PLoS One 10, e0126210. <http://dx.plos.org/10.1371/journal.pone.0126210>`_
	'''
	return np.log(y[t1, assay_strain]/(y[t0, assay_strain])) / np.log(y[t1, ref_strain]/y[t0, ref_strain])




