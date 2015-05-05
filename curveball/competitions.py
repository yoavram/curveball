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
import seaborn as sns
sns.set_style("ticks")


# def richards_ode(y, t, r, K, nu):    
#     return r * y * (1 - (y/K)**nu)


# def double_richards_ode(y, t, r, K, nu):    
#     dydt = r[0] * y[0] * (1 - (y.sum() / K[0])**nu), r[1] * y[1] * (1 - (y.sum() / K[1])**nu[1])
#     assert dydt.shape == y.shape
#     return dydt


def baranyi_roberts_ode(y, t, r, K, nu, q0, v):
    alfa = q0 / (q0 + np.exp(-v * t))
    return alfa * r * y * (1 - (y/K)**nu)


def double_baranyi_roberts_ode(y, t, r, K, nu, q0, v):
    alfa = q0[0] / (q0[0] + np.exp(-v[0] * t)), q0[1] / (q0[1] + np.exp(-v[1] * t))
    dydt = alfa[0] * r[0] * y[0] * (1 - (y.sum() / K[0])**nu[0]), alfa[1] * r[1] * y[1] * (1 - (y.sum() / K[1])**nu[1])
    return dydt


def compete(m1, m2, hours=24, num_of_points=100, ax=None, PLOT=False):
	t = np.linspace(0, hours, num_of_points)
	y0 = np.array(m1.best_values['y0'], m2.best_values['y0'])
	y0 = np.mean(y0), np.mean(y0)
	K = m1.best_values['K'], m2.best_values['K']
	r = m1.best_values['r'], m2.best_values['r']
	nu = m1.best_values.get('nu', 1.0), m1.best_values.get('nu', 1.0)
	q0 = m1.best_values.get('q0', 1.0), m1.best_values.get('q0', 1.0)
	v = m1.best_values.get('v', 1e6), m1.best_values.get('v', 1e6)

	y = odeint(double_baranyi_roberts_ode, y0, t, args=(r, K, nu, q0, v))

	if PLOT:
		if ax is None:
			fig,ax = plt.subplots(1,1)
		else:
			fig = ax.get_figure()
		ax.plot(t, y)
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




