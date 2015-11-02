#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of curveball.
# https://github.com/yoavram/curveball

# Licensed under the MIT license:
# http://www.opensource.org/licenses/MIT-license
# Copyright (c) 2015, Yoav Ram <yoav@yoavram.com>
from __future__ import division
from builtins import range
from past.utils import old_div
import warnings
import curveball
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import pandas as pd
import lmfit
import seaborn as sns
sns.set_style("ticks")

def _alfa(t, q0, v):
	if np.isinf(v):
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
	y : float
		population size
	t : float
		time, usually in hours
	r : float
		initial per capita growth rate
	K : float
		maximum population size (:math:`K>0`)
	nu : float
		curvature of the logsitic term (:math:`\nu>0`)
	q0 : float
		initial adjustment to current environment (:math:`0<q_0<1`)
	v : float
		adjustment rate (:math:`v>0`)

	Returns
	-------
	float
		population growth rate.

	References
	----------
	.. [1] Baranyi, J., Roberts, T. A., 1994. `A dynamic approach to predicting bacterial growth in food <www.ncbi.nlm.nih.gov/pubmed/7873331>`_. Int. J. Food Microbiol.
	"""
	alfa = _alfa(t, q0[0], v[0]), _alfa(t, q0[1], v[1])
	dydt = alfa[0] * r[0] * y[0] * (1 - (old_div((y[0] + y[1]), K[0]))**nu[0]), alfa[1] * r[1] * y[1] * (1 - (old_div((y[0] + y[1]), K[1]))**nu[1])
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
	dydt = alfa[0] * r[0] * y[0] * (1 - (old_div(y[0], K[0]) + old_div(y[1], K[1]))**nu[0]), alfa[1] * r[1] * y[1] * (1 - (old_div(y[0], K[0]) + old_div(y[1], K[1]))**nu[1])
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
	dydt = alfa[0] * r[0] * y[0] * (1 - (old_div(y[0], K[0]))**nu[0] - (old_div(y[1], K[1]))**nu[1]), alfa[1] * r[1] * y[1] * (1 - (old_div(y[0], K[0]))**nu[0] - (old_div(y[1], K[1]))**nu[1])
	return dydt


def compete(m1, m2, y0=None, hours=24, nsamples=1, lag_phase=True, ode=double_baranyi_roberts_ode1, num_of_points=100, ci=95, colors=None, ax=None, PLOT=False):
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
	y0 : tuple, optional
		initial population sizes/densities. Defaults to a tuple with twice the average of the ``y0`` parameter of `m1` and `m2`.
	hours : int, optional
		how many hours should the competition proceed, defaults to 24.
	nsamples : int, optional
		how many replicates of the competition should be simulated; if `nsamples` = 1, only one competition is simulated with the estimated parameters; otherwise `nsamples` competitions are simulated with parameters drawn from a distribution based on the covariance matrix of the parameter estimates (see :py:func:`curveball.models.sample_params`). Defaults to 1.
	lag_phase : bool, optional
		if :py:const:`True`, use lag phase as given by `m1` and `m2`. Otherwise, override the lag phase parameters to prevent a lag phase. Defaults to :py:const:`True`.
	ode : func, optional
		an ordinary differential systems system defined by a function that accepts ``y``, ``t``, and additional arguments, and returns the derivate of ``y`` with respect to ``t``. Defaults to :py:func:`.double_baranyi_roberts_ode0`.
	num_of_points : int, optional
		number of time points to use, defaults to 100.
	ci : float, optional
		confidence interval size, in (0, 100), only applicable when `PLOT` is :py:const:`True`, defaults to 95%.
	colors : sequence of str, optional
		if `PLOT` is :py:const:`True`, this sets the colors of the drawn lines. `colors[0]` will be used for `m1`; `colors[1]` for `m2`. If not provided, defaults to the current pallete.
	ax : matplotlib.axes.Axes, optional
		if `PLOT` is :py:const:`True`, an axes to plot into; if not provided, a new one is created.
	PLOT : bool, optional
		if :py:const:`True`, the function will plot the curves of *y* as a function of *t*. Defaults to :py:const:`False`.

	Returns
	-------
	t : numpy.ndarray
		1d (or 2d, if `nsamples`>1) array of time points, in hours.
	y: numpy.ndarray
		2d (or 3d, if `nsamples`>1) array of strain frequencies. First axis is time, second axis is strain, third axis (if applicable) is sample.
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

	Example
	-------
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

	t = np.linspace(0, hours, num_of_points)
	if y0 is None:		
		y0 = np.array([m1.best_values['y0'], m2.best_values['y0']])		
		y0 = old_div(y0.mean(),2), old_div(y0.mean(),2)
		assert y0[0] == y0[1]		
	if nsamples > 1:
		m1_samples = curveball.models.sample_params(m1, nsamples)
		m2_samples = curveball.models.sample_params(m2, nsamples)
		min_nsamples = min(len(m1_samples), len(m2_samples))
		if nsamples > min_nsamples:
			warnings.warn("{0} resamples lost".format(nsamples - min_nsamples))
			nsamples = min_nsamples
	else:
		nsamples = 1
		m1_samples = pd.DataFrame([m1.best_values])
		m2_samples = pd.DataFrame([m2.best_values])
		assert len(m1_samples) == len(m2_samples)

	y = np.zeros((num_of_points, 2, nsamples))
	#infodict = [None]*nsamples # DEBUG
	
	for i in range(nsamples):
		r = m1_samples.iloc[i]['r'], m2_samples.iloc[i]['r']
		K = m1_samples.iloc[i]['K'], m2_samples.iloc[i]['K']
		nu = m1_samples.iloc[i].get('nu', 1.0), m2_samples.iloc[i].get('nu', 1.0)		
		if lag_phase:
			q0 = m1_samples.iloc[i].get('q0', 1.0), m2_samples.iloc[i].get('q0', 1.0)
			v = m1_samples.iloc[i].get('v', np.inf), m2_samples.iloc[i].get('v', np.inf)
		else:
			q0 = 1.0, 1.0
			v = np.inf, np.inf
		args = (r, K, nu, q0, v)
		
		y[:,:,i] = odeint(ode, y0, t, args=args)

		# DEBUG
		# _y_,info = odeint(double_baranyi_roberts_ode, y0, t, args=args, full_output=1)        
		# info['args'] = (y0,) + args
		# infodict[i] = info
		# if info['message'] == 'Integration successful.':
		#    y[:,:,i] = _y_

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
		
		if not colors is None:
			colors = {i:c for i,c in enumerate(colors)}
		sns.tsplot(df, time='Time', unit='Replicate', condition='Strain', value='y', 
						ci=ci, color=colors, ax=ax)
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
		if `PLOT` is :py:const:`True`, an axes to plot into; if not provided, a new one is created.    
	PLOT : bool, optional
		if :py:const:`True`, the function will plot the curve of *s* as a function of *t*.

	Returns
	-------
	svals : numpy.ndarray
		the selection coefficients of the assay strain relative to the reference strain over time.
	fig : matplotlib.figure.Figure
		figure object.
	ax : numpy.ndarray
		array of :py:class:`matplotlib.axes.Axes` objects.

	Notes
	-----
	This formula assumes that the frequencies of the strains follow a logistic curve. 
	Lag phases, interactions, etc. may cause this formula to become irrelevant.

	References
	----------
	.. [12] Chevin, L-M. 2011. `On Measuring Selection in Experimental Evolution <http://dx.doi.org/10.1098/rsbl.2010.0580>`_. Biology Letters.	
	"""
	svals = np.gradient(np.log(old_div(y[:,0],y[:,1])), t)
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


def fitness_LTEE(y, ref_strain=0, assay_strain=1, t0=0, t1=-1, ci=0):
	r"""Calculate relative fitness according to the definition used in the *Long Term Evolutionary Experiment* (LTEE) [3]_,
	where :math:`A(t), B(t)` are population densities of assay strain *A* and reference strain *B* at time *t*:

	.. math::

		\omega = \frac{\log{(A(t)/A(0))}}{\log{(B(t)/B(0))}}

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
	ci : bool, optional
		if :py:const:`True`, a confidence interval will be calculated using the third axis of `y` as replicates.
	
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
		At0, Bt0 = y[t0,assay_strain,i], y[t0,ref_strain,i]
		At1, Bt1 = y[t1,assay_strain,i], y[t1,ref_strain,i]
		w[i] = (old_div(np.log(old_div(At1,At0)), np.log(old_div(Bt1,Bt0))))

	if ci == 0:
		return w.mean()
	else:
		margin = (1 - ci) * 50
		return w.mean(), np.percentile(w, margin), np.percentile(w, ci * 100 + margin)
