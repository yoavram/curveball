#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of curveball.
# https://github.com/yoavram/curveball

# Licensed under the MIT license:
# http://www.opensource.org/licenses/MIT-license
# Copyright (c) 2015, Yoav Ram <yoav@yoavram.com>
from __future__ import print_function
from __future__ import division
import sys
from warnings import warn
import numbers
import numpy as np
from scipy.optimize import minimize
from statsmodels.nonparametric.smoothers_lowess import lowess
import matplotlib.pyplot as plt
import pandas as pd
import lmfit
import sympy


MIN_VALUE = 1e-4
STEP_RATIO = 0.1
USE_STEP_FUNC = True


def smooth(x, y, PLOT=False):
	"""Lowess smoothing function.

	Parameters
	----------
	x : numpy.ndarray
		array of floats for the independent variable
	y : numpy.ndarray
		array of floats for the dependent variable

	Returns
	-------
	numpy.ndarray
		array of floats for the smoothed dependent variable
	"""
	yhat = lowess(y, x, 0.1, return_sorted=False)
	if PLOT:
		fig, ax = plt.subplots(1, 1)
		ax.plot(x, yhat, 'k--')
		ax.plot(x, y, 'ko')
	return yhat


def baranyi_roberts_function(t, y0, K, r, nu, q0, v):
	r"""The Baranyi-Roberts growth model is an extension of the Richards model that adds a lag phase [1]_.

	.. math::

		\frac{dy}{dt} = r \alpha(t) y \Big( 1 - \Big(\frac{y}{K}\Big)^{\nu} \Big) \Rightarrow

		y(t) = \frac{K}{\Big[1 - \Big(1 - \Big(\frac{K}{y_0}\Big)^{\nu}\Big) e^{-r \nu A(t)}\Big]^{1/\nu}}

		A(t) = \int_0^t{\alpha(s)ds} = \int_0^t{\frac{q_0}{q_0 + e^{-v s}} ds} = t + \frac{1}{v} \log{\Big( \frac{e^{-v t} + q0}{1 + q0} \Big)}


	- :math:`y_0`: initial population size
	- r: initial per capita growth rate
	- K: maximum population size
	- :math:`\nu`: curvature of the logsitic term
	- :math:`q_0`: initial adjustment to current environment
	- v: adjustment rate

	Parameters
	----------
	t : numpy.ndarray
		array of floats for time, usually in hours (:math:`t>0`)
	y0 : float
		initial population size (:math:`y_0>0`)
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
	numpy.ndarray
		population size per time point in `t`.

	References
	----------
	.. [1] Baranyi, J., Roberts, T. A., 1994. `A dynamic approach to predicting bacterial growth in food <www.ncbi.nlm.nih.gov/pubmed/7873331>`_. Int. J. Food Microbiol.
	"""
	if np.isposinf(q0) or np.isposinf(v):
		At = t
	else:
		At = t + (1.0 / v) * np.log((np.exp(-v * t) + q0) / (1.0 + q0))
	return K / ((1 - (1 - (K / y0)**nu) * np.exp(-r * nu * At))**(1.0/nu))


def baranyi_roberts_step_function(t, y0, K, r, nu, q0, v):
	r"""TODO
	"""
	S = STEP_RATIO
	if np.isposinf(q0) or np.isposinf(v):
		# without lag
		return K / ((1 - (1 - (K / y0)**nu) * np.exp(-r * nu * t))**(1.0/nu))
	# with lag: find time where (y-y0)/(K-y0) = S
	tS = (1.0/r) * np.log( ((S * (K - y0) / y0 + 1)*(1 + q0) - 1) / q0 )
	y = np.zeros(len(t))
	# small population, no logistic term (1-(N/K)**nu)
	y[t < tS] = y0 * (1 + q0 * np.exp(r * t[t < tS])) / (1 + q0)
	# big population, standard baranyi-roberts function
	At = t[t >= tS] + (1.0 / v) * np.log((np.exp(-v * t[t >= tS]) + q0)/(1 + q0))
	y[t >= tS] = K / (1 - (1 - (K / y0)**nu) * np.exp(-r * nu * At))**(1.0 / nu)
	return y


def guess_nu(t, N, K=None, PLOT=False, PRINT=False):
	r"""Guesses the value of :math:`\nu` from the shape of the growth curve.

	Following [4]_:

	.. math::

		N_{max} = K (1 + \nu)^{-\frac{1}{\nu}}


	- :math:`N_{max}`: population size when the population growth rate (:math:`\frac{dN}{dt}`) is maximum
	- r: initial per capita growth rate 
	- K: maximum population size
	- :math:`\nu`: curvature of the logsitic term

	Parameters
	----------
	t : numpy.ndarray
		time
	N : numpy:ndarray
		`N[i]` is the population size at time `t[i]`
	K : float, optional
		a guess of `K`, the maximum population size. If not given, it is guessed.
	PLOT : bool, optional
		if :py:const:`True`, the function will plot the calculations.
	PRINT : bool, optional
		if :py:const:`True`, the function will print intermediate results of the calculations.

	Returns
	-------
	x : float
		the guess of :math:`\nu`.
	fig : matplotlib.figure.Figure
		if the argument `PLOT` was :py:const:`True`, the generated figure.
	ax : matplotlib.axes.Axes
		if the argument `PLOT` was :py:const:`True`, the generated axis.

	References
	----------
	.. [4] Richards, F. J. 1959. `A Flexible Growth Function for Empirical Use <http://dx.doi.org/10.1093/jxb/10.2.290>`_. Journal of Experimental Botany
	"""
	if K is None:
		K = N.max()
	# only use the first part of the curve up to N=K/e (inflexion point for nu=0)
	idx = N >= K/np.exp(1)
	t = t[idx]
	N = N[idx]
	# smooth and calculate derivative
	N_smooth = smooth(t, N)
	dNdt = np.gradient(N_smooth, t[1]-t[0])   
	dNdt_smooth = smooth(t, dNdt)
	# find inflexion point
	i = dNdt_smooth.argmax()
	Nmax = N[i]    
	# find nu that gives this inflexion point
	def target(nu):
		return np.abs((1 + nu)**(-1.0 / nu) - Nmax / K)
	opt_res = minimize(target, x0=1, bounds=[(0, None)])
	x = opt_res.x
	y = target(x)
	y1 = target(1.0)
	
	if PRINT:
		print("f(1)=%.4f, f(%.4f)=%.4f" % (y1, x, y))
	if not opt_res.success and not np.allclose(y, 0):
		warn("Minimization warning in %s: %s\nGuessed nu=%.4f with f(nu)=%.4f" % (sys._getframe().f_code.co_name, opt_res.message, x, y))
	if y1 < y:
		print("f(1)=%.4f < f(%.4f)=%.4f, Setting nu=1" % (y1, x, y))
		x = 1.0
	if PLOT:
		fs = plt.rcParams['figure.figsize']
		fig, ax = plt.subplots(1, 2, figsize=(fs[0] * 2, fs[1]))
		ax1,ax2 = ax
		ax1.plot(t, dNdt, 'ok')
		ax1.plot(t, dNdt_smooth, '--k')
		ax1.axvline(t[i], color='k', ls='--')
		ax1.axhline(dNdt[i], color='k', ls='--')
		ax1.set_xlabel('Time')
		ax1.set_ylabel('dN/dt')
		
		ax2.plot(np.logspace(-3,3), target(np.logspace(-3, 3)), 'k-')
		ax2.set_xlabel(r'$\nu$')
		ax2.set_ylabel('Target function')
		ax2.set_xscale('log')
		
		fig.tight_layout()        
		return x, fig, ax
	return x[0]


def guess_r(t, N, nu=None, K=None):
	r"""Guesses the value of *r* from the shape of the growth curve.

	Following [5]_:

	.. math::

		\frac{dN}{dt}_{max} = r K \nu (1 + \nu)^{-\frac{1 + \nu}{\nu}}


	- :math:`\frac{dN}{dt}_{max}`: maximum population growth rate
	- r: initial per capita growth rate 
	- K: maximum population size
	- :math:`\nu`: curvature of the logsitic term

	Parameters
	----------
	t : numpy.ndarray
		time
	N : numpy:ndarray
		`N[i]` is the population size at time `t[i]`
	nu : float, optional
		a guess of `nu`, the maximum population size. If not given, it is guessed.
	K : float, optional
		a guess of `K`, the curvature of the logsitic term. If not given, it is guessed.

	Returns
	-------
	float
		the guess of *r*.

	References
	----------
	.. [5] Richards, F. J. 1959. `A Flexible Growth Function for Empirical Use <http://dx.doi.org/10.1093/jxb/10.2.290>`_. Journal of Experimental Botany
	"""
	if K is None:
		K = N.max()
	if nu is None:
		nu = guess_nu(t, N, K)
	idx = N >= K/np.exp(1)
	t = t[idx]
	N = N[idx]

	dNdt = np.gradient(N, t[1]-t[0])
	smoothed = smooth(t, dNdt)
	dNdtmax = smoothed.max()    
	
	return dNdtmax / (K * nu * (1 + nu)**(-(1 + nu) / nu))


def guess_q0_v(t, N, param_guess):		
	param_fix = {'y0', 'K', 'r', 'nu'}
	param_guess = dict(param_guess)
	if 'q0' in param_guess:
		param_fix.add('q0')
	if 'v' in param_guess:
		param_fix.add('v')
	param_guess['q0'] = param_guess.get('q0', 1)
	param_guess['v'] = param_guess.get('v', 1)
	model = BaranyiRobertsModel()
	params = model.guess(data=N, t=t, param_guess=param_guess, param_fix=param_fix)
	result = model.fit(data=N, t=t, params=params)	
	return result.best_values['q0'], result.best_values['v']


class BaranyiRobertsModel(lmfit.model.Model):
	"""TODO
	"""

	def __init__(self, *args, **kwargs):
		if kwargs.get('stepfunc', USE_STEP_FUNC):
			super(BaranyiRobertsModel, self).__init__(baranyi_roberts_step_function, *args, **kwargs)
		else:
			super(BaranyiRobertsModel, self).__init__(baranyi_roberts_function, *args, **kwargs)


	def guess(self, data, t, param_guess=None, param_min=None, param_max=None, param_fix=None):		
		if (sorted(t) != t).all():
			raise ValueError("Time argument t must be sorted")

		param_guess = dict() if param_guess is None else dict(param_guess)
		param_min = dict() if param_min is None else dict(param_min)
		param_max = dict() if param_max is None else dict(param_max)
		param_fix = set() if param_fix is None else set(param_fix)
		
		if 'y0' not in param_guess:
			param_guess['y0'] = np.min(data)
		if 'K' not in param_guess:
			param_guess['K'] = np.max(data)
		if 'nu' not in param_guess:
			param_guess['nu'] = guess_nu(t, data, K=param_guess['K'])
		if 'r' not in param_guess:
			param_guess['r'] = guess_r(t, data, K=param_guess['K'], nu=param_guess['nu'])
		if 'v' not in param_guess and 'v' in param_fix:
			param_guess['v'] = param_guess['r']
		if ('v' in param_guess and np.isposinf(param_guess['v'])) or ('q0' in param_guess and np.isposinf(param_guess['q0'])):
			param_guess['q0'], param_guess['v'] = np.inf, np.inf
		if 'v' not in param_guess or 'q0' not in param_guess:
			param_guess['q0'], param_guess['v'] = guess_q0_v(t, data, param_guess)

		params = lmfit.parameter.Parameters()
		for pname in baranyi_roberts_function.__code__.co_varnames:
			if pname in param_guess:
				params.add(
					name 	= pname,
					value 	= param_guess[pname],
					min 	= param_min.get(pname, MIN_VALUE),
					max 	= param_max.get(pname, np.inf),
					vary 	= pname not in param_fix
				)
		if 'v' in param_fix and param_guess['v'] == param_guess['r']:
			params['v'].set(expr='r')
		return params


	def get_sympy_expr(self, params):	
		t, y0, K, r, nu, q0, v = sympy.symbols('t y0 K r nu q0 v')
		args = [y0, K, r, nu, q0, v]
		if not params['y0'].vary:
			args.remove(y0)
			y0 = params['y0'].value
		if not params['K'].vary:			
			args.remove(K)			
			K = params['K'].value
		if not params['r'].vary:
			args.remove(r)
			r = params['r'].value
		if not params['nu'].vary:
			args.remove(nu)
			nu = params['nu'].value
		if not params['q0'].vary:
			args.remove(q0)
			q0 = params['q0'].value
		if not params['v'].vary:
			args.remove(v)
			v = params['v'].value
		if (isinstance(q0, float) and np.isinf(q0)) or (isinstance(v, float) and np.isinf(v)):
			At = t
		else:
			At = t + 1.0 / v * sympy.log((sympy.exp(-v * t) + q0) / (1 + q0))
		dNdt = K / (1.0 - (1.0 - (K / y0)**nu) * sympy.exp(-r * nu * At))**(1.0 / nu)
		return dNdt, t, tuple(args)



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


def randomize(t=12, y0=0.1, K=1.0, r=0.1, nu=1, q0=np.inf, v=np.inf, reps=1, noise_std=0.02, noise_func=noisify_lognormal_multiplicative, random_seed=None, as_df=True):
	if isinstance(t, numbers.Number):
		t = np.linspace(0, t)
	y = baranyi_roberts_function(t, y0, K, r, nu, q0, v)
	y.resize((len(t),))
	y = y.repeat(reps)
	y.resize((len(t), reps))
	y = noise_func(y, noise_std, random_seed)
	y[y < 0] = 0
	y = y.flatten()
	t = t.repeat(reps)
	if as_df:
		return pd.DataFrame({'OD': y, 'Time': t})
	else:
		return t, y


def nvarys(params):
	return len([p for p in params.values() if p.vary])


def lag(result):
	q0 = result.best_values['q0']
	v = result.best_values['v']
	if np.isinf(q0) or np.isinf(v):
		return 0
	return np.log(1.0 + 1.0 / q0) / v


if __name__ == '__main__':
	t, y = randomize(t=12, y0=0.12, K=0.56, r=0.8, nu=3.0, q0=0.2, v=0.8, reps=1, as_df=False, random_seed=0)
	plt.plot(t, y, 'o')
	plt.show()
	print("Step", USE_STEP_FUNC)

	br6_model = BaranyiRobertsModel()
	br6_params = br6_model.guess(data=y, t=t)
	assert nvarys(br6_params) == 6
	br6_result = br6_model.fit(data=y, t=t, params=br6_params)
	assert br6_result.nvarys == 6
	print(br6_result.model.name, br6_result.nvarys, br6_result.bic, lag(br6_result))

	br5_model = BaranyiRobertsModel()
	br5_params = br5_model.guess(data=y, t=t, param_guess={'nu':1}, param_fix=['nu'])
	assert nvarys(br5_params) == 5, br5_params
	assert br5_params['nu'].value == 1
	br5_result = br5_model.fit(data=y, t=t, params=br5_params)
	assert br5_result.nvarys == 5
	assert br5_result.best_values['nu'] == 1
	print(br5_result.model.name, br5_result.nvarys, br5_result.bic, lag(br5_result))


	br5b_model = BaranyiRobertsModel()
	br5b_params = br5b_model.guess(data=y, t=t, param_fix=['v'])
	assert nvarys(br5b_params) == 5, br5b_params
	assert br5b_params['v'].value == br5b_params['r'].value
	br5b_result = br5b_model.fit(data=y, t=t, params=br5b_params)
	assert br5b_result.nvarys == 5
	assert br5b_result.best_values['v'] == br5b_result.best_values['r']
	print(br5b_result.model.name, br5b_result.nvarys, br5b_result.bic, lag(br5b_result))


	br4_model = BaranyiRobertsModel()
	br4_params = br4_model.guess(data=y, t=t, param_guess={'nu':1}, param_fix=['nu', 'v'])
	assert nvarys(br4_params) == 4, br4_params
	assert br4_params['nu'].value == 1
	assert br4_params['v'].value == br4_params['r'].value
	br4_result = br4_model.fit(data=y, t=t, params=br4_params)
	assert br4_result.nvarys == 4
	assert br4_result.best_values['v'] == br4_result.best_values['r']
	print(br4_result.model.name, br4_result.nvarys, br4_result.bic, lag(br4_result))

	richards_model = BaranyiRobertsModel()
	richards_params = richards_model.guess(data=y, t=t, param_guess={'v':np.inf}, param_fix=['q0', 'v'])
	assert nvarys(richards_params) == 4
	assert np.isposinf(richards_params['v'].value)
	assert np.isposinf(richards_params['q0'].value)
	richards_result = richards_model.fit(data=y, t=t, params=richards_params)
	assert richards_result.nvarys == 4
	assert np.isposinf(richards_result.best_values['v'])
	assert np.isposinf(richards_result.best_values['q0'])
	print(richards_result.model.name, richards_result.nvarys, richards_result.bic, lag(richards_result))

	logistic_model = BaranyiRobertsModel()
	logistic_params = logistic_model.guess(data=y, t=t, param_guess={'nu':1, 'v':np.inf}, param_fix=['nu', 'v', 'q0'])
	assert nvarys(logistic_params) == 3
	assert logistic_params['nu'].value == 1
	assert np.isposinf(logistic_params['v'].value)
	assert np.isposinf(logistic_params['q0'].value)
	logistic_result = logistic_model.fit(data=y, t=t, params=logistic_params)
	assert logistic_result.nvarys == 3
	assert logistic_result.best_values['nu'] == 1
	assert np.isposinf(logistic_result.best_values['v'])
	assert np.isposinf(logistic_result.best_values['q0'])
	print(logistic_result.model.name, logistic_result.nvarys, logistic_result.bic, lag(logistic_result))
	