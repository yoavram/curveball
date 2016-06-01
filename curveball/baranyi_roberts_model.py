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
import inspect
from warnings import warn
import numpy as np
from scipy.optimize import minimize
from scipy.misc import derivative
import matplotlib.pyplot as plt
import pandas as pd
import lmfit
import sympy
import curveball
from curveball.utils import smooth


MIN_VALUES = {
	# based on nothing
	'y0': 1e-4,
	'K': 1e-4,
	'r': 1e-4,	
	'nu': 1e-1,
	'q0': 1e-4,
	'v': 1e-4
}

MAX_VALUES = {
	# based on min doubling time of ~5 minutes
	# r = 60 * log(2) / x where x is doubling time in minutes
	'r': 8,
	# based on nothing
	'nu': 10,
}


def lag(model_result=None, q0=None, v=None):
	r"""Calculate lag duration from model Parameters.

	Lag duration :math:`\lambda` is calculated as [Baranyi1994]_:

	.. math::

		\lambda = \frac{1}{v} \log{\Big( 1 + \frac{1}{q_0} \Big)}

	This definition follows from the Baranyi-Roberts model definition of :math:`A(t)`:

	.. math::

		A(t) - t = \frac{1}{v} \log{\Big(\frac{e^{-vt} + q_0}{1 + q_0} \Big)} \to_{t \to \infty} \lambda

	References
	----------
	.. [Baranyi1994] Baranyi, J., Roberts, T. A., 1994. `A dynamic approach to predicting bacterial growth in food <www.ncbi.nlm.nih.gov/pubmed/7873331>`_. Int. J. Food Microbiol.

	"""
	if model_result is not None:
		q0 = model_result.best_values.get('q0', np.inf)
		v = model_result.best_values.get('v', model_result.best_values['r'])
	elif q0 is None or v is None:
		raise ValueError("Either model_result or q0 and v should be given")
	if np.isinf(q0) or np.isinf(v):
		return 0.0
	return np.log(1.0 + 1.0 / q0) / v


def baranyi_roberts_function(t, y0, K, r, nu, q0, v):
	r"""The Baranyi-Roberts growth model is an extension of the logistic and Richards model that adds a lag phase [Baranyi1994]_.

	.. math::

		\frac{dy}{dt} = r \alpha(t) y \Big( 1 - \Big(\frac{y}{K}\Big)^{\nu} \Big) \Rightarrow

		y(t) = \frac{K}{\Big[1 - \Big(1 - \Big(\frac{K}{y_0}\Big)^{\nu}\Big) e^{-r \nu A(t)}\Big]^{1/\nu}}

		A(t) = \int_0^t{\alpha(s)ds} = \int_0^t{\frac{q_0}{q_0 + e^{-v s}} ds} = t + \frac{1}{v} \log{\Big( \frac{e^{-v t} + q0}{1 + q0} \Big)}


	- :math:`y_0`: initial population size
	- r: initial per capita growth rate
	- K: maximum population size
	- :math:`\nu`: curvature of the logsitic term
	- :math:`q_0`: initial physiological state of the population
	- v: rate of physiological state adjustment

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
		initial physiological state of the population (:math:`0<q_0<1`)
	v : float
		rate of physiological state adjustment (:math:`v>0`)

	Returns
	-------
	numpy.ndarray
		population size per time point in `t`.

	References
	----------
	.. [Baranyi1994] Baranyi, J., Roberts, T. A., 1994. `A dynamic approach to predicting bacterial growth in food <www.ncbi.nlm.nih.gov/pubmed/7873331>`_. Int. J. Food Microbiol.
	"""
	if np.isposinf(v):
		v = r
	if np.isposinf(q0):
		At = t
	else:
		At = t + (1.0 / v) * np.log((np.exp(-v * t) + q0) / (1.0 + q0))
	if np.isposinf(K):
		return y0 * np.exp(r * nu * At)
	else:
		return K / ((1 - (1 - (K / y0)**nu) * np.exp(-r * nu * At))**(1.0/nu))


def guess_nu(t, N, K=None, PLOT=False, PRINT=False):
	r"""Guesses the value of :math:`\nu` from the shape of the growth curve.

	Following [Richards1959]_:

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
		if :const:`True`, the function will plot the calculations.
	PRINT : bool, optional
		if :const:`True`, the function will print intermediate results of the calculations.

	Returns
	-------
	x : float
		the guess of :math:`\nu`.
	fig : matplotlib.figure.Figure
		if the argument `PLOT` was :const:`True`, the generated figure.
	ax : matplotlib.axes.Axes
		if the argument `PLOT` was :const:`True`, the generated axis.

	References
	----------
	.. [Richards1959] Richards, F. J. 1959. `A Flexible Growth Function for Empirical Use <http://dx.doi.org/10.1093/jxb/10.2.290>`_. Journal of Experimental Botany
	"""
	t, N = np.array(t), np.array(N)
	if K is None:
		K = N.max()
	# smooth and calculate derivative
	N_smooth = smooth(t, N) 	
	# calculate derivative	
	dNdt = derivative(N_smooth, t)
	assert len(t) > 0
	assert len(t) == len(dNdt)
	# only use the second part of the curve starting from N=K/e (inflexion point for nu=0)
	idx = N >= K * np.exp(-1)
	if idx.sum() < 10:
		warn('Less than 10 data points above K/e!')	
	t, N, dNdt = t[idx], N[idx], dNdt[idx]
	# find N at inflexion point
	i = dNdt.argmax()
	Ninf = N_smooth(t[i])
	# find nu that gives Ninf at inflexion point
	def target(nu):
		return np.abs(K * (1 + nu)**(-1.0 / nu)  - Ninf)
	opt_res = minimize(target, x0=1, bounds=[(0, None)])
	x = opt_res.x
	y = target(x)
	y1 = target(1.0)
	
	if PRINT:
		print("f(1)=%.4f, f(%.4f)=%.4f" % (y1, x, y))
	if not opt_res.success and not np.allclose(y, 0):
		warn("Minimization warning in %s: %s\nGuessed nu=%.4f with f(nu)=%.4f" % (sys._getframe().f_code.co_name, opt_res.message, x, y))
	if y1 < y:
		warn("f(1)=%.4f < f(%.4f)=%.4f, Setting nu=1" % (y1, x, y))
		x = 1.0
	if PLOT:
		fs = plt.rcParams['figure.figsize']
		fig, ax = plt.subplots(1, 3, figsize=(fs[0] * 3, fs[1]))
		ax0, ax1, ax2 = ax
		
		N_t = N_smooth(t)
		ax0.plot(t, N_t, '-k')
		ax0.axvline(t[i], color='k', ls='--')
		ax0.axhline(N_t[i], color='k', ls='--')
		ax0.set_xlabel('Time')
		ax0.set_ylabel('N')
		
		ax1.plot(t, dNdt, '-k')
		ax1.axvline(t[i], color='k', ls='--')
		ax1.axhline(dNdt[i], color='k', ls='--')
		ax1.set_xlabel('Time')
		ax1.set_ylabel('dN/dt')
		dN = (dNdt.max() - dNdt.min()) / 10
		ax1.set_ylim(dNdt.min() - dN, dNdt.max() + dN)
		
		ax2.plot(np.logspace(-3,3), target(np.logspace(-3, 3)), 'k-')
		ax2.axvline(x, color='k', ls='--')
		ax2.axhline(y, color='k', ls='--')
		ax2.set_xlabel(r'$\nu$')
		ax2.set_ylabel('Target function')
		ax2.set_xscale('log')
		
		fig.tight_layout()        
		return x[0], fig, ax
	return x[0]


def guess_r(t, N, nu=None, K=None, PLOT=False):
	r"""Guesses the value of *r* from the shape of the growth curve.

	Following [Richards1959]_:

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
	.. [Richards1959] Richards, F. J. 1959. `A Flexible Growth Function for Empirical Use <http://dx.doi.org/10.1093/jxb/10.2.290>`_. Journal of Experimental Botany
	"""
	t, N = np.array(t), np.array(N)
	if K is None:
		K = N.max()
	if nu is None:
		nu = guess_nu(t, N, K, PLOT=False, PRINT=False)
	
	# smooth and calculate derivative
	N_smooth = smooth(t, N) 	
	# calculate derivative	
	dNdt = derivative(N_smooth, t)
	# limit max search
	idx = N >= K * np.exp(-1) / 4
	t, dNdt = t[idx], dNdt[idx]
	i = dNdt.argmax()
	dNdtmax = dNdt[i]
	r = dNdtmax / (K * nu * (1 + nu)**(-(1 + nu) / nu))
	if PLOT:
		fs = plt.rcParams['figure.figsize']
		fig, ax = plt.subplots(1, 3, figsize=(fs[0] * 3, fs[1]))
		ax0, ax1, ax2 = ax
		
		N_t = N_smooth(t)
		ax0.plot(t, N_t, '-k')
		ax0.axvline(t[i], color='k', ls='--')
		ax0.axhline(N_t[i], color='k', ls='--')
		ax0.set_xlabel('Time')
		ax0.set_ylabel('N')
		
		ax1.plot(t, dNdt, '-k')
		ax1.axvline(t[i], color='k', ls='--')
		ax1.axhline(dNdt[i], color='k', ls='--')
		ax1.set_xlabel('Time')
		ax1.set_ylabel('dN/dt')
		dN = (dNdt.max() - dNdt.min()) / 10
		ax1.set_ylim(dNdt.min() - dN, dNdt.max() + dN)
		
		rs = dNdt / (K * nu * (1 + nu)**(-(1 + nu) / nu))
		ax2.plot(t, rs, '-k')
		ax2.axvline(t[i], color='k', ls='--')
		ax2.axhline(r, color='k', ls='--')
		ax2.set_xlabel('Time')
		ax2.set_ylabel('r')		

		fig.tight_layout()        
		return r, fig, ax
	return r


def guess_q0_v(t, N, param_guess, PLOT=False):
    r"""Guesses the values of :math:`q_0` and :math:`v` by fitting a model to a curve with other parameters fixed.
    
    Parameters
    ----------
    t : numpy.ndarray
        time
    N : numpy:ndarray
        `N[i]` is the population size at time `t[i]`
    param_guess : dict
        dictionary of parameter guesses for ``K``, ``y0``, ``nu``, ``r`` and possibly ``v``.
    PLOT : bool, optional
        if :const:`True`, the function will plot the calculations.
    
    Returns
    -------
    q0, v : float
        the guess of :math:`\q_0` and ``v``.
    fig : matplotlib.figure.Figure
        if the argument `PLOT` was :const:`True`, the generated figure.
    ax : matplotlib.axes.Axes
        if the argument `PLOT` was :const:`True`, the generated axis.
    """
    param_guess = dict(param_guess)
    param_fix = set(param_guess.keys())    
    param_guess['q0'] = param_guess.get('q0', 1) # default guess is 1
    param_guess['v'] = param_guess.get('v', 1) # default guess is 1
    model = curveball.baranyi_roberts_model.BaranyiRoberts()
    params = model.guess(data=N, t=t, param_guess=param_guess, param_fix=param_fix)
    result = model.fit(data=N, t=t, params=params)
    if PLOT:
        ax = result.plot_fit()
        return result.best_values['q0'], result.best_values['v'], ax.figure, ax
    return result.best_values['q0'], result.best_values['v']


class BaranyiRoberts(lmfit.model.Model):
	"""The Baranyi-Roberts growth model is an extension of the logistic and Richards model that adds a lag phase.	
	"""

	def __init__(self, *args, **kwargs):
		if args:
			func, args = args[0], args[1:]
		else:
			func = baranyi_roberts_function
		super(BaranyiRoberts, self).__init__(func, *args, **kwargs)
		self.name = self.__class__.__name__
		self.nested_models = {
			'nu': LogisticLag2,
			'lag': Richards
		}


	def guess(self, data, t, param_guess=None, param_min=None, param_max=None, param_fix=None):
		"""Use heuristics to guess model parameters.

		Parameters
		----------
		data : numpy.ndarray
			population size or density data
		t : numpy.ndarray
			time, usually in hours
		param_guess : dict, optional
			mapping parameter names to values used as guesses to override the heuristic guess
		param_min, param_max : dict, optional
			mapping parameter names to values used to bound the parametrs in the fit procedure
		param_fix : set, optional
			names of parameters to fix at the guessed value, i.e., no to estimate in the fit procedure

		Returns
		-------
		lmfit.parameter.Parameters
			a dictionary of the model parameters ready to be used by the `fit` method

		See also
		--------
		lmfit.model.Model.guess
		"""	
		if (sorted(t) != t).all():
			raise ValueError("Time argument t must be sorted")

		param_guess = dict() if param_guess is None else dict(param_guess)
		param_min = dict() if param_min is None else dict(param_min)
		param_max = dict() if param_max is None else dict(param_max)
		param_fix = set() if param_fix is None else set(param_fix)
		
		if 'y0' not in param_guess:
			param_guess['y0'] = (data[t == t.min()]).mean()
		if 'K' not in param_guess:
			param_guess['K'] = (data[t == t.max()]).mean()
		if 'nu' not in self.param_names:
			nu = 1.0
		elif 'nu' not in param_guess:
			# guess_nu has not been performing well, just use 1.0
			# param_guess['nu'] = guess_nu(t, data, K=param_guess['K'], PLOT=False, PRINT=False)
			param_guess['nu'] = 1.0 
			nu = param_guess['nu']
		else:
			nu = param_guess['nu']
		if 'r' not in param_guess:
			param_guess['r'] = guess_r(t, data, K=param_guess['K'], nu=nu)		
		if 'q0' not in param_guess:
			if 'v' not in self.param_names or 'v' in param_guess:
				param_guess['q0'], _ = guess_q0_v(t, data, param_guess)
			else:
				param_guess['q0'], param_guess['v'] = guess_q0_v(t, data, param_guess)

		params = lmfit.parameter.Parameters()
		for pname in self.param_names:		
			params.add(
				name 	= pname,
				value 	= param_guess[pname],
				min 	= param_min.get(pname, MIN_VALUES.get(pname, 0)),
				max 	= param_max.get(pname, MAX_VALUES.get(pname, np.inf)),
				vary 	= pname not in param_fix
			)
		# if 'q0' in params:
		# 	if 'v' in params:
		# 		params.add('lag', value=_lag(q0=param_guess['q0'], v=param_guess['v']), min=1.0/60.0, vary=params['q0'].vary)
		# 		params['q0'].set(expr='1/(exp(v / 60.0) - 1)')
		# 	else:
		# 		params.add('lag', value=_lag(q0=param_guess['q0'], v=param_guess['r']), min=1.0/60.0, vary=params['q0'].vary)
		# 		params['q0'].set(expr='1/(exp(r / 60.0) - 1)')
			
		return params


	def get_sympy_expr(self, params):
		"""Generate the required variables for creating a Dfun to be used in the fitting procedure.

		Parameters
		----------
		params : dict
			a dictionary of the model parameters, possibly created by :py:method:`guess`.

		Returns
		-------
			dNdt : sympy.expr.Expr
				expression for the derivate of the model function
			t : sympy.symbol.symbol
			ma	time symbol
			args : tuple of sympy.symbol.symbol
				tuple of symbols of model parameters

		See also
		--------
		curveball.models.make_Dfun
		"""
		t, y0, K, r, nu, q0, v = sympy.symbols('t y0 K r nu q0 v')
		args = [y0, K, r, nu, q0, v]
		# remove fixed params and replace their symbols by their values
		# for params that don't exist (in inherited models), remove them and replace with a default value
		if not params['y0'].vary:
			args.remove(y0)
			y0 = params['y0'].value
		if 'K' not in self.param_names:
			args.remove(K)
			K = np.inf
		elif not params['K'].vary:
			args.remove(K)
			K = params['K'].value
		if not params['r'].vary:
			args.remove(r)
			r = params['r'].value
		if 'nu' not in self.param_names:
			args.remove(nu)
			nu = 1.0
		elif not params['nu'].vary:
			args.remove(nu)
			nu = params['nu'].value
		if 'q0' not in self.param_names:
			args.remove(q0)
			q0 = np.inf
		elif not params['q0'].vary:
			args.remove(q0)
			q0 = params['q0'].value
		if 'v' not in self.param_names:
			args.remove(v)
			v = r
		elif not params['v'].vary:
			args.remove(v)
			v = params['v'].value
		if (isinstance(q0, float) and np.isinf(q0)) or (isinstance(v, float) and np.isinf(v)):
			At = t
		else:
			At = t + 1.0 / v * sympy.log((sympy.exp(-v * t) + q0) / (1 + q0))
		dNdt = K / (1.0 - (1.0 - (K / y0)**nu) * sympy.exp(-r * nu * At))**(1.0 / nu)
		return dNdt, t, tuple(args)


class Richards(BaranyiRoberts):
	r"""The Richards model [Richards1959]_ extends the logistic model by including the :math:`nu` parameter that controls the
	curvature of the response of the growth rate to population size/density. Also knows as the :math:`\theta`-logistic model[Gilpin1973]_.

	References
	----------
	.. [Richards1959] Richards, F. J. 1959. "A Flexible Growth Function for Empirical Use." Journal of Experimental Botany 10 (2): 290-301. doi:10.1093/jxb/10.2.290.
	.. [Gilpin1973] Gilpin, Michael E., and Francisco J. Ayala. 1973. "Global Models of Growth and Competition." Proceedings of the National Academy of Sciences of the United States of America 70 (12 Pt 1-2): 3590–3593. doi:10.1073/pnas.70.12.3590.
	"""
	def __init__(self, *args, **kwargs):
		def func(t, y0, K, r, nu):			
			return baranyi_roberts_function(t, y0, K, r, nu, np.inf, np.inf)
		super(Richards, self).__init__(func, *args, **kwargs)
		self.nested_models = {
			'nu': Logistic
		}

class RichardsLag1(BaranyiRoberts):
	r"""This is an extension of the :py:class:`Richards` that includes a single lag parameter :math:`q_0` (and sets :math:`v=r`).
	"""
	def __init__(self, *args, **kwargs):
		def func(t, y0, K, r, nu, q0): 
			return baranyi_roberts_function(t, y0, K, r, nu, q0, r)
		super(RichardsLag1, self).__init__(func, *args, **kwargs)
		self.nested_models = {
			'nu': LogisticLag1,
			'lag': Richards
		}

class LogisticLag2(BaranyiRoberts):
	r"""This is an extension of the :py:class:`Logistic` model that includes a two lag parameters :math:`q_0` and :math:`v`).	
	"""
	def __init__(self, *args, **kwargs):
		def func(t, y0, K, r, q0, v): 
			return baranyi_roberts_function(t, y0, K, r, 1.0, q0, v)
		super(LogisticLag2, self).__init__(func, *args, **kwargs)
		self.nested_models = {
			'lag': Logistic
		}

class LogisticLag1(BaranyiRoberts):
	r"""This is an extension of the :py:class:`Logistic` that includes a single lag parameter :math:`q_0` (and sets :math:`v=r`).
	"""
	def __init__(self, *args, **kwargs):
		def func(t, y0, K, r, q0):
			return baranyi_roberts_function(t, y0, K, r, 1.0, q0, r)
		super(LogisticLag1, self).__init__(func, *args, **kwargs)
		self.nested_models = {
			'lag': Logistic
		}

class Logistic(BaranyiRoberts):
	r"""This is the simplest growth model with a stationary phase, defined by the ODE:

	.. math::
	
		\frac{dN}{dt} = r N \Big( 1 - \frac{N}{K} \Big)
	
	Notes
	-----
	`Wikipedia <https://en.wikipedia.org/wiki/Logistic_function#In_ecology:_modeling_population_growth>`_
	"""
	def __init__(self, *args, **kwargs):
		def func(t, y0, K, r):
			return baranyi_roberts_function(t, y0, K, r, 1.0, np.inf, np.inf)
		super(Logistic, self).__init__(func, *args, **kwargs)
		self.nested_models = {}


if __name__ == '__main__':
	def nvarys(params): 
		return len([p for p in params.values() if p.vary])
	from curveball.models import randomize

	t, y = randomize(t=12, y0=0.12, K=0.56, r=0.8, nu=3.0, q0=0.2, v=0.8, reps=1, as_df=False, random_seed=0)
	plt.plot(t, y, 'o')
	plt.show()
	print("Step", USE_STEP_FUNC)

	br6_model = BaranyiRoberts()
	br6_params = br6_model.guess(data=y, t=t)
	assert nvarys(br6_params) == 6
	br6_result = br6_model.fit(data=y, t=t, params=br6_params)
	assert br6_result.nvarys == 6
	print(br6_result.model.name, br6_result.nvarys, br6_result.bic, _lag(br6_result))

	br5_model = LogisticLag2()
	br5_params = br5_model.guess(data=y, t=t, param_guess={'nu':1}, param_fix=['nu'])
	assert nvarys(br5_params) == 5, br5_params
	assert br5_params['nu'].value == 1
	br5_result = br5_model.fit(data=y, t=t, params=br5_params)
	assert br5_result.nvarys == 5
	assert br5_result.best_values['nu'] == 1
	print(br5_result.model.name, br5_result.nvarys, br5_result.bic, _lag(br5_result))


	br5b_model = RichardsLag1()
	br5b_params = br5b_model.guess(data=y, t=t, param_fix=['v'])
	assert nvarys(br5b_params) == 5, br5b_params
	assert br5b_params['v'].value == br5b_params['r'].value
	br5b_result = br5b_model.fit(data=y, t=t, params=br5b_params)
	assert br5b_result.nvarys == 5
	assert br5b_result.best_values['v'] == br5b_result.best_values['r']
	print(br5b_result.model.name, br5b_result.nvarys, br5b_result.bic, _lag(br5b_result))


	br4_model = LogisticLag1()
	br4_params = br4_model.guess(data=y, t=t, param_guess={'nu':1}, param_fix=['nu', 'v'])
	assert nvarys(br4_params) == 4, br4_params
	assert br4_params['nu'].value == 1
	assert br4_params['v'].value == br4_params['r'].value
	br4_result = br4_model.fit(data=y, t=t, params=br4_params)
	assert br4_result.nvarys == 4
	assert br4_result.best_values['v'] == br4_result.best_values['r']
	print(br4_result.model.name, br4_result.nvarys, br4_result.bic, _lag(br4_result))

	richards_model = Richards()
	richards_params = richards_model.guess(data=y, t=t, param_guess={'v':np.inf}, param_fix=['q0', 'v'])
	assert nvarys(richards_params) == 4
	assert np.isposinf(richards_params['v'].value)
	assert np.isposinf(richards_params['q0'].value)
	richards_result = richards_model.fit(data=y, t=t, params=richards_params)
	assert richards_result.nvarys == 4
	assert np.isposinf(richards_result.best_values['v'])
	assert np.isposinf(richards_result.best_values['q0'])
	print(richards_result.model.name, richards_result.nvarys, richards_result.bic, _lag(richards_result))

	logistic_model = Logistic()
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
	print(logistic_result.model.name, logistic_result.nvarys, logistic_result.bic, _lag(logistic_result))