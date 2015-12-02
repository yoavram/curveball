#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of curveball.
# https://github.com/yoavram/curveball

# Licensed under the MIT license:
# http://www.opensource.org/licenses/MIT-license
# Copyright (c) 2015, Yoav Ram <yoav@yoavram.com>
"""Benchmark Curveball model fitting.

See the `asv docs <http://asv.readthedocs.org/en/latest/writing_benchmarks.html#writing-benchmarks>`_ for more information.

Run in dev mode with (to just use benchmark function X of latest commit):
>>> asv dev -b X
Or in prod mode:
>>> asv run "HEAD^..HEAD"
"""
from __future__ import division
from builtins import str
import os
import pickle
from warnings import warn
import numpy as np
import curveball


CI = os.environ.get('CI', 'false').lower() == 'true'
RANDOM_SEED = int(os.environ.get('RANDOM_SEED', 0))
REPS = 30
NOISE_STD = 0.04
NUM_RUNS = 1000
TIMEOUT = 60.0 * 3


def relative_error(exp, obs):
	return np.abs(exp - obs) / exp


class TimingSuite:
	timeout = TIMEOUT
	params = ([True, False], [True, False])
	param_names = ['use_weights', 'use_Dfun']


	def setup(self, use_weights, use_Dfun):
		# from gh issue #22
		self.df = curveball.models.randomize(t=48, y0=0.1, K=1, r=0.75, nu=5.0, q0=0.1, v=0.1, reps=REPS, noise_std=NOISE_STD, random_seed=RANDOM_SEED)

	
	def time_fit_model(self, use_weights, use_Dfun):
		curveball.models.fit_model(self.df, use_weights=use_weights, use_Dfun=use_Dfun, PLOT=False, PRINT=False)


class ModelResultCopy:
	def __init__(self, fit_result):
		self.best_values = fit_result.best_values
		self.ndata = fit_result.ndata
		self.nvarys = fit_result.nvarys
		self.covar = fit_result.covar
		self.chisqr = fit_result.chisqr


def aggregate(agg):
	def aggregation_decorator(operator):		
		# do not use @functools.wraps as it will hide the decorated function from asv discovery
		def func_wrapper(self, cache, use_weights, use_Dfun):
			output = np.empty(NUM_RUNS)
			for i, (params, fit_result) in enumerate(cache[(use_weights, use_Dfun)]):				
				output[i] = operator(self, cache, use_weights, use_Dfun, params, fit_result)
			return agg(output)
		return func_wrapper
	return aggregation_decorator


class TrackFittingSuite:
	timeout = TIMEOUT
	params = ([True, False], [True, False])
	param_names = ['use_weights', 'use_Dfun']	

	def setup_cache(self):
		data = [None] * NUM_RUNS		
		for i in range(NUM_RUNS):
			random_seed = RANDOM_SEED + i
			rng = np.random.RandomState(random_seed)
			y0 = rng.uniform(0.1, 0.2)
			K = rng.uniform(0.4, 0.8)
			r = rng.uniform(1e-2, 1)
			nu = rng.uniform(0.1, 10)
			q0 = rng.uniform(0, 1)
			v = rng.uniform(1e-2, 1)
			params = {'y0':y0, 'K':K, 'r':r, 'nu':nu, 'q0':q0, 'v':v}
			data[i] = curveball.models.randomize(reps=REPS, noise_std=NOISE_STD, random_seed=random_seed, **params)

		cache = dict()
		for use_weights in self.params[0]:
			for use_Dfun in self.params[1]:				
				key = use_weights, use_Dfun
				cache[key] = [None] * NUM_RUNS
				for i in range(NUM_RUNS):
					fit_results = curveball.models.fit_model(data[i], use_weights=use_weights, use_Dfun=use_Dfun, PRINT=False, PLOT=False)
					best_fit = fit_results[0]
					cache[key][i] = (params, ModelResultCopy(best_fit))

		return cache

	
	@aggregate(np.mean)
	def track_mean_param_relative_error(self, cache, use_weights, use_Dfun, params, fit_result): 
		return np.mean([relative_error(pval, params[pname]) for pname, pval in fit_result.best_values.items()])

	@aggregate(np.std)
	def track_std_param_relative_error(self, cache, use_weights, use_Dfun, params, fit_result): 
		return np.mean([relative_error(pval, params[pname]) for pname, pval in fit_result.best_values.items()])

	@aggregate(np.mean)
	def track_mean_K_error(self, cache, use_weights, use_Dfun, params, fit_result):
		return params['K'] - fit_result.best_values.get('K', np.nan)

	@aggregate(np.std)
	def track_std_K_error(self, cache, use_weights, use_Dfun, params, fit_result):
		return params['K'] - fit_result.best_values.get('K', np.nan)

	@aggregate(np.mean)
	def track_mean_r_error(self, cache, use_weights, use_Dfun, params, fit_result):
		return params['r'] - fit_result.best_values.get('r', np.nan)

	@aggregate(np.std)
	def track_std_r_error(self, cache, use_weights, use_Dfun, params, fit_result):
		return params['r'] - fit_result.best_values.get('r', np.nan)

	@aggregate(np.mean)
	def track_mean_nu_error(self, cache, use_weights, use_Dfun, params, fit_result):
		return params['nu'] - fit_result.best_values.get('nu', np.nan)

	@aggregate(np.std)
	def track_std_nu_error(self, cache, use_weights, use_Dfun, params, fit_result):
		return params['nu'] - fit_result.best_values.get('nu', np.nan)

	@aggregate(np.mean)
	def track_mean_q0_error(self, cache, use_weights, use_Dfun, params, fit_result):
		return params['q0'] - fit_result.best_values.get('q0', np.nan)

	@aggregate(np.std)
	def track_std_q0_error(self, cache, use_weights, use_Dfun, params, fit_result):
		return params['q0'] - fit_result.best_values.get('q0', np.nan)
		
	@aggregate(np.mean)
	def track_mean_v_error(self, cache, use_weights, use_Dfun, params, fit_result):
		return params['v'] - fit_result.best_values.get('v', np.nan)

	@aggregate(np.std)
	def track_std_v_error(self, cache, use_weights, use_Dfun, params, fit_result):
		return params['v'] - fit_result.best_values.get('v', np.nan)

	@aggregate(np.mean)
	def track_mean_K_error(self, cache, use_weights, use_Dfun, params, fit_result):
		return params['K'] - fit_result.best_values.get('K', np.nan)

	@aggregate(np.std)
	def track_std_K_error(self, cache, use_weights, use_Dfun, params, fit_result):
		return params['K'] - fit_result.best_values.get('K', np.nan)
		
	@aggregate(np.mean)
	def track_mean_square_deviation(self, cache, use_weights, use_Dfun, params, fit_result):
		return fit_result.chisqr / fit_result.ndata

	@aggregate(np.std)
	def track_mean_square_deviation(self, cache, use_weights, use_Dfun, params, fit_result):
		return fit_result.chisqr / fit_result.ndata

	@aggregate(np.mean)
	def track_mean_covar_norm(self, cache, use_weights, use_Dfun, params, fit_result):
		return (np.linalg.norm(fit_result.covar)**2) / fit_result.nvarys
