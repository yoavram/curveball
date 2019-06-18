#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of curveball.
# https://github.com/yoavram/curveball

# Licensed under the MIT license:
# http://www.opensource.org/licenses/MIT-license
# Copyright (c) 2015, Yoav Ram <yoav@yoavram.com>
from __future__ import division
from builtins import str
from past.utils import old_div
from unittest import TestCase, main

import sys
import os
import types
import shutil
from scipy.integrate import odeint
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lmfit.model import Model, ModelResult
import matplotlib
from PIL import Image
import tempfile
import curveball


CI = os.environ.get('CI', 'false').lower() == 'true'
RANDOM_SEED = int(os.environ.get('RANDOM_SEED', 0))
REPS = 30
NOISE_STD = 0.04


def check_image(filename):      
	im = Image.open(filename)
	bands = im.split()
	return not all(band.getextrema() == (255, 255) for band in bands)


def logistic_ode(y, t, K, r, nu, q0, v):
	return r * y * (1 - y/K)


def richards_ode(y, t, K, r, nu, q0, v):
	return r * y * (1 - (y/K)**nu)


def baranyi_roberts_ode(y, t, K, r, nu, q0, v):
	alfa = q0 / (q0 + np.exp(-v * t))
	return alfa * r * y * (1 - (y/K)**nu)


def compare_curves(y1, y2):
	return (abs(y1 - y2) / y1).mean()


def relative_error(exp, obs):
	return abs(exp - obs) / exp


def mean_residual(model_result):
	return (np.abs(model_result.data - model_result.best_fit)).mean()


class FunctionsTestCase(TestCase):
	_multiprocess_can_split_ = True


	def test_logistic(self):
		y0=0.1; r=0.75; K=1.0
		t = np.linspace(0,12)
		y_curve = curveball.baranyi_roberts_model.baranyi_roberts_function(t, y0, r, K, 1, np.inf, np.inf)
		y_ode = odeint(logistic_ode, y0, t, args=(r, K, 1, np.inf, np.inf))
		y_ode.resize((len(t),))
		err = compare_curves(y_ode, y_curve)
		self.assertTrue(err < 1e-6)
	
	
	def test_richards(self):
		y0=0.1; r=0.75; K=1.0; nu=0.5
		t = np.linspace(0,12)
		y_curve = curveball.baranyi_roberts_model.baranyi_roberts_function(t, y0, r, K, nu, np.inf, np.inf)
		y_ode = odeint(richards_ode, y0, t, args=(r, K, nu, np.inf, np.inf))
		y_ode.resize((len(t),))
		err = compare_curves(y_ode, y_curve)
		self.assertTrue(err < 1e-6)
	
	
	def test_baranyi_roberts(self):
		y0=0.1; r=0.75; K=1.0; nu=0.5; q0=0.1; v=0.1
		t = np.linspace(0,12)
		y_curve = curveball.baranyi_roberts_model.baranyi_roberts_function(t, y0, r, K, nu, q0, v)
		y_ode = odeint(baranyi_roberts_ode, y0, t, args=(r, K, nu, q0, v))
		y_ode.resize((len(t),))
		err = compare_curves(y_ode, y_curve)
		self.assertTrue(err < 1e-6)


class ModelSelectionTestCase(TestCase):
	_multiprocess_can_split_ = True


	def setUp(self):
		if CI:
			self.folder = tempfile.mkdtemp()
		else:
			self.folder = '.'


	def tearDown(self):
		plt.close("all")
		if CI:
			shutil.rmtree(self.folder)


	def test_fit_model_logistic(self):
		df = curveball.models.randomize(t=12, y0=0.1, K=1, r=0.75, nu=1, reps=REPS, noise_std=NOISE_STD, random_seed=RANDOM_SEED)
		models,fig,ax = curveball.models.fit_model(df, PLOT=True, PRINT=True)
		self.assertIsInstance(fig, matplotlib.figure.Figure)
		filename = sys._getframe().f_code.co_name + ".png"
		fig.savefig(filename)
		self.assertTrue(check_image(filename))
		self.assertIsNotNone(models)
		for mod in models:
			self.assertIsInstance(mod, ModelResult)
		self.assertTrue(mean_residual(models[0]) < NOISE_STD)
		

	def test_fit_model_logistic_with_param_min(self):
		df = curveball.models.randomize(t=12, y0=0.1, K=1, r=0.75, nu=1, reps=REPS, noise_std=NOISE_STD, random_seed=RANDOM_SEED)       
		models,fig,ax = curveball.models.fit_model(df, PLOT=True, PRINT=True, param_min={'y0': 0.2})
		self.assertIsInstance(fig, matplotlib.figure.Figure)
		filename = sys._getframe().f_code.co_name + ".png"
		fig.savefig(filename)
		self.assertTrue(check_image(filename))
		self.assertIsNotNone(models)
		for mod in models:
			self.assertIsInstance(mod, ModelResult)
		self.assertTrue(models[0].best_values['y0'] >= 0.2)


	def test_fit_model_logistic_with_param_max(self):
		df = curveball.models.randomize(t=12, y0=0.1, K=1, r=0.75, nu=1, reps=REPS, noise_std=NOISE_STD, random_seed=RANDOM_SEED)
		models,fig,ax = curveball.models.fit_model(df, PLOT=True, PRINT=True, param_max={'K': 0.5})
		self.assertIsInstance(fig, matplotlib.figure.Figure)
		filename = sys._getframe().f_code.co_name + ".png"
		fig.savefig(filename)
		self.assertTrue(check_image(filename))
		self.assertIsNotNone(models)
		for mod in models:
			self.assertIsInstance(mod, ModelResult)
		self.assertTrue(models[0].best_values['K'] <= 0.5)


	def test_fit_model_logistic_with_param_fix(self):
		df = curveball.models.randomize(t=12, y0=0.1, K=1, r=0.75, nu=1, reps=REPS, noise_std=NOISE_STD, random_seed=RANDOM_SEED)
		models,fig,ax = curveball.models.fit_model(df, PLOT=True, PRINT=True, param_guess={'K': 1}, param_fix={'K'})
		self.assertIsInstance(fig, matplotlib.figure.Figure)
		filename = sys._getframe().f_code.co_name + ".png"
		fig.savefig(filename)
		self.assertTrue(check_image(filename))
		self.assertIsNotNone(models)
		for mod in models:
			self.assertIsInstance(mod, ModelResult)
		self.assertEqual(models[0].nvarys, 2) # one less param as we fixed K
		self.assertEqual(models[0].best_values['K'], 1)
		self.assertEqual(models[0].best_values.get('nu', 1), 1)
		self.assertEqual(models[0].best_values.get('v', np.inf), np.inf)
		self.assertEqual(models[0].best_values.get('q0', np.inf), np.inf)
		self.assertTrue(mean_residual(models[0]) < NOISE_STD)


	def test_fit_model_logistic_single_rep(self):
		df = curveball.models.randomize(t=12, y0=0.1, K=1, r=0.75, nu=1, reps=1, noise_std=NOISE_STD, random_seed=RANDOM_SEED)
		models,fig,ax = curveball.models.fit_model(df, PLOT=True, PRINT=True)
		self.assertIsInstance(fig, matplotlib.figure.Figure)
		filename = sys._getframe().f_code.co_name + ".png"
		fig.savefig(filename)
		self.assertTrue(check_image(filename))
		self.assertIsNotNone(models)
		for mod in models:
			self.assertIsInstance(mod, ModelResult)
		self.assertTrue(mean_residual(models[0]) < NOISE_STD)



	def test_fit_model_richards_nu_05(self):
		df = curveball.models.randomize(t=24, y0=0.1, K=1, r=0.75, nu=0.5, reps=REPS, noise_std=NOISE_STD, random_seed=RANDOM_SEED)
		models,fig,ax = curveball.models.fit_model(df, PLOT=True, PRINT=True)
		self.assertIsInstance(fig, matplotlib.figure.Figure)
		filename = sys._getframe().f_code.co_name + ".png"
		fig.savefig(filename)
		self.assertTrue(check_image(filename))
		self.assertIsNotNone(models)
		for mod in models:
			self.assertIsInstance(mod, ModelResult)
		self.assertTrue(mean_residual(models[0]) < NOISE_STD)


	def test_fit_model_logistic_lag(self):        
		df = curveball.models.randomize(t=48, y0=0.1, K=1, r=0.75, nu=1, q0=0.1, v=0.1, reps=REPS, noise_std=NOISE_STD, random_seed=RANDOM_SEED)
		models,fig,ax = curveball.models.fit_model(df, PLOT=True, PRINT=True)
		self.assertIsInstance(fig, matplotlib.figure.Figure)
		filename = sys._getframe().f_code.co_name + ".png"
		fig.savefig(filename)
		self.assertTrue(check_image(filename))    
		self.assertIsNotNone(models)
		for mod in models:
			self.assertIsInstance(mod, ModelResult)
		self.assertTrue(mean_residual(models[0]) < NOISE_STD)


	def test_fit_model_baranyi_roberts_nu_5(self):        
		df = curveball.models.randomize(t=24, y0=0.1, K=1, r=0.75, nu=5, q0=0.1, v=0.1, reps=REPS, noise_std=NOISE_STD, random_seed=RANDOM_SEED)
		models,fig,ax = curveball.models.fit_model(df, PLOT=True, PRINT=True)
		self.assertIsInstance(fig, matplotlib.figure.Figure)
		filename = sys._getframe().f_code.co_name + ".png"
		fig.savefig(filename)
		self.assertTrue(check_image(filename))    
		self.assertIsNotNone(models)
		for mod in models:
			self.assertIsInstance(mod, ModelResult)
		self.assertTrue(mean_residual(models[0]) < NOISE_STD)


class FindKTestCase(TestCase):
	_multiprocess_can_split_ = True

	def setUp(self):
		if CI:
			self.folder = tempfile.mkdtemp()
		else:
			self.folder = '.'


	def tearDown(self):
		plt.close("all")
		if CI:
			shutil.rmtree(self.folder)

	
	def test_find_K_ci_logistic(self):
		df = curveball.models.randomize(t=12, y0=0.1, K=1, r=0.75, nu=1, reps=REPS, noise_std=NOISE_STD, random_seed=RANDOM_SEED, as_df=True)
		model = curveball.baranyi_roberts_model.Logistic()
		params = model.guess(data=df.OD, t=df.Time)
		model_fit = model.fit(data=df.OD, t=df.Time, params=params)
		param_samples = curveball.models.bootstrap_params(df, model_fit, 100)
		K_est = model_fit.params['K'].value
		K_low, K_high = curveball.models.find_K_ci(param_samples)
		self.assertTrue(K_low < K_est < K_high, "K is {0}, K CI is ({1},{2})".format(K_est, K_low, K_high))


class FindLagTestCase(TestCase):
	_multiprocess_can_split_ = True

	def setUp(self):
		if CI:
			self.folder = tempfile.mkdtemp()
		else:
			self.folder = '.'


	def tearDown(self):
		plt.close("all")
		if CI:
			shutil.rmtree(self.folder)


	def test_find_lag_logistic(self):
		t, y = curveball.models.randomize(t=12, y0=0.1, K=1, r=0.75, nu=1, reps=REPS, noise_std=NOISE_STD, random_seed=RANDOM_SEED, as_df=False)
		model = curveball.baranyi_roberts_model.Logistic()
		params = model.guess(data=y, t=t)
		result = model.fit(data=y, t=t, params=params)
		lam = curveball.models.find_lag(result)
		self.assertTrue(lam < 1, "Lambda is " + str(lam))


	def test_find_lag_ci_logistic(self):
		df = curveball.models.randomize(t=12, y0=0.1, K=1, r=0.75, nu=1, reps=REPS, noise_std=NOISE_STD, random_seed=RANDOM_SEED, as_df=True)
		model = curveball.baranyi_roberts_model.Logistic()
		params = model.guess(data=df.OD, t=df.Time)
		model_fit = model.fit(data=df.OD, t=df.Time, params=params)
		param_samples = curveball.models.bootstrap_params(df, model_fit, 100)
		lam_est = curveball.models.find_lag(model_fit)
		lam_low, lam_high = curveball.models.find_lag_ci(model_fit, param_samples)
		self.assertTrue(lam_low < lam_est < lam_high, "Lambda is {0}, Lambda CI is ({1},{2})".format(lam_est, lam_low, lam_high))


	def test_find_lag_richards(self):
		for nu in [1.0, 2.0, 5.0]:
			t, y = curveball.models.randomize(t=12, y0=0.1, K=1, r=0.75, nu=nu, reps=REPS, noise_std=NOISE_STD, random_seed=RANDOM_SEED, as_df=False)
			model = curveball.baranyi_roberts_model.Richards()
			params = model.guess(data=y, t=t)
			result = model.fit(data=y, t=t, params=params)
			lam = curveball.models.find_lag(result)
			self.assertTrue(lam < 1, "Lambda is " + str(lam))
	

	def test_find_lag_baranyi_roberts(self):
		r = 0.75
		v = r
		for nu in [1.0, 2.0, 5.0]:
			for _lam in [2., 3., 4.]:
				q0 = 1.0 / (np.exp(_lam * v) - 1.0)
				t, y = curveball.models.randomize(t=16, y0=0.1, K=1, r=r, nu=nu, q0=q0, v=r, reps=REPS, noise_std=NOISE_STD, random_seed=RANDOM_SEED, as_df=False)
				model = curveball.baranyi_roberts_model.BaranyiRoberts()
				params = model.guess(data=y, t=t)
				result = model.fit(data=y, t=t, params=params)
				lam = curveball.models.find_lag(result)			      
				self.assertTrue((_lam + 1) > lam > (_lam - 1), "Lambda is " + str(lam) + " but should be " + str(_lam))

	### fails because find_lag_ci depends on sample_params which fails when covar is weird?
	# def test_find_lag_ci_baranyi_roberts(self):        
	# 	t = np.linspace(0, 24)
	# 	y0=0.1; r=0.75; K=1.0
	# 	v = r
	# 	for nu in [1.0, 2.0, 5.0]:
	# 		for _lam in [2., 3., 4.]:
	# 			q0 = 1.0 / (np.exp(_lam * v) - 1.0)
	# 			df = curveball.models.randomize(baranyi_roberts_ode, t=t, y0=y0, r=r, K=K, nu=nu, q0=q0, v=v, reps=1)
	# 			t = df.Time.as_matrix()
	# 			y = df.OD.as_matrix()
	# 			model = curveball.baranyi_roberts_model.BaranyiRobertsModel()
	# 			params = model.guess(data=y, t=t)
	# 			result = model.fit(data=y, t=t, params=params)
	# 			print('nu={}, lam={}'.format(nu, _lam))
	# 			print(result.fit_report())				
	# 			lam = curveball.models.find_lag(result)
	# 			print('lam:', lam)
	# 			lam_low,lam_high = curveball.models.find_lag_ci(result, nsamples=1000)  
	# 			self.assertTrue(lam_low < lam < lam_high, "Lambda is {2}, Lambda CI is ({0},{1})".format(lam, lam_low, lam_high))


class FindDoublingTimeTestCase(TestCase):
	_multiprocess_can_split_ = True

	def setUp(self):
		if CI:
			self.folder = tempfile.mkdtemp()
		else:
			self.folder = '.'


	def tearDown(self):
		plt.close("all")
		if CI:
			shutil.rmtree(self.folder)


	def test_find_doubling_time_logistic(self):
		t, y = curveball.models.randomize(t=12, y0=0.1, K=1, r=0.75, nu=1, reps=REPS, noise_std=NOISE_STD, random_seed=RANDOM_SEED, as_df=False)
		model = curveball.baranyi_roberts_model.Logistic()
		params = model.guess(data=y, t=t)
		result = model.fit(data=y, t=t, params=params)
		dbl = curveball.models.find_min_doubling_time(result)
		self.assertTrue(dbl > 1, "Doubling time is " + str(dbl))


	def test_find_doubling_time_ci_logistic(self):
		df = curveball.models.randomize(t=12, y0=0.1, K=1, r=0.75, nu=1, reps=REPS, noise_std=NOISE_STD, random_seed=RANDOM_SEED, as_df=True)
		model = curveball.baranyi_roberts_model.Logistic()
		params = model.guess(data=df.OD, t=df.Time)
		model_fit = model.fit(data=df.OD, t=df.Time, params=params)
		est = curveball.models.find_min_doubling_time(model_fit)
		param_samples = curveball.models.bootstrap_params(df, model_fit, 100)
		low, high = curveball.models.find_min_doubling_time_ci(model_fit, param_samples)
		self.assertTrue(low < est < high, "Doubling time is {2}, CI is ({0},{1})".format(est, low, high))


class FindMaxGrowthTestCase(TestCase):
	_multiprocess_can_split_ = True

	def setUp(self):
		if CI:
			self.folder = tempfile.mkdtemp()
		else:
			self.folder = '.'


	def tearDown(self):
		plt.close("all")
		if CI:
			shutil.rmtree(self.folder)


	def test_find_max_growth_logistic(self):
		y0 = 0.1
		K = 1.0
		r = 0.75
		df = curveball.models.randomize(t=12, y0=y0, K=K, r=r, nu=1, reps=REPS, noise_std=NOISE_STD, random_seed=RANDOM_SEED)
		model = curveball.baranyi_roberts_model.Logistic()
		model_fit = model.fit(df.OD, t=df.Time, y0=y0, K=K, r=r)
		  
		t1,y1,a,t2,y2,mu = curveball.models.find_max_growth(model_fit)

		self.assertTrue(relative_error(K / 2.0, y1) < 1, "y1=%.4g, K/2=%.4g" % (y1, K / 2.0))
		self.assertTrue(relative_error(K * r / 4.0, a) < 1, "a=%.4g, Kr/4=%.4g" % (a, K * r / 4.0))
		self.assertTrue(relative_error(y0, y2) < 1, "y2=%.4g, y0=%.4g" % (y2, y0))
		self.assertTrue(relative_error(r * (1 - y0 / K), mu) < 1, "mu=%.4g, r(1-y0/K)=%.4g" % (mu, r * (1 - y0 / K)))


	def test_find_max_growth_ci_logistic(self):
		y0 = 0.1
		K = 1.0
		r = 0.75
		df = curveball.models.randomize(t=12, y0=y0, K=K, r=r, nu=1, reps=REPS, noise_std=NOISE_STD, random_seed=RANDOM_SEED)
		model = curveball.baranyi_roberts_model.Logistic()
		model_fit = model.fit(df.OD, t=df.Time, y0=y0, K=K, r=r)
		  
		param_samples = curveball.models.bootstrap_params(df, model_fit, 100)
		_, _, a_est, _, _, mu_est = curveball.models.find_max_growth(model_fit)
		a_low, a_high, mu_low, mu_high = curveball.models.find_max_growth_ci(model_fit, param_samples)
		self.assertTrue(a_low < a_est < a_high, "a is {2}, a CI is ({0},{1})".format(a_low, a_high, a_est))
		self.assertTrue(mu_low < mu_est < mu_high, "mu is {2}, mu CI is ({0},{1})".format(mu_low, mu_high, mu_est))


	def test_find_max_growth_logistic_lag(self):
		y0 = 0.1
		K = 1.0
		r = 0.75
		lam = 3.0
		v = r
		q0 = 1.0 /(np.exp(lam * v) - 1)		
		df = curveball.models.randomize(t=12, y0=y0, K=K, r=r, nu=1, q0=q0, v=v, reps=REPS, noise_std=NOISE_STD, random_seed=RANDOM_SEED)
		model = curveball.baranyi_roberts_model.LogisticLag2()		
		model_fit = model.fit(df.OD, t=df.Time, y0=y0, K=K, r=r, q0=q0, v=v)        
			 
		t1,y1,a,t2,y2,mu = curveball.models.find_max_growth(model_fit)
		 
		self.assertTrue(K > y1 > old_div(K, 2), "y1=%.4g, K/2=%.4g" % (y1, old_div(K, 2)))
		self.assertTrue(K * r / 8 < a < K * r / 4, "a=%.4g, Kr/4=%.4g" % (a, K * r / 4))
		self.assertTrue(y0 < y2 < y1, "y0=%.4g, y1=%.4g, y2=%.4g," % (y0, y1, y2))
		self.assertTrue(0 < t2 < t1, "t1=%.4g, t2=%.4g," % (t1, t2))
		self.assertTrue(a < mu < r * (1 - old_div(y0,K)), "a = %.4g, mu=%.4g, r(1-y0/K)=%.4g" % (a, mu, r * (1-old_div(y0,K))))


	def test_find_max_growth_richards(self):
		y0 = 0.1
		K = 1
		r = 0.75
		nu = 0.5
		t, y = curveball.models.randomize(t=12, y0=y0, K=K, r=r, nu=nu, reps=REPS, noise_std=NOISE_STD, random_seed=RANDOM_SEED, as_df=False)
		model = curveball.baranyi_roberts_model.Richards()
		model_fit = model.fit(data=y, t=t, y0=y0, K=K, r=r, nu=nu)
			 
		t1,y1,a,t2,y2,mu = curveball.models.find_max_growth(model_fit)
		    
		exp_y1 = K * (nu + 1)**(-1.0 / nu)
		self.assertTrue(relative_error(exp_y1, y1) < 1, "y1=%.4g, K/(nu+1)**(1/nu)=%.4g" % (y1, exp_y1))
		exp_a = r * K * nu * (nu + 1)**(- 1 - 1.0 / nu)
		self.assertTrue(relative_error(exp_a, a) < 1, "a=%.4g, rKnu/(nu+1)**(1+1/nu)=%.4g" % (a, exp_a))
		self.assertTrue(relative_error(y0, y2) < 1, "y2=%.4g, y0=%.4g" % (y2, y0))
		self.assertTrue(relative_error(r * (1 - (y0 / K)**nu), mu) < 1, "mu=%.4g, r(1-(y0/K)**nu)=%.4g" % (mu, r * (1 - (y0 / K)**nu)))


class LRTestTestCase(TestCase):
	_multiprocess_can_split_ = True

	def setUp(self):
		if CI:
			self.folder = tempfile.mkdtemp()
		else:
			self.folder = '.'


	def tearDown(self):
		plt.close("all")
		if CI:
			shutil.rmtree(self.folder)


	def test_lrtest(self):
		rng = np.random.RandomState(RANDOM_SEED)
		a,b = 1,1
		a_init,b_init = 2,1
	
		alfa = 0.05
		noise = 0.03
		t = np.linspace(0,12)
		def f(t, a, b): 
			return b + np.exp(-a * t)
		y = f(t,a,b) + rng.normal(0, noise, len(t))
		model = Model(f)
		params = model.make_params(a=a_init, b=b_init)
	
		two_var_fit = model.fit(y, t=t, params=params)
	
		params['a'].set(vary=False)
		params['b'].set(vary=True)
		one_var_fit = model.fit(y, t=t, params=params)
	
		prefer_m1,pval,D,ddf = curveball.models.lrtest(one_var_fit, two_var_fit, alfa)
		self.assertTrue(prefer_m1)


	def test_has_lag_logistic(self):
		df = curveball.models.randomize(t=12, y0=0.1, K=1, r=0.75, nu=1, reps=REPS, noise_std=NOISE_STD, random_seed=RANDOM_SEED)

		models = curveball.models.fit_model(df, PLOT=False, PRINT=True)
		lag = curveball.models.has_lag(models)
		self.assertFalse(lag)


	def test_has_lag_richards(self):
		df = curveball.models.randomize(t=6, y0=0.1, K=1, r=0.75, nu=5.0, reps=REPS, noise_std=NOISE_STD, random_seed=RANDOM_SEED)

		models,fig,ax = curveball.models.fit_model(df, PLOT=True, PRINT=True)
		self.assertIsInstance(fig, matplotlib.figure.Figure)
		filename = sys._getframe().f_code.co_name + ".png"
		fig.savefig(filename)
		self.assertTrue(check_image(filename))   
		lag = curveball.models.has_lag(models)
		self.assertFalse(lag)


	def test_has_lag_baranyi_roberts(self):
		df = curveball.models.randomize(t=24, y0=0.1, K=1, r=0.75, nu=5.0, q0=0.1, v=0.1, reps=REPS, noise_std=NOISE_STD, random_seed=RANDOM_SEED)
		
		models,fig,ax = curveball.models.fit_model(df, PLOT=True, PRINT=True)
		self.assertIsInstance(fig, matplotlib.figure.Figure)
		filename = sys._getframe().f_code.co_name + ".png"
		fig.savefig(filename)
		self.assertTrue(check_image(filename))
		lag = curveball.models.has_lag(models)
		self.assertTrue(lag)


	def test_has_nu_logistic(self):
		df = curveball.models.randomize(t=12, y0=0.1, K=1, r=0.75, nu=1, reps=REPS, noise_std=NOISE_STD, random_seed=RANDOM_SEED)

		models = curveball.models.fit_model(df, PLOT=False, PRINT=True)
		result = curveball.models.has_nu(models)
		self.assertFalse(result)

	
	def test_has_nu_richards(self):
		df = curveball.models.randomize(t=6, y0=0.1, K=1, r=0.75, nu=5.0, reps=REPS, noise_std=NOISE_STD, random_seed=RANDOM_SEED)

		models, fig, ax = curveball.models.fit_model(df, PLOT=True, PRINT=True)
		self.assertIsInstance(fig, matplotlib.figure.Figure)
		filename = sys._getframe().f_code.co_name + ".png"
		fig.savefig(filename)
		self.assertTrue(check_image(filename))
		result = curveball.models.has_nu(models)
		self.assertTrue(result)

	## FIXME this is too hard, nu easiliy compensates for lag params

	# def test_has_nu_logistic_lag(self):
	# 	df = curveball.models.randomize(t=36, y0=0.1, K=1, r=0.75, nu=1, q0=0.1, v=0.1, reps=REPS, noise_std=NOISE_STD, random_seed=RANDOM_SEED)

	# 	models, fig, ax = curveball.models.fit_model(df, PLOT=True, PRINT=True)
	# 	self.assertIsInstance(fig, matplotlib.figure.Figure)
	# 	filename = sys._getframe().f_code.co_name + ".png"
	# 	fig.savefig(filename)
	# 	self.assertTrue(check_image(filename))
	# 	result = curveball.models.has_nu(models)
	# 	self.assertFalse(result)

	# def test_has_nu_baranyi_roberts_nu_1(self):
	# 	df = curveball.models.randomize(t=32, y0=0.1, K=1, r=0.75, nu=1.0, q0=0.1, v=0.1, reps=REPS, noise_std=NOISE_STD, random_seed=RANDOM_SEED)
		
	# 	models,fig,ax = curveball.models.fit_model(df, PLOT=True, PRINT=True)
	# 	self.assertIsInstance(fig, matplotlib.figure.Figure)
	# 	filename = sys._getframe().f_code.co_name + ".png"
	# 	fig.savefig(filename)
	# 	self.assertTrue(check_image(filename))
	# 	result = curveball.models.has_nu(models, PRINT=True)
	# 	self.assertFalse(result)


	# def test_has_nu_baranyi_roberts_nu_01(self):
	# 	df = curveball.models.randomize(t=120, y0=0.1, K=1, r=0.75, nu=0.1, q0=0.1, v=0.1, reps=REPS, noise_std=NOISE_STD, random_seed=RANDOM_SEED)		
		
	# 	models,fig,ax = curveball.models.fit_model(df, PLOT=True, PRINT=True)
	# 	self.assertIsInstance(fig, matplotlib.figure.Figure)
	# 	filename = sys._getframe().f_code.co_name + ".png"
	# 	fig.savefig(filename)
	# 	self.assertTrue(check_image(filename))       
	# 	result = curveball.models.has_nu(models, PRINT=True)
	# 	self.assertTrue(result)


	def test_has_nu_baranyi_roberts_nu_5(self):
		df = curveball.models.randomize(t=32, y0=0.1, K=1, r=0.75, nu=5.0, q0=0.1, v=0.1, reps=REPS, noise_std=NOISE_STD, random_seed=RANDOM_SEED)

		models,fig,ax = curveball.models.fit_model(df, PLOT=True, PRINT=True)
		self.assertIsInstance(fig, matplotlib.figure.Figure)
		filename = sys._getframe().f_code.co_name + ".png"
		fig.savefig(filename)
		self.assertTrue(check_image(filename))
		result = curveball.models.has_nu(models, PRINT=True)
		self.assertTrue(result)


class BenchmarkTestCase(TestCase):
	_multiprocess_can_split_ = True

	def setUp(self):
		if CI:
			self.folder = tempfile.mkdtemp()
		else:
			self.folder = '.'


	def tearDown(self):
		plt.close("all")
		if CI:
			shutil.rmtree(self.folder)


class OutliersTestCase(TestCase):
	_multiprocess_can_split_ = True


	def setUp(self):
		self.filename = os.path.join("data", "Tecan_210115.csv")
		self.df = pd.read_csv(self.filename)
		self.df = self.df[self.df.Strain == 'R']
		self.model_fit = curveball.models.fit_model(self.df, PLOT=False, PRINT=False)[0]
		if CI:
			self.folder = tempfile.mkdtemp()
		else:
			self.folder = '.'


	def tearDown(self):
		plt.close("all")
		if CI:
			shutil.rmtree(self.folder)


	def test_cooks_distance(self):
		D = curveball.models.cooks_distance(self.df, self.model_fit)
		self.assertEqual(set(D.keys()), set(self.df.Well), msg=D.keys())
		distances = np.array(list(D.values()))
		for d in distances:
			self.assertIsInstance(d, float)
		self.assertTrue( (distances > 0).all(), msg=D.values() )


	def test_find_outliers(self):		
		outliers,fig,ax = curveball.models.find_outliers(self.df, self.model_fit, PLOT=True)
		self.assertIsInstance(fig, matplotlib.figure.Figure)
		filename = sys._getframe().f_code.co_name + ".png"
		fig.savefig(filename)
		self.assertTrue(check_image(filename))       
		self.assertTrue(pd.Series(outliers).isin(self.df.Well).all())
		self.assertTrue(len(outliers) < len(self.df.Well.unique()))


	def test_find_all_outliers(self):        
		outliers,fig,ax = curveball.models.find_all_outliers(self.df, self.model_fit, PLOT=True)
		self.assertIsInstance(fig, matplotlib.figure.Figure)
		filename = sys._getframe().f_code.co_name + ".png"
		fig.savefig(filename)
		self.assertTrue(check_image(filename))       
		self.assertIsNotNone(outliers)
		self.assertTrue(len(outliers) > 0)
		for v in outliers: self.assertTrue(len(v) > 0)
		self.assertTrue(pd.Series(sum(outliers, [])).isin(self.df.Well).all())
		self.assertTrue(len(sum(outliers, [])) < len(self.df.Well.unique()))


class SamplingTestCase(TestCase):
	_multiprocess_can_split_ = True


	def test_sample_params(self):
		t, y = curveball.models.randomize(t=12, y0=0.1, K=1, r=0.75, nu=1.0, reps=1, noise_std=NOISE_STD, random_seed=RANDOM_SEED, as_df=False)
		model = curveball.baranyi_roberts_model.Logistic()
		params = model.guess(data=y, t=t)
		model_fit = model.fit(data=y, t=t, params=params)
		self.assertIsNotNone(model_fit.covar)
		sample_params = curveball.models.sample_params(model_fit, 100)
		self.assertIsNotNone(sample_params)
		self.assertEqual(sample_params.shape, (100, 3))


	def test_sample_params_with_params(self):
		t, y = curveball.models.randomize(t=12, y0=0.1, K=1, r=0.75, nu=1.0, reps=1, noise_std=NOISE_STD, random_seed=RANDOM_SEED, as_df=False)
		model = curveball.baranyi_roberts_model.Logistic()
		params = model.guess(data=y, t=t)
		model_fit = model.fit(data=y, t=t, params=params)
		self.assertIsNotNone(model_fit.covar)
		sample_params = curveball.models.sample_params(model_fit, 100, params={'K': 1.0})
		self.assertIsNotNone(sample_params)
		self.assertEqual(sample_params.shape, (100, 3))


	def test_sample_params_with_covar(self):
		t, y = curveball.models.randomize(t=12, y0=0.1, K=1, r=0.75, nu=1.0, reps=1, noise_std=NOISE_STD, random_seed=RANDOM_SEED, as_df=False)
		model = curveball.baranyi_roberts_model.Logistic()
		params = model.guess(data=y, t=t)
		model_fit = model.fit(data=y, t=t, params=params)
		self.assertIsNotNone(model_fit.covar)
		covar = model_fit.covar
		rng = np.random.RandomState(RANDOM_SEED)
		covar = rng.exponential(0.001, (3, 3))
		sample_params = curveball.models.sample_params(model_fit, 100, covar=covar)
		self.assertIsNotNone(sample_params)
		self.assertEqual(sample_params.shape[1], 3)
		self.assertTrue(95 <= sample_params.shape[0] <= 100)


	def test_bootstrap_params(self):
		plate = pd.read_csv('plate_templates/G-RG-R.csv')
		df = curveball.ioutils.read_tecan_xlsx('data/Tecan_280715.xlsx', plate=plate)
		model_result = curveball.models.fit_model(
			df, models=curveball.baranyi_roberts_model.BaranyiRoberts, 
			PLOT=False, PRINT=False)[0]
		sample_params = curveball.models.bootstrap_params(df, model_result, 10)
		self.assertIsInstance(sample_params, pd.DataFrame)
		self.assertEqual(sample_params.shape, (10, 6))


class GuessTestCase(TestCase):
	_multiprocess_can_split_ = True


	def test_guess_nu(self):
		df = curveball.models.randomize(t=32, y0=0.1, K=1, r=0.75, nu=1.0, reps=30, noise_std=NOISE_STD, random_seed=RANDOM_SEED)
		nu, fig, ax = curveball.baranyi_roberts_model.guess_nu(t=df.Time, N=df.OD, PLOT=True, PRINT=True)
		self.assertIsInstance(fig, matplotlib.figure.Figure)
		filename = sys._getframe().f_code.co_name + ".png"
		fig.savefig(filename)
		self.assertTrue(check_image(filename))               
		self.assertIsInstance(nu, float)
		self.assertTrue(nu > 0)


class WeightsTestCase(TestCase):
	_multiprocess_can_split_ = True


	def test_calc_weights(self):
		df = curveball.models.randomize(t=32, y0=0.1, K=1, r=0.75, nu=1.0, reps=30, noise_std=NOISE_STD, random_seed=RANDOM_SEED)
		weights, fig, ax = curveball.models.calc_weights(df, PLOT=True)
		self.assertIsInstance(fig, matplotlib.figure.Figure)
		filename = sys._getframe().f_code.co_name + ".png"
		fig.savefig(filename)
		self.assertTrue(check_image(filename))
		self.assertIsNotNone(weights)		
		self.assertIsInstance(weights, np.ndarray)
		self.assertTrue(len(weights) == df.shape[0])


class IssuesTestCase(TestCase):
	'''Tests that came up from bugs and other issues.

	See `Curveball issues <https://github.com/yoavram/curveball>`_.
	'''
	_multiprocess_can_split_ = True

	## I no longer care if covar is None; I can and should use bootstrap_params instead of sample_params
	# def test_covar_exists_issue27(self):
	# 	'''`Issue 27 <https://github.com/yoavram/curveball/issues/27>`_.
	# 	'''
	# 	plate = pd.read_csv('plate_templates/G-RG-R.csv')
	# 	df = curveball.ioutils.read_tecan_xlsx('data/Tecan_280715.xlsx', plate=plate)
	# 	self.assertTrue('R' in df.Strain.unique())
	# 	df = df[df.Strain == 'R']
	# 	models_R,fig,ax = curveball.models.fit_model(df[df.Time<=16], PLOT=True, PRINT=True)
	# 	filename = sys._getframe().f_code.co_name + ".png"
	# 	fig.savefig(filename)
	# 	self.assertIsNotNone(models_R[0].covar)

	## FIXME this is too hard, nu easiliy compensates for lag params

	# def test_has_nu_issue22(self):
	# 	'''`Issue 22 <https://github.com/yoavram/curveball/issues/22>`_.
	# 	'''
	# 	df = curveball.models.randomize(t=48, y0=0.1, K=1, r=0.75, nu=5.0, q0=0.1, v=0.1, reps=REPS, noise_std=NOISE_STD, random_seed=RANDOM_SEED)

	# 	models, fig, ax = curveball.models.fit_model(df, PLOT=True, PRINT=True)
	# 	self.assertIsInstance(fig, matplotlib.figure.Figure)
	# 	filename = sys._getframe().f_code.co_name + ".png"
	# 	fig.savefig(filename)
	# 	self.assertTrue(check_image(filename))  

	# 	self.assertTrue('nu' in models[0].best_values)
	# 	nu = models[0].best_values['nu']
	# 	self.assertTrue(1.0 < nu < 10.0, nu)
	# 	has = curveball.models.has_nu(models)
	# 	self.assertTrue(has)


	def test_Dfun_works(self):
		t, y = curveball.models.randomize(as_df=False)
		for model_class in curveball.models.get_models(curveball.baranyi_roberts_model):
			model = model_class()
			params = model.guess(data=y, t=t)
			dfun = curveball.models.make_Dfun(model, params)
			self.assertTrue(hasattr(dfun, '__call__'))
			result = model.fit(data=y, t=t, params=params, fit_kws={'Dfun': dfun, 'col_deriv':True})
			self.assertIsInstance(result, ModelResult)


	def test_is_model(self):
		for model in curveball.models.get_models(curveball.baranyi_roberts_model):
			self.assertTrue(curveball.models.is_model(model))


	def test_make_Dfun(self):
		model = curveball.baranyi_roberts_model.BaranyiRoberts()
		params = model.make_params(y0=0.1, K=1, r=1, nu=1, v=1, q0=1)
		Dfun = curveball.models.make_Dfun(model, params)
		self.assertIsInstance(Dfun, types.FunctionType)

		t, y = curveball.models.randomize(as_df=False)
		res = Dfun(params, y, None, t)
		self.assertIsInstance(res, np.ndarray)
		self.assertEqual(res.shape, (len(params), len(t)))


if __name__ == '__main__':
	main()
