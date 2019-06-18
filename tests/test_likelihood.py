#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of curveball.
# https://github.com/yoavram/curveball

# Licensed under the MIT license:
# http://www.opensource.org/licenses/MIT-license
# Copyright (c) 2015, Yoav Ram <yoav@yoavram.com>
from unittest import TestCase, main
import os
import types
import numpy as np
import matplotlib as mpl
import curveball
from curveball.likelihood import *

CI = os.environ.get('CI', 'false').lower() == 'true'
RANDOM_SEED = int(os.environ.get('RANDOM_SEED', 0))


class LoglikTestCase(TestCase):


	def setUp(self):
		self.params = dict(y0=0.1, K=1, r=1, nu=1, q0=np.inf, v=np.inf)
		self.df = curveball.models.randomize(reps=30, random_seed=RANDOM_SEED, as_df=True,
											 **self.params)
		self.t = self.df.Time.unique()
		self.y = self.df.groupby(by='Time').OD.mean()
		self.y_sig = self.df.groupby(by='Time').OD.std()				

	def test_loglik(self):
		ll = loglik(self.t, self.y, self.y_sig, 
					f=curveball.baranyi_roberts_model.baranyi_roberts_function,
					**self.params)
		self.assertIsInstance(ll, float)


	def test_ridge_regularization_lam0(self):
		penalty = ridge_regularization(0, **self.params)
		self.assertIsInstance(penalty, types.FunctionType)
		self.assertEqual(penalty(**self.params), 0)
		self.assertEqual(penalty(K=0.6), 0)


	def test_ridge_regularization_lam1(self):
		penalty = ridge_regularization(1, **self.params)
		self.assertIsInstance(penalty, types.FunctionType)
		self.assertEqual(penalty(**self.params), 0)
		self.assertTrue(penalty(K=0.6) > 0)


	def test_loglik_penalty(self):
		penalty = ridge_regularization(1, K=0.7)
		ll0 = loglik(self.t, self.y, self.y_sig, 
					f=curveball.baranyi_roberts_model.baranyi_roberts_function,
					**self.params)
		ll1 = loglik(self.t, self.y, self.y_sig, 
					f=curveball.baranyi_roberts_model.baranyi_roberts_function,
					penalty=penalty,
					**self.params)
		self.assertTrue(ll0 > ll1)


	def test_loglik_r_nu(self):
		L = loglik_r_nu(np.logspace(-2, 2, 10), np.logspace(-2, 2, 10), 
			self.df, 
			f=curveball.baranyi_roberts_model.baranyi_roberts_function, 
			**self.params)
		self.assertIsInstance(L, np.ndarray)
		self.assertTrue(L.ndim, 2)
		self.assertTrue(L.shape, (10, 10))
		self.assertIsInstance(L[0,0], float)


	def test_loglik_r_q0(self):
		L = loglik_r_q0(np.logspace(-2, 2, 10), np.logspace(-2, 2, 10), 
			self.df, 
			f=curveball.baranyi_roberts_model.baranyi_roberts_function, 
			**self.params)
		self.assertIsInstance(L, np.ndarray)
		self.assertTrue(L.ndim, 2)
		self.assertTrue(L.shape, (10, 10))
		self.assertIsInstance(L[0,0], float)


	def test_plot_loglik(self):
		L = loglik_r_nu(np.logspace(-2, 2, 10), np.logspace(-2, 2, 10), 
			self.df, 
			f=curveball.baranyi_roberts_model.baranyi_roberts_function, 
			**self.params)
		fig, ax = plot_loglik([L, L], np.logspace(-2, 2, 10), np.logspace(-2, 2, 10), 
					xlabel='r', ylabel='nu', columns=2, 
					fig_title='test', normalize=True,
                	ax_titles=['1', '2'], cmap='jet', colorbar=True, 
                	ax_width=4, ax_height=4)
		self.assertIsInstance(fig, mpl.figure.Figure)
		self.assertEqual(ax.size, 2)


	def test_plot_model_loglik(self):
		m = curveball.models.fit_model(self.df, PLOT=False, PRINT=False, 
									   models=curveball.baranyi_roberts_model.BaranyiRoberts)
		m = m[0]
		fig, ax = plot_model_loglik(m, self.df)
		self.assertIsInstance(fig, mpl.figure.Figure)