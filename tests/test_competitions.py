#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of curveball.
# https://github.com/yoavram/curveball

# Licensed under the MIT license:
# http://www.opensource.org/licenses/MIT-license
# Copyright (c) 2015, Yoav Ram <yoav@yoavram.com>
from unittest import TestCase, main
import sys
import os
import numpy as np
import curveball
import matplotlib


CI = os.environ.get('CI', 'false').lower() == 'true'
RANDOM_SEED = int(os.environ.get('RANDOM_SEED', 0))


def logistic(t, y0, r, K):
	return K / (1 + (K/y0 - 1) * np.exp(-r * t))


class CompetitionTestCase(TestCase):

	def setUp(self):
		t = np.linspace(0, 24, 20)
		y0 = 0.1
		r1, r2 = 0.3, 0.4
		K = 1.0

		y1 = logistic(t, y0, r1, K)
		y2 = logistic(t, y0, r2, K)

		params = curveball.models.logistic_model.make_params(y0=y0, r=r1, K=K)
		self.m1 = curveball.models.logistic_model.fit(data=y1, t=t, params=params)
		params = curveball.models.logistic_model.make_params(y0=y0, r=r2, K=K)
		self.m2 = curveball.models.logistic_model.fit(data=y2, t=t, params=params)


	def tearDown(self):
		pass


	def test_compete(self):
		t, y = curveball.competitions.compete(self.m1, self.m2, PLOT=False)
		self.assertEquals(t.shape[0], y.shape[0])
		self.assertEquals(y.shape[1], 2)
		self.assertTrue(y[-1,0] > y[0,0])
		self.assertTrue(y[-1,1] > y[0,1])


	def test_compete_plot(self):
		t, y, fig, ax = curveball.competitions.compete(self.m1, self.m2, colors=['r','b'], PLOT=True)
		self.assertEquals(t.shape[0], y.shape[0])
		self.assertEquals(y.shape[1], 2)
		self.assertTrue(y[-1,0] > y[0,0])
		self.assertTrue(y[-1,1] > y[0,1])
		self.assertIsInstance(fig, matplotlib.figure.Figure)
		self.assertIsInstance(ax, matplotlib.axes.Axes)
		if not CI:            
			func_name = sys._getframe().f_code.co_name
			fig.savefig(func_name + ".png")


	def test_compete_resample(self):
		nsamples = 100
		t, y = curveball.competitions.compete(self.m1, self.m2, nsamples=nsamples, PLOT=False)
		self.assertEquals(t.shape[0], y.shape[0])
		self.assertEquals(y.shape[1], 2)
		self.assertEquals(y.shape[2], nsamples)
		self.assertTrue((y[-1,0,:] > y[0,0,:]).all())
		self.assertTrue((y[-1,1,:] > y[0,1,:]).all())


	def test_compete_plot_resample(self):
		nsamples = 100
		self.m1.covar = np.ones((3,3)) * 0.01
		self.m2.covar = np.ones((3,3)) * 0.01

		t, y, fig, ax = curveball.competitions.compete(self.m1, self.m2, nsamples=nsamples, PLOT=True)
		self.assertEquals(t.shape[0], y.shape[0])
		self.assertEquals(y.shape[1], 2)
		self.assertEquals(y.shape[2], nsamples)
		# self.assertTrue((y[-1,0,:] > y[0,0,:]).all())
		# self.assertTrue((y[-1,1,:] > y[0,1,:]).all())
		self.assertIsInstance(fig, matplotlib.figure.Figure)
		self.assertIsInstance(ax, matplotlib.axes.Axes)
		if not CI:            
			func_name = sys._getframe().f_code.co_name
			fig.savefig(func_name + ".png")


class FitnessTestCase(TestCase):

	def test_fitness_LTEE(self):
		t = np.linspace(0, 24)
		y1 = logistic(t, 0.1, 0.3, 1)
		y2 = logistic(t, 0.1, 0.4, 1)
		y = np.array((y1, y2)).T
		shape = y.shape
		w = curveball.competitions.fitness_LTEE(y)
		self.assertEquals(y.shape, shape)
		self.assertTrue(1 < w < 1.1)


	def test_fitness_LTEE_ci(self):
		nsamples = 100
		rand = lambda: np.random.normal(0, 0.01)
		t = np.linspace(0, 24)
		y = np.zeros((len(t), 2, nsamples))
		for i in range(nsamples):
			y[:,0,i] = logistic(t, 0.1, 0.3 + rand(), 1 + rand())
			y[:,1,i] = logistic(t, 0.1, 0.4 + rand(), 1 + rand())
		shape = y.shape
		w, low, high = curveball.competitions.fitness_LTEE(y, ci=0.95)
		self.assertEquals(y.shape, shape)
		self.assertTrue(1 < w < 1.1)
		self.assertTrue(low < w < high)


if __name__ == '__main__':
	main()
