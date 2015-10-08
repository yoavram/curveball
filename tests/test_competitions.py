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


def logistic(t, y0, r, K):
	return K / (1 + (K/y0 - 1) * np.exp(-r * t))


class TestCase(TestCase):
	def setUp(self):
		pass


	def tearDown(self):
		pass


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
