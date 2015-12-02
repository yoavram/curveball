#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of curveball.
# https://github.com/yoavram/curveball

# Licensed under the MIT license:
# http://www.opensource.org/licenses/MIT-license
# Copyright (c) 2015, Yoav Ram <yoav@yoavram.com>

# See "Writing benchmarks" in the asv docs for more information.
# http://asv.readthedocs.org/en/latest/writing_benchmarks.html#writing-benchmarks
from __future__ import division
from builtins import str
import os
from warnings import warn
import numpy as np
import curveball


CI = os.environ.get('CI', 'false').lower() == 'true'
RANDOM_SEED = int(os.environ.get('RANDOM_SEED', 0))
REPS = 30
NOISE_STD = 0.04


def relative_error(exp, obs):
	return np.abs(exp - obs) / exp


class GuessNuSuite:
	params = ([0.5, 1.0, 5.0], [0, 0.04], [0.1, 0.5, 0.66])
	param_names = ['nu', 'noise', 'frac']


	def setup(self, nu, noise, frac):
		self.df = curveball.models.randomize(t=6, y0=0.1, K=1, r=0.75, nu=nu, reps=REPS, noise_std=noise, random_seed=RANDOM_SEED)		


	def track_guess_nu(self, nu, noise, frac):				
		return curveball.baranyi_roberts_model.guess_nu(t=self.df.Time, N=self.df.OD, frac=frac, PLOT=False, PRINT=False)


class GuessRSuite:
	params = ([0.01, 0.1, 1.0], [0.5, 1.0, 5.0], [0, 0.04])
	param_names = ['r', 'nu', 'noise']


	def setup(self, r, nu, noise):
		self.df = curveball.models.randomize(t=6, y0=0.1, K=1, r=r, nu=nu, reps=REPS, noise_std=noise, random_seed=RANDOM_SEED)		


	def track_guess_r(self, r, nu, noise):				
		return curveball.baranyi_roberts_model.guess_r(t=self.df.Time, N=self.df.OD, nu=nu)

	def track_guess_nu(self, r, nu, noise):				
		return curveball.baranyi_roberts_model.guess_nu(t=self.df.Time, N=self.df.OD)
