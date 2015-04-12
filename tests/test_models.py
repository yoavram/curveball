#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of curveball.
# https://github.com/yoavram/curveball

# Licensed under the MIT license:
# http://www.opensource.org/licenses/MIT-license
# Copyright (c) 2015, Yoav Ram <yoavram+github@gmail.com>

from unittest import TestCase, main

import curveball
from scipy.integrate import odeint
import numpy as np


def logistic_ode(y, t, r, K):
    return r * y * (1 - y/K)


def richards_ode(y, t, r, K, nu):
    return r * y * (1 - (y/K)**nu)


def baranyi_roberts_ode(y, t, r, K, nu, q0, v):
    alfa = q0 / (q0 + np.exp(-v * t))
    return alfa * r * y * (1 - (y/K)**nu)


def compare_curves(y1, y2):
    return (abs(y1 - y2) / y1).mean()


class ModelsTestCase(TestCase):
    def setUp(self):
        self.t = np.linspace(0, 1)
        self.y0 = 0.1
        self.r = 7.
        self.K = 1.
        self.nu = 0.5
        self.q0 = 0.1
        self.v = 0.1

    def tearDown(self):
        pass

    def test_logistic(self):
        y_curve = curveball.models.logistic_function(self.t, self.y0, self.r, self.K)
        y_ode = odeint(logistic_ode, self.y0, self.t, args=(self.r, self.K))
        y_ode.resize((len(self.t),))
        err = compare_curves(y_ode, y_curve)
        self.assertTrue(err < 1e-6)


    def test_richards(self):
        y_curve = curveball.models.richards_function(self.t, self.y0, self.r, self.K, self.nu)
        y_ode = odeint(richards_ode, self.y0, self.t, args=(self.r, self.K, self.nu))
        y_ode.resize((len(self.t),))
        err = compare_curves(y_ode, y_curve)
        self.assertTrue(err < 1e-6)


    def test_baranyi_roberts(self):
        y_curve = curveball.models.baranyi_roberts_function(self.t, self.y0, self.r, self.K, self.nu, self.q0, self.v)
        y_ode = odeint(baranyi_roberts_ode, self.y0, self.t, args=(self.r, self.K, self.nu, self.q0, self.v))
        y_ode.resize((len(self.t),))
        err = compare_curves(y_ode, y_curve)
        self.assertTrue(err < 1e-6)


if __name__ == '__main__':
    main()
