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
import pandas as pd
from lmfit import Model
from lmfit.model import ModelFit

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
        self.t = np.linspace(0, 12, 100)
        self.y0 = 0.1
        self.r = 0.75
        self.K = 1.
        self.nu = 0.5
        self.q0 = 0.1
        self.v = 0.1
        self.noise = 0.02

    def tearDown(self):
        pass


    # def test_lrtest(self):
    #     a,b = 1,1
    #     a_init,b_init = 2,1
    #
    #     alfa=0.05
    #
    #     f = lambda t,a,b: b + np.exp(-a * t)
    #     y = f(self.t,a,b) + np.random.normal(0, self.noise, len(self.t))
    #     model = Model(f)
    #     params = model.make_params(a=a_init, b=b_init)
    #
    #     two_var_fit = model.fit(y, t=self.t, params=params)
    #
    #     params['a'].set(vary=False)
    #     params['b'].set(vary=True)
    #     one_var_fit = model.fit(y, t=self.t, params=params)
    #
    #     prefer_m1,pval,D,ddf = curveball.models.lrtest(one_var_fit, two_var_fit, alfa)
    #     self.assertTrue(prefer_m1)

    # def test_logistic(self):
    #     y_curve = curveball.models.logistic_function(self.t, self.y0, self.r, self.K)
    #     y_ode = odeint(logistic_ode, self.y0, self.t, args=(self.r, self.K))
    #     y_ode.resize((len(self.t),))
    #     err = compare_curves(y_ode, y_curve)
    #     self.assertTrue(err < 1e-6)
    #
    #
    # def test_richards(self):
    #     y_curve = curveball.models.richards_function(self.t, self.y0, self.r, self.K, self.nu)
    #     y_ode = odeint(richards_ode, self.y0, self.t, args=(self.r, self.K, self.nu))
    #     y_ode.resize((len(self.t),))
    #     err = compare_curves(y_ode, y_curve)
    #     self.assertTrue(err < 1e-6)
    #
    #
    # def test_baranyi_roberts(self):
    #     y_curve = curveball.models.baranyi_roberts_function(self.t, self.y0, self.r, self.K, self.nu, self.q0, self.v)
    #     y_ode = odeint(baranyi_roberts_ode, self.y0, self.t, args=(self.r, self.K, self.nu, self.q0, self.v))
    #     y_ode.resize((len(self.t),))
    #     err = compare_curves(y_ode, y_curve)
    #     self.assertTrue(err < 1e-6)


    def test_fit_model_logistic(self):
        y = odeint(logistic_ode, self.y0, self.t, args=(self.r, self.K))
        y.resize((len(self.t),))
        y += np.random.normal(0, self.noise, len(self.t))
        df = pd.DataFrame({'OD':y, 'Time': self.t})
        models,fig,ax = curveball.models.fit_model(df, PLOT=True, PRINT=False)
        fig.savefig("test_fit_model_logistic.png")
        self.assertIsNotNone(models)
        self.assertEquals(len(models), 4)
        for mod in models:
            self.assertIsInstance(mod, ModelFit)
        self.assertEquals(models[0].nvarys, 3, msg = "The simplest model should win")
        self.assertEquals(models[0].model, curveball.models.logistic_model, msg = "The logistic model should win")


if __name__ == '__main__':
    main()
