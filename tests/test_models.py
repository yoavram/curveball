#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of curveball.
# https://github.com/yoavram/curveball

# Licensed under the MIT license:
# http://www.opensource.org/licenses/MIT-license
# Copyright (c) 2015, Yoav Ram <yoavram+github@gmail.com>

from unittest import TestCase, main
import sys

import curveball
from scipy.integrate import odeint
import numpy as np
import pandas as pd
from lmfit import Model
from lmfit.model import ModelFit

def logistic_ode(y, t, r, K, nu, q0, v):
    return r * y * (1 - y/K)


def richards_ode(y, t, r, K, nu, q0, v):
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
        self.noise = 0.03 # "measurement" standard deviation
        self.reps = 30 # replicates

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
    #     y_ode = odeint(logistic_ode, self.y0, self.t, args=(self.r, self.K, self.nu, self.q0, self.v))
    #     y_ode.resize((len(self.t),))
    #     err = compare_curves(y_ode, y_curve)
    #     self.assertTrue(err < 1e-6)
    #
    #
    # def test_richards(self):
    #     y_curve = curveball.models.richards_function(self.t, self.y0, self.r, self.K, self.nu)
    #     y_ode = odeint(richards_ode, self.y0, self.t, args=(self.r, self.K, self.nu, self.q0, self.v))
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


    def _randomize_data(self, func_ode):
        y = odeint(func_ode, self.y0, self.t, args=(self.r, self.K, self.nu, self.q0, self.v))
        y.resize((len(self.t),))
        y = y.repeat(self.reps).reshape((len(self.t), self.reps)) + np.random.normal(0, self.noise, (len(self.t), self.reps))
        y[y < 0] = 0
        return pd.DataFrame({'OD': y.flatten(), 'Time': self.t.repeat(self.reps)})


    def test_fit_model_logistic(self):
        df = self._randomize_data(logistic_ode)
        models,fig,ax = curveball.models.fit_model(df, PLOT=True, PRINT=False)
        func_name = sys._getframe().f_code.co_name
        fig.savefig(func_name + ".png")
        self.assertIsNotNone(models)
        self.assertEquals(len(models), 4)
        for mod in models:
            self.assertIsInstance(mod, ModelFit)
        self.assertEquals(models[0].model, curveball.models.logistic_model)
        self.assertEquals(models[0].nvarys, 3)


    def test_fit_model_logistic_single_rep(self):
        _reps = self.reps
        self.reps = 1
        df = self._randomize_data(logistic_ode)
        self.reps = _reps
        models,fig,ax = curveball.models.fit_model(df, PLOT=True, PRINT=False)
        func_name = sys._getframe().f_code.co_name
        fig.savefig(func_name + ".png")
        self.assertIsNotNone(models)
        self.assertEquals(len(models), 4)
        for mod in models:
            self.assertIsInstance(mod, ModelFit)
        self.assertEquals(models[0].model, curveball.models.logistic_model)
        self.assertEquals(models[0].nvarys, 3)


    def test_fit_model_richards(self):
        df = self._randomize_data(richards_ode)
        models,fig,ax = curveball.models.fit_model(df, PLOT=True, PRINT=False)
        func_name = sys._getframe().f_code.co_name
        fig.savefig(func_name + ".png")
        self.assertIsNotNone(models)
        self.assertEquals(len(models), 4)
        for mod in models:
            self.assertIsInstance(mod, ModelFit)
        self.assertEquals(models[0].model, curveball.models.richards_model)
        self.assertEquals(models[0].nvarys, 4)


    def test_fit_model_logistic_lag(self):
        self.nu, _nu = 1.0, self.nu
        self.t, _t = np.linspace(0,36), self.t
        df = self._randomize_data(baranyi_roberts_ode)
        self.nu = _nu
        self.t = _t
        models,fig,ax = curveball.models.fit_model(df, PLOT=True, PRINT=False)
        func_name = sys._getframe().f_code.co_name
        fig.savefig(func_name + ".png")
        self.assertIsNotNone(models)
        self.assertEquals(len(models), 4)
        for mod in models:
            self.assertIsInstance(mod, ModelFit)
        self.assertEquals(models[0].model, curveball.models.baranyi_roberts_model)
        self.assertEquals(models[0].nvarys, 5)


if __name__ == '__main__':
    main()
