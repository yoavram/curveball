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


def randomize_data(func_ode, t=None, y0=0.1, r=0.75, K=1.0, nu=0.5, q0=0.1, v=0.1, reps=30, noise=0.03):
    if t is None:
        t = np.linspace(0, 12)
    y = odeint(func_ode, y0, t, args=(r, K, nu, q0, v))
    y.resize((len(t),))
    y = y.repeat(reps).reshape((len(t), reps)) + np.random.normal(0, noise, (len(t), reps))
    y[y < 0] = 0
    return pd.DataFrame({'OD': y.flatten(), 'Time': t.repeat(reps)})


class ModelsTestCase(TestCase):
    def setUp(self):
        pass


    def tearDown(self):
        pass


    def test_lrtest(self):
        a,b = 1,1
        a_init,b_init = 2,1
    
        alfa = 0.05
        noise = 0.03
        t = np.linspace(0,12)
        f = lambda t,a,b: b + np.exp(-a * t)
        y = f(t,a,b) + np.random.normal(0, noise, len(t))
        model = Model(f)
        params = model.make_params(a=a_init, b=b_init)
    
        two_var_fit = model.fit(y, t=t, params=params)
    
        params['a'].set(vary=False)
        params['b'].set(vary=True)
        one_var_fit = model.fit(y, t=t, params=params)
    
        prefer_m1,pval,D,ddf = curveball.models.lrtest(one_var_fit, two_var_fit, alfa)
        self.assertTrue(prefer_m1)


    def test_logistic(self):
        y0=0.1; r=0.75; K=1.0
        t = np.linspace(0,12)
        y_curve = curveball.models.logistic_function(t, y0, r, K)
        y_ode = odeint(logistic_ode, y0, t, args=(r, K, 0, 0, 0))
        y_ode.resize((len(t),))
        err = compare_curves(y_ode, y_curve)
        self.assertTrue(err < 1e-6)
    
    
    def test_richards(self):
        y0=0.1; r=0.75; K=1.0; nu=0.5
        t = np.linspace(0,12)
        y_curve = curveball.models.richards_function(t, y0, r, K, nu)
        y_ode = odeint(richards_ode, y0, t, args=(r, K, nu, 0, 0))
        y_ode.resize((len(t),))
        err = compare_curves(y_ode, y_curve)
        self.assertTrue(err < 1e-6)
    
    
    def test_baranyi_roberts(self):
        y0=0.1; r=0.75; K=1.0; nu=0.5; q0=0.1; v=0.1
        t = np.linspace(0,12)
        y_curve = curveball.models.baranyi_roberts_function(t, y0, r, K, nu, q0, v)
        y_ode = odeint(baranyi_roberts_ode, y0, t, args=(r, K, nu, q0, v))
        y_ode.resize((len(t),))
        err = compare_curves(y_ode, y_curve)
        self.assertTrue(err < 1e-6)   
      

    def test_fit_model_logistic(self):
        df = randomize_data(logistic_ode)
        models,fig,ax = curveball.models.fit_model(df, PLOT=True, PRINT=False)
        func_name = sys._getframe().f_code.co_name
        fig.savefig(func_name + ".png")
        self.assertIsNotNone(models)
        for mod in models:
            self.assertIsInstance(mod, ModelFit)
        self.assertEquals(models[0].model, curveball.models.logistic_model)
        self.assertEquals(models[0].nvarys, 3)


    def test_fit_model_logistic_single_rep(self):
        df = randomize_data(logistic_ode, reps=1)        
        models,fig,ax = curveball.models.fit_model(df, PLOT=True, PRINT=False)
        func_name = sys._getframe().f_code.co_name
        fig.savefig(func_name + ".png")
        self.assertIsNotNone(models)
        for mod in models:
            self.assertIsInstance(mod, ModelFit)
        self.assertEquals(models[0].model, curveball.models.logistic_model)
        self.assertEquals(models[0].nvarys, 3)


    def test_fit_model_richards(self):
        df = randomize_data(richards_ode)
        models,fig,ax = curveball.models.fit_model(df, PLOT=True, PRINT=False)
        func_name = sys._getframe().f_code.co_name
        fig.savefig(func_name + ".png")
        self.assertIsNotNone(models)
        for mod in models:
            self.assertIsInstance(mod, ModelFit)
        self.assertEquals(models[0].model, curveball.models.richards_model)
        self.assertEquals(models[0].nvarys, 4)


    def test_fit_model_logistic_lag(self):        
        df = randomize_data(baranyi_roberts_ode, t=np.linspace(0,36), nu=1.0)
        models,fig,ax = curveball.models.fit_model(df, PLOT=True, PRINT=False)
        func_name = sys._getframe().f_code.co_name
        fig.savefig(func_name + ".png")
        self.assertIsNotNone(models)
        for mod in models:
            self.assertIsInstance(mod, ModelFit)
        self.assertEquals(models[0].model, curveball.models.baranyi_roberts_model)
        self.assertEquals(models[0].nvarys, 5)


    def test_fit_model_baranyi_roberts(self):        
        df = randomize_data(baranyi_roberts_ode, t=np.linspace(0,36), nu=5.0)
        models,fig,ax = curveball.models.fit_model(df, PLOT=True, PRINT=False)
        func_name = sys._getframe().f_code.co_name
        fig.savefig(func_name + ".png")
        self.assertIsNotNone(models)
        for mod in models:
            self.assertIsInstance(mod, ModelFit)
        self.assertEquals(models[0].model, curveball.models.baranyi_roberts_model)
        self.assertEquals(models[0].nvarys, 6)


    def test_find_lag_logistic(self):
        y0=0.1; r=0.75; K=1.0
        t = np.linspace(0,12)
        df = randomize_data(logistic_ode, t=t, y0=y0, r=r, K=K, reps=1)
        model_fit = curveball.models.logistic_model.fit(df.OD, t=df.Time, y0=y0, K=K, r=r)        
        res = curveball.models.find_lag(model_fit, PLOT=True)
        self.assertIsNotNone(res)
        self.assertTrue(len(res) == 4)
        lam,fig,ax1,ax2 = res
        self.assertIsNotNone(lam)
        self.assertIsNotNone(fig)
        self.assertIsNotNone(ax1)
        self.assertIsNotNone(ax2)
        func_name = sys._getframe().f_code.co_name
        fig.savefig(func_name + ".png")
        self.assertTrue(lam < 1)

if __name__ == '__main__':
    main()
