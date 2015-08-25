#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of curveball.
# https://github.com/yoavram/curveball

# Licensed under the MIT license:
# http://www.opensource.org/licenses/MIT-license
# Copyright (c) 2015, Yoav Ram <yoavram+github@gmail.com>

from unittest import TestCase, main
import sys
import os
import curveball
from scipy.integrate import odeint
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lmfit import Model
from lmfit.model import ModelResult


CI = os.environ.get('CI', 'false').lower() == 'true'
RANDOM_SEED = int(os.environ.get('RANDOM_SEED', 0))


def logistic_ode(y, t, r, K, nu, q0, v):
    return r * y * (1 - y/K)


def richards_ode(y, t, r, K, nu, q0, v):
    return r * y * (1 - (y/K)**nu)


def baranyi_roberts_ode(y, t, r, K, nu, q0, v):
    alfa = q0 / (q0 + np.exp(-v * t))
    return alfa * r * y * (1 - (y/K)**nu)


def compare_curves(y1, y2):
    return (abs(y1 - y2) / y1).mean()


def relative_error(exp, obs):
        return abs(exp - obs) / exp


def randomize_data(func_ode, t=None, y0=0.1, r=0.75, K=1.0, nu=0.5, q0=0.1, v=0.1, reps=30, noise=0.02):
    rng = np.random.RandomState(RANDOM_SEED)
    if t is None:
        t = np.linspace(0, 12)
    y = odeint(func_ode, y0, t, args=(r, K, nu, q0, v))
    y.resize((len(t),))
    y = y.repeat(reps).reshape((len(t), reps)) + rng.normal(0, noise, (len(t), reps))
    y[y < 0] = 0
    return pd.DataFrame({'OD': y.flatten(), 'Time': t.repeat(reps)})


class FunctionsTestCase(TestCase):
    _multiprocess_can_split_ = True


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


class ModelSelectionTestCase(TestCase):
    _multiprocess_can_split_ = True

    def tearDown(self):
        plt.close("all")


    def test_fit_model_logistic(self):
        df = randomize_data(logistic_ode)
        if not CI:
            models,fig,ax = curveball.models.fit_model(df, PLOT=True, PRINT=False)
            func_name = sys._getframe().f_code.co_name
            fig.savefig(func_name + ".png")
        else:
            models = curveball.models.fit_model(df, PLOT=False, PRINT=False)
        self.assertIsNotNone(models)
        for mod in models:
            self.assertIsInstance(mod, ModelResult)
        self.assertEquals(models[0].model, curveball.models.logistic_model)
        self.assertEquals(models[0].nvarys, 3)


    def test_fit_model_logistic_single_rep(self):
        df = randomize_data(logistic_ode, reps=1)
        if not CI:       
            models,fig,ax = curveball.models.fit_model(df, PLOT=True, PRINT=False)
            func_name = sys._getframe().f_code.co_name
            fig.savefig(func_name + ".png")
        else:
            models = curveball.models.fit_model(df, PLOT=False, PRINT=False)        
        self.assertIsNotNone(models)
        for mod in models:
            self.assertIsInstance(mod, ModelResult)
        self.assertEquals(models[0].model, curveball.models.logistic_model)
        self.assertEquals(models[0].nvarys, 3)


    def test_fit_model_richards(self):
        df = randomize_data(richards_ode)
        if not CI:
            models,fig,ax = curveball.models.fit_model(df, PLOT=True, PRINT=False)
            func_name = sys._getframe().f_code.co_name
            fig.savefig(func_name + ".png")
        else:
            models = curveball.models.fit_model(df, PLOT=False, PRINT=False)
        self.assertIsNotNone(models)
        for mod in models:
            self.assertIsInstance(mod, ModelResult)
        self.assertEquals(models[0].model, curveball.models.richards_model)
        self.assertEquals(models[0].nvarys, 4)


    def test_fit_model_logistic_lag(self):        
        df = randomize_data(baranyi_roberts_ode, t=np.linspace(0,36), nu=1.0)
        if not CI:
            models,fig,ax = curveball.models.fit_model(df, PLOT=True, PRINT=False)
            func_name = sys._getframe().f_code.co_name
            fig.savefig(func_name + ".png")
        else:
            models = curveball.models.fit_model(df, PLOT=False, PRINT=False)
        self.assertIsNotNone(models)
        for mod in models:
            self.assertIsInstance(mod, ModelResult)
        self.assertEquals(models[0].model, curveball.models.baranyi_roberts_model)
        self.assertEquals(models[0].nvarys, 5)


    def test_fit_model_baranyi_roberts(self):        
        df = randomize_data(baranyi_roberts_ode, t=np.linspace(0,30), nu=2.5)
        if not CI:
            models,fig,ax = curveball.models.fit_model(df, PLOT=True, PRINT=False)
            func_name = sys._getframe().f_code.co_name
            fig.savefig(func_name + ".png")
        else:
            models = curveball.models.fit_model(df, PLOT=False, PRINT=False)
        self.assertIsNotNone(models)
        for mod in models:
            self.assertIsInstance(mod, ModelResult)
        self.assertEquals(models[0].model, curveball.models.baranyi_roberts_model)
        self.assertEquals(models[0].nvarys, 6)


class FindLagTestCase(TestCase):
    _multiprocess_can_split_ = True

    def tearDown(self):
        plt.close("all")


    def test_find_lag_logistic(self):
        y0=0.1; r=0.75; K=1.0
        t = np.linspace(0,12)
        df = randomize_data(logistic_ode, t=t, y0=y0, r=r, K=K, reps=1)
        model_fit = curveball.models.logistic_model.fit(df.OD, t=df.Time, y0=y0, K=K, r=r)        
        if not CI:
            lam,fig,ax1,ax2 = curveball.models.find_lag(model_fit, PLOT=True)
            func_name = sys._getframe().f_code.co_name
            fig.savefig(func_name + ".png")
        else:
            lam = curveball.models.find_lag(model_fit, PLOT=False)
        self.assertTrue(lam < 1, "Lambda is " + str(lam))


    def test_find_lag_richards(self):
        y0=0.1; r=0.75; K=1.0
        t = np.linspace(0,12)
        for nu in [0.5,1.0,2.0]:
            df = randomize_data(richards_ode, t=t, y0=y0, r=r, K=K, nu=nu, reps=1)
            model_fit = curveball.models.richards_model.fit(df.OD, t=df.Time, y0=y0, K=K, r=r, nu=nu)
            if not CI:
                lam,fig,ax1,ax2 = curveball.models.find_lag(model_fit, PLOT=True)
                func_name = sys._getframe().f_code.co_name
                fig.savefig(func_name + ".png")
            else:
                lam = curveball.models.find_lag(model_fit, PLOT=False)
            self.assertTrue(lam < 1, "Lambda is " + str(lam))


    def test_find_lag_baranyi_roberts(self):
        t = np.linspace(0,16)
        y0=0.1; r=0.75; K=1.0
        v=r;
        for nu in [0.5, 1.0, 2.0]: 
            for _lam in [2., 3., 4.]:
                q0 = 1/(np.exp(_lam * v) - 1)
                df = randomize_data(baranyi_roberts_ode, t=t, y0=y0, r=r, K=K, nu=nu, q0=q0, v=v, reps=1)
                model_fit = curveball.models.baranyi_roberts_model.fit(df.OD, t=df.Time, y0=y0, K=K, r=r, nu=nu, q0=q0, v=v)
                if not CI:
                    lam,fig,ax1,ax2 = curveball.models.find_lag(model_fit, PLOT=True)
                    func_name = sys._getframe().f_code.co_name + ".nu.%.1f.lam.%d" % (nu, lam)
                    fig.savefig(func_name + ".png")
                else:
                    lam = curveball.models.find_lag(model_fit, PLOT=False)
                self.assertTrue((_lam + 1) > lam > (_lam - 1), "Lambda is " + str(lam) + " but should be " + str(_lam))


class FindMaxGrowthTestCase(TestCase):
    _multiprocess_can_split_ = True

    def tearDown(self):
        plt.close("all")


    def test_find_max_growth_logistic(self):
        y0=0.1; r=0.75; K=1.0
        t = np.linspace(0,12)
        df = randomize_data(logistic_ode, t=t, y0=y0, r=r, K=K, reps=10)
        model_fit = curveball.models.logistic_model.fit(df.OD, t=df.Time, y0=y0, K=K, r=r)
        if not CI:       
            t1,y1,a,t2,y2,mu,fig,ax1,ax2 = curveball.models.find_max_growth(model_fit, PLOT=True)
            func_name = sys._getframe().f_code.co_name
            fig.savefig(func_name + ".png")
        else:
            t1,y1,a,t2,y2,mu = curveball.models.find_max_growth(model_fit, PLOT=False)
        self.assertTrue(relative_error(K / 2, y1) < 1, "y1=%.4g, K/2=%.4g" % (y1, K / 2))
        self.assertTrue(relative_error(K * r / 4, a) < 1, "a=%.4g, Kr/4=%.4g" % (a, K * r / 4))
        self.assertTrue(relative_error(y0, y2) < 1, "y2=%.4g, y0=%.4g" % (y2, y0))
        self.assertTrue(relative_error(r * (1 - y0/K), mu) < 1, "mu=%.4g, r(1-y0/K)=%.4g" % (mu, r * (1-y0/K)))


    def test_find_max_growth_logistic_lag(self):
        y0=0.1; r=0.75; K=1.0; nu=1.0
        v=r; lam=3.0
        q0 = 1/(np.exp(lam * v) - 1)
        t = np.linspace(0,12)
        df = randomize_data(baranyi_roberts_ode, t=t, y0=y0, r=r, K=K, nu=nu, q0=q0, v=v, reps=10)
        model_fit = curveball.models.baranyi_roberts_model.fit(df.OD, t=df.Time, y0=y0, K=K, r=r, nu=nu, q0=q0, v=v)        
        if not CI:       
            t1,y1,a,t2,y2,mu,fig,ax1,ax2 = curveball.models.find_max_growth(model_fit, PLOT=True)
            func_name = sys._getframe().f_code.co_name
            fig.savefig(func_name + ".png")
        else:
            t1,y1,a,t2,y2,mu = curveball.models.find_max_growth(model_fit, PLOT=False)
        self.assertTrue(K > y1 > K / 2, "y1=%.4g, K/2=%.4g" % (y1, K / 2))
        self.assertTrue(K * r / 8 < a < K * r / 4, "a=%.4g, Kr/4=%.4g" % (a, K * r / 4))
        self.assertTrue(y0 < y2 < y1, "y0=%.4g, y1=%.4g, y2=%.4g," % (y0, y1, y2))
        self.assertTrue(0 < t2 < t1, "t1=%.4g, t2=%.4g," % (t1, t2))
        self.assertTrue(a < mu < r * (1 - y0/K), "a = %.4g, mu=%.4g, r(1-y0/K)=%.4g" % (a, mu, r * (1-y0/K)))


    def test_find_max_growth_richards(self):
        y0=0.1; r=0.75; K=1.0; nu=0.5
        t = np.linspace(0,12)
        df = randomize_data(richards_ode, t=t, y0=y0, r=r, K=K, nu=nu, reps=10)
        model_fit = curveball.models.richards_model.fit(df.OD, t=df.Time, y0=y0, K=K, r=r, nu=nu)        
        if not CI:       
            t1,y1,a,t2,y2,mu,fig,ax1,ax2 = curveball.models.find_max_growth(model_fit, PLOT=True)
            func_name = sys._getframe().f_code.co_name
            fig.savefig(func_name + ".png")
        else:
            t1,y1,a,t2,y2,mu = curveball.models.find_max_growth(model_fit, PLOT=False)
        exp_y1 = K * (nu + 1)**(-1/nu)
        self.assertTrue(relative_error(exp_y1, y1) < 1, "y1=%.4g, K/(nu+1)**(1/nu)=%.4g" % (y1, exp_y1))
        exp_a = r * K * nu * (nu + 1)**(- 1 - 1/nu)
        self.assertTrue(relative_error(exp_a, a) < 1, "a=%.4g, rKnu/(nu+1)**(1+1/nu)=%.4g" % (a, exp_a))
        self.assertTrue(relative_error(y0, y2) < 1, "y2=%.4g, y0=%.4g" % (y2, y0))
        self.assertTrue(relative_error(r * (1 - (y0/K)**nu), mu) < 1, "mu=%.4g, r(1-(y0/K)**nu)=%.4g" % (mu, r * (1 - (y0/K)**nu)))


class LRTestTestCase(TestCase):
    _multiprocess_can_split_ = True

    def tearDown(self):
        plt.close("all")


    def test_lrtest(self):
        rng = np.random.RandomState(RANDOM_SEED)
        a,b = 1,1
        a_init,b_init = 2,1
    
        alfa = 0.05
        noise = 0.03
        t = np.linspace(0,12)
        f = lambda t,a,b: b + np.exp(-a * t)
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
        df = randomize_data(logistic_ode)
        models = curveball.models.fit_model(df, PLOT=False, PRINT=False)
        lag = curveball.models.has_lag(models)
        self.assertFalse(lag)


    def test_has_lag_richards(self):
        df = randomize_data(richards_ode)
        models = curveball.models.fit_model(df, PLOT=False, PRINT=False)
        lag = curveball.models.has_lag(models)
        self.assertFalse(lag)


    def test_has_lag_logistic_lag(self):
        df = randomize_data(baranyi_roberts_ode, t=np.linspace(0,36), nu=1.0)
        models = curveball.models.fit_model(df, PLOT=False, PRINT=False)
        lag = curveball.models.has_lag(models)
        self.assertTrue(lag)


    def test_has_lag_baranyi_roberts(self):
        df = randomize_data(baranyi_roberts_ode, t=np.linspace(0,32))
        models = curveball.models.fit_model(df, PLOT=False, PRINT=False)
        lag = curveball.models.has_lag(models)
        self.assertTrue(lag)


    def test_has_nu_logistic(self):
        df = randomize_data(logistic_ode)
        models = curveball.models.fit_model(df, PLOT=False, PRINT=False)
        result = curveball.models.has_nu(models)
        self.assertFalse(result)


    def test_has_nu_richards(self):
        df = randomize_data(richards_ode)
        models = curveball.models.fit_model(df, PLOT=False, PRINT=False)
        result = curveball.models.has_nu(models)
        self.assertTrue(result)


    def test_has_nu_logistic_lag(self):
        df = randomize_data(baranyi_roberts_ode, t=np.linspace(0,36), nu=1.0)
        models = curveball.models.fit_model(df, PLOT=False, PRINT=False)
        result = curveball.models.has_nu(models)
        self.assertFalse(result)


    def test_has_nu_baranyi_roberts_nu_01(self):
        df = randomize_data(baranyi_roberts_ode, t=np.linspace(0,120), nu=0.1)
        if not CI:
            models,fig,ax = curveball.models.fit_model(df, PLOT=True, PRINT=False)
            func_name = sys._getframe().f_code.co_name
            fig.savefig(func_name + ".png")
        else:
            models = curveball.models.fit_model(df, PLOT=False, PRINT=False)
        result = curveball.models.has_nu(models, PRINT=True)
        self.assertTrue(result)


    def test_has_nu_baranyi_roberts_nu_1(self):
        df = randomize_data(baranyi_roberts_ode, t=np.linspace(0,32), nu=1.0)
        if not CI:
            models,fig,ax = curveball.models.fit_model(df, PLOT=True, PRINT=False)
            func_name = sys._getframe().f_code.co_name
            fig.savefig(func_name + ".png")
        else:
            models = curveball.models.fit_model(df, PLOT=False, PRINT=False)
        result = curveball.models.has_nu(models, PRINT=True)
        self.assertFalse(result)


    def test_has_nu_baranyi_roberts_nu_5(self):
        df = randomize_data(baranyi_roberts_ode, t=np.linspace(0,32), nu=5.0)
        if not CI:
            models,fig,ax = curveball.models.fit_model(df, PLOT=True, PRINT=False)
            func_name = sys._getframe().f_code.co_name
            fig.savefig(func_name + ".png")
        else:
            models = curveball.models.fit_model(df, PLOT=False, PRINT=False)
        result = curveball.models.has_nu(models, PRINT=True)
        self.assertTrue(result)


class BenchmarkTestCase(TestCase):
    _multiprocess_can_split_ = True

    def tearDown(self):
        plt.close("all")


    def test_benchmark_success(self):
        y0=0.1; r=0.75; K=1.0; nu=5.0
        v=r; lam=3.0
        q0 = 1/(np.exp(lam * v) - 1)
        t = np.linspace(0,12)
        df = randomize_data(baranyi_roberts_ode, t=t, r=r, y0=y0, K=K, nu=nu, q0=q0, v=v, reps=1)
        params = curveball.models.baranyi_roberts_model.make_params(r=0.1, y0=df.OD.min(), K=df.OD.max(), nu=1.0, q0=1.0, v=1.0)
        model_fit = curveball.models.baranyi_roberts_model.fit(data=df.OD, t=df.Time, params=params)
        if not CI:
            success,fig,ax = curveball.models.benchmark(model_fit, PLOT=True)
            func_name = sys._getframe().f_code.co_name
            fig.savefig(func_name + ".png")                
        else:
            success = curveball.models.benchmark(model_fit, PLOT=False)
        self.assertEquals(success, True)
        

    def test_benchmark_failure(self):
        y0=0.1; r=0.75; K=1.0; nu=5.0
        v=r; lam=3.0
        q0 = 1/(np.exp(lam * v) - 1)
        t = np.linspace(0,12)
        df = randomize_data(baranyi_roberts_ode, t=t, r=r, y0=y0, K=K, nu=nu, q0=q0, v=v, reps=1)
        params = curveball.models.logistic_model.make_params(r=0.1, y0=df.OD.min(), K=df.OD.max())
        model_fit = curveball.models.logistic_model.fit(data=df.OD, t=df.Time, params=params)
        if not CI:
            success,fig,ax = curveball.models.benchmark(model_fit, PLOT=True)
            func_name = sys._getframe().f_code.co_name
            fig.savefig(func_name + ".png")                
        else:
            success = curveball.models.benchmark(model_fit, PLOT=False)     
        self.assertEquals(success, True)
        

class OutliersTestCase(TestCase):
    _multiprocess_can_split_ = True

    def setUp(self):
        self.filename = os.path.join("data", "yoavram", "Tecan_210115.csv")
        self.df = pd.read_csv(self.filename)
        self.df = self.df[self.df.Strain == 'R']
        self.model_fit = curveball.models.fit_model(self.df, PLOT=False, PRINT=False)[0]


    def tearDown(self):
        plt.close("all")        


    def test_cooks_distance(self):
        D = curveball.models.cooks_distance(self.df, self.model_fit)
        self.assertEquals(set(D.keys()), set(self.df.Well))
        self.assertTrue( (np.array(D.values()) < 14).all() )
        self.assertTrue( (np.array(D.values()) > 11).all() )


    def test_find_outliers(self):
        if not CI:
            outliers,fig,ax = curveball.models.find_outliers(self.df, self.model_fit, PLOT=True)
            func_name = sys._getframe().f_code.co_name
            fig.savefig(func_name + ".png")
        else:
            outliers = curveball.models.find_outliers(self.df, self.model_fit, PLOT=False)
        self.assertTrue(pd.Series(outliers).isin(self.df.Well).all())
        self.assertTrue(len(outliers) < len(self.df.Well.unique()))


    def test_find_all_outliers(self):
        if not CI:
            outliers,fig,ax = curveball.models.find_all_outliers(self.df, self.model_fit, PLOT=True)
            func_name = sys._getframe().f_code.co_name
            fig.savefig(func_name + ".png")
        else:
            outliers = curveball.models.find_all_outliers(self.df, self.model_fit, PLOT=False)
        self.assertIsNotNone(outliers)
        self.assertTrue(len(outliers) > 0)
        for v in outliers: self.assertTrue(len(v) > 0)
        self.assertTrue(pd.Series(sum(outliers, [])).isin(self.df.Well).all())
        self.assertTrue(len(sum(outliers, [])) < len(self.df.Well.unique()))


class SamplingTestCase(TestCase):
    _multiprocess_can_split_ = True


    def test_sample_params(self):
        df = randomize_data(logistic_ode)
        params = curveball.models.logistic_model.make_params(r=0.1, y0=df.OD.min(), K=df.OD.max())
        model_fit = curveball.models.logistic_model.fit(data=df.OD, t=df.Time, params=params)
        sample_params = curveball.models.sample_params(df, model_fit, 100)
        self.assertIsNotNone(sample_params)
        self.assertEquals(sample_params.shape, (100, 3))

        

if __name__ == '__main__':
    main()
