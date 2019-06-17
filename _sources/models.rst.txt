Growth Models
=============

The :py:mod:`.models` module contains functions for fitting and selecting growth models to growth curve data.

This module contains:

- :py:func:`.fit_model` is the main function of the model; it fits growth models to growth curve data.
- **Growth functions** (``*_function``) are Python function in which the first argumet is time and the rest of the argument are model parameters. Time can be given an a :py:class:`numpy.ndarray`.
- **Growth models** (``*_model``) are :py:class:`lmfit.model.Model` objects from the `lmfit <http://lmfit.github.io/lmfit-py/>`_ package. These objects wrap the growth functions and are used to fit models to data with parameter constraints and some useful statistics and models selection methods.
- **Parameter guessing functions** (``guess_*``) are function that use the shape of a growth curve to guess growth parameters **without** fitting a model to the data, but rather through analytic properties of the growth functions.
- **Hypothesis testing** (``has_*``) are functions that perform a statistical test to decide if a property of one model can be significantly observed in the data.
- **Outlier functions** (``find_outliers*``) are functions for identification of outlier growth curve in high-throughput data sets.
- **Indirect trait calculation** (``find_*``) are functions for calculating growth parameters that are indirectly represented by the models.
- and some other utilities.

Models
------

Curveball uses the logistic model and its derivaties, the Richards model and the Baranyi-Roberts model:

Logistic model
^^^^^^^^^^^^^^

Also known as the `Valhulst model <https://en.wikipedia.org/wiki/Logistic_function#In_ecology:_modeling_population_growth>`_,
the logistic model includes three parameters and is the simplest and most commonly used population growth model.

The logistic model is defined by an ordinary differential equation (ODE) which also has a closed form solution:

.. math::

	\frac{dN}{dt} = r N \Big(1 - \frac{N}{K}\Big) \Rightarrow

    N(t) = \frac{K}{1 - \Big(1 - \frac{K}{N_0} \Big)e^{-r t}}


- N: population size
- :math:`N_0`: initial population size
- r: initial per capita growth rate 
- K: maximum population size

Richards model
^^^^^^^^^^^^^^

Also known as the `Generalised logistic model <http://en.wikipedia.org/wiki/Generalised_logistic_function>`_, 
Richards model [Richards1959]_ (or in its discrete time version, the :math:`\theta`-logistic model [Gilpin1973]_) 
extends the logistic model by including the curvature parameter :math:`\nu`:

.. math::

        \frac{dN}{dt} = r N \Big( 1 - \Big(\frac{N}{K}\Big)^{\nu} \Big) \Rightarrow

        N(t) = \frac{K}{\Big[1 - \Big(1 - \Big(\frac{K}{N_0}\Big)^{\nu}\Big) e^{-r \nu t}\Big]^{1/\nu}}

- :math:`y_0`: initial population size
- r: initial per capita growth rate
- K: maximum population size    
- :math:`\nu`: curvature of the logsitic term

The logistic model is then a special case of the Richards model for :math:`\nu=1`, 
that is, **the Logistic model is nested in the Richards model**.

When :math:`\nu>1`, the effect of the logistic term :math:`(1-(\frac{N}{K})^{\nu})`
increases in comparison to the logistic model, 
and the transition from fast growth to slow growth is faster.

When :math:`0<\nu<1`, the effect of the logistic term :math:`(1-(\frac{N}{K})^{\nu})`
decreases in comparison to the logistic model, 
and the transition from fast growth to slow growth is faster.

Baranyi-Roberts model
^^^^^^^^^^^^^^^^^^^^^

The Baranyi-Roberts model [Baranyi1994]_ extends Richards model by introducing a lag phase
in which the population growth is slower than expected while it is adjusting to a new environment.
This extension is done by introducing and adjustment function, :math:`\alpha(t)`:

.. math::

	\alpha(t) = \frac{q_0}{q_0 + e^{-v t}}


where :math:`q_0` is the initial ajdustment of the population and :math:`v` is the adjustment rate.

The model is then described by the following ODE and exact solution:

.. math::
		
		\frac{dN}{dt} = r \alpha(t) N \Big( 1 - \Big(\frac{N}{K}\Big)^{\nu} \Big) \Rightarrow

		N(t) = \frac{K}{\Big[1 - \Big(1 - \Big(\frac{K}{N_0}\Big)^{\nu}\Big) e^{-r \nu A(t)}\Big]^{1/\nu}}

    A(t) = \int_0^t{\alpha(s)ds} = \int_0^t{\frac{q_0}{q_0 + e^{-v s}} ds} = 
    t + \frac{1}{v} \log{\Big( \frac{e^{-v t} + q_0}{1 + q_0} \Big)}


- :math:`N_0`: initial population size
- r: initial per capita growth rate
- K: maximum population size
- :math:`\nu`: curvature of the logsitic term
- :math:`q_0`: initial adjustment to current environment
- v: adjustment rate

Note that :math:`A(t) - t \to  \lambda` as :math:`t \to \infty`; this :math:`\lambda` is called the *lag duration*.

Richards model is then a special case of the Baranyi-Roberts model for :math:`1/v \to 0`, 
that is, **Richards model is nested in Baranyi-Roberts**. 
Therefore, **Logistic is also nested in Baranyi-Roberts** by :math:`\nu=1, 1/v \to 0`.

Logistic models with lag phase
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We define tow additional models: Baranyi-Roberts with :math:`\nu=1` or a logistic model with lag phase, 
and the same model with :math:`v=r` [Baty2004]_.
These models have five and four and parameters, respectively.

Model hierarchy and the likelihood ratio test
---------------------------------------------

The nesting hierarchy between the models is given in Fig. 1. 
This nesting is used for the Likelihood ratio test using the function :py:func:`curveball.models.lrtest` 
in which the nested model (to which the arrows point) is the first argument ``m0``, 
and the nesting model (from which the arrow points) is the second argument ``m1``.

.. figure:: /_static/LRTest_map.png
	
	Fig. 1 - Map of model nesting for the likelihood ratio test.

Parameter guessing
------------------

Before fitting models to the data, Curveball attempts to guess the growth parameters from the shape of the curve. These guesses are used as initial parameters in the model fitting procedure.

``y0`` and ``K`` are guessed from the minimum and maximum of the growth curve, respectively.

``r`` and ``nu`` are guessed in :py:func:`guess_r` and :py:func:`guess_nu`, respectively, using formulas from [Richards1959]_:

.. math::

    N_{max} = K (1 + \nu)^{-\frac{1}{\nu}} \Rightarrow 

    \frac{dN}{dt}_{max} = r K \nu (1 + \nu)^{-\frac{1 + \nu}{\nu}}


- :math:`\frac{dN}{dt}_{max}`: maximum population growth rate
- :math:`N_{max}`: population size/density when the population growth rate (:math:`\frac{dN}{dt}`) is maximum    
- r: initial per capita growth rate 
- K: maximum population size
- :math:`\nu`: curvature of the logsitic term

``q0`` and ``v``, the lag phase parameters, are guessed by fitting a Baranyi-Roberts model with fixed 
``y0``, ``r``, ``K``, and ``nu`` (based on guesses) to the data.

Model selection
---------------

Model selection is done by comparing the `Bayesian Information Criteria (BIC) <http://en.wikipedia.org/wiki/Bayesian_information_criterion>`_ 
of all model fits and choosing the model fit with the **lowest** BIC value. 
BIC is a common method to measure the quality of a model fit [Kaas1995]_,
balancing between model fit (distance from data) and complexity (number of parameters).
See :py:func:`lmfit.model.ModelFit.bic` for more information.

Curveball also calculates the *weighted BIC* each model fitted to the same data.
This can be interpreted as the weight of evidence for each model.

Example
-------

>>> import pandas as pd
>>> import curveball
>>> plate = pd.read_csv('plate_templates/G-RG-R.csv')
>>> df = curveball.ioutils.read_tecan_xlsx('data/Tecan_280715.xlsx', label='OD', plate=plate)
>>> models, fig, ax = curveball.models.fit_model(df[df.Strain == 'G'], PLOT=True, PRINT=False)

.. figure:: /_static/example_model_fitting.svg

   Fig. 2 - Model fitting and selection. The figure is the result of calling ``fig.savefig()`` on the result of the above example code.
   Red error bars: mean and standard deviation of data; Green solid line: model fit; Blue dashed line: initial guess.

References
----------

.. [Richards1959] Richards, F. J., 1959. `A Flexible Growth Function for Empirical Use <http://dx.doi.org/10.1093/jxb/10.2.290>`_. Journal of Experimental Botany
.. [Baranyi1994] Baranyi, J., Roberts, T. A., 1994. `A dynamic approach to predicting bacterial growth in food <www.ncbi.nlm.nih.gov/pubmed/7873331>`_. Int. J. Food Microbiol.
.. [Kaas1995] Kass, R., Raftery, A., 1995. `Bayes Factors <http://www.tandfonline.com/doi/abs/10.1080/01621459.1995.10476572>`_. J. Am. Stat. Assoc.
.. [Gilpin1973] Gilpin, M. E., Ayala, F. J., 1973. `Global Models of Growth and Competition <https://dx.doi.org/10.1073/pnas.70.12.3590>`_. Proc. Nat. Acad. Sci. U S A.
.. [Baty2004] Baty, Florent, and Marie-Laure Delignette-Muller. 2004. `Estimating the Bacterial Lag Time: Which Model, Which Precision? <http://linkinghub.elsevier.com/retrieve/pii/S016816050300429X>`_. Intl. J. Food Microbiol.

Members
-------

.. automodule:: curveball.models
   :members:
