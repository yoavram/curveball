Competition Models
==================

The :py:mod:`.competitions` module contains functions for predicting 
the results of competitions between different species or strains
and analysing the competitions results to extract the fitness 
or selection coefficients of the competing strains.

The model contains:

- **Competition function** (:py:func:`.compete`) performs a numerical simulation of a competition.
- **Competition models** (``dobule_*_ode*``) are *Python* implementations of ODE systems, consistent with *Scipy*'s :py:func:`odeint <scipy.integrate.odeint>` requirement. These ODE systems define competition models in which two strains/species compete for resources while growing in a well-mixed homogeneous vessel.
- **Fitness and selection coefficient analysis** (``fitness_*``, ``selection_coefs_*``) are functions for infering fitness and selection coefficients from competitions time-series.


How competitions work
---------------------

Consider the following double logistic competition model in the form of a system of ordinary differential equations:

.. math::

	\frac{dN_i}{dt} = r_i N_i (1 - \frac{\sum_{j}{N_j}}{K_i})


- :math:`N_i`: population size of strain *i*.
- :math:`r_i`: initial per capita growth rate of strain *i*
- :math:`K_i`: maximum population size of strain *i*

To simulate a competition between two strains we need to set the values of 
:math:`N_i(0), r_i, K_i` for :math:`i \in {1,2}`
where *1* and *2* define two strains.
These values can be estimated from growth curve data using :py:func:`curveball.models.fit_model`, 
which fits growth models to data and returns model fit results.
These model fit results can be passed to the :py:func:`.compete` function to simulate a competition:

>>> import pandas as pd
>>> import curveball
>>> plate = pd.read_csv('plate_templates/G-RG-R.csv')
>>> df = curveball.ioutils.read_tecan_xlsx('data/Tecan_280715.xlsx', label='OD', plate=plate)
>>> green = curveball.models.fit_model(df[df.Strain == 'G'], PLOT=False, PRINT=False)[0]
>>> red = curveball.models.fit_model(df[df.Strain == 'R'], PLOT=False, PRINT=False)[0]
>>> t,y,fig,ax = curveball.competitions.compete(green, red, PLOT=True, colors=['green', 'red'])

The result of :py:func:`.compete`, ``t`` and ``y`` are the time array and the frequencies array of the competition: 
specifically, ``y`` is a 2-dimensional array, containing the frequency of each strain at each time point.

.. figure:: /_static/example_competition.svg
	
	Fig. 1 - Competition simulation. The figure is the result of calling ``fig.savefig()`` on the result of the above example code.

Members
-------

.. automodule:: curveball.competitions
   :members:
