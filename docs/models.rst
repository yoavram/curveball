Growth Models
=============

This module contains several growth functions and models.

- Functions (``x_function``) are Python function in which the first argumet is time and the rest of the argument are model parameters. Time can be given an a :py:class:`numpy.ndarray`.
- Models (``x_model``) are :py:class:`lmfit.Model` objects from the `lmfit <http://lmfit.github.io/lmfit-py/>`_ package. These objects are used to fit models to data with parameter constraints and some useful statistics and models selection methods.

Model selection
---------------

TBD

Likelihood ratio test
---------------------

The nesting between the models is given in Fig. 1. This nesting is used for the Likelihood ratio test using the function :py:func:`curveball.models.lrtest` in which the nested model (to which the arrows point) is the first argument `m0`, and the nesting model (from which the arrow points) is the second argument `m1`.

.. figure:: /_static/LRTest_map.png
	
	Fig. 1 - Map of model nesting for the likelihood ratio test.

	BR6: Baranyi-Roberts with 6 parameters; BR5: Baranyi-Roberts with 5 parameters - :math:`\nu=1`; R4: Richards with 4 parameters (no lag); L3: logistic with 3 parameters (no lag, :math:`\nu=1`). The arrows point from nesting to nested or from the test alternative hypothesis to the null hypothesis. The arrow label defines the parameter that is set to create the nesting. :math:`\nu=1` suggests a standard logistic reduction in growth rate; :math:`\frac{1}{v}=0` suggests that there is no lag phase :math:`\lambda \approx 0` (:math:`0<q_0<1` but otherwise doesn't matter).


Members
-------

.. automodule:: curveball.models
   :members:
