Models
======

This module contains several growth functions and models.

- Functions (``x_function``) are Python function in which the first argumet is time and the rest of the argument are model parameters. Time can be given an a :py:class:`numpy.ndarray`.
- Models (``x_model``) are :py:class:`lmfit.Model` objects from the `lmfit <http://lmfit.github.io/lmfit-py/>`_ package. These objects are used to fit models to data with parameter constraints and some useful statistics and models selection methods.

Members
-------

.. automodule:: curveball.models
   :members:
