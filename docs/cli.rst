CLI
============

The :py:mod:`cli <curveball.scripts.cli>` module implements a command line interface that allows to interact with Curveball
through a command line or terminal, without any knowledge in Python programming.

The :doc:`tutorial` includes a complete walkthrough on using the CLI to analyse growth cruves data.

Fit-and-compete
---------------

Use ``curveball fit_compete`` to estimate competition coefficients from mixed-culture data (requires a plate template that includes a ``total`` strain):

>>> curveball fit_compete mixed.xlsx --analyse_file=summary.csv --ref_strain=G --total=RG

The output CSV contains:

- ``a1``, ``a2``: competition coefficients from :py:func:`curveball.competitions.fit_and_compete`
- ``MSE``: mean squared error of the mixed-culture fit

Analyse output columns
----------------------

The ``curveball analyse`` command outputs a CSV table with one row per strain. Key columns include:

- ``folder``, ``filename``: input file location and base name.
- ``strain``: strain label from the plate template.
- ``model``: best-fit model name.
- ``RSS``, ``RMSD``, ``NRMSD``, ``CV(RMSD)``: fit error metrics.
- ``bic``, ``aic``, ``weighted_bic``, ``weighted_aic``: model selection scores.
- ``y0``, ``K``, ``r``, ``nu``, ``q0``, ``v``: fitted growth parameters.
- ``has_lag``, ``has_nu``: model selection flags.
- ``max_growth_rate``, ``min_doubling_time``, ``lag``: derived metrics.
- ``w``: relative fitness vs. reference strain.

If ``--ci`` is used, additional ``*_low``/``*_high`` columns are included for confidence intervals (e.g., ``lag_low``, ``lag_high``, ``K_low``, ``K_high``).

Members
-------

.. automodule:: curveball.scripts.cli
   :members:
