.. curveball documentation master file, created by
   sphinx-quickstart on Sun Apr 12 13:58:38 2015.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Curveball: Predicting competition results from growth curves
============================================================
**Author**: `Yoav Ram <http://www.yoavram.com>`_.

Curveball is an open source software for analysis and visualization of high-throughput growth curve data 
and prediction of competition experiment results.

Curveball:

* fits **growth models** to **growth curve data** to estimate values of **growth traits**
* uses estimated **growth traits** and **competition models** to predict results of **competition experiments**
* infers **fitness** and **selection coefficients** from predicted **competition results**

Who is this for?
----------------

Curveball is for researchers who want to analyze growth curve data using a framework that integrates **population dynamics** and **population genetics**,
allowing the inference and interpretation of **differences in fitness** between strains in terms of **differences in growth traits**.

Curveball provides a **command line interface** (CLI) and an **programmatic interface** (API) 
that can directly work with collections of growth curve measurements (e.g., 96-well plates).

**No programmings skills** are required for using the CLI; 
basic familiarity with the **Python programming language** is recommended for using the API.

You can find a few examples that load, plot, and analyse growth curve data and predict competitions results in the gallery page.

If you like what you see, then proceed to the installation page and then to the tutorial.

Contents:
---------

.. toctree::
   :maxdepth: 1

   install
   tutorial
   gallery
   ioutils
   plots
   models
   competitions
   cli

Resources:
----------

* Documentation: `Read the docs <http://curveball.rtfd.org/>`_
* Source code: `GitHub  <https://github.com/yoavram/curveball>`_
* Comments or questions: `Gitter <https://gitter.im/yoavram/curveball>`_, `Twitter <https://twitter.com/yoavram>`_, `Email <mailto:yoav@yoavram.com>`_
* Bugs & feature requests: `GitHub Issues <https://github.com/yoavram/curveball/issues>`_
* `Change log <https://github.com/yoavram/curveball/tree/master>`_
* `Contributing <https://github.com/yoavram/curveball/blob/master/CONTRIBUTING.md>`_
* `License <https://github.com/yoavram/curveball/blob/master/LICENCE.txt>`_

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Logo `designed by Freepik <http://www.freepik.com/free-vector/ball-of-wool_762106.htm>`_.
