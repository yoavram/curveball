.. curveball documentation master file, created by
   sphinx-quickstart on Sun Apr 12 13:58:38 2015.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Curveball: Predicting competition results from growth curves
============================================================

|Anaconda| |Install| |Supported Python versions| |License| 
|Build Status| |Docs| |codecov.io| 
|Code Issues| |Join the chat at https://gitter.im/yoavram/curveball|

**Author**: `Yoav Ram <http://www.yoavram.com>`_

Curveball is an open-source software for analysis and  
visualization of high-throughput growth curve data 
and prediction of competition experiment results.

Curveball:

* fits **growth models** to **growth curve data** to estimate values of **growth traits**
* uses estimated **growth traits** and **competition models** to predict results of **competition experiments**
* infers **fitness** and **selection coefficients** from predicted **competition results**


Who is this for?
----------------

Curveball is for researchers who want to analyze growth curve data using 
a framework that integrates **population dynamics** and **population genetics**,
allowing the inference and interpretation of **differences in fitness** 
between strains in terms of **differences in growth traits**.

Curveball provides a **command line interface** (CLI) and an **programmatic interface** (API) 
that can directly work with collections of growth curve measurements (e.g., 96-well plates).

**No programmings skills** are required for using the CLI; 
basic familiarity with the **Python programming language** is recommended for using the API.

.. note::
   
   This documentation provides technical details on using Curveball.
   For more information on the theoretical and computational aspects of Curveball, read the preprint:

   .. pull-quote::

      Ram et al. 2015. `Predicting competition results from growth curves <https://dx.doi.org/10.1101/022640>`_. *bioRxiv*. doi:10.1101/022640.


Quickstart
----------

Install `Anaconda <https://www.continuum.io/downloads>`_, then run:

>>> conda install -c https://conda.anaconda.org/t/yo-766bbd1c-8edd-4b45-abea-85cf58129278/yoavram curveball 
>>> curveball --help

For more detailed instructions, Proceed to the :doc:`Installation instructions <install>` and then to the :doc:`tutorial`.


Contents:
---------

.. toctree::
   :maxdepth: 1

   install
   tutorial
   ioutils
   plots
   models
   baranyi_roberts_model
   likelihood
   competitions
   cli
   troubleshooting

API
---

* :ref:`genindex`
* :ref:`modindex`

Resources:
----------

* `Documentation <https://curveball.divshot.io>`_
* Source code: `GitHub  <https://github.com/yoavram/curveball>`_
* Comments or questions: `Gitter <https://gitter.im/yoavram/curveball>`_, `Twitter <https://twitter.com/yoavram>`_, `Email <mailto:yoav@yoavram.com>`_
* Bugs & feature requests: `GitHub Issues <https://github.com/yoavram/curveball/issues>`_
* Buildbot: `Travis-CI <https://magnum.travis-ci.com/yoavram/curveball>`_
* Code coverage: `Codecov <http://codecov.io/github/yoavram/curveball>`_
* Code quality: `quantifiedcode <https://www.quantifiedcode.com/app/project/fb3dfaa863494b8fa9e3242c542304f6>`_
* `Change log <https://github.com/yoavram/curveball/tree/master/CHANGELOG.md>`_
* `Contributing <https://github.com/yoavram/curveball/blob/master/CONTRIBUTING.md>`_

.. note::
   
   Curveball source code and examples are licensed under the terms of the `MIT license <http://opensource.org/licenses/MIT>`_.
   
   Curveball documentation, examples, and other materials are licensed under the terms of the `Attribution 4.0 International (CC BY 4.0) license <https://creativecommons.org/licenses/by/4.0/>`_.

.. |Anaconda| image:: https://anaconda.org/yoavram/curveball/badges/version.svg   
   :target: https://anaconda.org/yoavram/curveball
.. |Install| image:: https://anaconda.org/yoavram/curveball/badges/installer/conda.svg   
   :target: https://anaconda.org/yoavram/curveball
.. |Supported Python versions| image:: https://img.shields.io/pypi/pyversions/curveball.svg
   :target: https://anaconda.org/yoavram/curveball
.. |License| image:: https://anaconda.org/yoavram/curveball/badges/license.svg
   :target: https://github.com/yoavram/curveball/blob/master/LICENCE.txt
.. |Build Status| image:: https://magnum.travis-ci.com/yoavram/curveball.svg?token=jdWtkbZwtnsj5TaFxVKJ&branch=travis
   :target: https://magnum.travis-ci.com/yoavram/curveball
.. |Docs| image:: https://img.shields.io/badge/docs-latest-yellow.svg
   :target: https://curveball.divshot.io
.. |codecov.io| image:: http://codecov.io/github/yoavram/curveball/coverage.svg?branch=master&token=PV0HysT5gx
   :target: http://codecov.io/github/yoavram/curveball?branch=master
.. |Code Issues| image:: https://www.quantifiedcode.com/api/v1/project/fb3dfaa863494b8fa9e3242c542304f6/badge.svg
   :target: https://www.quantifiedcode.com/app/project/fb3dfaa863494b8fa9e3242c542304f6
.. |Join the chat at https://gitter.im/yoavram/curveball| image:: https://badges.gitter.im/Join%20Chat.svg
   :target: https://gitter.im/yoavram/curveball?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge

Logo `designed by Freepik <http://www.freepik.com/free-vector/ball-of-wool_762106.htm>`_.
