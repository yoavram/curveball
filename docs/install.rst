Installation
============

Curveball is a Python package. As such, it requires a Python interpreter
to work. In addition, Curveball uses several scientific and data
analysis libraries. All the dependencies are listed below, followed by a
section containing instructions on how to install all the dependencies
on the different operating systems.

Dependencies
------------

-  Python 2.7.x (2.6.x might work, 3.x to be supported soon)
-  numpy
-  scipy
-  pandas
-  matplotlib
-  seaborn
-  lmfit
-  statsmodels
-  sympy
-  xlrd
-  lxml
-  click

Installing
------------

The recommended way to install the dependencies is to download and
install `Anaconda <https://www.continuum.io/downloads>`__, available for
Windows, OS X, and Linux. After installing Anaconda, open a terminal or
command line (Windows: click the Start button, write ``cmd`` and click
``Enter``), and write the following commands:

::

    conda update conda --yes
    conda install --yes requests pip numpy scipy matplotlib dateutil pandas statsmodels lxml seaborn sympy xlrd
    pip install lmfit
    pip install git+https://github.com/yoavram/curveball.git

To verify the installation, run this command:

::

    curveball --version

This should output ``curveball, version x.x.x``, where ``x.x.x`` will be
replaced by the stable version number.

Note on versions
^^^^^^^^^^^^^^^^

The versions of the dependencies used to develop Curveball are
documented in the `conda environment
file <https://github.com/yoavram/curveball/blob/master/environment.yml>`__.
However, this file includes packages that are not required for *running* 
Curveball. Some packages are required to test Curveball and build the
documents: for example, nose, coverage, and sphinx.

Note on conda and pip
^^^^^^^^^^^^^^^^^^^^^

The same installation can be achieved using ``pip`` instead of
``conda``, but on Windows and sometimes Linux, too, it is easier to use
conda. Therefore, a ``requirements.txt`` file is not provided.

For Developers
------------

Follow the
`.travis.yml <https://github.com/yoavram/curveball/blob/master/.travis.yml>`__
file for a description of how to install Curveball and all the
dependencies (including those required to test and build the docs) and how to run the tests using nose.
Replace ``$TRAVIS_PYTHON_VERSION`` with the version you want (probably 2.7.10).

If you intend to **contribute to Curveball**, then please do! We encourage
contributions, both to the documentation - from new sections and
examples to typo fixes and rephrasing - and to the source code - 
from new file format parser to new growth and competition models.

Please see the `guidelines for
contributing <https://github.com/yoavram/curveball/blob/master/CONTRIBUTING.md>`__
for instructions and best practices and feel free to contact me on
`Gitter <https://gitter.im/yoavram/curveball>`__,
`Twitter <https://twitter.com/yoavram>`__, and
`Email <mailto:yoav@yoavram.com>`__.
