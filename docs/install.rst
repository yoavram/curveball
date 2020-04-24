Installation
============

Curveball is a Python package. As such, it requires a Python interpreter to work. 
In addition, Curveball uses several scientific and data analysis libraries. 
All the dependencies are listed below, 
followed by a section containing instructions on how to install all the dependencies 
on the different operating systems.

Dependencies
------------

-  Python 3
-  numpy
-  scipy
-  pandas
-  matplotlib
-  seaborn
-  lmfit
-  scikit-learn
-  sympy
-  xlrd
-  lxml
-  click
-  future
-  python-dateutil
       

.. hint::

	The minimal debian (**Ubuntu**, etc.) dependencies for Anaconda are:

	>>> apt-get install libsm6 libxrender1 libfontconfig1


Users
-----

The recommended way to install the dependencies is to download and install 
`Anaconda with Python 3 <https://www.anaconda.com/downloads>`_,
available for free on Windows, OS X, and Linux.

After installing Anaconda, open a terminal or command line, and write the following commands to install the dependencies:

>>> conda update --yes conda
>>> python -m pip install curveball

.. tip::

	To open a command line (or terminal) in:

	- **Windows**: click the *Start* button, type :command:`cmd.exe` and click *Enter*.
	- **Linux**: click *Ctrl-T*.
  	- **OS X**: search for :command:`terminal` in *Spotlight*.


Verify installation
^^^^^^^^^^^^^^^^^^^

To verify the installation, run this command:

>>> curveball --version
curveball, version x.x.x

where ``x.x.x`` will be replaced by the current version number (|release|).


Updating
^^^^^^^^

To update Curveball:

>>> python -m pip install -U curveball

.. attention::

	If you previously installed Curveball using `conda` (prior to July 2019), please remove the installation and re-install using `pip`:

	>>> conda remove curveball
	>>> python -m pip install curveball


Dependencies versions
^^^^^^^^^^^^^^^^^^^^^

The versions of the dependencies used to develop Curveball are documented in the `conda environment file <https://github.com/yoavram/curveball/blob/master/environment.yml>`_.


Contributing
^^^^^^^^^^^^

Please do! We encourage contributions, both to the documentation - 
from new sections and examples to typo fixes and rephrasing - 
and to the source code - 
from new file format parser to new growth and competition models.

Please see the `guidelines for contributing <https://github.com/yoavram/curveball/blob/master/CONTRIBUTING.md>`_
for instructions and best practices and feel free to contact me via 
`Email <mailto:yoav@yoavram.com>`_ and `Twitter <https://twitter.com/yoavram>`_.
