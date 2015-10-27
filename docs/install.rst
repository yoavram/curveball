Installation
============

Curveball is a Python package. As such, it requires a Python interpreter to work. 
In addition, Curveball uses several scientific and data analysis libraries. 
All the dependencies are listed below, 
followed by a section containing instructions on how to install all the dependencies 
on the different operating systems.

Dependencies
------------

-  Python 3.4 and 2.7
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
-  future
-  dateutil
       

.. hint::

	The minimal debian (**Ubuntu**, etc.) dependencies for Anaconda are:

	>>> apt-get install libsm6 libxrender1 libfontconfig1


Users
-----

The recommended way to install the dependencies is to download and install 
`Anaconda <https://www.continuum.io/downloads>`_ (Python 3.4 or 2.7),
available for free on Windows, OS X, and Linux.

After installing Anaconda, open a terminal or command line, and write the following commands to install the dependencies:

>>> conda update --yes conda
>>> conda install -c https://conda.anaconda.org/t/yo-766bbd1c-8edd-4b45-abea-85cf58129278/yoavram curveball 

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

>>> conda update -c https://conda.anaconda.org/t/yo-766bbd1c-8edd-4b45-abea-85cf58129278/yoavram curveball 


Dependencies versions
---------------------

The versions of the dependencies used to develop Curveball are documented in the `conda environment file <https://github.com/yoavram/curveball/blob/master/environment.yml>`_.


Installing on Linux servers and clusters
----------------------------------------

The following method **does not** require :command:`sudo` permissions and is suitable for installing Curveball on remote Linux machines using the terminal.
We install a minimal Anaconda distribution (a.k.a *miniconda*) and then install Curveball and its dependencies.

>>> wget http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
>>> chmod +x miniconda.sh
>>> ./miniconda.sh -b
>>> export PATH=$HOME/miniconda3/bin:$PATH
>>> conda update conda --yes
>>> conda config --add channels https://conda.anaconda.org/t/yo-766bbd1c-8edd-4b45-abea-85cf58129278/yoavram
>>> conda install curveball --yes
>>> curveball --where
$HOME/miniconda3/lib/python3.4/site-packages/curveball

where the ``$HOME`` will be replaced by the path to your home folder (and possibly with a different Python version specifier if you chose to install Python 2.7).

.. tip::

	- If you prefer Python 2.7 over 3.4, change ``Miniconda3`` to ``Miniconda`` in the 1st command and ``miniconda3`` to ``miniconda`` in the 4th line.
	- On some machines you will need to replace ``export PATH=...`` with ``setenv PATH ...``, depending on your shell.
	- You might want to add the 4th command to your :file:`.profile` or :file:`.bashrc` file so that you will always use the installed Python distribution instead of any other pre-installed Python. Otherwise, you will need to run this command on every new session.
	- If you're not sure that you are using the Python installation that has Curveball, type ``which python``, it should give you ``$HOME/miniconda3/bin/python``.
	- If you use Curveball in a Python script on the remote Linux machine and get a runtime error about an ``Invalid DISPLAY variable``, then add these lines the top of your script (this will change :py:mod:`matplotlib`'s default plotting backend):
	>>> import matplotlib
	>>> matplotlib.use("Agg")
