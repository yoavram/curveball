Tutorial
========

This Curveball tutorial walks through loading, processing, and analysing a read growth curve dataset.

About this tutorial
-------------------

There is `no better way <https://csvkit.readthedocs.org/en/0.9.1/tutorial/1_getting_started.html>`_ 
to learn how to use a new tool than to see it applied in a real world situation. 
This tutorial will explain the workings of most of Curveball in the context of analysing a real growth curve dataset.

The data will be using is an *Excel* file (:file:`Tecan_280715.xlsx`), 
the result of me growing two bacteria strains (*DH5α*, denoted by ``G`` and *TG1*, denoted by ``R``) 
in a 96-well plate (:numref:`fig-plate`) inside a Tecan Infinity plate reader over 17 hours at the Berman Lab in Tel-Aviv University. 


.. _fig-plate:

.. figure:: /_static/example_plot_plate.svg

	Plate template for the **Tecan_280715** experiment, generated from the :file:`G-RG-R.csv` plate template file. Green is for **DH5α**; Red is for **TG1**; Blue is for wells with both strains; White is for blank wells.


This tutorial assumes you are comfortable in the command line, 
but does not assume any prior experience doing data processing or analysis or with Python programming.

To follow the tutorial, go ahead and open a command line window (or a terminal).


.. note::

  To open a command line (or terminal) in:

  - **Windows**: click the *Start* button, type :command:`cmd.exe` and click *Enter*.
  - **Linux**: click *Ctrl-T*.
  - **OS X**: search for :command:`terminal` in *Spotlight*.


Installing Curveball
--------------------

Use the :doc:`Installation instructions <install>` and check that Curveball was successfully installed:


>>> curveball --version
curveball, version x.x.x


where ``x.x.x`` will be replaced by the current version number (|release|).

Getting the data
----------------

The dataset we will be using is available online.
Let's start by creating a new folder for the tutorial.
On **Windows**:


>>> mkdir curveball-tutorial
>>> cd curveball-tutorial

On **Linux** and **OS X**:


>>> mkdir curveball-tutorial
>>> cd curveball-tutorial

Now download the `data file <https://github.com/yoavram/curveball/raw/master/data/Tecan_280715.xlsx>`_ and the `plate template file <https://github.com/yoavram/curveball/raw/master/plate_templates/G-RG-R.csv>`_.
Download the files using the links above and place themin the new folder.
On **Linux** and **OS X** you can also download directly from the terminal:


>>> curl -L https://github.com/yoavram/curveball/raw/master/data/Tecan_280715.xlsx -o Tecan_280715.xlsx
>>> curl -L https://github.com/yoavram/curveball/raw/master/plate_templates/G-RG-R.csv -o G-RG-R.csv


Analysing the data
------------------

Now we can proceed to analyse the data using Curveball.

For this, we will use the :command:`curveball analyse` command:


>>> curveball analyse Tecan_280715.xlsx --plate_file=G-RG-R.csv --plate_folder=. --ref_strain=G


This command will:

- Load the data from the file
- Fit growth models to the data separately for each strain
- Select the best model fit for each strain
- Use the best model fits to simulate a competition between the strains
- Infer the fitness of the strains from the simulated competition


.. note::
	Some interesting options we used:

	- ``--plate_file``: sets the plate template file to be :file:`G-RG-R.csv` (:numref:`fig-plate`). Plate template files can be generated with `Plato <http://plato.yoavram.com>`_.
	- ``--plate_folder``: this tells Curveball where to find the plate file; by default it will look is a special plate templates folder.
	- ``--ref_strain``: sets the green strain (``G``) to be the reference strain when infering fitness; *i.e.*, the fitness of ``G`` is set to 1 and other strains are compared to it.


It will result in the creation of several figures (in ``.png`` files):


.. _fig-wells:

.. figure:: /_static/Tecan_280715_wells.png

	showing the growth curve in each well of the plate. 	


.. _fig-strains:

.. figure:: /_static/Tecan_280715_strains.png

	showing the mean curve of each strain. 	


.. _fig-strain-G:

.. figure:: /_static/Tecan_280715_strain_G.png

	showing the model fitting and selection plot of strain G.


.. _fig-R_vs_G:

.. figure:: /_static/Tecan_280715_R_vs_G.png

	showing the results of the simulated competition.


Also, it prints out a table that contains a summary for each strain,
including all the growth parameters estimated by Curveball.

Here is the summary table:

.. csv-table:: 
  :file: _static/summary.csv

.. note::
  
  We can run :command:`curveball` again, this time with the ``-o summary.csv`` option, 
  which will cause this table to be saved to a file named :file:`summary.csv` instead of printing to the command line.


Additional commands and options
-------------------------------

Let's see which commands and options :command:`curveball` supports:


>>> curveball --help
Usage: curveball-script.py [OPTIONS] COMMAND [ARGS]...
.   	
Options:
  -v, --verbose / -V, --no-verbose
  -l, --plot / -L, --no-plot
  -p, --prompt / -P, --no-prompt
  --where                         prints the path where Curveball is installed
  --version                       Show the version and exit.
  --help                          Show this message and exit.
.
Commands:
  analyse  Analyse growth curves using Curveball.
  plate    Read and output a plate from a plate file.


We've already seen ``--version``, ``--where``, and now ``--help``.
As for the other options:

- ``--verbose`` allows us to get more information printed from :command:`curveball`; this is useful for bug hunting when we don't get the results we think we should get.
- ``--no-plot`` turns off plotting; no plot files will be created, so :command:`curveball` will finish faster.
- ``--prompt`` turns on prompting; :command:`curveball` will ask for confirmation, for example, when choosing the plate template file.

We can also list the options each command, such as :command:`analyse` and :command:`plate`, can get:


>>> curveball analyse --help
Usage: curveball-script.py analyse [OPTIONS] PATH
.
  Analyse growth curves using Curveball. Outputs estimated growth traits and
  fitness of all strains in all files in folder PATH or matching the pattern
  PATH.
.
Options:
  --max_time FLOAT            omit data after max_time hours
  --ref_strain TEXT           reference strain for competitions
  --blank_strain TEXT         blank strain for background calibration
  -o, --output_file FILENAME  output csv file path
  --plate_file TEXT           plate templates csv file
  --plate_folder PATH         plate templates default folder
  --help                      Show this message and exit.


Getting help
------------

Please don't hesitate to contact me (`Yoav Ram <http://www.yoavram.com>`_) with any questions, comments, or suggestions:

- `Email <mailto:yoav@yoavram.com>`_
- `Twitter <https://twitter.com/yoavram>`_
- `GitHub Issues <https://github.com/yoavram/curveball/issues>`_
