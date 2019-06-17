I/O Utils
=========

The :py:mod:`ioutils <curveball.ioutils>` module contains functions for reading data from automatic plate readers.
The different functions read the data files and generate a data table of type :py:class:`pandas.DataFrame`
which contains all the relevant data: the read from every well at every time point. 

This data table is in a `tidy data <http://vita.had.co.nz/papers/tidy-data.html>`_ format, meaning that each row in the table contains a single measurement with the following values (as columns):

-  ``Time``: in hours (mandatory)
-  ``OD``: optical density which is a proxy for cell density (mandatory)
-  ``Well``: as in the name of the well such as "A1" or "H12" (optional)
-  ``Row``, ``Col``: the row and column of the well in the plate (optional)
-  ``Strain``: the name of the strain (optional)
-  ``Color``: the color that should be given to graphs of the data from this well (optional)

Any other columns can also be provided (for example, ``Cycle Nr.`` and ``Temp. [°C]`` are provided by Tecan Infinity).

Example of a :py:class:`pandas.DataFrame` generated using the :py:mod:`ioutils` module functions:

.. csv-table:: 
  :file: _static/Tecan_210115.head6.csv

Plate template
^^^^^^^^^^^^^^

Normally, the output of a plate reader doesn't include information about the strain in each well.
To integrate that information 
(as well as the colors that should be used for plotting the data from each well),
you must provide a plate definition CSV file. 

This plate template file is a table in which each row has four values: 
``Row``, ``Col``, ``Strain``, and ``Color``.
The ``Row`` and ``Col`` values define the wells; the ``Strain`` and ``Color`` values 
define the names of the strains and their respective colors (for plotting purposes).
These template files can be created using the 
`Plato web app <http://plato.yoavram.com>`_, using Excel (save as ``.csv``), 
or in any other way that is convinient to you.

Curveball is also shipped with some plate templates files - 
type ``curveball plate list`` in the :py:mod:`command line <curveball.scripts.cli>` for a list of the builtin plate templates:

::

	> curveball plate --list
	checkerboard.csv
	checkerboard2.csv
	DH5a-s12-TG1.csv
	DH5a-TG1.csv
	G-RG-R.csv
	nine-strains.csv
	six-strains.csv

Example of the first 5 rows of a plate template file:

.. csv-table:: 
  :file: _static/plate.head6.csv

A full example can be viewed by typing ``curveball plate`` in the :py:mod:`command line <curveball.scripts.cli>`.

Members
-------

.. automodule:: curveball.ioutils
   :members:
