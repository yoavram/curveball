Tutorial
========

This Curveball tutorial walks through loading, processing, and analysing a read growth curve dataset.

Getting started
---------------

About this tutorial
^^^^^^^^^^^^^^^^^^^

There is `no better way <https://csvkit.readthedocs.org/en/0.9.1/tutorial/1_getting_started.html>`_ to learn how to use a new tool than to see it applied in a real world situation. This tutorial will explain the workings of most of  Curveball in the context of analyzing a real growth curve dataset.

The data will be using is an Excel file (``Tecan_280715.xlsx``) file, the result of growing two bacteria strains (*DH5α*, denoted by ``G`` and *TG1*, denoted by ``R``) in a 96-well plate inside a Tecan Infinity plate reader over 17 hours at the Berman Lab in Tel-Aviv University. The experiment was done by `Yoav Ram <http://www.yoavram.com>`. The plate template for this experiment can be seen in Figure 1.

.. figure:: /_static/example_plot_plate.svg

	Fig. 1. Plate template for the *Tecan_280715* experiment, generated from the *G-RG-R.csv* plate template file. Green is for *DH5α*; Red is for *TG1*; Blue is for wells with both strains; White is for blank wells.




This tutorial assumes you are comfortable in the command line, but does not assume any prior experience doing data processing or analysis.