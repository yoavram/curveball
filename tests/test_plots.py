#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of curveball.
# https://github.com/yoavram/curveball

# Licensed under the MIT license:
# http://www.opensource.org/licenses/MIT-license
# Copyright (c) 2015, Yoav Ram <yoav@yoavram.com>
from unittest import TestCase, main
import pkg_resources
import shutil
import tempfile
import os
import pandas as pd
import numpy as np
from PIL import Image
import seaborn as sns
import curveball
import matplotlib


CI = os.environ.get('CI', 'false').lower() == 'true'


class MplTestCase(TestCase):
	def test_mpl(self):
		if CI:
			self.assertEquals(matplotlib.rcParams['backend'].lower(), 'agg')


class PlotsTestCase(TestCase):
	def setUp(self):
		self.folder = tempfile.mkdtemp()
		self.output_filename = os.path.join(self.folder, 'plate.png')


	def tearDown(self):
		shutil.rmtree(self.folder)


	def check_image(self):		
		im = Image.open(self.output_filename)
		bands = im.split()
		self.assertFalse(all(band.getextrema() == (255, 255) for band in bands))
		

	def test_plot_wells(self):
		df = pd.read_csv(pkg_resources.resource_filename("data", "Tecan_210115.csv"))
		g = curveball.plots.plot_wells(df, output_filename=self.output_filename)
		self.assertIsInstance(g, sns.Grid)
		self.check_image()


	def test_plot_strains(self):
		df = pd.read_csv(pkg_resources.resource_filename("data", "Tecan_210115.csv"))
		g = curveball.plots.plot_strains(df, output_filename=self.output_filename)
		self.assertIsInstance(g, sns.Grid)
		self.check_image()


	def test_plot_strains_no_output(self):
		df = pd.read_csv(pkg_resources.resource_filename("data", "Tecan_210115.csv"))
		g = curveball.plots.plot_strains(df)
		self.assertIsInstance(g, sns.Grid)
		with self.assertRaises(IOError):
			self.check_image()


	def test_tsplot(self):
		df = pd.read_csv(pkg_resources.resource_filename("data", "Tecan_210115.csv"))		
		g = curveball.plots.tsplot(df, output_filename=self.output_filename)
		self.assertIsInstance(g, matplotlib.axes.Axes)
		self.check_image()


	def test_plot_plate(self):
		df = pd.read_csv(pkg_resources.resource_filename("plate_templates", "checkerboard.csv"))		
		fig, ax = curveball.plots.plot_plate(df, output_filename=self.output_filename)
		self.assertIsInstance(fig, matplotlib.figure.Figure)
		self.check_image()


	def test_plot_params_distribution(self):
		df = pd.DataFrame(np.random.normal(0, 1, size=(100, 4)), columns=list('ABCD'))
		g = curveball.plots.plot_params_distribution(df)
		self.assertIsInstance(g, sns.Grid)


if __name__ == '__main__':
	main()
