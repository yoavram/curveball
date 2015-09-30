#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of curveball.
# https://github.com/yoavram/curveball

# Licensed under the MIT license:
# http://www.opensource.org/licenses/MIT-license
# Copyright (c) 2015, Yoav Ram <yoav@yoavram.com>
from unittest import TestCase, main
from nose.plugins.skip import SkipTest
import os
import shutil
import pkg_resources
from click.testing import CliRunner # See reference on testing Click applications: http://click.pocoo.org/5/testing/
from curveball.scripts import cli
import curveball


CI = os.environ.get('CI', 'false').lower() == 'true'


class SimpleTestCase(TestCase):
	_multiprocess_can_split_ = True

	def setUp(self):
		self.runner = CliRunner()


	def tearDown(self):
		pass


	def test_help(self):
		result = self.runner.invoke(cli.cli, ['--help'])
		self.assertEquals(result.exit_code, 0)


	def test_version(self):
		result = self.runner.invoke(cli.cli, ['--version'])
		self.assertEquals(result.exit_code, 0)
		self.assertIn(curveball.__version__, result.output)


class PlateTestCase(TestCase):
	_multiprocess_can_split_ = True


	def setUp(self):
		self.runner = CliRunner()

	def test_default_plate(self):
		result = self.runner.invoke(cli.cli, ['plate'])
		self.assertEquals(result.exit_code, 0)
		newlines = result.output.count("\n")
		self.assertEquals(newlines, 98) # 96 wells, 1 header, 1 empty line


	def test_non_default_plate(self):
		result = self.runner.invoke(cli.cli, ['plate', '--plate_file=G-RG-R.csv'])
		self.assertEquals(result.exit_code, 0)
		newlines = result.output.count("\n")
		self.assertEquals(newlines, 98) # 96 wells, 1 header, 1 empty line


def setup_with_context_manager(testcase, cm):
    """Use a contextmanager to setUp a test case.
    See http://nedbatchelder.com/blog/201508/using_context_managers_in_test_setup.html
    """
    val = cm.__enter__()
    testcase.addCleanup(cm.__exit__, None, None, None)
    return val


class AnalysisTestCase(TestCase):
	_multiprocess_can_split_ = True
	filename = 'Tecan_280715.xlsx'


	def setUp(self):
		self.runner = CliRunner()
		self.ctx = setup_with_context_manager(self, self.runner.isolated_filesystem())
		src = pkg_resources.resource_filename('data', self.filename)
		dst = os.path.join('data', self.filename)
		os.makedirs(dst)
		shutil.copy(src, dst)		
		self.filepath = dst		
		self.assertTrue(os.path.exists(self.filepath))
		self.assertTrue(os.path.exists(os.path.join(os.getcwd(), self.filepath)))


	def tearDown(self):
		pass


	def test_process_file(self):
		result = self.runner.invoke(cli.cli, ['--no-plot', '--no-verbose', '--no-prompt', 'analyse', self.filepath, '--plate_file=G-RG-R.csv', '--ref_strain=G'])
		self.assertEquals(result.exit_code, 0, result.output)


if __name__ == '__main__':
    main()
