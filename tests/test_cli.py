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
# See reference on testing Click applications: http://click.pocoo.org/5/testing/
from click.testing import CliRunner
from curveball.scripts import cli
import curveball


CI = os.environ.get('CI', 'false').lower() == 'true'


class SimpleTestCase(TestCase):
	_multiprocess_can_split_ = True

	def setUp(self):
		pass


	def tearDown(self):
		pass


	def test_help(self):
		runner = CliRunner()
		result = runner.invoke(cli.cli, ['--help'])
		self.assertEquals(result.exit_code, 0)


	def test_version(self):
		runner = CliRunner()
		result = runner.invoke(cli.cli, ['--version'])
		self.assertEquals(result.exit_code, 0)
		self.assertIn(curveball.__version__, result.output)


class PlateTestCase(TestCase):
	_multiprocess_can_split_ = True


	def test_default_plate(self):
		runner = CliRunner()
		result = runner.invoke(cli.cli, ['plate'])
		self.assertEquals(result.exit_code, 0)
		newlines = result.output.count("\n")
		self.assertEquals(newlines, 98) # 96 wells, 1 header, 1 empty line


class AnalysisTestCase(TestCase):
	_multiprocess_can_split_ = True


	def test_process_file(self):
		pass


if __name__ == '__main__':
    main()
