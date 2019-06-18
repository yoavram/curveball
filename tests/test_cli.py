from builtins import map
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
import glob
import io
import shutil
import pkg_resources
import pandas as pd
from click.testing import CliRunner # See reference on testing Click applications: http://click.pocoo.org/5/testing/
from curveball.scripts import cli
import curveball
import click


CI = os.environ.get('CI', 'false').lower() == 'true'


def is_csv(data):
	lines = data.splitlines()
	data  = [line.split(',') for line in lines]
	lengths = list(map(len, data))
	return all(x==lengths[0] for x in lengths)


class SimpleTestCase(TestCase):
	_multiprocess_can_split_ = True

	def setUp(self):
		self.runner = CliRunner()


	def tearDown(self):
		pass


	def test_help(self):
		result = self.runner.invoke(cli.cli, ['--help'])
		self.assertEqual(result.exit_code, 0)


	def test_version(self):
		result = self.runner.invoke(cli.cli, ['--version'])
		self.assertEqual(result.exit_code, 0)
		self.assertIn(curveball.__version__, result.output)


	def test_where(self):
		result = self.runner.invoke(cli.cli, ['--where'])
		self.assertEqual(result.exit_code, 0)
		self.assertIn("curveball", result.output.lower())
		path = result.output.strip()
		self.assertTrue(os.path.exists(path), msg=path)


class PlateTestCase(TestCase):
	_multiprocess_can_split_ = True


	def setUp(self):
		self.runner = CliRunner()


	def _is_plate_csv(self, data):
		self.assertTrue(is_csv(data))
		newlines = data.count("\n")
		self.assertEqual(newlines, 97) # 96 wells, 1 header line


	def test_default_plate(self):
		result = self.runner.invoke(cli.cli, ['plate'])
		self.assertEqual(result.exit_code, 0)
		self._is_plate_csv(result.output)


	def test_non_default_plate(self):
		result = self.runner.invoke(cli.cli, ['plate', '--plate_file=G-RG-R.csv'])
		self.assertEqual(result.exit_code, 0)
		self._is_plate_csv(result.output)


	def test_default_plate_to_file(self):
		filename = 'plate.csv'
		with self.runner.isolated_filesystem():
			result = self.runner.invoke(cli.cli, ['plate', '--output_file={0}'.format(filename)])
			self.assertEqual(result.exit_code, 0)
			with open(filename, 'r') as f:
				data = f.read()		
			self._is_plate_csv(data)


	def test_plate_not_found(self):
		result = self.runner.invoke(cli.cli, ['plate', '--plate_file=untitled.csv'])
		self.assertNotEquals(result.exit_code, 0)
		self.assertIn('untitled.csv', result.output)


	def test_bad_plate_file(self):
		filename = 'bad_plate.csv'
		with self.runner.isolated_filesystem():
			with open(filename, 'w') as f:
				import this
				f.write(this.s)
			result = self.runner.invoke(cli.cli, ['plate', '--plate_file={0}'.format(filename)])
			self.assertNotEquals(result.exit_code, 0, result.output)
			self.assertIn(filename, result.output)


	def test_plate_list(self):
		result = self.runner.invoke(cli.cli, ['plate', '--list'])
		self.assertEqual(result.exit_code, 0)
		self.assertIn('G-RG-R.csv', result.output)
		self.assertTrue(result.output.count('\n') > 2)


	def test_plate_plot(self):
		result = self.runner.invoke(cli.cli, ['plate', '--show'])
		self.assertEqual(result.exit_code, 0)		


	def test_plate_plot_to_file(self):
		filename = 'plate.png'
		with self.runner.isolated_filesystem():
			result = self.runner.invoke(cli.cli, ['plate', '--show', '--output_file={0}'.format(filename)])
			self.assertEqual(result.exit_code, 0)
			self.assertTrue(os.path.exists(filename))		


class AnalysisTestCase(TestCase):
	_multiprocess_can_split_ = True


	def setup_with_context_manager(self, cm):
	    """Use a contextmanager to setUp a test case.
	    See http://nedbatchelder.com/blog/201508/using_context_managers_in_test_setup.html
	    """
	    val = cm.__enter__()
	    self.addCleanup(cm.__exit__, None, None, None)
	    return 


	def setUp(self):
		self.files = pkg_resources.resource_listdir('data', '')
		self.files = [fn for fn in self.files if os.path.splitext(fn)[-1] in ['.xlsx', '.mat']]
		self.files = [fn for fn in self.files if not fn.lower().startswith('sunrise')]
		self.files = [fn for fn in self.files if not fn.lower().startswith('biotek')]
		self.files.sort()
		self.runner = CliRunner()
		self.ctx = self.setup_with_context_manager(self.runner.isolated_filesystem())
		self.dirpath = os.getcwd()
		self.assertTrue(os.path.exists(self.dirpath))
		self.assertTrue(os.path.isdir(self.dirpath))

		print("files:", self.files)
		for fn in self.files:
			src = pkg_resources.resource_filename('data', fn)
			shutil.copy(src, '.')
			self.assertTrue(os.path.exists(os.path.join(self.dirpath, fn)))
			self.assertTrue(os.path.isfile(os.path.join(self.dirpath, fn)))
		self.filepath = os.path.join(self.dirpath, self.files[1])
		

	def tearDown(self):
		pass	


	def test_process_file(self):
		result = self.runner.invoke(cli.cli, ['--no-plot', '--verbose', '--no-prompt', 'analyse', self.filepath, '--plate_file=G-RG-R.csv', '--ref_strain=G'])
		self.assertEqual(result.exit_code, 0, "Code: {}\n{}".format(result.exit_code, result.output))
		lines = [line for line in result.output.splitlines() if len(line) > 0] 
		data = os.linesep.join(lines[-4:]) # only last 4 lines
		self.assertTrue(is_csv(data))


	def test_process_file2(self):
		filepath = os.path.join(self.dirpath, self.files[1])
		result = self.runner.invoke(cli.cli, ['--no-plot', '--verbose', '--no-prompt', 'analyse', filepath, '--plate_file=G-RG-R.csv', '--ref_strain=G'])
		self.assertEqual(result.exit_code, 0, "Code: {}\n{}".format(result.exit_code, result.output))
		lines = [line for line in result.output.splitlines() if len(line) > 0] 
		data = os.linesep.join(lines[-4:]) # only last 4 lines
		self.assertTrue(is_csv(data))


	def test_process_file_to_file(self):
		result = self.runner.invoke(cli.cli, ['--no-plot', '--verbose', '--no-prompt', 'analyse', self.filepath, '--plate_file=G-RG-R.csv', '--ref_strain=G', '--output_file=summary.csv'])
		self.assertEqual(result.exit_code, 0, "Code: {}\n{}".format(result.exit_code, result.output))
		with open('summary.csv', 'r') as f:
			data = f.read()
		self.assertTrue(is_csv(data))


	def test_process_file_with_guess(self):
		result = self.runner.invoke(cli.cli, ['--no-plot', '--verbose', '--no-prompt', 'analyse', self.filepath, '--guess', 'K', '0.7', '--guess', 'nu', '2.0', '--plate_file=G-RG-R.csv', '--ref_strain=G'])
		self.assertEqual(result.exit_code, 0, "Code: {}\n{}".format(result.exit_code, result.output))
		lines = [line for line in result.output.splitlines() if len(line) > 0] 
		data = os.linesep.join(lines[-4:]) # only last 4 lines
		self.assertTrue(is_csv(data))


	def test_process_file_with_param_max(self):
		result = self.runner.invoke(cli.cli, ['--no-plot', '--verbose', '--no-prompt', 'analyse', self.filepath, '--param_max', 'K', '0.7', '--param_max', 'nu', '2.0', '--param_max', 'v', '100', '--plate_file=G-RG-R.csv', '--ref_strain=G'])
		self.assertEqual(result.exit_code, 0, "Code: {}\n{}".format(result.exit_code, result.output))
		lines = [line for line in result.output.splitlines() if len(line) > 0] 
		data = os.linesep.join(lines[-4:]) # only last 4 lines
		self.assertTrue(is_csv(data))


	def test_process_file_with_param_min(self):
		result = self.runner.invoke(cli.cli, ['--no-plot', '--verbose', '--no-prompt', 'analyse', self.filepath, '--param_min', 'K', '0.2', '--param_min', 'nu', '0.1', '--param_min', 'v', '0.1', '--plate_file=G-RG-R.csv', '--ref_strain=G'])
		self.assertEqual(result.exit_code, 0, "Code: {}\n{}".format(result.exit_code, result.output))
		lines = [line for line in result.output.splitlines() if len(line) > 0] 
		data = os.linesep.join(lines[-4:]) # only last 4 lines
		self.assertTrue(is_csv(data))


	def test_process_file_with_param_fix(self):
		result = self.runner.invoke(cli.cli, ['--no-plot', '--verbose', '--no-prompt', 'analyse', self.filepath, '--param_fix', 'K', '--plate_file=G-RG-R.csv', '--ref_strain=G'])
		#self.assertEqual(result.exit_code, 0, "Code: {}\n{}".format(result.exit_code, result.output))
		lines = [line for line in result.output.splitlines() if len(line) > 0] 
		data = os.linesep.join(lines[-4:]) # only last 4 lines
		self.assertTrue(is_csv(data))


	def test_process_file_without_weights(self):
		result = self.runner.invoke(cli.cli, ['--no-plot', '--verbose', '--no-prompt', 'analyse', self.filepath, '--no-weights', '--plate_file=G-RG-R.csv', '--ref_strain=G'])
		self.assertEqual(result.exit_code, 0, "Code: {}\n{}".format(result.exit_code, result.output))
		lines = [line for line in result.output.splitlines() if len(line) > 0] 
		data = os.linesep.join(lines[-4:]) # only last 4 lines
		self.assertTrue(is_csv(data))


	# this test works but takes too long (10 min) see #129
	def test_process_file_with_ci(self):
		result = self.runner.invoke(cli.cli, [
			'--no-plot', '--verbose', '--no-prompt',
			'analyse', self.filepath, '--ci', '--nsamples=3',
			'--plate_file=G-RG-R.csv', '--ref_strain=G'
		])
		self.assertEqual(result.exit_code, 0, "Code: {}\n{}".format(result.exit_code, result.output))
		lines = [line for line in result.output.splitlines() if len(line) > 0]
		data = os.linesep.join(lines[-4:]) # only last 4 lines
		self.assertTrue(is_csv(data))


	# FIXME - fails on CI, succeeds on local
	# def test_create_plots(self):
	# 	result = self.runner.invoke(cli.cli, ['--plot', '--verbose', '--no-prompt', 'analyse', self.filepath, '--plate_file=G-RG-R.csv', '--ref_strain=G'])
	# 	self.assertEqual(result.exit_code, 0, result.output)
	# 	path,ext = os.path.splitext(self.filepath)		
	# 	plot_files = glob.glob(path + "_*.png")
	# 	self.assertNotEqual(len(plot_files), 0, result.output)


	def test_process_folder(self):
		num_files = len(self.files)
		result = self.runner.invoke(cli.cli, ['--no-plot', '--verbose', '--no-prompt', 
			'analyse', self.dirpath, '--plate_file=G-RG-R.csv', '--ref_strain=G'])
		self.assertEqual(result.exit_code, 0, "Code: {}\n{}".format(result.exit_code, result.output))
		lines = [line for line in result.output.splitlines() if len(line) > 0] 
		num_lines =  num_files * 3 + 1
		data = os.linesep.join(lines[-num_lines:])
		self.assertTrue(is_csv(data), result.output)
		

	def test_path_not_found(self):
		result = self.runner.invoke(cli.cli, ['analyse', 'untitled.xlsx'])
		self.assertNotEquals(result.exit_code, 0)
		self.assertIn('untitled.xlsx', result.output)


	def test_no_files_in_folder(self):
		for fn in self.files:
			os.remove(fn)
		self.assertEqual(len(glob.glob("*")), 0)
		result = self.runner.invoke(cli.cli, ['--no-plot', '--verbose', '--no-prompt', 'analyse', '.'])
		self.assertNotEquals(result.exit_code, 0)
		self.assertIn('.', result.output)


	def test_bad_data_file(self):
		shutil.copyfile(__file__, 'untitled.xlsx')
		result = self.runner.invoke(cli.cli, ['analyse', 'untitled.xlsx'])
		self.assertNotEquals(result.exit_code, 0)
		self.assertIn('untitled.xlsx', result.output)


if __name__ == '__main__':
    main()
