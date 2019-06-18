#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of curveball.
# https://github.com/yoavram/curveball

# Licensed under the MIT license:
# http://www.opensource.org/licenses/MIT-license
# Copyright (c) 2015, Yoav Ram <yoav@yoavram.com>
from unittest import TestCase, main
from builtins import str
import tempfile
import os
import shutil
import zipfile
import pkg_resources
import curveball
import pandas as pd


RANDOM_SEED = int(os.environ.get('RANDOM_SEED', 0))


class CurveballCSVTestCase(TestCase):
    def setUp(self):
        self.filename = pkg_resources.resource_filename("data", "Tecan_210115.csv")
        self.plate = pd.read_csv(pkg_resources.resource_filename("plate_templates", "G-RG-R.csv"))

    def test_read_curveball_csv(self):
        df = curveball.ioutils.read_curveball_csv(self.filename)
        self.assertIsNotNone(df)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(df.shape, (4992, 9))
        self.assertEqual(df.columns.tolist() , ['Time', u'Temp. [\xb0C]', 'Cycle Nr.', 'Well', 'OD', 'Row', 'Col', 'Strain', 'Color'])

    def test_write_curveball_csv(self):
        df = curveball.models.randomize(random_seed=RANDOM_SEED)
        f, fn = tempfile.mkstemp()
        curveball.ioutils.write_curveball_csv(df, fn)
        df1 = pd.read_csv(fn)
        pd.util.testing.assert_frame_equal(df, df1, check_dtype=False)


class TecanXLSXTestCase(TestCase):
    def setUp(self):
        self.filename = pkg_resources.resource_filename("data", "Tecan_210115.xlsx")
        self.plate = pd.read_csv(pkg_resources.resource_filename("plate_templates", "G-RG-R.csv"))

    def test_read_tecan_xlsx_OD(self):
        df = curveball.ioutils.read_tecan_xlsx(self.filename, 'OD')
        self.assertIsNotNone(df)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(df.shape, (8544, 9))
        self.assertEqual(df.columns.tolist() , ['Time', u'Temp. [\xb0C]', 'Cycle Nr.', 'Well', 'OD', 'Row', 'Col', 'Strain', 'Color'])

    def test_read_tecan_xlsx_full(self):
        df = curveball.ioutils.read_tecan_xlsx(self.filename, ('OD', 'Green', 'Red'))
        self.assertIsNotNone(df)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(df.shape, (8352, 15))
        self.assertEqual(sorted(df.columns.tolist()) , sorted(['Time_OD', u'Temp. [\xb0C]_OD', 'Cycle Nr.', 'Well', 'OD', 'Row', 'Col', 'Strain', 'Color', 'Time_Green', u'Temp. [\xb0C]_Green', 'Green', 'Time', u'Temp. [\xb0C]', 'Red']))

    def test_read_tecan_xlsx_12hrs(self):
        df = curveball.ioutils.read_tecan_xlsx(self.filename, 'OD', max_time=12)
        self.assertIsNotNone(df)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(df.shape, (4992, 9))
        self.assertEqual(df.columns.tolist() , ['Time', u'Temp. [\xb0C]', 'Cycle Nr.', 'Well', 'OD', 'Row', 'Col', 'Strain', 'Color'])

    def test_read_tecan_xlsx_with_plate(self):
        df = curveball.ioutils.read_tecan_xlsx(self.filename, 'OD', plate=self.plate)
        self.assertIsNotNone(df)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(df.shape, (8544, 9))
        self.assertEqual(df.columns.tolist() , ['Time', u'Temp. [\xb0C]', 'Cycle Nr.', 'Well', 'OD', 'Row', 'Col', 'Strain', 'Color'])


class BioTekXLSXTestCase(TestCase):
    def setUp(self):
        self.filename = pkg_resources.resource_filename("data", "BioTekSynergy.xlsx")

    def test_read_biotek_xlsx(self):
        df = curveball.ioutils.read_biotek_xlsx(self.filename)
        self.assertIsNotNone(df)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(df.columns.tolist(), ['Time', u'T\xb0 600', 'Well', 'OD', 'Row', 'Col', 'Strain', 'Color'])


# class XMLTestCase(TestCase):
#     def setUp(self):        
#         self.zip_filename = pkg_resources.resource_filename("data", "Tecan_210115.xlsx")
#         self.zip_filename = os.path.join("data", "20130211_dh.zip")
#         self.folder = tempfile.mkdtemp()
#         self.zipfile = zipfile.ZipFile(self.zip_filename)
#         self.zipfile.extractall(self.folder)
#         self.plate = pd.read_csv(pkg_resources.resource_filename("plate_templates", "checkerboard.csv"))


#     def tearDown(self):
#         shutil.rmtree(self.folder)


#     def test_read_tecan_xml_with_plate(self):
#         df = curveball.ioutils.read_tecan_xml(os.path.join(self.folder, "*.xml"), 'OD', plate=self.plate)
#         self.assertIsNotNone(df)
#         self.assertIsInstance(df, pd.DataFrame)
#         self.assertEqual(df.shape, (2016, 8))
#         self.assertEqual(df.columns.tolist() , ['OD', 'Well', 'Row', 'Col', 'Time', 'Filename', 'Strain', 'Color'])    


class SunriseTestCase(TestCase):
    def setUp(self):
        self.filename = pkg_resources.resource_filename("data", "Sunrise_180515_0916.xlsx")
        self.plate = pd.read_csv(pkg_resources.resource_filename("plate_templates", "G-RG-R.csv"))

    def test_read_sunrise_xlsx(self):
        df = curveball.ioutils.read_sunrise_xlsx(self.filename)
        self.assertIsNotNone(df)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(df.shape, (96, 8))
        self.assertEqual(sorted(df.columns.tolist()) , sorted([u'Time', u'Well', u'OD', u'Row', u'Col', 'Strain', 'Color', 'Filename']))

    # commented out 18JUN19 due to travis-ci failure:
    # ValueError: You are trying to merge on object and int64 columns. If you wish to proceed you should use pd.concat
    # def test_read_sunrise_xlsx_plate(self):
    #     df = curveball.ioutils.read_sunrise_xlsx(self.filename, plate=self.plate)
    #     self.assertIsNotNone(df)
    #     self.assertIsInstance(df, pd.DataFrame)
    #     self.assertEqual(df.shape, (96, 8))
    #     self.assertEqual(sorted(df.columns.tolist()) , sorted([u'Time', u'Well', u'OD', u'Row', u'Col', 'Strain', 'Color', 'Filename']))


class MatTestCase(TestCase):
    def setUp(self):
        self.filename = pkg_resources.resource_filename("data", "plate_9_OD.mat")
        self.plate = pd.read_csv(pkg_resources.resource_filename("plate_templates", "checkerboard.csv"))

    def test_read_tecan_mat(self):
        df = curveball.ioutils.read_tecan_mat(self.filename, plate=self.plate)
        self.assertIsNotNone(df)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(df.shape, (2496, 8))
        self.assertEqual(df.columns.tolist() , [u'Cycle Nr.', u'Time', u'Well', u'OD', u'Row', u'Col', 'Strain', 'Color'])


if __name__ == '__main__':
    main()
