#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of curveball.
# https://github.com/yoavram/curveball

# Licensed under the MIT license:
# http://www.opensource.org/licenses/MIT-license
# Copyright (c) 2015, Yoav Ram <yoavram+github@gmail.com>

from unittest import TestCase, main
import os
import zipfile

import curveball
import pandas as pd

class XLSXTestCase(TestCase):
    def setUp(self):
        self.filename = os.path.join("data", "yoavram", "Tecan_210115.xlsx")
        if not os.path.exists(self.filename):
            raise IOError("Data file not found: %s" % self.filename)
        self.plate = pd.read_csv(os.path.join("plate_templates", "G-RG-R.csv"))       


    def test_read_tecan_xlsx_OD(self):
        df = curveball.ioutils.read_tecan_xlsx(self.filename, 'OD')
        self.assertIsNotNone(df)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEquals(df.shape, (8448, 9))
        self.assertEquals(df.columns.tolist() , ['Time', u'Temp. [\xb0C]', 'Cycle Nr.', 'Well', 'OD', 'Row', 'Col', 'Strain', 'Color'])


    def test_read_tecan_xlsx_full(self):
        df = curveball.ioutils.read_tecan_xlsx(self.filename, ('OD', 'Green', 'Red'))
        self.assertIsNotNone(df)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEquals(df.shape, (8352, 15))
        self.assertEquals(df.columns.tolist() , ['Time_OD', u'Temp. [\xb0C]_OD', 'Cycle Nr.', 'Well', 'OD', 'Row', 'Col', 'Strain', 'Color', 'Time_Green', u'Temp. [\xb0C]_Green', 'Green', 'Time', u'Temp. [\xb0C]', 'Red'])


    def test_read_tecan_xlsx_12hrs(self):
        df = curveball.ioutils.read_tecan_xlsx(self.filename, 'OD', max_time=12)
        self.assertIsNotNone(df)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEquals(df.shape, (4992, 9))
        self.assertEquals(df.columns.tolist() , ['Time', u'Temp. [\xb0C]', 'Cycle Nr.', 'Well', 'OD', 'Row', 'Col', 'Strain', 'Color'])



    def test_read_tecan_xlsx_with_plate(self):
        df = curveball.ioutils.read_tecan_xlsx(self.filename, 'OD', plate=self.plate)
        self.assertIsNotNone(df)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEquals(df.shape, (8448, 9))
        self.assertEquals(df.columns.tolist() , ['Time', u'Temp. [\xb0C]', 'Cycle Nr.', 'Well', 'OD', 'Row', 'Col', 'Strain', 'Color'])


class XMLTestCase(TestCase):
    def setUp(self):            
        self.folder = os.path.join("data", "dorith")
        self.zip_filename = os.path.join(self.folder, "20140911_dorit.zip")
        if not os.path.exists(self.zip_filename):
            raise IOError("Data file not found: %s" % self.zip_filename)
        self.zipfile = zipfile.ZipFile(self.zip_filename)
        self.zipfile.extractall(self.folder)
        self.plate = pd.read_csv(os.path.join("plate_templates", "checkerboard.csv"))


    def test_read_tecan_xml_with_plate(self):
        df = curveball.ioutils.read_tecan_xml(os.path.join(self.folder, "*.xml"), 'OD', plate=self.plate)
        self.assertIsNotNone(df)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEquals(df.shape, (2016, 8))
        self.assertEquals(df.columns.tolist() , ['OD', 'Well', 'Row', 'Col', 'Time', 'Filename', 'Strain', 'Color'])


    def tearDown(self):
        for f in self.zipfile.filelist:
            os.remove(os.path.join(self.folder, f.filename))


    def test_read_tecan_xml(self):
        pass


if __name__ == '__main__':
    main()
