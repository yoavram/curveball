#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of curveball.
# https://github.com/yoavram/curveball

# Licensed under the MIT license:
# http://www.opensource.org/licenses/MIT-license
# Copyright (c) 2015, Yoav Ram <yoavram+github@gmail.com>

from unittest import TestCase, main
import os.path

import curveball
import pandas as pd

class IOTestCase(TestCase):
    def setUp(self):
        self.filename = 'data/yoavram/Tecan_210115.xlsx'
        if not os.path.exists(self.filename):
            raise IOError("Data file not found: %s" % self.filename)
        self.plate = pd.read_csv("plate_templates/G-RG-R.csv")


    def test_read_tecan_xlsx_OD(self):
        df = curveball.io.read_tecan_xlsx(self.filename, 'OD')
        self.assertIsNotNone(df)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEquals(df.shape, (8448, 9))
        self.assertEquals(df.columns.tolist() , ['Time', u'Temp. [\xb0C]', 'Cycle Nr.', 'Well', 'OD', 'Row', 'Col', 'Strain', 'Color'])


    def test_read_tecan_xlsx_full(self):
        df = curveball.io.read_tecan_xlsx(self.filename, ('OD', 'Green', 'Red'))
        self.assertIsNotNone(df)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEquals(df.shape, (8448, 15))
        self.assertEquals(df.columns.tolist() , ['Time_OD', u'Temp. [\xb0C]_OD', 'Cycle Nr.', 'Well', 'OD', 'Row', 'Col', 'Strain', 'Color', 'Time_Green', u'Temp. [\xb0C]_Green', 'Green', 'Time', u'Temp. [\xb0C]', 'Red'])


    def test_read_tecan_xlsx_12hrs(self):
        df = curveball.io.read_tecan_xlsx(self.filename, 'OD', max_time=12)
        self.assertIsNotNone(df)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEquals(df.shape, (4992, 9))
        self.assertEquals(df.columns.tolist() , ['Time', u'Temp. [\xb0C]', 'Cycle Nr.', 'Well', 'OD', 'Row', 'Col', 'Strain', 'Color'])



    def test_read_tecan_xlsx_with_plate(self):
        df = curveball.io.read_tecan_xlsx(self.filename, 'OD', plate=self.plate)
        self.assertIsNotNone(df)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEquals(df.shape, (8448, 9))
        self.assertEquals(df.columns.tolist() , ['Time', u'Temp. [\xb0C]', 'Cycle Nr.', 'Well', 'OD', 'Row', 'Col', 'Strain', 'Color'])


if __name__ == '__main__':
    main()
