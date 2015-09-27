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


CI = os.environ.get('CI', 'false').lower() == 'true'


class SimpleTestCase(TestCase):
	_multiprocess_can_split_ = True


    def setUp(self):
        pass


    def tearDown(self):
    	pass


    def test_read_tecan_xlsx_OD(self):
    	pass


if __name__ == '__main__':
    main()
