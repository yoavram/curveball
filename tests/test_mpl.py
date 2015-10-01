#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of curveball.
# https://github.com/yoavram/curveball

# Licensed under the MIT license:
# http://www.opensource.org/licenses/MIT-license
# Copyright (c) 2015, Yoav Ram <yoav@yoavram.com>
from unittest import TestCase, main
import matplotlib
import os


CI = os.environ.get('CI', 'false').lower() == 'true'


class MplTestCase(TestCase):
	def test_mpl(self):
		if CI:
			self.assertEquals(matplotlib.rcParams['backend'].lower(), 'agg')
		