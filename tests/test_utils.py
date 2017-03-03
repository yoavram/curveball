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
import pkg_resources
import site

import curveball


class ConfigTestCase(TestCase):
    def setUp(self):
        CFG_FILENAME = os.environ.get('CURVEBALL_CFG_NAME', 'curveball.json')
        self.cfg_fname = tempfile.mktemp()
        shutil.copy(os.path.join(site.getuserbase(), CFG_FILENAME), self.cfg_fname)

    def test_read_config(self):
        cfg = curveball.utils.read_config(fname=self.cfg_fname)
        self.assertTrue(isinstance(cfg, dict))
        self.assertNotEmpty(cfg)

    def test_write_config(self):
    	org = {'a': 1}
    	curveball.utils.write_config(org, fname=self.cfg_fname)
    	cfg = curveball.utils.read_config(fname=self.cfg_fname)
    	self.assertEquals(cfg, org)