#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of curveball.
# https://github.com/yoavram/curveball

# Licensed under the MIT license:
# http://www.opensource.org/licenses/MIT-license
# Copyright (c) 2015, Yoav Ram <yoav@yoavram.com>

from __future__ import absolute_import
import warnings
# catch warning caused by seaborn v0.6.0
warnings.filterwarnings(action="ignore", message="axes.color_cycle is deprecated and replaced with axes.prop_cycle; please use the latter.")
# catch warning caused by seaborn v0.7
warnings.filterwarnings(action="ignore", message="The `IPython.html` package has been deprecated. You should import from `notebook` instead. `IPython.html.widgets` has moved to `ipywidgets`.")

__license__ = u'MIT'
from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

from . import ioutils
from . import plots
from . import models
from . import competitions
from . import likelihood

__citation_bibtex__ = u"""@article{
Ram2015,
author = {Ram, Yoav, Dellus-Gur, Eynat, Obolski, Uri, Bibi, Maayan, Berman, Judith, and Hadany, Lilach},
doi = {10.1101/022640},						
journal = {bioRxiv},
keywords = {experimental,fitness,mathematical model,selection coefficient},
month = jul,
title = {{Predicting competition results from growth curves}},
url = {http://biorxiv.org/lookup/doi/10.1101/022640},
year = {2015}
}"""
__citation__ = u'Ram, Yoav, Eynat Dellus-Gur, Uri Obolski, Maayan Bibi, Judith Berman, and Lilach Hadany. 2015. "Predicting Competition Results from Growth Curves." bioRxiv (July 23). doi:10.1101/022640.'

