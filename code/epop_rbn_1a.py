#!/usr/bin/env python
#This code is used to generate the RBN-GOES map for the Space Weather feature article.

import sys
import os

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import numpy as np
import pandas as pd

from davitpy import gme
import datetime

import rbn_lib
import handling

output_path = os.path.join('output','firori')
handling.prepare_output_dirs({0:output_path},clear_output_dirs=True)

#sTime = datetime.datetime(2014,9,10)
#eTime = datetime.datetime(2014,9,15)
sTime = datetime.datetime(2015,3,11)
eTime = datetime.datetime(2015,3,12)
sat_nr = 15
