#!/usr/bin/env python
#Code to calcuate and make contour plots of foF2 from RBN data over a specified region and time

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

#Specify output filename
outFile=''
#create output directory if none exists
output_dir='output/epop'
output_path = os.path.join('output','rbn')
#handling.prepare_output_dirs({0:output_dir},clear_output_dirs=True)
#try: 
#    os.makedirs(output_dir)
#except:
#    pass 
