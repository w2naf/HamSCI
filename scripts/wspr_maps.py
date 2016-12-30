#!/usr/bin/env python
#This code is used to generate the RBN-GOES map for the Space Weather feature article.

import sys
import os
import datetime
import multiprocessing

import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt

import numpy as np
import pandas as pd

from hamsci import wspr_lib
from hamsci import handling


if __name__ == '__main__':
    multiproc   = False 
    sTime = datetime.datetime(2016,11,1)
    wspr_obj = wspr_lib.WsprObject(sTime) 
    wspr_obj.active.calc_reflection_points(reflection_type='miller2015')
    #For iPython
#    os.system('sudo python setup.py install')
