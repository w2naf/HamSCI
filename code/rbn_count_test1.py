#!/usr/bin/env python
#This code is to get rbn data for a given date and time interval and plot the number of counts per unit time

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

#check this line! It may not be what I want to use
#output path=os.path.join('output', 'rbn_counts')
#data_dir=os.path.join('data','rbn')
import ipdb; ipdb.set_trace()

#specify times
sTime=datetime.datetime(2014,9,10,16,45)
eTime=datetime.datetime(2014, 9,10, 18, 30)

import ipdb; ipdb.set_trace()
#call function to get rbn data, find de_lat, de_lon, dx_lat, dx_lon for the data
rbn_df=rbn_lib.read_rbn(sTime, eTime, data_dir='data/rbn')

df1=rbn_df

count=np.ones((len(df1['date']),1))

import ipdb; ipdb.set_trace()
df1['count']=count

#group counts together by time

import ipdb; ipdb.set_trace()
