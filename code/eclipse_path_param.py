#!/usr/bin/env python
#Determine distance parameters of eclipse path from the data used to make my maps for the proposal and AGU

import sys
sys.path.append('/data/mypython')
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
import eclipse_lib

#get eclipse path coordinates into dataframes
df_cl=eclipse_lib.eclipse_get_path('ds_CL.csv')                                                                                                                        
df_nl=eclipse_lib.eclipse_get_path('ds_NL.csv')                                                                                                             
df_sl=eclipse_lib.eclipse_get_path('ds_SL.csv')                                                                                                             

#Get Latitudinal Width of path along eclipse path
df_width=df_nl.eLat-df_sl.eLat  
width_max=df_width.max()
width_min=df_width.min()                                                                                                                                              

#Get Longitudinal Width of path along eclipse path
lon_width=df_cl.eLon.max()-df_cl.eLon.min()                                                                                                                           
#df_nl.eLon.max()-df_nl.eLon.min()                                                                                                                           
#df_sl.eLon.max()-df_sl.eLon.min()                                                                                                                           

#df_nl.eLat.iloc[1]                                                                                                                                          
#df_sl.eLat.iloc[1]                                                                                                                                          
#df_nl.eLat.iloc[1]-df_sl.eLat.iloc[1]
#df_width.iloc[1]


