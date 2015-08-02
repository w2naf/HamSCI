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
rbnMap=''
fof2Map=''

#create output directory if none exists
#output_dir='output'
output_path = os.path.join('output','rbn','foF2')
#handling.prepare_output_dirs({0:output_dir},clear_output_dirs=True)
try: 
    os.makedirs(output_path)
except:
    pass 

#Specify start and end times
#sTime = datetime.datetime(2015,9,10)
#eTime = datetime.datetime(2015,9,15)
sTime = datetime.datetime(2015,6,28,01,12)
eTime = datetime.datetime(2015,6,28,01,22)

#Read RBN data 
rbn_df  = rbn_lib.read_rbn(map_sTime,map_eTime,data_dir='data/rbn')

#Select Region

#Evaluate each link
for :
    #Calculate the midpoint and the distance between the two stations

    #Find Kp, Ap, and SSN for that location and time

    #Get hmF2 from the IRI using geomagnetic indices 

    #Get 
