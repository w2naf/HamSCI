#!/usr/bin/env python
#This code is intended to download the RBN data from the ePOP Satellite pass on Field Day 2015
#and find the RBN recievers that heard the callsigns recorded by ePOP during from 0116-0118UT on 28 June 2015

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

#ePOP data input file
inPath="data/epop"
fname="Callsigns_7MHz.csv"
infile = os.path.join(inPath,fname)
print infile
#create output directory if none exists
output_dir='output/epop'
#try: 
#    os.makedirs(output_dir)
#except:
#    pass 

#output_path = os.path.join('output','firori')
#handling.prepare_output_dirs({0:output_path},clear_output_dirs=True)

#Time of ePOP pass
sTime = datetime.datetime(2015,6,28,01,16)
eTime = datetime.datetime(2015,6,28,01,18)

#Get RBN data 
rbn_df=rbn_lib.read_rbn(sTime, eTime,data_dir='data/rbn')

#Get epop callsign data
epop_df=pd.DataFrame.from_csv(infile)
import ipdb; ipdb.set_trace()

i=1
while i<epop_df.Index(:end):
    callsign=epop_df.Call[:1]
    print callsign
    import ipdb; ipdb.set_trace()
    df_temp=pd.DataFrame(callsign, ['Callsign'])
    for i in range(0,len(rbn_df)-1):
        if rbn_df.callsign[i]==callsign:
            df.=rbn_df=



#Interval
#df_temp=
#Export to text file
df.to_csv(outfile, index=False)
