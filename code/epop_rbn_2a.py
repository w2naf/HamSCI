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

i=0
#callsign=epop_df.Call[i]
#print callsign
#import ipdb; ipdb.set_trace()
#df=pd.DataFrame(callsign, ['Callsign'])

#for n in range(0,len(rbn_df)-1):
#    if rbn_df.callsign[n]==callsign:
#        if n=0:
#            df=rbn_df[n]
#        else:
#            df=concat[df, rbn_df[n]]
#        ['Lat']=rbn_df.de_lat[n]
i=0
for i in range(0,len(epop_df)-1):
#while i<len(epop_df):
    epopCall=epop_df.Call[i]
    print epopCall
#    import ipdb; ipdb.set_trace()
#    df_temp=pd.DataFrame(callsign, ['Callsign'])
#    for n in range(0,len(rbn_df)-1):
#        if rbn_df.callsign[n]==callsign:
#            df_temp=rbn_df.de_lat[n]
#            df=rbn
    for n in range(0,len(rbn_df)-1):
        if rbn_df.callsign.iloc[n]==epopCall:
            if i==0:
                df=rbn_df[n]
                import ipdb; ipdb.set_trace()
            else:
                df=concat[df, rbn_df[n]]
#                import ipdb; ipdb.set_trace()


#end of loop

#Interval
import ipdb; ipdb.set_trace()
#df_temp=
fname='rbn_and_epop_calls'
outfile=os.path.join(output_dir,fname)
#Export to text file
df.to_csv(outfile, index=False)
