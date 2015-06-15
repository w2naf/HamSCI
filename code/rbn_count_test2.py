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
index=0
#specify times
sTime=datetime.datetime(2014,9,10,16,45)
eTime=datetime.datetime(2014, 9,10, 18, 30)
#specify time interval for spot counts
tDelta=datetime.timedelta(minutes=1)

import ipdb; ipdb.set_trace()
curr_time=sTime
times=[sTime]
while curr_time < eTime:
    curr_time+=tDelta
    times.append(curr_time)

import ipdb; ipdb.set_trace()

#group counts together by time
#write a while loop to get data for RBN
#call function to get rbn data, find de_lat, de_lon, dx_lat, dx_lon for the data

#index=0
#time=np.arange(sTime, eTime+tDelta, tDelta,dtype=str)
spots=np.zeros(len(times))

#import ipdb; ipdb.set_trace()
cTime=sTime
endTime=cTime
#rbn_df=rbn_lib.read_rbn(sTime, eTime, data_dir='data/rbn')
#import ipdb; ipdb.set_trace()

while cTime<eTime:
    endTime += tDelta
    if index>0:
        df1=df2

    df2=rbn_lib.read_rbn(cTime, endTime, data_dir='data/rbn')
    #store spot count for the given time interval in an array 
    spots[index]=len(df2)
    #concatenate rbn results to the the rbn_df
    if index>0:
        frames=[rbn_df, df2]
        rbn_df=pd.concat(frames)
    else:
        rbn_df=df2
        
    #spot_df['Count'][index:index+1]=len(rbn_df.loc[cTime:endTime])  
    #import ipdb; ipdb.set_trace()
    cTime=endTime
    index=index+1

spot_df=pd.DataFrame(data=times, columns=['date'])
spot_df['Count']=spots
#spot_df=pd.DataFrame(data=spots, columns=['Count'])
df1=rbn_df
import ipdb; ipdb.set_trace()

#count=np.ones((len(df1['date']),1))

#import ipdb; ipdb.set_trace()
#df1['count']=1

#group counts together by time

import ipdb; ipdb.set_trace()
