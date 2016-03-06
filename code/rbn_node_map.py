#!/usr/bin/env python

import sys
import os

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.markers as mmarkers


import numpy as np
import pandas as pd

from davitpy import gme
from davitpy.utils import *
import datetime

import rbn_lib
import handling


#Specify regional/spatial limits for links 
latMin=25
latMax=52  
lonMin=-130
lonMax=-65

#2014 ARRL CW SS
sTime = datetime.datetime(2014,11,2,01,00, 00)
eTime = datetime.datetime(2014,11,2,03,00, 00)
eTime = datetime.datetime(2014,11,2,04,00, 00)
contest="cwSS"

#Set delta time
deltaTime=datetime.timedelta(minutes=15)
map_sTime=sTime
map_eTime=map_sTime+deltaTime

#File path for output file
#File with list of stations
data_dir=os.path.join('data','rbn')
#if world:
    #p_filename = 'rbn_station_list'+sTime.strftime('%Y%m%d%H%M-')+eTime.strftime('%Y%m%d%H%M.p')
#else:
p_filename = 'rbn_station_US_list'+sTime.strftime('%Y%m%d%H%M-')+eTime.strftime('%Y%m%d%H%M.p')
p_filepath = os.path.join(data_dir,p_filename)

#Map file
output_path = os.path.join('output','rbn','station_map')
try: 
    os.makedirs(output_path)
except:
    pass 

filename = 'rbn_US_station_map'+sTime.strftime('%Y%m%d%H%M-')+eTime.strftime('%Y%m%d%H%M.png')
#filename = 'rbn_station_map'+sTime.strftime('%Y%m%d%H%M-')+eTime.strftime('%Y%m%d%H%M.png')
filepath=os.path.join(output_path, filename)

## for downloading data in 1 hour increments 
#pickle_sTime=sTime
##pickle_eTime=pickle_sTime+datetime.timedelta(hours=1)
#pickle_eTime=pickle_sTime+datetime.timedelta(minutes=15)
#kk=0
#
#import ipdb; ipdb.set_trace()
##Read RBN data into pickle files
#while pickle_sTime<eTime:
#    print "Processing RBN Data for Interval #"+str(kk)
##    rbn_df  = rbn_lib.read_rbn_std(pickle_sTime,pickle_eTime,data_dir='data/rbn')
#    rbn_df  = rbn_lib.read_rbn_std(pickle_sTime,pickle_eTime,data_dir='data/rbn')
#    import ipdb; ipdb.set_trace()
#    pickle_sTime=pickle_eTime
##    import ipdb; ipdb.set_trace()
##    pickle_eTime=pickle_sTime+datetime.timedelta(hours=1)
#    pickle_eTime=pickle_sTime+datetime.timedelta(minutes=15)
#    kk=kk+1
#    import ipdb; ipdb.set_trace()

#    #Get RBN data 
nodes=[]
while map_sTime<eTime:
    rbn_df  = rbn_lib.read_rbn_std(map_sTime,map_eTime,data_dir='data/rbn')
#    rbn_df=rbn_lib.read_rbn(sTime, eTime,data_dir='data/rbn')  
#    if len(nodes)==0:
    call=rbn_df['callsign'].unique()
    for de in call:
        nodes.append(de)
    #Increment time 
    map_sTime=map_eTime
    map_eTime=map_sTime+deltaTime
#    import ipdb; ipdb.set_trace()
#    else:
#        for call in nodes: 
#            if df[df['callsign'=call]]==0:
#                nodes.append(call)

#Re-sort Callsigns 
df=pd.DataFrame({'callsign': nodes})
rbn_nodes=df['callsign'].unique()

#Save information in dataframe
del df
df=pd.DataFrame({'callsign': rbn_nodes})

df=rbn_lib.station_loc(df,data_dir='data/rbn')
df = rbn_lib.rbn_region(df, latMin=latMin, latMax=latMax, lonMin=lonMin, lonMax=lonMax, constr_de=True, constr_dx=False)

import ipdb; ipdb.set_trace()
df.to_pickle(p_filepath)

#Map RBN Nodes
m,fig=rbn_lib.rbn_map_node(df, sTime, eTime, m=None,ax=None, llcrnrlon=lonMin, llcrnrlat=latMin, urcrnrlon=lonMax, urcrnrlat=latMax, plot_paths=False, eclipse=True)
#m,fig=rbn_lib.rbn_map_node(df, sTime, eTime, m=None,ax=None, plot_paths=False)
fig.savefig(filepath,bbox_inches='tight')
