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
from davitpy.utils import *
import datetime

import rbn_lib
import handling

#Specify output filename
outFile=''
rbnMap=''
fof2Map=''

#Specify spatial limits for links 
latMin=25
latMax=52  
lonMin=-130
lonMax=-65

#Map Properties 
#define map projection 
mapProj='cyl'
llcrnrlon=lonMin-5 
llcrnrlat=latMin-5
urcrnrlon=lonMax+5
urcrnrlat=latMax+5
#llcrnrlon=-130 
#llcrnrlat=25
#urcrnrlon=-65
#urcrnrlat=52 

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
#sTime = datetime.datetime(2015,6,28,01,12)
#eTime = datetime.datetime(2015,6,28,01,22)
#sTime = datetime.datetime(2015,6,28,01,16)
#eTime = datetime.datetime(2015,6,28,01,18)
sTime = datetime.datetime(2015,6,28,01,17, 00)
eTime = datetime.datetime(2015,6,28,01,17, 05)

map_sTime=sTime
map_eTime=eTime

#Read RBN data 
rbn_df  = rbn_lib.read_rbn(map_sTime,map_eTime,data_dir='data/rbn')
#import ipdb; ipdb.set_trace()

#Select Region
rbn_df2 = rbn_lib.rbn_region(rbn_df, latMin=latMin, latMax=latMax, lonMin=lonMin, lonMax=lonMax, constr_de=True, constr_dx=True)
#import ipdb; ipdb.set_trace()

#Evaluate each link
midLat=np.zeros([len(rbn_df2), 1])
#import ipdb; ipdb.set_trace()
midLon=np.zeros([len(rbn_df2), 1])
dist=np.zeros([len(rbn_df2), 1])
m_dist=np.zeros([len(rbn_df2), 1])
h=np.zeros([len(rbn_df2), 1])
theta=np.zeros([len(rbn_df2), 1])
fp=np.zeros([len(rbn_df2), 1])

#midLon=[]
#dist=[]
#m_dist=[]
for i in range(0, len(rbn_df2)-1): 
    #Isolate the ith link
    deLat=rbn_df2.de_lat.iloc[i]
    deLon=rbn_df2.de_lon.iloc[i]
    dxLat=rbn_df2.dx_lat.iloc[i]
    dxLon=rbn_df2.dx_lon.iloc[i]
    time=rbn_df2.date.iloc[i]
#    import ipdb; ipdb.set_trace()
    
    #Calculate the midpoint and the distance between the two stations
    midLat[i], midLon[i],dist[i],m_dist[i] =rbn_lib.path_mid(deLat, deLon, dxLat, dxLon)
#    import ipdb; ipdb.set_trace()

    #Find Kp, Ap, and SSN for that location and time
#    norm_sTime=sTime-sTime.hour-sTime.minute
    import ipdb; ipdb.set_trace()
    kp, ap, kpSum, apMean, ssn=rbn_lib.get_geomagInd(sTime, eTime)

    #Get hmF2 from the IRI using geomagnetic indices 
#    outf,oarr = iri.iri_sub(jf,jmag,alati,along,iyyyy,mmdd,dhour,heibeg,heiend,heistp,oarr)
    #outf and oarr are output and stored but right now they are changed each loop and not saved (Can change this in the future)
    h[i],outf,oarr=rbn_lib.get_hmF2(sTime=time, lat=midLat[i], lon=midLon[i],ssn=None)
    import ipdb; ipdb.set_trace()
    #test foF2
#    iri_fof2=np.sqrt(oarr[0]/(1.24e10))


    #Calculate theta (radians) from h=hmF2 and distance
    theta[i]=np.arctan(h[i]/m_dist[i])

    #Calculate foF2 from link frequency (MUF) and theta
    fp[i]=rbn_df.freq.iloc[i]*np.cos(theta[i])

#Save information in data frame
rbn_df2['midLat']=midLat
rbn_df2['midLon']=midLon
rbn_df2['link_dist']=dist
rbn_df2['m_dist']=m_dist
rbn_df2['hmF2']=h
rbn_df2['Elev_Ang(rad)']=theta
rbn_df2['Freq_plasma(kHz)']=fp



#Test plots
#Plot on map
fig = plt.figure(figsize=(8,4))
ax0  = fig.add_subplot(1,1,1)
m, fig=rbn_lib.rbn_map_plot(rbn_df2,legend=False,ax=ax0,tick_font_size=9,ncdxf=True, llcrnrlon=llcrnrlon, llcrnrlat=llcrnrlat, urcrnrlon=urcrnrlon, urcrnrlat=urcrnrlat)
#leg = rbn_lib.band_legend(fig,loc='center',bbox_to_anchor=[0.48,0.505],ncdxf=True,ncol=4)
filename='RBN_linkLimit_test4.jpg'
filepath    = os.path.join(output_path,filename)
fig.savefig(filepath,bbox_inches='tight')
fig.savefig(filepath[:-3]+'pdf',bbox_inches='tight')
plt.clf()
#import ipdb; ipdb.set_trace()

#Test of path_mid function
fig = plt.figure(figsize=(8,4))
ax0  = fig.add_subplot(1,1,1)
#df=rbn_df2
#df=rbn_df2.head(1)
df=rbn_df2.head(15)
df=rbn_df2.tail(15)
#j=(len(rbn_df2)-1)/2
#15)
j=(len(rbn_df2)-1)/2+4
k=j+1
#import ipdb; ipdb.set_trace()
df=rbn_df2.iloc[j:k]
import ipdb; ipdb.set_trace()
m, fig=rbn_lib.rbn_map_plot(df,legend=False,ax=ax0,tick_font_size=9,ncdxf=True, llcrnrlon=llcrnrlon, llcrnrlat=llcrnrlat, urcrnrlon=urcrnrlon, urcrnrlat=urcrnrlat)
color='m'
for i in range(0, len(df)): 
    #Isolate the ith link
    deLat=df.de_lat.iloc[i]
    deLon=df.de_lon.iloc[i]
    midLat=df.midLat.iloc[i]
    midLon=df.midLon.iloc[i]
    line, = m.drawgreatcircle(deLon,deLat, midLon, midLat, color=color)
    midpoint    = m.scatter(midLon, midLat,color='r',marker='o',s=2,zorder=100)

filename='RBN_linkMidpoint_test4.jpg'
filepath    = os.path.join(output_path,filename)
fig.savefig(filepath,bbox_inches='tight')
fig.savefig(filepath[:-3]+'pdf',bbox_inches='tight')
plt.clf()
import ipdb; ipdb.set_trace()
