#!/usr/bin/env python
#Code to calcuate and make contour plots of the critical frequency (fc) from RBN data over a specified region and time
#Based on foF2.py code

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


#Specify regional/spatial limits for links 
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

#Specify Ionosonde that the calculated critical frequencies will be compared to 
isond=[37.93, -75.47]
#Specify radius(km) for the area to evaluate over
radius=100

#create output directory if none exists
#output_dir='output'
output_path = os.path.join('output','rbn','foF2', 'ver2')
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
#Test of Code on Field Days at the same time for different years 
#sTime = datetime.datetime(2013,6,23,01,17, 00)
#eTime = datetime.datetime(2013,6,23,01,17, 05)
#sTime = datetime.datetime(2014,6,29,01,17, 00)
#eTime = datetime.datetime(2014,6,29,01,17, 05)
#sTime = datetime.datetime(2015,6,28,01,17, 00)
#eTime = datetime.datetime(2015,6,28,01,17, 05)
#Test of Code during the IARU at the same time 
#sTime = datetime.datetime(2015,7,12,01,17, 00)
#eTime = datetime.datetime(2015,7,12,01,22, 00)

sTime = datetime.datetime(2015,6,28,01,00, 00)
eTime = datetime.datetime(2015,6,28,02,00, 00)
map_sTime=sTime+datetime.timedelta(minutes=15)
map_eTime=map_sTime+datetime.timedelta(minutes=15)

#Specify output filename
outfile='rbn_wal_'+map_sTime.strftime('%H%M - ')+'-'+map_eTime.strftime('%H%M UT')
rbnMap='RBN_WAL_1a.png'
#fof2Map=''

#Read RBN data 
rbn_df  = rbn_lib.read_rbn(map_sTime,map_eTime,data_dir='data/rbn')
#import ipdb; ipdb.set_trace()

##Get Geomagnetic Indicies Data
##NEED to add code to make sure that multiple days of data can be processed!
#kp, ap, kpSum, apMean, ssn=rbn_lib.get_geomagInd(sTime, eTime)

#Select Region
rbn_df2 = rbn_lib.rbn_region(rbn_df, latMin=latMin, latMax=latMax, lonMin=lonMin, lonMax=lonMax, constr_de=True, constr_dx=True)
#import ipdb; ipdb.set_trace()

#Evaluate each link
midLat=np.zeros([len(rbn_df2), 1])
#import ipdb; ipdb.set_trace()
midLon=np.zeros([len(rbn_df2), 1])
l_dist=np.zeros([len(rbn_df2), 1])
m_dist=np.zeros([len(rbn_df2), 1])
dist=np.zeros([len(rbn_df2), 1])
h=np.zeros([len(rbn_df2), 1])
theta=np.zeros([len(rbn_df2), 1])
fp=np.zeros([len(rbn_df2), 1])

#midLon=[]
#dist=[]
#m_dist=[]
#for i in range(0, len(rbn_df2)-1): 
for i in range(0, len(rbn_df2)): 
    #Isolate the ith link
    deLat=rbn_df2.de_lat.iloc[i]
    deLon=rbn_df2.de_lon.iloc[i]
    dxLat=rbn_df2.dx_lat.iloc[i]
    dxLon=rbn_df2.dx_lon.iloc[i]
    time=rbn_df2.date.iloc[i]
#    import ipdb; ipdb.set_trace()
    
    #Calculate the midpoint and the distance between the two stations
    midLat[i], midLon[i],l_dist[i],m_dist[i] =rbn_lib.path_mid(deLat, deLon, dxLat, dxLon)
    #Convert l_dist and m_dist to km
    l_dist[i]=(l_dist[i])/1e3
    m_dist[i]=(m_dist[i])/1e3

    #Calculate the distance of the midpoint from the ionosonde/center of the reference area
    dist[i]=rbn_lib.greatCircleKm(isond[0],isond[1], midLat[i],midLon[i])
#    import ipdb; ipdb.set_trace()

#    #Find Kp, Ap, and SSN for that location and time
##    norm_sTime=sTime-sTime.hour-sTime.minute
#    import ipdb; ipdb.set_trace()

#    #Get hmF2 from the IRI using geomagnetic indices 
##    outf,oarr = iri.iri_sub(jf,jmag,alati,along,iyyyy,mmdd,dhour,heibeg,heiend,heistp,oarr)
#    #outf and oarr are output and stored but right now they are changed each loop and not saved (Can change this in the future)
#    h[i],outf,oarr=rbn_lib.get_hmF2(sTime=time, lat=midLat[i], lon=midLon[i],ssn=ssn)
##    import ipdb; ipdb.set_trace()
#    #test foF2
##    iri_fof2=np.sqrt(oarr[0]/(1.24e10))
#
#
#    #Calculate theta (radians) from h=hmF2 and distance
#    theta[i]=np.arctan(h[i]/m_dist[i])
#
#    #Calculate foF2 from link frequency (MUF) and theta
#    fp[i]=rbn_df.freq.iloc[i]*np.cos(theta[i])

#Save information in data frame
rbn_df2['midLat']=midLat
rbn_df2['midLon']=midLon
rbn_df2['link_dist']=l_dist
rbn_df2['m_dist']=m_dist
rbn_df2['dist']=dist
#rbn_df2['hmF2']=h
#rbn_df2['ssn']=ssn*np.ones(len(fp),1)
#rbn_df2['kp']=kp*np.ones(len(fp),1)
#rbn_df2['ap']=ap*np.ones(len(fp),1)
#Elevation Angle in Radians
#rbn_df2['Elev_Ang']=theta
#Plasma Frequency in kHz
#rbn_df2['Freq_plasma']=fp
#rbn_df2['foP']=fp
import ipdb; ipdb.set_trace()
#Limit links to those with a midpoint within the radius of the isond
rbn_links=rbn_df2[rbn_df2.dist<=radius]

#Export df of `links to csv file
rbn_links.to_csv(outfile, index=False)

#Plot on map
fig = plt.figure(figsize=(8,4))
ax0  = fig.add_subplot(1,1,1)
m, fig=rbn_lib.rbn_map_plot(rbn_links,legend=True,ax=ax0,tick_font_size=9,ncdxf=True, llcrnrlon=llcrnrlon, llcrnrlat=llcrnrlat, urcrnrlon=urcrnrlon, urcrnrlat=urcrnrlat)
midpoint    = m.scatter(rbn_links.midLon, rbn_links.midLat,color='m',marker='s',s=2,zorder=100)
loc_isond    = m.scatter(isond[1],isond[0],color='k',marker='*',s=12,zorder=100)
#leg = rbn_lib.band_legend(fig,loc='center',bbox_to_anchor=[0.48,0.505],ncdxf=True,ncol=4)
filename=rbnMap
filepath    = os.path.join(output_path,filename)
fig.savefig(filepath,bbox_inches='tight')
fig.savefig(filepath[:-3]+'pdf',bbox_inches='tight')
plt.clf()


##Seperate by band
##freq1=
##df_freq1=rbn_df2[rbn_df2['foP']<=freq1+1000]
##df_freq1=df_freq1[df_freq1['foP']=>freq1+1000]
#
##40m
#df_40m=rbn_df2[rbn_df2['foP']<=8000]
#df_40m=df_40m[6000<=df_40m['foP']]
#df_80m=rbn_df2[rbn_df2['foP']<=4000]
#df_80m=df_80m[2000<=df_80m['foP']]
#
##for I in range(0,len(df2)-1):
##    if df2.freq.iloc[I]>(freq1-500) and df2.freq.iloc[I]<(freq1+500):
#
##df_temp=df_temp[df_temp['band']=='40m']
#
##Test plots
#
##Plot foF2 values on map
##Working on new function to plot foF2 over the US
##rbn_lib.rbn_map_foF2()
#fig = plt.figure(figsize=(8,4))
#ax0  = fig.add_subplot(1,1,1)
##m, fig=rbn_lib.rbn_map_foF2(df_40m,legend=False,ax=ax0,tick_font_size=9,ncdxf=True, llcrnrlon=llcrnrlon, llcrnrlat=llcrnrlat, urcrnrlon=urcrnrlon, urcrnrlat=urcrnrlat)
#m, fig=rbn_lib.rbn_map_foF2(rbn_df2,legend=True,ssn=ssn, kp=kp,ax=ax0,tick_font_size=9,ncdxf=True, llcrnrlon=llcrnrlon, llcrnrlat=llcrnrlat, urcrnrlon=urcrnrlon, urcrnrlat=urcrnrlat)
#
##for i in range(0, len(df_40m)): 
##    midLat=df_40m.midLat.iloc[i]
##    midLon=df_40m.midLon.iloc[i]
###    midpoint    = m.scatter(midLon, midLat,color='r',marker='o',s=2,zorder=100)
##    fof2_pt    = m.scatter(midLon,midLat,color='r',marker='o',s=2,zorder=100)
##
##for i in range(0, len(df_80m)): 
##    midLat=df_80m.midLat.iloc[i]
##    midLon=df_80m.midLon.iloc[i]
###    midpoint    = m.scatter(midLon, midLat,color='r',marker='o',s=2,zorder=100)
##    fof2_pt    = m.scatter(midLon,midLat,color='b',marker='o',s=2,zorder=100)
#filename='RBN_foF2_map_test4.jpg'
#filepath    = os.path.join(output_path,filename)
#fig.savefig(filepath,bbox_inches='tight')
#fig.savefig(filepath[:-3]+'pdf',bbox_inches='tight')
#plt.clf()
#
##Generate Graph of foF2 Values
#fig = plt.figure(figsize=(8,4))
#num_bins=len(rbn_df2)-1
### the histogram of the data
##freq=rbn_df2['Freq_plasma(kHz)']
##freq=rbn_df2['Freq_plasma']
#freq=rbn_df2['foP']
#import ipdb; ipdb.set_trace()
##n, bins, patches = plt.hist(rbn_df2.Freq_plasma, num_bins, normed=1, facecolor='green', alpha=0.5)
#n, bins, patches = plt.hist(fp, num_bins, normed=1, facecolor='green', alpha=0.5)
#import ipdb; ipdb.set_trace()
## add a 'best fit' line
##y = mlab.normpdf(bins, mu, sigma)
##plt.plot(bins, y, 'r--')
#plt.xlabel('foF2')
#plt.ylabel('Counts')
#plt.title('Histogram of Plasma Frequency from RBN')
#import ipdb; ipdb.set_trace()
#
###Graph foF2 values on 40m
##fig = plt.figure(figsize=(8,4))
##num_bins=len(rbn_df2)-1
#### the histogram of the data
###freq=rbn_df2['Freq_plasma(kHz)']
###freq=rbn_df2['Freq_plasma']
##index,row=df_40m.iterrows()
##freq=row['foP']
##import ipdb; ipdb.set_trace()
###n, bins, patches = plt.hist(rbn_df2.Freq_plasma, num_bins, normed=1, facecolor='green', alpha=0.5)
##n, bins, patches = plt.hist(freq, num_bins, normed=1, facecolor='green', alpha=0.5)
##import ipdb; ipdb.set_trace()
### add a 'best fit' line
###y = mlab.normpdf(bins, mu, sigma)
###plt.plot(bins, y, 'r--')
##plt.xlabel('foF2')
##plt.ylabel('Counts')
##plt.title('Histogram of foF2 from RBN')
##import ipdb; ipdb.set_trace()
### Tweak spacing to prevent clipping of ylabel
##plt.subplots_adjust(left=0.15)
##filename='RBN_foF2_40m_test1.jpg'
##filepath    = os.path.join(output_path,filename)
##fig.savefig(filepath,bbox_inches='tight')
##fig.savefig(filepath[:-3]+'pdf',bbox_inches='tight')
##
##
##Plot on map
#fig = plt.figure(figsize=(8,4))
#ax0  = fig.add_subplot(1,1,1)
#m, fig=rbn_lib.rbn_map_plot(rbn_df2,legend=False,ax=ax0,tick_font_size=9,ncdxf=True, llcrnrlon=llcrnrlon, llcrnrlat=llcrnrlat, urcrnrlon=urcrnrlon, urcrnrlat=urcrnrlat)
##leg = rbn_lib.band_legend(fig,loc='center',bbox_to_anchor=[0.48,0.505],ncdxf=True,ncol=4)
#filename='RBN_linkLimit_test5.jpg'
#filepath    = os.path.join(output_path,filename)
#fig.savefig(filepath,bbox_inches='tight')
#fig.savefig(filepath[:-3]+'pdf',bbox_inches='tight')
#plt.clf()
##import ipdb; ipdb.set_trace()
#
##Test of path_mid function
#fig = plt.figure(figsize=(8,4))
#ax0  = fig.add_subplot(1,1,1)
##df=rbn_df2
##df=rbn_df2.head(1)
#df=rbn_df2.head(15)
#df=rbn_df2.tail(15)
##j=(len(rbn_df2)-1)/2
##15)
#j=(len(rbn_df2)-1)/2+4
#k=j+1
##import ipdb; ipdb.set_trace()
#df=rbn_df2.iloc[j:k]
#import ipdb; ipdb.set_trace()
#m, fig=rbn_lib.rbn_map_plot(df,legend=False,ax=ax0,tick_font_size=9,ncdxf=True, llcrnrlon=llcrnrlon, llcrnrlat=llcrnrlat, urcrnrlon=urcrnrlon, urcrnrlat=urcrnrlat)
#color='m'
#for i in range(0, len(df)): 
#    #Isolate the ith link
#    deLat=df.de_lat.iloc[i]
#    deLon=df.de_lon.iloc[i]
#    midLat=df.midLat.iloc[i]
#    midLon=df.midLon.iloc[i]
#    line, = m.drawgreatcircle(deLon,deLat, midLon, midLat, color=color)
#    midpoint    = m.scatter(midLon, midLat,color='r',marker='o',s=2,zorder=100)
#
#filename='RBN_linkMidpoint_test5.jpg'
#filepath    = os.path.join(output_path,filename)
#fig.savefig(filepath,bbox_inches='tight')
#fig.savefig(filepath[:-3]+'pdf',bbox_inches='tight')
#plt.clf()
#
#import ipdb; ipdb.set_trace()
