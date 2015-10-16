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


#Specify frequncies to evaluate at
freq1=14000
freq2=7000

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
eTime = datetime.datetime(2015,6,28,03,00, 00)
#map_sTime=sTime+datetime.timedelta(minutes=15)
#map_eTime=map_sTime+datetime.timedelta(minutes=15)

#Specify output filename
#csvfname='rbn_wal_'+map_sTime.strftime('%H%M - ')+'-'+map_eTime.strftime('%H%M UT')
#outfile=os.path.join(output_path,csvfname)
rbnMap='RBN_WAL_'
graphfile='RBN_WAL_count_'+sTime.strftime('%H%M - ')+'-'+eTime.strftime('%H%M UT')
graphfile1='RBN_WAL_'+sTime.strftime('%H%M - ')+'-'+eTime.strftime('%H%M UT')
#fof2Map=''

fig = plt.figure(figsize=(8,4))
#fig         = plt.figure(figsize=(nx_plots*xsize,ny_plots*ysize)) # Create figure with the appropriate size.
#Generate Maps
subplot_nr  = 0 # Counter for the subplot
letters = 'abcdef'

good_count  = 0
total_count = 0
kk=0
fig_inx=1
map_sTime=sTime
map_eTime=map_sTime+datetime.timedelta(minutes=15)

i=0
hv=[]
hvB=[]
fc=[]
fc2=[]
fcB=[]
f1=[]
f2=[]
d1=[]
d2=[]
count=[0,0]
time=[]
count1=[]
count2=[]
output=[0,0,0,0,0,0]

#for kk,map_sTime in enumerate(map_times):
while map_sTime<eTime:
    csvfname='rbn_wal_'+map_sTime.strftime('%H%M - ')+'-'+map_eTime.strftime('%H%M UT')
    outfile=os.path.join(output_path,csvfname)
    kk= kk + 1
    time.append(map_sTime)

    #If the maximum number of plots has been placed on the figure then make a new figure
    if kk>4:
        filename=rbnMap+str(fig_inx)+'_a.jpg'
        filepath    = os.path.join(output_path,filename)
        fig.savefig(filepath,bbox_inches='tight')
        fig.savefig(filepath[:-3]+'pdf',bbox_inches='tight')
        plt.clf()
        kk=1
        fig = plt.figure(figsize=(8,4))
        fig_inx=fig_inx+1

    #add subplot
    ax0     = fig.add_subplot(2,2,kk)
    #Read RBN data 
    rbn_df  = rbn_lib.read_rbn(map_sTime,map_eTime,data_dir='data/rbn')
#    import ipdb; ipdb.set_trace()

    #Select Region
    rbn_df2 = rbn_lib.rbn_region(rbn_df, latMin=latMin, latMax=latMax, lonMin=lonMin, lonMax=lonMax, constr_de=True, constr_dx=True)
    #import ipdb; ipdb.set_trace()
    ##Limit links to those with a midpoint within the radius of the isond
    #rbn_links=rbn_df2[rbn_df2.dist<=radius]
    rbn_df2, rbn_links=rbn_lib.getLinks(rbn_df2,isond,radius) 
#    import ipdb; ipdb.set_trace()
    #Find average frequency and distance/band
#    df1,df2,count1[i], count2[i], f1[i], f2[i], d1[i], d2[i]=rbn_lib.band_averages(rbn_links, freq1, freq2) 
#    df1,df2,count1[i], count2[i], f1[i], f2[i], d1[i], d2[i]=rbn_lib.band_averages(rbn_links, freq1, freq2) 
    df1,df2, count=rbn_lib.band_averages(rbn_links, freq1, freq2) 

    #Save averages in arrays
    count1.append(count[0])
    count2.append(count[1])
    f1.append(df1.freq)
    f2.append(df2.freq)
    d1.append(df1.dist)
    d2.append(df2.dist)

    #See if have enough information to solve for critical frequency
#    if df1.freq.isempty() or df2.freq.isempty():
#    if f1==0 or f2==0:
    if df1.freq==0 or df2.freq==0:
        fc.append('NA')
        hv.append('NA')
    else:
        #Solve the critical frequency equation
#        hv.append(abs((np.sqrt((np.square(f2[i]*d1[i])-np.square(f1[i]*d2[i]))))/(np.square(f1[i])-np.square(f2[i])))/2)
#        height=abs((np.square(f2[i]*d1[i])-np.square(f1[i]*d2[i]))))/(np.square(f1[i])-np.square(f2[i])))/2
        numer=abs(np.square(f2[i]*d1[i])-np.square(f1[i]*d2[i]))
        den=abs(np.square(f1[i])-np.square(f2[i]))
        hv.append(np.sqrt(abs(np.square(f2[i]*d1[i])-np.square(f1[i]*d2[i]))/abs(np.square(f1[i])-np.square(f2[i])))/2)
        hvB.append(np.sqrt(numer/den)/2)
#        den=(np.square(f1[i])-np.square(f2[i]))
#        import ipdb; ipdb.set_trace()
        fc.append(np.sqrt(np.square(f1[i])/(1+np.square(d1[i]/(2*hv[i])))))
        fc2.append(np.sqrt(np.square(f2[i])/(1+np.square(d2[i]/(2*hv[i])))))
        fcB.append(np.sqrt(np.square(f1[i])/(1+np.square(d1[i]/(2*hvB[i])))))
#        import ipdb; ipdb.set_trace()
        i=i+1
#        h[i]=np.sqrt((np.square(df2.freq*df1.dist)-np.square(df1.freq*df2.dist))/(np.square(df1.freq)-np.square(df2.freq)))/2
#        import ipdb; ipdb.set_trace()
#        fc[i]=np.sqrt(np.square(df1.freq)/(1+(df1.dist/(2*h[i]))))
#        import ipdb; ipdb.set_trace()
#        hv.append(np.sqrt((np.square(df2.freq*df1.dist)-np.square(df1.freq*df2.dist))/(np.square(df1.freq)-np.square(df2.freq)))/2)
#        import ipdb; ipdb.set_trace()
#        fc.append(np.sqrt(np.square(df1.freq)/(1+(df1.dist/(2*h[i])))))
#        import ipdb; ipdb.set_trace()

    #Export df of `links to csv file
    rbn_links.to_csv(outfile, index=False)
    #Concatinate in a dataframe
    if kk==1:
        df_links=rbn_links
    else:
        df_links=pd.concat([df_links, rbn_links])

    #Plot on map
#    fig = plt.figure(figsize=(8,4))
#    ax0  = fig.add_subplot(1,1,1)
    m, fig=rbn_lib.rbn_map_plot(rbn_links,legend=True,ax=ax0,tick_font_size=9,ncdxf=True, llcrnrlon=llcrnrlon, llcrnrlat=llcrnrlat, urcrnrlon=urcrnrlon, urcrnrlat=urcrnrlat)
    midpoint    = m.scatter(rbn_links.midLon, rbn_links.midLat,color='m',marker='s',s=2,zorder=100)
    loc_isond    = m.scatter(isond[1],isond[0],color='k',marker='*',s=12,zorder=100)
    #leg = rbn_lib.band_legend(fig,loc='center',bbox_to_anchor=[0.48,0.505],ncdxf=True,ncol=4)
    map_sTime=map_eTime
    map_eTime=map_sTime+datetime.timedelta(minutes=15)
#    import ipdb; ipdb.set_trace()

#    if kk==4:
#        filename=rbnMap+str(fig_inx)+'_a.jpg'
#        filepath    = os.path.join(output_path,filename)
#        fig.savefig(filepath,bbox_inches='tight')
#        fig.savefig(filepath[:-3]+'pdf',bbox_inches='tight')
#        plt.clf()
#        kk=0
#        fig = plt.figure(figsize=(8,4))
#        fig_inx=fig_inx+1

df_full=pd.DataFrame({'date':time, 'count1': count1, 'count2': count2, 'd1': d1,'d2': d2,'f1': f1,'f2': f2,'fc1': fc,'fc2':fc2})
csvfname='info_rbn_wal_'+sTime.strftime('%H%M - ')+'-'+eTime.strftime('%H%M UT')
#Drop NaNs (QSOs without Lat/Lons)
df = df_full.dropna(subset=['d1', 'd2', 'f1', 'f2'])
import ipdb; ipdb.set_trace()
filename=rbnMap+str(fig_inx)+'_a.jpg'
filepath    = os.path.join(output_path,filename)
fig.savefig(filepath,bbox_inches='tight')
fig.savefig(filepath[:-3]+'pdf',bbox_inches='tight')
plt.clf()

#Plot Data and Calculated values
nx_plots=1
ny_plots=4
xsize=8
ysize=4
fig         = plt.figure(figsize=(nx_plots*xsize,ny_plots*ysize)) # Create figure with the appropriate size.
#plot 
#ax0     = fig.add_subplot(ny_plots,1)
#ax1     = fig.add_subplot(1,ny_plots,2)
#ax2     = fig.add_subplot(1,ny_plots,3)
#ax3     = fig.add_subplot(1,ny_plots,4)
fig, ((ax0),(ax1),(ax2),(ax3))=plt.subplots(4,1,sharex=True,sharey=False)
ax0.plot(df['date'], df['count1'], '-y',df['date'], df['count2'], '-g')
ax1.plot(df['date'], df['d1'], '-y',df['date'], df['d2'], '-g')
ax2.plot(df['date'], df['f1'], '-y',df['date'], df['f2'], '-g')
ax3.plot(df['date'], df['fc1'], '-y',df['date'], df['fc2'], '-g')

fig.tight_layout()
filepath    = os.path.join(output_path,graphfile1)
fig.savefig(filepath)
fig.savefig(filepath[:-3]+'pdf',bbox_inches='tight')
plt.clf()

#Make count plot
fig2, ax1, ax2, ax3=rbn_lib.count_band(df_links, sTime, eTime,Inc_eTime=True,freq1=7000, freq2=14000, freq3=3000,dt=15,unit='minutes',xRot=False, ret_lim=False, rti_plot=False)
#fig.tight_layout()
filepath    = os.path.join(output_path,graphfile)
fig2.savefig(filepath)
#fig.savefig(filepath[:-3]+'pdf',bbox_inches='tight')
plt.clf()

##Critical Frequency calculation
#
##Find average distance of links
#
##Find average frequency of links on each band
#
##For 20 and 40m 


##Read RBN data 
#rbn_df  = rbn_lib.read_rbn(map_sTime,map_eTime,data_dir='data/rbn')
##import ipdb; ipdb.set_trace()
#
##Select Region
#rbn_df2 = rbn_lib.rbn_region(rbn_df, latMin=latMin, latMax=latMax, lonMin=lonMin, lonMax=lonMax, constr_de=True, constr_dx=True)
##import ipdb; ipdb.set_trace()

#Evaluate each link
#midLat=np.zeros([len(rbn_df2), 1])
##import ipdb; ipdb.set_trace()
#midLon=np.zeros([len(rbn_df2), 1])
#l_dist=np.zeros([len(rbn_df2), 1])
#m_dist=np.zeros([len(rbn_df2), 1])
#dist=np.zeros([len(rbn_df2), 1])
#h=np.zeros([len(rbn_df2), 1])
#theta=np.zeros([len(rbn_df2), 1])
#fp=np.zeros([len(rbn_df2), 1])
#
#for i in range(0, len(rbn_df2)): 
#    #Isolate the ith link
#    deLat=rbn_df2.de_lat.iloc[i]
#    deLon=rbn_df2.de_lon.iloc[i]
#    dxLat=rbn_df2.dx_lat.iloc[i]
#    dxLon=rbn_df2.dx_lon.iloc[i]
#    time=rbn_df2.date.iloc[i]
##    import ipdb; ipdb.set_trace()
#    
#    #Calculate the midpoint and the distance between the two stations
#    midLat[i], midLon[i],l_dist[i],m_dist[i] =rbn_lib.path_mid(deLat, deLon, dxLat, dxLon)
#    #Convert l_dist and m_dist to km
#    l_dist[i]=(l_dist[i])/1e3
#    m_dist[i]=(m_dist[i])/1e3
#
#    #Calculate the distance of the midpoint from the ionosonde/center of the reference area
#    dist[i]=rbn_lib.greatCircleKm(isond[0],isond[1], midLat[i],midLon[i])
##    import ipdb; ipdb.set_trace()
#
#
##Save information in data frame
#rbn_df2['midLat']=midLat
#rbn_df2['midLon']=midLon
#rbn_df2['link_dist']=l_dist
#rbn_df2['m_dist']=m_dist
#rbn_df2['dist']=dist
##Plasma Frequency in kHz
##rbn_df2['Freq_plasma']=fp
##rbn_df2['foP']=fp
#import ipdb; ipdb.set_trace()
#
###Limit links to those with a midpoint within the radius of the isond
##rbn_links=rbn_df2[rbn_df2.dist<=radius]
#rbn_df2, rbn_links=rbn_lib.getLinks(rbn_df2,isond,radius) 
#
##Export df of `links to csv file
#rbn_links.to_csv(outfile, index=False)
#
##Plot on map
#fig = plt.figure(figsize=(8,4))
#ax0  = fig.add_subplot(1,1,1)
#m, fig=rbn_lib.rbn_map_plot(rbn_links,legend=True,ax=ax0,tick_font_size=9,ncdxf=True, llcrnrlon=llcrnrlon, llcrnrlat=llcrnrlat, urcrnrlon=urcrnrlon, urcrnrlat=urcrnrlat)
#midpoint    = m.scatter(rbn_links.midLon, rbn_links.midLat,color='m',marker='s',s=2,zorder=100)
#loc_isond    = m.scatter(isond[1],isond[0],color='k',marker='*',s=12,zorder=100)
##leg = rbn_lib.band_legend(fig,loc='center',bbox_to_anchor=[0.48,0.505],ncdxf=True,ncol=4)
#filename=rbnMap
#filepath    = os.path.join(output_path,filename)
#fig.savefig(filepath,bbox_inches='tight')
#fig.savefig(filepath[:-3]+'pdf',bbox_inches='tight')
#plt.clf()
#
#filename=rbnMap+'1a.jpg'
#filepath    = os.path.join(output_path,filename)

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
