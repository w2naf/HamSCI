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


##Constants
##Radius of Earth
#r=6371.

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
output_path = os.path.join('output','rbn','foF2_curve')
#handling.prepare_output_dirs({0:output_dir},clear_output_dirs=True)
try: 
    os.makedirs(output_path)
except:
    pass 

#Specify start and end times
#Test of Code 2014 CW SS 
sTime = datetime.datetime(2014,11,1,23,00, 00)
eTime = datetime.datetime(2014,11,1,23,14, 00)
#sTime = datetime.datetime(2014,11,1,20,00, 00)
#eTime = datetime.datetime(2014,11,1,20,14, 00)
##First test of code on a quiet day 
#sTime = datetime.datetime(2016,7,6,23,00, 00)
#eTime = datetime.datetime(2016,7,6,23,14, 00)
#sTime = fdatetime.datetime(2016,7,7,01,00, 00)
#eTime = datetime.datetime(2016,7,7,01,14, 00)

map_sTime=sTime
map_eTime=eTime

#Specify output filename
#outFile=''
#rbnMap=''
#fof2Map=''
filename='RBN_foF2_map_'+map_sTime.strftime('%Y%b%d%H%M-')+map_eTime.strftime('%Y%b%d%H%M')+'.png'
filename5='RBN_midpt_map_'+map_sTime.strftime('%Y%b%d%H%M-')+map_eTime.strftime('%Y%b%d%H%M')+'.png'
#filename3='RBN_foF2_colormap_'+map_sTime.strftime('%Y%b%d%H%M-')+map_eTime.strftime('%Y%b%d%H%M')+'.png'
filename2='RBN_foF2_hist_'+map_sTime.strftime('%Y%b%d%H%M-')+map_eTime.strftime('%Y%b%d%H%M')+'.png'
filename4='RBN_link_freq_hist_'+map_sTime.strftime('%Y%b%d%H%M-')+map_eTime.strftime('%Y%b%d%H%M')+'.png'
filename6='Scatter_foF2_vs_theta_'+map_sTime.strftime('%Y%b%d%H%M-')+map_eTime.strftime('%Y%b%d%H%M')+'.png'
filename7='Scatter_foF2_vs_freq_'+map_sTime.strftime('%Y%b%d%H%M-')+map_eTime.strftime('%Y%b%d%H%M')+'.png'
filename8='Scatter_foF2_vs_LinkDist_'+map_sTime.strftime('%Y%b%d%H%M-')+map_eTime.strftime('%Y%b%d%H%M')+'.png'
csvfname='rbn_foF2_file_'+map_sTime.strftime('%H%M - ')+'-'+map_eTime.strftime('%H%M UT')
filepath    = os.path.join(output_path,filename)
#filepath2    = os.path.join(output_path,filename2)
filepath2    = os.path.join(output_path,'clip_foF2',filename2)
#filepath3    = os.path.join(output_path,filename3)
#filepath4    = os.path.join(output_path,filename4)
filepath4    = os.path.join(output_path, 'clip_foF2',filename4)
filepath5    = os.path.join(output_path,filename5)
filepath6    = os.path.join(output_path,filename6)
filepath7    = os.path.join(output_path,filename7)
filepath8    = os.path.join(output_path,filename8)
csvpath    = os.path.join(output_path,'csv_files',csvfname)

#Read RBN data 
rbn_df  = rbn_lib.read_rbn_std(map_sTime,map_eTime,data_dir='data/rbn')
#rbn_df  = rbn_lib.read_rbn(map_sTime,map_eTime,data_dir='data/rbn')

#Get Geomagnetic Indicies Data
#NEED to add code to make sure that multiple days of data can be processed!
kp, ap, kpSum, apMean, ssn=rbn_lib.get_geomagInd(sTime, eTime)

#Select Region
rbn_df2 = rbn_lib.rbn_region(rbn_df, latMin=latMin, latMax=latMax, lonMin=lonMin, lonMax=lonMax, constr_de=True, constr_dx=True)
#rbn_df2=rbn_df

#Evaluate each link
#Setup numpy arrays for properties
midLat=np.zeros([len(rbn_df2), 1])
midLon=np.zeros([len(rbn_df2), 1])
dist=np.zeros([len(rbn_df2), 1])
m_dist=np.zeros([len(rbn_df2), 1])
h=np.zeros([len(rbn_df2), 1])
theta=np.zeros([len(rbn_df2), 1])
fp=np.zeros([len(rbn_df2), 1])
iri_fp=np.zeros([len(rbn_df2), 1])
tdate=[]
#phi=np.zeros([len(rbn_df2), 1])

extrema=[]

#for i in range(0, len(rbn_df2)-1): 
for i in range(0, len(rbn_df2)): 
    #Isolate the ith link
    deLat=rbn_df2.de_lat.iloc[i]
    deLon=rbn_df2.de_lon.iloc[i]
    dxLat=rbn_df2.dx_lat.iloc[i]
    dxLon=rbn_df2.dx_lon.iloc[i]
    time=rbn_df2.date.iloc[i]
    freq=rbn_df2.freq.iloc[i]

    #Save time in array
    tdate.append(time)

    #Calculate foF2 
#    fp[i],theta[i],h[i],phi[i],x[i],z[i] = rbn_lib.rbn_fof2(sTime, freq, deLat, deLon, dxLat,dxLon)
    fp[i],theta[i],h[i],iri_fp[i] = rbn_lib.rbn_fof2(sTime, freq, deLat, deLon, dxLat,dxLon)

   #Calculate the midpoint and the distance between the two stations
    midLat[i], midLon[i],dist[i],m_dist[i] =rbn_lib.path_mid(deLat, deLon, dxLat, dxLon)

    #Check if outside limits
    fp_band=np.floor(fp[i]/1000.)
    if fp_band<1 or 28<fp_band:
        extrema.append(i)
#
#    #Get parameters to calculate theta (the angle of reflection)
#    phi[i] = (greatCircleDist(deLat, deLon, dxLat, dxLon))/2
#    alpha=(np.pi+phi[i])/2
#    x[i] = r*np.sqrt(2*(1-np.cos(phi[i])))
#    h[i],outf,oarr=rbn_lib.get_hmF2(sTime=time, lat=midLat[i], lon=midLon[i],ssn=ssn)
#    z=np.sqrt(h^2+x^2-2*h*x*np.cos(alpha))
#    theta=np.arcsin((x/z)*np.sin(alpha)) 

#    #Find Kp, Ap, and SSN for that location and time
##    norm_sTime=sTime-sTime.hour-sTime.minute
#    #Calculate foF2 from link frequency (MUF) and theta
#    fp[i]=rbn_df.freq.iloc[i]*np.cos(theta[i])

#Save information in data frame
rbn_df2['midLat']=midLat
rbn_df2['midLon']=midLon
rbn_df2['link_dist']=dist
rbn_df2['m_dist']=m_dist
rbn_df2['hmF2']=h
#Incidence Angle in Radians
rbn_df2['Inc_Ang']=theta
#Plasma Frequency in kHz
rbn_df2['foP']=fp
rbn_df2['iri_foP']=iri_fp
import ipdb; ipdb.set_trace()
df=pd.concat([rbn_df2.freq.iloc[extrema],rbn_df2.foP.iloc[extrema],rbn_df2.midLat.iloc[extrema],rbn_df2.midLon.iloc[extrema],rbn_df2.link_dist.iloc[extrema],rbn_df2.iri_foP.iloc[extrema],rbn_df2.hmF2.iloc[extrema]],axis=1)
df['band']  = np.array((np.floor(df['freq']/1000.)),dtype=np.int)
df['foP_band']  = np.array((np.floor(df['foP']/1000.)),dtype=np.int)
df['iri_band']  = np.array((np.floor(df['iri_foP']/1000.)),dtype=np.int)
import ipdb; ipdb.set_trace()
#rbn_df2['ssn']=ssn*np.ones(len(fp),1)
#rbn_df2['kp']=kp*np.ones(len(fp),1)
#rbn_df2['ap']=ap*np.ones(len(fp),1)

#Export df of times, midpoints,and foF2 and hmF2 values to csv file (then delete temporary df)
#df_fof2=pd.DataFrame({'date':rbn_df2.date.values, 'mid_Lat':midLat,'mid_Lon':midLon,'fof2':fp, 'hmF2':h})
#df_fof2=pd.DataFrame({'date':tdate, 'mid_Lat':midLat,'mid_Lon':midLon,'fof2':fp, 'hmF2':h})

#df_fof2=pd.concat([rbn_df2.date,rbn_df2.midLat,rbn_df2.midLon,rbn_df2.foP,rbn_df2.link_dist,rbn_df2.iri_foP,rbn_df2.hmF2],axis=1)
#df_fof2=pd.concat([rbn_df2.date,rbn_df2.midLat,rbn_df2.midLon,rbn_df2.foP,rbn_df2.hmF2],axis=1)
df_fof2=pd.concat([rbn_df2.date,rbn_df2.midLat,rbn_df2.midLon,rbn_df2.foP],axis=1)

#df_fof2=rbn_df2['date']
#df_fof2['midLat']=midLat
#df_fof2['midLon']=midLon
#df_fof2['foF2']=fp
#df_fof2['hmF2']=h
#index=np.arange(0,len(fp))
#df_fof2=pd.DataFrame({'mid_Lat':midLat,'mid_Lon':midLon,'fof2':fp, 'hmF2':h}, index=index)
import ipdb; ipdb.set_trace()
df_fof2.to_csv(csvpath, index=False)
del df_fof2

##Group plasma frequencies by band/frequency Range
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
#Test plots

#Plot Data in different ways
#Add dataframe columns for plotting 
rbn_df2['foP_mhz']  = np.array((rbn_df2['foP']/1000.),dtype=np.float)
rbn_df2['freq_mhz']  = np.array((rbn_df2['freq']/1000.),dtype=np.float)
rbn_df2['dist_km']  = np.array((rbn_df2['link_dist']/1000.),dtype=np.float)
rbn_df2['d_theta']  = np.array((rbn_df2['Inc_Ang']*180./np.pi),dtype=np.float)


#Clip fof2 frequencies 
df_clip=rbn_df2[20<rbn_df2['freq_mhz']]

#Generate Graph of foF2 Values
#del fig
fig = plt.figure(figsize=(8,4))

## the histogram of the data
##freq=rbn_df2['Freq_plasma(kHz)']
##freq=rbn_df2['Freq_plasma']
#freq=rbn_df2['foP']
#num_bins=len(rbn_df2)-1
num_bins=len(fp)-1
##n, bins, patches = plt.hist(rbn_df2.Freq_plasma, num_bins, normed=1, facecolor='green', alpha=0.5)
#Plot QSO frequency 
fig = plt.figure(figsize=(8,4))
qso_freq=rbn_df2.freq.values
qso_freq_clip=df_clip.freq_mhz.values
n, bins, patches = plt.hist(qso_freq_clip, num_bins, facecolor='blue', alpha=0.5)
plt.xlabel('Link Frequency')
plt.ylabel('Counts')
plt.title('Histogram of Link Frequency from RBN (clipped foF2)')
fig.savefig(filepath4,bbox_inches='tight')
#Plot foF2
del fig
fig = plt.figure(figsize=(8,4))
fp_clip=df_clip.foP_mhz.values
n, bins, patches = plt.hist(fp_clip, num_bins, facecolor='green', alpha=0.5)
#n, bins, patches = plt.hist(fp, num_bins, normed=1, facecolor='green', alpha=0.5)
#n, bins, patches = plt.hist(fp, num_bins, normed=len(rbn_df2['foP']), facecolor='green', alpha=0.5)
# add a 'best fit' line
#y = mlab.normpdf(bins, mu, sigma)
#plt.plot(bins, y, 'r--')
plt.xlabel('foF2')
plt.ylabel('Counts')
plt.title('Histogram of Plasma Frequency from RBN(>20MHz)')
fig.savefig(filepath2,bbox_inches='tight')
#fig.savefig(filepath2[:-3]+'pdf',bbox_inches='tight')
plt.clf()
import ipdb; ipdb.set_trace()

#make scatter plot
del fig
#del ax0
#fig = plt.figure(figsize=(8,4))
fig = plt.figure()
ax0  = fig.add_subplot(1,1,1)
ax0,fig=rbn_lib.plot_band(rbn_df2, x_param='freq_mhz', y_param='foP_mhz', ax=ax0, fig=fig, title='foF2 vs Link Frequency', x_ax='Link Freq (MHz)', y_ax='foF2 (MHz)')
#fp_mhz=fp/1e3
#freq_mhz=rbn_df2.freq.values/1e3
#ax0=rbn_lib.color_band(freq_mhz, freq_mhz,fp_mhz, ax=ax0)
#ax0.scatter(freq_mhz, fp_mhz,color=color)
#ax0.set_title('foF2 vs Link Frequency')
#ax0.set_xlabel('Link Frequency (MHz)')
#ax0.set_ylabel('foF2 (MHz)')
fig.savefig(filepath7,bbox_inches='tight')

del fig
del ax0
#fig = plt.figure(figsize=(8,4))
fig = plt.figure()
ax0  = fig.add_subplot(1,1,1)

d_theta=180*theta/np.pi
import ipdb; ipdb.set_trace()
#ax0.plot(d_theta, fp_mhz, 'bo')
#ax0.plot(d_theta, fp_mhz)
#ax0=rbn_lib.color_band(freq_mhz,d_theta, fp_mhz, ax0)
ax0, fig=rbn_lib.plot_band(rbn_df2, x_param='d_theta', y_param='foP_mhz', ax=ax0, fig=fig, title='foF2 vs Theta',x_ax='Theta (degrees)', y_ax='foF2 (MHz)')
#ax0.set_title('foF2 vs Angle of Incidence')
#ax0.set_xlabel('Theta (degrees)')
#ax0.set_ylabel('foF2 (MHz)')
fig.savefig(filepath6,bbox_inches='tight')

del fig
del ax0
#fig = plt.figure(figsize=(8,4))
fig = plt.figure()
ax0  = fig.add_subplot(1,1,1)
#dist_km=dist/1e3
#ax0=rbn_lib.color_band(freq_mhz,dist_km, fp_mhz,ax=ax0) 
ax0, fig=rbn_lib.plot_band(rbn_df2, x_param='dist_km', y_param='foP_mhz', ax=ax0, fig=fig, title='foF2 vs Link Distance',x_ax='Likn Distance (km)', y_ax='foF2 (MHz)')
#ax0.set_title('foF2 vs Link Distance')
#ax0.set_xlabel('Link Distance (km)')
#ax0.set_ylabel('foF2 (MHz)')
fig.savefig(filepath8,bbox_inches='tight')
import ipdb; ipdb.set_trace()

#Plot foF2 values on map
#Working on new function to plot foF2 over the US
#rbn_lib.rbn_map_foF2()
del fig
del ax0
fig = plt.figure(figsize=(8,4))
ax0  = fig.add_subplot(1,1,1)
#m, fig=rbn_lib.rbn_map_foF2(df_40m,legend=False,ax=ax0,tick_font_size=9,ncdxf=True, llcrnrlon=llcrnrlon, llcrnrlat=llcrnrlat, urcrnrlon=urcrnrlon, urcrnrlat=urcrnrlat)

m, fig=rbn_lib.rbn_map_foF2(rbn_df2,legend=True,ssn=ssn, kp=kp,ax=ax0,tick_font_size=9,ncdxf=True, llcrnrlon=llcrnrlon, llcrnrlat=llcrnrlat, urcrnrlon=urcrnrlon, urcrnrlat=urcrnrlat)
fig.savefig(filepath,bbox_inches='tight')

#plot link midpoints 
del fig
del m
del ax0
fig = plt.figure(figsize=(8,4))
ax0  = fig.add_subplot(1,1,1)
m, fig=rbn_lib.rbn_map_midpt(rbn_df2,ax=ax0,legend=True,tick_font_size=9,ncdxf=False, llcrnrlon=llcrnrlon, llcrnrlat=llcrnrlat, urcrnrlon=urcrnrlon, urcrnrlat=urcrnrlat)
#m, fig=rbn_lib.rbn_map_midpt(rbn_df2,legend=True,ssn=ssn, kp=kp,ax=ax0,tick_font_size=9,ncdxf=True, llcrnrlon=llcrnrlon, llcrnrlat=llcrnrlat, urcrnrlon=urcrnrlon, urcrnrlat=urcrnrlat)
fig.savefig(filepath5,bbox_inches='tight')


#del fig
#fig = plt.figure(figsize=(8,4))
#ax0  = fig.add_subplot(1,1,1)
##Plot Colormap of foF2
#m, fig=rbn_lib.rbn_colormap_foF2(rbn_df2,legend=False,ssn=ssn, kp=kp,cmapName='Blues',ax=ax0,tick_font_size=9,ncdxf=True, llcrnrlon=llcrnrlon, llcrnrlat=llcrnrlat, urcrnrlon=urcrnrlon, urcrnrlat=urcrnrlat)
#fig.savefig(filepath3,bbox_inches='tight')
##fig.savefig(filepath[:-3]+'pdf',bbox_inches='tight')
#plt.clf()

#for i in range(0, len(df_40m)): 
#    midLat=df_40m.midLat.iloc[i]
#    midLon=df_40m.midLon.iloc[i]
##    midpoint    = m.scatter(midLon, midLat,color='r',marker='o',s=2,zorder=100)
#    fof2_pt    = m.scatter(midLon,midLat,color='r',marker='o',s=2,zorder=100)
#
#for i in range(0, len(df_80m)): 
#    midLat=df_80m.midLat.iloc[i]
#    midLon=df_80m.midLon.iloc[i]
##    midpoint    = m.scatter(midLon, midLat,color='r',marker='o',s=2,zorder=100)
#    fof2_pt    = m.scatter(midLon,midLat,color='b',marker='o',s=2,zorder=100)
#fig.savefig(filepath,bbox_inches='tight')
##fig.savefig(filepath[:-3]+'pdf',bbox_inches='tight')
#plt.clf()
import ipdb; ipdb.set_trace()

### the histogram of the data
###freq=rbn_df2['Freq_plasma(kHz)']
###freq=rbn_df2['Freq_plasma']
##freq=rbn_df2['foP']
##num_bins=len(rbn_df2)-1
#num_bins=len(fp)-1
###n, bins, patches = plt.hist(rbn_df2.Freq_plasma, num_bins, normed=1, facecolor='green', alpha=0.5)
##Plot QSO frequency 
#fig = plt.figure(figsize=(8,4))
#qso_freq=rbn_df2.freq.values
#n, bins, patches = plt.hist(qso_freq, num_bins, facecolor='blue', alpha=0.5)
#plt.xlabel('Link Frequency')
#plt.ylabel('Counts')
#plt.title('Histogram of Link Frequency from RBN')
#fig.savefig(filepath4,bbox_inches='tight')
##Plot foF2
#del fig
#fig = plt.figure(figsize=(8,4))
#n, bins, patches = plt.hist(fp, num_bins, facecolor='green', alpha=0.5)
##n, bins, patches = plt.hist(fp, num_bins, normed=1, facecolor='green', alpha=0.5)
##n, bins, patches = plt.hist(fp, num_bins, normed=len(rbn_df2['foP']), facecolor='green', alpha=0.5)
## add a 'best fit' line
##y = mlab.normpdf(bins, mu, sigma)
##plt.plot(bins, y, 'r--')
#plt.xlabel('foF2')
#plt.ylabel('Counts')
#plt.title('Histogram of Plasma Frequency from RBN')
#fig.savefig(filepath2,bbox_inches='tight')
##fig.savefig(filepath2[:-3]+'pdf',bbox_inches='tight')
#plt.clf()
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
###
###
###Plot on map
##fig = plt.figure(figsize=(8,4))
##ax0  = fig.add_subplot(1,1,1)
##m, fig=rbn_lib.rbn_map_plot(rbn_df2,legend=False,ax=ax0,tick_font_size=9,ncdxf=True, llcrnrlon=llcrnrlon, llcrnrlat=llcrnrlat, urcrnrlon=urcrnrlon, urcrnrlat=urcrnrlat)
###leg = rbn_lib.band_legend(fig,loc='center',bbox_to_anchor=[0.48,0.505],ncdxf=True,ncol=4)
##filename='RBN_linkLimit_test5.jpg'
##filepath    = os.path.join(output_path,filename)
##fig.savefig(filepath,bbox_inches='tight')
##fig.savefig(filepath[:-3]+'pdf',bbox_inches='tight')
##plt.clf()
###import ipdb; ipdb.set_trace()
##
###Test of path_mid function
##fig = plt.figure(figsize=(8,4))
##ax0  = fig.add_subplot(1,1,1)
###df=rbn_df2
###df=rbn_df2.head(1)
##df=rbn_df2.head(15)
##df=rbn_df2.tail(15)
###j=(len(rbn_df2)-1)/2
###15)
##j=(len(rbn_df2)-1)/2+4
##k=j+1
###import ipdb; ipdb.set_trace()
##df=rbn_df2.iloc[j:k]
##import ipdb; ipdb.set_trace()
##m, fig=rbn_lib.rbn_map_plot(df,legend=False,ax=ax0,tick_font_size=9,ncdxf=True, llcrnrlon=llcrnrlon, llcrnrlat=llcrnrlat, urcrnrlon=urcrnrlon, urcrnrlat=urcrnrlat)
##color='m'
##for i in range(0, len(df)): 
##    #Isolate the ith link
##    deLat=df.de_lat.iloc[i]
##    deLon=df.de_lon.iloc[i]
##    midLat=df.midLat.iloc[i]
##    midLon=df.midLon.iloc[i]
##    line, = m.drawgreatcircle(deLon,deLat, midLon, midLat, color=color)
##    midpoint    = m.scatter(midLon, midLat,color='r',marker='o',s=2,zorder=100)
##
##filename='RBN_linkMidpoint_test5.jpg'
##filepath    = os.path.join(output_path,filename)
##fig.savefig(filepath,bbox_inches='tight')
##fig.savefig(filepath[:-3]+'pdf',bbox_inches='tight')
##plt.clf()
#
import ipdb; ipdb.set_trace()
