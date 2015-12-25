#!/usr/bin/env python
#Code to calcuate and make contour plots of the critical frequency (fc) from RBN data over a specified region and time
#Based on foF2.py code

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

#Map Properties 
#Specify whether to make a map
make_map=False
make_fullmap=True
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
#Arbitary point in South Britain
#isond=[52, -2]
#Specify radius(km) for the area to evaluate over
radius=150


#Specify frequncies to evaluate at
freq1=14000
#freq1=10000
#freq1=3500
#freq2=7000
freq2=21000

##create output directory if none exists
##output_dir='output'
#output_path = os.path.join('output','rbn','foF2', 'ver2', contest`)
##handling.prepare_output_dirs({0:output_dir},clear_output_dirs=True)
#try: 
#    os.makedirs(output_path)
#except:
#    pass 

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

##Field day
#sTime = datetime.datetime(2015,6,28,01,00, 00)
#eTime = datetime.datetime(2015,6,28,03,00, 00)
#contest="FD"
##2014 ARRL CW SS
#sTime = datetime.datetime(2014,11,2,01,00, 00)
#eTime = datetime.datetime(2014,11,2,03,00, 00)
#contest="cwSS"
#Validation Test Case
sTime = datetime.datetime(2014, 8,2,00,00, 00)
#sTime = datetime.datetime(2014, 8,2,00,15, 00)
#eTime = datetime.datetime(2014, 8,2,23,00, 00)
eTime = datetime.datetime(2014, 8,3,00,00, 00)
#sTime = datetime.datetime(2014, 8,2,00,40, 00)
##sTime = datetime.datetime(2014, 8,2,00,15, 00)
#eTime = datetime.datetime(2014, 8,2,01,40, 00)
contest="Code_Test"
##RSGB Eclipse QSO Party 2015
##Actual time 0800-1130
#sTime = datetime.datetime(2015,3,20,07,00, 00)
#eTime = datetime.datetime(2015,3,20,12,00, 00)
#contest="RSGB_EclipseQP"

#Set delta time
deltaTime=datetime.timedelta(minutes=15)
map_sTime=sTime
map_eTime=map_sTime+deltaTime

#map_sTime=sTime+datetime.timedelta(minutes=15)
#map_eTime=map_sTime+datetime.timedelta(minutes=15)

#Specify output filenames
csvfname='rbn_wal_'+map_sTime.strftime('%H%M - ')+'-'+map_eTime.strftime('%H%M UT')
#'+sTime.strftime('%Y')+'_'+contest+'_'
#rbnMap='RBN_WAL_2014_cwSS__'
rbnMap='RBN_WAL_'+sTime.strftime('%Y')+'_'+contest+'_'+str(freq1)+'MHz&'+str(freq2)+'MHz_'
graphfile='RBN_WAL_count_'+sTime.strftime('%Y')+'_'+contest+'_'+sTime.strftime('%H%M - ')+'-'+eTime.strftime('%H%M UT')+'_'+str(radius)+'km'+str(freq1)+'MHz&'+str(freq2)+'MHz'
graphfile1='RBN_WAL_'+sTime.strftime('%Y')+'_'+contest+'_'+sTime.strftime('%H%M - ')+'-'+eTime.strftime('%H%M UT')+'_'+str(radius)+'km'+str(freq1)+'MHz&'+str(freq2)+'MHz'+'.png'
graphfile2='RBN_WAL_freq_division_'+sTime.strftime('%Y')+'_'+contest+'_'+sTime.strftime('%H%M - ')+'-'+eTime.strftime('%H%M UT')+'_'+str(radius)+'km'+str(freq1)+'MHz&'+str(freq2)+'MHz'+'.png'
lgraph1='RBN_WAL_fc1_'+sTime.strftime('%Y')+'_'+contest+'_'+sTime.strftime('%H%M - ')+'-'+eTime.strftime('%H%M UT')+str(freq1)+'MHz&'+str(freq2)+'MHz'+'.png'
lgraph2='RBN_WAL_fc2_'+sTime.strftime('%Y')+'_'+contest+'_'+sTime.strftime('%H%M - ')+'-'+eTime.strftime('%H%M UT')+str(freq1)+'MHz&'+str(freq2)+'MHz'+'.png'
hgraph1='RBN_WAL_f1_'+sTime.strftime('%Y')+'_'+contest+'_'+sTime.strftime('%H%M - ')+'-'+eTime.strftime('%H%M UT')+str(freq1)+'MHz&'+str(freq2)+'MHz'+'.png'
hgraph2='RBN_WAL_f2_'+sTime.strftime('%Y')+'_'+contest+'_'+sTime.strftime('%H%M - ')+'-'+eTime.strftime('%H%M UT')+str(freq1)+'MHz&'+str(freq2)+'MHz'+'.png'
#hgraph2='RBN_WAL_f2_2014_cwSS'+sTime.strftime('%H%M - ')+'-'+eTime.strftime('%H%M UT')+'.png'
#fof2Map=''

#create output directory if none exists
#output_dir='output'
output_path = os.path.join('output','rbn','foF2', 'ver2', contest)
#handling.prepare_output_dirs({0:output_dir},clear_output_dirs=True)
try: 
    os.makedirs(output_path)
except:
    pass 

fig = plt.figure(figsize=(8,4))
#fig         = plt.figure(figsize=(nx_plots*xsize,ny_plots*ysize)) # Create figure with the appropriate size.
#Generate Maps
subplot_nr  = 0 # Counter for the subplot
letters = 'abcdef'

good_count  = 0
total_count = 0
kk=0
fig_inx=1

# for downloading data in 1 hour increments 
pickle_sTime=sTime
pickle_eTime=pickle_sTime+datetime.timedelta(hours=1)

##Read RBN data into pickle files
#while pickle_sTime<eTime:
#    print "Processing RBN Data for Interval #"+str(kk)
#    rbn_df  = rbn_lib.read_rbn_std(pickle_sTime,pickle_eTime,data_dir='data/rbn')
#    pickle_sTime=pickle_eTime
##    import ipdb; ipdb.set_trace()
#    pickle_eTime=pickle_sTime+datetime.timedelta(hours=1)
#import ipdb; ipdb.set_trace()

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
hmF2=[]
count=[0,0]
#time=[]
count1=[]
count2=[]
output=[0,0,0,0,0,0]

#rbn_df  = rbn_lib.read_rbn_std(sTime,eTime,data_dir='data/rbn')
#Compile Calculated values into a new dataframe
#df_full=pd.DataFrame({'date':time, 'count1': count1, 'count2': count2, 'd1': d1,'d2': d2,'f1': f1,'f2': f2,'hv': hv, 'fc1': fc,'fc2':fc2, 'hmF2':hmF2})
#df_full=pd.DataFrame({'date':[], 'count1':[] , 'count2':[] , 'd1':[] ,'d2':[] ,'f1':[] ,'f2':[] ,'hv':[] , 'fc1': [],'fc2':[], 'hmF2':[]})
#for kk,map_sTime in enumerate(map_times):
while map_sTime<eTime:
#    csvfname='rbn_wal_2014_cwSS__'+map_sTime.strftime('%H%M - ')+'-'+map_eTime.strftime('%H%M UT')
#    outfile=os.path.join(output_path,csvfname)
    kk= kk + 1
#    time.append(map_sTime)
#    time.append(map_eTime)

    if make_map==True:
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
    print "Processing RBN Data for Interval #"+str(kk)+': '+map_sTime.strftime('%H:%M')+'-'+map_eTime.strftime('%H:%M')
    rbn_df  = rbn_lib.read_rbn_std(map_sTime,map_eTime,data_dir='data/rbn')
#    rbn_df  = rbn_lib.read_rbn(map_sTime,map_eTime,data_dir='data/rbn')

    #Select Region
#    rbn_df2 = rbn_lib.rbn_region(rbn_df, latMin=latMin, latMax=latMax, lonMin=lonMin, lonMax=lonMax, constr_de=True, constr_dx=True)
    rbn_df2 = rbn_df 
#    import ipdb; ipdb.set_trace()
    #import ipdb; ipdb.set_trace()

    ##Limit links to those with a midpoint within the radius of the isond
    #rbn_links=rbn_df2[rbn_df2.dist<=radius]
    rbn_df2, rbn_links=rbn_lib.getLinks(rbn_df2,isond,radius) 
#    import ipdb; ipdb.set_trace()
#
#    #Calculate Critical frequency
    df_fc=rbn_lib.rbn_crit_freq(rbn_links, time=[map_sTime, map_eTime],coord_center=isond, freq1=freq1, freq2=freq2)
#    import ipdb; ipdb.set_trace()
    print df_fc.count1;
    print df_fc.count2;

#    #Find average frequency and distance/band
##    df1,df2,count1[i], count2[i], f1[i], f2[i], d1[i], d2[i]=rbn_lib.band_averages(rbn_links, freq1, freq2) 
##    df1,df2,count1[i], count2[i], f1[i], f2[i], d1[i], d2[i]=rbn_lib.band_averages(rbn_links, freq1, freq2) 
#    df1,df2, count=rbn_lib.band_averages(rbn_links, freq1, freq2) 
#
#    #Save averages in arrays
#    count1.append(count[0])
#    count2.append(count[1])
#    f1.append(df1.freq)
#    f2.append(df2.freq)
#    d1.append(df1.link_dist)
#    d2.append(df2.link_dist)
#
#    #See if have enough information to solve for critical frequency
##    if df1.freq.isempty() or df2.freq.isempty():
##    if f1==0 or f2==0:
#    if df1.freq==0 or df2.freq==0:
#        fc.append('NA')
#        hv.append('NA')
#        hmF2.append('NA')
#    else:
#        #Solve the critical frequency equation
##        hv.append(abs((np.sqrt((np.square(f2[i]*d1[i])-np.square(f1[i]*d2[i]))))/(np.square(f1[i])-np.square(f2[i])))/2)
##        height=abs((np.square(f2[i]*d1[i])-np.square(f1[i]*d2[i]))))/(np.square(f1[i])-np.square(f2[i])))/2
#        numer=abs(np.square(f2[i]*d1[i])-np.square(f1[i]*d2[i]))
#        den=abs(np.square(f1[i])-np.square(f2[i]))
#        hv.append(np.sqrt(abs(np.square(f2[i]*d1[i])-np.square(f1[i]*d2[i]))/abs(np.square(f1[i])-np.square(f2[i])))/2)
#        hvB.append(np.sqrt(numer/den)/2)
##        den=(np.square(f1[i])-np.square(f2[i]))
##        import ipdb; ipdb.set_trace()
#        fc.append(np.sqrt(np.square(f1[i])/(1+np.square(d1[i]/(2*hv[i])))))
#        fc2.append(np.sqrt(np.square(f2[i])/(1+np.square(d2[i]/(2*hv[i])))))
#        fcB.append(np.sqrt(np.square(f1[i])/(1+np.square(d1[i]/(2*hvB[i])))))
##        import ipdb; ipdb.set_trace()
#        #Get hmF2 for the start time
#        hmF2.append(rbn_lib.get_hmF2(map_sTime,isond[0], isond[1], output=False)) 
#        i=i+1
##        h[i]=np.sqrt((np.square(df2.freq*df1.dist)-np.square(df1.freq*df2.dist))/(np.square(df1.freq)-np.square(df2.freq)))/2
##        import ipdb; ipdb.set_trace()
##        fc[i]=np.sqrt(np.square(df1.freq)/(1+(df1.dist/(2*h[i]))))
##        import ipdb; ipdb.set_trace()
##        hv.append(np.sqrt((np.square(df2.freq*df1.dist)-np.square(df1.freq*df2.dist))/(np.square(df1.freq)-np.square(df2.freq)))/2)
##        import ipdb; ipdb.set_trace()
##        fc.append(np.sqrt(np.square(df1.freq)/(1+(df1.dist/(2*h[i])))))
##        import ipdb; ipdb.set_trace()
#
#    df_fc=pd.DataFrame({'date':[time[i]], 'count1': [count1[i-1]], 'count2': [count2[i-1]], 'd1': [d1[i-1]],'d2': [d2[i-1]],'f1': [f1[i-1]],'f2': [f2[i-1]],'hv': [hv[i-1]], 'fc1': [fc[i-1]],'fc2':[fc2[i-1]], 'hmF2':[hmF2[i-1]]})

#    #Export df of `links to csv file
#    rbn_links.to_csv(outfile, index=False)

    #Concatinate in a dataframe
    if kk==1:
        df_links=rbn_links
        df_full=df_fc
    else:
        df_links=pd.concat([df_links, rbn_links])
        df_full=pd.concat([df_full, df_fc], ignore_index=True)

    if make_map==True:
        #Plot on map
    #    fig = plt.figure(figsize=(8,4))
    #    ax0  = fig.add_subplot(1,1,1)
        if (rbn_links.date.min()!=rbn_links.date.max()):
            m, fig=rbn_lib.rbn_map_plot(rbn_links,legend=True,ax=ax0,tick_font_size=9,ncdxf=True, llcrnrlon=llcrnrlon, llcrnrlat=llcrnrlat, urcrnrlon=urcrnrlon, urcrnrlat=urcrnrlat)
            midpoint    = m.scatter(rbn_links.midLon, rbn_links.midLat,color='m',marker='s',s=2,zorder=100)
            loc_isond    = m.scatter(isond[1],isond[0],color='k',marker='*',s=12,zorder=100)
            #leg = rbn_lib.band_legend(fig,loc='center',bbox_to_anchor=[0.48,0.505],ncdxf=True,ncol=4)

    #Increment time 
    map_sTime=map_eTime
    map_eTime=map_sTime+deltaTime

    #    if kk==4:
    #        filename=rbnMap+str(fig_inx)+'_a.jpg'
    #        filepath    = os.path.join(output_path,filename)
    #        fig.savefig(filepath,bbox_inches='tight')
    #        fig.savefig(filepath[:-3]+'pdf',bbox_inches='tight')
    #        plt.clf()
    #        kk=0
    #        fig = plt.figure(figsize=(8,4))
    #        fig_inx=fig_inx+1

if make_map==True:
    #Plot last map
    filename=rbnMap+str(fig_inx)+'_a.jpg'
    filepath    = os.path.join(output_path,filename)
    fig.savefig(filepath,bbox_inches='tight')
    fig.savefig(filepath[:-3]+'pdf',bbox_inches='tight')
    plt.clf()
if make_fullmap==True:
#if (rbn_links.date.min()!=rbn_links.date.max()):
    fig = plt.figure(figsize=(8,4))
    ax0     = fig.add_subplot(1,1,1)
    m, fig=rbn_lib.rbn_map_plot(df_links,legend=True,ax=ax0,tick_font_size=9,ncdxf=True, llcrnrlon=llcrnrlon, llcrnrlat=llcrnrlat, urcrnrlon=urcrnrlon, urcrnrlat=urcrnrlat)
    midpoint    = m.scatter(df_links.midLon, df_links.midLat,color='m',marker='s',s=2,zorder=100)
    loc_isond    = m.scatter(isond[1],isond[0],color='k',marker='*',s=12,zorder=100)
    filename=rbnMap+sTime.strftime('%H%M - ')+'-'+eTime.strftime('%H%M UT')+'_a.jpg'
    filepath    = os.path.join(output_path,filename)
    fig.savefig(filepath,bbox_inches='tight')
    fig.savefig(filepath[:-3]+'pdf',bbox_inches='tight')
    plt.clf()
#Compile Calculated values into a new dataframe
#df_full=pd.DataFrame({'date':time, 'count1': count1, 'count2': count2, 'd1': d1,'d2': d2,'f1': f1,'f2': f2,'hv': hv, 'fc1': fc,'fc2':fc2, 'hmF2':hmF2})
#import ipdb; ipdb.set_trace()
outfile=os.path.join(output_path,csvfname)
df_links.to_csv(outfile)
#csvfname='info_rbn_wal_'+sTime.strftime('%h%m - ')+'-'+etime.strftime('%h%m ut')

#Make Count Graph for test
df=df_full.dropna(subset=['count1','count2'])
cgraph='RBN_WAL_count_'+sTime.strftime('%Y')+'_'+contest+'_'+'time_step='+str(deltaTime)+'_'+sTime.strftime('%H%M - ')+'-'+eTime.strftime('%H%M UT')+'.png'
fig         = plt.figure(figsize=(8,4)) # Create figure with the appropriate size.
line1=plt.plot(df['date'], df['count1'], '*-y',df['date'], df['count2'], '*-g')
filepath    = os.path.join(output_path,cgraph)
fig.savefig(filepath,bbox_inches='tight')
fig.savefig(filepath[:-3]+'pdf',bbox_inches='tight')
import ipdb; ipdb.set_trace()

#Drop NaNs (times that did not have enough data to preform the calculations)
df = df_full.dropna(subset=['d1', 'd2', 'f1', 'f2'])
#import ipdb; ipdb.set_trace()

#Plot Data and Calculated values
nx_plots=1
ny_plots=5
#xsize=8
#ysize=8
xsize=8
ysize=2
#fig=rbn_lib.fc_stack_plot(df, sTime, eTime, freq1, freq2, nx_plots,ny_plots, xsize, ysize, ncol=None, plot_legend=True)
#Uncomment the next 57 lines to get original plotting code
fig         = plt.figure(figsize=(nx_plots*xsize,ny_plots*ysize)) # Create figure with the appropriate size.
#plot 
ax0     = fig.add_subplot(ny_plots,nx_plots,1)
ax1     = fig.add_subplot(ny_plots,nx_plots,2)
ax2     = fig.add_subplot(ny_plots,nx_plots,3)
ax3     = fig.add_subplot(ny_plots,nx_plots,4)
ax4     = fig.add_subplot(ny_plots,nx_plots,5)
#ax1     = fig.add_subplot(ny_plots,1,2)
#ax2     = fig.add_subplot(1,ny_plots,3)
#ax3     = fig.add_subplot(1,ny_plots,4)
#fig, ((ax0),(ax1),(ax2),(ax3))=plt.subplots(4,1,sharex=True,sharey=False)
#fig, ((ax0),(ax1),(ax2),(ax3), (ax4))=plt.subplots(5,1,sharex=True,sharey=False)
#fig, ((ax5),(ax0),(ax1),(ax2),(ax3))=plt.subplots(5,1,sharex=True,sharey=False)
#m, fig=rbn_lib.rbn_map_plot(df_links,legend=True,ax=ax5,tick_font_size=9,ncdxf=True, llcrnrlon=llcrnrlon, llcrnrlat=llcrnrlat, urcrnrlon=urcrnrlon, urcrnrlat=urcrnrlat)
#plot data on the same figure
ax0.plot(df['date'], df['count1'], '*-y',df['date'], df['count2'], '*-g')
ax1.plot(df['date'], df['d1'], '*-y',df['date'], df['d2'], '*-g')
ax2.plot(df['date'], df['f1'], '*-y',df['date'], df['f2'], '*-g')
ax3.plot(df['date'], df['hv'], '*-m',df['date'], df['hmF2'],'*-r')
#ax3.plot(df['date'], df['hv'], '*-m')
ax4.plot(df['date'], df['fc1'], '*-y',df['date'], df['fc2'], '*-g')
#ax3.plot(df['date'], df['fc1'], '-y',df['date'], df['fc2'], '-g')
#Alternate color plots
#ax0.plot(df['date'], df['count1'], '-r',df['date'], df['count2'], '-b')
#ax1.plot(df['date'], df['d1'], '-r',df['date'], df['d2'], '-b')
#ax2.plot(df['date'], df['f1'], '-r',df['date'], df['f2'], '-b')
#ax3.plot(df['date'], df['hv'], '-g')
#ax4.plot(df['date'], df['fc1'], '-r',df['date'], df['fc2'], '-b')
##ax3.plot(df['date'], df['fc1'], '-r',df['date'], df['fc2'], '-b')

#Set the title and labels for the plots
ax0.set_title('RBN Spots per Unit Time\n'+sTime.strftime('%d %b %Y %H%M UT - ')+eTime.strftime('%d %b %Y %H%M UT'))
#ax1.set_title('Average Link Distance per Unit Time\n'+sTime.strftime('%d %b %Y %H%M UT - ')+eTime.strftime('%d %b %Y %H%M UT'))
#ax2.set_title('Average Link Frequency per Unit Time\n'+sTime.strftime('%d %b %Y %H%M UT - ')+eTime.strftime('%d %b %Y %H%M UT'))
###ax3.set_title('Calculated Virtual Height per unit timeper Unit Time\n'+sTime.strftime('%d %b %Y %H%M UT - ')+eTime.strftime('%d %b %Y %H%M UT'))
###ax4.set_title('Calculated Critical Frequency per unit timeper Unit Time\n'+sTime.strftime('%d %b %Y %H%M UT - ')+eTime.strftime('%d %b %Y %H%M UT'))
###ax4.set_xlabel('Time [UT]')
#ax3.set_title('Calculated Critical Frequency per Unit Time\n'+sTime.strftime('%d %b %Y %H%M UT - ')+eTime.strftime('%d %b %Y %H%M UT'))
#ax3.set_xlabel('Time [UT]')

#ax0.set_title('RBN Spots per Unit Time')
ax1.set_title('Average Link Distance per Unit Time')
ax2.set_title('Average Link Frequency per Unit Time')
ax3.set_title('Calculated Virtual Height per Unit Time')
ax4.set_title('Calculated Critical Frequency per Unit Time')
#ax3.set_title('Calculated Critical Frequency per Unit Time')

#set labels
ax0.set_ylabel('Count')
ax1.set_ylabel('Distance (km)')
ax2.set_ylabel('Freqency (kHz)')
ax3.set_ylabel('Height (km)')
ax4.set_ylabel('Freqency (kHz)')
ax4.set_xlabel('Time [UT]')
#ax3.set_xlabel('Time [UT]')
#ax0.set_legend(

##legend=plt.legend(handles=[ax0], loc=1)
#legend=plt.legend([line1, line2],['20m','40m'],loc=1)
#
#Add Legend
color1='y'
color2='g'
label1='14MHz'
#label1=str(freq1/'14MHz'
label2='7MHz'
handles=[]
labels=[]
#if fig is None: fig = plt.gcf() 
handles.append(mpatches.Patch(color=color1,label=label1))
labels.append(label1)
handles.append(mpatches.Patch(color=color2,label=label2))
labels.append(label2)
fig_tmp = plt.figure()
ax_tmp = fig_tmp.add_subplot(111)
ax_tmp.set_visible(False)
#if ncol is None:
ncol = len(labels)
#loc='lower center'
loc='lower right'
markerscale=0.5
prop={'size':10}
title=None
bbox_to_anchor=None
legend = fig.legend(handles,labels,ncol=ncol,loc=loc,markerscale=markerscale,prop=prop,title=title,bbox_to_anchor=bbox_to_anchor,scatterpoints=1)
#import ipdb; ipdb.set_trace()
ax = plt.gca().add_artist(legend)
#57 uncomment for original


#title_prop = {'weight':'bold','size':22}
##fig.text(0.525,1.025,'HF Communication Paths',ha='center',**title_prop)
#fig.text(0.525,1.000,'Reverse Beacon Network\n'+sTime.strftime('%d %b %Y %H%M UT - ')+eTime.strftime('%d %b %Y %H%M UT'),ha='center',**title_prop)
fig.tight_layout()
filepath    = os.path.join(output_path,graphfile1)
fig.savefig(filepath)
fig.savefig(filepath[:-3]+'pdf',bbox_inches='tight')
plt.clf()

##Make plot of fc1 and fc2 only
#fig         = plt.figure(figsize=(8,4)) # Create figure with the appropriate size.
#line1=plt.plot(df['date'], df['fc1'], color1, label='14MHz')
#filepath    = os.path.join(output_path,lgraph1)
#fig.savefig(filepath,bbox_inches='tight')
#fig.savefig(filepath[:-3]+'pdf',bbox_inches='tight')
#fig         = plt.figure(figsize=(8,4)) # Create figure with the appropriate size.
#line2=plt.plot(df['date'], df['fc2'], color2, label='7MHz')
#filepath    = os.path.join(output_path,lgraph2)
#fig.savefig(filepath,bbox_inches='tight')
#fig.savefig(filepath[:-3]+'pdf',bbox_inches='tight')
###legend=plt.legend([line1, line2],['20m','40m'],loc=1)
##filepath    = os.path.join(output_path,'Critical Frequency.png')
#
# the histogram of the data

#link_f1=df_links.freq[df_links.freq>freq1-500]
#import ipdb; ipdb.set_trace()
##link_f1=link_f1.freq[link_f1.freq<freq1+500]
#link_f1=link_f1[link_f1<freq1+500]
#import ipdb; ipdb.set_trace()
#link_f2=df_links.freq[df_links.freq>freq2-500]
##link_f2=link_f2.freq[link_f2.freq<freq2+500]
#link_f2=link_f2[link_f2<freq2+500]
#
##First Frequency Band  Histograms
#import ipdb; ipdb.set_trace()
#link1=link_f1.tolist()
#num_bins=len(link_f1)-1
##n, bins, patches = plt.hist(rbn_df2.Freq_plasma, num_bins, normed=1, facecolor='green', alpha=0.5)
#n, bins, patches = plt.hist(link1, num_bins, normed=1, facecolor=color1, alpha=0.5)
### add a 'best fit' line
###y = mlab.normpdf(bins, mu, sigma)
###plt.plot(bins, y, 'r--')
#plt.xlabel('Frequency (kHz)')
#plt.ylabel('Counts')
#plt.title('Histogram of '+str(freq1)+'kHz Band seen by RBN')
#plt.title('Histogram of Frequency from RBN')
#### Tweak spacing to prevent clipping of ylabel
#plt.subplots_adjust(left=0.15)
#filepath    = os.path.join(output_path,hgraph1)
#fig.savefig(filepath,bbox_inches='tight')
#fig.savefig(filepath[:-3]+'pdf',bbox_inches='tight')
#
##Second Frequency Band Histogram
#link2=link_f2.tolist()
#num_bins=len(link_f2)-1
##n, bins, patches = plt.hist(link2, num_bins, normed=1, facecolor=color2, alpha=0.5)
#n, bins, patches = plt.hist(link2, num_bins, facecolor=color2, alpha=0.5)
### add a 'best fit' line
###y = mlab.normpdf(bins, mu, sigma)
###plt.plot(bins, y, 'r--')
#plt.xlabel('Frequency (kHz)')
#plt.ylabel('Counts')
#plt.title('Histogram of '+str(freq2)+'kHz Band seen by RBN')
#### Tweak spacing to prevent clipping of ylabel
#plt.subplots_adjust(left=0.15)
#filepath    = os.path.join(output_path,hgraph2)
#fig.savefig(filepath,bbox_inches='tight')
#fig.savefig(filepath[:-3]+'pdf',bbox_inches='tight')

#Make plot of fc and hv only
#fig         = plt.figure(figsize=(8,4)) # Create figure with the appropriate size.
##line1=plt.plot(df['date'], df['fc1'], '-y', label='14MHz')
#line2=plt.plot(df['date'], df['fc2'], '-g', label='7MHz')
##legend=plt.legend([line1, line2],['20m','40m'],loc=1)
#filepath    = os.path.join(output_path,'Critical Frequency.png')

##make plots seperated by frequency
#nx_plots=1
#nx2=nx_plots+1
#ny_plots=5
##xsize=8
##ysize=8
#xsize=8
#ysize=2
#fig         = plt.figure(figsize=(nx2*xsize,ny_plots*ysize)) # Create figure with the appropriate size.
##plot 
#ax0     = fig.add_subplot(ny_plots,nx_plots,1)
#ax1     = fig.add_subplot(ny_plots,nx_plots,2)
#ax2     = fig.add_subplot(ny_plots,nx_plots,3)
#ax3     = fig.add_subplot(ny_plots,nx_plots,4)
#ax4     = fig.add_subplot(ny_plots,nx_plots,5)
#ax5     = fig.add_subplot(ny_plots,nx2,6)
#ax6     = fig.add_subplot(ny_plots,nx2,7)
#ax7     = fig.add_subplot(ny_plots,nx2,8)
#ax8     = fig.add_subplot(ny_plots,nx2,9)
#ax9     = fig.add_subplot(ny_plots,nx2,10)
#ax0.plot(df['date'], df['count1'], '-y')
#ax1.plot(df['date'], df['d1'], '-y')
#ax2.plot(df['date'], df['f1'], '-y')
#ax3.plot(df['date'], df['hv'], '-m')
#ax4.plot(df['date'], df['fc1'], '-y')
#ax5.plot(df['date'], df['count2'], '-g')
#ax6.plot(df['date'], df['d2'], '-g')
#ax7.plot(df['date'], df['f2'], '-g')
#ax8.plot(df['date'], df['hv'], '-m')
#ax9.plot(df['date'], df['fc2'], '-g')
#filepath    = os.path.join(output_path,graphfile2)
#fig.savefig(filepath)
##fig.savefig(filepath[:-3]+'pdf',bbox_inches='tight')
#plt.clf()

##Make count plot
#fig2, ax1, ax2, ax3=rbn_lib.count_band(df_links, sTime, eTime,Inc_eTime=True,freq1=7000, freq2=14000, freq3=3000,dt=15,unit='minutes',xRot=False, ret_lim=False, rti_plot=False)
##fig.tight_layout()
#filepath    = os.path.join(output_path,graphfile)
#fig2.savefig(filepath)
##fig.savefig(filepath[:-3]+'pdf',bbox_inches='tight')
#plt.clf()


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
