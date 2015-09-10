#!/usr/bin/env python
#This code is to get rbn data for a given date and time interval and plot the number of counts per unit time

import sys
import os

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib import patches

import numpy as np
import pandas as pd

from davitpy import gme
import datetime

import rbn_lib
import rti_magda

#create output directory if none exists
output_dir='output/bks_rbn'
try: 
    os.makedirs(output_dir)
except:
    pass 

#CARSON CHANGES: Variable declarations for desired frequencies
#freq0=3000
freq1=7000
freq2=14000
freq3=28000
#END

#Specify using Blackstone Radar
radars=['bks']
#plot groundscatter in gray (True) or in color (False)
gs=False

#check this line! It may not be what I want to use
#output path=os.path.join('output', 'rbn_counts')
#data_dir=os.path.join('data','rbn')
#import ipdb; ipdb.set_trace()
#Specify Several Inputs for the code
##specify index for vectors later in the code
#index=0
#specify unit time (in minutes) to make count/unit time
#Note: to change units of unit time then change the expression in tDelta assignment!
dt=10
unit='minutes'
#Specify whether to rotate the x labels on the plots
xRot=True
##specify filename for output graph's file
#graphfile='K4KDJ_rbnCount_'+eTime.strftime()+str(index)

#specify times
sTime=datetime.datetime(2015,7,11,10)
eTime=datetime.datetime(2015, 7,12, 20)
#specify time interval for plots
plotDelta=datetime.timedelta(hours=5)

#Specify Test times
#start_time[datetime.datetime(2015,7)
fname='/home/km4ege/HamSCI/code/bks_test.csv'
#bks_df=pd.DataFrame.from_csv(fname, parse_dates=True)
#bks_df=pd.read_csv(fname,parse_dates=[10])
sTest=datetime.datetime(2015, 7,11, 15, 00, 00)
eTest=datetime.datetime(2015, 7,12, 15, 30, 00)
testDelta=datetime.timedelta(hours=1)
curr_time=sTest
bks_test=[curr_time]
curr_time+=testDelta
while curr_time < eTest:
#    Times.append(curr_time)
    bks_test.append(curr_time)
    curr_time+=testDelta
#if bks_test[len(bks_test)-1]>=eTest:
##    import ipdb; ipdb.set_trace()
#    if Inc_eTime==True:
#print 'Choice Include Endpoint=True'
#bks_test.remove(bks_test[len(bks_test-1)])

bks_test.append(eTest)
##        t_end=bks_test[len(bks_test)-1]
#    else:
#        print 'Choice Include Endpoint=False'
#        #The end time is now the second to last value in the Times array
#        #Change t_end and clip Times array
##        t_end=bks_test[len(bks_test)-2]
#        bks_test.remove(bks_test[len(bks_test-1)])
bks_df=bks_test
#import ipdb; ipdb.set_trace()
#Even values of inx point to the time the radar is on and Odd values point/index the times it is off
inx=0
bks_off=[bks_test[inx]]
inx=1
bks_on=[bks_test[inx]]
inx=2
while inx<len(bks_test)-1:
    bks_off.append(bks_test[inx])
    inx=inx+1
    bks_on.append(bks_test[inx])
    inx=inx+1

#import ipdb; ipdb.set_trace()
##Specify whether to include eTime in the count if tDelta results in an end time greater than eTime
Inc_eTime=True
curr_time=sTime
#import ipdb; ipdb.set_trace()
Times=[curr_time]
#import ipdb; ipdb.set_trace()

while curr_time < eTime:
#    Times.append(curr_time)
    curr_time+=plotDelta
    Times.append(curr_time)
if Times[len(Times)-1]>=eTime:
#    import ipdb; ipdb.set_trace()
    if Inc_eTime==True:
        print 'Choice Include Endpoint=True'
        #must do so all contacts in the last time interval are counted, if not then it will skew data by not including a portion of the count in the final interval
        t_end=Times[len(Times)-1]
    else:
        print 'Choice Include Endpoint=False'
        #The end time is now the second to last value in the Times array
        #Change t_end and clip Times array
        t_end=Times[len(Times)-2]
        Times.remove(Times[len(Times-1)])

#import ipdb; ipdb.set_trace()
#Read RBN data for given dates/times
rbn_df=rbn_lib.k4kdj_rbn(sTime, datetime.datetime(2015, 07, 12, 00), data_dir='data/rbn')
#import ipdb; ipdb.set_trace()
df=rbn_lib.k4kdj_rbn(datetime.datetime(2015, 07, 12, 00), t_end, data_dir='data/rbn')
#import ipdb; ipdb.set_trace()
rbn_df=pd.concat([rbn_df, df])
#rbn_df=rbn_lib.k4kdj_rbn(sTime, t_end, data_dir='data/rbn')
#import ipdb; ipdb.set_trace()
##create data frame for the loop
#df1=rbn_df[rbn_df['callsign']=='K4KDJ']
#import ipdb; ipdb.set_trace()
#rbn_df2=rbn_df

#start conditions for loop
index=0
#import ipdb; ipdb.set_trace()
plot_sTime=Times[0]
plot_eTime=Times[1]

#Index for radar
sInx=0
eInx=0
flag1 =True
flag2 =True

while plot_sTime < Times[len(Times)-1]:
    plot_eTime=Times[index+1]

    #Get count plot
#    import ipdb; ipdb.set_trace()
    fig, ax1, ax2, ax3, ax4=rbn_lib.count_band(df1=rbn_df,sTime=plot_sTime, eTime=plot_eTime, freq1=freq1,freq2=freq2, freq3=freq3,dt=dt, unit=unit,xRot=xRot, rti_plot=True) 
#    import ipdb; ipdb.set_trace()

#    plt.xticks(rotation=30)

    #Make RTI plot for Blackstone
#    import ipdb; ipdb.set_trace()
    rti_magda.plotRti(sTime=plot_sTime, eTime=plot_eTime, ax=ax4, rad=radars[0], params=['power'],yrng=[0,40], gsct=gs, cax=None, xtick_size=10,ytick_size=10)
#    import ipdb; ipdb.set_trace()
    
    #get axis limits
    DumLim1=ax1.get_ylim()
    DumLim2=ax2.get_ylim()
    DumLim3=ax3.get_ylim()
#    import ipdb; ipdb.set_trace()

    #Insert appropriate indicators of radar state
    while plot_sTime<=bks_off[eInx]<=plot_eTime: 
        if flag1==True:
#            inx=eInx
            #Plot When radar turned off
            ax1.vlines(bks_off[eInx] ,DumLim1[0],DumLim1[1],color='r')
            ax2.vlines(bks_off[eInx] ,DumLim2[0],DumLim2[1],color='r')
            ax3.vlines(bks_off[eInx] ,DumLim3[0],DumLim3[1],color='r')
            eInx=eInx+1
            if eInx>=len(bks_off):
                flag1= False
                bks_off.append(bks_off[0])
#                eInx=eInx-1
#        import ipdb; ipdb.set_trace()
        
    while plot_sTime<=bks_on[sInx]<=plot_eTime:
        if flag2==True:
            #Draw lines for times off and on
            #Even values of inx point to the time the radar is on and Odd values point/index the times it is off
            ax1.vlines(bks_on[sInx],DumLim1[0],DumLim1[1],color='g')
            ax2.vlines(bks_on[sInx],DumLim2[0],DumLim2[1],color='g')
            ax3.vlines(bks_on[sInx],DumLim3[0],DumLim3[1],color='g')
            sInx=sInx+1
            if sInx>=len(bks_on):
                flag2= False
                bks_on.append(bks_on[0])
#                sInx=sInx-1
#    import ipdb; ipdb.set_trace()

    #specify filename for output graph's file
    graphfile='Plot'+str(index)+'K4KDJ_rbnCount_and_RTI_'+plot_sTime.strftime('%H_%M')+'-'+plot_eTime.strftime('%H_%M')+'Plot'

    #Save Figure
    fig.tight_layout()
    filename=os.path.join(output_dir, graphfile)
    # 'rbnCount_5min_line1.png')
    fig.savefig(filename)
    #    import ipdb; ipdb.set_trace()

    #increment 
    index=index+1
    plot_sTime=Times[index]
    #    import ipdb; ipdb.set_trace()


#import ipdb; ipdb.set_trace()
#Get arrays reset
bks_off.remove(bks_off[len(bks_off)-1])
bks_on.remove(bks_on[len(bks_on)-1])
#Get count plot
fig,ax1, ax2, ax3,ax4=rbn_lib.count_band(df1=rbn_df,sTime=sTime, eTime=eTime, freq1=freq1,freq2=freq2, freq3=freq3,dt=dt, unit=unit,xRot=xRot,ret_lim=False, rti_plot=True) 
#import ipdb; ipdb.set_trace()
#specify filename for output graph's file
graphfile='Plot'+str(index)+'FullTime_K4KDJ_rbnCount_and_RTI_'+sTime.strftime('%H_%M')+'-'+eTime.strftime('%H_%M')+'Plot'
#import ipdb; ipdb.set_trace()

DumLim1=ax1.get_ylim()
DumLim2=ax2.get_ylim()
DumLim3=ax3.get_ylim()
#Draw lines for times off and on
#Even values of inx point to the time the radar is on and Odd values point/index the times it is off
ax1.vlines(bks_on,DumLim1[0],DumLim1[1],color='g')
ax2.vlines(bks_on,DumLim2[0],DumLim2[1],color='g')
ax3.vlines(bks_on,DumLim3[0],DumLim3[1],color='g')
#ax2.plot(bks_on,np.array(DumLim2[0],DumLim2[1]),color='r')
#ax3.plot(bks_on,np.array(DumLim3[0],DumLim3[1]),color='r')

#Plot When radar turned off
ax1.vlines(bks_off ,DumLim1[0],DumLim1[1],color='r')
ax2.vlines(bks_off ,DumLim2[0],DumLim2[1],color='r')
ax3.vlines(bks_off ,DumLim3[0],DumLim3[1],color='r')
#Make RTI plots for Blackstone Radar
#ax      = fig.add_axes(ax_dim)
rti_magda.plotRti(sTime=sTime, eTime=eTime, ax=ax4, rad=radars[0], params=['power'],yrng=[0,40], gsct=gs, cax=None, xtick_size=10,ytick_size=10)
#Save Figure
fig.tight_layout()
# 'rbnCount_5min_line1.png')
filename=os.path.join(output_dir, graphfile)
fig.savefig(filename)
