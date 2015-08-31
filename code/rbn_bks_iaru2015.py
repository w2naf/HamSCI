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

#check this line! It may not be what I want to use
#output path=os.path.join('output', 'rbn_counts')
#data_dir=os.path.join('data','rbn')
#import ipdb; ipdb.set_trace()
#Specify Several Inputs for the code
#specify index for vectors later in the code
index=0
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
import ipdb; ipdb.set_trace()
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

import ipdb; ipdb.set_trace()
##Specify whether to include eTime in the count if tDelta results in an end time greater than eTime
Inc_eTime=True
curr_time=sTime
Times=[curr_time]

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
import ipdb; ipdb.set_trace()
df=rbn_lib.k4kdj_rbn(datetime.datetime(2015, 07, 12, 00), t_end, data_dir='data/rbn')
import ipdb; ipdb.set_trace()
rbn_df=pd.concat([rbn_df, df])
#rbn_df=rbn_lib.k4kdj_rbn(sTime, t_end, data_dir='data/rbn')
import ipdb; ipdb.set_trace()
##create data frame for the loop
#df1=rbn_df[rbn_df['callsign']=='K4KDJ']
#import ipdb; ipdb.set_trace()
#rbn_df2=rbn_df

#start conditions for loop
plot_sTime=Times[0]
plot_eTime=Times[1]

#Index for radar
sInx=0
eInx=0

while plot_sTime < Times[len(Times)-1]:
    plot_eTime=Times[index+1]

    #Get count plot
    fig,ax1, ax2, ax3=rbn_lib.count_band(df1=rbn_df,sTime=plot_sTime, eTime=plot_eTime, freq1=freq1,freq2=freq2, freq3=freq3,dt=dt, unit=unit,xRot=xRot) 

    plt.xticks(rotation=30)

    #specify filename for output graph's file
    graphfile='Plot'+str(index)+'K4KDJ_rbnCount_'+plot_sTime.strftime('%H_%M')+'-'+plot_eTime.strftime('%H_%M')+'Plot'

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


#Get count plot
fig,ax1, ax2, ax3,DumLim1, DumLim2, DumLim3=rbn_lib.count_band(df1=rbn_df,sTime=sTime, eTime=eTime, freq1=freq1,freq2=freq2, freq3=freq3,dt=dt, unit=unit,xRot=xRot,ret_lim=True) 
import ipdb; ipdb.set_trace()
#specify filename for output graph's file
graphfile='Plot'+str(index)+'FullTime_K4KDJ_rbnCount_'+sTime.strftime('%H_%M')+'-'+eTime.strftime('%H_%M')+'Plot'

fig.tight_layout()
# 'rbnCount_5min_line1.png')
fig.savefig(filename)
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
#Save Figure
filename=os.path.join(output_dir, graphfile)
#ax1.plot(bks_off,np.array(DumLim1[0],DumLim1[1]),color='g')
#ax2.plot(bks_off,np.array(DumLim2[0],DumLim2[1]),color='g')
#ax3.plot(bks_off,np.array(DumLim3[0],DumLim3[1]),color='g')
#while inx<len(bks_test)-1:
#    #Plot When radar turned off
#    ax1.plot(bks_test[inx],np.array(DumLim1[0],DumLim1[1]),color='r')
#    ax2.plot(bks_test[inx],np.array(DumLim2[0],DumLim2[1]),color='r')
#    ax3.plot(bks_test[inx],np.array(DumLim3[0],DumLim3[1]),color='r')
#    
#    #Now Increase inx so that we can get the time that the radar is turned on 
#    inx=inx+1
#    
#    #Plot When radar turned on
#    ax1.plot(bks_test[inx],np.array(DumLim1[0],DumLim1[1]),color='g')
#    ax2.plot(bks_test[inx],np.array(DumLim2[0],DumLim2[1]),color='g')
#    ax3.plot(bks_test[inx],np.array(DumLim3[0],DumLim3[1]),color='g')
#
#    #Increment inx so that it points to the next "turn on" time
##    import ipdb; ipdb.set_trace()
#    inx=inx+1
#    import ipdb; ipdb.set_trace()
#
#ax1.vlines(bks_df['t_on'],[0],[90],color='g')
#ax1.vlines(bks_df['t_off'],[0],[90],color='r')
#ax2.vlines(bks_df['t_on'],[0],[90],color='g')
#ax2.vlines(bks_df['t_off'],[0],[90],color='r')
#ax3.vlines(bks_df['t_on'],[0],[90],color='g')
#ax3.vlines(bks_df['t_off'],[0],[90],color='r')
##specify filename for output graph's file
#graphfile='Plot'+str(index)+'FullTime_K4KDJ_rbnCount_'+sTime.strftime('%H_%M')+'-'+eTime.strftime('%H_%M')+'Plot'
#
##Save Figure
#fig.tight_layout()
#filename=os.path.join(output_dir, graphfile)
## 'rbnCount_5min_line1.png')
#fig.savefig(filename)
  

##Redundant/old Code (next two lines)
##specify filename for output graph's file
##graphfile='rbnCount_time_step'+tDetlat.strftime('%M')
#
##Generate a time/date vector
#curr_time=sTime
##Two ways to have time labels for each count for the graph of counts  vs time: 
#    #1) the number of counts and the time at which that count started 
#    #2) the number of counts and the time at which that count ended [the number of counts in a 5min interval stamped with the time the interval ended and the next interval began]
##For option 1: uncomment line 48 and comment line 49 (uncomment the line after these notes and comment the one after it)
##For option 2: uncomment line 49 and comment line 48 (comment the following line and uncomment the one after it)
##times=[sTime]
#curr_time += tDelta
#times=[curr_time]
##if using option 2 then delete "=" sign in the following line!!!!
##AND flip the commented "times.append(curr_time)"
#while curr_time < eTime:
##    times.append(curr_time)
#    curr_time+=tDelta
#    times.append(curr_time)
#
##added the following code to ensure times does not contain any values > eTime
#i_tmax=len(times)
##if the last time in the time array is greater than the end Time (eTime)  originally specified
##Then must decide whether to expand time range (times) to include eTime or clip times to exclude times greater than eTime
##This situation arises when the time step results in the final value in the times array that is greater than eTime
##times_max=times[len(times-1)]#times_max is the maximum time value in the list
#import ipdb; ipdb.set_trace()
#if times[len(times)-1]>=eTime:
#    import ipdb; ipdb.set_trace()
#    if Inc_eTime==True:
#        print 'Choice Include Endpoint=True'
#        #must do so all contacts in the last time interval are counted, if not then it will skew data by not including a portion of the count in the final interval
#        t_end=times[len(times)-1]
#    else:
#        print 'Choice Include Endpoint=False'
#        #The end time is now the second to last value in the times array
#        #Change t_end and clip times array
#        t_end=times[len(times)-2]
#        times.remove(times[len(times-1)])
#
##import ipdb; ipdb.set_trace()
#
##Group counts together by unit time
##index=0
##define array to hold spot count
#spots=np.zeros(len(times))
#
##CARSON VARIABLES: Spot counters for previous frequencies
##spots0=np.zeros(len(times))
#spots1=np.zeros(len(times))
#spots2=np.zeros(len(times))
#spots3=np.zeros(len(times))
##END
#
##import ipdb; ipdb.set_trace()
#cTime=sTime
#endTime=cTime
##Read RBN data for given dates/times
##call function to get rbn data, find de_lat, de_lon, dx_lat, dx_lon for the data
#rbn_df=rbn_lib.k4kdj_rbn(sTime, t_end, data_dir='data/rbn')
##create data frame for the loop
#df1=rbn_df[rbn_df['callsign']=='K4KDJ']
#import ipdb; ipdb.set_trace()
#rbn_df2=rbn_df
##import ipdb; ipdb.set_trace()
#J=0
#
#while cTime < t_end:
#    endTime += tDelta
#   # import ipdb; ipdb.set_trace()
#    #rbn_df2=rbn_df
#    df1['Lower']=cTime
#    df1['Upper']=endTime
#    #import ipdb; ipdb.set_trace()
#    #Clip according to the range of time for this itteration
#    df2=df1[(df1.Lower <= df1.date) & (df1.date < df1.Upper)]
#    #store spot count for the given time interval in an array 
#    spots[index]=len(df2)
#
#    for I in range(0,len(df2)-1):
#        if df2.freq.iloc[I]>(freq1-500) and df2.freq.iloc[I]<(freq1+500):
#            J=J+1
#            spots1[index]+=1
#        elif df2.freq.iloc[I]>(freq2-500) and df2.freq.iloc[I]<(freq2+500): 
#            J=J+1
#            spots2[index]+=1
#        elif df2.freq.iloc[I]>(freq3-500) and df2.freq.iloc[I]<(freq3+500):
#            J=J+1
#            spots3[index]+=1
#       # elif df2.freq.iloc[I]>(freq0-500) and df2.freq.iloc[I]<(freq0+500):
#       #     spots0[index]+=1
#    #Itterate current time value and index
#    cTime=endTime
#    index=index+1
#
##create Data Frame from spots and times vectors
#spot_df=pcolor=d.DataFrame(data=times, columns=['dates'])
##spot_df['Count_F0']=spots0
#spot_df['Count_F1']=spots1
#spot_df['Count_F2']=spots2
#spot_df['Count_F3']=spots3
##spot_df=pd.DataFrame(data=spots, columns=['Count'])
##import ipdb; ipdb.set_trace()
#
##now isolate those on the day side
##now we need to constrain the data to those contacts that are only on the day side 
##will need to make this more elegant and universal
##I just wrote a quick code to isolate it for ONE EXAMPLE
#
#
#
##Plot figures
##fig=plt.figure()#generate a figure
#fig, ((ax1),(ax2),(ax3))=plt.subplots(3,1,sharex=True,sharey=False)
##ax.plot(spot_df['dates'], spot_df['Count_F1'],'r*-',spot_df['dates'],spot_df['Count_F2'],'b*-',spot_df['dates'],spot_df['Count_F3'],'g*-')
##ax0.plot(spot_df['dates'], spot_df['Count_F0'],'y*-')
#ax1.plot(spot_df['dates'], spot_df['Count_F1'],'r*-')
#ax2.plot(spot_df['dates'], spot_df['Count_F2'],'b*-')
#ax3.plot(spot_df['dates'], spot_df['Count_F3'],'g*-')
#
#ax1.set_title('RBN Spots per Unit Time\n'+sTime.strftime('%d %b %Y %H%M UT - ')+eTime.strftime('%d %b %Y %H%M UT'))
##ax0.set_ylabel(str(freq0/1000)+' MHz')
#ax1.set_ylabel(str(freq1/1000)+' MHz')
#ax2.set_ylabel(str(freq2/1000)+' MHz')
#ax3.set_ylabel(str(freq3/1000)+' MHz')
#ax3.set_xlabel('Time [UT]')
##Freq1=patches.Patch(color='red',label='3 MHz')
##Freq2=patches.Patch(color='blue',label='14 MHz')
##Freq3=patches.Patch(color='green',label='28 MHz')
##plt.legend(['3 MHz','14 MHz','28 MHz'])
#
##ax.text(spot_df.dates.min(),spot_df.Count.min(),'Unit Time: '+str(dt)+' '+unit)
##ax.text(spot_df.dates[10],spot_df.Count.max(),'Unit Time: '+str(dt)+' '+unit)

#fig=rbn_lib.count_band(sTime=sTime, eTime=eTime, freq1=freq1,freq2=freq2, freq3=freq3,dt=dt, unit=unit) 
#fig.tight_layout()
#filename=os.path.join(output_dir, graphfile)
## 'rbnCount_5min_line1.png')
#fig.savefig(filename)

#count=np.ones((len(df1['date']),1))

#import ipdb; ipdb.set_trace()
