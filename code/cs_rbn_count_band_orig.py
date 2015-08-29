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
output_dir='output'
try: 
    os.makedirs(output_dir)
except:
    pass 

#CARSON CHANGES: Variable declarations for desired frequencies
freq1=6000
freq2=14000
freq3=28000
#END

#check this line! It may not be what I want to use
#output path=os.path.join('output', 'rbn_counts')
#data_dir=os.path.join('data','rbn')
import ipdb; ipdb.set_trace()
#Specify Several Inputs for the code
#specify index for vectors later in the code
index=0
#specify unit time (in minutes) to make count/unit time
#Note: to change units of unit time then change the expression in tDelta assignment!
dt=10
unit='minutes'
#specify filename for output graph's file
graphfile='rbnCount_timeStep_'+str(dt)+' '+unit

#specify times
sTime=datetime.datetime(2014,9,10,16,45)
eTime=datetime.datetime(2014, 9,10, 18, 30)
#specify time interval for spot counts
tDelta=datetime.timedelta(minutes=dt)
#Specify whether to include eTime in the count if tDelta results in an end time greater than eTime
Inc_eTime=True

#Redundant/old Code (next two lines)
#specify filename for output graph's file
#graphfile='rbnCount_time_step'+tDetlat.strftime('%M')

#Generate a time/date vector
curr_time=sTime
#Two ways to have time labels for each count for the graph of counts  vs time: 
    #1) the number of counts and the time at which that count started 
    #2) the number of counts and the time at which that count ended [the number of counts in a 5min interval stamped with the time the interval ended and the next interval began]
#For option 1: uncomment line 48 and comment line 49 (uncomment the line after these notes and comment the one after it)
#For option 2: uncomment line 49 and comment line 48 (comment the following line and uncomment the one after it)
#times=[sTime]
curr_time += tDelta
times=[curr_time]
#if using option 2 then delete "=" sign in the following line!!!!
#AND flip the commented "times.append(curr_time)"
while curr_time < eTime:
#    times.append(curr_time)
    curr_time+=tDelta
    times.append(curr_time)

#added the following code to ensure times does not contain any values > eTime
i_tmax=len(times)
#if the last time in the time array is greater than the end Time (eTime)  originally specified
#Then must decide whether to expand time range (times) to include eTime or clip times to exclude times greater than eTime
#This situation arises when the time step results in the final value in the times array that is greater than eTime
#times_max=times[len(times-1)]#times_max is the maximum time value in the list
if times[len(times)-1]>eTime:
    if Inc_eTime==True:
        print 'Choice Include Endpoint=True'
        #must do so all contacts in the last time interval are counted, if not then it will skew data by not including a portion of the count in the final interval
        t_end=times[len(times)-1]
    else:
        print 'Choice Include Endpoint=False'
        #The end time is now the second to last value in the times array
        #Change t_end and clip times array
        t_end=times[len(times)-2]
        times.remove(times[len(times-1)])


import ipdb; ipdb.set_trace()

#Group counts together by unit time
#index=0
#define array to hold spot count
spots=np.zeros(len(times))

#CARSON VARIABLES: Spot counters for previous frequencies
spots1=np.zeros(len(times))
spots2=np.zeros(len(times))
spots3=np.zeros(len(times))
#END

#import ipdb; ipdb.set_trace()
cTime=sTime
endTime=cTime
#Read RBN data for given dates/times
#call function to get rbn data, find de_lat, de_lon, dx_lat, dx_lon for the data
rbn_df=rbn_lib.read_rbn(sTime, t_end, data_dir='data/rbn')
#create data frame for the loop
df1=rbn_df
rbn_df2=rbn_df
#import ipdb; ipdb.set_trace()
J=0

while cTime < t_end:
    endTime += tDelta
   # import ipdb; ipdb.set_trace()
    #rbn_df2=rbn_df
    df1['Lower']=cTime
    df1['Upper']=endTime
    #import ipdb; ipdb.set_trace()
    #Clip according to the range of time for this itteration
    df2=df1[(df1.Lower <= df1.date) & (df1.date < df1.Upper)]
    #store spot count for the given time interval in an array 
    spots[index]=len(df2)

    for I in range(0,len(df2)-1):
        if df2.freq.iloc[I]>(freq1-500) and df2.freq.iloc[I]<(freq1+500):
            J=J+1
            spots1[index]+=1
        elif df2.freq.iloc[I]>(freq2-500) and df2.freq.iloc[I]<(freq2+500): 
            J=J+1
            spots2[index]+=1
        elif df2.freq.iloc[I]>(freq3-500) and df2.freq.iloc[I]<(freq3+500):
            J=J+1
            spots3[index]+=1
    #Itterate current time value and index
    cTime=endTime
    index=index+1

#create Data Frame from spots and times vectors
spot_df=pd.DataFrame(data=times, columns=['dates'])
spot_df['Count_F1']=spots1
spot_df['Count_F2']=spots2
spot_df['Count_F3']=spots3
#spot_df=pd.DataFrame(data=spots, columns=['Count'])
import ipdb; ipdb.set_trace()

#now isolate those on the day side
#now we need to constrain the data to those contacts that are only on the day side 
#will need to make this more elegant and universal
#I just wrote a quick code to isolate it for ONE EXAMPLE



#Plot figures
fig=plt.figure()#generate a figure
ax=fig.add_subplot(111)
ax.plot(spot_df['dates'], spot_df['Count_F1'],'r*-',spot_df['dates'],spot_df['Count_F2'],'b*-',spot_df['dates'],spot_df['Count_F3'],'g*-')
#ax2=fig.add_subplot(112,sharex=ax1)
#ax2.plot(spot_df['dates'], spot_df['Count_F2'],'b*-')
#ax3=fig.add_subplot(113,sharex=ax2)
#ax3.plot(spot_df['dates'], spot_df['Count_F3'],'g*-')

ax.set_title('RBN Spots per Unit Time\n'+sTime.strftime('%d %b %Y %H%M UT - ')+eTime.strftime('%d %b %Y %H%M UT'))
ax.set_ylabel('Spots/minute')
ax.set_xlabel('Time [UT]')
#Freq1=patches.Patch(color='red',label='3 MHz')
#Freq2=patches.Patch(color='blue',label='14 MHz')
#Freq3=patches.Patch(color='green',label='28 MHz')
plt.legend(['3 MHz','14 MHz','28 MHz'])

#ax.text(spot_df.dates.min(),spot_df.Count.min(),'Unit Time: '+str(dt)+' '+unit)
#ax.text(spot_df.dates[10],spot_df.Count.max(),'Unit Time: '+str(dt)+' '+unit)


fig.tight_layout()
filename=os.path.join(output_dir, graphfile)
# 'rbnCount_5min_line1.png')
fig.savefig(filename)

#count=np.ones((len(df1['date']),1))

import ipdb; ipdb.set_trace()
