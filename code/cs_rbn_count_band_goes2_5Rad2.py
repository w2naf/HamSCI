 #!/usr/bin/env python
#This code is to get rbn data for a given date and time interval and plot the number of counts per unit time

import sys
import os

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib import patches
from matplotlib import gridspec as grd

import numpy as np
import pandas as pd

from davitpy import gme
import datetime
import rbn_lib
import davitpy

#create output directory if none exists
output_dir='output'
try: 
    os.makedirs(output_dir)
except:
    pass 

#CARSON CHANGES: Variable declarations for desired frequencies
freq5=3500
freq4=7000
freq3=14000
freq1=28000
freq2=21000
sat_nr= 15
Local=1
NA=1
latMin=30
latMax=80
lonMin=-130
lonMax=-60
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


#specify time interval for spot counts
tDelta=datetime.timedelta(minutes=dt)
#Specify whether to include eTime in the count if tDelta results in an end time greater than eTime
Inc_eTime=True

sT=datetime.datetime(2015,3,11)
eT=datetime.datetime(2015,3,12)
goes_data=gme.sat.read_goes(sT,eT,sat_nr)
flares=gme.sat.find_flares(goes_data,min_class='X1',window_minutes=60)
#import ipdb; ipdb.set_trace()
for T in range(0,len(flares)):
    Flar=flares.index[T]
    if Local==0:
        graphfile=str(Flar)+' Global '+'('+str(latMin)+','+str(latMax)+')'+' ['+str(lonMin)+','+str(lonMax)+']'
    else:
        graphfile=str(Flar)+' Local '+'('+str(latMin)+','+str(latMax)+')'+' ['+str(lonMin)+','+str(lonMax)+']'
    #import ipdb; ipdb.set_trace()


    Delt=datetime.timedelta(hours=3)
    sTime=Flar-Delt
    eTime=Flar+Delt
    Dum=datetime.datetime.strptime(str(Flar),'%Y-%m-%d %H:%M:%S')
    Var=Dum.strftime('%Y %m %d %H %M %S')
    Year=int(str.split(Var)[0])
    Month=int(str.split(Var)[1])
    Day=int(str.split(Var)[2])
    Hour=int(str.split(Var)[3])
    Min=int(str.split(Var)[4])
    #import ipdb;ipdb.set_trace()                     


    if Hour<21 and Hour>14:


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
        else:
            t_end=eTime
        #CARSON CHANGES
        #GET GOES Data
        goes_data   = gme.sat.read_goes(sTime+tDelta,t_end,sat_nr)
        #END FOR NOw

        #import ipdb; ipdb.set_trace()

        #Group counts together by unit time
        #index=0
        #define array to hold spot count
        spots=np.zeros(len(times))

        #CARSON VARIABLES: Spot counters for previous frequencies
        spots1=np.zeros(len(times))
        spots2=np.zeros(len(times))
        spots3=np.zeros(len(times))
        spots4=np.zeros(len(times))
        spots5=np.zeros(len(times))
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
        index=0
        while cTime < t_end:
            endTime += tDelta
            #import ipdb; ipdb.set_trace()
            #rbn_df2=rbn_df
            df1['Lower']=cTime
            df1['Upper']=endTime
            #import ipdb; ipdb.set_trace()
            #Clip according to the range of time for this itteration
            df2=df1[(df1.Lower <= df1.date) & (df1.date < df1.Upper)]
            #store spot count for the given time interval in an array 
            spots[index]=len(df2)

            for I in range(0,len(df2)-1):
                DumLatx=df2.dx_lat.iloc[I]
                DumLonx=df2.dx_lon.iloc[I]
                DumLate=df2.de_lat.iloc[I]
                DumLone=df2.de_lon.iloc[I]
        
                if Local==1:
                    if (DumLatx>latMin and DumLatx<latMax and DumLonx>lonMin and DumLonx<lonMax) and (DumLate>latMin and DumLate<latMax and DumLone<lonMax and DumLone>lonMin):
                        NA=1
                    else:
                        NA=0


                Dummy=df2.freq.iloc[I]
                if NA==1 and Dummy>(freq1-500) and Dummy<(freq1+500):
                    spots1[index]+=1
                elif NA==1 and Dummy>(freq2-500) and Dummy<(freq2+500): 
                    spots2[index]+=1
                elif NA==1 and Dummy>(freq3-500) and Dummy<(freq3+500):
                    spots3[index]+=1
                elif NA==1 and Dummy>(freq4-500) and Dummy<(freq4+500):
                    spots4[index]+=1
                elif NA==1 and Dummy>(freq5-500) and Dummy<(freq5+500):
                    spots5[index]+=1

            #Itterate current time value and index
            cTime=endTime
            index=index+1

        #create Data Frame from spots and times vectors
        spot_df=pd.DataFrame(data=times, columns=['dates'])
        spot_df['Count_F1']=spots1
        spot_df['Count_F2']=spots2
        spot_df['Count_F3']=spots3
        spot_df['Count_F4']=spots4
        spot_df['Count_F5']=spots5
        #spot_df=pd.DataFrame(data=spots, columns=['Count'])
        #import ipdb; ipdb.set_trace()

        #now isolate those on the day side
        #now we need to constrain the data to those contacts that are only on the day side 
        #will need to make this more elegant and universal
        #I just wrote a quick code to isolate it for ONE EXAMPLE

        #FLARE TIME
        ARR=np.array([Flar,Flar])
        #END

        #Plot figures
        #================================================================================================================================
        #================================================================================================================================
        fig=plt.figure(figsize=(11,15))#generate a figure
        gs=grd.GridSpec(7,1)#specify grid squares for plots to populate
        #Plotting for the highest frequency==============================================================================================
        
        ax1=plt.subplot(gs[2,:])
        ax1.plot(spot_df['dates'],spot_df['Count_F1'],'r*-',linewidth=3.0)
        axes=plt.gca()
        #DumLim=axes.get_ylim()
        #ax1.set_ylim([0,80])----If you need to manually set the ylimits
        DumLim=axes.get_ylim()
        ax1.plot(ARR,np.array([DumLim[0],DumLim[1]]),'k-',linewidth=3.0)
        ax1.grid(b=True, which='major', color='k', linestyle='--')
        labels=ax1.get_xticklabels()
        for label in labels:
            label.set_visible(False)
        labels=ax1.get_yticklabels()
        for label in labels:
            label.set_fontsize(16)

        #Plotting for the GOES data======================================================================================================
        ax0=plt.subplot(gs[:2,:],sharex=ax1)
        gme.sat.goes_plot(goes_data,ax=ax0,sTime=sTime,eTime=t_end)
        axes=plt.gca()
        DumLim=axes.get_ylim()
        ax0.plot(ARR,np.array([DumLim[0],DumLim[1]]),'k-',linewidth=2.0)
        labels=ax0.get_xticklabels()
        for label in labels:
            label.set_visible(False)
        labels=ax0.get_yticklabels()
        for label in labels:
            label.set_fontsize(16)

        #================================================================================================================================
        ax2=plt.subplot(gs[3,:],sharex=ax1)
        ax2.plot(spot_df['dates'], spot_df['Count_F2'],color='orange',linewidth=3.0)
        axes=plt.gca()
        DumLim=axes.get_ylim()
        #ax2.set_ylim([0,100])
        DumLim=axes.get_ylim()
        ax2.plot(ARR,np.array([DumLim[0],DumLim[1]]),'k-',linewidth=2.0)
        ax2.grid(b=True, which='major', color='k', linestyle='--')
        labels=ax2.get_xticklabels()
        for label in labels:
            label.set_visible(False)
        labels=ax2.get_yticklabels()
        for label in labels:
            label.set_fontsize(16)

        #================================================================================================================================
        ax3=plt.subplot(gs[4,:],sharex=ax1)
        ax3.plot(spot_df['dates'], spot_df['Count_F3'],'y*-',linewidth=3.0)
        axes=plt.gca()
        DumLim=axes.get_ylim()
        #ax3.set_ylim([0,150])
        DumLim=axes.get_ylim()
        ax3.plot(ARR,np.array([DumLim[0],DumLim[1]]),'k-',linewidth=2.0)
        ax3.grid(b=True, which='major', color='k', linestyle='--')
        labels=ax3.get_xticklabels()
        for label in labels:
            label.set_visible(False)
        labels=ax3.get_yticklabels()
        for label in labels:
            label.set_fontsize(16)

        #================================================================================================================================
        ax4=plt.subplot(gs[5,:],sharex=ax1)
        ax4.plot(spot_df['dates'],spot_df['Count_F4'],'g*-',linewidth=3.0)
        axes=plt.gca()
        DumLim=axes.get_ylim()
        #ax4.set_ylim([0,100])
        DumLim=axes.get_ylim()
        ax4.plot(ARR,np.array([DumLim[0],DumLim[1]]),'k-',linewidth=2.0)
        ax4.grid(b=True,which='major',color='k',linestyle='--')
        labels=ax4.get_xticklabels()
        for label in labels:
            label.set_visible(False)
        labels=ax4.get_yticklabels()
        for label in labels:
            label.set_fontsize(16)

        #===============================================================================================================================
        ax5=plt.subplot(gs[6,:],sharex=ax1)
        ax5.plot(spot_df['dates'], spot_df['Count_F5'],'b*-',linewidth=3.0)
        axes=plt.gca()
        DumLim=axes.get_ylim()
        ax5.plot(ARR,np.array([DumLim[0],DumLim[1]]),'k-',linewidth=2.0)
        ax5.grid(b=True,which='major',color='k',linestyle='--')
        
        labels=[item.get_text() for item in ax5.get_xticklabels()]
        T=Hour-8
        for I in range(0,len(labels)):
            labels[I]=str(T)+':00'
            T+=1
        ax5.set_xticklabels(labels)
        labels=ax5.get_xticklabels()
        for label in labels:
            label.set_fontsize(25)
            label.set_rotation(30)

        labels=ax5.get_yticklabels()
        for label in labels:
            label.set_fontsize(16)



        #================================================================================================================================
        #Set subplot and axis labels
        ax0.set_title('RBN Data for '+str(Hour)+':'+str(Min)+' Flare\n'+sTime.strftime('%d %b %Y %H%M UT - ')+eTime.strftime('%d %b %Y %H%M UT'),fontsize=28)
        #ax0.set_xlabel('Time, Central Time (UTC-6)',fontsize=28)
        ax0.set_ylabel('W/m^2',fontsize=23)
        ax0.set_xlabel('')
        
        ax1.set_ylabel('[NR. Spots]\n'+str(freq1/1000)+' MHz',fontsize=22)
        ax2.set_ylabel('[NR. Spots]\n'+str(freq2/1000)+' MHz',fontsize=22)
        ax3.set_ylabel('[NR. Spots]\n'+str(freq3/1000)+' MHz',fontsize=22)
        ax4.set_ylabel('[NR. Spots]\n'+str(freq4/1000)+' MHz',fontsize=22)
        ax5.set_ylabel('[NR. Spots]\n'+str(freq5/1000)+' MHz',fontsize=22)
        ax5.set_xlabel('Time, Central Time (UTC-6)',fontsize=28)


        fig.tight_layout()
        filename=os.path.join(output_dir, graphfile)
        fig.savefig(filename)


        #import ipdb; ipdb.set_trace()
