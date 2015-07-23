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
myLat=50
myLon=-95
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
#specify filename for output graph's file
#graphfile='rbnCount_timeStep_'+str(dt)+' '+unit

#specify times
#Flar=datetime.datetime(2015,3,10,16,22)
#Delt=datetime.timedelta(hours=3)
#sTime=Flar-Delt
#eTime=Flar+Delt
#sTime=datetime.datetime(2015,3,10,13,22)
#eTime=datetime.datetime(2015,3,10,19,22)
#sTime=datetime.datetime(2014,9,10,16,45)
#eTime=datetime.datetime(2014, 9,10, 18, 30)
#specify time interval for spot counts
tDelta=datetime.timedelta(minutes=dt)
#Specify whether to include eTime in the count if tDelta results in an end time greater than eTime
Inc_eTime=True

sT=datetime.datetime(2013,1,1)
eT=datetime.datetime(2015,3,30)
goes_data=gme.sat.read_goes(sT,eT,sat_nr)
flares=gme.sat.find_flares(goes_data,min_class='X1',window_minutes=60)
#import ipdb; ipdb.set_trace()
for T in range(0,len(flares)):
    Flar=flares.index[T]
    if Local==0:
        graphfile=str(Flar)+' Tlag Global '+'('+str(latMin)+','+str(latMax)+')'+' ['+str(lonMin)+','+str(lonMax)+']'
    else:
        graphfile=str(Flar)+' Tlag Local '+'('+str(latMin)+','+str(latMax)+')'+' ['+str(lonMin)+','+str(lonMax)+']'

    #CARSON FIND FLARE TIME AND GET sTime and eTime from
    #sT=datetime.datetime(2014,1,1)
    #eT=datetime.datetime(2014,12,30)
    #goes_data   = gme.sat.read_goes(sT,eT,sat_nr)
    #flares      = gme.sat.find_flares(goes_data,min_class='X1',window_minutes=60)
    #import ipdb; ipdb.set_trace()


    #Flar=flares.index[T]
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
    #if Hour>=20:
    #    D=Day+1
    #    H=Hour-23
    #else:
    #    D=Day
    #    H=Hour
    #sTime=datetime.datetime(Year,Month,Day,H-3)
    #eTime=datetime.datetime(Year,Month,D, H+3)

    if Hour<21 and Hour>13:


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
        b=goes_data.values()[0]
        Time=[]
        Time1=[]
        for I in range(0,len(b.index)):
            Dum=b.index[I]
            #goes_data.values()[0].index[I]-=datetime.timedelta(hours=6)
            Time.append(datetime.datetime.strptime(str(Dum),'%Y-%m-%d %H:%M:%S'))
            Dum=Time[I].strftime('%Y %m %d %H %M %S')
            Time1.append(float(str.split(Dum)[3]) + float(str.split(Dum)[4])/60)
        Mag=b.B_AVG




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


        #Int1=np.interp(float(Time),float(datetime.datetime.fromtimestamp(times)),spots1)
        Times=np.zeros(len(times))
        for I in range(0,len(times)):
            times[I]=datetime.datetime.strptime(str(times[I]),"%Y-%m-%d %H:%M:%S")
            #times[I]-=datetime.timedelta(hours=6)
            Dum=times[I].strftime('%Y %m %d %H %M %S')
            Times[I]=float(str.split(Dum)[3]) + float(str.split(Dum)[4])/60
            #times[I]-=datetime.timedelta(hours=6)

        Spots1=np.interp(Time1,Times,spots1)
        Spots2=np.interp(Time1,Times,spots2)
        Spots3=np.interp(Time1,Times,spots3)
        Spots4=np.interp(Time1,Times,spots4)
        Spots5=np.interp(Time1,Times,spots5)
        #import ipdb; ipdb.set_trace()


        #Correlation
        # Needed for normalization
        Mag = (Mag - np.mean(Mag)) / (np.std(Mag) * len(Mag))
        Spots1 = (Spots1 - np.mean(Spots1)) /  np.std(Spots1)
        Spots2 = (Spots2 - np.mean(Spots2)) /  np.std(Spots2)
        Spots3 = (Spots3 - np.mean(Spots3)) /  np.std(Spots3)
        Spots4 = (Spots4 - np.mean(Spots4)) /  np.std(Spots4)
        Spots5 = (Spots5 - np.mean(Spots5)) /  np.std(Spots5)

        #The correlation happens here.
        corr1 = np.correlate(Mag,Spots1,mode='same')
        corr2 = np.correlate(Mag,Spots2,mode='same')
        corr3 = np.correlate(Mag,Spots3,mode='same')
        corr4 = np.correlate(Mag,Spots4,mode='same')
        corr5 = np.correlate(Mag,Spots5,mode='same')


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
        #Flar=datetime.datetime(2014,9,10,17,45)
        ARR=np.array([Flar,Flar])
        #END


        #=================================================
        #Calc Terminator
        #def daynight_terminator(date, lons):
            #               """
            #date is datetime object (assumed UTC).
            #                """
            #import mpl_toolkits.basemap.solar as solar
            #dg2rad = np.pi/180.
            # compute greenwich hour angle and solar declination
            # from datetime object (assumed UTC).
            #tau, dec = solar.epem(date)
            # compute day/night terminator from hour angle, declination.
            #longitude = lons + tau
            #lats = np.arctan(-np.cos(longitude*dg2rad)/np.tan(dec*dg2rad))/dg2rad
            #return lats,tau,dec

        #daylight = np.ones([len(Time),len(myLat)],np.bool)
        #for tm_inx in range(len(Time)):    
            #tm = dt_list[tm_inx]
            #term_lats,tau,dec   = daynight_terminator(tm,myLon)
            #if dec > 0: # NH Summer
                #day_inx = np.where(myLat < term_lats)[0]
            #else:
                #day_inx = np.where(myLat > term_lats)[0]

            #if day_inx.size != 0:
                #daylight[tm_inx,day_inx] = False
                                                                                                                                                                                                                                                                                                                                                                                                
        #from numpy import ma
        #daylight = ma.array(daylight,mask=daylight)
        #ax.pcolormesh(X, Y, daylight.T, lw=0,alpha=0.10,cmap=matplotlib.cm.binary_r,zorder=99)
        #import ipdb;ipdb.set_trace()

        #Plot figures
        #================================================================================================================================
        #================================================================================================================================
        fig=plt.figure(figsize=(11,20))#generate a figure
        gs=grd.GridSpec(12,1)#specify grid squares for plots to populate
        #gs.update(wspace=0.01)

        #fig, ((ax0),(ax1),(ax2),(ax3))=plt.subplots(4,1,sharex=True,sharey=False)
        #ax.plot(spot_df['dates'], spot_df['Count_F1'],'r*-',spot_df['dates'],spot_df['Count_F2'],'b*-',spot_df['dates'],spot_df['Count_F3'],'g*-')

        #Plotting for the highest frequency==============================================================================================
        ax1=plt.subplot(gs[2,:])
        ax1.plot(spot_df['dates'],spot_df['Count_F1'],'g*-')
        #ax1.plot(Time,Spots1,'b*-')
        axes=plt.gca()
        DumLim=axes.get_ylim()
        ax1.plot(ARR,np.array([DumLim[0],DumLim[1]]),'k-',linewidth=2.0)
        ax1.grid(b=True, which='major', color='k', linestyle='--')
        labels=ax1.get_xticklabels()
        for label in labels:
            label.set_visible(False)

        ax6=plt.subplot(gs[3,:])
        ax6.plot(Time,corr1)
        M=min(corr1)
        I=np.argmin(corr1)
        ax6.plot(Time[I],M,'b*')
        axes=plt.gca()
        DumLim=axes.get_ylim()
        ax6.plot(ARR,np.array([DumLim[0],DumLim[1]]),'k-',linewidth=2.0)
        ax6.grid(b=True, which='major',color='k', linestyle='--')
        labels=ax6.get_xticklabels()
        for label in labels:
            label.set_visible(False)
        ax6.set_ylabel('r Max='+str(abs(round(M,3))))
        #props=dict(boxstyle='square',facecolor='white',alpha=0.5)
        #ax6.text(0.95,0.95,'r Max = '+str(M),horizontalalignment='right',verticalalignment='top',bbox=props)

        #Plotting for the GOES data======================================================================================================
        ax0=plt.subplot(gs[:2,:],sharex=ax1)
        gme.sat.goes_plot(goes_data,ax=ax0,sTime=sTime,eTime=t_end)
        axes=plt.gca()
        DumLim=axes.get_ylim()
        ax0.plot(ARR,np.array([DumLim[0],DumLim[1]]),'k-',linewidth=2.0)
        #gme.sat.goes_plot(goes_data,ax=ax0,sTime=sTime,eTime=t_end)
        labels=ax0.get_xticklabels()
        for label in labels:
            label.set_visible(False)

        #================================================================================================================================
        ax2=plt.subplot(gs[4,:],sharex=ax1)
        ax2.plot(spot_df['dates'], spot_df['Count_F2'],'r*-')
        axes=plt.gca()
        DumLim=axes.get_ylim()
        ax2.plot(ARR,np.array([DumLim[0],DumLim[1]]),'k-',linewidth=2.0)
        ax2.grid(b=True, which='major', color='k', linestyle='--')
        labels=ax2.get_xticklabels()
        for label in labels:
            label.set_visible(False)

        
        ax7=plt.subplot(gs[5,:],sharex=ax1)
        ax7.plot(Time,corr2)
        M=min(corr2)
        I=np.argmin(corr2)
        ax7.plot(Time[I],M,'b*')
        axes=plt.gca()
        DumLim=axes.get_ylim()
        ax7.plot(ARR,np.array([DumLim[0],DumLim[1]]),'k-',linewidth=2.0)
        ax7.grid(b=True, which='major',color='k',linestyle='--')
        labels=ax7.get_xticklabels()
        for label in labels:
            label.set_visible(False)
        ax7.set_ylabel('r Max='+str(abs(round(M,3))))

        #================================================================================================================================
        ax3=plt.subplot(gs[6,:],sharex=ax1)
        ax3.plot(spot_df['dates'], spot_df['Count_F3'],'b*-')
        axes=plt.gca()
        DumLim=axes.get_ylim()
        ax3.plot(ARR,np.array([DumLim[0],DumLim[1]]),'k-',linewidth=2.0)
        ax3.grid(b=True, which='major', color='k', linestyle='--')
        labels=ax3.get_xticklabels()
        for label in labels:
            label.set_visible(False)

        ax8=plt.subplot(gs[7,:],sharex=ax1)
        ax8.plot(Time,corr3)
        M=min(corr3)
        I=np.argmin(corr3)
        ax8.plot(Time[I],M,'b*')
        axes=plt.gca()
        DumLim=axes.get_ylim()
        ax8.plot(ARR,np.array([DumLim[0],DumLim[1]]),'k-',linewidth=2.0)
        ax8.grid(b=True,which='major',color='k',linestyle='--')
        labels=ax8.get_xticklabels()
        for label in labels:
            label.set_visible(False)
        ax8.set_ylabel('r Max='+str(abs(round(M,3))))

        #================================================================================================================================
        ax4=plt.subplot(gs[8,:],sharex=ax1)
        ax4.plot(spot_df['dates'],spot_df['Count_F4'],'y*-')
        axes=plt.gca()
        DumLim=axes.get_ylim()
        ax4.plot(ARR,np.array([DumLim[0],DumLim[1]]),'k-',linewidth=2.0)
        ax4.grid(b=True,which='major',color='k',linestyle='--')
        labels=ax4.get_xticklabels()
        for label in labels:
            label.set_visible(False)

        ax9=plt.subplot(gs[9,:],sharex=ax1)
        ax9.plot(Time,corr4)
        M=min(corr4)
        I=np.argmin(corr4)
        ax9.plot(Time[I],M,'b*')
        axes=plt.gca()
        DumLim=axes.get_ylim()
        ax9.plot(ARR,np.array([DumLim[0],DumLim[1]]),'k-',linewidth=2.0)
        ax9.grid(b=True,which='major',color='k',linestyle='--')
        labels=ax9.get_xticklabels()
        for label in labels:
            label.set_visible(False)
        ax9.set_ylabel('r Max='+str(abs(round(M,3))))

        #===============================================================================================================================
        ax5=plt.subplot(gs[10,:],sharex=ax1)
        ax5.plot(spot_df['dates'], spot_df['Count_F5'],'g*-')
        axes=plt.gca()
        DumLim=axes.get_ylim()
        ax5.plot(ARR,np.array([DumLim[0],DumLim[1]]),'k-',linewidth=2.0)
        ax5.grid(b=True,which='major',color='k',linestyle='--')
        labels=ax5.get_xticklabels()
        for label in labels:
            label.set_visible(False)


        ax10=plt.subplot(gs[11,:],sharex=ax1)
        ax10.plot(Time,corr5)
        M=min(corr5)
        I=np.argmin(corr5)
        ax10.plot(Time[I],M,'b*')
        axes=plt.gca()
        DumLim=axes.get_ylim()
        ax10.plot(ARR,np.array([DumLim[0],DumLim[1]]),'k-',linewidth=2.0)
        ax10.grid(b=True,which='major',color='k',linestyle='--')
        #import ipdb;ipdb.set_trace()
        T=Hour-8
        labels=[item.get_text() for item in ax10.get_xticklabels()]
        for I in range(0,len(labels)):
            labels[I]=str(T)+':00'
            T+=1
        ax10.set_xticklabels(labels)
        labels=ax10.get_xticklabels()
        for label in labels:
            label.set_rotation(30)
        ax10.set_ylabel('r Max='+str(abs(round(M,3))))
      

        #ax0.set_yticks(np.arange(min(Mag)/2, max(Mag)*1.2, (max(Mag)-min(Mag))/4))
        #ax1.set_yticks(np.arange(min(spot_df['Count_F1'])/2,max(spot_df['Count_F1'])*1.2,(max(spot_df['Count_F1'])-min(spot_df['Count_F1']))/5))

        #ax1.set_yticks(np.arange(200,19400,200))
        #ax2.set_yticks(np.arange(400,1200,200))
        #ax3.set_yticks(np.arange(0,350,50))



        #================================================================================================================================
        #Set subplot and axis labels
        ax0.set_title('RBN Spots with GOES Data for '+str(Hour)+':'+str(Min)+' Flare\n'+sTime.strftime('%d %b %Y %H%M UT - ')+eTime.strftime('%d %b %Y %H%M UT'),fontsize=28)
        ax0.set_xlabel('')
        ax1.set_ylabel('[NR. Spots]\n'+str(freq1/1000)+' MHz')
        ax2.set_ylabel('[NR. Spots]\n'+str(freq2/1000)+' MHz')
        ax3.set_ylabel('[NR. Spots]\n'+str(freq3/1000)+' MHz')
        ax4.set_ylabel('[NR. Spots]\n'+str(freq4/1000)+' MHz')
        ax5.set_ylabel('[NR. Spots]\n'+str(freq5/1000)+' MHz')
        ax10.set_xlabel('Time, Central Time (UTC-6)',fontsize=28)

        #labels = ax3.get_xticklabels()
        #for label in labels:
        #        label.set_rotation(30) 
        #ax1.set_xlabel('Time [UT]')


        #ax0.set_xticklabels([])
        #ax1.set_xticklabels([])
        #ax2.set_xticklabels([])
        #ax0.xaxis.set_visible(False)
        #ax1.xaxis.set_visible(False)
        #ax2.xaxis.set_visible(False)
        #plt.legend(['3 MHz','14 MHz','28 MHz'])

        #ax.text(spot_df.dates.min(),spot_df.Count.min(),'Unit Time: '+str(dt)+' '+unit)
        #ax.text(spot_df.dates[10],spot_df.Count.max(),'Unit Time: '+str(dt)+' '+unit)
        #fig.text(0.06, 0.5, '[NR. Spots]', ha='center', va='center', rotation='vertical')

        fig.tight_layout()
        filename=os.path.join(output_dir, graphfile)
        # 'rbnCount_5min_line1.png')
        fig.savefig(filename)

        #count=np.ones((len(df1['date']),1))

        #import ipdb; ipdb.set_trace()
