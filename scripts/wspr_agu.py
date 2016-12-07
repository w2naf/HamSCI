#!/usr/bin/env python
#Codes to make graphs for AGU. Also to test wspr_lib prior to AGU 2016

from hamsci import wspr_lib
#from hamsci import gridsquare
from hamsci import rbn_lib

import numpy as np
import pandas as pd 
from matplotlib import pyplot as plt

def find_pair(df,prefix='', prefix2=''):
    import pandas as pd

    print 'Finding calls\n'
    #Find 1st region's calls
    calls=wspr_lib.calls_by_grid(df, prefix=prefix,col_call='call_sign')
    df_calls=pd.DataFrame({'calls':calls})
#    calls=np.unique(calls)
    del calls
    calls=df_calls.calls.unique()
    import ipdb; ipdb.set_trace()
    del df_calls
    import ipdb; ipdb.set_trace()
    
#    reciever=wspr_lib.calls_by_grid(df, prefix=prefix,col_call='reporter')
    ref_tx=0
    ref_rx=0
    tx_call=[]
    print 'Checking Region 1 Callsigns'
    for callsign in calls:
#        tx  =  df['call_sign'][df['call_sign']==callsign]
#        num_tx = len(tx)
        num_tx  =   len(df['call_sign'][df['call_sign']==callsign])
        
        #Check given tx station recieving stations in 2nd specified region
        rx  =   df[df['reporter']==callsign]
#        rx  =   rx['call_sign'].unique()
        num_rx = len(rx)
        rx2 = wspr_lib.calls_by_grid(rx, prefix=prefix2, col_call='call_sign')
        df_calls=pd.DataFrame({'calls':rx2})
        del rx2
        rx2=df_calls.calls.unique()
        num_rx2=len(rx2)
        if callsign == 'W2GRK':
            import ipdb; ipdb.set_trace()

        if ref_tx < num_tx and not df_calls.empty:
            if ref_rx<num_rx2:
                tx_call=[]
                rx2_calls=[]
                ref_tx=num_tx
                tx_call.append(callsign)
                rx2_calls.append(rx2)
                print 'New Max'
                print callsign
                print str(num_tx)+'\n'
            
        elif ref_tx == num_tx and not df_calls.empty:
            tx_call.append(callsign)
            rx2_calls.append(rx2)
            print 'Multiple!!!'
            print callsign
            print str(num_tx)+'\n'
        del df_calls

    return tx_call, ref_tx, rx2_calls

def find_links(df,prefix='', prefix2=''):
    import pandas as pd
    print 'Finding calls\n'
    #Find 1st region's calls
    calls=wspr_lib.calls_by_grid(df, prefix=prefix,col_call='call_sign')
    df_calls=pd.DataFrame({'calls':calls})
    del calls
    tx_calls=df_calls.calls.unique()
    del df_calls
    import ipdb; ipdb.set_trace()
    
    print 'Checking Region 1 Callsigns'
    rx_calls=[]
    for callsign in tx_calls:
        #Check given tx station recieving stations in 2nd specified region
        rx  =   df[df['reporter']==callsign]
        rx2 = wspr_lib.calls_by_grid(rx, prefix=prefix2, col_call='call_sign')
        df_calls=pd.DataFrame({'calls':rx2})
        del rx2
        rx2=df_calls.calls.unique()
        for rxcall in rx2:
            rx_calls.append(rxcall)
    return tx_calls, rx_calls


#def grid_links(df,prefix='', prefix2='')
#    import numpy as np
#    tx1,rx1=find_links(df,prefix,prefix2)
#    tx2,rx2=find_links(df,prefix2,prefix)
#
#    flag=True
#    for tx in tx1:
#        for rx in rx1:
#            if flag:
#                df2=df[np.logical_or(np.logical_and(df['call_sign']==tx1, df['reporter']==rx
#                flag=False
#

#    rx_count=0
#    tx_count=0
#    for this_tx in tx:
#        for this_rx in rx:
#            tx

def plot_wspr_snr(df, fig=None, ax=None, by_pwr=True, loc_col='grid',x_unit='est', legend=True, raw_time=False):
    """Scatter Plot WSPR SNR reports

    Parameters
    ----------
    new_data_set : str
        Name for the new data_set object.
    comment : str
        Comment describing the new data_set object.

    Returns
    -------
    new_data_set_obj : data_set 
        Copy of the original data_set with new name and history entry.

    Written by Magdalina L. Moses, Fall 2016
    """
    from matplotlib import pyplot as plt

    #Get locations and create string for title
    location=df[loc_col].unique()
    str_location = ''
    for locality in location: 
        str_location    =  str_location + str(locality) + '_'
    str_location=str_location[0:len(str_location)-1]

#    #Convert to local time if desired
#    if x_unit == 'est':
#      #really should have eastern time
#            for time in df.timestamp.unique():
##                dt=datetime.timedelta(hours=4)
##                df=df.replace({'timestamp':{time: time-dt}})
#                if np.datetime64(2016, 11,6) < time:
#                    dt=datetime.timedelta(hours=5)
#                else:             
#                    dt=datetime.timedelta(hours=4)
#                df=df.replace({'timestamp':{time: time-dt}})

    #Get hours
    try:
        test = df['hour']
        del test
    except:
        df=wspr_lib.find_hour(df)
    for hour in df.hour.unique():
        df=df.replace({'hour':{hour: (hour-4)}})

    df=df.replace({'hour':{-5: (24-5)}})
    df=df.replace({'hour':{-4: (24-4)}})
    df=df.replace({'hour':{-3: (24-3)}})
    df=df.replace({'hour':{-2: (24-2)}})
    df=df.replace({'hour':{-1: (24-1)}})

    lstyle=('solid', 'dashed')
    xsize=8
    ysize=4

    #Get Band Data
    band_data = rbn_lib.BandData()
    band_list = band_data.band_dict.keys()
    band_list.sort()


    #Will likely need to bin powers, but need to check how far appart the different powers are
    #Plot
    if by_pwr:
#        grouped     = df.groupby('power')
#        if fig == None:
#            fig = plt.figure()
#        if ax == None:
#            ax=fig.add_subplot(111)
        df = df.sort(columns='power', ascending=False)
#        df = df.sort(columns='hour')
        pwr_grouped     = df.groupby('power')
        import ipdb; ipdb.set_trace()
        if 4 <= len(pwr_grouped):
            ny_plots=4
            nx_plots=len(pwr_grouped)/4
            if len(pwr_grouped) % 4 != 0:
                nx_plots =nx_plots+1

        else:
            nx_plots=1
            ny_plots=len(pwr_grouped) 

        if fig==None: 
            fig        = plt.figure(figsize=(nx_plots*xsize,ny_plots*ysize))

        inx=1
        for pwr in df.power.unique():
            pwr_group=pwr_grouped.get_group(pwr)
#            df = df.sort(columns='hour')
            pwr_group = pwr_group.sort(columns='hour')


            grouped=pwr_group.groupby('band')
            print 'Index is: '+str(inx+3)
            ax         = fig.add_subplot(ny_plots, nx_plots,inx)
            for band in band_list:
                try:
                    this_group = grouped.get_group(band)
                except:
                    continue

#                    ax=fig.add_subplot(len(),1,inx)

                color       = band_data.band_dict[band]['color']
                label       = band_data.band_dict[band]['freq_name']

                label1=label+ location[0] + ' to ' + location[1] #' NJ to VA'
                label2=label+ location [1] + ' to ' + location[0] #' VA to NJ'

                tx=this_group[this_group[loc_col]==location[0]]
                tx2=this_group[this_group[loc_col]==location[1]]

                if location[0]=='FN20':
                    marker1='*'
                    marker2='o'
                    lstyle1,lstyle2=lstyle
                else:
                    marker1='o'
                    marker2='*'
                    lstyle2,lstyle1=lstyle
                line1=ax.plot(tx.hour, tx.snr,color=color, marker=marker1, label=label1, linestyle=lstyle1)
                line2=ax.plot(tx2.hour, tx2.snr,color=color, marker=marker2, label=label2, linestyle=lstyle2)
                if legend: ax.legend()
                plt.title(str(pwr)+  ' W (between '+ str_location +')')
                ax.set_ylabel('SNR')
            inx=inx+1
            plt.close()
    else:
#        if by_grid == True:
#            nx_plots=1
#            ny_plots=2
#        else:
#            nx_plots=1
#            ny_plots=1
        nx_plots=1
        ny_plots=1
        if fig==None: 
            fig        = plt.figure(figsize=(nx_plots*xsize,ny_plots*ysize))

        ax         = fig.add_subplot(ny_plots, nx_plots,1)

        df=df.sort(columns='hour')
        grouped=df.groupby('band')
        for band in band_list:
            try:
                this_group = grouped.get_group(band)
            except:
                continue

#                    ax=fig.add_subplot(len(),1,inx)

            color       = band_data.band_dict[band]['color']
            label       = band_data.band_dict[band]['freq_name']

            label1=label+ location[0] + ' to ' + location[1] #' NJ to VA'
            label2=label+ location [1] + ' to ' + location[0] #' VA to NJ'
#            label1=label+ ' NJ to VA'
#            label2=label+ ' VA to NJ'

            tx=this_group[this_group[loc_col]==location[0]]
            tx2=this_group[this_group[loc_col]==location[1]]
            if raw_time:
                time='timestamp'
            else:
                time = 'hour'
            tx=tx.sort(time)
            tx2=tx2.sort(time)

            if location[0]=='FN20':
                marker1='*'
                marker2='o'
                lstyle1,lstyle2=lstyle
            else:
                marker1='o'
                marker2='*'
                lstyle2,lstyle1=lstyle
#            line1=ax.plot(tx.hour, tx.snr,color=color, marker=marker1, label=label1, linestyle=lstyle1)
#            line2=ax.plot(tx2.hour, tx2.snr,color=color, marker=marker2, label=label2, linestyle=lstyle2)
        
            line1=ax.plot(tx[time], tx.snr,color=color, marker=marker1, label=label1, linestyle=lstyle1)
            line2=ax.plot(tx2[time], tx2.snr,color=color, marker=marker2, label=label2, linestyle=lstyle2)
            if legend: ax.legend()
            plt.title('Between '+ str_location)
            ax.set_ylabel('SNR (W)')
        plt.close()
    ax.set_xlabel('Time (EST)')
    return fig


def average_dB(df, col='snr'):

    sn=np.power(10,df[col]/10)
    avg=sn.mean()
#    df=df.drop(col, axis=1)
#    df[col] = 10*np.log10(avg)
    #Could make this function just return the value and not put it in the dataframe yet!
    avg_dB = 10*np.log10(avg)
    
    return avg_dB

def snr_avg(df):
    mybands=[]
    mytime=[]
    myavg=[]
    mygrid=[]

    #Get Band Data
    band_data = rbn_lib.BandData()
    band_list = band_data.band_dict.keys()
    band_list.sort()

    for gridsq in df['grid'].unique():
        df_temp=df[df['grid']==gridsq]
        grouped=df_temp.groupby('band')
        
        for band in band_list:
            try:
                this_group = grouped.get_group(band)
            except:
                continue

            hour_grouped = this_group.groupby('hour')
            for hr in this_group.hour.unique():
                this_hour= hour_grouped.get_group(hr)
                avg_param = this_hour.snr.mean()

                mybands.append(band)
                mytime.append(hr)
                myavg.append(avg_param)
                mygrid.append(gridsq)

    #Save Values in new dataframe
    df_avg=pd.DataFrame({'hour':mytime, 'band':mybands, 'snr':myavg, 'grid':mygrid})
    import ipdb; ipdb.set_trace()
    return df_avg

def wspr_avg(df, t_div='hour', param='snr', groups=['band', 'tx_pwr'], bining=None):
    """Average Values over a given period of time  

    Parameters
    ----------
    df : Pandas Dataframe
        Dataframe containing test data.

    Returns
    -------
    new_data_set_obj : data_set 
        Copy of the original data_set with new name and history entry.

    Written by Magda Moses, Fall 2016
    """

#    #Group and Sort by time
#    srt = df.sort([t_div])
#    grouped = srt.groupby(t_div)

    mybands=[]
    mypwr=[]
    mytime=[]
    myavg=[]
    mygrid=[]

    for gridsq in df['grid'].unique():
        df_temp=df[df['grid']==gridsq]

        #Group and Sort by time
        srt = df_temp.sort([t_div])
        grouped = srt.groupby(t_div)

        for time in df_temp[t_div].unique():
            time_group=grouped.get_group(time)
            
            pwr_grouped=time_group.groupby('power')

            for pwr in time_group['power'].unique():
#                try:
                pwr_group=pwr_grouped.get_group(pwr)
                band_grouped=pwr_group.groupby('band')
#                except:
#                    continue
#                try:
                for band in pwr_group['band'].unique():
                    band_group=band_grouped.get_group(band)
#                    avg_param = wspr_lib.average_dB(band_group, col=param)
                    avg_param = average_dB(band_group, col=param)

                    mybands.append(band)
                    mypwr.append(pwr)
                    mytime.append(time)
                    myavg.append(avg_param)
                    mygrid.append(gridsq)
#                except: 
#                    continue
        import ipdb; ipdb.set_trace()

    #Save Values in new dataframe
    df_avg=pd.DataFrame({'hour':mytime, 'band':mybands, 'power':mypwr, param:myavg, 'grid':mygrid})
    import ipdb; ipdb.set_trace()
    return df_avg

def plot_avg_snr(df, fig=None, ax=None, by_pwr=True, loc_col='grid',x_unit='est', legend=True):
    """Scatter Plot WSPR SNR reports

    Parameters
    ----------
    new_data_set : str
        Name for the new data_set object.
    comment : str
        Comment describing the new data_set object.

    Returns
    -------
    new_data_set_obj : data_set 
        Copy of the original data_set with new name and history entry.

    Written by Magdalina L. Moses, Fall 2016
    """
    from matplotlib import pyplot as plt

    #Get locations and create string for title
    location=df[loc_col].unique()
    str_location = ''
    for locality in location: 
        str_location    =  str_location + str(locality) + '_'
    str_location=str_location[0:len(str_location)-1]

    #Convert to local time if desired
    if x_unit == 'est':
      #really should have eastern time
#            for time in df.timestamp.unique():
#                dt=datetime.timedelta(hours=4)
#                df=df.replace({'timestamp':{time: time-dt}})

#                if np.datetime64(2016, 11,6) < time:
#                    dt=datetime.timedelta(hours=5)
#                else:             
#                    dt=datetime.timedelta(hours=4)
#                df=df.replace({'timestamp':{time: time-dt}})
#        #Get hours
#        df=wspr_lib.find_hour(df)
        for hour in df.hour.unique():
            df=df.replace({'hour':{hour: (hour-4)}})

        df=df.replace({'hour':{-5: (24-5)}})
        df=df.replace({'hour':{-4: (24-4)}})
        df=df.replace({'hour':{-3: (24-3)}})
        df=df.replace({'hour':{-2: (24-2)}})
        df=df.replace({'hour':{-1: (24-1)}})

    lstyle=('solid', 'dashed')
    xsize=8
    ysize=4
    #Will likely need to bin powers, but need to check how far appart the different powers are
    #Plot
    if by_pwr:
#        grouped     = df.groupby('power')
#        if fig == None:
#            fig = plt.figure()
#        if ax == None:
#            ax=fig.add_subplot(111)
        df = df.sort(columns='power', ascending=False)
#        df = df.sort(columns='hour')
        pwr_grouped     = df.groupby('power')
        if 4 <= len(pwr_grouped):
            ny_plots=4
            nx_plots=len(pwr_grouped)/4
            if len(pwr_grouped) % 4 != 0:
                nx_plots =nx_plots+1

        else:
            nx_plots=1
            ny_plots=len(pwr_grouped) 

        if fig==None: 
            fig        = plt.figure(figsize=(nx_plots*xsize,ny_plots*ysize))

        #Get Band Data
        band_data = rbn_lib.BandData()
        band_list = band_data.band_dict.keys()
        band_list.sort()

        row=1
        col=0
        for pwr in df.power.unique():
#            if row % ny_plots ==0 :
            if row % 3 ==0 :
                import ipdb; ipdb.set_trace()
                col=col+1
                row = 1
            if row ==1:
                inx=row+1
            else:
                inx=row+col+3
#            inx=row+col
            pwr_group=pwr_grouped.get_group(pwr)
#            df = df.sort(columns='hour')
            pwr_group = pwr_group.sort(columns='hour')


            grouped=pwr_group.groupby('band')
            print 'Index is: '+str(inx)
                
            ax         = fig.add_subplot(ny_plots, nx_plots,inx)
            for band in band_list:
                try:
                    this_group = grouped.get_group(band)
                except:
                    continue

#                    ax=fig.add_subplot(len(),1,inx)

                color       = band_data.band_dict[band]['color']
                label       = band_data.band_dict[band]['freq_name']

                label1=label+ location[0] + ' to ' + location[1] #' NJ to VA'
                label2=label+ location [1] + ' to ' + location[0] #' VA to NJ'
#                label1=label+ ' NJ to VA'
#                label2=label+ ' VA to NJ'

                tx=this_group[this_group[loc_col]==location[0]]
                tx2=this_group[this_group[loc_col]==location[1]]

                if location[0]=='FN20':
                    marker1='*'
                    marker2='o'
                    lstyle1,lstyle2=lstyle
                else:
                    marker1='o'
                    marker2='*'
                    lstyle2,lstyle1=lstyle
                line1=ax.plot(tx.hour, tx.snr,color=color, marker=marker1, label=label1, linestyle=lstyle1)
                line2=ax.plot(tx2.hour, tx2.snr,color=color, marker=marker2, label=label2, linestyle=lstyle2)
#                ax.legend()
                plt.title(str(pwr)+  ' W (between '+ str_location +')')
                ax.set_ylabel('Average SNR (dB)')
            row=row+1
    else:
        nx_plots=1
        ny_plots=1
        if fig==None: 
            fig        = plt.figure(figsize=(nx_plots*xsize,ny_plots*ysize))

        #Get Band Data
        band_data = rbn_lib.BandData()
        band_list = band_data.band_dict.keys()
        band_list.sort()

        df=df.sort(columns='hour')
        grouped=df.groupby('band')
            
        ax         = fig.add_subplot(ny_plots, nx_plots,1)
        for band in band_list:
            try:
                this_group = grouped.get_group(band)
            except:
                continue

            color       = band_data.band_dict[band]['color']
            label       = band_data.band_dict[band]['freq_name']

#            label1=label+ ' NJ to VA'
#            label2=label+ ' VA to NJ'
            label1=label+ location[0] + ' to ' + location[1] #' NJ to VA'
            label2=label+ location [1] + ' to ' + location[0] #' VA to NJ'

            tx=this_group[this_group[loc_col]==location[0]]
            tx2=this_group[this_group[loc_col]==location[1]]

            if location[0]=='FN20':
                marker1='*'
                marker2='o'
                lstyle1,lstyle2=lstyle
            else:
                marker1='o'
                marker2='*'
                lstyle2,lstyle1=lstyle
            line1=ax.plot(tx.hour, tx.snr,color=color, marker=marker1, label=label1, linestyle=lstyle1)
            line2=ax.plot(tx2.hour, tx2.snr,color=color, marker=marker2, label=label2, linestyle=lstyle2)
            if legend: ax.legend()
            plt.title('Between '+ str_location)
            ax.set_ylabel('Average SNR (W)')
    
    ax.set_xlabel('Time (EST)')
    return fig


#def plot_wspr_snr(df, fig=None, ax=None, by_pwr=True, loc_col='grid',x_unit='est'):
#        """Scatter Plot WSPR SNR reports
#
#        Parameters
#        ----------
#        new_data_set : str
#            Name for the new data_set object.
#        comment : str
#            Comment describing the new data_set object.
#
#        Returns
#        -------
#        new_data_set_obj : data_set 
#            Copy of the original data_set with new name and history entry.
#
#        Written by Magdalina L. Moses, Fall 2016
#        """
#        from matplotlib import pyplot as plt
#
#        location=df[loc_col].unique()
#        df=wspr_lib.find_hour(df)
#        if x_unit == 'est':
#          #really should have eastern time
#          for hour in df.hour.unique():
#              df=df.replace({'hour':{hour: (hour-4)}})
##            df1=sounder_lib.manip_time(df[df.timestamp<datetime.datetime(2016, 11,6)],t_delta=4)
##            df2=sounder_lib.manip_time(df[df.timestamp>datetime.datetime(2016, 11,6)],t_delta=5)
##            del df
##            df=pd.concat([df1,df2])
##            del df1
##            del df2
#
#        #Will likely need to bin powers, but need to check how far appart the different powers are
#        #Plot
##        if by_pwr:
#        lstyle=('solid', 'dashed')
#        if fig == None:
#            fig = plt.figure()
#        if ax == None:
#            ax=fig.add_subplot(111)
#        df = df.sort(columns='hour')
#        grouped=df.groupby('band')
#
#        band_data = rbn_lib.BandData()
#        band_list = band_data.band_dict.keys()
#        band_list.sort()
#        import ipdb; ipdb.set_trace()
#
#        inx=1
#        for band in band_list:
#            try:
#                this_group = grouped.get_group(band)
#            except:
#                continue
#
#            inx=inx+1
#            color       = band_data.band_dict[band]['color']
#            label       = band_data.band_dict[band]['freq_name']
#
#            label1=label+ ' NJ to VA'
#            label2=label+ ' VA to NJ'
#
#            tx=df[df[loc_col]==location[0]]
#            tx2=df[df[loc_col]==location[1]]
#            if location[0]=='FN20':
#                marker1='*'
#                marker2='o'
#                lstyle1,lstyle2=lstyle
#            else:
#                marker1='o'
#                marker2='*'
#                lstyle2,lstyle1=lstyle
#            line1=ax.plot(tx.hour, tx.snr,color=color, marker=marker1, label=label1, linestyle=lstyle1)
#            line2=ax.plot(tx2.hour, tx2.snr,color=color, marker=marker2, label=label2, linestyle=lstyle2)
#            ax.legend()
#            ax.set_ylabel('SNR')
#        
#        ax.set_xlabel('Time (EST')
#        return fig

def run_plot(df_filt, gridsq, sTime, eTime, note=None):
    import datetime
    import os
    #Plot figure 
#    fig=plot_wspr_snr(df_filt)
#    fig=plot_avg_snr(df_filt, by_pwr=False)
    fig=plot_wspr_snr(df_filt, by_pwr=False, legend=False)

    output_dir=os.path.join('output', 'wspr')
    if note:
        output_file= 'wspr_test'+'_'+note+'_'+gridsq[0]+'_'+gridsq[1]+'_'+sTime.strftime('%Y%m%d_')+eTime.strftime('%Y%m%d')+'.png'
    else:
        output_file= 'wspr_test'+'_'+gridsq[0]+'_'+gridsq[1]+'_'+sTime.strftime('%Y%m%d_')+eTime.strftime('%Y%m%d')+'.png'
    output_path=os.path.join(output_dir,output_file)
    if not os.path.exists(output_path):
         try:    # Create the output directory, but fail silently if it already exists
             os.makedirs(output_dir) 
         except:
             pass

    fig.savefig(output_path)
    return fig


if __name__ == '__main__':
    import datetime
    import os
    import sys
    import pickle
    import copy 

    sTime       = datetime.datetime(2014,2,1)
    eTime       = datetime.datetime(2014,2,28)

    sTime       = datetime.datetime(2016,8,27)
    eTime       = datetime.datetime(2016,8,28)
    sTime       = datetime.datetime(2016,11,11)
    eTime       = datetime.datetime(2016,11,18)

#    sTime       = datetime.datetime(2016,10,1,0)
#    eTime       = datetime.datetime(2016,10,31,23,59, 59)

    sTime       = datetime.datetime(2016,11,1,0)
    sTime       = datetime.datetime(2016,10,1,0)
    eTime       = datetime.datetime(2016,11,2,0)
#    eTime       = datetime.datetime(2016,11,17,0)
#    eTime       = datetime.datetime(2016,11,30,0)
    eTime       = datetime.datetime(2016,12,1,0)

#    sTime       = datetime.datetime(2016,12,1,0)
##    eTime       = datetime.datetime(2016,12,5,0)
#    eTime       = datetime.datetime(2016,12,6,0)
    data_dir    = 'data/wspr' 

    #Select only stations within two lat/lon areas (near VT and NJIT)
    #   K2MFF 'FN20vr' (40.7429,-74.1770)
    #   KM4EGE 'EM97tf' 

    #For simplicity in this proof-of-concept application, only chose stations in the following gridsquares:
    #   FN20 and FN21 (or FN30 and FN31)
    #   Need to select from wider area for southern station 
    gridsq=['FN20', 'EM97']
    gridsq=['FN20', 'EM98']

    #Test Code for VM
    print str(sys.argv[1])
    if str(sys.argv[1]) == 'usePickle': 
        import ipdb; ipdb.set_trace()
        p_dir='data/wspr/filtered_wspr'
        p_filename = 'wspr_'+gridsq[0]+'-'+gridsq[1]+'_'+sTime.strftime('%Y%m%d-')+eTime.strftime('%Y%m%d.p')
        p_filepath = os.path.join(p_dir,p_filename)
        print p_filepath
        with open(p_filepath,'rb') as fl:
            df_filt = pickle.load(fl)
        import ipdb; ipdb.set_trace()

    if str(sys.argv[1]) == 'useFile': 
        p_dir='data/wspr/filtered_wspr'
        p_filename = 'wspr_'+gridsq[0]+'-'+gridsq[1]+'_'+sTime.strftime('%Y%m%d-')+eTime.strftime('%Y%m%d.csv')
        p_filepath = os.path.join(p_dir,p_filename)
        print p_filepath
        df_filt=pd.read_csv(p_filepath)
        import ipdb; ipdb.set_trace()
        df_filt['timestamp']=df_filt.timestamp.astype(datetime.datetime)

    #Original Code
    elif str(sys.argv[1]) == 'original':
#        import ipdb; ipdb.set_trace()
        #    df = wspr_lib.read_wspr(sTime,eTime,data_dir, overwrite=True)
        df = wspr_lib.read_wspr(sTime,eTime,data_dir)
#        import ipdb; ipdb.set_trace()
       
    #    #Find the pairs of stations with most links between them
    #    tx, num_tx, rx= find_pair(df, prefix='FN20', prefix2='EM95')

    #    #Found stations were KK4WJF and K3EA
    #    stations = ['KK4WJF', 'K3EA']
    #    stations = [tx[0], rx[0][0]]
    #    #Filter to only include links between two specified stations
    #    df_filt  =   wspr_lib.select_pair(df, stations)
        
        #Filter to only include links between stations in specific grid sqares
#        df_filt=wspr_lib.filter_grid_pair(df, gridsq, redef=True, precision=4) 
        df_filt=wspr_lib.filter_grid_pair(df, gridsq)
        df_filt=wspr_lib.redefine_grid(df_filt, precision=4)
#        #Redefine grid and filter by gridsquare
#        df=wspr_lib.redefine_grid(df, precision=4)
#        df_filt=wspr_lib.filter_grid_pair(df, gridsq, redef=False, precision=4)


    #    fig=plot_wspr_snr(df[df.power==30])
    df_filt = wspr_lib.dB_to_Watt(df_filt)
    #Plot figure 
#    fig=plot_wspr_snr(df_filt, by_pwr=False, legend=False)
    fig=plot_wspr_snr(df_filt, by_pwr=False, legend=True)
#    fig=plot_wspr_snr(df_filt, by_pwr=False, legend=False, raw_time=True)

    df_avg = snr_avg(df_filt)
    fig2=plot_avg_snr(df_avg, by_pwr=False, legend=False)

#  Need to write code to plot raw times 

    output_file= 'wspr_test'+'_'+'plot'+'_'+gridsq[0]+'_'+gridsq[1]+'_'+sTime.strftime('%Y%m%d_')+eTime.strftime('%Y%m%d')+'.png'
    output_file2= 'wspr_test'+'_'+'avg'+'_'+gridsq[0]+'_'+gridsq[1]+'_'+sTime.strftime('%Y%m%d_')+eTime.strftime('%Y%m%d')+'.png'

##    note = str(sys.argv[2])
#    if note:
#        output_file= 'wspr_test'+'_'+note+'_'+gridsq[0]+'_'+gridsq[1]+'_'+sTime.strftime('%Y%m%d_')+eTime.strftime('%Y%m%d')+'.png'
#    else:
#        output_file= 'wspr_test'+'_'+gridsq[0]+'_'+gridsq[1]+'_'+sTime.strftime('%Y%m%d_')+eTime.strftime('%Y%m%d')+'.png'
    output_dir=os.path.join('output', 'wspr')
    output_path=os.path.join(output_dir, output_file)
    output_path2=os.path.join(output_dir, output_file2)
#    output_path=os.path.join(output_dir, 'wspr_test'+gridsq[0]+'_'+gridsq[1]+'_'+sTime.strftime('%d%b%Y%H%MUT-')+eTime.strftime('%d%b%Y%H%MUT')+'.png')
    if not os.path.exists(output_path):
         try:    # Create the output directory, but fail silently if it already exists
             os.makedirs(output_dir) 
         except:
             pass

    fig.savefig(output_path)
    fig2.savefig(output_path2)
    import ipdb; ipdb.set_trace()

#   #Plot second 
#    df_filt=wspr_lib.filter_grid_pair(df, ['FN20', 'EM96']) 
#    gridsq=['FN20', 'EM96']
#    fig=plot_wspr_snr(df_filt)
#    output_dir=os.path.join('output', 'wspr')
#    output_path=os.path.join(output_dir, 'wspr_test'+gridsq[0]+'_'+gridsq[1]+'.png')
#    fig.savefig(output_path)
    import ipdb; ipdb.set_trace()
