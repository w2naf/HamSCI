#!/usr/bin/env python
dxf_prop        = {'marker':'*','color':'blue'}
dxf_leg_size    = 150
dxf_plot_size   = 50

def read_rbn(sTime,eTime=None,data_dir=None,
             qrz_call='km4ege',qrz_passwd='ProjectEllie_2014'):
    import os               # Provides utilities that help us do os-level operations like create directories
    import datetime         # Really awesome module for working with dates and times.
    import zipfile
    import urllib2          # Used to automatically download data files from the web.
    import pickle

    import numpy as np      #Numerical python - provides array types and operations
    import pandas as pd     #This is a nice utility for working with time-series type data.

    from hamtools import qrz

    #import ipdb; ipdb.set_trace()
    if data_dir is None: data_dir = os.getenv('DAVIT_TMPDIR')

    qz      = qrz.Session(qrz_call,qrz_passwd)

    ymd_list    = [datetime.datetime(sTime.year,sTime.month,sTime.day)]
    eDay        =  datetime.datetime(eTime.year,eTime.month,eTime.day)
    while ymd_list[-1] < eDay:
        ymd_list.append(ymd_list[-1] + datetime.timedelta(days=1))

    for ymd_dt in ymd_list:
        ymd         = ymd_dt.strftime('%Y%m%d')
        data_file   = '{0}.zip'.format(ymd)
        data_path   = os.path.join(data_dir,data_file)  

        time_0      = datetime.datetime.now()
        print 'Starting RBN processing on <%s> at %s.' % (data_file,str(time_0))

        ################################################################################
        # Make sure the data file exists.  If not, download it and open it.
        if not os.path.exists(data_path):
             try:    # Create the output directory, but fail silently if it already exists
                 os.makedirs(data_dir) 
             except:
                 pass
             # File downloading code from: http://stackoverflow.com/questions/22676/how-do-i-download-a-file-over-http-using-python
             url = 'http://www.reversebeacon.net/raw_data/dl.php?f='+ymd

             #import ipdb; ipdb.set_trace()
             u = urllib2.urlopen(url)
             f = open(data_path, 'wb')
             meta = u.info()
             file_size = int(meta.getheaders("Content-Length")[0])
             print "Downloading: %s Bytes: %s" % (data_path, file_size)
         
             file_size_dl = 0
             block_sz = 8192
             while True:
                 buffer = u.read(block_sz)
                 if not buffer:
                     break
         
                 file_size_dl += len(buffer)
                 f.write(buffer)
                 status = r"%10d  [%3.2f%%]" % (file_size_dl, file_size_dl * 100. / file_size)
                 status = status + chr(8)*(len(status)+1)
                 print status,
             f.close()
             status = 'Done downloading!  Now converting to Pandas dataframe and plotting...'
             print status

        p_filename = 'rbn_'+sTime.strftime('%Y%m%d%H%M-')+eTime.strftime('%Y%m%d%H%M.p')
        p_filepath = os.path.join(data_dir,p_filename)
        if not os.path.exists(p_filepath):
            # Load data into dataframe here. ###############################################
            with zipfile.ZipFile(data_path,'r') as z:   #This block lets us directly read the compressed gz file into memory.  The 'with' construction means that the file is automatically closed for us when we are done.
                with z.open(ymd+'.csv') as fl:
                    df          = pd.read_csv(fl,parse_dates=[10])

            # Create columns for storing geolocation data.
            df['dx_lat'] = np.zeros(df.shape[0],dtype=np.float)*np.nan
            df['dx_lon'] = np.zeros(df.shape[0],dtype=np.float)*np.nan
            df['de_lat'] = np.zeros(df.shape[0],dtype=np.float)*np.nan
            df['de_lon'] = np.zeros(df.shape[0],dtype=np.float)*np.nan

            # Trim dataframe to just the entries we need.
            df = df[np.logical_and(df['date'] >= sTime,df['date'] < eTime)]

            # Look up lat/lons in QRZ.com
            errors  = 0
            success = 0
            for index,row in df.iterrows():
                if index % 50   == 0:
                    print index,datetime.datetime.now()-time_0,row['date']
                de_call = row['callsign']
                dx_call = row['dx']
                try:
                    de      = qz.qrz(de_call)
                    dx      = qz.qrz(dx_call)

                    row['de_lat'] = de['lat']
                    row['de_lon'] = de['lon']
                    row['dx_lat'] = dx['lat']
                    row['dx_lon'] = dx['lon']
                    df.loc[index] = row
    #                print '{index:06d} OK - DX: {dx} DE: {de}'.format(index=index,dx=dx_call,de=de_call)
                    success += 1
                except:
    #                print '{index:06d} LOOKUP ERROR - DX: {dx} DE: {de}'.format(index=index,dx=dx_call,de=de_call)
                    errors += 1
            total   = success + errors
            pct     = success / float(total) * 100.
            print '{0:d} of {1:d} ({2:.1f} %) call signs geolocated via qrz.com.'.format(success,total,pct)
            df.to_pickle(p_filepath)
        else:
            with open(p_filepath,'rb') as fl:
                df = pickle.load(fl)

        #import ipdb; ipdb.set_trace()
        return df

def read_rbn_std(sTime,eTime=None,data_dir=None,
             qrz_call='km4ege',qrz_passwd='ProjectEllie_2014'):
    import os               # Provides utilities that help us do os-level operations like create directories
    import datetime         # Really awesome module for working with dates and times.
    import zipfile
    import urllib2          # Used to automatically download data files from the web.
    import pickle

    import numpy as np      #Numerical python - provides array types and operations
    import pandas as pd     #This is a nice utility for working with time-series type data.

    from hamtools import qrz

    #import ipdb; ipdb.set_trace()
    if data_dir is None: data_dir = os.getenv('DAVIT_TMPDIR')

    qz      = qrz.Session(qrz_call,qrz_passwd)

    ymd_list    = [datetime.datetime(sTime.year,sTime.month,sTime.day)]
    eDay        =  datetime.datetime(eTime.year,eTime.month,eTime.day)
    while ymd_list[-1] < eDay:
        ymd_list.append(ymd_list[-1] + datetime.timedelta(days=1))

    for ymd_dt in ymd_list:
        ymd         = ymd_dt.strftime('%Y%m%d')
        data_file   = '{0}.zip'.format(ymd)
        data_path   = os.path.join(data_dir,data_file)  

        time_0      = datetime.datetime.now()
        print 'Starting RBN processing on <%s> at %s.' % (data_file,str(time_0))

        ################################################################################
        # Make sure the data file exists.  If not, download it and open it.
        if not os.path.exists(data_path):
             try:    # Create the output directory, but fail silently if it already exists
                 os.makedirs(data_dir) 
             except:
                 pass

             qz      = qrz.Session(qrz_call,qrz_passwd)
             # File downloading code from: http://stackoverflow.com/questions/22676/how-do-i-download-a-file-over-http-using-python
             url = 'http://www.reversebeacon.net/raw_data/dl.php?f='+ymd

             #import ipdb; ipdb.set_trace()
             u = urllib2.urlopen(url)
             f = open(data_path, 'wb')
             meta = u.info()
             file_size = int(meta.getheaders("Content-Length")[0])
             print "Downloading: %s Bytes: %s" % (data_path, file_size)
         
             file_size_dl = 0
             block_sz = 8192
             while True:
                 buffer = u.read(block_sz)
                 if not buffer:
                     break
         
                 file_size_dl += len(buffer)
                 f.write(buffer)
                 status = r"%10d  [%3.2f%%]" % (file_size_dl, file_size_dl * 100. / file_size)
                 status = status + chr(8)*(len(status)+1)
                 print status,
             f.close()
             status = 'Done downloading!  Now converting to Pandas dataframe and plotting...'
             print status

#        #Original Implementation
#        std_sTime=datetime.datetime(sTime.year,sTime.month,sTime.day, sTime.hour)
#        if eTime.minute == 0 and eTime.second == 0:
#          std_eTime=datetime.datetime(eTime.year,eTime.month,eTime.day, eTime.hour)
#        else:
#          std_eTime=datetime.datetime(eTime.year,eTime.month,eTime.day, eTime.hour+1)

#        #New Implementation
#        std_sTime=datetime.datetime(sTime.year,sTime.month,sTime.day, sTime.hour)
#        if eTime.minute == 0 and eTime.second == 0:
#            hourly_eTime=datetime.datetime(eTime.year,eTime.month,eTime.day, eTime.hour)
#        else:
#            hourly_eTime=datetime.datetime(eTime.year,eTime.month,eTime.day, eTime.hour+1)
#
#        if hourly_eTime.hour-std_sTime.hour !=1:
#            std_eTime=datetime.datetime(eTime.year,eTime.month,eTime.day, std_sTime.hour+1)
#        else:
#            std_eTime=hourly_eTime

        #New Implementation
        std_sTime=datetime.datetime(sTime.year,sTime.month,sTime.day, sTime.hour)
        if eTime.minute == 0 and eTime.second == 0:
            hourly_eTime=datetime.datetime(eTime.year,eTime.month,eTime.day, eTime.hour)
        else:
            hourly_eTime=eTime+datetime.timedelta(hours=1)
            hourly_eTime=datetime.datetime(hourly_eTime.year,hourly_eTime.month,hourly_eTime.day, hourly_eTime.hour)
#            hourly_eTime=datetime.datetime(eTime.year,eTime.month,eTime.day, eTime.hour+1)

        std_eTime=std_sTime+datetime.timedelta(hours=1)
#        if hourly_eTime.hour-std_sTime.hour !=1:
#            std_eTime=datetime.datetime(eTime.year,eTime.month,eTime.day, std_sTime.hour+1)
#        else:
#            std_eTime=hourly_eTime
#
#        #New Implementation
#        std_sTime=datetime.datetime(sTime.year,sTime.month,sTime.day, sTime.hour)
#        if eTime.minute == 0 and eTime.second == 0:
#            hourly_eTime=datetime.datetime(eTime.year,eTime.month,eTime.day, eTime.hour)
#        else:
##            if sTime.hour+1==24:
##                hourly_eTime=datetime.datetime(eTime.year,eTime.month,eTime.day+1, 0)
##            else:
#            hourly_eTime=datetime.datetime(eTime.year,eTime.month,eTime.day, eTime.hour+1)
#
##        if hourly_eTime.hour==24:
##                hourly_eTime=datetime.datetime(eTime.year,eTime.month,eTime.day+1, 0)
#
#        if hourly_eTime.day==std_sTime.day and hourly_eTime.hour-std_sTime.hour !=1:
#            std_eTime=datetime.datetime(eTime.year,eTime.month,eTime.day, std_sTime.hour+1)
#        elif hourly_eTime.day!=std_sTime.day and 24 + std_sTime.hour !=24:
#
#        else:
#            std_eTime=hourly_eTime

        #flag 
        hour_flag=0
        while std_eTime<=hourly_eTime:
                p_filename = 'rbn_'+std_sTime.strftime('%Y%m%d%H%M-')+std_eTime.strftime('%Y%m%d%H%M.p')
                p_filepath = os.path.join(data_dir,p_filename)
                if not os.path.exists(p_filepath):
                    # Load data into dataframe here. ###############################################
                    with zipfile.ZipFile(data_path,'r') as z:   #This block lets us directly read the compressed gz file into memory.  The 'with' construction means that the file is automatically closed for us when we are done.
                        with z.open(ymd+'.csv') as fl:
                            df          = pd.read_csv(fl,parse_dates=[10])

                    # Create columns for storing geolocation data.
                    df['dx_lat'] = np.zeros(df.shape[0],dtype=np.float)*np.nan
                    df['dx_lon'] = np.zeros(df.shape[0],dtype=np.float)*np.nan
                    df['de_lat'] = np.zeros(df.shape[0],dtype=np.float)*np.nan
                    df['de_lon'] = np.zeros(df.shape[0],dtype=np.float)*np.nan

                    # Trim dataframe to just the entries in a 1 hour time period.
                    df = df[np.logical_and(df['date'] >= std_sTime,df['date'] < std_eTime)]

                    # Look up lat/lons in QRZ.com
                    errors  = 0
                    success = 0
                    for index,row in df.iterrows():
                        if index % 50   == 0:
                            print index,datetime.datetime.now()-time_0,row['date']
                        de_call = row['callsign']
                        dx_call = row['dx']
                        try:
                            de      = qz.qrz(de_call)
                            dx      = qz.qrz(dx_call)

                            row['de_lat'] = de['lat']
                            row['de_lon'] = de['lon']
                            row['dx_lat'] = dx['lat']
                            row['dx_lon'] = dx['lon']
                            df.loc[index] = row
            #                print '{index:06d} OK - DX: {dx} DE: {de}'.format(index=index,dx=dx_call,de=de_call)
                            success += 1
                        except:
            #                print '{index:06d} LOOKUP ERROR - DX: {dx} DE: {de}'.format(index=index,dx=dx_call,de=de_call)
                            errors += 1
                    total   = success + errors
                    pct     = success / float(total) * 100.
                    print '{0:d} of {1:d} ({2:.1f} %) call signs geolocated via qrz.com.'.format(success,total,pct)
                    df.to_pickle(p_filepath)
                else:
                    with open(p_filepath,'rb') as fl:
                        df = pickle.load(fl)

                if hour_flag==0:
                    df_comp=df
#                    df_comp=pd.df.copy(deep=True)
                    hour_flag=hour_flag+1
#                if hour_flag!=0:
                #When specified start/end times cross over the hour mark
                else:
                    df_comp=pd.concat([df_comp, df])

                std_sTime=std_eTime
                std_eTime=std_sTime+datetime.timedelta(hours=1)
#                std_eTime=datetime.datetime(eTime.year,eTime.month,eTime.day, std_eTime.hour+1)

        
        # Trim dataframe to just the entries we need.
        df = df_comp[np.logical_and(df_comp['date'] >= sTime,df_comp['date'] < eTime)]
#        df = df[np.logical_and(df['date'] >= sTime,df['date'] < eTime)]
        return df

def station_loc(df, data_dir=None,
             qrz_call='km4ege',qrz_passwd='ProjectEllie_2014'):
    import os               # Provides utilities that help us do os-level operations like create directories
    import datetime         # Really awesome module for working with dates and times.
    import zipfile
    import urllib2          # Used to automatically download data files from the web.
    import pickle

    import numpy as np      #Numerical python - provides array types and operations
    import pandas as pd     #This is a nice utility for working with time-series type data.

    from hamtools import qrz

    #import ipdb; ipdb.set_trace()
    if data_dir is None: data_dir = os.getenv('DAVIT_TMPDIR')

    #Get Station lat/lon
    qz      = qrz.Session(qrz_call,qrz_passwd)
#    df['dx_lat'] = np.zeros(df.shape[0],dtype=np.float)*np.nan
#    df['dx_lon'] = np.zeros(df.shape[0],dtype=np.float)*np.nan
    df['de_lat'] = np.zeros(df.shape[0],dtype=np.float)*np.nan
    df['de_lon'] = np.zeros(df.shape[0],dtype=np.float)*np.nan
    # Look up lat/lons in QRZ.com
    errors  = 0
    success = 0
    for index,row in df.iterrows():
        de_call = row['callsign']
        try:
            de      = qz.qrz(de_call)

            row['de_lat'] = de['lat']
            row['de_lon'] = de['lon']

            df.loc[index] = row
    #                print '{index:06d} OK - DX: {dx} DE: {de}'.format(index=index,dx=dx_call,de=de_call)
            success += 1
        except:
    #                print '{index:06d} LOOKUP ERROR - DX: {dx} DE: {de}'.format(index=index,dx=dx_call,de=de_call)
            errors += 1
    total   = success + errors
    pct     = success / float(total) * 100.
    print '{0:d} of {1:d} ({2:.1f} %) call signs geolocated via qrz.com.'.format(success,total,pct)
    return df

def k4kdj_rbn(sTime,eTime=None,data_dir=None,
             qrz_call='km4ege',qrz_passwd='ProjectEllie_2014'):
    import os               # Provides utilities that help us do os-level operations like create directories
    import datetime         # Really awesome module for working with dates and times.
    import zipfile
    import urllib2          # Used to automatically download data files from the web.
    import pickle

    import numpy as np      #Numerical python - provides array types and operations
    import pandas as pd     #This is a nice utility for working with time-series type data.

    from hamtools import qrz

    #import ipdb; ipdb.set_trace()
    if data_dir is None: data_dir = os.getenv('DAVIT_TMPDIR')

    qz      = qrz.Session(qrz_call,qrz_passwd)

    ymd_list    = [datetime.datetime(sTime.year,sTime.month,sTime.day)]
    eDay        =  datetime.datetime(eTime.year,eTime.month,eTime.day)
    while ymd_list[-1] < eDay:
        ymd_list.append(ymd_list[-1] + datetime.timedelta(days=1))

    for ymd_dt in ymd_list:
        ymd         = ymd_dt.strftime('%Y%m%d')
        data_file   = '{0}.zip'.format(ymd)
        data_path   = os.path.join(data_dir,data_file)  

        time_0      = datetime.datetime.now()
        print 'Starting RBN processing on <%s> at %s.' % (data_file,str(time_0))

        ################################################################################
        # Make sure the data file exists.  If not, download it and open it.
        if not os.path.exists(data_path):
             try:    # Create the output directory, but fail silently if it already exists
                 os.makedirs(data_dir) 
             except:
                 pass
             # File downloading code from: http://stackoverflow.com/questions/22676/how-do-i-download-a-file-over-http-using-python
             url = 'http://www.reversebeacon.net/raw_data/dl.php?f='+ymd

             #import ipdb; ipdb.set_trace()
             u = urllib2.urlopen(url)
             f = open(data_path, 'wb')
             meta = u.info()
             file_size = int(meta.getheaders("Content-Length")[0])
             print "Downloading: %s Bytes: %s" % (data_path, file_size)
         
             file_size_dl = 0
             block_sz = 8192
             while True:
                 buffer = u.read(block_sz)
                 if not buffer:
                     break
         
                 file_size_dl += len(buffer)
                 f.write(buffer)
                 status = r"%10d  [%3.2f%%]" % (file_size_dl, file_size_dl * 100. / file_size)
                 status = status + chr(8)*(len(status)+1)
                 print status,
             f.close()
             status = 'Done downloading!  Now converting to Pandas dataframe and plotting...'
             print status

        p_filename = 'rbn_'+sTime.strftime('%Y%m%d%H%M-')+eTime.strftime('%Y%m%d%H%M.p')
        p_filepath = os.path.join(data_dir,p_filename)
        if not os.path.exists(p_filepath):
            # Load data into dataframe here. ###############################################
            with zipfile.ZipFile(data_path,'r') as z:   #This block lets us directly read the compressed gz file into memory.  The 'with' construction means that the file is automatically closed for us when we are done.
                with z.open(ymd+'.csv') as fl:
                    df          = pd.read_csv(fl,parse_dates=[10])

#            # Create columns for storing geolocation data.
#            df['dx_lat'] = np.zeros(df.shape[0],dtype=np.float)*np.nan
#            df['dx_lon'] = np.zeros(df.shape[0],dtype=np.float)*np.nan
#            df['de_lat'] = np.zeros(df.shape[0],dtype=np.float)*np.nan
#            df['de_lon'] = np.zeros(df.shape[0],dtype=np.float)*np.nan

            # Trim dataframe to just the entries we need.
            df = df[np.logical_and(df['date'] >= sTime,df['date'] < eTime)]
            # Limit to stations heard by K4KDJ 
            import ipdb; ipdb.set_trace()
            df = df[df['callsign']=='K4KDJ']
            import ipdb; ipdb.set_trace()

#            # Look up lat/lons in QRZ.com
#            errors  = 0
#            success = 0
#            for index,row in df.iterrows():
#                if index % 50   == 0:
#                    print index,datetime.datetime.now()-time_0,row['date']
#                de_call = row['callsign']
#                dx_call = row['dx']
#                try:
#                    de      = qz.qrz(de_call)
#                    dx      = qz.qrz(dx_call)
#
#                    row['de_lat'] = de['lat']
#                    row['de_lon'] = de['lon']
#                    row['dx_lat'] = dx['lat']
#                    row['dx_lon'] = dx['lon']
#                    df.loc[index] = row
#    #                print '{index:06d} OK - DX: {dx} DE: {de}'.format(index=index,dx=dx_call,de=de_call)
#                    success += 1
#                except:
#    #                print '{index:06d} LOOKUP ERROR - DX: {dx} DE: {de}'.format(index=index,dx=dx_call,de=de_call)
#                    errors += 1
#            total   = success + errors
#            pct     = success / float(total) * 100.
#            print '{0:d} of {1:d} ({2:.1f} %) call signs geolocated via qrz.com.'.format(success,total,pct)
            df.to_pickle(p_filepath)
        else:
            with open(p_filepath,'rb') as fl:
                df = pickle.load(fl)

        #import ipdb; ipdb.set_trace()
        return df


# Set up a dictionary which identifies which bands we want and some plotting attributes for each band
band_dict       = {}
band_dict[28]   = {'name': '10 m',  'freq': '28 MHz',  'color':'red'}
band_dict[21]   = {'name': '15 m',  'freq': '21 MHz',  'color':'orange'}
band_dict[14]   = {'name': '20 m',  'freq': '14 MHz',  'color':'yellow'}
band_dict[7]    = {'name': '40 m',  'freq': '7 MHz',   'color':'green'}
band_dict[3]    = {'name': '80 m',  'freq': '3.5 MHz', 'color':'blue'}
band_dict[1]    = {'name': '160 m', 'freq': '1.8 MHz', 'color':'aqua'}

bandlist        = band_dict.keys()
bandlist.sort(reverse=True)


def band_legend(fig=None,loc='lower center',markerscale=0.5,prop={'size':10},title=None,bbox_to_anchor=None,ncdxf=False,ncol=None,addons=None):
    #Note (Magda Moses): Added variable addons to enable adding additional markers to legend if necessary (30 May 2016)
    #addons is a dictionary with the following format
    #addons[add]={'label': legend_label, 'marker': symbol,'color': symbol_color}
        #[add]: index of addons, hence, the legend (with respect to the addons) will be arrandged alphabetically! 
        #[legend_label]: The label that will appear on the legend
        #[symbol]: Symbol (if referencing a line/color, then use '')
        #[color]: Color of symbol/line that will appear on the legend
    #Also Note that you can change where the addons will appear in the legend by moving the if-statement in which they are created

    from matplotlib import pyplot as plt
    import matplotlib.patches as mpatches

    if fig is None: fig = plt.gcf() 

    handles = []
    labels  = []
    for band in bandlist:
        color = band_dict[band]['color']
        label = band_dict[band]['freq']
        handles.append(mpatches.Patch(color=color,label=label))
        labels.append(label)

    import matplotlib.markers as mmarkers
    fig_tmp = plt.figure()
    ax_tmp = fig_tmp.add_subplot(111)
    ax_tmp.set_visible(False)

    #Use the following block to add additional markers to map
    if addons!=None:
        for add in addons.keys():
            marker=addons[add]['marker']
            color=addons[add]['color']
            label=addons[add]['label']
            #VTARA
            if add=='vtara':
                scat = ax_tmp.scatter(0,0,s=dxf_leg_size,marker=marker, color=color)
                labels.append(label)
                handles.append(scat)
#                scat = ax_tmp.scatter(0,0,s=dxf_leg_size,marker=addons['vtara']['marker'], color=addons['vtara']['color'])
#                labels.append(addons['vtara']['label'])
#                handles.append(scat)
            #Eclipse
            if add=='eclipse':
                handles.append(mpatches.Patch(color=color,label=label))
                labels.append(label)

    #Add rest of markers
    scat = ax_tmp.scatter(0,0,color='k',s=50)
    labels.append('RBN Receiver')
    handles.append(scat)
    if ncdxf:
        scat = ax_tmp.scatter(0,0,s=dxf_leg_size,**dxf_prop)
        labels.append('NCDXF Beacon')
        handles.append(scat)

    if ncol is None:
        ncol = len(labels)
    
    legend = fig.legend(handles,labels,ncol=ncol,loc=loc,markerscale=markerscale,prop=prop,title=title,bbox_to_anchor=bbox_to_anchor,scatterpoints=1)
    return legend

def rbn_map_plot(df,ax=None,legend=True,tick_font_size=None,ncdxf=False,plot_paths=True,
        llcrnrlon=-180.,llcrnrlat=-90,urcrnrlon=180.,urcrnrlat=90.,proj='cyl',basemapType=True,m=None,eclipse=False,path_alpha=None):
    """Plot Reverse Beacon Network data.

    **Args**:
        * **[sTime]**: datetime.datetime object for start of plotting.
        * **[eTime]**: datetime.datetime object for end of plotting.
        * **[ymin]**: Y-Axis minimum limit
        * **[ymax]**: Y-Axis maximum limit
        * **[legendSize]**: Character size of the legend

    **Returns**:
        * **fig**:      matplotlib figure object that was plotted to

    .. note::
        If a matplotlib figure currently exists, it will be modified by this routine.  If not, a new one will be created.

    Written by Nathaniel Frissell 2014 Sept 06
    """
    import datetime
    
    from matplotlib import pyplot as plt
    from mpl_toolkits.basemap import Basemap

    import numpy as np
    import pandas as pd

    import eclipse_lib

    from davitpy.pydarn.radar import *
    from davitpy.pydarn.plotting import *
    from davitpy.utils import *

    if ax is None:
        fig     = plt.figure(figsize=(10,6))
        ax      = fig.add_subplot(111)
    else:
        fig     = ax.get_figure()

    #Drop NaNs (QSOs without Lat/Lons)
    df = df.dropna(subset=['dx_lat','dx_lon','de_lat','de_lon'])

    ##Sort the data by band and time, then group by band.
    df['band']  = np.array((np.floor(df['freq']/1000.)),dtype=np.int)
    srt         = df.sort(['band','date'])
    grouped     = srt.groupby('band')

    sTime       = df['date'].min()
    eTime       = df['date'].max()

#    import ipdb; ipdb.set_trace()
    half_time   = datetime.timedelta(seconds= ((eTime - sTime).total_seconds()/2.) )
    plot_mTime = sTime + half_time

    if m==None: #added to allow rbn to be plotted over maps of other data 
        if basemapType:
            m = Basemap(llcrnrlon=llcrnrlon,llcrnrlat=llcrnrlat,urcrnrlon=urcrnrlon,urcrnrlat=urcrnrlat,resolution='l',area_thresh=1000.,projection=proj,ax=ax)
        else:
            m = plotUtils.mapObj(llcrnrlon=llcrnrlon,llcrnrlat=llcrnrlat,urcrnrlon=urcrnrlon,urcrnrlat=urcrnrlat,resolution='l',area_thresh=1000.,projection=proj,ax=ax,fillContinents='None', fix_aspect=True)

#    title = sTime.strftime('%H%M - ')+eTime.strftime('%H%M UT')
#    title = sTime.strftime('Reverse Beacon Network %Y %b %d %H%M UT - ')+eTime.strftime('%Y %b %d %H%M UT')
#    if m==None:
        title = sTime.strftime('RBN: %d %b %Y %H%M UT - ')+eTime.strftime('%d %b %Y %H%M UT')
        ax.set_title(title)

        # draw parallels and meridians.
        m.drawparallels(np.arange(-90.,91.,45.),color='k',labels=[False,True,True,False],fontsize=tick_font_size)
        m.drawmeridians(np.arange(-180.,181.,45.),color='k',labels=[True,False,False,True],fontsize=tick_font_size)
        m.drawcoastlines(color='0.65')
        m.drawmapboundary(fill_color='w')
        m.nightshade(plot_mTime,color='0.82')
    #if plotting the 2017 eclipse map then also draw state boundaries
    if eclipse:
        m.drawcountries(color='0.65')#np.arange(-90.,91.,45.),color='k',labels=[False,True,True,False],fontsize=tick_font_size)
        m.drawstates(color='0.65')
    
    de_list = []
    dx_list = []
    for band in bandlist:
        try:
            this_group = grouped.get_group(band)
        except:
            continue

        color = band_dict[band]['color']
        label = band_dict[band]['name']

        for index,row in this_group.iterrows():
            #Yay stack overflow! - http://stackoverflow.com/questions/13888566/python-basemap-drawgreatcircle-function
            de_lat = row['de_lat']
            de_lon = row['de_lon']
            dx_lat = row['dx_lat']
            dx_lon = row['dx_lon']

            if row['callsign'] not in de_list: de_list.append(row['callsign'])
            if row['dx'] not in dx_list: dx_list.append(row['dx'])

            rx    = m.scatter(de_lon,de_lat,color='k',s=2,zorder=100)
            if plot_paths:
                line, = m.drawgreatcircle(dx_lon,dx_lat,de_lon,de_lat,color=color, alpha=path_alpha)

                p = line.get_path()
                # find the index which crosses the dateline (the delta is large)
                cut_point = np.where(np.abs(np.diff(p.vertices[:, 0])) > 200)[0]
                if cut_point:
                    cut_point = cut_point[0]

                    # create new vertices with a nan inbetween and set those as the path's vertices
                   # import ipdb; ipdb.set_trace()
                    new_verts = np.concatenate(
                                               [p.vertices[:cut_point, :], 
                                                [[np.nan, np.nan]], 
                                                p.vertices[cut_point+1:, :]]
                                               )
                    p.codes = None
                    p.vertices = new_verts
                #
#                cut_point_lat = np.where(np.abs(np.diff(p.vertices[:, 1])) > 90)[0]
#                if cut_point_lat:
#                    cut_point_lat = cut_point_lat[0]
#
#                    # create new vertices with a nan inbetween and set those as the path's vertices
#                    import ipdb; ipdb.set_trace()
#                    new_verts = np.concatenate(
#                                               [p.vertices[:cut_point_lat,:], 
#                                                [[np.nan, np.nan]], 
#                                                p.vertices[cut_point_lat+1:,:]]
#                                               )
#                    p.codes = None
#                    p.vertices = new_verts
#                    import ipdb; ipdb.set_trace()
    if ncdxf:
        dxf_df = pd.DataFrame.from_csv('ncdxf.csv')
        m.scatter(dxf_df['lon'],dxf_df['lat'],s=dxf_plot_size,**dxf_prop)

    #if eclipse:
     #   df_cl=eclipse_lib.eclipse_get_path(fname='ds_CL.csv')
     #   m.plot(df_cl['eLon'],df_cl['eLat'],'m--',label='2017 Eclipse Central Line', linewidth=2, latlon=True)

#    import ipdb; ipdb.set_trace()
    text = []
    text.append('TX Stations: {0:d}'.format(len(dx_list)))
    text.append('RX Stations: {0:d}'.format(len(de_list)))
    text.append('Plotted Paths: {0:d}'.format(len(df)))

    props = dict(facecolor='white', alpha=0.9,pad=6)
    ax.text(0.02,0.05,'\n'.join(text),transform=ax.transAxes,ha='left',va='bottom',size=9,bbox=props)

    if legend:
        band_legend()

    return m,fig



def dx_legend(dx_dict, dxlist, fig=None,loc='lower center',markerscale=0.5,prop={'size':10},title=None,bbox_to_anchor=None,ncdxf=False,ncol=None):
    from matplotlib import pyplot as plt
    import matplotlib.patches as mpatches

    if fig is None: fig = plt.gcf() 

    handles = []
    labels  = []
    for dx in dxlist:
        color = dx_dict[dx]['color']
        label = dx_dict[dx]['name']
        handles.append(mpatches.Patch(color=color,label=label))
        labels.append(label)

    import matplotlib.markers as mmarkers
    fig_tmp = plt.figure()
    ax_tmp = fig_tmp.add_subplot(111)
    ax_tmp.set_visible(False)
    scat = ax_tmp.scatter(0,0,color='k',s=50)
    labels.append('RBN Receiver')
    handles.append(scat)
    if ncdxf:
        scat = ax_tmp.scatter(0,0,s=dxf_leg_size,**dxf_prop)
        labels.append('NCDXF Beacon')
        handles.append(scat)

    if ncol is None:
        ncol = len(labels)
    
    legend = fig.legend(handles,labels,ncol=ncol,loc=loc,markerscale=markerscale,prop=prop,title=title,bbox_to_anchor=bbox_to_anchor,scatterpoints=1)
    return legend

def set_dx_dict(dx_call, color_array=None):
    """Create dictionary for dx callsigns identified in RBN to plot 

    **Args**:
        * **[dx_call]**: An array of unique callsigns
    """
    from matplotlib import colors as color
    # Set up a dictionary which identifies which calls we want and some plotting attributes for each band
#    st_color=['red', 'orange', 'yellow', 'green', 'blue', 'aqua']
##    if len(colors) != len(dx_call):
##        print "ERROR: Not enough colors"
#
#    #Define colors
##    colors=[]
##    (red, green, blue)=color.to_rgb('b')
##    red=color.ColorConverter.to_rgb('red')
##    green=color('green')
##    blue=color('blue')
#
##    for i in range (0,len(dx_call)-1):
##        if i<3:
##            colors[i]=(red, green, blue)
##
##       elif i>3:
##           color
#
#    dx_dict       = {}
#    i=0
#    for dx in dx_call:
#    red=0
#    green=.5
#    blue=.2
#    for dx in dx_call:
#        if i>=6:
#            if red<1:
#                colors[i]=(red+0.05,green, blue)
#
#            elif blue<1:
#                colors[i]=(red,green, blue+0.05)
#
#
#        dx_dict[i]   = {'name': dx_call[i],  'color':colors[i]}
#        i=+1
#
##    dx_dict[21]   = {'name': '15 m',  'freq': '21 MHz',  'color':'orange'}
##    dx_dict[14]   = {'name': '20 m',  'freq': '14 MHz',  'color':'yellow'}
##    dx_dict[7]    = {'name': '40 m',  'freq': '7 MHz',   'color':'green'}
##    dx_dict[3]    = {'name': '80 m',  'freq': '3.5 MHz', 'color':'blue'}
##    dx_dict[1]    = {'name': '160 m', 'freq': '1.8 MHz', 'color':'aqua'}

#may need this next line for a more general code
#    dx__call=[]
#    dx_call=e

    dx_dict       = {}
    i=0
    for dx in dx_call:
#        import ipdb; ipdb.set_trace()
        call=dx
        color=color_array[i]
        dx_dict[i+1]   = {'name': call,  'color':color}
#        import ipdb; ipdb.set_trace()
        i=i+1

#    import ipdb; ipdb.set_trace()
    dxlist        = dx_dict.keys()
    dxlist.sort(reverse=True)
    return dx_dict, dxlist
    
#color_dict={'c': (0.0, 0.75, 0.75), 'b': (0.0, 0.0, 1.0), 'g': (0.0, 0.5, 0.0), 'y': (0.75, 0.75, 0), 'r': (1.0, 0.0, 0.0), 'm': (0.75, 0, 0.75)}
#for color in color_dict:

def rbn_map_byDX(df,dx_dict=None, dxlist=None,dx_call=None, color_array=None,ax=None,legend=True,tick_font_size=None,ncdxf=False,plot_paths=True,
        llcrnrlon=-180.,llcrnrlat=-90,urcrnrlon=180.,urcrnrlat=90.,proj='cyl',basemapType=True,eclipse=False,path_alpha=None):
    """Plot Reverse Beacon Network data.

    **Args**:
        * **[sTime]**: datetime.datetime object for start of plotting.
        * **[eTime]**: datetime.datetime object for end of plotting.
        * **[ymin]**: Y-Axis minimum limit
        * **[ymax]**: Y-Axis maximum limit
        * **[legendSize]**: Character size of the legend

    **Returns**:
        * **fig**:      matplotlib figure object that was plotted to

    .. note::
        If a matplotlib figure currently exists, it will be modified by this routine.  If not, a new one will be created.

    Written by Nathaniel Frissell 2014 Sept 06
    Modified by Magda Moses 2015 July 17
    """
    import datetime
    
    from matplotlib import pyplot as plt
    from mpl_toolkits.basemap import Basemap

    import numpy as np
    import pandas as pd

    import eclipse_lib

    from davitpy.pydarn.radar import *
    from davitpy.pydarn.plotting import *
    from davitpy.utils import *

    if ax is None:
        fig     = plt.figure(figsize=(10,6))
        ax      = fig.add_subplot(111)
    else:
        fig     = ax.get_figure()

    #Drop NaNs (QSOs without Lat/Lons)
    df = df.dropna(subset=['dx_lat','dx_lon','de_lat','de_lon'])

    ##Sort the data by dx call and time, then group by call.
#    df['band']  = np.array((np.floor(df['freq']/1000.)),dtype=np.int)
    srt         = df.sort(['dx','date'])
    grouped     = srt.groupby('dx')

    sTime       = df['date'].min()
    eTime       = df['date'].max()

    half_time   = datetime.timedelta(seconds= ((eTime - sTime).total_seconds()/2.) )
    plot_mTime = sTime + half_time

    if dx_dict==None:
        dx_dict, dxlist=set_dx_dict(dx_call, color_array)

    if basemapType:
        m = Basemap(llcrnrlon=llcrnrlon,llcrnrlat=llcrnrlat,urcrnrlon=urcrnrlon,urcrnrlat=urcrnrlat,resolution='l',area_thresh=1000.,projection=proj,ax=ax)
    else:
        m = plotUtils.mapObj(llcrnrlon=llcrnrlon,llcrnrlat=llcrnrlat,urcrnrlon=urcrnrlon,urcrnrlat=urcrnrlat,resolution='l',area_thresh=1000.,projection=proj,ax=ax,fillContinents='None', fix_aspect=True)

#    title = sTime.strftime('%H%M - ')+eTime.strftime('%H%M UT')
#    title = sTime.strftime('Reverse Beacon Network %Y %b %d %H%M UT - ')+eTime.strftime('%Y %b %d %H%M UT')
    title = sTime.strftime('RBN: %d %b %Y %H%M UT - ')+eTime.strftime('%d %b %Y %H%M UT')
    ax.set_title(title)

    # draw parallels and meridians.
    m.drawparallels(np.arange(-90.,91.,45.),color='k',labels=[False,True,True,False],fontsize=tick_font_size)
    m.drawmeridians(np.arange(-180.,181.,45.),color='k',labels=[True,False,False,True],fontsize=tick_font_size)
    m.drawcoastlines(color='0.65')
    m.drawmapboundary(fill_color='w')
    m.nightshade(plot_mTime,color='0.82')
    #if plotting the 2017 eclipse map then also draw state boundaries
    if eclipse:
        m.drawcountries(color='0.65')#np.arange(-90.,91.,45.),color='k',labels=[False,True,True,False],fontsize=tick_font_size)
        m.drawstates(color='0.65')
    
#    for idx in range(0, 100)
        
#    color_idx=0
#    import ipdb; ipdb.set_trace()
    de_list = []
    dx_list = []
    for dx  in dxlist:
#        import ipdb; ipdb.set_trace()
        label = dx_dict[dx]['name']
        try:
            this_group = grouped.get_group(label)
        except:
            continue

#        color = color_array[color_idx]
        color = dx_dict[dx]['color']
#        label = dx_dict[dx]['name']

        for index,row in this_group.iterrows():
            #Yay stack overflow! - http://stackoverflow.com/questions/13888566/python-basemap-drawgreatcircle-function
#            import ipdb; ipdb.set_trace()
            de_lat = row['de_lat']
            de_lon = row['de_lon']
            dx_lat = row['dx_lat']
            dx_lon = row['dx_lon']

            if row['callsign'] not in de_list: de_list.append(row['callsign'])
            if row['dx'] not in dx_list: dx_list.append(row['dx'])

            rx    = m.scatter(de_lon,de_lat,color='k',s=2,zorder=100)
            if plot_paths:
                line, = m.drawgreatcircle(dx_lon,dx_lat,de_lon,de_lat,color=color, alpha=path_alpha)

                p = line.get_path()
                # find the index which crosses the dateline (the delta is large)
                cut_point = np.where(np.abs(np.diff(p.vertices[:, 0])) > 200)[0]
                if cut_point:
                    cut_point = cut_point[0]

                    # create new vertices with a nan inbetween and set those as the path's vertices
                   # import ipdb; ipdb.set_trace()
                    new_verts = np.concatenate(
                                               [p.vertices[:cut_point, :], 
                                                [[np.nan, np.nan]], 
                                                p.vertices[cut_point+1:, :]]
                                               )
                    p.codes = None
                    p.vertices = new_verts
                #
#                cut_point_lat = np.where(np.abs(np.diff(p.vertices[:, 1])) > 90)[0]
#                if cut_point_lat:
#                    cut_point_lat = cut_point_lat[0]
#
#                    # create new vertices with a nan inbetween and set those as the path's vertices
#                    import ipdb; ipdb.set_trace()
#                    new_verts = np.concatenate(
#                                               [p.vertices[:cut_point_lat,:], 
#                                                [[np.nan, np.nan]], 
#                                                p.vertices[cut_point_lat+1:,:]]
#                                               )
#                    p.codes = None
#                    p.vertices = new_verts
#                    import ipdb; ipdb.set_trace()
    if ncdxf:
        dxf_df = pd.DataFrame.from_csv('ncdxf.csv')
        m.scatter(dxf_df['lon'],dxf_df['lat'],s=dxf_plot_size,**dxf_prop)

    #if eclipse:
     #   df_cl=eclipse_lib.eclipse_get_path(fname='ds_CL.csv')
     #   m.plot(df_cl['eLon'],df_cl['eLat'],'m--',label='2017 Eclipse Central Line', linewidth=2, latlon=True)

#    import ipdb; ipdb.set_trace()
    text = []
    text.append('TX Stations: {0:d}'.format(len(dx_list)))
    text.append('RX Stations: {0:d}'.format(len(de_list)))
    text.append('Plotted Paths: {0:d}'.format(len(df)))

    props = dict(facecolor='white', alpha=0.9,pad=6)
    ax.text(0.02,0.05,'\n'.join(text),transform=ax.transAxes,ha='left',va='bottom',size=9,bbox=props)

    if legend:
        dx_legend(dx_dict, dxlist, ncol=4)
#        dx_legend(dx_dict, dxlist,fig=fig, loc='center',bbox_to_anchor=[0.48,0.505],ncdxf=True)

    return m,fig

def rbn_map_overlay(df,m=None, scatter_rbn=False, ax=None,legend=True,tick_font_size=None,ncdxf=False,plot_paths=True,
        llcrnrlon=-180.,llcrnrlat=-90,urcrnrlon=180.,urcrnrlat=90.,proj='cyl',basemapType=True,eclipse=False,path_alpha=None):
    """Plot Reverse Beacon Network data.

    **Args**:
        * **[sTime]**: datetime.datetime object for start of plotting.
        * **[eTime]**: datetime.datetime object for end of plotting.
        * **[ymin]**: Y-Axis minimum limit
        * **[ymax]**: Y-Axis maximum limit
        * **[legendSize]**: Character size of the legend

    **Returns**:
        * **fig**:      matplotlib figure object that was plotted to

    .. note::
        If a matplotlib figure currently exists, it will be modified by this routine.  If not, a new one will be created.

    Written by Nathaniel Frissell 2014 Sept 06
    Modified by Magda Moses July 22, 2015
    """
    import datetime
    
    from matplotlib import pyplot as plt
    from mpl_toolkits.basemap import Basemap

    import numpy as np
    import pandas as pd

    import eclipse_lib

    from davitpy.pydarn.radar import *
    from davitpy.pydarn.plotting import *
    from davitpy.utils import *

    if ax is None:
        fig     = plt.figure(figsize=(10,6))
        ax      = fig.add_subplot(111)
    else:
        fig     = ax.get_figure()

    #Drop NaNs (QSOs without Lat/Lons)
    df = df.dropna(subset=['dx_lat','dx_lon','de_lat','de_lon'])

    ##Sort the data by band and time, then group by band.
    df['band']  = np.array((np.floor(df['freq']/1000.)),dtype=np.int)
    srt         = df.sort(['band','date'])
    grouped     = srt.groupby('band')

    sTime       = df['date'].min()
    eTime       = df['date'].max()

    half_time   = datetime.timedelta(seconds= ((eTime - sTime).total_seconds()/2.) )
    plot_mTime = sTime + half_time

#    if scatter_rbn==False:
    if m==None:
        if basemapType:
            m = Basemap(llcrnrlon=llcrnrlon,llcrnrlat=llcrnrlat,urcrnrlon=urcrnrlon,urcrnrlat=urcrnrlat,resolution='l',area_thresh=1000.,projection=proj,ax=ax)
        else:
            m = plotUtils.mapObj(llcrnrlon=llcrnrlon,llcrnrlat=llcrnrlat,urcrnrlon=urcrnrlon,urcrnrlat=urcrnrlat,resolution='l',area_thresh=1000.,projection=proj,ax=ax,fillContinents='None', fix_aspect=True)

    #    title = sTime.strftime('%H%M - ')+eTime.strftime('%H%M UT')
    #    title = sTime.strftime('Reverse Beacon Network %Y %b %d %H%M UT - ')+eTime.strftime('%Y %b %d %H%M UT')
        title = sTime.strftime('RBN: %d %b %Y %H%M UT - ')+eTime.strftime('%d %b %Y %H%M UT')
        ax.set_title(title)

        # draw parallels and meridians.
        m.drawparallels(np.arange(-90.,91.,45.),color='k',labels=[False,True,True,False],fontsize=tick_font_size)
        m.drawmeridians(np.arange(-180.,181.,45.),color='k',labels=[True,False,False,True],fontsize=tick_font_size)
        m.drawcoastlines(color='0.65')
        m.drawmapboundary(fill_color='w')
        m.nightshade(plot_mTime,color='0.82')
        #if plotting the 2017 eclipse map then also draw state boundaries
        if eclipse:
            m.drawcountries(color='0.65')#np.arange(-90.,91.,45.),color='k',labels=[False,True,True,False],fontsize=tick_font_size)
            m.drawstates(color='0.65')
    
    de_list = []
    dx_list = []
    for band in bandlist:
        try:
            this_group = grouped.get_group(band)
        except:
            continue

        color = band_dict[band]['color']
        label = band_dict[band]['name']

        for index,row in this_group.iterrows():
            #Yay stack overflow! - http://stackoverflow.com/questions/13888566/python-basemap-drawgreatcircle-function
            de_lat = row['de_lat']
            de_lon = row['de_lon']
            dx_lat = row['dx_lat']
            dx_lon = row['dx_lon']

            if row['callsign'] not in de_list: de_list.append(row['callsign'])
            if row['dx'] not in dx_list: dx_list.append(row['dx'])

            rx    = m.scatter(de_lon,de_lat,color='k',s=2,zorder=100)
            if plot_paths:
                line, = m.drawgreatcircle(dx_lon,dx_lat,de_lon,de_lat,color=color, alpha=path_alpha)

                p = line.get_path()
                # find the index which crosses the dateline (the delta is large)
                cut_point = np.where(np.abs(np.diff(p.vertices[:, 0])) > 200)[0]
                if cut_point:
                    cut_point = cut_point[0]

                    # create new vertices with a nan inbetween and set those as the path's vertices
                   # import ipdb; ipdb.set_trace()
                    new_verts = np.concatenate(
                                               [p.vertices[:cut_point, :], 
                                                [[np.nan, np.nan]], 
                                                p.vertices[cut_point+1:, :]]
                                               )
                    p.codes = None
                    p.vertices = new_verts
                #
#                cut_point_lat = np.where(np.abs(np.diff(p.vertices[:, 1])) > 90)[0]
#                if cut_point_lat:
#                    cut_point_lat = cut_point_lat[0]
#
#                    # create new vertices with a nan inbetween and set those as the path's vertices
#                    import ipdb; ipdb.set_trace()
#                    new_verts = np.concatenate(
#                                               [p.vertices[:cut_point_lat,:], 
#                                                [[np.nan, np.nan]], 
#                                                p.vertices[cut_point_lat+1:,:]]
#                                               )
#                    p.codes = None
#                    p.vertices = new_verts
#                    import ipdb; ipdb.set_trace()
    if scatter_rbn==False:
        if ncdxf:
            dxf_df = pd.DataFrame.from_csv('ncdxf.csv')
            m.scatter(dxf_df['lon'],dxf_df['lat'],s=dxf_plot_size,**dxf_prop)

        #if eclipse:
         #   df_cl=eclipse_lib.eclipse_get_path(fname='ds_CL.csv')
         #   m.plot(df_cl['eLon'],df_cl['eLat'],'m--',label='2017 Eclipse Central Line', linewidth=2, latlon=True)

#        import ipdb; ipdb.set_trace()
        text = []
        text.append('TX Stations: {0:d}'.format(len(dx_list)))
        text.append('RX Stations: {0:d}'.format(len(de_list)))
        text.append('Plotted Paths: {0:d}'.format(len(df)))

        props = dict(facecolor='white', alpha=0.9,pad=6)
        ax.text(0.02,0.05,'\n'.join(text),transform=ax.transAxes,ha='left',va='bottom',size=9,bbox=props)

        if legend:
            band_legend()

    return m,fig

def rbn_map_node(df, sTime, eTime,m=None, ax=None, tick_font_size=None,ncdxf=False,plot_paths=True,
        llcrnrlon=-180.,llcrnrlat=-90,urcrnrlon=180.,urcrnrlat=90.,proj='cyl',basemapType=True,eclipse=False,path_alpha=None):
    """Plot Reverse Beacon Network reciever nodes from RBN data. 

    **Args**:
        * **[ymin]**: Y-Axis minimum limit
        * **[ymax]**: Y-Axis maximum limit
        * **[legendSize]**: Character size of the legend

    **Returns**:
        * **fig**:      matplotlib figure object that was plotted to

    .. note::
        If a matplotlib figure currently exists, it will be modified by this routine.  If not, a new one will be created.

    Written by Magda Moses and  Nathaniel Frissell 2016 March 05
    """
    import datetime
    
    from matplotlib import pyplot as plt
    from mpl_toolkits.basemap import Basemap

    import numpy as np
    import pandas as pd

    import eclipse_lib

    from davitpy.pydarn.radar import *
    from davitpy.pydarn.plotting import *
    from davitpy.utils import *

    if ax is None:
        fig     = plt.figure(figsize=(10,6))
        ax      = fig.add_subplot(111)
    else:
        fig     = ax.get_figure()

    #Drop NaNs (QSOs without Lat/Lons)
    df = df.dropna(subset=['de_lat','de_lon'])

#    if scatter_rbn==False:
    if m==None:
        if basemapType:
            m = Basemap(llcrnrlon=llcrnrlon,llcrnrlat=llcrnrlat,urcrnrlon=urcrnrlon,urcrnrlat=urcrnrlat,resolution='l',area_thresh=1000.,projection=proj,ax=ax)
        else:
            m = plotUtils.mapObj(llcrnrlon=llcrnrlon,llcrnrlat=llcrnrlat,urcrnrlon=urcrnrlon,urcrnrlat=urcrnrlat,resolution='l',area_thresh=1000.,projection=proj,ax=ax,fillContinents='None', fix_aspect=True)

    #    title = sTime.strftime('%H%M - ')+eTime.strftime('%H%M UT')
    #    title = sTime.strftime('Reverse Beacon Network %Y %b %d %H%M UT - ')+eTime.strftime('%Y %b %d %H%M UT')
        title = sTime.strftime('RBN: %d %b %Y %H%M UT - ')+eTime.strftime('%d %b %Y %H%M UT')
        ax.set_title(title)

        # draw parallels and meridians.
        m.drawparallels(np.arange(-90.,91.,45.),color='k',labels=[False,True,True,False],fontsize=tick_font_size)
        m.drawmeridians(np.arange(-180.,181.,45.),color='k',labels=[True,False,False,True],fontsize=tick_font_size)
        m.drawcoastlines(color='0.65')
        m.drawmapboundary(fill_color='w')
#        m.nightshade(plot_mTime,color='0.82')
        #if plotting the 2017 eclipse map then also draw state boundaries
        if eclipse:
            m.drawcountries(color='0.65')#np.arange(-90.,91.,45.),color='k',labels=[False,True,True,False],fontsize=tick_font_size)
            m.drawstates(color='0.65')

    m.scatter(df['de_lon'],df['de_lat'])

    return m,fig

def rbn_map_foF2(df,ax=None,legend=True,ssn='', kp='', tick_font_size=None,ncdxf=False,plot_paths=True,
        llcrnrlon=-180.,llcrnrlat=-90,urcrnrlon=180.,urcrnrlat=90.,proj='cyl',basemapType=True,eclipse=False,path_alpha=None):
    """Plot foF2 values derived from Reverse Beacon Network data for the midpoints between two stations.

    **Args**:
        * **[sTime]**: datetime.datetime object for start of plotting.
        * **[eTime]**: datetime.datetime object for end of plotting.
        * **[ymin]**: Y-Axis minimum limit
        * **[ymax]**: Y-Axis maximum limit
        * **[legendSize]**: Character size of the legend
        * **[df]**: DataFrame with 'midLat','midLon','foP' attributes/sections

    **Returns**:
        * **fig**:      matplotlib figure object that was plotted to

    .. note::
        If a matplotlib figure currently exists, it will be modified by this routine.  If not, a new one will be created.

    Written by Magda Moses and Nathaniel Frissell 2015 Sept 12
    """
    import datetime
    
    from matplotlib import pyplot as plt
    from mpl_toolkits.basemap import Basemap

    import numpy as np
    import pandas as pd

    import eclipse_lib

    from davitpy.pydarn.radar import *
    from davitpy.pydarn.plotting import *
    from davitpy.utils import *

    if ax is None:
        fig     = plt.figure(figsize=(10,6))
        ax      = fig.add_subplot(111)
    else:
        fig     = ax.get_figure()
    
    #If spliting up into sub-bands
    #Write Code later
    #Drop NaNs (QSOs without Lat/Lons)
    df = df.dropna(subset=['dx_lat','dx_lon','de_lat','de_lon'])

    ##Sort the data by band and time, then group by band.
    df['band']  = np.array((np.floor(df['foP']/1000.)),dtype=np.int)
    srt         = df.sort(['band','date'])
    grouped     = srt.groupby('band')
    
    sTime       = df['date'].min()
    eTime       = df['date'].max()

    half_time   = datetime.timedelta(seconds= ((eTime - sTime).total_seconds()/2.) )
    plot_mTime = sTime + half_time
    
    #Create Map
    #Use basemap if want to plot superdarn radars or other things with superdarn code on map
    if basemapType:
        m = Basemap(llcrnrlon=llcrnrlon,llcrnrlat=llcrnrlat,urcrnrlon=urcrnrlon,urcrnrlat=urcrnrlat,resolution='l',area_thresh=1000.,projection=proj,ax=ax)
    else:
        m = plotUtils.mapObj(llcrnrlon=llcrnrlon,llcrnrlat=llcrnrlat,urcrnrlon=urcrnrlon,urcrnrlat=urcrnrlat,resolution='l',area_thresh=1000.,projection=proj,ax=ax,fillContinents='None', fix_aspect=True)

#    title = sTime.strftime('%H%M - ')+eTime.strftime('%H%M UT')
#    title = sTime.strftime('Reverse Beacon Network %Y %b %d %H%M UT - ')+eTime.strftime('%Y %b %d %H%M UT')
    title = sTime.strftime('RBN Derived Plasma Frequency: %d %b %Y %H%M UT - ')+eTime.strftime('%d %b %Y %H%M UT')
    ax.set_title(title)

    # draw parallels and meridians.
    m.drawparallels(np.arange(-90.,91.,45.),color='k',labels=[False,True,True,False],fontsize=tick_font_size)
    m.drawmeridians(np.arange(-180.,181.,45.),color='k',labels=[True,False,False,True],fontsize=tick_font_size)
    m.drawcoastlines(color='0.65')
    m.drawmapboundary(fill_color='w')
    m.nightshade(plot_mTime,color='0.82')
    #if plotting the 2017 eclipse map then also draw state boundaries
    if eclipse:
        m.drawcountries(color='0.65')#np.arange(-90.,91.,45.),color='k',labels=[False,True,True,False],fontsize=tick_font_size)
        m.drawstates(color='0.65')
#    fof2_pt    = m.scatter(Lon,Lat,color=color,s=2,zorder=100)
#    Lat = row['midLat']
#    Lon = row['midLon']
    de_list = []
    dx_list = []
    for band in bandlist:
        try:
            this_group = grouped.get_group(band)
        except:
            continue

        color = band_dict[band]['color']
        label = band_dict[band]['name']

        for index,row in this_group.iterrows():
            #Yay stack overflow! - http://stackoverflow.com/questions/13888566/python-basemap-drawgreatcircle-function
            Lat = row['midLat']
            Lon = row['midLon']

            if row['callsign'] not in de_list: de_list.append(row['callsign'])
            if row['dx'] not in dx_list: dx_list.append(row['dx'])

            fof2_pt    = m.scatter(Lon,Lat,color=color,marker='s', s=2,zorder=100)

    text = []
    text.append('TX Stations: {0:d}'.format(len(dx_list)))
    text.append('RX Stations: {0:d}'.format(len(de_list)))
    text.append('Plotted Paths: {0:d}'.format(len(df)))
#    text.append('KP: '+kp[0])
    text.append('SSN: '+str(ssn))

    props = dict(facecolor='white', alpha=0.9,pad=6)
    ax.text(0.02,0.05,'\n'.join(text),transform=ax.transAxes,ha='left',va='bottom',size=9,bbox=props)

    if legend:
        band_legend()
#
    return m,fig

def rbn_region(df, latMin, lonMin, latMax, lonMax, constr_de=True, constr_dx=True):
    import numpy as np
    import pandas as pd
    """Limit the RBN links to a specific region
    **Args**:
        * **[df]: Data Frame with the format output by read_rbn  
        * **[latMin]: Lower Latitude Limit
        * **[lonMin]: Lower Longitude Limit
        * **[latMax]: Upper Latitude Limit
        * **[lonMax]: Upper Longitude Limit
        * **[constr_de]: Constrain the RBN recievers to the specified Lat/Lon limits
        * **[constr_dx]: Constrain the dx stations to the specified Lat/Lon limits
    **Returns**:
        * **[df2]: Dataframe containing only those links within the specified limits
    .. note:: Only Default conditions tested! By default constrains links to a given region but can be used to constrain only the de or dx stations by changing the args
    Written by Magda Moses and Carson Squibb 2015 August 03
    """
    import numpy as np
    import pandas as pd
    #Select which locations to constrain
    #Constrain Links
    if constr_de and constr_dx:
#        for i in range(0, len(df)-1): 
#        df2=df[latMin<=df['de_lat']<=latMax and lonMin<=df['de_lon']<=lonMax and latMin<=df['dx_lat']<=latMax and lonMin<=df['dx_lon']<=lonMax] 
#        df2=df[latMin<df['de_lat']<latMax and lonMin<df['de_lon']<lonMax and latMin<=df['dx_lat']<=latMax and lonMin<=df['dx_lon']<=lonMax] 
#        df2=df[latMin<=df['de_lat']<=latMax] 
#        df2=df2[lonMin<=df['de_lon']<=lonMax]
#        df2=df2[latMin<=df['dx_lat']<=latMax] 
#        df2=df2[lonMin<=df['dx_lon']<=lonMax] 

        df2=df[df['de_lat']>latMin] 
        df2=df2[df2['de_lat']<latMax] 
        df2=df2[df2['de_lon']>lonMin]
        df2=df2[df2['de_lon']<lonMax]
        df2=df2[df2['dx_lat']>latMin] 
        df2=df2[df2['dx_lat']<latMax] 
        df2=df2[df2['dx_lon']>lonMin]
        df2=df2[df2['dx_lon']<lonMax]

    #Constrain RBN recievers only
    elif constr_de and constr_dx==False:
        df2=df[df['de_lat']>latMin] 
        df2=df2[df2['de_lat']<latMax] 
        df2=df2[df2['de_lon']>lonMin]
        df2=df2[df2['de_lon']<lonMax]
#        for i in range(0, len(df)-1):
#        df2=df[latMin<=df['de_lat']<=latMax and lonMin<=df['de_lon']<=lonMax]

    #Constrain dx stations only
    elif constr_de==False and constr_dx:
        df2=df2[df2['dx_lat']>latMin] 
        df2=df2[df2['dx_lat']<latMax] 
        df2=df2[df2['dx_lon']>lonMin]
        df2=df2[df2['dx_lon']<lonMax]
#        for i in range(0, len(df)-1):
#        df2=df[latMin<=df['dx_lat']<=latMax and lonMin<=df['dx_lon']<=lonMax]

    #Cannnot constrain
    elif constr_de==False and constr_dx==False:
        print "Constraint False"
        
    return df2

def path_mid(de_lat, de_lon, dx_lat, dx_lon):
    """Find the latitude and longitude of the midpoint between the de and dx stations
    **Args**:
        * **[de_lat]:Latitude of the RBN reciever   
        * **[de_lon]:Longitude of the RBN reciever   
        * **[dx_lat]:Latitude of the dx station   
        * **[dx_lon]:Longitude of the dx station   
    **Returns**:
        * **[mlat]: Midpoint Latitude
        * **[mlon]: Midpoint Longitude
        * **[linkDist]: Link distance in meters  
        * **[dist]: Midpoint distance in meters
  
    .. note:: !

     dist
  
    .. note:: !

    Written by Magda Moses 2015 August 02
    """

    from davitpy.utils import *

    import numpy as np      #Numerical python - provides array types and operations
    import pandas as pd     #This is a nice utility for working with time-series type data.

    #Calculate the midpoint and the distance between the two stations
    d=greatCircleDist(de_lat, de_lon, dx_lat, dx_lon)
    azm=greatCircleAzm(de_lat, de_lon, dx_lat, dx_lon)
    mid=d/2
#    (mlat, mlon)=greatCircleMove(de_lat, de_lon, mid, azm)
    #The following is a slightly modified form of greatCircleMove from davitpy.utils.geoPack
    alt=0
    Re=6371.
    Re_tot = (Re + alt) * 1e3
    dist=mid*Re_tot
    linkDist=dist*2
    origLat=de_lat
    origLon=de_lon
    az=azm
#    dist = dist * 1e3
    lat1 = numpy.radians(origLat) 
    lon1 = numpy.radians(origLon)
    az = numpy.radians(az)
    
    lat2 = numpy.arcsin(numpy.sin(lat1)*numpy.cos(dist/Re_tot) +\
    numpy.cos(lat1)*numpy.sin(dist/Re_tot)*numpy.cos(az))
    lon2 = lon1 + numpy.arctan2(numpy.sin(az)*numpy.sin(dist/Re_tot)*numpy.cos(lat1),\
    numpy.cos(dist/Re_tot)-numpy.sin(lat1)*numpy.sin(lat2))

    ret_lat = numpy.degrees(lat2)
    ret_lon = numpy.degrees(lon2)
    
#    ret_lon = ret_lon % 360. 
#
#    tf = ret_lon > 180.
#    ret_lon[tf] = ret_lon - 360.
    mlat=ret_lat
    mlon=ret_lon
#    import ipdb; ipdb.set_trace()
    return mlat, mlon, linkDist, dist

def greatCircleKm(lat1,lon1,lat2,lon2, alt=0):
#    """Find the distance in km between two points the latitude and longitude of the midpoint between the de and dx stations
    """Calculates the distance in km along a great circle path between two points.
    **Args**:
        * **lat1**:  latitude [deg]
        * **lon1**:  longitude [deg]
        * **lat2**:  latitude [deg]
        * **lon2**:  longitude [deg]
        * **alt**:      altitude [km] (added to default Re = 6371 km)(6378.1 km)
    **Returns**:
        * **kmDist**:  distance [km]

    Written by Magda Moses 2015 October 05
    """

    from davitpy.utils import *

    import numpy as np      #Numerical python - provides array types and operations
    import pandas as pd     #This is a nice utility for working with time-series type data.
    
#    (mlat, mlon)=greatCircleMove(de_lat, de_lon, mid, azm)
    #The following is an expansion of the davitpy util greatCircleDist using parts of a slightly modified form of greatCircleMove from davitpy.utils.geoPack
    d=greatCircleDist(lat1,lon1,lat2,lon2)
    alt=0
    Re=6371.
#    Re_tot = (Re + alt) * 1e3
    Re_tot = (Re + alt)
    kmDist=d*Re_tot

    return kmDist

def get_geomagInd(sTime, eTime=None):
    """Get KP, AP, and SSN data for a date
    **Args**:
        * **[sTime]:The earliest time you want data for 
        * **[eTime]:The latest time you want data for (for our puposes it should be same as sTime)
    **Returns**:
        * **[]: 
        
    .. note:: Untested!???

    Written by Magda Moses 2015 August 06
    """
    import numpy as np
    import pandas as pd

    from davitpy import gme
    import datetime

##Normalize Time and correct day
##    realday=sTime.day-1
#    import ipdb; ipdb.set_trace()
##    norm_sTime=sTime.replace(day=realday, hour=0, minute=0, second=0)
#    norm_sTime=sTime.replace(hour=0, minute=0, second=0)
#    import ipdb; ipdb.set_trace()
#    if eTime==None:
#        norm_eTime=eTime
#    else:
#        realday=eTime.day-1
#        norm_eTime=eTime.replace(day=realday, hour=0, minute=0, second=0)
#        import ipdb; ipdb.set_trace()

#get Data
    aa=gme.ind.kp.readKp(sTime, eTime)
#    aa=gme.ind.kp.readKp(norm_sTime, norm_eTime)
#    =gme.ind.kp.readKp(dt.datetime(2015, 6, 28),dt.datetime(2015,  6, 28))
#    date=dt.datetime(2015, 6, 28)

#Check if the data was found by usual methods or by ftp and assign accordingly
    temp=aa[0]
#    import ipdb; ipdb.set_trace()
    realday=sTime.day+1
#    import ipdb; ipdb.set_trace()
    norm_sTime=sTime.replace(day=realday, hour=0, minute=0, second=0)
#    import ipdb; ipdb.set_trace()
    if norm_sTime == temp.time:
            print 'Found by usual methods'
            bb=temp
    else:
            #Convert sTime and eTime date into day of year
            t0 = sTime.timetuple()
#            t1 = eTime.timetuple()
#            import ipdb; ipdb.set_trace()
            b=t0.tm_yday
#            c=t1.tm_yday
#            import ipdb; ipdb.set_trace()
            day=b-1
            #Get data for desired day (sTime only for now!)
            bb=aa[day]
#            import ipdb; ipdb.set_trace()
#            day=c-1
#            cc=aa[day]
# Extract kp, ap and ssn values as integers
    kp=bb.kp
    ap=bb.ap
    kpSum=int(bb.kpSum)
    apMean=int(bb.apMean)
#    import ipdb; ipdb.set_trace()
    #Check if Sunspot data is availible or not for given sTime
    if bb.sunspot==None:
        print "No Sunspot Data Avalible from this source"
        ssn=None
#        import ipdb; ipdb.set_trace()
    else:
        ssn=int(bb.sunspot)
#        import ipdb; ipdb.set_trace()

    return kp, ap, kpSum, apMean, ssn

def get_hmF2(sTime,lat, lon, ssn=None, output=True):
    """Get hmF2 data for midpoint of RBN link
    **Args**:
        * **[sTime]:The earliest time you want data for 
        * **[eTime]:The latest time you want data for (for our puposes it should be same as sTime)
        * **[lat]: Latitude
        * **[lon]: Longitude
        * **[ssn]: Rz12 sunspot number
        * **[output]: Select output values. True=output all. False=Only output hmF2.
        * **[]: 
    **Returns**:
        * **[hmF2]: The height of the F2 layer 
        * **[outf]: An array with the output of irisub.for 
        * **[oarr]: Array with input parameters and array with additional output of irisub.for
        
    .. note:: Untested!????? (It probably has been at this point.)

    Written by Magda Moses 2015 August 06
    """
    import numpy as np
    import pandas as pd


    from davitpy.models import *
    from davitpy import utils
    import datetime

    # Inputs
    jf = [True]*50
    #jf[1]=False
    #uncomment next line to input ssn
    jf[2:6] = [False]*4
    jf[20] = False
    jf[22] = False
    jf[27:30] = [False]*3
    jf[32] = False
    jf[34] = False

    #Create Array for input variables(will also hold output values later) 
    oarr = np.zeros(100)

    #Decide if to have user input ssn
    if ssn!=None:
        jf[16]=False
        oarr[32]=ssn
    else: 
        jf[17]=True
#    import ipdb; ipdb.set_trace()
#    geographic   = 1 geomagnetic coordinates
    jmag = 0.
    #ALATI,ALONG: LATITUDE NORTH AND LONGITUDE EAST IN DEGREES
    alati = lat 
    along = lon
    # IYYYY: Year as YYYY, e.g. 1985
    iyyyy = sTime.year
#    import ipdb; ipdb.set_trace()
    #MMDD (-DDD): DATE (OR DAY OF YEAR AS A NEGATIVE NUMBER)
    t0 = sTime.timetuple()
    #Day of Year (doy)
    doy=t0.tm_yday
    mmdd = -doy 
    #DHOUR: LOCAL TIME (OR UNIVERSAL TIME + 25) IN DECIMAL HOURS
    decHour=float(sTime.hour)+float(sTime.minute)/60+float(sTime.second)/3600
    #Acording to the irisub.for comments, this should be equivilent to local time input
    #Need Testing !!!!!!
#    dhour=12+decHour-5
    dhour=decHour+25
#    import ipdb; ipdb.set_trace()
    #HEIBEG,HEIEND,HEISTP: HEIGHT RANGE IN KM; maximal 100 heights, i.e. int((heiend-heibeg)/heistp)+1.le.100
    heibeg, heiend, heistp = 80., 500., 10. 
#    heibeg, heiend, heistp = 350, 350., 0. 
    outf=np.zeros(20)
    outf,oarr = iri.iri_sub(jf,jmag,alati,along,iyyyy,mmdd,dhour,heibeg,heiend,heistp,oarr)
    hmF2=oarr[1]
#    foF2=np.sqrt(oarr[0]/(1.24e10))
#    import ipdb; ipdb.set_trace()
    if(output):
        return hmF2, outf, oarr
    else:
        return hmF2

#def rbn_fof2():
    
#    return
def count_band(df1, sTime, eTime,Inc_eTime=True,freq1=7000, freq2=14000, freq3=28000,dt=10,unit='minutes',xRot=False, ret_lim=False, rti_plot=False):
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
    tDelta=datetime.timedelta(minutes=dt)
    index=0
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
#    import ipdb; ipdb.set_trace()
    if times[len(times)-1]>=eTime:
#        import ipdb; ipdb.set_trace()
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

    #import ipdb; ipdb.set_trace()
    
    #Group counts together by unit time
    #index=0
    #define array to hold spot count
    spots=np.zeros(len(times))
    
    #CARSON VARIABLES: Spot counters for previous frequencies
    #spots0=np.zeros(len(times))
    spots1=np.zeros(len(times))
    spots2=np.zeros(len(times))
    spots3=np.zeros(len(times))
    #END

    #import ipdb; ipdb.set_trace()
    cTime=sTime
    endTime=cTime
#    #Read RBN data for given dates/times
#    #call function to get rbn data, find de_lat, de_lon, dx_lat, dx_lon for the data
#    rbn_df=rbn_lib.k4kdj_rbn(sTime, t_end, data_dir='data/rbn')
#    #create data frame for the loop
#    df1=rbn_df[rbn_df['callsign']=='K4KDJ']
#    import ipdb; ipdb.set_trace()
#    rbn_df2=rbn_df
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
           # elif df2.freq.iloc[I]>(freq0-500) and df2.freq.iloc[I]<(freq0+500):
           #     spots0[index]+=1
        #Itterate current time value and index
        cTime=endTime
        index=index+1

    #create Data Frame from spots and times vectors
    spot_df=pd.DataFrame(data=times, columns=['dates'])
    #spot_df['Count_F0']=spots0
    spot_df['Count_F1']=spots1
    spot_df['Count_F2']=spots2
    spot_df['Count_F3']=spots3
    #spot_df=pd.DataFrame(data=spots, columns=['Count'])
    #import ipdb; ipdb.set_trace()

    #now isolate those on the day side
    #now we need to constrain the data to those contacts that are only on the day side 
    #will need to make this more elegant and universal
    #I just wrote a quick code to isolate it for ONE EXAMPLE



    #Plot figures
    #fig=plt.figure()#generate a figure
    if rti_plot==False:
        fig, ((ax1),(ax2),(ax3))=plt.subplots(3,1,sharex=True,sharey=False)
    elif rti_plot==True:
        fig, ((ax1),(ax2),(ax3),ax4)=plt.subplots(4,1,sharex=True,sharey=False)
    if xRot==True:
        plt.xticks(rotation=30)
    #ax.plot(spot_df['dates'], spot_df['Count_F1'],'r*-',spot_df['dates'],spot_df['Count_F2'],'b*-',spot_df['dates'],spot_df['Count_F3'],'g*-')
    #ax0.plot(spot_df['dates'], spot_df['Count_F0'],'y*-')
    ax1.plot(spot_df['dates'], spot_df['Count_F1'],'r*-')
    axes=plt.gca()
#    import ipdb; ipdb.set_trace()
#    DumLim1=[0,spot_df['Count_F1'].max()]
    DumLim1=ax1.get_ylim()
    ax2.plot(spot_df['dates'], spot_df['Count_F2'],'b*-')
#    import ipdb; ipdb.set_trace()
#    axes=plt.gca()
#    import ipdb; ipdb.set_trace()
#    DumLim2=[0,spot_df['Count_F2'].max()]
    DumLim2=ax2.get_ylim()
#    axes=plt.gca()
#    DumLim3=[0,spot_df['Count_F3'].max()]
    DumLim3=ax3.get_ylim()
    ax3.plot(spot_df['dates'], spot_df['Count_F3'],'g*-')

    ax1.set_title('RBN Spots per Unit Time\n'+sTime.strftime('%d %b %Y %H%M UT - ')+eTime.strftime('%d %b %Y %H%M UT'))
    #ax0.set_ylabel(str(freq0/1000)+' MHz')
    ax1.set_ylabel(str(freq1/1000)+' MHz')
    ax2.set_ylabel(str(freq2/1000)+' MHz')
    ax3.set_ylabel(str(freq3/1000)+' MHz')
    ax3.set_xlabel('Time [UT]')
    #Freq1=patches.Patch(color='red',label='3 MHz')
    #Freq2=patches.Patch(color='blue',label='14 MHz')
    #Freq3=patches.Patch(color='green',label='28 MHz')
    #plt.legend(['3 MHz','14 MHz','28 MHz'])

    #ax.text(spot_df.dates.min(),spot_df.Count.min(),'Unit Time: '+str(dt)+' '+unit)
    #ax.text(spot_df.dates[10],spot_df.Count.max(),'Unit Time: '+str(dt)+' '+unit)
    #Need to Rewrite the return statements
    if rti_plot==True:
        return fig, ax1, ax2, ax3, ax4
    if ret_lim==True: 
        return fig, ax1, ax2, ax3, DumLim1, DumLim2, DumLim3
    else:
        return fig, ax1, ax2, ax3

def getLinks(df, center, radius): 
    """Gets links with midpoints within a specified radius of an iononsonde or other reference point
    **Args**:
        * **df**:  rbn data frame
        * **[center]**:  a lat/lon array for the center point [deg]
        * **radius**:  radius of the region to evaluate[km]
    **Returns**:
        * **df**:  data frame with added columns for the midpoint lat/lon, the link distance, the midipoint legnth and the midpoint distance from the center [km]
        * **link_df**: data frame with only those links whose midpoints fall within the specified radius from center
  
    .. note:: 

    Written by Magda Moses 2015 October 05
    """
    import numpy as np
    import pandas as pd

    from davitpy import gme
    from davitpy.utils import *
    import datetime

    import rbn_lib
        #Evaluate each link
    midLat=np.zeros([len(df), 1])
    #import ipdb; ipdb.set_trace()
    midLon=np.zeros([len(df), 1])
    l_dist=np.zeros([len(df), 1])
    m_dist=np.zeros([len(df), 1])
    dist=np.zeros([len(df), 1])
    h=np.zeros([len(df), 1])
    theta=np.zeros([len(df), 1])
    fp=np.zeros([len(df), 1])

    for i in range(0, len(df)): 
        #Isolate the ith link
        deLat=df.de_lat.iloc[i]
        deLon=df.de_lon.iloc[i]
        dxLat=df.dx_lat.iloc[i]
        dxLon=df.dx_lon.iloc[i]
        time=df.date.iloc[i]
    #    import ipdb; ipdb.set_trace()
        
        #Calculate the midpoint and the distance between the two stations
        midLat[i], midLon[i],l_dist[i],m_dist[i] =rbn_lib.path_mid(deLat, deLon, dxLat, dxLon)
        #Convert l_dist and m_dist to km
        l_dist[i]=(l_dist[i])/1e3
        m_dist[i]=(m_dist[i])/1e3

        #Calculate the distance of the midpoint from the ionosonde/center of the reference area
        dist[i]=rbn_lib.greatCircleKm(center[0],center[1], midLat[i],midLon[i])

    #Save information in data frame
    df['midLat']=midLat
    df['midLon']=midLon
    df['link_dist']=l_dist
    df['m_dist']=m_dist
    df['dist']=dist
    #Plasma Frequency in kHz
    #df['Freq_plasma']=fp
    #df['foP']=fp
#    import ipdb; ipdb.set_trace()

    #Limit links to those with a midpoint within the radius of the center
    link_df=df[df.dist<=radius]
    return df, link_df

def band_averages(df, freq1, freq2):
    """Divides the rbn link data by band and Gets the average distance and frequency of each band.  
    **Args**:
        * **df**:  rbn data frame
        * **freq1**:  frequencies interested in
        * **freq2**:  frequencies interested in
        * **dt**:  time interval averages taken over.
    **Returns**:
        * **df_avg**:  data frame with the average frequency and distance for each band

    .. note:: !

    Written by Magda Moses 2015 October 10
    """
    import numpy as np
    import pandas as pd 

    import datetime

    import rbn_lib

#    tDelta=datetime.timedelta(minutes=dt)

    #split into two dataframes based on frequency
    df1=df[df.freq>freq1-500]
    df1=df1[df1.freq<freq1+500]
    df2=df[df.freq>freq2-500]
    df2=df2[df2.freq<freq2+500]
#    import ipdb; ipdb.set_trace()
#    import ipdb; ipdb.set_trace()
#    df2=df[df.freq>freq2-500 && df.freq<freq2+500]
    count1=len(df1)
    count2=len(df2)
    df1=df1.mean()
#    import ipdb; ipdb.set_trace()
    df2=df2.mean()
    f1=df1.freq
    f2=df2.freq
    d1=df1.link_dist
    d2=df2.link_dist
#    import ipdb; ipdb.set_trace()
#    d1=df1.dist
#    d2=df2.dist
#    import ipdb; ipdb.set_trace()
#    output=[count1, count2, f1, f2, d1, d2]
    count=[count1, count2]
#    import ipdb; ipdb.set_trace()
#    import ipdb; ipdb.set_trace()

#    #Solve the critical frequncy equation
    
    
#    R
#    spots=np.zeros(len(times))
#    
#    #CARSON VARIABLES: Spot counters for previous frequencies
#    #spots0=np.zeros(len(times))
#    spots1=np.zeros(len(times))
#    spots2=np.zeros(len(times))
#    spots3=np.zeros(len(times))
#    for I in range(0,len(df)-1):
#        if df.freq.iloc[I]>(freq1-500) and df.freq.iloc[I]<(freq1+500):
#            J=J+1
#            spots1[index]+=1
#        elif df.freq.iloc[I]>(freq2-500) and df.freq.iloc[I]<(freq2+500): 
#            J=J+1
#            spots2[index]+=1

#    #Group by band and get the averages for each band
#    df_band=df.groupby['band']
#    d_avg=df_band.dist.mean()
#    f_avg=df_band.freq.mean()
#    #create 
#
#    df['band']  = np.array((np.floor(df['freq']/1000.)),dtype=np.int)
#    srt         = df.sort(['band','date'])
#    grouped     = srt.groupby('band')
#
#    for band in bandlist:
#        try:
#            this_group = grouped.get_group(band)
#        except:
#            continue
#
##    df_avg=pd.DataFrame({'d_avg':[davg.iloc[0],davg.iloc[1],davg.iloc[2]]})
#    df_avg=
#    import ipdb; ipdb.set_trace()
#    df_avg=df.band
#    #Sort the data by band
##    for band in bandlist: 
##        if band==None: 
#            
#     
##    df['']
##    ##Sort the data by band and time, then group by band.
##    df['band']  = np.array((np.floor(df['freq']/1000.)),dtype=np.int)
##    srt         = df.sort(['band','date'])
##    grouped     = srt.groupby('band')
#
##    for band in grouped:
##    for band in bandlist:
##        try:
##            this_group = grouped.get_group(band)
##        except:
##            continue
##
##        color = band_dict[band]['color']
##        label = band_dict[band]['name']
##
##        for index,row in this_group.iterrows():
#    return df_avg
#    return df1, df2, count1, count2, f1, f2, d1, d2
#    return df1, df2,output
    return df1, df2,count

def fc_stack_plot(df,sTime, eTime, freq1, freq2,nx_plots=1,ny_plots=5, xsize=8, ysize=4, ncol=None,plot_legend=False):
    """Creates stack plots of the average values used in the critical frequency calculations and the virtual height and critical frequency obtained from the rbn data
    **Args**:
        * **df**:  critical frequncy output dataframe
        * **xsize**:  size of the x axis of the plots 
        * **ysize**:  size of the y axis of the plots 
        * **ncol**:  the number of columns for the legend (should only be 2) 
    **Returns**:
        * **fig**: figure with stack plots 

    .. note:: Untested!

    Written by Magda Moses and Nathaniel Frissell 2015 October 20
    """
    from matplotlib import pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.markers as mmarkers

    import numpy as np
    import pandas as pd

    ##Sort the data by band to determine color 
#    band1=[]
#    band2=[]
#    df['band']  = np.array((np.floor(df['freq']/1000.)),dtype=np.int)
#    band1=(np.floor(df['f1']/1000.)),dtype=np.int)
    import ipdb; ipdb.set_trace()
    band1=np.int(np.floor(freq1/1000.))
    band2=np.int(np.floor(freq2/1000.))
#    band2.append(np.array((np.floor(df['f2']/1000.)),dtype=np.int))
#    band1  = np.floor(df['f1']/1000.)
#    band2  = np.floor(df['f2']/1000.)

    color1 = band_dict[band1]['color']
    color2 = band_dict[band2]['color']
#    label1 = band_dict[band1]['name']
#    label2 = band_dict[band2]['name']
    label1 = band_dict[band1]['freq']
    label2 = band_dict[band2]['freq']
    import ipdb; ipdb.set_trace()

#    color3='magenta'
    color3='m'
#    color4='red'
    color4='r'
#    color4=(0.75, 0.25, 0.75)
    label3='h_virtual'
    label4='hmF2'

#    import ipdb; ipdb.set_trace()
#    marker1='*-'
#    freq1_prop        = {'marker':'*','color':color1,'linestyle':'_draw_solid'}
#    freq2_prop        = {'marker':'*','color':color2,'linestyle':'_draw_solid'}
#    hv_prop           = {'marker':'*','color':color3,'linestyle':'_draw_solid'}
#    hmF2_prop         = {'marker':'*','color':color4,'linestyle':'_draw_solid'}

    #Uncomment the next 57 lines to get original plotting code
    fig         = plt.figure(figsize=(nx_plots*xsize,ny_plots*ysize)) # Create figure with the appropriate size.
    #plot 
    ax0     = fig.add_subplot(ny_plots,nx_plots,1)
    ax1     = fig.add_subplot(ny_plots,nx_plots,2)
    ax2     = fig.add_subplot(ny_plots,nx_plots,3)
    ax3     = fig.add_subplot(ny_plots,nx_plots,4)
    ax4     = fig.add_subplot(ny_plots,nx_plots,5)
#    #ax1     = fig.add_subplot(ny_plots,1,2)
#    #ax2     = fig.add_subplot(1,ny_plots,3)
#    #ax3     = fig.add_subplot(1,ny_plots,4)
#    #fig, ((ax0),(ax1),(ax2),(ax3))=plt.subplots(4,1,sharex=True,sharey=False)
#    #fig, ((ax0),(ax1),(ax2),(ax3), (ax4))=plt.subplots(5,1,sharex=True,sharey=False)
#    #fig, ((ax5),(ax0),(ax1),(ax2),(ax3))=plt.subplots(5,1,sharex=True,sharey=False)
#    #m, fig=rbn_lib.rbn_map_plot(df_links,legend=True,ax=ax5,tick_font_size=9,ncdxf=True, llcrnrlon=llcrnrlon, llcrnrlat=llcrnrlat, urcrnrlon=urcrnrlon, urcrnrlat=urcrnrlat)
#    #plot data on the same figure
#    ax0.plot(df['date'], df['count1'], '*-y',df['date'], df['count2'], '*-g')
#    ax1.plot(df['date'], df['d1'], '*-y',df['date'], df['d2'], '*-g')
#    ax2.plot(df['date'], df['f1'], '*-y',df['date'], df['f2'], '*-g')
#    ax3.plot(df['date'], df['hv'], '*-m',df['date'], df['hmF2'],'*-r')
#    #ax3.plot(df['date'], df['hv'], '*-m')
#    ax4.plot(df['date'], df['fc1'], '*-y',df['date'], df['fc2'], '*-g')
#    #ax3.plot(df['date'], df['fc1'], '-y',df['date'], df['fc2'], '-g')
#    #Alternate color plots
#    #ax0.plot(df['date'], df['count1'], '-r',df['date'], df['count2'], '-b')
#    #ax1.plot(df['date'], df['d1'], '-r',df['date'], df['d2'], '-b')
#    #ax2.plot(df['date'], df['f1'], '-r',df['date'], df['f2'], '-b')
#    #ax3.plot(df['date'], df['hv'], '-g')
#    #ax4.plot(df['date'], df['fc1'], '-r',df['date'], df['fc2'], '-b')
#    ##ax3.plot(df['date'], df['fc1'], '-r',df['date'], df['fc2'], '-b')
#    #plot data on the same figure
#    import ipdb; ipdb.set_trace()
##    ax0.plot(df['date'], df['count1'], s=freq1_prop, df['date'], df['count2'], s=freq2_prop)
#    ax0.plot(df['date'], df['count1'], freq1_prop,  df['date'], df['count2'], 'color', color2, 'marker',marker1)
#    import ipdb; ipdb.set_trace()
#    ax1.plot(df['date'], df['d1'], freq1_prop, df['date'], df['d2'], freq2_prop)
#    import ipdb; ipdb.set_trace()
#    ax2.plot(df['date'], df['f1'], freq1_prop, df['date'], df['f2'], freq2_prop)
#    import ipdb; ipdb.set_trace()
#    ax3.plot(df['date'], df['hv'], hv_prop, df['date'],  df['hmF2'],hmF2_prop)
#    import ipdb; ipdb.set_trace()
#    ax4.plot(df['date'], df['fc1'], freq1_prop,df['date'], df['fc2'], freq2_prop)
#    import ipdb; ipdb.set_trace()
    #plot data on the same figure
#    ax0.plot(df['date'], df['count1'], '-*'+color1,df['date'], df['count2'], color2)
#    ax1.plot(df['date'], df['d1'], color=freq1_prop['color'], linestyle=freq1_prop['linestyle'], marker=freq1_prop['marker'],df['date'], df['d2'], color2)
#    ax2.plot(df['date'], df['f1'], color1,df['date'], df['f2'], color2)
#    ax3.plot(df['date'], df['hv'], '-m')
#    ax4.plot(df['date'], df['fc1'], color1,df['date'], df['fc2'], color2)
#
#    fig         = plt.figure(figsize=(nx_plots*xsize,ny_plots*ysize)) # Create figure with the appropriate size.
#    #plot 
#    ax0     = fig.add_subplot(ny_plots,nx_plots,1)
#    ax1     = fig.add_subplot(ny_plots,nx_plots,2)
#    ax2     = fig.add_subplot(ny_plots,nx_plots,3)
#    ax3     = fig.add_subplot(ny_plots,nx_plots,4)
#    ax4     = fig.add_subplot(ny_plots,nx_plots,5)
#    #ax1     = fig.add_subplot(ny_plots,1,2)
#    #ax2     = fig.add_subplot(1,ny_plots,3)
#    #ax3     = fig.add_subplot(1,ny_plots,4)
#    #fig, ((ax0),(ax1),(ax2),(ax3))=plt.subplots(4,1,sharex=True,sharey=False)
#    #fig, ((ax0),(ax1),(ax2),(ax3), (ax4))=plt.subplots(5,1,sharex=True,sharey=False)
#    #fig, ((ax5),(ax0),(ax1),(ax2),(ax3))=plt.subplots(5,1,sharex=True,sharey=False)
#    #m, fig=rbn_lib.rbn_map_plot(df_links,legend=True,ax=ax5,tick_font_size=9,ncdxf=True, llcrnrlon=llcrnrlon, llcrnrlat=llcrnrlat, urcrnrlon=urcrnrlon, urcrnrlat=urcrnrlat)
#
    #plot data on the same figure
    ax0.plot(df['date'], df['count1'], color1,df['date'], df['count2'], color2)
    ax1.plot(df['date'], df['d1'], color1,df['date'], df['d2'], color2)
    ax2.plot(df['date'], df['f1'], color1,df['date'], df['f2'], color2)
    ax3.plot(df['date'], df['hv'], color3,df['date'], df['hmF2'],color4)
    ax4.plot(df['date'], df['fc1'], color1,df['date'], df['fc2'], color2)
    
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
#    #Set the title and labels for the plots
#    ax0.set_title('RBN Spots per Unit Time\n'+sTime.strftime('%d %b %Y %H%M UT - ')+eTime.strftime('%d %b %Y %H%M UT'))
#    #ax1.set_title('Average Link Distance per Unit Time\n'+sTime.strftime('%d %b %Y %H%M UT - ')+eTime.strftime('%d %b %Y %H%M UT'))
#    #ax2.set_title('Average Link Frequency per Unit Time\n'+sTime.strftime('%d %b %Y %H%M UT - ')+eTime.strftime('%d %b %Y %H%M UT'))
#    ###ax3.set_title('Calculated Virtual Height per unit timeper Unit Time\n'+sTime.strftime('%d %b %Y %H%M UT - ')+eTime.strftime('%d %b %Y %H%M UT'))
#    ###ax4.set_title('Calculated Critical Frequency per unit timeper Unit Time\n'+sTime.strftime('%d %b %Y %H%M UT - ')+eTime.strftime('%d %b %Y %H%M UT'))
#    ###ax4.set_xlabel('Time [UT]')
#    #ax3.set_title('Calculated Critical Frequency per Unit Time\n'+sTime.strftime('%d %b %Y %H%M UT - ')+eTime.strftime('%d %b %Y %H%M UT'))
#    #ax3.set_xlabel('Time [UT]')
#    #ax0.set_title('RBN Spots per Unit Time')
#    ax1.set_title('Average Link Distance per Unit Time')
#    ax2.set_title('Average Link Frequency per Unit Time')
#    ax3.set_title('Calculated Virtual Height per Unit Time')
#    ax4.set_title('Calculated Critical Frequency per Unit Time')
#    #ax3.set_title('Calculated Critical Frequency per Unit Time')
#
#    #set labels
#    ax0.set_ylabel('Count')
#    ax1.set_ylabel('Distance (km)')
#    ax2.set_ylabel('Freqency (kHz)')
#    ax3.set_ylabel('Height (km)')
#    ax4.set_ylabel('Freqency (kHz)')
#    ax4.set_xlabel('Time [UT]')
#    #ax3.set_xlabel('Time [UT]')
    
    #Add Legend
    handles=[]
    labels=[]
    if plot_legend:
#        if fig is None: fig = plt.gcf() 
        handles.append(mpatches.patch(color=color1,label=label1))
        labels.append(label1)
        handles.append(mpatches.patch(color=color2,label=label2))
        labels.append(label2)

        fig_tmp = plt.figure()
        ax_tmp = fig_tmp.add_subplot(111)
        ax_tmp.set_visible(False)
        if ncol is None:
            ncol = len(labels)
        
        legend = fig.legend(handles,labels,ncol=ncol,loc=loc,markerscale=markerscale,prop=prop,title=title,bbox_to_anchor=bbox_to_anchor,scatterpoints=1)

    return fig

#def freq_legend():
#    if fig is None: fig = plt.gcf() 
#    handles.append(mpatches.patch(color=color1,label=label1))
#    labels.append(label1)
#    handles.append(mpatches.patch(color=color2,label=label2))
#    labels.append(label2)
#
#    fig_tmp = plt.figure()
#    ax_tmp = fig_tmp.add_subplot(111)
#    ax_tmp.set_visible(False)
#    if ncol is None:
#        ncol = len(labels)
#    
#    legend = fig.legend(handles,labels,ncol=ncol,loc=loc,markerscale=markerscale,prop=prop,title=title,bbox_to_anchor=bbox_to_anchor,scatterpoints=1)
#
#    return legend

def rbn_crit_freq(df, time, coord_center, freq1=14000, freq2=7000):
    """Calculate ionospheric plasma characteristics from rbn data
    **Args**:
        * **df**:  rbn_links data frame
        * **time**:  the start and end ut time of the time interval calculations taken over.
        * **coord_center**: center lat and lon of the region 
        * **freq1**:  frequencies interested in
        * **freq2**:  frequencies interested in
    **Returns**:
        * **df_fc**:  data frame with the output values (calculated critial frequencies and virtual height and iri hmF2
                        df_fc=pd.DataFrame({'date':time[1], 'count1': count1, 'count2': count2, 'd1': d1,'d2': d2,'f1': f1,'f2': f2,'hv': hv, 'fc1': fc,'fc2':fc2, 'hmF2':hmF2})

    .. note:: !????? (It probably has been tested at this point.)

    Written by Magda Moses 2015 October 10
    """
    import numpy as np
    import pandas as pd 

    import datetime

    import rbn_lib
    

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
    print_time=[]
    count1=[]
    count2=[]
    output=[0,0,0,0,0,0]

#    #Find average frequency and distance/band
#    df1,df2,count1[i], count2[i], f1[i], f2[i], d1[i], d2[i]=rbn_lib.band_averages(rbn_links, freq1, freq2) 
#    df1,df2,count1[i], count2[i], f1[i], f2[i], d1[i], d2[i]=rbn_lib.band_averages(rbn_links, freq1, freq2) 
    df1,df2, count=rbn_lib.band_averages(df, freq1, freq2) 

    #Save averages in arrays
    count1.append(count[0])
    count2.append(count[1])
    f1.append(df1.freq)
    f2.append(df2.freq)
    d1.append(df1.link_dist)
    d2.append(df2.link_dist)

    #See if have enough information to solve for critical frequency
#    if df1.freq.isempty() or df2.freq.isempty():
#    if f1==0 or f2==0:
    if df1.freq==0 or df2.freq==0:
        fc.append('NA')
        hv.append('NA')
        hmF2.append('NA')
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
#        i=i+1
#        h[i]=np.sqrt((np.square(df2.freq*df1.dist)-np.square(df1.freq*df2.dist))/(np.square(df1.freq)-np.square(df2.freq)))/2
#        import ipdb; ipdb.set_trace()
#        fc[i]=np.sqrt(np.square(df1.freq)/(1+(df1.dist/(2*h[i]))))
#        import ipdb; ipdb.set_trace()
#        hv.append(np.sqrt((np.square(df2.freq*df1.dist)-np.square(df1.freq*df2.dist))/(np.square(df1.freq)-np.square(df2.freq)))/2)
#        import ipdb; ipdb.set_trace()
#        fc.append(np.sqrt(np.square(df1.freq)/(1+(df1.dist/(2*h[i])))))
#        import ipdb; ipdb.set_trace()

        #Get hmF2 for the start and end time
        h_start=rbn_lib.get_hmF2(time[0],coord_center[0], coord_center[1], output=False) 
        h_end=rbn_lib.get_hmF2(time[1],coord_center[0], coord_center[1], output=False) 
        hmF2.append((h_start+h_end)/2)
    #    hmF2.append(rbn_lib.get_hmF2(map_sTime,coord_center[0], coord_center[1], output=False)) 

    print_time.append(time[1])

    #save results in dataframe
#    df_fc=pd.DataFrame({'date':time[1], 'count1': count1, 'count2': count2, 'd1': d1,'d2': d2,'f1': f1,'f2': f2,'hv': hv, 'fc1': fc,'fc2':fc2, 'hmF2':hmF2})
    df_fc=pd.DataFrame({'date':print_time, 'count1': count1, 'count2': count2, 'd1': d1,'d2': d2,'f1': f1,'f2': f2,'hv': hv, 'fc1': fc,'fc2':fc2, 'hmF2':hmF2})

    return df_fc

