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


def band_legend(fig=None,loc='lower center',markerscale=0.5,prop={'size':10},title=None,bbox_to_anchor=None,ncdxf=False,ncol=None):
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
    if ncdxf:
        dxf_df = pd.DataFrame.from_csv('ncdxf.csv')
        m.scatter(dxf_df['lon'],dxf_df['lat'],s=dxf_plot_size,**dxf_prop)

    #if eclipse:
     #   df_cl=eclipse_lib.eclipse_get_path(fname='ds_CL.csv')
     #   m.plot(df_cl['eLon'],df_cl['eLat'],'m--',label='2017 Eclipse Central Line', linewidth=2, latlon=True)

    import ipdb; ipdb.set_trace()
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

    if scatter_rbn==False:
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

        import ipdb; ipdb.set_trace()
        text = []
        text.append('TX Stations: {0:d}'.format(len(dx_list)))
        text.append('RX Stations: {0:d}'.format(len(de_list)))
        text.append('Plotted Paths: {0:d}'.format(len(df)))

        props = dict(facecolor='white', alpha=0.9,pad=6)
        ax.text(0.02,0.05,'\n'.join(text),transform=ax.transAxes,ha='left',va='bottom',size=9,bbox=props)

        if legend:
            band_legend()

    return m,fig

def rbn_map_foF2(df,ax=None,legend=True,tick_font_size=None,ncdxf=False,plot_paths=True,
        llcrnrlon=-180.,llcrnrlat=-90,urcrnrlon=180.,urcrnrlat=90.,proj='cyl',basemapType=True,eclipse=False,path_alpha=None):
    """Plot foF2 values derived from Reverse Beacon Network data for the midpoints between two stations.

    **Args**:
        * **[sTime]**: datetime.datetime object for start of plotting.
        * **[eTime]**: datetime.datetime object for end of plotting.
        * **[ymin]**: Y-Axis minimum limit
        * **[ymax]**: Y-Axis maximum limit
        * **[legendSize]**: Character size of the legend
        * **[df]**: DataFrame with 'midLat','midLon','Freq_plasma' attributes/sections

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

    #Drop NaNs (QSOs without Lat/Lons)
    df = df.dropna(subset=['dx_lat','dx_lon','de_lat','de_lon'])

    ##Sort the data by frequency sub-bands and time, then group by band.
    df['sub_band']  = np.array((np.floor(df['Freq_plasma']/1000.)),dtype=np.int)
    srt         = df.sort(['band','date'])
    grouped     = srt.groupby('band')

    sTime       = df['date'].min()
    eTime       = df['date'].max()

    half_time   = datetime.timedelta(seconds= ((eTime - sTime).total_seconds()/2.) )
    plot_mTime = sTime + half_time

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

        color = pf_band_dict[sub_band]['color']
        #Need to change this
        label = pf_band_dict[sub_band]['name']

        for index,row in this_group.iterrows():
            #Yay stack overflow! - http://stackoverflow.com/questions/13888566/python-basemap-drawgreatcircle-function
            Lat = row['midLat']
            Lon = row['midLon']
#            dx_lat = row['dx_lat']
#            dx_lon = row['dx_lon']

            if row['callsign'] not in de_list: de_list.append(row['callsign'])
            if row['dx'] not in dx_list: dx_list.append(row['dx'])

            fof2_pt    = m.scatter(Lon,Lat,color=color,s=2,zorder=100)
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

    import ipdb; ipdb.set_trace()
    text = []
    text.append('TX Stations: {0:d}'.format(len(dx_list)))
    text.append('RX Stations: {0:d}'.format(len(de_list)))
    text.append('Plotted Paths: {0:d}'.format(len(df)))

    props = dict(facecolor='white', alpha=0.9,pad=6)
    ax.text(0.02,0.05,'\n'.join(text),transform=ax.transAxes,ha='left',va='bottom',size=9,bbox=props)

    if legend:
        subBand_legend()

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
        * **[mid_lat]: Midpoint Latitude
        * **[mid_lon]: Midpoint Longitude
        
    .. note:: Untested!

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

def get_geomagInd(sTime, eTime=None):
    """Get KP, AP, and SSN data for a date
    **Args**:
        * **[sTime]:The earliest time you want data for 
        * **[eTime]:The latest time you want data for (for our puposes it should be same as sTime)
    **Returns**:
        * **[]: 
        
    .. note:: Untested!

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

def get_hmF2(sTime,lat, lon, ssn=None):
    """Get hmF2 data for midpoint of RBN link
    **Args**:
        * **[sTime]:The earliest time you want data for 
        * **[eTime]:The latest time you want data for (for our puposes it should be same as sTime)
        * **[lat]: Latitude
        * **[lon]: Longitude
        * **[ssn]: Rz12 sunspot number
        * **[]: 
    **Returns**:
        * **[hmF2]: The height of the F2 layer 
        * **[outf]: An array with the output of irisub.for 
        * **[oarr]: Array with input parameters and array with additional output of irisub.for
        
    .. note:: Untested!

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
    return hmF2, outf, oarr

#def rbn_fof2():
    
#    return
def count_band(df1, sTime, eTime,Inc_eTime=True,freq1=7000, freq2=14000, freq3=28000,dt=10,unit='minutes',xRot=False, ret_lim=False, rti_plot=None):
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
    if rti_plot==None:
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
