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
        llcrnrlon=-180.,llcrnrlat=-90,urcrnrlon=180.,urcrnrlat=90.):
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

    m = Basemap(llcrnrlon=llcrnrlon,llcrnrlat=llcrnrlat,urcrnrlon=urcrnrlon,urcrnrlat=urcrnrlat,resolution='l',area_thresh=1000.,projection='cyl',ax=ax)

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
                line, = m.drawgreatcircle(dx_lon,dx_lat,de_lon,de_lat,color=color)

                p = line.get_path()
                # find the index which crosses the dateline (the delta is large)
                cut_point = np.where(np.abs(np.diff(p.vertices[:, 0])) > 200)[0]
                if cut_point:
                    cut_point = cut_point[0]

                    # create new vertices with a nan inbetween and set those as the path's vertices
                    import ipdb; ipdb.set_trace()
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

    text = []
    text.append('TX Stations: {0:d}'.format(len(dx_list)))
    text.append('RX Stations: {0:d}'.format(len(de_list)))
    text.append('Plotted Paths: {0:d}'.format(len(df)))

    props = dict(facecolor='white', alpha=0.9,pad=6)
    ax.text(0.02,0.05,'\n'.join(text),transform=ax.transAxes,ha='left',va='bottom',size=9,bbox=props)

    if legend:
        band_legend()

    return fig
