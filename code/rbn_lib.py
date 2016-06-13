#!/usr/bin/env python
de_prop         = {'marker':'^','color':'k'}
dxf_prop        = {'marker':'*','color':'blue'}
dxf_leg_size    = 150
dxf_plot_size   = 50

import geopack
import os               # Provides utilities that help us do os-level operations like create directories
import datetime         # Really awesome module for working with dates and times.
import zipfile
import urllib2          # Used to automatically download data files from the web.
import pickle

import numpy as np      #Numerical python - provides array types and operations
import pandas as pd     #This is a nice utility for working with time-series type data.

from hamtools import qrz

import matplotlib
matplotlib.use('Agg')

from matplotlib import pyplot as plt
from mpl_toolkits.basemap import Basemap
import matplotlib.patches as mpatches
import matplotlib.markers as mmarkers
from matplotlib.collections import PolyCollection

def read_rbn(sTime,eTime=None,data_dir=None,
             qrz_call='w2naf',qrz_passwd='hamscience'):
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

#            tf = df['callsign'] == 'N7TR'
##            tf = df['callsign'] == 'AC0C'
#            df = df[tf]


#        lat1, lon1 = 41.0, -75.0
#        lat2, lon2 = 41.0, -123.0
        lat1, lon1  = df['de_lat'],df['de_lon']
        lat2, lon2  = df['dx_lat'],df['dx_lon']
        sp_mid_lat, sp_mid_lon = geopack.midpoint(lat1,lon1,lat2,lon2)

        df['sp_mid_lat']    = sp_mid_lat
        df['sp_mid_lon']    = sp_mid_lon

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

    if fig is None: fig = plt.gcf() 

    handles = []
    labels  = []
    for band in bandlist:
        color = band_dict[band]['color']
        label = band_dict[band]['freq']
        handles.append(mpatches.Patch(color=color,label=label))
        labels.append(label)

    fig_tmp = plt.figure()
    ax_tmp = fig_tmp.add_subplot(111)
    ax_tmp.set_visible(False)
    scat = ax_tmp.scatter(0,0,s=50,**de_prop)
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

def latlon_filt(df,lat_col='sp_mid_lat',lon_col='sp_mid_lon',
        llcrnrlon=-180.,llcrnrlat=-90,urcrnrlon=180.,urcrnrlat=90.):
    """
    Return an RBN Dataframe with entries only within a specified lat/lon box.
    """
    df  = df.copy()
    lat_tf      = np.logical_and(df[lat_col] >= llcrnrlat,df[lat_col] < urcrnrlat)
    lon_tf      = np.logical_and(df[lon_col] >= llcrnrlon,df[lon_col] < urcrnrlon)
    tf          = np.logical_and(lat_tf,lon_tf)
    df          = df[tf]
    return df

def dedx_list(df):
    """
    Return unique, sorted lists of DE and DX stations in a dataframe.
    """
    de_list = df['callsign'].unique().tolist()
    dx_list = df['dx'].unique().tolist()

    de_list.sort()
    dx_list.sort()

    return (de_list,dx_list)

class RbnGeoGrid(object):
    """
    Define a geographic grid and bin RBN data.
    """
    def __init__(self,df=None,lat_col='sp_mid_lat',lon_col='sp_mid_lon',
        lat_min=-90. ,lat_max=90. ,lat_step=1.0,
        lon_min=-180.,lon_max=180.,lon_step=1.0):

        lat_vec         = np.arange(lat_min,lat_max,lat_step)
        lon_vec         = np.arange(lon_min,lon_max,lon_step)

        self.lat_min    = lat_min
        self.lat_max    = lat_max
        self.lat_step   = lat_step
        self.lon_min    = lon_min
        self.lon_max    = lon_max
        self.lon_step   = lon_step
        self.lat_vec    = lat_vec
        self.lon_vec    = lon_vec
        self.df         = df
        self.lat_col    = lat_col
        self.lon_col    = lon_col

    def grid_mean(self):
        df          = self.df
        lat_vec     = self.lat_vec
        lon_vec     = self.lon_vec
        lat_step    = self.lat_step
        lon_step    = self.lon_step
        lat_col     = self.lat_col
        lon_col     = self.lon_col

        data_arr    = np.ndarray((lat_vec.size,lon_vec.size),dtype=np.float)
        data_arr[:] = np.nan

        for lat_inx,lat in enumerate(lat_vec):
            for lon_inx,lon in enumerate(lon_vec):
                lat_tf  = np.logical_and(df[lat_col] >= lat, df[lat_col] < lat+lat_step)
                lon_tf  = np.logical_and(df[lon_col] >= lon, df[lon_col] < lon+lon_step)
                tf      = np.logical_and(lat_tf,lon_tf)
                if np.count_nonzero(tf) <= 0: continue
            
                cell_freq                   = df[tf]['freq'].mean()
                data_arr[lat_inx,lon_inx]   = cell_freq

        data_arr        = data_arr/1000.
        self.data_arr   = data_arr
        
##        data_arr    = np.ndarray

class RbnMap(object):
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
    def __init__(self,df,ax=None,tick_font_size=9,
            llcrnrlon=-180.,llcrnrlat=-90,urcrnrlon=180.,urcrnrlat=90.):
        md                  = {}
        self.metadata       = md
        self.latlon_bnds    = {'llcrnrlon':llcrnrlon,'llcrnrlat':llcrnrlat,'urcrnrlon':urcrnrlon,'urcrnrlat':urcrnrlat}

        self.__prep_dataframes__(df)
        self.__setup_map__(ax=ax,tick_font_size=tick_font_size,**self.latlon_bnds)

    def default_plot(self):
        self.plot_de()
        self.plot_midpoints()
#        self.plot_paths()
        self.plot_ncdxf()
        self.plot_link_stats()
        self.plot_band_legend()

    def __setup_map__(self,ax=None,tick_font_size=None,llcrnrlon=-180.,llcrnrlat=-90,urcrnrlon=180.,urcrnrlat=90.):
        sTime       = self.metadata['sTime']
        eTime       = self.metadata['eTime']

        if ax is None:
            fig     = plt.figure(figsize=(10,6))
            ax      = fig.add_subplot(111)
        else:
            fig     = ax.get_figure()

        m = Basemap(llcrnrlon=llcrnrlon,llcrnrlat=llcrnrlat,urcrnrlon=urcrnrlon,urcrnrlat=urcrnrlat,resolution='l',area_thresh=1000.,projection='cyl',ax=ax)

        title = sTime.strftime('RBN: %d %b %Y %H%M UT - ')+eTime.strftime('%d %b %Y %H%M UT')
        ax.set_title(title)

        # draw parallels and meridians.
        m.drawparallels(np.arange(-90.,91.,45.),color='k',labels=[False,True,True,False],fontsize=tick_font_size)
        m.drawmeridians(np.arange(-180.,181.,45.),color='k',labels=[True,False,False,True],fontsize=tick_font_size)
        m.drawcoastlines(color='0.65')
        m.drawmapboundary(fill_color='w')
        
        # Overlay nighttime terminator.
        half_time   = datetime.timedelta(seconds= ((eTime - sTime).total_seconds()/2.) )
        plot_mTime  = sTime + half_time
        m.nightshade(plot_mTime,color='0.50')

        # Expose select objects
        self.fig        = fig
        self.ax         = ax
        self.m          = m

    def __prep_dataframes__(self,df):
        #Drop NaNs (QSOs without Lat/Lons)
        df          = df.dropna(subset=['dx_lat','dx_lon','de_lat','de_lon'])
        df_all_nona = df.copy()

        # Filter the dataframe by map lat/lon bounds.
        # The midpoints are what will be plotted. However, keep track of de and dx stations on map
        # and overall for informational purposes.
        latlon_bnds = self.latlon_bnds
        df          = latlon_filt(df_all_nona,lat_col='sp_mid_lat',lon_col='sp_mid_lon',**latlon_bnds)
        df_de       = latlon_filt(df_all_nona,lat_col='de_lat',lon_col='de_lon',**latlon_bnds)
        df_dx       = latlon_filt(df_all_nona,lat_col='dx_lat',lon_col='dx_lon',**latlon_bnds)

        ##Sort the data by band and time, then group by band.
        df['band']  = np.array((np.floor(df['freq']/1000.)),dtype=np.int)
        srt         = df.sort(['band','date'])
        band_groups = srt.groupby('band')

        sTime       = df['date'].min()
        eTime       = df['date'].max()

        md          = self.metadata
        md['sTime'] = sTime
        md['eTime'] = eTime

        self.df_all_nona    = df_all_nona
        self.df             = df
        self.df_de          = df_de
        self.df_dx          = df_dx
        self.band_groups    = band_groups

    def plot_de(self,s=1,zorder=150):
        m       = self.m
        df_de   = self.df_de
        rx      = m.scatter(df_de['de_lon'],df_de['de_lat'],s=s,zorder=zorder,**de_prop)


    def plot_midpoints(self):
        for band in bandlist:
            try:
                this_group = self.band_groups.get_group(band)
            except:
                continue

            color = band_dict[band]['color']
            label = band_dict[band]['name']

            mid   = self.m.scatter(this_group['sp_mid_lon'],this_group['sp_mid_lat'],
                    alpha=0.25,facecolors=color,color=color,s=6,zorder=100)

    def plot_paths(self):
        m   = self.m
        for band in bandlist:
            try:
                this_group = self.band_groups.get_group(band)
            except:
                continue

            color = band_dict[band]['color']
            label = band_dict[band]['name']

            for index,row in this_group.iterrows():
                #Yay stack overflow! - http://stackoverflow.com/questions/13888566/python-basemap-drawgreatcircle-function
                de_lat  = row['de_lat']
                de_lon  = row['de_lon']
                dx_lat  = row['dx_lat']
                dx_lon  = row['dx_lon']
                line, = m.drawgreatcircle(dx_lon,dx_lat,de_lon,de_lat,color=color)

                p = line.get_path()
                # find the index which crosses the dateline (the delta is large)
                cut_point = np.where(np.abs(np.diff(p.vertices[:, 0])) > 200)[0]
                if cut_point:
                    cut_point = cut_point[0]

                    # create new vertices with a nan inbetween and set those as the path's vertices
                    new_verts = np.concatenate(
                                               [p.vertices[:cut_point, :], 
                                                [[np.nan, np.nan]], 
                                                p.vertices[cut_point+1:, :]]
                                               )
                    p.codes = None
                    p.vertices = new_verts

    def plot_ncdxf(self):
        dxf_df = pd.DataFrame.from_csv('ncdxf.csv')
        self.m.scatter(dxf_df['lon'],dxf_df['lat'],s=dxf_plot_size,**dxf_prop)

    def plot_link_stats(self):
        de_list_all, dx_list_all    = dedx_list(self.df_all_nona)
        de_list_map, _              = dedx_list(self.df_de)
        _, dx_list_map              = dedx_list(self.df_dx)

        text = []
        text.append('TX All: {0:d}; TX Map: {1:d}'.format( len(dx_list_all), len(dx_list_map) ))
        text.append('RX All: {0:d}; RX Map: {1:d}'.format( len(de_list_all), len(de_list_map) ))
        text.append('Plotted links: {0:d}'.format(len(self.df)))

        props = dict(facecolor='white', alpha=0.25,pad=6)
        self.ax.text(0.02,0.05,'\n'.join(text),transform=self.ax.transAxes,
                ha='left',va='bottom',size=9,zorder=500,bbox=props)

    def plot_band_legend(self,*args,**kw_args):
        band_legend(*args,**kw_args)

    def overlay_rbn_grid(self,rbn_grid_obj,color='0.8'):
        """
        Overlay the grid from an RbnGeoGrid object.
        """
        self.m.drawparallels(rbn_grid_obj.lat_vec,color=color)
        self.m.drawmeridians(rbn_grid_obj.lon_vec,color=color)

        m   = self.m
        rgo = rbn_grid_obj
        lat_vec = rgo.lat_vec
        lon_vec = rgo.lon_vec
        lat_step = rgo.lat_step
        lon_step = rgo.lon_step
        data_arr= rgo.data_arr

        scan    = []
        verts   = []
        for lat_inx,lat in enumerate(lat_vec):
            for lon_inx,lon in enumerate(lon_vec):
                data    = data_arr[lat_inx,lon_inx]
                if np.isnan(data): continue
                scan.append(data)

                x1,y1 = m(lon,lat)
                x2,y2 = m(lon,lat+lat_step)
                x3,y3 = m(lon+lon_step,lat+lat_step)
                x4,y4 = m(lon+lon_step,lat)
                verts.append(((x1,y1),(x2,y2),(x3,y3),(x4,y4),(x1,y1)))

#        scale   = (0.,30.)
#        cmap    = matplotlib.cm.jet
#        bounds  = np.linspace(scale[0],scale[1],256)
#        norm    = matplotlib.colors.BoundaryNorm(bounds,cmap.N)
#        pcoll   = PolyCollection(np.array(verts),edgecolors='face',closed=False,cmap=cmap,norm=norm,zorder=99)
#        pcoll.set_array(np.array(scan))
#        self.ax.add_collection(pcoll,autolim=False)

def rbn_map_plot(df,ax=None,legend=True,tick_font_size=9,ncdxf=False,plot_paths=True,
        llcrnrlon=-180.,llcrnrlat=-90,urcrnrlon=180.,urcrnrlat=90.):
    """
    Convenience wrapper to RbnMap to maintain backward compatibility with old
    rbn_map_plot().
    """

    rbn_map = RbnMap(df,ax=ax,tick_font_size=tick_font_size,
            llcrnrlon=llcrnrlon,llcrnrlat=llcrnrlat,urcrnrlon=urcrnrlon,urcrnrlat=urcrnrlat)

    rbn_map.plot_de()
    rbn_map.plot_midpoints()
    if plot_paths:
        rbn_map.plot_paths()
    if ncdxf:
        rbn_map.plot_ncdxf()
    rbn_map.plot_link_stats()
    if legend:
        rbn_map.plot_band_legend()

    return rbn_map
