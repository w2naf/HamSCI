#!/usr/bin/env python
de_prop         = {'marker':'^','edgecolor':'k','facecolor':'white'}
dxf_prop        = {'marker':'*','color':'blue'}
dxf_leg_size    = 150
dxf_plot_size   = 50

import geopack
import os               # Provides utilities that help us do os-level operations like create directories
import datetime         # Really awesome module for working with dates and times.
import zipfile
import urllib2          # Used to automatically download data files from the web.
import pickle
import copy

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

def read_rbn(sTime,eTime=None,data_dir=None,qrz_call='w2naf',qrz_passwd='hamscience'):
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

        std_sTime=datetime.datetime(sTime.year,sTime.month,sTime.day, sTime.hour)
        if eTime.minute == 0 and eTime.second == 0:
            hourly_eTime=datetime.datetime(eTime.year,eTime.month,eTime.day, eTime.hour)
        else:
            hourly_eTime=eTime+datetime.timedelta(hours=1)
            hourly_eTime=datetime.datetime(hourly_eTime.year,hourly_eTime.month,hourly_eTime.day, hourly_eTime.hour)

        std_eTime=std_sTime+datetime.timedelta(hours=1)

        hour_flag=0
        while std_eTime<=hourly_eTime:
                p_filename = 'rbn_'+std_sTime.strftime('%Y%m%d%H%M-')+std_eTime.strftime('%Y%m%d%H%M.p')
                p_filepath = os.path.join(data_dir,p_filename)
                if not os.path.exists(p_filepath):
                    # Load data into dataframe here. ###############################################
                    with zipfile.ZipFile(data_path,'r') as z:   #This block lets us directly read the compressed gz file into memory.
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
                    hour_flag=hour_flag+1
                #When specified start/end times cross over the hour mark
                else:
                    df_comp=pd.concat([df_comp, df])

                std_sTime=std_eTime
                std_eTime=std_sTime+datetime.timedelta(hours=1)
        
        # Trim dataframe to just the entries we need.
        df = df_comp[np.logical_and(df_comp['date'] >= sTime,df_comp['date'] < eTime)]

#        df = df[np.logical_and(df['date'] >= sTime,df['date'] < eTime)]
#            tf = df['callsign'] == 'n7tr'
##            tf = df['callsign'] == 'ac0c'
#            df = df[tf]

#        lat1, lon1 = 41.0, -75.0
#        lat2, lon2 = 41.0, -123.0
        lat1, lon1  = df['de_lat'],df['de_lon']
        lat2, lon2  = df['dx_lat'],df['dx_lon']
        sp_mid_lat, sp_mid_lon = geopack.midpoint(lat1,lon1,lat2,lon2)

        df.loc[:,'sp_mid_lat']    = sp_mid_lat
        df.loc[:,'sp_mid_lon']    = sp_mid_lon

        df['band']  = np.array((np.floor(df['freq']/1000.)),dtype=np.int)

        return df

class RbnObject(object):
    def __init__(self,sTime=None,eTime=None,data_dir=None,
            qrz_call=None,qrz_passwd=None,comment='Raw Data',df=None):

        if df is None:
            df = read_rbn(sTime=sTime,eTime=eTime,data_dir=data_dir,
                    qrz_call=qrz_call,qrz_passwd=qrz_passwd)

        #Make metadata block to hold information about the processing.
        metadata = {}

        data_set                 = 'DS000'
        metadata['data_set_name'] = data_set
        metadata['serial']      = 0
        cmt     = '[{}] {}'.format(data_set,comment)
        #Save data to be returned as self.variables
        
        rbn_ds  = RbnDataSet(df,parent=self,comment=comment)
        setattr(self,data_set,rbn_ds)
        setattr(rbn_ds,'metadata',metadata)

        rbn_ds.dropna()


    def get_data_sets(self):
        """Return a sorted list of musicDataObj's contained in this musicArray.

        Returns
        -------
        data_sets : list of str
            Names of musicDataObj's contained in this musicArray.

        Written by Nathaniel A. Frissell, Summer 2016
        """

        attrs = dir(self)

        data_sets = []
        for item in attrs:
            if item.startswith('DS'):
                data_sets.append(item)
        data_sets.sort()
        return data_sets

class RbnDataSet(object):
    def __init__(self, df, comment=None, parent=0, **metadata):
        self.parent = parent

        self.df     = df
        self.metadata = {}
        self.metadata.update(metadata)

        self.history = {datetime.datetime.now():comment}

    def dropna(self,new_data_set='dropna',comment='Remove Non-Geolocated Spots'):
        new_ds      = self.copy(new_data_set,comment)
        new_ds.df   = new_ds.df.dropna(subset=['dx_lat','dx_lon','de_lat','de_lon'])
        new_ds.set_active()
        return new_ds

    def latlon_filt(self,lat_col='sp_mid_lat',lon_col='sp_mid_lon',
        llcrnrlon=-180.,llcrnrlat=-90,urcrnrlon=180.,urcrnrlat=90.):

        arg_dct = {'lat_col':lat_col,'lon_col':lon_col,'llcrnrlon':llcrnrlon,'llcrnrlat':llcrnrlat,'urcrnrlon':urcrnrlon,'urcrnrlat':urcrnrlat}
        new_ds  = self.apply(latlon_filt,arg_dct)

        md_up   = {'llcrnrlon':llcrnrlon,'llcrnrlat':llcrnrlat,'urcrnrlon':urcrnrlon,'urcrnrlat':urcrnrlat}
        new_ds.metadata.update(md_up)
        return new_ds

    def get_band_group(self,band):
        if not hasattr(self,'band_groups'):
            srt                 = self.df.sort_values(by=['band','date'])
            self.band_groups    = srt.groupby('band')

        try:
            this_group  = self.band_groups.get_group(band)
        except:
            this_group  = None

        return this_group

    def dedx_list(self):
        """
        Return unique, sorted lists of DE and DX stations in a dataframe.
        """
        de_list = self.df['callsign'].unique().tolist()
        dx_list = self.df['dx'].unique().tolist()

        de_list.sort()
        dx_list.sort()

        return (de_list,dx_list)

    def apply(self,function,arg_dct,new_data_set=None,comment=None):
        if new_data_set is None:
            new_data_set = function.func_name

        if comment is None:
            comment = str(arg_dct)

        new_ds      = self.copy(new_data_set,comment)
        new_ds.df   = function(self.df,**arg_dct)
        new_ds.set_active()

        return new_ds

    def copy(self,new_data_set,comment):
        """Copy a RbnDataSet object.  This deep copies data and metadata, updates the serial
        number, and logs a comment in the history.  Methods such as plot are kept as a reference.

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

        Written by Nathaniel A. Frissell, Summer 2016
        """

        serial = self.metadata['serial'] + 1
        new_data_set = '_'.join(['DS%03d' % serial,new_data_set])

        new_data_set_obj    = copy.copy(self)
        setattr(self.parent,new_data_set,new_data_set_obj)

        new_data_set_obj.df         = copy.deepcopy(self.df)
        new_data_set_obj.metadata   = copy.deepcopy(self.metadata)
        new_data_set_obj.history    = copy.deepcopy(self.history)

        new_data_set_obj.metadata['data_set_name']  = new_data_set
        new_data_set_obj.metadata['serial']         = serial
        new_data_set_obj.history[datetime.datetime.now()] = '['+new_data_set+'] '+comment
        
        return new_data_set_obj
  
    def set_active(self):
        """Sets this as the currently active data_set.

        Written by Nathaniel A. Frissell, Summer 2016
        """
        self.parent.active = self

    def print_metadata(self):
        """Nicely print all of the metadata associated with the current data_set.

        Written by Nathaniel A. Frissell, Summer 2016
        """
        keys = self.metadata.keys()
        keys.sort()
        for key in keys:
            print key+':',self.metadata[key]

    def append_history(self,comment):
        """Add an entry to the processing history dictionary of the current data_set object.

        Parameters
        ----------
        comment : string
            Infomation to add to history dictionary.

        Written by Nathaniel A. Frissell, Summer 2016
        """
        self.history[datetime.datetime.now()] = '['+self.metadata['data_set_name']+'] '+comment

    def print_history(self):
        """Nicely print all of the processing history associated with the current data_set object.

        Written by Nathaniel A. Frissell, Summer 2016
        """
        keys = self.history.keys()
        keys.sort()
        for key in keys:
            print key,self.history[key]


def band_legend(fig=None,loc='lower center',markerscale=0.5,prop={'size':10},
        title=None,bbox_to_anchor=None,ncdxf=False,ncol=None):

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
    df          = df.copy()
    lat_tf      = np.logical_and(df[lat_col] >= llcrnrlat,df[lat_col] < urcrnrlat)
    lon_tf      = np.logical_and(df[lon_col] >= llcrnrlon,df[lon_col] < urcrnrlon)
    tf          = np.logical_and(lat_tf,lon_tf)
    df          = df[tf]
    return df

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
    def __init__(self,rbn_obj,data_set='active',data_set_all='DS001_dropna',ax=None,tick_font_size=9,
            llcrnrlon=None,llcrnrlat=None,urcrnrlon=None,urcrnrlat=None):
        self.rbn_obj        = rbn_obj
        self.data_set       = getattr(rbn_obj,data_set)
        self.data_set_all   = getattr(rbn_obj,data_set_all)

        ds                  = self.data_set
        ds_md               = self.data_set.metadata

        llb = {}
        if llcrnrlon is None:
           llb['llcrnrlon'] = ds_md.get('llcrnrlon',-180.) 
        if llcrnrlat is None:
           llb['llcrnrlat'] = ds_md.get('llcrnrlat', -90.) 
        if urcrnrlon is None:
           llb['urcrnrlon'] = ds_md.get('urcrnrlon', 180.) 
        if urcrnrlat is None:
           llb['urcrnrlat'] = ds_md.get('urcrnrlat',  90.) 

        self.latlon_bnds    = llb

        self.metadata       = {}
        self.metadata['sTime'] = ds.df['date'].min()
        self.metadata['eTime'] = ds.df['date'].max()

        self.__setup_map__(ax=ax,tick_font_size=tick_font_size,**self.latlon_bnds)

    def default_plot(self,
            plot_de         = True,
            plot_midpoints  = True,
            plot_paths      = False,
            plot_ncdxf      = False,
            plot_stats      = True,
            plot_legend     = True):

        if plot_de:
            self.plot_de()
        if plot_midpoints:
            self.plot_midpoints()
        if plot_paths:
            self.plot_paths()
        if plot_ncdxf:
            self.plot_ncdxf()
        if plot_stats:
            self.plot_link_stats()
        if plot_legend:
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

    def plot_de(self,s=25,zorder=150):
        m       = self.m
        df      = self.data_set.df
        rx      = m.scatter(df['de_lon'],df['de_lat'],
                s=s,zorder=zorder,**de_prop)

    def plot_midpoints(self):
        for band in bandlist:
            this_group = self.data_set.get_band_group(band)
            if this_group is None: continue

            color = band_dict[band]['color']
            label = band_dict[band]['name']

            mid   = self.m.scatter(this_group['sp_mid_lon'],this_group['sp_mid_lat'],
                    alpha=0.25,facecolors=color,color=color,s=6,zorder=100)

    def plot_paths(self):
        m   = self.m
        for band in bandlist:
            this_group = self.data_set.get_band_group(band)
            if this_group is None: continue

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
        de_list_all, dx_list_all    = self.data_set_all.dedx_list()
        de_list_map, dx_list_map    = self.data_set.dedx_list() 

        text = []
        text.append('TX All: {0:d}; TX Map: {1:d}'.format( len(dx_list_all), len(dx_list_map) ))
        text.append('RX All: {0:d}; RX Map: {1:d}'.format( len(de_list_all), len(de_list_map) ))
        text.append('Plotted links: {0:d}'.format(len(self.data_set.df)))

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
