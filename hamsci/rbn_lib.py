#!/usr/bin/env python
de_prop         = {'marker':'^','edgecolor':'k','facecolor':'white'}
dxf_prop        = {'marker':'*','color':'blue'}
dxf_leg_size    = 150
dxf_plot_size   = 50
Re              = 6371  # Radius of the Earth

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

import davitpy

import matplotlib
matplotlib.use('Agg')

from matplotlib import pyplot as plt
from mpl_toolkits.basemap import Basemap
import matplotlib.patches as mpatches
import matplotlib.markers as mmarkers
from matplotlib.collections import PolyCollection

import gridsquare

def cc255(color):
    cc = matplotlib.colors.ColorConverter().to_rgb
    trip = np.array(cc(color))*255
    trip = [int(x) for x in trip]
    return tuple(trip)

class BandData(object):
    def __init__(self,cmap='HFRadio',vmin=0.,vmax=30.):
        if cmap == 'HFRadio':
            self.cmap   = self.hf_cmap(vmin=vmin,vmax=vmax)
        else:
            self.cmap   = matplotlib.cm.get_cmap(cmap)

        self.norm   = matplotlib.colors.Normalize(vmin=vmin,vmax=vmax)

        # Set up a dictionary which identifies which bands we want and some plotting attributes for each band
        bands   = []
        bands.append((28.0,  '10 m'))
        bands.append((21.0,  '15 m'))
        bands.append((14.0,  '20 m'))
        bands.append(( 7.0,  '40 m'))
        bands.append(( 3.5,  '80 m'))
        bands.append(( 1.8, '160 m'))

        self.__gen_band_dict__(bands)

    def __gen_band_dict__(self,bands):
        dct = {}
        for freq,name in bands:
            key = int(freq)
            tmp = {}
            tmp['name']         = name
            tmp['freq']         = freq
            tmp['freq_name']    = '{:g} MHz'.format(freq)
            tmp['color']        = self.get_rgba(freq)
            dct[key]            = tmp
        self.band_dict          = dct

    def get_rgba(self,freq):
        nrm     = self.norm(freq)
        rgba    = self.cmap(nrm)
        return rgba

    def hf_cmap(self,name='HFRadio',vmin=0.,vmax=30.):
	fc = {}
        my_cdict = fc
	fc[ 0.0] = (  0,   0,   0)
	fc[ 1.8] = cc255('violet')
	fc[ 3.0] = cc255('blue')
	fc[ 8.0] = cc255('aqua')
	fc[10.0] = cc255('green')
	fc[13.0] = cc255('green')
	fc[17.0] = cc255('yellow')
	fc[21.0] = cc255('orange')
	fc[28.0] = cc255('red')
	fc[30.0] = cc255('red')
        cmap    = cdict_to_cmap(fc,name=name,vmin=vmin,vmax=vmax)
	return cmap

def cdict_to_cmap(cdict,name='CustomCMAP',vmin=0.,vmax=30.):
	norm = matplotlib.colors.Normalize(vmin=vmin,vmax=vmax)
	
	red   = []
	green = []
	blue  = []
	
	keys = cdict.keys()
	keys.sort()
	
	for x in keys:
	    r,g,b, = cdict[x]
	    x = norm(x)
	    r = r/255.
	    g = g/255.
	    b = b/255.
	    red.append(   (x, r, r))
	    green.append( (x, g, g))
	    blue.append(  (x, b, b))
	cdict = {'red'   : tuple(red),
		 'green' : tuple(green),
		 'blue'  : tuple(blue)}
	cmap  = matplotlib.colors.LinearSegmentedColormap(name, cdict)
	return cmap

def read_rbn(sTime,eTime=None,data_dir='data/rbn',qrz_call=None,qrz_passwd=None):
    if data_dir is None: data_dir = os.getenv('DAVIT_TMPDIR')

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
             status = 'Done downloading!  Now converting to Pandas dataframe...'
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
                    if total == 0:
                        print "No call signs geolocated."
                    else:
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

        # Calculate Total Great Circle Path Distance
        lat1, lon1          = df['de_lat'],df['de_lon']
        lat2, lon2          = df['dx_lat'],df['dx_lon']
        R_gc                = Re*geopack.greatCircleDist(lat1,lon1,lat2,lon2)
        df.loc[:,'R_gc']    = R_gc

        # Calculate Band
        df.loc[:,'band']        = np.array((np.floor(df['freq']/1000.)),dtype=np.int)

        return df

class RbnObject(object):
    """
    gridsquare_precision:   Even number, typically 4 or 6
    reflection_type:        Model used to determine reflection point in ionopshere.
                            'sp_mid': spherical midpoint
    """
    def __init__(self,sTime=None,eTime=None,data_dir='data/rbn',
            qrz_call=None,qrz_passwd=None,comment='Raw Data',df=None,
            gridsquare_precision=4,reflection_type='sp_mid'):

        if df is None:
            df = read_rbn(sTime=sTime,eTime=eTime,data_dir=data_dir,
                    qrz_call=qrz_call,qrz_passwd=qrz_passwd)

        #Make metadata block to hold information about the processing.
        metadata = {}
        data_set                            = 'DS000'
        metadata['data_set_name']           = data_set
        metadata['serial']                  = 0
        cmt     = '[{}] {}'.format(data_set,comment)
        
        rbn_ds  = RbnDataSet(df,parent=self,comment=cmt)
        setattr(self,data_set,rbn_ds)
        setattr(rbn_ds,'metadata',metadata)

        rbn_ds  = rbn_ds.dropna()

        if reflection_type == 'sp_mid':
            rbn_ds = rbn_ds.calc_reflection_sp_mid()

        rbn_ds.grid_data(gridsquare_precision=gridsquare_precision) 

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

    def geo_loc_stats(self,verbose=True):
        # Figure out how many records properly geolocated.
        good_loc        = rbn_obj.DS001_dropna.df
        good_count_map  = good_loc['callsign'].count()
        total_count_map = len(rbn_obj.DS000.df)
        good_pct_map    = float(good_count_map) / total_count_map * 100.
        print 'Geolocation success: {0:d}/{1:d} ({2:.1f}%)'.format(good_count_map,total_count_map,good_pct_map)

def make_list(item):
    """ Force something to be iterable. """
    item = np.array(item)
    if item.shape == ():
        item.shape = (1,)

    return item.tolist()

class RbnDataSet(object):
    def __init__(self, df, comment=None, parent=0, **metadata):
        self.parent = parent

        self.df     = df
        self.metadata = {}
        self.metadata.update(metadata)

        self.history = {datetime.datetime.now():comment}

    def compute_grid_stats(self,hgt=300.):
        """
        Create a dataframe with statistics for each grid square.

        hgt: Assumed altitude of reflection [km]
        """

        # Group the dataframe by grid square.
        gs_grp  = self.df.groupby('refl_grid')

        # Get a list of the gridsquares in use.
        grids   = gs_grp.indices.keys()

        # Pull out the desired statistics.
        dct     = {}
        dct['counts']       = gs_grp.freq.count()
        dct['f_max_MHz']    = gs_grp.freq.max()/1000.
        dct['R_gc_min']     = gs_grp.R_gc.min()
        dct['R_gc_max']     = gs_grp.R_gc.max()
        dct['R_gc_mean']    = gs_grp.R_gc.mean()
        dct['R_gc_std']     = gs_grp.R_gc.std()

        # Compute Zenith Angle Theta and FoF2.
        lambda_by_2         = dct['R_gc_min']/Re
        theta               = np.arctan( np.sin(lambda_by_2)/( (Re+hgt)/Re - np.cos(lambda_by_2) ) )
        foF2                = dct['f_max_MHz']*np.cos(theta)
        dct['theta']        = theta
        dct['foF2']         = foF2

        # Put into a new dataframe organized by grid square.
        grid_data       = pd.DataFrame(dct,index=grids)

        # Attach the new dataframe to the RbnDataObj and return.
        self.grid_data  = grid_data
        return grid_data

    def gridsquare_grid(self,precision=None,mesh=True):
        """
        Return a grid square grid.

        precision:
            None:           Use the gridded precsion of this dataset.
            Even integer:   Use specified precision.
        """
        if precision is None:
            precision   = self.metadata.get('gridsquare_precision')

        grid    = gridsquare.gridsquare_grid(precision=precision)
        if mesh:
            ret_val = grid
        else:
            xx = grid[:,0]
            yy = grid[0,:]

            ret_val = (xx,yy)
        return ret_val 

    def grid_latlons(self,precision=None,position='center',mesh=True):
        """
        Return a grid of gridsquare-based lat/lons.

        precision:
            None:           Use the gridded precsion of this dataset.
            Even integer:   Use specified precision.

        Position Options:
            'center'
            'lower left'
            'upper left'
            'upper right'
            'lower right'
        """
        gs_grid     = self.gridsquare_grid(precision=precision,mesh=mesh)
        lat_lons    = gridsquare.gridsquare2latlon(gs_grid,position=position)
       
        if mesh is False:
            lats        = lat_lons[0][1,:]
            lons        = lat_lons[1][0,:]
            lat_lons    = (lats,lons)

        return lat_lons

    def dropna(self,new_data_set='dropna',comment='Remove Non-Geolocated Spots'):
        """
        Removes spots that do not have geolocated Transmitters or Recievers.
        """
        new_ds      = self.copy(new_data_set,comment)
        new_ds.df   = new_ds.df.dropna(subset=['dx_lat','dx_lon','de_lat','de_lon'])
        new_ds.set_active()
        return new_ds

    def calc_reflection_sp_mid(self,new_data_set='sp_mid',comment='Great Circle Midpoints'):
        """
        Determine the path reflection point using a simple great circle
        midpoint method.

        This method creates a new dataset.
        """
        new_ds                  = self.copy(new_data_set,comment)
        df                      = new_ds.df
        md                      = new_ds.metadata
        lat1, lon1              = df['de_lat'],df['de_lon']
        lat2, lon2              = df['dx_lat'],df['dx_lon']
        refl_lat, refl_lon      = geopack.midpoint(lat1,lon1,lat2,lon2)
        df.loc[:,'refl_lat']    = refl_lat
        df.loc[:,'refl_lon']    = refl_lon

        md['reflection_type']   = 'sp_mid'
        new_ds.set_active()

        return new_ds

    def calc_reflection_miller2015(self,hgt=300.):
        lbd_gc_max              = 2*np.arccos( Re/(Re+hgt) )
        R_F_gc_max              = Re*lbd_gc_max
        N_F                     = np.floor(R_gc/R_F_gc_max)
        R_gc_mean               = R_gc/N_F
        import ipdb; ipdb.set_trace()

    def grid_data(self,gridsquare_precision=4,
            lat_key='refl_lat',lon_key='refl_lon',grid_key='refl_grid'):
        """
        Determine gridsquares for the data.

        The method appends gridsquares to current dataframe and does
        NOT create a new dataset.
        """
        df                          = self.df
        md                          = self.metadata
        lats                        = df[lat_key]
        lons                        = df[lon_key]
        df.loc[:,grid_key]          = gridsquare.latlon2gridsquare(lats,lons,
                                        precision=gridsquare_precision)
        md['gridsquare_precision']  = gridsquare_precision

        return self
        
    def filter_calls(self,calls,call_type='de',new_data_set='filter_calls',comment=None):
        """
        Filter data frame for specific calls.

        Calls is not case sensitive and may be a single call
        or a list.

        call_type is 'de' or 'dx'
        """

        if calls is None:
            return self

        if call_type == 'de': key = 'callsign'
        if call_type == 'dx': key = 'dx'

        df          = self.df
        df_calls    = df[key].apply(str.upper)

        calls       = make_list(calls)
        calls       = [x.upper() for x in calls]
        tf          = np.zeros((len(df),),dtype=np.bool)
        for call in calls:
            tf = np.logical_or(tf,df[key] == call)

        df = df[tf]

        if comment is None:
            comment = '{}: {!s}'.format(call_type.upper(),calls)

        new_ds      = self.copy(new_data_set,comment)
        new_ds.df   = df
        new_ds.set_active()
        return new_ds

    def latlon_filt(self,lat_col='refl_lat',lon_col='refl_lon',
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

    def create_geo_grid(self):
        self.geo_grid = RbnGeoGrid(self.df)
        return self.geo_grid

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

    def plot_spot_counts(self,sTime=None,eTime=None,
            integration_time=datetime.timedelta(minutes=15),
            plot_all        = True,     all_lw  = 2,
            plot_by_band    = False,    band_lw = 3,
            band_data=None,
            plot_legend=True,legend_loc='upper left',legend_lw=None,
            plot_title=True,format_xaxis=True,
            xticks=None,
            ax=None):
        """
        Plots counts of RBN data.
        """
        if sTime is None:
            sTime = self.df['date'].min()
        if eTime is None:
            eTime = self.df['date'].max()
            
        if ax is None:
            ax  = plt.gca()

        if plot_by_band:
            if band_data is None:
                band_data = BandData()

            band_list = band_data.band_dict.keys()
            band_list.sort()
            for band in band_list:
                this_group = self.get_band_group(band)
                if this_group is None: continue

                color       = band_data.band_dict[band]['color']
                label       = band_data.band_dict[band]['freq_name']

                counts      = rolling_counts_time(this_group,sTime=sTime,window_length=integration_time)
                ax.plot(counts.index,counts,color=color,label=label,lw=band_lw)

        if plot_all:
            counts  = rolling_counts_time(self.df,sTime=sTime,window_length=integration_time)
            ax.plot(counts.index,counts,color='k',label='All Spots',lw=all_lw)

        ax.set_ylabel('RBN Counts')

        if plot_legend:
            leg = ax.legend(loc=legend_loc,ncol=7)

            if legend_lw is not None:
                for legobj in leg.legendHandles:
                    legobj.set_linewidth(legend_lw)

        if plot_title:
            title   = []
            title.append('Reverse Beacon Network')
            date_fmt    = '%Y %b %d %H%M UT'
            date_str    = '{} - {}'.format(sTime.strftime(date_fmt), eTime.strftime(date_fmt))
            title.append(date_str)
            ax.set_title('\n'.join(title))

        if xticks is not None:
            ax.set_xticks(xticks)

        if format_xaxis:
            ax.set_xlabel('UT')
            ax.set_xlim(sTime,eTime)
            xticks  = ax.get_xticks()
            xtls    = []
            for xtick in xticks:
                xtd = matplotlib.dates.num2date(xtick)
                if xtd.hour == 0 and xtd.minute == 0:
                    xtl = xtd.strftime('%H%M\n%d %b %Y')
                else:
                    xtl = xtd.strftime('%H%M')
                xtls.append(xtl)
            ax.set_xticklabels(xtls)

            for tl in ax.get_xticklabels():
                tl.set_ha('left')

def band_legend(fig=None,loc='lower center',markerscale=0.5,prop={'size':10},
        title=None,bbox_to_anchor=None,ncdxf=False,ncol=None,band_data=None):

    if fig is None: fig = plt.gcf() 

    if band_data is None:
        band_data = BandData()

    handles = []
    labels  = []

    # Force freqs to go low to high regardless of plotting order.
    band_list   = band_data.band_dict.keys()
    band_list.sort()
    for band in band_list:
        color = band_data.band_dict[band]['color']
        label = band_data.band_dict[band]['freq_name']
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

def latlon_filt(df,lat_col='refl_lat',lon_col='refl_lon',
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

def rolling_counts_time(df,sTime=None,window_length=datetime.timedelta(minutes=15)):
    """
    Rolling counts of a RBN dataframe using a time-based data window.
    """
    eTime = df['date'].max().to_datetime()

    if sTime is None:
        sTime = df['date'].min().to_datetime()
        
    this_time   = sTime
    next_time   = this_time + window_length
    date_list, val_list = [], []
    while next_time <= eTime:
        tf  = np.logical_and(df['date'] >= this_time, df['date'] < next_time)
        val = np.count_nonzero(tf)
        
        date_list.append(this_time)
        val_list.append(val)

        this_time = next_time
        next_time = this_time + window_length

    return pd.Series(val_list,index=date_list)

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
    def __init__(self,rbn_obj,data_set='active',data_set_all='DS001_dropna',ax=None,
            sTime=None,eTime=None,
            llcrnrlon=None,llcrnrlat=None,urcrnrlon=None,urcrnrlat=None,
            nightshade=False,solar_zenith=True,solar_zenith_dict={},
            band_data=None,default_plot=True):

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

        if sTime is None:
            sTime = ds.df['date'].min()
        if eTime is None:
            eTime = ds.df['date'].max()

        self.metadata['sTime'] = sTime
        self.metadata['eTime'] = eTime

        if band_data is None:
            band_data = BandData()

        self.band_data = band_data

        self.__setup_map__(ax=ax,**self.latlon_bnds)
        if nightshade:
            self.plot_nightshade()

        if solar_zenith:
            self.plot_solar_zenith_angle(**solar_zenith_dict)

        if default_plot:
            self.default_plot()

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
            self.plot_band_legend(band_data=self.band_data)

    def __setup_map__(self,ax=None,llcrnrlon=-180.,llcrnrlat=-90,urcrnrlon=180.,urcrnrlat=90.):
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
        # This is now done in the gridsquare overlay section...
#        m.drawparallels(np.arange( -90., 91.,45.),color='k',labels=[False,True,True,False])
#        m.drawmeridians(np.arange(-180.,181.,45.),color='k',labels=[True,False,False,True])
        m.drawcoastlines(color='0.65')
        m.drawmapboundary(fill_color='w')

        # Expose select object
        self.fig        = fig
        self.ax         = ax
        self.m          = m

    def center_time(self):
        # Overlay nighttime terminator.
        sTime       = self.metadata['sTime']
        eTime       = self.metadata['eTime']
        half_time   = datetime.timedelta(seconds= ((eTime - sTime).total_seconds()/2.) )
        return (sTime + half_time)
        
    def plot_nightshade(self,color='0.60'):
        self.m.nightshade(self.center_time(),color=color)
        
    def plot_solar_zenith_angle(self,
            cmap=None,vmin=0,vmax=180,plot_colorbar=False):

        if cmap is None:
            fc = {}
            fc[vmin] = cc255('white')
            fc[82]   = cc255('white')
            fc[90]   = cc255('0.80')
            fc[95]   = cc255('0.70')
            fc[vmax] = cc255('0.30')
            cmap = cdict_to_cmap(fc,name='term_cmap',vmin=vmin,vmax=vmax)

        llcrnrlat   = self.latlon_bnds['llcrnrlat'] 
        llcrnrlon   = self.latlon_bnds['llcrnrlon'] 
        urcrnrlat   = self.latlon_bnds['urcrnrlat'] 
        urcrnrlon   = self.latlon_bnds['urcrnrlon'] 
        plot_mTime  = self.center_time()

        nlons       = int((urcrnrlon-llcrnrlon)*4)
        nlats       = int((urcrnrlat-llcrnrlat)*4)
        lats, lons, zen, term = davitpy.utils.calcTerminator( plot_mTime,
                [llcrnrlat,urcrnrlat], [llcrnrlon,urcrnrlon],nlats=nlats,nlons=nlons )

        x,y         = self.m(lons,lats)
        xx,yy       = np.meshgrid(x,y)
        z           = zen[:-1,:-1]
        Zm          = np.ma.masked_where(np.isnan(z),z)

        pcoll       = self.ax.pcolor(xx,yy,Zm,cmap=cmap,vmin=vmin,vmax=vmax)

        if plot_colorbar:
            term_cbar   = plt.colorbar(pcoll,label='Solar Zenith Angle',shrink=0.8)

    def plot_de(self,s=25,zorder=150):
        m       = self.m
        df      = self.data_set.df
        rx      = m.scatter(df['de_lon'],df['de_lat'],
                s=s,zorder=zorder,**de_prop)

    def plot_midpoints(self,s=20):
        band_data   = self.band_data
        band_list   = band_data.band_dict.keys()
        band_list.sort(reverse=True)
        for band in band_list:
            this_group = self.data_set.get_band_group(band)
            if this_group is None: continue

            color = band_data.band_dict[band]['color']
            label = band_data.band_dict[band]['name']

            mid   = self.m.scatter(this_group['refl_lon'],this_group['refl_lat'],
                    alpha=0.50,edgecolors='none',facecolors=color,color=color,s=s,zorder=100)

    def plot_paths(self,band_data=None):
        m   = self.m
        if band_data is None:
            band_data = BandData()

        band_list   = band_data.band_dict.keys()
        band_list.sort(reverse=True)
        for band in band_list:
            this_group = self.data_set.get_band_group(band)
            if this_group is None: continue

            color = band_data.band_dict[band]['color']
            label = band_data.band_dict[band]['name']

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


    def overlay_gridsquares(self,
            major_precision = 2,    major_style = {'color':'k',   'dashes':[1,1]}, 
            minor_precision = None, minor_style = {'color':'0.8', 'dashes':[1,1]},
            label_precision = 2,    label_fontdict=None, label_zorder = 100):
        """
        Overlays a grid square grid.

        Precsion options:
            None:       Gridded resolution of data
            0:          No plotting/labling
            Even int:   Plot or label to specified precision
        """
    
        # Get the dataset and map object.
        ds          = self.data_set
        m           = self.m
        ax          = self.ax

        # Determine the major and minor precision.
        if major_precision is None:
            maj_prec    = ds.metadata.get('gridsquare_precision',0)
        else:
            maj_prec    = major_precision

        if minor_precision is None:
            min_prec    = ds.metadata.get('gridsquare_precision',0)
        else:
            min_prec    = minor_precision

        if label_precision is None:
            label_prec  = ds.metadata.get('gridsquare_precision',0)
        else:
            label_prec  = label_precision

	# Draw Major Grid Squares
        if maj_prec > 0:
            lats,lons   = ds.grid_latlons(maj_prec,position='lower left',mesh=False)

            m.drawparallels(lats,labels=[False,True,True,False],**major_style)
            m.drawmeridians(lons,labels=[True,False,False,True],**major_style)

	# Draw minor Grid Squares
        if min_prec > 0:
            lats,lons   = ds.grid_latlons(min_prec,position='lower left',mesh=False)

            m.drawparallels(lats,labels=[False,False,False,False],**minor_style)
            m.drawmeridians(lons,labels=[False,False,False,False],**minor_style)

	# Label Grid Squares
	lats,lons   = ds.grid_latlons(label_prec,position='center')
        grid_grid   = ds.gridsquare_grid(label_prec)
	xx,yy = m(lons,lats)
	for xxx,yyy,grd in zip(xx.ravel(),yy.ravel(),grid_grid.ravel()):
	    ax.text(xxx,yyy,grd,ha='center',va='center',clip_on=True,
                    fontdict=label_fontdict, zorder=label_zorder)

    def overlay_gridsquare_data(self, param='f_max_MHz',
            cmap=None,vmin=None,vmax=None,label=None,
            band_data=None):
        """
        Overlay gridsquare data on a map.
        """

        grid_data   = self.data_set.grid_data

        param_info = {}
        key                 = 'f_max_MHz'
        tmp                 = {}
        param_info[key]     = tmp
        tmp['cbar_ticks']   = [1.8,3.5,7.,10.,14.,21.,24.,28.]
        tmp['label']        = 'F_max [MHz]'

        key                 = 'counts'
        tmp                 = {}
        param_info[key]     = tmp
        tmp['label']        = 'Counts'
        tmp['vmin']         = 0
        tmp['vmax']         = int(grid_data.counts.mean() + 3.*grid_data.counts.std())
        tmp['cmap']         = matplotlib.cm.jet
        
        key                 = 'theta'
        tmp                 = {}
        param_info[key]     = tmp
        tmp['label']        = 'Zenith Angle Theta'
        tmp['vmin']         = 0
        tmp['vmax']         = 90.
        tmp['cbar_ticks']   = np.arange(0,91,10)
        tmp['cmap']         = matplotlib.cm.jet

        key                 = 'foF2'
        tmp                 = {}
        param_info[key]     = tmp
        tmp['vmin']         = 0
        tmp['vmax']         = 30
        tmp['cbar_ticks']   = np.arange(0,31,5)
        tmp['label']        = 'RBN foF2 [MHz]'

        for stat in ['min','max','mean']:
            key                 = 'R_gc_{}'.format(stat)
            tmp                 = {}
            param_info[key]     = tmp
            tmp['label']        = '{} R_gc [km]'.format(stat)
            tmp['vmin']         = 0
#            tmp['vmax']         = int(grid_data[key].mean() + 3.*grid_data[key].std())
            tmp['vmax']         = 10000.
            tmp['cbar_ticks']   = np.arange(0,10001,1000)
            tmp['cmap']         = matplotlib.cm.jet

        param_dict  = param_info.get(param,{})
        if band_data is None:
            band_data   = param_dict.get('band_data',BandData())
        if cmap is None:
            cmap        = param_dict.get('cmap',band_data.cmap)
        if vmin is None:
            vmin        = param_dict.get('vmin',band_data.norm.vmin)
        param_info = {}
        if vmax is None:
            vmax        = param_dict.get('vmax',band_data.norm.vmax)
        if label is None:
            label       = param_dict.get('label',param)

        cbar_ticks  = param_dict.get('cbar_ticks')

        fig         = self.fig
        ax          = self.ax
        m           = self.m

        ll                  = gridsquare.gridsquare2latlon
        lats_ll, lons_ll    = ll(grid_data.index,'lower left')
        lats_lr, lons_lr    = ll(grid_data.index,'lower right')
        lats_ur, lons_ur    = ll(grid_data.index,'upper right')
        lats_ul, lons_ul    = ll(grid_data.index,'upper left')

        coords  = zip(lats_ll,lons_ll,lats_lr,lons_lr,
                      lats_ur,lons_ur,lats_ul,lons_ul)

        verts   = []
        for lat_ll,lon_ll,lat_lr,lon_lr,lat_ur,lon_ur,lat_ul,lon_ul in coords:
            x1,y1 = m(lon_ll,lat_ll)
            x2,y2 = m(lon_lr,lat_lr)
            x3,y3 = m(lon_ur,lat_ur)
            x4,y4 = m(lon_ul,lat_ul)
            verts.append(((x1,y1),(x2,y2),(x3,y3),(x4,y4),(x1,y1)))

        vals    = grid_data[param]

        if param == 'theta':
            vals = (180./np.pi)*vals # Convert to degrees

        bounds  = np.linspace(vmin,vmax,256)
        norm    = matplotlib.colors.BoundaryNorm(bounds,cmap.N)

        pcoll   = PolyCollection(np.array(verts),edgecolors='face',closed=False,cmap=cmap,norm=norm,zorder=99)
        pcoll.set_array(np.array(vals))
        ax.add_collection(pcoll,autolim=False)

        cbar    = fig.colorbar(pcoll,label=label)
        if cbar_ticks is not None:
            cbar.set_ticks(cbar_ticks)
