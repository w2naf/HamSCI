#!/usr/bin/env python
#Including the above line as the first line of the script allows this script to be run
#directly from the command line without first calling python/ipython.
de_prop         = {'marker':'^','edgecolor':'k','facecolor':'white'}
dx_prop         = {'marker':'o','edgecolor':'k','facecolor':'white'}
dxf_prop        = {'marker':'*','color':'blue'}
dxf_leg_size    = 150
dxf_plot_size   = 50
Re              = 6371  # Radius of the Earth

import geopack
import os               # Provides utilities that help us do os-level operations like create directories
import datetime         # Really awesome module for working with dates and times.
import gzip             # Allows us to read from gzipped files directly!
import urllib2          # Used to automatically download data files from the web.
import pickle
import sys
import copy 


import numpy as np      #Numerical python - provides array types and operations
import pandas as pd     #This is a nice utility for working with time-series type data.

# Some view options for debugging.
pd.set_option('display.height', 1000)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

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
#        bands.append((28.0,  '10 m'))
#        bands.append((21.0,  '15 m'))
#        bands.append((18.0,  '17 m'))
#        bands.append((14.0,  '20 m'))
#        bands.append((10.0,  '30 m'))
#        bands.append(( 7.0,  '40 m'))
#        bands.append(( 3.5,  '80 m'))
#        bands.append(( 1.8, '160 m'))

##        bands.append((144.0,  '2 m'))
#        bands.append((50.0,  '6 m'))
        bands.append((28.0,  '10 m'))
        bands.append((24.0,  '12 m'))
        bands.append((21.0,  '15 m'))
        bands.append((18.0,  '17 m'))
        bands.append((14.0,  '20 m'))
        bands.append((10.0,  '30 m'))
        bands.append(( 7.0,  '40 m'))
#        bands.append(( 5.0,  '60 m'))
        bands.append(( 3.5,  '80 m'))
        bands.append(( 1.8, '160 m'))
#        bands.append(( 0.5, '700 m'))
#        bands.append(( 0.1, '1600 m'))
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
#	fc[ 0.1] = cc255('magenta')
	fc[ 0.5] = cc255('magenta')
	fc[ 1.8] = cc255('violet')
	fc[ 3.0] = cc255('blue')
#	fc[ 5.5] = cc255('blue')
	fc[ 8.0] = cc255('aqua')
#	fc[10.0] = cc255('green')
	fc[13.0] = cc255('green')
	fc[17.0] = cc255('yellow')
	fc[21.0] = cc255('orange')
	fc[28.0] = cc255('red')
	fc[30.0] = cc255('red')

#	fc[ 0.0] = (  0,   0,   0)
##	fc[ 0.1] = cc255('magenta')
#        import ipdb; ipdb.set_trace()
#	fc[ 0.1] = cc255('magenta')
##        fc[0.5] = (130, 130, 238)
##	fc[ 1.8] = cc255('violet')
#	fc[ 0.5] = cc255('violet')
#        fc[ 1.8] = (130, 130, 238)
#	fc[ 3.0] = cc255('blue')
#	fc[ 5.5] = (130, 255,255/2)
#	fc[ 8.0] = cc255('aqua')
##	fc[10.0] = cc255('green')
#	fc[10.0] = (0, 225,128)
#	fc[13.0] = (0,128,225)
#        fc[17.0] = cc255('green')
#	fc[21.0] = cc255('yellow')
#	fc[24.5] = cc255('orange')
#	fc[28.0] = cc255('red')
#	fc[30.0] = cc255('red')
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


#Now we import libraries that are not "built-in" to python.
def __add_months(sourcedate,months=1):
    """
    Add 1 month to a datetime object.
    """
    import calendar
    import datetime

    month = sourcedate.month - 1 + months
    year = sourcedate.year + month / 12
    month = month % 12 + 1
    day = min(sourcedate.day,calendar.monthrange(year,month)[1])
    return datetime.date(year,month,day)

def ham_band_errorbars(freqs):
    """
    Return error bars based on ham radio band discretization.

    Upper error bar is the bottom of the next highest ham radio band.
    Lower error bar is 90% of the original frequency.
    """

    freqs   = np.array(freqs)
    if freqs.shape == (): freqs.shape = (1,)

    bands   = [ 1.80,  3.5,  7.0,  10.0,  14.0,  18.1,  21.0,
               24.89, 28.0, 50.0, 144.0, 220.0, 440.0]
    bands   = [ 1.80,  3.5,  7.0, 14.0,  18.1,  21.0,
               24.89, 28.0, 50.0, 144.0, 220.0, 440.0]
#    bands   = [ 1.80,  3.5, 5.0, 7.0,  10.0,  14.0,  18.1,  21.0,
#               24.89, 28.0, 50.0, 144.0, 220.0, 440.0]
    bands   = np.array(bands)

    low_lst = []
    upp_lst = []

    for freq in freqs:
        diff    = np.abs(bands - freq)
        argmin  = diff.argmin()

        lower   = 0.10 * freq
        low_lst.append(lower)

        upper   = bands[argmin+1] - freq
        upp_lst.append(upper)
    
    return (np.array(low_lst),np.array(upp_lst))

def read_wspr(sTime,eTime=None,data_dir='data/wspr', overwrite=False, refresh=False):
     #refresh is keyword to tell function to download data from wspr website even if already have it on computer (if downloaded data from current month once before and now it is updated)
        #Will NOT overwrite pickle files!!!!

     #overwrite is keyword to tell function to act as if no data files exist all ready

    if data_dir is None: data_dir = os.getenv('DAVIT_TMPDIR')

    if eTime is None: eTime = sTime + datetime.timedelta(days=1)

    #Determine which months of data to download.
    ym_list     = [datetime.date(sTime.year,sTime.month,1)]
    eMonth      = datetime.date(eTime.year,eTime.month,1)
    while ym_list[-1] < eMonth:
        ym_list.append(__add_months(ym_list[-1]))

#    df = None
    for year_month in ym_list:
        data_file   = 'wsprspots-%s.csv.gz' % year_month.strftime('%Y-%m')
        data_path   = os.path.join(data_dir,data_file)  

        time_0      = datetime.datetime.now()
        print 'Starting WSPRNet histogram processing on <%s> at %s.' % (data_file,str(time_0))

        ################################################################################
        # Make sure the data file exists.  If not, download it and open it.
        if not os.path.exists(data_path) or overwrite or refresh:
             try:    # Create the output directory, but fail silently if it already exists
                 os.makedirs(data_dir) 
             except:
                 pass
             # File downloading code from: http://stackoverflow.com/questions/22676/how-do-i-download-a-file-over-http-using-python
             url = 'http://wsprnet.org/archive/'+data_file
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

        print 'Loading Dataframe'

        # Load data into dataframe here. ###############################################
        # Here I define the column names of the data file, and also specify which ones to load into memory.  By only loading in some, I save time and memory.
        names       = ['spot_id', 'timestamp', 'reporter', 'rep_grid', 'snr', 'freq', 'call_sign', 'grid', 'power', 'drift', 'dist', 'azm', 'band', 'version', 'code']

        std_sTime=datetime.datetime(sTime.year,sTime.month,sTime.day, sTime.hour)
        if eTime.minute == 0 and eTime.second == 0:
            hourly_eTime=datetime.datetime(eTime.year,eTime.month,eTime.day, eTime.hour)
        else:
            hourly_eTime=eTime+datetime.timedelta(hours=1)
            hourly_eTime=datetime.datetime(hourly_eTime.year,hourly_eTime.month,hourly_eTime.day, hourly_eTime.hour)

        std_eTime=std_sTime+datetime.timedelta(hours=1)

        hour_flag=0
        extract=True
        print 'Initial interval: '+std_sTime.strftime('%Y%m%d%H%M-')+std_eTime.strftime('%Y%m%d%H%M')
        print 'End: '+hourly_eTime.strftime('%Y%m%d%H%M')
        ref_month=std_eTime.month
        while std_eTime<=hourly_eTime:
            p_filename = 'wspr_'+std_sTime.strftime('%Y%m%d%H%M-')+std_eTime.strftime('%Y%m%d%H%M.p')
            p_filepath = os.path.join(data_dir,p_filename)
            if not os.path.exists(p_filepath) or overwrite:
                # Load data into dataframe here. ###############################################
#                if std_eTime.month != ref_month or hour_flag == 0:
                # Reset flag to extract file to dataframe if looking at a new month 
                if std_sTime.month != ref_month:
                    extract = True
                    ref_month = std_sTime.month
#                # Reset flag to extract file to dataframe if looking at a new month 
#                if std_eTime.month != ref_month:
#                    extract = True
#                    ref_month = std_eTime.month
                if extract: 
                    with gzip.GzipFile(data_path,'rb') as fl:   #This block lets us directly read the compressed gz file into memory.  The 'with' construction means that the file is automatically closed for us when we are done.
    #                        df_tmp      = pd.read_csv(fl,names=names,index_col='spot_id')
                        print 'Loading '+str(data_path)+' into Pandas Dataframe'
                        df_tmp      = pd.read_csv(fl,names=names,index_col='spot_id')
#                    df=df_tmp

#                    if df is None:
#                        df = df_tmp
#                    else:
#                        df = df.append(df_tmp)
                    df_tmp['timestamp'] = pd.to_datetime(df_tmp['timestamp'],unit='s')
                    extract = False

                # Trim dataframe to just the entries in a 1 hour time period.
                df = df_tmp[np.logical_and(df_tmp['timestamp'] >= std_sTime,df_tmp['timestamp'] < std_eTime)]

#        df = df[np.logical_and(df['timestamp'] >= sTime, df['timestamp'] < eTime)]


#                sys.stdout.write('\r'+'\r'+'WSPR Data: '+std_sTime.strftime('%Y%m%d%H%M - ')+std_eTime.strftime('%Y%m%d%H%M')) 
#                sys.stdout.write('# Entries: '+str(len(df['call_sign'])))
#                sys.stdout.flush()
                print 'WSPR Data: '+std_sTime.strftime('%Y%m%d%H%M - ')+std_eTime.strftime('%Y%m%d%H%M') 
                print '# Entries: '+str(len(df['call_sign']))+'\n'
#                    print '# Entries: '+str(len(df['call_sign'].unique())

#                    if total == 0:
#                        print "No call signs geolocated."
#                    else:
#                        pct     = success / float(total) * 100.
#                        print '{0:d} of {1:d} ({2:.1f} %) call signs geolocated via qrz.com.'.format(success,total,pct)

                df.to_pickle(p_filepath)
            else:
                print 'Found Pickle File for '+std_sTime.strftime('%Y%m%d%H%M - ')+std_eTime.strftime('%Y%m%d%H%M') 
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
        df = df_comp[np.logical_and(df_comp['timestamp'] >= sTime,df_comp['timestamp'] < eTime)]

        df['timestamp']=df['timestamp'].astype(datetime.datetime)

#        # Calculate Band
#        df.loc[:,'band']        = np.array((np.floor(df['freq']/1000.)),dtype=np.int)

        return df

def fix_wspr_band(df):
    df = df.replace(to_replace = {'band': {0:0.5}})
    df = df.replace(to_replace = {'band': {-1:0.1}})
    return df
#
#def fix_wspr_band(df):
#    grouped = df.groupby('band')
#    all_bands = df['band'].unique()
#    df=None
#    for band in all_bands:
#        tmp = grouped.get_group(band)
#        import ipdb; ipdb.set_trace()
#        if band == 0 or band == -1:
#            tmp.loc[:,'band'] = np.array(np.round(tmp['freq'], decimals=1))
#        if df is None: df = tmp
#        else: df = pd.concat([df,tmp])
#    return df

class WsprObject(object):
    """
    gridsquare_precision:   Even number, typically 4 or 6
    reflection_type:        Model used to determine reflection point in ionopshere.
                            'sp_mid': spherical midpoint

        Written by Magdalina Moses, Fall 2016 
        (In part based on code written by Nathaniel A. Frissell, Summer 2016)
    """
    def __init__(self,sTime=None,eTime=None,data_dir='data/wspr',
            overwrite=False,refresh=False,qrz_call=None,qrz_passwd=None,comment='Raw Data',df=None,
            gridsquare_precision=4,reflection_type='sp_mid'):

        if df is None:
            df = read_wspr(sTime=sTime,eTime=eTime,data_dir=data_dir,
                    overwrite=overwrite, refresh=refresh)
        df = fix_wspr_band(df)

        #Make metadata block to hold information about the processing.
        metadata = {}
        data_set                            = 'DS000'
        metadata['data_set_name']           = data_set
        metadata['serial']                  = 0
        cmt     = '[{}] {}'.format(data_set,comment)
        
        wspr_ds  = WsprDataSet(df,parent=self,comment=cmt)
        setattr(self,data_set,wspr_ds)
        setattr(wspr_ds,'metadata',metadata)
        wspr_ds.set_active()

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

def make_list(item):
    """ Force something to be iterable. """
    item = np.array(item)
    if item.shape == ():
        item.shape = (1,)

    return item.tolist()

#    def geo_loc_stats(self,verbose=True):
#        # Figure out how many records properly geolocated.
#        good_loc        = rbn_obj.DS001_dropna.df
#        good_count_map  = good_loc['callsign'].count()
#        total_count_map = len(rbn_obj.DS000.df)
#        good_pct_map    = float(good_count_map) / total_count_map * 100.
#        print 'Geolocation success: {0:d}/{1:d} ({2:.1f}%)'.format(good_count_map,total_count_map,good_pct_map)

class WsprDataSet(object):
    def __init__(self, df, comment=None, parent=0, **metadata):
        self.parent = parent

        self.df     = df
        self.metadata = {}
        self.metadata.update(metadata)

        self.history = {datetime.datetime.now():comment}

#    def qth_latlon_data(self,gridsquare_precision=4,
#            dx_key='grid',de_key='rep_grid',grid_key=None):
#        """
#        Determine latitde and longitude data from the reported gridsquares for the data.
#
#        The method appends de and dx lat and lons to current dataframe and does
#        NOT create a new dataset.
#        """
##        df                          = self.df
#        df                          = self.df
#        md                          = self.metadata
#        gridsq                   = df[grid_key]
#
#        lat, lon              =gridsquare2latlon(gridsquare=gridsq,position=position)
#        df.loc[:,loc_key[0]]           = lat 
#        df.loc[:,loc_key[1]]           = lon 
#
#        df.loc[:,grid_key]          = gridsquare.latlon2gridsquare(lats,lons,
#                                        precision=gridsquare_precision)
#        md['position']  = position
#        md                          = self.metadata
#        dx_gridsq                   = df[dx_key]
#        de_gridsq                   = df[de_key]
#        self.latlon_data()
#        gs_0        = np.array(gridsquare)
#        gs_1        = gs_0.flatten()
#        gs_good_tf  = gs_1 != ''
#        gs_2        = gs_1[gs_good_tf]
#        gss = np.char.array(gs_2).lower()
#        precs       = np.array([len(x) for x in gss.ravel()])
#
#        return self

#    def dropna(self,new_data_set='dropna',comment='Remove Non-Geolocated Spots'):
#        """
#        Removes spots that do not have geolocated Transmitters or Recievers.
#        """
#        new_ds      = self.copy(new_data_set,comment)
#        new_ds.df   = new_ds.df.dropna(subset=['dx_lat','dx_lon','de_lat','de_lon'])
#        new_ds.set_active()
#        return new_ds

    def select_interval(self, sTime, eTime=None, dt = 5, replace = False, new_data_set = None, comment = None): 
        """
        Parameters
        ----------
        sTime : datetime
            
        eTime : datetime

        dt : int
            Interval in minutes

        replace : boolean
            Specifies when to replace current data set object
                True: Replace current data set object 
                False: (Default) Create new data set object 
        new_data_set : string
            Name of new WsprObj data set. (Default to date with start and end times)
        comment : 
            Comment for new WsprObj data set. (Default to date with interval in minutes.)

        Returns
        -------
        new_data_set_obj : data_set
            New data set object  

        Written by Magdalina Moses, January 2017
        """

        if sTime is None:
            sTime = ds.df['timestamp'].min()
        if eTime is None:
            eTime = sTime + datetime.timedelta(minutes = dt) 

        if not replace:
            if new_data_set is None: new_data_set=sTime.strftime('%Y%m%d_%H%M_')+eTime.strftime('%H%M')
            else: new_data_set = new_data_set
            if comment is None: comment = sTime.strftime('%Y%m%d')+' WSPR data over '+str(dt)+'minutes'
            else: comment = comment 
            new_ds  = self.copy(new_data_set, comment)
            df = new_ds.df
#        else: new_ds = self
#        else: df = self.df
        else:
            new_ds = self
            df = self.df

        #Replace following with code to check dataset name and decide
        try: 
            df['timestamp']
            time = 'timestamp'
        except:
            time = 'date'
        # Clip to times need
        print time
        df = df[np.logical_and(df[time]>=sTime, df[time] < eTime)]
        new_ds.df = df
        if not replace: new_ds.set_active()
#        if not replace: 
#            new_ds.df = df
#            new_ds.set_active()
#            return new_ds
#
#        else: 
#            self.df = df
#            return self
        return new_ds

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
        dct['f_max_MHz']    = gs_grp.freq.max()
        dct['R_gc_min']     = gs_grp.R_gc.min()
        dct['R_gc_max']     = gs_grp.R_gc.max()
        dct['R_gc_mean']    = gs_grp.R_gc.mean()
        dct['R_gc_std']     = gs_grp.R_gc.std()

        # Error bar info.
        f_max               = dct['f_max_MHz']
        lower,upper         = ham_band_errorbars(f_max)

        # Compute Zenith Angle Theta and FoF2.
        lambda_by_2         = dct['R_gc_min']/Re
        theta               = np.arctan( np.sin(lambda_by_2)/( (Re+hgt)/Re - np.cos(lambda_by_2) ) )
        foF2                = dct['f_max_MHz']*np.cos(theta)
        foF2_err_low        = lower*np.cos(theta)
        foF2_err_up         = upper*np.cos(theta)
        dct['theta']        = theta
        dct['foF2']         = foF2
        dct['foF2_err_low'] = foF2_err_low
        dct['foF2_err_up']  = foF2_err_up

        # Put into a new dataframe organized by grid square.
        grid_data       = pd.DataFrame(dct,index=grids)

#        fig     = plt.figure()
#        ax  = fig.add_subplot(111)
#        ax.plot(foF2.tolist(),label='foF2')
#        ax.plot(foF2_err_low.tolist(),label='foF2_err_low')
#        ax.plot(foF2_err_up.tolist(),label='foF2_err_up')
#        ax.set_ylabel('foF2 [MHz]')
#        ax.set_xlabel('Grid Square')
#        ax.legend(loc='upper right')
#        ax.set_ylim(0,50)
#        fig.savefig('error.png',bbox_inches='tight')

        # Attach the new dataframe to the WsprDataObj and return.
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

    def rbn_compatible(self,new_data_set='rbncomp',comment='RBN code compatible WSPR data'):
        """
        Rename certain columns of wspr object dataframe and convert units to be compatible with a rbn object

        Written by : Magdalina Moses January 2017
        """
        new_ds      = self.copy(new_data_set,comment)
        new_ds.df   = new_ds.df.rename(columns = {'timestamp' : 'date', 'reporter' : 'callsign', 'call_sign' : 'dx'})
        new_ds.df['freq'] = new_ds.df['freq']*1000.
        new_ds.set_active()
        return new_ds

    def calc_reflection_points(self,reflection_type='sp_mid',**kwargs):
        """
        Determine ionospheric reflection points of RBN data.

        reflection_type: Method used to determine reflection points. Choice of:
            'sp_mid':
                Determine the path reflection point using a simple great circle
                midpoint method.

            'miller2015':
                Determine the path reflection points using the multipoint scheme described
                by Miller et al. [2015].

        **kwargs:
            'new_data_set':
                Name of new RbnObj data set. Defaults to reflection_type.
            'comment':
                Comment for new data set. Default varies based on reflection type.
            'hgt':
                Assumed height [km] used in the 'miller2015' model. Defaults to 300 km.
        """
        if reflection_type == 'sp_mid':
            new_data_set            = kwargs.get('new_data_set',reflection_type)
            comment                 = kwargs.get('comment','Great Circle Midpoints')
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

        if reflection_type == 'miller2015':
            try:
                df['R_gc'] 
            except:
                self.calc_greatCircle_dist()
            new_data_set            = kwargs.get('new_data_set',reflection_type)
            comment                 = kwargs.get('comment','Miller et al 2015 Reflection Points')
            hgt                     = kwargs.get('hgt',300.)

            new_ds                  = self.copy(new_data_set,comment)
            df                      = new_ds.df
            md                      = new_ds.metadata

            R_gc                    = df['R_gc']
        
            azm                     = geopack.greatCircleAzm(df.de_lat,df.de_lon,df.dx_lat,df.dx_lon)

            lbd_gc_max              = 2*np.arccos( Re/(Re+hgt) )
            R_F_gc_max              = Re*lbd_gc_max
            N_hops                  = np.array(np.ceil(R_gc/R_F_gc_max),dtype=np.int)
            R_gc_mean               = R_gc/N_hops

            df['azm']               = azm
            df['N_hops']            = N_hops
            df['R_gc_mean']         = R_gc_mean

            new_df_list = []
            for inx,row in df.iterrows():
#                print ''
#                print '<<<<<---------->>>>>'
#                print 'DE: {!s} DX: {!s}'.format(row.callsign,row.dx)
#                print '        Old DE: {:f}, {:f}; DX: {:f},{:f}'.format(row.de_lat,row.de_lon,row.dx_lat,row.dx_lon)
                for hop in range(row.N_hops):
                    new_row = row.copy()

                    new_de  = geopack.greatCircleMove(row.de_lat,row.de_lon,(hop+0)*row.R_gc_mean,row.azm)
                    new_dx  = geopack.greatCircleMove(row.de_lat,row.de_lon,(hop+1)*row.R_gc_mean,row.azm)
                    
                    new_row['de_lat']   = float(new_de[0])
                    new_row['de_lon']   = float(new_de[1])
                    new_row['dx_lat']   = float(new_dx[0])
                    new_row['dx_lon']   = float(new_dx[1])
                    new_row['hop_nr']   = hop

                    new_df_list.append(new_row)

#                    print '({:02d}/{:02d}) New DE: {:f}, {:f}; DX: {:f},{:f}'.format(
#                            row.N_hops,hop,new_row.de_lat,new_row.de_lon,new_row.dx_lat,new_row.dx_lon)

            new_df                      = pd.DataFrame(new_df_list)
            new_ds.df                   = new_df

            lat1, lon1                  = new_df['de_lat'],new_df['de_lon']
            lat2, lon2                  = new_df['dx_lat'],new_df['dx_lon']
            refl_lat, refl_lon          = geopack.midpoint(lat1,lon1,lat2,lon2)
            new_df.loc[:,'refl_lat']    = refl_lat
            new_df.loc[:,'refl_lon']    = refl_lon

            md['reflection_type']       = 'miller2015'
            new_ds.set_active()
            return new_ds
        else:
            raise Exception('Error: Invalid reflection_type Input!')

    def calc_greatCircle_dist(self):
        """
        Determine great circle distance from lat and lons

        The method appends de and dx lat and lons to current dataframe and does
        NOT create a new dataset.
        """
#        try self.metadata['position']:
        try: 
            self.df['dx_lat'].head()
        except:
            self.dxde_gs_latlon()
#
#        except:
#            self.dxde_gs_latlon()

        df                          = self.df
        # Calculate Total Great Circle Path Distance
        lat1, lon1          = df['de_lat'],df['de_lon']
        lat2, lon2          = df['dx_lat'],df['dx_lon']
        R_gc                = Re*geopack.greatCircleDist(lat1,lon1,lat2,lon2)
        df.loc[:,'R_gc']    = R_gc

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

        if call_type == 'de': key = 'reporter'
        if call_type == 'dx': key = 'call_sign'

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

    def dxde_gs_latlon(self,pos='center'):
        """
        Determine latitde and longitude data for dx and de stations from the reported gridsquares for the data.

        The method appends de and dx lat and lons to current dataframe and does
        NOT create a new dataset.
        Written by Magdalina Moses, Fall 2016
        """
        print 'Finding dx lat/lon....'
        self.latlon_data(position=pos,grid_key='grid',loc_key=['dx_lat','dx_lon'])
        print 'Finding de lat/lon....'
        self.latlon_data(position=pos,
            grid_key='rep_grid',loc_key=['de_lat','de_lon'])
        print 'Found all lat/lon!'
        self.calc_greatCircle_dist()

        return self
        
    def latlon_data(self,position='center',
            grid_key='grid',loc_key=['dx_lat','dx_lon']):
        """
        Determine latitde and longitude data from the reported gridsquares for the data.

        The method appends de and dx lat and lons to current dataframe and does
        NOT create a new dataset.
        Written by: Magalina Moses Winter 2016/2017
        """
        df                          = self.df
        md                          = self.metadata
        gridsq                   = df[grid_key]
        lat, lon              = gridsquare.gridsquare2latlon(gridsquare=gridsq,position=position)
        df.loc[:,loc_key[0]]           = lat 
        df.loc[:,loc_key[1]]           = lon 

#        df.loc[:,grid_key]          = gridsquare.latlon2gridsquare(lats,lons,
#                                        precision=gridsquare_precision)
        md['position']  = position

        return self

    def filter_pathlength(self,min_length=None,max_length=None,
            new_data_set='pathlength_filter',comment=None):
        """
        """

        if min_length is None and max_length is None:
            return self

        if comment is None:
            comment = 'Pathlength Filter: {!s}'.format((min_length,max_length))

        new_ds                  = self.copy(new_data_set,comment)
        df                      = new_ds.df

        if min_length is not None:
            tf  = df.R_gc >= min_length
            df  = df[tf]

        if max_length is not None:
            tf  = df.R_gc < max_length
            df  = df[tf]
        
        new_ds.df = df
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
            srt                 = self.df.sort_values(by=['band','timestamp'])
            self.band_groups    = srt.groupby('band')

        try:
            this_group  = self.band_groups.get_group(band)
        except:
            this_group  = None

        return this_group

    def get_band_group(self,band):
        if not hasattr(self,'band_groups'):
            srt                 = self.df.sort_values(by=['band','timestamp'])
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
        de_list = self.df['reporter'].unique().tolist()
        dx_list = self.df['call_sign'].unique().tolist()

        de_list.sort()
        dx_list.sort()

        return (de_list,dx_list)

#    def create_geo_grid(self):
#        self.geo_grid = RbnGeoGrid(self.df)
#        return self.geo_grid

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
        """Copy a WsprDataSet object.  This deep copies data and metadata, updates the serial
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
        Plots counts of WSPR data.
        """
        if sTime is None:
            sTime = self.df['timestamp'].min()
        if eTime is None:
            eTime = self.df['timestamp'].max()
            
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

        ax.set_ylabel('WSPR Counts')

        if plot_legend:
            leg = ax.legend(loc=legend_loc,ncol=7)

            if legend_lw is not None:
                for legobj in leg.legendHandles:
                    legobj.set_linewidth(legend_lw)

        if plot_title:
            title   = []
            title.append(' WSPR Net')
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
        title=None,bbox_to_anchor=None,wspr_rx=True,ncdxf=False,ncol=None,band_data=None):

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
    if wspr_rx:
        scat = ax_tmp.scatter(0,0,s=50,**de_prop)
        labels.append('WSPR Receiver')
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
    Return an WSPR Dataframe with entries only within a specified lat/lon box.
    """
    df          = df.copy()
    lat_tf      = np.logical_and(df[lat_col] >= llcrnrlat,df[lat_col] < urcrnrlat)
    lon_tf      = np.logical_and(df[lon_col] >= llcrnrlon,df[lon_col] < urcrnrlon)
    tf          = np.logical_and(lat_tf,lon_tf)
    df          = df[tf]
    return df

def select_interval(df, sTime=None, eTime=None, dt = 5, replace = False, new_data_set = None, comment = None): 
    """
    Parameters
    ----------
    sTime : datetime
        
    eTime : datetime

    dt : int
        Interval in minutes

    replace : boolean
        Specifies when to replace current data set object
            True: Replace current data set object 
            False: (Default) Create new data set object 
    new_data_set : string
        Name of new WsprObj data set. (Default to date with start and end times)
    comment : 
        Comment for new WsprObj data set. (Default to date with interval in minutes.)

    Returns
    -------
    new_data_set_obj : data_set
        New data set object  

    Written by Magdalina Moses, January 2017
    """

    if sTime is None:
        sTime = df['timestamp'].min()
    if eTime is None:
        eTime = sTime + datetime.timedelta(minutes = dt) 

    #Replace following with code to check dataset name and decide
    try: 
        df['timestamp']
        time = 'timestamp'
    except:
        time = 'date'
    # Clip to times need
    print time
    df = df[np.logical_and(df[time]>=sTime, df[time] < eTime)]
    return df

def rolling_counts_time(df,sTime=None,window_length=datetime.timedelta(minutes=15)):
    """
    Rolling counts of a RBN dataframe using a time-based data window.
    """
    eTime = df['timestamp'].max().to_datetime()

    if sTime is None:
        sTime = df['timestamp'].min().to_datetime()
        
    this_time   = sTime
    next_time   = this_time + window_length
    date_list, val_list = [], []
    while next_time <= eTime:
        tf  = np.logical_and(df['timestamp'] >= this_time, df['timestamp'] < next_time)
        val = np.count_nonzero(tf)
        
        date_list.append(this_time)
        val_list.append(val)

        this_time = next_time
        next_time = this_time + window_length

    return pd.Series(val_list,index=date_list)

class WsprMap(object):
    """Plot WSPRNet data.

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

    Written by Magdalina Moses Jan 2017 and Nathaniel Frissell 2014 Sept 06
    """
    def __init__(self,wspr_obj,data_set='active',data_set_all='DS000',ax=None,
            sTime=None,eTime=None,
            llcrnrlon=None,llcrnrlat=None,urcrnrlon=None,urcrnrlat=None,
            coastline_color='0.65',coastline_zorder=10,
            nightshade=False,solar_zenith=True,solar_zenith_dict={},
            band_data=None,default_plot=True, other_plot=None):

#        rcp = matplotlib.rcParams
#        rcp['axes.titlesize']     = 'large'
#        rcp['axes.titleweight']   = 'bold'
        self.wspr_obj        = wspr_obj
        self.data_set       = getattr(wspr_obj,data_set)
        self.data_set_all   = getattr(wspr_obj,data_set_all)

        ds                  = self.data_set
        ds_md               = self.data_set.metadata

        if sTime is None:
            sTime = ds.df['timestamp'].min()
        if eTime is None:
            eTime = ds.df['timestamp'].max()
        #Added this because in RBN code there did not appear to be any place where times were selected maybe it is unecessary
        else:
#            wspr_obj = wspr_obj.active.select_interval(sTime, eTime, replace = True)
            wspr_obj.active.select_interval(sTime, eTime)
            print sTime.strftime('%Y%h%d %H%M')
            print eTime.strftime('%Y%h%d %H%M')

            self.wspr_obj        = wspr_obj
            self.data_set       = getattr(wspr_obj,data_set)
    #        self.data_set_all   = getattr(wspr_obj,data_set_all)

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

        self.metadata['sTime'] = sTime
        self.metadata['eTime'] = eTime

        if band_data is None:
            band_data = BandData()

        self.band_data = band_data

        self.__setup_map__(ax=ax,
                coastline_color=coastline_color,coastline_zorder=coastline_zorder,
                **self.latlon_bnds)
        if nightshade:
            self.plot_nightshade()

        if solar_zenith:
            self.plot_solar_zenith_angle(**solar_zenith_dict)

        if default_plot:
            self.default_plot()
#            self.default_plot(plot_de=True, plot_midpoints = False, plot_paths = True, plot_ncdxf = True, plot_stats=False)
        if other_plot == 'plot_paths':
            self.path_plot()
        if other_plot == 'plot_mid':
            self.mid_plot()


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

    def path_plot(self,
            plot_de         = True,
            plot_midpoints  = False,
            plot_paths      = True,
            plot_ncdxf      = True,
            plot_stats      = False,
            plot_legend     = True):

        if plot_de:
            self.plot_de()
        if plot_midpoints:
            self.plot_midpoints()
        if plot_paths:
            self.plot_paths()
        if plot_ncdxf:
            self.plot_ncdxf()
#        if plot_stats:
#            self.plot_link_stats()
        if plot_legend:
            self.plot_band_legend(band_data=self.band_data)

    def mid_plot(self,
            plot_de         = True,
            plot_midpoints  = True,
            plot_paths      = False,
            plot_ncdxf      = True,
            plot_stats      = False,
            plot_legend     = True):

        if plot_de:
            self.plot_de()
        if plot_midpoints:
            self.plot_midpoints()
        if plot_paths:
            self.plot_paths()
        if plot_ncdxf:
            self.plot_ncdxf()
#        if plot_stats:
#            self.plot_link_stats()
        if plot_legend:
            self.plot_band_legend(band_data=self.band_data)

    def __setup_map__(self,ax=None,llcrnrlon=-180.,llcrnrlat=-90,urcrnrlon=180.,urcrnrlat=90.,
            coastline_color='0.65',coastline_zorder=10):
        sTime       = self.metadata['sTime']
        eTime       = self.metadata['eTime']

        if ax is None:
            fig     = plt.figure(figsize=(10,6))
            ax      = fig.add_subplot(111)
        else:
            fig     = ax.get_figure()

        m = Basemap(llcrnrlon=llcrnrlon,llcrnrlat=llcrnrlat,urcrnrlon=urcrnrlon,urcrnrlat=urcrnrlat,resolution='l',area_thresh=1000.,projection='cyl',ax=ax)

        title = sTime.strftime('WSPR: %d %b %Y %H%M UT - ')+eTime.strftime('%d %b %Y %H%M UT')
#        fontdict = {'size':matplotlib.rcParams['axes.titlesize'],'weight':matplotlib.rcParams['axes.titleweight']}
        fontdict = {'size':matplotlib.rcParams['axes.titlesize'],'weight':'bold'}
        ax.text(0.5,1.075,title,fontdict=fontdict,transform=ax.transAxes,ha='center')

        rft         = self.data_set.metadata.get('reflection_type')
        if rft == 'sp_mid':
            rft = 'Great Circle Midpoints'
        elif rft == 'miller2015':
            rft = 'Multihop'

        subtitle    = 'Reflection Type: {}'.format(rft)
        fontdict    = {'weight':'normal'}
        ax.text(0.5,1.025,subtitle,fontdict=fontdict,transform=ax.transAxes,ha='center')

        # draw parallels and meridians.
        # This is now done in the gridsquare overlay section...
#        m.drawparallels(np.arange( -90., 91.,45.),color='k',labels=[False,True,True,False])
#        m.drawmeridians(np.arange(-180.,181.,45.),color='k',labels=[True,False,False,True])
        m.drawcoastlines(color=coastline_color,zorder=coastline_zorder)
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

        # Only plot the actual receiver location.
        if 'hop_nr' in df.keys():
            tf  = df.hop_nr == 0
            df  = df[tf]

        rx      = m.scatter(df['de_lon'],df['de_lat'],
                s=s,zorder=zorder,**de_prop)

    def plot_dx(self,s=25,zorder=150):
        m       = self.m
        df      = self.data_set.df

        # Only plot the actual receiver location.
        if 'hop_nr' in df.keys():
            tf  = df.hop_nr == 0
            df  = df[tf]

        tx      = m.scatter(df['dx_lon'],df['dx_lat'],
                s=s,zorder=zorder,**dx_prop)

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
        text.append('Relfection Points: {0:d}'.format(len(self.data_set.df)))

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
#        tmp['cbar_ticks']   = [1.8,3.5,7.,10.,14.,21.,24.,28.]
        tmp['cbar_ticks']   = [1.8,3.5, 5., 7.,10.,14.,18.1, 21.,24.,28., 50.]
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
#End of WSPR Class Code

def find_hour(df):
    hours=[]
    for inx in range(0,len(df)):
        hours.append(df['timestamp'].iloc[inx].hour)

    df['hour']=hours
    return df

def find_hour(df, timebin=datetime.timedelta(minutes=30)):
    hours=[]
    for inx in range(0,len(df)):
        hours.append(df['timestamp'].iloc[inx].hour)

    df['hour']=hours
    return df

#def bin_time(df, timebin=datetime.timedelta(minutes=30)):
#    binned=[]
#
#    df=df.sort('timestamp')
#    grouped=df.groupby('hour')
##    for hour in df.hour.unique():
#    sTime=df['timestamp'].min()
#    sTime=df['timestamp'].max()
#    bin1=sTime
#    bin2=sTime+timebin
#    
#    df['time']=df.timestamp.copy()
#
#    while bin1 < eTime:
#        df_temp=df[np.logical_and(bin1<=df['timestamp'], df['timestamp']<bin2)]
#        for inx in range(0,len(df)):
#            hour=df['timestamp'].iloc[inx].hour
#            minute=min
#        hours.append(r)
#        df['time']
#
#
#
#
#        df[bin1<df.timestamp<
#
#        for
#
#    for inx in range(0,len(df)):
#        hours.append(df['timestamp'].iloc[inx].hour)
#
#    df['hour']=hours
#    return df

def redefine_grid(df,precision=4):
    """Define the number of characters used in the grid square reporting

    Parameters
    ----------
    df  :   dataframe
        Dataframe to redefine gridsquares over
    precision   :   int
        Number of characters in grid square

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
    import sys

    print 'Redefining Grid Square Precision to '+str(precision)
    print 'Converting '+str(len(df.grid.unique()))+' grids'
    i=1
    for grid in df.grid.unique():
        new_grid=grid[0:precision]
        if new_grid != grid:
            df=df.replace(to_replace={'grid': {grid:new_grid}})
#        print '{0}\r'.format(str(i)+' grid squares converted'),
        sys.stdout.write('\r'+str(i)+' grid squares converted')
        sys.stdout.flush()
        i=i+1

    print '\nConverting '+str(len(df.rep_grid.unique()))+' reporter grids'
    i=1
    for rep_grid in df.rep_grid.unique():
        new_repgrid=rep_grid[0:precision]
        if new_repgrid != rep_grid:
            df=df.replace(to_replace={'rep_grid': {rep_grid:new_repgrid}})
#        print '{0}\r'.format(str(i)+' grid squares converted'),
        sys.stdout.write('\r'+str(i)+' reporter grid squares converted')
        sys.stdout.flush()
        i=i+1
#            if len(df.rep_grid.unique()) % 100 == 0:
#                print str(i)+' records remaining\n'
#    for inx in range(0,len(df)):
#            grid=df['grid'].iloc[inx][0:precision]
#            rep_grid=df['rep_grid'].iloc[inx][0:precision]  
#            df['rep_grid'].iloc[inx]=df['rep_grid'].iloc[inx][0:precision]
#            df['grid'].iloc[inx]=df['grid'].iloc[inx][0:precision]
    print '\nGrid Square Precision Task Complete'  
    return df

#def filter_grid_pair(df, gridsq, redef=False, precision=4):
#    """Filter to spots with stations only in specified two grid squares
#
#    Parameters
#    ----------
#    df  :   dataframe
#        Dataframe of wspr data
#    gridsq  :   list or numpy.array of str
#        Pair of grid squares to limit stations to 
#    redef   :   boolean
#        Flag to indicate if user would like to redefine grid before filtering
#    precision   :   int
#        Number of characters in grid square
#
#    new_data_set : str
#        Name for the new data_set object.
#    comment : str
#        Comment describing the new data_set object.
#
#    Returns
#    -------
#    new_data_set_obj : data_set 
#        Copy of the original data_set with new name and history entry.
#
#    Written by Magdalina L. Moses, Fall 2016
#    """
#    import wspr_lib
#    if redef:
#        df=wspr_lib.redefine_grid(df, precision=precision)
#   
#    df0=df[np.logical_and(df['grid']==gridsq[0], df['rep_grid']==gridsq[1])]
#    df0=pd.concat([df0,df[np.logical_and(df['grid']==gridsq[1], df['rep_grid']==gridsq[0])]])
#    return df0

#Write new filter code that checks uniques callsigns and copies/concatenates ones matiching the filter into new dataframe
#Combining the two current redefining and filtering functions
def filter_grid_pair(df, gridsq, precision=4):
    """Filter links to those between specified gridsquares

    Parameters
    ----------
    df  :   dataframe
        Dataframe to redefine gridsquares over
    gridsq  :   list or numpy.array of str
        Pair of grid squares to limit stations to 
    precision   :   int
        Number of characters in grid square

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
    import sys

    cond1=np.logical_and(df['grid'].str.startswith(gridsq[0]), df['rep_grid'].str.startswith(gridsq[1]))
    cond2=np.logical_and(df['grid'].str.startswith(gridsq[1]), df['rep_grid'].str.startswith(gridsq[0]))

    print 'Fetching Requested Entries...'

#    cond = []
#    for grid in gridsq:

    df=df[np.logical_or(cond1, cond2)]

    print 'Success!'

    return df

def calls_by_grid(df, prefix='', col='grid', col_call='call_sign'):
    """Find calls of stations in a certain gridsquare 

    Parameters
    ----------
    df  :   dataframe
        Dataframe to search
    prefix  : str
        Prefix of gridsquare
    col : str
        Dataframe column with gridsquares
    col_call : str
        Dataframe column with callsigns

    new_data_set : str
        Name for the new data_set object.
    comment : str
        Comment describing the new data_set object.

    Returns
    -------
    calls : list 
        List of callsigns within the specified gridsquare

    Written by Magdalina L. Moses, Fall 2016
    """
    calls=[]
    precision = len(prefix)
    for inx in range(0,len(df)):
        if df[col].iloc[inx][0:precision]==prefix:
            calls.append(df[col_call].iloc[inx])
##        call=df[col][df[col].iloc[inx][precision]==prefix]
#        if call.any():
#            calls.append(call[:])

    return calls

#def save_wspr(sTime,eTime=None,data_dir='data/wspr'):
#
#    return None
#
#def grsq_latlon(df,geoloc='gridsquare'):
#        """Select nodes based on reporter or transmitter geographic location.
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
#        return df
#
#
#def select_geo(df,node_type='reporter', grsq=None):
#        """Select nodes based on reporter or transmitter geographic location.
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
##    if grdsq ==
##    rbn_lib.latlon_filt(df, )
#
#    return df

def select_pair(df, station):

    df0=df[np.logical_and(df['call_sign']==station[0], df['reporter']==station[1])]
    df=pd.concat([df0,df[np.logical_and(df['call_sign']==station[1], df['reporter']==station[0])]])
    del df0

    return df

def average_dB(df, col='snr'):

    sn=np.power(10,df[col]/10)
    avg=sn.mean()
    df=df.drop(col, axis=1)
    df[col] = 10*np.log10(avg)
    #Could make this function just return the value and not put it in the dataframe yet!
    
    return df

def dB_to_Watt(df, col='snr'):
    """Convert dB values in dataframe column to watts

    Parameters
    ----------
    df  :   dataframe
        Dataframe of wspr data
    col :   str 
        Dataframe column to convert

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
    df[col]=np.power(10,df[col]/10)

    return df

#def station_counts():
#    tx  =   df['call_sign'].unique()
#    rx  =   df['reciever'].unique()
#
#    rx_count=0
#    tx_count=0
#    for this_tx in tx:
#        for this_rx in rx:
#            tx
#    return counts





def plot_wspr_histograms(df):
    import matplotlib       # Plotting toolkit
    matplotlib.use('Agg')   # Anti-grain geometry backend.
                            # This makes it easy to plot things to files without having to connect to X11.
                            # If you want to avoid using X11, you must call matplotlib.use('Agg') BEFORE calling anything that might use pyplot
                            # I don't like X11 because I often run my code in terminal or web server environments that don't have access to it.
    import matplotlib.pyplot as plt #Pyplot gives easier acces to Matplotlib.  
    import pandas as pd     #This is a nice utility for working with time-series type data.
    import os
    import datetime 

    output_dir  = 'output'

    #Pick off the start time and end times.
    sTime       = pd.to_datetime(df['timestamp'].min())
    eTime       = pd.to_datetime(df['timestamp'].max())

    #Sort the data by band and time, then group by band.
    srt         = df.sort(['band','timestamp'])
    grouped     = srt.groupby('band')

    # Plotting section #############################################################
    try:    # Create the output directory, but fail silently if it already exists
        os.makedirs(output_dir) 
    except:
        pass

    # Set up a dictionary which identifies which bands we want and some plotting attributes for each band
    bands       = {}
    bands[28]   = {'name': '10 m',  'color':'red'}
    bands[21]   = {'name': '15 m',  'color':'orange'}
    bands[14]   = {'name': '20 m',  'color':'yellow'}
    bands[7]    = {'name': '40 m',  'color':'green'}
    bands[3]    = {'name': '80 m',  'color':'blue'}
    bands[1]    = {'name': '160 m', 'color':'aqua'}

    # Determine the aspect ratio of each histogram.
    xsize       = 8.
    ysize       = 2.5
    nx_plots    = 1                     # Let's just do 1 panel across.
    ny_plots    = len(bands.keys())     # But we will do a stackplot with one panel for each band of interest.

    fig         = plt.figure(figsize=(nx_plots*xsize,ny_plots*ysize)) # Create figure with the appropriate size.
    subplot_nr  = 0 # Counter for the subplot
    xrng        = (0,15000)
    for band_key in sorted(bands.keys(),reverse=True):   # Now loop through the bands and create 1 histogram for each.
        subplot_nr += 1 # Increment subplot number... it likes to start at 1.
        ax      = fig.add_subplot(ny_plots,nx_plots,subplot_nr)
        grouped.get_group(band_key)['dist'].hist(bins=100,range=xrng,
                    ax=ax,color=bands[band_key]['color'],label=bands[band_key]['name']) #Pandas has a built-in wrapper for the numpy and matplotlib histogram function.
        ax.legend(loc='upper right')
        ax.set_xlim(xrng)
        ax.set_ylabel('WSPR Soundings')

        if subplot_nr == 1:
            txt = []
            txt.append('WSPRNet Distances')
            txt.append(sTime.strftime('%d %b %Y %H%M UT - ')+eTime.strftime('%d %b %Y %H%M UT'))
            ax.set_title('\n'.join(txt)) #\n creates a new line... here I'm joining two strings in a list to form a single string with \n as the joiner

        if subplot_nr == len(bands.keys()):
            ax.set_xlabel('WSPR Reported Distance [km]')

    fig.tight_layout()  #This often cleans up subplot spacing when you have multiple panels.

#    filename    = os.path.join(output_dir,'%s_histogram.png' % year_month)
#    fig.savefig(filename,bbox_inches='tight') # bbox_inches='tight' removes whitespace at the edge of the figure.  Very useful when creating PDFs for papers.

#    filename    = os.path.join(output_dir,'%s_histogram.pdf' % year_month)
#    fig.savefig(filename,bbox_inches='tight') # Now we save as a scalar-vector-graphics PDF ready to drop into PDFLatex

    time_1      = datetime.datetime.now()

#    print 'Total processing time is %.1f s.' % (time_1-time_0).total_seconds()
    return fig

if __name__ == '__main__':
    import datetime

    sTime       = datetime.datetime(2014,2,1)
    eTime       = datetime.datetime(2014,2,28)

    sTime       = datetime.datetime(2016,8,27)
    eTime       = datetime.datetime(2016,8,28)

    sTime       = datetime.datetime(2016,11,11)
    eTime       = datetime.datetime(2016,11,18)
    data_dir    = 'data/wspr' 

    df = read_wspr(sTime,eTime,data_dir)
    import ipdb; ipdb.set_trace()
