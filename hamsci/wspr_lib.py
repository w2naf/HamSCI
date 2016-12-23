#!/usr/bin/env python
#Including the above line as the first line of the script allows this script to be run
#directly from the command line without first calling python/ipython.
de_prop         = {'marker':'^','edgecolor':'k','facecolor':'white'}
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

        return df

class WsprObject(object):
    """
    gridsquare_precision:   Even number, typically 4 or 6
    reflection_type:        Model used to determine reflection point in ionopshere.
                            'sp_mid': spherical midpoint

        Written by Magdalina Moses and Nathaniel A. Frissell, Summer 2016 and Winter 2016/17
    """
    def __init__(self,sTime=None,eTime=None,data_dir='data/wspr',
            overwrite=False,refresh=False,qrz_call=None,qrz_passwd=None,comment='Raw Data',df=None,
            gridsquare_precision=4,reflection_type='sp_mid'):

        if df is None:
            df = read_wspr(sTime=sTime,eTime=eTime,data_dir=data_dir,
                    overwrite=overwrite, refresh=refresh)

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

    def dxde_gs_latlon_data(self,pos='center'):
        """
        Determine latitde and longitude data for dx and de stations from the reported gridsquares for the data.

        The method appends de and dx lat and lons to current dataframe and does
        NOT create a new dataset.
        """
        print 'Finding dx lat/lon....'
        self.latlon_data(position=pos,grid_key='grid',loc_key=['dx_lat','dx_lon'])
        print 'Finding de lat/lon....'
        self.latlon_data(position=pos,
            grid_key='rep_grid',loc_key=['de_lat','de_lon'])

        return self
        
    def latlon_data(self,position='center',
            grid_key='grid',loc_key=['dx_lat','dx_lon']):
        """
        Determine latitde and longitude data from the reported gridsquares for the data.

        The method appends de and dx lat and lons to current dataframe and does
        NOT create a new dataset.
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
