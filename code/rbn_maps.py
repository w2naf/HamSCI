#!/usr/bin/env python
#Including the above line as the first line of the script allows this script to be run
#directly from the command line without first calling python/ipython.

#Now we import libraries that are not "built-in" to python.
import os               # Provides utilities that help us do os-level operations like create directories
import datetime         # Really awesome module for working with dates and times.
import zipfile
import urllib2          # Used to automatically download data files from the web.
import sys
import pickle

from functools import partial

import matplotlib       # Plotting toolkit
matplotlib.use('Agg')   # Anti-grain geometry backend.
                        # This makes it easy to plot things to files without having to connect to X11.
                        # If you want to avoid using X11, you must call matplotlib.use('Agg') BEFORE calling anything that might use pyplot
                        # I don't like X11 because I often run my code in terminal or web server environments that don't have access to it.
import matplotlib.pyplot as plt #Pyplot gives easier acces to Matplotlib.  

from mpl_toolkits.basemap import Basemap

import numpy as np      #Numerical python - provides array types and operations
import pandas as pd     #This is a nice utility for working with time-series type data.

import hamtools
from hamtools import qrz

call    = 'w2naf'
passwd  = 'hamscience'
qz      = qrz.Session(call,passwd)

data_dir    = os.path.join('data','rbn')
output_dir  = os.path.join('output','rbn')
dt          = datetime.datetime(2014,1,2)
ymd         = dt.strftime('%Y%m%d')

plot_dict       = {}
plot_dict[0]    = {'sTime': datetime.datetime(2014,10,25,16,45), 'eTime': datetime.datetime(2014,10,25,17)}
plot_dict[1]    = {'sTime': datetime.datetime(2014,10,25,17,15), 'eTime': datetime.datetime(2014,10,25,17,30)}
plot_dict[2]    = {'sTime': datetime.datetime(2015,3,11,15,55), 'eTime': datetime.datetime(2015,3,11,16,10)}
plot_dict[3]    = {'sTime': datetime.datetime(2015,3,11,16,30), 'eTime': datetime.datetime(2015,3,11,16,45)}

#Using os.path.join uses the correct path delimiters regardless of OS
#e.g., Linux/Mac = '/' while Windows = '\'
#http://www.reversebeacon.net/raw_data/dl.php?f=20140101
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
#     url = 'http://wsprnet.org/archive/'+data_file
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

for kk in range(len(plot_dict)):
    data_time_0 = plot_dict[kk]['sTime']
    data_time_1 = plot_dict[kk]['eTime']
    p_filename = 'rbn_'+data_time_0.strftime('%Y%m%d%H%M-')+data_time_1.strftime('%Y%m%d%H%M.p')
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
        df = df[np.logical_and(df['date'] >= data_time_0,df['date'] < data_time_1)]

        # Look up lat/lons in QRZ.com
        errors = 0
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
            except:
#                print '{index:06d} LOOKUP ERROR - DX: {dx} DE: {de}'.format(index=index,dx=dx_call,de=de_call)
                errors += 1
        print 'Total errors: {0:d}'.format(errors)
        df.to_pickle(p_filepath)
    else:
        with open(p_filepath,'rb') as fl:
            df = pickle.load(fl)

    plot_dict[kk]['df'] = df

# Plotting section #############################################################
try:    # Create the output directory, but fail silently if it already exists
    os.makedirs(output_dir) 
except:
    pass

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

## Determine the aspect ratio of subplot.
xsize       = 8
ysize       = 5.5
nx_plots    = 2
ny_plots    = 2

fig         = plt.figure(figsize=(nx_plots*xsize,ny_plots*ysize)) # Create figure with the appropriate size.
subplot_nr  = 0 # Counter for the subplot

for kk in range(len(plot_dict)):
    #Drop NaNs (QSOs without Lat/Lons)
    df = plot_dict[kk]['df'].dropna(subset=['dx_lat','dx_lon','de_lat','de_lon'])

    ##Sort the data by band and time, then group by band.
    df['band']  = np.array((np.floor(df['freq']/1000.)),dtype=np.int)
    srt         = df.sort(['band','date'])
    grouped     = srt.groupby('band')

    plt_inx = kk + 1
    axis = fig.add_subplot(ny_plots,nx_plots,plt_inx)

    plot_sTime = plot_dict[kk]['sTime']
    plot_eTime = plot_dict[kk]['eTime']
    half_time   = datetime.timedelta(seconds= ((plot_eTime - plot_sTime).total_seconds()/2.) )
    plot_mTime = plot_sTime + half_time

    m = Basemap(llcrnrlon=-180.,llcrnrlat=-90,urcrnrlon=180.,urcrnrlat=90.,resolution='l',area_thresh=1000.,projection='cyl',ax=axis)

    title = plot_sTime.strftime('%H%M - ')+plot_eTime.strftime('%H%M UT')
    axis.set_title(title)

    # draw parallels and meridians.
    m.drawparallels(np.arange(-90.,91.,45.),color='k',labels=[False,True,True,False])
    m.drawmeridians(np.arange(-180.,181.,45.),color='k',labels=[True,False,False,True])
    m.drawcoastlines(color='0.65')
    m.drawmapboundary(fill_color='w')
    m.nightshade(plot_mTime,color='0.82')
    
    for band in bandlist:
        try:
            this_group = grouped.get_group(band)
        except:
            continue
#        de_lat = grouped.get_group(band)['de_lat']
#        de_lon = grouped.get_group(band)['de_lon']
#        dx_lat = grouped.get_group(band)['dx_lat']
#        dx_lon = grouped.get_group(band)['dx_lon']

        color = band_dict[band]['color']
        label = band_dict[band]['name']

#        map(partial(m.drawgreatcircle,color=color,label=label),dx_lon,dx_lat,de_lon,de_lat)

        for index,row in this_group.iterrows():
            #Yay stack overflow! - http://stackoverflow.com/questions/13888566/python-basemap-drawgreatcircle-function
            de_lat = row['de_lat']
            de_lon = row['de_lon']
            dx_lat = row['dx_lat']
            dx_lon = row['dx_lon']

            line, = m.drawgreatcircle(dx_lon,dx_lat,de_lon,de_lat,color=color)
#            map(partial(m.drawgreatcircle,color=color,label=label),dx_lon,dx_lat,de_lon,de_lat)

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


import matplotlib.patches as mpatches
handles = []
labels  = []
for band in bandlist:
    color = band_dict[band]['color']
    label = band_dict[band]['freq']
    handles.append(mpatches.Patch(color=color,label=label))
    labels.append(label)
fig.legend(handles,labels,ncol=len(labels),loc='lower center',markerscale=0.5,prop={'size':10})

title = '\n'.join(['Reverse Beacon Network Data',plot_sTime.strftime('%Y %B %d')])
fig.text(0.5,0.95,title,ha='center')

fig.tight_layout()  #This often cleans up subplot spacing when you have multiple panels.
filename    = os.path.join(output_dir,'%s_map.png' % ymd)
fig.savefig(filename,bbox_inches='tight') # bbox_inches='tight' removes whitespace at the edge of the figure.  Very useful when creating PDFs for papers.

time_1      = datetime.datetime.now()
print 'Total processing time is %.1f s.' % (time_1-time_0).total_seconds()
