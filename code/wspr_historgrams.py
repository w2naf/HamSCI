#!/usr/bin/env python
#Including the above line as the first line of the script allows this script to be run
#directly from the command line without first calling python/ipython.

#Now we import libraries that are not "built-in" to python.
import os               # Provides utilities that help us do os-level operations like create directories
import datetime         # Really awesome module for working with dates and times.
import gzip             # Allows us to read from gzipped files directly!
import urllib2          # Used to automatically download data files from the web.
import sys

import matplotlib       # Plotting toolkit
matplotlib.use('Agg')   # Anti-grain geometry backend.
                        # This makes it easy to plot things to files without having to connect to X11.
                        # If you want to avoid using X11, you must call matplotlib.use('Agg') BEFORE calling anything that might use pyplot
                        # I don't like X11 because I often run my code in terminal or web server environments that don't have access to it.
import matplotlib.pyplot as plt #Pyplot gives easier acces to Matplotlib.  

import numpy as np      #Numerical python - provides array types and operations
import pandas as pd     #This is a nice utility for working with time-series type data.


data_dir    = 'data'
output_dir  = 'output'
year_month  = '2014-02'

#Using os.path.join uses the correct path delimiters regardless of OS
#e.g., Linux/Mac = '/' while Windows = '\'
data_file   = 'wsprspots-%s.csv' % year_month
data_path   = os.path.join(data_dir,data_file)  

time_0      = datetime.datetime.now()
print 'Starting WSPRNet histogram processing on <%s> at %s.' % (data_file,str(time_0))

 sys.exit()
 ################################################################################
 # Make sure the data file exists.  If not, download it and open it.
 if not os.path.exists(data_path):
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
     status = 'Done downloading!  Now converting to Pandas dataframe and plotting...'
     print status

# Load data into dataframe here. ###############################################
# Here I define the column names of the data file, and also specify which ones to load into memory.  By only loading in some, I save time and memory.
names       = ['spot_id', 'timestamp', 'reporter', 'rep_grid', 'snr', 'freq', 'call_sign', 'grid', 'power', 'drift', 'dist', 'azm', 'band', 'version', 'code']
usecols     = ['spot_id', 'timestamp', 'reporter', 'call_sign', 'dist', 'band']

# with gzip.GzipFile(data_path,'rb') as fl:   #This block lets us directly read the compressed gz file into memory.  The 'with' construction means that the file is automatically closed for us when we are done.
#     df          = pd.read_csv(fl,names=names,index_col='spot_id',usecols=usecols)

sys.exit()
# If you wanted to read an plain old csv file, you can just use the following command:
df          = pd.read_csv(data_path,names=names,index_col='spot_id',usecols=usecols)

df['timestamp'] = pd.to_datetime(df['timestamp'],unit='s',utc=True)

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

filename    = os.path.join(output_dir,'%s_histogram.png' % year_month)
fig.savefig(filename,bbox_inches='tight') # bbox_inches='tight' removes whitespace at the edge of the figure.  Very useful when creating PDFs for papers.

filename    = os.path.join(output_dir,'%s_histogram.pdf' % year_month)
fig.savefig(filename,bbox_inches='tight') # Now we save as a scalar-vector-graphics PDF ready to drop into PDFLatex

time_1      = datetime.datetime.now()

print 'Total processing time is %.1f s.' % (time_1-time_0).total_seconds()
