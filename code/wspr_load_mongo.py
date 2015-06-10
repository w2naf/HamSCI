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

import pymongo
mongo       = pymongo.MongoClient()
db          = mongo.wspr


data_dir    = 'data'
output_dir  = 'output'
year_month  = '2014-02'

#Using os.path.join uses the correct path delimiters regardless of OS
#e.g., Linux/Mac = '/' while Windows = '\'
data_file   = 'wsprspots-%s.csv.gz' % year_month
data_path   = os.path.join(data_dir,data_file)  

time_0      = datetime.datetime.now()
print 'Starting WSPRNet histogram processing on <%s> at %s.' % (data_file,str(time_0))

# sys.exit()
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

with gzip.GzipFile(data_path,'rb') as fl:   #This block lets us directly read the compressed gz file into memory.  The 'with' construction means that the file is automatically closed for us when we are done.
     df          = pd.read_csv(fl,names=names,index_col='spot_id',usecols=usecols)

# If you wanted to read an plain old csv file, you can just use the following command:
#df          = pd.read_csv(data_path,names=names,index_col='spot_id',usecols=usecols)

df['timestamp'] = pd.to_datetime(df['timestamp'],unit='s',utc=True)

import json
records = json.loads(df.T.to_json()).values()
