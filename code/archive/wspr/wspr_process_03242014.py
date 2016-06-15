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

import h5py


# 

def write_wspr_h5(grouped,band_in, out_file):
	# putting everything into a single numpy array
	rx_grid = grouped.get_group(band_in)['rx_grid'][:]
	tx_grid = grouped.get_group(band_in)['tx_grid'][:]

	rx_call = grouped.get_group(band_in)['rx_call'][:]
	tx_call = grouped.get_group(band_in)['tx_call'][:]

	timestamp = grouped.get_group(band_in)['timestamp'][:]
	freq = grouped.get_group(band_in)['freq'][:]
	snr = grouped.get_group(band_in)['snr'][:]
	dist = grouped.get_group(band_in)['dist'][:]


	# testing
	# grab the slice using the pandas method
	rx_lon1 = rx_grid.str.slice(0,1)
	rx_lat1 = rx_grid.str.slice(1,2)
	rx_lon2 = rx_grid.str.slice(2,3)
	rx_lat2 = rx_grid.str.slice(3,4)
	rx_lon3 = rx_grid.str.slice(4,5)
	rx_lat3 = rx_grid.str.slice(5,6)


	tx_lon1 = tx_grid.str.slice(0,1)
	tx_lat1 = tx_grid.str.slice(1,2)
	tx_lon2 = tx_grid.str.slice(2,3)
	tx_lat2 = tx_grid.str.slice(3,4)
	tx_lon3 = tx_grid.str.slice(4,5)
	tx_lat3 = tx_grid.str.slice(5,6)



	#converting into a numpy array
	rx_lon1 = rx_lon1.values.astype('str')
	rx_lat1 = rx_lat1.values.astype('str')
	rx_lon2 = rx_lon2.values.astype('str')
	rx_lat2 = rx_lat2.values.astype('str')

	tx_lon1 = tx_lon1.values.astype('str')
	tx_lat1 = tx_lat1.values.astype('str')
	tx_lon2 = tx_lon2.values.astype('str')
	tx_lat2 = tx_lat2.values.astype('str')



	# note, I have to treat empty slots.  I know that I will at least have grids
	# with four characters and generally 6, but I need to figure out how to get around
	# the empty slots for 6
	# rx_lon3 = rx_lon3.values.astype('str')
	# rx_lat3 = rx_lat3.values.astype('str')
	# have to use map function on the ord to convert
	# http://stackoverflow.com/questions/7368789/convert-all-strings-in-a-list-to-int

	rx_lon1 = map(ord, rx_lon1)
	rx_lat1 = map(ord, rx_lat1)
	rx_lon2 = map(ord, rx_lon2)
	rx_lat2 = map(ord, rx_lat2)

	tx_lon1 = map(ord, tx_lon1)
	tx_lat1 = map(ord, tx_lat1)
	tx_lon2 = map(ord, tx_lon2)
	tx_lat2 = map(ord, tx_lat2)


	# now that this is a list, I need to convert back to a numpy array
	rx_lon1 = np.array(rx_lon1)
	rx_lat1 = np.array(rx_lat1)
	rx_lon2 = np.array(rx_lon2)
	rx_lat2 = np.array(rx_lat2)

	tx_lon1 = np.array(tx_lon1)
	tx_lat1 = np.array(tx_lat1)
	tx_lon2 = np.array(tx_lon2)
	tx_lat2 = np.array(tx_lat2)

	# will have to update to include the final maidenhead coordinates for a grid square of
	# 6 values

	rx_lon1 = (rx_lon1 - ord('A'))*20 - 180
	rx_lat1 = (rx_lat1- ord('A'))*10 - 90
	rx_lon2 = (rx_lon2 - ord('0'))*2
	rx_lat2 = (rx_lat2 - ord('0'))*1

	tx_lon1 = (tx_lon1 - ord('A'))*20 - 180
	tx_lat1 = (tx_lat1- ord('A'))*10 - 90
	tx_lon2 = (tx_lon2 - ord('0'))*2
	tx_lat2 = (tx_lat2 - ord('0'))*1


	rx_lon = rx_lon1+rx_lon2
	rx_lat = rx_lat1+rx_lat2

	tx_lon = tx_lon1+tx_lon2
	tx_lat = tx_lat1+tx_lat2

	# do some logic as well for outlier points which make no sense if we try to map them
	
	print '------Outliers------'
	print rx_lon[rx_lon < -180]
	print rx_lon[rx_lon > 180]
	print rx_lat[rx_lat < -90]
	print rx_lat[rx_lat > 90]
	print tx_lon[tx_lon < -180]
	print tx_lon[tx_lon > 180]
	print tx_lat[tx_lat < -90]
	print tx_lat[tx_lat > 90]
	
	rx_lon[rx_lon < -180] = -180
	rx_lon[rx_lon > 180] = 180
	rx_lat[rx_lat < -90] = -90
	rx_lat[rx_lat > 90] = 90
	
	tx_lon[tx_lon < -180] = -180
	tx_lon[tx_lon > 180] = 180
	tx_lat[tx_lat < -90] = -90
	tx_lat[tx_lat > 90] = 90
	
	
	


# 	sys.exit()



	rx_call = rx_call.values.astype('str')
	tx_call = tx_call.values.astype('str')
	rx_grid = rx_grid.values.astype('str')
	tx_grid = tx_grid.values.astype('str')
# 	sys.exit()

	freq = freq.values.astype('f')
	snr = snr.values.astype('f')
	dist = dist.values.astype('f')
	timestamp = timestamp.values.astype('int')



	# with lon and lat now going to write the hdf5 file

	# create the hdf5 file
	f = h5py.File(out_file, "w")

	#create a group called data or data15 m or whatever

	grp = f.create_group("data")
	dset = grp.create_dataset('freq', data=freq)
	dset2 = grp.create_dataset('rx_call', data=rx_call)
	dset3 = grp.create_dataset('tx_call', data=tx_call)
	dset4 = grp.create_dataset('rx_grid', data=rx_grid)
	dset5 = grp.create_dataset('tx_grid', data=tx_grid)
	dset6 = grp.create_dataset('dist', data=dist)
	dset7 = grp.create_dataset('timestamp', data=timestamp)
	dset8 = grp.create_dataset('rx_lon', data=rx_lon)
	dset9 = grp.create_dataset('rx_lat', data=rx_lat)
	dset10 = grp.create_dataset('tx_lon', data=tx_lon)
	dset11 = grp.create_dataset('tx_lat', data=tx_lat)


	f.close()


	return 0


	

####################################
# main program
###################################
data_dir    = '/Users/srkaeppler/research/data/wspr/022014/'
output_dir  = 'output'
year_month  = '2014-02'

#Using os.path.join uses the correct path delimiters regardless of OS
#e.g., Linux/Mac = '/' while Windows = '\'
data_file   = 'wsprspots-%s.csv' % year_month
data_path   = os.path.join(data_dir,data_file)  

time_0      = datetime.datetime.now()
print 'Starting WSPRNet histogram processing on <%s> at %s.' % (data_file,str(time_0))

# sys.exit()
# ################################################################################
# # Make sure the data file exists.  If not, download it and open it.
# if not os.path.exists(data_path):
#     try:    # Create the output directory, but fail silently if it already exists
#         os.makedirs(data_dir) 
#     except:
#         pass
#     # File downloading code from: http://stackoverflow.com/questions/22676/how-do-i-download-a-file-over-http-using-python
#     url = 'http://wsprnet.org/archive/'+data_file
#     u = urllib2.urlopen(url)
#     f = open(data_path, 'wb')
#     meta = u.info()
#     file_size = int(meta.getheaders("Content-Length")[0])
#     print "Downloading: %s Bytes: %s" % (data_path, file_size)
# 
#     file_size_dl = 0
#     block_sz = 8192
#     while True:
#         buffer = u.read(block_sz)
#         if not buffer:
#             break
# 
#         file_size_dl += len(buffer)
#         f.write(buffer)
#         status = r"%10d  [%3.2f%%]" % (file_size_dl, file_size_dl * 100. / file_size)
#         status = status + chr(8)*(len(status)+1)
#         print status,
#     f.close()
#     status = 'Done downloading!  Now converting to Pandas dataframe and plotting...'
#     print status

# Load data into dataframe here. ###############################################
# Here I define the column names of the data file, and also specify which ones to load into memory.  By only loading in some, I save time and memory.
names       = ['spot_id', 'timestamp', 'rx_call', 'rx_grid', 'snr', 'freq', 'tx_call', 'tx_grid', 'power', 'drift', 'dist', 'azm', 'band', 'version', 'code']
usecols     = ['spot_id', 'timestamp', 'rx_call', 'rx_grid', 'tx_call', 'tx_grid', 'dist', 'band', 'freq', 'snr','power']

# with gzip.GzipFile(data_path,'rb') as fl:   #This block lets us directly read the compressed gz file into memory.  The 'with' construction means that the file is automatically closed for us when we are done.
#     df          = pd.read_csv(fl,names=names,index_col='spot_id',usecols=usecols)


# If you wanted to read an plain old csv file, you can just use the following command:
df          = pd.read_csv(data_path,names=names,index_col='spot_id',usecols=usecols)


###############################################################
# commented this out because we don't need to worry about - causes time tagging trouble
# unless I rewrite all of this into python, then it might be useful
# sys.exit()
#df['timestamp'] = pd.to_datetime(df['timestamp'],unit='s')#,utc=True)

#Pick off the start time and end times.
#sTime       = pd.to_datetime(df['timestamp'].min())
#sTime       = pd.to_datetime(df['timestamp'].max())
######################################################


#sys.exit()
#Sort the data by band and time, then group by band.
srt         = df.sort(['band','timestamp'])
grouped     = srt.groupby('band')




print 'start 80 m...'
out_file = os.path.join(data_dir,'test_data_80m.hdf5')
band3 = write_wspr_h5(grouped, 3, out_file)
time_1      = datetime.datetime.now()
print 'Total processing time is %.1f s.' % (time_1-time_0).total_seconds()

print 'start 40 m...'
out_file = os.path.join(data_dir,'test_data_40m.hdf5')
band3 = write_wspr_h5(grouped, 7, out_file)
time_1      = datetime.datetime.now()
print 'Total processing time is %.1f s.' % (time_1-time_0).total_seconds()

print 'start 20 m...'
out_file = os.path.join(data_dir,'test_data_20m.hdf5')
band3 = write_wspr_h5(grouped, 14, out_file)
time_1      = datetime.datetime.now()
print 'Total processing time is %.1f s.' % (time_1-time_0).total_seconds()

print 'start 15 m...'
out_file = os.path.join(data_dir,'test_data_15m.hdf5')
band3 = write_wspr_h5(grouped, 21, out_file)
time_1      = datetime.datetime.now()
print 'Total processing time is %.1f s.' % (time_1-time_0).total_seconds()


print 'start 10 m...'
out_file = os.path.join(data_dir,'test_data_10m.hdf5')
band3 = write_wspr_h5(grouped, 28, out_file)
time_1      = datetime.datetime.now()
print 'Total processing time is %.1f s.' % (time_1-time_0).total_seconds()

print 'start 30 m...'
out_file = os.path.join(data_dir,'test_data_30m.hdf5')
band3 = write_wspr_h5(grouped, 10, out_file)
time_1      = datetime.datetime.now()
print 'Total processing time is %.1f s.' % (time_1-time_0).total_seconds()
# 4




time_1      = datetime.datetime.now()

print 'Total processing time is %.1f s.' % (time_1-time_0).total_seconds()

sys.exit()



