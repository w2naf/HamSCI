#!/usr/bin/env python
#Including the above line as the first line of the script allows this script to be run
#directly from the command line without first calling python/ipython.

# started by N. Frissell and written by SR Kaeppler 09 01 2014

#Now we import libraries that are not "built-in" to python.
import os               # Provides utilities that help us do os-level operations like create directories
import datetime         # Really awesome module for working with dates and times.
import gzip             # Allows us to read from gzipped files directly!
import urllib2          # Used to automatically download data files from the web.
import sys

#import matplotlib       # Plotting toolkit
#matplotlib.use('Agg')   # Anti-grain geometry backend.
                        # This makes it easy to plot things to files without having to connect to X11.
                        # If you want to avoid using X11, you must call matplotlib.use('Agg') BEFORE calling anything that might use pyplot
                        # I don't like X11 because I often run my code in terminal or web server environments that don't have access to it.
import matplotlib.pyplot as plt #Pyplot gives easier acces to Matplotlib.  

import numpy as np      #Numerical python - provides array types and operations
import pandas as pd     #This is a nice utility for working with time-series type data.
import scipy
import h5py
from mpl_toolkits.basemap import Basemap
from functools import partial
from matplotlib import colors
import matplotlib as mpl
import matplotlib.gridspec as grd

from subprocess import call

def lon_lat_convert(grid):
	n = len(grid)
# 	lon,lat = np.float()
	
	if (n == 4):
		lon1 = np.float(ord(grid[0]) - ord('A'))*20 - 180
		lat1 = np.float(ord(grid[1])- ord('A'))*10 - 90
		lon2 = np.float(ord(grid[2]) - ord('0'))*2
		lat2 = np.float(ord(grid[3]) - ord('0'))*1
		
		lon = lon1+lon2
		lat = lat1+lat2
	
	if(n == 6):
		lon1 = np.float(ord(grid[0]) - ord('A'))*20 - 180
		lat1 = np.float(ord(grid[1])- ord('A'))*10 - 90
		lon2 = np.float(ord(grid[2]) - ord('0'))*2
		lat2 = np.float(ord(grid[3]) - ord('0'))*1
		lon3 = np.float(ord(grid[4]) - ord('A'))*5/60
		lat3 = np.float(ord(grid[5]) - ord('A'))*2.5/60
		lon = lon1+lon2+lon3
		lat = lat1+lat2+lat3

	if(lon > 180):
		lon = 180
	if(lon < -180):
		lon = -180
	if(lat > 90):
		lat = 90
	if(lat < -90):
		lat = -90
	
	return lon, lat
	
# 


def plt_map(df,t0,tend,dt,out_path_map,data_dir):
	tmp_t = t0
	t =[0,0]
	while( tmp_t < tend):
		time_loop0      = datetime.datetime.now()
	#	print tmp_t
		t1 = tmp_t
		t2 = tmp_t+datetime.timedelta(minutes=dt)
		print t1, t2	




		#Sort the data by band and time, then group by band.
		srt         = df.sort(['band','timestamp'])
		grouped     = srt.groupby('band')

	# condition for selecting 
		cond0 = ( (df['timestamp']>t1) & (df['timestamp']<t2) ) 
# 		(df['rx_lat'] > 30) & (df['rx_lat'] < 70) &
# 		(df['tx_lat'] > 30) & (df['tx_lat'] < 70) &
# 		(df['rx_lon'] < 50) & (df['rx_lon'] > -20) &
# 		(df['tx_lon'] < 50) & (df['tx_lon'] > -20) )
	
		# Set up a dictionary which identifies which bands we want and some plotting attributes for each band
		bands       = {}
		bands[50] = {'name': '6 m / 50 MHz',  'color':'red', 'i':7, 'f':50}
		bands[28]   = {'name': '10 m / 28 MHz',  'color':'pink', 'i':5, 'f':28}
		bands[21]   = {'name': '15 m / 21 MHz',  'color':'orange','i':3,'f':21}
		bands[14]   = {'name': '20 m / 14 MHz',  'color':'yellow','i':1,'f':14}
		bands[10]   = {'name': '30 m / 10 MHz',  'color':'green', 'i':6, 'f':10} 
		bands[7]    = {'name': '40 m / 7 MHz',  'color':'blue', 'i':4, 'f':7}
		bands[3]    = {'name': '80 m / 3.5 MHz',  'color':'purple', 'i':2, 'f':3}
		bands[1]    = {'name': '160 m / 1.5 MHz', 'color':'black', 'i':0, 'f':1}

		# Determine the aspect ratio of each histogram.
		xsize       = 10.
		ysize       = 2.5
		nx_plots    = 2                    # Let's just do 1 panel across.
		ny_plots    = len(bands.keys())     # But we will do a stackplot with one panel for each band of interest.

		fig         = plt.figure(figsize=(nx_plots*xsize,ny_plots*ysize)) # Create figure with the appropriate size.
		subplot_nr  = 0 # Counter for the subplot
		xrng        = (0,5000)

		#make a color map
		# from http://stackoverflow.com/questions/9707676/defining-a-discrete-colormap-for-imshow-in-matplotlib
		colorList = [bands[x]['color'] for x in sorted(bands.keys())]
		cmap = colors.ListedColormap(colorList)
		bounds=[np.int(bands[x]['f']) for x in sorted(bands.keys())]
		bounds.insert(0,0)
		norm = colors.BoundaryNorm(bounds, cmap.N)
		cmap.set_over('0.25')
		cmap.set_under('0.75')

		# make basemap
		fig2 = plt.figure(figsize=(8,8))

		gs = grd.GridSpec(2, 2,width_ratios =[50,1],height_ratios=[1,1],wspace=0.1, hspace=0.25)

		ax2 = plt.subplot(gs[0])
		ax3 = plt.subplot(gs[1])


		m = Basemap(llcrnrlon=-179.,llcrnrlat=-80.,urcrnrlon=179.,urcrnrlat=80,\
					rsphere=(6378137.00,6356752.3142),\
					resolution='i',projection='merc',\
					ax=ax2)
		m.drawcountries()
		m.drawcoastlines()
		m.drawstates()
		m.fillcontinents(color='#bbbbbf')
		CS=m.nightshade(t1)
		# draw parallels
		m.drawparallels(np.arange(-90,90,10),labels=[1,1,0,1])
		# draw meridians
		m.drawmeridians(np.arange(-180,180,20),labels=[1,1,0,1])
	

		ax2.set_title('WSPR Map: '+t1.strftime('%d %b %Y %H%M UT - ')+t2.strftime('%d %b %Y %H%M UT'))
	
	
		cb1 = mpl.colorbar.ColorbarBase(ax3, cmap=cmap, norm=norm,boundaries=bounds,
		spacing='uniform',orientation='vertical')
		cb1.set_label('Frequency (MHz)')
		plot_time = datetime.datetime.utcnow()
		txt_2 = ('Plotted: '+plot_time.strftime('%d %b %Y %H%M UT'))

		ax2.set_xlabel('\n'+txt_2)

		gs00 = grd.GridSpecFromSubplotSpec(4, 2, subplot_spec=gs[2])
	
		#sys.exit()
		# want to include shareex somewhere in here...?
		for band_key in sorted(bands.keys()): # Now loop through the bands and create 1 histogram for each.
	 
			#subplot_nr += 1 # Increment subplot number... it likes to start at 1.# do the calculation, cond0 is defined above
			subplot_nr = np.int(bands[band_key]['i'])
			print band_key,subplot_nr
			ax = plt.Subplot(fig2, gs00[subplot_nr])
			# ax      = fig2.add_subplot(ny_plots,nx_plots,subplot_nr)
			fig2.add_subplot(ax)
			dist= grouped.get_group(band_key)['dist'][cond0]
			bins_in 	= np.arange(0,5000,200)
			hist,bins_out =np.histogram(dist, bins=bins_in)
			#code from: http://stackoverflow.com/questions/17874063/is-there-a-parameter-in-matplotlib-pandas-to-have-the-y-axis-of-a-histogram-as-p
			ax.bar(bins_out[:-1], hist.astype(np.float32), width=(bins_out[1]-bins_out[0]), color=bands[band_key]['color'], label=bands[band_key]['name'])
			ax.set_ylim(0,30)
			#    grouped.get_group(band_key)['dist'][cond0].hist(bin=bins_in,norm=True,
			#                ax=ax,color=bands[band_key]['color'],label=bands[band_key]['name']) #Pandas has a built-in wrapper for the numpy and matplotlib histogram function.
			ax.legend(loc='upper right')
			ax.set_xlim(xrng)
			ax.set_ylabel('WSPR Spots (#)')
			tx_lon = grouped.get_group(band_key)['tx_lon'][cond0].values
			rx_lon = grouped.get_group(band_key)['rx_lon'][cond0].values
			tx_lat = grouped.get_group(band_key)['tx_lat'][cond0].values
			rx_lat = grouped.get_group(band_key)['rx_lat'][cond0].values
	
			if (subplot_nr == 0) | (subplot_nr == 1):
				txt = []
				txt.append('WSPRNet Distances')
				txt.append(t1.strftime('%d %b %Y %H%M UT - ')+t2.strftime('%d %b %Y %H%M UT'))
				ax.set_title('\n'.join(txt)) #\n creates a new line... here I'm joining two strings in a list to form a single string with \n as the joiner
			if subplot_nr == len(bands.keys()):
				txt_2 = ('Plotted: '+plot_time.strftime('%d %b %Y %H%M UT'))
				ax.set_xlabel('WSPR Reported Distance [km]'+txt_2)
	
		
	
			# setup mercator map projection.
	
			print tx_lon
	
	
			map(partial(m.drawgreatcircle,color=bands[band_key]['color']),tx_lon,tx_lat,rx_lon,rx_lat)
	

		#fig2.tight_layout()  #This often cleans up subplot spacing when you have multiple panels.

	# 	filename    = os.path.join(out_path_hist,'%s.png' % t1.strftime('%H%MUT_WSPRHist') )
		#sys.exit()
		#fig.savefig(filename)#,bbox_inches='tight') # bbox_inches='tight' removes whitespace at the edge of the figure.  Very useful when creating PDFs for papers.
		file_map    = os.path.join(out_path_map, '%s.png' % t1.strftime('%m%d_%H%MUT')) #Y%m%d
		fig2.savefig(file_map)#, bbox_inches='tight')

		plt.close('all')
		tmp_t = tmp_t+datetime.timedelta(minutes=dt)

	# os.chdir(out_path_map)
# 	call(['ffmpeg', '-framerate', '4','-pattern_type','glob','-i', '*.png',
# 					 '-s:v','1280x720','-c:v','libx264','-profile:v','high',
# 					 '-crf','23','-pix_fmt','yuv420p','-r','30','%s.mp4'% t1.strftime('%m%d')])
# 
# 	os.chdir(data_dir)
	# out_files = os.path.join(out_path_hist,'*.png')
	# out_mov = os.path.join(out_path_hist,'test.mp4')
	# command1 = 'ffmpeg -framerate 4 -pattern_type glob  -i '+out_files+' -s:v 1280x720 -c:v libx264 -profile:v high -crf 23 -pix_fmt yuv420p -r 30 '+ out_mov
	# call(command1, shell=True)
	time_1      = datetime.datetime.now()

	print 'Total processing time is %.1f s.' % (time_1-time_0).total_seconds()
	return None
	

####################################
# main program
###################################

data_dir    = '/Users/srkaeppler/research/data/hamscience/wspr/'
output_dir  = 'output'
in_dir ='raw_data'
month = '01'
day = '01'
year = '2014'
# 32 for 31 day months
#31 for 30 day months
day_var = range(1,3)
day_str = ['%02d' % x for x in day_var]

year_month = year+'-'+month

data_file   = 'wsprspots-%s.csv' % year_month
data_path   = os.path.join(data_dir,in_dir,data_file) 

date_str= year+month+day
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
# 
# Load data into dataframe here. ###############################################
# Here I define the column names of the data file, and also specify which ones to load into memory.  By only loading in some, I save time and memory.
names       = ['spot_id', 'timestamp', 'rx_call', 'rx_grid', 'snr', 'freq', 'tx_call', 'tx_grid', 'power', 'drift', 'dist', 'azm', 'band', 'version', 'code']
usecols     = ['spot_id', 'timestamp', 'rx_call', 'rx_grid', 'tx_call', 'tx_grid', 'dist', 'band', 'freq', 'snr','power']
# 
# with gzip.GzipFile(data_path,'rb') as fl:   #This block lets us directly read the compressed gz file into memory.  The 'with' construction means that the file is automatically closed for us when we are done.
#     df          = pd.read_csv(fl,names=names,index_col='spot_id',usecols=usecols)
# 
# 
# If you wanted to read an plain old csv file, you can just use the following command:
df          = pd.read_csv(data_path,names=names,index_col='spot_id',usecols=usecols)
# 
# 
# #process to get the rx lon/lat
# 
# print 'process rx lon/lat'
rx_lon_lat = np.array(map(lon_lat_convert, df['rx_grid']))
df['rx_lon'] = rx_lon_lat[:,0]
df['rx_lat'] = rx_lon_lat[:,1]
# 
# print 'process tx lon/lat'
tx_lon_lat = np.array(map(lon_lat_convert, df['tx_grid'])) 
df['tx_lon'] = tx_lon_lat[:,0]
df['tx_lat'] = tx_lon_lat[:,1]
# 
# ###############################################################
# commented this out because we don't need to worry about - causes time tagging trouble
# unless I rewrite all of this into python, then it might be useful
# sys.exit()
df['timestamp'] = pd.to_datetime(df['timestamp'],unit='s')#,utc=True)
# 

######################################################


i=0
while(i < day_var[-1]):
	
	if(i == day_var[-2]):
		t0 = datetime.datetime(int(year),int(month),int(day_var[i]),00,00,00)
		tend = datetime.datetime(int(year),int(month),int(day_var[i]),23,59,00)
		out_path_map = os.path.join(data_dir,output_dir,'world',year,month,day_str[i])
	else:
		t0 = datetime.datetime(int(year),int(month),int(day_var[i]),00,00,00)
		tend = datetime.datetime(int(year),int(month),int(day_var[i]),23,59,00)
		out_path_map = os.path.join(data_dir,output_dir,'world',year,month,day_str[i])
		print t0,tend	
	# out_path_map = os.path.join(data_dir,output_dir,year,month,day_str[0])
	# #sys.exit()
	# #make sure output directories exist
	# # this section has been grabbed directly from nathaniel's code	
	# # Plotting section #############################################################	
	try:# Create the output directory, but fail silently if it already exists
		os.makedirs(out_path_map)
	except:
			pass
	# 
	foo = plt_map(df,t0,tend,59,out_path_map,data_dir)

	print i,i+1,t0,tend,out_path_map
	i = i+1









