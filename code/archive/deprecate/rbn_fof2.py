#!/usr/bin/env python
#This code is used to generate the RBN-GOES map for the Space Weather feature article.

import sys
sys.path.append('/data/mypython')
import os

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import numpy as np
import pandas as pd

import gme
import datetime

import rbn_lib
import handling

output_path = os.path.join('output','rbn')
handling.prepare_output_dirs({0:output_path},clear_output_dirs=True)
## Determine the aspect ratio of subplot.
xsize       = 6.5
ysize       = 5.5
nx_plots    = 1
ny_plots    = 1
plot_nr     = 0

map_sTime = datetime.datetime(2013,5,13,15,5)
map_eTime = datetime.datetime(2013,5,13,15,20)

filename    = map_sTime.strftime('rbn_fof2_%Y%m%d_%H%M.png')
filepath    = os.path.join(output_path,filename)

fig         = plt.figure(figsize=(nx_plots*xsize,ny_plots*ysize)) # Create figure with the appropriate size.

good_count  = 0
total_count = 0

plot_nr     += 1
ax0     = fig.add_subplot(ny_plots,nx_plots,plot_nr)

print ''
print '################################################################################'
print 'Plotting RBN Map: {0} - {1}'.format(map_sTime.strftime('%d %b %Y %H%M UT'),map_eTime.strftime('%d %b %Y %H%M UT'))

rbn_df  = rbn_lib.read_rbn(map_sTime,map_eTime,data_dir='data/rbn')

# Figure out how many records properly geolocated.
good_loc        = rbn_df.dropna(subset=['dx_lat','dx_lon','de_lat','de_lon'])
good_count_map  = good_loc['callsign'].count()
total_count_map = len(rbn_df)
good_pct_map    = float(good_count_map) / total_count_map * 100.

good_count      += good_count_map
total_count     += total_count_map

print 'Geolocation success: {0:d}/{1:d} ({2:.1f}%)'.format(good_count_map,total_count_map,good_pct_map)

# Go plot!!
bounds = dict(llcrnrlon=-135.,llcrnrlat=20,urcrnrlon=-60.,urcrnrlat=60.)
rbn_lib.rbn_map_plot(rbn_df,legend=True,ax=ax0,tick_font_size=9,ncdxf=True,**bounds)
title = map_sTime.strftime('%H%M - ')+map_eTime.strftime('%H%M UT')
ax0.set_title(title)

#leg = rbn_lib.band_legend(fig,loc='center',bbox_to_anchor=[0.48,0.360],ncdxf=True)

title_prop = {'weight':'bold','size':22}
fig.text(0.525,1.025,'Reverse Beacon Network',ha='center',**title_prop)

fig.tight_layout(h_pad=2.5,w_pad=3.5)
fig.savefig(filepath,bbox_inches='tight')

good_pct = float(good_count)/total_count * 100.
print ''
print 'Final stats for: {0}'.format(filepath)
print 'Geolocation success: {0:d}/{1:d} ({2:.1f}%)'.format(good_count,total_count,good_pct)
import ipdb; ipdb.set_trace()
