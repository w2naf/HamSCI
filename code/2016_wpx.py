#!/usr/bin/env python
#This code is used to generate the RBN-GOES map for the Space Weather feature article.

import sys
import os

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import numpy as np
import pandas as pd

import datetime

import rbn_lib
import handling

output_path = os.path.join('output','wpx2016')
handling.prepare_output_dirs({0:output_path},clear_output_dirs=True)

sTime   = datetime.datetime(2016,5,28,23)
eTime   = datetime.datetime(2016,5,29,1)

cTimes  = [datetime.datetime(2016,5,29)]

#cTime_0 = datetime.datetime(2016,5,28,23)
#cTime_1 = datetime.datetime(2016,5,29,1)
#cTime_dt = datetime.timedelta(minutes=15)
#cTimes  = [cTime_0]
#while cTimes[-1] <= cTime_1:
#    cTimes.append(cTimes[-1]+cTime_dt)

##sTime   = datetime.datetime(2016,6,3,23)
##eTime   = datetime.datetime(2016,6,4,1)
##cTimes  = []
##cTimes.append(datetime.datetime(2016,6,4))

## Determine the aspect ratio of subplot.
xsize       = 10.0
ysize       = 6.0
nx_plots    = 1
ny_plots    = 1

for cTime in cTimes:
    if True:
        filename    = cTime.strftime('rbn_%Y%m%d_%H%M.png')
        filepath    = os.path.join(output_path,filename)

        int_min     = 30
        map_times   = [cTime]

        fig         = plt.figure(figsize=(nx_plots*xsize,ny_plots*ysize)) # Create figure with the appropriate size.
        subplot_nr  = 0 # Counter for the subplot

        good_count  = 0
        total_count = 0
        for kk,map_sTime in enumerate(map_times):
            plt_inx = kk + 1
            ax0     = fig.add_subplot(1,1,plt_inx)

            map_eTime = map_sTime + datetime.timedelta(minutes=15)

            print ''
            print '################################################################################'
            print 'Plotting RBN Map: {0} - {1}'.format(map_sTime.strftime('%d %b %Y %H%M UT'),map_eTime.strftime('%d %b %Y %H%M UT'))

            rbn_df  = rbn_lib.read_rbn(map_sTime,map_eTime,data_dir='data/rbn')

            # Figure out how many records properly geolocated.
            good_loc    = rbn_df.dropna(subset=['dx_lat','dx_lon','de_lat','de_lon'])
            good_count_map  = good_loc['callsign'].count()
            total_count_map = len(rbn_df)
            good_pct_map    = float(good_count_map) / total_count_map * 100.

            good_count      += good_count_map
            total_count     += total_count_map

            print 'Geolocation success: {0:d}/{1:d} ({2:.1f}%)'.format(good_count_map,total_count_map,good_pct_map)


            # Go plot!!
            latlon_bounds  = {'llcrnrlat':0.,'llcrnrlon':-180.,'urcrnrlat':90.,'urcrnrlon':0.}
            rbn_map         = rbn_lib.RbnMap(rbn_df,ax=ax0,**latlon_bounds)
            rbn_map.default_plot()

            rbn_grid        = rbn_lib.RbnGeoGrid(rbn_map.df)
            rbn_grid.grid_mean()

            rbn_map.overlay_rbn_grid(rbn_grid)

            print map_sTime


        fig.savefig(filepath,bbox_inches='tight')
        plt.clf()

        good_pct = float(good_count)/total_count * 100.
        print ''
        print 'Final stats for: {0}'.format(filepath)
        print 'Geolocation success: {0:d}/{1:d} ({2:.1f}%)'.format(good_count,total_count,good_pct)
