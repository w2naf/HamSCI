#!/usr/bin/env python
#This code is used to generate the RBN-GOES map for the Space Weather feature article.

import sys
import os
import datetime

import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt

import numpy as np
import pandas as pd

import rbn_lib
import handling
event_dir   = None

def gen_time_list(sTime,eTime,cTime_dt=datetime.timedelta(minutes=60)):
    """ Generate a list of datetime.datetimes spaced cTime_dt apart.  """
    cTimes  = [sTime]
    while cTimes[-1] <= eTime:
        cTimes.append(cTimes[-1]+cTime_dt)

    return cTimes

# 2014 Nov Sweepstakes
#seTimes = ( datetime.datetime(2014,11,8), datetime.datetime(2014,11,9) )

# 2015 Nov Sweepstakes
#seTimes = ( datetime.datetime(2015,11,7), datetime.datetime(2015,11,10) )
#seTimes = ( datetime.datetime(2015,11,7,12), datetime.datetime(2015,11,8) )
seTimes = ( datetime.datetime(2015,11,8), datetime.datetime(2015,11,8,15) )

# 2016 CQ WPX CW
#seTimes = ( datetime.datetime(2016,5,28,18), datetime.datetime(2016,5,29,6) )


# Script processing begins here. ###############################################
cTimes      = gen_time_list(*seTimes)
cTimes      = cTimes[:1]

if event_dir is None:
    event_dir = '{:%Y%m%d.%H%M}-{:%Y%m%d.%H%M}'.format(*seTimes[:2])
output_path = os.path.join('output',event_dir)
handling.prepare_output_dirs({0:output_path},clear_output_dirs=True)

## Determine the aspect ratio of subplot.
xsize       = 15.0
ysize       = 6.5
nx_plots    = 1
ny_plots    = 1

rcp = mpl.rcParams
rcp['axes.titlesize']     = 'large'
rcp['axes.titleweight']   = 'bold'

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

            rbn_obj = rbn_lib.RbnObject(map_sTime,map_eTime,data_dir='data/rbn')
            
            # Figure out how many records properly geolocated.
            good_loc        = rbn_obj.DS001_dropna.df
            good_count_map  = good_loc['callsign'].count()
            total_count_map = len(rbn_obj.DS000.df)
            good_pct_map    = float(good_count_map) / total_count_map * 100.

            good_count      += good_count_map
            total_count     += total_count_map

            print 'Geolocation success: {0:d}/{1:d} ({2:.1f}%)'.format(good_count_map,total_count_map,good_pct_map)

            # Go plot!!
#            latlon_bounds  = {'llcrnrlat':0.,'llcrnrlon':-180.,'urcrnrlat':90.,'urcrnrlon':0.}
#            latlon_bounds  = {'llcrnrlat':-90.,'llcrnrlon':-180.,'urcrnrlat':90.,'urcrnrlon':180.}
            latlon_bounds  = {'llcrnrlat':20.,'llcrnrlon':-130.,'urcrnrlat':55.,'urcrnrlon':-65.}

            rbn_obj.active.latlon_filt(**latlon_bounds)
#            rbn_obj.active.filter_calls('n7tr')

            rbn_map         = rbn_lib.RbnMap(rbn_obj,ax=ax0)
            rbn_map.default_plot()

            rbn_grid        = rbn_obj.active.create_geo_grid()
            rbn_map.overlay_grid(rbn_grid)
#            rbn_grid.grid_mean()
#            rbn_map.overlay_grid_data(rbn_grid)

            print map_sTime

        fig.savefig(filepath,bbox_inches='tight')
        plt.clf()

        good_pct = float(good_count)/total_count * 100.
        print ''
        print 'Final stats for: {0}'.format(filepath)
        print 'Geolocation success: {0:d}/{1:d} ({2:.1f}%)'.format(good_count,total_count,good_pct)
