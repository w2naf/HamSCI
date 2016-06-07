#!/usr/bin/env python
#This code is used to generate the RBN-GOES map for the Space Weather feature article.

import sys
import os

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import numpy as np
import pandas as pd

from davitpy import gme
import datetime

import rbn_lib
import handling

output_path = os.path.join('output','wpx2016')
handling.prepare_output_dirs({0:output_path},clear_output_dirs=True)

#sTime   = datetime.datetime(2016,5,28,23)
#eTime   = datetime.datetime(2016,5,29,1)
#cTimes  = []
#cTimes.append(datetime.datetime(2016,5,29))

sTime   = datetime.datetime(2016,6,3,23)
eTime   = datetime.datetime(2016,6,4,1)
cTimes  = []
cTimes.append(datetime.datetime(2016,6,4))

## Determine the aspect ratio of subplot.
xsize       = 6.5
ysize       = 5.5
nx_plots    = 2
ny_plots    = 2

for cTime in cTimes:
    if True:
        filename    = cTime.strftime('rbn_%Y%m%d_%H%M.png')
        filepath    = os.path.join(output_path,filename)

        int_min     = 30
        map_times   = [cTime - datetime.timedelta(minutes=(int_min*2))]
        for x in range(3):
            map_times.append(map_times[-1] + datetime.timedelta(minutes=int_min))

        fig         = plt.figure(figsize=(nx_plots*xsize,ny_plots*ysize)) # Create figure with the appropriate size.
        subplot_nr  = 0 # Counter for the subplot
        letters = 'abcd'

        good_count  = 0
        total_count = 0
        for kk,map_sTime in enumerate(map_times):
            plt_inx = kk + 1
            ax0     = fig.add_subplot(3,2,plt_inx)

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
            rbn_lib.rbn_map_plot(rbn_df,legend=False,ax=ax0,tick_font_size=9,ncdxf=True)
            title = map_sTime.strftime('%H%M - ')+map_eTime.strftime('%H%M UT')
            ax0.set_title(title)
            letter_prop = {'weight':'bold','size':20}
            ax0.text(.015,.90,'({0})'.format(letters[kk]),transform=ax0.transAxes,**letter_prop)

            print map_sTime

        leg = rbn_lib.band_legend(fig,loc='center',bbox_to_anchor=[0.48,0.180],ncdxf=True)

        title_prop = {'weight':'bold','size':22}
        fig.text(0.525,0.925,'Reverse Beacon Network',ha='center',**title_prop)

        fig.savefig(filepath,bbox_inches='tight')
        fig.savefig(filepath[:-3]+'pdf',bbox_inches='tight')
        plt.clf()

        good_pct = float(good_count)/total_count * 100.
        print ''
        print 'Final stats for: {0}'.format(filepath)
        print 'Geolocation success: {0:d}/{1:d} ({2:.1f}%)'.format(good_count,total_count,good_pct)
