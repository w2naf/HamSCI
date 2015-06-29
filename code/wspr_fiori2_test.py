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

import wspr_lib
import handling

output_path = os.path.join('output','firori')
handling.prepare_output_dirs({0:output_path},clear_output_dirs=True)

sTime = datetime.datetime(2014,9,10)
eTime = datetime.datetime(2014,9,15)
#sTime = datetime.datetime(2015,3,11)
#eTime = datetime.datetime(2015,3,12)
sat_nr = 15

goes_data = gme.sat.read_goes(sTime,eTime,sat_nr)

##Look at the weekends only...
#inx = [x.weekday() in [5,6] for x in goes_data['xray'].index]
#goes_data['xray'] = goes_data['xray'][inx]

flares = gme.sat.find_flares(goes_data,min_class='X1')

fig = plt.figure(figsize=(8,4))
ax  = fig.add_subplot(1,1,1)
gme.sat.goes_plot(goes_data,ax=ax)
fig.tight_layout()
filename = os.path.join(output_path,'goes.png')
fig.savefig(filename,bbox_inches='tight')
plt.close(fig)
import ipdb; ipdb.set_trace()


## Determine the aspect ratio of subplot.
xsize       = 6.5
ysize       = 5.5
nx_plots    = 2
ny_plots    = 2

for inx,flare in flares.iterrows():
#    try:
    if True:
        filename    = inx.strftime('rbn_%Y%m%d_%H%M.png')
        filepath    = os.path.join(output_path,filename)

        int_min     = 30
        map_times   = [inx - datetime.timedelta(minutes=(int_min*2))]
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

           # import ipdb; ipdb.set_trace()
            # Go plot!!
            rbn_lib.rbn_map_plot(rbn_df,legend=False,ax=ax0,tick_font_size=9,ncdxf=True)
            title = map_sTime.strftime('%H%M - ')+map_eTime.strftime('%H%M UT')
            ax0.set_title(title)
            letter_prop = {'weight':'bold','size':20}
            ax0.text(.015,.90,'({0})'.format(letters[kk]),transform=ax0.transAxes,**letter_prop)

#            for item in (ax0.get_xticklabels() + ax0.get_yticklabels()):
#                item.set_fontsize(4)

            print flare
            print map_sTime

        import ipdb; ipdb.set_trace()
        ax      = fig.add_subplot(3,1,3)
        
        ax.plot(inx,flare['B_AVG'],'o',label='{0} Class Flare @ {1}'.format(flare['class'],inx.strftime('%H%M UT')))
        goes_map_sTime = inx.to_datetime() - datetime.timedelta(hours=3)
        goes_map_eTime = inx.to_datetime() + datetime.timedelta(hours=3)

#        goes_sTime = datetime.datetime(2013,5,13,16) - datetime.timedelta(hours=3)
#        goes_eTime = datetime.datetime(2013,5,13,16) + datetime.timedelta(hours=3)

        goes_data_map = gme.sat.read_goes(goes_map_sTime,goes_map_eTime,sat_nr=sat_nr)

        gme.sat.goes_plot(goes_data_map,ax=ax)
        leg = rbn_lib.band_legend(fig,loc='center',bbox_to_anchor=[0.48,0.360],ncdxf=True)

        title_prop = {'weight':'bold','size':22}
        fig.text(0.525,1.025,'Reverse Beacon Network',ha='center',**title_prop)
        fig.text(0.525,0.995,flare.name.strftime('%d %B %Y'),ha='center',size=18)

        fig.tight_layout(h_pad=2.5,w_pad=3.5)
        x0, y0, width, height = ax.get_position().bounds
        
        width   = 0.55
        x0      = (1.-width) / 2. + 0.025
        y0      = .050
        height  = 0.25
        ax.set_position([x0,y0,width,height])

        ax.text(-0.0256,-0.125,flare.name.strftime('%d %b %Y'),transform=ax.transAxes)
        ax.set_xlabel('Time [UT]')
        ax.set_title('NOAA GOES 15')
        ax.text(.015,.90,'(e)',transform=ax.transAxes,**letter_prop)

        xticks  = []
        for x in range(7):
            xticks.append(goes_map_sTime + datetime.timedelta(hours=(1*x)))
        ax.xaxis.set_ticks(xticks)

        ax.title.set_fontsize(title_prop['size'])
        ax.title.set_weight(title_prop['weight'])

        xticklabels = []
        for x,tick in enumerate(xticks):
            xticklabels.append(tick.strftime('%H%M'))
        ax.set_xticklabels(xticklabels)

        fig.savefig(filepath,bbox_inches='tight')
        fig.savefig(filepath[:-3]+'pdf',bbox_inches='tight')
        plt.clf()

        good_pct = float(good_count)/total_count * 100.
        print ''
        print 'Final stats for: {0}'.format(filepath)
        print 'Geolocation success: {0:d}/{1:d} ({2:.1f}%)'.format(good_count,total_count,good_pct)
#    except:
#        pass
