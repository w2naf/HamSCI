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

from davitpy import gme
import datetime

import rbn_lib
import handling
import eclipse_lib

sTime = datetime.datetime(2013,5,12)
eTime = datetime.datetime(2013,5,14)
#sTime = datetime.datetime(2014,9,10)
#eTime = datetime.datetime(2014,9,11)
sat_nr = 15

goes_data = gme.sat.read_goes(sTime,eTime,sat_nr)

##Look at the weekends only...
#inx = [x.weekday() in [5,6] for x in goes_data['xray'].index]
#goes_data['xray'] = goes_data['xray'][inx]

flares = gme.sat.find_flares(goes_data,min_class='X2')

output_path = os.path.join('output','rbn')
handling.prepare_output_dirs({0:output_path},clear_output_dirs=True)
## Determine the aspect ratio of subplot.
xsize       = 8.0
ysize       = 6.0
nx_plots    = 1
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

        map_times = []
        map_times.append(datetime.datetime(2013,5,13,15,5))
        map_times.append(datetime.datetime(2013,5,13,16,5))
        for kk,map_sTime in enumerate(map_times):
            plt_inx = kk + 1
            ax0     = fig.add_subplot(3,1,plt_inx)

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
            m,fig=rbn_lib.rbn_map_plot(rbn_df,legend=False,ax=ax0,tick_font_size=9,ncdxf=True,llcrnrlon=-130 ,llcrnrlat=20, urcrnrlon=-60, urcrnrlat=60 , eclipse=True)
            #Plot Eclipse cetral line on map
            #cl_color='green'
            m,fig=eclipse_lib.eclipse_map_plot(infile='ds_CL.csv',mapobj=m, fig=fig, style='--m')
            m,fig=eclipse_lib.eclipse_map_plot(infile='ds_NL.csv',mapobj=m, fig=fig, style='--m')
            m,fig=eclipse_lib.eclipse_map_plot(infile='ds_SL.csv',mapobj=m, fig=fig, style='--m')
            #Titles and other propertites
            title = map_sTime.strftime('%H%M - ')+map_eTime.strftime('%H%M UT')
            ax0.set_title(title,loc='center')
            ax0.set_title(map_sTime.strftime('%d %b %Y'),loc='right')
            if kk == 0:
                ax0.set_title('Preflare',loc='left')
            else:
                ax0.set_title('Flare Peak',loc='left')

            letter_prop = {'weight':'bold','size':20}
#            ax0.text(.015,.90,'({0})'.format(letters[kk]),transform=ax0.transAxes,**letter_prop)

#            for item in (ax0.get_xticklabels() + ax0.get_yticklabels()):
#                item.set_fontsize(4)

            print flare
            print map_sTime

        ax      = fig.add_subplot(3,1,3)
        ax.plot(inx,flare['B_AVG'],'o',label='{0} Class Flare @ {1}'.format(flare['class'],inx.strftime('%H%M UT')))
#        goes_map_sTime = datetime.datetime(inx.year,inx.month,inx.day)
#        goes_map_eTime = goes_map_sTime + datetime.timedelta(days=1)

        goes_sTime = datetime.datetime(2013,5,13,16) - datetime.timedelta(hours=3)
        goes_eTime = datetime.datetime(2013,5,13,16) + datetime.timedelta(hours=3)

        goes_data_map = gme.sat.read_goes(goes_sTime,goes_eTime,sat_nr=sat_nr)

        gme.sat.goes_plot(goes_data_map,ax=ax,legendLoc='lower right')
        leg = rbn_lib.band_legend(fig,loc='center',bbox_to_anchor=[0.48,0.305],ncdxf=True,ncol=4)
        title_prop = {'weight':'bold','size':22}
#        fig.text(0.525,1.025,'HF Communication Paths',ha='center',**title_prop)
        fig.text(0.525,1.000,'Reverse Beacon Network\nSolar Flare HF Communication Paths',ha='center',**title_prop)
#        fig.text(0.525,0.995,flare.name.strftime('%d %B %Y'),ha='center',size=18)

        fig.tight_layout(h_pad=2.5,w_pad=3.5)
        x0, y0, width, height = ax.get_position().bounds

        ax0_bounds = ax0.get_position().bounds
        ax_bounds = ax.get_position().bounds
        
        width   = 0.80
        x0      = (1.-width) / 2. + 0.050
#        y0      = .050
        y0      = .080
        height  = 0.200
        ax.set_position([x0,y0,width,height])

        ax.text(-0.0320,-0.140,flare.name.strftime('%d %b %Y'),transform=ax.transAxes)
        ax.set_xlabel('Time [UT]')
        ax.set_title('NOAA GOES 15')
#        ax.text(.015,.90,'(e)',transform=ax.transAxes,**letter_prop)

        xticks  = []
        for x in range(7):
            xticks.append(goes_sTime + datetime.timedelta(hours=(1*x)))
        ax.xaxis.set_ticks(xticks)

        ax.title.set_fontsize(title_prop['size'])
        ax.title.set_weight(title_prop['weight'])

        xticklabels = []
        for x,tick in enumerate(xticks):
            xticklabels.append(tick.strftime('%H%M'))
        ax.set_xticklabels(xticklabels)

        ax.vlines(map_times,0,1,linestyle='--',color='b')

        fig.savefig(filepath,bbox_inches='tight')
        fig.savefig(filepath[:-3]+'pdf',bbox_inches='tight')
        plt.clf()

        good_pct = float(good_count)/total_count * 100.
        print ''
        print 'Final stats for: {0}'.format(filepath)
        print 'Geolocation success: {0:d}/{1:d} ({2:.1f}%)'.format(good_count,total_count,good_pct)
#    except:
#        pass
