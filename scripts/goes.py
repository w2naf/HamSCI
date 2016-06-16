#!/usr/bin/env python
#This code will search through an extended period of GOES data, find large solar flares,
#and then plot the GOES data Reverse Beacon Network data around the flare.

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

sTime = datetime.datetime(2013,1,1)
eTime = datetime.datetime(2014,9,1)
sat_nr = 15

goes_data = gme.sat.read_goes(sTime,eTime,sat_nr)

##Look at the weekends only...
#inx = [x.weekday() in [5,6] for x in goes_data['xray'].index]
#goes_data['xray'] = goes_data['xray'][inx]

flares = gme.sat.find_flares(goes_data,min_class='X1')
import ipdb; ipdb.set_trace()

output_path = os.path.join('output','rbn')
handling.prepare_output_dirs({0:output_path},clear_output_dirs=True)
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
        for kk,map_sTime in enumerate(map_times):
            plt_inx = kk + 1
            ax      = fig.add_subplot(3,2,plt_inx)

            map_eTime = map_sTime + datetime.timedelta(minutes=15)

            rbn_df  = rbn_lib.read_rbn(map_sTime,map_eTime,data_dir='data/rbn')
            rbn_lib.rbn_map_plot(rbn_df,legend=False,ax=ax)
            print flare
            print map_sTime

        ax      = fig.add_subplot(3,1,3)
        
        ax.plot(inx,flare['B_AVG'],'o',label='{0} Class Flare @ {1}'.format(flare['class'],inx.strftime('%H%M UT')))
        goes_map_sTime = datetime.datetime(inx.year,inx.month,inx.day)
        goes_map_eTime = goes_map_sTime + datetime.timedelta(days=1)
        goes_data_map = gme.sat.read_goes(goes_map_sTime,goes_map_eTime,sat_nr=sat_nr)

        gme.sat.goes_plot(goes_data_map,ax=ax)

        minor_ticks = [goes_map_sTime + datetime.timedelta(hours=(x*3)) for x in range(1,8)]
        ax.minorticks_on()
        ax.xaxis.set_ticks(minor_ticks,minor=True)
        ax.grid(which='minor',axis='x')

        rbn_lib.band_legend(fig,loc=(0.25,0.98))
        fig.tight_layout(h_pad=2.5)
        fig.savefig(filepath,bbox_inches='tight')
        plt.clf()
#    except:
#        pass
