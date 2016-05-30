#!/usr/bin/env python
#This code is used to generate the Eclipse-RBN-SuperDARN-VTARA Site map for CEDAR 2016.

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

import rti_magda
from davitpy.pydarn.radar import *
from davitpy.pydarn.plotting import *
from davitpy.utils import *

from davitpy import pydarn
from pylab import gca
from matplotlib.patches import Polygon 

#Map Properties 
#define map projection 
mapProj='cyl'
llcrnrlon=-130 
llcrnrlat=25
urcrnrlon=-65
urcrnrlat=52 

x1=0.125
w1=0.775
h1=0.235
y1=0.6647
space=0.05
y2=y1+h1+space
map_ax=[[x1,y1, w1, h1], [x1, y2, w1, h1]]

#Define Eclipse Path limits
eLimits=['ds_NL.csv', 'ds_SL.csv']
import ipdb; ipdb.set_trace()
#Define visual 
eColor=(0.75,0.25,0.5)
#pZorder is the zorder of the eclipse path with higher zorder=on top
pZorder=9

#define RBN alpha
path_alpha=0.3

#Define SuperDARN radars want on the map 
#Note: Data for the RTI plot will be taken from beam of radars[0]
radars=['fhw', 'fhe','cvw','cve']
beam=7
#Define visual properties of Radars on the map 
fovColor=(0.5,0,0.75)
#fovZorder is the zorder of the FOV with higher zorder=on top
fovZorder=10

#Specify start and end time
sTime = datetime.datetime(2013,5,12)
eTime = datetime.datetime(2013,5,14)
##sTime = datetime.datetime(2014,9,10)
##eTime = datetime.datetime(2014,9,11)
#sat_nr = 15

output_path = os.path.join('output','cedar')
#handling.prepare_output_dirs({0:output_path},clear_output_dirs=True)
## Determine the aspect ratio of subplot.
xsize       = 8.0
#ysize       = 6.0
ysize       = 4.0
nx_plots    = 1
ny_plots    = 1

#int_min     = 30
#map_times   = [inx - datetime.timedelta(minutes=(int_min*2))]
#for x in range(3):
#    map_times.append(map_times[-1] + datetime.timedelta(minutes=int_min))

fig         = plt.figure(figsize=(nx_plots*xsize,ny_plots*ysize)) # Create figure with the appropriate size.

good_count  = 0
total_count = 0

map_sTime=datetime.datetime(2013,5,13,15,5)
map_eTime = map_sTime + datetime.timedelta(minutes=15)

filename    = 'cedar_2017eclipse_rbn_superdarn_'+sTime.strftime('%Y%m%d_%H%M')+'.png'
filepath    = os.path.join(output_path,filename)

#        map_times = []
#        map_times.append(datetime.datetime(2013,5,13,15,5))
#        map_times.append(datetime.datetime(2013,5,13,16,5))

ax0     = fig.add_subplot(1,1,1)

print ''
print '################################################################################'
print 'Plotting RBN Map: {0} - {1}'.format(map_sTime.strftime('%d %b %Y %H%M UT'),map_eTime.strftime('%d %b %Y %H%M UT'))

rbn_df  = rbn_lib.read_rbn(map_sTime,map_eTime,data_dir='data/rbn')
#rbn_df  = rbn_lib.read_rbn_std(map_sTime,map_eTime,data_dir='data/rbn')

# Figure out how many records properly geolocated.
good_loc    = rbn_df.dropna(subset=['dx_lat','dx_lon','de_lat','de_lon'])
good_count_map  = good_loc['callsign'].count()
total_count_map = len(rbn_df)
good_pct_map    = float(good_count_map) / total_count_map * 100.

good_count      += good_count_map
total_count     += total_count_map

print 'Geolocation success: {0:d}/{1:d} ({2:.1f}%)'.format(good_count_map,total_count_map,good_pct_map)

# Go plot!!
m,fig=rbn_lib.rbn_map_plot(rbn_df,legend=False,ax=ax0,tick_font_size=9,ncdxf=True,llcrnrlon=llcrnrlon, llcrnrlat=llcrnrlat, urcrnrlon=urcrnrlon, urcrnrlat=urcrnrlat, proj=mapProj, basemapType=False, eclipse=True,path_alpha=path_alpha)
#m,fig=rbn_lib.rbn_map_plot(rbn_df,legend=True,ax=ax0,tick_font_size=9,ncdxf=True,llcrnrlon=llcrnrlon, llcrnrlat=llcrnrlat, urcrnrlon=urcrnrlon, urcrnrlat=urcrnrlat, proj=mapProj, basemapType=False, eclipse=True,path_alpha=path_alpha)

#Plot Eclipse path swath on map
m,fig=eclipse_lib.eclipse_swath(infile=eLimits,mapobj=m, fig=fig, pathColor=eColor, pZorder=pZorder)

#Plot VTARA Field stations
m,fig=eclipse_lib.plot_vtara_stations(m=m, color='c', fig=fig, ax=ax0) 

#Plot SuperDARN Radars of interest on map (Fort Hayes West and Fort Hayes East on the map)
#for code in radars:
#            overlayRadar(m,fontSize=12,codes=radars,dateTime=map_sTime)
#            overlayFov(m, codes=radars, maxGate=40, beams=beam)
#First plot radar with the beam shown in the FOV which we will plot RTI data from later (radars[0])
overlayRadar(m,fontSize=12,codes=radars[0],dateTime=map_sTime)
overlayFov(m, codes=radars[0], maxGate=40, beams=beam,model='GS', fovColor=fovColor,zorder=fovZorder)
#Next plot the rest of the radars (no beams will be plotted)
overlayRadar(m,fontSize=12,codes=radars[1:],dateTime=map_sTime)
overlayFov(m, codes=radars[1:], maxGate=40, beams=None,model='GS', fovColor=fovColor,zorder=fovZorder)


#Titles and other propertites
#title = map_sTime.strftime('%H%M - ')+map_eTime.strftime('%H%M UT')
title='2017 Solar Eclipse\nSuperDARN Radars and Reverse Beacon Network'
title='2017 Solar Eclipse'
ax0.set_title(title,loc='center')
print map_sTime


shift=0.070
shift=0.035
#        yl=y2-shift-0.025
#        leg = rbn_lib.band_legend(fig,loc='center',bbox_to_anchor=[0.48,yl],ncdxf=True,ncol=4)
x0, y0, width, height = ax0.get_position().bounds
#y_pos=y0-shift-0.025
y_pos=y0-shift
leg = rbn_lib.band_legend(fig,loc='center',bbox_to_anchor=[0.45,y_pos],ncdxf=True,ncol=4)
#leg = rbn_lib.band_legend(fig,loc='center',bbox_to_anchor=[0.48,0.360],ncdxf=True)
title_prop = {'weight':'bold','size':22}
#        fig.text(0.525,1.025,'HF Communication Paths',ha='center',**title_prop)
#fig.text(0.525,1.000,'2017 Solar Eclipse\nSuperDARN Radars and Reverse Beacon Network',ha='center',**title_prop)
#fig.text('2017 Solar Eclipse\nSuperDARN Radars and Reverse Beacon Network',ha='center',**title_prop)
#        fig.text(0.525,0.995,flare.name.strftime('%d %B %Y'),ha='center',size=18)

fig.savefig(filepath,bbox_inches='tight')
fig.savefig(filepath[:-3]+'pdf',bbox_inches='tight')
plt.clf()

good_pct = float(good_count)/total_count * 100.
print ''
print 'Final stats for: {0}'.format(filepath)
print 'Geolocation success: {0:d}/{1:d} ({2:.1f}%)'.format(good_count,total_count,good_pct)
#    except:
#        pass
