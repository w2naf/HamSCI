#!/usr/bin/env python
#Make SuperDARN RTI plots to study terminator effects

import sys
sys.path.append('/data/mypython')
import os

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

#import numpy as np
#import pandas as pd
#
#from davitpy import gme
import datetime

#import rbn_lib
import handling
#import eclipse_lib

#import rti_magda
from davitpy.pydarn.radar import *
from davitpy.pydarn.plotting import *
from davitpy.utils import *

#from davitpy import pydarn
#from pylab import gca
#from matplotlib.patches import Polygon 
# should use functions from calcSun
#call("avconv -i %06d-symlink.png "+outFile,shel

sTime=datetime.datetime(2013,5,13,23,0)
sTime=datetime.datetime(2013,3,13,23,0)
sTime=datetime.datetime(2015,8,21,23,0)
sTime=datetime.datetime(2013,5,14,0,0)
sTime=datetime.datetime(2013,5,14,1,0)
eTime = sTime + datetime.timedelta(hours=4) #minutes=15)
eTime = sTime + datetime.timedelta(hours=5) #minutes=15)
import ipdb; ipdb.set_trace()

radar='fhw'
#radar='cvw'
#beam=23
beam=8
beam=19
beam=21
#Define visual properties of Radars on the map 
fovColor=(0.5,0,0.75)
#fovZorder is the zorder of the FOV with higher zorder=on top
fovZorder=10

mapProj='cyl'
llcrnrlon=-130 
llcrnrlat=25
urcrnrlon=-65
urcrnrlat=52 

output_path = os.path.join('output', 'superdarn')
filename='RTI_Terminator_'+radar+'_'+sTime.strftime('%Y%m%d_%H%M')+'-'+eTime.strftime('%Y%m%d_%H%M')+'.png'
filename2='Map_RTI_Terminator_'+radar+'_'+sTime.strftime('%Y%m%d_%H%M')+'-'+eTime.strftime('%Y%m%d_%H%M')+'.png'
filepath=os.path.join(output_path,filename)
filepath2=os.path.join(output_path,filename2)

## Determine the aspect ratio of subplot.
xsize       = 8.0
#ysize       = 6.0
ysize       = 4.0
nx_plots    = 1
ny_plots    = 1

#figure(figsize=(10,10))
#width = 111e3*40
fig         = plt.figure(figsize=(nx_plots*xsize,ny_plots*ysize)) # Create figure with the appropriate size.
ax0     = fig.add_subplot(1,1,1)
m = plotUtils.mapObj(llcrnrlon=llcrnrlon,llcrnrlat=llcrnrlat,urcrnrlon=urcrnrlon,urcrnrlat=urcrnrlat,resolution='l',area_thresh=1000.,projection=mapProj,ax=ax0,fillContinents='None', fix_aspect=True)

# Plotting some radars\n",
#overlayRadar(m, fontSize=12, codes=code)
# Plot radar fov\n",
#overlayFov(m, codes=code, maxGate=75, beams=[0,4,7,8,23])
overlayRadar(m,fontSize=12,codes=radar,dateTime=sTime)
fan.plotFan(sTime,['fhe','fhw', 'cve', 'cvw'],param='power',gsct=True, show=False, png=True)
import ipdb; ipdb.set_trace()
#plotFan(sTime,rad,interval=60,fileType='fitex',param='power',filtered=False, show=False)
#overlayFov(m, codes=radar, maxGate=40, model='GS', fovColor=fovColor,zorder=fovZorder)
overlayFov(m, codes=radar, maxGate=40, beams=beam, model='GS', fovColor=fovColor,zorder=fovZorder)
fig.savefig(filepath2,bbox_inches='tight')

#overlayRadar(m,fontSize=12,codes=radar,dateTime=sTime)
#overlayFov(m, codes=radar, maxGate=75) #, beams=[0,4,7,8,23])
##overlayFov(m, codes=radar, maxGate=40, beams=beam) #, model='GS', fovColor=fovColor,zorder=fovZorder)
#fig.savefig(filepath2,bbox_inches='tight')

del fig

fig = plt.figure(figsize=(14,12)) #Define a figure with a custom size.\n",
plotRti(sTime, rad=radar, eTime=eTime, bmnum=beam, figure=fig, coords='geo',plotTerminator=True)
#plotRti(sTime, rad=radar, eTime=eTime, bmnum=beam, figure=fig) #,plotTerminator=True)
#plotRti(sTime, rad=radar, eTime=eTime, bmnum=beam)

#fig.savefig(filepath,bbox_inches='tight')
fig.savefig(filepath)
