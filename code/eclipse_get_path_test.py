#!/usr/bin/env python
import eclipse_lib

import rti_magda
from davitpy.pydarn.radar import *
from davitpy.pydarn.plotting import *
from davitpy.utils import *
import datetime
import os
import matplotlib.pyplot as plt

#create output directory if none exists
output_dir='output'
try: 
    os.makedirs(output_dir)
except:
    pass 

outfile='Countour_Test'

#Define Figure and Axes
fig     = plt.figure(figsize=(10,6))
ax      = fig.add_subplot(2,1,1)
ax0      = fig.add_subplot(2,1,2)


eLimits=['ds_NL.csv', 'ds_SL.csv']
import ipdb; ipdb.set_trace()
m = plotUtils.mapObj(llcrnrlon=-180.,llcrnrlat=-90,urcrnrlon=180.,urcrnrlat=90.,resolution='l',area_thresh=1000.,projection='cyl',ax=ax,fillContinents='None')
m,fig=eclipse_lib.eclipse_swath(infile=eLimits,mapobj=m, fig=fig, style='--m')
import ipdb; ipdb.set_trace()

fig.tight_layout()
filename=os.path.join(output_dir, outfile)
fig.savefig(filename)

import ipdb; ipdb.set_trace()
