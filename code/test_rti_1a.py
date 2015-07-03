#!/usr/bin/env python
#This code is to test a modified version of the rti.py functions from davitpy (modified file is rti_magda.py in ~/HamSCI/code)

import rti_magda


import datetime
import os
import matplotlib.pyplot as plt


#Define SuperDARN radars want on the map
radars=['fhw', 'fhe']

#Specify start and end time
sTime = datetime.datetime(2013,5,12)
eTime = datetime.datetime(2013,5,14)
#sTime = datetime.datetime(2014,9,10)
#eTime = datetime.datetime(2014,9,11)
sat_nr = 15

#specify Beam #
beam = 7

#create output directory if none exists
output_dir='output'
try: 
    os.makedirs(output_dir)
except:
    pass 

outfile='RTIplot_'+radars[0]+'_'+radars[1]

#Define Figure and Axes
fig     = plt.figure(figsize=(10,6))
ax      = fig.add_subplot(2,1,1)
ax0      = fig.add_subplot(2,1,2)

rti_magda.plotRti(sTime=sTime, eTime=eTime, ax=ax, rad=radars[0], params=['power'])
rti_magda.plotRti(sTime=sTime, eTime=eTime, ax=ax0, rad=radars[1], params=['power'])

fig.tight_layout()
filename=os.path.join(output_dir, outfile)
# 'rbnCount_5min_line1.png')
fig.savefig(filename)
