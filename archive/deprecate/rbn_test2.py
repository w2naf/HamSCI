#!/usr/bin/env python
import os
import datetime
import pickle

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import numpy as np
import pandas as pd

import utils
import rbn_lib

#sTime   = datetime.datetime(2010,11,19)
#eTime   = datetime.datetime(2010,11,19)
#
data_dir    = os.path.join('data','rbn')
#rbn_df  = rbn_lib.read_rbn(sTime,eTime,data_dir=data_dir)

datafile = 'rbn_20101119.p'
with open(os.path.join(data_dir,datafile),'rb') as fl:
    rbn_df = pickle.load(fl)

import ipdb; ipdb.set_trace()
rbn_df  = rbn_df.dropna(subset=['dx_lat','dx_lon','de_lat','de_lon'],how='any')
grp     = rbn_df.groupby('callsign')


fig     = plt.figure(figsize=(10,8))
ax      = fig.add_subplot(111)

aa  = grp.count().date
aa  = aa.sort(inplace=False)

call    = 'WZ7I'
sub_df  = rbn_df[rbn_df.callsign == call]

lat_lon = zip(sub_df['de_lat'].tolist(),sub_df['de_lon'].tolist(),sub_df['dx_lat'].tolist(),sub_df['dx_lon'].tolist())
#dist    = utils.geoPack.greatCircleDist(sub_df['de_lat'],sub_df['de_lon'],sub_df['dx_lat'],sub_df['dx_lon'])
dist    = np.array([utils.geoPack.greatCircleDist(*x) for x in lat_lon]) * utils.Re

sub_df['dist'] = dist
sub_df['mhz'] = np.floor(sub_df['freq']/1000.)
for band,dct in rbn_lib.band_dict.iteritems():
    tmp = sub_df[sub_df.mhz == band]

    xx = np.array(tmp.date.tolist())
    ax.plot(xx, tmp.dist, color=dct['color'],label=dct['name'])

ax.legend()
ax.set_xlim(datetime.datetime(2010,11,19,12),datetime.datetime(2010,11,19,15))

outfile = os.path.join('output','rbn','rbn_test.png')
fig.savefig(outfile,bbox_inches='tight')

#fig     = plt.figure(figsize=(10,8))
#ax      = fig.add_subplot(111)
#
#rbn_lib.rbn_map_plot(rbn_df,legend=False,ax=ax,plot_paths=False)
#outfile = os.path.join('output','rbn','rbn_test.png')
#fig.savefig(outfile,bbox_inches='tight')
