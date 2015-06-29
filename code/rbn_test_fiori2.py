#!/usr/bin/env python
import os
import datetime

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import rbn_lib

sTime   = datetime.datetime(2015,3,11, 16, 22)
eTime   = datetime.datetime(2015,3,11, 16, 52)

data_dir    = os.path.join('data','rbn')
rbn_df  = rbn_lib.read_rbn(sTime,eTime,data_dir=data_dir)
import ipdb; ipdb.set_trace()

fig     = plt.figure(figsize=(10,8))
ax      = fig.add_subplot(111)

rbn_lib.rbn_map_plot(rbn_df,legend=False,ax=ax)
outfile = os.path.join('output','rbn','rbn_test_fiori2.png')
fig.savefig(outfile,bbox_inches='tight')
