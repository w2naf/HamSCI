#!/usr/bin/env python
import os
import datetime

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import wspr_lib
#plot histograms for first plot of fiori2 Sept.
sTime   = datetime.datetime(2014,9,10,16,45)
eTime   = datetime.datetime(2014,9,11,17,00)

data_dir    = os.path.join('data','wspr')
wspr_df  = wspr_lib.read_wspr(sTime,eTime,data_dir=data_dir)
import ipdb; ipdb.set_trace()

#fig     = plt.figure(figsize=(10,8))
#ax      = fig.add_subplot(111)

wspr_lib.plot_wspr_histograms(wspr_df)
outfile = os.path.join('output','wspr','wspr_test.png')
fig.savefig(outfile,bbox_inches='tight')
