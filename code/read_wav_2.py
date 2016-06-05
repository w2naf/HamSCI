#!/usr/bin/env python
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import datetime

fname   = 'B2_20151128_003000_7091kHz.wav'
#stime   = datetime.datetime(2015,11,28,0,30)

#wav     = sf.read(fname)
#data    = wav[0]
#samp_rt = wav[1]


figsize = (10,8)
fig     = plt.figure(figsize=figsize)
ax      = fig.add_subplot(111)

#sec     = np.arange(data.shape[0])/float(samp_rt)
#mt      = sec / 60.
#xvals   = mt
#yvals   = np.sqrt(data[:,0]**2 + data[:,1]**2)
#
#tf      = xvals <= 1.
#xvals   = xvals[tf]
#yvals   = yvals[tf]

xvals   = np.arange(100)
yvals   = np.arange(100)
ax.plot(xvals,yvals)
ax.set_xlabel('Index')
ax.set_ylabel('Voltage Magnitude')

fig.savefig('rbn_voltage.png')
import ipdb; ipdb.set_trace()
