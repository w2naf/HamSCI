#!/usr/bin/env python
#Generate various signals for input into fft test functions

#import sys
import os

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import numpy as np
#import pandas as pd

##########################################
#Define signal parameters
f1=100
#f2=1000

theta1=0
#theta2=pi/2

time=np.arrange(0,1,100)
##########################################

##########################################
#Define signals
sig1=[]
#sig2={}
for t in time:
    sig1.append(np.cos(2*pi*f1*t))
#    sig2.append()
##########################################

##########################################
#Plot Signal
#fig = plt.figure(figsize=(8,4))
fig = plt.figure()
ax0 =   fig.add_subplot(1,1,1)

filepath    = os.path.join(output_path,cgraph)
fig.savefig(filepath,bbox_inches='tight')
