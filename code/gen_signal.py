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
#define Constan and files
output_path = os.path.join('output','fft')
try: 
    os.makedirs(output_path)
except:
    pass 

filename='test_signal1.png'
filepath    = os.path.join(output_path,filename)
##########################################

##########################################
#Define signal parameters
f1=100
#f2=1000

theta1=0
#theta2=pi/2

#import ipdb; ipdb.set_trace()
#time=np.arange(0,4*np.pi,.1)
time=np.arange(0,.03,1e-5)
##########################################

##########################################
#Define signals
sig1=[]
sig2=[]
sig3=[] 
sig4=[] 
sig5=[] 
sig6=[] 
sig7=[] 
sig8=[] 

i=0

for t in time:
    sig1.append(np.cos(2*np.pi*f1*t))
    sig2.append(np.cos(2*np.pi*f1*t+np.pi/2))
    sig3.append(sig1[i]*sig2[i])
    sig4.append(sig1[i]*sig3[i])
    sig5.append(np.cos(2*np.pi*f1*t-np.pi/4))
    sig6.append(np.cos(2*np.pi*f1*t-np.pi/2))
    sig7.append(sig5[i]*sig6[i])
    sig8.append(sig7[i]*sig5[i])
    i=i+1
##########################################

##########################################
#Plot Signal
#fig = plt.figure(figsize=(8,4))
#fig = plt.figure()
fig, ((ax0), (ax1))=plt.subplots(2,1,sharex=True, sharey=False)
#ax0 =   fig.add_subplot(1,1,1)
#ax1 =   fig.add_subplot(2,1,1)
ax0.plot(time,sig1,'-m', time, sig2, '-b',time, sig3, '-r',time, sig4,'-g')
ax1.plot(time, sig5,'c', time,sig6, '-g', time, sig7, '-r', time, sig8, '-b')
fig.savefig(filepath,bbox_inches='tight')
