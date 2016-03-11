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
#define files
output_path = os.path.join('output','fft')
try: 
    os.makedirs(output_path)
except:
    pass 

filename='test_fft.png'
filename2='test_signal_and_fft.png'
filepath    = os.path.join(output_path,filename)
##########################################

###########################################
#FFT Test code from iPython`
print np.fft.fft(np.exp(2j * np.pi * np.arange(8) / 8))
#array([ -3.44505240e-16 +1.14383329e-17j,
#             8.00000000e+00 -5.71092652e-15j,
#                      2.33482938e-16 +1.22460635e-16j,
#                               1.64863782e-15 +1.77635684e-15j,
#                                        9.95839695e-17 +2.33482938e-16j,
#                                                 0.00000000e+00 +1.66837030e-15j,
#                                                          1.14383329e-17 +1.22460635e-16j,
#                                                                   -1.64863782e-15 +1.77635684e-15j])
t = np.arange(256)
sp = np.fft.fft(np.sin(t))
freq = np.fft.fftfreq(t.shape[-1])
fig = plt.figure()
plt.plot(freq, sp.real, 'b', freq, sp.imag,'g')
#plt.show()
fig.savefig(filepath,bbox_inches='tight')
import ipdb; ipdb.set_trace()
###########################################

##########################################
#Define signal parameters
T=1.0/800.0
N=800
f1=80
f2=50
a1=1
a2=5
theta1=-np.pi/2
theta2=-np.pi/2
#time=np.arange(0,4*np.pi,.1)
#time=np.arange(0,.03,1e-5)
time=np.linspace(0.0,N*T, N)
#time=np.arange(100)
##########################################

##########################################
#Generate test signals
signal='sinusoid'
if signal=='sinusoid'or signal=='mix':
    #Define signals
    sig1=[]
    sig2=[]
    sig3=[] 
    sig4=[] 
    sig9=[]
    sig10=[]

    i=0
    j=complex(0,1)

    for t in time:
#        sig1.append(a1*np.cos(2*np.pi*t+theta1))
#        sig2.append(a2*np.cos(2*np.pi*t+theta2))
        sig1.append(a1*np.cos(2*np.pi*f1*t+theta1))
        sig2.append(a2*np.cos(2*np.pi*f2*t+theta2))
        sig3.append(sig1[i]+sig2[i])
#        sig4.append(a1*np.exp((0+j*2*np.pi*f1*t))+a2*np.exp((0+j*2*np.pi*f2*t)))
        sig4.append(a1*np.exp((0+j*2*np.pi*f1*t)))
#        sig4.append(sig1[i]*sig3[i])
#        sig9.append(np.cos(2*np.pi*f2*t))
#        sig10.append(sig1[i]*sig9[i])
        i=i+1

#    Uncomment to use exp code
    mysig=np.array(sig4)
    sig_gen=np.ones((len(mysig),2))
    sig_gen[:,0]=mysig.real
    sig_gen[:,1]=mysig.imag

#    sig=np.linspace(0.0,1.0,sig_gen.shape[-2])
#    sig=np.zeros(sig_gen.shape[-2])
    sig=[]
    i=0

    while i<sig_gen.shape[-2]:
        sig.append(np.complex(sig_gen[i,0],sig_gen[i,1]))
        i=i+1

    sig=np.array(sig)

    sig=np.array(sig1)

    

if signal=='square' or signal=='mix':
    sqr=np.ones(100)
    fr=9
    ind=10
    while ind<len(sqr):
        sqr[ind:ind+fr]=0
        ind=ind+20
    if signal=='square':
        sig=sqr
        time=np.arange(len(sig))

if signal=='mix':
    i=0
    mix=[]
     
    while i<len(sig) and i<len(sqr):
        mix.append(sqr[i]*sig[i])
        i=i+1

    sig=np.array(mix)
    time=np.arange(len(sig))
##########################################

##########################################
#Preform FFT
#t = np.arange(256)
spec = np.fft.fft(sig)
freq = np.fft.fftfreq(time.shape[-1])
#freq = np.fft.fftfreq(sig.shape[-1])
import ipdb; ipdb.set_trace()

amp=np.abs(spec)
phase=np.angle(spec)
pwr=np.abs(spec)**2
import ipdb; ipdb.set_trace()
##########################################

##########################################
#Plot Signal and FFT output
#fig = plt.figure(figsize=(8,4))
#fig = plt.figure()
#fig, ((ax0), (ax1), (ax2), (ax3), (ax4))=plt.subplots(5,1,sharex=False, sharey=False)
#fig, ((ax0), (ax1), (ax2), (ax3))=plt.subplots(4,1,sharex=False, sharey=False)
ny_plots=5
nx_plots=1
xsize=8
ysize=2
fig         = plt.figure(figsize=(nx_plots*xsize,ny_plots*ysize)) # Create figure with the appropriate size.
ax0     = fig.add_subplot(ny_plots,nx_plots,1)
ax1     = fig.add_subplot(ny_plots,nx_plots,2)
ax2     = fig.add_subplot(ny_plots,nx_plots,3)
ax3     = fig.add_subplot(ny_plots,nx_plots,4)
ax4     = fig.add_subplot(ny_plots,nx_plots,5)


#ax0.plot(time,sig1,'-m', time, sig2, '-b',time, sig3, '-r',time, sig4,'-g')
ax0.plot(time,sig,'-m')
#ax0.plot(time,sig,'-m', time, sig1, 'b', time, sig2, 'r')
ax1.plot(freq, spec.real, 'b', freq, spec.imag,'g')
ax2.plot(freq, amp, 'r')
ax3.plot(freq, phase,'c')
ax4.plot(freq, pwr, 'orange')

ax0.set_title('Input Signal')
ax1.set_title('FFT Output')
ax2.set_title('Amplitude')
ax3.set_title('Phase')
ax4.set_title('Power')

filepath    = os.path.join(output_path,filename2)
fig.savefig(filepath,bbox_inches='tight')
import ipdb; ipdb.set_trace()
###############################################
