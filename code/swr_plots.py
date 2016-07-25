#!/usr/bin/env python 
#plot SWR vs freq for a given antenna 

import sys
sys.path.append('/data/mypython')
import os

import matplotlib
#matplotlib.use('Agg')
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.markers as mmarkers

import numpy as np
import pandas as pd
def find_lim(df_a,df_b):
    if df_a.swr.min() < df_b.vswr.min():
        xmin=df_a.swr.min()
    else:
        xmin=df_b.vswr.min()
    if df_a.swr.max() < df_b.vswr.max():
        xmax=df_b.vswr.max()
    else:
        xmax=df_a.swr.max()

    return xmin, xmax
 
input_file = 'ocf_dipole_test.csv'
input_file2 = 'VSWR_earle.csv'
input_path = os.path.join('data','antenna',input_file)
input_path2 = os.path.join('data','antenna','earle_antenna',input_file2)
output_dir='output/ocf_dipole'
filename = os.path.join(output_dir,'ocf_dipole_vs_end_dipole_swr3.png')
#filename = os.path.join(output_dir,'complete_ocf_dipole_vs_end_dipole_swr.png')

#Get data
df = pd.read_csv(input_path,header=4)
df2 = pd.read_csv(input_path2, sep=' ',skipinitialspace=True, index_col=False)

#plot SWR
fig=plt.figure()
ax = fig.add_subplot(111)
#ax.plot(df.freq, df.swr,'b', df2.freq,df2.vswr,'r')
line1=ax.semilogy(df.freq, df.swr,'b', label='OCF') 
line2=ax.semilogy(df2.freq,df2.vswr,'r', label='End-Feed')
ax.legend()

##x_min,x_max=find_lim(df.freq(),df2.freq())
#y_min,y_max=find_lim(df,df2)
##ax.set_xlim(df.freq.min(), df.freq.max())
#ax.set_xlim(df.freq.min(), 30.0)
##ax.set_ylim(y_min,y_max)
#ax.set_ylim(1,10)
#ax.set_xlim(5, 30.0)

ax.set_title('OCF Dipole vs End Dipole')
ax.set_xlabel('Frequency (MHz)')
ax.set_ylabel('SWR')

fig.tight_layout()  #Some things get clipped if you don't do this.
fig.savefig(filename)
