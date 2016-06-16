#!/usr/bin/env python
#This code is used to generate the RBN-GOES map for the Space Weather feature article.

import sys
import os
import datetime

import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt

import numpy as np
import pandas as pd

from hamsci import rbn_lib
from hamsci import handling

def rbn_counts(sTime,eTime,
#        llcrnrlon=-180., llcrnrlat=-90, urcrnrlon=180., urcrnrlat=90.,
        call_filt_de = None, call_filt_dx = None, integration_time=datetime.timedelta(minutes=15),
        output_dir = 'output'):

#    latlon_bnds = {'llcrnrlat':llcrnrlat,'llcrnrlon':llcrnrlon,'urcrnrlat':urcrnrlat,'urcrnrlon':urcrnrlon}

    filename    = 'rbn_counts-{:%Y%m%d.%H%M}-{:%Y%m%d.%H%M}.png'.format(sTime,eTime)
    filepath    = os.path.join(output_dir,filename)

    rbn_obj     = rbn_lib.RbnObject(sTime,eTime)
#    rbn_obj.active.latlon_filt(**latlon_bnds)
    rbn_obj.active.filter_calls(call_filt_de,call_type='de')
    rbn_obj.active.filter_calls(call_filt_dx,call_type='dx')

    rbn_grid   = rbn_obj.active.create_geo_grid()
#    rbn_grid.grid_mean()

    # Go plot!! ############################ 
    ## Determine the aspect ratio of subplot.
    xsize       = 15.0
    ysize       = 6.5
    nx_plots    = 1
    ny_plots    = 2

    rcp = mpl.rcParams
    rcp['axes.titlesize']       = 'large'
    rcp['axes.titleweight']     = 'bold'
    rcp['axes.labelweight']     = 'bold'
    rcp['font.weight']          = 'bold'

    fig     = plt.figure(figsize=(nx_plots*xsize,ny_plots*ysize))

    ax0     = fig.add_subplot(ny_plots,nx_plots,1)
    rbn_cnt = rbn_lib.RbnCounts(rbn_obj,ax=ax0,plot_by_band=False,plot_all=True,
                    integration_time=integration_time)
    ax0.set_ylim(0,25000)

    ax0     = fig.add_subplot(ny_plots,nx_plots,2)
    rbn_cnt = rbn_lib.RbnCounts(rbn_obj,ax=ax0,plot_by_band=True,plot_all=False,
                    integration_time=integration_time)
    ax0.set_ylim(0,14000)

    fig.tight_layout()
    fig.savefig(filepath,bbox_inches='tight')
    plt.close(fig)

if __name__ == '__main__':
#    # 2014 Nov Sweepstakes
    sTime   = datetime.datetime(2014,11,1)
    eTime   = datetime.datetime(2014,11,4)
#    eTime   = datetime.datetime(2014,11,2)

##    # 2015 Nov Sweepstakes
#    sTime   = datetime.datetime(2015,11,7)
##    eTime   = datetime.datetime(2015,11,8)
#    eTime   = datetime.datetime(2015,11,10)

#    # 2016 CQ WPX CW
#    sTime   = datetime.datetime(2016,5,28)
#    eTime   = datetime.datetime(2016,5,29)

    dct = {}
    dct.update({'llcrnrlat':20.,'llcrnrlon':-130.,'urcrnrlat':55.,'urcrnrlon':-65.})

#    event_dir           = '{:%Y%m%d.%H%M}-{:%Y%m%d.%H%M}'.format(sTime,eTime)
    output_dir          = os.path.join('output','counts')
    handling.prepare_output_dirs({0:output_dir},clear_output_dirs=False)
    dct['output_dir']   = output_dir

    rbn_counts(sTime,eTime,output_dir=output_dir)
