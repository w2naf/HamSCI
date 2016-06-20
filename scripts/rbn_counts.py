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
        llcrnrlon=-180., llcrnrlat=-90, urcrnrlon=180., urcrnrlat=90.,
        call_filt_de = None, call_filt_dx = None, integration_time=datetime.timedelta(minutes=15),
        output_dir = 'output'):

    latlon_bnds = {'llcrnrlat':llcrnrlat,'llcrnrlon':llcrnrlon,'urcrnrlat':urcrnrlat,'urcrnrlon':urcrnrlon}

    latlon_str  = 'Lat Range: {:.0f} to {:.0f} N; Lon Range: {:.0f} to {:.0f} E'.format(llcrnrlat,urcrnrlat,llcrnrlon,urcrnrlon) 

    filename    = 'rbn_counts-{:%Y%m%d.%H%M}-{:%Y%m%d.%H%M}.png'.format(sTime,eTime)
    filepath    = os.path.join(output_dir,filename)

    rbn_obj     = rbn_lib.RbnObject(sTime,eTime)

#    # Filter things East of 80 deg lon.
#    tf = rbn_obj.active.df['de_lon'] <= -80.
#    rbn_obj.active.df = rbn_obj.active.df[tf]

    rbn_obj.active.latlon_filt(**latlon_bnds)
    rbn_obj.active.filter_calls(call_filt_de,call_type='de')
    rbn_obj.active.filter_calls(call_filt_dx,call_type='dx')

    rbn_grid   = rbn_obj.active.create_geo_grid()
#    rbn_grid.grid_mean()

    # Go plot!! ############################ 
    ## Determine the aspect ratio of subplot.
    xsize       = 19.0
    ysize       = 6.5
    nx_plots    = 1
    ny_plots    = 2

    rcp = mpl.rcParams
    rcp['axes.titlesize']   = 'xx-large'
    rcp['axes.labelsize']   = 'xx-large'
    rcp['xtick.labelsize']  = 'xx-large'
    rcp['ytick.labelsize']  = 'xx-large'
    rcp['legend.fontsize']  = 'xx-large'
    rcp['legend.columnspacing'] = 1.8
    rcp['axes.titleweight'] = 'bold'
    rcp['axes.labelweight'] = 'bold'
    rcp['font.weight']      = 'bold'

    fig     = plt.figure(figsize=(nx_plots*xsize,ny_plots*ysize))

    ax0     = fig.add_subplot(ny_plots,nx_plots,1)
    rbn_obj.active.plot_spot_counts(sTime=sTime,eTime=eTime,
            integration_time=integration_time,
            plot_by_band=False,plot_all=True,
            ax=ax0)
    ax0.set_ylim(0,16000)
    ax0.set_xlabel('')
    
    title = []
    title.append('Reverse Beacon Network')
    title.append(latlon_str)
    ax0.set_title('\n'.join(title))

    ax0.grid(True)

    ax0     = fig.add_subplot(ny_plots,nx_plots,2)
    rbn_obj.active.plot_spot_counts(sTime=sTime,eTime=eTime,
            integration_time=integration_time,
            plot_by_band=True,plot_all=False,legend_lw=7,
            ax=ax0)
    ax0.set_ylim(0,10000)
    ax0.set_title('')
    ax0.grid(True)

    fig.tight_layout()
    fig.savefig(filepath,bbox_inches='tight')
    plt.close(fig)

if __name__ == '__main__':
##    # 2014 Nov Sweepstakes
    sTime   = datetime.datetime(2014,11,1)
    eTime   = datetime.datetime(2014,11,4)
#    eTime   = datetime.datetime(2014,11,2)

    # 2015 nov sweepstakes
#    sTime   = datetime.datetime(2015,11,7)
##    etime   = datetime.datetime(2015,11,8)
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

    rbn_counts(sTime,eTime,**dct)
