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

import rbn_lib
import handling

def gen_time_list(sTime,eTime,interval_time):
    """ Generate a list of datetime.datetimes spaced interval_time apart.  """
    cTimes  = [sTime]
    while cTimes[-1] < eTime:
        next_time = cTimes[-1]+interval_time
        if next_time >= eTime:
            break
        cTimes.append(next_time)

    return cTimes

def loop_info(map_sTime,map_eTime):
    print ''
    print '################################################################################'
    print 'Plotting RBN Map: {0} - {1}'.format(map_sTime.strftime('%d %b %Y %H%M UT'),map_eTime.strftime('%d %b %Y %H%M UT'))

    
def geoloc_info(rbn_obj):
    # Figure out how many records properly geolocated.
    good_loc        = rbn_obj.DS001_dropna.df
    good_count  = good_loc['callsign'].count()
    total_count = len(rbn_obj.DS000.df)
    good_pct    = float(good_count) / total_count * 100.
    print 'Geolocation success: {0:d}/{1:d} ({2:.1f}%)'.format(good_count,total_count,good_pct)

    return {'good_count':good_count,'total_count':total_count}

def rbn_map(sTime,eTime,
        llcrnrlon=-180., llcrnrlat=-90, urcrnrlon=180., urcrnrlat=90.,
        call_filt_de = None, call_filt_dx = None,
        output_dir = 'output'):

    latlon_bnds = {'llcrnrlat':llcrnrlat,'llcrnrlon':llcrnrlon,'urcrnrlat':urcrnrlat,'urcrnrlon':urcrnrlon}

    filename    = 'rbn_map-{:%Y%m%d.%H%M}-{:%Y%m%d.%H%M}.png'.format(sTime,eTime)
    filepath    = os.path.join(output_dir,filename)

    li          = loop_info(sTime,eTime)

    rbn_obj    = rbn_lib.RbnObject(sTime,eTime)
    rbn_obj.active.latlon_filt(**latlon_bnds)
    rbn_obj.active.filter_calls(call_filt_de,call_type='de')
    rbn_obj.active.filter_calls(call_filt_dx,call_type='dx')

    gli        = geoloc_info(rbn_obj)

    rbn_grid   = rbn_obj.active.create_geo_grid()
#    rbn_grid.grid_mean()

    # Go plot!! ############################ 
    ## Determine the aspect ratio of subplot.
    xsize       = 15.0
    ysize       = 6.5
    nx_plots    = 1
    ny_plots    = 1

    rcp = mpl.rcParams
    rcp['axes.titlesize']     = 'large'
    rcp['axes.titleweight']   = 'bold'

    fig        = plt.figure(figsize=(nx_plots*xsize,ny_plots*ysize))
    ax0        = fig.add_subplot(1,1,1)
    rbn_map    = rbn_lib.RbnMap(rbn_obj,ax=ax0)

    rbn_map.overlay_grid(rbn_grid)
#   rbn_map.overlay_grid_data(rbn_grid)

    fig.savefig(filepath,bbox_inches='tight')
    plt.close(fig)

def gen_map_run_list(sTime,eTime,integration_time,interval_time,**kw_args):
    dct_list    = []
    this_sTime  = sTime
    while this_sTime+integration_time < eTime:
        this_eTime   = this_sTime + integration_time

        tmp = {}
        tmp['sTime']    = this_sTime
        tmp['eTime']    = this_eTime
        tmp.update(kw_args)
        dct_list.append(tmp)

        this_sTime      = this_sTime + interval_time

    return dct_list


if __name__ == '__main__':
#    # 2014 Nov Sweepstakes
#    sTime   = datetime.datetime(2014,11,1)
#    eTime   = datetime.datetime(2014,11,4)

    # 2015 Nov Sweepstakes
    sTime   = datetime.datetime(2015,11,9)
    eTime   = datetime.datetime(2015,11,10)

#    # 2016 CQ WPX CW
#    sTime   = datetime.datetime(2016,5,28)
#    eTime   = datetime.datetime(2016,5,29)

    dct = {}
    dct.update({'llcrnrlat':20.,'llcrnrlon':-130.,'urcrnrlat':55.,'urcrnrlon':-65.})

    integration_time    = datetime.timedelta(minutes=15)
    interval_time       = datetime.timedelta(minutes=60)

    event_dir           = '{:%Y%m%d.%H%M}-{:%Y%m%d.%H%M}'.format(sTime,eTime)
    output_dir          = os.path.join('output',event_dir)
    handling.prepare_output_dirs({0:output_dir},clear_output_dirs=True)
    dct['output_dir']   = output_dir

    run_list            = gen_map_run_list(sTime,eTime,integration_time,interval_time,**dct)

    for run_dct in run_list:
        rbn_map(**run_dct)
