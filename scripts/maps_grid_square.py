#!/usr/bin/env python
#This code is used to generate the RBN-GOES map for the Space Weather feature article.

import sys
import os
import datetime
import multiprocessing

import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt

import numpy as np
import pandas as pd

from hamsci import rbn_lib
from hamsci import handling

def loop_info(map_sTime,map_eTime):
    print ''
    print '################################################################################'
    print 'Plotting RBN Map: {0} - {1}'.format(map_sTime.strftime('%d %b %Y %H%M UT'),map_eTime.strftime('%d %b %Y %H%M UT'))

    
def geoloc_info(rbn_obj):
    # Figure out how many records properly geolocated.
    good_loc        = rbn_obj.DS001_dropna.df
    good_count      = good_loc['callsign'].count()
    total_count     = len(rbn_obj.DS000.df)
    good_pct        = float(good_count) / total_count * 100.
    print 'Geolocation success: {0:d}/{1:d} ({2:.1f}%)'.format(good_count,total_count,good_pct)

    return {'good_count':good_count,'total_count':total_count}

def rbn_map(sTime,eTime,
        llcrnrlon=-180., llcrnrlat=-90, urcrnrlon=180., urcrnrlat=90.,
        call_filt_de = None, call_filt_dx = None,
        plot_de                 = True,
        plot_midpoints          = True,
        plot_paths              = False,
        plot_ncdxf              = False,
        plot_stats              = True,
        plot_legend             = True,
        overlay_gridsquares     = True,
        overlay_gridsquare_data = True,
        gridsquare_data_param   = 'f_max_MHz',
        fname_tag               = None,
        output_dir              = 'output'):
    """
    Creates a nicely formated RBN data map.
    """

    latlon_bnds = {'llcrnrlat':llcrnrlat,'llcrnrlon':llcrnrlon,'urcrnrlat':urcrnrlat,'urcrnrlon':urcrnrlon}

    if fname_tag is None:
        fname_tag = gridsquare_data_param
    filename    = '{}-{:%Y%m%d.%H%M}-{:%Y%m%d.%H%M}.png'.format(fname_tag,sTime,eTime)
    output_dir  = os.path.join(output_dir,fname_tag)
    filepath    = os.path.join(output_dir,filename)
    handling.prepare_output_dirs({0:output_dir},clear_output_dirs=False)

    li          = loop_info(sTime,eTime)

    t0 = datetime.datetime.now()
    rbn_obj     = rbn_lib.RbnObject(sTime,eTime,gridsquare_precision=4)

    rbn_obj.active.latlon_filt(**latlon_bnds)
    rbn_obj.active.filter_calls(call_filt_de,call_type='de')
    rbn_obj.active.filter_calls(call_filt_dx,call_type='dx')

    rbn_obj.active.compute_grid_stats()

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

    rbn_map_obj= rbn_lib.RbnMap(rbn_obj,ax=ax0,default_plot=False)
    if plot_de:
        rbn_map_obj.plot_de()
    if plot_midpoints:
        rbn_map_obj.plot_midpoints()
    if plot_paths:
        rbn_map_obj.plot_paths()
    if plot_ncdxf:
        rbn_map_obj.plot_ncdxf()
    if plot_stats:
        rbn_map_obj.plot_link_stats()
    if plot_legend:
        rbn_map_obj.plot_band_legend(band_data=rbn_map_obj.band_data)
    if overlay_gridsquares:
        rbn_map_obj.overlay_gridsquares()
    if overlay_gridsquare_data:
        rbn_map_obj.overlay_gridsquare_data(param=gridsquare_data_param)

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

def rbn_map_dct_wrapper(run_dct):
    rbn_map(**run_dct)

def rbn_map_multiview(run_dct):
    run_dct['plot_de']                  = True
    run_dct['plot_ncdxf']               = False
    run_dct['plot_stats']               = True
    run_dct['plot_legend']              = True
    run_dct['plot_paths']               = False
    run_dct['overlay_gridsquares']      = True

    run_dct['plot_midpoints']           = True
    run_dct['overlay_gridsquare_data']  = False
    run_dct['fname_tag']                = 'midpoints'
    rbn_map(**run_dct)

    run_dct['plot_midpoints']           = False
    run_dct['overlay_gridsquare_data']  = True
    run_dct['gridsquare_data_param']    = 'f_max_MHz'
    run_dct['fname_tag']                = None
    rbn_map(**run_dct)

    run_dct['plot_midpoints']           = False
    run_dct['overlay_gridsquare_data']  = True
    run_dct['gridsquare_data_param']    = 'counts'
    run_dct['fname_tag']                = None
    rbn_map(**run_dct)

if __name__ == '__main__':
    multiproc   = False 

#    # 2014 Nov Sweepstakes
#    sTime   = datetime.datetime(2014,11,1)
#    eTime   = datetime.datetime(2014,11,4)
    sTime   = datetime.datetime(2014,11,1,23)
    eTime   = datetime.datetime(2014,11,2)

    # 2015 Nov Sweepstakes
#    sTime   = datetime.datetime(2015,11,7)
#    eTime   = datetime.datetime(2015,11,10)

#    # 2016 CQ WPX CW
#    sTime   = datetime.datetime(2016,5,28)
#    eTime   = datetime.datetime(2016,5,29)

    dct = {}
    dct.update({'llcrnrlat':20.,'llcrnrlon':-130.,'urcrnrlat':55.,'urcrnrlon':-65.})

    integration_time    = datetime.timedelta(minutes=15)
    interval_time       = datetime.timedelta(minutes=60)

    event_dir           = '{:%Y%m%d.%H%M}-{:%Y%m%d.%H%M}'.format(sTime,eTime)
    output_dir          = os.path.join('output','maps',event_dir)
    handling.prepare_output_dirs({0:output_dir},clear_output_dirs=True)
    dct['output_dir']   = output_dir
#    dct['call_filt_de'] = 'aa4vv'

    run_list            = gen_map_run_list(sTime,eTime,integration_time,interval_time,**dct)

    if multiproc:
        pool = multiprocessing.Pool()
        pool.map(rbn_map_dct_wrapper,run_list)
        pool.close()
        pool.join()
    else:
        for run_dct in run_list:
#            rbn_map_dct_wrapper(run_dct)
            rbn_map_multiview(run_dct)


