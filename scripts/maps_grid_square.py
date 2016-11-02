#!/usr/bin/env python
#This code is used to generate the RBN-GOES map for the Space Weather feature article.

import sys
import os
import glob
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
        reflection_type         = 'sp_mid',
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
    rbn_obj     = rbn_lib.RbnObject(sTime,eTime,gridsquare_precision=4,reflection_type=reflection_type)

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
    serial  = 0

    run_dct['plot_de']                  = True
    run_dct['plot_ncdxf']               = False
    run_dct['plot_stats']               = True
    run_dct['plot_legend']              = True
    run_dct['plot_paths']               = False
    run_dct['overlay_gridsquares']      = True

    run_dct['plot_midpoints']           = True
    run_dct['overlay_gridsquare_data']  = False
    param                               = 'midpoints'
    run_dct['fname_tag']                = '{:03d}-{}'.format(serial,param)
    rbn_map(**run_dct)
    serial  += 1


    params  = ['foF2','f_max_MHz','theta','R_gc_min','R_gc_max','R_gc_mean','counts']
    for param in params:
        run_dct['plot_midpoints']           = False
        run_dct['overlay_gridsquare_data']  = True
        run_dct['gridsquare_data_param']    = param
        run_dct['fname_tag']                = '{:03d}-{}'.format(serial,param)
        rbn_map(**run_dct)
        serial  += 1

def create_webview(tags=None,html_fname='0001-multiview.html',
        output_dir='output',width=500):
    # Get the names of the directories in the output_dir.
    dirs    = os.walk(output_dir).next()[1] 
    dirs.sort()
    if tags is None:
        fname_tags  = dirs
    else:
        # Find the actual directory matching the user-specified tags.
        fname_tags  = []
        for tag in tags:
            matching = [s for s in dirs if tag in s]
            if matching != []:
                fname_tags.append(matching[0])

    # Identify all of the time slots we have some plots for.
    file_codes  = []
    for fname_tag in fname_tags:
        these_files = glob.glob(os.path.join(output_dir,fname_tag,'*.png'))
        length      = len(fname_tag)
        tmp = [os.path.basename(x)[length:] for x in these_files]
        file_codes  += tmp
    file_codes = np.unique(file_codes)

    # Create the HTML.
    html = []
    html.append('<html>')
    html.append(' <head>')
    html.append(' </head>')
    html.append(' <body>')
    html.append('   <table border=1>')

    # Add in the headers.
    html.append('     <tr>')
    for fname_tag in fname_tags:
        html.append('       <th>{}</th>'.format(fname_tag[4:]))
    html.append('     </tr>')

    # Create row for each date and insert img.
    for file_code in file_codes:
        html.append('     <tr>')
        for fname_tag in fname_tags:
            fpath = os.path.join(fname_tag,'{}{}'.format(fname_tag,file_code))
            txt = '<a href="'+fpath+'"><img src="'+fpath+'" width="'+'{:.0f}'.format(width)+'px" /></a>'
            html.append('       <td>{}</td>'.format(txt))
        html.append('     </tr>')


    html.append('   </table>')
    html.append(' </body>')
    html.append('</html>')

    # Write out html file.
    html_fpath  = os.path.join(output_dir,html_fname)
    with open(html_fpath,'w') as fl:
        fl.write('\n'.join(html))

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

    integration_time        = datetime.timedelta(minutes=15)
    interval_time           = datetime.timedelta(minutes=60)

    event_dir               = '{:%Y%m%d.%H%M}-{:%Y%m%d.%H%M}'.format(sTime,eTime)
    output_dir              = os.path.join('output','maps',event_dir)
    handling.prepare_output_dirs({0:output_dir},clear_output_dirs=True)

    dct['output_dir']       = output_dir
    dct['reflection_type']  = 'miller2015'
#    dct['call_filt_de'] = 'aa4vv'

    run_list            = gen_map_run_list(sTime,eTime,integration_time,interval_time,**dct)

    if multiproc:
        pool = multiprocessing.Pool()
#        pool.map(rbn_map_dct_wrapper,run_list)
        pool.map(rbn_map_multiview,run_list)
        pool.close()
        pool.join()
    else:
        for run_dct in run_list:
#            rbn_map_dct_wrapper(run_dct)
            rbn_map_multiview(run_dct)

    create_webview(output_dir=output_dir)

    name    = '0001-fof2.html'
    tags    = ['foF2','f_max_MHz','theta','R_gc_min','counts','midpoints']
    create_webview(tags=tags,html_fname=name,output_dir=output_dir)

    name    = '0001-rgc.html'
    tags    = ['R_gc_min','R_gc_max','R_gc_mean']
    create_webview(tags=tags,html_fname=name,output_dir=output_dir)
