#!/usr/bin/env python
#This code is used to generate the RBN-GOES map for the Space Weather feature article.

import sys
import os
import glob
import datetime
import multiprocessing
import pickle

import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
import matplotlib.dates as mdates

import numpy as np
import pandas as pd

import hamsci
from hamsci import rbn_lib
from hamsci import handling

# Set default gridsquare precision
gridsquare_precision = 4

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
    run_dct['plot_paths']               = True
    run_dct['overlay_gridsquares']      = True

    run_dct['plot_midpoints']           = True
    run_dct['overlay_gridsquare_data']  = False
    param                               = 'paths'
    run_dct['fname_tag']                = '{:03d}-{}'.format(serial,param)
    rbn_map(**run_dct)
    serial  += 1

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
        run_dct['plot_legend']              = False
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
        output_dir              = 'output',
        rbn_fof2_dir            = 'data/rbn_fof2'):
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

    t0          = datetime.datetime.now()
    # Load in data.
    rbn_fof2_fp = get_rbn_obj_path(rbn_fof2_dir,reflection_type,sTime,eTime)
    with open(rbn_fof2_fp,'rb') as fl:
        rbn_obj = pickle.load(fl)
    
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

    rbn_map_obj= rbn_lib.RbnMap(rbn_obj,ax=ax0,
            coastline_color='0.25',coastline_zorder=100,
            default_plot=False)
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
    if plot_legend is True:
        rbn_map_obj.plot_band_legend(band_data=rbn_map_obj.band_data,rbn_rx=False)

    if overlay_gridsquares:
        rbn_map_obj.overlay_gridsquares()
    if overlay_gridsquare_data:
        rbn_map_obj.overlay_gridsquare_data(param=gridsquare_data_param)

    ecl         = hamsci.eclipse.Eclipse2017()
    line, lbl   = ecl.overlay_umbra(rbn_map_obj.m,color='k')
    handles     = [line]
    labels      = [lbl]

    fig_tmp     = plt.figure()
    ax_tmp      = fig_tmp.add_subplot(111)
    ax_tmp.set_visible(False)
    scat        = ax_tmp.scatter(0,0,s=50,**rbn_lib.de_prop)
    labels.append('RBN Receiver')
    handles.append(scat)
    
    leg = ax0.legend(handles,labels,loc='upper left',fontsize='small',scatterpoints=1)
    leg.set_zorder(100)

    fig.savefig(filepath,bbox_inches='tight')
    plt.close(fig)

def create_rbn_obj_dct_wrapper(run_dct):
    create_rbn_obj(**run_dct)

def get_rbn_obj_path(output_dir,reflection_type,sTime,eTime):
    filename    = '{}-{:%Y%m%d.%H%M}-{:%Y%m%d.%H%M}.p'.format(reflection_type,sTime,eTime)
    output_dir  = os.path.join(rbn_fof2_dir,reflection_type)
    filepath    = os.path.join(output_dir,filename)
    return filepath

def create_rbn_obj(sTime,eTime,
        llcrnrlon=-180., llcrnrlat=-90, urcrnrlon=180., urcrnrlat=90.,
        call_filt_de = None, call_filt_dx = None,
        reflection_type         = 'sp_mid',
        rbn_fof2_dir            = 'data/rbn_fof2',
        **kwargs):

    latlon_bnds = {'llcrnrlat':llcrnrlat,'llcrnrlon':llcrnrlon,'urcrnrlat':urcrnrlat,'urcrnrlon':urcrnrlon}

    filepath    = get_rbn_obj_path(rbn_fof2_dir,reflection_type,sTime,eTime)
    output_dir  = os.path.split(filepath)[0]
    handling.prepare_output_dirs({0:output_dir},clear_output_dirs=False)

    rbn_obj     = rbn_lib.RbnObject(sTime,eTime)

    rbn_obj.active.dropna()
    rbn_obj.active.filter_pathlength(500.)
    rbn_obj.active.calc_reflection_points(reflection_type)
    rbn_obj.active.grid_data(gridsquare_precision)
    rbn_obj.active.latlon_filt(**latlon_bnds)
    rbn_obj.active.filter_calls(call_filt_de,call_type='de')
    rbn_obj.active.filter_calls(call_filt_dx,call_type='dx')

    rbn_obj.active.compute_grid_stats()
    with open(filepath,'wb') as fl:
        pickle.dump(rbn_obj,fl)

def plot_grid_timeseries(run_list,
        lat                     =  37.9,    # Wallops Island VA
        lon                     = 284.5,    # Wallops Island VA
        data_set                = 'active',
        gridsquare_data_param   = 'foF2',
        clear_cache             = True,
        output_dir              = 'output',):
    """
    Creates a nicely formated RBN data map.
    """

    gs_param    = gridsquare_data_param
    grid_square = str(hamsci.gridsquare.latlon2gridsquare(lat,lon,gridsquare_precision))

    fname_tag   = gridsquare_data_param
    filename    = '{}-{:%Y%m%d.%H%M}-{:%Y%m%d.%H%M}.png'.format(
            '_'.join([grid_square,gridsquare_data_param]),sTime,eTime)
    output_dir  = os.path.join(output_dir,fname_tag)
    filepath    = os.path.join(output_dir,filename)
    handling.prepare_output_dirs({0:output_dir},clear_output_dirs=False)

    t0          = datetime.datetime.now()

    # Load in data.
    cache_dir       = os.path.join('data','cache')
    handling.prepare_output_dirs({0:cache_dir},clear_output_dirs=clear_cache)
    cache_file      = '{}-{:%Y%m%d.%H%M}-{:%Y%m%d.%H%M}.cache.p'.format(fname_tag,sTime,eTime)
    cache_fpath     = os.path.join(cache_dir,cache_file)

    if not os.path.exists(cache_fpath):
        data_list   = []
        for run_dct in run_list:
            rbn_fof2_dir    = run_dct.get('rbn_fof2_dir')
            reflection_type = run_dct.get('reflection_type')
            this_sTime      = run_dct.get('sTime')
            this_eTime      = run_dct.get('eTime')

            rbn_fof2_fp = get_rbn_obj_path(rbn_fof2_dir,reflection_type,this_sTime,this_eTime)
            with open(rbn_fof2_fp,'rb') as fl:
                rbn_obj = pickle.load(fl)

            ds              = getattr(rbn_obj,data_set)

            if grid_square in ds.grid_data.index:
                tmp                     = {}
                tmp['time_ut']          = (this_eTime-this_sTime)/2 + this_sTime
                tmp['rbn_'+gs_param]    = ds.grid_data.loc[grid_square,gs_param]
                tmp['rbn_'+gs_param+'_err_low'] = ds.grid_data.loc[grid_square,gs_param+'_err_low']
                tmp['rbn_'+gs_param+'_err_up']  = ds.grid_data.loc[grid_square,gs_param+'_err_up']
                data_list.append(tmp)

        df_ts   = pd.DataFrame(data_list)

        with open(cache_fpath,'wb') as fl:
            pickle.dump(df_ts,fl)
    else:
        with open(cache_fpath,'rb') as fl:
            df_ts   = pickle.load(fl)

    # Go plot!! ############################ 
    ## Determine the aspect ratio of subplot.
    xsize       = 15.0
    ysize       = 6.5
    nx_plots    = 1
    ny_plots    = 1

    rcp = mpl.rcParams
    rcp['axes.titlesize']       = 'x-large'
    rcp['axes.titleweight']     = 'bold'
    rcp['axes.labelsize']       = 'large'
    rcp['axes.labelweight']     = 'bold'

    fig        = plt.figure(figsize=(nx_plots*xsize,ny_plots*ysize))
    ax         = fig.add_subplot(1,1,1)

    # RBN foF2 #############################
    xvals       = df_ts['time_ut']
    yvals       = df_ts['rbn_'+gs_param]
    yerr_0      = df_ts['rbn_'+gs_param+'_err_low']
    yerr_1      = df_ts['rbn_'+gs_param+'_err_up']
    label       = 'RBN {!s} ({!s})'.format(gs_param,grid_square)
#    ax.plot(xvals,yvals,marker='o',label=label)
    ax.errorbar(xvals,yvals,yerr=[yerr_0,yerr_1],fmt='-o',label=label,zorder=10)

    # Wallops Island Ionosonde #############
    iono_path   = 'data/ionograms/wal_viper.txt'
    iono_df     = pd.read_csv(iono_path,skiprows=10,header=None,sep=None)

    new_lst     = []
    for inx,row in iono_df.iterrows():

        date_code   = ' '.join([row[0],row[1]])
        this_dt     = datetime.datetime.strptime(date_code,'%Y-%m-%d %H:%M')
        foF2        = float(row[4])

        tmp = {'date':this_dt,'foF2':foF2}
        new_lst.append(tmp)

    iono_df = pd.DataFrame(new_lst)

    xvals       = iono_df['date']
    yvals       = iono_df['foF2']
    label       = 'Wallops VIPER'
#    ax.plot(xvals,yvals,marker='o',label=label)
    ax.errorbar(xvals,yvals,yerr=0.1*yvals,fmt='-o',label=label)

    # Take care of some labeling. ##########
    ax.set_xlabel('Time [UT]')

    if gs_param == 'foF2':
        ylabel = 'foF2 [MHz]'
    ax.set_ylabel(ylabel)

    ax.legend(loc='upper right',fontsize='large')


    title   = '{:%d %b %Y %H:%M} - {:%d %b %Y %H:%M}'.format(sTime,eTime)
    ax.set_title(title)

    fmt = mdates.DateFormatter('%d %b\n%H:%M')
    ax.xaxis.set_major_formatter(fmt)
    
    for xtl in ax.xaxis.get_ticklabels():
        xtl.set_rotation(70)
#        xtl.set_verticalalignment('top')
        xtl.set_horizontalalignment('center')
        xtl.set_fontsize('large')
        xtl.set_fontweight('bold')

    for ytl in ax.yaxis.get_ticklabels():
        ytl.set_fontsize('large')
        ytl.set_fontweight('bold')

    fig.savefig(filepath,bbox_inches='tight')
    plt.close(fig)

if __name__ == '__main__':
    multiproc           = True
    create_rbn_objs     = False
    plot_maps           = True
    plot_foF2           = True
    clear_foF2_cache    = True

    reflection_type     = 'miller2015'
#    reflection_type     = 'sp_mid'

#    # 2014 Nov Sweepstakes
    sTime   = datetime.datetime(2014,11,1)
    eTime   = datetime.datetime(2014,11,4)
#    sTime   = datetime.datetime(2014,11,2,12)
#    eTime   = datetime.datetime(2014,11,2,13)

    # 2015 Nov Sweepstakes
#    sTime   = datetime.datetime(2015,11,7)
#    eTime   = datetime.datetime(2015,11,10)

#    # 2016 CQ WPX CW
#    sTime   = datetime.datetime(2016,5,28)
#    eTime   = datetime.datetime(2016,5,29)

    dct = {}
    dct.update({'llcrnrlat':20.,'llcrnrlon':-130.,'urcrnrlat':55.,'urcrnrlon':-65.})

    integration_time        = datetime.timedelta(minutes=15)
#    interval_time           = datetime.timedelta(minutes=60)
    interval_time           = datetime.timedelta(minutes=15)

    event_dir               = '{:%Y%m%d.%H%M}-{:%Y%m%d.%H%M}-{}'.format(sTime,eTime,reflection_type)
    output_dir              = os.path.join('output',event_dir)
    rbn_fof2_dir            = os.path.join('data','rbn_fof2',event_dir)

    dct['output_dir']       = output_dir
    dct['rbn_fof2_dir']     = rbn_fof2_dir
    dct['reflection_type']  = reflection_type
#    dct['call_filt_de'] = 'aa4vv'

    run_list            = gen_map_run_list(sTime,eTime,integration_time,interval_time,**dct)

    # Create RBN Object ###############################################
    if create_rbn_objs:
        handling.prepare_output_dirs({0:rbn_fof2_dir},clear_output_dirs=True)
        if multiproc:
            pool = multiprocessing.Pool()
            pool.map(create_rbn_obj_dct_wrapper,run_list)
            pool.close()
            pool.join()
        else:
            for run_dct in run_list:
                create_rbn_obj_dct_wrapper(run_dct)
 
    # Plot Maps ####################################################################
    if plot_maps:
        handling.prepare_output_dirs({0:output_dir},clear_output_dirs=True)
        if multiproc:
            pool = multiprocessing.Pool()
            pool.map(rbn_map_multiview,run_list)
            pool.close()
            pool.join()
        else:
            for run_dct in run_list:
                rbn_map_multiview(run_dct)

        create_webview(output_dir=output_dir)

        name    = '0001-fof2.html'
        tags    = ['foF2','f_max_MHz','theta','R_gc_min','counts','midpoints']
        create_webview(tags=tags,html_fname=name,output_dir=output_dir)

        name    = '0001-rgc.html'
        tags    = ['R_gc_min','R_gc_max','R_gc_mean']
        create_webview(tags=tags,html_fname=name,output_dir=output_dir)

    if plot_foF2:
        plot_grid_timeseries(run_list,output_dir=output_dir,clear_cache=clear_foF2_cache)
