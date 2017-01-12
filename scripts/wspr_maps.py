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

from hamsci import wspr_lib
from hamsci import handling

def loop_info(map_sTime,map_eTime):
    print ''
    print '################################################################################'
    print 'Plotting WSPR Map: {0} - {1}'.format(map_sTime.strftime('%d %b %Y %H%M UT'),map_eTime.strftime('%d %b %Y %H%M UT'))

#def wspr_full_map(sTime,eTime,
#        llcrnrlon=-180., llcrnrlat=-90, urcrnrlon=180., urcrnrlat=90.,
#        call_filt_de = None, call_filt_dx = None,
#        reflection_type         = 'sp_mid',
#        plot_de                 = True,
#        plot_midpoints          = True,
#        plot_paths              = False,
#        plot_ncdxf              = False,
#        plot_stats              = True,
#        plot_legend             = True,
#        overlay_gridsquares     = True,
#        overlay_gridsquare_data = True,
#        gridsquare_data_param   = 'f_max_MHz',
#        fname_tag               = None,
#        output_dir              = 'output',
#        output_dir = 'output/wspr'):
#    """
#    Creates 
#    """

def wspr_path_map(sTime,eTime,
        filt_type='sp_mid',  llcrnrlon=-180., llcrnrlat=-90, urcrnrlon=180., urcrnrlat=90.,
        call_filt_de = None, call_filt_dx = None,
        output_dir = 'output/wspr'):

    filename    = 'wspr_map-{:%Y%m%d.%H%M}-{:%Y%m%d.%H%M}.png'.format(sTime,eTime)
    filepath    = os.path.join(output_dir,filename)

    wspr_obj = wspr_lib.WsprObject(sTime,eTime) 
    wspr_obj.active.dxde_gs_latlon()
    wspr_obj.active.calc_reflection_points(reflection_type=filt_type)
    latlon_bnds = {'llcrnrlat':llcrnrlat,'llcrnrlon':llcrnrlon,'urcrnrlat':urcrnrlat,'urcrnrlon':urcrnrlon}
    wspr_obj.active.latlon_filt(**latlon_bnds)

    # Go plot!! ############################ 
    ## Determine the aspect ratio of subplot.
    xsize       = 15.0
    ysize       = 6.5
    nx_plots    = 1
    ny_plots    = 1

    rcp = mpl.rcParams
    rcp['axes.titlesize']     = 'large'
#    rcp['axes.titleweight']   = 'bold'

    fig        = plt.figure(figsize=(nx_plots*xsize,ny_plots*ysize))
    ax0        = fig.add_subplot(1,1,1)

    map_obj=wspr_lib.WsprMap(wspr_obj, ax=ax0,nightshade=term[0], solar_zenith=term[1], other_plot='plot_paths', default_plot=False)
    map_obj.plot_link_stats()

    fig.savefig(filepath,bbox_inches='tight')
    plt.close(fig)

def wspr_mid_map(sTime,eTime,
        filt_type='sp_mid',  llcrnrlon=-180., llcrnrlat=-90, urcrnrlon=180., urcrnrlat=90.,
        call_filt_de = None, call_filt_dx = None,
        output_dir = 'output/wspr'):

    filename    = 'wspr_map-{:%Y%m%d.%H%M}-{:%Y%m%d.%H%M}.png'.format(sTime,eTime)
    filepath    = os.path.join(output_dir,filename)

    wspr_obj = wspr_lib.WsprObject(sTime,eTime) 
    wspr_obj.active.dxde_gs_latlon()
    wspr_obj.active.calc_reflection_points(reflection_type=filt_type)
    latlon_bnds = {'llcrnrlat':llcrnrlat,'llcrnrlon':llcrnrlon,'urcrnrlat':urcrnrlat,'urcrnrlon':urcrnrlon}
    wspr_obj.active.latlon_filt(**latlon_bnds)

    # Go plot!! ############################ 
    ## Determine the aspect ratio of subplot.
    xsize       = 15.0
    ysize       = 6.5
    nx_plots    = 1
    ny_plots    = 1

    rcp = mpl.rcParams
    rcp['axes.titlesize']     = 'large'
#    rcp['axes.titleweight']   = 'bold'

    fig        = plt.figure(figsize=(nx_plots*xsize,ny_plots*ysize))
    ax0        = fig.add_subplot(1,1,1)

    wspr_lib.WsprMap(wspr_obj, ax=ax0,nightshade=term[0], solar_zenith=term[1], other_plot='plot_mid', default_plot=False)

    fig.savefig(filepath,bbox_inches='tight')
    plt.close(fig)

def wspr_default_map(sTime,eTime,
        filt_type='sp_mid',  llcrnrlon=-180., llcrnrlat=-90, urcrnrlon=180., urcrnrlat=90.,
        call_filt_de = None, call_filt_dx = None,
        output_dir = 'output/wspr'):

    filename    = 'wspr_map-{:%Y%m%d.%H%M}-{:%Y%m%d.%H%M}.png'.format(sTime,eTime)
    filepath    = os.path.join(output_dir,filename)

    wspr_obj = wspr_lib.WsprObject(sTime,eTime) 
    wspr_obj.active.dxde_gs_latlon()
    wspr_obj.active.calc_reflection_points(reflection_type=filt_type)
    latlon_bnds = {'llcrnrlat':llcrnrlat,'llcrnrlon':llcrnrlon,'urcrnrlat':urcrnrlat,'urcrnrlon':urcrnrlon}
    wspr_obj.active.latlon_filt(**latlon_bnds)
    import ipdb; ipdb.set_trace()

    # Go plot!! ############################ 
    ## Determine the aspect ratio of subplot.
    xsize       = 15.0
    ysize       = 6.5
    nx_plots    = 1
    ny_plots    = 1

    rcp = mpl.rcParams
    rcp['axes.titlesize']     = 'large'
#    rcp['axes.titleweight']   = 'bold'

    fig        = plt.figure(figsize=(nx_plots*xsize,ny_plots*ysize))
    ax0        = fig.add_subplot(1,1,1)

    wspr_lib.WsprMap(wspr_obj, ax=ax0,nightshade=term[0], solar_zenith=term[1], default_plot=True)

    fig.savefig(filepath,bbox_inches='tight')
    plt.close(fig)

def wspr_map(sTime,eTime,
        filt_type='sp_mid',  llcrnrlon=-180., llcrnrlat=-90, urcrnrlon=180., urcrnrlat=90.,
        call_filt_de = None, call_filt_dx = None,
        output_dir = 'output'):

    latlon_bnds = {'llcrnrlat':llcrnrlat,'llcrnrlon':llcrnrlon,'urcrnrlat':urcrnrlat,'urcrnrlon':urcrnrlon}

    filename    = 'wspr_map-{:%Y%m%d.%H%M}-{:%Y%m%d.%H%M}.png'.format(sTime,eTime)
    filepath    = os.path.join(output_dir,filename)

    li          = loop_info(sTime,eTime)

    wspr_obj     = wspr_lib.WsprObject(sTime,eTime)
    wspr_obj.active.dxde_gs_latlon()
    if filt_type == 'sp_mid' or filt_type == 'miller2015':
        lat_col='refl_lat'
        lon_col='refl_lon'
        wspr_obj.active.calc_reflection_points(reflection_type=filt_type)

        latlon_bnds.update({'lat_col': lat_col, 'lon_col':lon_col})
        import ipdb; ipdb.set_trace()
        wspr_obj.active.latlon_filt(**latlon_bnds)
    elif filt_type == 'dx' or filt_type == 'de' or filt_type == 'dxde':
        if filt_type == 'dx' or filt_type == 'dxde':
            lat_col='dx_lat'
            lon_col='dx_lon'
            latlon_bnds.update({'lat_col': lat_col, 'lon_col':lon_col})
            wspr_obj.active.latlon_filt(**latlon_bnds)
        if filt_type =='de' or filt_type == 'dxde':
            lat_col='de_lat'
            lon_col='de_lon'
            latlon_bnds.update({'lat_col': lat_col, 'lon_col':lon_col})
            wspr_obj.active.latlon_filt(**latlon_bnds)




        latlon_bnds.update({'lat_col': lat_col, 'lon_col':lon_col})
        import ipdb; ipdb.set_trace()
        wspr_obj.active.latlon_filt(**latlon_bnds)


#    wspr_obj.active.filter_calls(call_filt_de,call_type='de')
#    wspr_obj.active.filter_calls(call_filt_dx,call_type='dx')

#    gli         = geoloc_info(wspr_obj)

    wspr_grid    = wspr_obj.active.create_geo_grid()

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
    wspr_map_obj= wspr_lib.WsprMap(wspr_obj,ax=ax0)

#    wspr_map_obj.overlay_grid(wspr_grid)
#    wspr_grid.grid_stat(stat='max',label='Max Frequency [MHz]')
#    wspr_map_obj.overlay_grid_data(wspr_grid)
    wspr_map_obj.overlay_gridsquares(wspr_obj)
#    wspr_grid.grid_stat(stat='max',label='Max Frequency [MHz]')
    wspr_map_obj.overlay_gridsquare_data(wspr_obj)

    fig.savefig(filepath,bbox_inches='tight')
    plt.close(fig)

def gen_map_run_list(sTime,eTime,integration_time,interval_time,**kw_args):
    dct_list    = []
    this_sTime  = sTime
    while this_sTime+integration_time <= eTime:
        this_eTime   = this_sTime + integration_time

        tmp = {}
        tmp['sTime']    = this_sTime
        tmp['eTime']    = this_eTime
        tmp.update(kw_args)
        dct_list.append(tmp)

#        this_sTime      = this_sTime + interval_time
        this_sTime  = this_eTime

    return dct_list

def wspr_map_dct_wrapper(run_dct):
    wspr_path_map(**run_dct)
#    wspr_mid_map(**run_dct)
#    wspr_default_map(**run_dct)
#    wspr_map(**run_dct)

if __name__ == '__main__':
    multiproc   = False 
#Initial WsprMap test code
    sTime = datetime.datetime(2016,5,13,15,5)
    eTime = datetime.datetime(2016,5,13,15,21)
    #Solar Flare Event
    sTime = datetime.datetime(2013,5,13,15,5)
    eTime = datetime.datetime(2013,5,13,17)
#    import inspect 
#    import mpl_toolkits
#    print inspect.getfile(mpl_toolkits)
#    import ipdb; ipdb.set_trace()

#    wspr_obj = wspr_lib.WsprObject(sTime,sTime+datetime.timedelta(minutes=15)) 
#    import ipdb; ipdb.set_trace()


    term=[True, False]
    dt=15
    integration_time    = datetime.timedelta(minutes=15)
    interval_time       = datetime.timedelta(minutes=60)

    dct = {}
    dct.update({'llcrnrlat':20.,'llcrnrlon':-130.,'urcrnrlat':55.,'urcrnrlon':-65.})

    filt_type='sp_mid'
#    filt_type='miller2015'

    map_sTime = sTime
#    map_eTime = map_sTime + datetime.timedelta(minutes = dt)
    map_eTime = map_sTime + interval_time

    event_dir           = '{:%Y%m%d.%H%M}-{:%Y%m%d.%H%M}'.format(sTime,eTime)
    output_dir          = os.path.join('output','wspr','maps','paths',event_dir)
#    output_dir          = os.path.join('output','wspr','maps','midpoints',event_dir)
#    output_dir          = os.path.join('output','wspr','maps','refl_points',event_dir)
#    output_dir          = os.path.join('output','wspr','maps','defaults_test','midpoints',event_dir)
#    output_dir          = os.path.join('output','wspr','maps','defaults_test','refl_points',event_dir)

    try:    # Create the output directory, but fail silently if it already exists
        os.makedirs(output_dir) 
    except:
        pass

    dct.update({'filt_type':filt_type})
    dct.update({'output_dir':output_dir})
    run_list    = gen_map_run_list(sTime,eTime,integration_time,interval_time,**dct)

    if multiproc:
        pool = multiprocessing.Pool()
        pool.map(wspr_map_dct_wrapper,run_list)
        pool.close()
        pool.join()
    else:
        for run_dct in run_list:
            import ipdb; ipdb.set_trace()
            wspr_map_dct_wrapper(run_dct)


#    wspr_path_map(map_sTime, map_eTime, filt_type = filt_type, output_dir=output_dir, **dct)


#    wspr_map.fig.savefig('output/wspr/WSPR_map_test.png')

#Adapted test code 
#    sTime = datetime.datetime(2016,11,1,0)
#    eTime = datetime.datetime(2016,11,1,1)
#    term=[True, False]
#    dt=15
#    integration_time    = datetime.timedelta(minutes=15)
#    interval_time       = datetime.timedelta(minutes=20)
##    interval_time       = datetime.timedelta(minutes=60)
#
#    dct = {}
#    dct.update({'llcrnrlat':20.,'llcrnrlon':-130.,'urcrnrlat':55.,'urcrnrlon':-65., 'filt_type':'sp_mid'})
##    dct.update({'llcrnrlat':20.,'llcrnrlon':-130.,'urcrnrlat':55.,'urcrnrlon':-65., 'output_dir': 'output/wspr'})
#
#    map_sTime = sTime
##    map_eTime = map_sTime + datetime.timedelta(minutes = dt)
#    map_eTime = map_sTime + interval_time
#
#    run_list            = gen_map_run_list(map_sTime,map_eTime,integration_time,interval_time,**dct)
#    if multiproc:
#        pool = multiprocessing.Pool()
#        pool.map(wspr_map_dct_wrapper,run_list)
#        pool.close()
#        pool.join()
#    else:
#        for run_dct in run_list:
#            wspr_map_dct_wrapper(run_dct)
#
##    wspr_map.fig.savefig('output/wspr/WSPR_map_test2.png')
#
##    mymap = wspr_map(sTime = map_sTime, eTime = map_eTime)

#Other?
#    dct = {}
#    dct.update({'llcrnrlat':20.,'llcrnrlon':-130.,'urcrnrlat':55.,'urcrnrlon':-65.})
#
#    integration_time    = datetime.timedelta(minutes=15)
#    interval_time       = datetime.timedelta(minutes=60)
#
#    event_dir           = '{:%Y%m%d.%H%M}-{:%Y%m%d.%H%M}'.format(sTime,eTime)
#    output_dir          = os.path.join('output','maps',event_dir)

#General Test Code
#    sTime = datetime.datetime(2016,11,1)
#    wspr_obj = wspr_lib.WsprObject(sTime) 
#    wspr_obj.active.calc_reflection_points(reflection_type='miller2015')
#    #For iPython
##    os.system('sudo python setup.py install')
