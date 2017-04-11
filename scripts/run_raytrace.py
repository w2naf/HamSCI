#!/usr/bin/env python3
import os
import bz2
import pickle
import datetime
import multiprocessing

import hamsci
from hamsci import raytrace
from hamsci import raytrace_plot
from hamsci.general_lib import prepare_output_dirs as prep_dirs

def gen_run_list(sTime,eTime,interval_time,**kw_args):
    """
    Generate a list of dictionaries containing the parameters necessary to
    define and analyze event periods.

    Args:
        sTime:              datetime.datetime object
        eTime:              datetime.datetime object
        integration_time:   datetime.timedelta object
                            How much time to include in an analysis period.
        interval_time:      datetime.timedelta object
                            How much time between consecutive startimes.

    Returns:
        dct_list:           List of dictionaries.

    """
    dct_list    = []
    this_sTime  = sTime
    while this_sTime < eTime:
        tmp = {}
        tmp['date'] = this_sTime
        tmp.update(kw_args)
        dct_list.append(tmp)

        this_sTime      = this_sTime + interval_time

    return dct_list

def gen_plot_list(rt_objs,output_dir='output'):
    """
    Generate list of dictionaries containing the rt_objs along with file
    and path names.

    This is to set the plotting up for multiprocessing.
    """
    dct_list    = []
    for rt_obj in rt_objs:
        dct                 = {}
        dct['rt_obj']       = rt_obj
        dct['fname']        = rt_obj.get_event_name()
        dct['output_dir']   = output_dir
        dct_list.append(dct)

    return dct_list

def run_rt(run_dct):
    """
    Calls the ray tracing routine and manages cache files.
    """
#    event_fname = run_dct.pop('event_fname')
    pkl_dir     = run_dct.pop('pkl_dir')
    rt_dir      = run_dct.pop('output_dir')
    use_cache   = run_dct.pop('use_cache')

    date        = run_dct.get('date')
    freq        = run_dct.get('frequency')
    tx_lat      = run_dct.get('tx_lat')
    tx_lon      = run_dct.get('tx_lon')
    tx_call     = run_dct.get('tx_call')
    rx_lat      = run_dct.get('rx_lat')
    rx_lon      = run_dct.get('rx_lon')
    rx_call     = run_dct.get('rx_call')

    fname       = raytrace.get_event_name(date,None,freq,tx_call,rx_call,
                    tx_lat,tx_lon,rx_lat,rx_lon)

    pkl_fname   = os.path.join(pkl_dir,'{}.p.bz2'.format(fname))
    if not use_cache or not os.path.exists(pkl_fname):
        rt_obj  = raytrace.TxRxRaytracer(**run_dct)

        with bz2.BZ2File(pkl_fname,'w') as fl:
            pickle.dump(rt_obj,fl)
    else:
        with bz2.BZ2File(pkl_fname,'r') as fl:
            rt_obj      = pickle.load(fl)

    return rt_obj

def plot_raytrace_and_power(rt_plt):
    output_dir  = rt_plt.get('output_dir','output')
    fname       = rt_plt.get('fname','raytrace_and_power')
    rt_obj      = rt_plt.get('rt_obj')

    fpath       = os.path.join(output_dir,fname+'.png')
    fig         = hamsci.raytrace_plot.plot_raytrace_and_power(rt_obj,output_file=fpath)

    return fpath

if __name__ == '__main__':
    multiproc   = False
    # Generate a dictionary of parameters to send to MATLAB.
#    date    = datetime.datetime(2017,2,2,21,53)
#
#    tx_lat                  =   44.425  # W7WHO
#    tx_lon                  = -121.238  # W7WHO
#    rx_lat                  =   40.907  # Jenny Jump
#    rx_lon                  =  -74.926  # Jenny Jump

    sTime                   = datetime.datetime(2014,11,1)
    eTime                   = datetime.datetime(2014,11,2)
    interval_time           = datetime.timedelta(minutes=5)

#    sTime                   = datetime.datetime(2014,11,3,12)
#    eTime                   = datetime.datetime(2014,11,3,15)
#    interval_time           = datetime.timedelta(hours=1)
    freq                    =   14.110
    tx_call                 =   'YV5B'
    tx_lat                  =    9.096      # YV5B
    tx_lon                  =  -67.824      # YV5B
    rx_call                 =   'KM3T'
    rx_lat                  =   42.821      # KM3T
    rx_lon                  =  -71.606      # KM3T

    event_fname             = raytrace.get_event_name(sTime,eTime,freq,tx_call,rx_call)

    # Prepare output directory.
    base_dir        = os.path.join('output','raytrace',event_fname)
    rx_ts_dir       = base_dir
    rt_dir          = os.path.join(base_dir,'raytrace')
    prep_dirs({0:base_dir,1:rx_ts_dir,2:rt_dir},clear_output_dirs=True)
    pkl_dir         = 'data/raytrace'
    prep_dirs({0:pkl_dir},clear_output_dirs=False,php_viewers=False)

    run_dct = {}
    run_dct['tx_call']      = tx_call
    run_dct['tx_lat']       = tx_lat
    run_dct['tx_lon']       = tx_lon
    run_dct['rx_call']      = rx_call
    run_dct['rx_lat']       = rx_lat
    run_dct['rx_lon']       = rx_lon
    run_dct['frequency']    = freq
    run_dct['pkl_dir']      = pkl_dir
    run_dct['output_dir']   = rt_dir
    run_dct['use_cache']    = True

    run_lst                 = gen_run_list(sTime,eTime,interval_time,**run_dct)
    if multiproc:
        pool    = multiprocessing.Pool(2)
        rt_objs = pool.map(run_rt,run_lst)
        pool.close()
        pool.join()
    else:
        rt_objs = []
        for this_dct in run_lst:
            rt_obj  = run_rt(this_dct)
            rt_objs.append(rt_obj)

    plt_lst                 = gen_plot_list(rt_objs,output_dir=rt_dir)
    if multiproc:
        pool    = multiprocessing.Pool()
        pool.map(plot_raytrace_and_power,plt_lst)
        pool.close()
        pool.join()
    else:
        for rt_plt in plt_lst:
            plot_raytrace_and_power(rt_plt)

    fname       = 'rxPwr_{}.png'.format(event_fname)
    fpath       = os.path.join(base_dir,fname)
    raytrace_plot.plot_rx_power_timeseries(rt_objs,sTime,eTime,output_file=fpath)

    fname       = 'rxPwr_{}.csv'.format(event_fname)
    fpath       = os.path.join(base_dir,fname)
    raytrace.rt_rx_pwr_to_csv(rt_objs,output_file=fpath)

    import ipdb; ipdb.set_trace()
