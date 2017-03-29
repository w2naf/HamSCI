#!/usr/bin/env python3
import os
import datetime

import hamsci
from hamsci import raytrace
from hamsci.general_lib import prepare_output_dirs as prep_dirs

def run_rt(run_dct):
    return raytrace.TxRxRaytracer(**run_dct)

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


if __name__ == '__main__':
    multiproc   = False
    # Generate a dictionary of parameters to send to MATLAB.
#    date    = datetime.datetime(2017,2,2,21,53)
#
#    tx_lat                  =   44.425  # W7WHO
#    tx_lon                  = -121.238  # W7WHO
#    rx_lat                  =   40.907  # Jenny Jump
#    rx_lon                  =  -74.926  # Jenny Jump

    sTime                   = datetime.datetime(2014,11,3)
    eTime                   = datetime.datetime(2014,11,4)
    interval_time           = datetime.timedelta(minutes=5)

#    sTime                   = datetime.datetime(2014,11,3,12)
#    eTime                   = datetime.datetime(2014,11,3,15)
#    interval_time           = datetime.timedelta(hours=1)
    freq                    =   21.150
    tx_call                 =   'YV5B'
    tx_lat                  =    9.096      # YV5B
    tx_lon                  =  -67.824      # YV5B
    rx_call                 =   'KM3T'
    rx_lat                  =   42.821      # KM3T
    rx_lon                  =  -71.606      # KM3T

    event_fname             = raytrace.get_event_fname(sTime,eTime,freq,tx_call,rx_call)

    # Prepare output directory.
    base_dir        = os.path.join('output','raytrace',event_fname)
    rx_ts_dir       = base_dir
    rt_dir          = os.path.join(base_dir,'ray_trace')
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
#    run_dct['event_fname']  = event_fname
#    run_dct['pkl_dir']      = pkl_dir
#    run_dct['output_dir']   = rt_dir
#    run_dct['use_cache']    = False

    run_lst                 = gen_run_list(sTime,eTime,interval_time,**run_dct)

    if multiproc:
        pool    = multiprocessing.Pool(2)
        rt_dcts = pool.map(run_ray_trace,run_lst)
        pool.close()
        pool.join()
    else:
        rt_objs = []
        for this_dct in run_lst:
            rt_obj  = run_rt(this_dct)
            rt_objs.append(rt_obj)
            import ipdb; ipdb.set_trace()

#    if multiproc:
#        pool    = multiprocessing.Pool()
#        pool.map(plot_raytrace_and_power,rt_dcts)
#        pool.close()
#        pool.join()
#    else:
#        for rt_dct in rt_dcts:
#            plot_raytrace_and_power(rt_dct)
#
#    plot_rx_power_timeseries(rt_dcts,sTime,eTime,output_dir=rx_ts_dir)
#    rt_rx_pwr_to_csv(rt_dcts,output_dir=rx_ts_dir)
#
#    import ipdb; ipdb.set_trace()
