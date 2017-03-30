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

def get_fname(sTime,eTime=None,freq=None,tx_call=None,rx_call=None,
        tx_lat=None,tx_lon=None,rx_lat=None,rx_lon=None,):

    sTime_str   = sTime.strftime('%Y%m%d.%H%M')

    if eTime is not None:
        eTime_str   = eTime.strftime('-%Y%m%d.%H%M')
    else:
        eTime_str   = ''

    date_str    = sTime_str + eTime_str + 'UT'

    if freq is not None:
        freq_str    = '{:.0f}'.format(freq*1000)+'kHz'
    else:
        freq_str    = ''

    if tx_call is None:
        tx_s    = 'tx'+lat_lon_fname(tx_lat,tx_lon)
    else:
        tx_s    = 'tx'+tx_call

    if rx_call is None:
        rx_s    = 'rx'+lat_lon_fname(rx_lat,rx_lon)
    else:
        rx_s    = 'rx'+rx_call

    fname           = '_'.join([date_str,freq_str,tx_s,rx_s])
    return fname

def lat_lon_fname(lat,lon):
    """
    Returns a string in the form of 000.000N000.000E
    for a given lat/lon pair.
    """
    if lat < 0:
        NS  = 'S'
    else:
        NS  = 'N'
    lat_s   = '{:07.3f}{}'.format(abs(lat),NS)

    if lon > 180: lon = lon - 360.

    if lon < 0:
        EW  = 'W'
    else:
        EW  = 'E'
    lon_s   = '{:07.3f}{}'.format(abs(lon),EW)

    ret     = ''.join([lat_s,lon_s])
    return ret

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
        md              = rt_obj.rt_dct['metadata']
        fname           = get_fname(md['date'],None,md['freq'],
                            md['tx_call'], md['rx_call'],
                            md['tx_lat'],  md['tx_lon'],
                            md['rx_lat'],  md['rx_lon'])

        dct                 = {}
        dct['rt_obj']       = rt_obj
        dct['fname']        = fname
        dct['output_dir']   = output_dir
        dct_list.append(dct)

    return dct_list

def rt_rx_pwr_to_csv(rt_dcts,print_header=True,output_dir='output',fname=None):
    """
    Writes Ray Trace Dictionary RX Power to CSV.
    """

    # Prepare file names.
    if fname is None:
        fname       = 'rxPwr_'+rt_dcts[0].get('event_fname','0')

    csv_fname       = '{}.csv'.format(fname)
    csv_path        = os.path.join(output_dir,csv_fname)

    keys    = []
    keys.append('event_fname')
    keys.append('date')
    keys.append('freq')
    keys.append('tx_call')
    keys.append('tx_lat')
    keys.append('tx_lon')
    keys.append('rx_call')
    keys.append('rx_lat')
    keys.append('rx_lon')
    keys.append('rx_range')
    keys.append('azm')
    keys.append('gain_tx_db')
    keys.append('gain_xx_db')
    keys.append('tx_power')

#    keys.append('Doppler_shift') 
#    keys.append('Doppler_spread') 
#    keys.append('TEC_path') 
#    keys.append('apogee') 
#    keys.append('deviative_absorption') 
#    keys.append('effective_range') 
#    keys.append('final_elev') 
#    keys.append('frequency') 
#    keys.append('geometric_path_length') 
#    keys.append('gnd_rng_to_apogee') 
#    keys.append('ground_range') 
#    keys.append('group_range') 
#    keys.append('initial_elev') 
#    keys.append('lat') 
#    keys.append('lon') 
#    keys.append('nhops_attempted') 
#    keys.append('phase_path') 
#    keys.append('plasma_freq_at_apogee') 
#    keys.append('ray_id') 
#    keys.append('ray_label') 
#    keys.append('rx_power_0_dB') 
#    keys.append('rx_power_O_dB') 
#    keys.append('rx_power_X_dB') 
#    keys.append('rx_power_dB') 
#    keys.append('virtual_height') 
#    keys.append('power_dbw')

    with open(csv_path,'w') as fl:
        if print_header:
            fl.write('# PHaRLAP Predicted Receive Power')
            fl.write('#\n')
            fl.write('## Metadata ####################################################################\n')

            for key in keys:
                val     = rt_dcts[0].get(key)
                line    = '# {!s}: {!s}\n'.format(key,val)
                fl.write(line)

            fl.write('#\n')
            fl.write('## Parameter Legend ############################################################\n')
            fl.write('# rx_power_0_dB: No Losses\n')
            fl.write('# rx_power_dB: Ground and Deviative Losses\n')
            fl.write('# rx_power_O_dB: O Mode\n')
            fl.write('# rx_power_X_dB: X Mode\n')

            fl.write('#\n')
            fl.write('## CSV Data ####################################################################\n')

    rx_power    = extract_rx_power(rt_dcts)
    rx_power.to_csv(csv_path,mode='a')

    return csv_path

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

    fname       = get_fname(date,None,freq,tx_call,rx_call,
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

    event_fname             = get_fname(sTime,eTime,freq,tx_call,rx_call)

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
#    run_dct['event_fname']  = event_fname
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
            break

    plt_lst                 = gen_plot_list(rt_objs,output_dir=rt_dir)
    if multiproc:
        pool    = multiprocessing.Pool()
        pool.map(plot_raytrace_and_power,plt_lst)
        pool.close()
        pool.join()
    else:
        for rt_plt in plt_lst:
            plot_raytrace_and_power(rt_plt)

#    plot_rx_power_timeseries(rt_dcts,sTime,eTime,output_dir=rx_ts_dir)
#    rt_rx_pwr_to_csv(rt_dcts,output_dir=rx_ts_dir)
#
#    import ipdb; ipdb.set_trace()
