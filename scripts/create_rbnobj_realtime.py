#!/usr/bin/env python
#This code is used to generate the RBN-GOES map for the Space Weather feature article.

import sys
import os
import glob
import datetime
import pickle

import numpy as np
import pandas as pd

import hamsci
from hamsci import rbn_lib
from hamsci import handling

# Set default gridsquare precision
gridsquare_precision = 4

def read_rbn_realtime(sTime,eTime,data_dir='realtime/rbn',
        file_dt=datetime.timedelta(minutes=1)):
    """
    Create a RBN data frame using files pulled from the real-time
    RBN Stream.
    """

    # Generate the list of files.
    sT  = datetime.datetime(sTime.year,sTime.month,sTime.day,
                            sTime.hour,sTime.minute)
    eT  = datetime.datetime(eTime.year,eTime.month,eTime.day,
                            eTime.hour,eTime.minute)
    flist       = []
    curr_time   = sT
    while curr_time < eT:
        fn      = curr_time.strftime('%Y_%m_%d-%H_%M_*.csv')
        gl_path = os.path.join(data_dir,fn)
        fl      = glob.glob(gl_path)
        if len(fl) > 0: flist += fl

        curr_time += file_dt

    # Load all data into a dataframe.
    df_comp     = None
    names       = ['dx','callsign','dx_lat','dx_lon','de_lat','de_lon','freq','db','date']
    for fl in flist:
        df          = pd.read_csv(fl,parse_dates=[8],names=names,header=0)
        if df_comp is None:
            df_comp = df
        else:
            df_comp     = pd.concat([df_comp, df],ignore_index=True)

    df = df_comp

    # Calculate Total Great Circle Path Distance
    lat1, lon1          = df['de_lat'],df['de_lon']
    lat2, lon2          = df['dx_lat'],df['dx_lon']
    R_gc                = rbn_lib.Re*hamsci.geopack.greatCircleDist(lat2,lon1,lat2,lon2)
    df.loc[:,'R_gc']    = R_gc

    # Calculate Band
    df.loc[:,'band']        = np.array((np.floor(df['freq']/1000.)),dtype=np.int)

    return df

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

def get_rbn_obj_path(output_dir,sTime,eTime):
    filename    = '{:%Y%m%d.%H%M}-{:%Y%m%d.%H%M}-rbnlive_rbnobj.p'.format(sTime,eTime)
    filepath    = os.path.join(output_dir,filename)
    return filepath

def create_rbn_obj(sTime,eTime,
        llcrnrlon=-180., llcrnrlat=-90, urcrnrlon=180., urcrnrlat=90.,
        call_filt_de = None, call_filt_dx = None,
        reflection_type     = 'sp_mid',
        input_dir           = None,
        output_dir          = None,
        **kwargs):

    latlon_bnds = {'llcrnrlat':llcrnrlat,'llcrnrlon':llcrnrlon,'urcrnrlat':urcrnrlat,'urcrnrlon':urcrnrlon}

    filepath    = get_rbn_obj_path(output_dir,sTime,eTime)
    handling.prepare_output_dirs({0:output_dir},clear_output_dirs=False)

    df          = read_rbn_realtime(sTime,eTime,data_dir=input_dir)

    rbn_obj     = rbn_lib.RbnObject(df=df)

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

    fl = os.path.join(output_dir,"web_plot_data.json")
    print('Saving to {}'.format(fl))
    with open(fl, "w") as output:
        df = rbn_obj.active.grid_data
        df["color"] = rbn_obj.active.get_grid_data_color(encoding="hex")
        output.write(df.T.to_json())

if __name__ == '__main__':
    integration_time        = datetime.timedelta(minutes=15)
    now                     = datetime.datetime.now()
    print('Running create_rbnobj_realtime.py for {!s}'.format(now))
    eTime                   = datetime.datetime(now.year,now.month,now.day,
                                                now.hour,now.minute)
    sTime                   = eTime - integration_time

    reflection_type     = 'miller2015'
#    reflection_type     = 'sp_mid'

    base                    = '/home/hamsci-live/scripts'
    input_dir               = os.path.join(base,'realtime','rbn')
    output_dir              = os.path.join(base,'realtime','rbnobj')

    dct = {}
#    dct.update({'llcrnrlat':20.,'llcrnrlon':-130.,'urcrnrlat':55.,'urcrnrlon':-65.})
    dct['sTime']            = sTime
    dct['eTime']            = eTime
    dct['output_dir']       = output_dir
    dct['input_dir']        = input_dir
    dct['reflection_type']  = reflection_type
#    dct['call_filt_de'] = 'aa4vv'

    create_rbn_obj(**dct)
