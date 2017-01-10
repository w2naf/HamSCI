#!/usr/bin/env python
#This code is used to generate the RBN-GOES map for the Space Weather feature article.

import sys
import os
import glob
import datetime
import multiprocessing
import pickle

import numpy as np
import pandas as pd

import hamsci
from hamsci import rbn_lib
from hamsci import handling

# Set default gridsquare precision
gridsquare_precision = 4

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

    # Make Json here, rbn_object.active
#    with open(filepath,'wb') as fl:
#        pickle.dump(rbn_obj,fl)
    with open("web_plot_data.json", "w") as output:
        df = rbn_obj.active.grid_data
        df["color"] = rbn_obj.active.get_grid_data_color(encoding="hex")
        output.write(df.T.to_json())

if __name__ == '__main__':
    multiproc           = False
    create_rbn_objs     = True

    reflection_type     = 'miller2015'

#    # 2014 Nov Sweepstakes
#    sTime   = datetime.datetime(2014,11,1)
#   eTime   = datetime.datetime(2014,11,4)
    sTime   = datetime.datetime(2014,11,2,12)
    eTime   = datetime.datetime(2014,11,2,13)

    dct = {}
    dct.update({'llcrnrlat':20.,'llcrnrlon':-130.,'urcrnrlat':55.,'urcrnrlon':-65.})

    integration_time        = datetime.timedelta(minutes=15)
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

