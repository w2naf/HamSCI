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

from hamsci import rbn_lib
from hamsci import handling

import pickle

def get_rbn_obj(sTime,eTime,tmp_dir='tmp'):
    filename    = '{:%Y%m%d.%H%M}-{:%Y%m%d.%H%M}.p'.format(sTime,eTime)
    filepath    = os.path.join(tmp_dir,filename)

    handling.prepare_output_dirs({0:tmp_dir},clear_output_dirs=False)

    if not os.path.exists(filepath):
        rbn_obj = rbn_lib.RbnObject(sTime,eTime)
        del rbn_obj.DS000
        with open(filepath,'wb') as pkl:
            pickle.dump(rbn_obj,pkl)
    else:
        with open(filepath,'rb') as pkl:
            rbn_obj = pickle.load(pkl)

    return rbn_obj

def rbn_counts(call_filt_de = None, call_filt_dx = None, integration_time=datetime.timedelta(minutes=15),
        output_dir = 'output'):


    filename    = 'rbn_counts_compare.png'
#    filename    = 'rbn_counts_compare-{:%Y%m%d.%H%M}-{:%Y%m%d.%H%M}.png'.format(sTime,eTime)
    filepath    = os.path.join(output_dir,filename)

#    latlon_bnds = {'llcrnrlat':llcrnrlat,'llcrnrlon':llcrnrlon,'urcrnrlat':urcrnrlat,'urcrnrlon':urcrnrlon}
#    rbn_obj.active.latlon_filt(**latlon_bnds)
#    rbn_obj.active.filter_calls(call_filt_de,call_type='de')
#    rbn_obj.active.filter_calls(call_filt_dx,call_type='dx')

    # Go plot!! ############################ 
    ## Determine the aspect ratio of subplot.
    xsize       = 19.0
    ysize       = 6.5
    nx_plots    = 1
    ny_plots    = 2

    rcp = mpl.rcParams
    rcp['axes.titlesize']       = 'xx-large'
    rcp['axes.labelsize']       = 'xx-large'
    rcp['xtick.labelsize']      = 'xx-large'
    rcp['ytick.labelsize']      = 'xx-large'
    rcp['legend.fontsize']      = 'xx-large'
    rcp['legend.columnspacing'] = 1.8
    rcp['axes.titleweight']     = 'bold'
    rcp['axes.labelweight']     = 'bold'
    rcp['font.weight']          = 'bold'


    # Choose only selected bands.
    band_data   = rbn_lib.BandData()
    bands       = [ 3, 7]
#    bands       = [14, 21]

    new_bd      = {}
    for key,item in band_data.band_dict.iteritems():
        if key not in bands: continue
        new_bd[key] = item

    band_data.band_dict = new_bd

    pnl_list    = []
    tmp = {}
#    # 2014 Nov Sweepstakes
    tmp['sTime']    = datetime.datetime(2014,11,1)
    tmp['eTime']    = datetime.datetime(2014,11,2)
    tmp['eTime']    = datetime.datetime(2014,11,4)
    pnl_list.append(tmp)

    tmp = {}
#    # 2015 Nov Sweepstakes
    tmp['sTime']    = datetime.datetime(2015,11,7)
    tmp['eTime']    = datetime.datetime(2015,11,8)
    tmp['eTime']    = datetime.datetime(2015,11,10)
    pnl_list.append(tmp)

    fig     = plt.figure(figsize=(nx_plots*xsize,ny_plots*ysize))
    
    for plt_nr,panel in enumerate(pnl_list):
        sTime   = panel['sTime']
        eTime   = panel['eTime']

        rbn_obj = get_rbn_obj(sTime,eTime)
        ax0     = fig.add_subplot(ny_plots,nx_plots,plt_nr+1)
        rbn_obj.active.plot_spot_counts(sTime=sTime,eTime=eTime,
                integration_time=integration_time,
                plot_by_band=True,plot_all=False,legend_lw=7,
                band_data=band_data,ax=ax0)
        ax0.set_ylim(0,15000)
        if plt_nr == 0:
            ax0.set_xlabel('')
            ax0.set_title('Revserse Beacon Network')
        if plt_nr > 0:
            ax0.set_title('')

        ystr        = '{:%Y}'.format(sTime)
        fontdict    = {'size':'36','weight':'bold'}
        ax0.text(-0.130,0.5,ystr,fontdict=fontdict,va='center',transform=ax0.transAxes,
                rotation=90)
        ax0.grid(True)

    fig.tight_layout()
    fig.savefig(filepath,bbox_inches='tight')
    plt.close(fig)

if __name__ == '__main__':
    dct = {}
    dct.update({'llcrnrlat':20.,'llcrnrlon':-130.,'urcrnrlat':55.,'urcrnrlon':-65.})

    output_dir          = os.path.join('output','counts')
    handling.prepare_output_dirs({0:output_dir},clear_output_dirs=False)
    dct['output_dir']   = output_dir

    rbn_counts(output_dir=output_dir)
