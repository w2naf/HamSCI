#!/usr/bin/env python
#Code for downloading and making pickle files of WSPR data

import sys
import os
import datetime

import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt

import numpy as np
import pandas as pd

from hamsci import wspr_lib
from hamsci import handling
# Set default gridsquare precision
gridsquare_precision = 4

def wspr_counts(sTime=None,eTime=None,
        llcrnrlon=-180., llcrnrlat=-90, urcrnrlon=180., urcrnrlat=90.,
        call_filt_de = None, call_filt_dx = None, integration_time=datetime.timedelta(minutes=15),
        reflection_type         = 'sp_mid',
        output_dir = 'output'):
#            plot_all        = True,     all_lw  = 2,
#            plot_by_band    = False,    band_lw = 3,
#            band_data=None,
#            plot_legend=True,legend_loc='upper left',legend_lw=None,
#            plot_title=True,format_xaxis=True,
#            xticks=None,
#            ax=None):
    """
    Make count plots of wspr data

    """

    latlon_bnds = {'llcrnrlat':llcrnrlat,'llcrnrlon':llcrnrlon,'urcrnrlat':urcrnrlat,'urcrnrlon':urcrnrlon}

    latlon_str  = 'Lat Range: {:.0f} to {:.0f} N; Lon Range: {:.0f} to {:.0f} E'.format(llcrnrlat,urcrnrlat,llcrnrlon,urcrnrlon) 

    filename    = 'wspr_counts-{:%Y%m%d.%H%M}-{:%Y%m%d.%H%M}.png'.format(sTime,eTime)
    filepath    = os.path.join(output_dir,filename)

    wspr_obj     = wspr_lib.WsprObject(sTime,eTime)
    import ipdb; ipdb.set_trace()
    #find lat/lon from gridsquares
    wspr_obj.active.dxde_gs_latlon()
    wspr_obj.active.filter_pathlength(500.)
    wspr_obj.active.calc_reflection_points(reflection_type=reflection_type)

    wspr_obj.active.latlon_filt(**latlon_bnds)
    wspr_obj.active.filter_calls(call_filt_de,call_type='de')
    wspr_obj.active.filter_calls(call_filt_dx,call_type='dx')

#    wspr_grid   = wspr_obj.active.create_geo_grid()
    wspr_obj.active.grid_data(gridsquare_precision)
    import ipdb; ipdb.set_trace()
    wspr_obj.active.compute_grid_stats()

    # Go plot!! ############################ 
    ## Determine the aspect ratio of subplot.
    xsize       = 19.0
    ysize       = 6.5
    nx_plots    = 1
    ny_plots    = 2

    rcp = mpl.rcParams
#    rcp['axes.titlesize']   = 'xx-large'
#    rcp['axes.labelsize']   = 'xx-large'
#    rcp['xtick.labelsize']  = 'xx-large'
#    rcp['ytick.labelsize']  = 'xx-large'
#    rcp['legend.fontsize']  = 'xx-large'
#    rcp['legend.columnspacing'] = 1.8
#    rcp['axes.titleweight'] = 'bold'
#    rcp['axes.labelweight'] = 'bold'
#    rcp['font.weight']      = 'bold'

    fig     = plt.figure(figsize=(nx_plots*xsize,ny_plots*ysize))

    ax0     = fig.add_subplot(ny_plots,nx_plots,1)
    wspr_obj.active.plot_spot_counts(sTime=sTime,eTime=eTime,
            integration_time=integration_time,
            plot_by_band=False,plot_all=True,
            ax=ax0)
#    ax0.set_ylim(0,16000)
#    ax0.set_xlabel('')
    
    title = []
    title.append(' WSPR Network ')
    title.append(latlon_str)
    ax0.set_title('\n'.join(title))

    ax0.grid(True)

    ax0     = fig.add_subplot(ny_plots,nx_plots,2)
    wspr_obj.active.plot_spot_counts(sTime=sTime,eTime=eTime,
            integration_time=integration_time,
            plot_by_band=True,plot_all=False,legend_lw=7,
            ax=ax0)
#    ax0.set_ylim(0,10000)
#    ax0.set_title('')
    ax0.grid(True)

    fig.tight_layout()
    fig.savefig(filepath,bbox_inches='tight')
    plt.close(fig)

def archive_wspr_data(sTime, eTime=None, pathlen=500., reflection_type='miller2015'):
    #Create wspr object
    wspr_obj = wspr_lib.WsprObject(sTime,eTime) 
    #find lat/lon from gridsquares
    wspr_obj.active.dxde_gs_latlon()
    #filter pathlegnth
    wspr_obj.active.filter_pathlength(500.)
    #Calculate reflection points of signals and apply lat/lon bounds
    wspr_obj.active.calc_reflection_points(reflection_type=reflection_type)
    latlon_bnds = {'llcrnrlat':llcrnrlat,'llcrnrlon':llcrnrlon,'urcrnrlat':urcrnrlat,'urcrnrlon':urcrnrlon}
    wspr_obj.active.latlon_filt(**latlon_bnds)

    #Calculate gridsquare data (f_max, fof2, etc.)
    wspr_obj.active.grid_data(gridsquare_precision)
    wspr_obj.active.compute_grid_stats()

if __name__ == '__main__':
    multiproc   = False 
    plot_de                 = True
    plot_dx                 = False
    plot_midpoints          = False
    plot_paths              = False
    plot_ncdxf              = False
    plot_stats              = True
    plot_legend             = False
    overlay_gridsquares     = True
    overlay_gridsquare_data = True
    gridsquare_data_param   = 'f_max_MHz'
#    gridsquare_data_param   = 'foF2'
    fname_tag               = None

    sTime = datetime.datetime(2014, 11,1)
    eTime = datetime.datetime(2014, 11,4)
    eTime = datetime.datetime(2014, 11,5)

    dct = {}
    dct.update({'llcrnrlat':20.,'llcrnrlon':-130.,'urcrnrlat':55.,'urcrnrlon':-65.})

#    event_dir           = '{:%Y%m%d.%H%M}-{:%Y%m%d.%H%M}'.format(sTime,eTime)
    output_dir          = os.path.join('output','wspr','counts')
    handling.prepare_output_dirs({0:output_dir},clear_output_dirs=False)
    dct['output_dir']   = output_dir

    wspr_counts(sTime,eTime,**dct)


