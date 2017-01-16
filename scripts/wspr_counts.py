#!/usr/bin/env python
#Code for downloading and making pickle files of WSPR data

import sys
import os
import datetime

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
