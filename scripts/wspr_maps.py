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

#def __setup_map__(self,ax=None,llcrnrlon=-180.,llcrnrlat=-90,urcrnrlon=180.,urcrnrlat=90.,
#        coastline_color='0.65',coastline_zorder=10):
#    sTime       = self.metadata['sTime']
#    eTime       = self.metadata['eTime']
#
#    if ax is None:
#        fig     = plt.figure(figsize=(10,6))
#        ax      = fig.add_subplot(111)
#    else:
#        fig     = ax.get_figure()
#
#    m = Basemap(llcrnrlon=llcrnrlon,llcrnrlat=llcrnrlat,urcrnrlon=urcrnrlon,urcrnrlat=urcrnrlat,resolution='l',area_thresh=1000.,projection='cyl',ax=ax)
#
#    title = sTime.strftime('WSPR: %d %b %Y %H%M UT - ')+eTime.strftime('%d %b %Y %H%M UT')
##        fontdict = {'size':matplotlib.rcParams['axes.titlesize'],'weight':matplotlib.rcParams['axes.titleweight']}
#    fontdict = {'size':matplotlib.rcParams['axes.titlesize'],'weight':'bold'}
#    ax.text(0.5,1.075,title,fontdict=fontdict,transform=ax.transAxes,ha='center')
#
#    rft         = self.data_set.metadata.get('reflection_type')
#    if rft == 'sp_mid':
#        rft = 'Great Circle Midpoints'
#    elif rft == 'miller2015':
#        rft = 'Multihop'
#
#    subtitle    = 'Reflection Type: {}'.format(rft)
#    fontdict    = {'weight':'normal'}
#    ax.text(0.5,1.025,subtitle,fontdict=fontdict,transform=ax.transAxes,ha='center')
#
#    # draw parallels and meridians.
#    # This is now done in the gridsquare overlay section...
##        m.drawparallels(np.arange( -90., 91.,45.),color='k',labels=[False,True,True,False])
##        m.drawmeridians(np.arange(-180.,181.,45.),color='k',labels=[True,False,False,True])
#    m.drawcoastlines(color=coastline_color,zorder=coastline_zorder)
#    m.drawmapboundary(fill_color='w')
#
#    # Expose select object
#    self.fig        = fig
#    self.ax         = ax
#    self.m          = m
#
#def plot_paths(self,band_data=None):
#    m   = self.m
#    if band_data is None:
#        band_data = BandData()
#
#    band_list   = band_data.band_dict.keys()
#    band_list.sort(reverse=True)
#    for band in band_list:
#        this_group = self.data_set.get_band_group(band)
#        if this_group is None: continue
#
#        color = band_data.band_dict[band]['color']
#        label = band_data.band_dict[band]['name']
#
#        for index,row in this_group.iterrows():
#            #Yay stack overflow! - http://stackoverflow.com/questions/13888566/python-basemap-drawgreatcircle-function
#            de_lat  = row['de_lat']
#            de_lon  = row['de_lon']
#            dx_lat  = row['dx_lat']
#            dx_lon  = row['dx_lon']
#            line, = m.drawgreatcircle(dx_lon,dx_lat,de_lon,de_lat,color=color)
#
#            p = line.get_path()
#            # find the index which crosses the dateline (the delta is large)
#            cut_point = np.where(np.abs(np.diff(p.vertices[:, 0])) > 200)[0]
#            if cut_point:
#                cut_point = cut_point[0]
#
#                # create new vertices with a nan inbetween and set those as the path's vertices
#                new_verts = np.concatenate(
#                                           [p.vertices[:cut_point, :], 
#                                            [[np.nan, np.nan]], 
#                                            p.vertices[cut_point+1:, :]]
#                                           )
#                p.codes = None
#                p.vertices = new_verts

def wspr_path_map(sTime,eTime,
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
#        latlon_bnds.update({'lat_col': lat_col, 'lon_col':lon_col})
#        import ipdb; ipdb.set_trace()
#        wspr_obj.active.latlon_filt(**latlon_bnds)
    wspr_obj.active.filter_calls(call_filt_de,call_type='de')
    wspr_obj.active.filter_calls(call_filt_dx,call_type='dx')

#    gli         = geoloc_info(wspr_obj)

#    wspr_grid    = wspr_obj.active.create_geo_grid()

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

    wspr_map_obj.overlay_grid(wspr_grid)
    wspr_grid.grid_stat(stat='max',label='Max Frequency [MHz]')
    wspr_map_obj.overlay_grid_data(wspr_grid)

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

    wspr_map_obj.overlay_grid(wspr_grid)
    wspr_grid.grid_stat(stat='max',label='Max Frequency [MHz]')
    wspr_map_obj.overlay_grid_data(wspr_grid)

    fig.savefig(filepath,bbox_inches='tight')
    plt.close(fig)

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

def wspr_map_dct_wrapper(run_dct):
    wspr_map(**run_dct)

if __name__ == '__main__':
    multiproc   = False 
    sTime = datetime.datetime(2016,11,1,0)
    eTime = datetime.datetime(2016,11,1,1)
    term=[True, False]
    dt=15
    integration_time    = datetime.timedelta(minutes=15)
    interval_time       = datetime.timedelta(minutes=20)
#    interval_time       = datetime.timedelta(minutes=60)

    dct = {}
    dct.update({'llcrnrlat':20.,'llcrnrlon':-130.,'urcrnrlat':55.,'urcrnrlon':-65., 'filt_type':'sp_mid'})
#    dct.update({'llcrnrlat':20.,'llcrnrlon':-130.,'urcrnrlat':55.,'urcrnrlon':-65., 'output_dir': 'output/wspr'})

    map_sTime = sTime
#    map_eTime = map_sTime + datetime.timedelta(minutes = dt)
    map_eTime = map_sTime + interval_time

    run_list            = gen_map_run_list(map_sTime,map_eTime,integration_time,interval_time,**dct)
    if multiproc:
        pool = multiprocessing.Pool()
        pool.map(wspr_map_dct_wrapper,run_list)
        pool.close()
        pool.join()
    else:
        for run_dct in run_list:
            wspr_map_dct_wrapper(run_dct)

#    wspr_map.fig.savefig('output/wspr/WSPR_map_test2.png')

#    mymap = wspr_map(sTime = map_sTime, eTime = map_eTime)
#    wspr_obj = wspr_lib.WsprObject(sTime,eTime) 
#    wspr_obj.active.dxde_gs_latlon()
#
#    wspr_map = wspr_lib.WsprMap(wspr_obj, sTime = map_sTime, eTime = map_eTime, nightshade=term[0], solar_zenith=term[1])
#    wspr_map.fig.savefig('output/wspr/WSPR_map_test.png')

#    dct = {}
#    dct.update({'llcrnrlat':20.,'llcrnrlon':-130.,'urcrnrlat':55.,'urcrnrlon':-65.})
#
#    integration_time    = datetime.timedelta(minutes=15)
#    interval_time       = datetime.timedelta(minutes=60)
#
#    event_dir           = '{:%Y%m%d.%H%M}-{:%Y%m%d.%H%M}'.format(sTime,eTime)
#    output_dir          = os.path.join('output','maps',event_dir)

#    sTime = datetime.datetime(2016,11,1)
#    wspr_obj = wspr_lib.WsprObject(sTime) 
#    wspr_obj.active.calc_reflection_points(reflection_type='miller2015')
#    #For iPython
##    os.system('sudo python setup.py install')
