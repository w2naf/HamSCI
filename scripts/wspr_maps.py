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


if __name__ == '__main__':
    multiproc   = False 
    sTime = datetime.datetime(2016,11,1)
    wspr_obj = wspr_lib.WsprObject(sTime) 
    wspr_obj.active.dxde_gs_latlon()
#    wspr_obj.active.

    map_sTime = sTime
    map_eTime = map_sTime + datetime.timedelta(minutes = 15)
    wspr_map = wspr_lib.WsprMap(wspr_obj, sTime = map_sTime, eTime = map_eTime)

#    sTime = datetime.datetime(2016,11,1)
#    wspr_obj = wspr_lib.WsprObject(sTime) 
#    wspr_obj.active.calc_reflection_points(reflection_type='miller2015')
#    #For iPython
##    os.system('sudo python setup.py install')
