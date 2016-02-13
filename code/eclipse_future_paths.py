#!/usr/bin/env python
#This code is used to generate the RBN-GOES map for the Space Weather feature article.

import sys
sys.path.append('/data/mypython')
import os

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches

import numpy as np
import pandas as pd

from davitpy import gme
import datetime

import rbn_lib
import handling
import eclipse_lib

import rti_magda
from davitpy.pydarn.radar import *
from davitpy.pydarn.plotting import *
from davitpy.utils import *

from davitpy import pydarn
from pylab import gca
from matplotlib.patches import Polygon 

#Map Properties 
#define map projection 
mapProj='cyl'
llcrnrlon=-130 
llcrnrlat=25
urcrnrlon=-65
urcrnrlat=52 

x1=0.125
w1=0.775
h1=0.235
y1=0.6647
space=0.05
y2=y1+h1+space
map_ax=[[x1,y1, w1, h1], [x1, y2, w1, h1]]

#Define Eclipse Path limits
eLimits=['ds_NL.csv', 'ds_SL.csv']
#Define future eclipse path central line coordinates
eFuture=['ds_CL_2024.txt', 'ds_CL_2044.txt', 'ds_CL_2045.txt', 'ds_CL_2052.txt']
import ipdb; ipdb.set_trace()
input_path='website'

#Define visual 
#for 2017
eColor=(0.75,0.25,0.5)
path_color=['r', 'orange', 'g', 'b']
epath=['-r', '-o', '-g', '-b']
#For others

tick_font_size=9
#pZorder is the zorder of the eclipse path with higher zorder=on top
pZorder=9

#define RBN alpha
path_alpha=0.3

#Define SuperDARN radars want on the map 
#Note: Data for the RTI plot will be taken from beam of radars[0]
radars=['fhw', 'fhe','cvw','cve']
beam=7
#Define visual properties of Radars on the map 
fovColor=(0.5,0,0.75)
#fovZorder is the zorder of the FOV with higher zorder=on top
fovZorder=10

#Define Properties of RTI plots
#plot groundscatter in gray (True) or in color (False)
gs=False
#ax_dim=[0.125,0.099999,0.7, 0.22]
#ax_dim=[0.125,0.099999,0.77, 0.22]
ax_dim=[0.125,0.099999,0.78, 0.22]
#cax_x=0.125+ax_dim[2]+.005
#cax_x=0.125+ax_dim[2]+.075
cax_x=0.125+ax_dim[2]+.055
#cax_w=0.08
#cax_w=0.075
cax_w=0.025
#cax_w=(1-cax_x-0.01)/2
cax_dim=[cax_x, ax_dim[1],cax_w, ax_dim[3]]
print ax_dim
print cax_dim
#import ipdb; ipdb.set_trace()
#Specify start and end time
sTime = datetime.datetime(2013,5,12)
eTime = datetime.datetime(2013,5,14)
#sTime = datetime.datetime(2014,9,10)
#eTime = datetime.datetime(2014,9,11)
sat_nr = 15

#goes_data = gme.sat.read_goes(sTime,eTime,sat_nr)

##Look at the weekends only...
#inx = [x.weekday() in [5,6] for x in goes_data['xray'].index]
#goes_data['xray'] = goes_data['xray'][inx]

#flares = gme.sat.find_flares(goes_data,min_class='X2')

output_path = os.path.join('website','output','figures')
#handling.prepare_output_dirs({0:output_path},clear_output_dirs=True)
## Determine the aspect ratio of subplot.
xsize       = 8.0
#ysize       = 6.0
ysize       = 4.0
nx_plots    = 1
ny_plots    = 1


fig         = plt.figure(figsize=(nx_plots*xsize,ny_plots*ysize)) # Create figure with the appropriate size.
#    subplot_nr  = 0 # Counter for the subplot
#    letters = 'abcd'
#
good_count  = 0
total_count = 0

map_sTime=datetime.datetime(2013,5,13,15,5)
map_eTime = map_sTime + datetime.timedelta(minutes=15)

filename    = '2017_2052_eclipses.png'
filepath    = os.path.join(output_path,filename)

#        map_times = []
#        map_times.append(datetime.datetime(2013,5,13,15,5))
#        map_times.append(datetime.datetime(2013,5,13,16,5))

ax0     = fig.add_subplot(1,1,1)
#        for kk,map_sTime in enumerate(map_times):
#plt_inx = kk + 1
#ax0     = fig.add_subplot(3,1,plt_inx)
#            ax0     = fig.add_axes(map_ax[plt_inx])


#print ''
#print '################################################################################'
#print 'Plotting RBN Map: {0} - {1}'.format(map_sTime.strftime('%d %b %Y %H%M UT'),map_eTime.strftime('%d %b %Y %H%M UT'))
#
#rbn_df  = rbn_lib.read_rbn(map_sTime,map_eTime,data_dir='data/rbn')
#
## Figure out how many records properly geolocated.
#good_loc    = rbn_df.dropna(subset=['dx_lat','dx_lon','de_lat','de_lon'])
#good_count_map  = good_loc['callsign'].count()
#total_count_map = len(rbn_df)
#good_pct_map    = float(good_count_map) / total_count_map * 100.
#
#good_count      += good_count_map
#total_count     += total_count_map
#
#print 'Geolocation success: {0:d}/{1:d} ({2:.1f}%)'.format(good_count_map,total_count_map,good_pct_map)
#
## Go plot!!
#m,fig=rbn_lib.rbn_map_plot(rbn_df,legend=False,ax=ax0,tick_font_size=9,ncdxf=True,llcrnrlon=llcrnrlon, llcrnrlat=llcrnrlat, urcrnrlon=urcrnrlon, urcrnrlat=urcrnrlat, proj=mapProj, basemapType=False, eclipse=True,path_alpha=path_alpha)
##m,fig=rbn_lib.rbn_map_plot(rbn_df,legend=True,ax=ax0,tick_font_size=9,ncdxf=True,llcrnrlon=llcrnrlon, llcrnrlat=llcrnrlat, urcrnrlon=urcrnrlon, urcrnrlat=urcrnrlat, proj=mapProj, basemapType=False, eclipse=True,path_alpha=path_alpha)
##############################################################
#Get Figure (taken from rbn_lib code)
fig     = ax0.get_figure()

m = plotUtils.mapObj(llcrnrlon=llcrnrlon,llcrnrlat=llcrnrlat,urcrnrlon=urcrnrlon,urcrnrlat=urcrnrlat,resolution='l',area_thresh=1000.,projection=mapProj,ax=ax0,fillContinents='None', fix_aspect=True)

#title='Eclipses over the Continetal US from 2017-2052'
#ax.set_title(title)
#
# draw parallels and meridians.
#m.drawparallels(np.arange(-90.,91.,45.),color='k',labels=[False,True,True,False],fontsize=tick_font_size)
#m.drawmeridians(np.arange(-180.,181.,45.),color='k',labels=[True,False,False,True],fontsize=tick_font_size)
m.drawcoastlines(color='0.65')
m.drawmapboundary(fill_color='w')
#m.nightshade(plot_mTime,color='0.82')
#if plotting the 2017 eclipse map then also draw state boundaries
m.drawcountries(color='0.65')#np.arange(-90.,91.,45.),color='k',labels=[False,True,True,False],fontsize=tick_font_size)
m.drawstates(color='0.65')
############################################################


#Plot 2017 Eclipse path swath on map
m,fig=eclipse_lib.eclipse_swath(infile=eLimits,mapobj=m, fig=fig, pathColor=eColor, pZorder=pZorder, filetype='csv')
#Plot future eclipse central line paths on map
cindex=0
for coord in eFuture:
    eCL=os.path.join('website',coord)
#    pathColor=path_color[cindex]
    m,fig=eclipse_lib.eclipse_map_plot(infile=eCL,mapobj=m, fig=fig, pathColor=path_color[cindex], filetype='txt')
    cindex=cindex+1

#Set up legend
import ipdb; ipdb.set_trace()
handles=[]
labels=[]

elabels=['2024', '2044', '2045', '2052']

cindex=0
handles.append(mpatches.Patch(color=eColor,label='2017'))
labels.append('2017')

while cindex<len(elabels):
    handles.append(mpatches.Patch(color=path_color[cindex],label=elabels[cindex]))
    labels.append(elabels[cindex])
    cindex+=1

fig_tmp = plt.figure()
ax_tmp = fig_tmp.add_subplot(111)
ax_tmp.set_visible(False)
#if ncol is None:
ncol = len(labels)
loc='lower center'
#loc='lower right'
markerscale=0.5
prop={'size':10}
title=None
bbox_to_anchor=None
#import ipdb; ipdb.set_trace()
legend = fig.legend(handles,labels,ncol=ncol,loc=loc,markerscale=markerscale,prop=prop,title=title,bbox_to_anchor=bbox_to_anchor,scatterpoints=1)
#import ipdb; ipdb.set_trace()
#import ipdb; ipdb.set_trace()
ax = plt.gca().add_artist(legend)

##Plot SuperDARN Radars of interest on map (Fort Hayes West and Fort Hayes East on the map)
##for code in radars:
##            overlayRadar(m,fontSize=12,codes=radars,dateTime=map_sTime)
##            overlayFov(m, codes=radars, maxGate=40, beams=beam)
##First plot radar with the beam shown in the FOV which we will plot RTI data from later (radars[0])
#overlayRadar(m,fontSize=12,codes=radars[0],dateTime=map_sTime)
#overlayFov(m, codes=radars[0], maxGate=40, beams=beam,model='GS', fovColor=fovColor,zorder=fovZorder)
##Next plot the rest of the radars (no beams will be plotted)
#overlayRadar(m,fontSize=12,codes=radars[1:],dateTime=map_sTime)
#overlayFov(m, codes=radars[1:], maxGate=40, beams=None,model='GS', fovColor=fovColor,zorder=fovZorder)

#    x,y, w, h=ax0.get_position().bounds
#    print "ax0 map 1-(x,y,width, height)="
#    print x,y,w,h
##            import ipdb; ipdb.set_trace()
#    #If this is the first map plotted, then save the axis as ax1
#    if plt_inx==1:
#        ax1=ax0
##                w1=w
##                h1=h
##                ax0.set_position([x1,y1,w1,h1])
##            else: 
##                y2=0.05+y1+w1
##                w1=w
##                h1=h
##                ax0.set_position([x1,y2, w1, h1])
##
#        print ax0.get_position().bounds
#        print map_sTime
##            import ipdb; ipdb.set_trace()

#Titles and other propertites
#title = map_sTime.strftime('%H%M - ')+map_eTime.strftime('%H%M UT')
#title='2017 Solar Eclipse\nSuperDARN Radars and Reverse Beacon Network'
title='Map of All Eclipses over Continental US From 2017-2052'
#import ipdb; ipdb.set_trace()
ax0.set_title(title,loc='center')
#ax0.set_title(map_sTime.strftime('%d %b %Y'),loc='right')

#            if kk == 0:
#                ax0.set_title('Preflare',loc='left')
#            else:
#                ax0.set_title('Flare Peak',loc='left')
#
#            letter_prop = {'weight':'bold','size':20}
#            ax0.set_position(
#            ax0.text(.015,.90,'({0})'.format(letters[kk]),transform=ax0.transAxes,**letter_prop)

#            for item in (ax0.get_xticklabels() + ax0.get_yticklabels()):
#                item.set_fontsize(4)

#            print flare
#print map_sTime

#        ax      = fig.add_subplot(3,1,3)
    #ax.plot(inx,flare['B_AVG'],'o',label='{0} Class Flare @ {1}'.format(flare['class'],inx.strftime('%H%M UT')))
#        goes_map_sTime = datetime.datetime(inx.year,inx.month,inx.day)
#        goes_map_eTime = goes_map_sTime + datetime.timedelta(days=1)

#        goes_sTime = datetime.datetime(2013,5,13,16) - datetime.timedelta(hours=2)
#        goes_eTime = datetime.datetime(2013,5,13,16) + datetime.timedelta(hours=2)
#
#        #goes_data_map = gme.sat.read_goes(goes_sTime,goes_eTime,sat_nr=sat_nr)
#
#        #Plot RTI plots for radars[0] (Radar at FHW)
##        ax      = fig.add_axes([0.125,0.099999,0.775, 0.22]) 
#        ax      = fig.add_axes(ax_dim)
##        ax      =fig.add_subplot(3,1,3)
#        cax     =fig.add_axes(cax_dim)
#        rti_magda.plotRti(sTime=goes_sTime, eTime=goes_eTime, ax=ax, rad=radars[0], params=['power'],yrng=[0,40], gsct=gs, cax=cax, xtick_size=18,ytick_size=18)
#       # ax2      = fig.add_subplot(3,1,3)
#       # rti_magda.plotRti(sTime=goes_sTime, eTime=goes_eTime, ax=ax2, rad=radars[1], params=['power'])
#
#        #gme.sat.goes_plot(goes_data_map,ax=ax,legendLoc='lower right')

#shift=0.070
#shift=0.035
##        yl=y2-shift-0.025
##        leg = rbn_lib.band_legend(fig,loc='center',bbox_to_anchor=[0.48,yl],ncdxf=True,ncol=4)
#x0, y0, width, height = ax0.get_position().bounds
##y_pos=y0-shift-0.025
#y_pos=y0-shift
#leg = rbn_lib.band_legend(fig,loc='center',bbox_to_anchor=[0.45,y_pos],ncdxf=True,ncol=4)
##leg = rbn_lib.band_legend(fig,loc='center',bbox_to_anchor=[0.48,0.360],ncdxf=True)
#title_prop = {'weight':'bold','size':22}
#        fig.text(0.525,1.025,'HF Communication Paths',ha='center',**title_prop)
#fig.text(0.525,1.000,'2017 Solar Eclipse\nSuperDARN Radars and Reverse Beacon Network',ha='center',**title_prop)
#fig.text('2017 Solar Eclipse\nSuperDARN Radars and Reverse Beacon Network',ha='center',**title_prop)
#        fig.text(0.525,0.995,flare.name.strftime('%d %B %Y'),ha='center',size=18)

#x0, y0, width, height = ax0.get_position().bounds
#
#        ax_bounds = ax.get_position().bounds
#        print "ax0 map2-(x,y,width, height)=" 
##        print ax0_bounds
##        import ipdb; ipdb.set_trace()
#        
#        #Set position of Bottommost plot
##        import ipdb; ipdb.set_trace()
#        width   = width #0.80
#        x0      = x0 #(1.-width) / 2. + 0.050
##        y0      = .050
#        y0      = y0 #.080
#        height  = height #0.200
#        ax.set_position([x0,y0,width,height])
#
#        #Set postion of maps
#        #Top Map
#        x1, y1, width1, height1= ax1.get_position().bounds
#        width1=width1
#        height1=height1
#        x1=x1
#        #increase shift to shift map down; decrease shift to shift map up
#        shift=0.030
#        y1 =1-shift-height1
#        ax1.set_position([x1,y1,width1, height1])
#        #Bottom Map
#        x2, y2, width2, height2= ax0.get_position().bounds
#        width2=width2
#        height2=height2
#        x2=x2
#        #increase shift to shift map down; decrease shift to shift map up
#        shift=0.0350+0.005
#        y2 = y1-shift-height2
#        #1-0.025-height2
#        ax0.set_position([x2,y2,width2, height2])

#        #Set Legend position
##        shift=0.070
#        shift=0.070+0.005
#        #wl is the width of the legend (right now it is an educated guess)
#        wl=0.025
#        #the lower left y coordinate of the legend (yl) is equal to the lower y coordinate of the bottom map minus the estimated legend width and the shift
#        yl=y2-shift-0.025
#        leg = rbn_lib.band_legend(fig,loc='center',bbox_to_anchor=[0.48,yl],ncdxf=True,ncol=4)

    #Insert line between 
#        axl_h=0.0025
#        sh_axl_y=0.005#
#        sh_axl=(yl+y0+height)/2
#        axl_y=(yl+y0+height)/2
##        axl_y=y0+height+sh_axl_y
##        axl=fig.add_axes([x0,axl_y,width, axl_h])
##        axl.set_visible(False)
#        plt.axhline(axl_y,xmin=0, xmax=0,hold=None,color= 'k', linewidth=2)
##        axl(linspace(0,1,100), axl_y*ones([1,100]), 'k-')

#        ax.text(-0.0320,-0.140,flare.name.strftime('%d %b %Y'),transform=ax.transAxes,size=14)
#        ax.set_xlabel('Time [UT]', size=18)
#        #Set title of SuperDARN RTI PLOT
#        r=pydarn.radar.network().getRadarByCode(radars[0])
#        ax.set_title(r.name+' (Beam: '+str(beam)+')')
#       # ax.text(.015,.90,'(e)',transform=ax.transAxes,**letter_prop)

#    xticks  = []
#        for x in range(7):
#            xticks.append(goes_sTime + datetime.timedelta(hours=(1*x)))
#        ax.xaxis.set_ticks(xticks)
#
#        ax.title.set_fontsize(title_prop['size'])
##        ax.title.set_weight(title_prop['weight'])
##
##        ax.title.set_fontsize(18)
#        ax.title.set_weight(title_prop['weight'])
#
#        xticklabels = []
##        ax.label.set_fontsize(18) 
##        rcParams.update({'font.size':18})
#        xfont={'size':18}
#        for x,tick in enumerate(xticks):
#            xticklabels.append(tick.strftime('%H%M'))
#        ax.set_xticklabels(xticklabels, fontdict=xfont)
#        for item in (ax.get_xticklabels()):
#            item.set_fontsize(18)
#
#        ax.vlines(map_times,0,1,linestyle='--',color='b')
#
#        fig.tight_layout(h_pad=2.5,w_pad=3.5)
import ipdb; ipdb.set_trace()
#fig.savefig(filepath,bbox_inches='tight')
fig.savefig(filepath)
fig.savefig(filepath[:-3]+'pdf',bbox_inches='tight')
#plt.clf()

#good_pct = float(good_count)/total_count * 100.
#print ''
#print 'Final stats for: {0}'.format(filepath)
#print 'Geolocation success: {0:d}/{1:d} ({2:.1f}%)'.format(good_count,total_count,good_pct)
#    except:
#        pass
