#!/usr/bin/env python
# Script to generate a summary plot of the geomagnetic environment.

import os
import datetime

import numpy as np
import scipy as sp

import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt

import hamsci

GME     = hamsci.geomagenv.GmeObject

class GeomagSummary(object):
    def __init__(self,sTime,eTime,output_dir='output'):
        self.sTime  = sTime
        self.eTime  = eTime

        filename    = 'geomag_summary-{:%Y%m%d.%H%M}-{:%Y%m%d.%H%M}.png'.format(sTime,eTime)
        filepath    = os.path.join(output_dir,filename)

        # Go plot!! ############################ 
        ## Determine the aspect ratio of subplot.
        self.xsize      = 10
        self.ysize      = 4
        self.nx_plots   = 1
        self.ny_plots   = 2
        self.plot_nr    = 0

        rcp = mpl.rcParams
        rcp['axes.titlesize']   = 'large'
        rcp['axes.titleweight'] = 'bold'
        rcp['axes.labelweight'] = 'bold'
        rcp['font.weight']      = 'bold'

        fig         = plt.figure(figsize=(self.nx_plots*self.xsize,
                                          self.ny_plots*self.ysize))
        self.fig    = fig

        self.plot_flow_np()
        self.plot_by_bz()

        # Close up plot. #######################
        fig.tight_layout()
        fig.savefig(filepath,bbox_inches='tight')
        plt.close(fig)

    def plot_flow_np(self,xlabels=True,ax=None):
        """
        OMNI Flow Speed and Number Densityi
        """
        if ax is None:
            ax = self.new_ax()

        sTime       = self.sTime
        eTime       = self.eTime

        lines       =[]
        data_obj    = GME(sTime,eTime,'omni_flowSpeed')
        ds          = data_obj.active
        tmp,        = ax.plot(ds.data.index,ds.data,label=ds.metadata['symbol'])
        lines.append(tmp)
        ax.set_ylabel(ds.metadata['gme_label'])

        ax_1        = ax.twinx()
        ax.set_zorder(ax_1.get_zorder()+1)
        ax.patch.set_visible(False)
        data_obj    = GME(sTime,eTime,'omni_np')
        ds          = data_obj.active
        color       = 'green'
        label       = ds.metadata['symbol']
        tmp,        = ax_1.plot(ds.data.index,ds.data,label=label,color=color)
        lines.append(tmp)
        ax_1.set_ylabel(ds.metadata['gme_label'],color=color)
        for tl in ax_1.get_yticklabels():
            tl.set_color(color)

        ax.legend(handles=lines,loc='upper right',ncol=2)
        self.xtick_time_fmt(ax,show_labels=False)

    def plot_by_bz(self,xlabels=True,ax=None):
        """
        By/Bz GSM
        """
        if ax is None:
            ax = self.new_ax()

        sTime       = self.sTime
        eTime       = self.eTime

        data_obj    = GME(sTime,eTime,'omni_bz')
        ds          = data_obj.active
        ax.plot(ds.data.index,ds.data,label=ds.metadata['symbol'])

        data_obj    = GME(sTime,eTime,'omni_by')
        ds          = data_obj.active
        ax.plot(ds.data.index,ds.data,label=ds.metadata['symbol'])
        ax.set_ylabel('B [nT]')

        ax.legend(loc='upper right',ncol=2)
        self.xtick_time_fmt(ax,show_labels=True)

    def new_ax(self):
        fig = self.fig
        self.plot_nr += 1
        ax = fig.add_subplot(self.ny_plots,self.nx_plots,self.plot_nr)
        return ax

    def xtick_time_fmt(self,ax,show_labels=False):
        ax.set_xlim(self.sTime,self.eTime)
        if not show_labels:
            for xtl in ax.get_xticklabels():
                xtl.set_visible(show_labels)
            return

        xticks  = ax.get_xticks()
        xtls    = []
        for xtick in xticks:
            xtd = mpl.dates.num2date(xtick)
            if xtd.hour == 0 and xtd.minute == 0:
                xtl = xtd.strftime('%H%M\n%d %b %Y')
            else:
                xtl = xtd.strftime('%H%M')
            xtls.append(xtl)
        ax.set_xticklabels(xtls)

        for xtl in ax.get_xticklabels():
            xtl.set_ha('left')
            xtl.set_visible(show_labels)

if __name__ == '__main__':
    output_dir          = os.path.join('output','geomag_summary')
    hamsci.general_lib.prepare_output_dirs({0:output_dir},clear_output_dirs=True)

    # 2014 Nov Sweepstakes
    sTime   = datetime.datetime(2014,11,1)
    eTime   = datetime.datetime(2014,11,4)

#    # 2015 Nov Sweepstakes
#    sTime   = datetime.datetime(2015,11,7)
#    eTime   = datetime.datetime(2015,11,10)

#    # 2016 CQ WPX CW
#    sTime   = datetime.datetime(2016,5,28)
#    eTime   = datetime.datetime(2016,5,29)

    dct = {}
    dct['output_dir']   = output_dir
    dct['sTime']        = sTime
    dct['eTime']        = eTime

    GeomagSummary(**dct)
