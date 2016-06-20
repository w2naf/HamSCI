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
    def __init__(self,sTime,eTime,
        llcrnrlon=-180., llcrnrlat=-90, urcrnrlon=180., urcrnrlat=90.,
        output_dir='output'):

        self.latlon_bnds = {'llcrnrlat':llcrnrlat,'llcrnrlon':llcrnrlon,'urcrnrlat':urcrnrlat,'urcrnrlon':urcrnrlon}
        self.sTime  = sTime
        self.eTime  = eTime

        filename    = 'geomag_summary-{:%Y%m%d.%H%M}-{:%Y%m%d.%H%M}.png'.format(sTime,eTime)
        filepath    = os.path.join(output_dir,filename)

        # Go plot!! ############################ 
        ## Determine the aspect ratio of subplot.
        self.xsize      = 17.5
        self.ysize      = 4
        self.nx_plots   = 1
        self.ny_plots   = 3
        self.plot_nr    = 0

        rcp = mpl.rcParams
        rcp['axes.titlesize']   = 'x-large'
        rcp['axes.labelsize']   = 'x-large'
        rcp['xtick.labelsize']  = 'x-large'
        rcp['ytick.labelsize']  = 'x-large'
        rcp['axes.titleweight'] = 'bold'
        rcp['axes.labelweight'] = 'bold'
        rcp['font.weight']      = 'bold'

        leg_dct = {}
        leg_dct['loc']      = 'best'
        leg_dct['ncol']     = 2
        self.legend_dict    = leg_dct
        self.legend_lw      = 4

        fig         = plt.figure(figsize=(self.nx_plots*self.xsize,
                                          self.ny_plots*self.ysize))
        self.fig    = fig

#        self.plot_flow_np(xlabels=False)
#        self.plot_by_bz(xlabels=False)
        self.plot_ae(xlabels=False)
        self.plot_symh_kp(xlabels=False)
#        self.plot_rbn_spot_counts(plot_all=False,plot_by_band=True,xlabels=True)
        self.plot_rbn_spot_counts(plot_all=True,plot_by_band=False,xlabels=True)

        # Close up plot. #######################
        fig.tight_layout()

        title   = []
        title.append('Geomagnetic Environment Summary')
        date_fmt    = '%Y %b %d %H%M UT'
        date_str    = '{} - {}'.format(sTime.strftime(date_fmt), eTime.strftime(date_fmt))
        title.append(date_str)
        fontdict    = {'size':'xx-large','weight':'bold'}
        fig.text(0.5,0.99,'\n'.join(title),fontdict=fontdict,ha='center')

        fig.savefig(filepath,bbox_inches='tight')
        plt.close(fig)

    def plot_flow_np(self,xlabels=True,ax=None):
        """
        OMNI Flow Speed and Number Density
        """
        if ax is None:
            ax = self.new_ax()

        sTime       = self.sTime
        eTime       = self.eTime
        lines       =[]

        # OMNI Flow Speed ######################
        data_obj    = GME(sTime,eTime,'omni_flowSpeed')
        ds          = data_obj.active
        tmp,        = ax.plot(ds.data.index,ds.data,label=ds.metadata['symbol'])
        lines.append(tmp)
        ax.set_ylabel(ds.metadata['gme_label'])

        # OMNI Number Density ##################
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

        dct = self.legend_dict.copy()
        dct.update({'handles':lines})
        leg = ax.legend(**dct)
        for legobj in leg.legendHandles:
            legobj.set_linewidth(self.legend_lw)

        self.xtick_time_fmt(ax,show_labels=xlabels)

    def plot_by_bz(self,xlabels=True,ax=None):
        """
        OMNI By/Bz GSM
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

        ax.axhline(0,ls='--',color='k')

        dct = self.legend_dict.copy()
        leg = ax.legend(**dct)
        for legobj in leg.legendHandles:
            legobj.set_linewidth(self.legend_lw)
        self.xtick_time_fmt(ax,show_labels=xlabels)

    def plot_symh_kp(self,xlabels=True,ax=None):
        """
        Sym-H and Kp
        """
        if ax is None:
            ax = self.new_ax()

        sTime       = self.sTime
        eTime       = self.eTime
        lines       =[]

        # SYM-H ################################
        data_obj    = GME(sTime,eTime,'omni_symh')
        ds          = data_obj.active
        tmp,        = ax.plot(ds.data.index,ds.data,label=ds.metadata['symbol'],color='k')
        ax.fill_between(ds.data.index,0,ds.data,color='0.75')
        lines.append(tmp)
        ax.set_ylabel(ds.metadata['gme_label'])
        ax.axhline(0,color='k',ls='--')


        # Kp ###################################
        ax_1        = ax.twinx()
#        ax.set_zorder(ax_1.get_zorder()+1)
        ax.patch.set_visible(False)
        kp          = hamsci.geomagenv.KpData(self.sTime,self.eTime) 
        low_color   = 'green'
        mid_color   = 'darkorange'
        high_color  = 'red'
        label       = ds.metadata['symbol']
        xvals       = kp.kp.index + datetime.timedelta(minutes=90)

        color       = low_color
        kp_markersize = 10
        markers,stems,base  = ax_1.stem(xvals,kp.kp,color=color)
        for stem in stems:
            stem.set_color(color)
        markers.set_color(color)
        markers.set_label('Kp Index')
        markers.set_markersize(kp_markersize)
        lines.append(markers)

        tf = kp.kp == 4
        if np.count_nonzero(tf) > 0:
            xx      = xvals[tf]
            yy      = kp.kp[tf]
            color   = mid_color
            markers,stems,base  = ax_1.stem(xx,yy,color=color)
            for stem in stems:
                stem.set_color(color)
            markers.set_color(color)
            markers.set_markersize(kp_markersize)
            lines.append(markers)

        tf = kp.kp > 5
        if np.count_nonzero(tf) > 0:
            xx      = xvals[tf]
            yy      = kp.kp[tf]
            color   = high_color
            markers,stems,base  = ax_1.stem(xx,yy,color=color)
            for stem in stems:
                stem.set_color(color)
            markers.set_color(color)
            markers.set_markersize(kp_markersize)
            lines.append(markers)

        ax_1.set_ylabel('Kp Index')
        ax_1.set_ylim(0,9)
        for tk,tl in zip(ax_1.get_yticks(),ax_1.get_yticklabels()):
            if tk < 4:
                color = low_color
            elif tk == 4:
                color = mid_color
            else:
                color = high_color
            tl.set_color(color)

        dct = self.legend_dict.copy()
        dct.update({'handles':lines})
        leg = ax.legend(**dct)
        for legobj in leg.legendHandles:
            legobj.set_linewidth(self.legend_lw)

        self.xtick_time_fmt(ax,show_labels=xlabels)

    def plot_ae(self,xlabels=True,ax=None):
        """
        Auroral Electrojet (AE)
        """
        if ax is None:
            ax = self.new_ax()

        sTime       = self.sTime
        eTime       = self.eTime

        data_obj    = GME(sTime,eTime,'ae')
        ds          = data_obj.active
        ax.plot(ds.data.index,ds.data,label=ds.metadata['symbol'])
        ax.set_ylabel(ds.metadata['gme_label'])

        dct = self.legend_dict.copy()
        leg = ax.legend(**dct)
        for legobj in leg.legendHandles:
            legobj.set_linewidth(self.legend_lw)
        self.xtick_time_fmt(ax,show_labels=xlabels)

    def plot_rbn_spot_counts(self,xlabels=True,ax=None,
            integration_time=datetime.timedelta(minutes=15),
            plot_by_band=True,plot_all=False,ylim=None):
        """
        OMNI By/Bz GSM
        """
        if ax is None:
            ax = self.new_ax()

        sTime       = self.sTime
        eTime       = self.eTime

        rbn_obj     = hamsci.rbn_lib.RbnObject(sTime,eTime)
        rbn_obj.active.latlon_filt(**self.latlon_bnds)
        rbn_obj.active.plot_spot_counts(sTime=sTime,eTime=eTime,
                integration_time=integration_time,
                plot_by_band=plot_by_band,plot_all=plot_all,
                legend_lw=self.legend_lw,ax=ax,plot_title=False)

        ax.set_ylim(ylim)
        self.xtick_time_fmt(ax,show_labels=xlabels)

        llcrnrlat   = self.latlon_bnds['llcrnrlat']
        urcrnrlat   = self.latlon_bnds['urcrnrlat']
        llcrnrlon   = self.latlon_bnds['llcrnrlon']
        urcrnrlon   = self.latlon_bnds['urcrnrlon']

        latlon_str  = 'Lat Range: {:.0f} to {:.0f}; N Lon Range: {:.0f} to {:.0f} E'.format(llcrnrlat,urcrnrlat,llcrnrlon,urcrnrlon) 
        ax.text(0.990,0.925,latlon_str,transform=ax.transAxes,ha='right')

    def new_ax(self):
        """
        Create a new axis.
        """
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

        ax.set_xlabel('UT')

if __name__ == '__main__':
    output_dir          = os.path.join('output','geomag_summary')
    hamsci.general_lib.prepare_output_dirs({0:output_dir},clear_output_dirs=True)

    # 2014 Nov Sweepstakes
    sTime   = datetime.datetime(2014,11,1)
    eTime   = datetime.datetime(2014,11,4)

##    # 2015 Nov Sweepstakes
#    sTime   = datetime.datetime(2015,11,7)
#    eTime   = datetime.datetime(2015,11,10)

#    # 2016 CQ WPX CW
#    sTime   = datetime.datetime(2016,5,28)
#    eTime   = datetime.datetime(2016,5,29)

    dct = {}
    dct['output_dir']   = output_dir
    dct['sTime']        = sTime
    dct['eTime']        = eTime
    dct.update({'llcrnrlat':20.,'llcrnrlon':-130.,'urcrnrlat':55.,'urcrnrlon':-65.})

    GeomagSummary(**dct)
