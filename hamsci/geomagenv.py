import os
import datetime
import copy
import re

import numpy as np
import scipy as sp
import pandas as pd

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from mpl_toolkits.basemap import solar

import davitpy.gme as gme

from general_lib import prepare_output_dirs

# script filename (usually with path)
import inspect
curr_file = inspect.getfile(inspect.currentframe()) 

class GmeObject(object):
    """
    Create a container object to help us keep track of changes made to a time series.
    """
    def __init__(self,sTime,eTime,gme_param,plot_info={},**kwargs):

        data_set                ='DS000'
        comment                 ='Raw Data'
        plot_info['data_set']   = data_set 
        plot_info['serial']     = 0
        plot_info['gme_param']  = gme_param
        plot_info['sTime']      = sTime
        plot_info['eTime']      = eTime
        comment_str             = '[{}] {}'.format(data_set,comment)

        d0      = GmeDataSet(sTime,eTime,gme_param,plot_info=plot_info,
                        comment=comment_str,parent=self,**kwargs)
        setattr(self,data_set,d0)
        d0.set_active()
        d0.set_secondary()

    def get_data_set(self,data_set='active'):
        """
        Get a data_set, even one that only partially matches the string.
        """
        lst = dir(self)

        if data_set not in lst:
            tmp = []
            for item in lst:
                if data_set in item:
                    tmp.append(item)
            if len(tmp) == 0:
                data_set = 'active'
            else:
                tmp.sort()
                data_set = tmp[-1]

        return getattr(self,data_set)

    def get_all_data_sets(self):
        """
        Return a list of all data set objects.
        """
        ds_names    = self.list_all_data_sets()
        data_sets   = [getattr(self,dsn) for dsn in ds_names]
        return data_sets
    
    def list_all_data_sets(self):
        """
        Return a list of the names of all data sets associated with this object.
        """
        lst = dir(self)
    
        data_sets   = []
        for item in lst:
            if re.match('DS[0-9]{3}',item):
                data_sets.append(item)

        return data_sets

    def update_all_metadata(self,**kwargs):
        """
        Update the metadata/plot_info dictionaries of ALL attached data sets.
        """
        data_sets   = self.get_all_data_sets()
        for ds in data_sets:
            ds.metadata.update(kwargs)

    def copy(self):
        return copy.deepcopy(self)

    def delete_not_active(self):
        """
        Delete all but the active dataset.
        """

        ds_names    = self.list_all_data_sets()
        for ds_name in ds_names:
            if getattr(self,ds_name) is not self.active:
                delattr(self,ds_name)
        return self

class GmeDataSet(object):
    def __init__(self,sTime,eTime,gme_param,
            oversamp_T=datetime.timedelta(minutes=1),
            plot_info=None,comment=None,parent=None):

        # Save the input plot_info to override default data later.
        _plot_info          = plot_info
        plot_info           = {}
        plot_info['sTime']  = sTime
        plot_info['eTime']  = eTime

        if 'omni' in gme_param: 
            ind_class   = gme.ind.readOmni(sTime,eTime,res=1)
            omni_list   = []
            omni_time   = []
            for xx in ind_class:
                tmp = {}

#                tmp['res']          = xx.res
#                tmp['timeshift']    = xx.timeshift
#                tmp['al']           = xx.al
#                tmp['au']           = xx.au
#                tmp['asyd']         = xx.asyd
#                tmp['asyh']         = xx.asyh
#                tmp['symd']         = xx.symd
#                tmp['beta']         = xx.beta
#                tmp['bye']          = xx.bye
#                tmp['bze']          = xx.bze
#                tmp['e']            = xx.e
#                tmp['flowSpeed']    = xx.flowSpeed
#                tmp['vxe']          = xx.vxe
#                tmp['vye']          = xx.vye
#                tmp['vzy']          = xx.vzy
#                tmp['machNum']      = xx.machNum
#                tmp['np']           = xx.np
#                tmp['temp']         = xx.temp
#                tmp['time']         = xx.time

                tmp['ae']           = xx.ae
                tmp['bMagAvg']      = xx.bMagAvg
                tmp['bx']           = xx.bx 
                tmp['bym']          = xx.bym
                tmp['bzm']          = xx.bzm
                tmp['pDyn']         = xx.pDyn
                tmp['symh']         = xx.symh
                tmp['flowSpeed']    = xx.flowSpeed
                tmp['np']           = xx.np
                tmp['temp']         = xx.temp
                
                omni_time.append(xx.time)
                omni_list.append(tmp)

            omni_df_raw         = pd.DataFrame(omni_list,index=omni_time)
            del omni_time
            del omni_list

            self.omni_df_raw    = omni_df_raw
            self.omni_df        = omni_df_raw.resample('T')
            self.omni_df        = self.omni_df.interpolate()

        plot_info['x_label']    = 'Date [UT]'
        if gme_param == 'ae':
            # Read data with DavitPy routine and place into numpy arrays.
            ind_class   = gme.ind.readAe(sTime,eTime,res=1)
            ind_data    = [(x.time, x.ae) for x in ind_class]

            df_raw              = pd.DataFrame(ind_data,columns=['time','raw'])
            df_raw              = df_raw.set_index('time')

            plot_info['title']        = 'Auroral Electrojet (AE) Index'
            plot_info['symbol']       = 'Auroral Electrojet (AE) Index'
            plot_info['gme_label']    = 'AE Index [nT]'

        elif (gme_param == 'omni_by'):
            df_raw  = pd.DataFrame(omni_df_raw['bym'])

            plot_info['symbol']     = 'OMNI By GSM'
            plot_info['gme_label']  = 'OMNI By GSM [nT]'

        elif gme_param == 'omni_bz':
            df_raw  = pd.DataFrame(omni_df_raw['bzm'])

            plot_info['symbol']     = 'OMNI Bz GSM'
            plot_info['gme_label']  = 'OMNI Bz GSM [nT]'

        elif gme_param == 'omni_flowSpeed':
            df_raw  = pd.DataFrame(omni_df_raw['flowSpeed'])

            plot_info['symbol']     = 'OMNI v'
            plot_info['gme_label']  = 'OMNI v [km/s]'

        elif gme_param == 'omni_np':
            df_raw  = pd.DataFrame(omni_df_raw['np'])

            plot_info['symbol']     = 'OMNI Np'
            plot_info['gme_label']  = 'OMNI Np [N/cm^3]'

        elif gme_param == 'omni_pdyn':
            df_raw  = pd.DataFrame(omni_df_raw['pDyn'])

            plot_info['symbol']     = 'OMNI pDyn'
            plot_info['gme_label']  = 'OMNI pDyn [nPa]'

        elif gme_param == 'omni_symh':
            df_raw  = pd.DataFrame(omni_df_raw['symh'])

            plot_info['title']        = 'OMNI Sym-H'
            plot_info['symbol']       = 'OMNI Sym-H'
            plot_info['gme_label']    = 'OMNI Sym-H\n[nT]'
        elif gme_param == 'omni_bmagavg':
            df_raw                              = pd.DataFrame(omni_df_raw['bMagAvg'])
            plot_info['symbol']       = 'OMNI |B|'
            plot_info['gme_label']    = 'OMNI |B| [nT]'
        elif gme_param == 'solar_dec':
            times   = []
            taus    = []
            decs    = []
            
            curr_time   = sTime
            while curr_time < eTime:
                tau, dec = solar.epem(curr_time)

                times.append(curr_time)
                taus.append(tau)
                decs.append(dec)

                curr_time += datetime.timedelta(days=1)

            df_raw = pd.DataFrame(decs,index=times)
            plot_info['symbol']       = 'Solar Dec.'
            plot_info['gme_label']    = 'Solar Dec.'

        if plot_info.get('title') is None:
            plot_info['title']  = '{}'.format(gme_param.upper())

        if _plot_info is not None:
            plot_info.update(_plot_info)

        # Enforce sTime, eTime
        tf              = np.logical_and(df_raw.index >= sTime, df_raw.index < eTime)
        df_raw          = df_raw[tf].copy()
        df_raw.columns  = ['raw']

        if parent is None:
            # This section is for compatibility with code that only uses 
            # the single level GmeDataSet.

            # Resample data.
            df_rsmp         = df_raw.resample(oversamp_T)
            df_rsmp         = df_rsmp.interpolate()
            df_rsmp.columns = ['processed']

            self.ind_df_raw     = df_raw
            self.sTime          = sTime
            self.eTime          = eTime
            self.ind_df         = df_rsmp
            self.ind_times      = df_rsmp.index.to_pydatetime()
            self.ind_vals       = df_rsmp['processed']
        else:
            # This section is for attributes of the new container-style
            # GmeObject class.
            self.data           = df_raw['raw']
            self.data.name      = gme_param

        self.parent         = parent
        self.gme_param      = gme_param
        self.history        = {datetime.datetime.now():comment}
        self.plot_info      = plot_info
        self.metadata       = plot_info #Create alias for compatibility with other code.

    def __sub__(self,other,data_set='subtract',comment=None,**kwargs):
        """
        Drop NaNs.
        """
        new_data    = self.data - other.data
        if comment is None:
            ds_0    = self.plot_info['data_set'][:5]
            ds_1    = other.plot_info['data_set'][:5]
            comment = '{} - {}'.format(ds_0,ds_1)
        new_do      = self.copy(data_set,comment,new_data)
        return new_do

    def ds_name(self):
        return self.plot_info.get('data_set')

    def resample(self,dt=datetime.timedelta(minutes=1),data_set='resample',comment=None):
        no_na_data  = self.data.dropna()
        new_data    = no_na_data.resample(dt)
        
        if comment is None: comment = 'dt = {!s}'.format(dt)
        new_do      = self.copy(data_set,comment,new_data)
        return new_do

    def interpolate(self,data_set='interpolate',comment=None):
        new_data    = self.data.interpolate()
        
        if comment is None: comment = 'Interpolate'
        new_do      = self.copy(data_set,comment,new_data)
        return new_do

    def simulate(self,wave_list=None,data_set='simulate',comment=None):
        """
        Generate fake sinusoidal data for testing of signal
        processing codes.
        """
        if wave_list is None:
            wd          = {}
            wd['T']     = datetime.timedelta(days=5.)
            wd['A']     = 100.
            wave_list   = [wd]

        t_0 = self.data.index.min().to_datetime()
        xx  = (self.data.index - t_0).total_seconds()
        yy  = self.data.values.copy() * 0.

        for wd in wave_list:
            T   = wd.get('T')
            A   = wd.get('A',1.)
            C   = wd.get('C',0.)

            f_c = 1./T.total_seconds()
            yy += A*np.sin(2*np.pi*f_c*xx) + C

        if comment is None:
            comment = 'Simulate'

        new_do          = self.copy(data_set,comment)
        new_do.data[:]  = yy

        return new_do

    def rolling(self,window,center=True,kind='mean',data_set=None,comment=None):
        """
        Apply a rolling pandas function.

        kind: String that matches pandas rolling function.
              See pandas.rolling_{kind}()...

        window: datetime.timedelta
        """
        dt          = self.sample_period()
        roll_win    = int(window.total_seconds() / dt.total_seconds())

        rlng        = getattr(pd,'rolling_{}'.format(kind))
        new_data    = rlng(self.data,roll_win,center=center)

        if data_set is None:
            if window < datetime.timedelta(hours=1):
                time_str = '{:.0f}_min'.format(window.total_seconds()/60.)
            elif window < datetime.timedelta(days=1):
                time_str = '{:.0f}_hr'.format(window.total_seconds()/3600.)
            else:
                time_str = '{:.0f}_day'.format(window.total_seconds()/86400.)

            data_set = 'rolling_{}_{}'.format(time_str,kind)
        
        if comment is None: comment = 'window = {!s}'.format(window)
        new_do      = self.copy(data_set,comment,new_data)

        if window < datetime.timedelta(hours=1):
            time_str = '{:.0f} Min'.format(window.total_seconds()/60.)
        elif window < datetime.timedelta(days=1):
            time_str = '{:.0f} Hr'.format(window.total_seconds()/3600.)
        else:
            time_str = '{:.0f} Day'.format(window.total_seconds()/86400.)
        new_do.plot_info['smoothing'] = '{} {} Smoothing'.format(time_str,kind.title())
        return new_do

    def dropna(self,data_set='dropna',comment='Remove NaNs',**kwargs):
        """
        Drop NaNs.
        """
        new_data    = self.data.dropna(**kwargs)
        new_do      = self.copy(data_set,comment,new_data)
        return new_do


    def apply_filter(self,data_set='filtered',comment=None,**kwargs):
        sig_obj = self
        data    = self.data.copy()

        filt        = Filter(sig_obj,**kwargs)
        filt_data   = sp.signal.lfilter(filt.ir,[1.0],data)

        data[:]     = filt_data

        shift       = len(filt.ir)/2
        t_0         = data.index.min()
        t_1         = data.index[shift]
        dt_shift    = (t_1-t_0)/2

        tf          = data.index > t_1
        data        = data[tf]
        data.index  = data.index - dt_shift

        new_data        = data
        if comment is None:
            comment = filt.comment
        new_do          = self.copy(data_set,comment,new_data)
        new_do.filter   = filt
        return new_do

    def copy(self,newsig,comment,new_data=None):
        """Copy object.  This copies data and metadata, updates the serial number, and logs a comment in the history.  Methods such as plot are kept as a reference.

        **Args**:
            * **newsig** (str): Name for the new musicDataObj object.
            * **comment** (str): Comment describing the new musicDataObj object.
        **Returns**:
            * **newsigobj** (:class:`musicDataObj`): Copy of the original musicDataObj with new name and history entry.

        Written by Nathaniel A. Frissell, Fall 2013
        """

        if self.parent is None:
            print 'No parent object; cannot copy.'
            return

        all_data_sets   = self.parent.get_all_data_sets()
        all_serials     = [x.plot_info['serial'] for x in all_data_sets]
        serial          = max(all_serials) + 1
        newsig          = '_'.join(['DS%03d' % serial,newsig])

        setattr(self.parent,newsig,copy.copy(self))
        newsigobj = getattr(self.parent,newsig)

        newsigobj.data          = self.data.copy()
        newsigobj.gme_param     = '{}'.format(self.gme_param)
        newsigobj.metadata      = self.metadata.copy()
        newsigobj.plot_info     = newsigobj.metadata
        newsigobj.history       = self.history.copy()

        newsigobj.metadata['data_set']  = newsig
        newsigobj.metadata['serial']    = serial
        newsigobj.history[datetime.datetime.now()] = '[{}] {}'.format(newsig,comment)

        if new_data is not None:
            newsigobj.data = new_data
        
        newsigobj.set_active()
        return newsigobj
  
    def set_active(self):
        """Sets this signal as the currently active signal.

        Written by Nathaniel A. Frissell, Fall 2013
        """
        self.parent.active = self

    def set_secondary(self):
        """
        Sets this signal as the secondary signal.
        """
        self.parent.secondary = self

    def nyquist_frequency(self,time_vec=None,allow_mode=False):
        """Calculate the Nyquist frequency of a vt sigStruct signal.

        **Args**:
            * [**time_vec**] (list of datetime.datetime): List of datetime.datetime to use instead of self.time.

        **Returns**:
            * **nq** (float): Nyquist frequency of the signal in Hz.

        Written by Nathaniel A. Frissell, Fall 2013
        """

        dt  = self.sample_period(time_vec=time_vec,allow_mode=allow_mode)
        nyq = float(1. / (2*dt.total_seconds()))
        return nyq

    def is_evenly_sampled(self,time_vec=None):
        if time_vec is None:
            time_vec = self.data.index.to_pydatetime()

        diffs       = np.diff(time_vec)
        diffs_unq   = np.unique(diffs)

        if len(diffs_unq) == 1:
            return True
        else:
            return False

    def sample_period(self,time_vec=None,allow_mode=False):
        """Calculate the sample period of a vt sigStruct signal.

        **Args**:
            * [**time_vec**] (list of datetime.datetime): List of datetime.datetime to use instead of self.time.

        **Returns**:
            * **samplePeriod** (float): samplePeriod: sample period of signal in seconds.

        Written by Nathaniel A. Frissell, Fall 2013
        """
        
        if time_vec is None: time_vec = self.data.index.to_pydatetime()

        diffs       = np.diff(time_vec)
        diffs_unq   = np.unique(diffs)
        self.diffs  = diffs_unq

        if len(diffs_unq) == 1:
            samplePeriod = diffs[0].total_seconds()
        else:
            diffs_sec   = np.array([x.total_seconds() for x in diffs])
            maxDt       = np.max(diffs_sec)
            avg         = np.mean(diffs_sec)
            mode        = sp.stats.mode(diffs_sec).mode[0]

            md          = self.metadata
            warn        = 'WARNING'
            if md.has_key('title'): warn = ' '.join([warn,'FOR','"'+md['title']+'"'])
            print warn + ':'
            print '   Date time vector is not regularly sampled!'
            print '   Maximum difference in sampling rates is ' + str(maxDt) + ' sec.'
            samplePeriod = mode

            if not allow_mode:
                raise()
            else:
                print '   Using mode sampling period of ' + str(mode) + ' sec.'
        
        smp = datetime.timedelta(seconds=samplePeriod)
        return smp

    def set_metadata(self,**metadata):
        """Adds information to the current musicDataObj's metadata dictionary.
        Metadata affects various plotting parameters and signal processing routinges.

        **Args**:
            * **metadata** (**kwArgs): keywords sent to matplot lib, etc.

        Written by Nathaniel A. Frissell, Fall 2013
        """
        self.metadata = dict(self.metadata.items() + metadata.items())

    def print_metadata(self):
        """Nicely print all of the metadata associated with the current musicDataObj object.

        Written by Nathaniel A. Frissell, Fall 2013
        """
        keys = self.metadata.keys()
        keys.sort()
        for key in keys:
            print key+':',self.metadata[key]

    def append_history(self,comment):
        """Add an entry to the processing history dictionary of the current musicDataObj object.

        **Args**:
            * **comment** (string): Infomation to add to history dictionary.

        Written by Nathaniel A. Frissell, Fall 2013
        """
        self.history[datetime.datetime.now()] = '['+self.metadata['dataSetName']+'] '+comment

    def print_history(self):
        """Nicely print all of the processing history associated with the current musicDataObj object.

        Written by Nathaniel A. Frissell, Fall 2013
        """
        keys = self.history.keys()
        keys.sort()
        for key in keys:
            print key,self.history[key]

    def get_short_name(self,sep='-'):
        """
        Returns the DS Number and GME Param for filenames, etc.
        """
        ds_num  = self.metadata['data_set'][:5]
        param   = self.metadata.get('processing_code')
        if param is None:
            param   = self.metadata['gme_param']

        sn      = sep.join([ds_num,param])
        return sn

    def plot_fft(self,ax=None,label=None,plot_target_nyq=False,
            xlim=None,xticks=None,xscale='log',phase=False,plot_legend=True,**kwargs):
        T_max   = kwargs.pop('T_max',datetime.timedelta(days=0.5))
        f_max   = 1./T_max.total_seconds()

        data    = self.data.copy()

        # Handle NaNs just for FFT purposes.
        if data.hasnans:
            data    = data.interpolate()
            data    = data.dropna()

        data    = data - data.mean()

        smp     = self.sample_period(data.index.to_pydatetime())
        nqf     = self.nyquist_frequency(data.index.to_pydatetime())

        hann    = np.hanning(data.size)
        data    = data * hann

        n_fft   = 2**(data.size).bit_length()
        sf      = sp.fftpack.fft(data,n=n_fft)

        freq    = sp.fftpack.fftfreq(n_fft,smp.total_seconds())
        T_s     = 1./freq
        T_d     = T_s / (24. * 60. * 60.)

#        if label is not None:
#            txt     = 'df = {!s} Hz'.format(freq[1])
#            label   = '\n'.join([label,txt])

        if ax is None:
            ax = plt.gca()

        xx      = freq[1:n_fft/2]
        yy      = sf[1:n_fft/2]
        
        if phase is not True:
            yy      = np.abs(yy)
            yy      = yy/yy.max()
            ylabel  = 'FFT |S(f)|'
        else:
            yy      = np.angle(yy,deg=True)
            ylabel  = 'FFT Phase [deg]'

        ax.plot(xx,yy,marker='.',label=label,**kwargs)

        #### Plot Nyquist Line for Target Sampling Rate (1 Day)
        if plot_target_nyq:
            dt_min  = datetime.timedelta(days=1)
            nyq_min = 1./(2.*dt_min.total_seconds())

            label   = '{!s}'.format(2*dt_min)
            ax.axvline(nyq_min,ls='--',color='g',label=label)

        #### Define xticks
#        set_spectrum_xaxis(f_max,ax=ax)
        if xscale is not None:
            ax.set_xscale(xscale)
        if xlim is not None:
            ax.set_xlim(xlim)
        if xticks is not None:
            ax.set_xticks(xticks)

        xts_hz  = ax.get_xticks()
        xts_d   = (1./xts_hz) / (24.*60.*60.)

        xtl     = []
        for xts_hz,xts_d in zip(xts_hz,xts_d):
            if np.isinf(xts_d):
                xtl.append('{:0.3g}\nInf'.format(xts_hz))
            else:
                xtl.append('{:0.3g}\n{:.1f}'.format(xts_hz,xts_d))
        ax.set_xticklabels(xtl)
        ax.set_xlabel('Frequency [Hz]\nPeriod [days]')

        #### A little more plot cleanup
        ax.set_ylabel(ylabel)
        if plot_legend:
            ax.legend(loc='upper right',fontsize='small')
        ax.grid(True)

    def plot_lsp(self,ax=None,n_freq=2**10,label=None,**kwargs):
        T_max   = kwargs.pop('T_max',datetime.timedelta(days=0.5))
        f_max   = 1./T_max.total_seconds()

        data    = self.data.copy()
        data    = data.dropna()

        data    = data - data.mean()

        smp     = self.sample_period(data.index.to_pydatetime(),allow_mode=True)
        nqf     = self.nyquist_frequency(data.index.to_pydatetime(),allow_mode=True)

        hann    = np.hanning(data.size)
        data    = data * hann

        t_0     = data.index.min().to_pydatetime()
        smp_vec = (data.index - t_0).total_seconds()
        
        freq    = np.arange(n_freq)/(n_freq-1.) * f_max
        freq    = freq[1:]
        omega   = 2.*np.pi*freq

        T_s     = 2.*np.pi/freq
        T_d     = T_s / (24. * 60. * 60.)

        lsp     = sp.signal.lombscargle(smp_vec,data.values,omega)

        if ax is None:
            ax = plt.gca()

        xx      = freq
        yy      = lsp
        yy      = 2.*np.sqrt(4.*(lsp/n_freq))
        ax.plot(xx,yy,marker='.',label=label,**kwargs)

        #### Define xticks
        set_spectrum_xaxis(f_max,ax=ax)

        ax.set_ylabel('LSP S(f)')
        ax.grid(True)

class KpData(object):
    def __init__(self,sTime,eTime):
        kp_obj  = gme.ind.readKp(sTime=sTime,eTime=eTime)
        times   = []
        vals    = []

        dt      = datetime.timedelta(hours=3)
        for kp_record in kp_obj:
            this_time   = kp_record.time

            if this_time < sTime or this_time >= eTime:
                continue

            for kp_str in kp_record.kp:
                kp_val  = self.kp_num(kp_str)
                times.append(this_time)
                vals.append(kp_val)

                this_time = this_time + dt

        kp_series   = pd.Series(vals,index=times)
        tf          = np.logical_and(kp_series.index >= sTime, kp_series.index < eTime)
        kp_series   = kp_series[tf]
        self.kp     = kp_series
                
    def kp_num(self,kp_str):
        if kp_str[-1] == '-':
            add = -0.33
        elif kp_str[-1] == '+':
            add = 0.33
        else:
            add = 0

        kp_val = float(kp_str[0]) + add
        return kp_val

    def plot_kp(self,ax=None):
        if ax is None:
            plt.gca()

        xvals   = self.kp.index + datetime.timedelta(minutes=90)
        markers,stems,base  = ax.stem(xvals,self.kp)
        ax.set_ylim(0,9)
