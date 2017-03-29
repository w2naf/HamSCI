#!/usr/bin/env python
import os
import datetime
import pickle as pickle
import bz2
import multiprocessing

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['text.usetex'] = True
from matplotlib import pyplot as plt
import mpl_toolkits.axes_grid.axes_size as Size
from mpl_toolkits.axes_grid import Divider
import matplotlib.dates as mdates

import matlab.engine
#eng = matlab.engine.start_matlab('-desktop')
#eng = matlab.engine.start_matlab()

import hamsci
from hamsci.general_lib import prepare_output_dirs as prep_dirs

class TxRxRaytracer(object):
    def __init__(self, date, frequency, nhops=3.,
                 tx_call      = None, tx_lat     = None,   tx_lon      = None,
                 rx_call      = None, rx_lat     = None,   rx_lon      = None,
                 R12          = -1,   kp         = 0.,     irregs_flag = 0.,
                 tx_power     = 1.,   gain_tx_db = 1.,     gain_rx_db  = 1.,
                 start_range  = 0.,   max_range  = 10000., num_range   = 201.,
                 start_height = 0.,   height_inc = 3.,     num_heights = 200., 
                 start_elev   = 2.,   max_elev   = 60.,    elev_step   = 0.5):
        """
        Ray trace a fan of elevation angles from a transmitter to a receiver on a specified frequency.
        The following are determined:
            * Most likely short-path ionospheric ray linking the TX and RX.
            * Predicted receiver power at RX and along path between RX and TX.

        PHaRLAP is used as the ray tracing engine.
        IRI2016 is used as the default ionosphere.

        Arguments:
        * date:         datetime.datetime object
        * frequency:    float [MHz]

        Keywords:
        * nhops:        Maximum number of hops to raytrace.
        * tx_call:      Transmitter callsign
        * tx_lat:       Transmitter latitude
        * tx_lon:       Transmitter longitude
        * rx_call:      Receiver callsign
        * rx_lat:       Receiver latitude
        * rx_lon:       Receiver longitude
        * R12:          scalar R12 index
            R12 = 1 - 200 :  IRI2016 is called with R12 (Zurich V1.0) input as the
                             user specified yearly smoothed monthly median sunspot
                             number. The foF2 storm model will be turned off
                             regardless of the setting of the optional input
                             iri_options.foF2_storm (see below). 
            R12 = -1      :  IRI2016 is called with ionospheric conditions (R12,
                             IG12, and F10.7) read from file (ig_rz.dat) based on
                             input epoch (UT) and may be historical or projected
                             conditions (dependent on the epoch). The foF2 storm
                             model defaults to "on" for this case (but may be
                             overridden). See the optional input
                             iri_options.foF2_storm below.   

        * kp:           kp not used if irregs_flag = 0.
        * irregs_flag:  (0 or 1) Turn on/off field aligned irregularities.
        * tx_power:     Transmit power [W]
        * gain_tx_db:   Transmitter gain [dB]
        * gain_rx_db:   Receiver gain [dB]

        * start_range:  start range for ionospheric grid [km]
        * max_range:    maximum range for sampling the ionosphere [km]
        * num_range:    number of ranges (must be < 2000)

        * start_height: start height for ionospheric grid [km]
        * height_inc:   height increment [km]
        * num_heights:  number of  heights (must be < 2000)

        * start_elev:   start takeoff angle [deg]
        * max_elev:     end takeoff angle   [deg]
        * elev_step:    takeoff angle step  [deg]
        """

        # doppler_flag must be set to generate ionosphere
        # 5 minutes later so that Doppler shift can be calculated.
        doppler_flag = 1

        eng = matlab.engine.start_matlab()
    #    eng = matlab.engine.start_matlab('-desktop')


        mld = {}    # Dictionary of values passed to MATLAB workspace
        d   = date
        mld['UT']               =  matlab.double([d.year,d.month,d.day,d.hour,d.minute])
        mld['R12']              =  R12                  # R12 index
        mld['speed_of_light']   =  2.99792458e8

        mld['tx_lat']           =  float(tx_lat)        # latitude of the start point of ray
        mld['tx_lon']           =  float(tx_lon)        # longitude of the start point of ray
        mld['rx_lat']           =  float(rx_lat)        # latitude of the receiver
        mld['rx_lon']           =  float(rx_lon)        # longitude of the receiver

        mld['tx_power']         =  float(tx_power)      # Transmitter power in Watts
        mld['gain_tx_db']       =  float(gain_tx_db)    # Transmit antenna gain in dB
        mld['gain_rx_db']       =  float(gain_rx_db)    # Receive antenna gain in dB

        mld['doppler_flag']     =  doppler_flag         # generate ionosphere 5 minutes later so that
                                                        # Doppler shift can be calculated
        mld['irregs_flag']      =  irregs_flag          # no irregularities - not interested in 
                                                        # Doppler spread or field aligned irregularities
        mld['kp']               =  kp                   # kp not used as irregs_flag = 0. Set it to a 
                                                        # dummy value 

        mld['start_range']      = start_range           # start range for ionospheric grid (km)
        mld['max_range']        = max_range             # maximum range for sampling the ionosphere (km)
        mld['num_range']        = num_range             # number of ranges (must be < 2000)

        mld['start_height']     = start_height          # start height for ionospheric grid (km)
        mld['height_inc']       = height_inc            # height increment (km)
        mld['num_heights']      = num_heights           # number of  heights (must be < 2000)

        mld['freq']             = float(frequency)      # frequency (MHz)
        mld['tol']              = 1e-7                  # ODE tolerance
        mld['nhops']            = nhops

        mld['elev_step']        = elev_step
        mld['elevs']            = matlab.double(np.arange(start_elev,max_elev,elev_step,
                                    dtype=np.double).tolist())

        for key,val in mld.items():
            eng.workspace[key] = val

        date_str    = date.strftime('%Y%m%d.%H%M')
        freq_str    = '{:.0f}'.format(mld['freq']*1000)

        # Call MATLAB / Run Ray Trace ########## 
        eng.ray_tracker_2d(nargout=0)

        # Retrieve Results and Store in Dictionary
        dr                          = eng.workspace['range_inc']
        r_0                         = eng.workspace['start_range']
        r_1                         = eng.workspace['max_range']
        rngs                        = np.arange(r_0,r_1+dr,dr)

        dh                          = eng.workspace['height_inc']
        h_0                         = eng.workspace['start_height']
        h_1                         = eng.workspace['num_heights'] * dh
        heights                     = np.arange(h_0,h_1,dh)

        rt_dct = {}
        rt_dct['date']              = date
        rt_dct['tx_call']           = tx_call
        rt_dct['tx_lat']            = float(eng.workspace['tx_lat'])
        rt_dct['tx_lon']            = float(eng.workspace['tx_lon'])
        rt_dct['rx_call']           = rx_call
        rt_dct['rx_lat']            = float(eng.workspace['rx_lat'])
        rt_dct['rx_lon']            = float(eng.workspace['rx_lon'])
        rt_dct['rx_range']          = float(eng.workspace['rx_range'])
        rt_dct['azm']               = float(eng.workspace['rx_azm'])
        rt_dct['freq']              = float(eng.workspace['freq'])
        rt_dct['R12']               = float(eng.workspace['R12'])
        rt_dct['doppler_flag']      = int(eng.workspace['doppler_flag'])
        rt_dct['irregs_flag']       = int(eng.workspace['irregs_flag'])
        rt_dct['kp']                = float(eng.workspace['kp'])
        rt_dct['ranges']            = rngs
        rt_dct['heights']           = heights
        rt_dct['tx_power']          = tx_power
        rt_dct['gain_tx_db']        = gain_tx_db
        rt_dct['gain_rx_db']        = gain_rx_db

        iono_params = ['iono_en_grid', 'iono_en_grid_5',
                       'iono_pf_grid', 'iono_pf_grid_5',
                       'collision_freq']
        for ip in iono_params:
            rt_dct[ip]  = np.array(eng.workspace[ip])

        # Get Ray Path Data ####################
        n_elevs                     = int(eng.workspace['num_elevs'])
        ray_path_data = None
        for ray_id in range(n_elevs):
            tmp     = eng.eval('ray_path_data({!s})'.format(ray_id+1))
            freq    = tmp.pop('frequency')
            elev    = tmp.pop('initial_elev')

            for key, val in tmp.items():
                tmp[key] = np.array(val).flatten()
            tmp_df   = pd.DataFrame(tmp)

            tmp_df['frequency']     = float(freq)
            tmp_df['initial_elev']  = float(elev)
            tmp_df['ray_id']        = ray_id + 1 # MATLAB indexes from 1

            if ray_path_data is None:
                ray_path_data   = tmp_df
            else:
                ray_path_data   = pd.concat([ray_path_data,tmp_df],axis=0,ignore_index=True)


        # Get Ray Turning Point and Reflection Data
        fns = eng.workspace['ray_data_fieldnames']
        fns.remove('FAI_backscatter_loss')
        fns.append('rx_power_0_dB')  # Received power with no ionospheric or ground losses
        fns.append('rx_power_dB')    # Includes deviative absorption, GS loss
        fns.append('rx_power_O_dB')  # Includes deviative absorption, GS loss, O mode absorption
        fns.append('rx_power_X_dB')  # Includes deviative absorption, GS loss, X mode absorption

        ray_data = {}
        ray_data['ray_id']  = np.array(eng.workspace['rd_id'],dtype=np.int).flatten()
        for fn in fns:
            tmp             = np.array(eng.workspace['rd_{}'.format(fn)])
            ray_data[fn]    = np.array(tmp).flatten()

        
        ray_data    = pd.DataFrame(ray_data)

        rt_dct['ray_path_data'] = ray_path_data
        rt_dct['ray_data']      = ray_data

        # Get Ray Path Data of Receiver Ray ####
        if eng.workspace['srch_ray_good'] != 0:
            srch_n_elevs       = int(eng.workspace['srch_num_elevs'])
            srch_ray_path_data = None
            for ray_id in range(srch_n_elevs):
                tmp     = eng.eval('srch_ray_path_data({!s})'.format(ray_id+1))
                freq    = tmp.pop('frequency')
                elev    = tmp.pop('initial_elev')

                for key, val in tmp.items():
                    tmp[key] = np.array(val).flatten()
                tmp_df   = pd.DataFrame(tmp)

                tmp_df['frequency']     = float(freq)
                tmp_df['initial_elev']  = float(elev)
                tmp_df['ray_id']        = ray_id + 1 # MATLAB indexes from 1

                if ray_path_data is None:
                    srch_ray_path_data   = tmp_df
                else:
                    srch_ray_path_data   = pd.concat([srch_ray_path_data,tmp_df],axis=0,ignore_index=True)

            # Get Ray Turning Point and Reflection Data
            fns = eng.workspace['srch_ray_data_fieldnames']
            fns.remove('FAI_backscatter_loss')
            fns.append('rx_power_0_dB')  # Received power with no ionospheric or ground losses
            fns.append('rx_power_dB')    # Includes deviative absorption, GS loss
            fns.append('rx_power_O_dB')  # Includes deviative absorption, GS loss, O mode absorption
            fns.append('rx_power_X_dB')  # Includes deviative absorption, GS loss, X mode absorption

            srch_ray_data = {}
            srch_ray_data['ray_id']  = np.array(eng.workspace['srch_rd_id'],dtype=np.int).flatten()
            for fn in fns:
                tmp             = np.array(eng.workspace['srch_rd_{}'.format(fn)])
                srch_ray_data[fn]    = np.array(tmp).flatten()

            srch_ray_data    = pd.DataFrame(srch_ray_data)

            rt_dct['srch_ray_path_data'] = srch_ray_path_data
            rt_dct['srch_ray_data']      = srch_ray_data

        # Reorganize the raytrace dictionary to make it more user friendly.
        # Attach the new raytrace dictionary to the object.
        keys = {}
        tmp = []
        tmp.append('date')
        tmp.append('freq')
        tmp.append('tx_lat')
        tmp.append('tx_lon')
        tmp.append('tx_call')
        tmp.append('tx_power')
        tmp.append('gain_tx_db')
        tmp.append('rx_lat')
        tmp.append('rx_lon')
        tmp.append('rx_call')
        tmp.append('gain_rx_db')
        tmp.append('R12')
        tmp.append('kp')
        tmp.append('irregs_flag')
        tmp.append('doppler_flag')
        tmp.append('rx_range')
        tmp.append('azm')
        keys['metadata']    = tmp

        tmp = []
        tmp.append('ranges')
        tmp.append('heights')
        keys['axes']        = tmp

        tmp = []
        tmp.append('iono_en_grid')
        tmp.append('iono_en_grid_5')
        tmp.append('iono_pf_grid_5')
        tmp.append('iono_pf_grid')
        tmp.append('collision_freq')
        keys['ionosphere']  = tmp

        self.rt_dct = {}

        for grp_key,var_keys in keys.items():
            self.rt_dct[grp_key] = {}
            for var_key in var_keys:
                self.rt_dct[grp_key][var_key] = rt_dct.get(var_key)

        rt  = {}
        self.rt_dct['raytrace']   = rt 
        rt['all_ray_paths']       = rt_dct.get('ray_path_data')
        rt['all_ray_data']        = rt_dct.get('ray_data')
        rt['connecting_ray_path'] = rt_dct.get('srch_ray_path_data')
        rt['connecting_ray_data'] = rt_dct.get('srch_ray_data')

#        view    = []
#    #    view.append('Doppler_shift') 
#    #    view.append('Doppler_spread') 
#    #    view.append('TEC_path') 
#    #    view.append('apogee') 
#        view.append('deviative_absorption') 
#        view.append('effective_range') 
#    #    view.append('final_elev') 
#    #    view.append('frequency') 
#    #    view.append('geometric_path_length') 
#    #    view.append('gnd_rng_to_apogee') 
#        view.append('ground_range') 
#    #    view.append('group_range') 
#        view.append('initial_elev') 
#        view.append('lat') 
#        view.append('lon') 
#    #    view.append('nhops_attempted') 
#    #    view.append('phase_path') 
#    #    view.append('plasma_freq_at_apogee') 
#        view.append('ray_id') 
#        view.append('ray_label') 
#        view.append('rx_power_0_dB') 
#        view.append('rx_power_O_dB') 
#        view.append('rx_power_X_dB') 
#        view.append('rx_power_dB') 
#        view.append('virtual_height') 
#    ##    view.append('power_dbw')
#
#        rd  = rt_dct['ray_data']
#        rdv = rd[view]

    def plot_iono_path_profile(tx_lat,tx_lon,ranges,heights,
            maxground=None, maxalt=None,Re=6371,
            iono_arr=None,iono_param=None,
            iono_cmap='viridis', iono_lim=None, iono_title='Ionospheric Parameter',
            ray_path_data=None, 
            srch_ray_path_data=None, 
            fig=None, rect=111, ax=None, aax=None,
            plot_colorbar=True,
            iono_rasterize=False,**kwargs):
        """
        Plot a 2d ionospheric profile along a path.
        """
        if maxground is None:
            maxground = np.max(ranges)

        if maxalt is None:
            maxalt = np.max(heights)

        # Set up axes
        if not ax and not aax:
            ax, aax = curvedEarthAxes(fig=fig, rect=rect, 
                maxground=maxground, maxalt=maxalt,Re=Re)

        cbax = None # Have something to return even if we don't plot a colorbar.

        # Convert linear range into angular distance.
        thetas  = ranges/Re

        # Plot background ionosphere. ################################################## 
        if (iono_arr is not None) or (iono_param is not None):
            if iono_param == 'iono_en_grid' or iono_param == 'iono_en_grid_5':
                if iono_lim is None: iono_lim = (10,12)
                if iono_title == 'Ionospheric Parameter':
                    iono_title = r"N$_{el}$ [$\log_{10}(m^{-3})$]"
                # Get the log10 and convert Ne from cm**(-3) to m**(-3)
                iono_arr    = np.log10(kwargs[iono_param]*100**3)
            elif iono_param == 'iono_pf_grid' or iono_param == 'iono_pf_grid_5':
                if iono_lim is None: iono_lim = (0,10)
                if iono_title == 'Ionospheric Parameter':
                    iono_title = r"Plasma Frequency [MHz]"
                iono_arr    = kwargs[iono_param]
            elif iono_param == 'collision_freq':
                if iono_lim is None: iono_lim = (0,8)
                if iono_title == 'Ionospheric Parameter':
                    iono_title = r"$\nu$ [$\log_{10}(\mathrm{Hz})$]"
                iono_arr    = np.log10(kwargs[iono_param])

            if iono_lim is None:
                iono_mean   = np.mean(iono_arr)
                iono_std    = np.std(iono_arr)

                iono_0      = 50000
                iono_1      = 90000

                iono_lim    = (iono_0, iono_1)


            X, Y    = np.meshgrid(thetas,heights+Re)
            im      = aax.pcolormesh(X, Y, iono_arr,
                        vmin=iono_lim[0], vmax=iono_lim[1],
                        cmap=iono_cmap,rasterized=iono_rasterize)

            # Add a colorbar
            if plot_colorbar:
                cbax    = addColorbar(im, ax)
                _       = cbax.set_ylabel(iono_title)

        # Plot Ray Paths ###############################################################
        freq_s  = 'None'
        if ray_path_data is not None:
            rpd         = ray_path_data
            ray_ids     = rpd.ray_id.unique()
            for ray_id in ray_ids:
                tf  = rpd.ray_id == ray_id
                xx  = rpd[tf].ground_range/Re
                yy  = rpd[tf].height + Re
                aax.plot(xx,yy,color='white')
            f   = rpd.frequency.unique()
            if f.size == 1:
                freq_s  = '{:0.3f} MHz'.format(float(f))
            else:
                freq_s  = 'multi'

        if srch_ray_path_data is not None:
            rpd         = srch_ray_path_data
            ray_ids     = rpd.ray_id.unique()
            for ray_id in ray_ids:
                tf  = rpd.ray_id == ray_id
                xx  = rpd[tf].ground_range/Re
                yy  = rpd[tf].height + Re
                aax.plot(xx,yy,color='red',zorder=100,lw=3)

        # Plot Receiver ################################################################ 
        if 'rx_lat' in kwargs and 'rx_lon' in kwargs:
            rx_lat      = kwargs.get('rx_lat')
            rx_lon      = kwargs.get('rx_lon')
            rx_label    = kwargs.get('rx_label','Receiver')

            rx_theta    = kwargs.get('rx_range')/Re
            
            hndl    = aax.scatter([rx_theta],[Re],s=250,marker='*',color='red',zorder=100,clip_on=False,label=rx_label)
            aax.legend([hndl],[rx_label],loc='upper right',scatterpoints=1,fontsize='small')

        
        # Add titles and other important information.
        title       = []
        date_s      = kwargs.get('date').strftime('%Y %b %d %H:%M UT')
        tx_lat_s     = '{:0.2f}'.format(tx_lat) + r'$^{\circ}$N'
        tx_lon_s     = '{:0.2f}'.format(tx_lon) + r'$^{\circ}$E'
        azm_s       = '{:0.1f}'.format(kwargs['azm'])   + r'$^{\circ}$'
        title.append(date_s)
        title.append('TX Origin: {}, {}; Azimuth: {}, Frequency: {}'.format(tx_lat_s,tx_lon_s,azm_s,freq_s))
        ax.set_title('\n'.join(title))

        title   = []
        tx_call = rt_dct.get('tx_call')
        if tx_call is not None:
            title.append('TX: {}'.format(tx_call))
        rx_call = rt_dct.get('rx_call')
        if rx_call is not None:
            title.append('RX: {}'.format(rx_call))
        ax.set_title('\n'.join(title),loc='left')

        return ax, aax, cbax

    def plot_power_path(rt_dct,ax,ylim=(-175,-100)):
        rd      = rt_dct['ray_data'].sort_values('ground_range')

        plot_dct            = {}
        tmp, key            = ({},'rx_power_0_dB')
        plot_dct[key]       = tmp
        tmp['label']        = 'No Losses'
        tmp['color']        = 'r'

        tmp, key            = ({},'rx_power_dB')
        plot_dct[key]       = tmp
        tmp['label']        = 'Ground and Deviative Losses'
        tmp['color']        = 'g'

        tmp, key            = ({},'rx_power_O_dB')
        plot_dct[key]       = tmp
        tmp['label']        = 'O Mode'
        tmp['color']        = 'b'

        tmp, key            = ({},'rx_power_X_dB')
        plot_dct[key]       = tmp
        tmp['label']        = 'X Mode'
        tmp['color']        = 'c'

        plot_list       = []
        plot_list.append('rx_power_0_dB')
        plot_list.append('rx_power_dB')
        plot_list.append('rx_power_O_dB')
        plot_list.append('rx_power_X_dB')

        handles     = []
        labels      = []
        for param in plot_list:
            xx      = rd.ground_range.tolist()
            yy      = rd[param].tolist()
            if 'srch_ray_data' in rt_dct:
                srd     = rt_dct['srch_ray_data']
                inx     = srd.ground_range.argmax()
                xx.append(srd.ground_range[inx])
                yy.append((srd[param])[inx])

                tmp = pd.DataFrame({'xx':xx,'yy':yy})
                tmp = tmp.sort_values('xx')
                xx  = tmp.xx
                yy  = tmp.yy
            label   = plot_dct[param].get('label',param)
            color   = plot_dct[param].get('color',None)
            handle, = ax.plot(xx,yy,label=label,marker='.',color=color)
            handles.append(handle)
            labels.append(label)

        if 'srch_ray_data' in rt_dct:
            for param in plot_list:
                inx     = srd.ground_range.argmax()
                xx      = [srd.ground_range[inx]]
                yy      = [(srd[param])[inx]]
                color   = plot_dct[param].get('color',None)
                handle, = ax.plot(xx,yy,marker='*',ls=' ',color=color,
                        ms=10,zorder=100)

        if 'rx_lat' in rt_dct and 'rx_lon' in rt_dct:
            # Mark the receiver location.
            rx_label    = rt_dct.get('rx_label','Receiver')
            rx_range    = rt_dct.get('rx_range')
            ax.axvline(rx_range,color='r',ls='--')
            trans   = matplotlib.transforms.blended_transform_factory( ax.transData, ax.transAxes)
            hndl    = ax.scatter([rx_range],[0],s=250,marker='*',color='red',zorder=100,clip_on=False,transform=trans)
    #        ax.legend([hndl],[rx_label],loc='upper right',scatterpoints=1,fontsize='small')

    #    ax.legend(loc='upper left',fontsize='small')
        ax.legend(handles,labels,loc='lower left',fontsize='small')

        ax.set_xlim(0,rt_dct['maxground'])
        ax.set_ylim(ylim)

        f       = rd.frequency.unique()
        if f.size == 1:
            freq_s  = '{:0.3f} MHz'.format(float(f))
        else:
            freq_s  = 'multi'

        title       = []
    #    date_s      = date.strftime('%Y %b %d %H:%M UT')
    #    tx_lat_s     = '{:0.2f}'.format(rt_dct['tx_lat'])  + r'$^{\circ}$N'
    #    tx_lon_s     = '{:0.2f}'.format(rt_dct['tx_lon']) + r'$^{\circ}$E'
    #    azm_s       = '{:0.1f}'.format(rt_dct['azm'])    + r'$^{\circ}$'
    #    title.append(date_s)
    #    title.append('TX Origin: {}, {}; Azimuth: {}, Frequency: {}'.format(tx_lat_s,tx_lon_s,azm_s,freq_s))
        line        = 'TX Power: {:0.1f} W, TX Gain: {:0.1f} dB, RX Gain: {:0.1f} dB'.format(
                    rt_dct['tx_power'], rt_dct['gain_tx_db'], rt_dct['gain_rx_db'])
        title.append(line)
        ax.set_title('\n'.join(title))

        ax.set_xlabel('Ground Range [km]')
        ax.set_ylabel('Power [dBW]')

    def plot_raytrace_and_power(rt_dct):
        """
        Plot both the ray trace and recieved power plots.
        """
        output_dir  = rt_dct.get('output_dir','output')
        fname       = rt_dct.get('fname','raytrace_and_power')
        ################################################################################
        fig                 = plt.figure(figsize=(15,10))
        x_0,x_w             = (0.0, 0.95)
        rt_rect             = [x_0, 0.50, x_w, 0.45]

        x_scale             = 0.925
        xs_w                = x_scale * x_w
        xs_0                = x_0 + (x_w-xs_w)/2.
        pwr_rect            = [xs_0, 0.275, xs_w, 0.30]

        y_scale             = 0.5
        iono_cbar_hgt       = y_scale*rt_rect[3]
        iono_cbar_ypos      = rt_rect[1] + (rt_rect[3]-iono_cbar_hgt)/2.
        iono_cbar_rect      = [0.960, iono_cbar_ypos, 0.025, iono_cbar_hgt]

        horiz               = [Size.Scaled(1.0)]
        vert                = [Size.Scaled(1.0)]
        rt_div              = Divider(fig,rt_rect,horiz,vert,aspect=False)
        pwr_div             = Divider(fig,pwr_rect,horiz,vert,aspect=False)
        iono_cbar_div       = Divider(fig,iono_cbar_rect,horiz,vert,aspect=False)
        
        pos                 = {}
        pos['rt']           = rt_div.new_locator(  nx=0, ny=0)
        pos['pwr']          = pwr_div.new_locator( nx=0, ny=0)
        pos['iono_cbax']    = iono_cbar_div.new_locator(nx=0, ny=0) 
        ################################################################################


        rt_dct['fig']           = fig
    #    rt_dct['iono_param']    = 'iono_en_grid'
    #    rt_dct['iono_param']    = 'collision_freq'
        rt_dct['iono_param']    = 'iono_pf_grid'
        rt_dct['maxground']     = 4000.
        rt_dct['maxalt']        = 400.

        ax, aax, cbax = plot_iono_path_profile(**rt_dct)
        ax.set_axes_locator(pos['rt'])
        cbax.set_axes_locator(pos['iono_cbax'])

        ax      = fig.add_subplot(111)
        plot_power_path(rt_dct,ax)
        ax.set_axes_locator(pos['pwr'])

    #    fig.tight_layout()
        fpath       = os.path.join(output_dir,fname+'.png')
        fig.savefig(fpath,bbox_inches='tight')
        plt.close(fig)
        return fpath

    def rt_rx_pwr_to_csv(rt_dcts,print_header=True,output_dir='output',fname=None):
        """
        Writes Ray Trace Dictionary RX Power to CSV.
        """

        # Prepare file names.
        if fname is None:
            fname       = 'rxPwr_'+rt_dcts[0].get('event_fname','0')

        csv_fname       = '{}.csv'.format(fname)
        csv_path        = os.path.join(output_dir,csv_fname)


        keys    = []
        keys.append('event_fname')
        keys.append('date')
        keys.append('freq')
        keys.append('tx_call')
        keys.append('tx_lat')
        keys.append('tx_lon')
        keys.append('rx_call')
        keys.append('rx_lat')
        keys.append('rx_lon')
        keys.append('rx_range')
        keys.append('azm')
        keys.append('gain_tx_db')
        keys.append('gain_xx_db')
        keys.append('tx_power')

        with open(csv_path,'w') as fl:
            if print_header:
                fl.write('# PHaRLAP Predicted Receive Power')
                fl.write('#\n')
                fl.write('## Metadata ####################################################################\n')

                for key in keys:
                    val     = rt_dcts[0].get(key)
                    line    = '# {!s}: {!s}\n'.format(key,val)
                    fl.write(line)

                fl.write('#\n')
                fl.write('## Parameter Legend ############################################################\n')
                fl.write('# rx_power_0_dB: No Losses\n')
                fl.write('# rx_power_dB: Ground and Deviative Losses\n')
                fl.write('# rx_power_O_dB: O Mode\n')
                fl.write('# rx_power_X_dB: X Mode\n')

                fl.write('#\n')
                fl.write('## CSV Data ####################################################################\n')

        rx_power    = extract_rx_power(rt_dcts)
        rx_power.to_csv(csv_path,mode='a')

        return csv_path

def addColorbar(mappable, ax):
    """ Append colorbar to axes

    Parameters
    ----------
    mappable :
        a mappable object
    ax :
        an axes object

    Returns
    -------
    cbax :
        colorbar axes object

    Notes
    -----
    This is mostly useful for axes created with :func:`curvedEarthAxes`.

    written by Sebastien, 2013-04

    """
    from mpl_toolkits.axes_grid1 import SubplotDivider, LocatableAxes, Size
    import matplotlib.pyplot as plt 

    fig1 = ax.get_figure()
    divider = SubplotDivider(fig1, *ax.get_geometry(), aspect=True)

    # axes for colorbar
    cbax = LocatableAxes(fig1, divider.get_position())

    h = [Size.AxesX(ax), # main axes
         Size.Fixed(0.1), # padding
         Size.Fixed(0.2)] # colorbar
    v = [Size.AxesY(ax)]

    _ = divider.set_horizontal(h)
    _ = divider.set_vertical(v)

    _ = ax.set_axes_locator(divider.new_locator(nx=0, ny=0))
    _ = cbax.set_axes_locator(divider.new_locator(nx=2, ny=0))

    _ = fig1.add_axes(cbax)

    _ = cbax.axis["left"].toggle(all=False)
    _ = cbax.axis["top"].toggle(all=False)
    _ = cbax.axis["bottom"].toggle(all=False)
    _ = cbax.axis["right"].toggle(ticklabels=True, label=True)

    _ = plt.colorbar(mappable, cax=cbax)

    return cbax

def curvedEarthAxes(rect=111, fig=None, minground=0., maxground=2000, minalt=0,
                    maxalt=500, Re=6371., nyticks=5, nxticks=4):
    """Create curved axes in ground-range and altitude

    Parameters
    ----------
    rect : Optional[int]
        subplot spcification
    fig : Optional[pylab.figure object]
        (default to gcf)
    minground : Optional[float]

    maxground : Optional[int]
        maximum ground range [km]
    minalt : Optional[int]
        lowest altitude limit [km]
    maxalt : Optional[int]
        highest altitude limit [km]
    Re : Optional[float] 
        Earth radius in kilometers
    nyticks : Optional[int]
        Number of y axis tick marks; default is 5
    nxticks : Optional[int]
        Number of x axis tick marks; deafult is 4

    Returns
    -------
    ax : matplotlib.axes object
        containing formatting
    aax : matplotlib.axes object
        containing data

    Example
    -------
        import numpy as np
        ax, aax = curvedEarthAxes()
        th = np.linspace(0, ax.maxground/ax.Re, 50)
        r = np.linspace(ax.Re+ax.minalt, ax.Re+ax.maxalt, 20)
        Z = exp( -(r - 300 - ax.Re)**2 / 100**2 ) * np.cos(th[:, np.newaxis]/th.max()*4*np.pi)
        x, y = np.meshgrid(th, r)
        im = aax.pcolormesh(x, y, Z.T)
        ax.grid()

    written by Sebastien, 2013-04

    """
    from matplotlib.transforms import Affine2D, Transform
    import mpl_toolkits.axisartist.floating_axes as floating_axes
    from matplotlib.projections import polar
    from mpl_toolkits.axisartist.grid_finder import FixedLocator, DictFormatter
    import numpy as np
    from pylab import gcf

    ang         = maxground / Re
    minang      = minground / Re
    angran      = ang - minang
    angle_ticks = [(0, "{:.0f}".format(minground))]
    while angle_ticks[-1][0] < angran:
        tang = angle_ticks[-1][0] + 1./nxticks*angran
        angle_ticks.append((tang, "{:.0f}".format((tang-minang)*Re)))

    grid_locator1   = FixedLocator([v for v, s in angle_ticks])
    tick_formatter1 = DictFormatter(dict(angle_ticks))

    altran      = float(maxalt - minalt)
    alt_ticks   = [(minalt+Re, "{:.0f}".format(minalt))]
    while alt_ticks[-1][0] < Re+maxalt:
        alt_ticks.append((altran / float(nyticks) + alt_ticks[-1][0], 
                          "{:.0f}".format(altran / float(nyticks) +
                                          alt_ticks[-1][0] - Re)))
    _ = alt_ticks.pop()
    grid_locator2   = FixedLocator([v for v, s in alt_ticks])
    tick_formatter2 = DictFormatter(dict(alt_ticks))

    tr_rotate       = Affine2D().rotate(np.pi/2-ang/2)
    tr_shift        = Affine2D().translate(0, Re)
    tr              = polar.PolarTransform() + tr_rotate

    grid_helper = \
        floating_axes.GridHelperCurveLinear(tr, extremes=(0, angran, Re+minalt,
                                                          Re+maxalt),
                                            grid_locator1=grid_locator1,
                                            grid_locator2=grid_locator2,
                                            tick_formatter1=tick_formatter1,
                                            tick_formatter2=tick_formatter2,)

    if not fig: fig = gcf()
    ax1 = floating_axes.FloatingSubplot(fig, rect, grid_helper=grid_helper)

    # adjust axis
    ax1.axis["left"].label.set_text(r"Alt. [km]")
    ax1.axis["bottom"].label.set_text(r"Ground range [km]")
    ax1.invert_xaxis()

    ax1.minground   = minground
    ax1.maxground   = maxground
    ax1.minalt      = minalt
    ax1.maxalt      = maxalt
    ax1.Re          = Re

    fig.add_subplot(ax1, transform=tr)

    # create a parasite axes whose transData in RA, cz
    aux_ax          = ax1.get_aux_axes(tr)

    # for aux_ax to have a clip path as in ax
    aux_ax.patch    = ax1.patch

    # but this has a side effect that the patch is drawn twice, and possibly
    # over some other artists. So, we decrease the zorder a bit to prevent this.
    ax1.patch.zorder=0.9

    return ax1, aux_ax

def extract_rx_power(rt_dcts,param='rx_power_dB'):
    """
    Extract the receive power and times from a list of rt_dcts.
    """
    params  = []
    params.append('rx_power_0_dB')
    params.append('rx_power_dB')
    params.append('rx_power_O_dB')
    params.append('rx_power_X_dB')

    rx_pwrs = []
    dates   = []
    for rt_dct in rt_dcts:
        tmp         = {}
        date        = dates.append(rt_dct.get('date'))

        srd         = rt_dct.get('srch_ray_data')
        for param in params:
            if srd is not None:
                inx         = srd.ground_range.argmax()
#                xx          = [srd.ground_range[inx]]
                tmp[param]  = (srd[param])[inx]
            else:
                tmp[param]  = np.nan

        rx_pwrs.append(tmp)

    rx_pwrs = pd.DataFrame(rx_pwrs,index=dates)
    return rx_pwrs.sort_index()

def get_event_fname(sTime,eTime,freq,tx_call=None,rx_call=None,
        tx_lat=None,tx_lon=None,rx_lat=None,rx_lon=None,):

    sTime_str   = sTime.strftime('%Y%m%d.%H%M')
    eTime_str   = eTime.strftime('%Y%m%d.%H%M')

    date_str    = sTime_str + '-' + eTime_str + 'UT'

    freq_str    = '{:.0f}'.format(freq*1000)+'kHz'

    if tx_call is None:
        tx_s    = 'tx'+lat_lon_fname(tx_lat,tx_lon)
    else:
        tx_s    = 'tx'+tx_call

    if rx_call is None:
        rx_s    = 'rx'+lat_lon_fname(rx_lat,rx_lon)
    else:
        rx_s    = 'rx'+rx_call

    fname           = '_'.join([date_str,freq_str,tx_s,rx_s])
    return fname

def lat_lon_fname(lat,lon):
    """
    Returns a string in the form of 000.000N000.000E
    for a given lat/lon pair.
    """
    if lat < 0:
        NS  = 'S'
    else:
        NS  = 'N'
    lat_s   = '{:07.3f}{}'.format(np.abs(lat),NS)

    if lon > 180: lon = lon - 360.

    if lon < 0:
        EW  = 'W'
    else:
        EW  = 'E'
    lon_s   = '{:07.3f}{}'.format(np.abs(lon),EW)

    ret     = ''.join([lat_s,lon_s])
    return ret

def plot_rx_power_timeseries(rt_dcts,sTime=None,eTime=None,output_dir='output',fname=None):
    """
    Plot RX Power Time Series
    """
    rx_power    = extract_rx_power(rt_dcts)

    freq        = rt_dcts[0].get('freq')
    tx_lat      = rt_dcts[0].get('tx_lat')
    tx_lon      = rt_dcts[0].get('tx_lon')
    rx_lat      = rt_dcts[0].get('rx_lat')
    rx_lon      = rt_dcts[0].get('rx_lon')

    if sTime is None or eTime is None:
        dates   = [x['date'] for x in rt_dcts]
        sTime   = min(dates)
        eTime   = max(dates)

    fig         = plt.figure(figsize=(10,6.5))
    ax          = fig.add_subplot(1,1,1)
    param       = 'rx_power_dB'
    xx          = rx_power.index
    yy          = rx_power[param]
    ax.plot(xx,yy,marker='.')
    ax.set_xlim(sTime,eTime)

    fmt = mdates.DateFormatter('%d %b\n%H:%M')
    ax.xaxis.set_major_formatter(fmt)
    
    for xtl in ax.xaxis.get_ticklabels():
        xtl.set_rotation(70)
#        xtl.set_verticalalignment('top')
        xtl.set_horizontalalignment('center')
        xtl.set_fontsize('large')
        xtl.set_fontweight('bold')

    for ytl in ax.yaxis.get_ticklabels():
        ytl.set_fontsize('large')
        ytl.set_fontweight('bold')

    title       = []
    date_s      = ' - '.join([sTime.strftime('%Y %b %d %H:%M UT'),eTime.strftime('%Y %b %d %H:%M UT')])
    tx_lat_s    = '{:0.2f}'.format(tx_lat) + r'$^{\circ}$N'
    tx_lon_s    = '{:0.2f}'.format(tx_lon) + r'$^{\circ}$E'
    rx_lat_s    = '{:0.2f}'.format(rx_lat) + r'$^{\circ}$N'
    rx_lon_s    = '{:0.2f}'.format(rx_lon) + r'$^{\circ}$E'
    freq_s      = '{:0.3f} MHz'.format(freq)
    title.append(date_s)
    title.append('TX: {}, {}; RX: {}, {}, Frequency: {}'.format(tx_lat_s,tx_lon_s,rx_lat_s,rx_lon_s,freq_s))

    tx_call = rt_dcts[0].get('tx_call')
    rx_call = rt_dcts[0].get('rx_call')
    if tx_call is not None and rx_call is not None:
        title.append('TX: {}; RX: {}'.format(tx_call,rx_call))

    ax.set_title('\n'.join(title))

    ax.set_xlabel('Time [UT]')
#    ax.set_ylabel(param)
    ax.set_ylabel('Predicted RX dB')

    fig.tight_layout()
    if fname is None:
        fname       = 'rxPwr_'+rt_dcts[0].get('event_fname','0')

    fpath       = os.path.join(output_dir,fname+'.png')
    fig.savefig(fpath,bbox_inches='tight')
    plt.close(fig)

if __name__ == '__main__':
    multiproc   = False
    # Generate a dictionary of parameters to send to MATLAB.
#    date    = datetime.datetime(2017,2,2,21,53)
#
#    tx_lat                  =   44.425  # W7WHO
#    tx_lon                  = -121.238  # W7WHO
#    rx_lat                  =   40.907  # Jenny Jump
#    rx_lon                  =  -74.926  # Jenny Jump

    sTime                   = datetime.datetime(2014,11,3)
    eTime                   = datetime.datetime(2014,11,4)
    interval_time           = datetime.timedelta(minutes=5)

#    sTime                   = datetime.datetime(2014,11,3,12)
#    eTime                   = datetime.datetime(2014,11,3,15)
#    interval_time           = datetime.timedelta(hours=1)
    freq                    =   21.150
    tx_call                 =   'YV5B'
    tx_lat                  =    9.096      # YV5B
    tx_lon                  =  -67.824      # YV5B
    rx_call                 =   'KM3T'
    rx_lat                  =   42.821      # KM3T
    rx_lon                  =  -71.606      # KM3T

    event_fname             = get_event_fname(sTime,eTime,freq,tx_call,rx_call)

    # Prepare output directory.
    base_dir        = os.path.join('output',event_fname)
    rx_ts_dir       = base_dir
    rt_dir          = os.path.join(base_dir,'ray_trace')
    prep_dirs({0:base_dir,1:rx_ts_dir,2:rt_dir},clear_output_dirs=True)
    pkl_dir     = 'pkl'
    prep_dirs({0:pkl_dir},clear_output_dirs=False,php_viewers=False)

    run_dct = {}
    run_dct['tx_call']      = tx_call
    run_dct['tx_lat']       = tx_lat
    run_dct['tx_lon']       = tx_lon
    run_dct['rx_call']      = rx_call
    run_dct['rx_lat']       = rx_lat
    run_dct['rx_lon']       = rx_lon
    run_dct['freq']         = freq
    run_dct['event_fname']  = event_fname
    run_dct['pkl_dir']      = pkl_dir
    run_dct['output_dir']   = rt_dir
    run_dct['use_cache']    = False

    run_lst                 = gen_run_list(sTime,eTime,interval_time,**run_dct)

    if multiproc:
        pool    = multiprocessing.Pool(2)
        rt_dcts = pool.map(run_ray_trace,run_lst)
        pool.close()
        pool.join()
    else:
        rt_dcts = []
        for this_dct in run_lst:
            rt_dct  = run_ray_trace(this_dct)
            rt_dcts.append(rt_dct)

    if multiproc:
        pool    = multiprocessing.Pool()
        pool.map(plot_raytrace_and_power,rt_dcts)
        pool.close()
        pool.join()
    else:
        for rt_dct in rt_dcts:
            plot_raytrace_and_power(rt_dct)

    plot_rx_power_timeseries(rt_dcts,sTime,eTime,output_dir=rx_ts_dir)
    rt_rx_pwr_to_csv(rt_dcts,output_dir=rx_ts_dir)

    import ipdb; ipdb.set_trace()
