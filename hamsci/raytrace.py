import os
import datetime

import numpy as np
import pandas as pd

import matlab.engine

import threading

class MatlabInstance(threading.local):
    def __init__(self):
        self.eng = matlab.engine.start_matlab()

mi  = MatlabInstance()

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

        eng = mi.eng
#        eng = matlab.engine.start_matlab()
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
        import ipdb; ipdb.set_trace()

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

    def get_event_name(self):
        md               = self.rt_dct['metadata']
        event_name       = get_event_name(md['date'],None,md['freq'],
                            md['tx_call'], md['rx_call'],
                            md['tx_lat'],  md['tx_lon'],
                            md['rx_lat'],  md['rx_lon'])
        return event_name


def get_event_name(sTime,eTime=None,freq=None,tx_call=None,rx_call=None,
        tx_lat=None,tx_lon=None,rx_lat=None,rx_lon=None,):

    sTime_str   = sTime.strftime('%Y%m%d.%H%M')

    if eTime is not None:
        eTime_str   = eTime.strftime('-%Y%m%d.%H%M')
    else:
        eTime_str   = ''

    date_str    = sTime_str + eTime_str + 'UT'

    if freq is not None:
        freq_str    = '{:.0f}'.format(freq*1000)+'kHz'
    else:
        freq_str    = ''

    if tx_call is None:
        tx_s    = 'tx'+lat_lon_str(tx_lat,tx_lon)
    else:
        tx_s    = 'tx'+tx_call

    if rx_call is None:
        rx_s    = 'rx'+lat_lon_str(rx_lat,rx_lon)
    else:
        rx_s    = 'rx'+rx_call

    fname           = '_'.join([date_str,freq_str,tx_s,rx_s])
    return fname

def lat_lon_str(lat,lon):
    """
    Returns a string in the form of 000.000N000.000E
    for a given lat/lon pair.
    """
    if lat < 0:
        NS  = 'S'
    else:
        NS  = 'N'
    lat_s   = '{:07.3f}{}'.format(abs(lat),NS)

    if lon > 180: lon = lon - 360.

    if lon < 0:
        EW  = 'W'
    else:
        EW  = 'E'
    lon_s   = '{:07.3f}{}'.format(abs(lon),EW)

    ret     = ''.join([lat_s,lon_s])
    return ret

def extract_rx_power(rt_objs,param='rx_power_dB'):
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
    for rt_obj in rt_objs:
        tmp         = {}
        date        = dates.append(rt_obj.rt_dct['metadata'].get('date'))

        srd         = rt_obj.rt_dct['raytrace'].get('connecting_ray_data')
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

def rt_rx_pwr_to_csv(rt_objs,print_header=True,output_file='output.csv'):
    """
    Writes Ray Trace Dictionary RX Power to CSV.
    """

    md0         = rt_objs[0].rt_dct['metadata']
    event_name  = rt_objs[0].get_event_name()

    keys    = []
#    keys.append('event_fname')
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
    keys.append('gain_rx_db')
    keys.append('tx_power')

#    keys.append('Doppler_shift') 
#    keys.append('Doppler_spread') 
#    keys.append('TEC_path') 
#    keys.append('apogee') 
#    keys.append('deviative_absorption') 
#    keys.append('effective_range') 
#    keys.append('final_elev') 
#    keys.append('frequency') 
#    keys.append('geometric_path_length') 
#    keys.append('gnd_rng_to_apogee') 
#    keys.append('ground_range') 
#    keys.append('group_range') 
#    keys.append('initial_elev') 
#    keys.append('lat') 
#    keys.append('lon') 
#    keys.append('nhops_attempted') 
#    keys.append('phase_path') 
#    keys.append('plasma_freq_at_apogee') 
#    keys.append('ray_id') 
#    keys.append('ray_label') 
#    keys.append('rx_power_0_dB') 
#    keys.append('rx_power_O_dB') 
#    keys.append('rx_power_X_dB') 
#    keys.append('rx_power_dB') 
#    keys.append('virtual_height') 
#    keys.append('power_dbw')

    with open(output_file,'w') as fl:
        if print_header:
            fl.write('# PHaRLAP Predicted Receive Power')
            fl.write('#\n')
            fl.write('## Metadata ####################################################################\n')

            line    ='# event_name: {}'.format(event_name)
            fl.write(line)
            for key in keys:
                val     = md0.get(key)
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

    rx_power    = extract_rx_power(rt_objs)
    rx_power.to_csv(output_file,mode='a')

    return output_file

