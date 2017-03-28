import datetime

import numpy as np
import pandas as pd
import matlab.engine

#eng = matlab.engine.start_matlab('-desktop')
#eng = matlab.engine.start_matlab()

class RxTxRaytace(object):
    def __init__(self,date,frequency,
           tx_call = None, tx_lat = None, tx_lon = None,
           rx_call = None, rx_lat = None, rx_lon = None,
           R12

        eng = matlab.engine.start_matlab()

        frequency               = run_dct.get('freq')
        R12                     = run_dct.get('R12',28)
        doppler_flag            = run_dct.get('doppler_flag',1)
        irregs_flag             = run_dct.get('irregs_flag',0)
        kp                      = run_dct.get('kp',0)
        fname                   = run_dct.get('fname')
        event_fname             = run_dct.get('event_fname')
        pkl_dir                 = run_dct.get('pkl_dir')
        output_dir              = run_dct.get('output_dir')
        use_cache               = run_dct.get('use_cache',False)

        tx_power                = run_dct.get('tx_power',1.)    # Watts
        gain_tx_db              = run_dct.get('gain_tx_db',1.)  # dB
        gain_rx_db              = run_dct.get('gain_rx_db',1.)  # dB

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

        mld['start_range']      = 0.                    # start range for ionospheric grid (km)
        mld['max_range']        = 10000.                # maximum range for sampling the ionosphere (km)
        mld['num_range']        = 201.                  # number of ranges (must be < 2000)

        mld['start_height']     = 0.                    # start height for ionospheric grid (km)
        mld['height_inc']       = 3.                    # height increment (km)
        mld['num_heights']      = 200.                  # number of  heights (must be < 2000)

        # call raytrace for a fan of rays
        mld['freq']             = float(frequency)      # frequency (MHz)
        mld['tol']              = 1e-7                  # ODE tolerance
        mld['nhops']            = 3.

        elev_step               = 0.5
        mld['elev_step']        = elev_step
        mld['elevs']            = matlab.double(np.arange(2,60,elev_step,
                                    dtype=np.double).tolist())

        for key,val in mld.items():
            eng.workspace[key] = val

        date_str    = date.strftime('%Y%m%d.%H%M')
        freq_str    = '{:.0f}'.format(mld['freq']*1000)

        tx_lls      = 'tx'+lat_lon_fname(tx_lat,tx_lon)
        rx_lls      = 'rx'+lat_lon_fname(rx_lat,rx_lon)
        if fname is None:
            fname           = '_'.join([date_str,event_fname[30:]])

        pkl_fname   = os.path.join(pkl_dir,'{}.p.bz2'.format(fname))
        if not use_cache or not os.path.exists(pkl_fname):
            eng.ray_tracker_2d(nargout=0)

            dr          = eng.workspace['range_inc']
            r_0         = eng.workspace['start_range']
            r_1         = eng.workspace['max_range']
            rngs        = np.arange(r_0,r_1+dr,dr)

            dh          = eng.workspace['height_inc']
            h_0         = eng.workspace['start_height']
            h_1         = eng.workspace['num_heights'] * dh
            heights     = np.arange(h_0,h_1,dh)

            rt_dct = {}
            rt_dct['event_fname']       = event_fname
            rt_dct['fname']             = fname
            rt_dct['output_dir']        = output_dir
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

            # Compute Received Power ###############
            rt_dct['tx_power']      = tx_power
            rt_dct['gain_tx_db']    = gain_tx_db
            rt_dct['gain_rx_db']    = gain_rx_db

            with bz2.BZ2File(pkl_fname,'w') as fl:
                pickle.dump(rt_dct,fl)
        else:
            with bz2.BZ2File(pkl_fname,'r') as fl:
                rt_dct      = pickle.load(fl)


        view    = []
    #    view.append('Doppler_shift') 
    #    view.append('Doppler_spread') 
    #    view.append('TEC_path') 
    #    view.append('apogee') 
        view.append('deviative_absorption') 
        view.append('effective_range') 
    #    view.append('final_elev') 
    #    view.append('frequency') 
    #    view.append('geometric_path_length') 
    #    view.append('gnd_rng_to_apogee') 
        view.append('ground_range') 
    #    view.append('group_range') 
        view.append('initial_elev') 
        view.append('lat') 
        view.append('lon') 
    #    view.append('nhops_attempted') 
    #    view.append('phase_path') 
    #    view.append('plasma_freq_at_apogee') 
        view.append('ray_id') 
        view.append('ray_label') 
        view.append('rx_power_0_dB') 
        view.append('rx_power_O_dB') 
        view.append('rx_power_X_dB') 
        view.append('rx_power_dB') 
        view.append('virtual_height') 
    ##    view.append('power_dbw')

        rd  = rt_dct['ray_data']
        rdv = rd[view]

        return rt_dct
