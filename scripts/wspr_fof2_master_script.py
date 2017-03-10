#!/usr/bin/env python
#This code is used to generate the RBN-GOES map for the Space Weather feature article.
#This code is the master file for wspr data processing to find fof2

import sys
import os
import datetime
import multiprocessing
import pickle
import shutil

import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt

import numpy as np
import pandas as pd

import hamsci
from hamsci import wspr_lib
from hamsci import handling
# Set default gridsquare precision
gridsquare_precision = 4

def loop_info(map_sTime,map_eTime):
    print ''
    print '################################################################################'
    print 'Plotting WSPR Map: {0} - {1}'.format(map_sTime.strftime('%d %b %Y %H%M UT'),map_eTime.strftime('%d %b %Y %H%M UT'))

def update_run_list(run_list,**kwargs):
    """
    Returns a copy of a list of dictionaries
    with new/updated items in each dictionary.
    """

    new_list    = []
    for item in run_list:
        item_copy   = item.copy() 
        item_copy.update(kwargs)
        new_list.append(item_copy)

    return new_list

### Create WSPR Object Codes #################################
def create_wspr_obj_dct_wrapper(run_dct):
    create_wspr_obj(**run_dct)

def get_wspr_obj_path(output_dir,reflection_type,sTime,eTime):
    filename    = '{}-{:%Y%m%d.%H%M}-{:%Y%m%d.%H%M}.p'.format(reflection_type,sTime,eTime)
    output_dir  = os.path.join(wspr_fof2_dir,reflection_type)
    filepath    = os.path.join(output_dir,filename)
    return filepath

def create_wspr_obj(sTime,eTime,
        llcrnrlon=-180., llcrnrlat=-90, urcrnrlon=180., urcrnrlat=90.,
        call_filt_de = None, call_filt_dx = None,
        reflection_type         = 'sp_mid',
        wspr_fof2_dir            = 'data/wspr_fof2',
        **kwargs):

    filepath = get_wspr_obj_path(wspr_fof2_dir, reflection_type,sTime,eTime)
    output_dir  = os.path.split(filepath)[0]
    handling.prepare_output_dirs({0:output_dir},clear_output_dirs=False)

    #Create wspr object
    wspr_obj = wspr_lib.WsprObject(sTime,eTime) 

    #find lat/lon from gridsquares
    wspr_obj.active.dxde_gs_latlon()
    #Filter Path 
    wspr_obj.active.filter_pathlength(500.)

    #Calculate Reflection points
    wspr_obj.active.calc_reflection_points(reflection_type=reflection_type)
    wspr_obj.active.grid_data(gridsquare_precision)

    #filter lat/lon and calls 
    latlon_bnds = {'llcrnrlat':llcrnrlat,'llcrnrlon':llcrnrlon,'urcrnrlat':urcrnrlat,'urcrnrlon':urcrnrlon}
    wspr_obj.active.latlon_filt(**latlon_bnds)
    wspr_obj.active.filter_calls(call_filt_de,call_type='de')
    wspr_obj.active.filter_calls(call_filt_dx,call_type='dx')

    #Gridsquare data 
    wspr_obj.active.compute_grid_stats()
    with open(filepath,'wb') as fl:
        pickle.dump(wspr_obj,fl)

### Mapping Codes ############################################
def wspr_map(sTime,eTime,
        llcrnrlon=-180., llcrnrlat=-90, urcrnrlon=180., urcrnrlat=90.,
        call_filt_de = None, call_filt_dx = None,
        reflection_type         = 'sp_mid',
        plot_de                 = True,
        plot_dx                 = True,
        plot_midpoints          = True,
        plot_paths              = False,
        plot_ncdxf              = False,
        plot_stats              = True,
        plot_legend             = True,
        overlay_gridsquares     = True,
        overlay_gridsquare_data = True,
        gridsquare_data_param   = 'f_max_MHz',
        fname_tag               = None,
        output_dir = 'output/wspr', 
        wspr_fof2_dir = 'data/wspr_fof2'):
    """
    Creates 
    """
    filename    = 'wspr_map-{:%Y%m%d.%H%M}-{:%Y%m%d.%H%M}.png'.format(sTime,eTime)
    filepath    = os.path.join(output_dir,filename)

    #Create wspr object
    wspr_obj = wspr_lib.WsprObject(sTime,eTime) 
    #find lat/lon from gridsquares
    wspr_obj.active.dxde_gs_latlon()
    wspr_obj.active.filter_pathlength(500.)
    wspr_obj.active.calc_reflection_points(reflection_type=reflection_type)
    latlon_bnds = {'llcrnrlat':llcrnrlat,'llcrnrlon':llcrnrlon,'urcrnrlat':urcrnrlat,'urcrnrlon':urcrnrlon}
    wspr_obj.active.latlon_filt(**latlon_bnds)

    # Go plot!! ############################ 
    ## Determine the aspect ratio of subplot.
    xsize       = 15.0
    ysize       = 6.5
    nx_plots    = 1
    ny_plots    = 1

    rcp = mpl.rcParams
    rcp['axes.titlesize']     = 'large'
#    rcp['axes.titleweight']   = 'bold'

    fig        = plt.figure(figsize=(nx_plots*xsize,ny_plots*ysize))
    ax0        = fig.add_subplot(1,1,1)

    map_obj=wspr_lib.WsprMap(wspr_obj, ax=ax0,nightshade=term[0], solar_zenith=term[1], default_plot=False)
    if plot_de:
        map_obj.plot_de()
    if plot_dx:
        map_obj.plot_dx()
    if plot_midpoints:
        map_obj.plot_midpoints()
    if plot_paths:
        map_obj.plot_paths()
    if plot_ncdxf:
        map_obj.plot_ncdxf()
    if plot_stats:
        map_obj.plot_link_stats()
    if plot_legend:
        map_obj.plot_band_legend(band_data=map_obj.band_data)

    if overlay_gridsquares:
        map_obj.overlay_gridsquares()
    if overlay_gridsquare_data:
        wspr_obj.active.grid_data(gridsquare_precision)
        wspr_obj.active.compute_grid_stats()
        map_obj.overlay_gridsquare_data(param=gridsquare_data_param)

    ecl         = hamsci.eclipse.Eclipse2017()
    line, lbl   = ecl.overlay_umbra(map_obj.m,color='k')
    handles     = [line]
    labels      = [lbl]

    fig.savefig(filepath,bbox_inches='tight')
    plt.close(fig)

def gen_map_run_list(sTime,eTime,integration_time,interval_time,**kw_args):
    dct_list    = []
    this_sTime  = sTime
    while this_sTime+integration_time <= eTime:
        this_eTime   = this_sTime + integration_time

        tmp = {}
        tmp['sTime']    = this_sTime
        tmp['eTime']    = this_eTime
        tmp.update(kw_args)
        dct_list.append(tmp)

#        this_sTime      = this_sTime + interval_time
        this_sTime  = this_eTime

    return dct_list

def wspr_map_dct_wrapper(run_dct):
    wspr_map(**run_dct)

### CSV Save Code #################################
def write_csv_dct(run_dct):
    """
    Dictionary wrapper for write_csv() to help with
    pool multiprocessing.
    """

    csv_path    = write_csv(**run_dct)
    return csv_path

def write_csv(sTime,eTime,reflection_type,output_dir,wspr_fof2_dir,data_set='active',dataframe='grid_data',
        print_header=True,**kwargs):
    """
    Write WSPR Obj data to a CSV File.
    """

    # Load in data.
    wspr_fof2_fp = get_wspr_obj_path(wspr_fof2_dir,reflection_type,sTime,eTime)
    with open(wspr_fof2_fp,'rb') as fl:
        wspr_obj = pickle.load(fl)

    ds          = getattr(wspr_obj,data_set)
    df          = getattr(ds,dataframe)

    # Prepare file names.
    data_set_name   = ds.metadata['data_set_name']
    filetag         = '{}.{}'.format(data_set_name,dataframe)

    output_dir      = os.path.join(output_dir,'csv',filetag)
    handling.prepare_output_dirs({0:output_dir},clear_output_dirs=False,php_viewers=False)

    csv_fname       = '{:%Y%m%d.%H%M}-{:%Y%m%d.%H%M}.{}.csv'.format(sTime,eTime,filetag)
    csv_path        = os.path.join(output_dir,csv_fname)

    with open(csv_path,'w') as fl:
        if print_header:
            fl.write('# Data Source: {!s}\n'.format(wspr_fof2_fp))
            fl.write('#\n')
            fl.write('## Data Set History ############################################################\n')

            keys = ds.history.keys()
            keys.sort()
            for key in keys:
                line = '# {!s}: {!s}\n'.format(key,ds.history[key])
                fl.write(line)

            fl.write('#\n')
            fl.write('## Data Set Metadata ###########################################################\n')

            keys = ds.metadata.keys()
            keys.sort()
            for key in keys:
                line = '# {!s}: {!s}\n'.format(key,ds.metadata[key])
                fl.write(line)

            fl.write('#\n')
            fl.write('## CSV Data ####################################################################\n')

    df.to_csv(csv_path,mode='a')
    return csv_path


### Main Code ###################
if __name__ == '__main__':
    multiproc   = False 
    create_wspr_objs=True
    test_wspr_objs=False
    gen_csv = False
    plot_maps = False

    plot_de                 = True
    plot_dx                 = False
    plot_midpoints          = False
    plot_paths              = False
    plot_ncdxf              = False
    plot_stats              = True
    plot_legend             = False
    overlay_gridsquares     = True
    overlay_gridsquare_data = True
#    gridsquare_data_param   = 'f_max_MHz'
    gridsquare_data_param   = 'foF2'
    fname_tag               = None
    #Initial WsprMap test code
    sTime = datetime.datetime(2016,11,1,22)
    eTime = datetime.datetime(2016,11,2,1)
    sTime = datetime.datetime(2016,11,1)
    eTime = datetime.datetime(2016,11,1,1)
    #CW Sweapstakes 2014
    sTime = datetime.datetime(2014, 11,1)
    eTime = datetime.datetime(2014, 11,4)
#    eTime = datetime.datetime(2014, 11,2)

#    eTime = datetime.datetime(2014, 11,2)
#    #CW Sweapstakes 2016
#    sTime = datetime.datetime(2016, 11,5)
#    eTime = datetime.datetime(2016, 11,7)
#    #Test CW Sweapstakes 2016
#    sTime = datetime.datetime(2016, 11,4)
#    eTime = datetime.datetime(2016, 11,7)


#    #Solar Flare Event
##    sTime = datetime.datetime(2016,5,13,15,5)
##    eTime = datetime.datetime(2016,5,13,15,21)
#    sTime = datetime.datetime(2013,5,13,15,5)
#    eTime = datetime.datetime(2013,5,13,17)

#    wspr_obj = wspr_lib.WsprObject(sTime,sTime+datetime.timedelta(minutes=15)) 

    term=[True, False]
    integration_time    = datetime.timedelta(minutes=15)
    interval_time       = datetime.timedelta(minutes=60)

    dct = {}
    dct.update({'llcrnrlat':20.,'llcrnrlon':-130.,'urcrnrlat':55.,'urcrnrlon':-65.})

    filt_type='sp_mid'
    filt_type='miller2015'
    reflection_type = filt_type

    map_sTime = sTime
#    map_eTime = map_sTime + datetime.timedelta(minutes = dt)
    map_eTime = map_sTime + interval_time
    map_eTime = map_sTime + integration_time

    #Create output directory based on start and end time and parameter ploted
    event_dir               = '{:%Y%m%d.%H%M}-{:%Y%m%d.%H%M}-{}'.format(sTime,eTime,reflection_type)
    output_dir              = os.path.join('output','wspr',event_dir)
    wspr_fof2_dir            = os.path.join('data','wspr_fof2',event_dir)

    dct['output_dir']       = output_dir
    dct['wspr_fof2_dir']     = wspr_fof2_dir
    dct['reflection_type']  = reflection_type

#    #Create output directory based on start and end time and parameter ploted
#    event_dir           = '{:%Y%m%d.%H%M}-{:%Y%m%d.%H%M}'.format(sTime,eTime)
##    filename    = '{}-{:%Y%m%d.%H%M}-{:%Y%m%d.%H%M}.png'.format(fname_tag,sTime,eTime)
#    if plot_midpoints:
#        if reflection_type == 'miller2015':
#            tag = 'refl_points'
##            output_dir = os.path.join('output','wspr','maps','refl_points',event_dir)
#        if reflection_type == 'sp_mid':
#            tag='midpoints'
#    if plot_paths: 
#        tag = 'paths'
##        output_dir          = os.path.join('output','wspr','maps','paths',event_dir)
#    if overlay_gridsquare_data:
#        if gridsquare_data_param   == 'foF2':
#            tag='fof2'
##            output_dir          = os.path.join('output','wspr','maps','fof2',event_dir)
#        else:
#            tag = 'gridsqs'
##            output_dir          = os.path.join('output','wspr','maps','gridsqs',event_dir)
#    output_dir          = os.path.join('output','wspr','maps',event_dir, tag)
#
##    output_dir          = os.path.join('output','wspr','maps','paths',event_dir)
##    output_dir          = os.path.join('output','wspr','maps','midpoints',event_dir)
##    output_dir          = os.path.join('output','wspr','maps','refl_points',event_dir)
##    output_dir          = os.path.join('output','wspr','maps','defaults_test','midpoints',event_dir)
##    output_dir          = os.path.join('output','wspr','maps','defaults_test','refl_points',event_dir)
#
    try:    # Create the output directory, but fail silently if it already exists
        os.makedirs(output_dir) 
    except:
        pass

# Create input parameter dictionary 
#    dct.update({'filt_type':filt_type})
    dct.update({'reflection_type':filt_type})
    dct.update({'output_dir':output_dir})
    dct.update({'plot_de':plot_de, 'plot_dx':plot_dx, 'plot_midpoints':plot_midpoints, 'plot_paths':plot_paths, 'plot_ncdxf': plot_ncdxf, 'plot_stats':plot_stats, 'plot_legend':plot_legend, 'overlay_gridsquares':overlay_gridsquares, 'overlay_gridsquare_data':overlay_gridsquare_data, 'gridsquare_data_param':gridsquare_data_param, 'fname_tag':fname_tag})

    #Generate list of input values for every interval to plot 
    run_list    = gen_map_run_list(sTime,eTime,integration_time,interval_time,**dct)
#    run_list    = gen_map_run_list(map_sTime,map_eTime,integration_time,interval_time,**dct)

    # Create WSPR Object ####
    if test_wspr_objs:
        #Check 
        for run_dct in run_list:
            if run_dct['sTime'] == datetime.datetime(2014,11,2,14,45):
                create_wspr_obj_dct_wrapper(run_dct)
    if create_wspr_objs:
        #Should check pathlegnth filter
        #Regular scripts
        if multiproc:
            pool = multiprocessing.Pool()
            pool.map(wspr_map_dct_wrapper,run_list)
            pool.close()
            pool.join()
        else:
            for run_dct in run_list:
                create_wspr_obj_dct_wrapper(run_dct)

    # Generate CSV Files ####
    if gen_csv:

        csv_requests = []
        csv_requests.append( {'data_set':'active', 'dataframe':'grid_data'} )
#        csv_requests.append( {'data_set':'active', 'dataframe':'grid_data','print_header':True} )
        csv_requests.append( {'data_set':'active', 'dataframe':'df'} )

        for csv_request in csv_requests:
            csv_list    = update_run_list(run_list,**csv_request)
            if multiproc:
                pool = multiprocessing.Pool()
                vals = pool.map(write_csv_dct,csv_list)
                pool.close()
                pool.join()
                csv_path = vals[-1]
            else:
                for csv_dct in csv_list:
                    csv_path = write_csv_dct(csv_dct)

            path        = os.path.split(csv_path)[0]
            shutil.make_archive(path, 'zip', path)

    # Plot Maps ####
    if plot_maps: 
        #Run map code
        if multiproc:
            pool = multiprocessing.Pool()
            pool.map(wspr_map_dct_wrapper,run_list)
            pool.close()
            pool.join()
        else:
            for run_dct in run_list:
                wspr_map_dct_wrapper(run_dct)


#    wspr_path_map(map_sTime, map_eTime, filt_type = filt_type, output_dir=output_dir, **dct)


#    wspr_map.fig.savefig('output/wspr/WSPR_map_test.png')

#Adapted test code 
#    sTime = datetime.datetime(2016,11,1,0)
#    eTime = datetime.datetime(2016,11,1,1)
#    term=[True, False]
#    dt=15
#    integration_time    = datetime.timedelta(minutes=15)
#    interval_time       = datetime.timedelta(minutes=20)
##    interval_time       = datetime.timedelta(minutes=60)
#
#    dct = {}
#    dct.update({'llcrnrlat':20.,'llcrnrlon':-130.,'urcrnrlat':55.,'urcrnrlon':-65., 'filt_type':'sp_mid'})
##    dct.update({'llcrnrlat':20.,'llcrnrlon':-130.,'urcrnrlat':55.,'urcrnrlon':-65., 'output_dir': 'output/wspr'})
#
#    map_sTime = sTime
##    map_eTime = map_sTime + datetime.timedelta(minutes = dt)
#    map_eTime = map_sTime + interval_time
#
#    run_list            = gen_map_run_list(map_sTime,map_eTime,integration_time,interval_time,**dct)
#    if multiproc:
#        pool = multiprocessing.Pool()
#        pool.map(wspr_map_dct_wrapper,run_list)
#        pool.close()
#        pool.join()
#    else:
#        for run_dct in run_list:
#            wspr_map_dct_wrapper(run_dct)
#
##    wspr_map.fig.savefig('output/wspr/WSPR_map_test2.png')
#
##    mymap = wspr_map(sTime = map_sTime, eTime = map_eTime)

#Other?
#    dct = {}
#    dct.update({'llcrnrlat':20.,'llcrnrlon':-130.,'urcrnrlat':55.,'urcrnrlon':-65.})
#
#    integration_time    = datetime.timedelta(minutes=15)
#    interval_time       = datetime.timedelta(minutes=60)
#
#    event_dir           = '{:%Y%m%d.%H%M}-{:%Y%m%d.%H%M}'.format(sTime,eTime)
#    output_dir          = os.path.join('output','maps',event_dir)

#General Test Code
#    sTime = datetime.datetime(2016,11,1)
#    wspr_obj = wspr_lib.WsprObject(sTime) 
#    wspr_obj.active.calc_reflection_points(reflection_type='miller2015')
#    #For iPython
##    os.system('sudo python setup.py install')
