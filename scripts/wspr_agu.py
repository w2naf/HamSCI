#!/usr/bin/env python
#Codes to make graphs for AGU. Also to test wspr_lib prior to AGU 2016

from hamsci import wspr_lib
#from hamsci import gridsquare
from hamsci import rbn_lib

#def find_pair(df):
#
#    tx  =   df['call_sign'].unique()
#    rx  =   df['reciever'].unique()
#
#    rx_count=0
#    tx_count=0
##    for this_tx in tx:
##        for this_rx in rx:
##            tx
#
#def plot_wspr_snr(df, by_pwr=True):
#        """Scatter Plot WSPR SNR reports
#
#        Parameters
#        ----------
#        new_data_set : str
#            Name for the new data_set object.
#        comment : str
#            Comment describing the new data_set object.
#
#        Returns
#        -------
#        new_data_set_obj : data_set 
#            Copy of the original data_set with new name and history entry.
#
#        Written by Magdalina L. Moses, Fall 2016
#        """
#        #Will likely need to bin powers, but need to check how far appart the different powers are
#        if by_pwr:
#
#
#
#
#    return fig


if __name__ == '__main__':
    import datetime

    sTime       = datetime.datetime(2014,2,1)
    eTime       = datetime.datetime(2014,2,28)

    sTime       = datetime.datetime(2016,8,27)
    eTime       = datetime.datetime(2016,8,28)
    sTime       = datetime.datetime(2016,11,11)
    eTime       = datetime.datetime(2016,11,18)

    sTime       = datetime.datetime(2016,11,1,0)
    eTime       = datetime.datetime(2016,11,1,2)
    data_dir    = 'data/wspr' 

    df = wspr_lib.read_wspr(sTime,eTime,data_dir)

    #Select only stations within two lat/lon areas (near VT and NJIT)
    #   K2MFF 'FN20vr' (40.7429,-74.1770)
    #   KM4EGE 'EM97tf' 

    #For simplicity in this proof-of-concept application, only chose stations in the following gridsquares:
    #   FN20 and FN21 (or FN30 and FN31)
    #   Need to select from wider area for southern station 
    #   


    #Find the pairs of stations with most links between them

    #Filter to only include links between two specified stations
    df  =   wspr_lib.select_pair(df, stations)
    
    fig=plot_wspr_snr(df)
    import ipdb; ipdb.set_trace()

