#!/usr/bin/env python
#This script downloads and geolocates RBN data.

import datetime
from hamsci import rbn_lib

def gen_time_list(sTime,eTime,cTime_dt=datetime.timedelta(minutes=60)):
    """ Generate a list of datetime.datetimes spaced cTime_dt apart.  """
    cTimes  = [sTime]
    while cTimes[-1] < eTime:
        next_time = cTimes[-1]+cTime_dt
        if next_time >= eTime:
            break
        cTimes.append(next_time)

    return cTimes

cTimes  = []
## 2015 Nov Sweepstakes
#seTimes = ( datetime.datetime(2015,11,7), datetime.datetime(2015,11,10) )
#cTimes      += gen_time_list(*seTimes)
#
## 2014 Nov Sweepstakes
#seTimes = ( datetime.datetime(2014,11,1), datetime.datetime(2014,11,4) )
#cTimes      += gen_time_list(*seTimes)

# 2016 CQ WPX CW
seTimes = ( datetime.datetime(2016,1,1), datetime.datetime(2016,6,15) )
cTimes      += gen_time_list(*seTimes)

# Script processing begins here. ###############################################

for sTime in cTimes:

    eTime = sTime + datetime.timedelta(minutes=59)

    print ''
    print '################################################################################'
    print 'Downloading and Geolocating RBN Data: {0} - {1}'.format(sTime.strftime('%d %b %Y %H%M UT'),eTime.strftime('%d %b %Y %H%M UT'))

    qrz_call    = 'w2naf'
    qrz_passwd  = "hamscience"
    rbn_df      = rbn_lib.read_rbn(sTime,eTime,data_dir='data/rbn',
                    qrz_call=qrz_call,qrz_passwd=qrz_passwd)

    # Figure out how many records properly geolocated.
    good_loc        = rbn_df.dropna(subset=['dx_lat','dx_lon','de_lat','de_lon'])
    good_count      = good_loc['callsign'].count()
    total_count     = len(rbn_df)
    if total_count == 0:
        print "No call signs geolocated."
    else:
        good_pct        = float(good_count) / total_count * 100.
        print 'Geolocation success: {0:d}/{1:d} ({2:.1f}%)'.format(good_count,total_count,good_pct)
