#This code is used to generate the RBN-GOES map for the Space Weather feature article.
import datetime
import rbn_lib

def gen_time_list(sTime,eTime,cTime_dt=datetime.timedelta(minutes=60)):
    """ Generate a list of datetime.datetimes spaced cTime_dt apart.  """
    cTimes  = [sTime]
    while cTimes[-1] <= eTime:
        cTimes.append(cTimes[-1]+cTime_dt)

    return cTimes

# 2014 Nov Sweepstakes
#seTimes = ( datetime.datetime(2014,11,8), datetime.datetime(2014,11,9) )

# 2015 Nov Sweepstakes
seTimes = ( datetime.datetime(2015,11,7), datetime.datetime(2015,11,10) )

# 2016 CQ WPX CW
#seTimes = ( datetime.datetime(2016,5,28,18), datetime.datetime(2016,5,29,6) )

# Script processing begins here. ###############################################
cTimes      = gen_time_list(*seTimes)

for cTime in cTimes:
    eTime = sTime + datetime.timedelta(minutes=15)

    print ''
    print '################################################################################'
    print 'Downloading and Geolocating RBN Data: {0} - {1}'.format(sTime.strftime('%d %b %Y %H%M UT'),eTime.strftime('%d %b %Y %H%M UT'))

    rbn_df  = rbn_lib.read_rbn(sTime,eTime,data_dir='data/rbn')

    # Figure out how many records properly geolocated.
    good_loc        = rbn_df.dropna(subset=['dx_lat','dx_lon','de_lat','de_lon'])
    good_count_map  = good_loc['callsign'].count()
    total_count_map = len(rbn_df)
    good_pct_map    = float(good_count_map) / total_count_map * 100.

    good_count      += good_count_map
    total_count     += total_count_map

    print 'Geolocation success: {0:d}/{1:d} ({2:.1f}%)'.format(good_count_map,total_count_map,good_pct_map)
