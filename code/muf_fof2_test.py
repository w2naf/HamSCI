#!/usr/bin/env python

import numpy as np
import datetime
import rbn_lib 

sTime=datetime.datetime(2015,8,21,18,00,00)
(de_lat, de_lon, dx_lat, dx_lon)=(44.9, -123.0, 43.74, -110.73)
 
 midlat, midlon, dist, mdist=rbn_lib.path_mid(de_lat, de_lon, dx_lat, dx_lon)
 h,outf,oarr=rbn_lib.get_hmF2(sTime=sTime, lat=midlat, lon=midlon)
 fof2, theta,h = rbn_lib.rbn_fof2(sTime, 13.4,de_lat, de_lon, dx_lat, dx_lon)
 foF2=np.sqrt(oarr[0]*(1e-2)**3)*(9e3)/1e6
 muf=(foF2/np.cos(theta))
