#!usr/bin/env python
#Functions to take davitpy code and output certain common values

import numpy as np

def NmF2_to_fof2(NmF2, Nunits='m', funits='kHz'):
    #Convert NmF2 (m-3) to fof2 (kHz)
#    inputs:
#        [NmF2]: NmF2 (default m-3)
#        [Nunits]:the base unit of the input NmF2 (ie if NmF2 input is in m-3, Nunits='m' and if NmF2 in cm-3, put Nunits='cm')

    #Convert NmF2 to cm-3
    if Nunits == 'm':
        Ncm=NmF2*(1e-2)**3
    elif Nunits == 'mm':
        Ncm=NmF2*(1e-3)**3
    elif Nunits != 'cm':
        print 'Error'

    #Calc fof2
    if funits == 'kHz':
        fof2=np.sqrt(Ncm)*(9e3)/1e3
    elif funits == 'MHz':
        fof2=np.sqrt(Ncm)*(9e3)/1e6
    elif funits == 'Hz':
        fof2=np.sqrt(Ncm)*(9e3)

    return fof2


def get_iri(sTime,lat, lon, ssn=None, output=['hmF2']):
    """Get hmF2 data for midpoint of RBN link
    **Args**:
        * **[sTime]:The earliest time you want data for 
        * **[eTime]:The latest time you want data for (for our puposes it should be same as sTime)
        * **[lat]: Latitude
        * **[lon]: Longitude
        * **[ssn]: Rz12 sunspot number
        * **[output]: Select output values. True=output all. False=Only output hmF2.
        * **[]: 
    **Returns**:
        * **[hmF2]: The height of the F2 layer 
        * **[outf]: An array with the output of irisub.for 
        * **[oarr]: Array with input parameters and array with additional output of irisub.for
        
    .. note:: Untested!????? (It probably has been at this point.)

    Written by Magda Moses 2015 August 06
    """
    import numpy as np
    import pandas as pd


    from davitpy.models import *
    from davitpy import utils
    import datetime

    # Inputs
    jf = [True]*50
    #jf[1]=False
    #uncomment next line to input ssn
    jf[2:6] = [False]*4
    jf[20] = False
    jf[22] = False
    jf[27:30] = [False]*3
    jf[32] = False
    jf[34] = False

    #Create Array for input variables(will also hold output values later) 
    oarr = np.zeros(100)

    #Decide if to have user input ssn
    if ssn!=None:
        jf[16]=False
        oarr[32]=ssn
    else: 
        jf[17]=True
#    import ipdb; ipdb.set_trace()
#    geographic   = 1 geomagnetic coordinates
    jmag = 0.
    #ALATI,ALONG: LATITUDE NORTH AND LONGITUDE EAST IN DEGREES
    alati = lat 
    along = lon
    # IYYYY: Year as YYYY, e.g. 1985
    iyyyy = sTime.year
#    import ipdb; ipdb.set_trace()
    #MMDD (-DDD): DATE (OR DAY OF YEAR AS A NEGATIVE NUMBER)
    t0 = sTime.timetuple()
    #Day of Year (doy)
    doy=t0.tm_yday
    mmdd = -doy 
    #DHOUR: LOCAL TIME (OR UNIVERSAL TIME + 25) IN DECIMAL HOURS
    decHour=float(sTime.hour)+float(sTime.minute)/60+float(sTime.second)/3600
    #Acording to the irisub.for comments, this should be equivilent to local time input
    #Need Testing !!!!!!
#    dhour=12+decHour-5
    dhour=decHour+25
#    import ipdb; ipdb.set_trace()
    #HEIBEG,HEIEND,HEISTP: HEIGHT RANGE IN KM; maximal 100 heights, i.e. int((heiend-heibeg)/heistp)+1.le.100
    heibeg, heiend, heistp = 80., 500., 10. 
#    heibeg, heiend, heistp = 350, 350., 0. 
    outf=np.zeros(20)
    outf,oarr = iri.iri_sub(jf,jmag,alati,along,iyyyy,mmdd,dhour,heibeg,heiend,heistp,oarr)

    #Extract Values from model output
    hmF2=oarr[1]
    #NmF2 in m-3
    NmF2=oarr[0]
    #Calculate iri fof2(kHz)
    foF2=np.sqrt(oarr[0]*(1e-2)**3)*(9e3)/1e3
#    foF2=np.sqrt(oarr[0]/(1.24e10))

    if(output):
        return hmF2, outf, oarr
    else:
        return hmF2
