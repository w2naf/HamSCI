#!/usr/bin/env python
#Functions for calculating ionospheric parameters from rbn data
import rbn_lib

imort sys
import os
import datetime
#import matplotlib as mpl

def muf():

    return 
def rbn_fof2():
    """Calculate foF2 at the midpoint between the two stations 
    **Args**:
        * **[time]:The time link occured (used in iri model)
        * **[muf]:The frequency of the link (taken to be the Maximum Usable frequency (MUF))
        * **[deLat]: Receiver Latitude
        * **[deLon]: Receiver Longitude
        * **[dxLat]: Transimitter Latitude
        * **[dxLon]: Transimitter Longitude
        * **[output]: Select output values. True=output all. False=Only output foF2,theta,h.

        * **Following parameter NOT implemented yet!
        * **[in_iri]: Array of additional input parameters to the IRI 
                    * **in_iri[0]=[ssn]: Rz12 sunspot number
    **Returns**:
        * **[foF2]: The critical frequency of the F2 layer 
        * **[theta]: The angle of incidence on the F2 layer 
        * **[h]: hmF2 
        * **[phi]: half the angular distance along the great cricle path of the link 
        * **[x]: Half of Cord legnth
        * **[z]: Distance from transmitter/reciever to the point of reflection
        
    .. note:: 
    Example (output=False): foF2[i],theta[i],h[i] = rbn_lib.rbn_fof2(sTime, freq, deLat, deLon, dxLat,dxLon)
    Example (output=True): foF2[i],theta[i],h[i],phi[i],x[i],z[i] = rbn_lib.rbn_fof2(sTime, freq, deLat, deLon, dxLat,dxLon, output=True)

    Written by Magda Moses 2016 July 07
    """
    return

def get_theta(deLoc, dxLoc,hv) 
    """Calculate the incident angle for the one-hop ray path 
    **Args**:
        * **[deLoc]: Receiver Latitude
        * **[deLon]: Receiver Longitude
        * **[dxLat]: Transimitter Latitude
        * **[dxLon]: Transimitter Longitude
        * **[output]: Select output values. True=output all. False=Only output foF2,theta,h.

        * **Following parameter NOT implemented yet!
        * **[in_iri]: Array of additional input parameters to the IRI 
                    * **in_iri[0]=[ssn]: Rz12 sunspot number
    **Returns**:
        * **[foF2]: The critical frequency of the F2 layer 
        * **[theta]: The angle of incidence on the F2 layer 
        * **[h]: hmF2 
        * **[phi]: half the angular distance along the great cricle path of the link 
        * **[x]: Half of Cord legnth
        * **[z]: Distance from transmitter/reciever to the point of reflection
        
    .. note:: 
    Example (output=False): foF2[i],theta[i],h[i] = rbn_lib.rbn_fof2(sTime, freq, deLat, deLon, dxLat,dxLon)
    Example (output=True): foF2[i],theta[i],h[i],phi[i],x[i],z[i] = rbn_lib.rbn_fof2(sTime, freq, deLat, deLon, dxLat,dxLon, output=True)

    Written by Magda Moses 2016 July 07
    """


    #Get parameters to calculate theta (the angle of reflection)
    phi = (greatCircleDist(deLat, deLon, dxLat, dxLon))/2
    alpha=(np.pi+phi)/2
    x = r*np.sqrt(2*(1-np.cos(phi)))
#    h,outf,oarr=rbn_lib.get_hmF2(sTime=time, lat=midLat, lon=midLon,ssn=ssn)
    h,outf,oarr=rbn_lib.get_hmF2(sTime=time, lat=midLat, lon=midLon)
    z=np.sqrt(h**2+x**2-2*h*x*np.cos(alpha))
    theta=np.arcsin((x/z)*np.sin(alpha)) 
