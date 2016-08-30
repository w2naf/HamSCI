#!/usr/bin/env python
#Functions for calculating ionospheric parameters from rbn data
#import rbn_lib

import sys
import os
import datetime

import davitpy

import numpy as np 

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
    import geopack
    if Re==None: Re=6371
    #may need to check alt value
#    midLat, midLon=geopack.midpoint(lat1,lon1,lat2,lon2)_

    theta=geopack.get_theta(lat1, lon1,  lat2, lon2)
    #Calculate foF2
    foF2 = muf*np.cos(theta)

    return foF2

#def get_theta(deLat,deLon, dxLat,  dxLon,h=350, iri=False):
#    """Calculate the incident angle (theta) for the one-hop path 
#    **Args**:
#        * **[deLat]: Receiver Latitude
#        * **[deLon]: Receiver Longitude
#        * **[dxLat]: Transimitter Latitude
#        * **[dxLon]: Transimitter Longitude
#        * **[h]: hmF2 
#        * **[iri]: Option to use the hmF2 output by the IRI for h. (Default is False)
#        * **[]:  
#
#    **Returns**:
#        * **[theta]: The angle of incidence on the F2 layer 
#
#    **Other Variables**:
#        * **[phi]: half the angular distance along the great cricle path of the link 
#        * **[x]: Half of Cord legnth
#        * **[z]: Distance from transmitter/reciever to the point of reflection
#        
#    .. note:: 
#    Example : theta = rbn_ionofun.get_theta()
#
#    Written by Magda Moses Fall 2016 
#    """
#
#    #Get parameters to calculate theta (the angle of reflection)
#    phi = (dagreatCircleDist(deLat, deLon, dxLat, dxLon))/2
#    alpha=(np.pi+phi)/2
#    x = r*np.sqrt(2*(1-np.cos(phi)))
#
#    #If want to use iri value of the height of the F2 Layer
#    if iri == True: 
#        hv,outf,oarr=rbn_lib.get_hmF2(sTime=time, lat=midLat, lon=midLon)
#    #Calculate theta
#    z=np.sqrt(hv**2+x**2-2*hv*x*np.cos(alpha))
#    theta=np.arcsin((x/z)*np.sin(alpha)) 
#
#    return theta
