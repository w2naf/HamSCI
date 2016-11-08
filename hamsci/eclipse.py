#!/usr/bin/env python
import os
import numpy as np
from pykml import parser

class Eclipse2017(object):
    def __init__(self,fname=None):
        """
        Provide lat/lons and plotting functions for the 2017 Eclipse.

        Data comes from KML/KMZ file from
        http://xjubier.free.fr/en/site_pages/SolarEclipsesGoogleEarth.html.
        """
        if fname is None:
            base    = os.path.split(__file__)[0]
            fname   = os.path.join(base,'2017_eclipse.kml')

        with open(fname) as fl:
            doc = parser.parse(fl)
        self.doc  = doc
        self.root = doc.getroot()
    
    def print_index(self):
        """
        Prints the names and indices of the folders and placemarks
        in the 2017 Eclipse KML File.
        """
        for fl_inx, folder in enumerate(self.root.Document.Folder):
            print '[{!s}] {}'.format(fl_inx,folder.name)
            if hasattr(folder,'Placemark'):
                for pm_inx,pm in enumerate(folder.Placemark):
                    print '    [{!s}] {}'.format(pm_inx,pm.name)
    
    def get_latlon(self,folder_inx=1,placemark_inx=1):
        """
        Return the (latitutes, longitudes) of the specified placemark
        index. Use print_index() for a directory of available 
        indices.
        
        Example:
        # Return lats and lons of Eclipse Central Umbral Line
        lat,lon = get_latlon(1,1)
        """
        
        str_coords   = self.root.Document.Folder[folder_inx].Placemark[placemark_inx].LineString.coordinates
        muxed_coords = str_coords.text.splitlines()[1].split()
        
        lats, lons = [],[]
        for mcrd in muxed_coords:
            lon,lat,alt = mcrd.split(',')
            lats.append(float(lat))
            lons.append(float(lon))
            
        return (np.array(lats),np.array(lons))
    
    def get_label(self,folder_inx,placemark_inx=None):
        
        if placemark_inx is None:
            label = self.root.Document.Folder[folder_inx].name
        else:
            label = self.root.Document.Folder[folder_inx].Placemark[placemark_inx].name

        return label

    def overlay_umbra(self,m,label='Eclipse Centerline',
            color='blue',bound_style='--',zorder=100):

        #Eclipse Centerline
        fl_inx, pm_inx = 1,1
        lats,lons       = self.get_latlon(fl_inx,pm_inx)
        line,           = m.plot(lons,lats,color=color,label=label,zorder=zorder)

        fl_inx, pm_inx = 1,0
        lats,lons      = self.get_latlon(fl_inx,pm_inx)
        m.plot(lons,lats,color=color,ls=bound_style,zorder=zorder)

        fl_inx, pm_inx = 1,2
        lats,lons      = self.get_latlon(fl_inx,pm_inx)
        m.plot(lons,lats,color=color,ls=bound_style,zorder=zorder)

        return (line, label)

if __name__ == '__main__':
    ecl = Eclipse2017()

    ecl.print_index()

    # Map Test #############################
    from matplotlib import pyplot as plt
    import matplotlib as mpl
    from mpl_toolkits.basemap import Basemap

    from gridsquare import *

    llcrnrlon = -130.
    llcrnrlat =  20.
    urcrnrlon =  -60.
    urcrnrlat =   60.

    fig = plt.figure(figsize=(20,15))
    ax  = fig.add_subplot(111)
    m   = Basemap(llcrnrlon=llcrnrlon,llcrnrlat=llcrnrlat,urcrnrlon=urcrnrlon,urcrnrlat=urcrnrlat,
                resolution='l',area_thresh=1000.,projection='cyl',ax=ax)

    # Draw Grid Squares
    grid_grid = gridsquare_grid(2)
    lats,lons = gridsquare2latlon(grid_grid,position='lower left')

    m.drawparallels(lats[0,:],color='k',labels=[False,True,True,False])
    m.drawmeridians(lons[:,0],color='k',labels=[True,False,False,True])
    m.drawcoastlines(color='0.65')
    m.drawmapboundary(fill_color='w')

    # Add in the precision-4 grid.
    grid_grid = gridsquare_grid(4)
    lats,lons = gridsquare2latlon(grid_grid,position='lower left')

    m.drawparallels(lats[0,:],color='0.85',labels=[False,False,False,False])
    m.drawmeridians(lons[:,0],color='0.85',labels=[False,False,False,False])

    ecl.overlay_umbra(m)
    ax.legend(loc='upper right')

    plt.show()
