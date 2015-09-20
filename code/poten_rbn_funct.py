#Functions that i started work on but which I think I will replace
#I'm saving the work on them in case my new pla does not work out
#Files functions from:
    #rbn_lib.py


#rbn_lib.py functions
def rbn_map_foF2(df,ax=None,legend=True,tick_font_size=None,ncdxf=False,plot_paths=True,
        llcrnrlon=-180.,llcrnrlat=-90,urcrnrlon=180.,urcrnrlat=90.,proj='cyl',basemapType=True,eclipse=False,path_alpha=None):
    """Plot foF2 values derived from Reverse Beacon Network data for the midpoints between two stations.

    **Args**:
        * **[sTime]**: datetime.datetime object for start of plotting.
        * **[eTime]**: datetime.datetime object for end of plotting.
        * **[ymin]**: Y-Axis minimum limit
        * **[ymax]**: Y-Axis maximum limit
        * **[legendSize]**: Character size of the legend
        * **[df]**: DataFrame with 'midLat','midLon','foP' attributes/sections

    **Returns**:
        * **fig**:      matplotlib figure object that was plotted to

    .. note::
        If a matplotlib figure currently exists, it will be modified by this routine.  If not, a new one will be created.

    Written by Magda Moses and Nathaniel Frissell 2015 Sept 12
    """
    import datetime
    
    from matplotlib import pyplot as plt
    from mpl_toolkits.basemap import Basemap

    import numpy as np
    import pandas as pd

    import eclipse_lib

    from davitpy.pydarn.radar import *
    from davitpy.pydarn.plotting import *
    from davitpy.utils import *

    if ax is None:
        fig     = plt.figure(figsize=(10,6))
        ax      = fig.add_subplot(111)
    else:
        fig     = ax.get_figure()

    #Drop NaNs (QSOs without Lat/Lons)
    df = df.dropna(subset=['dx_lat','dx_lon','de_lat','de_lon'])

    ##Sort the data by frequency sub-bands and time, then group by band.
#    df['sub_band']  = np.array((np.floor(df['Freq_plasma']/1000.)),dtype=np.int)
    df['sub_band']  = np.array((np.floor(df['foP']/1000.)),dtype=np.int)
    srt         = df.sort(['band','date'])
    grouped     = srt.groupby('band')

    sTime       = df['date'].min()
    eTime       = df['date'].max()

    half_time   = datetime.timedelta(seconds= ((eTime - sTime).total_seconds()/2.) )
    plot_mTime = sTime + half_time

    if basemapType:
        m = Basemap(llcrnrlon=llcrnrlon,llcrnrlat=llcrnrlat,urcrnrlon=urcrnrlon,urcrnrlat=urcrnrlat,resolution='l',area_thresh=1000.,projection=proj,ax=ax)
    else:
        m = plotUtils.mapObj(llcrnrlon=llcrnrlon,llcrnrlat=llcrnrlat,urcrnrlon=urcrnrlon,urcrnrlat=urcrnrlat,resolution='l',area_thresh=1000.,projection=proj,ax=ax,fillContinents='None', fix_aspect=True)

#    title = sTime.strftime('%H%M - ')+eTime.strftime('%H%M UT')
#    title = sTime.strftime('Reverse Beacon Network %Y %b %d %H%M UT - ')+eTime.strftime('%Y %b %d %H%M UT')
    title = sTime.strftime('RBN: %d %b %Y %H%M UT - ')+eTime.strftime('%d %b %Y %H%M UT')
    ax.set_title(title)

    # draw parallels and meridians.
    m.drawparallels(np.arange(-90.,91.,45.),color='k',labels=[False,True,True,False],fontsize=tick_font_size)
    m.drawmeridians(np.arange(-180.,181.,45.),color='k',labels=[True,False,False,True],fontsize=tick_font_size)
    m.drawcoastlines(color='0.65')
    m.drawmapboundary(fill_color='w')
    m.nightshade(plot_mTime,color='0.82')
    #if plotting the 2017 eclipse map then also draw state boundaries
    if eclipse:
        m.drawcountries(color='0.65')#np.arange(-90.,91.,45.),color='k',labels=[False,True,True,False],fontsize=tick_font_size)
        m.drawstates(color='0.65')
    
    de_list = []
    dx_list = []
    for band in bandlist:
        try:
            this_group = grouped.get_group(band)
        except:
            continue

        color = pf_band_dict[sub_band]['color']
        #Need to change this
        label = pf_band_dict[sub_band]['name']

        for index,row in this_group.iterrows():
            #Yay stack overflow! - http://stackoverflow.com/questions/13888566/python-basemap-drawgreatcircle-function
            Lat = row['midLat']
            Lon = row['midLon']
#            dx_lat = row['dx_lat']
#            dx_lon = row['dx_lon']

            if row['callsign'] not in de_list: de_list.append(row['callsign'])
            if row['dx'] not in dx_list: dx_list.append(row['dx'])

            fof2_pt    = m.scatter(Lon,Lat,color=color,s=2,zorder=100)
            if plot_paths:
                line, = m.drawgreatcircle(dx_lon,dx_lat,de_lon,de_lat,color=color, alpha=path_alpha)

                p = line.get_path()
                # find the index which crosses the dateline (the delta is large)
                cut_point = np.where(np.abs(np.diff(p.vertices[:, 0])) > 200)[0]
                if cut_point:
                    cut_point = cut_point[0]

                    # create new vertices with a nan inbetween and set those as the path's vertices
                   # import ipdb; ipdb.set_trace()
                    new_verts = np.concatenate(
                                               [p.vertices[:cut_point, :], 
                                                [[np.nan, np.nan]], 
                                                p.vertices[cut_point+1:, :]]
                                               )
                    p.codes = None
                    p.vertices = new_verts
                #
#                cut_point_lat = np.where(np.abs(np.diff(p.vertices[:, 1])) > 90)[0]
#                if cut_point_lat:
#                    cut_point_lat = cut_point_lat[0]
#
#                    # create new vertices with a nan inbetween and set those as the path's vertices
#                    import ipdb; ipdb.set_trace()
#                    new_verts = np.concatenate(
#                                               [p.vertices[:cut_point_lat,:], 
#                                                [[np.nan, np.nan]], 
#                                                p.vertices[cut_point_lat+1:,:]]
#                                               )
#                    p.codes = None
#                    p.vertices = new_verts
#                    import ipdb; ipdb.set_trace()
    if ncdxf:
        dxf_df = pd.DataFrame.from_csv('ncdxf.csv')
        m.scatter(dxf_df['lon'],dxf_df['lat'],s=dxf_plot_size,**dxf_prop)

    #if eclipse:
     #   df_cl=eclipse_lib.eclipse_get_path(fname='ds_CL.csv')
     #   m.plot(df_cl['eLon'],df_cl['eLat'],'m--',label='2017 Eclipse Central Line', linewidth=2, latlon=True)

    import ipdb; ipdb.set_trace()
    text = []
    text.append('TX Stations: {0:d}'.format(len(dx_list)))
    text.append('RX Stations: {0:d}'.format(len(de_list)))
    text.append('Plotted Paths: {0:d}'.format(len(df)))

    props = dict(facecolor='white', alpha=0.9,pad=6)
    ax.text(0.02,0.05,'\n'.join(text),transform=ax.transAxes,ha='left',va='bottom',size=9,bbox=props)

    if legend:
        subBand_legend()

    return m,fig
##Copy
def rbn_map_foF2(df,ax=None,legend=True,tick_font_size=None,ncdxf=False,plot_paths=True,
        llcrnrlon=-180.,llcrnrlat=-90,urcrnrlon=180.,urcrnrlat=90.,proj='cyl',basemapType=True,eclipse=False,path_alpha=None):
    """Plot foF2 values derived from Reverse Beacon Network data for the midpoints between two stations.

    **Args**:
        * **[sTime]**: datetime.datetime object for start of plotting.
        * **[eTime]**: datetime.datetime object for end of plotting.
        * **[ymin]**: Y-Axis minimum limit
        * **[ymax]**: Y-Axis maximum limit
        * **[legendSize]**: Character size of the legend
        * **[df]**: DataFrame with 'midLat','midLon','foP' attributes/sections

    **Returns**:
        * **fig**:      matplotlib figure object that was plotted to

    .. note::
        If a matplotlib figure currently exists, it will be modified by this routine.  If not, a new one will be created.

    Written by Magda Moses and Nathaniel Frissell 2015 Sept 12
    """
    import datetime
    
    from matplotlib import pyplot as plt
    from mpl_toolkits.basemap import Basemap

    import numpy as np
    import pandas as pd

    import eclipse_lib

    from davitpy.pydarn.radar import *
    from davitpy.pydarn.plotting import *
    from davitpy.utils import *

    if ax is None:
        fig     = plt.figure(figsize=(10,6))
        ax      = fig.add_subplot(111)
    else:
        fig     = ax.get_figure()

    #Drop NaNs (QSOs without Lat/Lons)
    df = df.dropna(subset=['dx_lat','dx_lon','de_lat','de_lon'])

    ##Sort the data by frequency sub-bands and time, then group by band.
#    df['sub_band']  = np.array((np.floor(df['Freq_plasma']/1000.)),dtype=np.int)
    df['sub_band']  = np.array((np.floor(df['foP']/1000.)),dtype=np.int)
    srt         = df.sort(['band','date'])
    grouped     = srt.groupby('band')

    sTime       = df['date'].min()
    eTime       = df['date'].max()

    half_time   = datetime.timedelta(seconds= ((eTime - sTime).total_seconds()/2.) )
    plot_mTime = sTime + half_time

    if basemapType:
        m = Basemap(llcrnrlon=llcrnrlon,llcrnrlat=llcrnrlat,urcrnrlon=urcrnrlon,urcrnrlat=urcrnrlat,resolution='l',area_thresh=1000.,projection=proj,ax=ax)
    else:
        m = plotUtils.mapObj(llcrnrlon=llcrnrlon,llcrnrlat=llcrnrlat,urcrnrlon=urcrnrlon,urcrnrlat=urcrnrlat,resolution='l',area_thresh=1000.,projection=proj,ax=ax,fillContinents='None', fix_aspect=True)

#    title = sTime.strftime('%H%M - ')+eTime.strftime('%H%M UT')
#    title = sTime.strftime('Reverse Beacon Network %Y %b %d %H%M UT - ')+eTime.strftime('%Y %b %d %H%M UT')
    title = sTime.strftime('RBN: %d %b %Y %H%M UT - ')+eTime.strftime('%d %b %Y %H%M UT')
    ax.set_title(title)

    # draw parallels and meridians.
    m.drawparallels(np.arange(-90.,91.,45.),color='k',labels=[False,True,True,False],fontsize=tick_font_size)
    m.drawmeridians(np.arange(-180.,181.,45.),color='k',labels=[True,False,False,True],fontsize=tick_font_size)
    m.drawcoastlines(color='0.65')
    m.drawmapboundary(fill_color='w')
    m.nightshade(plot_mTime,color='0.82')
    #if plotting the 2017 eclipse map then also draw state boundaries
    if eclipse:
        m.drawcountries(color='0.65')#np.arange(-90.,91.,45.),color='k',labels=[False,True,True,False],fontsize=tick_font_size)
        m.drawstates(color='0.65')
    
    de_list = []
    dx_list = []
    for band in bandlist:
        try:
            this_group = grouped.get_group(band)
        except:
            continue

        color = pf_band_dict[sub_band]['color']
        #Need to change this
        label = pf_band_dict[sub_band]['name']

        for index,row in this_group.iterrows():
            #Yay stack overflow! - http://stackoverflow.com/questions/13888566/python-basemap-drawgreatcircle-function
            Lat = row['midLat']
            Lon = row['midLon']
#            dx_lat = row['dx_lat']
#            dx_lon = row['dx_lon']

            if row['callsign'] not in de_list: de_list.append(row['callsign'])
            if row['dx'] not in dx_list: dx_list.append(row['dx'])

            fof2_pt    = m.scatter(Lon,Lat,color=color,s=2,zorder=100)
            if plot_paths:
                line, = m.drawgreatcircle(dx_lon,dx_lat,de_lon,de_lat,color=color, alpha=path_alpha)

                p = line.get_path()
                # find the index which crosses the dateline (the delta is large)
                cut_point = np.where(np.abs(np.diff(p.vertices[:, 0])) > 200)[0]
                if cut_point:
                    cut_point = cut_point[0]

                    # create new vertices with a nan inbetween and set those as the path's vertices
                   # import ipdb; ipdb.set_trace()
                    new_verts = np.concatenate(
                                               [p.vertices[:cut_point, :], 
                                                [[np.nan, np.nan]], 
                                                p.vertices[cut_point+1:, :]]
                                               )
                    p.codes = None
                    p.vertices = new_verts
                #
#                cut_point_lat = np.where(np.abs(np.diff(p.vertices[:, 1])) > 90)[0]
#                if cut_point_lat:
#                    cut_point_lat = cut_point_lat[0]
#
#                    # create new vertices with a nan inbetween and set those as the path's vertices
#                    import ipdb; ipdb.set_trace()
#                    new_verts = np.concatenate(
#                                               [p.vertices[:cut_point_lat,:], 
#                                                [[np.nan, np.nan]], 
#                                                p.vertices[cut_point_lat+1:,:]]
#                                               )
#                    p.codes = None
#                    p.vertices = new_verts
#                    import ipdb; ipdb.set_trace()
    if ncdxf:
        dxf_df = pd.DataFrame.from_csv('ncdxf.csv')
        m.scatter(dxf_df['lon'],dxf_df['lat'],s=dxf_plot_size,**dxf_prop)

    #if eclipse:
     #   df_cl=eclipse_lib.eclipse_get_path(fname='ds_CL.csv')
     #   m.plot(df_cl['eLon'],df_cl['eLat'],'m--',label='2017 Eclipse Central Line', linewidth=2, latlon=True)

    import ipdb; ipdb.set_trace()
    text = []
    text.append('TX Stations: {0:d}'.format(len(dx_list)))
    text.append('RX Stations: {0:d}'.format(len(de_list)))
    text.append('Plotted Paths: {0:d}'.format(len(df)))

    props = dict(facecolor='white', alpha=0.9,pad=6)
    ax.text(0.02,0.05,'\n'.join(text),transform=ax.transAxes,ha='left',va='bottom',size=9,bbox=props)

    if legend:
        subBand_legend()

    return m,fig

