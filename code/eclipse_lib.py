#!/usr/bin/env python

#Set dictionary for proposed VTARA field stations
vtara_dict     = {}
vtara_dict[1]  = {'name': '1',  'lat': 44.9,  'lon': -123}
vtara_dict[2]  = {'name': '2',  'lat': 43.74,  'lon': -110.73}
vtara_dict[3]  = {'name': '3',  'lat': 39.4,  'lon': -95.8}
vtara_dict[4]  = {'name': '4',  'lat': 33.9,  'lon': -80.34}

#band_dict[band]['color']

def eclipse_get_path(fname='ds_CL.csv', data_dir=None,filetype='csv'):
    #Inputs: 
    #fname: specify filename (the file is in the same folder as the code)
    #data_dir: directory the file is in, could have directory as input in future version too
    #could have directory as input in future version too
    import sys 
    import os 
    #import matplotlib
    #matplotlib.use
    import pandas as pd
    import numpy as np 
    #From original code
#    fname='ds_CL.csv'
    #Optional input path 
    #(Caution: folder 'eclipse' does not exist yet!!!! make this folder and put file in it BEFORE using this code section!)
    #input_path=os.path.join('data','eclipse')
    #fpath=os.path.join(inpuit,fname)
    #Main body
    #Make data frame (make sure there is an index column that is seperate from your data unless part of the data is an index!) 
#    df_cl=pd.DataFrame.from_csv(fname,delim_whitespace)
    if filetype=='csv':
        df_cl=pd.DataFrame.from_csv(fname)

    if filetype=='txt':
        df_cl=pd.DataFrame.from_csv(fname,sep=None, index_col=None)
#        df_cl=pd.read_csv(fname,sep=None,engine='python', index_col=None)

    df_cl.columns=['eLat','eLon']
#    import ipdb; ipdb.set_trace()
    return df_cl


def eclipse_map_plot(infile=None,mapobj=None, fig=None, pathColor='--m', lw=2,filetype='csv'):
    import datetime
    from matplotlib import pyplot as plt
    from mpl_toolkits.basemap import Basemap
    import numpy as np
    import pandas as pd
    #
    #Plot  
    df=eclipse_get_path(fname=infile, filetype=filetype)
    mapobj.plot(df['eLon'],df['eLat'], color=pathColor ,linewidth=lw, latlon=True)
    return mapobj,fig 

def eclipse_swath(infile=None, mapobj=None, fig=None, pathColor='m', lw=2, pZorder=0,filetype='csv'):
    import datetime
    from matplotlib import pyplot as plt
    from matplotlib import path
    from matplotlib import patches
    from pylab import gca
    from matplotlib.patches import Polygon 
    from mpl_toolkits.basemap import Basemap
    import numpy as np
    import pandas as pd

    i=0
    for inputF in infile: 
        df0=eclipse_get_path(fname=inputF, filetype=filetype)
#        import ipdb; ipdb.set_trace()
        if i==0:
            df=df0
        else: 
            #may need to change ignore_index
            sort_df0=df0.sort(columns='eLon',ascending=False)
            frame= [df,sort_df0]
            df=pd.concat(frame, ignore_index=True)
            import ipdb; ipdb.set_trace()
#            import ipdb; ipdb.set_trace()
            
        i+=1
    #df=concat([df, df[ 

#    Z=np.meshgrid(df['eLon'],df['eLat'])
#    import ipdb; ipdb.set_trace()
#    mapobj.contourf(x,y,Z, style, linewidth=lw, latlon=True)
#    mapobj.plot(df['eLon'], df['eLat'],style, linewidth=lw, latlon=True)
    #make polygon
    verticies=zip(df['eLon'],df['eLat'])
    verticies.append(verticies[0])
#    import ipdb; ipdb.set_trace()
#    codes=[path.MOVETO]
#    c=np.ones(len(df)-1)
#    code_df=pd.Dataframe(c,columns='Code')
##    code_df['Code']=path.LINETO
#    codes.append(code_df);
#    import ipdb; ipdb.set_trace()
#    codes.append(path.CLOSEPOLY)    
#    path=path(verticies,codes)
#    patch=patches.PathPatch(verticies,facecolor='m', lw=lw)
    patch=Polygon(verticies,color=pathColor, lw=lw,zorder=pZorder)
    gca().add_patch(patch)
    return mapobj, fig

def eclipse_limits_legend():
    return 

#def plot_vtara_stations(m,symbol='^',color='r',,ax=None,legend=True,tick_font_size=None,ncdxf=False,plot_paths=True,
#        llcrnrlon=-180.,llcrnrlat=-90,urcrnrlon=180.,urcrnrlat=90.,proj='cyl',basemapType=True,m=None,eclipse=False,path_alpha=None):
def plot_vtara_stations(m=None,symbol='^',color='r',fig=None,ax=None,legend=True,tick_font_size=None,
        llcrnrlon=-180.,llcrnrlat=-90,urcrnrlon=180.,urcrnrlat=90.,proj='cyl',basemapType=True,eclipse=False):
    """Plot VTARA Filed Stations

    **Args**:
        * **[m]**: Basemap object
        * **[symbol]**: Symbol used on map

    **Returns**:
        * **fig**:      matplotlib figure object that was plotted to

    .. note::
        If a matplotlib figure currently exists, it will be modified by this routine.  If not, a new one will be created.

    Written by Magda Moses 2016 May 28
    (Based on rbn_lib.rbn_map_plot() code by Nathaniel Frissell 2014 Sept 06)
    """
    from matplotlib import pyplot as plt
    from mpl_toolkits.basemap import Basemap

    import numpy as np
#    import pandas as pd

#    if ax is None:
#        fig     = plt.figure(figsize=(10,6))
#        ax      = fig.add_subplot(111)
#    else:
#        fig     = ax.get_figure()
#
    if m==None: #added to allow rbn to be plotted over maps of other data 
        if ax is None:
            fig     = plt.figure(figsize=(10,6))
            ax      = fig.add_subplot(111)
        else:
            fig     = ax.get_figure()

        if basemapType:
            m = Basemap(llcrnrlon=llcrnrlon,llcrnrlat=llcrnrlat,urcrnrlon=urcrnrlon,urcrnrlat=urcrnrlat,resolution='l',area_thresh=1000.,projection=proj,ax=ax)
        else:
            m = plotUtils.mapObj(llcrnrlon=llcrnrlon,llcrnrlat=llcrnrlat,urcrnrlon=urcrnrlon,urcrnrlat=urcrnrlat,resolution='l',area_thresh=1000.,projection=proj,ax=ax,fillContinents='None', fix_aspect=True)

        # draw parallels and meridians.
        m.drawparallels(np.arange(-90.,91.,45.),color='k',labels=[False,True,True,False],fontsize=tick_font_size)
        m.drawmeridians(np.arange(-180.,181.,45.),color='k',labels=[True,False,False,True],fontsize=tick_font_size)
        m.drawcoastlines(color='0.65')
        m.drawmapboundary(fill_color='w')
#        m.nightshade(plot_mTime,color='0.82')

        #if plotting the 2017 eclipse map then also draw state boundaries
        if eclipse:
            m.drawcountries(color='0.65')#np.arange(-90.,91.,45.),color='k',labels=[False,True,True,False],fontsize=tick_font_size)
            m.drawstates(color='0.65')
    
    #Get VTARA sites from dictionary
    vlat=[]
    vlon=[]
    for i in np.arange(1,5):
        vlat.append(vtara_dict[i]['lat']) 
        vlon.append(vtara_dict[i]['lon'])
    
    #plot stations
#    rx    = m.scatter(lon,lat,color='r',s=2,zorder=100)
#    rx    = m.scatter(vlon,vlat,marker=symbol, color=color,zorder=100)
    rx    = m.scatter(vlon,vlat,marker=symbol, color=color,s=50,zorder=100)
    
    return m,fig


