#!/usr/bin/env python
def eclipse_get_path(fname='ds_CL.csv', data_dir=None):
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
    df_cl=pd.DataFrame.from_csv(fname)
    df_cl.columns=['eLat','eLon']
#    import ipdb; ipdb.set_trace()
    return df_cl


def eclipse_map_plot(infile=None,mapobj=None, fig=None, style='--m', lw=2):
    import datetime
    from matplotlib import pyplot as plt
    from mpl_toolkits.basemap import Basemap
    import numpy as np
    import pandas as pd
    #
    #Plot  
    df=eclipse_get_path(fname=infile)
    mapobj.plot(df['eLon'],df['eLat'],style, linewidth=lw, latlon=True)
    return mapobj,fig 

def eclipse_swath(infile=None, mapobj=None, fig=None, pathColor='m', lw=2, pZorder=0):
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
        df0=eclipse_get_path(fname=inputF)
        import ipdb; ipdb.set_trace()
        if i==0:
            df=df0
        else: 
            #may need to change ignore_index
            sort_df0=df0.sort(columns='eLon',ascending=False)
            frame= [df,sort_df0]
            df=pd.concat(frame, ignore_index=True)
            import ipdb; ipdb.set_trace()
            
        i+=1
    #df=concat([df, df[ 

#    Z=np.meshgrid(df['eLon'],df['eLat'])
#    import ipdb; ipdb.set_trace()
#    mapobj.contourf(x,y,Z, style, linewidth=lw, latlon=True)
#    mapobj.plot(df['eLon'], df['eLat'],style, linewidth=lw, latlon=True)
    #make polygon
    verticies=zip(df['eLon'],df['eLat'])
    verticies.append(verticies[0])
    import ipdb; ipdb.set_trace()
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

