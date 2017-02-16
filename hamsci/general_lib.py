import os
import shutil
import datetime

import collections

import numpy as np
import matplotlib

# Colormap Routines ############################################################
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=None,name=None):
    if n is None:
        n = cmap.N

    if name is None:
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval)
    
    cmap_list   = cmap(np.linspace(minval, maxval, n))
    lsc         = matplotlib.colors.LinearSegmentedColormap
    new_cmap    = lsc.from_list(name,cmap_list)
    
    return new_cmap

def combine_cmaps(cmaps,minval=0.0,maxval=1.0,n=None,name='Combined CMAP'):
    lsc             = matplotlib.colors.LinearSegmentedColormap
    new_list        = np.array([])
    new_list.shape  = (0,4)

    for cmap in get_iterable(cmaps):
        if n is None:
            N = cmap.N
        else:
            N = n

        cmap_list   = cmap(np.linspace(minval, maxval, N))
        new_list    = np.append(new_list,cmap_list,axis=0)
    
    new_cmap    = lsc.from_list(name,new_list)
    return new_cmap

def get_custom_cmap(name='blue_red'):
    if name == 'blue_red':
        l_cut       = 0.30
        u_cut       = 0.0
        blues       = truncate_colormap(matplotlib.cm.Blues_r,minval=u_cut,maxval=1.-l_cut)
        reds        = truncate_colormap(matplotlib.cm.Reds,minval=l_cut,maxval=1.-u_cut)
        combine     = [blues,reds]
        new_cmap    = combine_cmaps(combine,name=name)

    return new_cmap

def cc255(color):
    """
    Convert any valid matplotlib color into a 255 bit
    RGB triplet (tuple).
    """
    cc = matplotlib.colors.ColorConverter().to_rgb
    trip = np.array(cc(color))*255
    trip = [int(x) for x in trip]
    return tuple(trip)

def cdict_to_cmap(cdict,name='CustomCMAP',vmin=0.,vmax=30.):
    """
    Generate a matplotlib cmap from a cdict that specifies
    colors for specifc values specified as 255 bit RGB triplets.

    Inputs:
        cdict: Color dictionary.
            Example:
            cdict       = {}
            cdict[ 0.0] = (  0,   0,   0)
            cdict[ 1.8] = cc255('violet')
            cdict[ 3.0] = cc255('blue')
            cdict[ 8.0] = cc255('aqua')
            cdict[10.0] = cc255('green')
            cdict[13.0] = cc255('green')
            cdict[17.0] = cc255('yellow')
            cdict[21.0] = cc255('orange')
            cdict[28.0] = cc255('red')
            cdict[30.0] = cc255('red')
        vmin: Bottom of color scale.
        vmax: Top of color scale.
    """
    norm = matplotlib.colors.Normalize(vmin=vmin,vmax=vmax)
    
    red   = []
    green = []
    blue  = []
    
    keys = cdict.keys()
    keys.sort()
    
    for x in keys:
        r,g,b, = cdict[x]
        x = norm(x)
        r = r/255.
        g = g/255.
        b = b/255.
        red.append(   (x, r, r))
        green.append( (x, g, g))
        blue.append(  (x, b, b))
    cdict = {'red'   : tuple(red),
             'green' : tuple(green),
             'blue'  : tuple(blue)}
    cmap  = matplotlib.colors.LinearSegmentedColormap(name, cdict)
    return cmap

# Misc. Functions ##############################################################
def get_iterable(x):
    if isinstance(x, collections.Iterable) and not isinstance(x,basestring):
        return x
    else:
        return [x]

def make_list(item):
    """
    Force something to be iterable.

    This should probably be deprecated in favor of get_iterable.
    """
    item = np.array(item)
    if item.shape == ():
        item.shape = (1,)

    return item.tolist()

def generate_radar_dict():
    rad_list = []
    rad_list.append(('bks', 39.6, -81.1))
    rad_list.append(('wal', 41.8, -72.2))
    rad_list.append(('fhe', 42.5, -95.0))
    rad_list.append(('fhw', 43.3, -102.7))
    rad_list.append(('cve', 46.4, -114.6))
    rad_list.append(('cvw', 47.9, -123.4))
    rad_list.append(('gbr', 58.4, -59.9))
    rad_list.append(('kap', 55.5, -85.0))
    rad_list.append(('sas', 56.1, -103.8))
    rad_list.append(('pgr', 58.0, -123.5))

    radar_dict = {}
    for radar,lat,lon in rad_list:
        tmp                 = {}
        tmp['lat']          = lat
        tmp['lon']          = lon
        radar_dict[radar]   = tmp

    return radar_dict

class TimeCheck(object):
    #import inspect
    #curr_file = inspect.getfile(inspect.currentframe()) # script filename (usually with path)
    #import logging
    #import logging.config
    #logging.filename    = curr_file+'.log'
    #logging.config.fileConfig("logging.conf")
    #log = logging.getLogger("root")

    def __init__(self,label=None,log=None):
        self.label  = label
        self.log    = log
        self.t0     = datetime.datetime.now()
    def check(self):
        self.t1 = datetime.datetime.now()
        dt      = self.t1 - self.t0

        txt = '{sec}'.format(sec=str(dt))

        if self.label is not None:
            txt = ': '.join([self.label,txt])

        if self.log is not None:
            log.info(txt)
        else:
            print txt

# File Handling ################################################################
def prepare_output_dirs(output_dirs={0:'output'},clear_output_dirs=False,width_100=False,img_extra='',
        php_viewers=True):

    if width_100:
        img_extra = "width='100%'"

    txt = []
    txt.append('<?php')
    txt.append('foreach (glob("*.png") as $filename) {')
    txt.append('    echo "<img src=\'$filename\' {img_extra}> ";'.format(img_extra=img_extra))
    txt.append('}')
    txt.append('?>')
    show_all_txt = '\n'.join(txt)

    txt = []
    txt.append('<?php')
    txt.append('foreach (glob("*.png") as $filename) {')
    txt.append('    echo "<img src=\'$filename\' {img_extra}> <br />";'.format(img_extra=img_extra))
    txt.append('}')
    txt.append('?>')
    show_all_txt_breaks = '\n'.join(txt)

    for value in output_dirs.itervalues():
        if clear_output_dirs:
            try:
                shutil.rmtree(value)
            except:
                pass
        try:
            os.makedirs(value)
        except:
            pass
        if php_viewers:
            with open(os.path.join(value,'0000-show_all.php'),'w') as file_obj:
                file_obj.write(show_all_txt)
            with open(os.path.join(value,'0000-show_all_breaks.php'),'w') as file_obj:
                file_obj.write(show_all_txt_breaks)
