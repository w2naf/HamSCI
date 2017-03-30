import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['text.usetex'] = True
from matplotlib import pyplot as plt
import mpl_toolkits.axes_grid.axes_size as Size
from mpl_toolkits.axes_grid import Divider
import matplotlib.dates as mdates

import matlab.engine

from . import raytrace
#eng = matlab.engine.start_matlab('-desktop')
#eng = matlab.engine.start_matlab()

def plot_iono_path_profile(rt_obj,
        iono_param='iono_pf_grid',  iono_arr=None,
        iono_cmap='viridis',        iono_lim=None,      iono_title='Ionospheric Parameter',
        maxground=4000.,            maxalt=400.,        Re=6371,
        rect=111,                   ax=None,            aax=None,
        fig=None,                   plot_colorbar=True, iono_rasterize=False,
        **kwargs):
    """
    Plot a 2d ionospheric profile along a path.
    """

    md          = rt_obj.rt_dct['metadata']
    axes        = rt_obj.rt_dct['axes']
    raytrace    = rt_obj.rt_dct['raytrace']
    ionosphere  = rt_obj.rt_dct['ionosphere']

    tx_lat      = md.get('tx_lat')
    tx_lon      = md.get('tx_lon')
    tx_call     = md.get('tx_call')
    rx_lat      = md.get('rx_lat')
    rx_lon      = md.get('rx_lon')
    rx_call     = md.get('rx_call')
    rx_range    = md.get('rx_range')
    azm         = md.get('azm')

    date        = md.get('date')

    ranges      = axes.get('ranges')
    heights     = axes.get('heights')

    all_ray_paths       = raytrace.get('all_ray_paths')
    connecting_ray_path = raytrace.get('connecting_ray_path')

    if maxground is None:
        maxground = np.max(ranges)

    if maxalt is None:
        maxalt = np.max(heights)

    # Set up axes
    if not ax and not aax:
        ax, aax = curvedEarthAxes(fig=fig, rect=rect, 
            maxground=maxground, maxalt=maxalt,Re=Re)

    cbax = None # Have something to return even if we don't plot a colorbar.

    # Convert linear range into angular distance.
    thetas  = ranges/Re

    # Plot background ionosphere. ################################################## 
    if (iono_arr is not None) or (iono_param is not None):
        if iono_param == 'iono_en_grid' or iono_param == 'iono_en_grid_5':
            if iono_lim is None: iono_lim = (10,12)
            if iono_title == 'Ionospheric Parameter':
                iono_title = r"N$_{el}$ [$\log_{10}(m^{-3})$]"
            # Get the log10 and convert Ne from cm**(-3) to m**(-3)
            iono_arr    = np.log10(ionosphere[iono_param]*100**3)
        elif iono_param == 'iono_pf_grid' or iono_param == 'iono_pf_grid_5':
            if iono_lim is None: iono_lim = (0,10)
            if iono_title == 'Ionospheric Parameter':
                iono_title = r"Plasma Frequency [MHz]"
            iono_arr    = ionosphere[iono_param]
        elif iono_param == 'collision_freq':
            if iono_lim is None: iono_lim = (0,8)
            if iono_title == 'Ionospheric Parameter':
                iono_title = r"$\nu$ [$\log_{10}(\mathrm{Hz})$]"
            iono_arr    = np.log10(ionosphere[iono_param])

        if iono_lim is None:
            iono_mean   = np.mean(iono_arr)
            iono_std    = np.std(iono_arr)

            iono_0      = 50000
            iono_1      = 90000

            iono_lim    = (iono_0, iono_1)


        X, Y    = np.meshgrid(thetas,heights+Re)
        im      = aax.pcolormesh(X, Y, iono_arr,
                    vmin=iono_lim[0], vmax=iono_lim[1],
                    cmap=iono_cmap,rasterized=iono_rasterize)

        # Add a colorbar
        if plot_colorbar:
            cbax    = addColorbar(im, ax)
            _       = cbax.set_ylabel(iono_title)

    # Plot Ray Paths ###############################################################
    freq_s  = 'None'
    if all_ray_paths is not None:
        rpd         = all_ray_paths
        ray_ids     = rpd.ray_id.unique()
        for ray_id in ray_ids:
            tf  = rpd.ray_id == ray_id
            xx  = rpd[tf].ground_range/Re
            yy  = rpd[tf].height + Re
            aax.plot(xx,yy,color='white')
        f   = rpd.frequency.unique()
        if f.size == 1:
            freq_s  = '{:0.3f} MHz'.format(float(f))
        else:
            freq_s  = 'multi'

    if connecting_ray_path is not None:
        rpd         = connecting_ray_path
        ray_ids     = rpd.ray_id.unique()
        for ray_id in ray_ids:
            tf  = rpd.ray_id == ray_id
            xx  = rpd[tf].ground_range/Re
            yy  = rpd[tf].height + Re
            aax.plot(xx,yy,color='red',zorder=100,lw=3)

    # Plot Receiver ################################################################ 
    if 'rx_lat' is not None and 'rx_lon' is not None:
        rx_label    = md.get(rx_call,'Receiver')
        rx_theta    = rx_range/Re
        
        hndl    = aax.scatter([rx_theta],[Re],s=250,marker='*',color='red',zorder=100,clip_on=False,label=rx_label)
        aax.legend([hndl],[rx_label],loc='upper right',scatterpoints=1,fontsize='small')

    
    # Add titles and other important information.
    title       = []
    date_s      = date.strftime('%Y %b %d %H:%M UT')
    tx_lat_s    = '{:0.2f}'.format(tx_lat) + r'$^{\circ}$N'
    tx_lon_s    = '{:0.2f}'.format(tx_lon) + r'$^{\circ}$E'
    azm_s       = '{:0.1f}'.format(azm)   + r'$^{\circ}$'
    title.append(date_s)
    title.append('TX Origin: {}, {}; Azimuth: {}, Frequency: {}'.format(tx_lat_s,tx_lon_s,azm_s,freq_s))
    ax.set_title('\n'.join(title))

    title   = []
    if tx_call is not None:
        title.append('TX: {}'.format(tx_call))
    if rx_call is not None:
        title.append('RX: {}'.format(rx_call))
    ax.set_title('\n'.join(title),loc='left')

    return ax, aax, cbax

def plot_power_path(rt_obj,ax,maxground=4000.,ylim=(-175,-100)):
    rd      = rt_obj.rt_dct['raytrace']['all_ray_data'].sort_values('ground_range')
    srd     = rt_obj.rt_dct['raytrace'].get('connecting_ray_data')
    md      = rt_obj.rt_dct['metadata']

    plot_dct            = {}
    tmp, key            = ({},'rx_power_0_dB')
    plot_dct[key]       = tmp
    tmp['label']        = 'No Losses'
    tmp['color']        = 'r'

    tmp, key            = ({},'rx_power_dB')
    plot_dct[key]       = tmp
    tmp['label']        = 'Ground and Deviative Losses'
    tmp['color']        = 'g'

    tmp, key            = ({},'rx_power_O_dB')
    plot_dct[key]       = tmp
    tmp['label']        = 'O Mode'
    tmp['color']        = 'b'

    tmp, key            = ({},'rx_power_X_dB')
    plot_dct[key]       = tmp
    tmp['label']        = 'X Mode'
    tmp['color']        = 'c'

    plot_list       = []
    plot_list.append('rx_power_0_dB')
    plot_list.append('rx_power_dB')
    plot_list.append('rx_power_O_dB')
    plot_list.append('rx_power_X_dB')

    handles     = []
    labels      = []
    for param in plot_list:
        xx      = rd.ground_range.tolist()
        yy      = rd[param].tolist()
        if srd is not None:
            inx     = srd.ground_range.argmax()
            xx.append(srd.ground_range[inx])
            yy.append((srd[param])[inx])

            tmp = pd.DataFrame({'xx':xx,'yy':yy})
            tmp = tmp.sort_values('xx')
            xx  = tmp.xx
            yy  = tmp.yy
        label   = plot_dct[param].get('label',param)
        color   = plot_dct[param].get('color',None)
        handle, = ax.plot(xx,yy,label=label,marker='.',color=color)
        handles.append(handle)
        labels.append(label)

    if srd is not None:
        for param in plot_list:
            inx     = srd.ground_range.argmax()
            xx      = [srd.ground_range[inx]]
            yy      = [(srd[param])[inx]]
            color   = plot_dct[param].get('color',None)
            handle, = ax.plot(xx,yy,marker='*',ls=' ',color=color,
                    ms=10,zorder=100)

    if 'rx_lat' in md and 'rx_lon' in md:
        # Mark the receiver location.
        rx_label    = md.get('rx_label','Receiver')
        rx_range    = md.get('rx_range')
        ax.axvline(rx_range,color='r',ls='--')
        trans   = matplotlib.transforms.blended_transform_factory( ax.transData, ax.transAxes)
        hndl    = ax.scatter([rx_range],[0],s=250,marker='*',color='red',zorder=100,clip_on=False,transform=trans)
#        ax.legend([hndl],[rx_label],loc='upper right',scatterpoints=1,fontsize='small')

#    ax.legend(loc='upper left',fontsize='small')
    ax.legend(handles,labels,loc='lower left',fontsize='small')

    ax.set_xlim(0,maxground)
    ax.set_ylim(ylim)

    f       = rd.frequency.unique()
    if f.size == 1:
        freq_s  = '{:0.3f} MHz'.format(float(f))
    else:
        freq_s  = 'multi'

    title       = []
#    date_s      = date.strftime('%Y %b %d %H:%M UT')
#    tx_lat_s     = '{:0.2f}'.format(rt_dct['tx_lat'])  + r'$^{\circ}$N'
#    tx_lon_s     = '{:0.2f}'.format(rt_dct['tx_lon']) + r'$^{\circ}$E'
#    azm_s       = '{:0.1f}'.format(rt_dct['azm'])    + r'$^{\circ}$'
#    title.append(date_s)
#    title.append('TX Origin: {}, {}; Azimuth: {}, Frequency: {}'.format(tx_lat_s,tx_lon_s,azm_s,freq_s))
    line        = 'TX Power: {:0.1f} W, TX Gain: {:0.1f} dB, RX Gain: {:0.1f} dB'.format(
                md['tx_power'], md['gain_tx_db'], md['gain_rx_db'])
    title.append(line)
    ax.set_title('\n'.join(title))

    ax.set_xlabel('Ground Range [km]')
    ax.set_ylabel('Power [dBW]')

def plot_raytrace_and_power(rt_obj,
        iono_param='iono_pf_grid',maxground=4000.,maxalt=400.,
        output_file='output.png'):
    """
    Plot both the ray trace and recieved power plots.

    Arguments:
    * rt_obj:       Raytrace Object containing data.

    Keywords:
    * iono_param:   Parameter of background ionosphere. Choice of:
                       'iono_en_grid'
                       'iono_pf_grid'
                       'collision_freq'
    *maxground:     Maximum ground range [km]
    *maxalt:        Maximum altitude [km]

    """
    fig                 = plt.figure(figsize=(15,6))
    x_0,x_w             = (0.0, 0.95)
    rt_rect             = [x_0, 0.40, x_w, 0.45]

    x_scale             = 0.925
    xs_w                = x_scale * x_w
    xs_0                = x_0 + (x_w-xs_w)/2.
#    pwr_rect            = [xs_0, 0.275, xs_w, 0.30]
    pwr_rect            = [xs_0, 0.000, xs_w, 0.35]

    y_scale             = 0.5
    iono_cbar_hgt       = y_scale*rt_rect[3]
    iono_cbar_ypos      = rt_rect[1] + (rt_rect[3]-iono_cbar_hgt)/2.
    iono_cbar_rect      = [0.960, iono_cbar_ypos, 0.025, iono_cbar_hgt]

    horiz               = [Size.Scaled(1.0)]
    vert                = [Size.Scaled(1.0)]
    rt_div              = Divider(fig,rt_rect,horiz,vert,aspect=False)
    pwr_div             = Divider(fig,pwr_rect,horiz,vert,aspect=False)
    iono_cbar_div       = Divider(fig,iono_cbar_rect,horiz,vert,aspect=False)
    
    pos                 = {}
    pos['rt']           = rt_div.new_locator(  nx=0, ny=0)
    pos['pwr']          = pwr_div.new_locator( nx=0, ny=0)
    pos['iono_cbax']    = iono_cbar_div.new_locator(nx=0, ny=0) 

    ################################################################################
    ax, aax, cbax       = plot_iono_path_profile(rt_obj,iono_param=iono_param,maxground=maxground,maxalt=maxalt)

    ax.set_axes_locator(pos['rt'])
    cbax.set_axes_locator(pos['iono_cbax'])

    ax      = fig.add_subplot(111)
    plot_power_path(rt_obj,ax,maxground=maxground)
    ax.set_axes_locator(pos['pwr'])

    fig.savefig(output_file,bbox_inches='tight')
    plt.close(fig)

    return output_file


def addColorbar(mappable, ax):
    """ Append colorbar to axes

    Parameters
    ----------
    mappable :
        a mappable object
    ax :
        an axes object

    Returns
    -------
    cbax :
        colorbar axes object

    Notes
    -----
    This is mostly useful for axes created with :func:`curvedEarthAxes`.

    written by Sebastien, 2013-04

    """
    from mpl_toolkits.axes_grid1 import SubplotDivider, LocatableAxes, Size
    import matplotlib.pyplot as plt 

    fig1 = ax.get_figure()
    divider = SubplotDivider(fig1, *ax.get_geometry(), aspect=True)

    # axes for colorbar
    cbax = LocatableAxes(fig1, divider.get_position())

    h = [Size.AxesX(ax), # main axes
         Size.Fixed(0.1), # padding
         Size.Fixed(0.2)] # colorbar
    v = [Size.AxesY(ax)]

    _ = divider.set_horizontal(h)
    _ = divider.set_vertical(v)

    _ = ax.set_axes_locator(divider.new_locator(nx=0, ny=0))
    _ = cbax.set_axes_locator(divider.new_locator(nx=2, ny=0))

    _ = fig1.add_axes(cbax)

    _ = cbax.axis["left"].toggle(all=False)
    _ = cbax.axis["top"].toggle(all=False)
    _ = cbax.axis["bottom"].toggle(all=False)
    _ = cbax.axis["right"].toggle(ticklabels=True, label=True)

    _ = plt.colorbar(mappable, cax=cbax)

    return cbax

def curvedEarthAxes(rect=111, fig=None, minground=0., maxground=2000, minalt=0,
                    maxalt=500, Re=6371., nyticks=5, nxticks=4):
    """Create curved axes in ground-range and altitude

    Parameters
    ----------
    rect : Optional[int]
        subplot spcification
    fig : Optional[pylab.figure object]
        (default to gcf)
    minground : Optional[float]

    maxground : Optional[int]
        maximum ground range [km]
    minalt : Optional[int]
        lowest altitude limit [km]
    maxalt : Optional[int]
        highest altitude limit [km]
    Re : Optional[float] 
        Earth radius in kilometers
    nyticks : Optional[int]
        Number of y axis tick marks; default is 5
    nxticks : Optional[int]
        Number of x axis tick marks; deafult is 4

    Returns
    -------
    ax : matplotlib.axes object
        containing formatting
    aax : matplotlib.axes object
        containing data

    Example
    -------
        import numpy as np
        ax, aax = curvedEarthAxes()
        th = np.linspace(0, ax.maxground/ax.Re, 50)
        r = np.linspace(ax.Re+ax.minalt, ax.Re+ax.maxalt, 20)
        Z = exp( -(r - 300 - ax.Re)**2 / 100**2 ) * np.cos(th[:, np.newaxis]/th.max()*4*np.pi)
        x, y = np.meshgrid(th, r)
        im = aax.pcolormesh(x, y, Z.T)
        ax.grid()

    written by Sebastien, 2013-04

    """
    from matplotlib.transforms import Affine2D, Transform
    import mpl_toolkits.axisartist.floating_axes as floating_axes
    from matplotlib.projections import polar
    from mpl_toolkits.axisartist.grid_finder import FixedLocator, DictFormatter
    import numpy as np
    from pylab import gcf

    ang         = maxground / Re
    minang      = minground / Re
    angran      = ang - minang
    angle_ticks = [(0, "{:.0f}".format(minground))]
    while angle_ticks[-1][0] < angran:
        tang = angle_ticks[-1][0] + 1./nxticks*angran
        angle_ticks.append((tang, "{:.0f}".format((tang-minang)*Re)))

    grid_locator1   = FixedLocator([v for v, s in angle_ticks])
    tick_formatter1 = DictFormatter(dict(angle_ticks))

    altran      = float(maxalt - minalt)
    alt_ticks   = [(minalt+Re, "{:.0f}".format(minalt))]
    while alt_ticks[-1][0] < Re+maxalt:
        alt_ticks.append((altran / float(nyticks) + alt_ticks[-1][0], 
                          "{:.0f}".format(altran / float(nyticks) +
                                          alt_ticks[-1][0] - Re)))
    _ = alt_ticks.pop()
    grid_locator2   = FixedLocator([v for v, s in alt_ticks])
    tick_formatter2 = DictFormatter(dict(alt_ticks))

    tr_rotate       = Affine2D().rotate(np.pi/2-ang/2)
    tr_shift        = Affine2D().translate(0, Re)
    tr              = polar.PolarTransform() + tr_rotate

    grid_helper = \
        floating_axes.GridHelperCurveLinear(tr, extremes=(0, angran, Re+minalt,
                                                          Re+maxalt),
                                            grid_locator1=grid_locator1,
                                            grid_locator2=grid_locator2,
                                            tick_formatter1=tick_formatter1,
                                            tick_formatter2=tick_formatter2,)

    if not fig: fig = gcf()
    ax1 = floating_axes.FloatingSubplot(fig, rect, grid_helper=grid_helper)

    # adjust axis
    ax1.axis["left"].label.set_text(r"Alt. [km]")
    ax1.axis["bottom"].label.set_text(r"Ground range [km]")
    ax1.invert_xaxis()

    ax1.minground   = minground
    ax1.maxground   = maxground
    ax1.minalt      = minalt
    ax1.maxalt      = maxalt
    ax1.Re          = Re

    fig.add_subplot(ax1, transform=tr)

    # create a parasite axes whose transData in RA, cz
    aux_ax          = ax1.get_aux_axes(tr)

    # for aux_ax to have a clip path as in ax
    aux_ax.patch    = ax1.patch

    # but this has a side effect that the patch is drawn twice, and possibly
    # over some other artists. So, we decrease the zorder a bit to prevent this.
    ax1.patch.zorder=0.9

    return ax1, aux_ax

def plot_rx_power_timeseries(rt_objs,sTime=None,eTime=None,output_file='output.png'):
    """
    Plot RX Power Time Series given a list of rt_objs.
    """
    rx_power    = raytrace.extract_rx_power(rt_objs)

    md0         = rt_objs[0].rt_dct['metadata']

    freq        = md0.get('freq')
    tx_lat      = md0.get('tx_lat')
    tx_lon      = md0.get('tx_lon')
    rx_lat      = md0.get('rx_lat')
    rx_lon      = md0.get('rx_lon')

    if sTime is None or eTime is None:
        dates   = [x['date'] for x in rt_dcts]
        sTime   = min(dates)
        eTime   = max(dates)

    fig         = plt.figure(figsize=(10,6.5))
    ax          = fig.add_subplot(1,1,1)
    param       = 'rx_power_dB'
    xx          = rx_power.index
    yy          = rx_power[param]
    ax.plot(xx,yy,marker='.')
    ax.set_xlim(sTime,eTime)

    fmt = mdates.DateFormatter('%d %b\n%H:%M')
    ax.xaxis.set_major_formatter(fmt)
    
    for xtl in ax.xaxis.get_ticklabels():
        xtl.set_rotation(70)
#        xtl.set_verticalalignment('top')
        xtl.set_horizontalalignment('center')
        xtl.set_fontsize('large')
        xtl.set_fontweight('bold')

    for ytl in ax.yaxis.get_ticklabels():
        ytl.set_fontsize('large')
        ytl.set_fontweight('bold')

    title       = []
    date_s      = ' - '.join([sTime.strftime('%Y %b %d %H:%M UT'),eTime.strftime('%Y %b %d %H:%M UT')])
    tx_lat_s    = '{:0.2f}'.format(tx_lat) + r'$^{\circ}$N'
    tx_lon_s    = '{:0.2f}'.format(tx_lon) + r'$^{\circ}$E'
    rx_lat_s    = '{:0.2f}'.format(rx_lat) + r'$^{\circ}$N'
    rx_lon_s    = '{:0.2f}'.format(rx_lon) + r'$^{\circ}$E'
    freq_s      = '{:0.3f} MHz'.format(freq)
    title.append(date_s)
    title.append('TX: {}, {}; RX: {}, {}, Frequency: {}'.format(tx_lat_s,tx_lon_s,rx_lat_s,rx_lon_s,freq_s))

    tx_call = md0.get('tx_call')
    rx_call = md0.get('rx_call')
    if tx_call is not None and rx_call is not None:
        title.append('TX: {}; RX: {}'.format(tx_call,rx_call))

    ax.set_title('\n'.join(title))

    ax.set_xlabel('Time [UT]')
#    ax.set_ylabel(param)
    ax.set_ylabel('Predicted RX dB')

    fig.tight_layout()

    fig.savefig(output_file,bbox_inches='tight')
    plt.close(fig)
