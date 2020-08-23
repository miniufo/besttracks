# -*- coding: utf-8 -*-
"""
Created on 2020.08.01

@author: MiniUFO
Copyright 2018. All rights reserved. Use is subject to license terms.
"""
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeat
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

undef = -9999

"""
Some useful functions for plotting
"""
def plot_tracks(TCs, figsize=(12,6), fontsize=15, size=60,
                linewidth=2, line_color=(0.4, 0.4, 0.4), add_legend=True,
                legend_loc='upper left', xlint=None, ylint=None, title=None,
                trackonly=False):
    """
    Plot the tracks on a map given TC data.

    Parameters
    ----------
    TCs: TCSet
        A set of TCs.

    Returns
    -------
    ax: axe
        Plot axe handle.
    """
    xmin, xmax = __get_TCs_range(TCs, 'LON')
    ymin, ymax = __get_TCs_range(TCs, 'LAT')
    
    xmin, xmax, ymin, ymax = __adjust_extent(xmin, xmax, ymin, ymax)
        
    reso = '10m'
    
    if xmax - xmin > 60:
        reso = '50m'
    
    _, ax = __prepare_background((xmin, xmax, ymin, ymax), reso,
                              xlint, ylint, figsize, fontsize)
    
    for i, tc in enumerate(TCs):
        if i != 0:
            add_legend = False
        
        plot_track(tc, ax=ax, figsize=figsize, fontsize=fontsize,
                   size=size, reso=reso, linewidth=linewidth, title='',
                   line_color=line_color, add_legend=add_legend,
                   legend_loc=legend_loc, xlint=xlint, ylint=ylint,
                   xlim=[xmin, xmax], ylim=[ymin, ymax],
                   trackonly=trackonly)
    
    if title == None:
        ax.set_title('TC tracks', fontsize=fontsize)
    else:
        ax.set_title(title, fontsize=fontsize)
    
    if add_legend:
        ax.scatter([xmin-1], [ymin-1], color='b', s=size/3,
                    transform=ccrs.PlateCarree(), zorder=1,
                    edgecolor='k', linewidth=0.5, label='TD stage')
        ax.scatter([xmin-1], [ymin-1], color=(0.3, 1, 0.3), s=size/2,
                    transform=ccrs.PlateCarree(), zorder=1,
                    edgecolor='k', linewidth=0.5, label='TS stage')
        ax.scatter([xmin-1], [ymin-1], color='r', s=size,
                    transform=ccrs.PlateCarree(), zorder=1,
                    edgecolor='k', linewidth=0.5, label='TY stage')
        ax.scatter([xmin-1], [ymin-1], color='yellow', s=size/2,
                    transform=ccrs.PlateCarree(), zorder=1,
                    edgecolor='k', linewidth=0.5, label='EC stage')
        ax.scatter([xmin-1], [ymin-1], color=(0.8, 0.8, 0.8), s=size/3,
                    transform=ccrs.PlateCarree(), zorder=1,
                    edgecolor='k', linewidth=0.5, label='Others')
        ax.legend(loc=legend_loc, fontsize=fontsize-3, edgecolor='k',
                  ncol=1, borderaxespad=0.8, frameon=True, shadow=False)
    
    return ax

def plot_track(TC, ax=None, figsize=(12,6), fontsize=15, size=60, reso='10m',
               linewidth=2, line_color=(0.4, 0.4, 0.4), add_legend=True,
               legend_loc='upper left', xlint=None, ylint=None, title=None,
               xlim=None, ylim=None, trackonly=False):
    """
    Plot the track on a map given a TC.

    Parameters
    ----------
    TC: TC
        A single TC.

    Returns
    -------
    ax: axe
        Plot axe handle.
    """
    if xlim == None and ylim == None:
        xmin, xmax = __get_TC_range(TC, 'LON')
        ymin, ymax = __get_TC_range(TC, 'LAT')
        
        xmin, xmax, ymin, ymax = __adjust_extent(xmin, xmax, ymin, ymax)
    else:
        xmin, xmax = xlim
        ymin, ymax = ylim
    
    if ax == None:
        _, ax = __prepare_background((xmin, xmax, ymin, ymax), reso,
                            xlint, ylint, figsize, fontsize)
        if title == None:
            ax.set_title('track of TC {0:s} ({1:s})'.format(TC.name, TC.ID),
                         fontsize=fontsize)
        else:
            ax.set_title(title, fontsize=fontsize)
        
    if add_legend:
        ax.scatter([xmin-1], [ymin-1], color='b', s=size/3,
                   transform=ccrs.PlateCarree(), zorder=1,
                   edgecolor='k', linewidth=0.5, label='TD stage')
        ax.scatter([xmin-1], [ymin-1], color=(0.3, 1, 0.3), s=size/2,
                   transform=ccrs.PlateCarree(), zorder=1,
                   edgecolor='k', linewidth=0.5, label='TS stage')
        ax.scatter([xmin-1], [ymin-1], color='r', s=size,
                   transform=ccrs.PlateCarree(), zorder=1,
                   edgecolor='k', linewidth=0.5, label='TY stage')
        ax.scatter([xmin-1], [ymin-1], color='yellow', s=size/2,
                   transform=ccrs.PlateCarree(), zorder=1,
                   edgecolor='k', linewidth=0.5, label='EC stage')
        ax.scatter([xmin-1], [ymin-1], color=(0.8, 0.8, 0.8), s=size/3,
                   transform=ccrs.PlateCarree(), zorder=1,
                   edgecolor='k', linewidth=0.5, label='Others')
        ax.legend(loc=legend_loc, fontsize=fontsize-3, edgecolor='k',
                  ncol=1, borderaxespad=0.8, frameon=True, shadow=False)
    
    recs = TC.records
    
    ax.plot(recs['LON'], recs['LAT'], color=line_color,
        transform=ccrs.PlateCarree(), zorder=0, linewidth=linewidth)
    
    if not trackonly:
        if 'TYPE' in recs:
            #################### TD ####################
            tctype = recs['TYPE'] == 'TD'
            data = recs.loc[tctype]
            ax.scatter(data['LON'], data['LAT'], color='b', s=size/3,
                       transform=ccrs.PlateCarree(), zorder=1,
                       edgecolor='k', linewidth=0.5)
            #################### TS ####################
            tctype = recs['TYPE'].isin(['TS', 'STS'])
            data = recs.loc[tctype]
            ax.scatter(data['LON'], data['LAT'], color=(0.3, 1, 0.3), s=size/2,
                       transform=ccrs.PlateCarree(), zorder=1,
                       edgecolor='k', linewidth=0.5)
            #################### TY ####################
            tctype = recs['TYPE'].isin(['TY', 'STY', 'HU'])
            data = recs.loc[tctype]
            ax.scatter(data['LON'], data['LAT'], color='r', s=size,
                       transform=ccrs.PlateCarree(), zorder=1,
                       edgecolor='k', linewidth=0.5)
            #################### EC ####################
            tctype = recs['TYPE'] == 'EC'
            data = recs.loc[tctype]
            ax.scatter(data['LON'], data['LAT'], color='yellow', s=size/2,
                       transform=ccrs.PlateCarree(), zorder=1,
                       edgecolor='k', linewidth=0.5)
            #################### OTHERS ####################
            tctype = recs['TYPE'].isin(['OTHERS', 'NR'])
            data = recs.loc[tctype]
            ax.scatter(data['LON'], data['LAT'], color=(0.8, 0.8, 0.8), s=size/3,
                       transform=ccrs.PlateCarree(), zorder=1,
                       edgecolor='k', linewidth=0.5)
        else:
            data = recs
            ax.scatter(data['LON'], data['LAT'], color=(0.3, 0.3, 0.3), s=size,
                           transform=ccrs.PlateCarree(), zorder=1,
                           edgecolor='k', linewidth=0.8)
    
    return ax


def plot_intensity(TC, ax=None, figsize=(10,5), fontsize=15):
    """
    Plot the intensity of the given TC.

    Parameters
    ----------
    TC: TC
        A single TC.
    ax: Axe
        Axe for plotting
    figsize: tuple
        Figure size of (12, 6)
    fontsize: int
        Size of the font (title, label and legend)

    Returns
    -------
    ax: Axe
        minimum and maximum values
    """
    recs = TC.records
    
    ax1 = None
    
    if ax == None:
        fig = plt.figure(figsize=figsize)
        ax1 = fig.add_subplot()
        ax2 = ax1.twinx()
    else:
        ax1 = ax
        ax2 = ax1.twinx()
    
    prslocs_m, wndlocs_m, prsylim_m, wndylim_m, \
        hasP, hasW = __get_intensity_range(TC)
    
    wnd = recs['WND' ].where(recs['PRS']!=undef)
    prs = recs['PRS' ].where(recs['PRS']!=undef)
    tim = recs['TIME']
    
    if hasP:
        if hasW:
            ax1.plot(tim, prs, 'b-', linewidth=2, label='Pmin')
            ax2.plot(tim, wnd, 'r-', linewidth=2, label='Wmax')
            
            h1, l1 = ax1.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            
            ax1.legend(h1+h2, l1+l2, loc='upper left', fontsize=fontsize)
            ax1.set_title('intensity for TC {0:s} ({1:s})'.format(TC.name, TC.ID),
                          fontsize=fontsize)
            ax1.tick_params(axis='both', labelsize=fontsize-2)
            ax2.tick_params(axis='both', labelsize=fontsize-2)
            ax1.grid(b=True)
            ax2.grid(b=True)
            ax1.set_yticks(prslocs_m)
            ax2.set_yticks(wndlocs_m)
            ax1.set_ylim(prsylim_m)
            ax2.set_ylim(wndylim_m)
            ax1.set_xlim([tim[0], tim.iloc[-1]])
            ax1.set_ylabel('Pressure (hPa)', fontsize=fontsize-2)
            ax2.set_ylabel('Wind speed ({0:s})'.format(TC.wndunit),
                           fontsize=fontsize-2)
            
            return ax1
            
        else:
            ax1.plot(tim, prs, 'b-', linewidth=2, label='Pmin')
            ax1.legend(loc='upper left', fontsize=fontsize)
            ax1.set_title('intensity for TC {0:s} ({1:s})'.format(TC.name, TC.ID),
                          fontsize=fontsize)
            ax1.tick_params(axis='both', labelsize=fontsize-2)
            ax1.grid(b=True)
            ax1.set_yticks(prslocs_m)
            ax1.set_ylim(prsylim_m)
            ax1.set_ylim([tim[0], tim.iloc[-1]])
            ax1.set_xlabel('Pressure (hPa)', fontsize=fontsize-2)
            
            return ax1
    else:
        if hasW:
            ax1.plot(tim, prs-prs, 'b-', linewidth=2, label='Pmin')
            ax2.plot(tim, wnd, 'r-', linewidth=2, label='Wmax')
            
            h1, l1 = ax1.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            
            ax1.legend(h1+h2, l1+l2, loc='upper left', fontsize=fontsize)
            ax1.set_title('intensity for TC {0:s} ({1:s})'.format(TC.name, TC.ID),
                          fontsize=fontsize)
            ax1.tick_params(axis='both', labelsize=fontsize-2)
            ax2.tick_params(axis='both', labelsize=fontsize-2)
            ax1.grid(b=True)
            ax2.grid(b=True)
            ax1.set_yticks(prslocs_m)
            ax2.set_yticks(wndlocs_m)
            ax1.set_ylim(prsylim_m)
            ax2.set_ylim(wndylim_m)
            ax1.set_xlim([tim[0], tim.iloc[-1]])
            ax1.set_ylabel('Pressure (hPa)', fontsize=fontsize-2)
            ax2.set_ylabel('Wind speed ({0:s})'.format(TC.wndunit),
                           fontsize=fontsize-2)
            
            return ax1
        else:
            print('no valid intensity data, nothing can be plotted')
            
            ax1.set_title('intensity for TC {0:s} ({1:s})'.format(TC.name, TC.ID),
                          fontsize=fontsize)
            ax1.tick_params(axis='both', labelsize=fontsize-2)
            ax2.tick_params(axis='both', labelsize=fontsize-2)
            ax1.set_ylabel('Pressure (hPa)', fontsize=fontsize-2)
            ax2.set_ylabel('Wind speed ({0:s})'.format(TC.wndunit),
                           fontsize=fontsize-2)
            
            return ax1


def plot(TC, ax=None, figsize=(12,6), fontsize=15, size=60, reso='10m',
         linewidth=2, line_color=(0.4, 0.4, 0.4), add_legend=True,
         legend_loc='upper left', xlint=None, ylint=None, title=None,
         xlim=None, ylim=None):
    """
    Plot the track and intensity of the given TC.

    Parameters
    ----------
    TC: TC
        A single TC.
    ax: Axe
        Axe for plotting
    fontsize: int
        Size of the font (title, label and legend)

    Returns
    -------
    ax: Axe
        minimum and maximum values
    """
    if xlim == None and ylim == None:
        xmin, xmax = __get_TC_range(TC, 'LON')
        ymin, ymax = __get_TC_range(TC, 'LAT')
        
        xmin, xmax, ymin, ymax = __adjust_extent(xmin, xmax, ymin, ymax)
    else:
        xmin, xmax = xlim
        ymin, ymax = ylim
    
    fig, ax1 = __prepare_background((xmin, xmax, ymin, ymax), reso,
                                    xlint, ylint, figsize, fontsize)
    
    ax1.set_title('track of TC {0:s} ({1:s})'.format(TC.name, TC.ID),
                  fontsize=fontsize)
    
    plot_track(TC, ax=ax1, fontsize=fontsize, size=size, reso=reso,
               linewidth=linewidth, line_color=line_color,
               add_legend=add_legend, legend_loc=legend_loc, xlint=xlint, 
               ylint=ylint, title=title, xlim=xlim, ylim=ylim)
    
    left, bottom, width, height = ax1.get_position().extents
    
    ax2 = fig.add_axes([width*1.1, bottom, width/1.8, height-bottom])
    
    plot_intensity(TC, ax=ax2, fontsize=fontsize)


"""
Helper (private) methods are defined below
"""
def __get_TCs_range(TCs, field):
    """
    Get the minimum and maximum values of the field in TCs.

    Parameters
    ----------
    TCs: TCSet
        A set of TCs.
    field: str
        A column field in TC.data, e.g., LON or LAT

    Returns
    -------
    re: tuple of float
        minimum and maximum values
    """
    data_min = []
    data_max = []

    for tc in TCs:
        data = tc.records[field]

        data_min.append(data.min())
        data_max.append(data.max())
    
    dmin = min(data_min)
    dmax = max(data_max)
    
    return dmin, dmax


def __prepare_background(extent, reso, xlint, ylint, figsize, fontsize):
    """
    Prepare a background plot for TC tracks.

    Parameters
    ----------
    extent: tuple
        Plot extent as (xmin, xmax, ymin, ymax).
    reso: str
        Resolution of land/ocean map, one of '10m', '50m', or '110m'.
    xlint: int
        Interval of x-labels.
    ylint: int
        Interval of y-labels.
    figsize: tuple
        Figure size of (width, height).

    Returns
    -------
    re: tuple of float
        minimum and maximum values
    """
    # guess full labels
    xmin, xmax, ymin, ymax = extent
    
    xlocs = __guess_lon_labels(xmax - xmin, interval=xlint)
    ylocs = __guess_lat_labels(ymax - ymin, interval=ylint)
    
    # select those in range
    xlocs = xlocs[np.logical_and(xlocs>xmin, xlocs<xmax)]
    ylocs = ylocs[np.logical_and(ylocs>ymin, ylocs<ymax)]
    
    proj = ccrs.PlateCarree(central_longitude=180)
    
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(projection=proj)

    ax.set_extent([xmin, xmax, ymin, ymax], ccrs.PlateCarree())
    ax.coastlines(resolution=reso)
    ax.add_feature(cfeat.OCEAN)
    ax.add_feature(cfeat.LAND)
    ax.set_xticks(xlocs, crs=ccrs.PlateCarree())
    ax.set_yticks(ylocs, crs=ccrs.PlateCarree())
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    ax.tick_params(axis='both', labelsize=fontsize-2)
    
    return fig, ax

def __get_intensity_range(TC):
    """
    Get the intensity plot range according to the peak intensity.

    Parameters
    ----------
    TC: TC
        A given TC.

    Returns
    ----------
    re: tuple
        A tuple of pressure labels, wind labels wndlocs_m,
        pressure limits, and wind limits wmdylim_m
        (prslocs_m, wndlocs_m, prsylim_m, wndylim_m)
    """
    prslocs_w = [985, 990, 995, 1000, 1005, 1010, 1015]
    wndlocs_w = [0, 10, 20, 30, 40, 50, 60]
    prsylim_w = [985, 1015]
    wndylim_w = [0, 60]
    
    prslocs_m = [950, 960, 970, 980, 990, 1000, 1010]
    wndlocs_m = [0, 15, 30, 45, 60, 75, 90]
    prsylim_m = [950, 1010]
    wndylim_m = [0, 90]
    
    prslocs_s = [900, 920, 940, 960, 980, 1000]
    wndlocs_s = [0, 25, 50, 75, 100, 125]
    prsylim_s = [900, 1010]
    wndylim_s = [0, 137.5]
    
    Pmin, Wmax = TC.peak_intensity()
    
    hasP, hasW = False, False
    
    if Wmax != undef:
        hasW = True
    
    if Pmin != undef:
        hasP = True
    
    if Wmax != undef:
        if Wmax <= 55:
            return prslocs_w, wndlocs_w, prsylim_w, wndylim_w, hasP, hasW
        elif Wmax < 85:
            return prslocs_m, wndlocs_m, prsylim_m, wndylim_m, hasP, hasW
        else:
            return prslocs_s, wndlocs_s, prsylim_s, wndylim_s, hasP, hasW
    elif Pmin != undef:
        if Pmin > 985:
            return prslocs_w, wndlocs_w, prsylim_w, wndylim_w, hasP, hasW
        elif Pmin > 960:
            return prslocs_m, wndlocs_m, prsylim_m, wndylim_m, hasP, hasW
        else:
            return prslocs_s, wndlocs_s, prsylim_s, wndylim_s, hasP, hasW
    else:
        print('warning: no valid intensity records')
        return prslocs_w, wndlocs_w, prsylim_w, wndylim_w, hasP, hasW
            
    


def __get_TC_range(TC, field):
    """
    Get the minimum and maximum values of the field in TCs.

    Parameters
    ----------
    TC: TC
        A single TC.
    field: str
        A column field in TC.data, e.g., LON or LAT

    Returns
    -------
    re: tuple of float
        minimum and maximum values
    """
    data = TC.records[field]
    return min(data), max(data)


def __adjust_extent(xmin, xmax, ymin, ymax):
    """
    Ajust the extent of the plot extent.
    
    Parameters
    ----------
    xmin: float
        minimum longitude.
    xmax: float
        maximum longitude.
    ymin: float
        minimum latitude.
    ymax: float
        maximum latitude.
    
    Returns
    -------
    re: tuple
        tuple of adjusted (xmin, xmax, ymin, ymax).
    """
    if xmax - xmin < 15:
        xmin -= 15
        xmax += 15
    elif xmax - xmin < 25:
        xmin -= 10
        xmax += 10
    else:
        xmin -= 5
        xmax += 5
    
    if ymax - ymin < 12:
        ymin -= 12
        ymax += 12
    elif ymax - ymin < 18:
        ymin -= 8
        ymax += 8
    else:
        ymin -= 4
        ymax += 4
    
    return xmin, xmax, ymin, ymax


def __guess_lon_labels(extent, interval=None):
    """
    Guess longitudinal labels given an extent.
    
    Parameters
    ----------
    extent : float
        Longitudinal extent = lon_max - lon_min.
    
    Returns
    -------
    result : np.array of floats
        Longitudinal labels.
    """
    if interval is None:
        if   extent >= 240 : interval = 60
        elif extent >= 160 : interval = 40
        elif extent >= 120 : interval = 30
        elif extent >= 80  : interval = 20
        elif extent >= 40  : interval = 10
        elif extent >= 20  : interval = 5
        elif extent >= 12  : interval = 3
        elif extent >= 8   : interval = 2
        elif extent >= 4   : interval = 1
        elif extent >= 2   : interval = 0.5
        elif extent >= 0.8 : interval = 0.2
        elif extent >= 0.4 : interval = 0.1
        elif extent >= 0.08: interval = 0.02
        else               : interval = 0.01
    
    return np.linspace(-360, 360, int(720.0/interval + 1))


def __guess_lat_labels(extent, interval=None):
    """
    Guess latitudinal labels given an extent.
    
    Parameters
    ----------
    extent : float
        latitudinal extent = lat_max - lat_min.
    
    Returns
    -------
    result : np.array of floats
        latitudinal labels.
    """
    if interval is None:
        if   extent >= 120 : interval = 30
        elif extent >= 80  : interval = 20
        elif extent >= 60  : interval = 15
        elif extent >= 40  : interval = 10
        elif extent >= 20  : interval = 5
        elif extent >= 12  : interval = 3
        elif extent >= 8   : interval = 2
        elif extent >= 4   : interval = 1
        elif extent >= 2   : interval = 0.5
        elif extent >= 0.8 : interval = 0.2
        elif extent >= 0.4 : interval = 0.1
        elif extent >= 0.08: interval = 0.02
        else               : interval = 0.01
    
    return np.linspace(-90, 90, int(180.0/interval + 1))

