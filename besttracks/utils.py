# -*- coding: utf-8 -*-
"""
Created on 2020.08.01

@author: MiniUFO
Copyright 2018. All rights reserved. Use is subject to license terms.
"""
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeat
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from .core import ParticleSet, TCSet, DrifterSet, Particle, TC, Drifter

undef = -9999

"""
Some useful functions for plotting
"""
def plot_tracks(ps, figsize=(12,6), fontsize=15, size=60,
                linewidth=2, line_color=(0.4, 0.4, 0.4), add_legend=False,
                scatter_color=(0.3, 0.3, 0.3),
                legend_loc='upper left', xlint=None, ylint=None, title=None,
                xlim=None, ylim=None, trackonly=False):
    """
    Plot the tracks on a map given particle dataset.

    Parameters
    ----------
    ps: ParticleSet
        A set of Lagrangian particles.

    Returns
    -------
    ax: axe
        Plot axe handle.
    """
    _, ax, _, (xmin, xmax, ymin, ymax) = \
            __prepare_background(ps, xlint, ylint, figsize, fontsize,
                                 xlim, ylim, adjust=True)
    
    for i, p in enumerate(ps):
        if i != 0:
            add_legend = False
        
        pplot, _ = __to_pplot(p)
        
        plot_track(pplot, ax=ax, figsize=figsize, fontsize=fontsize,
                   size=size, linewidth=linewidth, title='',
                   line_color=line_color, add_legend=add_legend,
                   legend_loc=legend_loc, xlint=xlint, ylint=ylint,
                   xlim=[xmin, xmax], ylim=[ymin, ymax],
                   trackonly=trackonly, scatter_color=scatter_color)
    
    if title == None:
        ax.set_title('tracks', fontsize=fontsize)
    else:
        ax.set_title(title, fontsize=fontsize)
    
    if add_legend:
        ofs = 90
        if 'SCALE' in ps[0].records:
            ax.scatter([xmin-ofs], [ymin-ofs], color='b', s=size/3,
                       transform=ccrs.PlateCarree(), zorder=1,
                       edgecolor='k', linewidth=0.5, label='TD stage')
            ax.scatter([xmin-ofs], [ymin-ofs], color=(0.3, 1, 0.3), s=size/3,
                       transform=ccrs.PlateCarree(), zorder=1,
                       edgecolor='k', linewidth=0.5, label='TS stage')
            ax.scatter([xmin-ofs], [ymin-ofs], color='yellow', s=size/2,
                       transform=ccrs.PlateCarree(), zorder=1,
                       edgecolor='k', linewidth=0.5, label='CAT1 stage')
            ax.scatter([xmin-ofs], [ymin-ofs], color='orange', s=size/2,
                       transform=ccrs.PlateCarree(), zorder=1,
                       edgecolor='k', linewidth=0.5, label='CAT2 stage')
            ax.scatter([xmin-ofs], [ymin-ofs], color='pink', s=size,
                       transform=ccrs.PlateCarree(), zorder=1,
                       edgecolor='k', linewidth=0.5, label='CAT3 stage')
            ax.scatter([xmin-ofs], [ymin-ofs], color='r', s=size,
                       transform=ccrs.PlateCarree(), zorder=1,
                       edgecolor='k', linewidth=0.5, label='CAT4 stage')
            ax.scatter([xmin-ofs], [ymin-ofs], color=(0.6, 0, 0), s=size,
                       transform=ccrs.PlateCarree(), zorder=1,
                       edgecolor='k', linewidth=0.5, label='CAT5 stage')
            ax.legend(loc=legend_loc, fontsize=fontsize-3, edgecolor='k',
                      ncol=1, borderaxespad=0.8, frameon=True, shadow=False)
        else:
            ax.scatter([xmin-ofs], [ymin-ofs], color='b', s=size/3,
                        transform=ccrs.PlateCarree(), zorder=1,
                        edgecolor='k', linewidth=0.5, label='TD stage')
            ax.scatter([xmin-ofs], [ymin-ofs], color=(0.3, 1, 0.3), s=size/2,
                        transform=ccrs.PlateCarree(), zorder=1,
                        edgecolor='k', linewidth=0.5, label='TS stage')
            ax.scatter([xmin-ofs], [ymin-ofs], color='r', s=size,
                        transform=ccrs.PlateCarree(), zorder=1,
                        edgecolor='k', linewidth=0.5, label='TY stage')
            ax.scatter([xmin-ofs], [ymin-ofs], color='yellow', s=size/2,
                        transform=ccrs.PlateCarree(), zorder=1,
                        edgecolor='k', linewidth=0.5, label='EC stage')
            ax.scatter([xmin-ofs], [ymin-ofs], color=(0.8, 0.8, 0.8), s=size/3,
                        transform=ccrs.PlateCarree(), zorder=1,
                        edgecolor='k', linewidth=0.5, label='Others')
            ax.legend(loc=legend_loc, fontsize=fontsize-3, edgecolor='k',
                      ncol=1, borderaxespad=0.8, frameon=True, shadow=False)
    
    return ax


def plot_track(p, ax=None, figsize=(12,6), fontsize=15, size=60,
               line_color=(0.4, 0.4, 0.4), add_legend=False,
               scatter_color=(0.3, 0.3, 0.3), linewidth=2,
               legend_loc='upper left', xlint=None, ylint=None, title=None,
               xlim=None, ylim=None, trackonly=False, scatteronly=False):
    """
    Plot the track on a map given a Lagrangian particle.

    Parameters
    ----------
    p: Particle
        A single Lagrangian particle.

    Returns
    -------
    ax: axe
        Plot axe handle.
    """
    if ax == None:
        _, ax, pplot, (xmin, xmax, ymin, ymax) = \
                    __prepare_background(p, xlint, ylint, figsize, fontsize,
                                         xlim, ylim, adjust=True)
        
        if title == None:
            ax.set_title('track of Particle ({1:s})'
                         .format(str(type(p)),str(pplot.ID)),
                         fontsize=fontsize)
        else:
            ax.set_title(title, fontsize=fontsize)
    else:
        pplot = p
        
        if xlim is not None and ylim is not None:
            xmin, xmax = xlim
            ymin, ymax = ylim
        else:
            xmin, xmax = __get_p_range(pplot, 'LON')
            ymin, ymax = __get_p_range(pplot, 'LAT')
    
    recs = pplot.records
    
    if add_legend:
        ofs = 90
        if 'SCALE' in recs:
            ax.scatter([xmin-ofs], [ymin-ofs], color='b', s=size/3,
                       transform=ccrs.PlateCarree(), zorder=1,
                       edgecolor='k', linewidth=0.5, label='TD stage')
            ax.scatter([xmin-ofs], [ymin-ofs], color=(0.3, 1, 0.3), s=size/3,
                       transform=ccrs.PlateCarree(), zorder=1,
                       edgecolor='k', linewidth=0.5, label='TS stage')
            ax.scatter([xmin-ofs], [ymin-ofs], color='yellow', s=size/2,
                       transform=ccrs.PlateCarree(), zorder=1,
                       edgecolor='k', linewidth=0.5, label='CAT1 stage')
            ax.scatter([xmin-ofs], [ymin-ofs], color='orange', s=size/2,
                       transform=ccrs.PlateCarree(), zorder=1,
                       edgecolor='k', linewidth=0.5, label='CAT2 stage')
            ax.scatter([xmin-ofs], [ymin-ofs], color='pink', s=size,
                       transform=ccrs.PlateCarree(), zorder=1,
                       edgecolor='k', linewidth=0.5, label='CAT3 stage')
            ax.scatter([xmin-ofs], [ymin-ofs], color='r', s=size,
                       transform=ccrs.PlateCarree(), zorder=1,
                       edgecolor='k', linewidth=0.5, label='CAT4 stage')
            ax.scatter([xmin-ofs], [ymin-ofs], color=(0.6, 0, 0), s=size,
                       transform=ccrs.PlateCarree(), zorder=1,
                       edgecolor='k', linewidth=0.5, label='CAT5 stage')
            ax.legend(loc=legend_loc, fontsize=fontsize-3, edgecolor='k',
                      ncol=1, borderaxespad=0.8, frameon=True, shadow=False)
        else:
            ax.scatter([xmin-ofs], [ymin-ofs], color='b', s=size/3,
                       transform=ccrs.PlateCarree(), zorder=1,
                       edgecolor='k', linewidth=0.5, label='TD stage')
            ax.scatter([xmin-ofs], [ymin-ofs], color=(0.3, 1, 0.3), s=size/2,
                       transform=ccrs.PlateCarree(), zorder=1,
                       edgecolor='k', linewidth=0.5, label='TS stage')
            ax.scatter([xmin-ofs], [ymin-ofs], color='r', s=size,
                       transform=ccrs.PlateCarree(), zorder=1,
                       edgecolor='k', linewidth=0.5, label='TY stage')
            ax.scatter([xmin-ofs], [ymin-ofs], color='yellow', s=size/2,
                       transform=ccrs.PlateCarree(), zorder=1,
                       edgecolor='k', linewidth=0.5, label='EC stage')
            ax.scatter([xmin-ofs], [ymin-ofs], color=(0.8, 0.8, 0.8), s=size/3,
                       transform=ccrs.PlateCarree(), zorder=1,
                       edgecolor='k', linewidth=0.5, label='Others')
            ax.legend(loc=legend_loc, fontsize=fontsize-3, edgecolor='k',
                      ncol=1, borderaxespad=0.8, frameon=True, shadow=False)
    
    if not scatteronly:
        ax.plot(recs['LON'], recs['LAT'], color=line_color,
                transform=ccrs.PlateCarree(), zorder=0, linewidth=linewidth)
    
    if not trackonly:
        if 'SCALE' in recs:
            #################### TD ####################
            tctype = recs['SCALE'] == 'TD'
            data = recs.loc[tctype]
            ax.scatter(data['LON'], data['LAT'], color='b', s=size/4,
                       transform=ccrs.PlateCarree(), zorder=1,
                       edgecolor='k', linewidth=0.5)
            #################### TS ####################
            tctype = recs['SCALE'] == 'TS'
            data = recs.loc[tctype]
            ax.scatter(data['LON'], data['LAT'], color=(0.3, 1, 0.3), s=size/4,
                       transform=ccrs.PlateCarree(), zorder=1,
                       edgecolor='k', linewidth=0.5)
            #################### TY ####################
            tctype = recs['SCALE'] == 'CAT1'
            data = recs.loc[tctype]
            ax.scatter(data['LON'], data['LAT'], color='yellow', s=size/3,
                       transform=ccrs.PlateCarree(), zorder=1,
                       edgecolor='k', linewidth=0.5)
            #################### EC ####################
            tctype = recs['SCALE'] == 'CAT2'
            data = recs.loc[tctype]
            ax.scatter(data['LON'], data['LAT'], color='orange', s=size/3,
                       transform=ccrs.PlateCarree(), zorder=1,
                       edgecolor='k', linewidth=0.5)
            #################### OTHERS ####################
            tctype = recs['SCALE'] == 'CAT3'
            data = recs.loc[tctype]
            ax.scatter(data['LON'], data['LAT'], color='pink', s=size/2,
                       transform=ccrs.PlateCarree(), zorder=1,
                       edgecolor='k', linewidth=0.5)
            #################### OTHERS ####################
            tctype = recs['SCALE'] == 'CAT4'
            data = recs.loc[tctype]
            ax.scatter(data['LON'], data['LAT'], color='r', s=size/2,
                       transform=ccrs.PlateCarree(), zorder=1,
                       edgecolor='k', linewidth=0.5)
            #################### OTHERS ####################
            tctype = recs['SCALE'] == 'CAT5'
            data = recs.loc[tctype]
            ax.scatter(data['LON'], data['LAT'], color=(0.6, 0, 0), s=size/2,
                       transform=ccrs.PlateCarree(), zorder=1,
                       edgecolor='k', linewidth=0.5)
            
        elif 'TYPE' in recs:
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
            tctype = recs['TYPE'].isin(['EC', 'EX'])
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
            ax.scatter(data['LON'], data['LAT'], color=scatter_color, s=size,
                           transform=ccrs.PlateCarree(), zorder=1,
                           edgecolor='k', linewidth=0.8)
    return ax


def plot_intensity(tc, ax=None, figsize=(10,5), fontsize=15):
    """
    Plot the intensity of the given TC.

    Parameters
    ----------
    tc: TC
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
    recs = tc.records
    
    ax1 = None
    
    if ax == None:
        fig = plt.figure(figsize=figsize)
        ax1 = fig.add_subplot()
        ax2 = ax1.twinx()
    else:
        ax1 = ax
        ax2 = ax1.twinx()
    
    prslocs_m, wndlocs_m, prsylim_m, wndylim_m, \
        hasP, hasW = __get_intensity_range(tc)
    
    wnd = recs['WND' ].where(recs['WND']!=undef)
    prs = recs['PRS' ].where(recs['PRS']!=undef)
    tim = recs['TIME']
    
    if hasP:
        if hasW:
            ax1.plot(tim, prs, 'b-', linewidth=2, label='Pmin')
            ax2.plot(tim, wnd, 'r-', linewidth=2, label='Wmax')
            
            h1, l1 = ax1.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            
            ax1.legend(h1+h2, l1+l2, loc='upper left', fontsize=fontsize)
            ax1.set_title('intensity for TC {0:s} ({1:s})'.format(tc.name, tc.ID),
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
            ax2.set_ylabel('Wind speed ({0:s})'.format(tc.wndunit),
                           fontsize=fontsize-2)
            
            return ax1
            
        else:
            ax1.plot(tim, prs, 'b-', linewidth=2, label='Pmin')
            ax1.legend(loc='upper left', fontsize=fontsize)
            ax1.set_title('intensity for TC {0:s} ({1:s})'.format(tc.name, tc.ID),
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
            ax1.set_title('intensity for TC {0:s} ({1:s})'.format(tc.name, tc.ID),
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
            ax2.set_ylabel('Wind speed ({0:s})'.format(tc.wndunit),
                           fontsize=fontsize-2)
            
            return ax1
        else:
            print('no valid intensity data, nothing can be plotted')
            
            ax1.set_title('intensity for TC {0:s} ({1:s})'.format(tc.name, tc.ID),
                          fontsize=fontsize)
            ax1.tick_params(axis='both', labelsize=fontsize-2)
            ax2.tick_params(axis='both', labelsize=fontsize-2)
            ax1.set_ylabel('Pressure (hPa)', fontsize=fontsize-2)
            ax2.set_ylabel('Wind speed ({0:s})'.format(tc.wndunit),
                           fontsize=fontsize-2)
            
            return ax1


def plot_intensities(tcs, ax=None, figsize=(10,5), fontsize=15):
    """
    Plot the intensity of the given TC.

    Parameters
    ----------
    tcs: TCSet
        A TCSet.
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
    ax1 = None
    
    if ax == None:
        fig = plt.figure(figsize=figsize)
        ax1 = fig.add_subplot()
        ax2 = ax1.twinx()
    else:
        ax1 = ax
        ax2 = ax1.twinx()
    
    prslocs_m, wndlocs_m, prsylim_m, wndylim_m, \
        hasP, hasW = __get_intensity_range(tcs)
    
    if hasP:
        if hasW:
            for i, tc in enumerate(tcs):
                recs = tc.records
                
                wnd = recs['WND' ].where(recs['WND']!=undef)
                prs = recs['PRS' ].where(recs['PRS']!=undef)
                tim = recs['TIME']
                
                ax1.plot(tim, prs, 'b-', linewidth=2, label='Pmin')
                ax2.plot(tim, wnd, 'r-', linewidth=2, label='Wmax')
                
                if i == 0:
                    h1, l1 = ax1.get_legend_handles_labels()
                    h2, l2 = ax2.get_legend_handles_labels()
            
            tstr = min([tc.records['TIME'].min() for tc in tcs])
            tend = max([tc.records['TIME'].max() for tc in tcs])
                
            ax1.legend(h1+h2, l1+l2, loc='upper left', fontsize=fontsize)
            ax1.set_title('intensity for TC {0:s} ({1:s})'.format(tc.name, tc.ID),
                          fontsize=fontsize)
            ax1.tick_params(axis='both', labelsize=fontsize-2)
            ax2.tick_params(axis='both', labelsize=fontsize-2)
            ax1.grid(b=True)
            ax2.grid(b=True)
            ax1.set_yticks(prslocs_m)
            ax2.set_yticks(wndlocs_m)
            ax1.set_ylim(prsylim_m)
            ax2.set_ylim(wndylim_m)
            ax1.set_xlim([tstr, tend])
            ax1.set_ylabel('Pressure (hPa)', fontsize=fontsize-2)
            ax2.set_ylabel('Wind speed ({0:s})'.format(tc.wndunit),
                           fontsize=fontsize-2)
            
            return ax1
            
        else:
            for i, tc in enumerate(tcs):
                recs = tc.records
                
                prs = recs['PRS' ].where(recs['PRS']!=undef)
                tim = recs['TIME']
                
                ax1.plot(tim, prs, 'b-', linewidth=2, label='Pmin')
                
                if i == 0:
                    h1, l1 = ax1.get_legend_handles_labels()
                    h2, l2 = ax2.get_legend_handles_labels()
            
            tstr = min([tc.records['TIME'].min() for tc in tcs])
            tend = max([tc.records['TIME'].max() for tc in tcs])
            
            ax1.legend(loc='upper left', fontsize=fontsize)
            ax1.set_title('intensity for TC {0:s} ({1:s})'.format(tc.name, tc.ID),
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
            for i, tc in enumerate(tcs):
                recs = tc.records
                
                wnd = recs['WND' ].where(recs['WND']!=undef)
                prs = recs['PRS' ].where(recs['PRS']!=undef)
                tim = recs['TIME']
                
                ax1.plot(tim, prs-prs, 'b-', linewidth=2, label='Pmin')
                ax2.plot(tim, wnd, 'r-', linewidth=2, label='Wmax')
                
                if i == 0:
                    h1, l1 = ax1.get_legend_handles_labels()
                    h2, l2 = ax2.get_legend_handles_labels()
            
            tstr = min([tc.records['TIME'].min() for tc in tcs])
            tend = max([tc.records['TIME'].max() for tc in tcs])
            
            ax1.legend(h1+h2, l1+l2, loc='upper left', fontsize=fontsize)
            ax1.set_title('intensity for TC {0:s} ({1:s})'.format(tc.name, tc.ID),
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
            ax2.set_ylabel('Wind speed ({0:s})'.format(tc.wndunit),
                           fontsize=fontsize-2)
            
            return ax1
        else:
            print('no valid intensity data, nothing can be plotted')
            
            ax1.set_title('intensity for TC {0:s} ({1:s})'.format(tc.name, tc.ID),
                          fontsize=fontsize)
            ax1.tick_params(axis='both', labelsize=fontsize-2)
            ax2.tick_params(axis='both', labelsize=fontsize-2)
            ax1.set_ylabel('Pressure (hPa)', fontsize=fontsize-2)
            ax2.set_ylabel('Wind speed ({0:s})'.format(tc.wndunit),
                           fontsize=fontsize-2)
            
            return ax1

def plot(tc, ax=None, figsize=(12,6), fontsize=15, size=60,
         linewidth=2, line_color=(0.4, 0.4, 0.4), add_legend=True,
         legend_loc='upper left', xlint=None, ylint=None, title=None,
         xlim=None, ylim=None, trackonly=False):
    """
    Plot the track and intensity of the given TC.

    Parameters
    ----------
    tc: TC
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
    fig, ax1, TCplt, (xmin, xmax, ymin, ymax) = \
            __prepare_background(tc, xlint, ylint, figsize, fontsize,
                                 xlim, ylim, adjust=True)
    
    if xlim == None:
        xlim = [xmin, xmax]
    
    if ylim == None:
        ylim = [ymin, ymax]
    
    ax1.set_title('track of TC {0:s} ({1:s})'.format(TCplt.name, TCplt.ID),
                  fontsize=fontsize)
    
    plot_track(TCplt, ax=ax1, fontsize=fontsize, size=size,
               linewidth=linewidth, line_color=line_color, trackonly=trackonly,
               add_legend=add_legend, legend_loc=legend_loc, xlint=xlint, 
               ylint=ylint, title=title, xlim=xlim, ylim=ylim)
    
    left, bottom, width, height = ax1.get_position().extents
    
    ax2 = fig.add_axes([width*1.1, bottom, width/1.8, height-bottom])
    
    plot_intensity(TCplt, ax=ax2, fontsize=fontsize)


def binning(lons, lats, var, ax=None, xlim=None, ylim=None, fontsize=15,
            xlint=None, ylint=None, figsize=(12,5), reso=1,
            title=None, add_sides=True):
    """
    Binning scatter data into a Eulerian statistical map.
    """
    if ax == None:
        fig, ax, _, (xmin, xmax, ymin, ymax) = \
        __prepare_background([lons.min(), lons.max(), lats.min(), lats.max()],
                             xlint, ylint, figsize, fontsize,
                             xlim, ylim, adjust=True)
    
    x, y, z = __kde2D(lons, lats, var, 1.0)
    
    ax.contourf(x, y, z, transform=ccrs.PlateCarree(),
                cmap=__transparent_jet(),
                levels=21, add_colorbar=True)
    
    if title == None:
        title = 'Gridding stat.'
    
    ax.set_title(title)
    
    if add_sides:
        left, bottom, width, height = ax.get_position().extents
        
        ax2 = fig.add_axes([width*1.05, bottom, width/4, height-bottom])
        ax3 = fig.add_axes([left, height*1.2, width-left, 0.2])
        
        xmean = xr.DataArray(z.sum(axis=1), dims='lon', coords={'lon': x[:,0]})
        ymean = xr.DataArray(z.sum(axis=0), dims='lat', coords={'lat': y[0,:]})
        
        xmean.plot(ax=ax3)
        ymean.plot(ax=ax2, y='lat')
        
        ax3.set_xlim([xmin, xmax])
        ax3.set_ylim([0, xmean.max()*1.1])
        ax2.set_ylim([ymin, ymax])
        ax2.set_xlim([0, ymean.max()*1.1])
        ax3.set_xlabel('')
        ax2.set_ylabel('')
    
    
    # import seaborn as sns
    
    # return sns.kdeplot(lon, lat, ax=ax, weights=vs, shade_lowest=False,
    #                    hist_kws={'weights': vs},
    #                    levels=20, thresh=0.05, bw=0.1)
    
    # return sns.jointplot(lon, lat, vs, kind='kde')
    
    # grid, binY, binX = np.histogram2d(lat, lon, weights=vs, bins=(binY, binX))
    
    # re = xr.DataArray(grid, name='ACE', dims=['lat','lon'],
    #                   coords={'lon':binX[:-1], 'lat':binY[:-1]})
    
    # return re
    
    return ax, (x, y, z)


def binning_particle(p, var=None, xlim=None, ylim=None, fontsize=15,
                     xlint=None, ylint=None, figsize=(12,5), reso=1,
                     title=None):
    """
    Binning the particle into a Eulerian statistical map.
    """
    lon = p.records['LON'].values
    lat = p.records['LAT'].values
    
    vs = None
    if var != None:
        vsm  = p.records[var].values
        vsm[vsm==undef] = 0
        
        vs  = vsm[vsm!=0]
        lon = lon[vsm!=0]
        lat = lat[vsm!=0]
    
    return binning(lon, lat, vs, None, xlim, ylim, fontsize, xlint, ylint,
                   figsize, reso, title, add_sides=True)


def binning_particles(ps, var=None, xlim=None, ylim=None, fontsize=15,
                xlint=None, ylint=None, figsize=(12,5), reso=1,
                title=None):
    """
    Binning the ParticleSet into a Eulerian statistical map.
    """
    lon = np.concatenate([p.records['LON'].values for p in ps])
    lat = np.concatenate([p.records['LAT'].values for p in ps])
    
    vs = None
    if var != None:
        vsm  = np.concatenate([p.records[var].values for p in ps])
        vsm[vsm==undef] = 0
        
        vs  = vsm[vsm!=0]
        lon = lon[vsm!=0]
        lat = lat[vsm!=0]
    
    return binning(lon, lat, vs, None, xlim, ylim, fontsize, xlint, ylint,
                     figsize, reso, title, add_sides=True)


def plot_timeseries(ps, freq='monthly', ax=None, figsize=(12,6), fontsize=16,
                    linewidth=2, add_legend=True, legend_loc='upper left'):
    """
    Plot the track and intensity of the given TC.

    Parameters
    ----------
    tc: TC
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
    import pandas as pd
    
    df_ace = [tc.records[['TIME','WND']] for tc in ps]
    df_ace = pd.concat(df_ace, axis=0).set_index('TIME')
    
    df_ace['ACE'] = df_ace['WND'] ** 2
    
    df_num = [tc.records.iloc[0] for tc in ps]
    df_num = pd.concat(df_num, axis=1).T.set_index('TIME')
    
    if freq == 'monthly':
        ace = df_ace.resample('M').sum()
        num = df_num.resample('M').count()
        
    elif freq == 'yearly':
        ace = df_ace.resample('Y').sum()
        num = df_num.resample('Y').count()
        
    elif freq == 'annual':
        ace = df_ace.groupby(df_ace.index.month).sum()
        num = df_num.groupby(df_num.index.month).count()
        
    else:
        raise Exception('unknown frequency: ' + freq +
                        ', should be one of [monthly, yearly, annual]')
    
    # print(ace)
    # print(num)
    
    import matplotlib.pyplot as plt
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    a1 = num['LAT'].plot.line(ax=ax, color='r', marker='o')
    ax.set_title('ACE and No. of TCs', fontsize=fontsize)
    ax.set_ylim(bottom=0)
    ax.set_xlabel('Time', fontsize=fontsize-1)
    ax.set_ylabel("Number of TCs", color="red", fontsize=fontsize-1)
    plt.setp(ax.get_xticklabels(), fontsize=fontsize-3)
    plt.setp(ax.get_yticklabels(), fontsize=fontsize-3)
    
    ax2=ax.twinx()
    a2 = (ace['ACE']/1e5).plot.line(ax=ax2, color='b', marker='x')
    ax2.set_ylim(bottom=0)
    ax2.set_xlabel('Time', fontsize=fontsize-1)
    ax2.set_ylabel('ACE', color='b', fontsize=fontsize-1)
    plt.setp(ax2.get_xticklabels(), fontsize=fontsize-3)
    plt.setp(ax2.get_yticklabels(), fontsize=fontsize-3)
    
    if add_legend:
        lns = [a1.get_lines()[0], a2.get_lines()[0]]
        
        plt.legend(lns, ['No. of TCs', 'ACE of TCs'], fontsize=fontsize-3,
                   loc=legend_loc)
    
    return ax




"""
Helper (private) methods are defined below
"""
def __transparent_jet():
    import matplotlib.colors as mcolors
    
    cmap = plt.cm.get_cmap('jet')
    clrs = cmap(np.arange(cmap.N))
    
    clrs[:16] = clrs[0] - clrs[0] + np.array([0, 0, 0, 0])
        
    return mcolors.LinearSegmentedColormap.from_list(cmap.name + "_trans",
                                                     clrs, cmap.N)

def __kde2D(x, y, weights=None, bandwidth=0.5, xbins=160j, ybins=160j, **kwargs): 
    """Build 2D kernel density estimate (KDE)."""
    from sklearn.neighbors import KernelDensity
    
    # create grid of sample locations (default: 100x100)
    xx, yy = np.mgrid[x.min()-2:x.max()+2:xbins, 
                      y.min()-2:y.max()+2:ybins]

    xy_sample = np.vstack([yy.ravel(), xx.ravel()]).T
    xy_train  = np.vstack([y, x]).T

    kde_skl = KernelDensity(bandwidth=bandwidth, **kwargs)
    kde_skl.fit(xy_train, sample_weight=weights)

    # score_samples() returns the log-likelihood of the samples
    z = np.exp(kde_skl.score_samples(xy_sample))
    
    return xx, yy, np.reshape(z, xx.shape)


def __to_pplot(p):
    """
    Change Particle to pplot so that crossing 0-degree line can be plotted.

    Parameters
    ----------
    p: Particle
        A given Lagrangian particle.

    Returns
    -------
    re: tuple of particle and cross0
        Particle after LON being modified and whether it crosses 0-degree line.
    """
    tmp = (np.abs(p.records['LON'].diff()) > 300)
    cross0 = tmp.any()
    count = tmp.sum()
    
    pplot = p
    
    if cross0:
        pplot = p.copy(copy_records=True)
        recs = pplot.records
        # print(recs['LON'])
        
        loops = 0
        
        while count != 0:
            idx = recs['LON'].loc[tmp].index
            
            if len(idx) == 1:
                beg = idx[0]
                end = len(recs)
            else:
                beg = idx[0]
                end = idx[1] - 1
            
            slc = slice(beg, end)
            # print('count == ' + str(count), beg, end, recs.loc[beg, 'LON'],
            #       recs.loc[beg-1, 'LON'])
            
            if  recs.loc[beg, 'LON'] - recs.loc[beg-1, 'LON'] < -300:
                recs.loc[slc, 'LON'] = recs.loc[slc,   'LON'] +  360
            elif recs.loc[beg,'LON'] - recs.loc[beg-1, 'LON'] >  300:
                recs.loc[slc, 'LON'] = recs.loc[slc,   'LON'] -  360
            else:
                raise Exception('should not reach here')
            
            tmp = (np.abs(recs['LON'].diff()) > 300)
            count = tmp.sum()
            
            loops += 1
            
            if loops > 20:
                raise Exception('dead loop')
        
        # print(pplot.records['LON'].values)
        
    return pplot, cross0


def __prepare_background(extent, xlint, ylint, figsize, fontsize,
                         xlim, ylim, adjust):
    """
    Prepare a background plot for particle tracks.
    
    Parameters
    ----------
    extent: tuple, Particle, or ParticleSet
        Plot extent as (xmin, xmax, ymin, ymax),
        or inferred from Particle or ParticleSet.
    xlint: int
        Interval of x-labels.
    ylint: int
        Interval of y-labels.
    figsize: tuple
        Figure size of (width, height).
    xlim: list
        List as [xmin, xmax].
    ylim: list
        List as [ymin, ymax].
    adjust: bool
        Whether adjust the extent.

    Returns
    -------
        re: tuple of figure, axis, pplot, extent
    """
    pplot = None
    central_lon = 180
    
    if isinstance(extent, tuple) or isinstance(extent, list):
        xmin, xmax, ymin, ymax = extent
    
    elif isinstance(extent, Particle):
        pplot, cross0 = __to_pplot(extent)
        
        if cross0:
            central_lon = 0
        
        xmin, xmax = __get_p_range(pplot, 'LON')
        ymin, ymax = __get_p_range(pplot, 'LAT')
        
        if xlim != None:
            xmin, xmax = xlim
            
        if ylim != None:
            ymin, ymax = ylim
    
    elif isinstance(extent, ParticleSet):
        xmin, xmax = __get_ps_range(extent, 'LON')
        ymin, ymax = __get_ps_range(extent, 'LAT')
        
        if xlim != None:
            xmin, xmax = xlim
        if ylim != None:
            ymin, ymax = ylim
    
    else:
        raise Exception('invalid type of extent ' + str(type(extent)))
    
    if adjust:
        xmin, xmax, ymin, ymax = __adjust_extent(xmin, xmax, ymin, ymax)
    
    # guess full labels
    xlocs = __guess_lon_labels(xmax - xmin, interval=xlint)
    ylocs = __guess_lat_labels(ymax - ymin, interval=ylint)
    
    # select those in range
    xlocs = xlocs[np.logical_and(xlocs>xmin, xlocs<xmax)]
    ylocs = ylocs[np.logical_and(ylocs>ymin, ylocs<ymax)]
    
    proj = ccrs.PlateCarree(central_longitude=central_lon)
    
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(projection=proj)
    
    ax.set_extent([xmin, xmax, ymin, ymax], ccrs.PlateCarree())
    ax.add_feature(cfeat.COASTLINE)
    ax.add_feature(cfeat.OCEAN)
    ax.add_feature(cfeat.LAND)
    ax.set_xticks(xlocs, crs=ccrs.PlateCarree())
    ax.set_yticks(ylocs, crs=ccrs.PlateCarree())
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    ax.tick_params(axis='both', labelsize=fontsize-2)
    
    return fig, ax, pplot, (xmin, xmax, ymin, ymax)


def __get_intensity_range(p):
    """
    Get the intensity plot range according to the peak intensity.

    Parameters
    ----------
    p: TC or TCSet
        A given TC or TCSet.

    Returns
    ----------
    re: tuple
        A tuple of pressure labels, wind labels wndlocs_m,
        pressure limits, and wind limits wmdylim_m
        (prslocs_m, wndlocs_m, prsylim_m, wndylim_m)
    """
    prslocs_w = np.array([985., 990., 995., 1000., 1005., 1010., 1015.])
    wndlocs_w = np.array([0., 10., 20., 30., 40., 50., 60.])
    prsylim_w = np.array([985., 1015.])
    wndylim_w = np.array([0., 60.])
    
    prslocs_m = np.array([950., 960., 970., 980., 990., 1000., 1010.])
    wndlocs_m = np.array([0., 15., 30., 45., 60., 75., 90.])
    prsylim_m = np.array([950., 1010.])
    wndylim_m = np.array([0., 90.])
    
    prslocs_s = np.array([900., 920., 940., 960., 980., 1000.])
    wndlocs_s = np.array([0., 25., 50., 75., 100., 125.])
    prsylim_s = np.array([900., 1010.])
    wndylim_s = np.array([0., 137.5])
    
    if isinstance(p, TC):
        unit = p.wndunit
        Pmin, Wmax = p.peak_intensity()
        
    elif isinstance(p, TCSet):
        unit = p[0].wndunit
        
        rep = [tc.peak_intensity()[0] for tc in p]
        rew = [tc.peak_intensity()[1] for tc in p]
        
        Pmin, Wmax = min(rep), max(rew)
    else:
        raise Exception('invalid type ' + str(type(p)))
    
    hasP, hasW = False, False
    
    if Wmax != undef:
        hasW = True
        
        if unit == 'm/s':
            Wmax /= 0.51444 # to knot
            
            wndlocs_w *= 0.6
            wndylim_w *= 0.6
            wndlocs_m *= 0.6
            wndylim_m *= 0.6
            wndlocs_s *= 0.6
            wndylim_s *= 0.6
    
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
#            print(prslocs_s, prsylim_s)
            return prslocs_s, wndlocs_s, prsylim_s, wndylim_s, hasP, hasW
    else:
        print('warning: no valid intensity records')
        return prslocs_w, wndlocs_w, prsylim_w, wndylim_w, hasP, hasW
            
    
def __get_ps_range(ps, field):
    """
    Get the minimum and maximum values of the field in a ParticleSet.

    Parameters
    ----------
    ps: ParticleSet
        A set of Lagrangian particles.
    field: str
        A column field in p.data, e.g., LON or LAT

    Returns
    -------
    re: tuple of float
        minimum and maximum values
    """
    data_min = []
    data_max = []

    for p in ps:
        data = p.records[field]

        data_min.append(data.min())
        data_max.append(data.max())
    
    dmin = min(data_min)
    dmax = max(data_max)
    
    return dmin, dmax


def __get_p_range(p, field):
    """
    Get the minimum and maximum values of the field in Particle.

    Parameters
    ----------
    p: Particle
        A single Lagrangian particle.
    field: str
        A column field in p.records, e.g., LON or LAT

    Returns
    -------
    re: tuple of float
        minimum and maximum values
    """
    data = p.records[field]
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
    
    return np.linspace(-420, 420, int(840.0/interval + 1))


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

