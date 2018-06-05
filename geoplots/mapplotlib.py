# -*- coding: utf-8 -*-
"""
@author: Wenchang Yang (yang.wenchang@uci.edu)
"""
# from .mypyplot import vcolorbar, hcolorbar

import numpy as np
import matplotlib.pyplot as plt
import os

def _get_grid_edges(x,y):
    '''Get grid edges from grid centers.'''
    if len(x)==2:
        # when x has only two elements: treat them as edges
        x_edge = x
    else:
        x_ = np.hstack((
            x[0]*2 - x[1],
            x,
            x[-1]*2 - x[-2]))
        x_edge = ( x_[:-1] + x_[1:] )/2.0
    if len(y)==2:
        y_edge = y
    else:
        y_ = np.hstack((
            y[0]*2 - y[1],
            y,
            y[-1]*2 - y[-2]))
        y_edge = ( y_[:-1] + y_[1:] )/2.0
    return x_edge, y_edge
def _load_coast():
    this_dir, this_filename = os.path.split(__file__)
    lon_path = os.path.join(this_dir, 'data', 'lon.npy')
    lat_path = os.path.join(this_dir, 'data', 'lat.npy')
    coast_lon = np.load(lon_path)
    coast_lat = np.load(lat_path)
    return coast_lon, coast_lat
def xticks2lon(new_xticks=None):
    '''Convert xticks to longitudes. '''
    if new_xticks is not None:
        plt.gca().set_xticks(new_xticks)
    current_xticks = plt.gca().get_xticks()
    current_xticklabels = plt.gca().get_xticklabels()
    new_xticklabels = current_xticklabels
    for i, x in enumerate(current_xticks):
        x = np.mod(x,360)
        if 0<x<180: #x>0 and x<180:
            # new_xticklabels[i] = str(int(x)) + '$^{\circ}$E'
            new_xticklabels[i] = '{}$^{{\circ}}$E'.format(x)
        elif 180<x<360: #x>180 and x<360:
            # new_xticklabels[i] = str(int(360-x))+'$^{\circ}$W'
            new_xticklabels[i] = '{}$^{{\circ}}$W'.format(360-x)
        elif -180<x<0:
            new_xticklabels[i] = '{}$^{{\circ}}$W'.format(-x)
        elif x==0 or x==180:
            # new_xticklabels[i] = str(int(x)) + '$^{\circ}$'
            new_xticklabels[i] = '{}$^{{\circ}}$'.format(x)
    plt.gca().set_xticklabels(new_xticklabels)
def yticks2lat(new_yticks=None):
    '''Convert yticks to latitudes. '''
    if new_yticks is not None:
        plt.gca().set_yticks(new_yticks)
    current_yticks = plt.gca().get_yticks()
    current_yticklabels = plt.gca().get_yticklabels()
    new_yticklabels = current_yticklabels
    for i, y in enumerate(current_yticks):
        if y>0:
            new_yticklabels[i] = str(int(y)) + '$^{\circ}$N'
        elif y<0 :
            new_yticklabels[i] = str(int(-y))+'$^{\circ}$S'
        else:
            new_yticklabels[i] = str(int(y)) + '$^{\circ}$'
    plt.gca().set_yticklabels(new_yticklabels)
#
# ######## plot basemap
def mapplot(lon=None, lat=None, **kw):
    '''Plot the basemap using coast data from Matlab.

    Parameters:
    -----------
        lon: vector-like longitude range.
        lat: vector-like latitude range.
        kw: optional parameters

    Optional parameters:
    --------------------
        ax: axis object, default=plt.gca()
        lonlatbox: [lon_start, lon_end, lat_start, lat_end] like array.
        lonlatbox_color: lonlat box color.
        lonlatbox_kw: dict parameters used in plotting lonlatbox.

        fill_continents: boolean, default is False.
        continents_color: color of continents, default is 0.33.
        coastlines_color: color of coastlines, default is 0.33.
        coastlines_width: line width of coastlines, default is 1.
    '''

    if lon is None:
        if plt.get_fignums():# figures already exist
            lon = plt.gca().get_xlim()
        else:
            lon = np.arange(0,360,2)
    if lat is None:
        if plt.get_fignums(): # figures already exist
            lat = plt.gca().get_ylim()
        else:
            lat = np.arange(-89, 90, 2)
    lon = np.squeeze(lon)
    lat = np.squeeze(lat)

    ax = kw.pop('ax', None)
    if ax is None:
        ax = plt.gca()
    plt.sca(ax)

    # get grid edges
    lon_edge, lat_edge = _get_grid_edges(lon, lat)
    lat_edge[lat_edge < -90] = -90
    lat_edge[lat_edge >  90] = 90

    # plot lonlatbox
    lonlatbox = kw.pop('lonlatbox', None)
    if lonlatbox is not None:
        lon0, lon1, lat0, lat1 = lonlatbox # unpack lonlatbox
        lon_ = np.array([
            np.linspace(lon0, lon1, 100),
            lon1*np.ones(100),
            np.linspace(lon1, lon0, 100),
            lon0*np.ones(100)
            ]).ravel()
        lat_ = np.array([
            lat0*np.ones(100),
            np.linspace(lat0, lat1, 100),
            lat1*np.ones(100),
            np.linspace(lat1, lat0, 100)
            ]).ravel()
        lonlatbox_kw = kw.pop('lonlatbox_kw', {})
        lonlatbox_color = kw.pop('lonlatbox_color', 'k')
        lonlatbox_color = lonlatbox_kw.pop('color', lonlatbox_color)
        plt.plot(lon_, lat_, color=lonlatbox_color, **lonlatbox_kw)

    # plot coast lines
    # load coast data
    lonlon, latlat = _load_coast()
    # fill continents or plot coast lines
    fill_continents = kw.pop('fill_continents', False)
    xticks = kw.pop('xticks', np.arange(-180, 360, 60))
    yticks = kw.pop('yticks', np.arange(-90, 91, 30))
    if fill_continents:
        # indices of western and eastern boundaries
        ivec = np.arange(lonlon.size)[latlat==-83.83]
        i0,i1,i2,i3,i4,i5 = ivec
        # correct the Antarctic map by adding coast points at the south pole
        lonlon = np.hstack((
            lonlon[:i0],
            lonlon[i0], lonlon[i0:i1+1], lonlon[i1],
            lonlon[i1+1:i2],
            lonlon[i2], lonlon[i2:i3+1], lonlon[i3],
            lonlon[i3+1:i4],
            lonlon[i4], lonlon[i4:i5+1], lonlon[i5],
            lonlon[i5+1:]))
        latlat = np.hstack((
            latlat[:i0],
            -90, latlat[i0:i1+1], -90,
            latlat[i1+1:i2],
            -90, latlat[i2:i3+1], -90,
            latlat[i3+1:i4],
            -90, latlat[i4:i5+1], -90,
            latlat[i5+1:]))
        continents_color = kw.pop('continents_color', '0.33')
        plt.fill(lonlon, latlat,
            color=continents_color, edgecolor='none',
            **kw)
    else:
        # plot coastlines
        coastlines_color = kw.pop('coastlines_color', '0.33')
        coastlines_width = kw.pop('linewidth', 1)
        coastlines_width = kw.pop('coastlines_width', coastlines_width)
        plt.plot(lonlon, latlat,
            color=coastlines_color, linewidth=coastlines_width,
            **kw)

    try:
        xticks2lon(xticks)
    except:
        pass#xticks2lon fails when some ticks have no ticklabels
    try:
        yticks2lat(yticks)
    except:
        pass#xticks2lon fails when some ticks have no ticklabels
    plt.xlim(min(lon_edge), max(lon_edge))
    plt.ylim(min(lat_edge), max(lat_edge))

    return
