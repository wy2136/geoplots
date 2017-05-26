"""
@author: Wenchang Yang (yang.wenchang@uci.edu)
"""
# from .mypyplot import vcolorbar, hcolorbar

import numpy as np
from numpy import ma
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .util import (xticks2lat, xticks2lon, xticks2month, xticks2dayofyear,
    yticks2p, yticks2lat, yticks2lon, yticks2month, yticks2dayofyear)

# from copy import deepcopy
try:
    import xarray as xr
except ImportError:
    pass


# ###### universal 2-D plot function
def fxyplot(data=None, x=None, y=None, **kw):
    '''Show 2D data.

    Parameters
    -----------
        data: array of shape (n_lat, n_lon), or [u_array, v_array]-like for (u,v)
            data or None(default) when only plotting the basemap.
        lon: n_lon length vector or None(default).
        lat: n_lat length vector or None(default).
        kw: dict parameters related to basemap or plot functions.

    Basemap related parameters:
    ----------------------------
        basemap_kw: dict parameter in the initialization of a Basemap.
        proj or projection: map projection name (default='moll')
            popular projections: 'ortho', 'np'(='nplaea'), 'sp'(='splaea')
                and other projections given from basemap.
        lon_0: map center longitude (None as default).
        lat_0: map center latitude (None as default).
        lonlatcorner: (llcrnrlon, urcrnrlon, llcrnrlat, urcrnrlat).
        boundinglat: latitude at the map out boundary (None as default).
        basemap_round or round: True(default) or False.

        fill_continents: bool value (False as default).
        continents_kw: dict parameter used in the Basemap.fillcontinents method.
        continents_color: color of continents ('0.5' as default).
        lake_color: color of lakes ('0.5' as default).

        coastlines_kw: dict parameter used in the Basemap.drawcoastlines method.
        coastlines_color: color of coast lines ('0.66' as default).

        parallels_kw: dict parameters used in the Basemap.drawparallels method.
        parallels: parallels to be drawn (None as default).
        parallels_color: color of parallels ('0.75' as default).
        parallels_labels:[0,0,0,0] or [1,0,0,0].

        meridians_kw: dict parameters used in the Basemap.drawmeridians method.
        meridians: meridians to be drawn (None as default).
        meridians_color: color of meridians ('0.75' as default).
        meridians_labels: [0,0,0,0] or [0,0,0,1].

        lonlatbox: None or (lon_start, lon_end, lat_start, lat_end).
        lonlatbox_kw: dict parameters in the plot of lon-lat box.
        lonlatbox_color:

    General plot parameters:
    -------------------------
        ax: axis object, default is plt.gca()
        plot_type: a string of plot type from ('pcolor', 'pcolormesh', 'imshow', 'contourf', 'contour', 'quiver', 'scatter') or None(default).
        cmap: pyplot colormap.
        clim: a tuple of colormap limit.
        levels: sequence, int or None (default=None)
        plot_kw: dict parameters used in the plot functions.

    Pcolor/Pcolormesh related parameters:
    -------------------------------
        rasterized: bool (default is True).

    Imshow related parameters
    ---------------------------
        origin: 'lower' or 'upper'.
        extent: horizontal range.
        interpolation: 'nearest' (default) or 'bilinear' or 'cubic'.

    Contourf related parameters:
    -------------------------------
        extend: 'both'(default).

    Contour related parameters:
    ----------------------------
        label_contour: False(default) or True.
            Whether to label contours or not.
        colors: contour color (default is 'gray').

    Quiver plot related parameters:
    --------------------------------
        stride: stride along lon and lat.
        stride_lon: stride along lon.
        stride_lat: stride along lat.
        quiver_scale: quiver scale.
        quiver_color: quiver color.
        quiver_kw: dict parameters used in the plt.quiver function.

        hide_qkey: bool value, whether to show the quiverkey plot.
        qkey_X: X parameter in the plt.quiverkey function (default is 0.85).
        qkey_Y: Y parameter in the plt.quiverkey function (default is 1.02).
        qkey_U: U parameter in the plt.quiverkey function (default is 2).
        qkey_label: label parameter in the plt.quiverkey function.
        qkey_labelpos: labelpos parameter in the plt.quiverkey function.
        qkey_kw: dict parameters used in the plt.quiverkey function.

    Scatter related parameters:
    ------------------------------
        scatter_data: None(default) or (lonvec, latvec).

    Hatch plot related parameters:
    ----------------------------------
        hatches: ['///'] is default.

    Colorbar related parameters:
    -------------------------------
        hide_cbar: bool value, whether to show the colorbar.
        cbar_type: 'vertical'(shorten as 'v') or 'horizontal' (shorten as 'h').
        cbar_extend: extend parameter in the plt.colorbar function.
            'neither' as default here.
        cbar_size: default '2.5%' for vertical colorbar,
            '5%' for horizontal colorbar.
        cbar_pad: default 0.1 for vertical colorbar,
            0.4 for horizontal colorbar.
        cbar_kw: dict parameters used in the plt.colorbar function.
        units: str
        long_name: str

    Returns
    --------
        basemap object if only basemap is plotted.
        plot object if data is shown.
        '''

    # target axis
    ax = kw.pop('ax', None)
    if ax is not None:
        plt.sca(ax)

    if isinstance(data, xr.DataArray):
        data_array = data.copy()
        data = data_array.values
        if np.any(np.isnan(data)):
            data = ma.masked_invalid(data)
        if x is None:
            xname = data_array.dims[1]
            x = data_array[xname].values
        if y is None:
            yname = data_array.dims[0]
            y = data_array[yname].values
        # guess data name
        try:
            data_name = data_array.attrs['long_name']
        except KeyError:
            try:
                data_name = data_array.name
                if data_name is None:
                    data_name = ''
            except AttributeError:
                data_name = ''
        # guess data units
        try:
            data_units = data_array.attrs['units']
        except KeyError:
            data_units = ''
    elif data is not None: # try to copy the input data
        try:
            data = data.copy()
        except:
            try:
                data = ( data[0].copy(), data[1].copy() )
            except:
                pass
    try:
        x = x.copy()
    except:
        pass
    try:
        y = y.copy()
    except:
        pass

    # data prepare
    input_data_have_two_components = isinstance(data, tuple) \
        or isinstance(data, list)
    if input_data_have_two_components:
        # input data is (u,v) or [u, v] where u, v are ndarray and two components of a vector
        assert len(data) == 2,'quiver data must contain only two componets u and v'
        u = data[0].squeeze()
        v = data[1].squeeze()
        assert u.ndim == 2, 'u component data must be two dimensional'
        assert v.ndim == 2, 'v component data must be two dimensional'
        data = np.sqrt( u**2 + v**2 ) # calculate wind speed
    else:# input data is a ndarray
        data = data.squeeze()
        assert data.ndim == 2, 'Input data must be two dimensional!'
    Ny, Nx = data.shape

    # x
    if x is None:
        x = np.arange(Nx)
        x_edge = np.arange(-0.5, Nx+0.5)
    else:
        x_edge = np.zeros((Nx+1,))
        x_edge[1:Nx] = ( x[0:Nx-1] + x[1:Nx] )/2.0
        x_edge[0] = x[0] - (x[1] - x[0])/2.0
        x_edge[Nx] = x[Nx-1] + (x[Nx-1] - x[Nx-2])/2.0

    # y
    if y is None:
        y = np.arange(Ny)
        y_edge = np.arange(-0.5, Ny+0.5)
    else:
        y_edge = np.zeros((Ny+1,))
        y_edge[1:Ny] = ( y[0:Ny-1] + y[1:Ny] )/2.0
        y_edge[0] = y[0] - (y[1] - y[0])/2.0
        y_edge[Ny] = y[Ny-1] + (y[Ny-1] - y[Ny-2])/2.0


    # ###### plot parameters
    # plot_type
    plot_type = kw.pop('plot_type', None)
    if plot_type is None:
        if input_data_have_two_components:
            plot_type = 'quiver'
        else:
            plot_type = 'pcolormesh'
            print ('plot_type **** pcolormesh **** is used.')

    # cmap
    cmap = kw.pop('cmap', None)
    if cmap is None:
         zz_max = data.max()
         zz_min = data.min()
         if zz_min >=0:
             try:
                 cmap = plt.get_cmap('viridis')
             except:
                 cmap = plt.get_cmap('OrRd')
         elif zz_max<=0:
             try:
                 cmap = plt.get_cmap('viridis')
             except:
                 cmap = plt.get_cmap('Blues_r')
         else:
             cmap = plt.get_cmap('RdBu_r')
    elif isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)

    # clim parameters
    clim = kw.pop('clim', None)
    robust = kw.pop('robust', False)
    if clim is None:
        if isinstance(data,np.ma.core.MaskedArray):
            data1d = data.compressed()
        else:
            data1d = data.ravel()
        notNaNs = np.logical_not(np.isnan(data1d))
        data1d = data1d[notNaNs]
        if robust:
            a = np.percentile(data1d,2)
            b = np.percentile(data1d,98)
        else:
            a = data1d.min()
            b = data1d.max()
        if a * b < 0:
            b = max(abs(a), abs(b))
            a = -b
        clim = a, b

    # levels
    levels = kw.pop('levels', None)
    if levels is None:
        if plot_type in ('contour', 'contourf', 'contourf+'):
            a, b = clim
            levels = np.linspace(a, b, 11)
    elif isinstance(levels, int):
        if plot_type in ('contour', 'contourf', 'contourf+'):
            a, b = clim
            levels = np.linspace(a, b, levels)
        elif plot_type in ('pcolor', 'pcolormesh', 'imshow'):
            cmap = plt.get_cmap(cmap.name, levels-1)
    else: # levels is a sequence
        if plot_type in ('pcolor', 'pcolormesh', 'imshow'):
            cmap = plt.get_cmap(cmap.name, len(levels)-1)
            clim = min(levels), max(levels)


    # colorbar parameters
    cbar_on = kw.pop('cbar_on', None)
    cbar_type = kw.pop('cbar_type', 'vertical')
    cbar_kw = kw.pop('cbar_kw', {})
    cbar_extend = kw.pop('cbar_extend', 'neither')
    cbar_extend = cbar_kw.pop('extend', cbar_extend)
    if cbar_type in ('v', 'vertical'):
        cbar_size = kw.pop('cbar_size', '2.5%')
        cbar_size = cbar_kw.pop('size', cbar_size)
        cbar_pad = kw.pop('cbar_pad', 0.1)
        cbar_pad = cbar_kw.pop('pad', cbar_pad)
        cbar_position = 'right'
        cbar_orientation = 'vertical'
    elif cbar_type in ('h', 'horizontal'):
        # cbar = hcolorbar(units=units)
        cbar_size = kw.pop('cbar_size', '5%')
        cbar_size = cbar_kw.pop('size', cbar_size)
        cbar_pad = kw.pop('cbar_pad', 0.4)
        cbar_pad = cbar_kw.pop('pad', cbar_pad)
        cbar_position = 'bottom'
        cbar_orientation = 'horizontal'
    # units in colorbar
    units = kw.pop('units', None)
    if units is None:
        try:
            units = data_units # input data is a DataArray
        except:
            units = ''
    # long_name in colorbar
    long_name = kw.pop('long_name', None)
    if long_name is None:
        try:
            long_name = data_name # if input data is a DataArray
        except:
            long_name = ''

    # xtype
    xtype = kw.pop('xtype', None)
    # xticks
    xticks = kw.pop('xticks', None)

    # ytype
    ytype = kw.pop('ytype', None)
    # yticks
    yticks = kw.pop('yticks', None)


    # ###### plot
    # pcolor
    if plot_type in ('pcolor',):
        rasterized = kw.pop('rasterized', True)
        plot_obj = plt.pcolor(x_edge, y_edge, data, cmap=cmap,
            rasterized=rasterized, **kw)

    # pcolormesh
    elif plot_type in ('pcolormesh',):
        rasterized = kw.pop('rasterized', True)
        plot_obj = plt.pcolormesh(x_edge, y_edge, data, cmap=cmap,
            rasterized=rasterized, **kw)

    # imshow
    elif plot_type in ('imshow',):
        if y_edge[-1] > y_edge[0]:
            origin = kw.pop('origin', 'lower')
        else:
            origin = kw.pop('origin', 'upper')
        extent = kw.pop('extent', [x_edge[0], x_edge[-1], y_edge[0], y_edge[-1]])
        interpolation = kw.pop('interpolation', 'nearest')
        plot_obj = plt.imshow(data, origin=origin, cmap=cmap, extent=extent,
            interpolation=interpolation, **kw)

    # contourf
    elif plot_type in ('contourf',):
        extend = kw.pop('extend', 'both')
        plot_obj = plt.contourf(x, y, data, extend=extend, cmap=cmap,
            levels=levels, **kw)

    # contour
    elif plot_type in ('contour',):
        colors = kw.pop('colors', 'k')
        if colors is not None:
            cmap = None
        alpha = kw.pop('alpha', 0.5)
        plot_obj = plt.contour(x, y, data, cmap=cmap, colors=colors,
            levels=levels, alpha=alpha, **kw)
        label_contour = kw.pop('label_contour', False)
        if label_contour:
            plt.clabel(plot_obj,plot_obj.levels[::2],fmt='%.2G')

    # contourf + contour
    elif plot_type in ('contourf+',):
        extend = kw.pop('extend', 'both')
        plot_obj = plt.contourf(x, y, data, extend=extend, cmap=cmap,
            levels=levels, **kw)
        colors = kw.pop('colors', 'k')
        if colors is not None:
            cmap = None
        alpha = kw.pop('alpha', 0.5)
        plt.contour(x, y, data, cmap=cmap, colors=colors, alpha=alpha,
            levels=levels, **kw)

    # quiverplot
    elif plot_type in ('quiver',):
        stride = kw.pop('stride', 1)
        stride_x = kw.pop('stride_x', stride)
        stride_y = kw.pop('stride_y', stride)
        x_ = x[::stride_x] # subset of lon
        y_ = y[::stride_y]
        u_ = u[::stride_y, ::stride_x]
        v_ = v[::stride_y, ::stride_x]
        quiver_color = kw.pop('quiver_color', 'g')
        quiver_scale = kw.pop('quiver_scale', None)
        hide_qkey = kw.pop('hide_qkey', False)
        qkey_kw = kw.pop('qkey_kw', {})
        qkey_X = kw.pop('qkey_X', 0.85)
        qkey_X = qkey_kw.pop('X', qkey_X)
        qkey_Y = kw.pop('qkey_Y', 1.02)
        qkey_Y = qkey_kw.pop('Y', qkey_Y)
        qkey_U = kw.pop('qkey_U', 2)
        qkey_U = qkey_kw.pop('U', qkey_U)
        qkey_label = kw.pop('qkey_label', '{:g} '.format(qkey_U) + units)
        qkey_label = qkey_kw.pop('label', qkey_label)
        qkey_labelpos = kw.pop('qkey_labelpos', 'W')
        qkey_labelpos = qkey_kw.pop('labelpos', qkey_labelpos)
        plot_obj = plt.quiver(x_, y_, u_, v_, color=quiver_color,
            scale=quiver_scale, **kw)
        if not hide_qkey:
            # quiverkey plot
            plt.quiverkey(plot_obj, qkey_X, qkey_Y, qkey_U,
                label=qkey_label, labelpos=qkey_labelpos, **qkey_kw)

    # hatch plot
    elif plot_type in ('hatch', 'hatches'):
        hatches = kw.pop('hatches', ['///'])
        plot_obj = plt.contourf(x, y, data, colors='none', hatches=hatches,
            extend='both', **kw)
    else:
        print('Please choose a right plot_type from ("pcolor", "contourf", "contour", "contourf+", "hatch")!')

    # set clim
    if plot_type in ('pcolor', 'pcolormesh', 'imshow'):
        plt.clim(clim)

    # plot colorbar
    if cbar_on is None:
        if plot_type in ('pcolor', 'pcolormesh', 'contourf', 'contourf+',
            'imshow'):
            cbar_on = True
        else:
            cbar_on = False
    if cbar_on:
        ax_current = plt.gca()
        divider = make_axes_locatable(ax_current)
        cax = divider.append_axes(cbar_position, size=cbar_size, pad=cbar_pad)
        cbar = plt.colorbar(plot_obj, cax=cax, extend=cbar_extend,
            orientation=cbar_orientation, **cbar_kw)
        if cbar_type in ('v', 'vertical'):
            # put the units on the top of the vertical colorbar
            cbar.ax.xaxis.set_label_position('top')
            cbar.ax.set_xlabel(units)
            cbar.ax.set_ylabel(long_name)
        elif cbar_type in ('h', 'horizontal'):
            # cbar.ax.yaxis.set_label_position('right')
            # cbar.ax.set_ylabel(units, rotation=0, ha='left', va='center')
            if long_name == '' or units =='':
                cbar.ax.set_xlabel('{}{}'.format(long_name, units))
            else:
                cbar.ax.set_xlabel('{} [{}]'.format(long_name, units))
        # set back the main axes as the current axes
        plt.sca(ax_current)


    # xaxis
    if xtype is None:
        pass
    elif xtype in ('lon',):
        xticks2lon(xticks=xticks)
    elif xtype in ('lat',):
        xticks2lat(xticks=xticks)
    elif xtype in ('month',):
        xticks2month()
    elif xtype in ('mon',):
        xticks2month(show_initials=True)
    elif xtype in ('dayofyear',):
        xticks2dayofyear()

    # yaxis
    if ytype is None:
        pass
    elif ytype in ('lat',):
        yticks2lat(yticks=yticks)
    elif ytype in ('p',):
        yticks2p(yticks=yticks)
    elif ytype in ('month',):
        yticks2month()
    elif ytype in ('mon',):
        yticks2month(show_initials=True)
    elif ytype in ('lon',):
        yticks2lon(yticks=yticks)
    elif ytype in ('dayofyear',):
        yticks2dayofyear()

    # xlim and ylim
    if plot_type in ('pcolormesh', 'pcolor', 'imshow'):
        xlim = x_edge[0], x_edge[-1]
        ylim = y_edge[0], y_edge[-1]
        plt.xlim(*xlim)
        plt.ylim(*ylim)


    return plot_obj
