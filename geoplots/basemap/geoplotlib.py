"""
@author: Wenchang Yang (yang.wenchang@uci.edu)
"""
# from .mypyplot import vcolorbar, hcolorbar

# needed for recent versions of basemap
import os, sys
if 'PROJ_LIB' not in os.environ: #and sys.version < '3.9':
    os.environ['PROJ_LIB'] = '/'.join ( os.__file__.split('/')[:-3] + ['share', 'proj'])
    print('[added]: PROJ_LIB =', os.environ['PROJ_LIB'])
import numpy as np
from numpy import ma
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.basemap import addcyclic, shiftgrid
# from copy import deepcopy
try:
    import xarray as xr
except ImportError:
    pass

# ###### universal 2-D plot function on the lon-lat plane
def geoplot(data=None, lon=None, lat=None, **kw):
    '''Show 2D data in a lon-lat plane.

    Parameters
    -----------
        data: array (ndarray or DataArray) of shape (n_lat, n_lon),
            or [u_array, v_array]-like for (u,v) data
            or None(default) when only plotting the basemap.
        lon: n_lon length vector or None(default).
        lat: n_lat length vector or None(default).
        kw: dict parameters related to basemap or plot functions.

    Basemap related parameters:
    ----------------------------
        basemap_kw: dict parameter in the initialization of a Basemap.
        proj or projection: map projection name (default is 'hammer')
            popular projections: 'cyl', 'ortho', 'hammer', 'lcc', 'laea'
                'np'(='nplaea'), 'sp'(='splaea')
                and other projections given from basemap.
        lon_0: map center longitude (0 as default).
        lat_0: map center latitude (0 as default).
        lonlatcorner: (llcrnrlon, urcrnrlon, llcrnrlat, urcrnrlat).
        boundinglat: latitude at the map out boundary (None as default).
        basemap_round or round: True(default) or False.
        basemap_width: width of basemap in projections like 'lcc' or 'laea'
        basemap_height: height of basemap in projections like 'lcc' or 'laea'

        land_on: bool value (False as default).
        land_kw: dict parameter used in the Basemap.fillcontinents method.
        land_color: color of land ('0.33' as default).
        lake_color: color of lakes ('none' as default).
        ocean_on: bool value (False as default).
        ocean_color: color of ocean ([ 0.59375, 0.71484375, 0.8828125] as default).

        coastline_on: bool value (default True if both land and ocean are not shown)
        coastline_kw: dict parameter used in the Basemap.drawcoastlines method.
        coastline_color: color of coast lines ('0.33' as default).

        grid_on: bool value (True as default).
        grid_label_on: bool value (False as default).

        parallel_kw: dict parameters used in the Basemap.drawparallels method.
        parallels: parallels to be drawn (None as default).
        parallel_color: color of parallels ('0.5' as default).
        parallel_labels:[0,0,0,0] or [1,0,0,0].

        meridian_kw: dict parameters used in the Basemap.drawmeridians method.
        meridians: meridians to be drawn (None as default).
        meridian_color: color of meridians (parallels_color as default).
        meridian_labels: [0,0,0,0] or [0,0,0,1].

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
        quiver_scale: quiver scale.
        quiver_color: quiver color.
        quiver_kw: dict parameters used in the plt.quiver function.
        regrid_shape: int or tuple.

        qkey_on: bool value, whether to show the quiverkey plot.
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
        cbar_on: bool value, whether to show the colorbar.
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

    #### extract data into ndarray dtypes (zz, lon, lat), might be modified after lon_0 is determined (lon-shifted)
    # data
    data_is_vector_field = isinstance(data, (list, tuple)) \
        and len(data) == 2
    if data_is_vector_field: # input data are (u,v) type, in this case u = data[0], v = data[1]
        data1 = data[1] # v
        data = data[0] # u
    # zz
    try:
        zz = data.values # data is an xarray.DataArray
    except:
        zz = data

    # uu, vv
    if data_is_vector_field: # input data are (u,v) type
        try: # data1 is DataArray
            vv = data1.values
        except: # data1 is ndarray
            vv = data1
        uu = zz
        zz = np.sqrt(uu**2 + vv**2) # magnitude of the vector field

    # lon
    if lon is None:
        try: # data is DataArray
            lon = [data[dim].values for dim in data.dims
                if dim in ['lon', 'longitude', 'X', 'Lon', 'Longitude']][0]
        except:
            try: # data and zz are ndarray
                nx = zz.shape[-1]
            except: # data and zz are None
                nx = 180
            lon = np.linspace(0, 360, nx+1)[:-1]
    # lat
    if lat is None:
        try: # data is DataArray
            lat = [data[dim].values for dim in data.dims
                if dim in ['lat', 'latitude', 'Y']][0]
        except:
            try: # data and zz are ndarray
                ny = zz.shape[0]
            except: # data and zz are None
                ny = 90
            lat_ = np.linspace(-90, 90, ny+1) # lat grid edges
            lat = (lat_[0:-1] + lat_[1:])/2
    if lat[-1] < lat[0]: #lat is in descending order
        lat = lat[-1::-1]
        zz = zz[-1::-1, :]
        if data_is_vector_field:
            uu = uu[-1::-1, :]
            vv = vv[-1::-1, :]

    #### target axis
    ax = kw.pop('ax', None)
    if ax is None:
        ax = plt.gca()
    else:
        plt.sca(ax)

    # #### basemap parameters
    basemap_kw = kw.pop('basemap_kw', {})
    # projection
    proj = kw.pop('proj', 'hammer')
    proj = kw.pop('projection', proj) # projection overrides the proj parameter
    proj = basemap_kw.pop('projection', proj)
    # short names for nplaea and splaea projections
    if proj in ('npolar', 'polar', 'np'):
        proj = 'nplaea'
    elif proj in ('spolar', 'sp'):
        proj = 'splaea'

    # lon_0: might shift the original lon and zz
    if proj in ('lcc', 'laea'):
        lon_0 = kw.pop('lon_0', -100) # center longitude prescribed
    else:
        if np.isclose( np.abs(lon[-1]-lon[0]+lon[-1]-lon[-2]), 360 ):
            lon_0 = kw.pop('lon_0', 0) # center longitude prescribed
        else:
            lon_0 = kw.pop('lon_0', (lon[0]+lon[-1])/2)
    if proj in ('moll', 'cyl', 'hammer', 'robin', 'kav7') \
        and np.isclose( np.abs(lon[-1]-lon[0]+lon[-1]-lon[-2]), 360 ):
        # modify lon_0
        lon_0 = basemap_kw.pop('lon_0', lon_0)
        lon_0_data = (lon[0] + lon[-1])/2.0 # center longitude of data coverage
        dlon = lon[1] - lon[0] # lon grid size of data
        d_lon_0 = lon_0 - lon_0_data # distance between specified center lon and data center lon
        lon_0 = float(int(d_lon_0/dlon)) * dlon + lon_0_data # modified version of the specified lon_0

        # shift grid: only needed in some projections
        lon_west_end = lon_0 - 180 + (lon[1] - lon[0])/2.0
        # make sure the longitude of west end within the lon
        if lon_west_end < lon.min():
            lon_west_end += 360
        elif lon_west_end > lon.max():
            lon_west_end -= 360
        try:
            zz, lon_ = shiftgrid(lon_west_end, zz, lon, start=True)
        except: # zz is None
            dummy, lon_ = shiftgrid(lon_west_end, lon, lon, start=True)
        if data_is_vector_field:
            uu, lon_ = shiftgrid(lon_west_end, uu, lon, start=True)
            vv, lon_ = shiftgrid(lon_west_end, vv, lon, start=True)
        lon = lon_
        # make sure lon_0 within the shifted lon
        if lon.min() > lon_0:
            lon -= 360
        elif lon.max() < lon_0:
            lon += 360

    # lat_0
    if proj in ('lcc', 'laea'):
        lat_0 = kw.pop('lat_0', 50)
    else:
        lat_0 = kw.pop('lat_0', 0)
    lat_0 = basemap_kw.pop('lat_0', lat_0)

    # lonlatcorner = (llcrnrlon, urcrnrlon, llcrnrlat, urcrnrlat)
    lonlatcorner = kw.pop('lonlatcorner', None)
    if lonlatcorner is not None:
        llcrnrlon, urcrnrlon, llcrnrlat, urcrnrlat = lonlatcorner
    else:
        llcrnrlon, urcrnrlon, llcrnrlat, urcrnrlat = (None,) * 4
    llcrnrlon = basemap_kw.pop('llcrnrlon', llcrnrlon)
    urcrnrlon = basemap_kw.pop('urcrnrlon', urcrnrlon)
    llcrnrlat = basemap_kw.pop('llcrnrlat', llcrnrlat)
    urcrnrlat = basemap_kw.pop('urcrnrlat', urcrnrlat)

    # boundinglat
    boundinglat = kw.pop('boundinglat', None)
    boundinglat = basemap_kw.pop('boundinglat', boundinglat)
    if boundinglat is None:
        if proj in ('npstere', 'nplaea', 'npaeqd'):
            boundinglat = 30
        elif proj in ('spstere', 'splaea', 'spaeqd'):
            boundinglat = -30

    # basemap round: True or False
    if proj in ('npstere', 'nplaea', 'npaeqd', 'spstere', 'splaea', 'spaeqd'):
        basemap_round = kw.pop('basemap_round', True)
    else:
        basemap_round = kw.pop('basemap_round', False)
    basemap_round = basemap_kw.pop('round', basemap_round)

    # basemap width and height
    if proj in ('lcc', 'laea'):
        basemap_width = kw.pop('basemap_width', 120e5)
        basemap_height = kw.pop('basemap_height', 90e5)
    else:
        basemap_width = kw.pop('basemap_width', None)
        basemap_height = kw.pop('basemap_height', None)
    basemap_width = basemap_kw.pop('width', basemap_width)
    basemap_height = basemap_kw.pop('height', basemap_height)

    # base map
    m = Basemap(projection=proj, lon_0=lon_0, lat_0=lat_0, boundinglat=boundinglat,
        round=basemap_round,
        llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon,
        llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,
        width=basemap_width, height=basemap_height,
        **basemap_kw)
    # test
    '''
    print('projection', proj)
    print('lon_0', lon_0)
    print('lat_0', lat_0)
    print('round', basemap_round)
    print('llcrnrlon', llcrnrlon)
    print('urcrnrlon', urcrnrlon)
    print('llcrnrlat', llcrnrlat)
    print('urcrnrlat', urcrnrlat)
    print('width', basemap_width)
    print('height', basemap_height)
    '''

    # show continents or plot coast lines
    # ocean
    ocean_on = kw.pop('ocean_on', False)
    default_ocean_color = [ 0.59375, 0.71484375, 0.8828125]
    ocean_color = kw.pop('ocean_color', default_ocean_color)
    if ocean_on:
        m.drawmapboundary(fill_color=ocean_color)

    # land
    land_on = kw.pop('land_on', False)
    land_kw = kw.pop('land_kw', {})
    land_color = kw.pop('land_color', '0.33')
    land_color = land_kw.pop('color', land_color)
    lake_color = kw.pop('lake_color', 'none')
    lake_color = land_kw.pop('lake_color', lake_color)
    if land_on :
        # use Basemap.fillcontinents method
        m.fillcontinents(color=land_color, lake_color=lake_color,
            **land_kw)

    # coastlines
    coastline_on = kw.pop('coastline_on', not land_on)
    if coastline_on:
        coastline_kw = kw.pop('coastline_kw', {})
        # use Basemap.drawcoastlines method
        coastline_color = kw.pop('coastline_color', '0.33')
        coastline_color = coastline_kw.pop('color', coastline_color)
        coastline_lw = kw.pop('coastline_lw', 0.5)
        coastline_lw = coastline_kw.pop('linewidth', coastline_lw)
        m.drawcoastlines(color=coastline_color, linewidth=coastline_lw,
            **coastline_kw)

    # parallels
    grid_on = kw.pop('grid_on', True)
    parallel_kw = kw.pop('parallel_kw', {})
    grid_label_on = kw.pop('grid_label_on', False)
    parallels = kw.pop('parallels', None)
    parallel_color = kw.pop('parallel_color', '0.5')
    parallel_color = parallel_kw.pop('color', parallel_color)
    parallel_lw = kw.pop('parallel_lw', 0.5)
    parallel_lw = parallel_kw.pop('linewidth', parallel_lw)
    parallel_labels = kw.pop('parallel_labels', None)
    parallel_labels = parallel_kw.pop('labels', parallel_labels)
    if parallel_labels is None:
        if grid_label_on:
            parallel_labels = [1, 0, 0, 0]
        else:
            parallel_labels = [0, 0, 0, 0]
    if parallels is not None:
        m.drawparallels(parallels, color=parallel_color,
            labels=parallel_labels,
            linewidth=parallel_lw, **parallel_kw)
    elif grid_on:
        m.drawparallels(np.arange(-90, 91, 30), color=parallel_color,
            linewidth=parallel_lw, labels=parallel_labels,
            **parallel_kw)

    # meridians
    meridians = kw.pop('meridians', None)
    meridian_kw = kw.pop('meridian_kw', {})
    meridian_color = kw.pop('meridian_color', parallel_color)
    meridian_color = meridian_kw.pop('color', meridian_color)
    meridian_lw = kw.pop('meridian_lw', parallel_lw)
    meridian_lw = meridian_kw.pop('linewidth',meridian_lw)
    meridian_labels = kw.pop('meridian_labels', None)
    meridian_labels = meridian_kw.pop('labels', meridian_labels)
    if meridian_labels is None:
        if grid_label_on:
            if proj in ('npstere', 'nplaea', 'npaeqd',
                'spstere', 'splaea', 'spaeqd'):
                meridian_labels = [1, 1, 0, 0]
            elif proj in ('hammer', 'moll'):
                meridian_labels = [0, 0, 0, 0]
            else:
                meridian_labels = [0, 0, 0, 1]
        else:
            meridian_labels = [0, 0, 0, 0]
    if meridians is not None:
        m.drawmeridians(meridians, color=meridian_color,
            label=meridian_labels,
            linewidth=meridian_lw, **meridian_kw)
    elif grid_on:
        m.drawmeridians(np.arange(0, 360, 30), color=meridian_color,
            labels=meridian_labels,
            linewidth=meridian_lw,
            **meridian_kw)

    # lonlatbox
    lonlatbox = kw.pop('lonlatbox', None)
    if lonlatbox is not None:
        lonlon = np.array([
            np.linspace(lonlatbox[0], lonlatbox[1], 100),
            lonlatbox[1]*np.ones(100),
            np.linspace(lonlatbox[1], lonlatbox[0], 100),
            lonlatbox[0]*np.ones(100)
            ]).ravel()
        latlat = np.array([
            lonlatbox[2]*np.ones(100),
            np.linspace(lonlatbox[2], lonlatbox[3], 100),
            lonlatbox[3]*np.ones(100),
            np.linspace(lonlatbox[3], lonlatbox[2], 100)
            ]).ravel()
        lonlatbox_kw = kw.pop('lonlatbox_kw', {})
        lonlatbox_color = kw.pop('lonlatbox_color', 'k')
        lonlatbox_color = lonlatbox_kw.pop('color', lonlatbox_color)
        m.plot(lonlon, latlat, latlon=True, color=lonlatbox_color, **lonlatbox_kw)

    # #### stop here and return the map object if data is None
    if data is None:
        return m


    # ###### plot parameters
    # plot_type
    plot_type = kw.pop('plot_type', None)
    if plot_type is None:
        if data_is_vector_field:
            plot_type = 'quiver'
        elif proj in ('nplaea', 'splaea', 'ortho'):
            # pcolormesh has a problem for these projections
            plot_type = 'pcolor'
        else:
            plot_type = 'pcolormesh'
    print ('plot_type **** {} **** is used.'.format(plot_type))

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
        if isinstance(zz,np.ma.core.MaskedArray):
            zz1d = zz.compressed()
        else:
            zz1d = zz.ravel()
        notNaNs = np.logical_not(np.isnan(zz1d))
        zz1d = zz1d[notNaNs]
        if robust:
            a = np.percentile(zz1d,2)
            b = np.percentile(zz1d,98)
        else:
            a = zz1d.min()
            b = zz1d.max()
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
    cbar_kw = kw.pop('cbar_kw', {})
    cbar_location = kw.pop('cbar_location', 'right')
    if cbar_location in ('right', 'left'):
        cbar_size = kw.pop('cbar_size', '2.5%')
        cbar_pad = kw.pop('cbar_pad', '2.5%')
    elif cbar_location in ('bottom', 'top'):
        cbar_size = kw.pop('cbar_size', '5%')
        cbar_pad = kw.pop('cbar_pad', '5%')

    # units in colorbar
    units = kw.pop('units', None)
    if units is None:
        try:
            units = data.attrs['units'] # input data is a DataArray
        except:
            units = ''
    # long_name in colorbar
    long_name = kw.pop('long_name', None)
    if long_name is None:
        try:
            long_name = data['long_name'] # if input data is a DataArray
            if long_name is None:
                long_name = ''
        except:
            long_name = ''

    # lon and lat edges
    lon_extend = np.hstack((2*lon[0] - lon[1], lon, 2*lon[-1] - lon[-2])) # size=nx+2
    lon_edge = (lon_extend[:-1] + lon_extend[1:])/2.0 # size=nx+1
    lat_extend = np.hstack((2*lat[0] - lat[1], lat, 2*lat[-1] - lat[-2])) # size=nx+2
    lat_edge = (lat_extend[:-1] + lat_extend[1:])/2.0 # size=nx+1


    # add cyclic point in some projections and plot types
    if plot_type in ('contourf', 'contour', 'contourf+') \
        and proj in ('ortho','npstere', 'nplaea', 'npaeqd', 'spstere',
        'splaea', 'spaeqd')\
        and np.isclose(np.abs(lon_edge[-1]-lon_edge[0]), 360):
        zz, lon = addcyclic(zz, lon)

    # final data preparation before plotting
    Lon, Lat = np.meshgrid(lon, lat)
    X, Y = m(Lon, Lat)
    Lon_edge, Lat_edge = np.meshgrid(lon_edge, lat_edge)
    X_edge, Y_edge = m(Lon_edge, Lat_edge)
    if np.any(np.isnan(zz)):
        zz = ma.masked_invalid(zz) # mask invalid values

    # ###### plot
    # pcolor
    if plot_type in ('pcolor',):
        rasterized = kw.pop('rasterized', True)
        plot_obj = m.pcolor(X_edge, Y_edge, zz, cmap=cmap,
            rasterized=rasterized, **kw)

    # pcolormesh
    elif plot_type in ('pcolormesh',):
        rasterized = kw.pop('rasterized', True)
        plot_obj = m.pcolormesh(X_edge, Y_edge, zz, cmap=cmap,
            rasterized=rasterized, **kw)

    # imshow
    elif plot_type in ('imshow',):
        if lat_edge[-1] > lat_edge[0]:
            origin = kw.pop('origin', 'lower')
        else:
            origin = kw.pop('origin', 'upper')
        extent = kw.pop('extent', [lon_edge[0], lon_edge[-1], lat_edge[0], lat_edge[-1]])
        interpolation = kw.pop('interpolation', 'nearest')
        plot_obj = m.imshow(zz, origin=origin, cmap=cmap, extent=extent,
            interpolation=interpolation, **kw)

    # contourf, contour and contourf+
    # contourf
    elif plot_type in ('contourf',):
        extend = kw.pop('extend', 'both')
        plot_obj = m.contourf(X, Y, zz, extend=extend, cmap=cmap,
            levels=levels, **kw)

    # contour
    elif plot_type in ('contour',):
        colors = kw.pop('colors', 'k')
        if colors is not None:
            cmap = None
        alpha = kw.pop('alpha', 0.5)
        plot_obj = m.contour(X, Y, zz, cmap=cmap, colors=colors,
            levels=levels, alpha=alpha, **kw)
        contour_label_on = kw.pop('contour_label_on', False)
        if contour_label_on:
            plt.clabel(plot_obj,plot_obj.levels[::2],fmt='%.2G')

    # contourf + contour
    elif plot_type in ('contourf+',):
        extend = kw.pop('extend', 'both')
        linewidths = kw.pop('linewidths', 1)
        plot_obj = m.contourf(X, Y, zz, extend=extend, cmap=cmap,
            levels=levels, **kw)
        colors = kw.pop('colors', 'k')
        if colors is not None:
            cmap = None
        alpha = kw.pop('alpha', 0.5)
        m.contour(X, Y, zz, cmap=cmap, colors=colors, alpha=alpha,
            levels=levels, linewidths=linewidths, **kw)

    # quiverplot
    elif plot_type in ('quiver',):
        regrid_shape = kw.pop('regrid_shape', (40, 40))
        if isinstance(regrid_shape, int):
            regrid_shape = (regrid_shape,) * 2
        nx, ny = regrid_shape
        # need to shift grid if lon is out of the [-180, 180] range
        if lon.max() > 180 or lon.min() < -180:
            uu, lon_ = shiftgrid(180.,uu,lon,start=False)
            vv, lon_ = shiftgrid(180.,vv,lon,start=False)
            lon = lon_
        uu, vv, X, Y = m.transform_vector(
            uu, vv, lon, lat, nx, ny, returnxy=True
        )
        magnitude_on = kw.pop('magnitude_on', False)
        quiver_color = kw.pop('quiver_color', 'g')
        quiver_scale = kw.pop('quiver_scale', None)

        # quiverkey params
        qkey_kw = kw.pop('qkey_kw', {})
        qkey_on = kw.pop('qkey_on', True)
        qkey_X = kw.pop('qkey_X', 0.9)
        qkey_X = qkey_kw.pop('X', qkey_X)
        qkey_Y = kw.pop('qkey_Y', 1.03)
        qkey_Y = qkey_kw.pop('Y', qkey_Y)
        qkey_U = kw.pop('qkey_U', np.nanmax(zz))
        qkey_U = qkey_kw.pop('U', qkey_U)
        qkey_label = kw.pop('qkey_label', '{:.2g} '.format(qkey_U) + units)
        qkey_label = qkey_kw.pop('label', qkey_label)
        qkey_labelpos = kw.pop('qkey_labelpos', 'W')
        qkey_labelpos = qkey_kw.pop('labelpos', qkey_labelpos)

        if magnitude_on:
            magnitude = np.sqrt(uu**2 + vv**2)
            plot_obj = m.quiver(X, Y, uu, vv, magnitude, color=quiver_color,
                scale=quiver_scale, **kw)
        else:
            plot_obj = m.quiver(X, Y, uu, vv, color=quiver_color,
                scale=quiver_scale, **kw)

        if qkey_on:
            # quiverkey plot
            plt.quiverkey(plot_obj, qkey_X, qkey_Y, qkey_U,
                label=qkey_label, labelpos=qkey_labelpos, **qkey_kw)


    # hatch plot
    elif plot_type in ('hatch', 'hatches'):
        # hatches = kw.pop('hatches', ['///'])
        hatches = kw.pop('hatches', ['...'])
        plot_obj = m.contourf(X, Y, zz, colors='none', hatches=hatches,
            extend='both', **kw)
    else:
        print('Please choose a right plot_type from ("pcolor", "contourf", "contour")!')

    # set clim
    if plot_type in ('pcolor', 'pcolormesh', 'imshow'):
        plt.clim(clim)

    # plot colorbar
    if cbar_on is None:
        if plot_type in ('pcolor', 'pcolormesh', 'contourf', 'contourf+',
            'imshow'):
            cbar_on = True
        elif (plot_type in ('quiver',) and magnitude_on):
            cbar_on = True
        else:
            cbar_on = False
    if cbar_on:
        cbar = m.colorbar(plot_obj, location=cbar_location,
            size=cbar_size, pad=cbar_pad, ax=ax, **cbar_kw)
        if cbar_location in ('right',):
            cbar.ax.xaxis.set_label_position('top')
            cbar.ax.set_xlabel(units)
            cbar.ax.set_ylabel(long_name)
        elif cbar_location in ('bottom',):
            if long_name == '' or units =='':
                cbar.ax.set_xlabel('{}{}'.format(long_name, units))
            else:
                cbar.ax.set_xlabel('{} [{}]'.format(long_name, units))

    return plot_obj
