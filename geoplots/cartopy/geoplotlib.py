"""
@author: Wenchang Yang (yang.wenchang@uci.edu)
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.util import add_cyclic_point

def geoplot(data=None, lon=None, lat=None, **kw):
    '''Show 2D data in a lon-lat plane'''

    #### projection related parameters
    proj = kw.pop('proj', 'moll')
    proj_kw = kw.pop('proj_kw', {})
    lon_0 = kw.pop('lon_0', None)
    if lon_0 is not None:
        proj_kw['central_longitude'] = lon_0
    lat_0 = kw.pop('lat_0', None)
    if lat_0 is not None:
        proj_kw['central_latitude'] = lat_0
    projections = {'ortho': ccrs.Orthographic,
        'moll': ccrs.Mollweide,
        'cyl': ccrs.PlateCarree,
        'np': ccrs.NorthPolarStereo,
        'sp': ccrs.SouthPolarStereo,
        'robin': ccrs.Robinson}
    ax = plt.axes(projection=projections[proj](**proj_kw))
    # extent
    if proj in ('np',):
        extent = kw.pop('extent', [-180+0.01, 180, 45, 90])
    elif proj in ('sp',):
        extent = kw.pop('extent', [-180+0.01, 180, -90, -45])
    else:
        extent = kw.pop('extent', None)
    if extent is None:
        ax.set_global()
    else:
        ax.set_extent(extent, crs=ccrs.PlateCarree())
    # add features, e.g. land, ocean, coastlines
    # ax.add_feature(cfeature.OCEAN, zorder=0)
    # land
    land_color = kw.pop('land_color', cfeature.COLORS['land'])
    land_on = kw.pop('land_on', False)
    if land_on:
        ax.add_feature(cfeature.LAND, zorder=0, facecolor=land_color)
    land_masked = kw.pop('land_masked', False)
    if land_masked:
        ax.add_feature(cfeature.LAND, zorder=100, facecolor=land_color)
    # ocean
    ocean_color = kw.pop('ocean_color', cfeature.COLORS['water'])
    ocean_on = kw.pop('ocean_on', False)
    if ocean_on:
        ax.add_feature(cfeature.OCEAN, zorder=0, facecolor=ocean_color)
    ocean_masked = kw.pop('ocean_masked', False)
    if ocean_masked:
        ax.add_feature(cfeature.OCEAN, zorder=100, facecolor=ocean_color)
    # coastlines
    coastline_color = kw.pop('coastline_color', 'k')
    coastline_on = kw.pop('coastline_on', None)
    if coastline_on is None:  # default value determined by other parameters
        if ocean_on or land_on or ocean_masked or land_masked:
            coastline_on = False
        else:
            coastline_on = True
    if coastline_on:
        ax.coastlines(color=coastline_color)


    if data is None:
        return ax

    #### prepare data to plot on the basemap
    # data
    data_is_vector_field = isinstance(data, (list, tuple)) and len(data) == 2
    if data_is_vector_field: # input data are (u,v) type, in this case u = data[0], v = data[1]
        data1 = data[1]
        data = data[0]
    else:
        data1 = None
    # zz
    try:
        zz = data.values # data is an xarray.DataArray
    except:
        zz = data
    # uu, vv
    if data_is_vector_field: # input data are (u,v) type
        try:
            vv = data1.values
        except:
            vv = data1
        uu = zz
        zz = np.sqrt(uu**2 + vv**2)
    # lon
    if lon is None:
        try:
            lon = [data[dim].values for dim in data.dims
                if dim in ['lon', 'longitude', 'X']][0]
        except:
            nx = zz.shape[-1]
            lon = np.linspace(0, 360, nx+1)[:-1]
    lon_extend = np.hstack((2*lon[0] - lon[1], lon, 2*lon[-1] - lon[-2])) # size=nx+2
    lon_edge = (lon_extend[:-1] + lon_extend[1:])/2.0 # size=nx+1
    # lat
    if lat is None:
        try:
            lat = [data[dim].values for dim in data.dims
                if dim in ['lat', 'latitude', 'Y']][0]
        except:
            ny = zz.shape[0]
            lat = np.linspace(-90, 90, ny+1) # lat grid edges
            lat = (lat[0:-1] + lat[1:])/2
    lat_extend = np.hstack((2*lat[0] - lat[1], lat, 2*lat[-1] - lat[-2])) # size=nx+2
    lat_edge = (lat_extend[:-1] + lat_extend[1:])/2.0 # size=nx+1


    #### plot parameters
    data_coordinate = ccrs.PlateCarree()
    if data_is_vector_field:
        plot_type = kw.pop('plot_type', 'quiver')
    else:
        plot_type = kw.pop('plot_type', 'pcolormesh')
    plot_funcs = {'pcolormesh': ax.pcolormesh,
        'pcolor': ax.pcolor,
        'contourf': ax.contourf,
        'contour': ax.contour,
        'quiver': ax.quiver,
        'streamplot': ax.streamplot}
    # colorbar
    cbar_orientation = kw.pop('cbar_orientation', 'vertical')
    cbar_extend = kw.pop('cbar_extend', 'neither')
    cbar_kw = kw.pop('cbar_kw', {})
    if cbar_orientation in ('v', 'vertical',):
        cbar_size = kw.pop('cbar_size', '2.5%')
        cbar_pad = kw.pop('cbar_pad', 0.1)
        cbar_position = 'right'
        cbar_orientation = 'vertical'
    elif cbar_orientation in ('h', 'horizontal'):
        cbar_size = kw.pop('cbar_size', '5%')
        cbar_pad = kw.pop('cbar_pad', 0.4)
        cbar_position = 'bottom'
        cbar_orientation = 'horizontal'

    #### plot
    if plot_type in ('pcolormesh', 'pcolor'):
        rasterized = kw.pop('rasterized', True)
        if lat_edge[0] < -90 or lat_edge[0] > 90:
            lat_edge = lat_edge[1:]
            zz = zz[1:, :]
        if lat_edge[-1] > 90 or lat_edge[-1] < -90:
            lat_edge = lat_edge[:-1]
            zz = zz[:-1, :]
        plot_obj = plot_funcs[plot_type](lon_edge, lat_edge, zz, 
            transform=data_coordinate, rasterized=rasterized, **kw)
    elif plot_type in ('contourf', 'contour'):
        if np.isclose(np.abs(lon_edge[-1]-lon_edge[0]), 360): # data has global coverage but miss the cyclic point
            zz, lon = add_cyclic_point(zz, coord=lon, axis=-1)
        plot_obj = plot_funcs[plot_type](lon, lat, zz,
            transform=data_coordinate, **kw)
    elif plot_type in ('quiver',):
        regrid_shape = kw.pop('regrid_shape', 20)
        quiver_color = kw.pop('quiver_color', 'g')
        magnitude_on = kw.pop('magnitude_on', False)
        if magnitude_on:
            plot_obj = plot_funcs[plot_type](lon, lat, uu, vv, zz, 
                transform=data_coordinate, regrid_shape=regridShape, **kw)
        else:
            plot_obj = plot_funcs[plot_type](lon, lat, uu, vv, 
                transform=data_coordinate, regrid_shape=regrid_shape, 
                color=quiver_color, **kw)
    elif plot_type in ('streamplot',):
        stream_color = kw.pop('stream_color', zz)
        plot_obj = plot_funcs[plot_type](lon, lat, uu, vv,
            transform=data_coordinate, color=stream_color, **kw)
    print('plot_type =', plot_type)

    #### plot colorbar
    # ax_phantom = plt.axes(ax.get_position(True), visible=False)
    # divider = make_axes_locatable(ax_phantom)
    # cax = divider.append_axes(cbarPosition, size=cbarSize, pad=cbarPad)
    # cbar = plt.colorbar(plotObj, cax=cax, extend=cbarExtend,
    #     orientation=cbarOrientation, **cbar_kw)
    # if cbarOrientation in ('v', 'vertical'):
    #     # put the units on the top of the vertical colorbar
    #     cbar.ax.xaxis.set_label_position('top')
    #     cbar.ax.set_xlabel(units)
    #     cbar.ax.set_ylabel(long_name)
    # elif cbarOrientation in ('h', 'horizontal'):
    #     if long_name == '' or units =='':
    #         cbar.ax.set_xlabel('{}{}'.format(long_name, units))
    #     else:
    #         cbar.ax.set_xlabel('{} [{}]'.format(long_name, units))

    return plot_obj
