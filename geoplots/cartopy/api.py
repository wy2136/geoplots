'''
author: Wenchang Yang (wenchang@princeton.edu)
'''
import matplotlib.pyplot as plt, cartopy.crs as ccrs, cartopy.feature as cfeature

def cartoproj(name=None, **kws):
    '''create the cartopy projection given its shorname and params.

    cyl: cartopy.crs.PlateCarree
    robin: cartopy.crs.Robinson
    moll: cartopy.crs.Mollweide
    ortho: cartopy.crs.Orthographic
    geo: cartopy.crs.Geostationary
    '''
    projs = {
        'cyl': ccrs.PlateCarree,
        'robin': ccrs.Robinson,
        'moll': ccrs.Mollweide,
        'ortho': ccrs.Orthographic,
        'pv': ccrs.NearsidePerspective,
        'geo': ccrs.Geostationary
    }

    if name is None:
        print('Please provide projection short names from:')
        print(list(projs.keys()))
        return

    proj = projs[name](**kws)

    return proj

def cartofeature(name=None):
    '''create the cartopy feature given its shortname and params.
    '''

    features = {
        'coastline': cfeature.COASTLINE,
        'land': cfeature.LAND,
        'borders': cfeature.BORDERS
    }

    if name is None:
        print('Please provide feature short names from:')
        print(list(features.keys()))
        return

    feature = features[name]

    return feature

def cartoplot(da, **kws):
    '''visulize data on basemap using the cartopy package'''
    # params: proj_kws, proj, dproj, ax, ...

    # figure
    figsize = kws.pop('figsize', None)
    if figsize is not None:
        fig = plt.figure(figsize=figsize)


    # plot types
    plot_type = kws.pop('plot_type', 'default')
    func_plot = {'default': da.plot,
        'contourf': da.plot.contourf,
        'contour': da.plot.contour,
        'pcolormesh': da.plot.pcolormesh,
        'imshow': da.plot.imshow}

    # proj
    proj_kws = kws.pop('proj_kws', {})
    proj = kws.pop('proj', 'robin')
    if type(proj) is str:
        proj = cartoproj(proj, **proj_kws)

    # data transform
    dproj = kws.pop('dproj', ccrs.PlateCarree())
    ax = kws.pop('ax', plt.axes(projection=proj))

    coastline_on = kws.pop('coastline_on', True)
    coastline_kws = kws.pop('coastline_kws', {})
    coastline_lw = coastline_kws.pop('linewidth', 0.5)

    land_on = kws.pop('land_on', False)
    land_kws = kws.pop('land_kws', {})


    plot_obj = func_plot[plot_type](ax=ax, transform=dproj, **kws)

    if coastline_on:
        ax.add_feature(cartofeature('coastline'),
            linewidth=coastline_lw,
            **coastline_kws)
    if land_on:
        ax.add_feature(cartofeature('land'), **land_kws)

    return plot_obj
