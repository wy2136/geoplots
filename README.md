# geoplots: customized python plots for geoscience

## Example

    from geoplots import geoplot
    import xarray as xr
  
    da = xr.open_dataarray('http://iridl.ldeo.columbia.edu/SOURCES/.NASA/.GPCP/.V2p2/.satellite-gauge/.prcp/T(Jan%201980)(Dec%202010)RANGE%2[T]average/dods')
    geoplot(da)

![GPCP Precipitation Climatology: 1980-2010](examples/gpcp_climatology.png)
