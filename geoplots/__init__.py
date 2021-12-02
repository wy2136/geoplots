'''
Author: Wenchang Yang (yang.wenchang@uci.edu)
'''

# from .geoplotlib import geoplot
try:
    from .basemap.geoplotlib import geoplot # .geoplotlib is an old version
except:
    print('There are some problems for the deprecated basemap package.')
from .fxyplotlib import fxyplot
from .mapplotlib import mapplot
from .util import (xticksmonth, xticksyear, xticks2lat, xticks2lon, xticks2month,
    xticks2dayofyear,
    yticks2p, yticks2lat, yticks2lon, yticks2month, yticks2dayofyear)
