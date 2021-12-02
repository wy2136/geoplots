'''
Author: Wenchang Yang (yang.wenchang@uci.edu)
'''
import matplotlib.pyplot as plt
from matplotlib import dates
import datetime
import numpy as np

def xticksmonth(bymonth=None, bymonthday=1, interval=1, tz=None, ax=None,
    fstr=None):
    """Make ticks on a given day of each year that is a multiple of base."""
    if ax is None:
        ax = plt.gca()
    if fstr is None:
        fstr = '%Y-%m'
    months = dates.MonthLocator(bymonth=bymonth, bymonthday=bymonthday,
        interval=interval, tz=tz)
    ax.xaxis.set_major_locator(months)

    # DateFormatter
    f = dates.DateFormatter(fstr)
    ax.xaxis.set_major_formatter(f)

def xticksyear(base=1, month=1, day=1, tz=None, ax=None):
    """Make ticks on a given day of each year that is a multiple of base."""
    if ax is None:
        ax = plt.gca()
    years = dates.YearLocator(base=base, month=month, day=day, tz=tz)
    ax.xaxis.set_major_locator(years)


def xticks2lon(xticks=None, **kw):
    '''Convert xticks to longitudes. '''
    ax = kw.pop('ax', plt.gca())

    if xticks is None:
        xticks = ax.get_xticks()
    else:
        ax.set_xticks(xticks)
    xticklabels = ax.get_xticklabels()

    for i, x in enumerate(xticks):
        x = np.mod(x,360)
        if 0<x<180: #x>0 and x<180:
            # new_xticklabels[i] = str(int(x)) + '$^{\circ}$E'
            xticklabels[i] = '{}$^{{\circ}}$E'.format(x)
        elif 180<x<360: #x>180 and x<360:
            # new_xticklabels[i] = str(int(360-x))+'$^{\circ}$W'
            xticklabels[i] = '{}$^{{\circ}}$W'.format(360-x)
        elif -180<x<0:
            xticklabels[i] = '{}$^{{\circ}}$W'.format(-x)
        elif x==0 or x==180:
            # new_xticklabels[i] = str(int(x)) + '$^{\circ}$'
            xticklabels[i] = '{}$^{{\circ}}$'.format(x)

    ax.set_xticklabels(xticklabels, **kw)

def yticks2lon(yticks=None, **kw):
    '''Convert xticks to longitudes. '''
    ax = kw.pop('ax', plt.gca() )

    if yticks is None:
        yticks = ax.get_yticks()
    else:
        ax.set_yticks(yticks)
    yticklabels = ax.get_yticklabels()

    for i, y in enumerate(yticks):
        y = np.mod(y,360)
        if 0<y<180: #x>0 and x<180:
            # new_xticklabels[i] = str(int(x)) + '$^{\circ}$E'
            yticklabels[i] = '{}$^{{\circ}}$E'.format(y)
        elif 180<y<360: #x>180 and x<360:
            # new_xticklabels[i] = str(int(360-x))+'$^{\circ}$W'
            yticklabels[i] = '{}$^{{\circ}}$W'.format(360-y)
        elif -180<y<0:
            yticklabels[i] = '{}$^{{\circ}}$W'.format(-y)
        elif y==0 or y==180:
            # new_xticklabels[i] = str(int(x)) + '$^{\circ}$'
            yticklabels[i] = '{}$^{{\circ}}$'.format(y)

    #plt.yticks(yticks, yticklabels, **kw)
    ax.set_yticklabels(yticklabels, **kw)

def xticks2lat(xticks=None, **kw):
    '''Convert yticks to latitudes. '''
    ax = kw.pop('ax', plt.gca())

    if xticks is None:
        xticks = ax.get_xticks()
    else:
        ax.set_xticks(xticks)
    xticklabels = ax.get_xticklabels()

    for i, x in enumerate(xticks):
        if x>0:
            xticklabels[i] = '{}$^{{\circ}}$N'.format(x)
        elif x<0 :
            xticklabels[i] = '{}$^{{\circ}}$S'.format(-x)
        else:
            xticklabels[i] = '0$^{\circ}$'

    #plt.xticks(xticks, xticklabels, **kw)
    ax.set_xticklabels(xticklabels, **kw)

def yticks2lat(yticks=None, **kw):
    '''Convert yticks to latitudes. '''
    ax = kw.pop('ax', plt.gca())

    if yticks is None:
        yticks = ax.get_yticks()
    else:
        ax.set_yticks(yticks)
    yticklabels = ax.get_yticklabels()

    for i, y in enumerate(yticks):
        if y>0:
            yticklabels[i] = '{}$^{{\circ}}$N'.format(y)
        elif y<0 :
            yticklabels[i] = '{}$^{{\circ}}$S'.format(-y)
        else:
            yticklabels[i] = '0$^{\circ}$'

    ax.set_yticklabels(yticklabels, **kw)

def yticks2p(yticks=None, **kw):
    '''Convert yticks to pressure level'''
    ax = kw.pop('ax', plt.gca())

    if yticks is not None:
        ax.set_yticks(yticks, **kw)
    ax.invert_yaxis()

def xticks2month(xticks=None, show_initials=False, **kw):
    '''Convert xticks to months. '''
    ax = kw.pop('ax', plt.gca())
    if xticks is None:
        xticks = range(1,13)
    month_numbers = [(m-1)%12 + 1 for m in xticks]
    months = [datetime.datetime(2000, month, 1).strftime('%b')
        for month in month_numbers]
    if show_initials:
        months = [mon[0] for mon in months]
    ax.set_xlim(xticks[0]-0.5, xticks[-1]+0.5)
    ax.set_xticks(xticks)
    ax.set_xticklabels(months, **kw)

def yticks2month(yticks=None, show_initials=False, **kw):
    '''Convert yticks to months. '''
    ax = kw.pop('ax', plt.gca())
    if yticks is None:
        yticks = range(1, 13)
    month_numbers = [(m-1)%12 + 1 for m in yticks]
    months = [datetime.datetime(2000, month, 1).strftime('%b')
        for month in month_numbers]
    if show_initials:
        months = [mon[0] for mon in months]
    ax.set_ylim(yticks[0]-0.5, yticks[-1]+0.5)
    ax.set_yticks(yticks)
    ax.set_yticklabels(months, **kw)
    ax.invert_yaxis()

def xticks2dayofyear(**kw):
    '''adjust xticks when it represents day of year.'''
    ax = kw.pop('ax', plt.gca())
    xticks = [datetime.datetime(2000, month, 1).timetuple().tm_yday
        for month in range(1, 13)]
    months = [datetime.datetime(2000, month, 1).strftime('%b')
        for month in range(1, 13)]
    ax.set_xticks(xticks)
    ax.set_xticklabels(months, ha='left', **kw)
    ax.set_xlim(1, 366)

def yticks2dayofyear(**kw):
    '''adjust yticks when it represents day of year.'''
    ax = kw.pop('ax', plt.gca())
    yticks = [datetime.datetime(2000, month, 1).timetuple().tm_yday
        for month in range(1, 13)]
    months = [datetime.datetime(2000, month, 1).strftime('%b')
        for month in range(1, 13)]
    ax.set_yticks(yticks)
    ax.set_yticklabels(months, **kw)
    ax.set_ylim(1, 366)
    ax.invert_yaxis()
