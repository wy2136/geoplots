'''
Author: Wenchang Yang (yang.wenchang@uci.edu)
'''
import matplotlib.pyplot as plt
from matplotlib import dates
import datetime


def xticksyear(base=1, month=1, day=1, tz=None, ax=None):
    """Make ticks on a given day of each year that is a multiple of base."""
    if ax is None:
        ax = plt.gca()
    years = dates.YearLocator(base=base, month=month, day=day, tz=tz)
    ax.xaxis.set_major_locator(years)


def xticks2lon(xticks=None, **kw):
    '''Convert xticks to longitudes. '''
    ax = plt.gca()

    if xticks is None:
        xticks = ax.get_xticks()
    else:
        plt.xticks(xticks)
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

    plt.xticks(xticks, xticklabels, **kw)
def yticks2lon(yticks=None, **kw):
    '''Convert xticks to longitudes. '''
    ax = plt.gca()

    if yticks is None:
        yticks = ax.get_yticks()
    else:
        plt.yticks(yticks)
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

    plt.yticks(yticks, yticklabels, **kw)

def xticks2lat(xticks=None, **kw):
    '''Convert yticks to latitudes. '''
    ax = plt.gca()

    if xticks is None:
        xticks = ax.get_xticks()
    else:
        plt.xticks(xticks)
    xticklabels = ax.get_xticklabels()

    for i, x in enumerate(xticks):
        if x>0:
            xticklabels[i] = '{}$^{{\circ}}$N'.format(x)
        elif x<0 :
            xticklabels[i] = '{}$^{{\circ}}$S'.format(-x)
        else:
            xticklabels[i] = '0$^{\circ}$'

    plt.xticks(xticks, xticklabels, **kw)
def yticks2lat(yticks=None, **kw):
    '''Convert yticks to latitudes. '''
    ax = plt.gca()

    if yticks is None:
        yticks = ax.get_yticks()
    else:
        plt.yticks(yticks)
    yticklabels = ax.get_yticklabels()

    for i, y in enumerate(yticks):
        if y>0:
            yticklabels[i] = '{}$^{{\circ}}$N'.format(y)
        elif y<0 :
            yticklabels[i] = '{}$^{{\circ}}$S'.format(-y)
        else:
            yticklabels[i] = '0$^{\circ}$'

    plt.yticks(yticks, yticklabels, **kw)

def yticks2p(yticks=None, **kw):
    '''Convert yticks to pressure level'''

    if yticks is not None:
        plt.yticks(yticks, **kw)

    ax = plt.gca()
    ax.invert_yaxis()

def xticks2month(show_initials=False, **kw):
    '''Convert xticks to months. '''
    months = [datetime.datetime(2000, month, 1).strftime('%b')
        for month in range(1, 13)]
    if show_initials:
        months = [mon[0] for mon in months]
    plt.xlim(.5,12.5)
    plt.xticks(range(1,13),months, **kw)
def yticks2month(show_initials=False, **kw):
    '''Convert yticks to months. '''
    months = [datetime.datetime(2000, month, 1).strftime('%b')
        for month in range(1, 13)]
    if show_initials:
        months = [mon[0] for mon in months]
    plt.ylim(.5,12.5)
    plt.yticks(range(1,13),months, **kw)
    plt.gca().invert_yaxis()

def xticks2dayofyear(**kw):
    '''adjust xticks when it represents day of year.'''
    xticks = [datetime.datetime(2000, month, 1).timetuple().tm_yday
        for month in range(1, 13)]
    months = [datetime.datetime(2000, month, 1).strftime('%b')
        for month in range(1, 13)]
    plt.xticks(xticks, months, ha='left', **kw)
    plt.xlim(1, 366)
def yticks2dayofyear(**kw):
    '''adjust yticks when it represents day of year.'''
    yticks = [datetime.datetime(2000, month, 1).timetuple().tm_yday
        for month in range(1, 13)]
    months = [datetime.datetime(2000, month, 1).strftime('%b')
        for month in range(1, 13)]
    plt.yticks(yticks, months, **kw)
    plt.ylim(1, 366)
    plt.gca().invert_yaxis()
