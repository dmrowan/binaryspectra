#!/usr/bin/env python

import brokenaxes
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib
import numpy as np

#Dom Rowan 2024

desc="""
Utility plotting functions
"""

colors = ["#3696ff", "#f70065", "#011a7c", "#761954", "#8800b2"]

def plotparams(ax, labelsize=15):
    '''
    Basic plot params

    :param ax: axes to modify

    :type ax: matplotlib axes object

    :returns: modified matplotlib axes object
    '''

    ax.minorticks_on()
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    ax.tick_params(direction='in', which='both', labelsize=labelsize)
    ax.tick_params('both', length=8, width=1.8, which='major')
    ax.tick_params('both', length=4, width=1, which='minor')
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(1.5)
    return ax

def plotparams_cbar(cbar):

    cbar.ax.tick_params(direction='out', which='both', labelsize=15)
    cbar.ax.tick_params('y', length=8, width=1.8, which='major')
    cbar.ax.tick_params('y', length=4, width=1, which='minor')

    for axis in ['top', 'bottom', 'left', 'right']:
        cbar.ax.spines[axis].set_linewidth(1.5)

    return cbar

def plt_return(created_fig, fig, ax, savefig, dpi=300):
    if created_fig:
        if savefig is not None:
            fig.savefig(savefig, dpi=dpi)
            return
        else:
            plt.show()
            return
    else:
        return ax

def fig_init(ax=None, use_plotparams=True, figsize=(12,6), tight=True, **kwargs):

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize, **kwargs)
        created_fig=True
    else:
        created_fig=False
        fig=None

    if created_fig and tight:
        fig.subplots_adjust(top=.98, right=.98)

    if isinstance(ax, brokenaxes.BrokenAxes):
        ax = plotparams_bax(ax)
    elif use_plotparams and not has_twin(ax):
        ax = plotparams(ax)

    return fig, ax, created_fig

def has_twin(ax):
    for other_ax in ax.figure.axes:
        if other_ax is ax:
            continue
        if other_ax.bbox.bounds == ax.bbox.bounds:
            return True
    return False

def format_latex_label(label):

    return label.replace('_', ' ')

def many_colors():
    
    colors = [(25,25,25),(0,92,49),(43,206,72),(255,204,153),
              (148,255,181),(143,124,0),(157,204,0),
              (255,0,16),(94,241,242),(0,153,143),(224,255,102),(116,10,255),
              (153,0,0),(255,80,5),
              (194,0,136),(0,51,128),(255,164,5),(66,102,0),
              (240,163,255),(0,117,220),(153,63,0),(76,0,92)][::-1]

    return colors

def plotparams_bax(bax):

    for i, ax in enumerate(bax.axs):

        ax.xaxis.set_ticks_position('both')
        ax.tick_params(direction='in', which='both', labelsize=15, axis='both', reset=True)
        ax.spines['top'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)

        ax.minorticks_on()

        if i == 0:
            ax.spines['left'].set_linewidth(1.5)
            ax.tick_params('both', length=8, width=1.8, which='major')
            ax.tick_params('both', length=4, width=1, which='minor')
        if i == len(bax.axs)-1:
            ax.spines['right'].set_linewidth(1.5)
            ax.yaxis.tick_right()
            ax.tick_params('both', length=8, width=1.8, which='major')
            ax.tick_params('both', length=4, width=1, which='minor')
            ax.yaxis.set_ticklabels([])
        else:
            pass

    for i in range(len(bax.axs)): 
        if (i != 0) and (i != len(bax.axs)-1): 
            bax.axs[i].tick_params(axis='y', which='minor', left=False) 
         
        bounds = bax.axs[i].get_position().bounds 
        size = bax.fig.get_size_inches() 
        ylen = bax.d*np.sin(bax.tilt*np.pi/180)*size[0]/size[1] 
        xlen = bax.d*np.cos(bax.tilt*np.pi/180) 
 
        d_kwargs=dict(transform=bax.fig.transFigure,  
                      color=bax.diag_color, clip_on=False, lw=1.5) 
 
        if i != len(bax.axs)-1: 
            xpos = bounds[0]+bounds[2] 
            ypos = bounds[1]+bounds[3] 
            bax.axs[i].plot((xpos-xlen, xpos+xlen), (ypos-ylen, ypos+ylen), **d_kwargs) 
            bax.axs[i].plot((xpos-xlen, xpos+xlen), (bounds[1]-ylen, bounds[1]+ylen), **d_kwargs) 
 
        if i != 0: 
            xpos = bounds[0] 
            ypos = bounds[1]+bounds[3] 
            bax.axs[i].plot((xpos-xlen, xpos+xlen), (ypos-ylen, ypos+ylen), **d_kwargs) 
            bax.axs[i].plot((xpos-xlen, xpos+xlen), (bounds[1]-ylen, bounds[1]+ylen), **d_kwargs) 
 
    return bax

def DoubleY(ax, colors=('black', 'black')):
    '''
    Create a double y axis with two seperate colors
    
    :param ax: axes to modify
    
    :type ax: matplotlib axes object 
    
    :param colors: 2-tuple of axes colors 
        
    :type colors: tuple length 2

    :returns: two axes, modified original and new y scale
    '''
    if (type(colors) != tuple) or (len(colors) != 2):
        raise TypeError("colors must be 2-tuple") 
    ax2 = ax.twinx()
    ax.minorticks_on()
    ax.xaxis.set_ticks_position('both')
    for a in [ax, ax2]:
        a.minorticks_on()
        a.tick_params(direction='in', which='both', labelsize=15)
        a.tick_params('both', length=8, width=1.8, which='major')
        a.tick_params('both', length=4, width=1, which='minor')
        for axis in ['top', 'bottom', 'left', 'right']:
            a.spines[axis].set_linewidth(1.5)
    ax.tick_params('y', colors=colors[0], which='both')
    ax2.tick_params('y', colors=colors[1], which='both')

    return ax, ax2

def autoscale_y(ax,margin=0.1):
    """This function rescales the y-axis based on the data that is visible given the current xlim of the axis.
    ax -- a matplotlib axes object
    margin -- the fraction of the total height of the y-data to pad the upper and lower ylims"""

    import numpy as np

    def get_bottom_top(line):
        xd = line.get_xdata()
        yd = line.get_ydata()
        lo, hi = ax.get_xlim()
        y_displayed = yd[((xd > lo) & (xd < hi))]
        
        # Filter out NaN and Inf values
        y_displayed = y_displayed[np.isfinite(y_displayed)]
        
        if len(y_displayed) == 0:
            return np.nan, np.nan

        h = np.nanmax(y_displayed) - np.nanmin(y_displayed)
        bot = np.nanmin(y_displayed) - margin * h
        top = np.nanmax(y_displayed) + margin * h
        return bot, top

    lines = ax.get_lines()
    bot,top = np.inf, -np.inf

    for line in lines:
        new_bot, new_top = get_bottom_top(line)
        if not np.isnan(new_bot):
            if new_bot < bot: 
                bot = new_bot

        if not np.isnan(new_top):
            if new_top > top: 
                top = new_top

    ax.set_ylim(bot,top)
    
    return ax

    
def get_colors(vals, cmap='plasma'):
    
    if isinstance(plt.get_cmap(cmap), matplotlib.colors.ListedColormap):
        return [ plt.get_cmap(cmap).colors[i] 
                 for i in np.linspace(
                        0, int(0.75*len(plt.get_cmap(cmap).colors)),
                        len(vals), dtype=int)]
    elif isinstance(plt.get_cmap(cmap), matplotlib.colors.LinearSegmentedColormap):
        return [ plt.get_cmap(cmap)(np.arange(0, plt.get_cmap(cmap).N))[i]
                 for i in np.linspace(
                        0, int(0.75*plt.get_cmap(cmap).N),
                        len(vals), dtype=int)]

def rv_phase_check(rgeb, savefig=None):
    '''
    quick plot to see if any of the spectra were taken during an eclipse
    if so they need to be skipped for spectral disentangling
    '''

    fig, ax = plt.subplots(2, 1, figsize=(8, 10))
    fig.subplots_adjust(top=.98, right=.98)

    ax[0] = rgeb.vsg.plot_lc(nphase=1, plot_binned=False, yvals='flux', errorbar=True, yerr='flux_err', ax=ax[0])

    rgeb.df_rv['phase'] = ((rgeb.df_rv.JD + 2.46e6 - rgeb.vsg.t0)%rgeb.vsg.period)/rgeb.vsg.period

    ax[1].scatter(rgeb.df_rv.phase, rgeb.df_rv.RV1, color='black')
    ax[1].scatter(rgeb.df_rv.phase, rgeb.df_rv.RV2, color='xkcd:red')

    ax[1] = plotparams(ax[1])
    ax[1].set_xlim(ax[0].get_xlim())

    for i in range(len(rgeb.df_rv)):
        ax[0].axvline(rgeb.df_rv.phase.iloc[i], color='gray', alpha=0.4)
        ax[1].axvline(rgeb.df_rv.phase.iloc[i], color='gray', alpha=0.4)

    ax[1].set_xlabel('Phase', fontsize=20)
    ax[1].set_ylabel('Radial Velocity (km/s)', fontsize=20)

    return plt_return(True, fig, ax, savefig)

def create_markers_dict(instrument_list):
    
    markers = ['o', 's', 'd', 'h', 'D', 'P']
    if len(instrument_list) > len(markers):
        raise ValueError(f'Not enough default markers ({len(markers)})for given instrument list ({len(instrument_list)})')
    else:
        markers_dict = dict(zip(instrument_list, markers[:len(instrument_list)]))

        return markers_dict



