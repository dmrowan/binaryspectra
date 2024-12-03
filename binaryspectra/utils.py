#!/usr/bin/env python

import time
from astropy import log
from astropy.io import fits
from astropy import constants as const
from astropy.coordinates import SkyCoord, EarthLocation
from astropy import units as u
from astropy.time import Time
from datetime import datetime
from math import log10, floor
import numpy as np
import os
import pandas as pd
import pkg_resources
import sys

#Dom Rowan 2024

def get_data_file(filename):
    return pkg_resources.resource_filename(__name__, '../data/' + filename)

def get_plot_file(filename):
    return pkg_resources.resource_filename(__name__, '../plots/' + filename)

def get_table_file(filename):
    return pkg_resources.resource_filename(__name__, '../tables/' + filename)

#Check if list tuple, np array, etc
def check_iter(var):
    if isinstance(var, str):
        return False
    elif hasattr(var, '__iter__'):
        return True
    else:
        return False

#utility function for reading in df from various extensions
def pd_read(table_path, low_memory=False):

    if (not isinstance(table_path, pd.core.frame.DataFrame)
        and check_iter(table_path)):
        df0 = pd_read(table_path[0])
        for i in range(1, len(table_path)):
            df0 = pd.concat([df0, pd_read(table_path[i])])
        return df0
    else:
        if type(table_path) == pd.core.frame.DataFrame:
            return table_path
        elif table_path.endswith('.csv') or table_path.endswith('.dat'):
            return pd.read_csv(table_path, low_memory=low_memory)
        elif table_path.endswith('.pickle'):
            return pd.read_pickle(table_path)
        else:
            raise TypeError("invalid extension")

#Utility function for standard table formatting
def write_latex_table(df, labels, outfile):

    df = df[list(labels.keys())]
    df.columns = [ '{'+ x[0] +'}' for x in list(labels.values()) ]

    df.to_latex(outfile, index=False,
                column_format= ' '.join([x[2] for x in list(labels.values())]),
                escape=False)

    with open(outfile, 'r') as f:
        lines = f.readlines()

    lines.insert(3, ' & '.join(['{'+x[1]+'}'
                                for x in list(labels.values())])+'\\\ \n')

    with open(outfile, 'w') as f:
        for line in lines:
            f.write(line)

def round_val_err(val, err, err_upper=None, as_string=False):

    if np.isnan(err):
        return str(round(val, 3))
    
    if err_upper is not None:
        return round_val_err_uneven(val, err, err_upper)
    else:
        n = int(-1*np.floor(np.log10(err)))

        if n < 0:
            if as_string:
                return r'$'+str(int(round(val, n)))+r'\pm'+str(int(round(err,n)))+'$'
            else:
                return int(round(val, n)), int(round(err, n))
        else:
            if as_string:
                return r'$'+str(round(val, n))+r'\pm'+str(round(err, n))+'$'
            return round(val, n), round(err, n)

def round_val_err_uneven(val, err_lower, err_upper):

    ndigits_lower = -int(floor(log10(abs(err_lower))))
    ndigits_upper = -int(floor(log10(abs(err_upper))))

    ndigits = np.max([ndigits_lower, ndigits_upper])

    if ndigits <= 0:
        val_string = str(int(round(val, ndigits)))
        upper_string = str(int(round(err_upper, ndigits)))
        lower_string = str(int(round(err_lower, ndigits)))
    else:
        val_string = '{0:.{1}f}'.format(val, ndigits)
        upper_string = '{0:.{1}f}'.format(err_upper, ndigits_upper)
        lower_string = '{0:.{1}f}'.format(err_lower, ndigits_lower)

    return r'$'+val_string+r'^{+'+upper_string+r'}_{-'+lower_string+r'}$'

def mass_function(P, K, e=0):

    """
    if type(P) != u.quantity.Quantity:
        raise TypeError("period must be astropy quantity")
    if type(K) != u.quantity.Quantity:
        raise TypeError("semiamplitude must be astropy quantity")
    """

    f = P.to('s')*(K.to('m/s')**3)/(2*np.pi*const.G) * np.power(1-np.power(e, 2), 3/2)
    #f *= (1-e**2)**(3/2)
    return f.to('M_sun')

def companion_mass(f, Mstar, inc, verbose=True):
        
    if inc > 2*np.pi: 
        if verbose:
            log.info('Using inclination in degrees')
        #convert to radians
        inc = inc*np.pi/180

    f = f.to('Msun').value
    Mstar = Mstar.to('Msun').value
    
    a = np.power(np.sin(inc),3)
    b = -1*f
    c = -2*Mstar*f
    d = -1*f*Mstar**2

    roots = np.roots([a,b,c,d])
    mc = roots[np.where(~np.iscomplex(roots))[0]][0].real
    return mc*u.Msun

class HiddenPrints:
    def __init__(self, verbose=False):
        self.verbose = verbose

    def __enter__(self):
        if not self.verbose:
            self._original_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.verbose:
            sys.stdout.close()
            sys.stdout = self._original_stdout

def manager_list_wrapper(func, L, *args, **kwargs):
    return_vals = func(*args, **kwargs)
    L.append(return_vals)
    return return_vals

def get_short_name(ra, dec):
    coord = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame='icrs')
    name = 'J'+coord.ra.to_string(u.hour, sep='', pad=True, precision=2)+coord.dec.to_string(u.degree, sep='', pad=True, precision=1)
    return name[:5]


def golden_ratio_search(func, x1, x4, tol=1e-6):

    x1_0 = x1
    x4_0 = x4

    z = (1+np.sqrt(5))/2

    midpoint = (x1+x4)/2
    x3 = x1+(1/z)*(x4-x1)
    dx3 = x3-midpoint
    x2 = x3-2*dx3

    xvals = [x3]
    if not ((func(x2) < np.min([func(x1), func(x4)])) or
            (func(x3) < np.min([func(x1), func(x4)]))):
        raise ValueError('invalid range')

    while (x4-x1) > tol:
        #new range x1 -- x3
        if func(x2) < func(x3):
            x4 = x3
            x3 = x2
            midpoint = (x1+x4)/2
            dx3 = x3-midpoint
            x2 = x3-2*dx3
            xvals.append(x2)
        #new range x2 -- x4
        else:
            x1 = x2
            x2 = x3
            midpoint = (x1+x4)/2
            dx2 = midpoint - x2
            x3 = x2+2*dx2
            xvals.append(x3)

    xmin = 0.5*(x2+x3)

    return xmin

#Binary search method for nonlinear equations
def binary_search(f, x1, x2, epsilon=1e-6, plot=False, ax=None):
    #Check if we can solve for a root in this interval
    if f(x1)*f(x2) > 0:
        return float('NaN')
    xp = .5*(x1+x2) #calculate x prime
    xvals = [xp] #save values as we iterate
        
    counter = 0
    #Loop until we reach precision (or stumble onto value)
    while (f(xp)!=0) and (abs(x1-x2)>epsilon):
        #Redefine our bracket
        if f(x1)*f(xp) < 0:
            x2 = xp
        else:
            x1 = xp
        xp = .5*(x1+x2)
        xvals.append(xp)
    
    #Option to plot
    if plot:
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(12, 6))
            ax = plotutils.plotparams(ax)
        ax.plot(xvals, color='xkcd:azure', label='Binary Search')

    #Returned solved value and plot
    if plot and ax is not None:
        return xp, ax
    else:
        return xp

def convert_jd_bjd(jd_utc, ra, dec, site):
    
    coord = SkyCoord(ra, dec, unit=(u.deg, u.deg), frame='icrs')
    site = EarthLocation.of_site(site)

    t = Time(jd_utc, format='jd', scale='utc', location=site)
    ltt_bary = t.light_travel_time(coord)

    jd_bjd = t.tdb + ltt_bary

    return jd_bjd.value


class MCMCData:
    
    def __init__(self, samples, parameter_names):
        
        if samples.shape[1] != len(parameter_names):
            raise ValueError(f"The length of parameter_names must match the \
                               number of dimensions in samples.")

        self.samples = samples
        self.parameter_names = parameter_names
        self.param_to_index = {name: idx for idx, name in enumerate(parameter_names)}

    def get_samples_by_name(self, name):
        
        if name not in self.param_to_index:
            raise ValueError(f'Parameter {name} not found')

        idx = self.param_to_index[name]

        return self.samples[:,idx]


def _delta_t_infconj_perpass(period, ecc, per0):
    """
    time shift between inferior conjuction and periastron passage
    """

    per0 = per0*np.pi/180

    ups_sc = 3*np.pi/2-per0
    E_sc = 2*np.arctan( np.sqrt((1-ecc)/(1+ecc)) * np.tan(ups_sc/2) )
    M_sc = E_sc - ecc*np.sin(E_sc)
    return period*(M_sc/2./np.pi)


def t0_perpass_to_infconj(t0_perpass, period, ecc, per0):

    return t0_perpass + _delta_t_infconj_perpass(period, ecc, per0)




