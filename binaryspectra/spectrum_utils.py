#!/usr/bin/env python

from astropy import units as u
from astropy.table import Table
import numpy as np
import os
import pandas as pd
import pickle
import spectres
from scipy import ndimage
import subprocess
import warnings

from . import base_spectrum
from . import spectra
from . import utils

import sys
sys.path.append(os.environ['ISPEC_DIR'])
import ispec

#Dom Rowan 2024

def default_apf_orders():
    
    return np.concatenate([np.arange(20, 45), [47, 48, 49, 51, 54, 55, 62, 64]])

def default_chiron_orders():
    
    return np.concatenate([np.arange(5, 29), [31, 32, 33, 34, 36, 37, 40, 41, 42, 49, 50, 52]])

def load_spec(fname):
    
    with open(fname, 'rb') as p:
        spec = pickle.load(p)

    return spec

def resample_spectrum(new_waves, current_waves, flux, err=None, fill=1, simple=False):
    
    if simple:
        resampled_flux = np.interp(new_waves, current_waves, flux, left=fill, right=fill)
        return resampled_flux
    else:
        if err is None:
            resampled_flux = spectres.spectres(
                    new_waves, current_waves, flux, spec_errs=None,
                    fill=1, verbose=False)
            return resampled_flux
        else:
            resampled_flux = spectres.spectres(
                    new_waves, current_waves, flux, spec_errs=err,
                    fill=1, verbose=False)
            return resampled_flux


def mask_lines(wavelength, flux, wavelength_unit):

    #convert wavelength to angstrom
    wavelength = (wavelength * wavelength_unit).to(u.AA).value

    lines = {'halpha':[6562.79, 20],
             'hbeta':[4861.35, 20]}

    df_lines = pd.DataFrame(lines).T
    df_lines.columns = ['wavelength', 'width']

    idx_mask = []
    for i in range(len(df_lines)):

        line_wavelength = df_lines.wavelength.iloc[i]
        line_width = df_lines.width.iloc[i]

        idx = np.where( (wavelength > line_wavelength-line_width) &
                        (wavelength < line_wavelength+line_width) )[0]
        if len(idx):
            idx_mask.append(idx)

    if len(idx_mask):
        idx_mask = np.concatenate([idx_mask])
        flux[idx_mask] = np.nan

    return flux

def convert_to_air(wavelength, wavelength_unit):
    
    wavelength = (wavelength*wavelength_unit).to('AA').value

    idx = np.where(wavelength > 2000)[0]

    #Using Ciddor96 equation
    sigma_2 = np.power(10**4 / wavelength[idx], 2)
    f = 1.0 + (0.05792105/(238.0185-sigma_2)) + (0.00167917/(57.362-sigma_2))

    wavelength[idx] = wavelength[idx]/f

    return wavelength

def organize_pepsi_files(spec_dir, skip_files=None):

    all_fnames = os.listdir(spec_dir)

    pepsi_fnames = [ f for f in all_fnames if f.startswith('pepsi') ]

    if skip_files is not None:

        if not utils.check_iter(skip_files):
            skip_files = [ skip_files ]

        pepsi_fnames = [ f for f in pepsi_fnames if f not in skip_files ]

    if len(pepsi_fnames) == 0:
        return 

    pepsi_both = [ f for f in pepsi_fnames if f.endswith('all') ]
    pepsi_blue = [ f for f in pepsi_fnames if f.startswith('pepsib') 
                     and (not f.endswith('all'))]
    pepsi_red = [ f for f in pepsi_fnames if f.startswith('pepsir')]

    #Check conditions
    if not (len(pepsi_both) + len(pepsi_blue) + len(pepsi_red) == len(pepsi_fnames)):
        raise ValueError(f'Issue organizing pepsi spectra in {spec_dir}')
    if (len(pepsi_blue) != len(pepsi_red)):
        raise ValueError(f'Number of blue and red pepsi spectra do not match in {spec_dir}')

    if len(pepsi_blue) > 1:
        blue_jds = [ spectra.PEPSIspec(
                os.path.join(spec_dir, x)).header['JD-OBS'] 
                     for x in pepsi_blue ]
        red_jds = [ spectra.PEPSIspec(
                os.path.join(spec_dir, x)).header['JD-OBS'] 
                    for x in pepsi_red ]

        js = []
        for i in range(len(blue_jds)):
            match = False
            for j in range(len(blue_jds)):
                if np.abs(blue_jds[i] - red_jds[j]) < 0.01:
                    if match:
                        raise ValueError(f"multiple matches found for {pepsi_blue[i]} in {spec_dir}")
                    else:
                        js.append(j)
                        match = True
            if not match:
                raise ValueError("no match found")
        final_list =  pepsi_both + [ (pepsi_blue[i], pepsi_red[j]) 
                                     for i,j in zip(range(len(blue_jds)), js) ]
    else:
        final_list =  pepsi_both + [ (pepsi_blue[0], pepsi_red[0]) ]

    final_list_full = []
    for path in final_list:
        
        if isinstance(path, str):
            final_list_full.append(os.path.join(spec_dir, path))
        else:
            final_list_full.append( (os.path.join(spec_dir, path[0]), 
                                     os.path.join(spec_dir, path[1])))

    return final_list_full

def get_template(teff, logg, met=0.0, wavelength_range=(4900, 5300)):
    
    template_dir = os.environ['PHOENIX_DIR']
    logg_str = f'{logg:.2f}'
    if met == 0:
        template_fname = f'lte{str(teff).zfill(5)}-{logg_str}-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits'
        template_fname = os.path.join(template_dir, template_fname)
    elif met == 0.5:
        template_fname = f'lte{str(teff).zfill(5)}-{logg_str}+0.5.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits'
        template_fname = os.path.join(template_dir, 'Zp5', template_fname)
    else:
        raise ValueError(f'unavailable met {0.0}')
    
    if os.path.isfile(template_fname):
        template = spectra.PHOENIXspec(template_fname)
        template.filter_wavelength_range(*wavelength_range)
        template.fit_continuum()
        
        return template
    else:
        return

def create_telluric_mask(unit=u.AA, outfile=None):
    
    ll_file = os.path.join(os.environ['ISPEC_DIR'], 'input/linelists/CCF/Synth.Tellurics.500_1100nm/mask.lst')
    telluric_linelist = ispec.read_telluric_linelist(ll_file, minimum_depth=0.0)

    df = pd.read_table(os.path.join(os.environ['ISPEC_DIR'], 
                       'input/spectra/templates/Synth.Tellurics.350_1100nm/template.txt.gz'),
                       delim_whitespace=True)
    wavelengths_nm = df.waveobs.values

    min_vel = -30
    max_vel = 30
    dfilter = telluric_linelist['depth'] > np.percentile(telluric_linelist['depth'], 50)
    with utils.HiddenPrints():
        tfilter = ispec.create_filter_for_regions_affected_by_tellurics(
                wavelengths_nm,
                telluric_linelist[dfilter], min_velocity=min_vel,
                max_velocity=max_vel)

    #Turn the filter into a list of wavelength ranges
    groups = []
    for i in range(len(tfilter)):
        if i == 0:
            if tfilter[i]:
                groups.append([i])
            else:
                continue
        elif i == len(tfilter)-1:
            if tfilter[i] & (not tfilter[i-1]):
                groups.append([i, i])
            elif tfilter[i] and tfilter[i-1]:
                groups[-1].append(i)
            elif (not tfilter[i]) and tfilter[i-1]:
                groups[-1].append(i-1)
        else:
            if tfilter[i] & (not tfilter[i-1]):
                groups.append([i])
            elif tfilter[i] and tfilter[i-1]:
                continue
            elif (not tfilter[i]) and tfilter[i-1]:
                groups[-1].append(i-1)

    groups_wavelength = []
    for g in groups:
        wmin = wavelengths_nm[g[0]] * u.nm
        wmax = wavelengths_nm[g[1]] * u.nm

        wmin = wmin.to(unit).value
        wmax = wmax.to(unit).value

        groups_wavelength.append([wmin, wmax])

    if outfile is not None:
        with open(outfile, 'wb') as p:
            pickle.dump(groups_wavelength, p)

    return groups_wavelength, unit


def process_apf_chunk(chunk, **kwargs):
    return [ spectra.APFspec(f, **kwargs) for f in chunk ]

def modified_z_score(intensity):
    '''
    z-score from towardsdatascience.com/removing-spikes-from-raman-spectra-8a9fdda0ac22
    '''
    median_int = np.nanmedian(intensity)
    mad_int = np.nanmedian([np.abs(intensity - median_int)])
    modified_z_scores = 0.6745 * (intensity - median_int) / mad_int
    return np.abs(modified_z_scores)

def quantile_filter(arr, window_size, quantile):
    """
    Apply a quantile filter to an array.
    
    Parameters:
        arr (numpy.ndarray): Input array.
        window_size (int): Size of the window for filtering.
        quantile (float): Quantile value (e.g., 0.9 for 90th percentile).
        
    Returns:
        numpy.ndarray: Filtered array.
    """
    filtered = np.zeros_like(arr)
    padding = window_size // 2
    
    for i in range(padding, len(arr) - padding):
        window = arr[i - padding:i + padding + 1]
        filtered[i] = np.percentile(window, quantile * 100)
        
    return filtered


def interpolate_spectrum(teff, logg, MH=0.0, vsini=0.0,
                         limb_darkening_coeff=0.6,
                         alpha=None, 
                         vmic=None, vmac=None,
                         wavelength_range=None):
    '''
    pass wavelength range in angstroms
    '''

    if wavelength_range is None:
        wavelength_range = (480, 520)
    else:
        wavelength_range = (wavelength_range[0]/10, wavelength_range[1]/10)

    if alpha is None:
        alpha = ispec.determine_abundance_enchancements(MH)

    if vmic is None:
        vmic = ispec.estimate_vmic(teff, logg, MH) # 1.07
    if vmac is None:
        vmac = ispec.estimate_vmac(teff, logg, MH) # 4.21

    vsini = vsini
    limb_darkening_coeff = 0.6
    resolution = 300000
    wave_step = 0.001

    wave_base = wavelength_range[0]
    wave_top = wavelength_range[1]

    code = "grid"
    precomputed_grid_dir = os.path.join(
            os.environ['ISPEC_DIR'],
            "input/grid/SPECTRUM_ATLAS9.Castelli_SPECTRUM.380_900nm")
    grid = ispec.load_spectral_grid(precomputed_grid_dir)

    atomic_linelist = None
    isotopes = None
    modeled_layers_pack = None
    solar_abundances = None
    fixed_abundances = None
    abundances = None
    atmosphere_layers = None
    regions = None

    # Validate parameters
    if not ispec.valid_interpolated_spectrum_target(grid, {'teff':teff, 'logg':logg, 'MH':MH, 'alpha':alpha, 'vmic': vmic}):
        msg = "The specified effective temperature, gravity (log g) and metallicity [M/H] \
                fall out of the spectral grid limits."
        print(msg)

    interpolated_spectrum = ispec.create_spectrum_structure(
            np.arange(wave_base, wave_top, wave_step))
    interpolated_spectrum['flux'] = ispec.generate_spectrum(
            interpolated_spectrum['waveobs'],
            atmosphere_layers, teff, logg, MH, alpha, atomic_linelist, isotopes, 
            abundances, fixed_abundances, microturbulence_vel=vmic,
            macroturbulence=vmac, vsini=vsini, 
            limb_darkening_coeff=limb_darkening_coeff,
            R=resolution, regions=regions, verbose=1,
            code=code, grid=grid)
    
    spec = base_spectrum.BaseSpectrum(interpolated_spectrum['waveobs'][1:-1]*10,
                                      interpolated_spectrum['flux'][1:-1])
    return spec

def fd3_worker(fd3_path, fname, verbose=False, write_log=False):

    if write_log:
        logfile = fname.replace('.in', '.log')

        with open(logfile, 'w', encoding='utf-8') as f:
            stdout_dest = f
            stderr_dest = subprocess.STDOUT
            result = subprocess.run([fd3_path], stdin=open(fname, 'r'),
                                    stdout=stdout_dest,
                                    stderr=stderr_dest,
                                    text=True)
    else:
        stdout_dest = None if verbose else subprocess.DEVNULL
        stderr_dest = None if verbose else subprocess.DEVNULL

        result = subprocess.run([fd3_path], stdin=open(fname, 'r'),
                            stdout=stdout_dest,
                            stderr=stderr_dest,
                            text=True)

    if result.returncode != 0:
        raise ValueError(f'fd3 failed to run {result.returncode}')


    return result

def convert_pepsi_to_echelle(spec, wmins, wmaxs, 
                             dlambdas=None, log=False, orders=None, 
                             wavelength_unit=u.AA): 

    '''
    For using FDBinary I need to convert a pepsi spec into an echelle spectra
    '''

    if dlambdas is None:
        dlambdas = [None]*len(wmins)

    if orders is None:
        orders = np.arange(len(wmins)) 

    if spec.has_red_arm_data:
        log.warning('Data from red arm is present. Check Order selection')

    Orders = []
    for wmin, wmax, dlambda, order in zip(wmins, wmaxs, dlambdas, orders):
        
        wavelengths = spec.df.wavelength.copy().values
        wavelengths = (wavelengths*spec.wavelength_unit).to(wavelength_unit).value
        fluxes = spec.df.flux.copy().values

        idx = np.where( (wavelengths > wmin) & (wavelengths < wmax) )[0]
        
        #Check that we have the full wavelength range in PEPSI for that order
        if np.min(wavelengths) > wmin:
            continue

        #This doesn't work if we still have the red arm in the data
        if np.max(wavelengths) < wmax:
            continue

        if len(idx):
            df = pd.DataFrame({'wavelength': wavelengths[idx],
                               'flux': fluxes[idx]})
            eo = base_spectrum.EchelleOrder(order, wavelengths[idx], fluxes[idx])
            
            if log:
                assert(dlambda is not None)
                eo.resample_log_wavelength(wmin, wmax, dlambda)

            Orders.append(eo)

    echelle_spec = base_spectrum.EchelleSpectrum(
            spec.df.wavelength.values,
            spec.df.flux.values,
            header=spec.header,
            wavelength_unit=wavelength_unit)
    
    #Copy over attributes
    attributes_to_copy = ['verbose', 'RA', 'DEC', 'JD', 'ObsName', '_barycentric_corrected',
                          'alt_header', '_copy_red_arm', 'fname', 'todcor_profile']
    for attr in attributes_to_copy:
        if hasattr(spec, attr):
            setattr(echelle_spec, attr, getattr(spec, attr))

    #Add orders
    echelle_spec.Orders = Orders

    #Flag as converted
    echelle_spec._converted = True

    #Set fdbinary mask (since it propogates to each order
    echelle_spec.fdbinary_mask = spec.fdbinary_mask

    return echelle_spec

class ispec_model:
    
    def __init__(self, xvals, yvals, mu, emu, amp):
        
        self.xvals = xvals
        self.yvals = yvals
        self.mu = mu
        self.emu = emu
        self.amp = amp

def convert_ispec_model_to_serializable(model, xvals):
    
    yvals = model._model_function(xvals)
    mu = model.mu()
    emu = model.emu()
    amp = model._model_function(mu)

    return ispec_model(xvals, yvals, mu, emu, amp)

def from_ispec(fname):
    
    df = pd.read_table(fname, delim_whitespace=True)

    wavelength = df.waveobs*10
    flux = df.flux

    spec = base_spectrum.BaseSpectrum(wavelength, flux)
    spec.fname = fname
    
    return spec

def parse_ispec_fname(fname):
    
    fname = os.path.split(fname)[-1]
    fname = os.path.splitext(fname)[0]
    
    params = fname.split('_')

    params = dict(teff=params[0],
                  logg=params[1],
                  met=params[2], 
                  alpha=params[3], 
                  vmic=params[4], 
                  vmac=params[5], 
                  vsini=params[6])

    params = { k:float(v) for k, v in params.items() }

    return params

def write_ispec_segments(spec, orders, outfile, trim=None):
    
    wave_base = []
    wave_top = []
    for n in orders:
        wavelength_range = spec.get_order(n).get_wavelength_range()
        if trim is None:
            wave_base.append(wavelength_range[0] / 10 )
            wave_top.append(wavelength_range[1] / 10 )
        else:
            wave_base.append( (wavelength_range[0]+trim)/10)
            wave_top.append( (wavelength_range[1]-trim)/10)

    df = pd.DataFrame({'wave_base':wave_base, 'wave_top':wave_top})
    df.to_csv(outfile, sep='\t', index=False)


