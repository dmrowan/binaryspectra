#!/usr/bin/env python

from astropy import units as u
from astropy import constants as const
from astropy import log
from barycorrpy import get_BC_vel
import cmasher as cmr
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib import rc
import numpy as np
import os
import pandas as pd
import pickle
from specutils.spectra import Spectrum1D
from scipy.ndimage import median_filter
from tqdm import tqdm
import warnings

import specmatchemp.library
import specmatchemp.plots as smplot
from specmatchemp import spectrum
from specmatchemp.specmatch import SpecMatch

from . import plotutils
from . import spectrum_utils
from . import utils
from . import measure_rvs
from . import rotbroadint
from . import profiles

import sys
sys.path.append(os.environ['ISPEC_DIR'])
import ispec

#Dom Rowan 2024

rc('text', usetex=True)

class BaseSpectrum:
    '''
    Basic Spectrum Class
    '''

    def __init__(self, wavelength, flux, flux_err=None,
                 wavelength_unit=u.AA,
                 flux_unit=u.dimensionless_unscaled,
                 name=None,
                 verbose=True,
                 header=None):

        if flux_err is None:
            flux_err = np.zeros(len(flux))

        self.df = pd.DataFrame({
                'wavelength':wavelength,
                'flux':flux,
                'err':flux_err})
        self.df = self.df.sort_values(by='wavelength', ascending=True).reset_index(drop=True)

        self.verbose = verbose

        self.wavelength_unit = wavelength_unit
        self.flux_unit = flux_unit

        self.header = header
        self.vb = None
        self.name = name

        self.fdbinary_mask = False

        self.continuum_regions = []

    @property
    def verbose(self):
        return self._verbose
    @verbose.setter
    def verbose(self, value):
        self._verbose = bool(value)
    def print(self, value):
        if self.verbose:
            print(value)

    @property
    def fdbinary_mask(self):
        return self._fdbinary_mask

    @fdbinary_mask.setter
    def fdbinary_mask(self, value):
        self._fdbinary_mask = value

    def get_barycentric_velocity(self):
        '''
        Use barycorrpy to determine the barycentric velocity
        '''

        vb, warnings, flag = get_BC_vel(JDUTC=self.JD, ra=self.RA, dec=self.DEC,
                                        obsname=self.ObsName, predictive=True)
        self.vb = (float(vb)/1000) * u.km/u.s

        return self.vb

    @property
    def barycentric_corrected(self):
        '''
        Flag to determine if spectra had already been barycentric corrected
        '''
        return self._barycentric_corrected

    def correct_barycentric_velocity(self):
        '''
        Correct spectra for barycentric motion 
        '''

        if not self.barycentric_corrected:

            if self.vb is None:
                self.get_barycentric_velocity()

            ss = Spectrum1D(flux=self.df.flux.to_numpy()*u.one,
                            spectral_axis=self.df.wavelength.to_numpy()*self.wavelength_unit,
                            radial_velocity=self.vb)
            self.df['wavelength'] = ss.spectral_axis.to_rest().value
            self._barycentric_corrected = True

    def shift_rv(self, rv, clip=True):

        '''
        Apply RV shift
        '''
        
        c = 299792458.0
        factor = np.sqrt((1.-(rv*1000.)/c)/(1.+(rv*1000./c)))

        shifted = np.interp(self.df.wavelength, 
                            self.df.wavelength/factor, 
                            self.df.flux,
                            left=np.nan, right=np.nan)

        self.df['flux'] = shifted
        self.df = self.df[~self.df.flux.isna()].reset_index(drop=True)

        return self

    def resample_spectrum(self, wavelength_grid, **kwargs):
        '''
        Resample spectrum over input wavelength grid
        '''

        resampled_fluxes = spectrum_utils.resample_spectrum(
                wavelength_grid, self.df.wavelength.values, self.df.flux.values,
                **kwargs)

        self.df = pd.DataFrame({'wavelength':wavelength_grid,
                                'flux':resampled_fluxes})
        
        return self

    def resample_log_wavelength(self, wmin, wmax, dlambda, **kwargs):
        '''
        Resample spectrum over logarithmic wavelength grid
        '''
        
        dlnwave = np.log(wmin + dlambda) - np.log(wmin)

        lnwavestart = np.log(wmin)
        lnwavestop = np.log(wmax)

        len_log_grid = (lnwavestop - lnwavestart) / dlnwave

        wavelength_grid = np.arange(len_log_grid)*dlnwave + lnwavestart

        exp_wavelength_grid = np.exp(wavelength_grid)

        self.resample_spectrum(exp_wavelength_grid, **kwargs)
        self.df['log_wavelength'] = wavelength_grid

    def trim_edges(self, trim):
        '''
        Remove edges of spectra by input amount
        '''
        
        wmin = self.df.wavelength.min()
        wmax = self.df.wavelength.max()
        idx = np.where( (self.df.wavelength > wmin+trim) &
                        (self.df.wavelength < wmax-trim) )[0]

        self.df = self.df.iloc[idx].reset_index(drop=True)

        return self

    def filter_wavelength_range(self, *args):
        '''
        Select wavelength range of spectrum
        '''

        if args[0] is None:
            self.df = self.df[ self.df.wavelength < args[1] ]
        elif args[1] is None:
            self.df = self.df[ self.df.wavelength > args[0] ]
        else:
            self.df = self.df[ (self.df.wavelength > args[0]) & (self.df.wavelength < args[1])]
        self.df = self.df.reset_index(drop=True)

        return self

    def get_wavelength_range(self):
        
        return self.df.wavelength.min(), self.df.wavelength.max()

    def get_wavelength_spacing(self):
        
        return np.quantile(np.diff(self.df.wavelength.values), 0.90)

    def apply_telluric_mask(self, regions=None, fill=1):
        '''
        Mask out telluric regions
        '''

        if regions is not None:
            if isinstance(regions, str):
                with open(regions, 'rb') as p:
                    regions = pickle.load(p)
        else:
            regions = spectrum_utils.create_telluric_mask()
            region_unit = regions[1]
            regions = regions[0]

        for group in regions:
            wmin = group[0] * region_unit
            wmax = group[1] * region_unit

            wmin = wmin.to(self.wavelength_unit).value
            wmax = wmax.to(self.wavelength_unit).value

            idx = np.where( (self.df.wavelength > wmin) & (self.df.wavelength < wmax) )[0]
            if len(idx):

                self.df.loc[idx, 'flux'] = fill

        return self


    def despike(self, nsigma=3, window=3, fill=1):
        '''
        Clip spikes in spectra from cosmic rays or stiching
        '''

        std = self.df.flux.std()

        idx = np.where(self.df.flux > self.df.flux.median() + nsigma*std)[0]

        idx_clip = []
        for i in idx:
            idx_clip.append(np.arange(len(self.df))[i-window:i+window+1])
        idx_clip = np.unique(np.concatenate(idx_clip))

        self.df.loc[idx_clip, 'flux'] = fill

        self.mask(idx_clip)

        return self

    def flux_filter(self, topflux):
        '''
        Interpolate over wavelengths where spike exceeds given flux threshold
        '''
            
        idx = np.where(self.df.flux < topflux)[0]

        if len(idx):
            wave_f = self.df.wavelength.copy().values[idx]
            flux_f = self.df.flux.copy().values[idx]

            self.df['flux'] = np.interp(self.df.wavelength, wave_f, flux_f)

        return self

    def broaden(self, vsini, **kwargs):
        '''
        Rotationally broaden spectrum
        '''

        self.vsini = vsini
        r = rotbroadint.rot_int_cmj(self.df.wavelength, self.df.flux, vsini, **kwargs)
        self.df['flux'] = r

        return self

    def fit_continuum(self, **kwargs):
        '''
        Use ispec to do continuum normalization
        '''

        with utils.HiddenPrints():
            #Create ispec spectrum
            wavelength = (self.df.wavelength.values*self.wavelength_unit).to('nm').value
            flux = self.df.flux.values
            err = self.df.err.values

            ispec_spectrum = np.array([
                    (x,y,z) for x,y,z in zip(wavelength, flux, err)],
                    dtype=[('waveobs', float), ('flux', float), ('err', float)])

            kwargs.setdefault('model',"Polynomy")
            kwargs.setdefault('degree',3)
            kwargs.setdefault('nknots',None)
            kwargs.setdefault('order','median+max')
            kwargs.setdefault('median_wave_range',0.05)
            kwargs.setdefault('max_wave_range',1.0)
            kwargs.setdefault('automatic_strong_line_detection', True)
            kwargs.setdefault('strong_line_probability', 0.5)
            kwargs.setdefault('use_errors_for_fitting', True)

            #Fit continuum
            star_continuum_model = ispec.fit_continuum(
                    ispec_spectrum, **kwargs)

            #Normalize Continuum
            ispec_spectrum = ispec.normalize_spectrum(
                   ispec_spectrum, star_continuum_model,
                   consider_continuum_errors=False)

            wavelength = ispec_spectrum['waveobs']
            flux = ispec_spectrum['flux']
            err = ispec_spectrum['err']

            self.df['wavelength'] = (wavelength*u.nm).to(self.wavelength_unit).value
            self.df['flux'] = flux
            self.df['err'] = err

    def add_continuum_region(self, regions):
       
        if isinstance(regions[0], (int, float)):
            regions = [regions]

        self.continuum_regions.extend(regions)

        self.continuum_regions.sort(key=lambda region: region[0])


        merged_regions = []
        for region in self.continuum_regions:
            if not merged_regions or merged_regions[-1][1] < region[0]:
                merged_regions.append(region)
            else:
                merged_regions[-1][1] = max(merged_regions[-1][1], region[1])
        self.continuum_regions = merged_regions

    def fit_continuum_regions(self, degree, plot=False, ax=None, savefig=None):

        wavelength = self.df.wavelength.values
        allflux = self.df.flux.values
        flux = self.df.flux.values

        if len(self.continuum_regions):
            idx = np.where( (self.df.wavelength > self.continuum_regions[0][0]) &
                            (self.df.wavelength < self.continuum_regions[0][1]) )[0]
            for i in range(1, len(self.continuum_regions)):
                idxi = np.where( (self.df.wavelength > self.continuum_regions[i][0]) &
                                 (self.df.wavelength < self.continuum_regions[i][1]) )[0]
                idx = np.concatenate([idx, idxi])
            wavelength = wavelength[idx]
            flux = flux[idx]

        popt = np.polyfit(wavelength, flux, 3)
        poly = np.poly1d(popt)
        continuum = poly(self.df.wavelength.values)

        self.df['flux'] = self.df.flux.values / continuum

        if plot or (ax is not None) or (savefig is not None):
           
            fig, ax, created_fig = plotutils.fig_init(ax=ax, figsize=(12, 6))

            ax.plot(self.df.wavelength, allflux, color='black')
            for r in self.continuum_regions:
                ax.axvspan(r[0], r[1], facecolor='xkcd:red', edgecolor='none', alpha=0.2)

            ax.plot(wavelength, flux, color='xkcd:azure')
            ax.plot(self.df.wavelength.values, continuum, color='xkcd:green', ls='--')

            ax.set_xlabel(f'Wavelength ({self.wavelength_unit:latex})', fontsize=20)
            if self.flux_unit == u.dimensionless_unscaled:
                ax.set_ylabel('Scaled Flux', fontsize=20)
            else:
                ax.set_ylabel(f'Flux ({self.flux_unit:latex})', fontsize=20)

            return plotutils.plt_return(created_fig, fig, ax, savefig)


    def plot(self, ax=None, savefig=None,
             plot_unit=None,
             offset=0,
             wavelength_range=None,
             plot_kwargs=None):

        '''
        Plot the spectrum
        '''

        fig, ax, created_fig = plotutils.fig_init(ax=ax)
        if created_fig:
            fig.subplots_adjust(top=.98, right=.98)

        if plot_kwargs is None:
            plot_kwargs = {}
        plot_kwargs.setdefault('color', 'black')
        plot_kwargs.setdefault('lw', 1)
        plot_kwargs.setdefault('alpha', 1.0)


        wavelength = self.df.wavelength.values
        flux = self.df.flux.values

        if plot_unit is not None:
            wavelength = wavelength * self.wavelength_unit
            wavelength = wavlength.to(plot_unit).value
        else:
            plot_unit = self.wavelength_unit

        if wavelength_range is not None:

            idx = np.where( (wavelength >= wavelength_range[0]) &
                            (wavelength <= wavelength_range[1]) )[0]

            if not len(idx):
                self.print(f'No data in wavelength range {wavelength_range}')

            wavelength = wavelength[idx]
            flux = flux[idx]

        ax.plot(wavelength, flux+offset, **plot_kwargs)

        ax.set_xlabel(f'Wavelength ({plot_unit:latex})', fontsize=20)
        if self.flux_unit == u.dimensionless_unscaled:
            ax.set_ylabel('Scaled Flux', fontsize=20)
        else:
            ax.set_ylabel(f'Flux ({self.flux_unit:latex})', fontsize=20)

        return plotutils.plt_return(created_fig, fig, ax, savefig)

    def plot_velocity(self, rest_wavelength, ax=None, savefig=None,
                      plot_unit=None, offset=0,
                      velocity_range=None,
                      plot_kwargs=None):
        
        '''
        Plot the spectrum in velocity space

        assume velocity range is in km/s
        '''

        fig, ax, created_fig = plotutils.fig_init(ax=ax)
        if created_fig:
            fig.subplots_adjust(top=.98, right=.98)

        if plot_kwargs is None:
            plot_kwargs = {}
        plot_kwargs.setdefault('color', 'black')
        plot_kwargs.setdefault('lw', 1)
        plot_kwargs.setdefault('alpha', 1.0)

        wavelength = self.df.wavelength.values * self.wavelength_unit
        flux = self.df.flux.values

        if not isinstance(rest_wavelength, u.quantity.Quantity):
            raise TypeError('line wavelength must be astropy quantity')

        vel = const.c * (wavelength - rest_wavelength) / rest_wavelength
        vel = vel.to('km/s').value

        if velocity_range is not None:
            idx = np.where( (vel >= velocity_range[0]) &
                            (vel <= velocity_range[1]) )[0]

            if not len(idx):
                self.print('No data in velocity range {velocity_range}')

            vel = vel[idx]
            flux = flux[idx]

        ax.plot(vel, flux+offset, **plot_kwargs)

        ax.set_xlabel(f'Velocity (km/s)', fontsize=20)
        if self.flux_unit == u.dimensionless_unscaled:
            ax.set_ylabel('Scaled Flux', fontsize=20)
        else:
            ax.set_ylabel(f'Flux ({self.flux_unit:latex})', fontsize=20)

        return plotutils.plt_return(created_fig, fig, ax, savefig)


    def output_ispec_format(self, outfile, err=None):
        '''
        Output the spectrum in ispec format
        '''

        out_df = self.df.copy()
        out_df['wavelength'] = (
                (out_df.wavelength.to_numpy() * self.wavelength_unit).to('nm').value)

        if err is not None:
            out_df['err'] = np.ones(len(out_df)) * err

        if outfile is not None:
            out_df[['wavelength', 'flux', 'err']].to_csv(
                    outfile,index=False,header=False,sep='\t')
        else:
            return out_df[['wavelength', 'flux', 'err']]

    def cross_correlation(self, template, 
                          lower_velocity_limit=-200, 
                          upper_velocity_limit=200,
                          velocity_step=0.5):
        '''
        Measure RVs with 1D-CCF
        '''

        #Create velocity grid
        shifts = np.arange(
                np.int32(np.floor(lower_velocity_limit)/velocity_step),
                np.int32(np.ceil(upper_velocity_limit)/velocity_step)+1)
        velocity = shifts * velocity_step

        wavelength_grid = measure_rvs.resample_spectrum_uniform_velocity_grid(
                (self.df.wavelength.min(), self.df.wavelength.max()),
                velocity_step)


        #depth = np.abs(template.df.flux.max() - template.df.flux).values
        depth = template.df.flux.values

        resampled_template = spectrum_utils.resample_spectrum(
                wavelength_grid, template.df.wavelength.values, depth,
                simple=True)

        resampled_template = 1 - resampled_template

        sg = np.sqrt(np.average(np.power(resampled_template, 2)))

        num_shifts = len(shifts)

            
        flux = spectrum_utils.resample_spectrum(
                wavelength_grid,
                self.df.wavelength.values,
                self.df.flux.values,
                err=self.df.err.values,
                simple=True)
            
        flux = 1 - flux 

        sf = np.sqrt(np.average(np.power(flux, 2)))

        ccf = np.zeros(num_shifts)

        for i, shift in enumerate(shifts):
            
            if shift == 0:
                shifted_template = resampled_template
            else:
                shifted_template = np.hstack((resampled_template[-1*shift:],
                              resampled_template[:-1*shift]))
            ccf[i] = np.nanmean(flux*shifted_template)

        df_ccf = pd.DataFrame({'RV':velocity, 'CCF':ccf/(sf*sg)})

        self.ccf = profiles.CCFProfile(df_ccf, nbins=len(flux))

    def todcor(self, template1, template2, 
               lower_velocity_limit, upper_velocity_limit,
               velocity_step=0.5):

        self.todcor_profile = measure_rvs.todcor(
                self, template1, template2,
                lower_velocity_limit, upper_velocity_limit,
                velocity_step, progress=self.verbose)

    
    def rv_pipeline(self, template1, template2, window=30, alpha_tolerance=0,
                    lower_velocity_limit=None, upper_velocity_limit=None):
        '''
        Combine 1D CCF with TODCOR to measure RVs
        '''

        self.cross_correlation(template1)
        self.ccf.model()

        if lower_velocity_limit is None:
            if len(self.ccf.models) == 2:
                lower_velocity_limit = [self.ccf.models[0].mu - window//2,
                                        self.ccf.models[1].mu - window//2]
                upper_velocity_limit = [l+window for l in lower_velocity_limit ]
            else:
                lower_velocity_limit = self.ccf.models[0].mu - window
                upper_velocity_limit = lower_velocity_limit + int(2*window)

        self.todcor(template1, template2, lower_velocity_limit, upper_velocity_limit)
        #Param to allow for alpha < 1 + alpha_tolerance 
        #Most applicable to todmor profiles
        self.todcor_profile.alpha_tolerance = alpha_tolerance

        self.todcor_profile.get_rvs_nm()
        self.slice1 = self.todcor_profile.get_slice(1, method='nm')
        self.slice2 = self.todcor_profile.get_slice(2, method='nm')

    def rv_pipeline_plot(self, savefig=None):
        '''
        Plot to view RV pipeline results
        '''

        fig, ax = plt.subplots(3, 2, figsize=(10, 12))
        fig.subplots_adjust(top=.98, right=.98)

        ax_ccf = ax[0,0]
        ax[0,1].axis('off')
        ax_todcor = ax[1,0]
        ax_alpha = ax[1,1]
        ax_slice1 = ax[2,0]
        ax_slice2 = ax[2,1]

        ax_ccf = self.ccf.plot_model(ax=ax_ccf)

        ax_todcor = self.todcor_profile.plot(ax=ax_todcor)

        ax_alpha = self.todcor_profile.alpha_profile.plot(ax=ax_alpha, vmin=0, vmax=1)

        ax_slice1 = self.slice1.plot(ax=ax_slice1)
        ax_slice2 = self.slice2.plot(ax=ax_slice2)

        ax_slice1.axvline(self.slice1.rv, color='xkcd:red')
        ax_slice1.axvline(self.slice1.rv - self.slice1.rv_err, color='xkcd:red', ls='--')
        ax_slice1.axvline(self.slice1.rv + self.slice1.rv_err, color='xkcd:red', ls='--')

        ax_slice2.axvline(self.slice2.rv, color='xkcd:red')
        ax_slice2.axvline(self.slice2.rv - self.slice2.rv_err, color='xkcd:red', ls='--')
        ax_slice2.axvline(self.slice2.rv + self.slice2.rv_err, color='xkcd:red', ls='--')

        if len(self.ccf.models) == 2:
            ax_slice1.axvline(self.ccf.models[0].mu, color='gray', ls='-')
            ax_slice1.axvline(self.ccf.models[0].mu - self.ccf.models[0].emu,
                             ls='--', color='gray')
            ax_slice1.axvline(self.ccf.models[0].mu + self.ccf.models[0].emu,
                             ls='--', color='gray')
            ax_slice2.axvline(self.ccf.models[1].mu, color='gray', ls='-')
            ax_slice2.axvline(self.ccf.models[1].mu - self.ccf.models[1].emu,
                             ls='--', color='gray')
            ax_slice2.axvline(self.ccf.models[1].mu + self.ccf.models[1].emu,
                             ls='--', color='gray')

        if savefig is not None:
            fig.savefig(savefig)
        else:
            plt.show()

    def estimate_params(self, shift_refs=None):
        
        with utils.HiddenPrints(self.verbose):

            wavelength_range = self.get_wavelength_range()
            wavlim = [wavelength_range[0]+5, wavelength_range[1]-5]
            
            if wavlim[0] < 4991:
                wavlim[0] = 4991

            if wavlim[1] > 6409:
                wavlim[1] = 6409

            if (wavlim[1] < wavlim[0]):
                self.sm = None
            else:

                try:
                    lib = specmatchemp.library.read_hdf(wavlim=wavlim)

                    flux = self.df.flux.values
                    wavelength = (self.df.wavelength.values * self.wavelength_unit).to('AA').value

                    data_spectrum = spectrum.Spectrum(wavelength, flux).cut(wavlim[0]-5, wavlim[1]+5)
                    data_spectrum.name = 'BaseSpectrum'
                    sm = SpecMatch(data_spectrum, lib, wavlim=wavlim)

                    sm.shift(shift_refs=shift_refs)
                    sm.match(wavlim=tuple(wavlim))
                    sm.lincomb()
                    self.sm = sm
                except Exception as e:
                    if isinstance(e, ValueError) and "NaN values detected in your input data" in str(e):
                        self.print(f'Skipping SM {self.n} due to NaN error')
                        self.sm = None
                    elif isinstance(e, ValueError) and "operands could not be broadcast together" in str(e):
                        self.print(f'Skipping SM {self.n} due to shape error')
                        self.sm = None
                    else:
                        raise


        return self.sm

    def plot_sm(self, ax=None, savefig=None):
        
        fig, ax, created_fig = plotutils.fig_init(ax=ax, figsize=(10, 6))
        if created_fig:
            fig.subplots_adjust(top=.98, right=.98)

        label = (r'Model: '+
                 r'$T_{\rm{eff}} = '+str(round(self.sm.results['Teff']))+r'$ K, '+
                 r'$\log g = {}$, '.format(round(self.sm.results['logg'], 2))+
                 r'[Fe/H] $= {}$'.format(round(self.sm.results['feh'], 2)))
        mt = self.sm.lincomb_matches[0]

        data_label = self.name
        if data_label is None:
            data_label = 'Data'
        ax.plot(mt.target.w, mt.target.s, color='black', lw=1, label=data_label)
        ax.plot(mt.modified.w, mt.modified.s, color='xkcd:red', lw=2, alpha=0.6, label=label)

        ax.legend(loc='lower left', edgecolor='black', fontsize=15)
        ax.set_xlabel(r'Wavelength, $(\mathring{A})$', fontsize=20)
        ax.set_ylabel('Normalized Flux', fontsize=20)

        return plotutils.plt_return(created_fig, fig, ax, savefig)

    def to_pickle(self, outfile):
        
        with open(outfile, 'wb') as p:
            pickle.dump(self, p)


class EchelleOrder(BaseSpectrum):
    '''
    Class for a single spectrum Echelle Order
    '''
    
    def __init__(self, n, wavelength, flux, **kwargs):
       
        self.n = n
        super().__init__(wavelength, flux, **kwargs)

    def filter_cosmic_rays(self, quantile=0.98, minimum_threshold=2):
        '''
        Remove large spikes from cosmic rays
        '''

        score = np.concatenate([[0],spectrum_utils.modified_z_score(np.diff(self.df.flux))])
        threshold = np.quantile(score, quantile)

        threshold = np.min([threshold, minimum_threshold])

        idx = np.where(score < threshold)[0]

        #If there is a floating index, remove
        #This is to try to get rid of double-cosmic rays
        idx_cleaned = []
        for j in range(1, len(idx)-1):
            i = idx[j]
            im = idx[j-1]
            ip = idx[j+1]

            if not ((ip - i > 1) and (i - im > 1)):
                idx_cleaned.append(idx[j])

        wave_crr = self.df.wavelength.copy()[idx_cleaned]
        flux_crr = self.df.flux.copy()[idx_cleaned]

        self.df['flux'] = np.interp(self.df.wavelength, wave_crr, flux_crr)

        return self

    def deblaze(self, window=100, percentile=0.95, degree=9,
                plot=False, ax=None, savefig=None):
        '''
        Deblaze spectrum by fitting legendre polynomial
        '''

        pfilter = spectrum_utils.quantile_filter(self.df.flux.values, window, percentile)

        pfilter = pfilter[window:-1*window]
        wave = self.df.wavelength.values[window:-1*window]
        flux = self.df.flux.values[window:-1*window]
        flux_err = self.df.err.values[window:-1*window]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            coeffs = np.polynomial.legendre.legfit(wave, pfilter, degree)
            legfit = np.polynomial.legendre.legval(wave, coeffs)

        flux = flux / legfit

        if plot:
            fig, ax, created_fig = plotutils.fig_init(ax=ax, figsize=(8, 6))
            ax.plot(self.df.wavelength, self.df.flux, color='black', alpha=0.6)
            ax.plot(wave, pfilter, color='xkcd:red', alpha=0.7)
            ax.plot(wave, legfit, color='xkcd:blue', alpha=0.7)

        self.df = pd.DataFrame({'wavelength':wave,
                                'flux':flux,
                                'err':flux_err})
        
        return self

class CHIRON_EchelleOrder(EchelleOrder):
    '''
    Class for CHIRON echelle order
    '''

    def deblaze(self, window=40, percentile=0.95, degree=4,
                plot=False, ax=None, savefig=None):
        '''
        CHIRON echelle order deblazing
        Uses regular polynomail degree 4 by default
        '''

        pfilter = spectrum_utils.quantile_filter(self.df.flux.values, window, percentile)

        pfilter = pfilter[window:-1*window]
        wave = self.df.wavelength.values[window:-1*window]
        flux = self.df.flux.values[window:-1*window]
        flux_err = self.df.err.values[window:-1*window]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            coeffs = np.polynomial.polynomial.polyfit(wave, pfilter, degree)
            polyfit = np.polynomial.polynomial.polyval(wave, coeffs)

        flux = flux / polyfit

        if plot:
            fig, ax, created_fig = plotutils.fig_init(ax=ax, figsize=(8, 6))
            ax.plot(self.df.wavelength, self.df.flux, color='black', alpha=0.6)
            ax.plot(wave, pfilter, color='xkcd:red', alpha=0.7)
            ax.plot(wave, polyfit, color='xkcd:azure', alpha=0.7)

        self.df = pd.DataFrame({'wavelength':wave,
                                'flux':flux,
                                'err':flux_err})
        
        return self

class EchelleSpectrum:

    '''
    Class for EchelleSpectrum with multiple orders
    '''
    
    def __init__(self, wavelength, flux, 
                 wavelength_unit=u.AA,
                 name=None,
                 flux_unit=u.dimensionless_unscaled,
                 verbose=True,
                 header=None):
                
        self.wavelength = wavelength
        self.flux = flux

        self.verbose = verbose
        self.wavelength_unit = wavelength_unit
        self.flux_unit = flux_unit
        self.header = header
        self.name = name

        self.fdbinary_mask = False

    def get_flux_arr(self):
        '''
        Make single flux array from all the orders
        '''
        
        list_of_fluxes = []
        for Order in self.Orders:
            list_of_fluxes.append(Order.df.flux.values)

        return np.concatenate(list_of_fluxes)

    def get_wavelength_arr(self):
        '''
        Make single array of wavelength out of all the orders
        '''
        list_of_wavelengths = []
        for Order in self.Orders:
            list_of_wavelengths.append(Order.df.wavelength.values)

        return np.concatenate(list_of_wavelengths)

    @property
    def verbose(self):
        return self._verbose
    @verbose.setter
    def verbose(self, value):
        self._verbose = bool(value)
        if hasattr(self, "Orders"):
            for Order in self.Orders:
                Order.verbose = value

    @property
    def name(self):
        return self._name
    @name.setter
    def name(self, value):
        self._name = value
        if hasattr(self, "Orders"):
            for Order in self.Orders:
                Order.name = value

    @property
    def fdbinary_mask(self):
        return self._fdbinary_mask
    @fdbinary_mask.setter
    def fdbinary_mask(self, value):
        self._fdbinary_mask = value
        if hasattr(self, "Orders"):
            for Order in self.Orders:
                Order.fdbinary_mask = value

    def print(self, value):
        if self.verbose:
            print(value)

    def reduce(self, 
               cr_quantile=None,
               deblaze_window=None, deblaze_percentile=None,
               deblaze_degree=None, trim=None, flux_filter=None,
               reduce=True):

        '''
        Reduce echelle spectrum for all orders
        '''

        if cr_quantile is None:
            cr_quantile = self.cr_quantile
        if deblaze_window is None:
            deblaze_window = self.deblaze_window
        if deblaze_percentile is None:
            deblaze_percentile = self.deblaze_percentile
        if deblaze_degree is None:
            deblaze_degree = self.deblaze_degree
        if trim is None:
            trim = self.trim
        if flux_filter is None:
            flux_filter = self.flux_filter
       
        if self.verbose:
            iterator = tqdm(range(self.flux.shape[0]))
        else:
            iterator = range(self.flux.shape[0])

        self.Orders = []
        for i in iterator:

            flux = self.flux[i,:].byteswap().newbyteorder()
            wave = self.wavelength[i,:].byteswap().newbyteorder()

            eo = EchelleOrder(i, wave, flux)
            eo.df['wavelength'] = eo.df.wavelength.astype(float)
            eo.df['flux'] = eo.df.flux.astype(float)
            if reduce:
                eo.filter_cosmic_rays(quantile=cr_quantile)
                eo.deblaze(window=deblaze_window, percentile=deblaze_percentile,
                           degree=deblaze_degree)
                eo.trim_edges(trim)
                eo.flux_filter(flux_filter)

            self.Orders.append(eo)

        self.verbose = self.verbose #make sure all orders are set with same level

    def get_barycentric_velocity(self):
        '''
        Use barycorrpy to determine the barycentric velocity
        '''

        vb, warnings, flag = get_BC_vel(JDUTC=self.JD, ra=self.RA, dec=self.DEC,
                                        obsname=self.ObsName, predictive=True)
        self.vb = (float(vb)/1000) * u.km/u.s

        return self.vb

    @property
    def barycentric_corrected(self):
        '''
        Flag to determine if spectra had already been barycentric corrected
        '''
        return self._barycentric_corrected

    @barycentric_corrected.setter
    def barycentric_corrected(self, value):
        
        self._barycentric_corrected = bool(value)
        for Order in self.Orders:
            Order._barycentric_corrected = bool(value)

    def correct_barycentric_velocity(self):
        '''
        Correct spectra for barycentric motion 
        '''

        if not self.barycentric_corrected:

            self.get_barycentric_velocity()

            for Order in self.Orders:
                Order.vb = self.vb
                Order.correct_barycentric_velocity()

            self._barycentric_corrected = True

    def filter_wavelength_range(self, *args):
        '''
        Select wavelength range. Drop orders where no data in specified wavelength range
        '''

        orders_to_drop = []
        for Order in self.Orders:
            idx = np.where( (Order.df.wavelength > args[0]) & 
                            (Order.df.wavelength < args[1]) )[0]
            if len(idx):
                Order.df = Order.df.iloc[idx].reset_index(drop=True)
            else:
                orders_to_drop.append(Order.n)

        if len(orders_to_drop):
            self.drop_orders(orders_to_drop)

        return self

    def get_wavelength_range(self):
        
        return self.Orders[0].df.wavelength.min(), self.Orders[-1].df.wavelength.max()

    def get_wavelength_spacing(self):
        
        spacing = []
        for Order in self.Orders:
            if len(Order.df > 10):
                spacing.append(Order.get_wavelength_spacing())

        return np.max(spacing)

    def resample_log_wavelength(self, **kwargs):
        '''
        Resample spectrum over uniform grid in log-wavelength
        '''
        
        for Order in self.Orders:
             
             wavelength_spacing = Order.get_wavelength_spacing()

             Order.resample_log_wavelength(
                    *Order.get_wavelength_range(), wavelength_spacing,
                    **kwargs)
        

    def drop_orders(self, orders_to_drop):

        self.Orders = [ O for O in self.Orders if O.n not in orders_to_drop ]

    def select_orders(self, selected_orders):

        self.Orders = [ O for O in self.Orders if O.n in selected_orders ]

    def get_order(self, n):
        '''
        Return selected order 
        '''
        
        for Order in self.Orders:
            if Order.n == n:
                return Order

        self.print(f"Order {n} not found")

    def apply_telluric_mask(self, **kwargs):
        
        for Order in self.Orders:
            Order.apply_telluric_max(**kwargs)

        return self

    def despike(self, **kwargs):
        
        for Order in self.Orders:
            Order.despike(**kwargs)
        
        return self

    def flux_filter(self, **kwargs):

        for Order in self.Orders:
            Order.flux_filter(**kwargs)

        return self

    def shift_rv(self, rv):
        
        for Order in self.Orders:
            Order.shift_rv(rv)

        return self

    def plot(self, ax=None, savefig=None, alternating_colors=True, plot_kwargs=None, **kwargs):
        
        fig, ax, created_fig = plotutils.fig_init(ax=ax)
        if created_fig:
            fig.subplots_adjust(top=.98, right=.98)


        if plot_kwargs is None:
            plot_kwargs = {}
        plot_kwargs.setdefault('color', 'black')
        plot_kwargs.setdefault('alpha', 0.7)
        plot_kwargs.setdefault('lw', 1)

        for Order in self.Orders:
            ax = Order.plot(ax=ax, plot_kwargs=plot_kwargs, **kwargs)
            if alternating_colors:
                if plot_kwargs['color'] == 'black':
                    plot_kwargs['color'] = 'xkcd:red'
                else:
                    plot_kwargs['color'] = 'black'

        return plotutils.plt_return(created_fig, fig, ax, savefig)

    def cross_correlation(self, template, 
                          lower_velocity_limit=-200, 
                          upper_velocity_limit=200,
                          velocity_step=0.5):

        #Create velocity grid
        shifts = np.arange(
                np.int32(np.floor(lower_velocity_limit)/velocity_step),
                np.int32(np.ceil(upper_velocity_limit)/velocity_step)+1)
        num_shifts = len(shifts)
        velocity = shifts * velocity_step

        profile_list = []
        for Order in self.Orders:

            wavelength_grid = measure_rvs.resample_spectrum_uniform_velocity_grid(
                    (Order.df.wavelength.min(), Order.df.wavelength.max()),
                    velocity_step)

            if not hasattr(template, 'Orders'):
                depth = template.df.flux.values
                wavelength_template = template.df.wavelength.values
            else:
                template_order = template.get_order(Order.n)
                #Do a quick check for wavelength range
                if ((template_order.df.wavelength.max() < Order.df.wavelength.min()) or
                    (template_order.df.wavelength.min() > Order.df.wavelength.max())):
                    raise ValueError('Wavelength range of template order does not match spec order')

                depth = template_order.df.flux.values
                wavelength_template = template_order.df.wavelength.values

            resampled_template = spectrum_utils.resample_spectrum(
                    wavelength_grid, wavelength_template, depth,
                    simple=True)

            resampled_template = 1 - resampled_template

            sg = np.sqrt(np.average(np.power(resampled_template, 2)))
            
            flux = spectrum_utils.resample_spectrum(
                    wavelength_grid,
                    Order.df.wavelength.values,
                    Order.df.flux.values,
                    err=Order.df.err.values,
                    simple=True)
            
            flux = 1 - flux 
            sf = np.sqrt(np.average(np.power(flux, 2)))
            ccf = np.zeros(num_shifts)

            for i, shift in enumerate(shifts):
                
                if shift == 0:
                    shifted_template = resampled_template
                else:
                    shifted_template = np.hstack((resampled_template[-1*shift:],
                                  resampled_template[:-1*shift]))

                ccf[i] = np.average(flux*shifted_template)

            df_ccf = pd.DataFrame({'RV':velocity, 'CCF':ccf/(sf*sg)})

            Order.ccf = profiles.BaseProfile(df_ccf, nbins=len(flux))

        #Combine the profiles
        M = len(self.Orders)

        if M > 1:

            C0 = self.Orders[0].ccf.df.CCF.values
            C1 = self.Orders[1].ccf.df.CCF.values

            x0 = 1 - np.power(C0, 2)
            x1 = 1 - np.power(C1, 2)

            product = np.multiply(x0, x1)

            i = 2

            while i < M:

                Ci = self.Orders[i].ccf.df.CCF.values
                xi = 1 - np.power(Ci, 2)

                product = np.multiply(product, xi)

                i += 1

            ML2 = 1 - np.power(product, (1/M) )

        else:
            raise NotImplementedError

        #Remove bad values
        idx = np.where(ML2 < 1.0)[0]

        selected_velocities = velocity[idx]
        ML_vals = np.sqrt(ML2[idx])

        df_ml = pd.DataFrame({'RV':velocity,
                              'ML':np.interp(velocity, selected_velocities, ML_vals)})
        df_ml['err'] = 1e-6

        self.ccf = profiles.CCFProfile(
                df_ml, 
                nbins=np.mean([O.ccf.nbins for O in self.Orders]), 
                norders=M)

    def todcor(self, template1, template2, 
               lower_velocity_limit=None,
               upper_velocity_limit=None,
               window=30, velocity_step=0.5):

        if lower_velocity_limit is None:
            if len(self.ccf.models) > 2:
                self.print('More than 2 components identified in 1D ccf')
                self.print('Using two highest amplitude to set todcor window')

                lower_velocity_limit = [self.ccf.models[0].mu - window//2,
                                        self.ccf.models[1].mu - window//2]
                upper_velocity_limit = [l+window for l in lower_velocity_limit ]

            elif len(self.ccf.models) == 2:
                lower_velocity_limit = [self.ccf.models[0].mu - window//2,
                                        self.ccf.models[1].mu - window//2]
                upper_velocity_limit = [l+window for l in lower_velocity_limit ]
            else:
                lower_velocity_limit = self.ccf.models[0].mu - window
                upper_velocity_limit = lower_velocity_limit + int(2*window)

        self.todcor_profile = measure_rvs.todmor(
                self, template1, template2,
                lower_velocity_limit,
                upper_velocity_limit,
                velocity_step,
                progress=self.verbose)

        return self.todcor_profile

    def rv_pipeline(self, template1, template2, alpha_tolerance=0, 
                    lower_velocity_limit=None,
                    upper_velocity_limit=None):

        self.cross_correlation(template1)
        self.ccf.model()
        self.todcor(template1, template2, 
                    lower_velocity_limit=lower_velocity_limit,
                    upper_velocity_limit=upper_velocity_limit)

        #Param to allow for alpha < 1 + alpha_tolerance 
        #Most applicable to todmor profiles
        self.todcor_profile.alpha_tolerance = alpha_tolerance

        self.todcor_profile.get_rvs_nm()
        self.slice1 = self.todcor_profile.get_slice(1, method='nm')
        self.slice2 = self.todcor_profile.get_slice(2, method='nm')

    def rv_pipeline_plot(self, savefig=None):
        '''
        Plot to view RV pipeline results
        '''

        fig, ax = plt.subplots(3, 2, figsize=(10, 12))
        fig.subplots_adjust(top=.98, right=.98)

        ax_ccf = ax[0,0]
        ax[0,1].axis('off')
        ax_todcor = ax[1,0]
        ax_alpha = ax[1,1]
        ax_slice1 = ax[2,0]
        ax_slice2 = ax[2,1]

        ax_ccf = self.ccf.plot_model(ax=ax_ccf)

        ax_todcor = self.todcor_profile.plot(ax=ax_todcor)

        ax_alpha = self.todcor_profile.alpha_profile.plot(ax=ax_alpha, vmin=0, vmax=1)

        ax_slice1 = self.slice1.plot(ax=ax_slice1)
        ax_slice2 = self.slice2.plot(ax=ax_slice2)

        ax_slice1.axvline(self.slice1.rv, color='xkcd:red')
        ax_slice1.axvline(self.slice1.rv - self.slice1.rv_err, color='xkcd:red', ls='--')
        ax_slice1.axvline(self.slice1.rv + self.slice1.rv_err, color='xkcd:red', ls='--')

        ax_slice2.axvline(self.slice2.rv, color='xkcd:red')
        ax_slice2.axvline(self.slice2.rv - self.slice2.rv_err, color='xkcd:red', ls='--')
        ax_slice2.axvline(self.slice2.rv + self.slice2.rv_err, color='xkcd:red', ls='--')

        if len(self.ccf.models) == 2:
            ax_slice1.axvline(self.ccf.models[0].mu, color='gray', ls='-')
            ax_slice1.axvline(self.ccf.models[0].mu - self.ccf.models[0].emu,
                             ls='--', color='gray')
            ax_slice1.axvline(self.ccf.models[0].mu + self.ccf.models[0].emu,
                             ls='--', color='gray')
            ax_slice2.axvline(self.ccf.models[1].mu, color='gray', ls='-')
            ax_slice2.axvline(self.ccf.models[1].mu - self.ccf.models[1].emu,
                             ls='--', color='gray')
            ax_slice2.axvline(self.ccf.models[1].mu + self.ccf.models[1].emu,
                             ls='--', color='gray')

        if savefig is not None:
            fig.savefig(savefig)
        else:
            plt.show()

    def select_rv_orders(self, N, method='ccf'):

        if method == 'ccf':
            return self._select_rv_orders_ccf(N)
        elif method == 'ml':
            return self._select_rv_orders_ml(N)
        elif method == 'todcor':
            return self._select_rv_orders_todcor(N)
        elif method == 'todcor_ml':
            return self._select_rv_orders_todcor_ml(N)
        else:
            raise ValueError(f'method {method} not recognized')

    def _select_rv_orders_ccf(self, N):
        
        #Get the velocities from todcor
        rvs = self.todcor_profile.get_rvs()

        R1 = []
        R2 = []
        #Go through orders
        for Order in self.Orders:
            r1 = Order.ccf.interp(rvs[0])
            r2 = Order.ccf.interp(rvs[1])
            R1.append(r1)
            R2.append(r2)

        df = pd.DataFrame({
            'order': [O.n for O in self.Orders],
            'R1': R1,
            'R2': R2})

        df_R1 = df.copy().sort_values(by='R1', ascending=False).reset_index(drop=True)
        df_R2 = df.copy().sort_values(by='R2', ascending=False).reset_index(drop=True)

        orders = np.unique(np.concatenate([
                df_R1.order.iloc[:N].values,
                df_R2.order.iloc[:N].values]))

        return np.sort(orders)

    def _select_rv_orders_todcor(self, N):
        
        idx = self.todcor_profile.get_rvs(return_idx=True)

        R = []
        for Order in self.Orders:
            R.append(Order.todcor_profile.arr[idx])

        df = pd.DataFrame({
                'order': [ O.n for O in self.Orders],
                'R': R})
        df = df.sort_values(by='R', ascending=False).reset_index(drop=True)

        return np.sort(df.order.iloc[:N].values)

    def _select_rv_orders_ml(self, N):

        rvs = self.todcor_profile.get_rvs()

        ML_drop_rv1 = []
        ML_drop_rv2 = []
        for i in range(len(self.Orders)):
            
            used_orders = [ O for O in self.Orders if O.n != self.Orders[i].n ]
            M = len(used_orders)

            C0 = used_orders[0].ccf.df.CCF.values
            C1 = used_orders[1].ccf.df.CCF.values

            x0 = 1 - np.power(C0, 2)
            x1 = 1 - np.power(C1, 2)

            product = np.multiply(x0, x1)

            i = 2

            while i < M:

                Ci = used_orders[i].ccf.df.CCF.values
                xi = 1 - np.power(Ci, 2)

                product = np.multiply(product, xi)

                i += 1

            ML2 = 1 - np.power(product, (1/M) )
            ML = np.sqrt(ML2)

            df_ml = pd.DataFrame({'RV':used_orders[0].ccf.df.RV.values,
                                  'ML':ML})
            ml_profile = profiles.CCFProfile(
                    df_ml, nbins=used_orders[0].ccf.nbins, norders=M)

            ML_drop_rv1.append(ml_profile.interp(rvs[0]))
            ML_drop_rv2.append(ml_profile.interp(rvs[1]))

        df = pd.DataFrame({
            'order': [O.n for O in self.Orders],
            'ML_drop_rv1': ML_drop_rv1,
            'ML_drop_rv2': ML_drop_rv2})

        df_rv1 = df.copy().sort_values(by='ML_drop_rv1', ascending=True)
        df_rv2 = df.copy().sort_values(by='ML_drop_rv2', ascending=True)

        df_rv1.reset_index(drop=True)
        df_rv2.reset_index(drop=True)

        orders = np.unique(np.concatenate([
                df_rv1.order.iloc[:N].values,
                df_rv2.order.iloc[:N].values]))

        return np.sort(orders)

    def _select_rv_orders_todcor_ml(self, N):

        ML_drop = []
        for i in range(len(self.Orders)):
            
            used_orders = [ O for O in self.Orders if O.n != self.Orders[i].n ]
            M = len(used_orders)
            
            T0 = used_orders[0].todcor_profile.arr
            T1 = used_orders[1].todcor_profile.arr
            
            x0 = 1 - np.power(T0, 2)
            x1 = 1 - np.power(T1, 2)

            product = np.multiply(x0, x1)

            i = 2
            while i < M:
                Ti = used_orders[i].todcor_profile.arr
                xi = 1 - np.power(Ti, 2)
                product = np.multiply(product, xi)

                i += 1

            ML2 = 1 - np.power(product, (1/M))
            ML = np.sqrt(ML2)
            
            ML_drop.append(ML[self.todcor_profile.get_rvs(return_idx=True)])
            
        df = pd.DataFrame({'order': [o.n for o in self.Orders]})
        df['ML_drop'] = ML_drop
        df = df.sort_values(by='ML_drop', ascending=True).reset_index(drop=True)
        orders = np.sort(df.order.iloc[:N].values)
        return orders

    def sm_params(self):

        teff_vals = []
        teff_errs = []
        logg_vals = []
        logg_errs = []
        feh_vals = []
        feh_errs = []

        for Order in self.Orders:
            if hasattr(Order, 'sm') and (Order.sm is not None):
                
                teff_vals.append(Order.sm.results['Teff'])
                teff_errs.append(Order.sm.results['u_Teff'])
                logg_vals.append(Order.sm.results['logg'])
                logg_errs.append(Order.sm.results['u_logg'])
                feh_vals.append(Order.sm.results['feh'])
                feh_errs.append(Order.sm.results['u_feh'])

        #Just take the median values
        self.sm_teff = np.median(teff_vals)
        self.sm_teff_err = np.std(teff_vals)
        self.sm_logg = np.median(logg_vals)
        self.sm_logg_err = np.std(logg_vals)
        self.sm_feh = np.median(feh_vals)
        self.sm_feh_err = np.std(feh_vals)

        return {'Teff': [self.sm_teff, self.sm_teff_err],
                'logg': [self.sm_logg, self.sm_logg_err],
                'feh': [self.sm_feh, self.sm_feh_err]}

    def sm_results_plot(self, param1, param2, ax=None, savefig=None):
        
        fig, ax, created_fig = plotutils.fig_init(ax=ax, figsize=(6, 6))
        if created_fig:
            fig.subplots_adjust(top=.98, right=.98)

        mean_wavelengths = []
        param1_vals = []
        param1_errs = []
        param2_vals = []
        param2_errs = []
        for Order in self.Orders:
            
            if hasattr(Order, 'sm') and (Order.sm is not None):
                
                param1_val = Order.sm.results[param1]
                param1_err = Order.sm.results['u_'+param1]

                param2_val = Order.sm.results[param2]
                param2_err = Order.sm.results['u_'+param2]

                param1_vals.append(param1_val)
                param2_vals.append(param2_val)
                param1_errs.append(param1_err)
                param2_errs.append(param2_err)

                mean_wavelengths.append(np.mean(Order.get_wavelength_range()))

        labels = {'Teff': r'$T_{\rm{eff}}$ (K)',
                  'logg': r'$\log g$',
                  'feh': r'[Fe/H]'}

        final_values = self.sm_params()

        ax.errorbar(param1_vals, param2_vals, 
                    xerr=param1_errs, yerr=param2_errs, 
                    zorder=2,alpha=0.7,
                    ls='', lw=2, capsize=3, color='black', ecolor='gray')

        sc = ax.scatter(param1_vals, param2_vals, s=90,
                        c=mean_wavelengths, marker='o', alpha=0.9, cmap=cmr.cosmic,
                        zorder=3, vmin=min(mean_wavelengths)-100)
        cbar = plt.colorbar(sc, ax=ax)

        cbar = plotutils.plotparams_cbar(cbar)
        cbar.set_label(r'Wavelength $(\mathring{A})$', fontsize=20)

        ax.errorbar(final_values[param1][0], final_values[param2][0],
                    xerr=final_values[param1][1], yerr=final_values[param2][1],
                    color='xkcd:red', marker='s', markersize=10, zorder=4, alpha=0.9,
                    ls='', lw=2, capsize=3, ecolor='xkcd:red')
        
        ax.set_xlabel(labels[param1], fontsize=20)
        ax.set_ylabel(labels[param2], fontsize=20)

        return plotutils.plt_return(created_fig, fig, ax, savefig)


    def output_ispec_format(self, outfile, orders=None, err=0):
        '''
        Output the spectrum in ispec format
        '''

        if orders is None:
            wavelength = (self.get_wavelength_arr()*self.wavelength_unit).to('nm').value
            flux = self.get_flux_arr()
            errs = np.ones(len(flux))*err

            out_df = pd.DataFrame({'wavelength':wavelength, 'flux':flux, 'err':errs})
            out_df[['wavelength', 'flux', 'err']].to_csv(
                    outfile,index=False,header=False,sep='\t')
        else:
            out_df = pd.concat( [self.get_order(order).output_ispec_format(None, err=err)
                                 for order in orders] )
            out_df[['wavelength', 'flux', 'err']].to_csv(
                    outfile, index=False, header=False, sep='\t')
