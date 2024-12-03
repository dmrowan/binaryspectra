#!/usr/bin/env python

from astropy import units as u
from astropy.io import fits
from astropy.io import votable
from astropy.time import Time
from astropy.coordinates import SkyCoord
from astropy.modeling import models
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
from specutils.manipulation import FluxConservingResampler
from specutils import Spectrum1D
from specutils.fitting import fit_generic_continuum
from tqdm import tqdm
import warnings

from binaryspec.base_spectrum import *
from binaryspec import utils, spectrum_utils, plotutils

#Dom Rowan 2024

class PEPSIspec(BaseSpectrum):

    def __init__(self, fname, verbose=True):
        
        if utils.check_iter(fname):
            if len(fname) != 2:
                raise ValueError(f'invalid PEPSI format: {fname}')
            spec0 = PEPSIspec(fname[0]) 
            spec1 = PEPSIspec(fname[1])

            self.fname = fname

            self.df = pd.concat([spec0.df, spec1.df])
            self.df = self.df.sort_values(by='wavelength', ascending=True)
            self.df = self.df.reset_index(drop=True)

            self.header = spec0.header
            self.alt_header = spec1.header

        else:
            hdul = fits.open(fname)

            self.fname = fname

            df = pd.DataFrame.from_records(hdul[1].data)
            df.columns = ['wavelength', 'flux', 'err', 'mask']

            df['wavelength'] = df.wavelength.to_numpy().byteswap().newbyteorder()
            df['flux'] = df.flux.to_numpy().byteswap().newbyteorder()
            df['err'] = df.err.to_numpy().byteswap().newbyteorder()
            df['mask'] = df['mask'].to_numpy().byteswap().newbyteorder()

            df = df.drop(columns=['mask'])

            self.df = df

            self.header = {}
            for k in list(hdul[0].header.keys()):
                if k != '':
                    self.header[k] = hdul[0].header[k]

        self.verbose = verbose
        self.wavelength_unit = u.AA
        self.flux_unit = u.dimensionless_unscaled

        coord = SkyCoord(self.header['RA2000']+' '+self.header['DE2000'],
                         unit=(u.hourangle, u.deg))
        self.RA = coord.ra.deg
        self.DEC = coord.dec.deg
        JD = self.header['JD-OBS'] #utc
        self.ObsName = 'Mount Graham International Observatory'
        self._barycentric_corrected=True

        self.JD = utils.convert_jd_bjd(JD, self.RA, self.DEC, self.ObsName)

        self.fdbinary_mask = False

    def apply_telluric_mask(self):
        
        #First make upper cut
        if self.df.wavelength.max() > 6860:
            self.filter_wavelength_range(self.df.wavelength.min(), 6860)

        mask_regions = ([5875, 5975], [6275, 6327], [6470, 6577])
        for region in mask_regions:
            wmin = region[0]
            wmax = region[1]

            idx = np.where( (self.df.wavelength > wmin) & (self.df.wavelength < wmax) )[0]
            if len(idx):
                self.df.loc[idx, 'flux'] = np.nan

    def remove_red_arm(self):
        '''
        Remove data from the RED pepsi arm (effectively a more flexible wavelength filter)
        '''
        
        #First get the CDs and their wavelength ranges
        cd0 = self.header['CROSDIS']

        if hasattr(self, 'alt_header'):
            cd1 = self.alt_header['CROSDIS']

            cd0_number = int(cd0.split(':')[0])
            cd1_number = int(cd1.split(':')[0])

            cd0_wavelength = cd0.split(':')[1].replace(' ', '').split('-')
            cd1_wavelength = cd1.split(':')[1].replace(' ', '').split('-')

            cd0_wavelength = [ int(x) for x in cd0_wavelength ]
            cd1_wavelength = [ int(x) for x in cd1_wavelength ]

            #Swap if necessary
            if cd0_wavelength[0] > cd1_wavelength[0]:
                
                cd_temp = cd0_wavelength
                cd0_wavelength = cd1_wavelength
                cd1_wavelength = cd_temp

            #Check that the swap worked
            assert(cd0_wavelength[0] < cd1_wavelength[0])
            assert(cd0_wavelength[0] != cd1_wavelength[0])

            #Determine and apply the cut
            red_cut = np.mean([cd0_wavelength[1], cd1_wavelength[0]])

            return self.filter_wavelength_range(None, red_cut)
        else:
            return self.filter_wavelength_range(None, 6000)

    @property
    def has_red_arm_data(self):
        
        return self.df.wavelength.min()*self.wavelength_unit > 544*u.nm

class APFspec(EchelleSpectrum):

    cr_quantile=0.98
    deblaze_window=100
    deblaze_percentile=0.95
    deblaze_degree=9
    trim=12
    flux_filter=1.3
    
    def __init__(self, fname, verbose=True, name=None):
        
        self.fname = fname
        hdul_wave = fits.open(utils.get_data_file('apf_wave_2022.fits'))
        self.wavelength = hdul_wave[0].data

        hdul = fits.open(fname)
        self.flux = hdul[0].data

        self.header = {}
        for k in list(hdul[0].header.keys()):
            if k != '':
                self.header[k] = hdul[0].header[k]

        self.wavelength_unit = u.AA
        self.flux_unit = u.dimensionless_unscaled

        self.verbose = verbose

        ra_hms = self.header['RA']
        dec_dms = self.header['DEC']
        coord = SkyCoord(ra_hms+' '+dec_dms, unit=(u.hourangle, u.deg))

        self.RA = coord.ra.deg
        self.DEC = coord.dec.deg
        JD = Time(self.header['THEMIDPT'], format='isot', scale='utc').jd
        self.ObsName = 'Lick Observatory'

        self.JD = utils.convert_jd_bjd(JD, self.RA, self.DEC, self.ObsName)

        self.reduce()
        self.barycentric_corrected = False
        self.name = name

        self.fdbinary_mask = False

class CHIRONspec(EchelleSpectrum):

    cr_quantile=0.98
    deblaze_window=40
    deblaze_percentile=0.95
    deblaze_degree=4
    trim=5
    flux_filter=1.3

    def __init__(self, fname, verbose=True, run_reduction=True):

        self.fname = fname
        hdul = fits.open(fname)

        self.wavelength = hdul[0].data[:,:,0]
        self.flux = hdul[0].data[:,:,1]

        self.header = {}
        for k in list(hdul[0].header.keys()):
            if k == 'GAINS':
                continue
            elif k != '':
                self.header[k] = hdul[0].header[k]

        self.wavelength_unit = u.AA
        self.flux_unit = u.dimensionless_unscaled

        self.verbose=verbose

        ra_hms = self.header['RA']
        dec_dms = self.header['DEC']
        coord = SkyCoord(ra_hms+' '+dec_dms, unit=(u.hourangle, u.deg))
        self.RA = coord.ra.deg
        self.DEC = coord.dec.deg
        JD = Time(self.header['EMMNWOB'], format='isot', scale='utc').jd
        self.ObsName = 'Cerro Tololo Interamerican Observatory'

        self.JD = utils.convert_jd_bjd(JD, self.RA, self.DEC, self.ObsName)
        if run_reduction:
            self.reduce()
        self.barycentric_corrected = False

        self.fdbinary_mask = False

    def reduce(self, 
               cr_quantile=None,
               deblaze_window=None, deblaze_percentile=None,
               deblaze_degree=None, trim=None, flux_filter=None):
 
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

            eo = CHIRON_EchelleOrder(i, wave, flux)
            eo.df['wavelength'] = eo.df.wavelength.astype(float)
            eo.df['flux'] = eo.df.flux.astype(float)
            eo.filter_cosmic_rays(quantile=cr_quantile)
            eo.deblaze(window=deblaze_window, percentile=deblaze_percentile,
                       degree=deblaze_degree)
            eo.trim_edges(trim)
            eo.flux_filter(flux_filter)

            self.Orders.append(eo)

class PHOENIXspec(BaseSpectrum):
    
    def __init__(self, fname, verbose=True):

        hdul = fits.open(fname)
        fluxes = hdul[0].data.byteswap().newbyteorder()

        hdul_wavelength = fits.open(utils.get_data_file(
                'WAVE_PHOENIX-ACES-AGSS-COND-2011.fits'))
        wavelength = hdul_wavelength[0].data.byteswap().newbyteorder()

        #PHOENIX wavelegnths are vacuum wavelength. Convert to air
        wavelength = spectrum_utils.convert_to_air(wavelength, u.AA)

        self.df = pd.DataFrame({'wavelength':wavelength,
                                'flux': fluxes})
        self.df['err'] = 0

        self.header = {}
        for k in list(hdul[0].header.keys()):
            if k != '':
                self.header[k] = hdul[0].header[k]

        self.wavelength_unit = u.AA
        self.flux_unit = u.Unit('erg/s/cm^2/cm')
        self.default_filter_size = 1000
        self.verbose=verbose

        self.df['mask'] = np.zeros(len(self.df))
