#!/usr/bin/env python

from astropy import units as u
from astropy.time import Time
import brokenaxes
import copy
import corner
import dill
import emcee
from functools import partial
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.legend_handler import HandlerTuple
import multiprocessing as mp
from numba import typed
from numba.types import unicode_type, float64
import numpy as np
import os
import pandas as pd
import pickle
import pkg_resources
import subprocess
from tqdm.autonotebook import tqdm

from . import astrometric_orbit_fitter
from . import plotutils
from . import rv_orbit_fitter
from . import spectra
from . import spectrum_utils
from . import utils
from . import fdbinary

#Dom Rowan 2024

class SpectroscopicBinary:

    def __init__(self, name, verbose=True):

        self.name = name
        self.verbose = verbose

        self._gamma_corrected=False
        self._spectra_dir = None

        self.burnin = 0
        self.lnprob_cutoff = -np.inf

    @property
    def verbose(self):
        return self._verbose
    @verbose.setter
    def verbose(self, value):
        self._verbose = bool(value)

        for spec in self.spectra:
            spec.verbose = value

        if hasattr(self, 'fdbinary'):
            self.fdbinary.verbose = value

    def print(self, value):
        if self.verbose:
            print(value)

    @property
    def spectra_dir(self):
        return self._spectra_dir
    @spectra_dir.setter
    def spectra_dir(self, value):
        if os.path.isdir(value):
            self._spectra_dir = value
        else:
            raise FileNotFoundError(f'Spectra dir not found {value}')

    def load_spectra(self, spectra_dir=None, skip_files=None):

        if spectra_dir is not None:
            self.spectra_dir = spectra_dir

        if self.spectra_dir is None:
            raise ValueError('Spectra dir must be defined first')

        if skip_files is not None:
            if not utils.check_iter(skip_files):
                skip_files = [ skip_files ]

        self._load_pepsi_spectra(skip_files=skip_files)
        self._load_apf_spectra(skip_files=skip_files)
        self._load_chiron_spectra(skip_files=skip_files)

        #Check the coordinates of all the spectra
        spectrum_utils.check_coordinates(self.spectra)


    def _load_pepsi_spectra(self, skip_files=None, **kwargs):

        pepsi_fnames = spectrum_utils.organize_pepsi_files(
                self.spectra_dir, skip_files=skip_files)
        if pepsi_fnames is None:
            self.pepsi_spectra = []
        else:
            self.pepsi_spectra = [ spectra.PEPSIspec(f, **kwargs) 
                                   for f in pepsi_fnames ]

    def _load_apf_spectra(self, skip_files=None, **kwargs):

        kwargs.setdefault('verbose', False)
        apf_fnames = [ os.path.join(self.spectra_dir, f) 
                       for f in os.listdir(self.spectra_dir)
                       if (not f.startswith('achi')) and
                          (f.endswith('.fits')) ]

        if skip_files is not None:
            apf_fnames = [ f for f in apf_fnames 
                           if os.path.split(f)[-1] not in skip_files ]

        if len(apf_fnames):
            if len(apf_fnames) < 4:
                self.apf_spectra = [ spectra.APFspec(f, **kwargs) 
                                     for f in tqdm(apf_fnames) ]
            else:
                num_processes = np.min([8, len(apf_fnames)//2])
                chunks = [ apf_fnames[i:i+len(apf_fnames)//num_processes]
                           for i in range(0, len(apf_fnames), 
                                          len(apf_fnames)//num_processes) ]

                with mp.Pool(processes=num_processes) as pool:
                    results = pool.map(
                            partial(spectrum_utils.process_apf_chunk, **kwargs),
                            chunks)

                self.apf_spectra = [ item for sublist in results 
                                     for item in sublist ]
        else:
            self.apf_spectra = []


    def _load_chiron_spectra(self, skip_files=None, **kwargs):

        chiron_fnames = [ os.path.join(self.spectra_dir, f)
                         for f in os.listdir(self.spectra_dir)
                         if f.startswith('achi') ]

        if skip_files is not None:
            chiron_fnames = [ f for f in chiron_fnames 
                              if os.path.split(f)[-1] not in skip_files ]

        self.chiron_spectra = [ spectra.CHIRONspec(f, **kwargs) 
                                for f in chiron_fnames ]

    @property
    def spectra(self):
        
        try:
            all_spectra = (self.pepsi_spectra + 
                           self.apf_spectra + 
                           self.chiron_spectra)
            all_jds = [ spec.JD for spec in all_spectra ]
            spectra = [ spec for _,spec in sorted(zip(all_jds, all_spectra)) ]
            return spectra

        except:
            return []

    @property
    def echelle_spectra(self):
        '''
        Get all of the echelle spectra for a given target
        Sorted by JD
        '''

        try:
            all_echelle_spectra = self.apf_spectra + self.chiron_spectra
            all_jds = [ spec.JD for spec in all_echelle_spectra ]
            spectra = [ spec for _,spec in sorted(zip(all_jds, 
                                                      all_echelle_spectra)) ]
            return spectra
        except:
            return []

    def get_wavelength_range(self, wavelength_unit=u.AA):

        spec = self.spectra[0]
        wmin, wmax = spec.get_wavelength_range()
        wmin = (wmin*spec.wavelength_unit).to(wavelength_unit).value
        wmax = (wmax*spec.wavelength_unit).to(wavelength_unit).value

        for i in range(1, len(self.spectra)):
            wmini, wmaxi = self.spectra[i].get_wavelength_range()
            wmini = (wmini*self.spectra[i].wavelength_unit).to(wavelength_unit).value
            wmaxi = (wmaxi*self.spectra[i].wavelength_unit).to(wavelength_unit).value

            if wmini < wmin:
                wmin = wmini
            if wmaxi > wmax:
                wmax = wmaxi

        return wmin, wmax

    def filter_wavelength_range(self, lower, upper, wavelength_unit=u.AA):

        for spec in self.spectra:
            
            if spec.wavelength_unit != wavelength_unit:
                lower = (lower*wavelength_unit).to(spec.wavelength_unit).value
                upper = (upper*wavelength_unit).to(spec.wavelength_unit).value

            spec.filter_wavelength_range(lower, upper)

    def select_apf_orders(self, orders):
        
        for spec in self.apf_spectra:
            spec.select_orders(orders)

    def drop_apf_orders(self, orders):
        
        for spec in self.apf_spectra:
            spec.drop_orders(orders)

    def select_chiron_orders(self, orders):
        
        for spec in self.chiron_spectra:
            spec.select_orders(orders)

    def drop_chiron_orders(self, orders):
        
        for spec in self.chiron_spectra:
            spec.drop_orders(orders)

    def remove_red_arm(self):
        
        for spec in self.pepsi_spectra:
            spec.remove_red_arm()

    def correct_barycentric_velocity(self):
        
        for spec in self.spectra:
            spec.correct_barycentric_velocity()
    
    def correct_system_velocity(self):

        if not self._gamma_corrected:
            rv_samples, _ = self.get_mcmc_chain()
            for spec in self.spectra:
                spec.shift_rv(-1*rv_samples.gamma.median())
            self._gamma_corrected = True

    @property
    def gamma_corrected(self):
        return self._gamma_corrected

    def load_rv_table(self, fname, time_offset=0):

        if isinstance(fname, pd.core.frame.DataFrame):
            self.df_rv = fname
        else:
            self.df_rv = pd.read_csv(fname)

        self.instrument_list = self.df_rv.Instrument.value_counts().index

        self.df_rv['JD'] = self.df_rv.JD - time_offset
        self.time_offset = time_offset

    def plot_rv_orbit_broken(self, fig=None, gs=None, 
                             savefig=None, gap=5, **kwargs):
        
        #Determine breaks
        df_rv = self.df_rv.copy().sort_values(by='JD', ascending=True)
        df_rv = df_rv.reset_index(drop=True)
        df_rv['gap'] = np.concatenate([np.zeros(1), np.diff(df_rv.JD)])

        #Period to use for break selection
        rv_samples, _ = self.get_mcmc_chain()
        period = rv_samples.period.median()
        df_rv['gap_period'] = df_rv.gap / period
        idx = np.where(df_rv.gap_period > gap)[0]
        xlims = []
        
        if len(idx):
            df_split = [ df_rv.iloc[:idx[0]] ]

            for i in range(1, len(idx)):
                df_split.append(df_rv[idx[i-1]:idx[i]])
            df_split.append(df_rv[idx[-1]:])

            #Now get xlims from split dfs
            for i in range(len(df_split)):
                xlims.append( (df_split[i].JD.min()-period*0.25, 
                               df_split[i].JD.max()+period*0.25))
        else:
            ax = fig.add_subplot(gs)
            return self.plot_rv_orbit(ax=ax, **kwargs)

        if fig is None:
            created_fig = True
            fig = plt.Figure(figsize=(12, 6))
            fig.subplots_adjust(top=.95, right=.98, left=0.08)
            bax = brokenaxes.brokenaxes(fig=fig, xlims=xlims, despine=False)
            bax = self.plot_rv_orbit(ax=bax, **kwargs)

            bax = plotutils.plotparams_bax(bax)

        else:
            created_fig = False
            bax = brokenaxes.brokenaxes(fig=fig, subplot_spec=gs, 
                                        xlims=xlims, despine=False)
            bax = self.plot_rv_orbit(ax=bax, **kwargs)

        if created_fig:
            if savefig is not None:
                fig.savefig(savefig)
            else:
                plt.show()
        else:
            return bax

    @property
    def burnin(self):
        return self._burnin

    @burnin.setter
    def burnin(self, value):
        self._burnin = value

    @property
    def lnprob_cutoff(self):
        return self._lnprob_cutoff

    @lnprob_cutoff.setter
    def lnprob_cutoff(self, value):
        self._lnprob_cutoff = value

    def _get_flat_mcmc_samples(self):

        if hasattr(self, '_mcmc_chain_path'):
            lnprob = self.loaded_lnprob
            samples = self.loaded_samples
        else:
            lnprob = self.rv_sampler.lnprobability.copy()
            samples = self.rv_sampler.get_chain(discard=0, thin=1)

        burnin = self.burnin
        lnprob_cutoff = self.lnprob_cutoff
        thin = 1

        samples = samples[burnin:, :, :][::thin, :, :] #n_steps, #n_walkers, #n_dim
        lnprob = lnprob[:, burnin:][:, ::thin] #n_walkers, #n_steps
        lnprob = lnprob.T #n_steps, #n_walkers

        samples_flat = samples.reshape(-1, samples.shape[2]) #nsteps*nwalkers, n_dim
        
        lnprob_flat = lnprob.reshape(-1) #n_steps*n_walkers

        lnprob_inds = np.where(lnprob_flat > lnprob_cutoff)[0]
        samples = samples_flat[lnprob_inds]
        lnprob = lnprob_flat[lnprob_inds]

        return samples, lnprob

    def save_mcmc_chain(self, outfile, full=True):

        if full:
            lnprob = self.rv_sampler.lnprobability.copy()
            samples = self.rv_sampler.get_chain(discard=0, thin=1)
        else:
            samples, lnprob = self.get_mcmc_chain()

        with open(outfile, 'wb') as p:
            pickle.dump((samples, lnprob), p)

    def load_mcmc_chain(self, infile):

        with open(infile, 'rb') as p:
            samples, lnprob = pickle.load(p)

        self.loaded_samples = samples
        self.loaded_lnprob = lnprob
        self._mcmc_chain_path = infile

    def plot_lnprob(self, ax=None, savefig=None):

        fig, ax, created_fig = plotutils.fig_init(ax=ax, figsize=(9, 6))

        if hasattr(self, '_mcmc_chain_path'):
            lnprob = self.loaded_lnprob
        else:
            lnprob = self.rv_sampler.lnprobability.copy()

        burnin = self.burnin
        lnprob_cutoff = self.lnprob_cutoff
        thin = 1

        lnprob = lnprob[:, burnin:][:, ::thin]
        lnprob[lnprob < lnprob_cutoff] = np.nan

        for i in range(lnprob.shape[0]):
            ax.plot(np.arange(len(lnprob[i, :]))+burnin, lnprob[i,:], lw=1, alpha=0.6, rasterized=True)

        ax.set_xlabel('Iterations', fontsize=20)
        ax.set_ylabel('lnprob', fontsize=20)

        return plotutils.plt_return(created_fig, fig, ax, savefig)

    def plot_mcmc_trace(self, savefig=None, labels=None):
        
        lnprob = self.rv_sampler.lnprobability.copy()
        samples = self.rv_sampler.get_chain(discard=0, thin=1).copy()

        burnin = self.burnin
        lnprob_cutoff = self.lnprob_cutoff
        thin = 1

        samples = samples[burnin:, :, :][::thin, :, :] #n_steps, #n_walkers, #n_dim
        lnprob = lnprob[:, burnin:][:, ::thin] #n_walkers, #n_steps
        lnprob = lnprob.T #n_steps, #n_walkers

        ndim = samples.shape[2]
        figsize = (10, ndim*4)

        if labels is None:
            labels = [f'param_{i}' for i in range(samples.shape[2])]

        fig, ax = plt.subplots(ndim, 1, figsize=figsize, sharex=True)
        fig.subplots_adjust(top=.98, right=.98, hspace=0.0)

        for k in range(samples.shape[2]):
            ax[k] = plotutils.plotparams(ax[k])
            ax[k].set_ylabel(labels[k], fontsize=20)
            for j in range(samples.shape[1]):

                samples_walker = samples[:, j, k]
                lnprob_walker = lnprob[:, j]
                idx = np.where(lnprob_walker < lnprob_cutoff)[0]
                samples_walker[idx] = np.nan
                ax[k].plot(samples_walker, alpha=0.6, rasterized=True)

        ax[-1].set_xlabel('Iterations - Burnin', fontsize=20)

        if savefig is not None:
            fig.savefig(savefig)
        else:
            plt.show()

    def to_dill(self, outfile):
        
        with open(outfile, 'wb') as p:
            dill.dump(self, p)

class SingleLinedSpectroscopicBinary(SpectroscopicBinary):

    def measure_rvs(self, template, **kwargs):
        
        current_verbose = self.verbose
        self.verbose = False
        for spec in tqdm(self.spectra):
            spec.cross_correlation(template, **kwargs)
            spec.ccf.model()
        self.verbose = current_verbose

    def build_rv_table(self, time_offset=0.0):

        jds = []
        rvs1 = []
        rv_errs1 = []
        instruments = []
        specs = []

        self.time_offset = time_offset

        spec_lists = [self.pepsi_spectra, 
                      self.apf_spectra, 
                      self.chiron_spectra]
        instrument_list = ['PEPSI', 'APF', 'CHIRON']

        for spec_list, instrument in zip(spec_lists, instrument_list):
            
            for spec in spec_list:
                
                jds.append(spec.JD)

                if len(spec.ccf.models):
                    if isinstance(spec, spectra.APFspec):
                        rvs1.append(spec.ccf.models[0].mu - 2.36)
                    else:
                        rvs1.append(spec.ccf.models[0].mu)

                    rv_errs1.append(spec.ccf.models[0].emu)
                else:
                    rvs1.append(np.nan)
                    rv_errs1.append(np.nan)
                instruments.append(instrument)
                specs.append(spec)

        self.df_rv = pd.DataFrame({'JD':jds,
                                   'RV1': rvs1,
                                   'RV1_err':rv_errs1,
                                   'Instrument':instruments,
                                   'spec':specs})

        self.df_rv = self.df_rv.sort_values(by='JD', ascending=True)
        self.df_rv = self.df_rv.reset_index(drop=True)

        paths = []
        for spec in self.df_rv.spec:
            if isinstance(spec.fname, tuple):
                paths.append( tuple(
                        [ os.path.split(t)[-1] for t in spec.fname ]))
            else:
                paths.append( os.path.split(spec.fname)[-1] )
        self.df_rv['path'] = paths

        self.df_rv['JD'] = self.df_rv.JD - self.time_offset
        self.instrument_list = self.df_rv.Instrument.value_counts().index

    def plot_rvs(self, ax=None, savefig=None, 
                 markers_dict=None, plot_kwargs=None):

        fig, ax, created_fig = plotutils.fig_init(ax, figsize=(10, 6))

        if markers_dict is None:
            markers_dict = plotutils.create_markers_dict(self.instrument_list)

        if plot_kwargs is None:
            plot_kwargs = {}

        plot_kwargs.setdefault('color', 'black')
        plot_kwargs.setdefault('ls', '')

        for instrument in self.instrument_list:
            
            df_plot = self.df_rv[self.df_rv.Instrument == instrument]
            ax.errorbar(df_plot.JD, df_plot.RV1, yerr=df_plot.RV1_err,
                        marker=markers_dict[instrument],
                        **plot_kwargs)

        if self.time_offset == 0:
            ax.set_xlabel(r'$\rm{JD}$', fontsize=20)
        else:
            ax.set_xlabel(r'$\rm{JD}-'+str(self.time_offset/1e6)+r'\times10^6$', fontsize=20)
        ax.set_ylabel(r'Radial Velocity [km/s]', fontsize=20)

        return plotutils.plt_return(created_fig, fig, ax, savefig)

    def joker_test(self):

        # Run the script within the specified conda environment
        script = pkg_resources.resource_filename(__name__, 'thejoker_rs_test.py')
        subprocess.run(f'source ~/anaconda3/etc/profile.d/conda.sh && conda activate thejoker2 && {script} && conda activate bhs', shell=True)
        

    def fit_joker(self, cleanup=True, **kwargs):
        
        '''
        Fitting thejoker in an external environment

        idea is to create a subprocess in a different conda env, excecute a script.
        '''

        #Run the rejection sampling
        np.random.seed()
        joker_input_file = f'tj_input_{np.random.randint(1e9)}.csv'
        joker_output_file = f'tj_output_{np.random.randint(1e9)}.csv'
        joker_kwargs_pickle = f'tj_kwargs_pickle_{np.random.randint(1e9)}.pickle'

        df_rv = self.df_rv.copy()
        df_rv['RV1_err'] = 1.0
        df_rv.to_csv(joker_input_file, index=False)

        script = pkg_resources.resource_filename(__name__, 'thejoker_rs.py')

        with open(joker_kwargs_pickle, 'wb') as p:
            joker_kwargs_dict = {'inputfile':joker_input_file,
                                 'outputfile':joker_output_file}

            pickle.dump({**joker_kwargs_dict, **kwargs}, p)

        # Run the script within the specified conda environment
        subprocess.run(f'source ~/anaconda3/etc/profile.d/conda.sh && conda activate thejoker2 && {script} "{os.path.abspath(joker_kwargs_pickle)}" && conda activate bhs', shell=True)

        # Load in joker results
        assert(os.path.isfile(joker_output_file))
        self.df_rs = pd.read_csv(joker_output_file)

        #Add fM column to df_rs

        self.df_rs['fM'] = [ utils.mass_function(self.df_rs.P.iloc[i]*u.day,
                                                 np.abs(self.df_rs.K.iloc[i])*u.km/u.s,
                                                 e=self.df_rs.e.iloc[i])
                             for i in range(len(self.df_rs)) ]

        # Cleanup temporary files
        if cleanup:
            for fname in [joker_input_file, joker_output_file, joker_kwargs_pickle]:
                subprocess.call(['rm', fname])

        self.rs_unimodal_p = utils.is_P_unimodal(self.df_rs.P.to_numpy(),
                                                 self.df_rv.JD.to_numpy())

    def plot_joker_solution(self, ax=None, savefig=None):

        if not hasattr(self, 'df_rs'):
            raise ValueError('Joker fit must be run first')

        fig, ax, created_fig = plotutils.fig_init(ax=ax, figsize=(10, 6))

        tvals = np.linspace(self.df_rv.JD.min() - 10, self.df_rv.JD.max()+10, 1000)

        if utils.is_P_unimodal(self.df_rs.P.to_numpy(), self.df_rv.JD.to_numpy()):
        
            K = self.df_rs.K.median()
            P = self.df_rs.P.median()
            omega = self.df_rs.omega.median()
            gamma = self.df_rs.v0.median()
            ecc = self.df_rs.e.median()
            M0 = self.df_rs.M0.median()
            logs = np.log(self.df_rs.s.median())
            #M0_new = rv_orbit_fitter_tj.redetermine_M0(self.df_rv.JD.to_numpy(), self.df_rv.RV.to_numpy(),
            #                        K, gamma, ecc, omega, P)
        
            models = np.array([[ rv_orbit_fitter.rv_model(tvals, K, gamma, M0, ecc, omega, P) ]])

        else:
            nmodels = min([100, len(self.df_rs)])
            models = np.zeros((nmodels, len(tvals)))
            
            if nmodels == 100:
                df_rs = self.df_rs.copy().sample(frac=1).reset_index(drop=True).iloc[:100]
            else:
                df_rs = self.df_rs.copy()

            JDarr = self.df_rv.JD.to_numpy()
            RVarr = self.df_rv.RV.to_numpy()
            for i in tqdm(range(nmodels)):

                K = self.df_rs.K.iloc[i]
                P = self.df_rs.P.iloc[i]
                omega = self.df_rs.omega.iloc[i]
                gamma = self.df_rs.v0.iloc[i]
                ecc = self.df_rs.e.iloc[i]
                M0 = self.df_rs.M0.iloc[i]
                logs = np.log(self.df_rs.s.iloc[i])

                #M0_new = rv_orbit_fitter_tj.redetermine_M0(JDarr, RVarr, K, gamma, ecc, omega, P)

                models[i] = rv_orbit_fitter_tj.rv_model(tvals, K, gamma, M0, ecc, omega, P)

        x0 = 1
        x1 = 50
        x2 = 100
        y0 = 0.9
        y1 = 0.4
        y2 = 0.2
        X = np.array([[x0**2, x0, 1], [x1**2, x1, 1], [x2**2, x2, 1]])
        Y = np.array([y0, y1, y2])
        a, b, c = np.dot(np.linalg.inv(X), Y)
        alpha = a*models.shape[0]**2 + b*models.shape[0]+c

        if models.shape[0] == 1:
            lw = 2
        else:
            lw = 1

        for i in range(models.shape[0]):

            ax.plot(tvals, models[i].reshape(-1), alpha=alpha, lw=lw, color='xkcd:red')

        #Plot Data
        ax.errorbar(self.df_rv.JD, self.df_rv.RV1,
                    yerr=self.df_rv.RV1_err, color='black', ls='', marker='o',
                    zorder=2)

        ax.set_xlabel('Julian Date', fontsize=20)
        ax.set_ylabel('RV (km/s)', fontsize=20)

        return plotutils.plt_return(created_fig, fig, ax, savefig)

    def fit_rvs(self, guess, niters=50000, 
                idx_mask=None, apsidal=False, **lnprob_kwargs):

        '''
        Fit RV model for single-lined spectroscopic bianry
        '''

        self._fit_apsidal = apsidal

        if guess == 'Gaia':
            gaia_orbit = utils.query_gaia_orbit(self.name)
            period = gaia_orbit['Per']
            K1 = gaia_orbit['K1']
            gamma = gaia_orbit['Vcm']
            ecc = gaia_orbit['ecc']
            omega = gaia_orbit['omega']*np.pi/180
            M0 = np.pi #temp

            pos = [ K1, gamma, M0, ecc, omega, period, -5]

            #Convert the gaia orbit dict to numba typed dictionary
            numba_dict = typed.Dict.empty(key_type=unicode_type, value_type=float64)
            for k, v in gaia_orbit.items():
                if k == 'Source':
                    continue
                else:
                    numba_dict[k] = v
        elif guess == 'Joker':
            period = self.df_rs.P.median()
            ecc = self.df_rs.e.median()
            K1 = self.df_rs.K.median()
            omega = self.df_rs.omega.median()

            if K1 < 0:
                K1 *= -1
                omega = omega - np.pi

            while omega < -np.pi:
                omega += 2*np.pi
            while omega > np.pi:
                omega -= 2*np.pi
            gamma = self.df_rs.v0.median()
            #M0 = self.df_rs.M0.median()

            M0 = np.pi

            pos = [ K1, gamma, M0, ecc, omega, period, -5]
            print(pos)

        else:
            if not apsidal:
                K1, gamma, M0, ecc, omega, period = guess
                pos = [K1, gamma, M0, ecc, omega, period, -5]
            else:
                K1, gamma, M0, ecc, omega, period, omegadot = guess
                pos = [K1, gamma, M0, ecc, omega, period, -5, omegadot]

        pos = pos + [ 0 for i in range(len(self.instrument_list)-1) ]

        nwalkers = len(pos)*2
        pos = np.array(pos)+1e-3*np.random.randn(nwalkers, len(pos))

        nwalkers, ndim = pos.shape

        #Mask specific RVs from orbit fit
        mask_rv = np.zeros(len(self.df_rv))
        if idx_mask is not None:
            mask_rv[idx_mask] = 1
        self.df_rv['mask_rv'] = mask_rv

        lnprob_args = (
                tuple([ self.df_rv[(self.df_rv.Instrument == instrument) & 
                                   (self.df_rv.mask_rv == 0)].JD.values
                        for instrument in self.instrument_list ]),
                tuple([ self.df_rv[(self.df_rv.Instrument == instrument) & 
                                   (self.df_rv.mask_rv == 0)].RV1.values
                        for instrument in self.instrument_list ]),
                tuple([ self.df_rv[(self.df_rv.Instrument == instrument) & 
                                   (self.df_rv.mask_rv == 0)].RV1_err.values
                        for instrument in self.instrument_list ]))

        if guess == 'Gaia':
            lnprob_kwargs['gaia_dict'] = numba_dict
        else:
            lnprob_kwargs.setdefault('apsidal', apsidal)
            lnprob_kwargs.setdefault('period_mu', period)
            lnprob_kwargs.setdefault('period_sigma', period*0.1)

        self.rv_sampler = emcee.EnsembleSampler(
                nwalkers, ndim, rv_orbit_fitter.lnprob_sb1,
                args=lnprob_args,
                kwargs=lnprob_kwargs)

        self.rv_sampler.run_mcmc(pos, niters, progress=self.verbose)

    def get_mcmc_chain(self):

        samples, lnprob = self._get_flat_mcmc_samples()

        mcmc_cols = ['K1', 'gamma', 'M0', 'ecc', 'omega', 'period', 'logs']
        if self._fit_apsidal:
            mcmc_cols.append('omegadot')
        mcmc_cols = mcmc_cols + [ f'rv_offset_{i}' 
                              for i in range(len(self.instrument_list)-1) ]

        rv_samples = pd.DataFrame(samples, columns=mcmc_cols)

        rv_samples['T0'] = rv_samples.M0 * rv_samples.period / (2*np.pi)

        #Add mass function column
        rv_samples['fM'] = utils.mass_function(
                rv_samples.period.to_numpy()*u.day,
                np.abs(rv_samples.K1.to_numpy())*u.km/u.s,
                e=rv_samples.ecc.to_numpy()).value

        return rv_samples, lnprob

    def plot_mcmc_trace(self, savefig=None):

        mcmc_cols = ['K1', 'gamma', 'M0', 'ecc', 'omega', 'period', 'logs']
        if self._fit_apsidal:
            mcmc_cols.append('omegadot')
        mcmc_cols = mcmc_cols + [ f'rv_offset_{i}' 
                              for i in range(len(self.instrument_list)-1) ]

        super().plot_mcmc_trace(savefig=savefig, labels=mcmc_cols)


    def plot_rv_corner(self, savefig=None, figsize=None, 
                       gaia_priors=False,
                       offset_numeric_label=True):

        samples, _ = self.get_mcmc_chain()
        samples = samples.copy()
        samples['M0'] = samples.T0.values
        samples = samples.drop(columns=['T0', 'fM'], errors='ignore')

        labels = [r'$K_1\ \rm{[km/s]}$', r'$\gamma\ \rm{[km/s]}$', 
                  r'$T_0\ \rm{[d]}$',
                  'Ecc', r'$\omega\ [\rm{rad}]$', r'$P \rm{[d]}$',
                  r'$\log s\ \rm{[km/s]}$']

        units = [r'$\rm{[km/s]}$', 
                 r'$\rm{[km/s]}$', 
                 r'$[\rm{d}]$',
                 '', 
                 r'$[\rm{rad}]$',
                 r'$[\rm{d}]$',
                 r'$\rm{[km/s]}$']

        if 'omegadot' in samples.columns:
            labels.append(r'$\dot{\omega}\ \rm{[deg/yr]}$')
            units.append(r'$\rm{[deg/yr]}$')

        for i in range(len(self.instrument_list)-1):
            instrument = self.instrument_list[i]
            if offset_numeric_label:
                labels.append(r'$\delta\rm{RV}_{'+str(i)+r'}\ \rm{[km/s]}$')
            else:
                labels.append((r'$\delta\rm{RV}_{\textnormal{\small '+
                               instrument+
                               r'}}\ \rm{[km/s]}$'))

            units.append(r'$\rm{(km/s)}$')

        if figsize is None:
            n = len(labels)+4
            figsize=(n,n)

        fig, ax = plt.subplots(samples.shape[1], samples.shape[1], 
                               figsize=figsize)

        fig = corner.corner(samples, labels=labels, fig=fig,
                            hist_kwargs=dict(lw=3, color='black'),
                            quantiles=[0.5],
                            label_kwargs=dict(fontsize=22))

        for a in ax.reshape(-1):
            a = plotutils.plotparams(a)

        if gaia_priors:
            gaia_dict = utils.query_gaia_orbit(self.name)

            keys = {'Per':'period', 
                    'ecc':'ecc', 
                    'Vcm': 'gamma', 
                    'K1':'K1', 
                    'omega': 'omega'}

            for k in keys.keys():
                
                samples_key = keys[k]
                mu = gaia_dict[k]
                emu = gaia_dict[f'e_{k}']

                idx = np.where(np.asarray(samples.columns) == samples_key)[0][0]
                random_samples = np.random.normal(loc=mu, scale=emu, size=len(samples))
                ax[idx,idx].hist(random_samples, histtype='step', color='xkcd:red', bins=20, lw=2, ls='--')

        #Set titles
        for i in range(samples.shape[1]):
            
            med_val = np.median(samples[samples.columns[i]])
            lower = np.quantile(samples[samples.columns[i]], 0.16)
            upper = np.quantile(samples[samples.columns[i]], 0.84)

            lower_err = med_val - lower
            upper_err = upper - med_val

            try:
                title = utils.round_val_err(
                        med_val, lower_err, upper_err) + ' ' + units[i]
            except:
                title = 'title failed'

            ax[i,i].set_title(title, fontsize=12)

        if savefig is None:
            plt.show()
        else:
            fig.savefig(savefig)

    def plot_rv_orbit(self, ax=None, savefig=None, legend=False, legend_kwargs=None,
                      markers_dict=None, tmin=None, tmax=None, 
                      period_text=False, fm_text=False):
        
        '''
        Plot single-lined RV model
        '''

        rv_samples, _ = self.get_mcmc_chain()

        fig, ax, created_fig = plotutils.fig_init(ax=ax, figsize=(10, 6))
        if created_fig and ((not period_text) and (not fm_text)):
            fig.subplots_adjust(top=.98, right=.98)
        elif created_fig and (period_text or fm_text):
            fig.subplots_adjust(top=.95, right=.98)

        if tmin is None:
            tmin = self.df_rv.JD.min()-0.1*rv_samples.period.median()
        if tmax is None:
            tmax = self.df_rv.JD.max()+0.1*rv_samples.period.median()
        tvals = np.linspace(tmin, tmax, int(1e4))

        rv_offsets = [0] + [ rv_samples[f'rv_offset_{i-1}'].median()
                             for i in range(1, len(self.instrument_list)) ]

        if markers_dict is None:
            markers_dict = plotutils.create_markers_dict(self.instrument_list)

        for instrument, offset in zip(self.instrument_list, rv_offsets):
            
            df_plot = self.df_rv[(self.df_rv.Instrument == instrument) & 
                                 (self.df_rv.mask_rv == 0)]
            df_plot_masked = self.df_rv[(self.df_rv.Instrument == instrument) &
                                        (self.df_rv.mask_rv == 1)]

            e1 = ax.errorbar(df_plot.JD, df_plot.RV1 - offset,
                        yerr=df_plot.RV1_err,
                        color='white', markeredgecolor='black', mew=2,
                        ecolor='black', 
                        marker=markers_dict[instrument], ls='',
                        label=instrument)

            if len(df_plot_masked):
                
                ax.errorbar(df_plot_masked.JD, df_plot_masked.RV1 - offset,
                            yerr = df_plot_masked.RV1_err,
                            color='white', markeredgecolor='black', mew=1,
                            ecolor='black',
                            marker=markers_dict[instrument], ls='', 
                            alpha=0.5)

        apsidal = 'omegadot' in rv_samples.columns

        for i in range(100):
            
            j = np.random.randint(0, len(rv_samples))
            sample = rv_samples.iloc[j].to_numpy()
            if not apsidal:
                K1, gamma, phi0, ecc, omega, period = sample[:6]
                model1 = rv_orbit_fitter.rv_model(
                        tvals, K1, gamma, phi0, ecc, omega, period)
            else:
                K1, gamma, phi0, ecc, omega, period, logs, omegadot = sample[:8]
                model1 = rv_orbit_fitter.rv_model(
                        tvals, K1, gamma, phi0, ecc, omega, period, omegadot=omegadot)

            ax.plot(tvals, model1, color='xkcd:red', lw=1, alpha=0.2, zorder=1)

        if self.time_offset == 0:
            ax.set_xlabel(r'$\rm{JD}$', fontsize=20)
        else:
            ax.set_xlabel(r'$\rm{JD}-'+str(self.time_offset/1e6)+r'\times10^6$', fontsize=20)
        ax.set_ylabel(r'$\rm{RV}\ [\rm{km/s}]$', fontsize=20)

        if period_text:
            ax.text(.05, 1.0, r'$P={}$ d'.format(round(rv_samples.period.median(), 2)),
                    ha='left', va='center', transform=ax.transAxes, fontsize=18,
                    bbox=dict(edgecolor='black', facecolor='white'))
        
        if fm_text:
            ax.text(.95, 1.0, r'$f(M)={}\ M_\odot$'.format(round(rv_samples.fM.median(), 2)),
                    ha='right', va='center', transform=ax.transAxes, fontsize=18,
                    bbox=dict(edgecolor='black', facecolor='white'))

        if not isinstance(ax, brokenaxes.BrokenAxes):
            ax.set_xlim(tmin, tmax)

        if legend:
            if legend_kwargs is None:
                legend_kwargs = {}

            legend_kwargs.setdefault('fontsize', 15)
            legend_kwargs.setdefault('edgecolor', 'black')
            legend_kwargs.setdefault('loc', 'lower left')

            if isinstance(ax, brokenaxes.BrokenAxes):

                if legend_kwargs['loc'].split()[1] == 'left':
                    axs = ax.axs[0]
                else:
                    axs =ax.axs[-1]

                ylim = axs.get_ylim()
                xlim = axs.get_xlim()

                handles = []
                for instrument in self.instrument_list:
                    
                    df_plot = self.df_rv[self.df_rv.Instrument == instrument]
                    e1 = axs.errorbar([-99], [99], yerr=[10], color='white', 
                                      markeredgecolor='black', mew=2, ecolor='black',
                                      ls='', marker=markers_dict[instrument])
                    handles.append(e1)

                axs.legend(handles, self.instrument_list,
                           **legend_kwargs)

                axs.set_xlim(xlim)
                axs.set_ylim(ylim)

            else:
                ax.legend(**legend_kwargs)

        return plotutils.plt_return(created_fig, fig, ax, savefig)

    def calculate_rv_residuals(self, idx=None):
        

        rv_samples, _ = self.get_mcmc_chain()

        if idx is None:
            idx = np.argsort(rv_samples.period.values)[len(rv_samples)//2]
        sample = rv_samples.iloc[idx].to_numpy()

        K1, gamma, phi0, ecc, omega, period, logs = sample[:7]
        rv_offset_values = np.concatenate([np.array([0]), sample[7:]])
        rv1_offset_corrected = []

        for i in range(len(self.df_rv)):
            
            instrument = self.df_rv.Instrument.iloc[i]
            j = np.where(np.asarray(self.instrument_list) == instrument)[0][0]
            offset = rv_offset_values[j]

            rv1_offset_corrected.append(self.df_rv.RV1.iloc[i] - offset)

        self.df_rv['RV1_offset'] = rv1_offset_corrected

        self.df_rv['model'] = rv_orbit_fitter.rv_model(
                self.df_rv.JD.values, K1, gamma, phi0, ecc, omega, period)
        self.df_rv['residual'] = self.df_rv.RV1_offset - self.df_rv.model

        s1 = np.exp(logs)

        self.df_rv['RV1_err_jitter'] = np.sqrt(
                np.power(self.df_rv.RV1_err, 2)+s1**2)

    def calculate_rv_chi2(self):

        self.calculate_rv_residuals()

        residuals = self.df_rv.residual.values
        errors = self.df_rv.RV1_err.values
        errors_with_jitter = self.df_rv.RV1_err_jitter.values

        idx_include = np.where(self.df_rv.mask_rv == 0)[0]

        self.rv_chi2 = np.sum(np.power(residuals[idx_include]/errors[idx_include],2))
        self.rv_chi2_jitter = np.sum(np.power(residuals[idx_include]/errors_with_jitter[idx_include],2))

        dof = 7 #K1, gamma, M0, ecc, omega, period, logs

        self.rv_chi2_nu = self.rv_chi2 / (len(residuals[idx_include]) - dof)
        self.rv_chi2_nu_jitter = self.rv_chi2_jitter / (len(residuals[idx_include]) - dof)

        return self.rv_chi2_nu, self.rv_chi2_nu_jitter

    def plot_rv_residuals(self, with_jitter=False, plot_masked=True,
                          ax=None, savefig=None, legend=False, legend_kwargs=None,
                          markers_dict=None,
                          plot_kwargs=None,
                          label_chi2=True):
        
        if 'residual' not in self.df_rv.columns:
            self.calculate_rv_residuals()

        fig, ax, created_fig = plotutils.fig_init(ax=ax, figsize=(10, 6))

        if markers_dict is None:
            markers_dict = plotutils.create_markers_dict(self.instrument_list)

        if plot_kwargs is None:
            plot_kwargs = {}
        plot_kwargs.setdefault('color', 'black')
        plot_kwargs.setdefault('ls', '')
        plot_kwargs.setdefault('alpha', 1.0)

        plot_kwargs_masked = copy.deepcopy(plot_kwargs)
        plot_kwargs_masked['alpha'] *= 0.5

        if with_jitter:
            yerr_column = 'RV1_err_jitter'
        else:
            yerr_column = 'RV1_err'

        for instrument in self.instrument_list:

            df_plot = self.df_rv[(self.df_rv.Instrument == instrument) & 
                                 (self.df_rv.mask_rv == 0)]
            df_plot_masked = self.df_rv[(self.df_rv.Instrument == instrument) &
                                        (self.df_rv.mask_rv == 1)]

            ax.errorbar(df_plot.JD, df_plot.residual, yerr=df_plot[yerr_column],
                        marker=markers_dict[instrument],
                        **plot_kwargs)

            if len(df_plot_masked) and plot_masked:

                ax.errorbar(df_plot_masked.JD, df_plot_masked.residual,
                            yerr=df_plot_masked[yerr_column],
                            marker=markers_dict[instrument],
                            **plot_kwargs_masked)

        if self.time_offset == 0:
            ax.set_xlabel(r'$\rm{JD}$', fontsize=20)
        else:
            ax.set_xlabel(r'$\rm{JD}-'+str(self.time_offset/1e6)+r'\times10^6$', fontsize=20)
        ax.set_ylabel(r'Residuals (km/s)', fontsize=20)

        ax.axhline(0.0, color='gray', ls='--')

        ylim = np.max(np.abs(ax.get_ylim()))
        ax.set_ylim(-1*ylim, ylim)

        if legend:

            if legend_kwargs is None:
                legend_kwargs = {}
            legend_kwargs.setdefault('fontsize', 15)
            legend_kwargs.setdefault('edgecolor', 'black')
            legend_kwargs.setdefault('loc', 'lower left')

            ax.legend(handles, self.instrument_list,
                      handler_map={tuple: HandlerTuple(ndivide=None)},
                      **legend_kwargs)

        if label_chi2:
            
            if with_jitter:
                chi2_val = self.rv_chi2_nu_jitter
            else:
                chi2_val = self.rv_chi2_nu

            label = r'$\chi^2_{\nu} = '+f'{chi2_val:.2f}'+r'$'
            ax.text(.95, .95, label, ha='right', va='top', 
                    fontsize=15, transform=ax.transAxes)

    def plot_orbit_separation(self, M1, incl, ax=None, savefig=None,
                              obs_date=None, distance=None, tmin=None, tmax=None,
                              obs_sep=None,
                              **kwargs):

        fig, ax, created_fig = plotutils.fig_init(ax=ax, figsize=(10, 6))

        if obs_date is not None:
            if not utils.check_iter(obs_date):
                obs_date = [obs_date]

            obs_jd = [ Time(od).jd - self.time_offset for od in obs_date ]

        if tmin is None:
            tmin = self.df_rv.JD.min()
            if obs_date is not None and (obs_jd[0] < tmin):
                tmin = obs_jd[0] - 100
        if tmax is None:
            tmax = self.df_rv.JD.max()
            if obs_date is not None and (obs_jd[-1] > tmax):
                tmax = obs_jd[-1] + 100

        tvals = np.linspace(tmin, tmax, 10000)

        if distance is None:
            distance = 1000

        incl = incl *np.pi/180

        rv_samples, _ = self.get_mcmc_chain()

        for i in tqdm(range(100)):
            j = -1*(i+1)
            K = rv_samples.K1.iloc[j]
            P = rv_samples.period.iloc[j]
            T0 = rv_samples.T0.iloc[j]
            ecc = rv_samples.ecc.iloc[j]
            omega = rv_samples.omega.iloc[j]

            seps = astrometric_orbit_fitter.projected_sep(
                    tvals, M1, K, P, T0, ecc, omega, incl, distance)

            ax.plot(tvals, seps, color='xkcd:red', lw=1, alpha=0.2, zorder=1)

        if obs_sep is not None:
            if distance == 1000:
                raise ValueError('Must supply distance if plotting observed seps')

            ax.scatter(obs_jd, obs_sep, marker='X',
                       color='xkcd:azure', s=150, edgecolor='black',
                       zorder=10)

        if self.time_offset == 0:
            ax.set_xlabel(r'$\rm{JD}$', fontsize=20)
        else:
            ax.set_xlabel(r'$\rm{JD}-'+str(self.time_offset/1e6)+r'\times10^6$', fontsize=20)

        if distance == 1000:
            ax.set_ylabel('Projected Separation [AU]', fontsize=25)
        else:
            ax.set_ylabel('Projected Angular Sep. [mas]', fontsize=25)

        return plotutils.plt_return(created_fig, fig, ax, savefig)

    def companion_mass_plot(self, M1vals, ax=None, savefig=None,
                            mass_compare=1.4, incl_min=30, label=True, 
                            legend_kwargs=None, **kwargs):

        fig, ax, created_fig = plotutils.fig_init(ax=ax, figsize=(9, 6))
        if created_fig:
            fig.subplots_adjust(bottom=0.15)

        incl_vals = np.linspace(incl_min, 90, 100)

        rv_samples, _ = self.get_mcmc_chain()
        fM = rv_samples.fM.median()*u.M_sun

        if not utils.check_iter(M1vals):
            M1vals = [M1vals]

        lss = ['-', '--', ':', 'dashdot'][:len(M1vals)]

        kwargs.setdefault('color', 'black')

        for M1, ls in zip(M1vals, lss):

            M2vals = [ utils.companion_mass(fM, M1*u.M_sun, incl).value for incl in incl_vals ]

            if label:
                ax.plot(incl_vals, M2vals, ls=ls, label=r'$M_1 = {}\ M_\odot$'.format(M1), **kwargs)
            else:
                ax.plot(incl_vals, M2vals, ls=ls, label='_nolegend_', **kwargs)

        ax.set_xlabel(r'Orbital Inclination [$^{\circ}$]', fontsize=25)
        ax.set_ylabel(r'Companion Mass [$M_\odot$]', fontsize=25)

        if legend_kwargs is None:
            legend_kwargs = {}
        legend_kwargs.setdefault('loc', 'upper right')
        legend_kwargs.setdefault('edgecolor', 'black')
        legend_kwargs.setdefault('fontsize', '15')
        ax.legend(**legend_kwargs)
        ax.set_xlim(incl_min, 90)

        return plotutils.plt_return(created_fig, fig, ax, savefig)

    def get_quadrature_spec(self):
        
        rv_samples, _ = self.get_mcmc_chain()
        gamma = rv_samples.gamma.median()
        idx = np.argmax(np.abs(self.df_rv.RV1 - gamma))

        return self.df_rv.spec.iloc[idx]

    def write_rv_table(self, outfile):

        df_rv = self.df_rv.copy()
        df_rv['JD'] = [ f'{x:.2f}' for x in df_rv.JD ]
        df_rv['RV1'] = [ f'{x:.1f}' for x in df_rv.RV1 ]
        df_rv['RV1_err'] = [ f'{x:.1f}' for x in df_rv.RV1_err ]

        labels = {}
        if self.time_offset == 0:
            labels['JD'] = [r'$\rm{JD}$', '[d]', 'l']
        else:
            labels['JD'] = [r'$\rm{JD}-'+str(self.time_offset/1e6)+r'\times10^6$', '[d]', 'l']

        labels['RV1'] = ['RV', r'[km/s]', 'r']
        labels['RV1_err'] = [r'$\sigma_{\rm{RV}}$', r'[km/s]', 'r']
        labels['Instrument'] = ['Reference', '', 'r']

        utils.write_latex_table(df_rv, labels, outfile)

class DoubleLinedSpectroscopicBinary(SpectroscopicBinary):

    def measure_rvs(self, template1, template2):

        #temporarily turn off verbose for spectra
        current_verbose = self.verbose
        self.verbose = False
        for spec in tqdm(self.spectra):
            spec.rv_pipeline(template1, template2)
        self.verbose = current_verbose

        self.build_rv_table()


    def build_rv_table(self, method='todcor'):
        
        jds = []
        rv1s = []
        rv2s = []
        rv_errs1 = []
        rv_errs2 = []
        instruments = []
        specs = []

        assert(method in ['todcor', 'ccf'])

        spec_lists = [self.pepsi_spectra, 
                      self.apf_spectra, 
                      self.chiron_spectra]
        instrument_list = ['PEPSI', 'APF', 'CHIRON']
        for spec_list, instrument in zip(spec_lists, instrument_list):
            for spec in spec_list:

                if method == 'todcor':
                    if hasattr(spec, 'slice1'):
                        jds.append(spec.JD)

                        #Add RV offset for APF data
                        if isinstance(spec, spectra.APFspec):
                            rv1s.append(spec.slice1.rv - 2.36)
                            rv2s.append(spec.slice2.rv - 2.36)
                        else:
                            rv1s.append(spec.slice1.rv)
                            rv2s.append(spec.slice2.rv)
                        rv_errs1.append(spec.slice1.rv_err)
                        rv_errs2.append(spec.slice2.rv_err)
                        instruments.append(instrument)
                        specs.append(spec)
                else:
                    jds.append(spec.JD)

                    #Add RV offset for APF data
                    if isinstance(spec, spectra.APFspec):
                        rv1s.append(spec.ccf.models[0].mu - 2.36)
                        if len(spec.ccf.models) > 1:
                            rv2s.append(spec.ccf.models[1].mu - 2.36)
                        else:
                            rv2s.append(np.nan)
                    else:
                        rv1s.append(spec.ccf.models[0].mu)
                        if len(spec.ccf.models) > 1:
                            rv2s.append(spec.ccf.models[1].mu)
                        else:
                            rv2s.append(np.nan)
                    rv_errs1.append(spec.ccf.models[0].emu)
                    if len(spec.ccf.models) > 1:
                        rv_errs2.append(spec.ccf.models[1].emu)
                    else:
                        rv_errs2.append(np.nan)
                    instruments.append(instrument)
                    specs.append(spec)


        self.df_rv = pd.DataFrame({'JD': jds, 
                                   'RV1': rv1s, 
                                   'RV2': rv2s,
                                   'RV1_err': rv_errs1,
                                   'RV2_err': rv_errs2,
                                   'Instrument':instruments, 
                                   'spec':specs})


        self.df_rv = self.df_rv.sort_values(by='JD', ascending=True)
        self.df_rv = self.df_rv.reset_index(drop=True)

        self.df_rv['JD'] = self.df_rv.JD - 2.46e6

        self.instrument_list = self.df_rv.Instrument.value_counts().index

    def flip_rv(self, i):
        
        if not utils.check_iter(i):
            i = [i]

        for ii in i:
            rv1 = self.df_rv.RV1.iloc[ii]
            self.df_rv.RV1.iat[ii] = self.df_rv.RV2.iloc[ii]
            self.df_rv.RV2.iat[ii] = rv1

    def flip_all_rvs(self, which_rv_higher):
        
        idx_to_flip = []
        for i in range(len(self.df_rv)):
            
            RV1 = self.df_rv.RV2.iloc[i]
            RV2 = self.df_rv.RV1.iloc[i]
            if which_rv_higher[i] == 1:
                if RV1 > RV2:
                    continue
                else:
                    idx_to_flip.append(i)
            elif which_rv_higher[i] == 2:
                if RV2 > RV1:
                    continue
                else:
                    idx_to_flip.append(i)
            else:
                raise ValueError(
                        f'invalid max RV component {which_rv_higher[i]}')

        self.flip_rv(idx_to_flip)

    def plot_rvs(self, ax=None, savefig=None):

        fig, ax, created_fig = plotutils.fig_init(ax=ax, figsize=(8, 6))
        if created_fig:
            fig.subplots_adjust(top=.98, right=.98)
        
        for instrument in self.instrument_list:
            
            df_plot = self.df_rv[self.df_rv.Instrument == instrument]

            ax.errorbar(df_plot.JD, df_plot.RV1, yerr=df_plot.RV1_err,
                        color='black', marker=markers[instrument],
                        ls='')

            ax.errorbar(df_plot.JD, df_plot.RV2, yerr=df_plot.RV2_err,
                        color='xkcd:red', marker=markers[instrument],
                        ls='')

        if self.time_offset == 0:
            ax.set_xlabel(r'$\rm{JD}$', fontsize=20)
        else:
            ax.set_xlabel(r'$\rm{JD}-'+str(self.time_offset/1e6)+r'\times10^6$', fontsize=20)
        ax.set_ylabel(r'Radial Velocity (km/s)', fontsize=20)

        return plotutils.plt_return(created_fig, fig, ax, savefig)

    def fit_rvs(self, guess, niters=50000, 
                idx_mask=None, **lnprob_kwargs):

        '''
        fit rvs for double-lined spectroscopic binary
        '''
        
        K1, K2, gamma, M0, ecc, omega, period = guess

        pos = [K1, K2, gamma, M0, ecc, omega, period, -5, -5]

        pos = pos + [ 0 for i in range(len(self.instrument_list)-1) ]

        nwalkers = len(pos)*2
        pos = np.array(pos)+1e-3*np.random.randn(nwalkers, len(pos))

        nwalkers, ndim = pos.shape

        #Mask specific RVs from orbit fit
        mask_rv = np.zeros(len(self.df_rv))
        if idx_mask is not None:
            mask_rv[idx_mask] = 1
        self.df_rv['mask_rv'] = mask_rv

        lnprob_args = (
                tuple([ self.df_rv[(self.df_rv.Instrument == instrument) & 
                                   (self.df_rv.mask_rv == 0)].JD.values
                        for instrument in self.instrument_list ]),
                tuple([ self.df_rv[(self.df_rv.Instrument == instrument) & 
                                   (self.df_rv.mask_rv == 0)].RV1.values
                        for instrument in self.instrument_list ]),
                tuple([ self.df_rv[(self.df_rv.Instrument == instrument) & 
                                   (self.df_rv.mask_rv == 0)].RV1_err.values
                        for instrument in self.instrument_list ]),
                tuple([ self.df_rv[(self.df_rv.Instrument == instrument) & 
                                   (self.df_rv.mask_rv == 0)].RV2.values
                        for instrument in self.instrument_list ]),
                tuple([ self.df_rv[(self.df_rv.Instrument == instrument) & 
                                   (self.df_rv.mask_rv == 0)].RV2_err.values
                        for instrument in self.instrument_list ]))

        lnprob_kwargs.setdefault('period_mu', period)
        lnprob_kwargs.setdefault('period_sigma', period*0.1)

        self.rv_sampler = emcee.EnsembleSampler(
                nwalkers, ndim, rv_orbit_fitter.lnprob_sb2,
                args=lnprob_args,
                kwargs=lnprob_kwargs)

        self.rv_sampler.run_mcmc(pos, niters, progress=self.verbose)

    def get_mcmc_chain(self):

        samples, lnprob = self._get_flat_mcmc_samples()

        mcmc_cols = ['K1', 'K2', 'gamma', 'M0', 'ecc', 'omega', 'period', 'logs1', 'logs2']
        mcmc_cols = mcmc_cols + [ f'rv_offset_{i}' for i in range(len(self.instrument_list)-1) ]

        rv_samples = pd.DataFrame(samples, columns=mcmc_cols)

        rv_samples['T0'] = rv_samples.M0 * rv_samples.period / (2*np.pi)

        self.calculate_rv_chi2()

        return rv_samples, lnprob

    def plot_mcmc_trace(self, savefig=None):
        
        mcmc_cols = ['K1', 'K2', 'gamma', 'M0', 'ecc', 'omega', 'period', 'logs1', 'logs2']
        mcmc_cols = mcmc_cols + [ f'rv_offset_{i}' for i in range(len(self.instrument_list)-1) ]

        super().plot_mcmc_trace(savefig=savefig, labels=mcmc_cols)


    def plot_rv_corner(self, savefig=None, figsize=None):
        
        samples, lnprob = self.get_mcmc_chain()
        samples = samples.copy()
        samples['M0'] = samples.T0
        samples = samples.drop(columns=['T0'], errors='ignore')


        labels = [r'$K_1\ \rm{(km/s)}$', r'$K_2\ \rm{(km/s)}$', 
                  r'$\gamma\ \rm{(km/s)}$', r'$T_0$', 'Ecc', 
                  r'$\omega\ (\rm{rad})$', r'$P \rm{(d)}$',
                  r'$\log s_1$', '$\log s_2$']

        units = [r'$\rm{(km/s)}$', r'$\rm{(km/s)}$', r'$\rm{(km/s)}$',
                 r'$(\rm{d})$', '', r'$(\rm{rad})$', r'$(\rm{d})$', 
                 r'$\rm{(km/s)}$',r'$\rm{(km/s)}$']


        for i in range(len(self.instrument_list)-1):
            instrument = self.instrument_list[i]
            labels.append(r'$\delta\rm{RV}_{\textnormal{\small '+instrument+r'}}(\rm{km/s})$')
            #labels.append(r'$\delta\rm{RV}_{'+str(i)+r'}\ (\rm{km/s})$')
            units.append(r'$\rm{(km/s)}$')

        if figsize is None:
            n = len(labels)+4
            figsize=(n,n)

        fig, ax = plt.subplots(samples.shape[1], samples.shape[1], figsize=figsize)

        fig = corner.corner(samples, labels=labels, fig=fig,
                            hist_kwargs=dict(lw=3, color='black'),
                            quantiles=[0.5],
                            label_kwargs=dict(fontsize=22))

        for a in ax.reshape(-1):
            a = plotutils.plotparams(a)

        #Set titles
        for i in range(samples.shape[1]):
            
            med_val = np.median(samples[samples.columns[i]])
            lower = np.quantile(samples[samples.columns[i]], 0.16)
            upper = np.quantile(samples[samples.columns[i]], 0.84)

            lower_err = med_val - lower
            upper_err = upper - med_val

            title = utils.round_val_err(med_val, lower_err, upper_err) + ' ' + units[i]

            ax[i,i].set_title(title, fontsize=12)

        if savefig is None:
            plt.show()
        else:
            fig.savefig(savefig)

    def plot_rv_orbit(self, ax=None, savefig=None, legend=False, legend_kwargs=None,
                          markers_dict=None):
        
        samples, _ = self.get_mcmc_chain()

        fig, ax, created_fig = plotutils.fig_init(ax=ax, figsize=(10, 6))
        if created_fig:
            fig.subplots_adjust(top=.98, right=.98)

        tvals = np.linspace(self.df_rv.JD.min()-10, self.df_rv.JD.max()+10,
                            int(1e4))

        rv_offsets = [0] + [ rv_samples[f'rv_offset_{i-1}'].median()
                             for i in range(1, len(self.instrument_list)) ]

        if markers_dict is None:
            markers_dict = plotutils.create_markers_dict(self.instrument_list) 
        
        handles = []
        for instrument, offset in zip(self.instrument_list, rv_offsets):
            
            df_plot = self.df_rv[(self.df_rv.Instrument == instrument) & 
                                 (self.df_rv.mask_rv == 0)]
            df_plot_masked = self.df_rv[(self.df_rv.Instrument == instrument) & 
                                        (self.df_rv.mask_rv == 1)]

            e1 = ax.errorbar(df_plot.JD, df_plot.RV1 - offset,
                        yerr=df_plot.RV1_err,
                        color='white', markeredgecolor='xkcd:red', mew=2,
                        ecolor='xkcd:red', 
                        marker=markers_dict[instrument], ls='')

            e2 = ax.errorbar(df_plot.JD, df_plot.RV2 - offset,
                        yerr=df_plot.RV2_err,
                        color='white', markeredgecolor='black', mew=2, 
                        ecolor='black',
                        marker=markers_dict[instrument], ls='')
            handles.append((e1,e2))

            if len(df_plot_masked):
                ax.errorbar(df_plot_masked.JD, df_plot_masked.RV1 - offset,
                            yerr=df_plot_masked.RV1_err,
                            color='white', markeredgecolor='xkcd:red', mew=1,
                            ecolor='xkcd:red', 
                            alpha=0.5,
                            marker=markers_dict[instrument], ls='')

                ax.errorbar(df_plot_masked.JD, df_plot_masked.RV2 - offset,
                            yerr=df_plot_masked.RV2_err,
                            color='white', markeredgecolor='black', mew=1, 
                            ecolor='black',
                            alpha=0.5,
                            marker=markers_dict[instrument], ls='')
                

        for i in range(100):
            
            sample = rv_samples.iloc[np.random.randint(0, len(rv_samples))].to_numpy()
            K1, K2, gamma, phi0, ecc, omega, period = sample[:7]

            model1 = rv_orbit_fitter.rv_model(
                    tvals, K1, gamma, phi0, ecc, omega, period)
            model2 = rv_orbit_fitter.rv_model(
                    tvals, -1*K2, gamma, phi0, ecc, omega, period)

            ax.plot(tvals, model1, color='black', lw=1, alpha=0.2, zorder=1)
            ax.plot(tvals, model2, color='xkcd:red', lw=1, alpha=0.2, zorder=1)

        if self.time_offset == 0:
            ax.set_xlabel(r'$\rm{JD}$', fontsize=20)
        else:
            ax.set_xlabel(r'$\rm{JD}-'+str(self.time_offset/1e6)+r'\times10^6$', fontsize=20)
        ax.set_ylabel(r'$\rm{RV}\ (\rm{km/s})$', fontsize=20)

        if legend:
            if legend_kwargs is None:
                legend_kwargs = {}
            legend_kwargs.setdefault('fontsize', 15)
            legend_kwargs.setdefault('edgecolor', 'black')
            legend_kwargs.setdefault('loc', 'lower left')

            if isinstance(ax, brokenaxes.BrokenAxes):

                if legend_kwargs['loc'].split()[1] == 'left':
                    axs = ax.axs[0]
                else:
                    axs =ax.axs[-1]

                ylim = axs.get_ylim()
                xlim = axs.get_xlim()

                handles = []
                for instrument in self.instrument_list:
                    
                    df_plot = self.df_rv[self.df_rv.Instrument == instrument]
                    e1 = axs.errorbar([-99], [99], [10], color='white', 
                                      markeredgecolor='xkcd:red', mew=2,
                                      ls='', marker=markers_dict[instrument])
                    e2 = axs.errorbar([-99], [99], [10], color='white', 
                                      markeredgecolor='black', mew=2,
                                      ls='', marker=markers_dict[instrument])

                    handles.append((e1,e2))

                axs.set_xlim(xlim)
                axs.set_ylim(ylim)

                axs.legend(handles, self.instrument_list, 
                          handler_map={tuple: HandlerTuple(ndivide=None)},
                          **legend_kwargs)

            else:
                ax.legend(handles, self.instrument_list, 
                          handler_map={tuple: HandlerTuple(ndivide=None)},
                          **legend_kwargs)


        return plotutils.plt_return(created_fig, fig, ax, savefig)


    def calculate_rv_residuals(self, idx=None):

        if not hasattr(self, 'rv_samples'):
            raise ValueError('RV orbit must be fit first')

        rv_samples, _ = self.get_mcmc_chain()

        #Get sample corresponding to the median period
        if idx is None:
            idx = np.argsort(rv_samples.period.values)[len(rv_samples)//2]
        sample = rv_samples.iloc[idx].to_numpy()

        K1, K2, gamma, phi0, ecc, omega, period, logs1, logs2 = sample[:9]
        rv_offset_values = np.concatenate([np.array([0]), sample[9:]])
        rv1_offset_corrected = []
        rv2_offset_corrected = []

        for i in range(len(self.df_rv)):
            
            instrument = self.df_rv.Instrument.iloc[i]
            j = np.where(np.asarray(self.instrument_list) == instrument)[0][0]
            offset = rv_offset_values[j]

            rv1_offset_corrected.append(self.df_rv.RV1.iloc[i] - offset)
            rv2_offset_corrected.append(self.df_rv.RV2.iloc[i] - offset)

        self.df_rv['RV1_offset'] = rv1_offset_corrected
        self.df_rv['RV2_offset'] = rv2_offset_corrected

        self.df_rv['model1'] = rv_orbit_fitter.rv_model(
                self.df_rv.JD.values, K1, gamma, phi0, ecc, omega, period)
        self.df_rv['model2'] = rv_orbit_fitter.rv_model(
                self.df_rv.JD.values, -1*K2, gamma, phi0, ecc, omega, period)

        self.df_rv['residual_1'] = self.df_rv.RV1_offset - self.df_rv.model1
        self.df_rv['residual_2'] = self.df_rv.RV2_offset - self.df_rv.model2

        s1 = np.exp(logs1)
        s2 = np.exp(logs2)
        self.df_rv['RV1_err_jitter'] = np.sqrt(
                np.power(self.df_rv.RV1_err, 2)+s1**2)
        self.df_rv['RV2_err_jitter'] = np.sqrt(
                np.power(self.df_rv.RV2_err, 2)+s2**2)

        #Calculate RV chi2
        residuals = np.concatenate([self.df_rv.residual_1.values, 
                                    self.df_rv.residual_2.values])
        errors = np.concatenate([self.df_rv.RV1_err.values,
                                 self.df_rv.RV2_err.values])
        errors_with_jitter = np.concatenate([self.df_rv.RV1_err_jitter.values,
                                             self.df_rv.RV2_err_jitter.values])

        return residuals, errors, errors_with_jitter

    def calculate_rv_chi2(self):

        residuals, errors, errors_with_jitter = self.calculate_rv_residuals()

        idx_include = np.where(self.df_rv.mask_rv == 0)[0]

        self.rv_chi2_1 = np.sum(np.power(self.df_rv.residual_1[idx_include]/self.df_rv.RV1_err[idx_include],2))
        self.rv_chi2_2 = np.sum(np.power(self.df_rv.residual_2[idx_include]/self.df_rv.RV2_err[idx_include],2))

        self.rv_chi2_jitter_1 = np.sum(np.power(self.df_rv.residual_1[idx_include]/self.df_rv.RV1_err_jitter[idx_include],2))
        self.rv_chi2_jitter_2 = np.sum(np.power(self.df_rv.residual_2[idx_include]/self.df_rv.RV2_err_jitter[idx_include],2))

        dof = 6 #K1, gamma, M0, ecc, omega, s1 (period is constrained from LC)
        
        self.rv_chi2_nu_1 = self.rv_chi2_1 / (len(self.df_rv.residual_1[idx_include]) - dof)
        self.rv_chi2_nu_jitter_1 = self.rv_chi2_jitter_1 / (len(self.df_rv.residual_1[idx_include]) - dof)

        self.rv_chi2_nu_2 = self.rv_chi2_2 / (len(self.df_rv.residual_2[idx_include]) - dof)
        self.rv_chi2_nu_jitter_2 = self.rv_chi2_jitter_2 / (len(self.df_rv.residual_2[idx_include]) - dof)

        return self.rv_chi2_nu_1, self.rv_chi2_nu_2


    def plot_rv_residuals(self, with_jitter=False, plot_masked=True,
                          ax=None, savefig=None, legend=False, legend_kwargs=None,
                          markers_dict=None, label_chi2=True):

        if 'residual_1' not in self.df_rv.columns:
            self.calculate_rv_residuals()

        fig, ax, created_fig = plotutils.fig_init(ax=ax, figsize=(8, 6))
        if created_fig:
            fig.subplots_adjust(top=.98, right=.98)

        if with_jitter:
            yerr_columns = ('RV1_err_jitter', 'RV2_err_jitter')
        else:
            yerr_columns = ('RV1_err', 'RV2_err')

        if markers_dict is None:
            markers_dict = plotutils.create_markers_dict(self.instrument_list)

        handles = []
        for instrument in self.instrument_list:

            df_plot = self.df_rv[(self.df_rv.Instrument == instrument) & 
                                 (self.df_rv.mask_rv == 0)]
            df_plot_masked = self.df_rv[(self.df_rv.Instrument == instrument) & 
                                        (self.df_rv.mask_rv == 1)]
            
            e1 = ax.errorbar(df_plot.JD, df_plot.residual_1, 
                             yerr=df_plot[yerr_columns[0]],
                             color='black', marker=markers_dict[instrument],
                             ls='')

            e2 = ax.errorbar(df_plot.JD, df_plot.residual_2, 
                             yerr=df_plot[yerr_columns[1]],
                             color='xkcd:red', marker=markers_dict[instrument],
                             ls='')

            handles.append((e1, e2))

            if len(df_plot_masked) and plot_masked:

                ax.errorbar(df_plot_masked.JD, df_plot_masked.residual_1, 
                            yerr=df_plot_masked[yerr_columns[0]],
                            color='black', marker=markers_dict[instrument], 
                            markeredgecolor='none',
                            ls='', alpha=0.3)

                ax.errorbar(df_plot_masked.JD, df_plot_masked.residual_2, 
                            yerr=df_plot_masked[yerr_columns[1]],
                            color='xkcd:red', marker=markers_dict[instrument], 
                            markeredgecolor='none',
                            ls='', alpha=0.3)

        if self.time_offset == 0:
            ax.set_xlabel(r'$\rm{JD}$', fontsize=20)
        else:
            ax.set_xlabel(r'$\rm{JD}-'+str(self.time_offset/1e6)+r'\times10^6$', fontsize=20)
        ax.set_ylabel(r'Residuals (km/s)', fontsize=20)

        ax.axhline(0.0, color='gray', ls='--')

        ylim = np.max(np.abs(ax.get_ylim()))
        ax.set_ylim(-1*ylim, ylim)

        if legend:
            if legend_kwargs is None:
                legend_kwargs = {}
            legend_kwargs.setdefault('fontsize', 15)
            legend_kwargs.setdefault('edgecolor', 'black')
            legend_kwargs.setdefault('loc', 'lower left')

            ax.legend(handles, self.instrument_list, 
                      handler_map={tuple: HandlerTuple(ndivide=None)},
                      **legend_kwargs)

        if label_chi2:
            if with_jitter:
                chi2_vals = [ self.rv_chi2_nu_jitter_1,
                              self.rv_chi2_nu_jitter_2 ]
            else:
                chi2_vals = [ self.rv_chi2_nu_1,
                              self.rv_chi2_nu_2 ]

            label = r'$\chi^2_{\nu,1} = '+str(round(chi2_vals[0],2))+r'$'
            label += r'$\chi^2_{\nu,2} = '+str(round(chi2_vals[1],2))+r'$'

            ax.text(.95, .95, label,
                    ha='right', va='top', 
                    fontsize=15, transform=ax.transAxes)

        return plotutils.plt_return(created_fig, fig, ax, savefig)

    def get_quadrature_spec(self):
        
        delta_rv = np.abs(self.df_rv.RV1 - self.df_rv.RV2)
        idx = np.argmax(delta_rv)

        return self.df_rv.spec.iloc[idx]

    def run_fdbinary(self, cleanup=True, use_mp=False, alpha=None, 
                     dont_resample=False):

        #Remove gamma offset
        self.correct_system_velocity()

        #Check that we have all the same type of echelle spectra
        first_type = type(self.echelle_spectra[0])
        if not all(type(spec) == first_type for spec in self.echelle_spectra):
            raise TypeError('Multiple echelle spectra detected')

        #Check that all echelle spectra have same orders
        echelle_orders = [ O.n for O in self.echelle_spectra[0].Orders ]
        for spec in self.echelle_spectra[1:]:
            spec_orders = [ O.n for O in spec.Orders ]
            if spec_orders != echelle_orders:
                raise ValueError('Echelle orders do not match for all spectra')

        #Get wavelength range and spacing for each order
        wmins = []
        wmaxs = []
        dlambdas = []

        for i in range(len(echelle_orders)):

            wmins.append(np.max([
                    spec.Orders[i].df.wavelength.min() 
                    for spec in self.echelle_spectra]))
            wmaxs.append(np.min([
                    spec.Orders[i].df.wavelength.max() 
                    for spec in self.echelle_spectra]))
            dlambdas.append(np.max([
                    spec.Orders[i].get_wavelength_spacing() 
                    for spec in self.echelle_spectra]))

        if dont_resample:
            pass
        else:
            #Resample echelle spectra on log wavelength grid
            for spec in self.echelle_spectra:
                
                for i, Order in enumerate(spec.Orders):
                    
                    wmin = wmins[i]
                    wmax = wmaxs[i]
                    dlambda = dlambdas[i]

                    Order.resample_log_wavelength(wmin, wmax, dlambda)

        #Mark which spectra should be omitted from disentangling
        for i in range(len(self.df_rv)):
            if self.df_rv.mask_rv.iloc[i] == 1.0:
                self.df_rv.spec.iloc[i].fdbinary_mask = True

        #Convert PEPSI spectra to echelle spectra on same wavelength grid
        if len(self.pepsi_spectra):
            for i in range(len(self.pepsi_spectra)):
                #Flag to check if it has already been converted
                if ((not hasattr(self.pepsi_spectra[i], '_converted')) or 
                    (not self.pepsi_spectra[i]._converted)):

                    self.pepsi_spectra[i] = spectrum_utils.convert_pepsi_to_echelle(
                            self.pepsi_spectra[i], wmins, wmaxs, dlambdas, 
                            log=True, orders=echelle_orders)


        n_masked = 0
        for spec in self.spectra:
            if spec.fdbinary_mask:
                n_masked += 1
        print(f'{n_masked} spectra masked during fdbinary')
                
        #Write obs files
        obs_fnames = []
        jd_skips = []
        for n in echelle_orders:

            log_wavelength = None
            ni_lw = 0 #index to get log_wavelength
            while (log_wavelength is None) and (ni_lw < len(self.echelle_spectra)):
                if self.echelle_spectra[ni_lw].fdbinary_mask:
                    ni_lw += 1
                else:
                    log_wavelength = self.echelle_spectra[0].get_order(n).df.log_wavelength.values
            if log_wavelength is None:
                raise ValueError('Unable to identify unmasked echelle spectra to set wavelength scale')

            fluxes = [ log_wavelength ]

            jd_skip = []

            for spec in self.spectra:
                
                Order = spec.get_order(n)
                if Order is not None: 
                    fdbinary_mask = Order.fdbinary_mask
                else:
                    fdbinary_mask = True #doesn't actually matter for the below line
                if (Order is not None) and (not fdbinary_mask):
                    if not np.all(Order.df.log_wavelength.values == log_wavelength):
                        raise ValueError('Failure to create uniform wavelength grid')
                    else:
                        fluxes.append(Order.df.flux.values)
                else:
                    #Need to keep track of the spectra that are missing for each order
                    jd_skip.append(spec.JD)

            fluxes = np.array(fluxes)

            #Write file
            obs_file = f'{self.name}_fd3_{n}.obs'
            np.savetxt(obs_file, fluxes.T, delimiter='\t')

            #Add line for shape
            with open(obs_file, 'r') as f:
                lines = f.readlines()
            lines.insert(0, f'# {fluxes.shape[0]} X {len(log_wavelength)} \n')

            #Re-write file
            with open(obs_file, 'w') as f:
                f.writelines(lines)

            obs_fnames.append(obs_file)
            jd_skips.append(jd_skip)

        #Write input files
        input_fnames = []


        rv_samples, _ = self.get_mcmc_chain()

        period = rv_samples.period.median()
        M0 = rv_samples.M0.median()
        #tper = M0*period/(2*np.pi)
        #tper_err = 0
        tper = self.df_rv.JD.median()
        tper_err = period/2
        ecc = rv_samples.ecc.median()
        omega = rv_samples.omega.median() * 180/np.pi
        K1 = rv_samples.K1.median()
        K2 = rv_samples.K2.median()

        #Calculate flux ratio
        if alpha is None:
            alpha_list = []
            for spec in self.spectra:
                if not spec.fdbinary_mask:
                    idx_rv = spec.todcor_profile.get_rvs(return_idx=True)
                    alpha = spec.todcor_profile.alpha_profile.arr[idx_rv]
                    alpha_list.append(alpha)
            alpha = np.mean(alpha_list) #F2/F1
        f1 = 1/(1+alpha)
        f2 = 1/(1/alpha + 1)

        print(alpha, f1, f2)

        for i, n in enumerate(echelle_orders):
            
            input_file = f'{self.name}_fd3_{n}.in'
            output_head = f'{self.name}_fd3_{n}'
            jd_skip = jd_skips[i]
            caught_skip = 0 #Check that all the skipped JDs get caught

            with open(input_file, 'w') as f:
                 
                f.write(f'{obs_fnames[i]} {np.log(wmins[i])} {np.log(wmaxs[i])} {output_head} 1 1 0')
                f.write('\n\n')

                for spec in self.spectra:
                    if spec.JD not in jd_skip:
                        f.write(f'{spec.JD - 2.46e6} 0 0.01 {f1} {f2}')
                        f.write('\n')
                    else:
                        caught_skip += 1
                f.write('\n')

                f.write('1 0   0 0   0 0   0 0   0 0   0 0\n')
                f.write(f'{period} 0.0 {tper} {tper_err} {ecc} 0.0 {omega} 0.0 {K1} 0.0 {K2} 0.0 0 0'+'\n')
                f.write('100 1000 0.00001')

            if caught_skip != len(jd_skip):
                raise ValueError('Not all JD skips caught')
            input_fnames.append(input_file)

        #Run FDBinary
        if use_mp:
            nprocesses = np.min([mp.cpu_count(), len(input_fnames)])
            pool = mp.Pool(processes=nprocesses)
            jobs = []
            iterator = range(len(input_fnames))
            
            def callback(result):
                progress_bar.update(1)

            progress_bar = tqdm(total=len(input_fnames))
        else:
            iterator = tqdm(range(len(input_fnames)))

        script_dir = os.path.dirname(os.path.abspath(__file__))
        fd3_path = os.path.join(script_dir, 'fd3')

        for i in iterator:

            if use_mp:
                job = pool.apply_async(
                        spectrum_utils.fd3_worker, 
                        (fd3_path, input_fnames[i],),
                        callback=callback)
                jobs.append(job)
            else:
                spectrum_utils.fd3_worker(fd3_path, input_fnames[i], 
                                          verbose=self.verbose)

        if use_mp:
            for job in jobs:
                job.get()
                
        #Create FDBinaryResult object
        self.fdbinary = fdbinary.FDBinaryResult(
                input_fnames, verbose=self.verbose,
                flux_ratio=alpha)
        if cleanup:
            self.fdbinary.cleanup()

