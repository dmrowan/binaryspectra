#!/usr/bin/env python

import corner
import emcee
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
import pandas as pd
from tqdm.autonotebook import tqdm

from . import astrometric_orbit_fitter
from . import binarytarget
from . import plotutils
from . import rv_orbit_fitter
from . import utils

#Dom Rowan 2025

class AstrometricSB1(binarytarget.SingleLinedSpectroscopicBinary):

    def set_distance(self, value):
        self.distance = value
    
    def load_ast_table(self, fname, time_offset=0):
        
        if isinstance(fname, pd.core.frame.DataFrame):
            self.df_ast = fname
        else:
            self.df_ast = pd.read_csv(fname)

        if hasattr(self, 'time_offset') and time_offset == self.time_offset:
            self.df_ast['JD'] = self.df_ast.JD - self.time_offset
        else:
            raise NotImplementedError('Different time offset for RV and AST passed')

    def plot_ast(self, ax=None, savefig=None, **plot_kwargs):
        
        fig, ax, created_fig = plotutils.fig_init(ax=ax, figsize=(6, 6))

        plot_kwargs.setdefault('color', 'black')
        plot_kwargs.setdefault('marker', '.')

        ax.scatter(self.df_ast.xpos, self.df_ast.ypos, **plot_kwargs)
        ax.scatter([0], [0], color='xkcd:red', marker='*')

        lim = np.max(np.abs(ax.get_xlim() + ax.get_ylim()))*1.1
        ax.set_xlim(-1*lim, lim)
        ax.set_ylim(-1*lim, lim)

        ax.set_xlabel(r'$\Delta x$ [mas]', fontsize=20)
        ax.set_ylabel(r'$\Delta y$ [mas]', fontsize=20)

        return plotutils.plt_return(created_fig, fig, ax, savefig)


    def fit_ast(self, guess, niters=50000, burnin=10000,
                  idx_mask=None, **lnprob_kwargs):

        pass

    def fit_ast_rvs(self, guess, niters=50000, burnin=10000,
                    idx_mask=None, **lnprob_kwargs):

        K1, gamma, M0, ecc, omega, period, A, B, F, G = guess
        logs = -5

        pos = [K1, gamma, M0, ecc, omega, period, logs, A, B, F, G]
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
                self.df_ast.JD.values,
                self.df_ast.xpos.values,
                self.df_ast.ypos.values,
                self.df_ast.xerr.values,
                self.df_ast.yerr.values,
                self.distance)

        lnprob_kwargs.setdefault('period_mu', period)
        lnprob_kwargs.setdefault('period_sigma', period*0.1)

        sampler = emcee.EnsembleSampler(
                nwalkers, ndim, astrometric_orbit_fitter.lnprob_sb1_astrometry,
                args=lnprob_args,
                kwargs=lnprob_kwargs)

        sampler.run_mcmc(pos, niters, progress=self.verbose)
        samples = sampler.get_chain(discard=burnin, flat=True)
        
        ast_rv_samples = pd.DataFrame(samples)

        columns = ['K1', 'gamma', 'phi0', 'ecc', 'omega', 'period', 'logs', 
                   'A_TI', 'B_TI', 'F_TI', 'G_TI']
        
        columns = columns + [ f'rv_offset_{i}' for i in range(len(self.instrument_list)-1) ]
        ast_rv_samples.columns = columns

        self.ast_rv_samples = ast_rv_samples

    def plot_ast_corner(self, savefig=None, figsize=None):
        
        pass

    def plot_ast_rv_corner(self, savefig=None, figsize=None,
                           gaia_priors=False,
                           offset_numeric_label=True):
            
        if not hasattr(self, 'ast_rv_samples'):
            raise ValueError('Ast+RV orbit must be fit first')

        samples = self.ast_rv_samples.copy()

        labels = [r'$K_1\ \rm{[km/s]}$', r'$\gamma\ \rm{[km/s]}$',
                  r'$M_0\ \rm{[rad]}$',
                  'Ecc', r'$\omega\ [\rm{rad}]$', r'$P \rm{[d]}$',
                  r'$\log s\ \rm{[km/s]}$',
                  r'A_{\rm{TI}} [mas]', 
                  r'B_{\rm{TI}} [mas]', 
                  r'F_{\rm{TI}} [mas]', 
                  r'G_{\rm{TI}} [mas]']

        units = [r'$\rm{[km/s]}$',
                 r'$\rm{[km/s]}$',
                 r'$[\rm{rad}]$',
                 '',
                 r'$[\rm{rad}]$',
                 r'$[\rm{d}]$',
                 r'$\rm{[km/s]}$',
                 r'$\rm{[mas]}$',
                 r'$\rm{[mas]}$',
                 r'$\rm{[mas]}$',
                 r'$\rm{[mas]}$']

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

            title = utils.round_val_err(
                    med_val, lower_err, upper_err) + ' ' + units[i]

            ax[i,i].set_title(title, fontsize=12)

        if savefig is None:
            plt.show()
        else:
            fig.savefig(savefig)

    def plot_rv_ast_orbit(self, **kwargs):

        if not hasattr(self, 'ast_rv_samples'):
            raise ValueError('Ast+RV orbit must be fit first')

        #backup the rv_samples dict
        if hasattr(self, 'rv_samples'):
            reset_samples = True
            backup = self.rv_samples.copy()

        self.rv_samples = self.ast_rv_samples.copy()

        r = self.plot_rv_orbit(**kwargs)

        if reset_samples:
            self.rv_samples = backup

        return r


    def plot_ast_orbit(self, ax=None, savefig=None, tmin=None, tmax=None,
                       **plot_kwargs):
        
        if not hasattr(self, 'ast_rv_samples'):
            raise ValueError('Ast+RV orbit must be fit first')

        fig, ax, created_fig = plotutils.fig_init(ax=ax, figsize=(6, 6))

        plot_kwargs.setdefault('zorder', 5)

        ax = self.plot_ast(ax=ax, **plot_kwargs)

        if tmin is None:
            tmin = self.df_ast.JD.min()-10
        if tmax is None:
            tmax = self.df_ast.JD.max()+10
        tvals = np.linspace(tmin, tmax, int(1e4))

        for i in range(100):
            
            sample = self.ast_rv_samples.iloc[np.random.randint(0, len(self.ast_rv_samples))].to_numpy()

            K1, gamma, phi0, ecc, omega, period, logs, A, B, F, G = sample[:11]

            xvals, yvals = astrometric_orbit_fitter.compute_xy(tvals, 0, 0, A, B, F, G, period, phi0, ecc)
            ax.plot(xvals*1000/self.distance, yvals*1000/self.distance, 
                    color='xkcd:azure', alpha=0.2, lw=1, zorder=1)

        ax.set_xlabel(r'$\Delta x$ [mas]', fontsize=20)
        ax.set_ylabel(r'$\Delta y$ [mas]', fontsize=20)

        return plotutils.plt_return(created_fig, fig, ax, savefig)





