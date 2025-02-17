#!/usr/bin/env python

from astropy import units as u
from astropy import constants as const
import corner
import emcee
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
import pandas as pd
from tqdm.autonotebook import tqdm

from . import astrospectro_orbit_fitter
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
        ax.invert_xaxis()
        ax.set_ylim(-1*lim, lim)

        ax.set_xlabel(r'$\Delta x$ [mas]', fontsize=20)
        ax.set_ylabel(r'$\Delta y$ [mas]', fontsize=20)

        return plotutils.plt_return(created_fig, fig, ax, savefig)

    def fit_ast_rvs(self, guess, niters=50000, burnin=10000,
                    idx_mask=None, **lnprob_kwargs):

        A, B, F, G, C, H, period, phi0, ecc, gamma = guess
        logs = -5

        pos = [A, B, F, G, C, H, period, phi0, ecc, gamma, logs]
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
                nwalkers, ndim, astrospectro_orbit_fitter.lnprob_astrosb1,
                args=lnprob_args,
                kwargs=lnprob_kwargs)

        sampler.run_mcmc(pos, niters, progress=self.verbose)
        print(sampler)
        samples = sampler.get_chain(discard=burnin, flat=True)
        
        ast_rv_samples = pd.DataFrame(samples)

        columns = ['A_TI', 'B_TI', 'F_TI', 'G_TI', 'C_TI', 'H_TI',
                   'period', 'M0', 'ecc', 'gamma', 'logs']
        
        columns = columns + [ f'rv_offset_{i}' for i in range(len(self.instrument_list)-1) ]
        ast_rv_samples.columns = columns

        self.ast_rv_samples = ast_rv_samples

        #Add columns
        '''
        M1, M2, incl, omega, Omega
        '''
        A = self.ast_rv_samples.A_TI.values
        B = self.ast_rv_samples.B_TI.values
        F = self.ast_rv_samples.F_TI.values
        G = self.ast_rv_samples.G_TI.values

        C = self.ast_rv_samples.C_TI.values
        H = self.ast_rv_samples.H_TI.values

        uu = (A**2 + B**2 + F**2 + G**2) / 2
        v = A*G - B*F

        a = np.sqrt(uu + np.sqrt((uu+v)*(uu-v)))

        omega_p_Omega = np.arctan((B-F)/(A+G)) 
        omega_m_Omega = np.arctan((B+F)/(G-A))

        if np.sin(omega_p_Omega[0])*(B[0]-F[0]) < 1:
            omega_p_Omega += np.pi
        if np.sin(omega_m_Omega[0])*(-B[0]-F[0]) < 1:
            omega_m_Omega += np.pi

        assert(np.sin(omega_p_Omega[0])*(B[0]-F[0]))
        assert(np.sin(omega_m_Omega[0])*(-B[0]-F[0]))

        d1 = np.abs((A+G)*np.cos(omega_m_Omega))
        d2 = np.abs((F-B)*np.sin(omega_m_Omega))

        incl = np.zeros(len(d1), dtype=np.float64)
        for i in range(len(A)):
            if d1[i] >= d2[i]:
                incl[i] = 2*np.arctan(np.sqrt(np.abs((A[i]-G[i])*np.cos(omega_p_Omega[i]))/d1[i]))
            else:
                incl[i] = 2*np.arctan(np.sqrt(np.abs((B[i]+F[i])*np.sin(omega_p_Omega[i]))/d2[i]))

        a1 = np.sqrt(C**2 + H**2) / np.sin(incl)

        period = self.ast_rv_samples.period.values
        M2 = 4*np.pi**2 * np.square(a*u.au) * (a1*u.AU) / (const.G*np.square(period*u.day))
        M2 = M2.to('M_sun').value

        Mtotal = 4*np.pi**2 * np.power(a*u.au, 3) / (const.G*np.square(period*u.day))
        M1 = Mtotal.to('M_sun').value - M2

        self.ast_rv_samples['sma'] = a
        self.ast_rv_samples['a1'] = a1
        self.ast_rv_samples['incl'] = incl * 180/np.pi
        self.ast_rv_samples['M1'] = M1
        self.ast_rv_samples['M2'] = M2

    def plot_ast_rv_corner(self, savefig=None, figsize=None,
                           params=None,
                           gaia_priors=False,
                           offset_numeric_label=True):
            
        if not hasattr(self, 'ast_rv_samples'):
            raise ValueError('Ast+RV orbit must be fit first')

        if params is None:
            params = ['A_TI', 'B_TI', 'F_TI', 'G_TI', 'C_TI', 'H_TI',
                      'period', 'M0', 'ecc', 'gamma', 'logs']

            params.extend([ f'rv_offset_{i}' for i in range(len(self.instrument_list)-1) ])

        samples = self.ast_rv_samples[params].copy()

        labels_units = {'A_TI': [r'A_{\rm{TI}} [mas]', r'$\rm{[mas]}$'],
                        'B_TI': [r'B_{\rm{TI}} [mas]', r'$\rm{[mas]}$'],
                        'F_TI': [r'F_{\rm{TI}} [mas]', r'$\rm{[mas]}$'],
                        'G_TI': [r'G_{\rm{TI}} [mas]', r'$\rm{[mas]}$'],
                        'C_TI': [r'C_{\rm{TI}} [mas]', r'$\rm{[mas]}$'],
                        'H_TI': [r'H_{\rm{TI}} [mas]', r'$\rm{[mas]}$'],
                        'period': [r'$P \rm{[d]}$', r'$[\rm{d}]$'],
                        'M0': [r'$M_0\ \rm{[rad]}$', r'$[\rm{rad}]$'],
                        'ecc': ['Ecc', ''],
                        'gamma': [r'$\gamma\ \rm{[km/s]}$', r'$\rm{[km/s]}$',],
                        'logs': [r'$\log s\ \rm{[km/s]}$', r'$\rm{[km/s]}$'],
                        'M1': [r'$M_1\ [M_\odot]$', r'$[M_\odot]$'],
                        'M2': [r'$M_2\ [M_\odot]$', r'$[M_\odot]$'],
                        'incl': [r'Incl [$^\circ$]', r'$[^\circ]$'],
                        'sma': [r'$a\ \rm{[AU]}$', r'$\rm{[AU]}$'],
                        'a1': [r'$a_1\ \rm{[AU]}$', r'$\rm{[AU]}$']}

        for i in range(len(self.instrument_list)-1):
            
            instrument = self.instrument_list[i]
            if offset_numeric_label:
                labels_units[f'rv_offset_{i}'] = [ r'$\delta\rm{RV}_{'+str(i)+r'}\ \rm{[km/s]}$',
                                                   r'$\rm{(km/s)}$' ]
            else:
                ll = r'$\delta\rm{RV}_{\textnormal{\small '+ instrument+ r'}}\ \rm{[km/s]}$'
                labels_units[f'rv_offset_{i}'] = [ ll, r'$\rm{(km/s)}$' ]

        labels = [ labels_units[p][0] for p in params ]
        units = [ labels_units[p][1] for p in params ]

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

    def plot_rv_ast_orbit(self, ax=None, savefig=None, legend=False, legend_kwargs=None,
                          markers_dict=None, tmin=None, tmax=None):

        if not hasattr(self, 'ast_rv_samples'):
            raise ValueError('Ast+RV orbit must be fit first')

        fig, ax, created_fig = plotutils.fig_init(ax=ax, figsize=(10, 6))
        if created_fig:
            fig.subplots_adjust(top=.98, right=.98)

        if tmin is None:
            tmin = self.df_rv.JD.min()-10
        if tmax is None:
            tmax = self.df_rv.JD.max()+10
        tvals = np.linspace(tmin, tmax, int(1e4))

        rv_offsets = [0] + [ self.ast_rv_samples[f'rv_offset_{i-1}'].median()
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

        for i in range(100):

            sample = self.ast_rv_samples.iloc[np.random.randint(0, len(self.ast_rv_samples))].to_numpy()
    
            A, B, F, G, C, H, period, phi0, ecc, gamma, logs = sample[:11]
            
            model1 = astrospectro_orbit_fitter.rv_model(tvals, C, H, period, phi0, ecc, gamma)
            ax.plot(tvals, model1, color='xkcd:red', lw=1, alpha=0.2, zorder=1)

        if self.time_offset == 0:
            ax.set_xlabel(r'$\rm{JD}$', fontsize=20)
        else:
            ax.set_xlabel(r'$\rm{JD}-'+str(self.time_offset/1e6)+r'\times10^6$', fontsize=20)
            ax.set_ylabel(r'$\rm{RV}\ [\rm{km/s}]$', fontsize=20)

            ax.set_xlim(tmin, tmax)

        if legend:
            if legend_kwargs is None:
                legend_kwargs = {}

            legend_kwargs.setdefault('fontsize', 15)
            legend_kwargs.setdefault('edgecolor', 'black')
            legend_kwargs.setdefault('loc', 'lower left')
            
            ax.legend(**legend_kwargs)

        return plotutils.plt_return(created_fig, fig, ax, savefig)

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
            A, B, F, G, C, H, period, phi0, ecc, gamma, logs = sample[:11]

            xvals, yvals = astrospectro_orbit_fitter.compute_xy(tvals, 0, 0, A, B, F, G, period, phi0, ecc)

            xvals = xvals * 1000/self.distance
            yvals = yvals * 1000/self.distance

            ax.plot(xvals, yvals, color='xkcd:azure', alpha=0.2, lw=1, zorder=1)

        ax.set_xlabel(r'$\Delta x$ [mas]', fontsize=20)
        ax.set_ylabel(r'$\Delta y$ [mas]', fontsize=20)

        return plotutils.plt_return(created_fig, fig, ax, savefig)





