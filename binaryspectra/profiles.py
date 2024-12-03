#!/usr/bin/env python

from astropy import log
import copy
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from scipy import ndimage
import statsmodels.api as sm
import sys

import sys
sys.path.append(os.environ['ISPEC_DIR'])
import ispec
from ispec.lines import __model_velocity_profile as model_velocity_profile
from ispec.modeling.mpfitmodels import GaussianModel
from ispec.common import find_local_min_values, find_local_max_values

from binaryspec import plotutils, utils
from binaryspec import measure_rvs
from binaryspec import spectrum_utils

#Dom Rowan 2024

class BaseProfile:

    def __init__(self, df, nbins=None):

        self.df = df
        self._ycolumn = self.df.columns[1]

        self.nbins = nbins

    def plot(self, fig=None, ax=None, savefig=None, **kwargs):

        fig, ax, created_fig = plotutils.fig_init(ax=ax, figsize=(6, 4))
        if created_fig:
            fig.subplots_adjust(top=.98, right=.98)

        kwargs.setdefault('color', 'black')
        kwargs.setdefault('lw', 2)

        ax.plot(self.df.RV, self.df[self._ycolumn], **kwargs)
        ax.set_xlabel('RV (km/s)', fontsize=20)

        return plotutils.plt_return(created_fig, fig, ax, savefig)

    def interp(self, RV):
        
        return np.interp(RV, self.df.RV.values, self.df[self._ycolumn].values)

    def model(self):
        
        ccf_struct = np.recarray((len(self.df), ), 
                                 dtype=[('x', float),('y', float), ('err', float)])
        ccf_struct['x'] = self.df.RV.values
        ccf_struct['y'] = self.df[self._ycolumn]
        ccf_struct['err'] = self.df.err


        if self.nbins is None:
            raise ValueError("BaseProfile.nbins cannot be none")
        with utils.HiddenPrints():
            self.models = model_velocity_profile(ccf_struct, self.nbins)

            self.models = [ spectrum_utils.convert_ispec_model_to_serializable(
                                    model, self.df.RV.values)
                            for model in self.models ]

        return self.models

    def plot_model(self, fig=None, ax=None, savefig=None, legend=True, model_colors=None, 
                   **kwargs):
        
        fig, ax, created_fig = plotutils.fig_init(ax=ax, figsize=(6, 4))
        if created_fig:
            fig.subplots_adjust(top=.98, right=.98, bottom=0.2)

        ax = self.plot(ax=ax, **kwargs)

        colors = ['xkcd:red', 'xkcd:azure', 'xkcd:violet']

        if model_colors is not None:
            if not utils.check_iter(model_colors):
                model_colors = [model_colors]

            for i in range(len(model_colors)):
                colors[i] = model_colors[i]

        xvals = self.df.RV.values
        for i in range(len(self.models)):
            
            xvals = self.models[i].xvals
            yvals = self.models[i].yvals

            l = utils.round_val_err(self.models[i].mu, self.models[i].emu, 
                                    as_string=True)+' km/s'
            ax.plot(xvals, yvals, color=colors[i], label=l)

        if legend:
            ax.legend(edgecolor='black', fontsize=15)

        return plotutils.plt_return(created_fig, fig, ax, savefig)

    def __len__(self):
        
        return len(self.df)

class CCFProfile(BaseProfile):

    def __init__(self, df, nbins=None, norders=None):
        
        self.norders = norders
        super().__init__(df, nbins=nbins)

    def model(self):

        xvals = self.df.RV.values
        yvals = self.df[self._ycolumn].values

        #Smooth the CCF
        smoothed_vals = ndimage.filters.gaussian_filter1d(yvals, 1)

        #Identify peaks and minima between the peaks
        peaks = find_local_max_values(yvals)
        base_points = find_local_min_values(yvals)

        #Remove duplicates (rare)
        idx_peaks = np.where(np.concatenate([[np.inf], np.diff(peaks)]) > 1)[0]
        idx_bases = np.where(np.concatenate([[np.inf], np.diff(base_points)]) > 1)[0]
        peaks = peaks[idx_peaks]
        base_points = base_points[idx_bases]

        #Identify selected peaks
        x_c = sm.add_constant(xvals, prepend=False)
        huber_t = sm.RLM(yvals, x_c, M=sm.robust.norms.HuberT())
        linear_model = huber_t.fit()

        peak_probability = 0.55
        poly_step = 0.01
        selected_peaks_indices = np.where(linear_model.weights[peaks] < 1. - peak_probability)[0]

        selected_peaks_indices = [ s for s in selected_peaks_indices if s not in [0, len(peaks)-1] ]

        self.models = []

        for i in selected_peaks_indices:
            
            peaks_left = peaks[i]-4
            peaks_right = peaks[i]+5
            
            #First estimate mu using a polynomial fit to the peak
            p = np.poly1d(np.polyfit(xvals[peaks_left:peaks_right], 
                                     yvals[peaks_left:peaks_right], 2))
            poly_vel = np.arange(xvals[peaks_left], xvals[peaks_right]+poly_step, poly_step)
            poly_yvals = p(poly_vel)

            mu = poly_vel[np.argmax(poly_yvals)]
            
            #select the region to do the Gaussian fit
            base_lower_idx = 0
            while xvals[base_points[base_lower_idx+1]] < xvals[peaks[i]]:
                base_lower_idx += 1
            base_upper_idx = base_lower_idx + 1
            
            #Run Gaussian LM fit
            gaussian_model = GaussianModel()

            #Initial values for parameters
            baseline = np.median(yvals[base_points])
            A = yvals[peaks[i]] - baseline
            #sig = np.abs(xvals[base_points[base_lower_idx]] - xvals[base_points[base_upper_idx]])/3.0
            sig = 10

            parinfo = [{'value':0., 'fixed':False, 
                        'limited':[False, False], 
                        'limits':[0., 0.]} for j in np.arange(4)]

            #Baseline
            parinfo[0]['value'] = baseline
            parinfo[0]['fixed'] = False
            parinfo[0]['limited'] = [True, True]
            parinfo[0]['limits'] = [ baseline - 5*np.std(yvals[base_points]),
                                     baseline + 5*np.std(yvals[base_points])]
            
            #Amplitude
            parinfo[1]['value'] = A 
            parinfo[1]['limited'] = [False, True]
            parinfo[1]['limits'] = [0., 1.]

            #Sigma
            parinfo[2]['value'] = sig 
            parinfo[2]['limited'] = [True, False]
            parinfo[2]['limits'] = [1.0e-10, 0.]

            #Mu
            parinfo[3]['value'] = mu 
            parinfo[3]['fixed'] = False
            parinfo[3]['limited'] = [True, True]
            parinfo[3]['limits'] = [mu-poly_step, mu+poly_step]
            
            #Weights
            f = yvals[base_points[base_lower_idx]:base_points[base_upper_idx]]
            max_flux = np.max(f)
            weights = f/max_flux
            weights -= np.min(weights)
            weights = weights/np.max(weights)

            gaussian_model.fitData(
                    xvals[base_points[base_lower_idx]:base_points[base_upper_idx]], 
                    yvals[base_points[base_lower_idx]:base_points[base_upper_idx]],
                    parinfo=copy.deepcopy(parinfo), weights=weights)

            
            #Calculate uncertainty
            sep = xvals[1] - xvals[0]
            first_derivative = np.gradient(yvals, sep)
            second_derivative = np.gradient(first_derivative, sep)

            peak = xvals.searchsorted(gaussian_model.mu())
            yvals_peak = yvals[peak]
            second_derivative_peak = second_derivative[peak]
            
            sharpness = second_derivative_peak / yvals_peak
            line_snr = yvals_peak**2 / (1-yvals_peak**2)

            
            M = self.norders
            N = self.nbins

            if M is None:
                M = 1
            
            rv_err = np.sqrt( -1*np.power(M*N*sharpness*line_snr, -1))

            if np.isnan(rv_err):
                print('Warning: NAN RV err')

            gaussian_model.set_emu(rv_err)

            self.models.append(gaussian_model)

        #Sort the models based on amplitude
        amps = [ model._model_function(model.mu()) for model in self.models ]
        self.models = np.array(self.models)[np.array(amps).argsort()[::-1]]

        #Convert model to serializable form
        self.models = [ spectrum_utils.convert_ispec_model_to_serializable(
                                model, xvals)
                        for model in self.models ]

class TODCORSlice(BaseProfile):

    def __init__(self, df, nbins=None, norders=None):
        
        self.norders = norders
        super().__init__(df, nbins=nbins)

    def get_err(self):

        #Use uncertainty equation from Zucker03
        velocity_step = self.df.RV.iloc[1] - self.df.RV.iloc[0]

        first_derivative = np.gradient(self.df[self._ycolumn].values, velocity_step)
        second_derivative = np.gradient(first_derivative, velocity_step)

        #idx_peak = np.argmax(self.df.CCF)
        #peak_value = self.df.CCF.iloc[idx_peak]
        #second_derivative_peak = second_derivative[idx_peak]

        peak_value = np.interp(self.rv, self.df.RV.values, self.df[self._ycolumn].values)
        second_derivative_peak = np.interp(self.rv, self.df.RV.values, second_derivative)


        nbins = self.nbins

        if peak_value >= 1:
            peak_value = 1-1e-5

        sharpness = second_derivative_peak / peak_value
        line_snr = (peak_value**2) / (1-peak_value**2)

        #error = np.sqrt(np.power(nbins * (-1*sharpness) * (line_snr), -1))

        M = self.norders
        if M is None:
            M = 1

        error = np.sqrt(-1*np.power(M*nbins*sharpness*line_snr, -1))

        self.rv_err = error

    def _model(self):
        
        idx_peak = np.argmax(self.df.CCF.values)
        #idx_peak = len(self.df)//2

        #Fit second degree polynomial to peak
        p = np.poly1d(np.polyfit(self.df.RV[idx_peak-4:idx_peak+5],
                                 self.df.CCF[idx_peak-4:idx_peak+5], 2))

        poly_vel = np.arange(self.df.RV[idx_peak-4], self.df.RV[idx_peak+4], 0.01)
        mu_poly = poly_vel[np.argmax(p(poly_vel))]

        gaussian_model = GaussianModel()

        baseline = np.min(self.df.CCF)
        A = np.max(p(poly_vel))
        sig = 5

        parinfo = [{'value':0., 'fixed':False, 'limited':[False, False], 'limits':[0., 0.]}
           for j in np.arange(5)]

        #Continuum
        parinfo[0]['value'] = baseline
        parinfo[0]['fixed'] = False
        parinfo[0]['limits'] = [0, A]
        parinfo[0]['limited'] = [True, True]

        #Amplitude
        parinfo[1]['value'] = A
        parinfo[1]['fixed'] = False
        parinfo[1]['limited'] = [False, False]
        parinfo[1]['limits'] = [0., 0.]

        #Sigma
        parinfo[2]['value'] = sig
        parinfo[2]['fixed'] = False
        parinfo[2]['limited'] = [True, False]
        parinfo[2]['limits'] = [1.0e-10, 0.]

        #Mu
        parinfo[3]['value'] = mu_poly
        parinfo[3]['fixed'] = False
        parinfo[3]['limited'] = [True, True]
        parinfo[3]['limits'] = [mu_poly-10, mu_poly+10]

        #Weight higher CCF values more heavily

        weights = self.df.CCF.values.min() - self.df.CCF.values + 0.01
        weights = np.min(weights)/weights

        gaussian_model.fitData(self.df.RV.values,
                               self.df.CCF.values,
                               parinfo=parinfo[:4], weights=weights)

        model_function = gaussian_model._model_function
        yvals = model_function(self.df.RV.values)

        #Use uncertainty equation from Zucker03
        velocity_step = self.df.RV.iloc[1] - self.df.RV.iloc[0]

        first_derivative = np.gradient(self.df.CCF.values, velocity_step)
        second_derivative = np.gradient(first_derivative, velocity_step)

        idx_peak = self.df.RV.values.searchsorted(gaussian_model.mu())
        peak_value = self.df.CCF.iloc[idx_peak]
        second_derivative_peak = second_derivative[idx_peak]

        nbins = self.nbins

        if peak_value >= 1:
            peak_value = 1-1e-5
        sharpness = second_derivative_peak / peak_value
        line_snr = (peak_value**2) / (1-peak_value**2)

        error = np.sqrt(np.power(nbins * (-1*sharpness) * (line_snr), -1))
        gaussian_model.set_emu(error)

        self.models = [gaussian_model]

        return gaussian_model

class Base2DProfile:

    def __init__(self, arr, s1_grid, s2_grid, verbose=True):

        self.arr = arr
        self.s1_grid = s1_grid
        self.s2_grid = s2_grid

        if np.any(np.isnan(self.arr.flatten())):
            self.nanify_neighbors()

        self._verbose = verbose

    @property
    def verbose(self):
        return self._verbose
    @verbose.setter
    def verbose(self, value):
        self._verbose = bool(value)
    def print(self, value):
        if self.verbose:
            print(value)

    def nanify_neighbors(self, radius=3):
        
        nan_indices = np.argwhere(np.isnan(self.arr))
        for idx in nan_indices:
            row, col = idx
            start_row = max(0, row - radius)
            end_row = min(self.arr.shape[0], row + radius + 1)
            start_col = max(0, col - radius)
            end_col = min(self.arr.shape[1], col + radius + 1)
            self.arr[start_row:end_row, start_col:end_col] = np.nan

    def plot(self, ax=None, savefig=None, cbar=False, **kwargs):

        fig, ax, created_fig = plotutils.fig_init(ax=ax, figsize=(6, 6))
        if created_fig:
            fig.subplots_adjust(top=.98, right=.98)

        extent = [self.s1_grid[0], self.s1_grid[-1],
                  self.s2_grid[0], self.s2_grid[-1]]

        kwargs.setdefault('cmap', 'viridis')

        im = ax.imshow(self.arr.T, extent=extent, origin='lower', **kwargs)

        contours = ax.contour(self.s1_grid, self.s2_grid, self.arr.T,
                              levels=5, colors='black')

        ax.set_xlabel(r'$\rm{RV}_1$ (km/s)', fontsize=20)
        ax.set_ylabel(r'$\rm{RV}_2$ (km/s)', fontsize=20)

        if cbar:
            cbar = ax.figure.colorbar(im,fraction=0.046, pad=0.04)
            cbar = plotutils.plotparams_cbar(cbar)

        return plotutils.plt_return(created_fig, fig, ax, savefig)

class TODCORprofile(Base2DProfile):
    
    def __init__(self, arr, s1_grid, s2_grid, alpha_arr=None, nbins=None, verbose=True,
                 spec=None, template1=None, template2=None):

        super().__init__(arr, s1_grid, s2_grid, verbose=verbose)
        
        self.nbins = nbins

        if alpha_arr is not None:
            self.alpha_profile = AlphaProfile(alpha_arr, s1_grid, s2_grid)
            self.alpha_tolerance = 0

        self.nm_result = None
        self.spec = spec
        self.templates = [template1, template2]

    def get_maxima(self):
        '''
        Find all local maxima in the array
        '''

        idx = find_maxima(self.arr)
        return idx

    def _clear_spec(self):
        self.spec = None
    def _clear_templates(self):
        self.templates = []

    @property
    def alpha_tolerance(self):
        return self._alpha_tolerance

    @alpha_tolerance.setter
    def alpha_tolerance(self, value):
        self._alpha_tolerance = value

    def get_rvs(self, return_idx=False, ignore_alpha=False):
        '''
        Select RV guess from grid

        The accuracy is limited by the size of the grid points

        For a more accurate result use get_rvs_nm
        '''
        
        idx_maxima = self.get_maxima()
        #Remove nans
        idx_maxima = [ idx for idx in idx_maxima if not np.isnan(self.arr[idx]) ]
        idx_guess = None
        for idx in idx_maxima:
            alpha = self.alpha_profile.arr[idx]
            if ignore_alpha:
                alpha_criteria = True
            else:
                alpha_criteria = alpha < 1+self.alpha_tolerance
            if alpha_criteria:
                if idx_guess is None:
                    idx_guess = idx
                elif self.arr[idx] > self.arr[idx_guess]:
                    idx_guess = idx
                else:
                    continue

        if idx_guess is None:
            if self.verbose:
                log.warning("Automatic TODCOR RV determination failed")
            return (np.nan, np.nan)
        else:
            if return_idx:
                return idx_guess
            else:
                return self.s1_grid[idx_guess[0]], self.s2_grid[idx_guess[1]]

    def get_rvs_nm(self, spec=None, template1=None, template2=None, rerun=False,
                   ignore_alpha=False):
        
        if (self.nm_result is None) or (rerun):

            if spec is None:
                spec = self.spec
            if spec is None:    
                raise ValueError('Spec argument required')

            if template1 is None:
                template1 = self.templates[0]
            if template1 is None:
                raise ValueError('template1 argument required')

            if template2 is None:
                template2 = self.templates[1]
            if template2 is None:
                raise ValueError('template2 argument required')
            
            self.nm_result = measure_rvs.todcor_nm(spec, template1, template2, self,
                                                   ignore_alpha=ignore_alpha)

        return self.nm_result.x
                    
    def plot(self, ax=None, savefig=None, cbar=False, 
             mark_result=True,
             mark_all=False, **kwargs):

        fig, ax, created_fig = plotutils.fig_init(ax=ax, figsize=(6, 6))
        if created_fig:
            fig.subplots_adjust(top=.98, right=.98)

        ax = super().plot(ax=ax, cbar=cbar, **kwargs)

        if cbar:
            im = ax.images
            cb = im[-1].colorbar
            cb.set_label(r'$\mathcal{R}(\rm{RV}_1,\ \rm{RV}_2,\ \hat{\alpha})$', 
                         fontsize=20)

        if mark_all:
            idx = self.get_maxima()
            idx0 = [ ij for ij in idx if self.alpha_profile.arr[ij] >= 1 ]
            idx1 = [ ij for ij in idx if self.alpha_profile.arr[ij] < 1 ]

            if len(idx0):
                ax.scatter(self.s1_grid[np.array([ij[0] for ij in idx0])],
                           self.s2_grid[np.array([ij[1] for ij in idx0])],
                           color='xkcd:red', marker='o')

            if len(idx1):
                ax.scatter(self.s1_grid[np.array([ij[0] for ij in idx1])],
                           self.s2_grid[np.array([ij[1] for ij in idx1])],
                           color='black', marker='o')


        if mark_result:
            rv1, rv2 = self.get_rvs()
            ax.scatter([rv1],
                       [rv2],
                       color='xkcd:red', marker='x', s=150)


        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        ax.plot(xlim, xlim, color='gray', ls='--')
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        return plotutils.plt_return(created_fig, fig, ax, savefig)

    def get_slice(self, component, method='nm'):
        '''
        Create the 1D slices through the maxima. Return as 1D profile objects
        '''

        if method == 'grid':
            return self._get_slice_grid(component)
        elif method == 'nm':
            return self._get_slice_nm(component)
        else:
            raise ValueError(f'Invalid method {method}')

    def _get_slice_grid(self, component):

        idx_max = self.get_rvs(return_idx=True)

        if component == 1:
            profile = TODCORSlice(pd.DataFrame({'RV': self.s1_grid,
                                                'CCF': self.arr[:, idx_max[1]]}),
                                               nbins=self.nbins)
            profile.rv = self.get_rvs()[0]
        elif component == 2:
            profile = TODCORSlice(pd.DataFrame({'RV': self.s2_grid,
                                                'CCF': self.arr[idx_max[0], :]}),
                                               nbins=self.nbins)
            profile.rv = self.get_rvs()[1]
        else:
            raise ValueError(f'Invalid component {component}')

        profile.df['err'] = 1e-4
        profile.get_err()
        return profile

    def _get_slice_nm(self, component):
        
        rvs = self.get_rvs_nm()

        if component == 1:
            
            profile = measure_rvs.todcor_slice(
                    self.spec, *self.templates, component, 
                    rvs[1], self.s1_grid)

            profile.rv = rvs[0]

        elif component == 2:
            profile = measure_rvs.todcor_slice(
                    self.spec, *self.templates, component,
                    rvs[0], self.s2_grid)

            profile.rv = rvs[1]

        profile.nbins = self.nbins
        profile.get_err()

        return profile

class TODMORprofile(TODCORprofile):

    def __init__(self, arr, s1_grid, s2_grid, alpha_arr=None, nbins=None, norders=None, 
                 verbose=True, spec=None, template1=None, template2=None):

        self.norders = norders
        super().__init__(arr, s1_grid, s2_grid, alpha_arr=alpha_arr, nbins=nbins,
                         verbose=verbose, spec=spec, 
                         template1=template1, template2=template2)

    def get_rvs_nm(self, spec=None, template1=None, template2=None, rerun=False,
                   ignore_alpha=False):
        
        if (self.nm_result is None) or (rerun):

            if spec is None:
                spec = self.spec
            if spec is None:    
                raise ValueError('Spec argument required')

            if template1 is None:
                template1 = self.templates[0]
            if template1 is None:
                raise ValueError('template1 argument required')

            if template2 is None:
                template2 = self.templates[1]
            if template2 is None:
                raise ValueError('template2 argument required')
            
            self.nm_result = measure_rvs.todmor_nm(spec, template1, template2, self,
                                                   ignore_alpha=ignore_alpha)

        return self.nm_result.x

    def _get_slice_nm(self, component):
        
        rvs = self.get_rvs_nm()

        if component == 1:
            
            profile = measure_rvs.todmor_slice(
                    self.spec, *self.templates, component, 
                    rvs[1], self.s1_grid)

            profile.rv = rvs[0]

        elif component == 2:
            profile = measure_rvs.todmor_slice(
                    self.spec, *self.templates, component,
                    rvs[0], self.s2_grid)

            profile.rv = rvs[1]

        profile.nbins = self.nbins
        profile.norders = self.norders
        profile.get_err()

        return profile

    def _get_slice_grid(self, component):

        profile = super()._get_slice_grid(component)
        profile.nbins = self.nbins
        profile.norders = self.norders
        profile.get_err()


class AlphaProfile(Base2DProfile):

    def __init__(self, arr, s1_grid, s2_grid, verbose=True):
        
        super().__init__(arr, s1_grid, s2_grid, verbose=verbose)

    def plot(self, ax=None, savefig=None, cbar=False, **kwargs):

        fig, ax, created_fig = plotutils.fig_init(ax=ax, figsize=(6, 6))
        if created_fig:
            fig.subplots_adjust(top=.98, right=.98)

        ax = super().plot(ax=ax, cbar=cbar, **kwargs)

        if cbar:
            im = ax.images
            cb = im[-1].colorbar
            cb.set_label(r'$\hat{\alpha}$', fontsize=20)

        return plotutils.plt_return(created_fig, fig, ax, savefig)


def find_maxima(arr):
    idx = []
    for i in range(1, arr.shape[0]-1):
        for j in range(1, arr.shape[1]-1):

            element = arr[i,j]
            if arr[i-1, j] > element:
                continue
            if arr[i+1, j] > element:
                continue
            if arr[i, j-1] > element:
                continue
            if arr[i, j+1] > element:
                continue

            idx.append((i, j))

    return idx
