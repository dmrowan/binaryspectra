#!/usr/bi/env python

import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
import pandas as pd
import pickle
import subprocess

from . import plotutils
from .base_spectrum import *

class FDBinaryResult:
    
    def __init__(self, input_files, flux_ratio=1, verbose=True):
        
        self.input_files = input_files
        self.flux_ratio = flux_ratio

        self.log_files = [] #log files
        self.mod_files = [] #model files
        self.obs_files = [] #actual data files
        self.res_files = [] #residuals of the fit
        self.rvs_files = [] #measured RVs

        for fname in self.input_files:
            
            pwd, f = os.path.split(fname)
            f = os.path.splitext(f)[0]
            self.log_files.append(os.path.join(pwd, f+'.log'))
            self.mod_files.append(os.path.join(pwd, f+'.mod'))
            self.obs_files.append(os.path.join(pwd, f+'.obs'))
            self.res_files.append(os.path.join(pwd, f+'.res'))
            self.rvs_files.append(os.path.join(pwd, f+'.rvs'))

        self._load_observed_spectra()
        self._load_residual_spectra()
        self._load_model_spectra()
        self._load_rv_results()

        self.verbose = verbose

    @property
    def verbose(self):
        return self._verbose
    @verbose.setter
    def verbose(self, value):
        self._verbose = value

        for spec in self.spectra:
            try:
                spec.verbose = value
            except:
                pass
        self.model1.verbose = value
        self.model2.verbose = value

        
    def __load_fd_spectra(self, fnames):

        '''
        the problem is that different orders have different number of spectra

        so i can the actual number of spectra by taking the max over all fnames
       
        but for a given order fname, if N spectra in that file is less than the total, 
        I won't know which spectra to assign it to

        so i think the easiest thing is to loop through all fnames

        but if I do have all the same number of spectra in each order I do want to re-construct the full spectra objects

        '''
        
        N_spectra = np.loadtxt(fnames[0]).shape[1] - 1
        all_same_number = True
        for i in range(1, len(fnames)):
            N_spectra_i = np.loadtxt(fnames[i]).shape[1] - 1
            if N_spectra_i != N_spectra:
                all_same_number = False
                break

        if all_same_number:
            N_orders = len(fnames)

            spectra = []
            for i in range(N_spectra):
                Orders = []
                for j in range(N_orders):
                    
                    n = int(os.path.splitext(os.path.split(fnames[j])[-1])[0].split('_')[-1])
                    data = np.loadtxt(fnames[j])
                    wavelength = np.exp(data[:,0])
                    flux = data[:,i+1]

                    Order = EchelleOrder(n, wavelength, flux)

                    Orders.append(Order)
                
                spec = EchelleSpectrum(None, None)
                spec.verbose = False
                spec.Orders = Orders
                spectra.append(spec)

            return spectra

        else:
            print('Variable number of spectra in each order')
            Orders = []
            for fname in fnames:
                
                n = int(os.path.splitext(os.path.split(fname)[-1])[0].split('_')[-1])
                data = np.loadtxt(fname)

                wavelength = np.exp(data[:,0])
                Orders_i = []
                for i in range(1, data.shape[1]):
                    flux = data[:,i]
                    spec = EchelleOrder(n, wavelength, flux)
                    Orders_i.append(spec)
                Orders.append(Orders_i)

            return Orders

    def _load_observed_spectra(self):

        self.spectra = self.__load_fd_spectra(self.obs_files)

    def _load_residual_spectra(self):

        self.residual_spectra = self.__load_fd_spectra(self.res_files)

    def _load_model_spectra(self):
        
        Orders1 = []
        Orders2 = []
        
        for i in range(len(self.mod_files)):
            
            data = np.loadtxt(self.mod_files[i])
            n = int(os.path.splitext(os.path.split(self.mod_files[i])[-1])[0].split('_')[-1])
            wavelength = np.exp(data[:,0])
            flux1 = data[:,1] #* 1/(1+self.flux_ratio)
            flux2 = data[:,2] #* 1/(1+(1/self.flux_ratio))
            Order1 = EchelleOrder(n, wavelength, flux1)
            Order2 = EchelleOrder(n, wavelength, flux2)

            Orders1.append(Order1)
            Orders2.append(Order2)

        self.model1 = EchelleSpectrum(None, None, name='Component 1')
        self.model1.Orders = Orders1

        self.model2 = EchelleSpectrum(None, None, name='Component 2')
        self.model2.Orders = Orders2

    def _read_input(self, fname):
        
        '''
        just getting the times
        '''

        with open(fname, 'r') as f:
            lines = f.readlines()

        idx_start = 2
        idx_end = idx_start
        while lines[idx_end] != '\n':
            idx_end += 1

        times = []
        for l in lines[idx_start:idx_end]:
            l = l.split()
            times.append(float(l[0]))

        return times

    def _load_rv_results(self):
        
        df_list = []
        for fname_rv, fname_input in zip(self.rvs_files, self.input_files):
            
            times = self._read_input(fname_input)
            
            df = pd.read_table(fname_rv, delim_whitespace=True, comment='#', header=None)
            df.columns = ['RV1', 'RV2']
            df['time'] = times
            df_list.append(df)

        all_times = np.unique(np.concatenate([ df.time.values for df in df_list ]))

        RV1 = []
        RV2 = []
        RV1_std = []
        RV2_std = []

        for time in all_times:

            RV1_i_vals = []
            RV2_i_vals = []

            for df in df_list:
                idx = np.where(df.time == time)[0]
                if not len(idx):
                    RV1_i_vals.append(np.nan)
                    RV2_i_vals.append(np.nan)
                else:
                    idx = idx[0]
                    RV1_i_vals.append(df.RV1.iloc[idx])
                    RV2_i_vals.append(df.RV2.iloc[idx])
            
            RV1.append(np.nanmedian(RV1_i_vals))
            RV2.append(np.nanmedian(RV2_i_vals))
            RV1_std.append(np.nanstd(RV1_i_vals))
            RV2_std.append(np.nanstd(RV2_i_vals))

        self.df_rv = pd.DataFrame({'times': all_times,
                                   'RV1': RV1, 'RV1_std':RV1_std,
                                   'RV2': RV2, 'RV2_std':RV2_std})


    def cleanup(self):

        for file_list in [self.input_files, self.log_files, 
                          self.mod_files, self.obs_files, 
                          self.res_files, self.rvs_files]:

            for f in file_list:
                subprocess.run(['rm', f])

    def plot(self, spec_index, order=None, ax=None, savefig=None, **kwargs):
    
        fig, ax, created_fig = plotutils.fig_init(ax=ax, figsize=(10, 6))

        ax = self.spectra[spec_index].plot(ax=ax, offset=0.75, alternating_colors=False,
                                           plot_kwargs=dict(color='black', alpha=0.7),
                                           **kwargs)
        ax = self.model1.plot(ax=ax, offset=0.5, alternating_colors=False,
                              plot_kwargs=dict(color='xkcd:red', alpha=0.7),
                              **kwargs)
        ax = self.model2.plot(ax=ax, offset=0.0, alternating_colors=False,
                              plot_kwargs=dict(color='xkcd:azure', alpha=0.7),
                              **kwargs)

        return plotutils.plt_return(created_fig, fig, ax, savefig)


    def get_chi2(self, wavelength_range):

        '''
        the residual spectra can either be a list of spectra
        or a list of lists of orders
        '''

        chi2 = 0
        npoints = 0

        if isinstance(self.residual_spectra[0], EchelleSpectrum):
            #Case 1
            for spec in self.residual_spectra:
                wavelength_arr = spec.get_wavelength_arr()
                residuals = spec.get_flux_arr()

                idx = np.where( (wavelength_arr > wavelength_range[0]) &
                                (wavelength_arr < wavelength_range[1]) )[0]
                if len(idx):
                    chi2 += np.sum(np.square(residuals[idx]))
                    npoints += len(idx)
        else:
            for i in range(len(self.residual_spectra)):
                
                for spec in self.residual_spectra[i]:
                    
                    wavelength_arr = spec.df.wavelength.values
                    residuals = spec.df.flux.values

                    idx = np.where( (wavelength_arr > wavelength_range[0]) &
                                    (wavelength_arr < wavelength_range[1]) )[0]
                    if len(idx):
                        chi2 += np.sum(np.square(residuals[idx]))
                        npoints += len(idx)

        
        return chi2, npoints

    def to_pickle(self, outfile):
        
        with open(outfile, 'wb') as p:
            pickle.dump(self, p)
