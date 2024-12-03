#!/usr/bin/env python

import astropy.constants as const
from astropy import units as u
import numpy as np
import numba
import pandas as pd
from scipy.optimize import minimize
from tqdm import tqdm
import warnings

from binaryspec import spectrum_utils, utils
from binaryspec import profiles

#Dom Rowan 2024

def cross_correlate_with_template(
        spec, template, 
        lower_velocity_limit, upper_velocity_limit, 
        velocity_step, flip_template=True):

    #Create velocity grid
    shifts = np.arange(
            np.int32(np.floor(lower_velocity_limit)/velocity_step), 
            np.int32(np.ceil(upper_velocity_limit)/velocity_step)+1)
    velocity = shifts * velocity_step

    df = spec.df.copy()

    #Wavelength grid corresponding to the unifrom velocity grid
    wavelength_grid = resample_spectrum_uniform_velocity_grid(
            (df.wavelength.min(), df.wavelength.max()),
            velocity_step)

    flux, err = spectrum_utils.resample_spectrum(
            wavelength_grid, 
            df.wavelength.values,
            df.flux.values,
            err=df.err.values)

    #Resample template spectrum
    if flip_template:
        depth = np.abs(template.df.flux.max()- template.df.flux).values
    else:
        depth = template.df.flux.values

    resampled_template = spectrum_utils.resample_spectrum(
            wavelength_grid, template.df.wavelength.values, depth)

    num_shifts = len(shifts)
    # Cross-correlation function
    ccf = np.zeros(num_shifts)
    ccf_err = np.zeros(num_shifts)

    for shift, i in zip(shifts, np.arange(num_shifts)):
        if shift == 0:
            shifted_template = resampled_template
        else:
            shifted_template = np.hstack((resampled_template[-1*shift:], 
                                          resampled_template[:-1*shift]))
        ccf[i] = np.average(flux*shifted_template)
        ccf_err[i] = np.average(err*shifted_template)

    max_ccf = np.max(ccf)
    ccf = ccf/max_ccf # Normalize

    nbins = len(flux)

    df_ccf = pd.DataFrame({'RV':velocity, 'CCF':ccf, 'err':ccf_err})

    return profiles.BaseProfile(df_ccf, nbins=len(flux))

def resample_spectrum_uniform_velocity_grid(wavelength_range, velocity_step):

    wave_base, wave_top = wavelength_range
    c = const.c.to('km/s').value

    i = int(np.ceil( (c * (wave_top - wave_base))/ (wave_base*velocity_step)))
    grid = wave_base * np.power((1 + (velocity_step/ c)), np.arange(i)+1)

    # Ensure wavelength limits since the "number of elements i" tends to be overestimated
    wfilter = grid <= wave_top
    grid = grid[wfilter]

    return np.asarray(grid)

def compute_rms(arr):
    squared_values = np.square(arr)
    mean_squared = np.mean(squared_values)
    return np.sqrt(mean_squared)

def todcor(spec, template1, template2, 
           lower_velocity_limit, upper_velocity_limit, 
           velocity_step, progress=True, alpha=None):

    #Different velocity grids
    if utils.check_iter(lower_velocity_limit):
        assert(utils.check_iter(upper_velocity_limit))

        s1_grid = np.arange(lower_velocity_limit[0], 
                            upper_velocity_limit[0]+velocity_step,
                            velocity_step)

        s2_grid = np.arange(lower_velocity_limit[1],
                            upper_velocity_limit[1]+velocity_step,
                            velocity_step)

    else:
        #Create velocity grid
        s1_grid = np.arange(lower_velocity_limit, upper_velocity_limit+velocity_step, 
                            velocity_step)
        s2_grid = np.arange(lower_velocity_limit, upper_velocity_limit+velocity_step, 
                            velocity_step)

    #Wavelength grid corresponding to the unifrom velocity grid
    wavelength_grid = resample_spectrum_uniform_velocity_grid(
            (spec.df.wavelength.min(), spec.df.wavelength.max()),
            velocity_step)

    #resampled_fluxes = 1 - spectrum_utils.resample_spectrum(
    #        wavelength_grid, spec.df.wavelength.values, spec.df.flux.values)
    resampled_fluxes = spectrum_utils.resample_spectrum(wavelength_grid, 
                                                        spec.df.wavelength.values, 
                                                        spec.df.flux.values)
    resampled_fluxes = np.abs(np.max(resampled_fluxes) - resampled_fluxes)

    sigma_f = np.sqrt(np.sum(np.square(resampled_fluxes))/len(resampled_fluxes))

    arr = np.zeros((len(s1_grid), len(s2_grid)))
    alpha_arr = np.zeros((len(s1_grid), len(s2_grid)))

    g1wavs = template1.df.wavelength.values
    #g1fluxes = 1 - template1.df.flux.values
    g1fluxes = np.abs(template1.df.flux.max() - template1.df.flux.values)
    g2wavs = template2.df.wavelength.values
    #g2fluxes = 1 - template2.df.flux.values
    g2fluxes = np.abs(template2.df.flux.max() - template2.df.flux.values)

    c = 299792458.0

    if progress:
        iterator = enumerate(tqdm(s1_grid))
    else:
        iterator = enumerate(s1_grid)
    for i, s1 in iterator:

        factor_g1 = np.sqrt((1.-(s1*1000.)/c)/(1.+(s1*1000./c)))
        g1_shifted = np.interp(wavelength_grid, g1wavs/factor_g1, 
                               g1fluxes, left=0.0, right=0.0)

        g1_unshifted = np.interp(wavelength_grid, g1wavs, g1fluxes, left=0.0, right=0.0)

        sigma_g1 = np.sqrt(np.sum(np.square(g1_shifted))/len(g1_shifted))
        sigma_g1_unshifted = np.sqrt(np.sum(np.square(g1_unshifted))/len(g1_unshifted))

        for j, s2 in enumerate(s2_grid):

            factor_g2 = np.sqrt((1.-(s2*1000.)/c)/(1.+(s2*1000./c)))
            g2_shifted = np.interp(wavelength_grid, g2wavs/factor_g2, 
                                   g2fluxes, left=0.0, right=0.0)

            factor_g2_12 = np.sqrt((1.-((s2-s1)*1000.)/c)/(1.+((s2-s1)*1000./c)))
            g2_12_shifted = np.interp(wavelength_grid, g2wavs/factor_g2_12,
                                      g2fluxes, left=0.0, right=0.0)

            sigma_g2 = np.sqrt(np.sum(np.square(g2_shifted))/len(g2_shifted))
            sigma_g2_12 = np.sqrt(np.sum(np.square(g2_12_shifted))/len(g2_12_shifted))

            arr[i,j], alpha_arr[i,j] = compute_R(
                    resampled_fluxes, g1_shifted, g2_shifted, g1_unshifted, g2_12_shifted,
                    sigma_f, sigma_g1, sigma_g2, sigma_g1_unshifted, sigma_g2_12)

    return profiles.TODCORprofile(arr, s1_grid, s2_grid,
                                  nbins=len(resampled_fluxes), 
                                  alpha_arr=alpha_arr,
                                  spec=spec,
                                  template1=template1,
                                  template2=template2)

def compute_R(f, g1_shifted, g2_shifted, g1_unshifted, g2_12_shifted,
              sigma_f, sigma_g1, sigma_g2, sigma_g1_unshifted, sigma_g2_12):

    N = len(f)

    c1 =(1/(N*sigma_f*sigma_g1)) * np.correlate(f, g1_shifted)[0]
    c2 = (1/(N*sigma_f*sigma_g2)) * np.correlate(f, g2_shifted)[0]
    c12 = (1/(N*sigma_g1_unshifted*sigma_g2_12)) * np.correlate(g1_unshifted, g2_12_shifted)[0]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        R_numerator = (c1**2) - (2*c1*c2*c12) + (c2**2)
        R_denominator = 1 - (c12**2)
        R = np.sqrt(R_numerator / R_denominator)

        alpha = (c1*c12 - c2) / (c2*c12 - c1)
        alpha = alpha * (sigma_g1 / sigma_g2)
        if R<0:
            return np.nan, alpha
        else:
            return R, alpha

def todcor_nm(spec, template1, template2, todcor_profile, ignore_alpha=False, rvs=None):

    if rvs is None:
        rvs = todcor_profile.get_rvs(ignore_alpha=ignore_alpha)
    velocity_step = 0.5

    c = 299792458.0

    #Wavelength grid corresponding to the uniform velocity grid
    wavelength_grid = resample_spectrum_uniform_velocity_grid(
            (spec.df.wavelength.min(), spec.df.wavelength.max()),
            velocity_step)

    resampled_fluxes = 1 - spectrum_utils.resample_spectrum(
            wavelength_grid, spec.df.wavelength.values, spec.df.flux.values)

    sigma_f = np.sqrt(np.sum(np.square(resampled_fluxes))/len(resampled_fluxes))

    g1wavs = template1.df.wavelength.values
    g1fluxes = 1 - template1.df.flux.values
    g2wavs = template2.df.wavelength.values
    g2fluxes = 1 - template2.df.flux.values

    g1_unshifted = np.interp(wavelength_grid, g1wavs, g1fluxes, left=0.0, right=0.0)
    sigma_g1_unshifted = np.sqrt(np.sum(np.square(g1_unshifted))/len(g1_unshifted))
    N = len(resampled_fluxes)

    def nm_func(s1s2):

        s1,s2 = s1s2

        factor_g1 = np.sqrt((1.-(s1*1000.)/c)/(1.+(s1*1000./c)))
        g1_shifted = np.interp(wavelength_grid, g1wavs/factor_g1,
                               g1fluxes, left=0.0, right=0.0)
        sigma_g1 = np.sqrt(np.sum(np.square(g1_shifted))/len(g1_shifted))


        factor_g2 = np.sqrt((1.-(s2*1000.)/c)/(1.+(s2*1000./c)))
        g2_shifted = np.interp(wavelength_grid, g2wavs/factor_g2,
                               g2fluxes, left=0.0, right=0.0)
        sigma_g2 = np.sqrt(np.sum(np.square(g2_shifted))/len(g2_shifted))

        factor_g2_12 = np.sqrt((1.-((s2-s1)*1000.)/c)/(1.+((s2-s1)*1000./c)))
        g2_12_shifted = np.interp(wavelength_grid, g2wavs/factor_g2_12,
                                  g2fluxes, left=0.0, right=0.0)

        sigma_g2_12 = np.sqrt(np.sum(np.square(g2_12_shifted))/len(g2_12_shifted))

        R, _ = compute_R(
                resampled_fluxes, g1_shifted, g2_shifted, g1_unshifted, g2_12_shifted,
                sigma_f, sigma_g1, sigma_g2, sigma_g1_unshifted, sigma_g2_12)

        if np.isnan(R):
            raise ValueError('R<0')

        return -1*R

    bounds=[(rvs[0]-10, rvs[0]+10),(rvs[1]-10, rvs[1]+10)]

    return minimize(nm_func, rvs, method='Nelder-Mead',
                    tol=1e-6,
                    bounds=bounds)

def todcor_slice(spec, template1, template2, component, s_fixed, grid):
    
    c = 299792458.0
    velocity_step = 0.5

    #Wavelength grid corresponding to the unifrom velocity grid
    wavelength_grid = resample_spectrum_uniform_velocity_grid(
            (spec.df.wavelength.min(), spec.df.wavelength.max()),
            velocity_step)

    resampled_fluxes = 1 - spectrum_utils.resample_spectrum(
            wavelength_grid, spec.df.wavelength.values, spec.df.flux.values)

    sigma_f = np.sqrt(np.sum(np.square(resampled_fluxes))/len(resampled_fluxes))

    g1wavs = template1.df.wavelength.values
    g1fluxes = 1 - template1.df.flux.values
    g2wavs = template2.df.wavelength.values
    g2fluxes = 1 - template2.df.flux.values

    N = len(resampled_fluxes)

    if component == 1:
        s2 = s_fixed

        factor_g2 = np.sqrt((1.-(s2*1000.)/c)/(1.+(s2*1000./c)))
        g2_shifted = np.interp(wavelength_grid, g2wavs/factor_g2,
                               g2fluxes, left=0.0, right=0.0)
        sigma_g2 = np.sqrt(np.sum(np.square(g2_shifted))/len(g2_shifted))

        def compute_R_slice(s1):
            factor_g1 = np.sqrt((1.-(s1*1000.)/c)/(1.+(s1*1000./c)))
            g1_shifted = np.interp(wavelength_grid, g1wavs/factor_g1,
                                   g1fluxes, left=0.0, right=0.0)

            g1_unshifted = np.interp(wavelength_grid, g1wavs, g1fluxes, left=0.0, right=0.0)

            factor_g2_12 = np.sqrt((1.-((s2-s1)*1000.)/c)/(1.+((s2-s1)*1000./c)))
            g2_12_shifted = np.interp(wavelength_grid, g2wavs/factor_g2_12,
                                      g2fluxes, left=0.0, right=0.0)

            sigma_g2_12 = np.sqrt(np.sum(np.square(g2_12_shifted))/len(g2_12_shifted))
            sigma_g1 = np.sqrt(np.sum(np.square(g1_shifted))/len(g1_shifted))
            sigma_g1_unshifted = np.sqrt(np.sum(np.square(g1_unshifted))/len(g1_unshifted))

            R, _ = compute_R(
                    resampled_fluxes, g1_shifted, g2_shifted, g1_unshifted, g2_12_shifted,
                    sigma_f, sigma_g1, sigma_g2, sigma_g1_unshifted, sigma_g2_12)

            return R

    elif component == 2:
        s1 = s_fixed

        factor_g1 = np.sqrt((1.-(s1*1000.)/c)/(1.+(s1*1000./c)))
        g1_shifted = np.interp(wavelength_grid, g1wavs/factor_g1,
                               g1fluxes, left=0.0, right=0.0)

        g1_unshifted = np.interp(wavelength_grid, g1wavs, g1fluxes, left=0.0, right=0.0)
        sigma_g1 = np.sqrt(np.sum(np.square(g1_shifted))/len(g1_shifted))
        sigma_g1_unshifted = np.sqrt(np.sum(np.square(g1_unshifted))/len(g1_unshifted))

        def compute_R_slice(s2):
            
            factor_g2 = np.sqrt((1.-(s2*1000.)/c)/(1.+(s2*1000./c)))
            g2_shifted = np.interp(wavelength_grid, g2wavs/factor_g2,
                                   g2fluxes, left=0.0, right=0.0)
            sigma_g2 = np.sqrt(np.sum(np.square(g2_shifted))/len(g2_shifted))
            factor_g2_12 = np.sqrt((1.-((s2-s1)*1000.)/c)/(1.+((s2-s1)*1000./c)))

            g2_12_shifted = np.interp(wavelength_grid, g2wavs/factor_g2_12,
                                      g2fluxes, left=0.0, right=0.0)
            sigma_g2_12 = np.sqrt(np.sum(np.square(g2_12_shifted))/len(g2_12_shifted))

            R, _ = compute_R(
                    resampled_fluxes, g1_shifted, g2_shifted, g1_unshifted, g2_12_shifted,
                    sigma_f, sigma_g1, sigma_g2, sigma_g1_unshifted, sigma_g2_12)

            return R

    
    R_vals = [ compute_R_slice(s) for s in grid ]

    return profiles.TODCORSlice(pd.DataFrame({
            'RV':grid, 'CCF':R_vals}))


def todmor(spec, template1, template2, 
           lower_velocity_limit, upper_velocity_limit, 
           velocity_step, progress=True):

    if spec.verbose:
        iterator = tqdm(spec.Orders)
    else:
        iterator = spec.Orders
    for Order in iterator:
        Order.todcor_profile = todcor(
                Order, template1, template2,
                lower_velocity_limit, upper_velocity_limit,
                0.5, progress=False)
    M = len(spec.Orders)
    if M > 1:
        T0 = spec.Orders[0].todcor_profile.arr
        T1 = spec.Orders[1].todcor_profile.arr

        x0 = 1 - np.power(T0, 2)
        x1 = 1 - np.power(T1, 2)

        product = np.multiply(x0, x1)

        i = 2
        while i < M:
            Ti = spec.Orders[i].todcor_profile.arr
            xi = 1 - np.power(Ti, 2)
            product = np.multiply(product, xi)

            i += 1

        ML2 = 1 - np.power(product, (1/M))
    else:
        raise NotImplementedError
    
    alpha_array_list = [ Order.todcor_profile.alpha_profile.arr
                         for Order in spec.Orders ]

    stacked_alpha_array = np.stack(alpha_array_list, axis=0)

    """
    averaged_alpha_array = np.average(
            stacked_alpha_array, axis=0,
            weights=np.array([np.max(O.todcor_profile.arr)**2 for O in spec.Orders]))
    """

    weights = np.array([ np.nanmax(O.todcor_profile.arr)**2 for O in spec.Orders])
    masked_alpha_array = np.ma.MaskedArray(
            stacked_alpha_array, mask=np.isnan(stacked_alpha_array))
    averaged_alpha_array = np.ma.average(masked_alpha_array, weights=weights, axis=0)

    return profiles.TODMORprofile(
            np.sqrt(ML2),
            spec.Orders[0].todcor_profile.s1_grid,
            spec.Orders[0].todcor_profile.s2_grid,
            nbins=np.mean([O.todcor_profile.nbins for O in spec.Orders]),
            norders=len(spec.Orders),
            alpha_arr=averaged_alpha_array,
            spec=spec,
            template1=template1,
            template2=template2)

def todmor_nm(spec, template1, template2, todcor_profile, ignore_alpha=False, rvs=None):

    if rvs is None:
        rvs = todcor_profile.get_rvs(ignore_alpha=ignore_alpha)
    velocity_step = 0.5
    c = 299792458.0

    def nm_func(s1s2):

        s1,s2 = s1s2

        Rs = []

        for Order in spec.Orders:

            wavelength_grid = resample_spectrum_uniform_velocity_grid(
                (Order.df.wavelength.min(), Order.df.wavelength.max()),
                velocity_step)

            resampled_fluxes = 1 - spectrum_utils.resample_spectrum(
                wavelength_grid, Order.df.wavelength.values, Order.df.flux.values)

            sigma_f = np.sqrt(np.sum(np.square(resampled_fluxes))/len(resampled_fluxes))

            g1wavs = template1.df.wavelength.values
            g1fluxes = 1 - template1.df.flux.values
            g2wavs = template2.df.wavelength.values
            g2fluxes = 1 - template2.df.flux.values

            g1_unshifted = np.interp(wavelength_grid, g1wavs, g1fluxes, left=0.0, right=0.0)
            sigma_g1_unshifted = np.sqrt(np.sum(np.square(g1_unshifted))/len(g1_unshifted))
            N = len(resampled_fluxes)

            factor_g1 = np.sqrt((1.-(s1*1000.)/c)/(1.+(s1*1000./c)))
            g1_shifted = np.interp(wavelength_grid, g1wavs/factor_g1,
                                   g1fluxes, left=0.0, right=0.0)
            sigma_g1 = np.sqrt(np.sum(np.square(g1_shifted))/len(g1_shifted))

            factor_g2 = np.sqrt((1.-(s2*1000.)/c)/(1.+(s2*1000./c)))
            g2_shifted = np.interp(wavelength_grid, g2wavs/factor_g2,
                                   g2fluxes, left=0.0, right=0.0)
            sigma_g2 = np.sqrt(np.sum(np.square(g2_shifted))/len(g2_shifted))

            factor_g2_12 = np.sqrt((1.-((s2-s1)*1000.)/c)/(1.+((s2-s1)*1000./c)))
            g2_12_shifted = np.interp(wavelength_grid, g2wavs/factor_g2_12,
                                      g2fluxes, left=0.0, right=0.0)

            sigma_g2_12 = np.sqrt(np.sum(np.square(g2_12_shifted))/len(g2_12_shifted))

            R, _ = compute_R(
                    resampled_fluxes, g1_shifted, g2_shifted, g1_unshifted, g2_12_shifted,
                    sigma_f, sigma_g1, sigma_g2, sigma_g1_unshifted, sigma_g2_12)


            Rs.append(R)

        M = len(spec.Orders)

        T0 = Rs[0]
        T1 = Rs[1]

        x0 = 1 - np.power(T0, 2)
        x1 = 1 - np.power(T1, 2)

        product = np.multiply(x0, x1)

        i = 2
        while i < M:
            Ti = Rs[i]
            xi = 1 - np.power(Ti, 2)
            product = np.multiply(product, xi)

            i += 1

        ML2 = 1 - np.power(product, (1/M))
        return -1*np.sqrt(ML2)


    bounds=[(rvs[0]-10, rvs[0]+10),(rvs[1]-10, rvs[1]+10)]
    return minimize(nm_func, rvs, method='Nelder-Mead',
                tol=1e-6,
                bounds=bounds)

def todmor_slice(spec, template1, template2, component, s_fixed, grid):

    c = 299792458.0
    velocity_step = 0.5

    g1wavs = template1.df.wavelength.values
    g1fluxes = 1 - template1.df.flux.values
    g2wavs = template2.df.wavelength.values
    g2fluxes = 1 - template2.df.flux.values

    if component == 1:
        s2 = s_fixed
        factor_g2 = np.sqrt((1.-(s2*1000.)/c)/(1.+(s2*1000./c)))

        R_orders = []
        for Order in spec.Orders:
            wavelength_grid = resample_spectrum_uniform_velocity_grid(
                    (Order.df.wavelength.min(), Order.df.wavelength.max()),
                    velocity_step)
            resampled_fluxes = 1 - spectrum_utils.resample_spectrum(
                    wavelength_grid, Order.df.wavelength.values, Order.df.flux.values)
            N = len(resampled_fluxes)
            sigma_f = np.sqrt(np.sum(np.square(resampled_fluxes))/len(resampled_fluxes))

            g2_shifted = np.interp(wavelength_grid, g2wavs/factor_g2,
                                   g2fluxes, left=0.0, right=0.0)
            sigma_g2 = np.sqrt(np.sum(np.square(g2_shifted))/len(g2_shifted))

            Rs = []
            for s1 in grid:
                factor_g1 = np.sqrt((1.-(s1*1000.)/c)/(1.+(s1*1000./c)))
                g1_shifted = np.interp(wavelength_grid, g1wavs/factor_g1,
                                       g1fluxes, left=0.0, right=0.0)

                g1_unshifted = np.interp(wavelength_grid, g1wavs, g1fluxes, left=0.0, right=0.0)

                factor_g2_12 = np.sqrt((1.-((s2-s1)*1000.)/c)/(1.+((s2-s1)*1000./c)))
                g2_12_shifted = np.interp(wavelength_grid, g2wavs/factor_g2_12,
                                          g2fluxes, left=0.0, right=0.0)

                sigma_g2_12 = np.sqrt(np.sum(np.square(g2_12_shifted))/len(g2_12_shifted))
                sigma_g1 = np.sqrt(np.sum(np.square(g1_shifted))/len(g1_shifted))
                sigma_g1_unshifted = np.sqrt(np.sum(np.square(g1_unshifted))/len(g1_unshifted))

                R, _ = compute_R(
                    resampled_fluxes, g1_shifted, g2_shifted, g1_unshifted, g2_12_shifted,
                    sigma_f, sigma_g1, sigma_g2, sigma_g1_unshifted, sigma_g2_12)
                Rs.append(np.asarray(R))

            R_orders.append(Rs)

    elif component == 2:
        s1 = s_fixed
        factor_g1 = np.sqrt((1.-(s1*1000.)/c)/(1.+(s1*1000./c)))

        R_orders = []
        for Order in spec.Orders:
            wavelength_grid = resample_spectrum_uniform_velocity_grid(
                    (Order.df.wavelength.min(), Order.df.wavelength.max()),
                    velocity_step)
            resampled_fluxes = 1 - spectrum_utils.resample_spectrum(
                    wavelength_grid, Order.df.wavelength.values, Order.df.flux.values)
            N = len(resampled_fluxes)
            sigma_f = np.sqrt(np.sum(np.square(resampled_fluxes))/len(resampled_fluxes))

            g1_shifted = np.interp(wavelength_grid, g1wavs/factor_g1,
                                   g1fluxes, left=0.0, right=0.0)

            g1_unshifted = np.interp(wavelength_grid, g1wavs, g1fluxes, left=0.0, right=0.0)
            sigma_g1 = np.sqrt(np.sum(np.square(g1_shifted))/len(g1_shifted))
            sigma_g1_unshifted = np.sqrt(np.sum(np.square(g1_unshifted))/len(g1_unshifted))

            Rs = []
            for s2 in grid:

                factor_g2 = np.sqrt((1.-(s2*1000.)/c)/(1.+(s2*1000./c)))
                g2_shifted = np.interp(wavelength_grid, g2wavs/factor_g2,
                                       g2fluxes, left=0.0, right=0.0)
                sigma_g2 = np.sqrt(np.sum(np.square(g2_shifted))/len(g2_shifted))

                factor_g2_12 = np.sqrt((1.-((s2-s1)*1000.)/c)/(1.+((s2-s1)*1000./c)))
                g2_12_shifted = np.interp(wavelength_grid, g2wavs/factor_g2_12,
                                          g2fluxes, left=0.0, right=0.0)

                sigma_g2_12 = np.sqrt(np.sum(np.square(g2_12_shifted))/len(g2_12_shifted))

                R, _ = compute_R(
                        resampled_fluxes, g1_shifted, g2_shifted, g1_unshifted, g2_12_shifted,
                        sigma_f, sigma_g1, sigma_g2, sigma_g1_unshifted, sigma_g2_12)
                Rs.append(np.asarray(R))

            R_orders.append(Rs)

    M = len(spec.Orders)
    T0 = R_orders[0]
    T1 = R_orders[1]
    x0 = 1 - np.power(T0, 2)
    x1 = 1 - np.power(T1, 2)

    product = np.multiply(x0, x1)
    i = 2
    while i < M:
        Ti = R_orders[i]
        xi = 1 - np.power(Ti, 2)
        product = np.multiply(product, xi)

        i += 1

    ML2 = 1 - np.power(product, (1/M))
    return profiles.TODCORSlice(pd.DataFrame({
            'RV':grid, 'ML':np.sqrt(ML2)}))
