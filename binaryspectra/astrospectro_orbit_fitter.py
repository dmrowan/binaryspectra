#!/usr/bin/env python

from astropy import units as u
from astropy import constants as const
import numpy as np
import numba

from . import utils
from . import rv_orbit_fitter
from . import astrometric_orbit_fitter

@numba.jit(nopython=True)
def rv_model(t, C, H, period, phi0, ecc, gamma):

    kappa = 2*np.pi*149597870.7/86400 #converts au/d to km/s
    K1 = kappa*np.sqrt(C**2+H**2)/(period*np.sqrt(1-ecc**2))

    cosw = H / np.sqrt(C**2 + H**2)
    sinw = C / np.sqrt(C**2 + H**2)

    M = ((2*np.pi*t)/period) - phi0
    E = np.zeros(len(M))
    for i in range(len(M)):
        Ei = rv_orbit_fitter.eccentric_anomaly(M[i], ecc)
        if Ei is None:
            print('Here', M[i], ecc)
        E[i] = Ei

    cosf = ( np.cos(E) - ecc ) / ( 1-ecc*np.cos(E) )
    sinf = ( np.sqrt(1-ecc**2) * np.sin(E) ) / ( 1-ecc*np.cos(E))

    model = gamma + K1 * (cosw*cosf - sinw*sinf + ecc*cosw)

    return model

@numba.jit(nopython=True)
def log_likelihood_sb1(theta, t, rv1, rv_err1):
    
    C, H, period, phi0, ecc, gamma, logs = theta[:7]
    rvoffsets = theta[7:]

    s = np.exp(logs)

    if isinstance(t, tuple):
        assert(len(rvoffsets) == len(t)-1)

        models_1 = [ rv_model(t[i], C, H, period, phi0, ecc, gamma) + 0
                   if i == 0 else
                   rv_model(t[i], C, H, period, phi0, ecc, gamma) + rvoffsets[i-1]
                   for i in range(len(t)) ]

        for i in range(len(models_1)):
            if np.any(np.isnan(models_1[i])):
                return -np.inf

        lnlike = 0
        for i in range(len(models_1)):

            lnlike += -0.5*np.sum(
                    ((np.power(rv1[i]-models_1[i],2))/(np.power(rv_err1[i],2)+s**2) ) +
                    np.log(2*np.pi*(np.power(rv_err1[i],2)+s**2)))

        return lnlike

    else:
        assert(len(rvoffsets) == 0)

        model_1 = rv_model(t, C, H, period, phi0, ecc, gamma)

        if np.any(np.isnan(model_1)):
            return -np.inf

        return -0.5*np.sum(
                ((np.power(rv1-model_1, 2))/(np.power(rv_err1,2)+s**2)) +
                np.log(2*np.pi*(np.power(rv_err1,2)+s**2)))



@numba.jit(nopython=True)
def log_likelihood_astrosb1(theta, trv, rv1, rv_err1, tast, astx, asty,
                            astx_err, asty_err, distance):

    A, B, F, G, C, H, period, phi0, ecc, gamma, logs = theta[:11]
    rvoffsets = theta[11:]

    #Compute the rv likelihood
    #build theta rv
    theta_rv = np.empty_like(np.arange(7+len(rvoffsets)), dtype=np.float64)
    theta_rv[0] = C
    theta_rv[1] = H
    theta_rv[2] = period
    theta_rv[3] = phi0
    theta_rv[4] = ecc
    theta_rv[5] = gamma
    theta_rv[6] = logs
    i = 7
    for j in range(len(rvoffsets)):
        theta_rv[i] = rvoffsets[j]
        i += 1

    lnlike_rv = log_likelihood_sb1(theta_rv, trv, rv1, rv_err1)

    #Compute ast likelihood
    model_x, model_y = astrometric_orbit_fitter.compute_xy(
            tast, 0, 0, A, B, F, G, period, phi0, ecc)
    
    model_x = model_x * 1000/distance
    model_y = model_y * 1000/distance

    lnlike_ast = -0.5*np.sum(np.square( (astx-model_x) / astx_err ))
    lnlike_ast += -0.5*np.sum(np.square( (asty-model_y) / asty_err ))

    lnlike = lnlike_rv + lnlike_ast

    return lnlike

@numba.jit(nopython=True)
def lnprior_astrosb1(theta, period_mu=None, period_sigma=None, ecc_max=0.9, period_max=3000):
    
    A, B, F, G, C, H, period, phi0, ecc, gamma, logs = theta[:11]
    rvoffsets = theta[11:]

    if np.any(np.isinf(theta)):
        return -np.inf

    A_condition = True
    B_condition = True
    F_condition = True
    G_condition = True

    C_condition = True
    H_condition = True

    period_condition = 0.5 <= period <= period_max
    phi0_condition = 0 <= phi0 <= 2*np.pi
    ecc_condition = 0 <= ecc <= ecc_max
    gamma_condition = True

    logs_condition = -10 < logs < 0

    conditions = [ A_condition, B_condition, F_condition, G_condition,
                   C_condition, H_condition,
                   period_condition, phi0_condition, ecc_condition, gamma_condition,
                   logs_condition ]

    lp = 0
    if len(rvoffsets):
        
        for rvo in rvoffsets:
            conditions.append( -5 <= rvo <= 5 )
            lp += -0.5*((rvo)/1.5)**2

    if period_mu is None:
        if np.all(np.array(conditions)):
            return lp
        else:
            return -np.inf
    else:
        if np.all(np.array(conditions)):
            return lp + -0.5*((period-period_mu)/period_sigma)**2
        else:
            return -np.inf

@numba.jit(nopython=True)
def lnprob_astrosb1(theta, trv, rv1, rv_err1, tast, astx, asty, astx_err, asty_err,
                    distance, period_mu=None, period_sigma=None, ecc_max=0.9, period_max=3000):

    lp = lnprior_astrosb1(theta, period_mu=period_mu, period_sigma=period_sigma, 
                          ecc_max=ecc_max, period_max=period_max)

    if np.isinf(lp):
        return -np.inf

    like = log_likelihood_astrosb1(theta, trv, rv1, rv_err1,
                                   tast, astx, asty, astx_err, asty_err, distance)
    if np.isinf(like):
        return -np.inf

    return lp+like
                        
