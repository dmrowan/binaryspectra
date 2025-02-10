#!/usr/bin/env python

from astropy import units as u
from astropy import constants as const
import numpy as np
import numba

from . import utils
from . import rv_orbit_fitter

def compute_sma(M1, M2, P):
    
    sma = np.power(const.G*(M1*u.M_sun+M2*u.M_sun)*(P*u.day)**2 / (4*np.pi**2), 1/3)
    sma = sma.to('AU')
    return sma.value

def projected_sep(t, M1, K, P, T0, ecc, omega, incl, distance):

    #Compute M2 from K, M1, P and ecc
    fM = utils.mass_function(P*u.day, K*u.km/u.s, e=ecc)
    M2 = utils.companion_mass(fM, M1*u.M_sun, incl*180/np.pi).value

    sma3 = const.G * (M1*u.M_sun+M2*u.M_sun) * (P*u.day)**2 / (4*np.pi**2)
    sma = np.power(sma3, 1/3).to('AU').value

    M = (2*np.pi / P) * (t-T0)
    E = np.zeros(len(M))

    for i in range(len(M)):
        Ei = rv_orbit_fitter.eccentric_anomaly(M[i], ecc)
        if Ei is None:
            print('Here', M[i], ecc)
        E[i] = Ei

    f = 2*np.arctan( np.sqrt( (1+ecc) / (1-ecc) ) * np.tan(E / 2))

    term1 = (1-ecc**2) / (1+ecc*np.cos(f))
    term2 = np.sqrt(1 -np.square(np.sin(incl))*np.square(np.sin(f+omega)))

    return sma * term1 * term2 * 1000 / distance

@numba.jit(nopython=True)
def compute_TI(sma, incl, omega, lan):
    
    A = sma*(np.cos(lan)*np.cos(omega) - np.cos(incl)*np.sin(lan)*np.sin(omega))
    B = sma*(np.sin(lan)*np.cos(omega) + np.cos(incl)*np.cos(lan)*np.sin(omega))
    F = sma*(-np.cos(lan)*np.sin(omega) - np.cos(incl)*np.sin(lan)*np.cos(omega))
    G = sma*(-np.sin(lan)*np.sin(omega) + np.cos(incl)*np.cos(lan)*np.cos(omega))
    
    return A, B, F, G

@numba.jit(nopython=True)
def compute_xy(t, xcm, ycm, A, B, F, G, P, phi0, ecc):
    
    #M = (2*np.pi / P) * (t-T0)
    M = ((2*np.pi*t)/P) - phi0
    u = np.zeros(len(M))
    for i in range(len(M)):
        Ei = rv_orbit_fitter.eccentric_anomaly(M[i], ecc)
        if Ei is None:
            print('Here', M[i], ecc)
        u[i] = Ei
    
    x = xcm - A*(np.cos(u) - ecc) - F*(1-ecc**2)**(1/2) * np.sin(u)
    y = xcm - B*(np.cos(u) - ecc) - G*(1-ecc**2)**(1/2) * np.sin(u)
    
    return x, y

'''
apsidal motion fit

K1, gamma, phi0, ecc, omega, period, logs, sma, lan, incl, omegadot

calculate omega = omega+omegadot*t (pay attention to units)

calculate TI ABFG

compute xy
'''

@numba.jit(nopython=True)
def log_likelihood_sb1_astrometry(theta, trv, rv1, rv_err1, tast, astx, asty, astx_err, asty_err, distance):

    K1, gamma, phi0, ecc, omega, period, logs, A, B, F, G = theta[:11]
    rvoffsets = theta[11:]

    #Build the theta for the RV likelihood
    #Have to do it this way because np concatenate doesn't work with numba

    #theta_rv = (K1, gamma, phi0, ecc, omega, period, logs)

    theta_rv = np.empty_like(np.arange(len(theta[:7])+len(rvoffsets)))
    for i in range(7):
        theta_rv[i] = theta[i]
    i = 7
    for j in range(len(rvoffsets)):
        theta_rv[i] = rvoffsets[j]
        i += 1

    #Compute the likelihood for the RV measurements
    lnlike_rv = rv_orbit_fitter.log_likelihood_sb1(theta_rv, trv, rv1, rv_err1)

    #Compute the likelihood for the relative astrometry
    model_x, model_y = compute_xy(tast, 0, 0, A, B, F, G, period, phi0, ecc)
    model_x = model_x * 1000/distance
    model_y = model_y * 1000/distance

    lnlike_ast = -0.5*np.sum(np.square( (astx-model_x) / astx_err ))
    lnlike_ast += -0.5*np.sum(np.square( (asty-model_y) / asty_err ))

    lnlike = lnlike_rv + lnlike_ast

    return lnlike

@numba.jit(nopython=True)
def lnprior_sb1_astrometry(theta, period_mu=None, period_sigma=None, ecc_max=None, period_max=3000):

    K1, gamma, phi0, ecc, omega, period, logs, A, B, F, G = theta[:11]
    rvoffsets = theta[11:]

    lnprior_rv = rv_orbit_fitter.lnprior_sb1(
            theta, period_mu=period_mu, period_sigma=period_sigma,
            ecc_max=ecc_max, period_max=period_max)

    A_condition = True
    B_condition = True
    F_condition = True
    G_condition = True

    conditions = [ A_condition, B_condition, F_condition, G_condition ]

    lp = lnprior_rv
    if np.isinf(lp):
        return -np.inf

    if np.all(np.array(conditions)):
        return lp
    

@numba.jit(nopython=True)
def lnprob_sb1_astrometry(theta, trv, rv1, rv_err1, tast, astx, asty, astx_err, asty_err,
                          distance, 
                          period_mu=None, period_sigma=None, ecc_max=None, period_max=3000):

    lp = lnprior_sb1_astrometry(theta, period_mu=period_mu, period_sigma=period_sigma,
                                ecc_max=ecc_max, period_max=period_max)
    
    if np.isinf(lp):
        return -np.inf

    like = log_likelihood_sb1_astrometry(theta, trv, rv1, rv_err1, tast, astx, asty, astx_err, asty_err, distance)
    if np.isinf(like):
        return -np.inf

    return lp+like

