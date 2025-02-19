#!/usr/bin/env python

from astropy import units as u
from astropy import constants as const
import numpy as np
import numba

from . import utils
from . import rv_orbit_fitter

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
def compute_TI(sma, incl, omega, lan):

    A = sma*(np.cos(lan)*np.cos(omega) - np.cos(incl)*np.sin(lan)*np.sin(omega))
    B = sma*(np.sin(lan)*np.cos(omega) + np.cos(incl)*np.cos(lan)*np.sin(omega))
    F = sma*(-np.cos(lan)*np.sin(omega) - np.cos(incl)*np.sin(lan)*np.cos(omega))
    G = sma*(-np.sin(lan)*np.sin(omega) + np.cos(incl)*np.cos(lan)*np.cos(omega))

    return A, B, F, G

@numba.jit(nopython=True)
def compute_xy(t, xcm, ycm, A, B, F, G, P, phi0, ecc):

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
                            astx_err, asty_err):

    A, B, F, G, C, H, period, phi0, ecc, gamma, logs, parallax = theta[:12]
    rvoffsets = theta[12:]

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
    model_x, model_y = compute_xy(tast, 0, 0, A, B, F, G, period, phi0, ecc)
    
    model_x = model_x * parallax
    model_y = model_y * parallax

    lnlike_ast = -0.5*np.sum(np.square( (astx-model_x) / astx_err ))
    lnlike_ast += -0.5*np.sum(np.square( (asty-model_y) / asty_err ))

    lnlike = lnlike_rv + lnlike_ast

    return lnlike

@numba.jit(nopython=True)
def lnprior_astrosb1(theta, period_mu=None, period_sigma=None, ecc_max=0.9, period_max=3000,
                     parallax_mu=None, parallax_sigma=None):
    
    A, B, F, G, C, H, period, phi0, ecc, gamma, logs, parallax = theta[:12]
    rvoffsets = theta[12:]

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
    parallax_condition = parallax > 0

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

    lp += -0.5*((parallax-parallax_mu)/parallax_sigma)**2

    if period_mu is None:
        if np.all(np.array(conditions)):
            return lp
        else:
            return -np.inf
    else:
        if np.all(np.array(conditions)):
            lp += -0.5*((period-period_mu)/period_sigma)**2
            return lp
        else:
            return -np.inf

@numba.jit(nopython=True)
def lnprob_astrosb1(theta, trv, rv1, rv_err1, tast, astx, asty, astx_err, asty_err,
                    period_mu=None, period_sigma=None, ecc_max=0.9, period_max=3000,
                    parallax_mu=None, parallax_sigma=None):

    lp = lnprior_astrosb1(theta, period_mu=period_mu, period_sigma=period_sigma, 
                          ecc_max=ecc_max, period_max=period_max,
                          parallax_mu=parallax_mu, parallax_sigma=parallax_sigma)

    if np.isinf(lp):
        return -np.inf

    like = log_likelihood_astrosb1(theta, trv, rv1, rv_err1,
                                   tast, astx, asty, astx_err, asty_err)
    if np.isinf(like):
        return -np.inf

    return lp+like
                        
