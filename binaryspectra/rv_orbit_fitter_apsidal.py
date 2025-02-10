#!/usr/bin/env python

import numba
import numpy as np

np.random.seed(32)

#Dom Rowan 2024

desc="""
Functions for fitting RV orbit with EMCEE
"""

@numba.jit(nopython=True)
def eccentric_anomaly(M, e):

    max_iterations = 1000
    tolerance = 1e-7
    E = M
    for _ in range(max_iterations):
        f = E - e * np.sin(E) - M

        f_prime = 1 - e * np.cos(E)
        delta = f / f_prime
        E -= delta

        if np.abs(delta) < tolerance:
            return E

@numba.jit(nopython=True)
def rv_model(t, K, gamma, phi0, ecc, omega, period, omegadot=0):

    cosw = np.cos(omega + omegadot*t/365)
    sinw = np.sin(omega + omegadot*t/365)

    M = ((2*np.pi*t)/period) - phi0
    E = np.zeros(len(M))
    for i in range(len(M)):
        Ei = eccentric_anomaly(M[i], ecc)
        if Ei is None:
            print('Here', M[i], ecc)
        E[i] = Ei

    cosf = ( np.cos(E) - ecc ) / ( 1-ecc*np.cos(E) )
    sinf = ( np.sqrt(1-ecc**2) * np.sin(E) ) / ( 1-ecc*np.cos(E))

    model = gamma + K * (cosw*cosf - sinw*sinf + ecc*cosw)

    return model

@numba.jit(nopython=True)
def log_likelihood_sb1(theta, t, rv1, rv_err1):

    K1, gamma, phi0, ecc, omega, period, logs, omegadot = theta[:8]
    rvoffsets = theta[8:]

    s = np.exp(logs)

    if isinstance(t, tuple):
        assert(len(rvoffsets) == len(t)-1)

        models_1 = [ rv_model(t[i], K1, gamma, phi0, ecc, omega, period, omegadot=omegadot) + 0
                   if i == 0 else
                   rv_model(t[i], K1, gamma, phi0, ecc, omega, period, omegadot=omegadot) + rvoffsets[i-1]
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

        model_1 = rv_model(t, K1, gamma, phi0, ecc, omega, period, omegadot=omegadot)

        if np.any(np.isnan(model_1)):
            return -np.inf

        return -0.5*np.sum(
                ((np.power(rv1-model_1, 2))/(np.power(rv_err1,2)+s**2)) +
                np.log(2*np.pi*(np.power(rv_err1,2)+s**2)))


@numba.jit(nopython=True)
def log_likelihood_sb2(theta, t, rv1, rv_err1, rv2, rv_err2):

    K1, K2, gamma, phi0, ecc, omega, period, logs1, logs2 = theta[:9]
    rvoffsets = theta[9:]

    s1 = np.exp(logs1)
    s2 = np.exp(logs2)

    if isinstance(t, tuple):
        assert(len(rvoffsets) == len(t)-1)



        models_1 = [ rv_model(t[i], K1, gamma, phi0, ecc, omega, period) + 0
                   if i == 0 else
                   rv_model(t[i], K1, gamma, phi0, ecc, omega, period) + rvoffsets[i-1]
                   for i in range(len(t)) ]
        models_2 = [ rv_model(t[i], -1*K2, gamma, phi0, ecc, omega, period) + 0
                   if i == 0 else
                   rv_model(t[i], -1*K2, gamma, phi0, ecc, omega, period) + rvoffsets[i-1]
                   for i in range(len(t)) ]

        for models in [models_1, models_2]:
            for i in range(len(models)):
                if np.any(np.isnan(models[i])):
                    return -np.inf

        lnlike = 0
        for i in range(len(models_1)):

            lnlike += -0.5*np.sum(
                    ((np.power(rv1[i]-models_1[i],2))/(np.power(rv_err1[i],2)+s1**2) +
                     (np.log(2*np.pi*(np.power(rv_err1[i],2)+s1**2)))))
            lnlike += -0.5*np.sum(
                    ((np.power(rv2[i]-models_2[i],2))/(np.power(rv_err2[i],2)+s2**2) +
                     (np.log(2*np.pi*(np.power(rv_err2[i],2)+s2**2)))))

        return lnlike

    else:

        assert(len(rvoffsets) == 0)

        model_1 = rv_model(t, K1, gamma, phi0, ecc, omega, period)
        model_2 = rv_model(t, -1*K2, gamma, phi0, ecc, omega, period)

        if np.any(np.isnan(model_1)):
            return -np.inf
        if np.any(np.isnan(model_2)):
            return -np.inf

        lnlike = -0.5*np.sum(
                ((np.power(rv1-model_1,2))/(np.power(rv_err1,2)+s1**2)+
                 (np.log(2*np.pi*(np.power(rv_err1,2)+s1**2)))))
        lnlike += -0.5*np.sum(
                ((np.power(rv2-model_2,2))/(np.power(rv_err2,2)+s2**2)+
                 (np.log(2*np.pi*(np.power(rv_err2,2)+s2**2)))))

        return lnlike

@numba.jit(nopython=True)
def lnprior_sb1(theta, period_mu=None, period_sigma=None, ecc_max=None, period_max=3000):

    K1, gamma, phi0, ecc, omega, period, logs, omegadot = theta[:8]
    rvoffsets = theta[8:]

    if np.any(np.isinf(theta)):
        return -np.inf

    K_condition = (K1 > 0)
    gamma_condition = True
    phi_condition = 0 <= phi0 <= 2*np.pi

    if ecc_max is None:
        ecc_condition = 0 <= ecc <= 0.9
    else:
        ecc_condition = 0 <= ecc <= ecc_max
    omega_condition = -1*np.pi <= omega <= np.pi

    period_condition = 0.5 <= period <= period_max

    logs_condition = -10 < logs < 0

    omegadot_condition = -2 < omegadot < 2

    conditions = [K_condition, gamma_condition, phi_condition, ecc_condition,
                  omega_condition, period_condition, logs_condition, omegadot_condition]

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
def lnprior_sb1_gaia(theta, gaia_dict, ecc_max=None, period_max=3000):

    K1, gamma, phi0, ecc, omega, period, logs = theta[:7]
    rvoffsets = theta[7:]

    if np.any(np.isinf(theta)):
        return -np.inf

    K_condition = (K1 > 0)
    gamma_condition = True
    phi_condition = 0 <= phi0 <= 2*np.pi

    if ecc_max is None:
        ecc_condition = 0 <= ecc <= 0.9
    else:
        ecc_condition = 0 <= ecc <= ecc_max
    omega_condition = -1*np.pi <= omega <= np.pi

    period_condition = 0.5 <= period <= period_max

    logs_condition = -10 < logs < 0

    conditions = [K_condition, gamma_condition, phi_condition, ecc_condition,
                  omega_condition, period_condition, logs_condition]

    if not np.all(np.array(conditions)):
        return -np.inf

    lp = 0
    if len(rvoffsets):
        for rvo in rvoffsets:
            conditions.append( -5 <= rvo <= 5 )
            lp += -0.5*((rvo)/1.5)**2

    #Gaia priors
    lp += -0.5*((period-gaia_dict['Per'])/gaia_dict['e_Per'])**2
    lp += -0.5*((ecc-gaia_dict['ecc'])/gaia_dict['e_ecc'])**2
    lp += -0.5*((gamma-gaia_dict['Vcm'])/gaia_dict['e_Vcm'])**2
    lp += -0.5*((K1-gaia_dict['K1'])/gaia_dict['e_K1'])**2
    lp += -0.5*((omega-gaia_dict['omega'])/gaia_dict['e_omega'])**2

    return lp


@numba.jit(nopython=True)
def lnprior_sb2(theta, period_mu=None, period_sigma=None, ecc_max=None, period_max=3000):

    K1, K2, gamma, phi0, ecc, omega, period, logs1, logs2 = theta[:9]
    rvoffsets = theta[9:]

    if np.any(np.isinf(theta)):
        return -np.inf

    K_condition = (K1 > 0) & (K2 > 0)
    gamma_condition = True
    phi_condition = 0 <= phi0 <= 2*np.pi

    if ecc_max is None:
        ecc_condition = 0 <= ecc <= 0.9
    else:
        ecc_condition = 0 <= ecc <= ecc_max
    omega_condition = -1*np.pi <= omega <= np.pi

    period_condition = 0.5 <= period <= period_max

    logs_condition = (-10 < logs1 < 0) and (-10 < logs2 < 0)

    conditions = [K_condition, gamma_condition, phi_condition, ecc_condition,
                  omega_condition, period_condition, logs_condition]

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
def lnprob_sb1(theta, t, rv1, rv_err1, 
               period_mu=None, period_sigma=None, ecc_max=None, period_max=3000, gaia_dict=None):

    if gaia_dict is None:
        lp = lnprior_sb1(theta, period_mu=period_mu, period_sigma=period_sigma, ecc_max=ecc_max,
                     period_max=period_max)
    else:
        lp = lnprior_sb1_gaia(theta, gaia_dict, ecc_max=ecc_max, period_max=period_max)

    if np.isinf(lp):
        return -np.inf

    like = log_likelihood_sb1(theta, t, rv1, rv_err1)
    if np.isinf(like):
        return -np.inf

    return lp+like

@numba.jit(nopython=True)
def lnprob_sb2(theta, t, rv1, rv_err1, rv2, rv_err2, 
               period_mu=None, period_sigma=None, ecc_max=None, period_max=3000):

    lp = lnprior_sb2(theta, period_mu=period_mu, period_sigma=period_sigma, ecc_max=ecc_max)

    if np.isinf(lp):
        return -np.inf

    like = log_likelihood_sb2(theta, t, rv1, rv_err1, rv2, rv_err2)
    if np.isinf(like):
        return -np.inf

    return lp+like
