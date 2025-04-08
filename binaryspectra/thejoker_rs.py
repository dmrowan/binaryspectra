#!/usr/bin/env python

import argparse
from astropy import units as u
import thejoker.units as xu
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import pymc as pm
import thejoker as tj
import subprocess


desc="""
Script to run thejoker rejection sampler
"""

def rejection_sampling(inputfile=None, outputfile=None, savefig=None, 
                       P_min=0.5, P_max=3000, sigma_K0=150, sigma_v=10, n_samples=10_000_000,
                       circular_orbit=False,
                       period_mu=None, period_sigma=None, kwargs_pickle=None):
    
    if kwargs_pickle is not None:
        with open(kwargs_pickle, 'rb') as p:
            kwargs_pickle = pickle.load(p)
        return rejection_sampling(**kwargs_pickle)
    
    df = pd.read_csv(inputfile)

    data = tj.RVData(t=df.JD.to_numpy(), rv=df.RV1.to_numpy()*u.km/u.s,rv_err=df.RV1_err.to_numpy()*u.km/u.s)

    with pm.Model() as model:

        #Extra uncertainty param
        s = xu.with_unit(pm.Lognormal('s', -2, 1), u.km/u.s)

        kwargs = {}
        #Period prior
        if (period_mu is not None) and (period_sigma is not None):
            P = xu.with_unit(pm.Normal('P', period_mu, period_sigma), u.day)
            kwargs['pars'] = {'P':P}
        else:
            kwargs['P_min'] = P_min*u.day
            kwargs['P_max'] = P_max*u.day

        if circular_orbit:
            e = xu.with_unit(pm.Constant('e', 0), u.one)
            if 'pars' in kwargs.keys():
                kwargs['pars']['e'] = e
            else:
                kwargs['pars'] = {'e':e}

        prior_joint = tj.JokerPrior.default(
                sigma_K0=sigma_K0*u.km/u.s,
                sigma_v=sigma_v*u.km/u.s,
                s=s,
                **kwargs)

    prior_samples_joint = prior_joint.sample(n_samples)
    joker_joint = tj.TheJoker(prior_joint)
    samples_joint = joker_joint.rejection_sample(data, prior_samples_joint, max_posterior_samples=512)

    if savefig is not None:
        plot_rs_results(data, samples_joint, savefig)

    samples_joint.tbl.to_pandas().to_csv(outputfile, index=False)

def plot_rs_results(data, samples_joint, savefig):
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    fig.subplots_adjust(top=.98, right=.98, hspace=0)

    _ = tj.plot_phase_fold(samples_joint.mean(), data, ax=axes[0], add_labels=False)
    _ = tj.plot_phase_fold(samples_joint.mean(), data, ax=axes[1], residual=True)

    axes[1].set_xlabel('Orbital Phase', fontsize=20)
    axes[0].set_ylabel('RV (km/s)', fontsize=20)

    axes[1].set_ylabel('RV Residuals (km/s)', fontsize=20)

    fig.savefig(savefig)
    plt.close()


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('kwargs_pickle', type=str)

    args = parser.parse_args()

    rejection_sampling(kwargs_pickle=args.kwargs_pickle)
