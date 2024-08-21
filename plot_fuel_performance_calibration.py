#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 22:35:23 2024

@author: robertgc
"""
# %%
import pickle
import openturns as ot
import numpy as np
import matplotlib.pyplot as plt
from corner import corner
from lib.custom_openturns import (
    log_pdf_x,
    mu_post_truncated_normal,
    var_post_truncated_normal,
)
import pandas as pd
import scipy.stats as stats

from copulogram import Copulogram

# %%

# Load meta models and data
from fuel_performance import FuelPerformance

fp = FuelPerformance()

# %%
# Defaulting gb_sweeping and athermal release
metamodels = fp.models

# %%

# Defining some likelihoods
likes = fp.likes

# %%

# Random vector for sampling of mu
mu_rv = ot.RandomVector(ot.TruncatedNormal())

# Random vector for sampling of sigma
var_rv = ot.RandomVector(ot.TruncatedDistribution(ot.InverseGamma(), 0.0, 1.0))


# %%


class PosteriorParametersMu(ot.OpenTURNSPythonFunction):
    """Outputs the parameters of the conditional posterior distribution of one
    of the mu parameters.
    This conditional posterior distribution is a TruncatedNormal distribution.

    Parameters
    ----------
    state : list of float
        Current state of the Markov chain.
        The posterior distribution is conditional to those values.

    Returns
    -------
    parameters : list of float
        Parameters of the conditional posterior distribution.
        In order: mean, standard deviation, lower bound, upper bound.
    """

    def __init__(self, dim=0, lb=-100, ub=100):
        # Get dimension and total number of dimensions
        self._dim = dim
        self._ndim = fp.Xtrain.getDimension()

        # Total number of experiments
        self._nexp = len(metamodels)

        # state description: mu values, then sigma values, then for each experiment x values
        state_length = (1 + 1 + self._nexp) * self._ndim
        super().__init__(state_length, 4)
        self._xindices = range(state_length)[2 * self._ndim :][dim :: self._ndim]

        # Get lower and upper bound
        self._lb = lb
        self._ub = ub

    def _exec(self, state):
        # posterior mean of mu = empirical mean of the x values
        post_mean = np.mean([state[i] for i in self._xindices])

        # posterior std of mu = prior sigma / sqrt(nexp)
        post_std = np.sqrt(state[self._ndim + self._dim] / self._nexp)

        # Hyperparameters of a truncated normal
        return [post_mean, post_std, self._lb, self._ub]


class PosteriorParametersSigmaSquare(ot.OpenTURNSPythonFunction):
    """Outputs the parameters of the conditional posterior distribution of one
    of the sigma parameters.
    This conditional posterior distribution is a Truncated InverseGamma distribution.

    Parameters
    ----------
    state : list of float
        Current state of the Markov chain.
        The posterior distribution is conditional to those values.

    Returns
    -------
    parameters : list of float
        Parameters of the conditional posterior distribution.
        In order: lambda, k (shape), lower bound, upper bound.
    """

    def __init__(self, dim=0, lb=1e-4, ub=100):
        # Get dimension and total number of dimensions
        self._dim = dim
        self._ndim = fp.Xtrain.getDimension()

        # Total number of experiments
        self._nexp = len(metamodels)

        # State description: mu values, then sigma values, then for each experiment x values
        state_length = (1 + 1 + self._nexp) * self._ndim
        super().__init__(state_length, 4)
        self._xindices = range(state_length)[2 * self._ndim :][dim :: self._ndim]

        # Get lower and upper bound
        self._lb = lb
        self._ub = ub

    def _exec(self, state):
        mu = state[self._dim]

        # Get squares of centered xvalues from the state
        squares = [(state[i] - mu) ** 2 for i in self._xindices]

        post_lambda = 2.0 / np.sum(squares)  # rate lambda =  1 / beta
        post_k = self._nexp / 2.0  # shape

        # Hyperparameters of a truncated inverse gamma
        return [post_lambda, post_k, self._lb, self._ub]


# Create a logpdf for the latent parameters
class PosteriorLogDensityX(ot.OpenTURNSPythonFunction):
    """Outputs the conditional posterior density (up to an additive constant)
    of the 3D latent variable x_i = (x_i1, x_i2, x_i3)
    corresponding to one experiment i.

    Parameters
    ----------
    state : list of float
        Current state of the Markov chain.
        The posterior distribution is conditional to those values.

    Returns
    -------
    log_density : list of one float
        PLog-density (up to an additive constant) of the conditional posterior.
    """

    def __init__(self, exp):
        # Get total number of dimensions
        self._ndim = fp.Xtrain.getDimension()

        # Total number of experiments
        self._nexp = len(metamodels)

        # State description: mu values, then sigma values, then for each experiment x values
        state_length = (1 + 1 + self._nexp) * self._ndim
        super().__init__(state_length, 1)
        self._xindices = range(state_length)[2 * self._ndim :][
            exp * self._ndim : (exp + 1) * self._ndim
        ]

        # Setup experiment number and associated model and likelihood
        self._exp = exp
        self._like = likes[exp]
        self._metamodel = metamodels[exp]

    def _exec(self, state):
        # Get the x indices of the experiment
        x = np.array([state[i] for i in self._xindices])

        # Get mu and sigma in order to normalize x
        mu = np.array([state[i] for i in range(self._ndim)])
        sig = np.sqrt([state[i] for i in range(self._ndim, 2 * self._ndim)])
        normalized = (x - mu) / sig

        # Compute the log-prior density
        logprior = np.sum(
            [ot.DistFunc.logdNormal(normalized[i]) for i in range(self._ndim)]
        )

        # Use the metamodel to predict the experiment and compute the log-likelihood
        pred = self._metamodel(x)
        loglikelihood = self._like.computeLogPDF(pred)

        # Return the log-posterior, i.e. the sum of the log-prior and the log-likelihood
        return [logprior + loglikelihood]


# %%

ndim = fp.Xtrain.getDimension()
nexp = fp.ytrain.getDimension()

lbs = [0.1, 0.1, 1e-4]
ubs = [40, 10, 1]

means = np.array([20.0, 5.0, 0.5])

initial_state = [
    9.78478,
    7.18681,
    0.292831,
    28.8707,
    0.518241,
    0.0262414,
    11.702,
    7.15259,
    0.504642,
    8.93815,
    7.20632,
    0.227912,
    10.7733,
    7.16673,
    0.423543,
    9.86638,
    7.16642,
    0.34344,
    9.52776,
    7.1736,
    0.19822,
    11.1469,
    7.12211,
    0.418397,
    9.08546,
    7.19024,
    0.14975,
    8.78541,
    7.22476,
    0.248703,
    10.3239,
    7.18289,
    0.342643,
    7.50655,
    7.23985,
    0.221591,
    9.47391,
    7.21092,
    0.235646,
    8.47214,
    7.22322,
    0.198574,
    9.8565,
    7.18337,
    0.308543,
    10.2439,
    7.17974,
    0.321506,
    8.07999,
    7.18148,
    0.280854,
    8.72788,
    7.20715,
    0.201784,
    11.8741,
    7.1397,
    0.468259,
    7.44071,
    7.2692,
    0.183411,
    9.93625,
    7.1716,
    0.308075,
    10.22,
    7.18294,
    0.343424,
    9.60504,
    7.19412,
    0.290279,
    16.2637,
    7.10373,
    0.572483,
    8.60862,
    7.20276,
    0.248138,
    8.75301,
    7.17151,
    0.241266,
    9.777,
    7.1634,
    0.317432,
    9.342,
    7.24039,
    0.116625,
    9.04912,
    7.17853,
    0.0566619,
    10.0503,
    7.19752,
    0.322601,
    15.0076,
    7.13199,
    0.536823,
    5.12208,
    7.224,
    0.141905,
    9.88428,
    7.19038,
    0.30269,
]  # 99

# initial_state = np.ones(2*ndim+ndim*nexp)

# initial_state[:ndim] = means
# # use different initial sigmas to distinguish logpdfs
# initial_state[ndim:2*ndim] = np.array([20.0, 15.0, 0.5])**2

# initial_state[2*ndim::3] = 19.0
# initial_state[2*ndim+1::3] = 4.0 
# initial_state[2*ndim+2::3] = 0.4


lbs_var = np.array([0.1, 0.1, 0.1]) ** 2
ubs_var = np.array([40, 20, 10]) ** 2

support = ot.Interval(
    lbs + lbs_var.tolist() + nexp * lbs, ubs + ubs_var.tolist() + nexp * ubs
)

# %%
# Remove restriction on the proposal probability of the origin
ot.ResourceMap.SetAsScalar("Distribution-QMin", 0.0)
ot.ResourceMap.SetAsScalar("Distribution-QMax", 1.0)

samplers = [
    ot.RandomVectorMetropolisHastings(
        mu_rv,
        initial_state,
        [i],
        ot.Function(PosteriorParametersMu(dim=i, lb=lbs[i], ub=ubs[i])),
    )
    for i in range(ndim)
]


samplers += [
    ot.RandomVectorMetropolisHastings(
        var_rv,
        initial_state,
        [ndim + i],
        ot.Function(
            PosteriorParametersSigmaSquare(dim=i, lb=lbs_var[i], ub=ubs_var[i])
        ),
    )
    for i in range(ndim)
]

for exp in range(nexp):
    base_index = 2 * ndim + ndim * exp

    samplers += [
        ot.RandomWalkMetropolisHastings(
            ot.Function(PosteriorLogDensityX(exp=exp)),
            support,
            initial_state,
            ot.Normal([0.0] * 3, [20.0, 5.0, 0.5]),
            [base_index + i for i in range(ndim)],
        )
    ]
# %%
real = [s.getRealization() for s in samplers]
# %%

sampler = ot.Gibbs(samplers)

# %%

samples = sampler.getSample(24000)

acceptance = [
    sampler.getMetropolisHastingsCollection()[i].getAcceptanceRate()
    for i in range(len(samplers))
]
print(acceptance)

# %%

names = ["diff", "gbsat", "crack"]

hypost = samples.asDataFrame().iloc[:, :6]  # interesting to look at whole sample
hypost.iloc[:, -3:] = hypost.iloc[:, -3:].apply(np.sqrt)


hypost.columns = [f"{p}_{{{n}}}" for p in ["$\mu$", "$\sigma$"] for n in names]

# %%

corner(hypost)
plt.show()

# %%

mu = hypost.iloc[:, :3]
sig = hypost.iloc[:, 3:6]

mu.columns = np.arange(3)
sig.columns = np.arange(3)

a = (lbs - mu) / sig
b = (ubs - mu) / sig

marg_samples = stats.truncnorm(loc=mu, scale=sig, a=a, b=b).rvs(mu.shape)

marg_samples_pd = pd.DataFrame(marg_samples)
marg_samples_pd.columns = ["diff", "gbsat", "crack"]

corner(marg_samples_pd)
plt.show()

# %%

mean_pred = np.array([metamodels[i](marg_samples).computeMean()[0] for i in range(31)])
ub_pred = np.array(
    [metamodels[i](marg_samples).computeQuantile(0.95)[0] for i in range(31)]
)
lb_pred = np.array(
    [metamodels[i](marg_samples).computeQuantile(0.05)[0] for i in range(31)]
)

yerr = np.abs(np.column_stack([lb_pred, ub_pred]).T - mean_pred)


l = np.linspace(0, 0.5)

plt.errorbar(fp.meas_v, mean_pred, yerr, fmt="o")
# plt.plot(meas_v,mean_pred,'o')
# plt.plot(meas_v,ub_pred,'o')
# plt.plot(meas_v,lb_pred,'o')

plt.xlabel("Measured fgr [-]")
plt.ylabel("GP predicted fgr [-]")


plt.plot(l, l, "--")
plt.show()

# %%
copulogram = Copulogram(hypost)
copulogram.draw(alpha=0.05, marker=".")


# %%
np.log(hypost["$\\mu$_{diff}"]).plot()
np.log(hypost["$\\mu$_{gbsat}"]).plot()
np.log(hypost["$\\mu$_{crack}"]).plot()

# %%
np.log(hypost["$\\sigma$_{diff}"]).plot()
np.log(hypost["$\\sigma$_{gbsat}"]).plot()
np.log(hypost["$\\sigma$_{crack}"]).plot()
plt.legend()

# %%
hypost["$\\sigma$_{diff}"].plot()
hypost["$\\sigma$_{gbsat}"].plot()
hypost["$\\sigma$_{crack}"].plot()
plt.legend()
