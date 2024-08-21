#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 22:35:23 2024

@author: robertgc
"""
# %%
import openturns as ot
from openturns.viewer import View
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats

# %%
# Load meta models and data
from fuel_performance import FuelPerformance

fp = FuelPerformance()
ndim = fp.Xtrain.getDimension()
nexp = fp.ytrain.getDimension()

# %%
# Defaulting gb_sweeping and athermal release
metamodels = fp.models

# %%

# Defining some likelihoods
likes = fp.likes

# %%

# Random vector for sampling of mu
mu_rv = ot.RandomVector(ot.TruncatedNormal())
mu_desc = ["$\\mu$_{{{}}}".format(label) for label in fp.Xtrain.getDescription()]

# Random vector for sampling of sigma
var_rv = ot.RandomVector(ot.TruncatedDistribution(ot.InverseGamma(), 0.0, 1.0))
sigma_desc = ["$\\sigma$_{{{}}}".format(label) for label in fp.Xtrain.getDescription()]


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

        # state description: mu values, then sigma values, then for each experiment x values
        state_length = (1 + 1 + nexp) * ndim
        super().__init__(state_length, 4)
        self._xindices = range(state_length)[2 * ndim :][dim::ndim]

        # Get lower and upper bound
        self._lb = lb
        self._ub = ub

    def _exec(self, state):
        # posterior mean of mu = empirical mean of the x values
        post_mean = np.mean([state[i] for i in self._xindices])

        # posterior std of mu = prior sigma / sqrt(nexp)
        post_std = np.sqrt(state[ndim + self._dim] / nexp)

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

        # State description: mu values, then sigma values, then for each experiment x values
        state_length = (1 + 1 + nexp) * ndim
        super().__init__(state_length, 4)
        self._xindices = range(state_length)[2 * ndim :][dim::ndim]

        # Get lower and upper bound
        self._lb = lb
        self._ub = ub

    def _exec(self, state):
        mu = state[self._dim]

        # Get squares of centered xvalues from the state
        squares = [(state[i] - mu) ** 2 for i in self._xindices]

        post_lambda = 2.0 / np.sum(squares)  # rate lambda =  1 / beta
        post_k = nexp / 2.0  # shape

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

        # State description: mu values, then sigma values, then for each experiment x values
        state_length = (1 + 1 + nexp) * ndim
        super().__init__(state_length, 1)
        self._xindices = range(state_length)[2 * ndim :][exp * ndim : (exp + 1) * ndim]

        # Setup experiment number and associated model and likelihood
        self._exp = exp
        self._like = likes[exp]
        self._metamodel = metamodels[exp]

    def _exec(self, state):
        # Get the x indices of the experiment
        x = np.array([state[i] for i in self._xindices])

        # Get mu and sigma in order to normalize x
        mu = np.array([state[i] for i in range(ndim)])
        sig = np.sqrt([state[i] for i in range(ndim, 2 * ndim)])
        normalized = (x - mu) / sig

        # Compute the log-prior density
        logprior = np.sum([ot.DistFunc.logdNormal(normalized[i]) for i in range(ndim)])

        # Use the metamodel to predict the experiment and compute the log-likelihood
        pred = self._metamodel(x)
        loglikelihood = self._like.computeLogPDF(pred)

        # Return the log-posterior, i.e. the sum of the log-prior and the log-likelihood
        return [logprior + loglikelihood]


# %%
# Lower and upper bounds for :math:`\mu_1, \mu_2, \mu_3`
lbs = [0.1, 0.1, 1e-4]
ubs = [40.0, 10.0, 1.0]

# %%
# Lower and upper bounds for :math:`\sigma_1^2, \sigma_2^2, \sigma_3^2`
lbs_var = np.array([0.1, 0.1, 0.1]) ** 2
ubs_var = np.array([40, 20, 10]) ** 2

# %%
# Initial state
initial_mus = [10.0, 5.0, 0.3]
initial_sigma_squares = [20.0 ** 2, 15.0 ** 2, 0.5 ** 2]
initial_x = np.repeat([[19.0, 4.0, 0.4]], repeats=nexp, axis=0).flatten().tolist()
initial_state = initial_mus + initial_sigma_squares + initial_x

# %%
# Support of the prior (and thus posterior) distribution
support = ot.Interval(
    lbs + lbs_var.tolist() + nexp * lbs, ubs + ubs_var.tolist() + nexp * ubs
)

# %%
# Remove restriction on the proposal probability of the origin
ot.ResourceMap.SetAsScalar("Distribution-QMin", 0.0)
ot.ResourceMap.SetAsScalar("Distribution-QMax", 1.0)

# %%
# Create the list of all samplers in the Gibbs algorithm.
# We start with the samplers of :math:`\mu_1, \mu_2, \mu_3`.
# We are able to directly sample these conditional distributions,
# hence we use the :class:`~openturns.RandomVectorMetropolisHastings` class.

samplers = [
    ot.RandomVectorMetropolisHastings(
        mu_rv,
        initial_state,
        [i],
        ot.Function(PosteriorParametersMu(dim=i, lb=lbs[i], ub=ubs[i])),
    )
    for i in range(ndim)
]

# %%
# We continue with the samplers of :math:`\sigma_1^2, \sigma_2^2, \sigma_3^2`.
# We are alse able to directly sample these conditional distributions.

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

# %%
# We finish with the samplers of the :math:`\vect{x}_i`, with :math:`1 \leq i \leq n_{exp}`.
# Each of these samplers outputs points in a :math:`n_{dim}`-dimensional space.
# We are not able to directly sample these conditional posterior distributions,
# so we resort to random walk Metropolis-Hastings.

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
# Run all samplers independently once to make sure they work.
sampler_realizations = [s.getRealization() for s in samplers]

# %%
# The Gibbs algorithm combines all these samplers.

sampler = ot.Gibbs(samplers)
x_desc = []
for i in range(1, nexp + 1):
    x_desc += ["x_{{{}, {}}}".format(label, i) for label in fp.Xtrain.getDescription()]
sampler.setDescription(mu_desc + sigma_desc + x_desc)

# %%
# Run this Metropolis-within-Gibbs algorithm and check the acceptance rates
# for the Random walk Metropolis-Hastings samplers.

samples = sampler.getSample(2000)
acceptance = [
    sampler.getMetropolisHastingsCollection()[i].getAcceptanceRate()
    for i in range(len(samplers))
]

print("Minimum acceptance rate = ", np.min(acceptance))
print("Maximum acceptance rate for random walk MH = ", np.max(acceptance[2 * ndim :]))

# %%
# 
names = ["diff", "gbsat", "crack"]

hypost = samples.asDataFrame().iloc[:, :6]  # interesting to look at whole sample
hypost.iloc[:, -3:] = hypost.iloc[:, -3:].apply(np.sqrt)


hypost.columns = [f"{p}_{{{n}}}" for p in ["$\\mu$", "$\\sigma$"] for n in names]


# %%

mu = hypost.iloc[:, :3]
sig = hypost.iloc[:, 3:6]


# %%
normal_collection = [ot.Normal(mu.iloc[i], sig.iloc[i]) for i in range(mu.shape[0])]
normal_mixture = ot.Mixture(normal_collection)
normal_mixture.setDescription(fp.Xtrain.getDescription())
rv_normal_mixture = ot.RandomVector(normal_mixture)
marg_samples = normal_mixture.getSample(nexp)
rv_models = [ot.CompositeRandomVector(model, rv_normal_mixture) for model in metamodels]
predictions = [rv.getSample(100) for rv in rv_models]
prediction_medians = [sam.computeMedian()[0] for sam in predictions]
prediction_lb = [sam.computeQuantile(0.05)[0] for sam in predictions]
prediction_ub = [sam.computeQuantile(0.95)[0] for sam in predictions]

yerr = np.abs(np.column_stack([prediction_lb, prediction_ub]).T - prediction_medians)

# %%

l = np.linspace(0, 0.5)

plt.errorbar(fp.meas_v, prediction_medians, yerr, fmt="o")

plt.xlabel("Measurements")
plt.ylabel("Prediction ranges")


plt.plot(l, l, "--")
plt.show()


# %%
# Represent the last sampled points (i.e. those which are least dependent on the initial state)
# We are only interested in the :math:`\mu` and :math:`\sigma` parameters.

reduced_samples = samples[1000:, 0:6]

# %%
# It is possible to quickly draw pair plots.
# Here we tweak the rendering a little.

pair_plots = ot.VisualTest.DrawPairs(reduced_samples)

for i in range(pair_plots.getNbRows()):
    for j in range(pair_plots.getNbColumns()):
        graph = pair_plots.getGraph(i, j)
        graph.setXTitle(pair_plots.getGraph(pair_plots.getNbRows() - 1, j).getXTitle())
        graph.setYTitle(pair_plots.getGraph(i, 0).getYTitle())
        pair_plots.setGraph(i, j, graph)

_ = View(pair_plots)
       
# %%
# Create an enhanced pair plots grid with histograms on the diagonal.

full_grid = ot.GridLayout(pair_plots.getNbRows() + 1, pair_plots.getNbColumns() + 1)

for i in range(full_grid.getNbRows()):
    hist = ot.HistogramFactory().build(reduced_samples.getMarginal(i))
    pdf = hist.drawPDF()
    pdf.setLegends([""])
    pdf.setTitle("")
    full_grid.setGraph(i, i, pdf)

for i in range(pair_plots.getNbRows()):
    for j in range(pair_plots.getNbColumns()):
        if len(pair_plots.getGraph(i, j).getDrawables()) > 0:
            full_grid.setGraph(i + 1, j, pair_plots.getGraph(i, j))
    
_ = View(full_grid)


# %%
# Finally superimpose contour plots of the 2D marginal PDFs on the pairplots.

ot.ResourceMap.SetAsBool("Contour-DefaultIsFilled", True)
ot.ResourceMap.SetAsString("Contour-DefaultColorMap", "viridis")

for i in range(1, full_grid.getNbRows()):
    for j in range(i):
        graph = full_grid.getGraph(i, j)
        bb = graph.getBoundingBox()
        cloud = graph.getDrawable(0).getImplementation()
        cloud.setPointStyle(".")
        data = cloud.getData()
        dist = ot.KernelSmoothing().build(data)
        contour = dist.drawPDF().getDrawable(0).getImplementation()
        contour.setLevels(np.linspace(0.0, contour.getLevels()[-1], 10))
        graph.setDrawables([contour, cloud])
        graph.setBoundingBox(bb)
        full_grid.setGraph(i, j, graph)

_ = View(full_grid, scatter_kw={"alpha": 0.2})


# %%
# ot.ResourceMap.Reload()
