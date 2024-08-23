"""
Bayesian calibration of a hierarchical fuel performance model
=============================================================
"""
# %%
import openturns as ot
from openturns.viewer import View
import numpy as np
import matplotlib.pyplot as plt

# %%
# Set random seed

ot.RandomGenerator.SetSeed(0)

# ot.ResourceMap.SetAsUnsignedInteger("RandomWalkMetropolisHastings-DefaultBurnIn", 10000)

# %%
# Load the models
import os

if "gibbs_bepu_2024_turns" in os.getcwd():
    from fission_gas import FissionGasRelease

    fgr = FissionGasRelease()
else:
    from openturns.usecases import fission_gas

    fgr = fission_gas.FissionGasRelease()
desc = fgr.get_input_description()  # description of the model inputs (diff, crack)
ndim = len(desc) # dimension of the model inputs: 2
nexp = fgr.measurement_values.getSize() # number of experiments (each has a specific model)
models = fgr.models  # the nexp models

# %%
# Each experiment :math:`i` produced one measurement value,
# which is used to define the likelihood of the associated model :math:`\mathcal{\model}_i`
# and latent variable :math:`\vect{x}_i = (x_{i, diff}, x_{i, crack})`.

likelihoods = [ot.Normal(v, fgr.measurement_uncertainty(v)) for v in fgr.measurement_values]

# %%
# The unobserved model inputs :math:`x_{\mathrm{diff}, i}, i=1...\sampleSize_{\mathrm{exp}}`
# (resp. :math:`x_{\mathrm{crack}, i}, i=1...\sampleSize_{\mathrm{exp}}`)
# are i.i.d. random variable which follow a normal distribution with
# mean parameter :math:`\mu_{\mathrm{diff}}` (resp. :math:`\mu_{\mathrm{crack}}`)
# and standard deviation parameter :math:`\sigma_{\mathrm{diff}}` (resp. :math:`\sigma_{\mathrm{crack}}`).
#
# The network plot from the page :ref:`fission_gas` can thus be updated:
#
# .. figure:: ../../../../_static/fission_gas_network_calibration.png
#     :align: center
#     :alt: use case geometry
#     :width: 50%
#
# In the network above, full arrows represent deterministic relationships and dashed arrows probabilistic relationships.
# More precisely, the conditional distribution of the node at the end of two dashed arrows when (only) the starting nodes are known
# is a normal distribution with parameters equal to these starting nodes.
#
# The goal of this study is to calibrate the parameters :math:`\mu_{\mathrm{diff}}`, :math:`\sigma_{\mathrm{diff}}`,
# resp. :math:`\mu_{\mathrm{crack}}` and :math:`\sigma_{\mathrm{crack}}`.
# To perform Bayesian calibration, we set a uniform prior distribution on :math:`\mu_{\mathrm{diff}}` and :math:`\mu_{\mathrm{crack}}`
# and the limit of a truncated inverse gamma distribution with parameters :math:`(\lambda, k)` when :math:`\lambda \to \infty` and :math:`k \to 0`.
# The parameters of the prior distributions are defined later.

# %%
# This choice of prior distributions means that the posterior is partially conjugate.
# For instance, the conditional posterior distribution of :math:`\mu_{\mathrm{diff}}`
# (resp. :math:`\mu_{\mathrm{crack}}`)
# is truncated normal with the following parameters (for :math:`\mu_{\mathrm{crack}}` simply replace :math:`\mathrm{diff}` with :math:`\mathrm{crack}` in what follows) :
#
# - The truncation parameters are the bounds of the prior uniform distribution.
# - The mean parameter is :math:`\frac{1}{\sampleSize_{\mathrm{exp}}} \sum_{i=1}^{\sampleSize_{\mathrm{exp}}} x_{\mathrm{diff}, i}`.
# - The standard deviation parameter is :math:`\sqrt{\frac{\sigma_{\mathrm{diff}}}{\sampleSize_{\mathrm{exp}}}}`.
#
# Let us prepare a random vector to sample the conditional posterior
# distributions of :math:`\mu_{diff}` and :math:`\mu_{crack}`.

mu_rv = ot.RandomVector(ot.TruncatedNormal())
mu_desc = ["$\\mu$_{{{}}}".format(label) for label in desc]

# %%
# The conditional posterior distribution of :math:`\sigma_{\mathrm{diff}}`
# (resp. :math:`\sigma_{\mathrm{crack}}`)
# is truncated inverse gamma with the following parameters (for :math:`\sigma_{\mathrm{crack}}` simply replace :math:`\mathrm{diff}` with :math:`\mathrm{crack}` in what follows) :
#
# - The truncation parameters are the truncation parameters of the prior distribution.
# - The :math:`\lambda` parameter is :math:`\frac{2}{\sum_{i=1}^{\sampleSize_{\mathrm{exp}}} \left(x_{\mathrm{diff}, i} - \mu_{\mathrm{diff}} \right)^2}`.
# - The :math:`k` parameter is :math:`\sqrt{\frac{\sampleSize_{\mathrm{exp}}}{2}}`.
#
# Let us prepare a random vector to sample the conditional posterior
# distribution of :math:`\sigma_{diff}^2` and :math:`\sigma_{crack}^2`.

sigma_square_rv = ot.RandomVector(ot.TruncatedDistribution(ot.InverseGamma(), 0.0, 1.0))
sigma_square_desc = ["$\\sigma$_{{{}}}^2".format(label) for label in desc]


# %%
# We define 3 function templates which produce:
#
# - the parameters of the conditional posterior distributions of the :math:`\mu` parameters
# - the parameters of the conditional posterior distributions of the :math:`\sigma` parameters
# - the conditional posterior log-PDF of the latent variables.


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

        return [post_lambda, post_k, self._lb, self._ub]


class PosteriorLogDensityX(ot.OpenTURNSPythonFunction):
    """Outputs the conditional posterior density (up to an additive constant)
    of the 2D latent variable x_i = (x_{i, diff}, x_{i, x_{i, crack})
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
        self._like = likelihoods[exp]
        self._model = models[exp]

    def _exec(self, state):
        # Get the x indices of the experiment
        x = np.array([state[i] for i in self._xindices])

        # Get mu and sigma in order to normalize x
        mu = np.array([state[i] for i in range(ndim)])
        sig = np.sqrt([state[i] for i in range(ndim, 2 * ndim)])
        normalized = (x - mu) / sig

        # Compute the log-prior density
        logprior = np.sum([ot.DistFunc.logdNormal(normalized[i]) for i in range(ndim)])

        # Use the model to predict the experiment and compute the log-likelihood
        pred = self._model(x)
        loglikelihood = self._like.computeLogPDF(pred)

        # Return the log-posterior, i.e. the sum of the log-prior and the log-likelihood
        return [logprior + loglikelihood]


# %%
# Lower and upper bounds for :math:`\mu_{diff}, \mu_{crack}`
lbs = [0.1, 1e-4]
ubs = [40.0, 1.0]

# %%
# Lower and upper bounds for :math:`\sigma_{diff}^2, \sigma_{crack}^2`
lbs_sigma_square = np.array([0.1, 0.1]) ** 2
ubs_sigma_square = np.array([40, 10]) ** 2

# %%
# Initial state
initial_mus = [10.0, 0.3]
initial_sigma_squares = [20.0 ** 2, 0.5 ** 2]
initial_x = np.repeat([[19.0, 0.4]], repeats=nexp, axis=0).flatten().tolist()
initial_state = initial_mus + initial_sigma_squares + initial_x

# %%
# Support of the prior (and thus posterior) distribution
support = ot.Interval(
    lbs + lbs_sigma_square.tolist() + nexp * lbs,
    ubs + ubs_sigma_square.tolist() + nexp * ubs,
)

# %%
# Remove the restriction on the proposal probability of the origin.
ot.ResourceMap.SetAsScalar("Distribution-QMin", 0.0)
ot.ResourceMap.SetAsScalar("Distribution-QMax", 1.0)

# %%
# Create the list of all samplers in the Gibbs algorithm.
# We start with the samplers of :math:`\mu_{diff}, \mu_{crack}`.
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
# We continue with the samplers of :math:`\sigma_{diff}^2, \sigma_{crack}^2`.
# We are alse able to directly sample these conditional distributions.

samplers += [
    ot.RandomVectorMetropolisHastings(
        sigma_square_rv,
        initial_state,
        [ndim + i],
        ot.Function(
            PosteriorParametersSigmaSquare(
                dim=i, lb=lbs_sigma_square[i], ub=ubs_sigma_square[i]
            )
        ),
    )
    for i in range(ndim)
]

# %%
# We finish with the samplers of the :math:`\vect{x}_i`, with :math:`1 \leq i \leq n_{exp}`.
# Each of these samplers outputs points in a :math:`\sampleSize_{\mathrm{exp}}`-dimensional space.
# We are not able to directly sample these conditional posterior distributions,
# so we resort to random walk Metropolis-Hastings.

for exp in range(nexp):
    base_index = 2 * ndim + ndim * exp

    samplers += [
        ot.RandomWalkMetropolisHastings(
            ot.Function(PosteriorLogDensityX(exp=exp)),
            support,
            initial_state,
            ot.Normal([0.0] * 2, [6.0, 0.15]),
            [base_index + i for i in range(ndim)],
        )
    ]

# %%
# The Gibbs algorithm combines all these samplers.

sampler = ot.Gibbs(samplers)
x_desc = []
for i in range(1, nexp + 1):
    x_desc += ["x_{{{}, {}}}".format(label, i) for label in desc]
sampler.setDescription(mu_desc + sigma_square_desc + x_desc)

# %%
# Run this Metropolis-within-Gibbs algorithm and check the acceptance rates
# for the Random walk Metropolis-Hastings samplers.

samples = sampler.getSample(2000)
acceptance = [
    sampler.getMetropolisHastingsCollection()[i].getAcceptanceRate()
    for i in range(6, len(samplers))
]

adaptation_factor = [
    sampler.getMetropolisHastingsCollection()[i]
    .getImplementation()
    .getAdaptationFactor()
    for i in range(6, len(samplers))
]

print("Minimum acceptance rate = ", np.min(acceptance))
print("Maximum acceptance rate for random walk MH = ", np.max(acceptance[2 * ndim :]))

# %%
# Represent the last sampled points (i.e. those which are least dependent on the initial state)
# We are only interested in the :math:`\mu` and :math:`\sigma` parameters.

reduced_samples = samples[:, 0:4]

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
# Create an enhanced pair plots grid with histograms of the marginals on the diagonal.

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
# Finally superimpose contour plots of the KDE-estimated 2D marginal PDFs on the pairplots.

ot.ResourceMap.SetAsBool("Contour-DefaultIsFilled", True)
ot.ResourceMap.SetAsString("Contour-DefaultColorMap", "viridis")

# sphinx_gallery_thumbnail_number = 3 
for i in range(1, full_grid.getNbRows()):
    for j in range(i):
        graph = full_grid.getGraph(i, j)
        cloud = graph.getDrawable(0).getImplementation()
        cloud.setPointStyle(".")
        data = cloud.getData()
        dist = ot.KernelSmoothing().build(data)
        contour = dist.drawPDF().getDrawable(0).getImplementation()
        contour.setLevels(np.linspace(0.0, contour.getLevels()[-1], 10))
        graph.setDrawables([contour, cloud])
        graph.setBoundingBox(contour.getBoundingBox())
        full_grid.setGraph(i, j, graph)

_ = View(full_grid, scatter_kw={"alpha": 0.1})


# %%
# Retrieve the :math:`\mu` and :math:`\sigma^2` columns in the sample.

mu = samples.getMarginal(["$\\mu$_{{{}}}".format(label) for label in desc])
sigma_square = np.sqrt(
    samples.getMarginal(["$\\sigma$_{{{}}}^2".format(label) for label in desc])
)


# %%
# Build the joint distribution of the latent variables :math:`x_{diff}, x_{crack}`
# obtained when :math:`\mu_{\mathrm{diff}}`, :math:`\sigma_{\mathrm{diff}}`,
# :math:`\mu_{\mathrm{crack}}` and :math:`\sigma_{\mathrm{crack}}` 
# follow their joint posterior distribution.
# It is estimated as a mixture of truncated :math:`\sampleSize_{\mathrm{exp}}`-dimensional normal distributions
# corresponding to the posterior samples of the :math:`\mu_{\mathrm{diff}}`, :math:`\mu_{\mathrm{crack}}`, 
# :math:`\sigma_{\mathrm{diff}}` and :math:`\sigma_{\mathrm{crack}}` parameters.

truncation_interval = ot.Interval(lbs, ubs)
normal_collection = [
    ot.TruncatedDistribution(ot.Normal(mean, std), truncation_interval)
    for (mean, std) in zip(mu, sigma_square)
]
normal_mixture = ot.Mixture(normal_collection)
normal_mixture.setDescription(desc)

# %%
# Build a collection of random vectors such that the distribution
# of each is the push-forward of the marginal distribution of :math:`(x_{diff}, x_{crack})`
# defined above through one of the nexp models.

rv_normal_mixture = ot.RandomVector(normal_mixture)
rv_models = [ot.CompositeRandomVector(model, rv_normal_mixture) for model in models]

# %%
# Get a Monte-Carlo estimate of the median, 0.05 quantile and 0.95 quantile
# of these push-forward distributions.

predictions = [rv.getSample(100) for rv in rv_models]
prediction_medians = [sam.computeMedian()[0] for sam in predictions]
prediction_lb = [sam.computeQuantile(0.05)[0] for sam in predictions]
prediction_ub = [sam.computeQuantile(0.95)[0] for sam in predictions]

# %%
# These push-forward distributions are the distributions
# of the model predictions when :math:`\mu_{\mathrm{diff}}`, :math:`\mu_{\mathrm{crack}}`, 
# :math:`\sigma_{\mathrm{diff}}` and :math:`\sigma_{\mathrm{crack}}` follow
# their joint posterior distribution.
# They can be compared to the actual measurements to represent predictive accuracy.

yerr = np.abs(np.column_stack([prediction_lb, prediction_ub]).T - prediction_medians)
plt.errorbar(fgr.measurement_values, prediction_medians, yerr, fmt="o")
plt.xscale("log")

l = np.linspace(0, 0.5)
plt.plot(l, l, "--")

plt.xlabel("Measurements")
plt.ylabel("Prediction ranges")

plt.show()


# %%
ot.ResourceMap.Reload()