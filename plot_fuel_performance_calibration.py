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

# %%
# Load the models
import os

if "gibbs_bepu_2024_turns" in os.getcwd():
    from fuel_performance import FuelPerformance
    fp = FuelPerformance()
else:
    from openturns.usecases import fuel_performance
    fp = fuel_performance.FuelPerformance()
ndim = fp.Xtrain.getDimension() # dimension of the model inputs: 3
desc = fp.Xtrain.getDescription() # description of the model inputs (diff, gb\_satutation, crack)
nexp = fp.ytrain.getDimension() # number of experiments (each has a specific model)
models = fp.models # the nexp models

# %%
# Each experiment :math:`i` produced one measurement value,
# which is used to define the likelihood of the associated model :math:`\mathcal{M}_i`
# and latent variable :math:`\vect{x}_i = (x_{i, diff}, x_{i, gb\_satutation}, x_{i, crack})`.

likes = fp.likes 

# %%
# Random vector to sample the conditional posterior
# distribution of :math:`\vect{\mu} = (\mu_{diff}, \mu_{gb\_satutation}, \mu_{crack})`

mu_rv = ot.RandomVector(ot.TruncatedNormal())
mu_desc = ["$\\mu$_{{{}}}".format(label) for label in desc]

# %%
# Random vector to sample the conditional posterior
# distribution of :math:`\vect{\sigma}^2 = (\sigma_{diff}^2, \sigma_{gb\_satutation}^2, \sigma_{crack}^2)`

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

        # Hyperparameters of a truncated inverse gamma
        return [post_lambda, post_k, self._lb, self._ub]


class PosteriorLogDensityX(ot.OpenTURNSPythonFunction):
    """Outputs the conditional posterior density (up to an additive constant)
    of the 3D latent variable x_i = (x_{i, diff}, x_{i, gb_saturation}, x_{i, crack})
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
# Lower and upper bounds for :math:`\mu_{diff}, \mu_{gb\_satutation}, \mu_{crack}`
lbs = [0.1, 0.1, 1e-4]
ubs = [40.0, 10.0, 1.0]

# %%
# Lower and upper bounds for :math:`\sigma_{diff}^2, \sigma_{gb\_satutation}^2, \sigma_{crack}^2`
lbs_sigma_square = np.array([0.1, 0.1, 0.1]) ** 2
ubs_sigma_square = np.array([40, 20, 10]) ** 2

# %%
# Initial state
initial_mus = [10.0, 5.0, 0.3]
initial_sigma_squares = [20.0 ** 2, 15.0 ** 2, 0.5 ** 2]
initial_x = np.repeat([[19.0, 4.0, 0.4]], repeats=nexp, axis=0).flatten().tolist()
initial_state = initial_mus + initial_sigma_squares + initial_x


initial_state = [7.25437,5.34333,0.2688,16.0572,0.0400135,0.0272261,11.9245,5.37405,0.525344,5.73046,5.38231,0.0911716,9.63216,5.24496,0.374362,14.0424,5.75832,0.271396,9.87644,5.03691,0.162653,9.82311,5.39674,0.390882,8.00765,5.09337,0.16456,2.65465,5.69722,0.394443,2.5703,5.23133,0.426503,5.8409,5.17407,0.0137091,7.6153,5.28807,0.192897,6.22031,5.09896,0.170041,11.6519,4.92299,0.286779,3.35373,5.27964,0.334533,7.88469,5.57834,0.41805,2.37791,5.35143,0.224999,8.20026,5.54231,0.454754,9.27286,5.4026,0.126513,9.47674,5.26752,0.376011,7.66051,4.99955,0.240198,6.52532,5.35201,0.268828,8.92401,5.2735,0.6278,7.10376,5.37863,0.190223,9.51549,5.62299,0.22081,4.92388,5.45614,0.330533,5.33167,5.2495,0.165869,7.62818,5.45446,0.0581072,10.2911,5.17247,0.23252,14.6922,5.15172,0.731592,0.369771,4.72366,0.20958,11.0594,5.43281,0.063014]

# %%
# Support of the prior (and thus posterior) distribution
support = ot.Interval(
    lbs + lbs_sigma_square.tolist() + nexp * lbs, ubs + ubs_sigma_square.tolist() + nexp * ubs
)

# %%
# Remove the restriction on the proposal probability of the origin.
ot.ResourceMap.SetAsScalar("Distribution-QMin", 0.0)
ot.ResourceMap.SetAsScalar("Distribution-QMax", 1.0)

# %%
# Create the list of all samplers in the Gibbs algorithm.
# We start with the samplers of :math:`\mu_{diff}, \mu_{gb\_satutation}, \mu_{crack}`.
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
# We continue with the samplers of :math:`\sigma_{diff}^2, \sigma_{gb\_satutation}^2, \sigma_{crack}^2`.
# We are alse able to directly sample these conditional distributions.

samplers += [
    ot.RandomVectorMetropolisHastings(
        sigma_square_rv,
        initial_state,
        [ndim + i],
        ot.Function(
            PosteriorParametersSigmaSquare(dim=i, lb=lbs_sigma_square[i], ub=ubs_sigma_square[i])
        ),
    )
    for i in range(ndim)
]

# %%
# We finish with the samplers of the :math:`\vect{x}_i`, with :math:`1 \leq i \leq n_{exp}`.
# Each of these samplers outputs points in a 3-dimensional space.
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
    x_desc += ["x_{{{}, {}}}".format(label, i) for label in desc]
sampler.setDescription(mu_desc + sigma_square_desc + x_desc)

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
# Retrieve the :math:`\vect{\mu}` and :math:`\vect{\sigma}^2` columns in the sample.

mu = samples.getMarginal(["$\\mu$_{{{}}}".format(label) for label in desc])
sigma_square = np.sqrt(samples.getMarginal(["$\\sigma$_{{{}}}^2".format(label) for label in desc]))


# %%
# Build the joint distribution of the latent variables :math:`x_{diff}, x_{gb\_satutation}, x_{crack}`
# obtained when the :math:`\mu` and :math:`\sigma` parameters follow
# their joint posterior distribution.
# It is estimated as a mixture of normal distributions
# corresponding to the posterior samples of the :math:`\mu` and :math:`\sigma` parameters.

normal_collection = [ot.Normal(mean, std) for (mean, std) in zip(mu, sigma_square)]
normal_mixture = ot.Mixture(normal_collection)
normal_mixture.setDescription(desc)

# %%
# Build a collection of random vectors such that the distribution
# of each is the push-forward of the marginal distribution of :math:`(x_{diff}, x_{gb\_satutation}, x_{crack})`
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
# of the model predictions when the :math:`\mu` and :math:`\sigma` parameters follow
# their joint posterior distribution.
# They can be compared to the actual measurements to represent predictive accuracy.

yerr = np.abs(np.column_stack([prediction_lb, prediction_ub]).T - prediction_medians)
plt.errorbar(fp.meas_v, prediction_medians, yerr, fmt="o")
plt.xscale("log")

l = np.linspace(0, 0.5)
plt.plot(l, l, "--")

plt.xlabel("Measurements")
plt.ylabel("Prediction ranges")

plt.show()


# %%
ot.ResourceMap.Reload()
