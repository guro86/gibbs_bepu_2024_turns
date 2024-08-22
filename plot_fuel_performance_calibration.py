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

ot.ResourceMap.SetAsUnsignedInteger("RandomWalkMetropolisHastings-DefaultBurnIn", 10000)

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

        print("dim = ", self._dim)
        print([post_lambda, post_k, self._lb, self._ub])
        print("empirical standard deviation = ",  np.sqrt(np.mean(squares)))
        trinvgamma = ot.TruncatedDistribution(ot.InverseGamma(), 0.0, 1.0)
        try:
            trinvgamma.setParameter([post_lambda, post_k, self._lb, self._ub])
        except:
            print("DID NOT WORK")
            print([post_lambda, post_k, self._lb, self._ub])
            invgamma = ot.InverseGamma(post_lambda, post_k)
            print(invgamma.computeCDF(float(self._lb)))
            print(invgamma.computeCDF(float(self._ub)))
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
lbs_sigma_square = np.array([0.1, 0.00001, 0.1]) ** 2
ubs_sigma_square = np.array([40, 20, 10]) ** 2

# %%
# Initial state
initial_mus = [10.0, 5.0, 0.3]
initial_sigma_squares = [20.0 ** 2, 15.0 ** 2, 0.5 ** 2]
initial_x = np.repeat([[19.0, 4.0, 0.4]], repeats=nexp, axis=0).flatten().tolist()
initial_state = initial_mus + initial_sigma_squares + initial_x


initial_state = [7.51188,8.51937,0.292952,9.37161,0.0516549,0.0290076,9.08868,8.45879,0.48073,6.34297,8.5543,0.150879,7.93481,8.65959,0.519287,4.31882,8.39354,0.525883,6.85787,8.70104,0.189518,7.52134,9.01983,0.422947,9.55284,8.76941,0.125033,6.16268,8.03187,0.165988,6.23355,8.53377,0.427718,1.34126,8.12576,0.463676,6.8223,8.25217,0.215145,4.32857,8.48588,0.235282,4.16995,8.64144,0.439799,12.6397,8.79266,0.493053,7.17511,9.2374,0.432577,6.79614,8.72714,0.184218,9.13748,8.19622,0.476752,4.74385,8.38604,0.189267,6.50082,8.48837,0.356241,10.7832,8.43997,0.251964,10.0711,7.98261,0.200379,16.7423,8.86075,0.527558,9.69518,8.77775,0.166641,8.56429,8.69221,0.199346,15.5955,8.38788,0.272872,10.2809,8.48477,0.132925,10.5913,8.3798,0.0507148,8.156,8.56329,0.237119,7.92834,8.12394,0.536364,1.47742,8.88654,0.161173,4.48622,8.71523,0.230233]

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
            ot.Normal([0.0] * 3, [6.0, 0.5, 0.15]),
            [base_index + i for i in range(ndim)],
        )
    ]

# %%
# Run all samplers independently once to make sure they work.
#sampler_realizations = [s.getRealization() for s in samplers]

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

samples = sampler.getSample(20000)
acceptance = [
    sampler.getMetropolisHastingsCollection()[i].getAcceptanceRate()
    for i in range(6, len(samplers))
]

adaptation_factor = [
    sampler.getMetropolisHastingsCollection()[i].getImplementation().getAdaptationFactor()
    for i in range(6, len(samplers))
]

print("Minimum acceptance rate = ", np.min(acceptance))
print("Maximum acceptance rate for random walk MH = ", np.max(acceptance[2 * ndim :]))

# %%
# Represent the last sampled points (i.e. those which are least dependent on the initial state)
# We are only interested in the :math:`\mu` and :math:`\sigma` parameters.

#reduced_samples = samples[samples.getSize()*3//4::20, 0:6]
#reduced_samples = samples[0::100, 0:6]
reduced_samples = samples[:, 0:6]

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
        cloud = graph.getDrawable(0).getImplementation()
        cloud.setPointStyle(".")
        data = cloud.getData()
        dist = ot.KernelSmoothing().build(data)
        contour = dist.drawPDF().getDrawable(0).getImplementation()
        contour.setLevels(np.linspace(0.0, contour.getLevels()[-1], 10))
        graph.setDrawables([contour, cloud])
        graph.setBoundingBox(contour.getBoundingBox())
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
from copulogram import Copulogram
import pandas as pd

reduced_df = pd.DataFrame(np.array(reduced_samples))
reduced_df.columns = [name for name in reduced_samples.getDescription()]
reduced_df["INDEX"] = reduced_df.index
#reduced_df.columns = reduced_samples.getDescription()
copulogram = Copulogram(reduced_df)
copulogram.draw(alpha=0.25, marker=".", hue="INDEX")


# %%
ot.ResourceMap.Reload()
