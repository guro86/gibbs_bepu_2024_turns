import data
import matplotlib.pyplot as plt
import numpy as np
import openturns as ot
import multiprocessing as mp
import os

# %%

# Training test split random state
train_test_split_kwargs = {"random_state": 19885}

# A data object
fgr_data = data.fgr.dakota_data(train_test_split_kwargs=train_test_split_kwargs)

# Preprocessing of the data (read and split etc.)
fgr_data.process()

# Training values
Xtrain = ot.Sample.BuildFromDataFrame(
    fgr_data.Xtrain[["diff", "crack"]]
)
ytrain = ot.Sample.BuildFromDataFrame(fgr_data.ytrain)
Xtrain.exportToCSVFile("fission_gas_Xtrain.csv")
ytrain.exportToCSVFile("fission_gas_ytrain.csv")


# Testing values
Xtest = ot.Sample.BuildFromDataFrame(fgr_data.Xtest[["diff", "crack"]])
ytest = ot.Sample.BuildFromDataFrame(fgr_data.ytest)


# Dimensions
dimension = Xtrain.getDimension()

# Constant basis
basis = ot.ConstantBasisFactory(dimension).build()

# Squared exponential kernel with dimension lenbght scales
# Unity amplitude
covarianceModel = ot.SquaredExponential([1.0] * dimension, [1.0])
# There are missing input dimensions, we account for them with a nugget effect
covarianceModel.activateNuggetFactor(True)

# Function returning the GP for exp = i
def func(exp):
    # We do the algorithm
    algo = ot.KrigingAlgorithm(Xtrain, ytrain.getMarginal(exp), covarianceModel, basis)

    # We run it
    algo.run()

    # Get results
    result = algo.getResult()

    print(f"returning metamodel {exp}")

    return result


with mp.Pool() as pool:
    metamodel_results = pool.map(func, range(31))

metamodels = [res.getMetaModel() for res in metamodel_results]
# %%
q2 = ot.Sample(len(metamodel_results), 1)
# Loop over all experiments
for exp in range(31):
    # plott test vs gp-predictions
    plt.plot(ytest.getMarginal(exp), metamodels[exp](Xtest), "o")
    val = ot.MetaModelValidation(ytest.getMarginal(exp), metamodels[exp](Xtest))
    q2[exp] = val.computeR2Score()

# Plot equalittyline
l = np.linspace(0, 0.4)
plt.plot(l, l, "--")

# and some labels
plt.xlabel("Test predictions [-]")
plt.ylabel("GP predictions [-]")


# %%
template_kernel = ot.SquaredExponential(dimension)
parameters = ot.Sample(
    ytrain.getDimension(), template_kernel.getFullParameter().getDimension()
)
parameters.setDescription(template_kernel.getFullParameterDescription())
for index, metamodel_result in enumerate(metamodel_results):
    parameters[index] = metamodel_result.getCovarianceModel().getFullParameter()
print(parameters)
parameters.exportToCSVFile("fission_gas_GPR_hyperparameters.csv")
