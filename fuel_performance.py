# %%
import os
import openturns as ot


class FuelPerformance:
    def __init__(self) -> None:
        self.Xtrain = ot.Sample.ImportFromTextFile(
            os.path.join("fuel_performance_data", "fuel_performance_Xtrain.csv"), ";"
        )
        self.ytrain = ot.Sample.ImportFromTextFile(
            os.path.join("fuel_performance_data", "fuel_performance_ytrain.csv"), ";"
        )
        self._hyperparameters = ot.Sample.ImportFromTextFile(
            os.path.join(
                "fuel_performance_data", "fuel_performance_GPR_hyperparameters.csv"
            ),
            ";",
        )

        self.models = []
        covariance = ot.SquaredExponential(self.Xtrain.getDimension())
        basis = ot.ConstantBasisFactory(self.Xtrain.getDimension()).build()
        for num, hyper in enumerate(self._hyperparameters):
            covariance.setFullParameter(hyper)
            gpr = ot.KrigingAlgorithm(
                self.Xtrain, self.ytrain.getMarginal(num), covariance, basis
            )
            gpr.setOptimizeParameters(False)
            gpr.run()
            self.models.append(gpr.getResult().getMetaModel())

        # Measurement array
        self.meas_v = [
            0.228,
            0.015,
            0.265,
            0.253,
            0.016,
            0.173,
            0.13,
            0.067,
            0.048,
            0.018,
            0.104,
            0.066,
            0.16,
            0.022,
            0.095,
            0.035,
            0.321,
            0.035,
            0.009,
            0.037,
            0.05,
            0.28,
            0.136,
            0.221,
            0.296,
            0.13,
            0.085,
            0.058,
            0.449,
            0.029,
            0.018,
        ]

        meas_unc = lambda v: ((v * 0.05) ** 2 + 0.01 ** 2) ** 0.5

        # Defining some likelihoods
        self.likes = [ot.Normal(v, meas_unc(v)) for v in self.meas_v]
