# %%
import os
import openturns as ot


class FissionGasRelease:
    r"""
    Data class for the fission gas release example.
    
    Attributes
    ----------

    measurement_values : :class:`~openturns.Point`
        Observed values :math:`y_i` (:math:`1 \leq i \leq \sampleSize_{\mathrm{exp}}`)

    measurement_uncertainty : :class:`~openturns.Function`
        Function :math:`y_i \mapsto \sigma_{y_i}`

    models : list of :class:`~openturns.Function`
        List of the :math:`\model_i` models (:math:`1 \leq i \leq \sampleSize_{\mathrm{exp}}`)

    Examples
    --------
    >>> from openturns.usecases import fission_gas
    >>> # Load the fission gas release models
    >>> fgr = fission_gas.FissionGasRelease()
    """
    def __init__(self) -> None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self._Xtrain = ot.Sample.ImportFromTextFile(
            os.path.join(current_dir, "fission_gas_Xtrain.csv"), ";"
        )
        self._ytrain = ot.Sample.ImportFromTextFile(
            os.path.join(current_dir, "fission_gas_ytrain.csv"), ";"
        )
        self._hyperparameters = ot.Sample.ImportFromTextFile(
            os.path.join(current_dir, "fission_gas_GPR_hyperparameters.csv"), ";"
        )

        self.models = []
        covariance = ot.SquaredExponential(self._Xtrain.getDimension())
        basis = ot.ConstantBasisFactory(self._Xtrain.getDimension()).build()
        for num, hyper in enumerate(self._hyperparameters):
            covariance.setFullParameter(hyper)
            gpr = ot.KrigingAlgorithm(
                self._Xtrain, self._ytrain.getMarginal(num), covariance, basis
            )
            gpr.setOptimizeParameters(False)
            gpr.run()
            self.models.append(gpr.getResult().getMetaModel())

        # Measurement array
        self.measurement_values = ot.Point([
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
        ])

        self.measurement_uncertainty = lambda v: ((v * 0.05) ** 2 + 0.01 ** 2) ** 0.5
        
    def get_input_description(self):
        """Get the description of the model inputs.

        Returns
        -------
        :class:`~openturns.Description`
            Model input names
        """
        return self._Xtrain.getDescription()
