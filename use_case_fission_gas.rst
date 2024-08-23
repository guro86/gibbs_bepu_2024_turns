.. _use-case-fission-gas:

Fission gas release
===================

This use case presents several simplified models of fission gas release,
as a fraction of what is created by the fission reaction.
They are derived from the work by [robertson2024]_.

Each of the models has two inputs:

- Single atom diffusion denoted by :math:`x_{\mathrm{diff}}`.
- Amount released due to micro-cracking, denoted by :math:`x_{\mathrm{crack}}`.

Each model corresponds to different sets of experimental conditions.
The number of different sets of environmental conditions is denoted by :math:`\sampleSize_{\mathrm{exp}}`,
and in the following we use the index :math:`i` (:math:`1 \leq i \leq \sampleSize_{\mathrm{exp}}`)
to identify a given set of environmental conditions.

For each set of environmental conditions :math:`i`, fission gas release
was measured, and the measured value is denoted by :math:`y_i`.
Measurement uncertainty is represented by a normal distribution
with mean :math:`y_i`, and its standard deviation :math:`\sigma_{y_i}` is a known function of :math:`y_i`:

.. math::
   \sigma_{y_i} = \sqrt{\left( \frac{y_i}{20} \right)^2 + 10^{-4}}

The model corresponding to the environmental conditions :math:`i` 
is denoted by :math:`\model_i`.
The values of single atom diffusion :math:`x_{\mathrm{diff}, i}`
and of the amount of gas released due to micro-cracking :math:`x_{\mathrm{crack}, i}`
corresponding to the measured :math:`y_i` are unobserved.

The relationships between these quantities are represented in the following network.
Full arrows represent deterministic relationships,
while dashed arrows represent probabilistic relationships (e.g. a normal distribution).

.. figure:: ../_static/fission_gas_network.png
    :align: center
    :alt: Relationships between the variables
    :width: 50%

References
----------

- [robertson2024]_


API documentation
-----------------

.. currentmodule:: openturns.usecases.fission_gas

.. autoclass:: FissionGasRelease
    :noindex:

Examples based on this use case
-------------------------------

.. minigallery:: openturns.usecases.fission_gas.FissionGasRelease
