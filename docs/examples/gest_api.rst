(New) Standardized Generators
=============================

libEnsemble now also supports all generators that implement the gest_api_ interface.

.. code-block:: python
  :linenos:
  :emphasize-lines: 17

  from gest_api.vocs import VOCS
  from optimas.generators import GridSamplingGenerator

  from libensemble.specs import GenSpecs

  vocs = VOCS(
      variables={
          "x0": [-3.0, 2.0],
          "x1": [1.0, 5.0],
      },
      objectives={"f": "MAXIMIZE"},
  )

  generator = GridSamplingGenerator(vocs=vocs, n_steps=[7, 15])

  gen_specs = GenSpecs(
      generator=generator,
      batch_size=4,
      vocs=vocs,
  )
  ...

Here we list many standard-adhering generators included with libEnsemble.

Sampling
--------

.. toctree::
  :maxdepth: 1
  :caption: Sampling
  :hidden:

  gest_api/sampling

- :doc:`Basic sampling<gest_api/sampling>`

  Various generators for sampling a space.

Optimization
------------

.. toctree::
  :maxdepth: 1
  :caption: Optimization
  :hidden:

  gest_api/aposmm

- :doc:`APOSMM<gest_api/aposmm>`

  Asynchronously Parallel Optimization Solver for finding Multiple Minima (paper_).

Modeling and Approximation
--------------------------

.. toctree::
  :maxdepth: 1
  :caption: Modeling and Approximation
  :hidden:

  gest_api/gpcam

- :doc:`gpCAM<gest_api/gpcam>`

  Gaussian Process-based adaptive sampling using gpcam_.

.. _gest_api: https://github.com/campa-consortium/gest-api
.. _gpcam: https://gpcam.lbl.gov/
.. _paper: https://link.springer.com/article/10.1007/s12532-017-0131-4
