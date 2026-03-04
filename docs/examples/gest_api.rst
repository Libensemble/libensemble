=============================
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

Included with libEnsemble
=========================

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

Verified Third Party Examples
=============================

Generators that implement the gest_api_ interface and are verified to work with libEnsemble.

The standardized interface was developed in partnership with their authors.

Xopt - https://github.com/xopt-org/Xopt
---------------------------------------

Examples:

`Expected Improvement`_

`Nelder Mead`_

Optimas - https://github.com/optimas-org/optimas
------------------------------------------------

Examples:

`Grid Sampling`_

`Ax Multi-fideltiy`_

.. _gest_api: https://github.com/campa-consortium/gest-api
.. _gpcam: https://gpcam.lbl.gov/
.. _paper: https://link.springer.com/article/10.1007/s12532-017-0131-4

.. _Expected Improvement: https://github.com/Libensemble/libensemble/blob/develop/libensemble/tests/regression_tests/test_xopt_EI.py

.. _Nelder Mead: https://github.com/Libensemble/libensemble/blob/develop/libensemble/tests/regression_tests/test_xopt_nelder_mead.py

.. _Grid Sampling: https://github.com/Libensemble/libensemble/blob/develop/libensemble/tests/regression_tests/test_optimas_grid_sample.py

.. _Ax Multi-fideltiy: https://github.com/Libensemble/libensemble/blob/develop/libensemble/tests/regression_tests/test_optimas_ax_mf.py
