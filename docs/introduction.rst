.. include:: ../README.rst
    :start-after: after_badges_rst_tag

See the :doc:`tutorial<tutorials/local_sine_tutorial>` for a step-by-step beginners guide.

See the `user guide`_ for more information.

.. example_packages

.. dropdown:: **Example Compatible Packages**
  :open:

  libEnsemble and the `Community Examples repository`_ include example generator
  functions for the following libraries:

  - APOSMM_ Asynchronously parallel optimization solver for finding multiple minima. Supported local optimization routines include:

    - DFO-LS_ Derivative-free solver for (bound constrained) nonlinear least-squares minimization
    - NLopt_ Library for nonlinear optimization, providing a common interface for various methods
    - `scipy.optimize`_ Open-source solvers for nonlinear problems, linear programming,
      constrained and nonlinear least-squares, root finding, and curve fitting.
    - `PETSc/TAO`_ Routines for the scalable (parallel) solution of scientific applications

  - DEAP_ Distributed evolutionary algorithms
  - Distributed optimization methods for minimizing sums of convex functions. Methods include:

    - Primal-dual sliding (https://arxiv.org/pdf/2101.00143).
    - Distributed gradient descent with gradient tracking (https://arxiv.org/abs/1908.11444).
    - Proximal sliding (https://arxiv.org/abs/1406.0919).

  - ECNoise_ Estimating Computational Noise in Numerical Simulations
  - Surmise_ Modular Bayesian calibration/inference framework
  - Tasmanian_ Toolkit for Adaptive Stochastic Modeling and Non-Intrusive ApproximatioN
  - VTMOP_ Fortran package for large-scale multiobjective multidisciplinary design optimization

  libEnsemble has also been used to coordinate many computationally expensive
  simulations. Select examples include:

  - OPAL_ Object Oriented Parallel Accelerator Library. (See this `IPAC manuscript`_.)
  - WarpX_ Advanced electromagnetic particle-in-cell code.

.. _APOSMM: https://link.springer.com/article/10.1007/s12532-017-0131-4
.. _Community Examples repository: https://github.com/Libensemble/libe-community-examples
.. _DEAP: https://deap.readthedocs.io/en/master/overview.html
.. _DFO-LS: https://github.com/numericalalgorithmsgroup/dfols
.. _ECNoise: https://www.mcs.anl.gov/~wild/cnoise/
.. _IPAC manuscript: https://doi.org/10.18429/JACoW-ICAP2018-SAPAF03
.. _NLopt: https://nlopt.readthedocs.io/en/latest/
.. _OPAL: http://amas.web.psi.ch/docs/opal/opal_user_guide-1.6.0.pdf
.. _PETSc/TAO: http://www.mcs.anl.gov/petsc
.. _scipy.optimize: https://docs.scipy.org/doc/scipy/reference/optimize.html
.. _Surmise: https://surmise.readthedocs.io/en/latest/index.html
.. _Tasmanian: https://github.com/ORNL/Tasmanian
.. _user guide: https://libensemble.readthedocs.io/en/latest/programming_libE.html
.. _VTMOP: https://github.com/Libensemble/libe-community-examples#vtmop
.. _WarpX: https://warpx.readthedocs.io/en/latest/
