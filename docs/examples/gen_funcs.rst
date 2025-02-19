Generator Functions
===================

Here we list many generator functions included with libEnsemble.

.. IMPORTANT::
  See the API for generator functions :ref:`here<api_gen_f>`.



Sampling
--------

.. toctree::
  :maxdepth: 1
  :caption: Sampling
  :hidden:

  sampling

- :doc:`sampling<sampling>`

  Various generators for sampling a space.

Optimization
------------

.. toctree::
  :maxdepth: 1
  :caption: Optimization
  :hidden:

  aposmm
  uniform_or_localopt
  ax_multitask<ax_multitask>
  VTMOP<https://libensemble.readthedocs.io/projects/libe-community-examples/en/latest/#module-vtmop>
  ytopt<https://libensemble.readthedocs.io/projects/libe-community-examples/en/latest/#module-ytopt_heffte.ytopt_asktell>
  consensus<https://libensemble.readthedocs.io/projects/libe-community-examples/en/latest/#module-consensus>
..   Ax<https://libensemble.readthedocs.io/projects/libe-community-examples/en/latest/#module-persistent_ax_multitask>
..   Dragonfly<https://libensemble.readthedocs.io/projects/libe-community-examples/en/latest/#module-persistent_gp>

- :doc:`APOSMM<aposmm>`

  APOSMM_ Asynchronously parallel optimization solver for finding multiple minima. Supported local optimization routines include:

    - DFO-LS_ Derivative-free solver for (bound constrained) nonlinear least-squares minimization
    - NLopt_ Library for nonlinear optimization, providing a common interface for various methods
    - `scipy.optimize`_ Open-source solvers for nonlinear problems, linear programming,
      constrained and nonlinear least-squares, root finding, and curve fitting.
    - `PETSc/TAO`_ Routines for the scalable (parallel) solution of scientific applications

- :doc:`uniform_or_localopt<uniform_or_localopt>`

  Performs a uniform random sample or a single persistent NLopt_ local optimization run.

- :doc:`Ax Multitask<ax_multitask>`

  Bayesian optimization with a Gaussian process and the multi-task algorithm of Ax_.

- :ref:`VTMOP<community:vtmop-link>`

  Generators using the VTMOP_ Fortran package for large-scale multiobjective multidisciplinary design optimization. (*community example*)

- :ref:`ytopt<community:ytopt-link>`

  Generators using ytopt_. A Bayesian Optimization package for determining optimal input parameter configurations for applications or other executables. (*community example*)

.. - :ref:`Dragonfly<community:dragonfly-link>`
..
..   Generators for.

- :ref:`Consensus<community:consensus-link>`

  Distributed optimization methods for minimizing sums of convex functions (*community example*).
  Methods include:
    - Primal-dual sliding (https://arxiv.org/pdf/2101.00143).
    - Distributed gradient descent with gradient tracking (https://arxiv.org/abs/1908.11444).
    - Proximal sliding (https://arxiv.org/abs/1406.0919).

Modeling and Approximation
--------------------------

.. toctree::
  :maxdepth: 1
  :caption: Modeling and Approximation
  :hidden:

  gpcam
  tasmanian
  fd_param_finder
  surmise
  DEAP-NSGA-II<https://libensemble.readthedocs.io/projects/libe-community-examples/en/latest/#persistent-deap-nsga2>

- :doc:`gpCAM<gpcam>`

  Generators for Gaussian Process-based adaptive sampling using gpcam_.

- :doc:`Tasmanian<tasmanian>`

  Generators using the Tasmanian_ sparse grid library
  (*Toolkit for Adaptive Stochastic Modeling and Non-Intrusive ApproximatioN*).

- :doc:`fd_param_finder<fd_param_finder>`

  Generator that loops through a set of suitable finite difference
  parameters for a mapping ``F`` from ``R^n`` to ``R^m``.

- :doc:`surmise<surmise>`

  Modular Bayesian calibration/inference framework using Surmise_.
  Has the option of cancelling previous issued simulations.

- :ref:`DEAP-NSGA-II<community:deap-link>`

  Distributed evolutionary algorithms (*community example*)



.. _libEnsemble Community Repository: https://github.com/Libensemble/libe-community-examples

.. _APOSMM: https://link.springer.com/article/10.1007/s12532-017-0131-4
.. _Ax: https://github.com/facebook/Ax
.. _Community Examples repository: https://github.com/Libensemble/libe-community-examples
.. _DEAP: https://deap.readthedocs.io/en/master/overview.html
.. _DFO-LS: https://github.com/numericalalgorithmsgroup/dfols
.. _ECNoise: https://www.mcs.anl.gov/~wild/cnoise/
.. _gpcam: https://gpcam.lbl.gov/
.. .. _heFFTe: https://github.com/icl-utk-edu/heffte
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
.. _ytopt: https://github.com/ytopt-team/ytopt


