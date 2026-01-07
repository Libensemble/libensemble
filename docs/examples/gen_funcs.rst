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
  persistent_sampling
  persistent_sampling_var_resources

- :doc:`Basic sampling<sampling>`

  Various generators for sampling a space. The non-persistent function is called as needed.

- :doc:`Persistent sampling<persistent_sampling>`

  Various persistent generators (persists on a worker) for sampling a space. After the initial
  batch each generator creates ``p`` new random points for every ``p`` points that are returned.

- :doc:`Persistent sampling with variable resources<persistent_sampling_var_resources>`

  Various persistent sampling generators that assign different resources to each simulation.

Optimization
------------

.. toctree::
  :maxdepth: 1
  :caption: Optimization
  :hidden:

  aposmm
  uniform_or_localopt
  ax_multitask<ax_multitask>
  VTMOP<https://libensemble.readthedocs.io/projects/libe-community-examples/en/latest/generators.html#module-vtmop>
  ytopt<https://libensemble.readthedocs.io/projects/libe-community-examples/en/latest/generators.html#module-ytopt_heffte.ytopt_asktell>
  consensus<https://libensemble.readthedocs.io/projects/libe-community-examples/en/latest/generators.html#gens.persistent_independent_optimize>
  DEAP-NSGA-II<https://libensemble.readthedocs.io/projects/libe-community-examples/en/latest/generators.html#persistent-deap-nsga2>

- :doc:`APOSMM<aposmm>`

  Asynchronously Parallel Optimization Solver for finding Multiple Minima (APOSMM_).

- :doc:`Ax Multitask<ax_multitask>`

  Bayesian optimization with a Gaussian process driven by an Ax_ multi-task algorithm.

- :ref:`DEAP-NSGA-II<community:deap-link>`

  Distributed evolutionary algorithms (*community example*)

- :ref:`Distributed optimization<community:consensus-link>`

  Distributed optimization methods for minimizing sums of convex functions. (*community example*)

- :doc:`uniform_or_localopt<uniform_or_localopt>`

  Samples uniformly in non-persistent mode then runs an NLopt_ local optimization runs in persistent mode.

- :ref:`VTMOP<community:vtmop-link>`

  Multiobjective multidisciplinary design optimization using the VTMOP_ Fortran package. (*community example*)

- :ref:`ytopt<community:ytopt-link>`

  Bayesian Optimization package for determining optimal input parameter configurations for applications/executables using ytopt_. (*community example*)

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

- :doc:`Finite-difference parameter finder<fd_param_finder>`

  Uses ECNoise_ to determine a suitable finite difference parameters for a mapping ``F`` from ``R^n`` to ``R^m``.

- :doc:`gpCAM<gpcam>`

  Gaussian Process-based adaptive sampling using gpcam_.

- :doc:`surmise<surmise>`

  Modular Bayesian calibration/inference framework using Surmise_ (demonstration of cancelling previous issued simulations).

- :doc:`Tasmanian<tasmanian>`

  Evaluates points generators by the Tasmanian_ sparse grid library

.. _libEnsemble Community Repository: https://github.com/Libensemble/libe-community-examples

.. _APOSMM: https://link.springer.com/article/10.1007/s12532-017-0131-4
.. _Ax: https://github.com/facebook/Ax
.. _Community Examples repository: https://github.com/Libensemble/libe-community-examples
.. _DEAP: https://deap.readthedocs.io/en/master/overview.html
.. _DFO-LS: https://github.com/numericalalgorithmsgroup/dfols
.. _ECNoise: https://www.mcs.anl.gov/~wild/cnoise/
.. _gpcam: https://gpcam.lbl.gov/
.. _IPAC manuscript: https://doi.org/10.18429/JACoW-ICAP2018-SAPAF03
.. _NLopt: https://nlopt.readthedocs.io/en/latest/
.. _OPAL: http://amas.web.psi.ch/docs/opal/opal_user_guide-1.6.0.pdf
.. _Surmise: https://surmise.readthedocs.io/en/latest/index.html
.. _Tasmanian: https://github.com/ORNL/Tasmanian
.. _user guide: https://libensemble.readthedocs.io/en/latest/programming_libE.html
.. _VTMOP: https://github.com/Libensemble/libe-community-examples#vtmop
.. _WarpX: https://warpx.readthedocs.io/en/latest/
.. _ytopt: https://github.com/ytopt-team/ytopt
