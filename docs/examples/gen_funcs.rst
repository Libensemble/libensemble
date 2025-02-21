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

- :doc:`sampling<sampling>`

  Various generators for sampling a space. Function runs once each call.

- :doc:`persistent sampling<persistent_sampling>`

  Various persistent generators (persists on a worker) for sampling a space. After the initial
  batch each generator creates N new random points for every N points that are returned.

- :doc:`persistent sampling with variable resources<persistent_sampling_var_resources>`

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
  VTMOP<https://libensemble.readthedocs.io/projects/libe-community-examples/en/latest/#module-vtmop>
  ytopt<https://libensemble.readthedocs.io/projects/libe-community-examples/en/latest/#module-ytopt_heffte.ytopt_asktell>
  consensus<https://libensemble.readthedocs.io/projects/libe-community-examples/en/latest/#module-consensus>
..   Ax<https://libensemble.readthedocs.io/projects/libe-community-examples/en/latest/#module-persistent_ax_multitask>
..   Dragonfly<https://libensemble.readthedocs.io/projects/libe-community-examples/en/latest/#module-persistent_gp>

- :doc:`APOSMM<aposmm>`

  Asynchronously Parallel Optimization Solver for finding Multiple Minima (APOSMM_) coordinates
  concurrent local optimization runs to identify many local minima faster on parallel hardware.

- :doc:`uniform_or_localopt<uniform_or_localopt>`

  Samples uniformly in non-persistent mode or runs an NLopt_ local optimization run in persistent mode.

- :doc:`Ax Multitask<ax_multitask>`

  Bayesian optimization with a Gaussian process and the multi-task algorithm of Ax_.

- :ref:`VTMOP<community:vtmop-link>`

  Generators using the VTMOP_ Fortran package for large-scale multiobjective multidisciplinary design optimization. (*community example*)

- :ref:`ytopt<community:ytopt-link>`

  Generators using ytopt_. A Bayesian Optimization package for determining optimal input parameter configurations for applications or other executables. (*community example*)

- :ref:`Consensus<community:consensus-link>`

  Distributed optimization methods for minimizing sums of convex functions (*community example*).


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
.. _Surmise: https://surmise.readthedocs.io/en/latest/index.html
.. _Tasmanian: https://github.com/ORNL/Tasmanian
.. _user guide: https://libensemble.readthedocs.io/en/latest/programming_libE.html
.. _VTMOP: https://github.com/Libensemble/libe-community-examples#vtmop
.. _WarpX: https://warpx.readthedocs.io/en/latest/
.. _ytopt: https://github.com/ytopt-team/ytopt


