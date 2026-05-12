.. _funcguides-sim:

Simulator Functions
===================

**Introduction** \|\| `Standardized Simulator (gest-api) <simulator_standardized.html>`__ \|\| `Legacy Simulator Function <simulator_legacy.html>`__

Simulator and :ref:`Generator functions<funcguides-gen>` have relatively similar interfaces.

Writing a Simulator
-------------------

.. note::
    The `gest-api` simulator interface is the recommended approach for new libEnsemble projects.
    The "Legacy Simulator Function" interface is supported for backward compatibility but may be deprecated in a future release.

Tutorial sections
-----------------

1. Introduction (this page)
2. :doc:`Standardized Simulator (gest-api) <simulator_standardized>`
3. :doc:`Legacy Simulator Function <simulator_legacy>`

Executor
--------

libEnsemble's Executors are commonly used within simulator functions to launch
and monitor applications. An excellent overview is already available
:doc:`here<../executor/ex_index>`.

See the :doc:`Ensemble with an MPI Application tutorial<../tutorials/executor_forces_tutorial>`
for an additional example to try out.

Persistent Simulators
---------------------

Simulator functions can also be written
in a persistent fashion. See the :ref:`here<persistent-gens>` for a general API overview
of writing persistent generators, since the interface is largely identical. The only
differences are to pass ``EVAL_SIM_TAG`` when instantiating a ``PersistentSupport``
class instance and to return ``FINISHED_PERSISTENT_SIM_TAG`` when the simulator
function returns.

.. note::
  An example routine using a persistent simulator can be found in test_persistent_sim_uniform_sampling_.

.. _test_persistent_sim_uniform_sampling: https://github.com/Libensemble/libensemble/blob/develop/libensemble/tests/functionality_tests/test_persistent_sim_uniform_sampling.py

.. toctree::
   :hidden:

   simulator_standardized
   simulator_legacy
