Utilities
=========

libEnsemble features several modules and tools to assist in writing consistent
calling scripts and user functions.

libE input consistency
----------------------

Users can check the formatting and consistency of ``exit_criteria`` and each
``specs`` dictionary with the ``check_inputs()`` function from the ``utils``
module. Provide any combination of these data structures as keyword arguments.
For example::

  from libensemble.libE import check_inputs

  check_inputs(sim_specs=my-sim_specs, gen_specs=my-gen_specs, exit_criteria=ec)
