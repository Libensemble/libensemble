Data Structures
===============

Users can check the formatting and consistency of ``exit_criteria`` and each ``specs``
dictionary with the ``check_inputs()`` function within the ``libE module``.
Provide any combination of these data structures as keyword arguments.
For example::

    from libensemble.libE import check_inputs

    check_inputs(sim_specs=my-sim_specs, gen_specs=my-gen_specs, exit_criteria=ec)

.. toctree::
   :maxdepth: 1
   :caption: libEnsemble Data Structures:

   history_array
   worker_array
   work_dict
   libE_specs
   sim_specs
   gen_specs
   exit_criteria
   alloc_specs
   persis_info
