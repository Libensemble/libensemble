Data Structures
===============

This section outlines the data structures used by libEnsemble.

.. note::
  Users can check the formatting and consistency of ``exit_criteria`` and each ``specs``
  dictionary with the ``check_inputs()`` function from the ``utils`` module.
  Provide any combination of these data structures as keyword arguments.
  For example::

    from libensemble.libE import check_inputs

    check_inputs(sim_specs=my-sim_specs, gen_specs=my-gen_specs, exit_criteria=ec)

.. toctree::
   :maxdepth: 2
   :caption: libEnsemble Data Structures:

   sim_specs
   gen_specs
   alloc_specs
   libE_specs
   persis_info
   exit_criteria


   history_array
   worker_array
   work_dict
