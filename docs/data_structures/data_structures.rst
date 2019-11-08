Data Structures
===============

This section outlines the data structures used by libEnsemble. We first discuss
the dictionaries that are input to libEnsemble to declare the
:ref:`simulation<datastruct-sim-specs>`,
:ref:`generation<datastruct-gen-specs>`,
and
:ref:`allocation<datastruct-alloc-specs>`
specifications, as well as specify
:ref:`exit criteria<datastruct-exit-criteria>`,
:ref:`persistent information<datastruct-persis-info>`, and other
:ref:`libEnsemble<datastruct-libe-specs>`
options.

We then discuss internal libEnsemble, including the
:ref:`history array<datastruct-history-array>`,
:ref:`worker array<datastruct-history-array>`,
and the
:ref:`work<datastruct-history-array>` dictionary produced by the allocation
function.

.. note::
  Users can check the formatting and consistency of ``exit_criteria`` and each ``specs``
  dictionary with the ``check_inputs()`` function from the ``utils`` module.
  Provide any combination of these data structures as keyword arguments.
  For example::

    from libensemble.libE import check_inputs

    check_inputs(sim_specs=my-sim_specs, gen_specs=my-gen_specs, exit_criteria=ec)

.. toctree::
   :maxdepth: 3
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
