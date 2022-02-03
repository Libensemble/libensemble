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

    Since version 0.7.0, libEnsemble performs an initial check that all ``'in'``
    fields in ``sim_specs``, ``gen_specs``, and ``alloc_specs`` correspond to
    a field in the initial history array, ``H0`` or
    at least one ``'out'`` field in the aforementioned data structures. This
    guarantees that the required inputs are available.

.. toctree::
   :maxdepth: 3
   :caption: libEnsemble Data Structures:

   history_array
   sim_specs
   gen_specs
   alloc_specs
   libE_specs
   persis_info
   exit_criteria

   worker_array
   work_dict
   calc_status
