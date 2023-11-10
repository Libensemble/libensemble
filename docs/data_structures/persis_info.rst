.. _datastruct-persis-info:

persis_info
===========

Holds persistent information that can be updated during the ensemble.

An initialized ``persis_info`` dictionary can be provided to the ``libE()`` call
or as an attribute of the ``Ensemble`` class.

Dictionary keys that have an integer value contain entries that are passed to
and from the corresponding workers. These are received in the ``persis_info``
argument of user functions, and returned as the optional second return value.

A typical example is a random number generator stream to be used in consecutive
calls to a generator (see
:meth:`add_unique_random_streams()<tools.add_unique_random_streams>`)

All other entries persist on the manager and can be updated in the calling script
between ensemble invocations, or in the allocation function.

Examples:

.. tab-set::

  .. tab-item:: RNG or reusable structures

    .. literalinclude:: ../../libensemble/gen_funcs/sampling.py
      :linenos:
      :start-at: def uniform_random_sample(_, persis_info, gen_specs):
      :end-before: def uniform_random_sample_with_variable_resources(_, persis_info, gen_specs):
      :emphasize-lines: 17
      :caption: libensemble/libensemble/gen_funcs/sampling.py

  .. tab-item:: Incrementing indexes or process counts

    .. literalinclude:: ../../libensemble/alloc_funcs/fast_alloc.py
       :linenos:
       :start-at:     for wid in support.avail_worker_ids():
       :end-before:         # Give sim work if possible
       :emphasize-lines: 3-4
       :caption: libensemble/alloc_funcs/fast_alloc.py

  .. tab-item:: Tracking running generators

    .. literalinclude:: ../../libensemble/alloc_funcs/start_only_persistent.py
       :linenos:
       :start-at:        avail_workers = support.avail_worker_ids(persistent=False, zero_resource_workers=True)
       :end-before:    return Work, persis_info, 0
       :emphasize-lines: 18
       :caption: libensemble/alloc_funcs/start_only_persistent.py

  .. tab-item:: Allocation function triggers shutdown

    .. literalinclude:: ../../libensemble/alloc_funcs/start_only_persistent.py
       :linenos:
       :start-at:    if gen_count < persis_info.get("num_gens_started", 0):
       :end-before:    # Give evaluated results back to a running persistent gen
       :emphasize-lines: 1
       :caption: libensemble/alloc_funcs/start_only_persistent.py

.. - Random number generators or other structures for use on consecutive calls
.. - Incrementing array row indexes or process counts
.. - Sending/receiving updated models from workers
.. - Keeping track of the number of generators started in an allocation function
.. - Triggering the shutdown of the ensemble (from the allocation function).

When there are repeated calls to ``libE()`` or ``ensemble.run()``, users may
need to modify or reset the contents of ``persis_info`` in some cases.

.. seealso::

  From: support.py_

  ..  literalinclude:: ../../libensemble/tests/regression_tests/support.py
      :start-at: persis_info_1
      :end-before: end_persis_info_rst_tag

.. _support.py: https://github.com/Libensemble/libensemble/blob/develop/libensemble/tests/regression_tests/support.py
