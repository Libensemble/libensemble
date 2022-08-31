.. _datastruct-work-dict:

work dictionary
===============

The work dictionary contains metadata that is used by the manager to send a packet
of work to a worker. The dictionary uses integer keys ``i`` and values that determine
the data given to worker ``i``. ``Work[i]`` has the following form::

    Work[i]: [dict]:

        Required keys:
        'H_fields' [list]: The field names of the history 'H' to be sent to worker 'i'
        'persis_info' [dict]: Any persistent info to be sent to worker 'i'
        'tag' [int]: 'EVAL_SIM_TAG'/'EVAL_GEN_TAG' if worker 'i' is to call sim/gen_func
        'libE_info' [dict]: Info sent to/from worker to help manager update the 'H' array

        libE_info contains the following:
        'H_rows' [list of ints]: History rows to send to worker 'i'
        'rset_team' [list of ints]: The resource sets to be assigned (if dynamic scheduling is used)
        'persistent' [bool]: True if worker 'i' will enter persistent mode (Default: False)

The work dictionary is typically set using the ``gen_work`` or ``sim_work``
:doc:`helper functions<../function_guides/allocator>` in the allocation function.
``H_fields``, for example, is usually packed from either ``sim_specs["in"]``, ``gen_specs["in"]``
or the equivalent "persis_in" variants.

.. seealso::
  For allocation functions giving work dictionaries using persistent workers,
  see `start_only_persistent.py`_ or `start_persistent_local_opt_gens.py`_.
  For a use case where the allocation and generator functions combine to do
  simulation evaluations with different resources, see
  `test_uniform_sampling_with_variable_resources.py`_.

.. _start_only_persistent.py: https://github.com/Libensemble/libensemble/blob/develop/libensemble/alloc_funcs/start_only_persistent.py
.. _start_persistent_local_opt_gens.py: https://github.com/Libensemble/libensemble/blob/develop/libensemble/alloc_funcs/start_persistent_local_opt_gens.py
.. _test_uniform_sampling_with_variable_resources.py: https://github.com/Libensemble/libensemble/blob/develop/libensemble/tests/regression_tests/test_uniform_sampling_with_variable_resources.py
