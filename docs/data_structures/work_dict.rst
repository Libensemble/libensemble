.. _datastruct-work-dict:

work dictionary
===============

The work dictionary contains integer keys ``i`` and dictionary values to be
given to worker ``i``. ``Work[i]`` has the following form::

    Work[i]: [dict]:

        Required keys:
        'persis_info' [dict]: Any persistent info to be sent to worker 'i'
        'H_fields' [list]: The field names of the history 'H' to be sent to worker 'i'
        'tag' [int]: 'EVAL_SIM_TAG'/'EVAL_GEN_TAG' if worker 'i' is to call sim/gen_func
        'libE_info' [dict]: Info sent to/from worker to help manager update the 'H'

        Optional keys are:
        'H_rows' [list of ints]: History rows to send to worker 'i'
        'blocking' [list of ints]: Workers to be blocked by this calculation
        'persistent' [bool]: True if worker 'i' will enter persistent mode

.. seealso::
  For allocation functions giving work dictionaries using persistent workers,
  see `start_only_persistent.py`_ or `start_persistent_local_opt_gens.py`_.
  For a use case where the allocation and generator functions combine to do
  simulation evaluations with different resources (blocking some workers), see
  `test_uniform_sampling_with_variable_resources.py`_.

.. _start_only_persistent.py: https://github.com/Libensemble/libensemble/blob/develop/libensemble/alloc_funcs/start_only_persistent.py
.. _start_persistent_local_opt_gens.py: https://github.com/Libensemble/libensemble/blob/develop/libensemble/alloc_funcs/start_persistent_local_opt_gens.py
.. _test_uniform_sampling_with_variable_resources.py: https://github.com/Libensemble/libensemble/blob/develop/libensemble/tests/regression_tests/test_uniform_sampling_with_variable_resources.py
