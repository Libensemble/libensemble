.. _datastruct-work-dict:

work dictionary
===============

Dictionary with integer keys ``i`` and dictionary values to be given to worker ``i``.
``Work[i]`` has the following form::


    Work[i]: [dict]:

        Required keys :
        'persis_info' [dict]: Any persistent info to be sent to worker 'i'
        'H_fields' [list]: The field names of the history 'H' to be sent to worker 'i'
        'tag' [int]: 'EVAL_SIM_TAG'/'EVAL_GEN_TAG') if worker 'i' is to call sim/gen_func
        'libE_info' [dict]: Info sent to/from worker to help manager update the 'H'

        Available keys are:
        'H_rows' [list of ints]: History rows to send to worker 'i'
        'blocking' [list of ints]: Workers to be blocked by this calculation
        'persistent' [bool]: True if worker 'i' will enter persistent mode


:Examples:

.. How to link directly to the file?

| For allocation functions using persistent workers, see
| ``libensemble/tests/regression_tests/test_6-hump_camel_persistent_uniform_sampling.py``
| or
| ``libensemble/tests/regression_tests/test_6-hump_camel_uniform_sampling_with_persistent_localopt_gens.py``
|
| For allocation functions giving work that blocks other workers, see
| ``libensemble/tests/regression_tests/test_6-hump_camel_with_different_nodes_uniform_sample.py``
