.. _datastruct-alloc-specs:

alloc_specs
===========

Allocation function specifications to be set in user calling script and passed
to main ``libE()`` routine::

    alloc_specs: [dict, optional] :

        'alloc_f' [func] :
            Default: give_sim_work_first
        'in' [list of strings] :
            Default: None
        'out' [list of tuples] :
            Default: [('allocated',bool)]
        'user' [dict]
            Default: {'batch_mode': True}

.. note::

  * The alloc_specs has the default keys as given above, but may be overidden by the user.
  * The tuples defined in the 'out' list are entered into the master :ref:`history array<datastruct-history-array>`

.. seealso::
  - `test_chwirut_uniform_sampling_one_residual_at_a_time.py`_ specifies fields
    to be used by the allocation function ``give_sim_work_first`` from
    fast_alloc_and_pausing.py_.

  ..  literalinclude:: ../../libensemble/tests/regression_tests/test_chwirut_uniform_sampling_one_residual_at_a_time.py
      :start-at: alloc_specs
      :end-before: end_alloc_specs_rst_tag

.. _test_chwirut_uniform_sampling_one_residual_at_a_time.py: https://github.com/Libensemble/libensemble/blob/develop/libensemble/tests/regression_tests/test_chwirut_uniform_sampling_one_residual_at_a_time.py
.. _fast_alloc_and_pausing.py: https://github.com/Libensemble/libensemble/blob/develop/libensemble/alloc_funcs/fast_alloc_and_pausing.py
