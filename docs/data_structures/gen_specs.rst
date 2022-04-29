.. _datastruct-gen-specs:

gen_specs
=========

Generation function specifications to be set in user calling script and passed
to main ``libE()`` routine::

    gen_specs: [dict]:

        'gen_f' [func]:
            generates inputs to sim_f
        'in' [list]:
            field names (as strings) that will be given to gen_f function when
            it is called (if not present, then no fields will be passed).
        'persis_in' [list]:
            field names (as strings) that will be given back to a persistent gen_f
        'out' [list of tuples (field name, data type, [size])]:
            gen_f outputs that will be stored in the libEnsemble history
        'user' [dict]:
            Data structure to contain problem specific constants and/or input data

.. note::

  * The user may define fields only in ``'user'`` to be passed to the generator function.
  * The tuples defined in the 'out' list are entered into the manager's :ref:`history array<datastruct-history-array>`.
  * The generator ``'out'`` field will usually include variable(s) appearing as a simulator 'in' field,
    in which case only the variable name is required for the simulator ``'in'``
    field. The example below, matches the corresponding
    :ref:`sim_specs example<sim-specs-example1>`, where ``'x'`` is defined in the gen_specs ``'out'`` field to give
    two-dimensional floats.

.. seealso::

  .. _gen-specs-example1:

  - test_uniform_sampling.py_ In this example, the
    generation function ``uniform_random_sample`` in sampling.py_ will generate 500 random
    points uniformly over the 2D domain defined by ``gen_specs['ub']`` and
    ``gen_specs['lb']``.

  ..  literalinclude:: ../../libensemble/tests/regression_tests/test_uniform_sampling.py
      :start-at: gen_specs
      :end-before: end_gen_specs_rst_tag

.. seealso::

    - test_persistent_aposmm_nlopt.py_ shows an example where ``gen_specs['in']`` is empty, but
      ``gen_specs['persis_in']`` specifies values to return to the persistent generator.

    - test_persistent_aposmm_with_grad.py_ shows a similar example where an ``H0`` is used to
      provide points from a previous run. In this case, ``gen_specs['in']`` is populated to provide
      the generator with data for the initial points.

    - In some cases you might be able to give different (perhaps fewer) fields in ``'persis_in'``
      than ``'in'``; you may not need to give ``x`` for example, as the persistent generator
      already has ``x`` for those points. See `more example uses`_ of ``persis_in``.

.. _sampling.py: https://github.com/Libensemble/libensemble/blob/develop/libensemble/gen_funcs/sampling.py
.. _test_uniform_sampling.py: https://github.com/Libensemble/libensemble/blob/develop/libensemble/tests/regression_tests/test_uniform_sampling.py
.. _test_persistent_aposmm_nlopt.py: https://github.com/Libensemble/libensemble/blob/develop/libensemble/tests/regression_tests/test_persistent_aposmm_nlopt.py
.. _test_persistent_aposmm_with_grad.py: https://github.com/Libensemble/libensemble/blob/develop/libensemble/tests/regression_tests/test_persistent_aposmm_with_grad.py
.. _more example uses: https://github.com/Libensemble/libensemble/wiki/Using-persis_in-field
