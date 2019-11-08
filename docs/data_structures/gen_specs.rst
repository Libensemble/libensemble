.. _datastruct-gen-specs:

gen_specs
=========

Generation function specifications to be set in user calling script and passed
to main ``libE()`` routine::

    gen_specs: [dict]:

        'gen_f' [func] :
            generates inputs to sim_f
        'in' [list] :
            field names (as strings) that will be given to gen_f
        'out' [list of tuples (field name, data type, [size])] :
            gen_f outputs that will be stored in the libEnsemble history
        'user' [dict]:
            Data structure to contain problem specific constants and/or input data

.. note::

  * The user may define fields only in ``'user'`` to be passed to the generator function.
  * The tuples defined in the 'out' list are entered into the master :ref:`history array<datastruct-history-array>`
  * The generator ``'out'`` field will generally include a variable(s) which is used for the simulator 'in' field,
    in which case only the variable name is required for the simulator ``'in'`` field.  E.g. The
    **test_6-hump_camel_uniform_sampling.py** example below, matches the corresponding
    :ref:`sim_specs example<sim-specs-exmple1>`, where ``'x'`` is defined in the gen_specs ``'out'`` field to give
    two-dimensional floats.

.. seealso::

  .. _gen-specs-exmple1:

  - test_6-hump_camel_uniform_sampling.py_ declares In this example, the
    generation function ``uniform_random_sample`` in sampling.py_ will generate 500 random
    points uniformly over the 2D domain defined by ``gen_specs['ub']`` and
    ``gen_specs['lb']``.

  ..  literalinclude:: ../../libensemble/tests/regression_tests/test_6-hump_camel_uniform_sampling.py
      :start-at: gen_specs
      :end-before: end_gen_specs_rst_tag

.. _sampling.py: https://github.com/Libensemble/libensemble/blob/develop/libensemble/gen_funcs/sampling.py
.. _test_6-hump_camel_uniform_sampling.py: https://github.com/Libensemble/libensemble/blob/develop/libensemble/tests/regression_tests/test_6-hump_camel_uniform_sampling.py
