.. _datastruct-sim-specs:

sim_specs
=========
Used to specify the simulation function, its inputs and outputs, and user data::

    sim_specs: [dict]:

        'sim_f' [func]:
            the simulation function being evaluated
        'in' [list]:
            field names (as strings) to be given to sim_f by alloc_f
        'persis_in' [list]:
            field names (as strings) that will be given back to a persistent sim_f
        'out' [list of tuples (field name, data type, [size])]:
            sim_f outputs to be stored in the libEnsemble history
        'user' [dict, optional]:
            Data structure to contain problem specific constants and/or input data

.. note::
  * The entirety of ``sim_specs`` is passed from the worker each time a
    simulation is requested by the allocation function.

  * The tuples in ``sim_specs['out']`` are entered into the manager's
    :ref:`history array<datastruct-history-array>`.

.. seealso::

  .. _sim-specs-example1:

  - test_uniform_sampling.py_ has a ``sim_specs``  that declares
    the name of the ``'in'`` field variable, ``'x'`` (as specified by the
    corresponding generator ``'out'`` field ``'x'`` from the :ref:`gen_specs
    example<gen-specs-example1>`).  Only the field name is required in
    ``sim_specs['in']``.

  ..  literalinclude:: ../../libensemble/tests/regression_tests/test_uniform_sampling.py
      :start-at: sim_specs
      :end-before: end_sim_specs_rst_tag

  - run_libe_forces.py_ has a longer ``sim_specs`` declaration with a number of
    user-specific fields. These are given to the corresponding sim_f, which
    can be found at forces_simf.py_.

  ..  literalinclude:: ../../libensemble/tests/scaling_tests/forces/forces_adv/run_libe_forces.py
      :start-at: sim_f
      :end-before: end_sim_specs_rst_tag

.. _forces_simf.py: https://github.com/Libensemble/libensemble/blob/develop/libensemble/tests/scaling_tests/forces/forces_simf.py
.. _run_libe_forces.py: https://github.com/Libensemble/libensemble/blob/develop/libensemble/tests/scaling_tests/forces/run_libe_forces.py
.. _test_uniform_sampling.py: https://github.com/Libensemble/libensemble/blob/develop/libensemble/tests/regression_tests/test_uniform_sampling.py
