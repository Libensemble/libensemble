*********************
Input data structures
*********************
We first describe the dictionaries given to libEnsemble to specify the
inputs/outputs of the ensemble of calculations to be performed.

sim_specs
=========
.. _datastruct-sim-specs:

Used to specify the simulation function, its inputs and outputs, and user data::

    sim_specs: [dict]:

        'sim_f' [func]:
            the simulation function being evaluated
        'in' [list]:
            field names (as strings) to be given to sim_f by alloc_f
        'out' [list of tuples (field name, data type, [size])]:
            sim_f outputs to be stored in the libEnsemble history
        'user' [dict, optional]:
            Data structure to contain problem specific constants and/or input data

.. note::
  * The entirety of ``sim_specs`` is passed from the worker each time a
    simulation is requested by the allocation function.

  * The tuples in ``sim_specs['out']`` are entered into the master
    :ref:`history array<datastruct-history-array>`

  * The ``libE_specs['sim_dir_prefix']`` option may be used to create
    simulation working directories in node local/scratch storage when workers
    are distributed. This may have a performance benefit with I/O heavy
    simulations.


.. seealso::

  .. _sim-specs-exmple1:

  From: ``libensemble/tests/regression_tests/test_6-hump_camel_uniform_sampling.py``

  ..  literalinclude:: ../../libensemble/tests/regression_tests/test_6-hump_camel_uniform_sampling.py
      :start-at: sim_specs
      :end-before: end_sim_specs_rst_tag


  The dimensions and type of the ``'in'`` field variable ``'x'`` is specified by the corresponding
  generator ``'out'`` field ``'x'`` (see :ref:`gen_specs example<gen-specs-exmple1>`).
  Only the variable name is then required in ``sim_specs['in']``.

  From: ``libensemble/tests/scaling_tests/forces/run_libe_forces.py``

  ..  literalinclude:: ../../libensemble/tests/scaling_tests/forces/run_libe_forces.py
      :start-at: sim_f
      :end-before: end_sim_specs_rst_tag

  This example uses a number of user specific fields, that will be dealt with in the corresponding sim f, which
  can be found at ``libensemble/tests/scaling_tests/forces/forces_simf.py``
