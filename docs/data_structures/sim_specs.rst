.. _datastruct-sim-specs:

sim_specs
=========

Simulation function specifications to be set in user calling script and passed to ``libE.libE()``::


    sim_specs: [dict]:

        Required keys :

        'sim_f' [func] :
            the simulation function being evaluated
        'in' [list] :
            field names (as strings) that will be given to sim_f
        'out' [list of tuples (field name, data type, [size])] :
            sim_f outputs that will be stored in the libEnsemble history


        Optional keys :

        'save_every_k' [int] :
            Save history array to file after every k simulated points.
        'sim_dir' [str] :
            Name of simulation directory which will be copied for each worker
        'sim_dir_prefix' [str] :
            A prefix path specifying where to create sim directories
        'sim_dir_suffix' [str] :
            A suffix to add to worker copies of sim_dir to distinguish runs.
         'profile' [Boolean] :
            Profile using cProfile. Default: False

        Additional entires in sim_specs will be given to sim_f

.. note::
  * The user may define other fields to be passed to the simulator function.
  * The tuples defined in the ``'out'`` list are entered into the master :ref:`history array<datastruct-history-array>`
  * The ``sim_dir_prefix`` option may be used to create simulation working directories in node local/scratch storage when workers are distributed. This may have a performance benefit with I/O heavy sim funcs.


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
