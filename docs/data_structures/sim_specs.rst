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

:Notes:

* The user may define other fields to be passed to the simulator function.
* The tuples defined in the ``'out'`` list are entered into the master :ref:`history array<datastruct-history-array>`
* The ``sim_dir_prefix`` option may be used to create simulation working directories in node local/scratch storage when workers are distributed. This may have a performance benefit with I/O heavy sim funcs.


:Examples:

.. _sim-specs-exmple1:

From: ``libensemble/tests/regression_tests/test_6-hump_camel_uniform_sampling.py``::

    sim_specs = {'sim_f': six_hump_camel, # This is the function whose output is being minimized
                 'in': ['x'],             # These keys will be given to the above function
                 'out': [('f',float)],    # This is the output from the function being minimized
                 'save_every_k': 400
                 }

Note that the dimensions and type of the ``'in'`` field variable ``'x'`` is specified by the corresponding
generator ``'out'`` field ``'x'`` (see :ref:`gen_specs example<gen-specs-exmple1>`).
Only the variable name is then required in sim_specs.

From: ``libensemble/tests/scaling_tests/forces/run_libe_forces.py``::

    sim_specs = {'sim_f': run_forces,             # This is the function whose output is being minimized (sim func)
                 'in': ['x'],                     # Name of input data structure for sim func
                 'out': [('energy', float)],      # Output from sim func
                 'keys': ['seed'],                # Key/keys for input data
                 'sim_dir': './sim',              # Simulation input dir to be copied for each worker (*currently empty)
                 'sim_dir_suffix': 'test',        # Suffix for copied sim dirs to indentify run (in case multiple)
                 'simdir_basename': 'forces',     # User attribute to name sim directories (forces_***)
                 'cores': 2,                      # User attribute to set number of cores for sim func runs (optional)
                 'sim_particles': 1e3,            # User attribute for number of particles in simulations
                 'sim_timesteps': 5,              # User attribute for number of timesteps in simulations
                 'sim_kill_minutes': 10.0,        # User attribute for max time for simulations
                 'kill_rate': 0.5,                # Between 0 and 1 for proportion of jobs that go bad (for testing kills)
                 'particle_variance': 0.2,        # Range over which particle count varies (for testing load imbalance)
                 'profile': False
                 }

This example uses a number of user specific fields, that will be dealt with in the corresponding sim f, which
can be found at ``libensemble/tests/scaling_tests/forces/forces_simf.py``
