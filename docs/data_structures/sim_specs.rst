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
        
        Additional entires in sim_specs will be given to sim_f
        
:Notes:

* The user may define other fields to be passed to the simulator function.
* The tuples defined in the ``'out'`` list are entered into the master :ref:`history array<datastruct-history-array>`

:Examples:

.. _sim-specs-exmple1:

From: ``libensemble/tests/regression_tests/test_6-hump_camel_uniform_sampling.py``

    sim_specs = {'sim_f': six_hump_camel, # This is the function whose output is being minimized
                 'in': ['x'],             # These keys will be given to the above function
                 'out': [('f',float)],    # This is the output from the function being minimized
                 'save_every_k': 400  
                 }

Note that the dimensions and type of the ``'in'`` field variable ``'x'`` is specified by the corresponding
generator ``'out'`` field ``'x'`` (see :ref:`gen_specs example<gen-specs-exmple1>`).
Only the variable name is then required in sim_specs.
