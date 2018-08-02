sim_specs
=========


Simulation function specifications to be set in user calling script and passed to libE.libE()::


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
            Save history array every k steps
        'sim_dir' [str] :
            Name of simulation directory which will be copied for each worker
        'sim_dir_prefix' [str] :
            A prefix path specifying where to create sim directories
        
        Additional entires in sim_specs will be given to sim_f
        
        
:Examples:

From: libensemble/tests/regression_tests/test_6-hump_camel_uniform_sampling.py::

    sim_specs = {'sim_f': six_hump_camel, # This is the function whose output is being minimized
                 'in': ['x'],             # These keys will be given to the above function
                 'out': [('f',float),     # This is the output from the function being minimized
                        ],
                 'save_every_k': 400  
                 }

:See Also:



