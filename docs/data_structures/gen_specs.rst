gen_specs
=========

Generation function specifications to be set in user calling script and passed to libE.libE()::

    gen_specs: [dict]:

        Required keys :     
        
        'gen_f' [func] : 
            generates inputs to sim_f
        'in' [list] : 
            field names (as strings) that will be given to gen_f
        'out' [list of tuples (field name, data type, [size])] :
            gen_f outputs that will be stored in the libEnsemble history
            
        Optional keys :
    
        'save_every_k' [int] :
            Save history array every k steps
        'queue_update_function' [func] :
            Additional entires in gen_specs will be given to gen_f

:Examples:

From: libensemble/tests/regression_tests/test_6-hump_camel_uniform_sampling.py::

    gen_specs = {'gen_f': uniform_random_sample,
                 'in': ['sim_id'],
                 'out': [('x',float,2),
                        ],
                 'lb': np.array([-3,-2]),
                 'ub': np.array([ 3, 2]),
                 'gen_batch_size': 500,
                 'batch_mode': True,
                 'num_inst':1,
                 'save_every_k': 300
                 }
 
:See Also:
