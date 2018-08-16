alloc_specs
===========

Allocation function specifications to be set in user calling script and passed to libE.libE()::

    alloc_specs: [dict, optional] :
        'alloc_f' [func] :
            Default: give_sim_work_first
        'out' [list of tuples] :
            Default: [('allocated',bool)]
        'batch_mode' [bool] :
            Default: []
        'num_inst' [int] :
            Default: []
            
        The 'batch_mode' and 'num_inst' are specific arguments for the allocation function give_sim_work_first

