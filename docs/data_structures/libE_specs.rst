.. _datastruct-libe-specs:

libE_specs
==========

Specifications for libEnsemble::

    libE_specs: [dict] :
        'comm' [MPI communicator] :
            libEnsemble communicator. Default: MPI.COMM_WORLD
        'color' [int] :
            Communicator color. Default: 0
        'manager_ranks' [set] :
            Default: [0]
        'worker_ranks' [set] :
            Default: [1 to comm.Get_size()-1]
        'queue_update_function' [func] :
            Default: []
            
