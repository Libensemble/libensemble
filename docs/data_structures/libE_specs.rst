.. _datastruct-libe-specs:

libE_specs
==========

Specifications for libEnsemble::

    libE_specs: [dict] :
        'comm' [MPI communicator] :
            libEnsemble communicator. Default: MPI.COMM_WORLD
        'color' [int] :
            Communicator color. Default: 0
        'queue_update_function' [func] :
            Default: []
