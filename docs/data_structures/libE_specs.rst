.. _datastruct-libe-specs:

libE_specs
==========

Specifications for libEnsemble::

    libE_specs: [dict] :
        'comms' [string] :
            Manager/Worker communications mode. Default: mpi
            Options are 'mpi', 'local', 'tcp'
        'nprocesses' [int]:
            Number of worker processes to spawn (in local/tcp modes)
        'comm' [MPI communicator] :
            libEnsemble communicator. Default: MPI.COMM_WORLD
        'color' [int] :
            Communicator color. Default: 0
        'abort_on_exception' [boolean] :
            In MPI mode, whether to call MPI_ABORT on an exception. Default: True
            IF False, an exception will be raised by the manager.
