.. _datastruct-libe-specs:

libE_specs
==========

Specifications for libEnsemble::

    libE_specs: [dict, optional] :
        'comms' [string] :
            Manager/Worker communications mode. Default: mpi
            Options are 'mpi', 'local', 'tcp'
        'nworkers' [int]:
            Number of worker processes to spawn (in local/tcp modes)
        'comm' [MPI communicator] :
            libEnsemble communicator. Default: MPI.COMM_WORLD
        'abort_on_exception' [boolean] :
            In MPI mode, whether to call MPI_ABORT on an exception. Default: True
            IF False, an exception will be raised by the manager.
        'save_every_k_sims' [int] :
            Save history array to file after every k simulated points.
        'save_every_k_gens' [int] :
            Save history array to file after every k generated points.
        'sim_dir' [str] :
            Name of simulation directory which will be copied for each worker
        'clean_jobs' [bool] :
            Clean up sim_dirs after libEnsemble completes. Default: False
        'sim_dir_prefix' [str] :
            A prefix path specifying where to create sim directories
        'sim_dir_suffix' [str] :
            A suffix to add to worker copies of sim_dir to distinguish runs.
        'profile_worker' [Boolean] :
            Profile using cProfile. Default: False

.. seealso::
  Example ``libE_specs`` from the forces_ scaling test, completely populated::

      libE_specs = {'comm': MPI.COMM_WORLD,
                    'comms': 'mpi',
                    'save_every_k_gens': 1000,
                    'sim_dir': './sim',
                    'profile_worker': False}

.. _forces: https://github.com/Libensemble/libensemble/blob/develop/libensemble/tests/scaling_tests/forces/run_libe_forces.py
