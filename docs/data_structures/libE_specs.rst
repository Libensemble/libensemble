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
        'sim_input_dir' [str] :
            Name of directory which will be copied for each sim calc
        'use_worker_dirs' [bool] :
            Divide calc_dirs into per_worker parent directories.
        'clean_jobs' [bool] :
            Clean up calc_dirs after libEnsemble completes. Default: False
        'ensemble_dir' [str] :
            A prefix path specifying where to create sim directories
        'ensemble_dir_suffix' [str] :
            A suffix to add to worker copies of sim_input_dir to distinguish runs.
        'copy_input_files' [list] :
            List of filenames to copy from the input dir. Ignore all others.
        'symlink_input_files' [list] :
            List of filenames to symlink instead of copy.
        'copy_input_to_parent' [bool] :
            Copy all input files to the parent dirs containing calc dirs. Default: False
        'profile_worker' [Boolean] :
            Profile using cProfile. Default: False

.. note::
    The ``ensemble_dir`` and ``sim_input_dir`` options can indicate working
    directories on local node or scratch storage. This may produce performance
    benefits on I/O heavy simulations, but will use more space.

.. seealso::
  Example ``libE_specs`` from the forces_ scaling test, completely populated::

      libE_specs = {'comm': MPI.COMM_WORLD,
                    'comms': 'mpi',
                    'save_every_k_gens': 1000,
                    'sim_input_dir': './sim',
                    'profile_worker': False}

.. _forces: https://github.com/Libensemble/libensemble/blob/develop/libensemble/tests/scaling_tests/forces/run_libe_forces.py
