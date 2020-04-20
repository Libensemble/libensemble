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
        'sim_dirs_make' [boolean] :
            Whether to make simulation-specific directories for each sim call.
            Default: False
        'sim_dir_path' [string] :
            Path to directory into which to put sim directories. Default: './ensemble'
        'sim_dirs_per_worker' [boolean] :
            Whether to organize sim dirs into directories by which worker performed sim.
            Default: False
        'sim_dir_copy_files' [list] :
            List of paths to files to copy into each sim dir.
        'sim_dir_symlink_files' [list] :
            List of paths to files to symlink into each sim dir.
        'sim_dir_copy_back' [boolean] :
            Whether to copy back directories within sim_dir_path back to launch location.
            Default: False
        'sim_input_dir' [string] :
            Copy this directory and it's contents for each simulation-specific directory.
        'profile_worker' [boolean] :
            Profile using cProfile. Default: False
        'disable_log_files' [boolean] :
            Disable the creation of 'ensemble.log' and 'libE_stats.txt' log files.
            Default: False

.. note::
    The ``sim_dir_path`` option can create working directories on local node or
    scratch storage. This may produce performance benefits on I/O heavy simulations.

.. seealso::
  Example ``libE_specs`` from the forces_ scaling test, completely populated::

      libE_specs = {'comm': MPI.COMM_WORLD,
                    'comms': 'mpi',
                    'save_every_k_gens': 1000,
                    'make_sim_dirs: True,
                    'sim_dir_path': '/scratch/ensemble'
                    'profile_worker': False}

.. _forces: https://github.com/Libensemble/libensemble/blob/develop/libensemble/tests/scaling_tests/forces/run_libe_forces.py
