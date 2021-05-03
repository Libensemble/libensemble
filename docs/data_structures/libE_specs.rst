.. _datastruct-libe-specs:

libE_specs
==========

Specifications for libEnsemble::

    libE_specs: [dict, optional]:
        'comms' [string]:
            Manager/Worker communications mode. Default: mpi
            Options are 'mpi', 'local', 'tcp'
        'nworkers' [int]:
            Number of worker processes to spawn (in local/tcp modes)
        'comm' [MPI communicator]:
            libEnsemble communicator. Default: MPI.COMM_WORLD
        'abort_on_exception' [boolean]:
            In MPI mode, whether to call MPI_ABORT on an exception. Default: True
            IF False, an exception will be raised by the manager.
        'save_every_k_sims' [int]:
            Save history array to file after every k simulated points.
        'save_every_k_gens' [int]:
            Save history array to file after every k generated points.
        'sim_dirs_make' [boolean]:
            Whether to make simulation-specific calculation directories for each sim call.
            This will create a directory for each simulation, even if no sim_input_dir is
            specified. If False, all workers operate within the ensemble directory
            described below.
            Default: True
        'gen_dirs_make' [boolean]:
            Whether to make generator-instance specific calculation directories for each
            gen call. This will create a directory for each generator call, even if no
            gen_input_dir is specified. If False, all workers operate within the ensemble
            directory.
            Default: True
        'ensemble_dir_path' [string]:
            Path to main ensemble directory containing calculation directories. Can serve
            as single working directory for workers, or contain calculation directories.
            Default: './ensemble'
        'use_worker_dirs' [boolean]:
            Whether to organize calculation directories under worker-specific directories.
            Default: False
        'sim_dir_copy_files' [list]:
            Paths to files or directories to copy into each sim dir, or ensemble dir.
        'sim_dir_symlink_files' [list]:
            Paths to files or directories to symlink into each sim dir.
        'gen_dir_copy_files' [list]:
            Paths to files or directories to copy into each gen dir, or ensemble dir.
        'gen_dir_symlink_files' [list]:
            Paths to files or directories to symlink into each gen dir.
        'ensemble_copy_back' [boolean]:
            Whether to copy back directories within ensemble_dir_path back to launch
            location. Useful if ensemble_dir placed on node-local storage.
            Default: False
        'sim_input_dir' [string]:
            Copy this directory and its contents for each simulation-specific directory.
            If not using calculation directories, contents are copied to the ensemble dir.
        'gen_input_dir' [string]:
            Copy this directory and its contents for each generator-instance specific dir.
            If not using calc directories, contents are copied to the ensemble directory.
        'profile_worker' [boolean]:
            Profile using cProfile. Default: False
        'disable_log_files' [boolean]:
            Disable the creation of 'ensemble.log' and 'libE_stats.txt' log files.
            Default: False
        'workers' [list]:
            TCP Only: A list of worker hostnames.
        'ip' [string]:
            TCP Only: IP address
        'port' [int]:
            TCP Only: Port number
        'authkey' [string]:
            TCP Only: Authkey
        'safe_mode' [boolean]:
            Prevents user functions from overwritting protected libE fields.
            Default: True
        'use_persis_return' [boolean]:
            Adds persistent function H return to managers history array.
            Default: False

.. note::
    The ``ensemble_dir_path`` option can create working directories on local node or
    scratch storage. This may produce performance benefits on I/O heavy simulations.

.. seealso::
  Example ``libE_specs`` from the forces_ scaling test, completely populated::

      libE_specs = {'comm': MPI.COMM_WORLD,
                    'comms': 'mpi',
                    'save_every_k_gens': 1000,
                    'sim_dirs_make: True,
                    'ensemble_dir_path': '/scratch/ensemble'
                    'profile_worker': False}

.. _forces: https://github.com/Libensemble/libensemble/blob/develop/libensemble/tests/scaling_tests/forces/run_libe_forces.py
