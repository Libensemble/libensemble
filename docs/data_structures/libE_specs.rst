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
            Prevents user functions from overwriting protected libE fields, but requires
            moderate overhead.
            Default: True
        'kill_canceled_sims' [boolean]:
            Will libE try to kill sims that user functions mark 'cancel_requested' as True.
            If False, the manager avoid this moderate overhead.
            Default: True
        'use_persis_return_gen' [boolean]:
            Adds persistent generator function H return to managers history array.
            Default: False
        'use_persis_return_sim' [boolean]:
            Adds persistent simulator function H return to managers history array.
            Default: False
        'final_fields' [list]:
            List of fields in H that the manager will return to persistent
            workers along with the PERSIS_STOP tag at the end of the libE run.
            Default: None
        'disable_resource_manager' [boolean]:
            Disable the built-in resource manager. If this is True, automatic resource detection
            and/or assignment of resources to workers is disabled. This also means that
            any entries in the ``resource_info`` option will be ignored.
            Default: False
        'num_resource_sets' [int]:
            The total number of resource sets. Resources will be divided into this number.
            Default: None. If None, resources will be divided by workers (excluding
            zero_resource_workers).
        'enforce_worker_core_bounds' [boolean]:
            If false, the Executor will permit submission of tasks with a
            higher processor count than the CPUs available to the worker as
            detected by the resource manager. Larger node counts are not allowed.
            When the libE_specs option `disable_resource_manager` is True,
            this argument is ignored. Default: False
        'dedicated_mode' [boolean]:
            If true, then running in dedicated mode, otherwise in distributed
            mode. Dedicated mode means libE processes (manager and workers) are
            grouped together and do not share nodes with applications.
            Distributed mode means workers share nodes with applications.
            Default: False
        'zero_resource_workers' [list of ints]:
            List of workers that require no resources.
        'resource_info' [dict]:
            Provide resource information that will override automatically detected resources.
            The allowable fields are given below in 'Overriding Auto-detection'
            Note that if ``disable_resource_manager`` is set then
            this option is ignored.

.. _resource_info:

Overriding Auto-detection
-------------------------

The allowable fields are::

    'cores_on_node' [tuple (int,int)]:
        Tuple (physical cores, logical cores) on nodes.
    'node_file' [string]:
        Name of file containing a node-list. Default is 'node_list'.
    'nodelist_env_slurm' [String]:
            The environment variable giving a node list in Slurm format
            (Default: Uses SLURM_NODELIST).  Note: This is queried only if
            a node_list file is not provided and the resource manager is
            enabled (default).
    'nodelist_env_cobalt' [String]:
            The environment variable giving a node list in Cobalt format
            (Default: Uses COBALT_PARTNAME) Note: This is queried only
            if a node_list file is not provided and the resource manager
            is enabled (default).
    'nodelist_env_lsf' [String]:
            The environment variable giving a node list in LSF format
            (Default: Uses LSB_HOSTS) Note: This is queried only
            if a node_list file is not provided and the resource manager
            is enabled (default).
    'nodelist_env_lsf_shortform' [String]:
            The environment variable giving a node list in LSF short-form
            format (Default: Uses LSB_MCPU_HOSTS) Note: This is queried only
            if a node_list file is not provided and the resource manager is
            enabled (default).

For example::

    customizer = {cores_on_node': (16, 64),
                  'node_file': 'libe_nodes'}

    libE_specs['resource_info'] = customizer

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
