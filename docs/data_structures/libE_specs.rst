.. _datastruct-libe-specs:

libE_specs
==========

libEnsemble is primarily customized by setting options within a ``libE_specs`` dictionary::

    libE_specs: [dict, optional]:

        General options:
        ----------------
        'comms' [str]:
            Manager/Worker communications mode. Default: mpi
            Options are 'mpi', 'local', 'tcp'
        'nworkers' [int]:
            Number of worker processes to spawn (only in local/tcp modes)
        'mpi_comm' [MPI communicator]:
            libEnsemble communicator. Default: MPI.COMM_WORLD
        'abort_on_exception' [bool]:
            In MPI mode, whether to call MPI_ABORT on an exception. Default: True
            If False, an exception will be raised by the manager.
        'save_every_k_sims' [int]:
            Save history array to file after every k simulated points.
        'save_every_k_gens' [int]:
            Save history array to file after every k generated points.
        'save_H_and_persis_on_abort' [bool]:
            Whether libEnsemble should save the states of H and persis_info on
            aborting after an error.
            Default: True
        'worker_timeout' [int]:
            When libEnsemble concludes and attempts to close down worker processes,
            the number of seconds after which workers are considered timed out. Worker
            processes are then terminated. multiprocessing default: 1
        'kill_canceled_sims' [bool]:
            Will libE try to kill sims that user functions mark 'cancel_requested' as True.
            If False, the manager avoids this moderate overhead.
            Default: True

        Directory management options:
        -----------------------------
        'ensemble_dir_path' [str]:
            Path to main ensemble directory containing calculation directories. Can serve
            as single working directory for workers, or contain calculation directories.
            Default: './ensemble'
        'ensemble_copy_back' [bool]:
            Whether to copy back directories within ensemble_dir_path back to launch
            location. Useful if ensemble_dir placed on node-local storage.
            Default: False
        'use_worker_dirs' [bool]:
            Whether to organize calculation directories under worker-specific directories.
            Default: False

        'sim_dirs_make' [bool]:
            Whether to make simulation-specific calculation directories for each sim call.
            This will create a directory for each simulation, even if no sim_input_dir is
            specified. If False, all workers operate within the ensemble directory
            described below.
            Default: False
        'sim_dir_copy_files' [list]:
            Paths to files or directories to copy into each sim directory, or ensemble directory.
        'sim_dir_symlink_files' [list]:
            Paths to files or directories to symlink into each sim directory.
        'sim_input_dir' [str]:
            Copy this directory and its contents for each simulation-specific directory.
            If not using calculation directories, contents are copied to the ensemble directory.

        'gen_dirs_make' [bool]:
            Whether to make generator-instance specific calculation directories for each
            gen call. This will create a directory for each generator call, even if no
            gen_input_dir is specified. If False, all workers operate within the ensemble
            directory. Note that if using a persistent generator function
            Default: False
        'gen_dir_copy_files' [list]:
            Paths to files or directories to copy into each gen directory, or ensemble directory.
        'gen_dir_symlink_files' [list]:
            Paths to files or directories to symlink into each gen directory.
        'gen_input_dir' [str]:
            Copy this directory and its contents for each generator-instance specific directory.
            If not using calc directories, contents are copied to the ensemble directory.

        Profiling/timing options:
        -------------------------
        'profile' [bool]:
            Profile manager and worker logic using cProfile. Default: False
        'disable_log_files' [bool]:
            Disable the creation of 'ensemble.log' and 'libE_stats.txt' log files.
            Default: False
        'safe_mode' [bool]:
            Prevents user functions from overwriting protected libE fields, but requires
            moderate overhead.
            Default: True
        'stats_fmt' [dict]:
            A dictionary of options for formatting the 'libE_stats.txt' output file.
            See 'Formatting Options for libE_stats File' for more options.

        TCP only options:
        -----------------
        'workers' [list]:
            TCP Only: A list of worker hostnames.
        'ip' [str]:
            TCP Only: IP address for Manager's system
        'port' [int]:
            TCP Only: Port number for Manager's system
        'authkey' [str]:
            TCP Only: Authkey for Manager's system
        'workerID' [int]:
            TCP Only: Worker ID number assigned to the new process.
        'worker_cmd' [list]:
            TCP Only: Split string corresponding to worker/client Python process invocation. Contains
            a local Python path, calling script, and manager/server format-fields for manager_ip,
            manager_port, authkey, and workerID. nworkers is specified normally.

        Options for history management with persistent workers:
        -------------------------------------------------------
        'use_persis_return_gen' [bool]:
            Adds persistent generator function H return to managers history array.
            Default: False
        'use_persis_return_sim' [bool]:
            Adds persistent simulator function H return to managers history array.
            Default: False
        'final_fields' [list]:
            List of fields in H that the manager will return to persistent
            workers along with the PERSIS_STOP tag at the end of the libE run.
            Default: None

        Resource management options:
        ----------------------------
        'disable_resource_manager' [bool]:
            Disable the built-in resource manager. If this is True, automatic resource detection
            and/or assignment of resources to workers is disabled. This also means that
            any entries in the "resource_info" option will be ignored.
            Default: False
        'num_resource_sets' [int]:
            The total number of resource sets. Resources will be divided into this number.
            Default: None. If None, resources will be divided by workers (excluding
            zero_resource_workers).
        'enforce_worker_core_bounds' [bool]:
            If False, the Executor will permit submission of tasks with a
            higher processor count than the CPUs available to the worker as
            detected by the resource manager. Larger node counts are not allowed.
            When the libE_specs option "disable_resource_manager" is True,
            this argument is ignored. Default: False
        'dedicated_mode' [bool]:
            If True, then running in dedicated mode, otherwise in distributed
            mode. Dedicated mode means libE processes (manager and workers) are
            grouped together and do not share nodes with applications.
            Distributed mode means workers share nodes with applications.
            Default: False
        'zero_resource_workers' [list of ints]:
            List of workers that require no resources. For when a fixed mapping of workers
            to resources is required. Otherwise, use "num_resource_sets".
            For use with supported allocation functions.
        'resource_info' [dict]:
            Provide resource information that will override automatically detected resources.
            The allowable fields are given below in 'Overriding Auto-detection'
            Note that if "disable_resource_manager" is set then
            this option is ignored.
        'scheduler_opts' [dict]:
            A dictionary of options for the resource scheduler.
            See 'Scheduler Options' for more options.

The following describe the dictionary options within ``libE_specs``.

Scheduler Options
-----------------

See options for :ref:`built-in scheduler<resources-scheduler>`.

.. _resource_info:

Overriding Resource Auto-Detection
----------------------------------

The allowable ``libE_specs["resource_info"]`` fields are::

    'cores_on_node' [tuple (int, int)]:
        Tuple (physical cores, logical cores) on nodes.
    'node_file' [str]:
        Name of file containing a node-list. Default is 'node_list'.
    'nodelist_env_slurm' [str]:
        The environment variable giving a node list in Slurm format
        (Default: Uses SLURM_NODELIST).  Note: This is queried only if
        a node_list file is not provided and the resource manager is
        enabled (default).
    'nodelist_env_cobalt' [str]:
        The environment variable giving a node list in Cobalt format
        (Default: Uses COBALT_PARTNAME) Note: This is queried only
        if a node_list file is not provided and the resource manager
        is enabled (default).
    'nodelist_env_lsf' [str]:
        The environment variable giving a node list in LSF format
        (Default: Uses LSB_HOSTS) Note: This is queried only
        if a node_list file is not provided and the resource manager
        is enabled (default).
    'nodelist_env_lsf_shortform' [str]:
        The environment variable giving a node list in LSF short-form
        format (Default: Uses LSB_MCPU_HOSTS) Note: This is queried only
        if a node_list file is not provided and the resource manager is
        enabled (default).

For example::

    customizer = {cores_on_node': (16, 64),
                  'node_file': 'libe_nodes'}

    libE_specs['resource_info'] = customizer

.. seealso::
  Example ``libE_specs`` from the forces_ scaling test, completely populated::

      libE_specs = {'comm': MPI.COMM_WORLD,
                    'comms': 'mpi',
                    'save_every_k_gens': 1000,
                    'sim_dirs_make: True,
                    'ensemble_dir_path': '/scratch/ensemble'
                    'profile_worker': False}

Formatting Options for libE_stats File
--------------------------------------

The allowable ``libE_specs["stats_fmt"]`` fields are::

    'task_timing' [bool]:
        Outputs elapsed time for each task launched by the executor.
        Default: False
    'task_datetime' [bool]:
        Outputs the elapsed time and start and end time for each task launched by the executor.
        Can be used with the 'plot_libe_tasks_util_v_time.py' to give task utilization plots.
        Default: False
    'show_resource_sets' [bool]:
        Shows the resource set IDs assigned to each worker for each call of the user function.
        Default: False

.. _forces: https://github.com/Libensemble/libensemble/blob/develop/libensemble/tests/scaling_tests/forces/run_libe_forces.py
