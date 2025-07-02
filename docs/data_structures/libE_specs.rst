.. _datastruct-libe-specs:

LibE Specs
==========

libEnsemble is primarily customized by setting options within a ``LibeSpecs`` class or dictionary.

.. code-block:: python

    from libensemble.specs import LibeSpecs

    specs = LibeSpecs(
        gen_on_manager=True,
        save_every_k_gens=100,
        sim_dirs_make=True,
        nworkers=4
    )

.. dropdown:: Settings by Category
    :open:

    .. tab-set::

        .. tab-item:: General

                **comms** [str] = ``"mpi"``:
                    Manager/Worker communications mode: ``'mpi'``, ``'local'``, or ``'tcp'``.
                    If ``nworkers`` is specified, then ``local`` comms will be used unless a
                    parallel MPI environment is detected.

                **nworkers** [int]:
                    Number of worker processes in ``"local"``, ``"threads"``, or ``"tcp"``.

                **gen_on_manager** [bool] = False
                    Instructs Manager process to run generator functions.
                    This generator function can access/modify user objects by reference.

                **mpi_comm** [MPI communicator] = ``MPI.COMM_WORLD``:
                    libEnsemble MPI communicator.

                **dry_run** [bool] = ``False``:
                    Whether libEnsemble should immediately exit after validating all inputs.

                **abort_on_exception** [bool] = ``True``:
                    In MPI mode, whether to call ``MPI_ABORT`` on an exception.
                    If ``False``, an exception will be raised by the manager.

                **worker_timeout** [int] = ``1``:
                    On libEnsemble shutdown, number of seconds after which workers considered timed out,
                    then terminated.

                **kill_canceled_sims** [bool] = ``False``:
                    Try to kill sims with ``cancel_requested`` set to ``True``.
                    If ``False``, the manager avoids this moderate overhead.

                **disable_log_files** [bool] = ``False``:
                    Disable ``ensemble.log`` and ``libE_stats.txt`` log files.

                **gen_workers** [list of ints]:
                    List of workers that should run only generators. All other workers will run
                    only simulator functions.

        .. tab-item:: Directories

            .. tab-set::

                .. tab-item:: General

                    **use_workflow_dir** [bool] = ``False``:
                        Whether to place *all* log files, dumped arrays, and default ensemble-directories in a
                        separate ``workflow`` directory. Each run is suffixed with a hash.
                        If copying back an ensemble directory from another location, the copy is placed here.

                    **workflow_dir_path** [str]:
                        Optional path to the workflow directory.

                    **ensemble_dir_path** [str] = ``"./ensemble"``:
                        Path to main ensemble directory. Can serve
                        as single working directory for workers, or contain calculation directories.

                        .. code-block:: python

                            LibeSpecs.ensemble_dir_path = "/scratch/my_ensemble"

                    **ensemble_copy_back** [bool] = ``False``:
                        Whether to copy back contents of ``ensemble_dir_path`` to launch
                        location. Useful if ``ensemble_dir_path`` is located on node-local storage.

                    **reuse_output_dir** [bool] = ``False``:
                        Whether to allow overwrites and access to previous ensemble and workflow directories in subsequent runs.
                        ``False`` by default to protect results.

                    **calc_dir_id_width** [int] = ``4``:
                        The width of the numerical ID component of a calculation directory name. Leading
                        zeros are padded to the sim/gen ID.

                    **use_worker_dirs** [bool] = ``False``:
                        Whether to organize calculation directories under worker-specific directories:

                        .. tab-set::

                            .. tab-item:: False

                                .. code-block::

                                    - /ensemble_dir
                                        - /sim0000
                                        - /gen0001
                                        - /sim0001
                                        ...

                            .. tab-item:: True

                                .. code-block::

                                    - /ensemble_dir
                                        - /worker1
                                            - /sim0000
                                            - /gen0001
                                            - /sim0004
                                            ...
                                        - /worker2
                                        ...

                .. tab-item:: Sims

                    **sim_dirs_make** [bool] = ``False``:
                        Whether to make calculation directories for each simulation function call.

                    **sim_dir_copy_files** [list]:
                        Paths to files or directories to copy into each sim directory, or ensemble directory.
                        List of strings or ``pathlib.Path`` objects.

                    **sim_dir_symlink_files** [list]:
                        Paths to files or directories to symlink into each sim directory, or ensemble directory.
                        List of strings or ``pathlib.Path`` objects.

                    **sim_input_dir** [str]:
                        Copy this directory's contents into the working directory upon calling the simulation function.
                        Forms the base of a simulation directory.

                .. tab-item:: Gens

                    **gen_dirs_make** [bool] = ``False``:
                        Whether to make generator-specific calculation directories for each generator function call.
                        *Each persistent generator creates a single directory*.

                    **gen_dir_copy_files** [list]:
                        Paths to copy into the working directory upon calling the generator function.
                        List of strings or ``pathlib.Path`` objects

                    **gen_dir_symlink_files** [list]:
                        Paths to files or directories to symlink into each gen directory.
                        List of strings or ``pathlib.Path`` objects

                    **gen_input_dir** [str]:
                        Copy this directory's contents into the working directory upon calling the generator function.
                        Forms the base of a generator directory.

        .. tab-item:: Profiling

                **profile** [bool] = ``False``:
                    Profile manager and worker logic using ``cProfile``.

                **safe_mode** [bool] = ``True``:
                    Prevents user functions from overwriting internal fields, but requires moderate overhead.

                **stats_fmt** [dict]:
                    A dictionary of options for formatting ``"libE_stats.txt"``.
                    See "Formatting Options for libE_stats.txt".

                **live_data** [LiveData] = None:
                    Add a live data capture object (e.g., for plotting).

        .. tab-item:: TCP

                **workers** [list]:
                    TCP Only: A list of worker hostnames.

                **ip** [str]:
                    TCP Only: IP address for Manager's system.

                **port** [int]:
                    TCP Only: Port number for Manager's system.

                **authkey** [str]:
                    TCP Only: Authkey for Manager's system.

                **workerID** [int]:
                    TCP Only: Worker ID number assigned to the new process.

                **worker_cmd** [list]:
                    TCP Only: Split string corresponding to worker/client Python process invocation. Contains
                    a local Python path, calling script, and manager/server format-fields for ``manager_ip``,
                    ``manager_port``, ``authkey``, and ``workerID``. ``nworkers`` is specified normally.

        .. tab-item:: History

                **save_every_k_sims** [int]:
                    Save history array to file after every k simulated points.

                **save_every_k_gens** [int]:
                    Save history array to file after every k generated points.

                **save_H_and_persis_on_abort** [bool] = ``True``:
                    Save states of ``H`` and ``persis_info`` to file on aborting after an exception.

                **save_H_on_completion** bool | None = ``False``
                    Save state of ``H`` to file upon completing a workflow. Also enabled when either ``save_every_k_sims``
                    or ``save_every_k_gens`` is set.

                **save_H_with_date** bool | None = ``False``
                    Save ``H`` filename contains date and timestamp.

                **H_file_prefix** str | None = ``"libE_history"``
                    Prefix for ``H`` filename.

                **use_persis_return_gen** [bool] = ``False``:
                    Adds persistent generator output fields to the History array on return.

                **use_persis_return_sim** [bool] = ``False``:
                    Adds persistent simulator output fields to the History array on return.

                **final_gen_send** [bool] = ``False``:
                    Send final simulation results to persistent generators before shutdown.
                    The results will be sent along with the ``PERSIS_STOP`` tag.

        .. tab-item:: Resources

                **disable_resource_manager** [bool] = ``False``:
                    Disable the built-in resource manager, including automatic resource detection
                    and/or assignment of resources to workers. ``"resource_info"`` will be ignored.

                **platform** [str]:
                    Name of a :ref:`known platform<known-platforms>`, e.g., ``LibeSpecs.platform = "perlmutter_g"``
                    Alternatively set the ``LIBE_PLATFORM`` environment variable.

                **platform_specs** [Platform|dict]:
                    A ``Platform`` object (or dictionary) specifying :ref:`settings for a platform.<platform-fields>`.
                    Fields not provided will be auto-detected. Can be set to a :ref:`known platform object<known-platforms>`.

                **num_resource_sets** [int]:
                    The total number of resource sets into which resources will be divided.
                    By default resources will be divided by workers (excluding
                    ``zero_resource_workers``).

                **gen_num_procs** [int] = ``0``:
                    The default number of processors (MPI ranks) required by generators. Unless
                    overridden by equivalent ``persis_info`` settings, generators will be allocated
                    this many processors for applications launched via the MPIExecutor.

                **gen_num_gpus** [int] = ``0``:
                    The default number of GPUs required by generators. Unless overridden by
                    the equivalent ``persis_info`` settings, generators will be allocated this
                    many GPUs.

                **gpus_per_group** [int]:
                    Number of GPUs for each group in the scheduler. This can be used when
                    running on nodes with different numbers of GPUs. In effect a
                    block of this many GPUs will be treated as a virtual node.
                    By default the GPUs on each node are treated as a group.

                **use_tiles_as_gpus** [bool] = ``False``:
                    If ``True`` then treat a GPU tile as one GPU, assuming
                    ``tiles_per_GPU`` is provided in ``platform_specs`` or detected.

                **enforce_worker_core_bounds** [bool] = ``False``:
                    Permit submission of tasks with a
                    higher processor count than the CPUs available to the worker.
                    Larger node counts are not allowed. Ignored when
                    ``disable_resource_manager`` is set.

                **dedicated_mode** [bool] = ``False``:
                    Instructs libEnsemble’s MPI executor not to run applications on nodes where
                    libEnsemble processes (manager and workers) are running.

                **zero_resource_workers** [list of ints]:
                    List of workers (by IDs) that require no resources. For when a fixed mapping of workers
                    to resources is required. Otherwise, use ``num_resource_sets``.
                    For use with supported allocation functions.

                **resource_info** [dict]:
                    Provide resource information that will override automatically detected resources.
                    The allowable fields are given below in "Overriding Resource Auto-Detection"
                    Ignored if ``disable_resource_manager`` is set.

                **scheduler_opts** [dict]:
                    Options for the resource scheduler.
                    See "Scheduler Options" for more options.

.. dropdown:: Complete Class API

    .. autopydantic_model:: libensemble.specs.LibeSpecs
        :model-show-json: False
        :model-show-config-member: False
        :model-show-config-summary: False
        :model-show-validator-members: False
        :model-show-validator-summary: False
        :field-list-validators: False
        :model-show-field-summary: False

Scheduler Options
-----------------

See options for :ref:`built-in scheduler<resources-scheduler>`.

.. _resource_info:

Overriding Resource Auto-Detection
----------------------------------

Note that ``"cores_on_node"`` and ``"gpus_on_node"`` are supported for backward
compatibility, but use of :ref:`Platform specification<datastruct-platform-specs>` is
recommended for these settings.

.. dropdown:: Resource Info Fields

    The allowable ``libE_specs["resource_info"]`` fields are::

        "cores_on_node" [tuple (int, int)]:
            Tuple (physical cores, logical cores) on nodes.

        "gpus_on_node" [int]:
            Number of GPUs on each node.

        "node_file" [str]:
            Name of file containing a node-list. Default is "node_list".

        "nodelist_env_slurm" [str]:
            The environment variable giving a node list in Slurm format
            (Default: Uses ``SLURM_NODELIST``).  Queried only if
            a ``node_list`` file is not provided and the resource manager is
            enabled.

        "nodelist_env_cobalt" [str]:
            The environment variable giving a node list in Cobalt format
            (Default: Uses ``COBALT_PARTNAME``) Queried only
            if a ``node_list`` file is not provided and the resource manager
            is enabled.

        "nodelist_env_lsf" [str]:
            The environment variable giving a node list in LSF format
            (Default: Uses ``LSB_HOSTS``) Queried only
            if a ``node_list`` file is not provided and the resource manager
            is enabled.

        "nodelist_env_lsf_shortform" [str]:
            The environment variable giving a node list in LSF short-form
            format (Default: Uses ``LSB_MCPU_HOSTS``) Queried only
            if a ``node_list`` file is not provided and the resource manager is
            enabled.

    For example::

        customizer = {cores_on_node": (16, 64),
                    "node_file": "libe_nodes"}

        libE_specs["resource_info"] = customizer

Formatting Options for libE_stats File
--------------------------------------

The allowable ``libE_specs["stats_fmt"]`` fields are::

    "task_timing" [bool] = ``False``:
        Outputs elapsed time for each task launched by the executor.

    "task_datetime" [bool] = ``False``:
        Outputs the elapsed time and start and end time for each task launched by the executor.
        Can be used with the ``"plot_libe_tasks_util_v_time.py"`` to give task utilization plots.

    "show_resource_sets" [bool] = ``False``:
        Shows the resource set IDs assigned to each worker for each call of the user function.
