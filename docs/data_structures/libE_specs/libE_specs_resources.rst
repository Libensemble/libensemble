Resources
=========

`Introduction <libE_specs.html>`__ \|\| `General <libE_specs_general.html>`__ \|\| `Directories <libE_specs_directories.html>`__ \|\| `Profiling <libE_specs_profiling.html>`__ \|\| `TCP <libE_specs_tcp.html>`__ \|\| `History <libE_specs_history.html>`__ \|\| **Resources**

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
    If ``True`` then treat a GPU tile as one GPU when GPU tiles
    are provided in ``platform_specs`` or auto-detected.

**enforce_worker_core_bounds** [bool] = ``False``:
    Permit submission of tasks with a
    higher processor count than the CPUs available to the worker.
    Larger node counts are not allowed. Ignored when
    ``disable_resource_manager`` is set.

**dedicated_mode** [bool] = ``False``:
    Instructs libEnsemble’s MPI executor not to run applications on nodes where
    libEnsemble processes (manager and workers) are running.

**resource_info** [dict]:
    Provide resource information that will override automatically detected resources.
    The allowable fields are given below in "Overriding Resource Auto-Detection"
    Ignored if ``disable_resource_manager`` is set.

**scheduler_opts** [dict]:
    Options for the resource scheduler.
    See "Scheduler Options" for more options.
