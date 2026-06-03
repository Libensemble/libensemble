.. _datastruct-libe-specs:

**Introduction** \|\| `General <libE_specs_general.html>`__ \|\| `Directories <libE_specs_directories.html>`__ \|\| `Profiling <libE_specs_profiling.html>`__ \|\| `TCP <libE_specs_tcp.html>`__ \|\| `History <libE_specs_history.html>`__ \|\| `Resources <libE_specs_resources.html>`__

LibE Specs
==========

libEnsemble is primarily customized by setting options within a ``LibeSpecs`` instance.

.. code-block:: python

    from libensemble.specs import LibeSpecs

    specs = LibeSpecs(save_every_k_gens=100, sim_dirs_make=True, nworkers=4)

.. toctree::
    :hidden:

    libE_specs_general
    libE_specs_directories
    libE_specs_profiling
    libE_specs_tcp
    libE_specs_history
    libE_specs_resources

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
