Dynamic Assignment of Resources
===============================

Overview
--------

libEnsemble comes with built-in resource management. This entails the
:ref:`detection of available resources<resource_detection>` (e.g., nodelists,
core counts, and GPUs), and the allocation of resources to workers.

By default, the available resources are divided evenly among workers.
libEnsemble's :doc:`MPI Executor<../executor/ex_index>` is aware of the
resources assigned to each worker, and if not given any of ``num_nodes``,
``num_procs``, or ``procs_per_node`` in the submit call, it will use all nodes
and CPU cores available to that worker.

Detected resources can be overridden using the ``libE_specs`` option
:ref:`resource_info<resource_info>`.

Variable resource assignment
----------------------------

Simulations often need different amounts of resources. The generator can
request processors and GPUs for each simulation by including ``num_procs``
and/or ``num_gpus`` in its output fields:

.. code-block:: python
    :emphasize-lines: 6-7

    gen_specs = {
        "gen_f": gen_f,
        "in": ["sim_id"],
        "out": [
            ("priority", float),
            ("num_procs", int),
            ("num_gpus", int),
            ("x", float, n),
        ],
    }

The generator then sets these values for every point it produces:

.. code-block:: python
    :emphasize-lines: 4-5

    H_o = np.zeros(b, dtype=gen_specs["out"])
    for i in range(b):
        H_o["x"][i] = x[i]
        H_o["num_procs"][i] = procs[i]
        H_o["num_gpus"][i] = gpus[i]

Values are initialized to zero, so set each field for every point. If
``num_procs`` is included but left at zero, that simulation is given no
resources.

The allocation function checks whether the requested resources are free
before giving a simulation to a worker. If not, the simulation waits until
enough resources become available. This is built into the supplied allocation
functions and generally requires no modification.

The nodes assigned to each simulation are chosen by the libEnsemble
:doc:`built-in scheduler<scheduler_module>`, although an alternative scheduler
can be provided via the :doc:`allocation function<../function_guides/allocator>`.
The scheduler prefers to fit each simulation onto a single node and uses even
splits across nodes when more than one node is needed.

The requested values are passed to the MPI Executor automatically, so they do
not need to be included in ``sim_specs["in"]`` or the submit call. GPUs are
assigned in the correct way for the given system (including Nvidia, AMD, and
Intel GPUs); you do not need to set ``CUDA_VISIBLE_DEVICES`` or equivalent.
See the :ref:`forces_gpu tutorial<var-resources-gpu>` and
`test_GPU_variable_resources.py`_ for complete examples.

Accessing resources from the simulation function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Most simulation functions do not need to inspect their resources; submitting
through the executor is sufficient. For custom launch logic, the resources
supplied to the worker can be
:doc:`interrogated directly via the resources class attribute<worker_resources>`:

.. code-block:: python

    from libensemble.resources.resources import Resources

    resources = Resources.resources.worker_resources
    num_nodes = resources.local_node_count
    slots_per_node = resources.slot_count
    resources.set_env_to_slots("CUDA_VISIBLE_DEVICES")

Slots are the partitions of a node assigned to this worker, numbered from
zero. ``set_env_to_slots`` sets the given environment variable to the assigned
slot numbers (``HIP_VISIBLE_DEVICES`` or ``ROCR_VISIBLE_DEVICES`` can be used on
AMD systems).

`var_resources.py`_ has further examples (``multi_points_with_variable_resources``
and ``CUDA_variable_resources``).

Varying generator resources
^^^^^^^^^^^^^^^^^^^^^^^^^^^

By default, generators are not allocated resources. Fixed resources for the
generator can be set using the ``libE_specs`` options ``gen_num_procs`` and
``gen_num_gpus``. If only ``gen_num_gpus`` is set, the number of processors
will be set to match.

To vary generator resources, set ``gen_num_procs`` and ``gen_num_gpus`` in
``persis_info`` in the allocation function before calling the ``gen_work``
support function. Persistent workers keep their resources until they leave
the persistent state.

.. _legacy-resource-sets:

Legacy: resource sets
-----------------------

Internally, the resource manager divides resources into **resource sets**,
the smallest unit that can be assigned (and dynamically reassigned) to
workers. By default, there is one resource set per worker. A ``num_procs`` or
``num_gpus`` request is rounded up to a whole number of resource sets.

.. _rsets-diagram:

Each resource set is either part of one node or one or more whole nodes. When
there are more resource sets than nodes, each node holds an equal share of them
(any remainder goes to the first nodes), numbered by **slot** from zero. When
there are fewer resource sets than nodes, each resource set covers one or more
whole nodes, and is slot zero on each.

The resource set IDs given to each worker can be shown in ``libE_stats.txt`` by
setting ``libE_specs["stats_fmt"] = {"show_resource_sets": True}``.

Setting the number of resource sets
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``num_resource_sets`` :class:`libE_specs<libensemble.specs.LibeSpecs>`
option sets the number of resource sets directly. If set, dynamic resource
assignment is always used. Example cases:

* **A worker runs the generator** (``gen_on_worker=True``) and simulations do
  not request resources. That worker would otherwise hold an equal share. Set
  ``num_resource_sets`` to the number of simulation workers.

* **Finer granularity**. Requests are rounded up to whole resource sets, so
  with few workers each set is large. Setting more resource sets than workers
  allows requests to be matched more closely and the spare resources to be used
  by larger simulations. For example, ``--nworkers 5 --nresource_sets 8`` on the
  command line (see :meth:`parse_args<tools.parse_args>`).

The generator's resources may also be given as a number of resource sets with
``persis_info["gen_resources"]``. When resource sets are larger than one GPU,
the ``multiplier`` argument of ``set_env_to_slots`` maps slots to devices.

Resource scheduler options
^^^^^^^^^^^^^^^^^^^^^^^^^^

The following options are available for the :doc:`built-in scheduler<scheduler_module>`
and can be set via ``libE_specs["scheduler_opts"]``:

 **split2fit** [boolean]
    Try to split resource sets across more nodes if space is not currently
    available on the minimum node count required. Allows more efficient
    scheduling.
    Default: True

 **match_slots** [boolean]:
    When splitting resource sets across multiple nodes, slot IDs must match.
    Useful if setting an environment variable such as ``CUDA_VISIBLE_DEVICES``
    to specific slot numbers, which should match over multiple nodes.
    Default: True

For example, if the next simulation requires four resource sets, and two
nodes each have two free slots but with different slot numbers, ``split2fit``
allows two to be used on each node. However, the simulation will not be
scheduled unless ``match_slots`` is ``False``:

.. code-block:: python

    libE_specs["scheduler_opts"] = {"match_slots": False}

This is only recommended if not mapping slots to devices (e.g., via
``CUDA_VISIBLE_DEVICES``). If six resource sets were requested on nodes
holding four each, they would be split three per node even if ``split2fit``
is ``False``, as this could otherwise never be scheduled.

.. _test_GPU_variable_resources.py: https://github.com/Libensemble/libensemble/blob/develop/libensemble/tests/regression_tests/test_GPU_variable_resources.py
.. _var_resources.py: https://github.com/Libensemble/libensemble/blob/develop/libensemble/sim_funcs/var_resources.py
