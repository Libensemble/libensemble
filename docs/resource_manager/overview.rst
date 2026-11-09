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

 **gpu_packing** [str]:
    Where to place tasks that use part of a GPU when
    :ref:`GPU sharing<gpu-sharing>` is enabled. ``"pack"`` fills partly used GPUs
    and nodes first, keeping whole GPUs free for larger tasks. ``"spread"``
    prefers the least used GPUs and nodes, giving tasks more room.
    Default: ``"pack"``

.. _gpu-sharing:

Sharing GPUs between tasks
--------------------------

Some simulations need only part of a GPU. ``gpu_sharing_factor`` lets several such
tasks run on the same GPU at once, so a node's GPUs can serve more concurrent
simulations.

Set the number of tasks that may share each GPU, and have the generator request
fractional GPUs through the ``num_gpus`` field:

.. code-block:: python
    :emphasize-lines: 2, 5

    ensemble.libE_specs = LibeSpecs(
        gpu_sharing_factor=2,  # Up to two tasks per GPU
    )

    ensemble.gen_specs = GenSpecs(outputs=[("x", float, 2), ("num_gpus", float)], ...)

with the generator setting, for example:

.. code-block:: python

    H_o["num_gpus"] = [0.5, 1.0, 1.5, 2.0]

GPUs are assigned and ``CUDA_VISIBLE_DEVICES`` (or equivalent) is set automatically,
as with whole-GPU requests. There is no need to set ``num_resource_sets``.

What tasks can request
^^^^^^^^^^^^^^^^^^^^^^

The sharing factor sets the step size of GPU requests. Requests are rounded up to
the nearest step.

.. list-table::
    :header-rows: 1

    * - ``gpu_sharing_factor``
      - Smallest request
      - Example requests
    * - 1 (default)
      - 1 GPU
      - 1, 2, 4
    * - 2
      - 0.5 GPU
      - 0.5, 1, 1.5, 2
    * - 4
      - 0.25 GPU
      - 0.25, 0.5, 0.75, 1.25

Requests can mix whole and partial GPUs. Each task gets its whole GPUs to itself.
The fractional part of a request always comes from one GPU, never split across
several. For example, with ``gpu_sharing_factor=4``, a task requesting ``0.75`` GPUs
gets three quarters of one GPU, not shares of two or three different GPUs.

With ``gpu_sharing_factor=2``, a task requesting ``1.5`` GPUs gets one GPU to itself
and half of a second GPU, which it may share with one other task. It sees two GPUs.

Generators can also use a fraction of a GPU, via ``gen_num_gpus``:

.. code-block:: python

    ensemble.libE_specs = LibeSpecs(gpu_sharing_factor=4, gen_num_gpus=0.25)

Example: a node with four GPUs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

With ``gpu_sharing_factor=2`` on a node with four GPUs, any of these sets of tasks
can run at once (using the default ``"pack"`` placement, below):

* Eight tasks requesting ``0.5`` GPUs (two per GPU).
* Four tasks requesting ``0.5`` GPUs and two requesting ``1`` GPU.
* Two tasks requesting ``1.5`` GPUs and two requesting ``0.5`` GPUs.

Run with enough workers to cover the concurrency you want. For eight concurrent
half-GPU tasks per node, use eight workers per node.

Choosing a sharing factor
^^^^^^^^^^^^^^^^^^^^^^^^^

Tasks sharing a GPU are not isolated: they share its memory and compute. Choose a
factor so that this many tasks fit in one GPU's memory at once. For example, with
40 GB GPUs and tasks needing about 8 GB, a factor of up to ``4`` is safe.

If your platform provides tools such as NVIDIA MPS for better GPU sharing, they can
be enabled as usual alongside ``gpu_sharing_factor``.

Each task's share of CPU cores also gets smaller: libEnsemble divides each node's
cores between all GPU shares. For example, on a node with 32 cores and four GPUs,
a factor of ``2`` gives four cores per share.

Placing partial-GPU tasks
^^^^^^^^^^^^^^^^^^^^^^^^^

By default, tasks using part of a GPU are packed onto GPUs that are already partly
used, keeping whole GPUs free for larger tasks. To instead spread such tasks over
the least used GPUs and nodes:

.. code-block:: python

    ensemble.libE_specs = LibeSpecs(
        gpu_sharing_factor=2,
        scheduler_opts={"gpu_packing": "spread"},
    )

Use ``"pack"`` (the default) when tasks of different sizes are mixed. With
``"spread"``, four half-GPU tasks on a four-GPU node take one share of every GPU,
so a following task requesting a whole GPU must wait. Use ``"spread"`` when tasks
are all the same size, or when they compete for memory or bandwidth and should be
kept apart.

Resource sets and errors
^^^^^^^^^^^^^^^^^^^^^^^^

When ``gpu_sharing_factor`` is greater than 1 and ``num_resource_sets`` is not set,
libEnsemble uses one resource set per GPU share and logs a message such as::

    gpu_sharing_factor=2: num_resource_sets defaulting to 16 (2 node(s) x 4 GPUs x 2 shares)

If you do set ``num_resource_sets``, it must provide at least one resource set per
GPU share on each node (for example, at least ``16`` in the case above). Fewer
raises an error. More is allowed, with the extra resource sets having no GPUs.

An error is also raised if the factor would leave a resource set with no CPU cores.

libEnsemble warns at startup if ``gpu_sharing_factor`` is set but tasks do not
request GPUs, or if no GPUs are detected.

For an example, see `test_GPU_sharing_variable_resources.py`_.

.. _test_GPU_sharing_variable_resources.py: https://github.com/Libensemble/libensemble/blob/develop/libensemble/tests/regression_tests/test_GPU_sharing_variable_resources.py
.. _test_GPU_variable_resources.py: https://github.com/Libensemble/libensemble/blob/develop/libensemble/tests/regression_tests/test_GPU_variable_resources.py
.. _var_resources.py: https://github.com/Libensemble/libensemble/blob/develop/libensemble/sim_funcs/var_resources.py
