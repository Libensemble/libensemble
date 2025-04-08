.. _zero_resource_workers:

Zero-resource workers
~~~~~~~~~~~~~~~~~~~~~

Users with persistent ``gen_f`` functions may notice that the persistent workers
are still automatically assigned resources. This can be wasteful if those workers
only run ``gen_f`` functions in-place (i.e., they do not use the Executor
to submit applications to allocated nodes). Suppose the user is using the
:meth:`parse_args()<tools.parse_args>` function and runs::

    python run_ensemble_persistent_gen.py --nworkers 3

If three nodes are available in the node allocation, the result may look like the
following.

    .. image:: ../images/persis_wasted_node.png
        :alt: persis_wasted_node
        :scale: 40
        :align: center

To avoid the the wasted node above, add an extra worker::

    python run_ensemble_persistent_gen.py --nworkers 4

and in the calling script (*run_ensemble_persistent_gen.py*), explicitly set the
number of resource sets to the number of workers that will be running simulations.

.. code-block:: python

    nworkers, is_manager, libE_specs, _ = parse_args()
    libE_specs["num_resource_sets"] = nworkers - 1

When the ``num_resource_sets`` option is used, libEnsemble will use the dynamic
resource scheduler, and any worker may assign work to any node. This works well
for most users.

    .. image:: ../images/persis_add_worker.png
        :alt: persis_add_worker
        :scale: 40
        :align: center

**Optional**: An alternative way to express the above would be to use the command
line::

    python run_ensemble_persistent_gen.py --comms local --nsim_workers 3

This would automatically set the ``num_resource_sets`` option and add a single
worker for the persistent generator - a common use-case.

In general, the number of resource sets should be set to enable the maximum
concurrency desired by the ensemble, taking into account generators and simulators.

Users can set generator resources using the *libE_specs* options
``gen_num_procs`` and/or ``gen_num_gpus``, which take integer values.
If only ``gen_num_gpus`` is set, then the number of processors is set to match.

To vary generator resources, ``persis_info`` settings can be used in allocation
functions before calling the ``gen_work`` support function. This takes the
same options (``gen_num_procs`` and ``gen_num_gpus``).

Alternatively, the setting ``persis_info["gen_resources"]`` can also be set to
a number of resource sets.

The available nodes are always divided by the number of resource sets, and there
may be multiple nodes or a partition of a node in each resource set. If the split
is uneven, resource sets are not split between nodes. For example, if there are
two nodes and five resource sets, one node will have three resource sets, and
the other will have two.

Placing zero-resource functions on a fixed worker
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If the generator must always be on worker one, then instead of using
``num_resource_sets``, use the ``zero_resource_workers`` *libE_specs* option:

.. code-block:: python

    libE_specs["zero_resource_workers"] = [1]

in the calling script and worker one will not be allocated resources. In general,
set the parameter ``zero_resource_workers`` to a list of worker IDs that should not
have resources assigned.

This approach can be useful if running in
:doc:`distributed mode<../platforms/platforms_index>`.

The use of the ``zero_resource_workers`` *libE_specs* option must be supported by
the allocation function, see :ref:`start_only_persistent<start_only_persistent_label>`)
