.. _zero_resource_workers:

Zero-resource workers
~~~~~~~~~~~~~~~~~~~~~

Users with persistent ``gen_f`` functions may notice that the persistent workers
are still automatically assigned system resources. This can be wasteful if those
workers only run ``gen_f`` routines in-place and don't use the Executor to submit
applications to allocated nodes:

    .. image:: ../images/persis_wasted_node.png
        :alt: persis_wasted_node
        :scale: 40
        :align: center

This can be resolved by using the libE_specs option ``zero_resource_workers``:

.. code-block:: python

    libE_specs['zero_resource_workers'] = [1]

in the calling script. Set the parameter ``zero_resource_workers`` to a list of
worker IDs that should not have system resources assigned.

Worker 1 will not be allocated resources. Note that additional worker
processes can be added to take advantage of the free resources (if using the
same resource set) for simulation instances:

    .. image:: ../images/persis_add_worker.png
        :alt: persis_add_worker
        :scale: 40
        :align: center

An alternative, when resource sets are being used, it to set the ``num_resource_sets``
libE_specs option explicitly to the required value. The difference with declaring
``zero_resource_workers`` is that a fixed worker will have zero resources (this must
be supported by the allocation function, see :ref:`start_only_persistent<start_only_persistent_label>`)
