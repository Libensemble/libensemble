Globus Compute Executor
=======================

`Overview <ex_overview.html>`__ || `Base Executor <ex_base.html>`__ || `MPI Executor <ex_mpi.html>`__ || **Globus Compute Executor**

The :class:`GlobusComputeExecutor<libensemble.executors.globus_compute_executor.GlobusComputeExecutor>`
submits Python callables to a remote `Globus Compute`_ endpoint instead of
launching local subprocesses. It can be used inside simulator functions in the
same way as the :doc:`MPI Executor<ex_mpi>`, retrieving it from
``libE_info["executor"]``.

See :ref:`Globus Compute - Remote User Functions<globus_compute_ref>` for an
overview of the two GC integration modes (manager-side GC-only and user-facing
executor).

.. note::

    ``globus-compute-sdk`` must be installed to use this executor::

        pip install globus-compute-sdk

    Users must also authenticate via Globus_ and have an active
    `Globus Compute endpoint`_ running on the target system.

GlobusComputeExecutor
---------------------

.. autoclass:: libensemble.executors.globus_compute_executor.GlobusComputeExecutor
    :members: register_app, submit, set_workerID, set_worker_info
    :show-inheritance:

    .. automethod:: __init__

GlobusComputeTask
-----------------

Tasks are created and returned by
:meth:`GlobusComputeExecutor.submit()<libensemble.executors.globus_compute_executor.GlobusComputeExecutor.submit>`.
Each task wraps a ``concurrent.futures.Future`` from the Globus Compute SDK
and exposes the same polling interface as other libEnsemble tasks.

.. autoclass:: libensemble.executors.globus_compute_executor.GlobusComputeTask
    :members: poll, wait, kill, result, running, done, cancelled

**Task states**: ``RUNNING`` | ``FINISHED`` | ``FAILED`` | ``USER_KILLED``

**Key attributes**:

:task.state: (string) Current task state - one of the values above.
:task.finished: (bool) True once the task has completed (successfully or not).
:task.success: (bool) True if the remote callable returned without raising.
:task.runtime: (float) Elapsed wall-clock seconds since submission.
:task.submit_time: (float) Time since epoch at submission.

.. _Globus Compute: https://www.globus.org/compute
.. _Globus: https://www.globus.org/
.. _Globus Compute endpoint: https://globus-compute.readthedocs.io/en/latest/endpoints.html
