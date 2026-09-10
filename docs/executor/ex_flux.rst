Flux Executor
==============

`Overview <ex_overview.html>`__ \|\| `Base Executor <ex_base.html>`__ \|\| `MPI Executor <ex_mpi.html>`__ \|\| **Flux Executor**

.. automodule:: flux_executor
    :no-undoc-members:

.. note::

    The ``FluxExecutor`` requires the ``flux-core`` Python bindings to be
    installed and importable (e.g., via ``conda install -c conda-forge flux-core``
    or Spack), and must either be run inside a Flux instance or be given a
    Flux URI to connect to. If these bindings aren't available, ``FluxExecutor``
    won't be importable, but the standard :doc:`MPI Executor <ex_mpi>` can
    still submit to Flux via ``mpi_runner="flux"`` (see below).

.. tab-set::

    .. tab-item:: Flux Executor

        .. autoclass:: libensemble.executors.flux_executor.FluxExecutor
            :members:
            :show-inheritance:
            :exclude-members: serial_setup, sim_default_app, gen_default_app, get_app, default_app, set_resources, get_task, set_workerID, set_worker_info, new_tasks_timing, add_platform_info, set_gen_procs_gpus

            .. automethod:: __init__

    .. tab-item:: Flux Task

        Like the base :ref:`Task <task_tag>`, ``FluxTask`` objects are created and
        returned by ``FluxExecutor.submit()``. ``poll()``, ``kill()``, and ``wait()``
        are overridden to query and control the job via Flux's own job-lifecycle
        API instead of subprocess/signal-based mechanisms.

        .. autoclass:: libensemble.executors.flux_executor.FluxTask
            :members:
            :show-inheritance:
            :exclude-members: reset

Two ways to run under Flux
---------------------------

libEnsemble offers two independent ways to run applications under Flux:

1. **MPI Executor with the Flux runner.** The standard :doc:`MPI Executor <ex_mpi>`
   can subprocess ``flux run`` like any other MPI launcher. This requires no
   Python bindings, works with the usual resource-manager and GPU
   auto-detection, and is a good default choice::

        from libensemble.executors import MPIExecutor

        exctr = MPIExecutor(custom_info={"mpi_runner": "flux"})

   or, equivalently, by selecting the built-in ``"flux"`` :ref:`platform<datastruct-platform-specs>`::

        libE_specs["platform"] = "flux"

2. **Flux Executor.** ``FluxExecutor`` instead submits jobs directly to Flux
   through its Python API (``flux.job.submit_async``), bypassing any launcher
   subprocess entirely. This is particularly useful inside containers or other
   environments where a standard MPI launcher isn't available, and gives more
   direct access to Flux's own job states. Since Flux handles aren't
   thread-safe, ``FluxExecutor`` is best suited to process-based libEnsemble
   runs (e.g., multiprocessing or MPI workers) rather than threaded workers.

**Basic usage**

.. code-block:: python

    from libensemble import Ensemble
    from libensemble.executors.flux_executor import FluxExecutor

    exctr = FluxExecutor()
    exctr.register_app(full_path="/home/user/forces.x", app_name="forces")
    ensemble = Ensemble(executor=exctr)

**In user simulation function**::

    def sim_func(H, persis_info, sim_specs, libE_info):
        exctr = libE_info["executor"]

        task = exctr.submit(
            app_name="forces",
            num_procs=8,
            num_nodes=2,
            stdout="out.txt",
            stderr="err.txt",
        )

        # Wait for task to complete
        task.wait()

GPUs can be requested with ``num_gpus``, and jobs can be submitted without
blocking on start-up via ``wait_on_start`` (an integer gives a timeout in
seconds). See the :doc:`Forces tutorial <../tutorials/executor_forces_tutorial>`
for a complete calling script and simulation function using the ``forces.x``
application; substituting ``FluxExecutor`` for ``MPIExecutor`` there requires
no other changes.
