.. _globus_compute_ref:

======================================
Globus Compute - Remote User Functions
======================================

`Globus Compute`_ (formerly funcX) is a distributed, high-performance
function-as-a-service platform. When libEnsemble is running on a resource with
internet access (laptops, login nodes, other servers, etc.), it can offload
simulator calls to remote Globus Compute endpoints:

    .. image:: ../images/funcxmodel.png
        :alt: running_with_globus_compute
        :scale: 50
        :align: center

This is useful for running ensembles across machines and heterogeneous resources.
There are **two approaches**, described below.

.. dropdown:: **Caveats**

    The following caveats apply to all Globus Compute modes:

        1. Simulator functions submitted to Globus Compute must be non-persistent,
        since manager-worker communicators cannot be serialized or used by a
        remote resource.

        2. ``Executor.manager_poll()`` is not available inside remotely executed
        functions. Control over remote work is limited to inspecting return
        values and exceptions when tasks complete.

        3. Globus Compute imposes a `handful of task-rate and data limits`_ on
        submitted functions.

        4. Users are responsible for authenticating via Globus_ and maintaining their
        `Globus Compute endpoints`_ on their target systems.

.. _gc_only_mode:

Manager-side GC (GC-only mode)
-------------------------------

The recommended approach for most use cases. When
``globus_compute_endpoint`` is set in :class:`SimSpecs<libensemble.specs.SimSpecs>`
and ``gen_on_worker`` is not set (the default), libEnsemble enters
**GC-only mode**: no local worker processes are launched. The manager
submits simulation work directly to Globus Compute and polls futures for
results. The generator still runs as a local thread on the manager.

``nworkers`` controls the maximum number of simultaneously in-flight
Globus Compute tasks (virtual concurrency). The default is 1.

This mode supports both the :ref:`gest-api simulator format<datastruct-sim-specs>`
(``SimSpecs.simulator``) and the legacy ``sim_f`` format.

.. code-block:: python

    from libensemble import Ensemble
    from libensemble.specs import ExitCriteria, GenSpecs, LibeSpecs, SimSpecs


    def my_sim(input_dict: dict, **kwargs) -> dict:
        """gest-api simulator - runs remotely on the GC endpoint."""
        return {"f": input_dict["x"] ** 2}


    sim_specs = SimSpecs(
        simulator=my_sim,
        vocs=vocs,
        globus_compute_endpoint="3af6dc24-3f27-4c49-8d11-e301ade15353",
    )

    libE_specs = LibeSpecs(nworkers=4)  # up to 4 concurrent GC tasks

    workflow = Ensemble(
        sim_specs=sim_specs,
        gen_specs=gen_specs,
        libE_specs=libE_specs,
        exit_criteria=ExitCriteria(sim_max=20),
    )
    H, _, _ = workflow.run()

Users can also define ``Executor`` instances within their remote simulator
functions and submit MPI applications normally, as long as libEnsemble and
the target application are accessible on the remote system::

    # Within the remote simulator function
    from libensemble.executors import MPIExecutor
    exctr = MPIExecutor()
    exctr.register_app(full_path="/home/user/forces.x", app_name="forces")
    task = exctr.submit(app_name="forces", num_procs=64)

.. note::

    Both the simulator callable and any VOCS object must be picklable,
    as they are serialized and shipped to the remote Globus Compute endpoint.

.. _gc_executor_approach:

GlobusComputeExecutor (user-facing)
------------------------------------

For workflows where the simulation function itself orchestrates remote
calls, like fanning out to multiple endpoints or mixing local
and remote work. Use the
:class:`GlobusComputeExecutor<libensemble.executors.globus_compute_executor.GlobusComputeExecutor>`
directly inside the simulator.

Create and register the executor in the top-level script:

.. code-block:: python

    from libensemble.executors import GlobusComputeExecutor

    exctr = GlobusComputeExecutor(endpoint_id="3af6dc24-3f27-4c49-8d11-e301ade15353")

Then use it inside the simulator function:

.. code-block:: python

    import time


    def my_sim(H, persis_info, sim_specs, libE_info):
        exctr = libE_info["executor"]

        task = exctr.submit(func=my_remote_func, app_args=H["x"][0])

        while not task.finished:
            task.poll()
            if exctr.manager_kill_received():
                task.kill()
                break
            time.sleep(0.1)

        return H_o, persis_info

See the :doc:`GlobusComputeExecutor API reference<../executor/ex_globus_compute>` for
the full interface including ``register_app``, ``submit``, and
:class:`GlobusComputeTask<libensemble.executors.globus_compute_executor.GlobusComputeTask>` methods.

.. _Globus Compute: https://www.globus.org/compute
.. _Globus Compute endpoints: https://globus-compute.readthedocs.io/en/latest/endpoints.html
.. _Globus: https://www.globus.org/
.. _handful of task-rate and data limits: https://globus-compute.readthedocs.io/en/latest/limits.html
