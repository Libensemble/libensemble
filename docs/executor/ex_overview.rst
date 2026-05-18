Overview
========

**Overview** \|\| `Base Executor <ex_base.html>`__ \|\| `MPI Executor <ex_mpi.html>`__

The **Executor** provides a portable interface for running applications on any system and
any number of compute resources.

.. dropdown:: Detailed description

    An **Executor** interface is provided by libEnsemble to remove the burden
    of system interaction from the user and improve workflow portability. Users
    first register their applications to Executor instances, which then return
    corresponding ``Task`` objects upon submission within user functions.

    **Task** attributes and retrieval functions can be queried to determine
    the status of running application instances. Functions are also provided
    to access and interrogate files in the task's working directory.

    libEnsemble's Executors and Tasks contain many familiar features and methods
    to Python's native `concurrent futures`_ interface. Executors feature the
    ``submit()`` function for launching apps (detailed below),  but currently do
    not support ``map()`` or ``shutdown()``. Tasks are much like ``futures``.
    They feature the ``cancel()``, ``cancelled()``, ``running()``, ``done()``,
    ``result()``, and ``exception()`` functions from the standard.

    The main ``Executor`` class can subprocess serial applications in place,
    while the ``MPIExecutor`` is used for running MPI applications.

    Typically, users choose and parameterize their ``Executor`` objects in their
    calling scripts, where each executable generator or simulation application is
    registered to it. Once in the user-side worker code (sim/gen func), the Executor
    can be retrieved without any need to specify the type.

    Once the Executor is retrieved, tasks can be submitted by specifying the
    ``app_name`` from registration in the calling script alongside other optional
    parameters described in the API.

**Basic usage**

To set up an MPI executor, register an MPI application, and add
to the ensemble object.

.. code-block:: python

    from libensemble import Ensemble
    from libensemble.executors import MPIExecutor

    exctr = MPIExecutor()
    exctr.register_app(full_path="/path/to/my/exe", app_name="sim1")
    ensemble = Ensemble(executor=exctr)

**In user simulation function**::

    def sim_func(H, persis_info, sim_specs, libE_info):

        input_param = str(int(H["x"][0][0]))
        exctr = libE_info["executor"]

        task = exctr.submit(
            app_name="sim1",
            num_procs=8,
            app_args=input_param,
            stdout="out.txt",
            stderr="err.txt",
        )

        # Wait for task to complete
        task.wait()

Example use-cases:

* :doc:`Electrostatic Forces example <../tutorials/executor_forces_tutorial>`: Launches the ``forces.x`` MPI application.

* :doc:`Forces example with GPUs <../tutorials/forces_gpu_tutorial>`: Auto-assigns GPUs via executor.

See :doc:`Running on HPC Systems<../platforms/platforms_index>` for illustrations
of how common options such as ``libE_specs["dedicated_mode"]`` affect the
run configuration on clusters and supercomputers.

**Advanced Features**

**Example of polling output and killing application:**

In simulation function (sim_f).

.. code-block:: python

    import time


    def sim_func(H, persis_info, sim_specs, libE_info):
        input_param = str(int(H["x"][0][0]))
        exctr = libE_info["executor"]

        task = exctr.submit(
            app_name="sim1",
            num_procs=8,
            app_args=input_param,
            stdout="out.txt",
            stderr="err.txt",
        )

        timeout_sec = 600
        poll_delay_sec = 1

        while not task.finished:
            # Has manager sent a finish signal
            if exctr.manager_kill_received():
                task.kill()
                my_cleanup()

            # Check output file for error and kill task
            elif task.stdout_exists():
                if "Error" in task.read_stdout():
                    task.kill()

            elif task.runtime > timeout_sec:
                task.kill()  # Timeout

            else:
                time.sleep(poll_delay_sec)
                task.poll()

        print(task.state)  # state may be finished/failed/killed

Users who wish to poll only for manager kill signals and timeouts don't necessarily
need to construct a polling loop like above, but can instead use the ``Executor``
built-in ``polling_loop()`` method. An alternative to the above simulation function
may resemble:

.. code-block:: python

    def sim_func(H, persis_info, sim_specs, libE_info):
        input_param = str(int(H["x"][0][0]))
        exctr = libE_info["executor"]

        task = exctr.submit(
            app_name="sim1",
            num_procs=8,
            app_args=input_param,
            stdout="out.txt",
            stderr="err.txt",
        )

        timeout_sec = 600
        poll_delay_sec = 1

        exctr.polling_loop(task, timeout=timeout_sec, delay=poll_delay_sec)

        print(task.state)  # state may be finished/failed/killed

The ``MPIExecutor`` autodetects system criteria such as the appropriate MPI launcher
and mechanisms to poll and kill tasks. It also has access to the resource manager,
which partitions resources among workers, ensuring that runs utilize different
resources (e.g., nodes). Furthermore, the ``MPIExecutor`` offers resilience via the
feature of re-launching tasks that fail to start because of system factors.

Remote Execution with Globus Compute
-------------------------------------

The :doc:`GlobusComputeExecutor<ex_globus_compute>` submits Python callables
to remote `Globus Compute`_ endpoints instead of launching local subprocesses.
It exposes the same ``submit()`` / ``poll()`` / ``kill()`` interface as other
libEnsemble executors and can be retrieved from ``libE_info["executor"]``
inside simulator functions.

See :ref:`Globus Compute - Remote User Functions<globus_compute_ref>` for an
overview of all three GC integration modes and the
:doc:`GlobusComputeExecutor API reference<ex_globus_compute>` for the full
interface.

.. _concurrent futures: https://docs.python.org/library/concurrent.futures.html
.. _Globus Compute: https://www.globus.org/compute
