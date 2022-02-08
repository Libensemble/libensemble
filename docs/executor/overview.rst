Executor Overview
=================

A typical libEnsemble workflow will include launching tasks from a
:ref:`sim_f<api_sim_f>` (or :ref:`gen_f<api_gen_f>`) running on a worker. We use
"task" to represent an application submission by libEnsemble to the system,
which may be the compute nodes of a supercomputer, cluster, or other compute resource.

An **Executor** interface is provided by libEnsemble to remove the burden of
system interaction from the user and ease the writing of portable user scripts that
launch applications. The Executor provides the key functions: ``submit()``,
``poll()``, ``wait()``, and ``kill()``. Task attributes can be queried to determine
the status following each of these commands. Functions are also provided to access
and interrogate files in the task's working directory.

The main ``Executor`` class is an abstract class and is inherited by the ``MPIExecutor``,
for direct running of MPI applications. We also provide a ``BalsamMPIExecutor``,
which submits an MPI run request from a worker running on a compute node to a
Balsam service running on a launch node (suitable for systems that do not allow
running MPI applications directly from compute nodes).

In a calling script, an ``Executor`` object is created, and the executable
generator or simulation applications are registered to it for submission. If an
alternative Executor like Balsam is used, then the applications can be
registered as in the example below. Once in the user-side worker code (sim/gen func),
an MPI-based Executor can be retrieved without any need to specify the type.

Once the Executor is retrieved, tasks can be submitted by specifying the ``app_name``
from registration in the calling script alongside other optional parameters
described in the API. A corresponding ``Task`` object instance is returned. As
can be seen in the examples below, a variety of ``Executor`` and ``Task`` attributes
and methods can be queried to effectively manage currently running applications
within user functions.

**Example usage (code runnable with or without a Balsam 0.5.0 backend):**

In calling function::

    sim_app = '/path/to/my/exe'
    USE_BALSAM = False

    if USE_BALSAM:
        from libensemble.executors.balsam_executor import BalsamMPIExecutor
        exctr = BalsamMPIExecutor()
    else:
        from libensemble.executors.mpi_executor import MPIExecutor
        exctr = MPIExecutor()

    exctr.register_app(full_path=sim_app, app_name='sim1')

.. note::
    The *Executor* set up in the calling script is stored as a class attribute and
    does **not** have to be passed to *libE*. It is extracted via *Executor.executor*
    in the sim function (regardless of type).

In user simulation function::

    import time
    from libensemble.executors.executor import Executor

    # Will return Executor (whether MPI or inherited such as Balsam).
    exctr = Executor.executor

    task = exctr.submit(app_name='sim1', num_procs=8, app_args='input.txt',
                        stdout='out.txt', stderr='err.txt')

    timeout_sec = 600
    poll_delay_sec = 1

    while(not task.finished):

        # Has manager sent a finish signal
        exctr.manager_poll()
        if exctr.manager_signal == 'finish':
            task.kill()
            my_cleanup()

        # Check output file for error and kill task
        elif task.stdout_exists():
            if 'Error' in task.read_stdout():
                task.kill()

        elif task.runtime > timeout_sec:
            task.kill()  # Timeout

        else:
            time.sleep(poll_delay_sec)
            task.poll()

    print(task.state)  # state may be finished/failed/killed

Users primarily concerned with running their tasks to completion without intermediate
evaluation don't necessarily need to construct a polling loop like above, but can
instead use an ``Executor`` instance's ``polling_loop()`` method. An alternative
to the above simulation function may resemble::

    import time
    from libensemble.executors.executor import Executor

    # Will return Executor (whether MPI or inherited such as Balsam).
    exctr = Executor.executor

    task = exctr.submit(app_name='sim1', num_procs=8, app_args='input.txt',
                        stdout='out.txt', stderr='err.txt')

    timeout_sec = 600
    poll_delay_sec = 1

    exctr.polling_loop(task, timeout=timeout_sec, delay=poll_delay_sec)

    print(task.state)  # state may be finished/failed/killed

See the :doc:`executor<executor>` interface for the complete API.

For a more realistic example see
the :doc:`Electrostatic Forces example <../tutorials/executor_forces_tutorial>`,
which launches the ``forces.x`` application as an MPI task.

.. note::
    Applications or tasks submitted via the Balsam Executor are referred to as
    **"jobs"** within Balsam, including within Balsam's database and when
    describing the state of a completed submission.

The ``MPIExecutor`` autodetects system criteria such as the appropriate MPI launcher
and mechanisms to poll and kill tasks. It also has access to the resource manager,
which partitions resources amongst workers, ensuring that runs utilize different
resources (e.g., nodes). Furthermore, the ``MPIExecutor`` offers resilience via the
feature of re-launching tasks that fail to start because of system factors.

Various back-end mechanisms may be used by the Executor to best interact
with each system, including proxy launchers or task management systems such as
Balsam_. Currently, these Executors launch at the application level within
an existing resource pool. However, submissions to a batch scheduler may be
supported in future Executors.

See :doc:`Running on HPC Systems<../platforms/platforms_index>` to see, with
diagrams, how common options such as ``libE_specs['dedicated_mode']`` affect the
run configuration on clusters and supercomputers.

.. _Balsam: https://balsam.readthedocs.io/en/latest/
