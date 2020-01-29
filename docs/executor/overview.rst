Executor Overview
=======================

Users who wish to launch tasks to a system from a :ref:`sim_f<api_sim_f>` (or :ref:`gen_f<api_gen_f>`)
running on a worker have several options.

Typically, an MPI task could be initialized with a subprocess call to
``mpirun`` or an alternative launcher such as ``aprun`` or ``jsrun``. The ``sim_f``
may then monitor this task, check output, and possibly kill the task. We use "task"
to represent an application submission by libEnsemble to the system, which may
be a supercomputer, cluster, or other compute resource.

An **executor** interface is provided by libEnsemble to remove the burden of
system interaction from the user and ease writing portable user scripts that
launch applications. The executor provides the key functions: ``submit()``,
``poll()``, and ``kill()``. Job attributes can be queried to determine status after
each poll. To implement these functions, libEnsemble autodetects system criteria
such as the MPI launcher and mechanisms to poll and kill tasks on supported systems.
libEnsemble's executor is resilient and can relaunch tasks that fail
because of system factors.

Functions are also provided to access and interrogate files in the task's working directory.

Various back-end mechanisms may be used by the executor to best interact
with each system, including proxy launchers or task management systems such as
Balsam_. Currently, these executors launch at the application level within
an existing resource pool. However, submissions to a batch scheduler may be
supported in the future.

In a calling script, an executor object is created, and the executable
generator or simulation applications are registered to it for submission. If an
alternative executor like Balsam is used, then the applications can be
registered as in the example below. Once in the user-side worker code (sim/gen func),
an MPI-based executor can be retrieved without any need to specify the type.

**Example usage (code runnable with or without a Balsam backend):**

In calling function::

    sim_app = '/path/to/my/exe'
    USE_BALSAM = False

    if USE_BALSAM:
        from libensemble.executors.balsam_executor import BalsamExecutor
        taskctrl = BalsamExecutor()
    else:
        from libensemble.executors.mpi_executor import MPI_Executor
        taskctrl = MPI_Executor()

    taskctrl.register_calc(full_path=sim_app, calc_type='sim')

In user sim func::

    import time

    # Will return executor (whether MPI or inherited such as Balsam).
    exctr = Executor.executor

    task = exctr.submit(calc_type='sim', num_procs=8, app_args='input.txt',
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

See the :doc:`executor<executor>` interface for API.

For a more realistic example see
the :doc:`Electrostatic Forces example <../examples/calling_scripts>`,
which launches the ``forces.x`` application as an MPI task.

.. _Balsam: https://balsam.readthedocs.io/en/latest/
