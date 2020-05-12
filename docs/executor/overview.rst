Executor Overview
=================

A typical libEnsemble workflow will include launching tasks from a
:ref:`sim_f<api_sim_f>` (or :ref:`gen_f<api_gen_f>`) running on a worker. We use
"task" to represent an application submission by libEnsemble to the system,
may be a supercomputer, cluster, or other compute resource.

The task could be launched via a subprocess call to ``mpirun`` or an alternative
launcher such as ``aprun`` or ``jsrun``. The ``sim_f`` may then monitor this task,
check output, and possibly kill the task.

An **Executor** interface is provided by libEnsemble to remove the burden of
system interaction from the user and ease the writing of portable user scripts that
launch applications. The Executor provides the key functions: ``submit()``,
``poll()``, and ``kill()``. Task attributes can be queried to determine the status
following each of these commands. Functions are also provided to access and
interrogate files in the task's working directory.

The main Executor class is an abstract class and is inherited by the MPIExecutor,
for direct running of MPI applications. Another Executor is the BalsamMPIExecutor,
which submits an MPI run request from a worker running on a compute node to a
Balsam process running on a launch node (suitable for systems that do not allow
running MPI applications directly from compute nodes).

In a calling script, an Executor object is created, and the executable
generator or simulation applications are registered to it for submission. If an
alternative Executor like Balsam is used, then the applications can be
registered as in the example below. Once in the user-side worker code (sim/gen func),
an MPI-based Executor can be retrieved without any need to specify the type.

**Example usage (code runnable with or without a Balsam backend):**

In calling function::

    sim_app = '/path/to/my/exe'
    USE_BALSAM = False

    if USE_BALSAM:
        from libensemble.executors.balsam_executor import BalsamMPIExecutor
        exctr = BalsamMPIExecutor()
    else:
        from libensemble.executors.mpi_executor import MPIExecutor
        exctr = MPIExecutor()

    exctr.register_calc(full_path=sim_app, calc_type='sim')

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

.. note::
    Applications or tasks submitted via the Balsam Executor are referred to as
    **"jobs"** within Balsam, including within Balsam's database and when
    describing the state of a completed submission.

The MPIExecutor autodetects system criteria such as the appropriate MPI launcher
and mechanisms to poll and kill tasks. It will also partition resources amongst
workers, ensuring that runs utilise different resources (e.g. nodes).
Furthermore, the MPIExecutor offers resilience via the feature of re-launching
tasks that fail because of system factors.

Various back-end mechanisms may be used by the Executor to best interact
with each system, including proxy launchers or task management systems such as
Balsam_. Currently, these Executors launch at the application level within
an existing resource pool. However, submissions to a batch scheduler may be
supported in future Executors.

.. _Balsam: https://balsam.readthedocs.io/en/latest/
