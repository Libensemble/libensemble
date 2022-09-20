Executor Overview
=================

Most computationally expensive libEnsemble workflows involve launching applications
from a :ref:`sim_f<api_sim_f>` or :ref:`gen_f<api_gen_f>` running on a worker to the
compute nodes of a supercomputer, cluster, or other compute resource.

An **Executor** interface is provided by libEnsemble to remove the burden of
system interaction from the user and improve workflow portability. Users first register
their applications to Executor instances, which then return corresponding ``Task``
objects upon submission within user functions.

**Task** attributes and retrieval functions can be queried to determine
the status of running application instances. Functions are also provided to access
and interrogate files in the task's working directory.

libEnsemble's Executors and Tasks contain many familiar features and methods to
Python's native `concurrent futures`_ interface. Executors feature the ``submit()``
function for launching apps (detailed below),  but currently do not support
``map()`` or ``shutdown()``. Tasks are much like ``futures``, except they correspond
to an application instance instead of a callable. They feature the ``cancel()``,
``cancelled()``, ``running()``, ``done()``, ``result()``, and ``exception()`` functions
from the standard.

The main ``Executor`` class is an abstract class, inherited by the ``MPIExecutor``
for direct running of MPI applications, and the ``BalsamExecutor``
for submitting MPI run requests from a worker running on a compute node to the
Balsam service. This second approach is suitable for
systems that don't allow submitting MPI applications from compute nodes.

Typically, users choose and parameterize their ``Executor`` objects in their
calling scripts, where each executable generator or simulation application is
registered to it. If an alternative Executor like Balsam is used, then the applications can be
registered as in the example below. Once in the user-side worker code (sim/gen func),
the Executor can be retrieved without any need to specify the type.

Once the Executor is retrieved, tasks can be submitted by specifying the ``app_name``
from registration in the calling script alongside other optional parameters
described in the API.

**Example usage (code runnable with or without a Balsam 0.5.0 backend):**

In calling script::

    sim_app = '/path/to/my/exe'
    USE_BALSAM = False

    if USE_BALSAM:
        from libensemble.executors.balsam_executors import LegacyBalsamMPIExecutor
        exctr = LegacyBalsamMPIExecutor()
    else:
        from libensemble.executors.mpi_executor import MPIExecutor
        exctr = MPIExecutor()

    exctr.register_app(full_path=sim_app, app_name='sim1')

Note that Executor instances in the calling script are also stored as class attributes, and
do **not** have to be passed to ``libE()``. They can be extracted via *Executor.executor*
in the sim function (regardless of type).

In user simulation function::

    import time
    from libensemble.executors import Executor

    # Will return Executor (whether MPI or inherited such as Balsam).
    exctr = Executor.executor

    task = exctr.submit(app_name='sim1', num_procs=8, app_args='input.txt',
                        stdout='out.txt', stderr='err.txt')

    timeout_sec = 600
    poll_delay_sec = 1

    while(not task.finished):

        # Has manager sent a finish signal
        exctr.manager_poll()
        if exctr.manager_signal in [MAN_SIGNAL_KILL, MAN_SIGNAL_FINISH]:
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

Executor instances can also be retrieved using Python's ``with`` context switching statement,
although this is effectively syntactical sugar to above::

    from libensemble.executors import Executor

    with Executor.executor as exctr:
        task = exctr.submit(app_name='sim1', num_procs=8, app_args='input.txt',
                            stdout='out.txt', stderr='err.txt')

    ...

Users primarily concerned with running their tasks to completion without intermediate
evaluation don't necessarily need to construct a polling loop like above, but can
instead use an ``Executor`` instance's ``polling_loop()`` method. An alternative
to the above simulation function may resemble::

    from libensemble.executors import Executor

    # Will return Executor (whether MPI or inherited such as Balsam).
    exctr = Executor.executor

    task = exctr.submit(app_name='sim1', num_procs=8, app_args='input.txt',
                        stdout='out.txt', stderr='err.txt')

    timeout_sec = 600
    poll_delay_sec = 1

    exctr.polling_loop(task, timeout=timeout_sec, delay=poll_delay_sec)

    print(task.state)  # state may be finished/failed/killed

Or put *yet another way*::

    from libensemble.executors import Executor

    # Will return Executor (whether MPI or inherited such as Balsam).
    exctr = Executor.executor

    task = exctr.submit(app_name='sim1', num_procs=8, app_args='input.txt',
                        stdout='out.txt', stderr='err.txt')

    print(task.result(timeout=600))  # returns state on completion

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
.. _`concurrent futures`: https://docs.python.org/3.8/library/concurrent.futures.html
