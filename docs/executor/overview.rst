Executor Overview
=================

Most computationally expensive libEnsemble workflows involve launching applications
from a :ref:`sim_f<api_sim_f>` or :ref:`gen_f<api_gen_f>` running on a worker to the
compute nodes of a supercomputer, cluster, or other compute resource.

libEnsemble's **Executor** interface provides **portable app-launches** represented as **Tasks**:

.. tab-set::

    .. tab-item:: Example Setup

        .. code-block:: python

            sim_app = "/path/to/my/exe"

            from libensemble.executors import MPIExecutor

            exctr = MPIExecutor()

            exctr.register_app(full_path=sim_app, app_name="sim1")

    .. tab-item:: Example Usage

        .. code-block:: python

            import time
            from libensemble.executors import Executor

            # Retrieve executor instance
            exctr = Executor.executor

            task = exctr.submit(app_name="sim1", num_procs=8, app_args="input.txt", stdout="out.txt", stderr="err.txt")

            while not task.finished:

                exctr.manager_poll()
                if exctr.manager_signal in [MAN_SIGNAL_KILL, MAN_SIGNAL_FINISH]:
                    task.kill()
                    my_cleanup()

                elif task.stdout_exists():
                    if "Error" in task.read_stdout():
                        task.kill()

                elif task.runtime > 600:
                    task.kill()

                else:
                    time.sleep(1)
                    task.poll()

            print(task.state)  # state may be finished/failed/killed

Executors and Tasks contain many familiar features and methods to
Python's native `concurrent futures`_ interface. Tasks are much like
``futures``, except they correspond to an application instance.

Executor instances do **not** have to be passed to  ``Ensemble.run()`` or ``libE()``.
They can be extracted via ``Executor.executor`` in the sim function (regardless of type).

Looping over a task's status isn't required. Some alternatives:

.. tab-set::

    .. tab-item:: Executor.polling_loop()

        .. code-block:: python

            from libensemble.executors import Executor

            # Will return Executor (whether MPI or inherited such as Balsam).
            exctr = Executor.executor

            task = exctr.submit(app_name="sim1", num_procs=8, app_args="input.txt", stdout="out.txt", stderr="err.txt")

            timeout_sec = 600
            poll_delay_sec = 1

            exctr.polling_loop(task, timeout=timeout_sec, delay=poll_delay_sec)

            print(task.state)  # state may be finished/failed/killed

    .. tab-item:: Task.result()

        .. code-block:: python

            from libensemble.executors import Executor

            # Will return Executor (whether MPI or inherited such as Balsam).
            exctr = Executor.executor

            task = exctr.submit(app_name="sim1", num_procs=8, app_args="input.txt", stdout="out.txt", stderr="err.txt")

            print(task.result(timeout=600))  # returns state on completion

See the :doc:`executor<executor>` interface for the complete API.

For a more realistic example see
the :doc:`Electrostatic Forces example <../tutorials/executor_forces_tutorial>`,
which launches the ``forces.x`` application as an MPI task.

The ``MPIExecutor`` autodetects MPI launchers
and mechanisms to poll and kill tasks, and can partition resources (including multiple nodes)
amongst workers.

Executors can interact with proxy launchers or task management systems such as Balsam_.

See :doc:`Running on HPC Systems<../platforms/platforms_index>` to see, with
diagrams, how common options such as ``libE_specs["dedicated_mode"]`` affect the
run configuration on clusters and supercomputers.

.. _Balsam: https://balsam.readthedocs.io/en/latest/
.. _`concurrent futures`: https://docs.python.org/3.8/library/concurrent.futures.html
