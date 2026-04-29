.. _executor_index:

Executors
=========

libEnsemble's Executors can be used within user functions to provide a simple,
portable interface for running and managing user applications.

.. tab-set::

    .. tab-item:: Overview

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

        .. _concurrent futures: https://docs.python.org/library/concurrent.futures.html

    .. tab-item:: Base Executor

        .. automodule:: executor
            :no-undoc-members:

        Only for running local serial-launched applications.
        To run MPI applications and use detected resources, use the `MPI Executor` tab.

        .. tab-set::

            .. tab-item:: Base Executor

                .. autoclass:: libensemble.executors.executor.Executor
                    :members:
                    :exclude-members: serial_setup, sim_default_app, gen_default_app, get_app, default_app, set_resources, get_task, set_workerID, set_worker_info, new_tasks_timing, add_platform_info, set_gen_procs_gpus, kill, poll

                    .. automethod:: __init__

            .. tab-item:: Task

                .. _task_tag:

                Tasks are created and returned by the Executor's ``submit()``. Tasks
                can be polled, killed, and waited on with the respective ``poll``, ``kill``, and ``wait`` functions.
                Task information can be queried through instance attributes and query functions.

                .. autoclass:: libensemble.executors.executor.Task
                    :members:
                    :exclude-members: calc_task_timing, check_poll

            .. tab-item:: Task Attributes

                .. note::
                    These should not be set directly. Tasks are launched by the Executor,
                    and task information can be queried through the task attributes
                    below and the query functions.

                :task.state: (string) The task status. One of
                            ("UNKNOWN"|"CREATED"|"WAITING"|"RUNNING"|"FINISHED"|"USER_KILLED"|"FAILED"|"FAILED_TO_START")

                :task.process: (process obj) The process object used by the underlying process
                              manager (e.g., return value of subprocess.Popen).
                :task.errcode: (int) The error code (or return code) used by the underlying process manager.
                :task.finished: (boolean) True means task has finished running - not whether it was successful.
                :task.success: (boolean) Did task complete successfully (e.g., the return code is zero)?
                :task.runtime: (int) Time in seconds that task has been running.
                :task.submit_time: (int) Time since epoch that task was submitted.
                :task.total_time: (int) Total time from task submission to completion (only available when task is finished).

                Run configuration attributes - some will be autogenerated:

                :task.workdir: (string) Work directory for the task
                :task.name: (string) Name of task - autogenerated
                :task.app: (app obj) Use application/executable, registered using exctr.register_app
                :task.app_args: (string) Application arguments as a string
                :task.stdout: (string) Name of file where the standard output of the task is written (in task.workdir)
                :task.stderr: (string) Name of file where the standard error of the task is written (in task.workdir)
                :task.dry_run: (boolean) True if task corresponds to dry run (no actual submission)
                :task.runline: (string) Complete, parameterized command to be subprocessed to launch app

    .. tab-item:: MPI Executor

        .. automodule:: mpi_executor
            :no-undoc-members:

        .. autoclass:: libensemble.executors.mpi_executor.MPIExecutor
            :show-inheritance:
            :inherited-members:
            :exclude-members: serial_setup, sim_default_app, gen_default_app, get_app, default_app, set_resources, get_task, set_workerID, set_worker_info, new_tasks_timing, add_platform_info, set_gen_procs_gpus, kill, poll

        **Class-specific Attributes**

        Class-specific attributes can be set directly to alter the behavior of the MPI
        Executor. However, they should be used with caution, because they may not
        be implemented in other executors.

        :max_submit_attempts: (int) Maximum number of launch attempts for a given
                              task. *Default: 5*.
        :fail_time: (int or float) *Only if wait_on_start is set.* Maximum run time to failure in
                    seconds that results in relaunch. *Default: 2*.
        :retry_delay_incr: (int or float) Delay increment between launch attempts in seconds.
                    *Default: 5*. (i.e., First retry after 5 seconds, then 10 seconds, then 15, etc...)

        Example. To increase resilience against submission failures::

            taskctrl = MPIExecutor()
            taskctrl.max_launch_attempts = 8
            taskctrl.fail_time = 5
            taskctrl.retry_delay_incr = 10

        .. _customizer:
