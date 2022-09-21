"""
This module launches and controls the running of MPI applications.

In order to create an MPI executor, the calling script should contain ::

    exctr = MPIExecutor()

See the executor API below for optional arguments.
"""

import os
import logging
import time

import libensemble.utils.launcher as launcher
from libensemble.resources.mpi_resources import get_MPI_variant
from libensemble.executors.executor import Executor, Task, ExecutorException
from libensemble.executors.mpi_runner import MPIRunner

logger = logging.getLogger(__name__)
# To change logging level for just this module
# logger.setLevel(logging.DEBUG)


class MPIExecutor(Executor):
    """The MPI executor can create, poll and kill runnable MPI tasks

    **Object Attributes:**

    :ivar list list_of_tasks: A list of tasks created in this executor
    :ivar int manager_signal: The most recent manager signal received since manager_poll() was called.
    """

    def __init__(self, custom_info={}):
        """Instantiate a new MPIExecutor instance.

        A new MPIExecutor is created with an application
        registry and configuration attributes.

        This is typically created in the user calling script. The
        MPIExecutor will use system resource information supplied by
        the libEsnemble resource manager when submitting tasks.

        Parameters
        ----------

        custom_info: dict, optional
            Provide custom overrides to selected variables that are usually
            auto-detected. See below.


        The MPIExecutor automatically detects MPI runners and launch
        mechanisms. However it is possible to override the detected
        information using the ``custom_info`` argument. This takes
        a dictionary of values.

        The allowable fields are::

            'mpi_runner' [string]:
                Select runner: 'mpich', 'openmpi', 'aprun', 'srun', 'jsrun', 'custom'
                All except 'custom' relate to runner classes in libEnsemble.
                Custom allows user to define their own run-lines but without parsing
                arguments or making use of auto-resources.
            'runner_name' [string]:
                Runner name: Replaces run command if present. All runners have a default
                except for 'custom'.
            'subgroup_launch' [boolean]:
                Whether MPI runs should be initiatied in a new process group. This needs
                to be correct for kills to work correctly. Use the standalone test at
                libensemble/tests/standalone_tests/kill_test to determine correct value
                for a system.

        For example::

            customizer = {'mpi_runner': 'mpich',
                          'runner_name': 'wrapper -x mpich'}

            from libensemble.executors.mpi_executor import MPIExecutor
            exctr = MPIExecutor(custom_info=customizer)

        """

        Executor.__init__(self)

        # MPI launch settings
        self.max_launch_attempts = 5
        self.fail_time = 2
        self.retry_delay_incr = 5  # Incremented wait after each launch attempt

        # Apply custom options
        mpi_runner_type = custom_info.get("mpi_runner", None)
        runner_name = custom_info.get("runner_name", None)
        subgroup_launch = custom_info.get("subgroup_launch", None)

        if not mpi_runner_type:
            mpi_runner_type = get_MPI_variant()
        self.mpi_runner = MPIRunner.get_runner(mpi_runner_type, runner_name)
        if subgroup_launch is not None:
            self.mpi_runner.subgroup_launch = subgroup_launch
        self.resources = None

    def set_resources(self, resources):
        self.resources = resources

    def _launch_with_retries(self, task, runline, subgroup_launch, wait_on_start):
        """Launch task with retry mechanism"""
        retry_count = 0
        while retry_count < self.max_launch_attempts:
            retry = False
            try:
                retry_string = f" (Retry {retry_count})" if retry_count > 0 else ""
                logger.info(f"Launching task {task.name}{retry_string}: {' '.join(runline)}")
                task.run_attempts += 1
                with open(task.stdout, "w") as out, open(task.stderr, "w") as err:
                    task.process = launcher.launch(
                        runline,
                        cwd="./",
                        stdout=out,
                        stderr=err,
                        start_new_session=subgroup_launch,
                    )
            except Exception as e:
                logger.warning(f"task {task.name} submit command failed on try {retry_count} with error {e}")
                retry = True
                retry_count += 1
            else:
                if wait_on_start:
                    self._wait_on_start(task, self.fail_time)

                if task.state == "FAILED":
                    logger.warning(
                        f"task {task.name} failed within fail_time on "
                        f"try {retry_count} with err code {task.errcode}"
                    )
                    retry = True
                    retry_count += 1

            if retry and retry_count < self.max_launch_attempts:
                logger.debug(f"Retry number {retry_count} for task {task.name}")
                time.sleep(retry_count * self.retry_delay_incr)
                task.reset()  # Some cases may require user cleanup
            else:
                break

    def submit(
        self,
        calc_type=None,
        app_name=None,
        num_procs=None,
        num_nodes=None,
        procs_per_node=None,
        machinefile=None,
        app_args=None,
        stdout=None,
        stderr=None,
        stage_inout=None,
        hyperthreads=False,
        dry_run=False,
        wait_on_start=False,
        extra_args=None,
    ):
        """Creates a new task, and either executes or schedules execution.

        The created task object is returned.

        Parameters
        ----------

        calc_type: String, optional
            The calculation type: 'sim' or 'gen'
            Only used if app_name is not supplied. Uses default sim or gen application.

        app_name: String, optional
            The application name. Must be supplied if calc_type is not.

        num_procs: int, optional
            The total number of MPI tasks on which to submit the task

        num_nodes: int, optional
            The number of nodes on which to submit the task

        procs_per_node: int, optional
            The processes per node for this task

        machinefile: string, optional
            Name of a machinefile for this task to use

        app_args: string, optional
            A string of the application arguments to be added to task
            submit command line

        stdout: string, optional
            A standard output filename

        stderr: string, optional
            A standard error filename

        stage_inout: string, optional
            A directory to copy files from; default will take from
            current directory

        hyperthreads: boolean, optional
            Whether to submit MPI tasks to hyperthreads

        dry_run: boolean, optional
            Whether this is a dry_run - no task will be launched; instead
            runline is printed to logger (at INFO level)

        wait_on_start: boolean, optional
            Whether to wait for task to be polled as RUNNING (or other
            active/end state) before continuing

        extra_args: String, optional
            Additional command line arguments to supply to MPI runner. If
            arguments are recognised as MPI resource configuration
            (num_procs, num_nodes, procs_per_node) they will be used in
            resources determination unless also supplied in the direct
            options.

        Returns
        -------

        task: obj: Task
            The launched task object


        Note that if some combination of num_procs, num_nodes, and
        procs_per_node is provided, these will be honored if
        possible. If resource detection is on and these are omitted,
        then the available resources will be divided among workers.
        """

        if app_name is not None:
            app = self.get_app(app_name)
        elif calc_type is not None:
            app = self.default_app(calc_type)
        else:
            raise ExecutorException("Either app_name or calc_type must be set")

        default_workdir = os.getcwd()
        task = Task(app, app_args, default_workdir, stdout, stderr, self.workerID)

        if not dry_run:
            self._check_app_exists(task.app.full_path)

        if stage_inout is not None:
            logger.warning("stage_inout option ignored in this " "executor - runs in-place")

        mpi_specs = self.mpi_runner.get_mpi_specs(
            task,
            num_procs,
            num_nodes,
            procs_per_node,
            machinefile,
            hyperthreads,
            extra_args,
            self.resources,
            self.workerID,
        )

        mpi_command = self.mpi_runner.mpi_command
        sglaunch = self.mpi_runner.subgroup_launch
        runline = launcher.form_command(mpi_command, mpi_specs)

        runline.extend(task.app.app_cmd.split())
        if task.app_args is not None:
            runline.extend(task.app_args.split())

        task.runline = " ".join(runline)  # Allow to be queried
        if dry_run:
            task.dry_run = True
            logger.info(f"Test (No submit) Runline: {' '.join(runline)}")
            task._set_complete(dry_run=True)
        else:
            # Launch Task
            self._launch_with_retries(task, runline, sglaunch, wait_on_start)

            if not task.timer.timing and not task.finished:
                task.timer.start()
                task.submit_time = task.timer.tstart  # Time not date - may not need if using timer.

        self.list_of_tasks.append(task)

        return task

    def set_worker_info(self, comm, workerid=None):
        """Sets info for this executor"""
        super().set_worker_info(comm, workerid)
