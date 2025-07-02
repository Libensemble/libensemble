"""
This module launches and controls the running of MPI applications.

In order to create an MPI executor, the calling script should contain:

.. code-block:: python

    exctr = MPIExecutor()

The MPIExecutor will use system resource information supplied by the libEnsemble
resource manager when submitting tasks.

"""

import logging
import os
import time

import libensemble.utils.launcher as launcher
from libensemble.executors.executor import Executor, ExecutorException, Task
from libensemble.executors.mpi_runner import MPIRunner
from libensemble.resources.mpi_resources import get_MPI_variant

logger = logging.getLogger(__name__)
# To change logging level for just this module
# logger.setLevel(logging.DEBUG)


class MPIExecutor(Executor):
    """
    The MPI executor can create, poll and kill runnable MPI tasks

    Parameters
    ----------

    custom_info: dict, Optional
        Provide custom overrides to selected variables that are usually
        auto-detected. See below.


    .. dropdown:: custom_info usage

        The MPIExecutor automatically detects MPI runners and launch
        mechanisms. However it is possible to override the detected
        information using the ``custom_info`` argument. This takes
        a dictionary of values.

        The allowable fields are:

        .. parsed-literal::

            **'mpi_runner'** [string]:
                Select runner: `'mpich'`, `'openmpi'`, `'aprun'`, `'srun'`, `'jsrun'`, `'custom'`
                All except `'custom'` relate to runner classes in libEnsemble.
                Custom allows user to define their own run-lines but without parsing
                arguments or making use of auto-resources.
            **'runner_name'** [string]:
                The literal string that appears at the front of the run command.
                This is typically 'mpirun', 'srun', etc., and can be a full path.
                Defaults exist for all runners except 'custom'.
            **'subgroup_launch'** [bool]:
                Whether MPI runs should be initiated in a new process group. This needs
                to be correct for kills to work correctly. Use the standalone test at
                `libensemble/tests/standalone_tests/kill_test` to determine correct value
                for a system.

    For example::

        customizer = {'mpi_runner': 'mpich',
                      'runner_name': 'wrapper -x mpich'}

        from libensemble.executors.mpi_executor import MPIExecutor
        exctr = MPIExecutor(custom_info=customizer)


    """

    def __init__(self, custom_info: dict = {}) -> None:
        """Instantiate a new MPIExecutor instance."""

        Executor.__init__(self)

        # MPI launch settings
        self.max_launch_attempts = 5
        self.fail_time = 2
        self.retry_delay_incr = 5  # Incremented wait after each launch attempt
        self.resources = None
        self.platform_info = None
        self.gen_nprocs = None
        self.gen_ngpus = None

        # Apply custom options
        self.mpi_runner_type = custom_info.get("mpi_runner")
        self.runner_name = custom_info.get("runner_name")
        self.subgroup_launch = custom_info.get("subgroup_launch")
        self.mpi_runner_obj = None  # Do not set here or will override platform

    def _create_mpi_runner_obj(self, mpi_runner_type, runner_name, subgroup_launch) -> MPIRunner:
        mpi_runner_obj = MPIRunner.get_runner(mpi_runner_type, runner_name, self.platform_info)
        if subgroup_launch is not None:
            mpi_runner_obj.subgroup_launch = subgroup_launch
        return mpi_runner_obj

    def _create_mpi_runner_from_config(self, mpi_config: dict = {}) -> MPIRunner:
        """Return an mpi_runner object from given info"""

        mpi_runner_type = mpi_config.get("mpi_runner")
        runner_name = mpi_config.get("runner_name")
        subgroup_launch = mpi_config.get("subgroup_launch")
        return self._create_mpi_runner_obj(mpi_runner_type, runner_name, subgroup_launch)

    def _create_mpi_runner_from_attr(self) -> MPIRunner:
        """Create mpi_runner_obj based on existing attributes

        If runner type has not been given, then detect
        """
        if not self.mpi_runner_type:
            self.mpi_runner_type = get_MPI_variant()
        return self._create_mpi_runner_obj(self.mpi_runner_type, self.runner_name, self.subgroup_launch)

    def add_platform_info(self, platform_info={}):
        """Add user supplied platform info to executor"""

        # Apply platform options (does not overwrite custom_info Executor options)
        if platform_info:
            self.mpi_runner_type = self.mpi_runner_type or platform_info.get("mpi_runner")
            self.runner_name = self.runner_name or platform_info.get("runner_name")
        self.platform_info = platform_info

        # If runner type has not been given, then detect
        self.mpi_runner_obj = self._create_mpi_runner_from_attr()

    def set_gen_procs_gpus(self, libE_info):
        """Add gen supplied procs and gpus"""
        self.gen_nprocs = libE_info.get("num_procs")
        self.gen_ngpus = libE_info.get("num_gpus")

    def set_resources(self, resources) -> None:
        self.resources = resources

    def _launch_with_retries(
        self, task: Task, subgroup_launch: bool, wait_on_start: bool, run_cmd: list[str], use_shell: bool
    ) -> None:
        """Launch task with retry mechanism"""
        retry_count = 0

        while retry_count < self.max_launch_attempts:
            retry = False
            try:
                retry_string = f" (Retry {retry_count})" if retry_count > 0 else ""
                logger.info(f"Launching task {task.name}{retry_string}: {' '.join(run_cmd)}")
                task.run_attempts += 1
                with open(task.stdout, "w") as out, open(task.stderr, "w") as err:
                    task.process = launcher.launch(
                        run_cmd,
                        cwd="./",
                        stdout=out,
                        stderr=err,
                        start_new_session=subgroup_launch,
                        shell=use_shell,
                    )
            except Exception as e:
                logger.warning(f"task {task.name} submit command failed on try {retry_count} with error {e}")
                task.state = "FAILED_TO_START"
                task.finished = True
                retry = True
                retry_count += 1
            else:
                if wait_on_start:
                    self._wait_on_start(task, self.fail_time)
                task.poll()

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
        calc_type: str | None = None,
        app_name: str | None = None,
        num_procs: int | None = None,
        num_nodes: int | None = None,
        procs_per_node: int | None = None,
        num_gpus: int | None = None,
        machinefile: str | None = None,
        app_args: str | None = None,
        stdout: str | None = None,
        stderr: str | None = None,
        stage_inout: str | None = None,
        hyperthreads: bool | None = False,
        dry_run: bool | None = False,
        wait_on_start: bool | None = False,
        extra_args: str | None = None,
        auto_assign_gpus: bool | None = False,
        match_procs_to_gpus: bool | None = False,
        env_script: str | None = None,
        mpi_runner_type: str | dict | None = None,
    ) -> Task:
        """Creates a new task, and either executes or schedules execution.

        The created :class:`task<libensemble.executors.executor.Task>` object is returned.

        The user must supply either the app_name or calc_type arguments (app_name
        is recommended). All other arguments are optional.

        Parameters
        ----------

        calc_type: str, Optional
            The calculation type: 'sim' or 'gen'
            Only used if app_name is not supplied. Uses default sim or gen application.

        app_name: str, Optional
            The application name. Must be supplied if calc_type is not.

        num_procs: int, Optional
            The total number of processes (MPI ranks)

        num_nodes: int, Optional
            The number of nodes

        procs_per_node: int, Optional
            The processes per node

        num_gpus: int, Optional
            The total number of GPUs

        machinefile: str, Optional
            Name of a machinefile

        app_args: str, Optional
            A string of the application arguments to be added to task
            submit command line

        stdout: str, Optional
            A standard output filename

        stderr: str, Optional
            A standard error filename

        stage_inout: str, Optional
            A directory to copy files from; default will take from
            current directory

        hyperthreads: bool, Optional
            Whether to submit MPI tasks to hyperthreads

        dry_run: bool, Optional
            Whether this is a dry_run - no task will be launched; instead
            runline is printed to logger (at INFO level)

        wait_on_start: bool, Optional
            Whether to wait for task to be polled as RUNNING (or other
            active/end state) before continuing.

        extra_args: str, Optional
            Additional command line arguments to supply to MPI runner. If
            arguments are recognized as MPI resource configuration
            (num_procs, num_nodes, procs_per_node) they will be used in
            resources determination unless also supplied in the direct
            options.

        auto_assign_gpus: bool, Optional
            Auto-assign GPUs available to this worker using either the method
            supplied in configuration or determined by detected environment.
            Default: False

        match_procs_to_gpus: bool, Optional
            For use with auto_assign_gpus. Auto-assigns MPI processors to match
            the assigned GPUs. Default: False unless auto_assign_gpus is True and
            no other CPU configuration is supplied.

        env_script: str, Optional
            The full path of a shell script to set up the environment for the
            launched task. This will be run in the subprocess, and not affect
            the worker environment. The script should start with a shebang.

        mpi_runner_type: (str|dict), Optional
            An MPI runner to be used for this submit only. Supply either a string
            for the MPI runner type or a dictionary for detailed configuration
            (see custom_info on MPIExecutor constructor). This will not change
            the default MPI runner for the executor.
            Example string inputs are "mpich", "openmpi", "srun", "jsrun", "aprun".

        Returns
        -------

        task: Task
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
        task = Task(app, app_args, default_workdir, stdout, stderr, self.workerID, dry_run)

        if not dry_run:
            self._check_app_exists(task.app.full_path)

        if stage_inout is not None:
            logger.warning("stage_inout option ignored in this " "executor - runs in-place")

        if not num_procs and not match_procs_to_gpus:
            num_procs = self.gen_nprocs

        if num_gpus is None:
            num_gpus = self.gen_ngpus

        if mpi_runner_type is not None:
            if isinstance(mpi_runner_type, str):
                mpi_config = {"mpi_runner": mpi_runner_type}
            else:
                mpi_config = mpi_runner_type
            mpi_runner_obj = self._create_mpi_runner_from_config(mpi_config)
        else:
            mpi_runner_obj = self.mpi_runner_obj or self._create_mpi_runner_from_attr()

        if env_script is None and mpi_runner_obj is None:
            raise ExecutorException("No valid MPI runner was found")

        mpi_specs = mpi_runner_obj.get_mpi_specs(
            task,
            num_procs,
            num_nodes,
            procs_per_node,
            num_gpus,
            machinefile,
            hyperthreads,
            extra_args,
            auto_assign_gpus,
            match_procs_to_gpus,
            self.resources,
            self.workerID,
        )

        mpi_command = mpi_runner_obj.mpi_command
        sglaunch = mpi_runner_obj.subgroup_launch
        runline = launcher.form_command(mpi_command, mpi_specs)

        runline.extend(task.app.app_cmd.split())
        if task.app_args is not None:
            runline.extend(task.app_args.split())

        task.runline = " ".join(runline)  # Allow to be queried

        if env_script is not None:
            run_cmd = Executor._process_env_script(task, runline, env_script)
            use_shell = True
        else:
            run_cmd = runline
            use_shell = False

        if dry_run:
            logger.info(f"Test (No submit) Runline: {' '.join(run_cmd)}")
            task._set_complete()
        else:
            # Set environment variables and launch task
            task._implement_env()

            # Launch Task
            self._launch_with_retries(task, sglaunch, wait_on_start, run_cmd, use_shell)

            if not task.timer.timing and not task.finished:
                task.timer.start()
                task.submit_time = task.timer.tstart  # Time not date - may not need if using timer.

        self.list_of_tasks.append(task)

        return task

    def set_worker_info(self, comm, workerid=None):
        """Sets info for this executor"""
        super().set_worker_info(comm, workerid)
