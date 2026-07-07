"""
This module provides a native Flux executor using the Flux Python API.

The FluxExecutor submits jobs directly to Flux using its Python bindings,
rather than wrapping `flux run` as a subprocess. This provides better
integration with Flux's job lifecycle management and is particularly
useful when running inside containers where MPI runners may not be available.

Usage::

    from libensemble.executors.flux_executor import FluxExecutor

    exctr = FluxExecutor()
    exctr.register_app(full_path="/path/to/my_app.x", app_name="my_app")

    # In your sim function:
    task = exctr.submit(app_name="my_app", num_procs=4, num_nodes=1)
    task.wait()

Requirements:
    - flux-core Python bindings must be installed
    - Must be running inside a Flux instance (FLUX_URI must be set)
"""

import logging
import os
import shlex
import time

from libensemble.executors.executor import (
    Application,
    Executor,
    ExecutorException,
    Task,
    jassert,
)

logger = logging.getLogger(__name__)

# Try to import flux - it's optional
try:
    import flux
    import flux.job
    from flux.job import JobspecV1

    FLUX_AVAILABLE = True
except ImportError:
    FLUX_AVAILABLE = False
    flux = None
    JobspecV1 = None


class FluxTask(Task):
    """
    Task subclass for Flux jobs using native Flux Python API.

    Overrides poll() and kill() to use Flux job management instead
    of subprocess operations.
    """

    def __init__(
        self,
        app=None,
        app_args=None,
        workdir=None,
        stdout=None,
        stderr=None,
        workerid=None,
        dry_run=False,
    ) -> None:
        super().__init__(app, app_args, workdir, stdout, stderr, workerid, dry_run)
        self.flux_handle = None
        self.flux_jobid = None
        self.flux_future = None

    def reset(self) -> None:
        super().reset()
        self.flux_jobid = None
        self.flux_future = None

    def _check_poll(self) -> bool:
        """Check whether polling this task makes sense."""
        jassert(
            self.flux_jobid is not None,
            f"task {self.name} has no Flux job ID - check task has been launched",
        )
        if self.finished:
            logger.debug(f"Polled task {self.name} has already finished. Not re-polling. Status is {self.state}")
            return False
        return True

    def poll(self) -> None:
        """Polls and updates the status attributes of the task using Flux job state."""
        if self.dry_run:
            self._set_complete()
            return

        if not self._check_poll():
            return

        try:
            info = flux.job.get_job(self.flux_handle, self.flux_jobid)
            jassert(info is not None, f"Flux job {self.flux_jobid} was not found")
            state = str(info.get("state", "UNKNOWN")).upper()

            # Map Flux states to libEnsemble states
            # Flux states: DEPEND, PRIORITY, SCHED, RUN, CLEANUP, INACTIVE
            if state in ("DEPEND", "PRIORITY", "SCHED"):
                self.state = "WAITING"
            elif state == "RUN":
                self.state = "RUNNING"
                self.runtime = self.timer.elapsed
            elif state in ("CLEANUP", "INACTIVE"):
                # Job has finished - check if successful
                self._handle_completion(info)
            else:
                self.state = "UNKNOWN"
                self.runtime = self.timer.elapsed

        except Exception as e:
            logger.warning(f"Error polling Flux job {self.flux_jobid}: {e}")
            self.state = "UNKNOWN"
            self.runtime = self.timer.elapsed

    def _handle_completion(self, info: dict) -> None:
        """Handle job completion and determine success/failure."""
        self.finished = True
        self.calc_task_timing()

        # Check result/exit status
        result = str(info.get("result", "")).upper()
        success = result == "COMPLETED" or info.get("returncode", 1) == 0

        if success:
            self.success = True
            self.state = "FINISHED"
            self.errcode = 0
        else:
            self.success = False
            self.state = "FAILED"
            # Try to get exit code from result
            self.errcode = info.get("returncode", 1)

        logger.info(f"Task {self.name} finished with state {self.state} (result={result})")

    def _set_complete(self) -> None:
        """Set task as complete (used for dry_run)."""
        self.finished = True
        if self.dry_run:
            self.success = True
            self.state = "FINISHED"
        else:
            self.calc_task_timing()
            self.success = self.errcode == 0
            self.state = "FINISHED" if self.success else "FAILED"
            logger.info(f"Task {self.name} finished with errcode {self.errcode} ({self.state})")

    def wait(self, timeout: float | None = None) -> None:
        """Waits on completion of the Flux job or raises TimeoutExpired exception."""
        from libensemble.executors.executor import TimeoutExpired

        if self.dry_run:
            self._set_complete()
            return

        if not self._check_poll():
            return

        try:
            # Wait for job to complete
            start_time = time.time()
            while True:
                self.poll()
                if self.finished:
                    break

                if timeout is not None:
                    elapsed = time.time() - start_time
                    if elapsed >= timeout:
                        raise TimeoutExpired(self.name, timeout)

                time.sleep(0.1)

        except TimeoutExpired:
            raise
        except Exception as e:
            logger.warning(f"Error waiting for Flux job {self.flux_jobid}: {e}")
            self.state = "FAILED"
            self.finished = True

    def kill(self, wait_time: int | None = 60) -> None:
        """Kills/cancels the Flux job.

        Parameters
        ----------
        wait_time: int, Optional
            Time in seconds to wait for cancellation.
            Note: Flux handles job cancellation internally.
        """
        self.poll()
        if self.dry_run:
            return

        if self.finished:
            logger.warning(f"Trying to kill task that is no longer running. Task {self.name}: Status is {self.state}")
            return

        if self.flux_jobid is None:
            logger.warning(f"Task {self.name} has no Flux job ID - cannot kill")
            return

        logger.info(f"Canceling Flux job {self.flux_jobid} for task {self.name}")

        try:
            # Cancel the job using Flux API
            flux.job.cancel(self.flux_handle, self.flux_jobid)

            # Wait briefly for cancellation to take effect
            if wait_time:
                deadline = time.time() + min(wait_time, 5)  # Don't wait too long
                while time.time() < deadline:
                    self.poll()
                    if self.finished:
                        break
                    time.sleep(0.1)

        except Exception as e:
            logger.warning(f"Error canceling Flux job {self.flux_jobid}: {e}")

        self.state = "USER_KILLED"
        self.finished = True
        self.calc_task_timing()


class FluxExecutor(Executor):
    """
    Native Flux executor using the Flux Python API.

    This executor submits jobs directly to Flux rather than wrapping
    `flux run` as a subprocess. It provides better integration with
    Flux's job lifecycle and is suitable for container-based workflows.

    Parameters
    ----------
    None

    Raises
    ------
    ExecutorException
        If flux Python bindings are not available or FLUX_URI is not set.

    Example
    -------
    ::

        from libensemble.executors.flux_executor import FluxExecutor

        exctr = FluxExecutor()
        exctr.register_app(full_path="/path/to/sim.x", app_name="sim")

        # In sim function:
        task = exctr.submit(app_name="sim", num_procs=4)
        task.wait()
    """

    def __init__(self) -> None:
        """Instantiate a new FluxExecutor instance."""
        if not FLUX_AVAILABLE:
            raise ExecutorException(
                "Flux Python bindings not available. "
                "Install flux-core or use MPIExecutor with mpi_runner='flux' instead."
            )

        if not os.environ.get("FLUX_URI"):
            raise ExecutorException(
                "FLUX_URI environment variable not set. " "FluxExecutor must be used inside a Flux instance."
            )

        super().__init__()

        # Connect to the Flux instance
        try:
            self.flux_handle = flux.Flux()
        except Exception as e:
            raise ExecutorException(f"Failed to connect to Flux instance: {e}")

        self.resources = None
        self.platform_info: dict = {}

    def set_resources(self, resources) -> None:
        """Set resources for the executor."""
        self.resources = resources

    def add_platform_info(self, platform_info: dict | None = None) -> None:
        """Add platform info to the executor."""
        self.platform_info = platform_info or {}

    def submit(
        self,
        calc_type: str | None = None,
        app_name: str | None = None,
        num_procs: int | None = None,
        num_nodes: int | None = None,
        procs_per_node: int | None = None,
        num_gpus: int | None = None,
        app_args: str | None = None,
        stdout: str | None = None,
        stderr: str | None = None,
        dry_run: bool = False,
        wait_on_start: bool = False,
        extra_args: str | None = None,
    ) -> FluxTask:
        """Submit a job to Flux.

        Returns :class:`FluxTask` object.

        Parameters
        ----------
        calc_type: str, Optional
            The calculation type: 'sim' or 'gen'

        app_name: str, Optional
            The application name.

        num_procs: int, Optional
            The total number of processes (MPI ranks)

        num_nodes: int, Optional
            The number of nodes

        procs_per_node: int, Optional
            The processes per node

        num_gpus: int, Optional
            The total number of GPUs

        app_args: str, Optional
            Application arguments

        stdout: str, Optional
            Standard output filename

        stderr: str, Optional
            Standard error filename

        dry_run: bool, Optional
            If True, don't actually submit the job

        wait_on_start: bool, Optional
            Whether to wait for job to start running

        extra_args: str, Optional
            Additional arguments (currently not used for native Flux)

        Returns
        -------
        task: FluxTask
            The submitted task object
        """
        app: Application | None = None
        if app_name is not None:
            app = self.get_app(app_name)
        elif calc_type is not None:
            app = self.default_app(calc_type)
        else:
            raise ExecutorException("Either app_name or calc_type must be set")

        assert app is not None

        default_workdir = os.getcwd()
        task = FluxTask(app, app_args, default_workdir, stdout, stderr, self.workerID, dry_run)
        task.flux_handle = self.flux_handle

        if not dry_run:
            self._check_app_exists(task.app)

        if extra_args:
            raise ExecutorException("extra_args is not supported by FluxExecutor")

        num_procs = num_procs or 1
        if num_nodes is None:
            if procs_per_node is not None:
                if num_procs % procs_per_node != 0:
                    raise ExecutorException("num_procs must be divisible by procs_per_node for FluxExecutor")
                num_nodes = num_procs // procs_per_node
            else:
                num_nodes = 1
        elif procs_per_node is not None and num_procs != num_nodes * procs_per_node:
            raise ExecutorException("num_procs must equal num_nodes * procs_per_node for FluxExecutor")

        command = shlex.split(task.app.app_cmd)
        if task.app_args:
            command.extend(shlex.split(task.app_args))

        command = self._set_sim_dir_env(task, command)
        task.runline = " ".join(command)

        if dry_run:
            logger.info(f"Test (No submit) Command: {task.runline}")
            logger.info(f"  num_procs={num_procs}, num_nodes={num_nodes}, procs_per_node={procs_per_node}")
            task._set_complete()
        else:
            # Create Flux jobspec
            try:
                gpus_per_task = None
                if num_gpus is not None:
                    if num_gpus < 0:
                        raise ExecutorException("num_gpus must be non-negative")
                    if num_gpus and num_gpus % num_procs != 0:
                        raise ExecutorException("num_gpus must be divisible by num_procs for FluxExecutor")
                    gpus_per_task = num_gpus // num_procs if num_gpus else 0

                jobspec = JobspecV1.from_command(
                    command,
                    num_tasks=num_procs,
                    num_nodes=num_nodes,
                    cores_per_task=1,
                    gpus_per_task=gpus_per_task,
                    cwd=task.workdir,
                    environment=dict(os.environ),
                )

                if stdout:
                    jobspec.stdout = os.path.join(task.workdir, stdout)
                if stderr:
                    jobspec.stderr = os.path.join(task.workdir, stderr)
                if gpus_per_task:
                    jobspec.setattr_shell_option("gpu-affinity", "per-task")

                logger.info(f"Submitting Flux job for task {task.name}: {task.runline}")
                task.flux_jobid = flux.job.submit(self.flux_handle, jobspec)
                logger.info(f"Task {task.name} submitted with Flux job ID {task.flux_jobid}")

                task.timer.start()
                task.submit_time = task.timer.tstart

                if wait_on_start:
                    self._wait_on_start(task)

            except Exception as e:
                logger.error(f"Failed to submit Flux job: {e}")
                task.state = "FAILED_TO_START"
                task.finished = True
                raise ExecutorException(f"Failed to submit Flux job: {e}")

        self.list_of_tasks.append(task)
        return task

    def _wait_on_start(self, task: FluxTask, timeout: float = 60.0) -> None:
        """Wait for a task to start running."""
        start = time.time()
        task.timer.start()
        task.submit_time = task.timer.tstart

        while task.state in ("CREATED", "WAITING"):
            time.sleep(0.1)
            task.poll()
            if time.time() - start > timeout:
                logger.warning(f"Timeout waiting for task {task.name} to start")
                break

        if not task.finished:
            task.timer.start()
            task.submit_time = task.timer.tstart

        logger.debug(f"Task {task.name} polled as {task.state} after {time.time() - start:.2f} seconds")
