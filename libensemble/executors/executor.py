"""
This module contains the classes ``Executor`` and ``Task``. An ``executor``
can create and manage multiple ``tasks``. ``Task`` attributes are queried to
determine status.
"""

import itertools
import logging
import os
import shutil
import stat
import sys
import time
from pathlib import Path
from typing import Any

import libensemble.utils.launcher as launcher
from libensemble.message_numbers import (
    MAN_KILL_SIGNALS,
    STOP_TAG,
    TASK_FAILED,
    TASK_FAILED_TO_START,
    UNSET_TAG,
    WORKER_DONE,
    WORKER_KILL_ON_TIMEOUT,
)
from libensemble.utils.timer import TaskTimer

logger = logging.getLogger(__name__)
# To change logging level for just this module
# logger.setLevel(logging.DEBUG)

STATES = """
UNKNOWN
CREATED
WAITING
RUNNING
FINISHED
USER_KILLED
FAILED
FAILED_TO_START""".split()

NOT_STARTED_STATES = """
CREATED
WAITING
""".split()

END_STATES = """
FINISHED
USER_KILLED
FAILED
FAILED_TO_START
""".split()


class ExecutorException(Exception):
    """Raised for any exception in the Executor"""


class TimeoutExpired(Exception):
    """Timeout exception raised when Timeout expires"""

    def __init__(self, task: str, timeout: float) -> None:
        self.task = task
        self.timeout = timeout

    def __str__(self):
        return f"Task {self.task} timed out after {self.timeout} seconds"


class Application:
    """An application is an executable user-program
    (e.g., implementing a sim/gen function)."""

    prefix = "libe_app"

    def __init__(
        self,
        full_path: str,
        name: str | None = None,
        calc_type: str | None = "sim",
        desc: str | None = None,
        pyobj: Any | None = None,
        precedent: str = "",
    ) -> None:
        """Instantiates a new Application instance."""
        self.full_path = full_path
        self.calc_type = calc_type
        self.calc_dir, self.exe = os.path.split(full_path)
        self.precedent = precedent

        if self.exe.endswith(".py"):
            if not precedent:
                self.precedent = sys.executable

        self.name = name or self.exe
        self.pyobj = pyobj
        self.desc = desc or (self.exe + " app")
        self.gname = "_".join([Application.prefix, self.name])
        self.app_cmd = " ".join(filter(None, [self.precedent, self.full_path]))


def jassert(test: Application | bool | None, *args) -> None:
    "Version of assert that raises a ExecutorException"
    if not test:
        raise ExecutorException(*args)


class Task:
    """
    Manages the creation, configuration and status of a launchable task
    """

    prefix = "libe_task"
    newid = itertools.count()

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
        """Instantiate a new Task instance.

        A new task object is created with an id, status and configuration
        attributes.  This will normally be created by the executor
        on a submission
        """
        self.id = next(Task.newid)

        self.reset()  # Set status attributes
        self.timer = TaskTimer()

        # Run attributes
        self.app = app
        self.app_args = app_args
        self.workerID = workerid

        jassert(app is not None, f"Task must be created with an app - no app found for task {self.id}")

        worker_name = f"_worker{self.workerID}" if self.workerID else ""
        self.name = Task.prefix + f"_{app.name}{worker_name}_{self.id}"
        self.stdout = stdout or self.name + ".out"
        self.stderr = stderr or self.name + ".err"
        self.workdir = workdir
        self.dry_run = dry_run
        self.runline = None
        self.run_attempts = 0
        self.env = {}
        self.ngpus_req = 0

    def reset(self) -> None:
        # Status attributes
        self.state = "CREATED"
        self.process = None
        self.errcode = None
        self.finished = False  # True means task ran, not that it succeeded
        self.success = False
        self.submit_time = None
        self.runtime = 0  # Time since task started to latest poll (or finished).
        self.total_time = None  # Time from task submission until polled as finished.
        self.ngpus_req = 0

    def _add_to_env(self, key, value):
        """Add to task environment - overwrites if already set"""
        self.env[key] = value

    def workdir_exists(self) -> bool | None:
        """Returns true if the task's workdir exists"""
        return self.workdir and os.path.exists(self.workdir)

    def file_exists_in_workdir(self, filename: str) -> bool:
        """Returns true if the named file exists in the task's workdir"""
        return self.workdir and os.path.exists(os.path.join(self.workdir, filename))

    def read_file_in_workdir(self, filename: str) -> str:
        """Opens and reads the named file in the task's workdir"""
        path = os.path.join(self.workdir, filename)
        if not os.path.exists(path):
            raise ValueError(f"{filename} not found in working directory")
        with open(path) as f:
            return f.read()

    def stdout_exists(self) -> bool:
        """Returns true if the task's stdout file exists in the workdir"""
        return self.file_exists_in_workdir(self.stdout)

    def read_stdout(self) -> str:
        """Opens and reads the task's stdout file in the task's workdir"""
        return self.read_file_in_workdir(self.stdout)

    def stderr_exists(self) -> bool:
        """Returns true if the task's stderr file exists in the workdir"""
        return self.file_exists_in_workdir(self.stderr)

    def read_stderr(self) -> str:
        """Opens and reads the task's stderr file in the task's workdir"""
        return self.read_file_in_workdir(self.stderr)

    def calc_task_timing(self) -> None:
        """Calculate timing information for this task"""
        if self.submit_time is None:
            logger.warning("Cannot calc task timing - submit time not set")
            return

        # Do not update if total_time is already set
        if self.total_time is None:
            self.timer.stop()
            self.runtime = self.timer.elapsed
            self.total_time = self.runtime  # For direct launched tasks

    def _implement_env(self):
        """Set environment variables for this task"""
        if self.env:
            logger.debug(f"Task: {self.name}: Setting environment vars {self.env}")
        for k, v in self.env.items():
            os.environ[k] = v

    def _check_poll(self) -> bool:
        """Check whether polling this task makes sense."""
        jassert(self.process is not None, f"task {self.name} has no process ID - check task has been launched")
        if self.finished:
            logger.debug(f"Polled task {self.name} has already finished. Not re-polling. Status is {self.state}")
            return False
        return True

    def _set_complete(self) -> None:
        """Set task as complete"""
        self.finished = True
        if self.dry_run:
            self.success = True
            self.state = "FINISHED"
        else:
            self.calc_task_timing()
            self.errcode = self.process.returncode
            self.success = self.errcode == 0
            self.state = "FINISHED" if self.success else "FAILED"
            logger.info(f"Task {self.name} finished with errcode {self.errcode} ({self.state})")

    def poll(self) -> None:
        """Polls and updates the status attributes of the task"""
        if self.dry_run:
            self._set_complete()
            return

        if not self._check_poll():
            return

        # Poll the task
        poll = self.process.poll()
        if poll is None:
            self.state = "RUNNING"
            self.runtime = self.timer.elapsed
            return

        self._set_complete()

    def wait(self, timeout: float | None = None) -> None:
        """Waits on completion of the task or raises TimeoutExpired exception

        Status attributes of task are updated on completion.

        Parameters
        ----------

        timeout: int or float,  Optional
            Time in seconds after which a TimeoutExpired exception is raised.
            If not set, then simply waits until completion.
            Note that the task is not automatically killed on timeout.
        """

        if self.dry_run:
            self._set_complete()
            return

        if not self._check_poll():
            return

        # Wait on the task
        rc = launcher.wait(self.process, timeout)
        if rc is None:
            raise TimeoutExpired(self.name, timeout)

        self._set_complete()

    def result(self, timeout: int | float | None = None) -> str:
        """Wrapper for task.wait() that also returns the task's status on completion.

        Parameters
        ----------

        timeout: int or float,  Optional
            Time in seconds after which a TimeoutExpired exception is raised.
            If not set, then simply waits until completion.
            Note that the task is not automatically killed on timeout.
        """

        self.wait(timeout=timeout)
        return self.state

    def exception(self, timeout: int | float | None = None):
        """Wrapper for task.wait() that instead returns the task's error code on completion.

        Parameters
        ----------

        timeout: int or float,  Optional
            Time in seconds after which a TimeoutExpired exception is raised.
            If not set, then simply waits until completion.
            Note that the task is not automatically killed on timeout.
        """

        self.wait(timeout=timeout)
        return self.errcode

    def running(self) -> bool:
        """Return ``True`` if task is currently running."""
        self.poll()
        return self.state == "RUNNING"

    def done(self) -> bool:
        """Return ``True`` if task is finished."""
        self.poll()
        return self.finished

    def kill(self, wait_time: int = 60) -> None:
        """Kills or cancels the supplied task

        Parameters
        ----------

        wait_time: int, Optional
            Time in seconds to wait for termination between sending
            SIGTERM and a SIGKILL signals.


        Sends SIGTERM, waits for a period of <wait_time> for graceful
        termination, then sends a hard kill with SIGKILL.  If <wait_time>
        is 0, we go immediately to SIGKILL; if <wait_time> is none, we
        never do a SIGKILL.
        """
        self.poll()
        if self.dry_run:
            return

        if self.finished:
            logger.warning(f"Trying to kill task that is no longer running. Task {self.name}: Status is {self.state}")
            return

        if self.process is None:
            time.sleep(0.2)
            jassert(
                self.process is not None,
                f"task {self.name} has no process ID - check task has been launched",
            )

        logger.info(f"Killing task {self.name}")
        launcher.cancel(self.process, wait_time)
        self.state = "USER_KILLED"
        self.finished = True
        self.calc_task_timing()

    def cancel(self) -> None:
        """Wrapper for task.kill() without waiting"""
        self.kill(wait_time=None)

    def cancelled(self) -> bool:
        """Return ```True`` if task successfully cancelled."""
        self.poll()
        return self.state == "USER_KILLED"


class Executor:
    """The executor can create, poll and kill runnable tasks

    **Class Attributes:**

    :cvar Executor: executor: The executor object is stored here and can be retrieved in user functions.

    """

    executor = None

    def _wait_on_start(self, task: Task, fail_time: int | None = None) -> None:
        """Called by submit when wait_on_start is True.

        Blocks until task polls as having started.
        If fail_time is supplied, will also block until either task is in an
        end state or fail_time has expired.
        """
        start = time.time()
        task.timer.start()  # To ensure a start time before poll - will be overwritten unless finished by poll.
        task.submit_time = task.timer.tstart
        while task.state in NOT_STARTED_STATES:
            time.sleep(0.001)
            task.poll()
        logger.debug(f"Task {task.name} polled as {task.state} after {time.time() - start} seconds")
        if not task.finished:
            task.timer.start()
            task.submit_time = task.timer.tstart
            if fail_time:
                remaining = fail_time - task.timer.elapsed
                while task.state not in END_STATES and remaining > 0:
                    time.sleep(min(0.01, remaining))
                    task.poll()
                    remaining = fail_time - task.timer.elapsed
                logger.debug(f"After {task.timer.elapsed} seconds: task {task.name} polled as {task.state}")

    def __init__(self) -> None:
        """Instantiate a new Executor instance.

        Returns
        -------

        Executor
            A new Executor object is created.
            This is typically created in the user calling script.

        """

        self.manager_signal = None
        self.default_apps = {"sim": None, "gen": None}
        self.apps = {}

        self.wait_time = 60
        self.list_of_tasks = []
        self.workerID = None
        self.comm = None
        self.last_task = 0
        Executor.executor = self

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        pass

    def serial_setup(self):
        """Set up to be called by only one process"""
        pass  # To be overloaded

    @property
    def sim_default_app(self) -> Application:
        """Returns the default simulation app"""
        return self.default_apps["sim"]

    @property
    def gen_default_app(self) -> Application:
        """Returns the default generator app"""
        return self.default_apps["gen"]

    def get_app(self, app_name: str) -> Application:
        """Gets the app for a given app_name or raise exception"""
        try:
            app = self.apps[app_name]
        except KeyError:
            app_keys = list(self.apps.keys())
            raise ExecutorException(
                f"Application {app_name} not found in registry", f"Registered applications: {app_keys}"
            )
        return app

    def default_app(self, calc_type: str) -> Application:
        """Gets the default app for a given calc type"""
        app = self.default_apps.get(calc_type)
        jassert(calc_type in ["sim", "gen"], "Unrecognized calculation type", calc_type)
        jassert(app, f"Default {calc_type} app is not set")
        return app

    def set_resources(self, resources):
        # Does not use resources
        pass

    def add_platform_info(self, platform_info={}):
        """Add user supplied platform info to executor

        Base executor does not currently use platform info
        """
        pass

    def set_gen_procs_gpus(self, libE_info):
        """Add gen supplied procs and gpus

        Base executor does not currently use procs and gpus
        """
        pass

    def register_app(
        self,
        full_path: str,
        app_name: str | None = None,
        calc_type: str | None = None,
        desc: str | None = None,
        precedent: str = "",
    ) -> None:
        """Registers a user application to libEnsemble.

        The ``full_path`` of the application must be supplied. Either
        ``app_name`` or ``calc_type`` can be used to identify the
        application in user scripts (in the **submit** function).
        ``app_name`` is recommended.

        Parameters
        ----------

        full_path: str
            The full path of the user application to be registered

        app_name: str, Optional
            Name to identify this application.

        calc_type: str, Optional
            Calculation type: Set this application as the default 'sim'
            or 'gen' function.

        desc: str, Optional
            Description of this application

        precedent: str, Optional
            Any str that should directly precede the application full path.
        """

        if not app_name:
            app_name = os.path.split(full_path)[1]
        self.apps[app_name] = Application(full_path, app_name, calc_type, desc, None, precedent)

        # Default sim/gen apps will be deprecated. Just use names.
        if calc_type is not None:
            jassert(calc_type in self.default_apps, "Unrecognized calculation type", calc_type)
            self.default_apps[calc_type] = self.apps[app_name]

    def manager_poll(self) -> int:
        """
        .. _manager_poll_label:

        Polls for a manager signal

        The executor manager_signal attribute will be updated.
        """

        self.manager_signal = None  # Reset

        # Check for messages; disregard anything but a stop signal
        if not self.comm.mail_flag():
            return
        mtag, man_signal = self.comm.recv()
        if mtag != STOP_TAG:
            return

        # Process the signal and push back on comm (for now)
        self.manager_signal = man_signal

        if man_signal in MAN_KILL_SIGNALS:
            # Only kill signals exist currently
            logger.info(f"Worker received kill signal {man_signal} from manager")
        else:
            logger.warning(f"Received unrecognized manager signal {man_signal} - ignoring")
        self.comm.push_to_buffer(mtag, man_signal)
        return man_signal

    def manager_kill_received(self) -> bool:
        """Return True if received kill signal from the manager"""
        man_signal = self.manager_poll()
        if man_signal in MAN_KILL_SIGNALS:
            return True
        return False

    def polling_loop(
        self, task: Task, timeout: int | None = None, delay: float = 0.1, poll_manager: bool = False
    ) -> int:
        """Optional, blocking, generic task status polling loop. Operates until the task
        finishes, times out, or is optionally killed via a manager signal. On completion, returns a
        presumptive :ref:`calc_status<funcguides-calcstatus>` integer. Useful
        for running an application via the Executor until it stops without monitoring
        its intermediate output.

        Parameters
        ----------

        task: object
            a Task object returned by the executor on submission

        timeout: int, Optional
            Maximum number of seconds for the polling loop to run. Tasks that run
            longer than this limit are killed. Default: No timeout

        delay: int, Optional
            Sleep duration between polling loop iterations. Default: 0.1 seconds

        poll_manager: bool, Optional
            Whether to also poll the manager for 'finish' or 'kill' signals.
            If detected, the task is killed. Default: False.

        Returns
        -------
        calc_status: int
            presumptive integer attribute describing the final status of a launched task
        """

        calc_status = UNSET_TAG

        while not task.finished:
            try:
                task.poll()
            except ExecutorException as e:
                logger.warning(f"Exception in polling_loop: {e}")
                break

            if poll_manager:
                man_signal = self.manager_poll()
                if self.manager_signal in MAN_KILL_SIGNALS:
                    task.kill()
                    calc_status = man_signal
                    break

            if timeout is not None and task.runtime > timeout:
                task.kill()
                calc_status = WORKER_KILL_ON_TIMEOUT
                break

            time.sleep(delay)

        if calc_status == UNSET_TAG:
            if task.state == "FINISHED":
                calc_status = WORKER_DONE
            elif task.state == "FAILED_TO_START":
                calc_status = TASK_FAILED_TO_START
            elif task.state == "FAILED":
                calc_status = TASK_FAILED
            else:
                logger.warning(f"Warning: Task {task.name} in unknown state {task.state}. Error code {task.errcode}")

        return calc_status

    def get_task(self, taskid: str | int) -> Task | None:
        """Returns the task object for the supplied task ID"""
        task = next((j for j in self.list_of_tasks if j.id == taskid), None)
        if task is None:
            logger.warning(f"Task {taskid} not found in tasklist")
        return task

    def new_tasks_timing(self, datetime=False) -> str:
        """Returns timing of new tasks as a str

        Parameters
        ----------

        datetime: bool
            If True, returns start and end times in addition to elapsed time.
        """

        timing_msg = ""
        if self.list_of_tasks:
            start_task = self.last_task
            for i, task in enumerate(self.list_of_tasks[start_task:]):
                if datetime:
                    timing_msg += f" Task {i}: {task.timer}"
                else:
                    timing_msg += f" Task {i}: {task.timer.summary()}"
                self.last_task += 1
        return timing_msg

    def set_workerID(self, workerid) -> None:
        """Sets the worker ID for this executor"""
        self.workerID = workerid

    def set_worker_info(self, comm=None, workerid=None) -> None:
        """Sets info for this executor"""
        self.workerID = workerid
        self.comm = comm

    def _check_app_exists(self, full_path: str) -> None:
        """Allows submit function to check if app exists and error if not"""
        if not os.path.isfile(full_path):
            raise ExecutorException(f"Application does not exist {full_path}")

    def submit(
        self,
        calc_type: str | None = None,
        app_name: str | None = None,
        app_args: str | None = None,
        stdout: str | None = None,
        stderr: str | None = None,
        dry_run: bool | None = False,
        wait_on_start: bool | None = False,
        env_script: str | None = None,
    ) -> Task:
        """Create a new task and run as a local serial subprocess.

        The created :class:`task<libensemble.executors.executor.Task>` object is returned.

        Parameters
        ----------

        calc_type: str, Optional
            The calculation type: 'sim' or 'gen'
            Only used if app_name is not supplied. Uses default sim or gen application.

        app_name: str, Optional
            The application name. Must be supplied if calc_type is not.

        app_args: str, Optional
            A str of the application arguments to be added to task
            submit command line

        stdout: str, Optional
            A standard output filename

        stderr: str, Optional
            A standard error filename

        dry_run: bool, Optional
            Whether this is a dry_run - no task will be launched; instead
            runline is printed to logger (at INFO level)

        wait_on_start: bool, Optional
            Whether to wait for task to be polled as RUNNING (or other
            active/end state) before continuing. If an integer N is supplied,
            wait at most N seconds.

        env_script: str, Optional
            The full path of a shell script to set up the environment for the
            launched task. This will be run in the subprocess, and not affect
            the worker environment. The script should start with a shebang.

        Returns
        -------

        task: Task
            The launched task object
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

        runline = task.app.app_cmd.split()
        if task.app_args is not None:
            runline.extend(task.app_args.split())

        if dry_run:
            logger.info(f"Test (No submit) Runline: {' '.join(runline)}")
        else:
            if env_script is not None:
                run_cmd = Executor._process_env_script(task, runline, env_script)
            else:
                run_cmd = runline

            # Set environment variables and launch task
            task._implement_env()

            # Launch Task
            logger.info(f"Launching task {task.name}: {' '.join(runline)}")
            with open(task.stdout, "w") as out, open(task.stderr, "w") as err:
                task.process = launcher.launch(
                    run_cmd,
                    cwd="./",
                    stdout=out,
                    stderr=err,
                    start_new_session=False,
                )
            if wait_on_start:
                self._wait_on_start(task, wait_on_start)

            if not task.timer.timing and not task.finished:
                task.timer.start()
                task.submit_time = task.timer.tstart  # Time not date - may not need if using timer.

            self.list_of_tasks.append(task)
        return task

    def poll(self, task: Task) -> None:
        """Polls the supplied task"""
        task.poll()

    def kill(self, task: Task) -> None:
        """Kills the supplied task"""
        jassert(isinstance(task, Task), "Invalid task has been provided")
        task.kill(self.wait_time)

    @staticmethod
    def _process_env_script(task, runline, env_script):
        """Merge users environment script with generated run-line"""
        sout_f = task.name + "_run.sh"
        p = Path(".")
        shutil.copy(env_script, p / sout_f)
        st = os.stat(sout_f)
        os.chmod(sout_f, st.st_mode | stat.S_IEXEC)
        run_line_str = " ".join(runline)

        with open(sout_f, "a") as sout:
            sout.write(run_line_str)

        run_str = "./" + sout_f
        run_cmd = run_str.split()
        return run_cmd
