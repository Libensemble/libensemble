"""
This module contains an
``executor`` and ``task``. The class ``Executor`` is a base class and not
intended for direct use. Instead one of the inherited classes should be used. Inherited
classes include MPI and Balsam variants. A ``executor`` can create and manage
multiple ``tasks``. The worker or user-side code can issue and manage ``tasks`` using the submit,
poll and kill functions. ``Task`` attributes are queried to determine status. Functions are
also provided to access and interrogate files in the ``task``'s working directory.

"""

import os
import sys
import logging
import itertools
import time

from libensemble.message_numbers import (UNSET_TAG, MAN_SIGNAL_FINISH,
                                         MAN_SIGNAL_KILL, WORKER_DONE,
                                         TASK_FAILED, WORKER_KILL_ON_TIMEOUT,
                                         WORKER_KILL, STOP_TAG)

import libensemble.utils.launcher as launcher
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
FAILED""".split()

NOT_STARTED_STATES = '''
CREATED
WAITING
'''.split()

END_STATES = '''
FINISHED
USER_KILLED
FAILED
'''.split()


class ExecutorException(Exception):
    "Raised for any exception in the Executor"


class TimeoutExpired(Exception):
    """Timeout exception raised when Timeout expires"""
    def __init__(self, task, timeout):
        self.task = task
        self.timeout = timeout

    def __str__(self):
        return ("Task {} timed out after {} seconds".format(self.task, self.timeout))


def jassert(test, *args):
    "Version of assert that raises a ExecutorException"
    if not test:
        raise ExecutorException(*args)


class Application:
    """An application is an executable user-program
    (e.g., implementing a sim/gen function)."""

    prefix = 'libe_app'

    def __init__(self, full_path, name=None, calc_type='sim', desc=None):
        """Instantiates a new Application instance."""
        self.full_path = full_path
        self.calc_type = calc_type
        self.calc_dir, self.exe = os.path.split(full_path)

        if self.exe.endswith('.py'):
            self.full_path = ' '.join((sys.executable, full_path))
        self.name = name or self.exe
        self.desc = desc or (self.exe + ' app')
        self.gname = '_'.join([Application.prefix, self.name])


class Task:
    """
    Manages the creation, configuration and status of a launchable task

    """

    prefix = 'libe_task'
    newid = itertools.count()

    def __init__(self, app=None, app_args=None, workdir=None,
                 stdout=None, stderr=None, workerid=None, dry_run=False):
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

        jassert(app is not None,
                "Task must be created with an app - no app found for task {}".
                format(self.id))

        worker_name = "_worker{}".format(self.workerID) if self.workerID else ""
        self.name = Task.prefix + "_{}{}_{}".format(app.name, worker_name, self.id)
        self.stdout = stdout or self.name + '.out'
        self.stderr = stderr or self.name + '.err'
        self.workdir = workdir
        self.dry_run = dry_run
        self.runline = None
        self.run_attempts = 0

    def reset(self):
        # Status attributes
        self.state = 'CREATED'
        self.process = None
        self.errcode = None
        self.finished = False  # True means task ran, not that it succeeded
        self.success = False
        self.submit_time = None
        self.runtime = 0  # Time since task started to latest poll (or finished).
        self.total_time = None  # Time from task submission until polled as finished.

    def workdir_exists(self):
        """Returns true if the task's workdir exists"""
        return self.workdir and os.path.exists(self.workdir)

    def file_exists_in_workdir(self, filename):
        """Returns true if the named file exists in the task's workdir"""
        return (self.workdir
                and os.path.exists(os.path.join(self.workdir, filename)))

    def read_file_in_workdir(self, filename):
        """Opens and reads the named file in the task's workdir """
        path = os.path.join(self.workdir, filename)
        if not os.path.exists(path):
            raise ValueError("{} not found in working directory".
                             format(filename))
        with open(path) as f:
            return f.read()

    def stdout_exists(self):
        """Returns true if the task's stdout file exists in the workdir"""
        return self.file_exists_in_workdir(self.stdout)

    def read_stdout(self):
        """Opens and reads the task's stdout file in the task's workdir"""
        return self.read_file_in_workdir(self.stdout)

    def stderr_exists(self):
        """Returns true if the task's stderr file exists in the workdir"""
        return self.file_exists_in_workdir(self.stderr)

    def read_stderr(self):
        """Opens and reads the task's stderr file in the task's workdir"""
        return self.read_file_in_workdir(self.stderr)

    def calc_task_timing(self):
        """Calculate timing information for this task"""
        if self.submit_time is None:
            logger.warning("Cannot calc task timing - submit time not set")
            return

        # Do not update if total_time is already set
        if self.total_time is None:
            self.timer.stop()
            self.runtime = self.timer.elapsed
            self.total_time = self.runtime  # For direct launched tasks

    def _check_poll(self):
        """Check whether polling this task makes sense."""
        jassert(self.process is not None,
                "Polled task {} has no process ID - check tasks been launched".
                format(self.name))
        if self.finished:
            logger.debug("Polled task {} has already finished. "
                         "Not re-polling. Status is {}".
                         format(self.name, self.state))
            return False
        return True

    def _set_complete(self, dry_run=False):
        """Set task as complete"""
        self.finished = True
        if dry_run:
            self.success = True
            self.state = 'FINISHED'
        else:
            self.calc_task_timing()
            self.errcode = self.process.returncode
            self.success = (self.errcode == 0)
            self.state = 'FINISHED' if self.success else 'FAILED'
            logger.info("Task {} finished with errcode {} ({})".
                        format(self.name, self.errcode, self.state))

    def poll(self):
        """Polls and updates the status attributes of the task"""
        if self.dry_run:
            return

        if not self._check_poll():
            return

        # Poll the task
        poll = self.process.poll()
        if poll is None:
            self.state = 'RUNNING'
            self.runtime = self.timer.elapsed
            return

        self._set_complete()

    def manager_poll(self, comm):
        """ Polls for a manager signal

        The task manager_signal attribute will be updated.

        Parameters
        ----------

        comm: Communicator object
            Communicator object from ``libE_info['comm']``

        Returns
        -------
        man_signal: int
            Exact signal received from manager

        """
        self.manager_signal = 'none'  # Reset

        # Check for messages; disregard anything but a stop signal
        if not comm.mail_flag():
            return
        mtag, man_signal = comm.recv()
        if mtag != STOP_TAG:
            return

        # Process the signal and push back on comm (for now)
        logger.info('Manager probe hit true')
        if man_signal == MAN_SIGNAL_FINISH:
            self.manager_signal = 'finish'
        elif man_signal == MAN_SIGNAL_KILL:
            self.manager_signal = 'kill'
        else:
            logger.warning("Received unrecognized manager signal {} - "
                           "ignoring".format(man_signal))
        comm.push_to_buffer(mtag, man_signal)
        return man_signal

    def polling_loop(self, time_limit=0, delay=1, poll_manager=False, comm=None):
        """ Optional, generic task status polling loop. Operates until the task
        either finishes or times out. Periodically polls the task
        and optionally the manager for signals. On completion, returns a
        :ref:`calc_status<datastruct-calc-status>` integer.

        Parameters
        ----------

        time_limit: int
            Maximum number of seconds for the polling loop to run. Tasks that run
            longer than this limit are killed.

        delay: int
            Sleep duration between polling loop iterations

        poll_manager: bool
            Whether to also poll the manager for 'finish' or 'kill' signals.
            If detected, the task is killed. Requires the user to also provide
            a communicator object from ``libE_info``.

        comm: Communicator object
            Communicator object from ``libE_info['comm']``

        Returns
        -------
        calc_status: int
            integer attribute describing the status of a launched task

        """
        calc_status = UNSET_TAG

        while not self.finished:
            self.poll()

            if poll_manager and comm:
                man_signal = self.manager_poll(comm)
                if self.manager_signal != 'none':
                    self.kill()
                    calc_status = man_signal
                    break

            if time_limit > 0 and self.runtime > time_limit:
                self.kill()
                calc_status = WORKER_KILL_ON_TIMEOUT
                break

            time.sleep(delay)

        if calc_status == UNSET_TAG:
            if self.state == 'FINISHED':
                calc_status = WORKER_DONE
            elif self.state == 'FAILED':
                calc_status = TASK_FAILED
            elif self.state == 'USER_KILLED':
                calc_status = WORKER_KILL
            else:
                logger.warning("Warning: Task {} in unknown state {}. Error code {}"
                               .format(self.name, self.state, self.errcode))

        return calc_status

    def wait(self, timeout=None):
        """Waits on completion of the task or raises TimeoutExpired exception

        Status attributes of task are updated on completion.

        Parameters
        ----------

        timeout:
            Time in seconds after which a TimeoutExpired exception is raised"""

        if self.dry_run:
            return

        if not self._check_poll():
            return

        # Wait on the task
        rc = launcher.wait(self.process, timeout)
        if rc is None:
            raise TimeoutExpired(self.name, timeout)

        self._set_complete()

    def kill(self, wait_time=60):
        """Kills or cancels the supplied task

        Sends SIGTERM, waits for a period of <wait_time> for graceful
        termination, then sends a hard kill with SIGKILL.  If <wait_time>
        is 0, we go immediately to SIGKILL; if <wait_time> is none, we
        never do a SIGKILL.
        """
        if self.dry_run:
            return

        if self.finished:
            logger.warning("Trying to kill task that is no longer running. "
                           "Task {}: Status is {}".format(self.name, self.state))
            return

        if self.process is None:
            time.sleep(0.2)
            jassert(self.process is not None,
                    "Attempting to kill task {} that has no process ID - "
                    "check tasks been launched".format(self.name))

        logger.info("Killing task {}".format(self.name))
        launcher.cancel(self.process, wait_time)
        self.state = 'USER_KILLED'
        self.finished = True
        self.calc_task_timing()


class Executor:
    """The executor can create, poll and kill runnable tasks

    **Class Attributes:**

    :cvar Executor: executor: The default executor.

    **Object Attributes:**

    :ivar int wait_time: Timeout period for hard kill
    :ivar list list_of_tasks: A list of tasks created in this executor
    :ivar int workerID: The workerID associated with this executor

    """

    executor = None

    def _wait_on_run(self, task, fail_time=None):
        '''Called by submit when wait_on_run is True.

        Blocks until task polls as having started.
        If fail_time is supplied, will also block until either task is in an
        end state or fail_time has expired.
        '''
        start = time.time()
        task.timer.start()  # To ensure a start time before poll - will be overwritten unless finished by poll.
        task.submit_time = task.timer.tstart
        while task.state in NOT_STARTED_STATES:
            time.sleep(0.2)
            task.poll()
        logger.debug("Task {} polled as {} after {} seconds".format(task.name, task.state, time.time()-start))
        if not task.finished:
            task.timer.start()
            task.submit_time = task.timer.tstart
            if fail_time:
                remaining = fail_time - task.timer.elapsed
                while task.state not in END_STATES and remaining > 0:
                    time.sleep(min(1.0, remaining))
                    task.poll()
                    remaining = fail_time - task.timer.elapsed
                logger.debug("After {} seconds: task {} polled as {}".format(task.timer.elapsed, task.name, task.state))

    def __init__(self):
        """Instantiate a new Executor instance.

        A new Executor object is created with an application
        registry and configuration attributes.

        This is typically created in the user calling script. If
        auto_resources is True, an evaluation of system resources is
        performance during this call.
        """
        self.top_level_dir = os.getcwd()
        self.manager_signal = 'none'
        self.default_apps = {'sim': None, 'gen': None}
        self.apps = {}

        self.wait_time = 60
        self.list_of_tasks = []
        self.workerID = None
        Executor.executor = self

    def _serial_setup(self):
        pass  # To be overloaded

    @property
    def sim_default_app(self):
        """Returns the default simulation app"""
        return self.default_apps['sim']

    @property
    def gen_default_app(self):
        """Returns the default generator app"""
        return self.default_apps['gen']

    def get_app(self, app_name):
        """Gets the app for a given app_name or raise exception"""
        try:
            app = self.apps[app_name]
        except KeyError:
            app_keys = list(self.apps.keys())
            raise ExecutorException("Application {} not found in registry".format(app_name),
                                    "Registered applications: {}".format(app_keys))
        return app

    def default_app(self, calc_type):
        """Gets the default app for a given calc type"""
        app = self.default_apps.get(calc_type)
        jassert(calc_type in ['sim', 'gen'],
                "Unrecognized calculation type", calc_type)
        jassert(app, "Default {} app is not set".format(calc_type))
        return app

    def register_calc(self, full_path, app_name=None, calc_type=None, desc=None):
        """Registers a user application to libEnsemble

        Parameters
        ----------

        app_name: String
            Name to identify this application.

        full_path: String
            The full path of the user application to be registered

        calc_type: String
            Calculation type: Set this application as the default 'sim'
            or 'gen' function.

        desc: String, optional
            Description of this application

        """
        if not app_name:
            app_name = os.path.split(full_path)[1]
        self.apps[app_name] = Application(full_path, app_name, calc_type, desc)

        # Default sim/gen apps will be deprecated. Just use names.
        if calc_type is not None:
            jassert(calc_type in self.default_apps,
                    "Unrecognized calculation type", calc_type)
            self.default_apps[calc_type] = self.apps[app_name]

    def manager_poll(self, comm):
        """ Polls for a manager signal

        The executor manager_signal attribute will be updated.

        """
        self.manager_signal = 'none'  # Reset

        # Check for messages; disregard anything but a stop signal
        if not comm.mail_flag():
            return
        mtag, man_signal = comm.recv()
        if mtag != STOP_TAG:
            return

        # Process the signal and push back on comm (for now)
        logger.info('Manager probe hit true')
        if man_signal == MAN_SIGNAL_FINISH:
            self.manager_signal = 'finish'
        elif man_signal == MAN_SIGNAL_KILL:
            self.manager_signal = 'kill'
        else:
            logger.warning("Received unrecognized manager signal {} - "
                           "ignoring".format(man_signal))
        comm.push_to_buffer(mtag, man_signal)

    def get_task(self, taskid):
        """ Returns the task object for the supplied task ID """
        task = next((j for j in self.list_of_tasks if j.id == taskid), None)
        if task is None:
            logger.warning("Task {} not found in tasklist".format(taskid))
        return task

    def set_workerID(self, workerid):
        """Sets the worker ID for this executor"""
        self.workerID = workerid

    def set_worker_info(self, workerid=None):
        """Sets info for this executor"""
        self.workerID = workerid

    def poll(self, task):
        "Polls a task"
        task.poll()

    def kill(self, task):
        "Kills a task"
        jassert(isinstance(task, Task), "Invalid task has been provided")
        task.kill(self.wait_time)
