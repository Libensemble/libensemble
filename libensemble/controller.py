"""
Module to launch and control running jobs.

Contains job_controller and job. The class JobController is a base class and not
intended for direct use. Instead one of the inherited classes should be used. Inherited
classes include MPI and Balsam variants. A job_controller can create and manage
multiple jobs. The worker or user-side code can issue and manage jobs using the launch,
poll and kill functions. Job attributes are queried to determine status. Functions are
also provided to access and interrogate files in the job's working directory.

"""

import os
import logging
import itertools
import time

from libensemble.message_numbers import STOP_TAG, MAN_SIGNAL_FINISH, MAN_SIGNAL_KILL
import libensemble.util.launcher as launcher
from libensemble.util.timer import JobTimer

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


class JobControllerException(Exception):
    "Raised for any exception in the JobController"


def jassert(test, *args):
    "Version of assert that raises a JobControllerException"
    if not test:
        raise JobControllerException(*args)


class Application:
    """An application is an executable user-program
    (e.g. Implementing a sim/gen function)."""

    def __init__(self, full_path, calc_type='sim', desc=None):
        """Instantiate a new Application instance."""
        self.full_path = full_path
        self.calc_type = calc_type
        self.calc_dir, self.exe = os.path.split(full_path)

        # Use this name to delete jobs in database - see del_apps(), del_jobs()
        self.name = self.exe + '.' + self.calc_type + 'func'
        self.desc = desc or (self.exe + ' ' + self.calc_type + ' function')


class Job:
    """
    Manage the creation, configuration and status of a launchable job.

    """

    newid = itertools.count()

    def __init__(self, app=None, app_args=None, workdir=None,
                 stdout=None, stderr=None, workerid=None):
        """Instantiate a new Job instance.

        A new job object is created with an id, status and configuration
        attributes.  This will normally be created by the job_controller
        on a launch
        """
        self.id = next(Job.newid)

        self.reset()  # Set status attributes
        self.timer = JobTimer()

        # Run attributes
        self.app = app
        self.app_args = app_args
        self.workerID = workerid

        jassert(app is not None,
                "Job must be created with an app - no app found for job {}".
                format(self.id))

        worker_name = "_worker{}".format(self.workerID) if self.workerID else ""
        self.name = "job_{}{}_{}".format(app.name, worker_name, self.id)
        self.stdout = stdout or self.name + '.out'
        self.stderr = stderr or self.name + '.err'
        self.workdir = workdir

    def reset(self):
        # Status attributes
        self.state = 'CREATED'
        self.process = None
        self.errcode = None
        self.finished = False  # True means job ran, not that it succeeded
        self.success = False
        self.launch_time = None
        self.runtime = 0  # Time since job started to latest poll (or finished).
        self.total_time = None  # Time from job submission until polled as finished.

    def workdir_exists(self):
        """Returns True if the job's workdir exists"""
        return self.workdir and os.path.exists(self.workdir)

    def file_exists_in_workdir(self, filename):
        """Returns True if the named file exists in the job's workdir"""
        return (self.workdir
                and os.path.exists(os.path.join(self.workdir, filename)))

    def read_file_in_workdir(self, filename):
        """Open and reads the named file in the job's workdir """
        path = os.path.join(self.workdir, filename)
        if not os.path.exists(path):
            raise ValueError("{} not found in working directory".
                             format(filename))
        with open(path) as f:
            return f.read()

    def stdout_exists(self):
        """Returns True if the job's stdout file exists in the workdir"""
        return self.file_exists_in_workdir(self.stdout)

    def read_stdout(self):
        """Open and reads the job's stdout file in the job's workdir"""
        return self.read_file_in_workdir(self.stdout)

    def stderr_exists(self):
        """Returns True if the job's stderr file exists in the workdir"""
        return self.file_exists_in_workdir(self.stderr)

    def read_stderr(self):
        """Open and reads the job's stderr file in the job's workdir"""
        return self.read_file_in_workdir(self.stderr)

    def calc_job_timing(self):
        """Calculate timing information for this job"""
        if self.launch_time is None:
            logger.warning("Cannot calc job timing - launch time not set")
            return

        # Do not update if total_time is already set
        if self.total_time is None:
            self.timer.stop()
            self.runtime = self.timer.elapsed
            self.total_time = self.runtime  # For direct launched jobs

    def check_poll(self):
        """Check whether polling this job makes sense."""
        jassert(self.process is not None,
                "Polled job {} has no process ID - check jobs been launched".
                format(self.name))
        if self.finished:
            logger.warning("Polled job {} has already finished. "
                           "Not re-polling. Status is {}".
                           format(self.name, self.state))
            return False
        return True

    def poll(self):
        """Polls and updates the status attributes of the job"""
        if not self.check_poll():
            return

        # Poll the job
        poll = self.process.poll()
        if poll is None:
            self.state = 'RUNNING'
            self.runtime = self.timer.elapsed
            return

        self.finished = True
        self.calc_job_timing()

        # Want to be more fine-grained about non-success (fail vs user kill?)
        self.errcode = self.process.returncode
        self.success = (self.errcode == 0)
        self.state = 'FINISHED' if self.success else 'FAILED'
        logger.info("Job {} completed with errcode {} ({})".
                    format(self.name, self.errcode, self.state))

    def kill(self, wait_time=60):
        """Kills or cancels the supplied job

        Sends SIGTERM, waits for a period of <wait_time> for graceful
        termination, then sends a hard kill with SIGKILL.  If <wait_time>
        is 0, we go immediately to SIGKILL; if <wait_time> is None, we
        never do a SIGKILL.
        """
        if self.finished:
            logger.warning("Trying to kill job that is no longer running. "
                           "Job {}: Status is {}".format(self.name, self.state))
            return

        if self.process is None:
            time.sleep(0.2)
            jassert(self.process is not None,
                    "Attempting to kill job {} that has no process ID - "
                    "check jobs been launched".format(self.name))

        logger.info("Killing job {}".format(self.name))
        launcher.cancel(self.process, wait_time)
        self.state = 'USER_KILLED'
        self.finished = True
        self.calc_job_timing()


class JobController:
    """The job_controller can create, poll and kill runnable jobs

    **Class Attributes:**

    :cvar JobController: controller: The default job_controller.

    **Object Attributes:**

    :ivar int wait_time: Timeout period for hard kill
    :ivar list list_of_jobs: A list of jobs created in this job controller
    :ivar int workerID: The workerID associated with this job controller

    """

    controller = None

    def _wait_on_run(self, job, fail_time=None):
        '''Called by launch when wait_on_run is True'''
        start = time.time()
        job.timer.start()  # To ensure a start time before poll - will be overwritten unless finished by poll.
        job.launch_time = job.timer.tstart
        while job.state in NOT_STARTED_STATES:
            time.sleep(0.2)
            job.poll()
        logger.debug("Job {} polled as {} after {} seconds".format(job.name, job.state, time.time()-start))
        if not job.finished:
            job.timer.start()
            job.launch_time = job.timer.tstart
            if fail_time:
                time.sleep(fail_time)
                job.poll()
                logger.debug("After {} seconds: job {} polled as {}".format(fail_time, job.name, job.state))

    def __init__(self):
        """Instantiate a new JobController instance.

        A new JobController object is created with an application
        registry and configuration attributes.

        This is typically created in the user calling script. If
        auto_resources is True, an evaluation of system resources is
        performance during this call.
        """
        self.top_level_dir = os.getcwd()
        self.manager_signal = 'none'
        self.default_apps = {'sim': None, 'gen': None}

        self.wait_time = 60
        self.list_of_jobs = []
        self.workerID = None
        JobController.controller = self

    @property
    def sim_default_app(self):
        """Return the default simulation app."""
        return self.default_apps['sim']

    @property
    def gen_default_app(self):
        """Return the default generator app."""
        return self.default_apps['gen']

    def default_app(self, calc_type):
        "Get the default app for a given calc type."
        app = self.default_apps.get(calc_type)
        jassert(calc_type in ['sim', 'gen'],
                "Unrecognized calculation type", calc_type)
        jassert(app, "Default {} app is not set".format(calc_type))
        return app

    def register_calc(self, full_path, calc_type='sim', desc=None):
        """Registers a user application to libEnsemble

        Parameters
        ----------

        full_path: String
            The full path of the user application to be registered.

        calc_type: String
            Calculation type: Is this application part of a 'sim'
            or 'gen' function.

        desc: String, optional
            Description of this application.

        """
        jassert(calc_type in self.default_apps,
                "Unrecognized calculation type", calc_type)
        jassert(self.default_apps[calc_type] is None,
                "Default {} app already set".format(calc_type))
        self.default_apps[calc_type] = Application(full_path, calc_type, desc)

    def manager_poll(self, comm):
        """ Polls for a manager signal

        The job controller manager_signal attribute will be updated.

        """

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
        comm.push_back(mtag)

    def get_job(self, jobid):
        """ Returns the job object for the supplied job ID """
        job = next((j for j in self.list_of_jobs if j.id == jobid), None)
        if job is None:
            logger.warning("Job {} not found in joblist".format(jobid))
        return job

    def set_workerID(self, workerid):
        """Sets the worker ID for this job_controller"""
        self.workerID = workerid

    def set_worker_info(self, workerid=None):
        """Sets info for this job_controller"""
        self.workerID = workerid

    def poll(self, job):
        "Polls a job"
        job.poll()

    def kill(self, job):
        "Kill a job"
        jassert(isinstance(job, Job), "Invalid job has been provided")
        job.kill(self.wait_time)
