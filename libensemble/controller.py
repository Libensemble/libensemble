"""
Module to launch and control running jobs.

Contains job_controller, job, and inherited classes. A job_controller can
create and manage multiple jobs. The worker or user-side code can issue
and manage jobs using the launch, poll and kill functions. Job attributes
are queried to determine status. Functions are also provided to access
and interrogate files in the job's working directory.

"""

import os
import logging
import itertools
import time

import libensemble.launcher as launcher
from libensemble.register import Register
from libensemble.mpi_resources import MPIResources

logger = logging.getLogger(__name__ + '(' + MPIResources.get_my_name() + ')')
#For debug messages in this module  - uncomment
#(see libE.py to change root logging level)
#logger.setLevel(logging.DEBUG)

STATES = """
UNKNOWN
CREATED
WAITING
RUNNING
FINISHED
USER_KILLED
FAILED""".split()


class JobControllerException(Exception): pass

def jassert(test, *args):
    if not test:
        raise JobControllerException(*args)


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

        #Status attributes
        self.state = 'CREATED' #: test1 docstring
        self.process = None
        self.errcode = None
        self.finished = False  # True means job ran, not that it succeeded
        self.success = False

        # Note: runtime, total_time, and time since launch may differ!
        self.launch_time = None
        self.runtime = None
        self.total_time = None

        #Run attributes
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

        #In case already been killed and set then
        if self.runtime is None:
            self.runtime = time.time() - self.launch_time

        #For direct launched jobs - these should be the same.
        if self.total_time is None:
            self.total_time = self.runtime

    def poll(self):
        """Polls and updates the status attributes of the job"""

        jassert(self.process is not None,
                "Polled job {} has no process ID - check jobs been launched".
                format(self.name))
        if self.finished:
            logger.warning("Polled job {} has already finished. "
                           "Not re-polling. Status is {}".
                           format(self.name, self.state))
            return

        #-------- Up to here should be common - can go in a baseclass ------#

        # Poll the job
        poll = self.process.poll()
        if poll is None:
            self.state = 'RUNNING'
            return

        self.finished = True
        self.calc_job_timing()

        # Want to be more fine-grained about non-success (fail vs user kill?)
        self.errcode = self.process.returncode
        self.success = (self.errcode == 0)
        self.state = 'FINISHED' if self.success else 'FAILED'
        logger.debug("Job {} completed with errcode {} ({})".
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

        logger.debug("Killing job {}".format(self.name))
        launcher.cancel(self.process, wait_time)
        self.state = 'USER_KILLED'
        self.finished = True
        self.calc_job_timing()


class JobController:
    """The job_controller can create, poll and kill runnable jobs

    **Class Attributes:**

    :cvar JobController: controller: The default job_controller.

    **Object Attributes:**

    :ivar Register registry: The registry associated with this job_controller
    :ivar int wait_time: Timeout period for hard kill
    :ivar list list_of_jobs: A list of jobs created in this job controller
    :ivar int workerID: The workerID associated with this job controller

    """

    controller = None

    def __init__(self, registry=None, auto_resources=True,
                 nodelist_env_slurm=None, nodelist_env_cobalt=None):
        """Instantiate a new JobController instance.

        A new JobController object is created with an application
        registry and configuration attributes. A registry object must
        have been created.

        This is typically created in the user calling script. If
        auto_resources is True, an evaluation of system resources is
        performance during this call.

        Parameters
        ----------
        registry: obj: Registry, optional
            A registry containing the applications to use in this
            job_controller (Default: Use Register.default_registry).

        auto_resources: Boolean, optional
            Auto-detect available processor resources and assign to jobs
            if not explicitly provided on launch.

        nodelist_env_slurm: String, optional
            The environment variable giving a node list in Slurm format
            (Default: Uses SLURM_NODELIST).  Note: This is only queried if
            a worker_list file is not provided and auto_resources=True.

        nodelist_env_cobalt: String, optional
            The environment variable giving a node list in Cobalt format
            (Default: Uses COBALT_PARTNAME) Note: This is only queried
            if a worker_list file is not provided and
            auto_resources=True.
        """

        self.registry = registry or Register.default_registry
        jassert(self.registry is not None, "Cannot find default registry")

        self.top_level_dir = os.getcwd()
        self.auto_resources = auto_resources
        self.manager_signal = 'none'

        if self.auto_resources:
            self.resources = \
              MPIResources(top_level_dir=self.top_level_dir,
                           nodelist_env_slurm=nodelist_env_slurm,
                           nodelist_env_cobalt=nodelist_env_cobalt)

        mpi_commands = {
            'mpich':   ['mpirun', '--env {env}', '-machinefile {machinefile}',
                        '-hosts {hostlist}', '-np {num_procs}',
                        '--ppn {ranks_per_node}'],
            'openmpi': ['mpirun', '-x {env}', '-machinefile {machinefile}',
                        '-host {hostlist}', '-np {num_procs}',
                        '-npernode {ranks_per_node}'],
        }
        self.mpi_command = mpi_commands[MPIResources.get_MPI_variant()]
        self.wait_time = 60
        self.list_of_jobs = []
        self.workerID = None
        JobController.controller = self


    def launch(self, calc_type, num_procs=None, num_nodes=None,
               ranks_per_node=None, machinefile=None, app_args=None,
               stdout=None, stderr=None, stage_inout=None,
               hyperthreads=False, test=False):
        """Creates a new job, and either launches or schedules launch.

        The created job object is returned.

        Parameters
        ----------

        calc_type: String
            The calculation type: 'sim' or 'gen'

        num_procs: int, optional
            The total number of MPI tasks on which to launch the job.

        num_nodes: int, optional
            The number of nodes on which to launch the job.

        ranks_per_node: int, optional
            The ranks per node for this job.

        machinefile: string, optional
            Name of a machinefile for this job to use.

        app_args: string, optional
            A string of the application arguments to be added to job
            launch command line.

        stdout: string, optional
            A standard output filename.

        stderr: string, optional
            A standard error filename.

        stage_inout: string, optional
            A directory to copy files from. Default will take from
            current directory.

        hyperthreads: boolean, optional
            Whether to launch MPI tasks to hyperthreads

        test: boolean, optional
            Whether this is a test - No job will be launched. Instead
            runline is printed to logger (At INFO level).


        Returns
        -------

        job: obj: Job
            The lauched job object.


        Note that if some combination of num_procs, num_nodes and
        ranks_per_node are provided, these will be honored if
        possible. If resource detection is on and these are omitted,
        then the available resources will be divided amongst workers.
        """

        app = self.registry.default_app(calc_type)
        jassert(calc_type in ['sim', 'gen'],
                "Unrecognized calculation type", calc_type)
        jassert(app, "Default {} app is not set".format(calc_type))


        #-------- Up to here should be common - can go in a baseclass ------#
        hostlist = None
        if machinefile is None and self.auto_resources:

            #kludging this for now - not nec machinefile if more than one node
            #- try a hostlist
            num_procs, num_nodes, ranks_per_node = \
              self.resources.get_resources(
                  num_procs=num_procs,
                  num_nodes=num_nodes, ranks_per_node=ranks_per_node,
                  hyperthreads=hyperthreads)

            if num_nodes > 1:
                #hostlist
                hostlist = self.resources.get_hostlist()
            else:
                #machinefile
                machinefile = "machinefile_autogen"
                if self.workerID is not None:
                    machinefile += "_for_worker_{}".format(self.workerID)
                mfile_created, num_procs, num_nodes, ranks_per_node = \
                  self.resources.create_machinefile(
                      machinefile, num_procs, num_nodes,
                      ranks_per_node, hyperthreads)
                jassert(mfile_created, "Auto-creation of machinefile failed")

        else:
            num_procs, num_nodes, ranks_per_node = \
              MPIResources.job_partition(num_procs, num_nodes,
                                         ranks_per_node, machinefile)

        default_workdir = os.getcwd()
        job = Job(app, app_args, default_workdir, stdout, stderr, self.workerID)

        if stage_inout is not None:
            logger.warning("stage_inout option ignored in this "
                           "job_controller - runs in-place")

        mpi_specs = {'num_procs': num_procs,
                     'num_nodes': num_nodes,
                     'ranks_per_node': ranks_per_node,
                     'machinefile': machinefile,
                     'hostlist': hostlist}
        runline = launcher.form_command(self.mpi_command, mpi_specs)
        runline.append(job.app.full_path)
        if job.app_args is not None:
            runline.extend(job.app_args.split())

        if test:
            logger.info('Test selected: Not launching job')
            logger.info('runline args are {}'.format(runline))
        else:
            logger.debug("Launching job {}: {}".
                         format(job.name, " ".join(runline))) #One line
            job.launch_time = time.time()
            job.process = launcher.launch(runline, cwd='./',
                                          stdout=open(job.stdout, 'w'),
                                          stderr=open(job.stderr, 'w'),
                                          start_new_session=True)
            self.list_of_jobs.append(job)

        return job


    def manager_poll(self):
        """ Polls for a manager signal

        The job controller manager_signal attribute will be updated.

        """

        #Will use MPI_MODE from settings.py but for now assume MPI
        from libensemble.message_numbers import \
            STOP_TAG, MAN_SIGNAL_FINISH, MAN_SIGNAL_KILL
        from mpi4py import MPI

        # Manager Signals
        # Stop tag may be manager interupt as diff kill/stop/pause....
        comm = MPI.COMM_WORLD
        status = MPI.Status()
        if comm.Iprobe(source=0, tag=STOP_TAG, status=status):
            logger.info('Manager probe hit true')
            man_signal = comm.recv(source=0, tag=STOP_TAG, status=status)
            if man_signal == MAN_SIGNAL_FINISH:
                self.manager_signal = 'finish'
            elif man_signal == MAN_SIGNAL_KILL:
                self.manager_signal = 'kill'
            else:
                logger.warning("Received unrecognized manager signal {} - "
                               "ignoring".format(man_signal))

    def get_job(self, jobid):
        """ Returns the job object for the supplied job ID """
        job = next((j for j in self.list_of_jobs if j.id == jobid), None)
        if job is None:
            logger.warning("Job {} not found in joblist".format(jobid))
        return job

    def set_workerID(self, workerid):
        """Sets the worker ID for this job_controller"""
        self.workerID = workerid

    def kill(self, job):
        "Kill a job"
        jassert(isinstance(job, Job), "Invalid job has been provided")
        job.kill(self.wait_time)
