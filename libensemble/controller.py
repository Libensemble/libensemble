#!/usr/bin/env python

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
from libensemble.resources import Resources

logger = logging.getLogger(__name__ + '(' + Resources.get_my_name() + ')')
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

    def __init__(self, app=None, app_args=None, num_procs=None, num_nodes=None,
                 ranks_per_node=None, machinefile=None, hostlist=None,
                 workdir=None, stdout=None, stderr=None, workerid=None):
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
        self.launch_time = None
        self.runtime = None
        self.total_time = None
        self.manager_signal = 'none'

        #Run attributes
        self.app = app
        self.app_args = app_args
        self.num_procs = num_procs
        self.num_nodes = num_nodes
        self.ranks_per_node = ranks_per_node
        self.machinefile = machinefile
        self.hostlist = hostlist
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

    #Note - this is currently only final job-time. May make running job time.
    #And prob want to use for polling in sim func - esp in balsam -
    #where want acutal runtime not time since launch
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


class JobController:
    """The job_controller can create, poll and kill runnable jobs

    **Class Attributes:**

    :cvar JobController: controller: A class attribute holding the default job_controller.

    **Object Attributes:**

    :ivar Register registry: The registry associated with this job_controller
    :ivar String manager_signal: Contains any signals received by manager ('none'|'finish'|'kill')
    :ivar String kill_signal: The kill signal to be sent to jobs
    :ivar boolean wait_and_kill: Whether running in wait_and_kill mode (If True a hard kill will be sent after a timeout period)
    :ivar int wait_time: Timeout period for hard kill, when wait_and_kill is set.
    :ivar list list_of_jobs: A list of jobs created in this job controller
    :ivar int workerID: The workerID associated with this job controller

    """

    controller = None

    @staticmethod
    def job_partition(num_procs, num_nodes, ranks_per_node, machinefile=None):
        """Takes provided nprocs/nodes/ranks and outputs working
        configuration of procs/nodes/ranks or error"""

        #If machinefile is provided - ignore everything else
        if machinefile:
            if num_procs or num_nodes or ranks_per_node:
                logger.warning("Machinefile provided - overriding "
                               "procs/nodes/ranks_per_node")
            return None, None, None

        if not num_procs:
            jassert(num_nodes and ranks_per_node,
                    "Need num_procs, num_nodes/ranks_per_node, or machinefile")
            num_procs = num_nodes * ranks_per_node

        elif not num_nodes:
            ranks_per_node = ranks_per_node or num_procs
            num_nodes = num_procs//ranks_per_node

        elif not ranks_per_node:
            ranks_per_node = num_procs//num_nodes

        jassert(num_procs == num_nodes*ranks_per_node,
                "num_procs does not equal num_nodes*ranks_per_node")
        return num_procs, num_nodes, ranks_per_node


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
            self.resources = Resources(top_level_dir=self.top_level_dir,
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
        self.mpi_command = mpi_commands[Resources.get_MPI_variant()]
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
              self.get_resources(num_procs=num_procs, num_nodes=num_nodes,
                                 ranks_per_node=ranks_per_node,
                                 hyperthreads=hyperthreads)

            if num_nodes > 1:
                #hostlist
                hostlist = self.get_hostlist()
            else:
                #machinefile
                machinefile = "machinefile_autogen"
                if self.workerID is not None:
                    machinefile += "_for_worker_{}".format(self.workerID)
                mfile_created, num_procs, num_nodes, ranks_per_node = \
                  self.create_machinefile(machinefile, num_procs, num_nodes,
                                          ranks_per_node, hyperthreads)
                jassert(mfile_created, "Auto-creation of machinefile failed")

        else:
            num_procs, num_nodes, ranks_per_node = \
              JobController.job_partition(num_procs, num_nodes,
                                          ranks_per_node, machinefile)

        default_workdir = os.getcwd()
        job = Job(app, app_args, num_procs, num_nodes, ranks_per_node,
                  machinefile, hostlist, default_workdir, stdout, stderr,
                  self.workerID)

        if stage_inout is not None:
            logger.warning("stage_inout option ignored in this "
                           "job_controller - runs in-place")

        runline = launcher.form_command(self.mpi_command, vars(job))
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


    def poll(self, job):
        """ Polls and updates the status attributes of the supplied job

        Parameters
        -----------

        job: obj: Job
            The job object.to be polled.

        """

        jassert(isinstance(job, Job), "Invalid job has been provided")
        jassert(job.process is not None,
                "Polled job {} has no process ID - check jobs been launched".
                format(job.name))
        if job.finished:
            logger.warning("Polled job {} has already finished. "
                           "Not re-polling. Status is {}".
                           format(job.name, job.state))
            return

        #-------- Up to here should be common - can go in a baseclass ------#

        # Poll the job
        poll = job.process.poll()
        if poll is None:
            job.state = 'RUNNING'
            return

        job.finished = True
        job.calc_job_timing()

        # Want to be more fine-grained about non-success (fail vs user kill?)
        job.errcode = job.process.returncode
        job.success = (job.errcode == 0)
        job.state = 'FINISHED' if job.success else 'FAILED'
        logger.debug("Job {} completed with errcode {} ({})".
                     format(job.name, job.errcode, job.state))


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


    def kill(self, job):
        """Kills or cancels the supplied job

        Parameters
        -----------

        job: obj: Job
            The job object.to be polled.

        Sends SIGTERM, waits for a period of <wait_time> for graceful
        termination, then sends a hard kill with SIGKILL.  If <wait_time>
        is 0, we go immediately to SIGKILL; if <wait_time> is None, we
        never do a SIGKILL.
        """

        jassert(isinstance(job, Job), "Invalid job has been provided")

        if job.finished:
            logger.warning("Trying to kill job that is no longer running. "
                           "Job {}: Status is {}".format(job.name, job.state))
            return

        if job.process is None:
            time.sleep(0.2)
            jassert(job.process is not None,
                    "Attempting to kill job {} that has no process ID - "
                    "check jobs been launched".format(job.name))

        logger.debug("Killing job {}".format(job.name))
        launcher.cancel(job.process, self.wait_time)
        job.state = 'USER_KILLED'
        job.finished = True
        job.calc_job_timing()


    def get_job(self, jobid):
        """ Returns the job object for the supplied job ID """
        if self.list_of_jobs:
            for job in self.list_of_jobs:
                if job.id == jobid:
                    return job
            logger.warning("Job {} not found in joblist".format(jobid))
            return None
        logger.warning("Job {} not found in joblist. Joblist is empty".
                       format(jobid))
        return None


    def set_workerID(self, workerid):
        """Sets the worker ID for this job_controller"""
        self.workerID = workerid


    #Reformat create_machinefile to use this and also use this for
    #non-machinefile cases when auto-detecting
    def get_resources(self, num_procs=None, num_nodes=None,
                      ranks_per_node=None, hyperthreads=False):
        """Reconciles user supplied options with available Worker
        resources to produce run configuration.

        Detects resources available to worker, checks if an existing
        user supplied config is valid, and fills in any missing config
        information (ie. num_procs/num_nodes/ranks_per_node)

        User supplied config options are honoured, and an exception is
        raised if these are infeasible.
        """

        node_list = self.resources.local_nodelist

        if hyperthreads:
            cores_avail_per_node = self.resources.logical_cores_avail_per_node
        else:
            cores_avail_per_node = self.resources.physical_cores_avail_per_node

        num_workers = self.resources.num_workers
        local_node_count = self.resources.local_node_count

        if num_workers > local_node_count:
            workers_per_node = self.resources.workers_per_node
            cores_avail_per_node_per_worker = \
              cores_avail_per_node//workers_per_node
        else:
            cores_avail_per_node_per_worker = cores_avail_per_node

        jassert(node_list, "Node list is empty - aborting")

        #If no decomposition supplied - use all available cores/nodes
        if not num_procs and not num_nodes and not ranks_per_node:
            num_nodes = local_node_count
            ranks_per_node = cores_avail_per_node_per_worker
            logger.debug("No decomposition supplied - "
                         "using all available resource. "
                         "Nodes: {}  ranks_per_node {}".
                         format(num_nodes, ranks_per_node))
        elif not num_nodes and not ranks_per_node:
            num_nodes = local_node_count
            #Here is where really want a compact/scatter option - go for
            #scatter (could get cores and say if less than one node - but then
            #hyperthreads complication if no psutil installed)
        elif not num_procs and not ranks_per_node:
            #Who would just put num_nodes???
            ranks_per_node = cores_avail_per_node_per_worker
        elif not num_procs and not num_nodes:
            num_nodes = local_node_count

        #checks config is consistent and sufficient to express -
        #does not check actual resources
        num_procs, num_nodes, ranks_per_node = \
          JobController.job_partition(num_procs, num_nodes, ranks_per_node)

        #Could just downgrade to those available with warning - for now error
        jassert(num_nodes <= local_node_count,
                "Not enough nodes to honour arguments. "
                "Requested {}. Only {} available".
                format(num_nodes, local_node_count))

        jassert(ranks_per_node <= cores_avail_per_node,
                "Not enough processors on a node to honour arguments. "
                "Requested {}. Only {} available".
                format(ranks_per_node, cores_avail_per_node))

        jassert(ranks_per_node <= cores_avail_per_node_per_worker,
                "Not enough processors per worker to honour arguments. "
                "Requested {}. Only {} available".
                format(ranks_per_node, cores_avail_per_node_per_worker))

        jassert(num_procs <= (cores_avail_per_node * local_node_count),
                "Not enough procs to honour arguments. "
                "Requested {}. Only {} available".
                format(num_procs, cores_avail_per_node*local_node_count))

        if num_nodes < local_node_count:
            logger.warning("User constraints mean fewer nodes being used "
                           "than available. {} nodes used. {} nodes available".
                           format(num_nodes, local_node_count))

        return num_procs, num_nodes, ranks_per_node


    def create_machinefile(self, machinefile=None, num_procs=None,
                           num_nodes=None, ranks_per_node=None,
                           hyperthreads=False):
        """Create a machinefile based on user supplied config options,
        completed by detected machine resources"""

        machinefile = machinefile or 'machinefile'
        if os.path.isfile(machinefile):
            try:
                os.remove(machinefile)
            except:
                pass

        node_list = self.resources.local_nodelist
        logger.debug("Creating machinefile with {} nodes and {} ranks per node".
                     format(num_nodes, ranks_per_node))

        with open(machinefile, 'w') as f:
            for node in node_list[:num_nodes]:
                f.write((node + '\n') * ranks_per_node)

        built_mfile = (os.path.isfile(machinefile)
                       and os.path.getsize(machinefile) > 0)
        return built_mfile, num_procs, num_nodes, ranks_per_node


    def get_hostlist(self):
        """Create a hostlist based on user supplied config options,
        completed by detected machine resources"""
        node_list = self.resources.local_nodelist
        hostlist_str = ",".join([str(x) for x in node_list])
        return hostlist_str
