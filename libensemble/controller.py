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
import subprocess
import logging
import signal
import itertools
import time
from libensemble.register import Register
from libensemble.resources import Resources

if Resources.am_I_manager():
    wrkid = 'Manager'
else:
    wrkid = 'w' + str(Resources.get_workerID())

logger = logging.getLogger(__name__ + '(' + wrkid + ')')
#For debug messages in this module  - uncomment (see libE.py to change root logging level)
#logger.setLevel(logging.DEBUG)

STATES = '''
UNKNOWN
CREATED
WAITING
RUNNING
FINISHED
USER_KILLED
FAILED'''.split()

SIGNALS = '''
SIGTERM
SIGKILL'''.split()


#I may want to use a top-level abstract/base class for maximum re-use
# - else inherited controller will be reimplementing common code

class JobControllerException(Exception): pass


class Job:

    '''
    Manage the creation, configuration and status of a launchable job.

    '''

    newid = itertools.count()

    def __init__(self, app=None, app_args=None, num_procs=None, num_nodes=None, ranks_per_node=None,
                 machinefile=None, hostlist=None, workdir=None, stdout=None, stderr=None, workerid=None):
        '''Instantiate a new Job instance.

        A new job object is created with an id, status and configuration attributes
        This will normally be created by the job_controller on a launch
        '''
        self.id = next(Job.newid)

        #Status attributes
        self.state = 'CREATED' #: test1 docstring
        self.process = None
        self.errcode = None
        self.finished = False  # True means job has run - not whether was successful
        self.success = False
        self.launch_time = None
        self.runtime = None
        self.total_time = None
        #self.manager_signal = 'none'

        #Run attributes
        self.app = app
        self.app_args = app_args
        self.num_procs = num_procs
        self.num_nodes = num_nodes
        self.ranks_per_node = ranks_per_node
        self.machinefile = machinefile
        self.hostlist = hostlist
        #self.stdout = stdout
        #self.stderr = stderr
        self.workerID = workerid

        if app is None:
            raise JobControllerException("Job must be created with an app - no app found for job {}".format(self.id))

        worker_name = "_worker{}".format(self.workerID) if self.workerID else ""
        self.name = "job_{}{}_{}".format(app.name, worker_name, self.id)
        self.stdout = stdout or self.name + '.out'
        self.stderr = stderr or self.name + '.err'        
        self.workdir = workdir

    def workdir_exists(self):
        ''' Returns True if the job's workdir exists, else False '''
        return self.workdir and os.path.exists(self.workdir)

    def file_exists_in_workdir(self, filename):
        ''' Returns True if the named file exists in the job's workdir, else False '''
        return self.workdir and os.path.exists(os.path.join(self.workdir, filename))

    def read_file_in_workdir(self, filename):
        ''' Open and reads the named file in the job's workdir '''
        path = os.path.join(self.workdir, filename)
        if not os.path.exists(path):
            raise ValueError("{} not found in working directory".format(filename))
        with open(path) as f:
            return f.read()

    def stdout_exists(self):
        ''' Returns True if the job's stdout file exists in the workdir, else False '''
        return self.file_exists_in_workdir(self.stdout)

    def read_stdout(self):
        ''' Open and reads the job's stdout file in the job's workdir '''
        return self.read_file_in_workdir(self.stdout)

    def stderr_exists(self):
        ''' Returns True if the job's stderr file exists in the workdir, else False '''
        return self.file_exists_in_workdir(self.stderr)

    def read_stderr(self):
        ''' Open and reads the job's stderr file in the job's workdir '''
        return self.read_file_in_workdir(self.stderr)

    #Note - this is currently only final job-time. May make running job time.
    #And prob want to use for polling in sim func - esp in balsam - where want acutal runtime not time since launch
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


class BalsamJob(Job):

    '''Wraps a Balsam Job from the Balsam service.

    The same attributes and query routines are implemented.

    '''

    #newid = itertools.count() #hopefully can use the one in Job

    def __init__(self, app=None, app_args=None, num_procs=None, num_nodes=None, ranks_per_node=None, machinefile=None, hostlist=None, workdir=None, stdout=None, stderr=None, workerid=None):
        '''Instantiate a new BalsamJob instance.

        A new BalsamJob object is created with an id, status and configuration attributes
        This will normally be created by the job_controller on a launch
        '''

        super().__init__(app, app_args, num_procs, num_nodes, ranks_per_node, machinefile, hostlist, workdir, stdout, workerid)

        self.balsam_state = None

        #prob want to override workdir attribute with Balsam value - though does it exist yet?
        #self.workdir = None #Don't know until starts running
        self.workdir = workdir #Default for libe now is to run in place.


    def read_file_in_workdir(self, filename):
        return self.process.read_file_in_workdir(filename)

    def read_stdout(self):
        return self.process.read_file_in_workdir(self.stdout)

    def read_stderr(self):
        return self.process.read_file_in_workdir(self.stderr)
    
    def calc_job_timing(self):
        """Calculate timing information for this job"""

        #Get runtime from Balsam
        self.runtime = self.process.runtime_seconds

        if self.launch_time is None:
            logger.warning("Cannot calc job total_time - launch time not set")
            return

        if self.total_time is None:
            self.total_time = time.time() - self.launch_time


class JobController:

    ''' The job_controller can create, poll and kill runnable jobs

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
    
    '''

    controller = None

    @staticmethod
    def job_partition(num_procs, num_nodes, ranks_per_node, machinefile=None):
        """ Takes provided nprocs/nodes/ranks and outputs working configuration of procs/nodes/ranks or error """

        #If machinefile is provided - ignore everything else
        if machinefile is not None:
            if num_procs is not None or num_nodes is not None or ranks_per_node is not None:
                logger.warning('Machinefile provided - overriding procs/nodes/ranks_per_node')
            num_procs = None
            num_nodes = None
            ranks_per_node = None
            return num_procs, num_nodes, ranks_per_node

        #If all set then check num_procs equals num_nodes*ranks_per_node and set values as given
        if num_procs is not None and num_nodes is not None and ranks_per_node is not None:
            if num_procs != num_nodes*ranks_per_node:
                raise JobControllerException("num_procs does not equal num_nodes*ranks_per_node")
            return num_procs, num_nodes, ranks_per_node

        #If num_procs not set then need num_nodes and ranks_per_node and set num_procs
        if num_procs is None:
            #Note this covers case where none are set - may want to use job_controller defaults in that case - not implemented yet.
            if num_nodes is None or ranks_per_node is None:
                raise JobControllerException("Must set either num_procs or num_nodes/ranks_per_node or machinefile")
            num_procs = num_nodes * ranks_per_node
            return num_procs, num_nodes, ranks_per_node

        #If num_procs is set - fill in any other values
        #if num_procs is not None:
        else:
            if num_nodes is None:
                if ranks_per_node is None:
                    #Currently not auto-detecting so if only num_procs - you are on 1 node
                    num_nodes = 1
                    ranks_per_node = num_procs
                else:
                    num_nodes = num_procs//ranks_per_node
            else:
                ranks_per_node = num_procs//num_nodes

        return num_procs, num_nodes, ranks_per_node

    #def _calc_job_timing(job):

        #if job.launch_time is None:
            #logger.warning("Cannot calc job timing - launch time not set")
            #return

        ##In case already been killed and set then
        #if job.runtime is None:
            #job.runtime = time.time() - job.launch_time

        ##For direct launched jobs - these should be the same.
        #if job.total_time is None:
            #if job.runtime is not None:
                #job.total_time = job.runtime
            #else:
                #job.total_time = time.time() - job.launch_time

    def __init__(self, registry=None, auto_resources=True, nodelist_env_slurm=None, nodelist_env_cobalt=None):
        '''Instantiate a new JobController instance.

        A new JobController object is created with an application registry and configuration attributes. A
        registry object must have been created.

        This is typically created in the user calling script. If auto_resources is True, an evaluation of system resources is performance during this call.

        Parameters
        ----------
        registry: obj: Registry, optional
            A registry containing the applications to use in this job_controller (Default: Use Register.default_registry).

        auto_resources: Boolean, optional
            Auto-detect available processor resources and assign to jobs if not explicitly provided on launch.

        nodelist_env_slurm: String, optional
            The environment variable giving a node list in Slurm format (Default: Uses SLURM_NODELIST)
            Note: This is only queried if a worker_list file is not provided and auto_resources=True.

        nodelist_env_cobalt: String, optional
            The environment variable giving a node list in Cobalt format (Default: Uses COBALT_PARTNAME)
            Note: This is only queried if a worker_list file is not provided and auto_resources=True.

        '''

        self.registry = registry or Register.default_registry
        if self.registry is None:
            raise JobControllerException("Cannot find default registry")

        self.top_level_dir = os.getcwd()
        self.auto_resources = auto_resources
        self.manager_signal = 'none'

        if self.auto_resources:
            self.resources = Resources(top_level_dir=self.top_level_dir,
                                       nodelist_env_slurm=nodelist_env_slurm,
                                       nodelist_env_cobalt=nodelist_env_cobalt)

        #logger.debug("top_level_dir is {}".format(self.top_level_dir))

        #todo Configure by autodetection
        #In fact it will be a sub-object - most likely with inhertience - based on detection or specification
        #Also the construction of the run-line itself will prob. be a function of that object
        #For now though - do like this:

        mpi_variant = Resources.get_MPI_variant()
        if mpi_variant == 'mpich':
            self.mpi_launcher = 'mpirun'
            self.mfile = '-machinefile'
            self.nprocs = '-np'
            self.nnodes = ''
            self.ppn = '--ppn'
            self.hostlist = '-hosts'
        elif mpi_variant == 'openmpi':
            self.mpi_launcher = 'mpirun'
            self.mfile = '-machinefile'
            self.nprocs = '-np'
            self.nnodes = ''
            self.ppn = '-npernode'
            self.hostlist = '-host'
        #self.mpi_launcher = 'srun'
        #self.mfile = '-m arbitrary'
        #self.nprocs = '--ntasks'
        #self.nnodes = '--nodes'
        #self.ppn = '--ntasks-per-node'
        #self.hostlist = '-w'

        #Job controller settings - can be set in user function.
        self.kill_signal = 'SIGTERM'
        self.wait_and_kill = True #If true - wait for wait_time after signal and then kill with SIGKILL
        self.wait_time = 60

        #list_of_jobs: Need to decide on reset... - reset for each calc?
        #and how link to libe job (or calc) class - if reset for each calc - could store this in job
        self.list_of_jobs = []
        self.workerID = None

        #self.auto_machinefile = True #Create a machinefile automatically

        JobController.controller = self

        #self.resources = Resources(top_level_dir = self.top_level_dir)

        #If this could share multiple launches could set default job parameters here (nodes/ranks etc...)


    # May change job_controller launch functions to use **kwargs and then init job empty - and use setattr
    #eg. To pass through args:
    #def launch(**kwargs):
    #...
    #job = Job()
    #for k,v in kwargs.items():
    #try:
        #getattr(job, k)
    #except AttributeError:
        #raise ValueError(f"Invalid field {}".format(k)) #Unless not passing through all
    #else:
        #setattr(job, k, v)

    def launch(self, calc_type, num_procs=None, num_nodes=None, ranks_per_node=None,
               machinefile=None, app_args=None, stdout=None, stderr=None, stage_inout=None, hyperthreads=False, test=False):
        ''' Creates a new job, and either launches or schedules to launch in the job controller

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
            A string of the application arguments to be added to job launch command line.

        stdout: string, optional
            A standard output filename.

        stderr: string, optional
            A standard error filename.

        stage_inout: string, optional
            A directory to copy files from. Default will take from current directory.

        hyperthreads: boolean, optional
            Whether to launch MPI tasks to hyperthreads

        test: boolean, optional
            Whether this is a test - No job will be launched. Instead runline is printed to logger (At INFO level).


        Returns
        -------

        job: obj: Job
            The lauched job object.


        Note that if some combination of num_procs, num_nodes and ranks_per_node are provided, these will be honored if possible. If resource detection is on and these are omitted, then the available resources will be divided amongst workers.

        '''

        # Find the default sim or gen app from registry.sim_default_app OR registry.gen_default_app
        # Could take optional app arg - if they want to supply here - instead of taking from registry
        if calc_type == 'sim':
            if self.registry.sim_default_app is None:
                raise JobControllerException("Default sim app is not set")
            app = self.registry.sim_default_app
        elif calc_type == 'gen':
            if self.registry.gen_default_app is None:
                raise JobControllerException("Default gen app is not set")
            app = self.registry.gen_default_app
        else:
            raise JobControllerException("Unrecognized calculation type", calc_type)


        #-------- Up to here should be common - can go in a baseclass and make all concrete classes inherit ------#
        hostlist = None
        if machinefile is None and self.auto_resources:

            #klugging this for now - not nec machinefile if more than one node - try a hostlist

            num_procs, num_nodes, ranks_per_node = self.get_resources(num_procs=num_procs, num_nodes=num_nodes, ranks_per_node=ranks_per_node, hyperthreads=hyperthreads)

            if num_nodes > 1:
                #hostlist
                hostlist = self.get_hostlist()
            else:
                #machinefile
                if self.workerID is not None:
                    machinefile = 'machinefile_autogen_for_worker_' + str(self.workerID)
                else:
                    machinefile = 'machinefile_autogen'
                mfile_created, num_procs, num_nodes, ranks_per_node = self.create_machinefile(machinefile, num_procs, num_nodes, ranks_per_node, hyperthreads)
                if not mfile_created:
                    raise JobControllerException("Auto-creation of machinefile failed")

        else:
            num_procs, num_nodes, ranks_per_node = JobController.job_partition(num_procs, num_nodes, ranks_per_node, machinefile)


        default_workdir = os.getcwd() #Will be possible to override with arg when implemented
        job = Job(app, app_args, num_procs, num_nodes, ranks_per_node, machinefile, hostlist, default_workdir, stdout, stderr, self.workerID)

        #Temporary perhaps - though when create workdirs - will probably keep output in place
        if stage_inout is not None:
            logger.warning('stage_inout option ignored in this job_controller - runs in-place')

        #Construct run line - possibly subroutine
        runline = [self.mpi_launcher]

        if job.machinefile is not None:
            #os.environ['SLURM_HOSTFILE'] = job.machinefile
            runline.append(self.mfile)
            runline.append(job.machinefile)

        #Should be else - if machine file - dont need any other config

        if job.hostlist is not None:
            #os.environ['SLURM_HOSTFILE'] = job.machinefile
            runline.append(self.hostlist)
            runline.append(job.hostlist)

        if job.num_procs is not None:
            runline.append(self.nprocs)
            runline.append(str(job.num_procs))

        #Not currently setting nodes
        #- as not always supported - but should always have the other two after calling _job_partition
        #if job.num_nodes is not None:
            #runline.append(self.nnodes)
            #runline.append(str(job.num_nodes))

        #Currently issues - command depends on mpich/openmpi etc...
        if job.ranks_per_node is not None:
            runline.append(self.ppn)
            runline.append(str(job.ranks_per_node))

        runline.append(job.app.full_path)

        if job.app_args is not None:
            runline.extend(job.app_args.split())

        if test:
            logger.info('Test selected: Not launching job')
            logger.info('runline args are {}'.format(runline))
        else:
            logger.debug("Launching job {}: {}".format(job.name, " ".join(runline))) #One line
            #logger.debug("Launching job {}:\n{}{}".format(job.name, " "*32, " ".join(runline))) #With newline

            #not good for timing job itself as dont know when finishes - if use this prob. change to date time or
            #use for timeout. For now using for timing with approx end....
            job.launch_time = time.time()

            #job.process = subprocess.Popen(runline, cwd='./', stdout=open(job.stdout, 'w'), stderr=open(job.stderr, 'w'), shell=False)
            job.process = subprocess.Popen(runline, cwd='./', stdout=open(job.stdout, 'w'), stderr=open(job.stderr, 'w'), shell=False, preexec_fn=os.setsid)

            #To test when have workdir
            #job.process = subprocess.Popen(runline, cwd=job.workdir, stdout=open(job.stdout, 'w'), stderr=open(job.stderr, 'w'), shell=False, preexec_fn=os.setsid)

            self.list_of_jobs.append(job)

        #return job.id
        return job


    def poll(self, job):
        ''' Polls and updates the status attributes of the supplied job

        Parameters
        -----------

        job: obj: Job
            The job object.to be polled.

        '''

        if not isinstance(job, Job):
            raise JobControllerException('Invalid job has been provided')

        # Check the jobs been launched (i.e. it has a process ID)
        if job.process is None:
            #logger.warning('Polled job has no process ID - returning stored state')
            #Prob should be recoverable and return state - but currently fatal
            raise JobControllerException('Polled job {} has no process ID - check jobs been launched'.format(job.name))

        # Do not poll if job already finished
        # Maybe should re-poll job to check (in case self.finished set in error!)???
        if job.finished:
            logger.warning('Polled job {} has already finished. Not re-polling. Status is {}'.format(job.name, job.state))
            return

        #-------- Up to here should be common - can go in a baseclass and make all concrete classes inherit ------#

        # Poll the job
        poll = job.process.poll()
        if poll is None:
            job.state = 'RUNNING'
        else:
            job.finished = True
            #logger.debug("Process {} Completed".format(job.process))

            job.calc_job_timing()

            if job.process.returncode == 0:
                job.success = True
                job.errcode = 0
                #logger.debug("Process {} completed successfully".format(job.process))
                logger.debug("Job {} completed successfully".format(job.name))
                job.state = 'FINISHED'
            else:
                #Need to differentiate failure from if job was user-killed !!!! What if remotely???
                #If this process killed the job it will already be set and if not re-polling will not get here.
                #But could query existing state here as backup?? - Also may add a REMOTE_KILL state???
                #Not yet remote killing so assume failed....
                job.errcode = job.process.returncode
                logger.debug("Job {} failed".format(job.name))
                job.state = 'FAILED'

        #Just updates job as provided
        #return job

    def manager_poll(self):
        ''' Polls for a manager signal

        The job controller manager_signal attribute will be updated.

        '''
        from libensemble.message_numbers import STOP_TAG, MAN_SIGNAL_FINISH, MAN_SIGNAL_KILL
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
                logger.warning("Received unrecognized manager signal {} - ignoring".format(man_signal))


    @staticmethod
    def _kill_process(process, signal):
        """Launch the process kill for this system"""
        time.sleep(0.1) # Without a small wait - kill signal can not work
        os.killpg(os.getpgid(process.pid), signal) # Kill using process group (see launch with preexec_fn=os.setsid)

        #process.send_signal(signal) # Kill by sending direct signal

    # Just for you, python2
    @staticmethod
    def _time_out(process, timeout):
        """Loop to wait for process to finish after a kill"""
        start_wait_time = time.time()
        while time.time() - start_wait_time < timeout:
            time.sleep(0.01)
            poll = process.poll()
            if poll is not None:
                return False # process has finished - no timeout
        return True # process has not finished - timeout


    def kill(self, job):
        ''' Kills or cancels the supplied job

        Parameters
        -----------

        job: obj: Job
            The job object.to be polled.


        The signal used is determined by the job_controller attirbute <kill_signal> will be send to the job,
        followed by a wait for the process to terminate. If the <wait_and_kill> attribute is True, then
        a SIGKILL will be sent if the job has not finished after <wait_time> seconds. The kill can be
        configured using the set_kill_mode function.

        '''

        if not isinstance(job, Job):
            raise JobControllerException('Invalid job has been provided')

        if job.finished:
            logger.warning('Trying to kill job that is no longer running. Job {}: Status is {}'.format(job.name, job.state))
            return

        if job.process is None:
            time.sleep(0.2)
            if job.process is None:
                #logger.warning('Polled job has no process ID - returning stored state')
                #Prob should be recoverable and return state - but currently fatal
                raise JobControllerException('Attempting to kill job {} that has no process ID - check jobs been launched'.format(job.name))

        logger.debug("Killing job {}".format(job.name))

        # Issue signal
        sig = {'SIGTERM': signal.SIGTERM, 'SIGKILL': signal.SIGKILL}
        if self.kill_signal not in sig:
            raise JobControllerException('Unknown kill signal')
        try:
            JobController._kill_process(job.process, sig[self.kill_signal])
        except ProcessLookupError:
            logger.warning("Tried to kill job {}. Process {} not found. May have finished".format(job.name, job.process.pid))

        # Wait for job to be killed
        if self.wait_and_kill:

            # My python2 method works ok for py2 and py3
            if JobController._time_out(job.process, self.wait_time):
                logger.warning("Kill signal {} timed out for job {}: Issuing SIGKILL".format(self.kill_signal, job.name))
                JobController._kill_process(job.process, signal.SIGKILL)
                job.process.wait()

            #Using subprocess timeout attribute where available (py3)
            #try:
                #job.process.wait(timeout=self.wait_time)
                ##stdout,stderr = self.process.communicate(timeout=self.wait_time) #Wait for process to finish
            #except TypeError: #eg. Python2
                ##logger.warning("TimeoutExpired not supported in this version of Python. Issuing SIGKILL to job {}".format(job.name))
                #if JobController._time_out(job.process, self.wait_time):
                    #logger.warning("Kill signal {} timed out for job {}: Issuing SIGKILL".format(self.kill_signal, job.name))
                    #JobController._kill_process(job.process, signal.SIGKILL)
                    #job.process.wait()
            #except subprocess.TimeoutExpired:
                #logger.warning("Kill signal {} timed out for job {}: Issuing SIGKILL".format(self.kill_signal, job.name))
                #JobController._kill_process(job.process, signal.SIGKILL)
                #job.process.wait()
        else:
            job.process.wait()

        job.state = 'USER_KILLED'
        job.finished = True
        job.calc_job_timing()

        #Need to test out what to do with
        #job.errcode #Can it be discovered after killing?
        #job.success #Could set to false but should be already - only set to true on success


    def set_kill_mode(self, signal=None, wait_and_kill=None, wait_time=None):
        ''' Configures the kill mode for the job_controller

        Parameters
        ----------

        signal: String, optional
            The signal type to be sent to kill job: 'SIGTERM' or 'SIGKILL'

        wait_and_kill: boolean, optional
            If True, a SIGKILL will be sent after <wait_time> seconds if the process has not terminated.

        wait_time: int, optional
            The number of seconds to wait for the job to finish before sending a SIGKILL when wait_and_kill is set.
            (Default is 60).


        '''
        if signal is not None:
            if signal not in SIGNALS:
                raise JobControllerException("Unknown signal {} supplied to set_kill_mode".format(signal))
            self.kill_signal = signal

        if wait_and_kill is not None:
            self.wait_and_kill = wait_and_kill

        if wait_time is not None:
            self.wait_time = wait_time
            if not wait_and_kill:
                logger.warning('wait_time set but will have no effect while wait_and_kill is False')


    def get_job(self, jobid):
        ''' Returns the job object for the supplied job ID '''
        if self.list_of_jobs:
            for job in self.list_of_jobs:
                if job.id == jobid:
                    return job
            logger.warning("Job {} not found in joblist".format(jobid))
            return None
        logger.warning("Job {} not found in joblist. Joblist is empty".format(jobid))
        return None


    def set_workerID(self, workerid):
        """Sets the worker ID for this job_controller"""
        self.workerID = workerid


    #Reformat create_machinefile to use this and also use this for non-machinefile cases when auto-detecting
    def get_resources(self, num_procs=None, num_nodes=None, ranks_per_node=None, hyperthreads=False):
        """
        Reconciles user supplied options with available Worker resources to produce run configuration.

        Detects resources available to worker, checks if an existing user supplied config is valid,
        and fills in any missing config information (ie. num_procs/num_nodes/ranks_per_node)

        User supplied config options are honoured, and an exception is raised if these are infeasible.
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
            cores_avail_per_node_per_worker = cores_avail_per_node//workers_per_node
        else:
            cores_avail_per_node_per_worker = cores_avail_per_node

        if not node_list:
            raise JobControllerException("Node list is empty - aborting")

        #If no decomposition supplied - use all available cores/nodes
        if num_procs is None and num_nodes is None and ranks_per_node is None:
            num_nodes = local_node_count
            ranks_per_node = cores_avail_per_node_per_worker
            #logger
            logger.debug("No decomposition supplied - using all available resource. Nodes: {}  ranks_per_node {}".format(num_nodes, ranks_per_node))
        elif num_nodes is None and ranks_per_node is None:
            #Got just num_procs
            num_nodes = local_node_count
            #Here is where really want a compact/scatter option - go for scatter (could get cores and say if less than one node - but then hyperthreads complication if no psutil installed)
        elif num_procs is None and ranks_per_node is None:
            #Who would just put num_nodes???
            ranks_per_node = cores_avail_per_node_per_worker
        elif num_procs is None and num_nodes is None:
            num_nodes = local_node_count

        #checks config is consistent and sufficient to express - does not check actual resources
        num_procs, num_nodes, ranks_per_node = JobController.job_partition(num_procs, num_nodes, ranks_per_node)

        if num_nodes > local_node_count:
            #Could just downgrade to those available with warning - for now error
            raise JobControllerException("Not enough nodes to honour arguments. Requested {}. Only {} available".format(num_nodes, local_node_count))

        elif ranks_per_node > cores_avail_per_node:
            #Could just downgrade to those available with warning - for now error
            raise JobControllerException("Not enough processors on a node to honour arguments. Requested {}. Only {} available".format(ranks_per_node, cores_avail_per_node))

        elif ranks_per_node > cores_avail_per_node_per_worker:
            #Could just downgrade to those available with warning - for now error
            raise JobControllerException("Not enough processors per worker to honour arguments. Requested {}. Only {} available".format(ranks_per_node, cores_avail_per_node_per_worker))

        elif num_procs > (cores_avail_per_node * local_node_count):
            #Could just downgrade to those available with warning - for now error
            raise JobControllerException("Not enough procs to honour arguments. Requested {}. Only {} available".format(num_procs, cores_avail_per_node*local_node_count))

        elif num_nodes < local_node_count:
            logger.warning("User constraints mean fewer nodes being used than available. {} nodes used. {} nodes available".format(num_nodes, local_node_count))

        return num_procs, num_nodes, ranks_per_node



    def create_machinefile(self, machinefile=None, num_procs=None, num_nodes=None, ranks_per_node=None, hyperthreads=False):
        """Create a machinefile based on user supplied config options, completed by detected machine resources"""

        #Maybe hyperthreads should be mpi_hyperthreads

        if machinefile is None:
            machinefile = 'machinefile'

        if os.path.isfile(machinefile):
            try:
                os.remove(machinefile)
            except:
                pass

        #num_procs, num_nodes, ranks_per_node = self.get_resources(num_procs=num_procs, num_nodes=num_nodes, ranks_per_node=ranks_per_node, hyperthreads=hyperthreads)
        node_list = self.resources.local_nodelist

        logger.debug("Creating machinefile with {} nodes and {} ranks per node".format(num_nodes, ranks_per_node))

        with open(machinefile, 'w') as f:
            for node in node_list[:num_nodes]:
                f.write((node + '\n') * ranks_per_node)

        #Return true if created and not empty
        built_mfile = os.path.isfile(machinefile) and os.path.getsize(machinefile) > 0

        #Return new values for num_procs,num_nodes,ranks_per_node - in case want to use
        return built_mfile, num_procs, num_nodes, ranks_per_node

    #will prob want to adjust based on input
    #def get_hostlist(self, machinefile=None, num_procs=None, num_nodes=None, ranks_per_node=None, hyperthreads=False):
    def get_hostlist(self):
        """Create a hostlist based on user supplied config options, completed by detected machine resources"""
        node_list = self.resources.local_nodelist
        hostlist_str = ",".join([str(x) for x in node_list])
        return hostlist_str


class BalsamJobController(JobController):

    '''Inherits from JobController and wraps the Balsam job management service

    .. note::  Job kills are currently not configurable in the Balsam job_controller.

    The set_kill_mode function will do nothing but print a warning.

    '''

    #controller = None

    def __init__(self, registry=None, auto_resources=True, nodelist_env_slurm=None, nodelist_env_cobalt=None):
        '''Instantiate a new BalsamJobController instance.

        A new BalsamJobController object is created with an application registry and configuration attributes
        '''

        #Will use super - atleast if use baseclass - but for now dont want to set self.mpi_launcher etc...

        self.registry = registry or Register.default_registry
        if self.registry is None:
            raise JobControllerException("Cannot find default registry")

        self.top_level_dir = os.getcwd()
        self.auto_resources = auto_resources
        self.manager_signal = 'none'
        
        if self.auto_resources:
            self.resources = Resources(top_level_dir=self.top_level_dir, central_mode=True,
                                       nodelist_env_slurm=nodelist_env_slurm,
                                       nodelist_env_cobalt=nodelist_env_cobalt)

        #-------- Up to here should be common - can go in a baseclass and make all concrete classes inherit ------#

        self.list_of_jobs = [] #Why did I put here? Will inherit

        #self.auto_machinefile = False #May in future use the auto_detect part though - to fill in procs/nodes/ranks_per_node

        JobController.controller = self
        #BalsamJobController.controller = self

    #def _calc_job_timing(job):
        ##Get runtime from Balsam
        #if job.launch_time is None:
            #logger.warning("Cannot calc job total_time - launch time not set")
            #return

        #if job.total_time is None:
            #job.total_time = time.time() - job.launch_time



    def launch(self, calc_type, num_procs=None, num_nodes=None, ranks_per_node=None, 
               machinefile=None, app_args=None, stdout=None, stderr=None, stage_inout=None, test=False, hyperthreads=False):
        ''' Creates a new job, and either launches or schedules to launch in the job controller

        The created job object is returned.
        '''
        import balsam.launcher.dag as dag

        # Find the default sim or gen app from registry.sim_default_app OR registry.gen_default_app
        # Could take optional app arg - if they want to supply here - instead of taking from registry
        if calc_type == 'sim':
            if self.registry.sim_default_app is None:
                raise JobControllerException("Default sim app is not set")
            else:
                app = self.registry.sim_default_app
        elif calc_type == 'gen':
            if self.registry.gen_default_app is not None:
                raise JobControllerException("Default gen app is not set")
            else:
                app = self.registry.gen_default_app
        else:
            raise JobControllerException("Unrecognized calculation type", calc_type)

        #-------- Up to here should be common - can go in a baseclass and make all concrete classes inherit ------#

        #Need test somewhere for if no breakdown supplied.... or only machinefile

        #Specific to this class
        if machinefile is not None:
            logger.warning("machinefile arg ignored - not supported in Balsam")
            if num_procs is None and num_nodes is None and ranks_per_node is None:
                raise JobControllerException("No procs/nodes provided - aborting")


        #Set num_procs, num_nodes and ranks_per_node for this job

        #Without resource detection
        #num_procs, num_nodes, ranks_per_node = JobController.job_partition(num_procs, num_nodes, ranks_per_node) #Note: not included machinefile option

        #With resource detection (may do only if under-specified?? though that will not tell if larger than possible
        #for static allocation - but Balsam does allow dynamic allocation if too large!!
        #For now allow user to specify - but default is True....
        if self.auto_resources:
            num_procs, num_nodes, ranks_per_node = self.get_resources(num_procs=num_procs, num_nodes=num_nodes, ranks_per_node=ranks_per_node, hyperthreads=hyperthreads)
        else:
            #Without resource detection
            num_procs, num_nodes, ranks_per_node = JobController.job_partition(num_procs, num_nodes, ranks_per_node) #Note: not included machinefile option

        #temp - while balsam does not accept a standard out name
        if stdout is not None or stderr is not None:
            logger.warning("Balsam does not currently accept a stdout or stderr name - ignoring")
            stdout = None
            stderr = None

        #Will be possible to override with arg when implemented (or can have option to let Balsam assign)
        default_workdir = os.getcwd()
        
        hostlist = None
        job = BalsamJob(app, app_args, num_procs, num_nodes, ranks_per_node, machinefile, hostlist, default_workdir, stdout, stderr, self.workerID)

        #This is not used with Balsam for run-time as this would include wait time
        #Again considering changing launch to submit - or whatever I chose before.....
        job.launch_time = time.time() #Not good for timing job - as I dont know when it finishes - only poll/kill est.

        add_job_args = {'name': job.name,
                        'workflow': "libe_workflow", #add arg for this
                        'user_workdir': default_workdir, #add arg for this
                        'application': app.name,
                        'args': job.app_args,
                        'num_nodes': job.num_nodes,
                        'ranks_per_node': job.ranks_per_node}

        if stage_inout is not None:
            #For now hardcode staging - for testing
            add_job_args['stage_in_url'] = "local:" + stage_inout + "/*"
            add_job_args['stage_out_url'] = "local:" + stage_inout
            add_job_args['stage_out_files'] = "*.out"

        job.process = dag.add_job(**add_job_args)

        logger.debug("Added job to Balsam database {}: Worker {} nodes {} ppn {}".format(job.name, self.workerID, job.num_nodes, job.ranks_per_node))

        #job.workdir = job.process.working_directory #Might not be set yet!!!!
        self.list_of_jobs.append(job)
        return job


    def poll(self, job):
        ''' Polls and updates the status attributes of the supplied job '''
        if not isinstance(job, BalsamJob):
            raise JobControllerException('Invalid job has been provided')

        # Check the jobs been launched (i.e. it has a process ID)
        if job.process is None:
            #logger.warning('Polled job has no process ID - returning stored state')
            #Prob should be recoverable and return state - but currently fatal
            raise JobControllerException('Polled job has no process ID - check jobs been launched')

        # Do not poll if job already finished
        if job.finished:
            logger.warning('Polled job has already finished. Not re-polling. Status is {}'.format(job.state))
            return

        #-------- Up to here should be common - can go in a baseclass and make all concrete classes inherit ------#

        # Get current state of jobs from Balsam database
        job.process.refresh_from_db()
        job.balsam_state = job.process.state #Not really nec to copy have balsam_state - already job.process.state...
        #logger.debug('balsam_state for job {} is {}'.format(job.id, job.balsam_state))

        import balsam.launcher.dag as dag #Might need this before get models - test
        from balsam.service import models

        if job.balsam_state in models.END_STATES:
            job.finished = True

            job.calc_job_timing()

            if job.workdir is None:
                job.workdir = job.process.working_directory
            if job.balsam_state == 'JOB_FINISHED':
                job.success = True
                job.state = 'FINISHED'
            elif job.balsam_state == 'PARENT_KILLED': #I'm not using this currently
                job.state = 'USER_KILLED'
                #job.success = False #Shld already be false - init to false
                #job.errcode = #Not currently returned by Balsam API - requested - else will remain as None
            elif job.balsam_state in STATES: #In my states
                job.state = job.balsam_state
                #job.success = False #All other end states are failrues currently - bit risky
                #job.errcode = #Not currently returned by Balsam API - requested - else will remain as None
            else:
                logger.warning("Job finished, but in unrecognized Balsam state {}".format(job.balsam_state))
                job.state = 'UNKNOWN'

        elif job.balsam_state in models.ACTIVE_STATES:
            job.state = 'RUNNING'
            if job.workdir is None:
                job.workdir = job.process.working_directory

        elif job.balsam_state in models.PROCESSABLE_STATES + models.RUNNABLE_STATES: #Does this work - concatenate lists
            job.state = 'WAITING'
        else:
            raise JobControllerException('Job state returned from Balsam is not in known list of Balsam states. Job state is {}'.format(job.balsam_state))

        # DSB: With this commented out, number of return args is inconsistent (returns job above)
        #return job

    def kill(self, job):
        ''' Kills or cancels the supplied job '''

        if not isinstance(job, BalsamJob):
            raise JobControllerException('Invalid job has been provided')

        import balsam.launcher.dag as dag
        dag.kill(job.process)

        #Could have Wait here and check with Balsam its killed - but not implemented yet.

        job.state = 'USER_KILLED'
        job.finished = True
        job.calc_job_timing()

        #Check if can wait for kill to complete - affect signal used etc....

    def set_kill_mode(self, signal=None, wait_and_kill=None, wait_time=None):
        ''' Not currently implemented for BalsamJobController.

        No action is taken
        '''
        logger.warning("set_kill_mode currently has no action with Balsam controller")
