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

logger = logging.getLogger(__name__)
formatter = logging.Formatter('%(name)s (%(levelname)s): %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

#For debug messages - uncomment
logger.setLevel(logging.DEBUG)

STATES = '''
UNKNOWN
CREATED
WAITING
RUNNING
FINISHED
USER_KILLED
FAILED'''.split()

#I may want to use a top-level abstract/base class for maximum re-use
# - else inherited controller will be reimplementing common code

class JobControllerException(Exception): pass


class Job:
    
    '''
    Manage the creation, configuration and status of a launchable job.

    '''

    newid = itertools.count()
    
    def __init__(self, app=None, app_args=None, num_procs=None, num_nodes=None, ranks_per_node=None, machinefile=None, hostlist=None, workdir = None, stdout = None, workerid = None):
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
        self.manager_signal = 'none'        
        
        #Run attributes
        self.app = app
        self.app_args = app_args      
        self.num_procs = num_procs
        self.num_nodes = num_nodes
        self.ranks_per_node = ranks_per_node
        self.machinefile = machinefile
        self.hostlist = hostlist
        self.stdout = stdout
        self.workerID = workerid

        
        if app is not None:
            if self.workerID is not None:
                self.name = 'job_' + app.name + '_worker' + str(self.workerID)  + '_' +  str(self.id)
            else:
                self.name = 'job_' + app.name + '_' + str(self.id)
        else:
            raise JobControllerException("Job must be created with an app - no app found for job {}".format(self.id))
        
        if stdout is not None:
            self.stdout = stdout
        else:
            self.stdout = self.name + '.out'
        
        #self.workdir = './' #Default -  run in place - setting to be implemented
        self.workdir = workdir

    def workdir_exists(self):
        ''' Returns True if the job's workdir exists, else False '''
        if self.workdir is None:
            return False
        if os.path.exists(self.workdir):
            return True
        else:
            return False
        
    def file_exists_in_workdir(self, filename):
        ''' Returns True if the named file exists in the job's workdir, else False '''
        if self.workdir is None:
            return False
        path = os.path.join(self.workdir, filename)
        if os.path.exists(path):
            return True
        else:
            return False 
        
    def read_file_in_workdir(self, filename):
        ''' Open and reads the named file in the job's workdir '''
        path = os.path.join(self.workdir,filename)
        if not os.path.exists(path):
            raise ValueError("%s not found in working directory".format(filename))
        else:
            return open(path).read()
                
    def stdout_exists(self):
        ''' Returns True if the job's stdout file exists in the workdir, else False '''
        if self.workdir is None:
            return False        
        path = os.path.join(self.workdir, self.stdout)
        if os.path.exists(path):
            return True
        else:
            return False
        
    def read_stdout(self):
        ''' Open and reads the job's stdout file in the job's workdir '''
        path = os.path.join(self.workdir, self.stdout)
        if not os.path.exists(path):
            raise ValueError("%s not found in working directory".format(self.stdout))
        else:
            return open(path).read()
        
        
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
            if self.runtime is not None:
                self.total_time = self.runtime
            else:
                self.total_time = time.time() - self.launch_time    


class BalsamJob(Job):
    
    '''Wraps a Balsam Job from the Balsam service.'''
    
    #newid = itertools.count() #hopefully can use the one in Job
    
    def __init__(self, app=None, app_args=None, num_procs=None, num_nodes=None, ranks_per_node=None, machinefile=None, hostlist=None, workdir = None, stdout = None, workerid = None):
        '''Instantiate a new BalsamJob instance.
        
        A new BalsamJob object is created with an id, status and configuration attributes
        This will normally be created by the job_controller on a launch
        '''
        super().__init__(app, app_args, num_procs, num_nodes, ranks_per_node, machinefile, hostlist, workdir, stdout, workerid)
        
        self.balsam_state = None
        
        #prob want to override workdir attribute with Balsam value - though does it exist yet?
        self.workdir = None #Don't know until starts running

    def read_file_in_workdir(self, filename):
        out = self.process.read_file_in_workdir(filename)
        return out
    
    def read_stdout(self):
        out = self.process.read_file_in_workdir(self.stdout)
        return out

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
    
    ''' The job_controller can create, poll and kill runnable jobs '''
    
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
            else:
                return num_procs, num_nodes, ranks_per_node

        #If num_procs not set then need num_nodes and ranks_per_node and set num_procs
        if num_procs is None:
            #Note this covers case where none are set - may want to use job_controller defaults in that case - not implemented yet.
            if num_nodes is None or ranks_per_node is None:
                raise JobControllerException("Must set either num_procs or num_nodes/ranks_per_node or machinefile")
            else:
                num_procs = num_nodes * ranks_per_node
                return num_procs, num_nodes, ranks_per_node
        
        #If num_procs is set - fill in any other values 
        if num_procs is not None:
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
    
    def __init__(self, registry=None, auto_resources=True):
        '''Instantiate a new JobController instance.
        
        A new JobController object is created with an application registry and configuration attributes
        '''
        
        if registry is None:
            self.registry = Register.default_registry #Error handling req.
        else:
            self.registry = registry
        
        if self.registry is None:
            raise JobControllerException("Cannot find default registry")
        
        self.top_level_dir = os.getcwd()
        self.auto_resources = auto_resources
        
        if self.auto_resources:
            self.resources = Resources(top_level_dir = self.top_level_dir)
        
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
    
    def launch(self, calc_type, num_procs=None, num_nodes=None, ranks_per_node=None, machinefile=None, app_args=None, stdout=None, stage_inout=None, test=False, hyperthreads=False):
        ''' Creates a new job, and either launches or schedules to launch in the job controller
        
        The created job object is returned.
        '''
        
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
        job = Job(app, app_args, num_procs, num_nodes, ranks_per_node, machinefile, hostlist, default_workdir, stdout, self.workerID)
        
        #Temporary perhaps - though when create workdirs - will probably keep output in place
        if stage_inout is not None:
            logger.warning('stage_inout option ignored in this job_controller - runs in-place')
         
        #Construct run line - possibly subroutine
        runline = []
        runline.append(self.mpi_launcher)
        
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
            app_args_list = job.app_args.split()
            for iarg in app_args_list:
                runline.append(iarg)
        
        if test:
            print('runline args are', runline)
            print('stdout to', stdout)
            #logger.info(runline)
        else:          
            logger.debug("Launching job: {}".format(" ".join(runline)))
            
            #not good for timing job itself as dont know when finishes - if use this prob. change to date time or
            #use for timeout. For now using for timing with approx end....
            job.launch_time = time.time()
            
            job.process = subprocess.Popen(runline, cwd='./', stdout = open(job.stdout,'w'), shell=False)
            
            #To test when have workdir
            #job.process = subprocess.Popen(runline, cwd=job.workdir, stdout = open(job.stdout,'w'), shell=False)
            
            self.list_of_jobs.append(job)
        
        #return job.id
        return job

    
    def poll(self, job):
        ''' Polls and updates the status attributes of the supplied job '''
        
        if job is None:
            raise JobControllerException('No job has been provided')

        # Check the jobs been launched (i.e. it has a process ID)
        if job.process is None:
            #logger.warning('Polled job has no process ID - returning stored state')
            #Prob should be recoverable and return state - but currently fatal
            raise JobControllerException('Polled job has no process ID - check jobs been launched')
        
        # Do not poll if job already finished
        # Maybe should re-poll job to check (in case self.finished set in error!)???
        if job.finished:
            logger.warning('Polled job has already finished. Not re-polling. Status is {}'.format(job.state))
            return job
        
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
                logger.debug("Process {} completed successfully".format(job.process))
                job.state = 'FINISHED'
            else:
                #Need to differentiate failure from if job was user-killed !!!! What if remotely???
                #If this process killed the job it will already be set and if not re-polling will not get here.
                #But could query existing state here as backup?? - Also may add a REMOTE_KILL state???
                #Not yet remote killing so assume failed....
                job.errcode = job.process.returncode
                job.state = 'FAILED'
        
        #Just updates job as provided
        #return job
                
    def manager_poll(self, job):
        ''' Polls for a manager signal '''
        
        #Will use MPI_MODE from settings.py but for now assume MPI
        from libensemble.message_numbers import STOP_TAG, MAN_SIGNAL_FINISH,MAN_SIGNAL_KILL
        from mpi4py import MPI
        
        # Manager Signals
        # Stop tag may be manager interupt as diff kill/stop/pause....
        comm = MPI.COMM_WORLD
        status = MPI.Status()
        if comm.Iprobe(source=0, tag=STOP_TAG, status=status):        
            logger.info('Manager probe hit true during job {}'.format(job.name))
            man_signal = comm.recv(source=0, tag=STOP_TAG, status=status)
            if man_signal == MAN_SIGNAL_FINISH:
                job.manager_signal = 'finish'
            elif man_signal == MAN_SIGNAL_KILL:
                job.manager_signal = 'kill'
            else:
                logger.warning("Received unrecognized manager signal {} - ignoring".format(man_signal))        
        
        
    def kill(self, job):
        ''' Kills or cancels the supplied job '''
        
        if job is None:
            raise JobControllerException('No job has been provided')
        
        #In here can set state to user killed!
        #- but if killed by remote job (eg. through balsam database) may be different .... 

        #Issue signal
        if self.kill_signal == 'SIGTERM':
            job.process.terminate()
        elif self.kill_signal == 'SIGKILL':
            job.process.kill()
        else:
            job.process.send_signal(signal.self.kill_signal) #Prob only need this line!
            
        #Wait for job to be killed
        if self.wait_and_kill:
            job.process.wait() #tmp - works python2
            #try:
                #job.process.wait(timeout=self.wait_time)
                ##stdout,stderr = self.process.communicate(timeout=self.wait_time) #Wait for process to finish
            #except subprocess.TimeoutExpired:
                #logger.warning("Kill signal {} timed out - issuing SIGKILL".format(self.kill_signal))
                #job.process.kill()
                #job.process.wait()
        else:
            #job.process.wait(timeout=self.wait_time)
            job.process.wait() #tmp - works python2

        job.state = 'USER_KILLED'
        job.finished = True
        
        job.calc_job_timing()
        
        #Need to test out what to do with
        #job.errcode #Can it be discovered after killing?
        #job.success #Could set to false but should be already - only set to true on success            
                
    def set_kill_mode(self, signal=None, wait_and_kill=None, wait_time=None):
        ''' Configures the kill mode for the job_controller '''
        if signal is not None:
            self.kill_signal = signal
            
        if wait_and_kill is not None:
            self.wait_and_kill = wait_and_kill
            
        if wait_time is not None: 
            self.wait_time = wait_time
    
    def get_job(self, jobid):
        ''' Returns the job object for the supplied job ID '''
        if self.list_of_jobs:
            for job in list_of_jobs:
                if job.id == jobid:
                    return job
            logger.warning("Job %s not found in joblist".format(jobid))
            return None
        logger.warning("Job %s not found in joblist. Joblist is empty".format(jobid))
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
            logger.debug("No decomposition supplied - using all available resource. Nodes: {}  ranks_per_node {}".format(num_nodes,ranks_per_node))
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
        
        else:
            if num_nodes < local_node_count:
                logger.warning("User constraints mean fewer nodes being used than available. {} nodes used. {} nodes available".format(num_nodes,local_node_count))
        
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
        
        logger.debug("Creating machinefile with {} nodes and {} ranks per node".format(num_nodes,ranks_per_node))
            
        node_count = 0
        with open(machinefile,'w') as f:
            for node in node_list:
                node_count += 1
                if node_count > num_nodes:
                    break
                for rank in range(ranks_per_node):
                    f.write(node + '\n')
    
        #Return true if created and not empty
        if os.path.isfile(machinefile) and os.path.getsize(machinefile) > 0:
            built_mfile = True
        else:
            built_mfile = False        
        
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
    
    '''Inherits from JobController and wraps the Balsam job management service'''
    
    #controller = None
      
    def __init__(self, registry=None, auto_resources=True):
        '''Instantiate a new BalsamJobController instance.
        
        A new BalsamJobController object is created with an application registry and configuration attributes
        '''        
        
        #Will use super - atleast if use baseclass - but for now dont want to set self.mpi_launcher etc...
        if registry is None:
            self.registry = Register.default_registry #Error handling req.
        else:
            self.registry = registry
        
        if self.registry is None:
            raise JobControllerException("Cannot find default registry")

        self.top_level_dir = os.getcwd()
        self.auto_resources = auto_resources
        
        if self.auto_resources:
            self.resources = Resources(top_level_dir = self.top_level_dir, central_mode=True)

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
        


    def launch(self, calc_type, num_procs=None, num_nodes=None, ranks_per_node=None, machinefile=None, app_args=None, stdout=None, stage_inout=None, test=False, hyperthreads=False):
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
        if auto_resources:
            num_procs, num_nodes, ranks_per_node = self.get_resources(num_procs=num_procs, num_nodes=num_nodes, ranks_per_node=ranks_per_node, hyperthreads=hyperthreads)
        else:
            #Without resource detection
            num_procs, num_nodes, ranks_per_node = JobController.job_partition(num_procs, num_nodes, ranks_per_node) #Note: not included machinefile option
        
        #temp - while balsam does not accept a standard out name
        if stdout is not None:
            logger.warning("Balsam does not currently accept a stdout name - ignoring")
            stdout = None
            
        default_workdir = None #Will be possible to override with arg when implemented (else wait for Balsam to assign)
        hostlist = None
        job = BalsamJob(app, app_args, num_procs, num_nodes, ranks_per_node, machinefile, hostlist, default_workdir, stdout, self.workerID)
       
        #Re-do debug launch line for balsam job
        #logger.debug("Launching job: {}".format(" ".join(runline)))
        #logger.debug("Added job to Balsam database: {}".format(job.id))
        
        logger.debug("Added job to Balsam database: Worker {} JobID {} nodes {} ppn {}".format(self.workerID, job.id, job.num_nodes, job.ranks_per_node))
        
        #This is not used with Balsam for run-time as this would include wait time
        #Again considering changing launch to submit - or whatever I chose before.....
        job.launch_time = time.time() #Not good for timing job - as I dont know when it finishes - only poll/kill est.
        
        if stage_inout is not None:
            #For now hardcode staging - for testing
            job.process = dag.add_job(name = job.name,
                                      workflow = "libe_workflow", #add arg for this
                                      application = app.name,
                                      application_args = job.app_args,                            
                                      num_nodes = job.num_nodes,
                                      ranks_per_node = job.ranks_per_node,
                                      #input_files = app.exe,
                                      stage_in_url = "local:" + stage_inout,
                                      stage_out_url = "local:" + stage_inout,
                                      stage_out_files = "*.out")
                                      #stage_out_files = "*") #Current fails if there are directories
                                      
            #job.process = dag.spawn_child(name = job.name,
                                      #workflow = "libe_workflow", #add arg for this
                                      #application = app.name,
                                      #application_args = job.app_args,                            
                                      #num_nodes = job.num_nodes,
                                      #ranks_per_node = job.ranks_per_node,
                                      ##input_files = app.exe,
                                      #stage_in_url = "local:" + stage_inout,
                                      #stage_out_url = "local:" + stage_inout,
                                      #stage_out_files = "*",
                                      #wait_for_parents=False)            
        else:
            #No staging
            job.process = dag.add_job(name = job.name,
                                      workflow = "libe_workflow", #add arg for this
                                      application = app.name,
                                      application_args = job.app_args,           
                                      num_nodes = job.num_nodes,
                                      ranks_per_node = job.ranks_per_node) 

            #job.process = dag.spawn_child(name = job.name,
                                      #workflow = "libe_workflow", #add arg for this
                                      #application = app.name,
                                      #application_args = job.app_args,           
                                      #num_nodes = job.num_nodes,
                                      #ranks_per_node = job.ranks_per_node,
                                      #input_files = app.exe,
                                      #wait_for_parents=False)
                
        #job.workdir = job.process.working_directory #Might not be set yet!!!!
        self.list_of_jobs.append(job)
        return job

    
    def poll(self, job):
        ''' Polls and updates the status attributes of the supplied job '''
        if job is None:
            raise JobControllerException('No job has been provided') 
        
        # Check the jobs been launched (i.e. it has a process ID)
        if job.process is None:
            #logger.warning('Polled job has no process ID - returning stored state')
            #Prob should be recoverable and return state - but currently fatal
            raise JobControllerException('Polled job has no process ID - check jobs been launched')
        
        # Do not poll if job already finished
        if job.finished:
            logger.warning('Polled job has already finished. Not re-polling. Status is {}'.format(job.state))
            return job 
        
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
            
            if job.workdir == None:
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
            if job.workdir == None:
                job.workdir = job.process.working_directory
            
        elif job.balsam_state in models.PROCESSABLE_STATES + models.RUNNABLE_STATES: #Does this work - concatenate lists
            job.state = 'WAITING'
        else:
            raise JobControllerException('Job state returned from Balsam is not in known list of Balsam states. Job state is {}'.format(job.balsam_state))
        
        #return job
    
    def kill(self, job):
        ''' Kills or cancels the supplied job '''
        import balsam.launcher.dag as dag
        dag.kill(job.process)

        #Could have Wait here and check with Balsam its killed - but not implemented yet.

        job.state = 'USER_KILLED'
        job.finished = True
        job.calc_job_timing()
        
        #Check if can wait for kill to complete - affect signal used etc....
    
    def set_kill_mode(self, signal=None, wait_and_kill=None, wait_time=None):
        ''' Not currently implemented for BalsamJobController'''
        logger.warning("set_kill_mode currently has no action with Balsam controller")
        
