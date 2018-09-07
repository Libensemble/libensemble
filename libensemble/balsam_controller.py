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
import time

from libensemble.register import Register
from libensemble.resources import Resources
from libensemble.controller import \
     Job, JobController, JobControllerException, jassert, STATES

logger = logging.getLogger(__name__ + '(' + Resources.get_my_name() + ')')
#For debug messages in this module  - uncomment (see libE.py to change root logging level)
#logger.setLevel(logging.DEBUG)


class BalsamJob(Job):
    """Wraps a Balsam Job from the Balsam service.

    The same attributes and query routines are implemented.

    """

    def __init__(self, app=None, app_args=None, num_procs=None, num_nodes=None,
                 ranks_per_node=None, machinefile=None, hostlist=None,
                 workdir=None, stdout=None, stderr=None, workerid=None):
        """Instantiate a new BalsamJob instance.

        A new BalsamJob object is created with an id, status and
        configuration attributes.  This will normally be created by the
        job_controller on a launch.
        """

        super().__init__(app, app_args, num_procs, num_nodes, ranks_per_node,
                         machinefile, hostlist, workdir, stdout, stderr,
                         workerid)

        self.balsam_state = None

        #prob want to override workdir attribute with Balsam value -
        #though does it exist yet?
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


class BalsamJobController(JobController):
    """Inherits from JobController and wraps the Balsam job management service

    .. note::  Job kills are currently not configurable in the Balsam job_controller.

    The set_kill_mode function will do nothing but print a warning.

    """

    #controller = None

    def __init__(self, registry=None, auto_resources=True,
                 nodelist_env_slurm=None, nodelist_env_cobalt=None):
        """Instantiate a new BalsamJobController instance.

        A new BalsamJobController object is created with an application
        registry and configuration attributes
        """

        #Will use super - atleast if use baseclass - but for now dont want to set self.mpi_launcher etc...

        self.registry = registry or Register.default_registry
        jassert(self.registry, "Cannot find default registry")

        self.top_level_dir = os.getcwd()
        self.auto_resources = auto_resources

        if self.auto_resources:
            self.resources = Resources(top_level_dir=self.top_level_dir, central_mode=True,
                                       nodelist_env_slurm=nodelist_env_slurm,
                                       nodelist_env_cobalt=nodelist_env_cobalt)

        #-------- Up to here should be common - can go in a baseclass and make all concrete classes inherit ------#

        self.list_of_jobs = [] #Why did I put here? Will inherit

        #self.auto_machinefile = False #May in future use the auto_detect part though - to fill in procs/nodes/ranks_per_node

        JobController.controller = self
        #BalsamJobController.controller = self


    def launch(self, calc_type, num_procs=None, num_nodes=None,
               ranks_per_node=None, machinefile=None, app_args=None,
               stdout=None, stderr=None, stage_inout=None,
               hyperthreads=False, test=False):
        """Creates a new job, and either launches or schedules to launch
        in the job controller

        The created job object is returned.
        """
        import balsam.launcher.dag as dag

        app = self.registry.default_app(calc_type)
        jassert(calc_type in ['sim', 'gen'],
                "Unrecognized calculation type", calc_type)
        jassert(app, "Default {} app is not set".format(calc_type))

        #-------- Up to here should be common - can go in a baseclass and make all concrete classes inherit ------#

        #Need test somewhere for if no breakdown supplied.... or only machinefile

        #Specific to this class
        if machinefile is not None:
            logger.warning("machinefile arg ignored - not supported in Balsam")
            jassert(num_procs or num_nodes or ranks_per_node,
                    "No procs/nodes provided - aborting")


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
            logger.warning("Balsam does not currently accept a stdout "
                           "or stderr name - ignoring")
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
        """Polls and updates the status attributes of the supplied job"""
        jassert(isinstance(job, BalsamJob), "Invalid job has been provided")

        # Check the jobs been launched (i.e. it has a process ID)
        #Prob should be recoverable and return state - but currently fatal
        jassert(job.process, "Polled job has no process ID - check jobs been launched")

        # Do not poll if job already finished
        if job.finished:
            logger.warning("Polled job has already finished. Not re-polling. "
                           "Status is {}".format(job.state))
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
        """ Kills or cancels the supplied job """

        jassert(isinstance(job, BalsamJob), "Invalid job has been provided")

        import balsam.launcher.dag as dag
        dag.kill(job.process)

        #Could have Wait here and check with Balsam its killed - but not implemented yet.

        job.state = 'USER_KILLED'
        job.finished = True
        job.calc_job_timing()

