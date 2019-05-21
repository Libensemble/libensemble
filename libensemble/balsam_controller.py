"""
Module to launch and control running jobs with Balsam.

"""

import os
import logging
import time
import datetime
from mpi4py import MPI

from libensemble.mpi_resources import MPIResources
from libensemble.controller import \
    Job, JobControllerException, jassert, STATES
from libensemble.mpi_controller import MPIJobController

import balsam.launcher.dag as dag
from balsam.core import models

logger = logging.getLogger(__name__)
# To change logging level for just this module
# logger.setLevel(logging.DEBUG)


class BalsamJob(Job):
    """Wraps a Balsam Job from the Balsam service.

    The same attributes and query routines are implemented.

    """

    def __init__(self, app=None, app_args=None, workdir=None,
                 stdout=None, stderr=None, workerid=None):
        """Instantiate a new BalsamJob instance.

        A new BalsamJob object is created with an id, status and
        configuration attributes.  This will normally be created by the
        job_controller on a launch.
        """
        # May want to override workdir with Balsam value when it exists
        Job.__init__(self, app, app_args, workdir, stdout, stderr, workerid)

    def read_file_in_workdir(self, filename):
        return self.process.read_file_in_workdir(filename)

    def read_stdout(self):
        return self.process.read_file_in_workdir(self.stdout)

    def read_stderr(self):
        return self.process.read_file_in_workdir(self.stderr)

    def _get_time_since_balsam_launch(self):
        """Return time since balam job entered RUNNING state"""
        
        # If wait_on_run then can could calculate runtime same a base controller
        # but otherwise that will return time from job submission. Get from Balsam.
        
        #self.runtime = self.process.runtime_seconds # Only reports at end of run currently
        balsam_launch_datetime = self.process.get_state_times().get('RUNNING', None)
        current_datetime = datetime.datetime.now()
        if balsam_launch_datetime:
            return (current_datetime - balsam_launch_datetime).total_seconds()
        else:
            return 0        

    def calc_job_timing(self):
        """Calculate timing information for this job"""

        # Get runtime from Balsam
        self.runtime = self._get_time_since_balsam_launch()

        if self.launch_time is None:
            logger.warning("Cannot calc job total_time - launch time not set")
            return

        if self.total_time is None:
            self.total_time = time.time() - self.launch_time

    def poll(self):
        """Polls and updates the status attributes of the supplied job"""
        if not self.check_poll():
            return

        # Get current state of jobs from Balsam database
        self.process.refresh_from_db()
        balsam_state = self.process.state
        self.runtime = self._get_time_since_balsam_launch()

        if balsam_state in models.END_STATES:
            self.finished = True
            self.calc_job_timing()
            self.workdir = self.workdir or self.process.working_directory
            self.success = (balsam_state == 'JOB_FINISHED')
            # self.errcode - requested feature from Balsam devs

            if balsam_state == 'JOB_FINISHED':
                self.state = 'FINISHED'
            elif balsam_state == 'PARENT_KILLED':  # Not currently used
                self.state = 'USER_KILLED'
            elif balsam_state in STATES:  # In my states
                self.state = balsam_state
            else:
                logger.warning("Job finished, but in unrecognized "
                               "Balsam state {}".format(balsam_state))
                self.state = 'UNKNOWN'

            logger.info("Job {} ended with state {}".
                        format(self.name, self.state))

        elif balsam_state in models.ACTIVE_STATES:
            self.state = 'RUNNING'
            self.workdir = self.workdir or self.process.working_directory

        elif (balsam_state in models.PROCESSABLE_STATES or
              balsam_state in models.RUNNABLE_STATES):
            self.state = 'WAITING'

        else:
            raise JobControllerException(
                "Job state returned from Balsam is not in known list of "
                "Balsam states. Job state is {}".format(balsam_state))

    def kill(self, wait_time=None):
        """ Kills or cancels the supplied job """

        dag.kill(self.process)

        # Could have Wait here and check with Balsam its killed -
        # but not implemented yet.

        logger.info("Killing job {}".format(self.name))
        self.state = 'USER_KILLED'
        self.finished = True
        self.calc_job_timing()


class BalsamJobController(MPIJobController):
    """Inherits from MPIJobController and wraps the Balsam job management service

    .. note::  Job kills are not configurable in the Balsam job_controller.

    """
    def __init__(self, auto_resources=True, central_mode=True,
                 nodelist_env_slurm=None, nodelist_env_cobalt=None):
        """Instantiate a new BalsamJobController instance.

        A new BalsamJobController object is created with an application
        registry and configuration attributes
        """

        if not central_mode:
            logger.warning("Balsam does not currently support distributed mode - running in central mode")
            central_mode = True

        super().__init__(auto_resources, central_mode,
                         nodelist_env_slurm, nodelist_env_cobalt)
        self.mpi_launcher = None
        if MPI.COMM_WORLD.Get_rank() == 0:
            BalsamJobController.del_apps()
            BalsamJobController.del_jobs()

    @staticmethod
    def del_apps():
        """Deletes all Balsam apps whose names contains .simfunc or .genfunc"""
        AppDef = models.ApplicationDefinition

        # Some error handling on deletes.... is it internal
        for app_type in ['.simfunc', '.genfunc']:
            deletion_objs = AppDef.objects.filter(name__contains=app_type)
            if deletion_objs:
                for del_app in deletion_objs.iterator():
                    logger.debug("Deleting app {}".format(del_app.name))
                deletion_objs.delete()

    @staticmethod
    def del_jobs():
        """Deletes all Balsam jobs whose names contains .simfunc or .genfunc"""
        for app_type in ['.simfunc', '.genfunc']:
            deletion_objs = models.BalsamJob.objects.filter(
                name__contains=app_type)
            if deletion_objs:
                for del_job in deletion_objs.iterator():
                    logger.debug("Deleting job {}".format(del_job.name))
                deletion_objs.delete()

        # May be able to use union function - to combine - see queryset help.
        # Eg (not tested)
        # del_simfuncs = Job.objects.filter(name__contains='.simfunc')
        # del_genfuncs = Job.objects.filter(name__contains='.genfunc')
        # deletion_objs = deletion_objs.union()

    @staticmethod
    def add_app(name, exepath, desc):
        """ Add application to Balsam database """
        AppDef = models.ApplicationDefinition
        app = AppDef()
        app.name = name
        app.executable = exepath
        app.description = desc
        # app.default_preprocess = '' # optional
        # app.default_postprocess = '' # optional
        app.save()
        logger.debug("Added App {}".format(app.name))

    def register_calc(self, full_path, calc_type='sim', desc=None):
        """Registers a user applications to libEnsemble and Balsam

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
        # OK to use Python 3 syntax (Balsam requires 3.6+)
        super().register_calc(full_path, calc_type, desc)

        # Get from one place - so always matches
        calc_name = self.default_apps[calc_type].name
        desc = self.default_apps[calc_type].desc

        if MPI.COMM_WORLD.Get_rank() == 0:
            self.add_app(calc_name, full_path, desc)

    def launch(self, calc_type, num_procs=None, num_nodes=None,
               ranks_per_node=None, machinefile=None, app_args=None,
               stdout=None, stderr=None, stage_inout=None,
               hyperthreads=False, test=False, wait_on_run=False):
        """Creates a new job, and either launches or schedules to launch
        in the job controller

        The created job object is returned.
        """
        app = self.default_app(calc_type)

        # Need test somewhere for if no breakdown supplied....
        # or only machinefile

        # Specific to this class
        if machinefile is not None:
            logger.warning("machinefile arg ignored - not supported in Balsam")
            jassert(num_procs or num_nodes or ranks_per_node,
                    "No procs/nodes provided - aborting")

        # Set num_procs, num_nodes and ranks_per_node for this job

        # Without resource detection
        # num_procs, num_nodes, ranks_per_node = JobController.job_partition(num_procs, num_nodes, ranks_per_node)  # Note: not included machinefile option

        # With resource detection (may do only if under-specified?? though that will not tell if larger than possible
        # for static allocation - but Balsam does allow dynamic allocation if too large!!
        # For now allow user to specify - but default is True....
        if self.auto_resources:
            num_procs, num_nodes, ranks_per_node = \
                self.resources.get_resources(
                    num_procs=num_procs,
                    num_nodes=num_nodes, ranks_per_node=ranks_per_node,
                    hyperthreads=hyperthreads)
        else:
            # Without resource detection (note: not included machinefile option)
            num_procs, num_nodes, ranks_per_node = \
                MPIResources.job_partition(num_procs, num_nodes, ranks_per_node)

        # temp - while balsam does not accept a standard out name
        if stdout is not None or stderr is not None:
            logger.warning("Balsam does not currently accept a stdout "
                           "or stderr name - ignoring")
            stdout = None
            stderr = None

        # Will be possible to override with arg when implemented
        # (or can have option to let Balsam assign)
        default_workdir = os.getcwd()
        job = BalsamJob(app, app_args, default_workdir,
                        stdout, stderr, self.workerID)

        # This is not used with Balsam for run-time as this would include wait time
        # Again considering changing launch to submit - or whatever I chose before.....
        #job.launch_time = time.time()  # Not good for timing job - as I dont know when it finishes - only poll/kill est.

        add_job_args = {'name': job.name,
                        'workflow': "libe_workflow",  # add arg for this
                        'user_workdir': default_workdir,  # add arg for this
                        'application': app.name,
                        'args': job.app_args,
                        'num_nodes': num_nodes,
                        'ranks_per_node': ranks_per_node}

        if stage_inout is not None:
            # For now hardcode staging - for testing
            add_job_args['stage_in_url'] = "local:" + stage_inout + "/*"
            add_job_args['stage_out_url'] = "local:" + stage_inout
            add_job_args['stage_out_files'] = "*.out"

        job.process = dag.add_job(**add_job_args)

        if (wait_on_run):
            self._wait_on_run(job)
            
        if not job.timer.timing:
            job.timer.start()
            job.launch_time = job.timer.tstart  # Time not date - may not need if using timer.
                
        logger.info("Added job to Balsam database {}: "
                    "nodes {} ppn {}".
                    format(job.name, num_nodes, ranks_per_node))

        # job.workdir = job.process.working_directory  # Might not be set yet!!!!
        self.list_of_jobs.append(job)
        return job
