"""
This module launches and controls the running of tasks with Balsam.

.. note:: Balsam is supported only when using ``mpi`` comms and requires Python 3.6 or higher.

In order to create a Balsam executor, the calling script should contain ::

    exctr = BalsamMPIExecutor()

The Balsam executor inherits from the MPI executor. See the
:doc:`MPIExecutor<mpi_executor>` for shared API. Any differences are
shown below.

"""

import os
import logging
import time
import datetime

from libensemble.resources.mpi_resources import MPIResources
from libensemble.executors.executor import \
    Task, ExecutorException, jassert, STATES
from libensemble.executors.mpi_executor import MPIExecutor

import balsam.launcher.dag as dag
from balsam.core import models

logger = logging.getLogger(__name__)
# To change logging level for just this module
# logger.setLevel(logging.DEBUG)


class BalsamTask(Task):
    """Wraps a Balsam Task from the Balsam service

    The same attributes and query routines are implemented.

    """

    def __init__(self, app=None, app_args=None, workdir=None,
                 stdout=None, stderr=None, workerid=None):
        """Instantiate a new BalsamTask instance.

        A new BalsamTask object is created with an id, status and
        configuration attributes.  This will normally be created by the
        executor on a submission.
        """
        # May want to override workdir with Balsam value when it exists
        Task.__init__(self, app, app_args, workdir, stdout, stderr, workerid)

    def read_file_in_workdir(self, filename):
        return self.process.read_file_in_workdir(filename)

    def read_stdout(self):
        return self.process.read_file_in_workdir(self.stdout)

    def read_stderr(self):
        return self.process.read_file_in_workdir(self.stderr)

    def _get_time_since_balsam_submit(self):
        """Return time since balsam task entered RUNNING state"""

        # If wait_on_run then can could calculate runtime same a base executor
        # but otherwise that will return time from task submission. Get from Balsam.

        # self.runtime = self.process.runtime_seconds # Only reports at end of run currently
        balsam_launch_datetime = self.process.get_state_times().get('RUNNING', None)
        current_datetime = datetime.datetime.now()
        if balsam_launch_datetime:
            return (current_datetime - balsam_launch_datetime).total_seconds()
        else:
            return 0

    def calc_task_timing(self):
        """Calculate timing information for this task"""

        # Get runtime from Balsam
        self.runtime = self._get_time_since_balsam_submit()

        if self.submit_time is None:
            logger.warning("Cannot calc task total_time - submit time not set")
            return

        if self.total_time is None:
            self.total_time = time.time() - self.submit_time

    def poll(self):
        """Polls and updates the status attributes of the supplied task"""
        if not self.check_poll():
            return

        # Get current state of tasks from Balsam database
        self.process.refresh_from_db()
        balsam_state = self.process.state
        self.runtime = self._get_time_since_balsam_submit()

        if balsam_state in models.END_STATES:
            self.finished = True
            self.calc_task_timing()
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
                logger.warning("Task finished, but in unrecognized "
                               "Balsam state {}".format(balsam_state))
                self.state = 'UNKNOWN'

            logger.info("Task {} ended with state {}".
                        format(self.name, self.state))

        elif balsam_state in models.ACTIVE_STATES:
            self.state = 'RUNNING'
            self.workdir = self.workdir or self.process.working_directory

        elif (balsam_state in models.PROCESSABLE_STATES or
              balsam_state in models.RUNNABLE_STATES):
            self.state = 'WAITING'

        else:
            raise ExecutorException(
                "Task state returned from Balsam is not in known list of "
                "Balsam states. Task state is {}".format(balsam_state))

    def kill(self, wait_time=None):
        """ Kills or cancels the supplied task """

        dag.kill(self.process)

        # Could have Wait here and check with Balsam its killed -
        # but not implemented yet.

        logger.info("Killing task {}".format(self.name))
        self.state = 'USER_KILLED'
        self.finished = True
        self.calc_task_timing()


class BalsamMPIExecutor(MPIExecutor):
    """Inherits from MPIExecutor and wraps the Balsam task management service

    .. note::  Task kills are not configurable in the Balsam executor.

    """
    def __init__(self, auto_resources=True,
                 allow_oversubscribe=True,
                 central_mode=True,
                 nodelist_env_slurm=None,
                 nodelist_env_cobalt=None,
                 nodelist_env_lsf=None,
                 nodelist_env_lsf_shortform=None,
                 custom_info={}):
        """Instantiate a new BalsamMPIExecutor instance.

        A new BalsamMPIExecutor object is created with an application
        registry and configuration attributes
        """

        if not central_mode:
            logger.warning("Balsam does not currently support distributed mode - running in central mode")
            central_mode = True

        if custom_info:
            logger.warning("The Balsam executor does not support custom_info - ignoring")

        super().__init__(auto_resources,
                         allow_oversubscribe,
                         central_mode,
                         nodelist_env_slurm,
                         nodelist_env_cobalt,
                         nodelist_env_lsf,
                         nodelist_env_lsf_shortform)

        self.workflow_name = "libe_workflow"

    def _serial_setup(self):
        """Balsam serial setup includes empyting database and adding applications"""
        BalsamMPIExecutor.del_apps()
        BalsamMPIExecutor.del_tasks()

        for calc_type in self.default_apps:
            if self.default_apps[calc_type] is not None:
                calc_name = self.default_apps[calc_type].name
                desc = self.default_apps[calc_type].desc
                full_path = self.default_apps[calc_type].full_path
                self.add_app(calc_name, full_path, desc)

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
    def del_tasks():
        """Deletes all Balsam tasks whose names contains .simfunc or .genfunc"""
        for app_type in ['.simfunc', '.genfunc']:
            deletion_objs = models.BalsamJob.objects.filter(
                name__contains=app_type)
            if deletion_objs:
                for del_task in deletion_objs.iterator():
                    logger.debug("Deleting task {}".format(del_task.name))
                deletion_objs.delete()

        # May be able to use union function - to combine - see queryset help.
        # Eg (not tested)
        # del_simfuncs = Task.objects.filter(name__contains='.simfunc')
        # del_genfuncs = Task.objects.filter(name__contains='.genfunc')
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

    def submit(self, calc_type, num_procs=None, num_nodes=None,
               ranks_per_node=None, machinefile=None, app_args=None,
               stdout=None, stderr=None, stage_inout=None,
               hyperthreads=False, dry_run=False, wait_on_run=False,
               extra_args=None):
        """Creates a new task, and either executes or schedules to execute
        in the executor

        The created task object is returned.
        """
        app = self.default_app(calc_type)

        # Specific to this class
        if machinefile is not None:
            logger.warning("machinefile arg ignored - not supported in Balsam")
            jassert(num_procs or num_nodes or ranks_per_node,
                    "No procs/nodes provided - aborting")

        # Extra_args analysis not done here - could pick up self.mpi_runner but possible
        # that Balsam finds a different runner.
        if self.auto_resources:
            num_procs, num_nodes, ranks_per_node = \
                self.resources.get_resources(
                    num_procs=num_procs,
                    num_nodes=num_nodes, ranks_per_node=ranks_per_node,
                    hyperthreads=hyperthreads)
        else:
            num_procs, num_nodes, ranks_per_node = \
                MPIResources.task_partition(num_procs, num_nodes, ranks_per_node)

        if stdout is not None or stderr is not None:
            logger.warning("Balsam does not currently accept a stdout "
                           "or stderr name - ignoring")
            stdout = None
            stderr = None

        # Will be possible to override with arg when implemented
        # (or can have option to let Balsam assign)
        default_workdir = os.getcwd()
        task = BalsamTask(app, app_args, default_workdir,
                          stdout, stderr, self.workerID)

        add_task_args = {'name': task.name,
                         'workflow': self.workflow_name,
                         'user_workdir': default_workdir,
                         'application': app.name,
                         'args': task.app_args,
                         'num_nodes': num_nodes,
                         'ranks_per_node': ranks_per_node,
                         'mpi_flags': extra_args}

        if stage_inout is not None:
            # For now hardcode staging - for testing
            add_task_args['stage_in_url'] = "local:" + stage_inout + "/*"
            add_task_args['stage_out_url'] = "local:" + stage_inout
            add_task_args['stage_out_files'] = "*.out"

        if dry_run:
            task.dry_run = True
            logger.info('Test (No submit) Runline: {}'.format(' '.join(add_task_args)))
            task.set_as_complete()
        else:
            task.process = dag.add_job(**add_task_args)

            if (wait_on_run):
                self._wait_on_run(task)

            if not task.timer.timing:
                task.timer.start()
                task.submit_time = task.timer.tstart  # Time not date - may not need if using timer.

            logger.info("Added task to Balsam database {}: "
                        "nodes {} ppn {}".
                        format(task.name, num_nodes, ranks_per_node))

            # task.workdir = task.process.working_directory  # Might not be set yet!
        self.list_of_tasks.append(task)
        return task
