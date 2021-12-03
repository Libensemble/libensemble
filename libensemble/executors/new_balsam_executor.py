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

from libensemble.resources import mpi_resources
from libensemble.executors.executor import \
    Application, Task, ExecutorException, TimeoutExpired, jassert, STATES
from libensemble.executors.mpi_executor import MPIExecutor
from libensemble.executors.executor import Application

from balsam.api import ApplicationDefinition, BatchJob, Job, EventLog

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

    def _get_time_since_balsam_submit(self):
        """Return time since balsam task entered RUNNING state"""

        # If wait_on_start then can could calculate runtime same a base executor
        # but otherwise that will return time from task submission. Get from Balsam.

        # self.runtime = self.process.runtime_seconds # Only reports at end of run currently
        # balsam_launch_datetime = self.process.get_state_times().get('RUNNING', None)
        balsam_launch_datetime = EventLog.objects.filter(
            job_id=self.process.job_id, to_state="RUNNING").timestamp
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

    def _set_complete(self, dry_run=False):
        """Set task as complete"""
        self.finished = True
        if dry_run:
            self.success = True
            self.state = 'FINISHED'
        else:
            balsam_state = self.process.state
            self.workdir = self.workdir or self.process.working_directory
            self.calc_task_timing()
            self.success = (balsam_state == 'JOB_FINISHED')
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

    def poll(self):
        """Polls and updates the status attributes of the supplied task"""
        if self.dry_run:
            return

        if not self._check_poll():
            return

        # Get current state of tasks from Balsam database
        self.process.refresh_from_db()
        balsam_state = self.process.state
        self.runtime = self._get_time_since_balsam_submit()

        if balsam_state in models.END_STATES:
            self._set_complete()

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
        start = time.time()
        self.process.refresh_from_db()
        while self.process.state not in models.END_STATES:
            time.sleep(0.2)
            self.process.refresh_from_db()
            if timeout and time.time() - start > timeout:
                self.runtime = self._get_time_since_balsam_submit()
                raise TimeoutExpired(self.name, timeout)

        self.runtime = self._get_time_since_balsam_submit()
        self._set_complete()

    def kill(self, wait_time=None):
        """ Kills or cancels the supplied task """

        dag.kill(self.process)

        # Could have Wait here and check with Balsam its killed -
        # but not implemented yet.

        logger.info("Killing task {}".format(self.name))
        self.state = 'USER_KILLED'
        self.finished = True
        self.calc_task_timing()

class NewBalsamMPIExecutor(MPIExecutor):
    """Inherits from MPIExecutor and wraps the Balsam task management service

    .. note::  Task kills are not configurable in the Balsam executor.

    """
    def __init__(self, custom_info={}):
        """Instantiate a new BalsamMPIExecutor instance.

        A new BalsamMPIExecutor object is created with an application
        registry and configuration attributes
        """

        if custom_info:
            logger.warning("The Balsam executor does not support custom_info - ignoring")

        super().__init__(custom_info)

        self.workflow_name = "libe_workflow"
        self.application_objs = {}

    def serial_setup(self):
        """Balsam serial setup includes empyting database and adding applications"""

        for app in self.apps.values():
            calc_name = app.gname
            desc = app.desc
            full_path = app.full_path
            site = app.site
            self.application_objs[calc_name] = self.add_app(calc_name, site, full_path, desc)


    def add_app(name, site, exepath, desc):
        """ Sync application with balsam service """

        class BalsamApplication(ApplicationDefinition):
            site = site
            command_template=exepath

        BalsamApplication.sync()
        logger.debug("Added App {}".format(name))
        return BalsamApplication

    def register_app(self, full_path, site, app_name=None, calc_type=None, desc=None):
        """Registers a user application to libEnsemble.

        The ``full_path`` of the application must be supplied. Either
        ``app_name`` or ``calc_type`` can be used to identify the
        application in user scripts (in the **submit** function).
        ``app_name`` is recommended.

        Parameters
        ----------

        full_path: String
            The full path of the user application to be registered

        site: String
            The Balsam site name for where to launch the app

        app_name: String, optional
            Name to identify this application.

        calc_type: String, optional
            Calculation type: Set this application as the default 'sim'
            or 'gen' function.

        desc: String, optional
            Description of this application

        """
        if not app_name:
            app_name = os.path.split(full_path)[1]
        self.apps[app_name] = Application(full_path, app_name, calc_type, desc, site)

        # Default sim/gen apps will be deprecated. Just use names.
        if calc_type is not None:
            jassert(calc_type in self.default_apps,
                    "Unrecognized calculation type", calc_type)
            self.default_apps[calc_type] = self.apps[app_name]

    def set_resources(self, resources):
        self.resources = resources

    def submit(self, calc_type=None, app_name=None, num_procs=None,
               num_nodes=None, procs_per_node=None, machinefile=None,
               app_args=None, stdout=None, stderr=None, stage_inout=None,
               hyperthreads=False, dry_run=False, wait_on_start=False, queue=None,
               project=None, wall_time_min=None, extra_args=''):
        """Creates a new task, and either executes or schedules to execute
        in the executor

        The created task object is returned.
        """

        if app_name is not None:
            app = self.get_app(app_name)
        elif calc_type is not None:
            app = self.default_app(calc_type)
        else:
            raise ExecutorException("Either app_name or calc_type must be set")

        # Specific to this class
        if machinefile is not None:
            logger.warning("machinefile arg ignored - not supported in Balsam")
            jassert(num_procs or num_nodes or procs_per_node,
                    "No procs/nodes provided - aborting")

        num_procs, num_nodes, procs_per_node = \
            mpi_resources.task_partition(num_procs, num_nodes, procs_per_node)

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
                         'application': app.gname,
                         'args': task.app_args,
                         'num_nodes': num_nodes,
                         'procs_per_node': procs_per_node,
                         'mpi_flags': extra_args}

        if dry_run:
            task.dry_run = True
            logger.info('Test (No submit) to Balsam: {}'.format(' '.join(add_task_args)))
            task._set_complete(dry_run=True)
        else:

            balsam_app_obj = self.application_objs[task.name]
            task.process = balsam_app_obj.submit(workdir=self.workflow_name)

            task.batchjob = BatchJob.objects.create(
                site_id=task.process.site_id,
                num_nodes=num_nodes,
                ranks_per_node=procs_per_node,
                wall_time_min=wall_time_min,
                job_mode="mpi",
                queue=queue,
                project=project
            )

            if (wait_on_start):
                self._wait_on_start(task)

            if not task.timer.timing:
                task.timer.start()
                task.submit_time = task.timer.tstart  # Time not date - may not need if using timer.

            logger.info("Added task to Balsam database {}: "
                        "nodes {} ppn {}".
                        format(task.name, num_nodes, procs_per_node))

            # task.workdir = task.process.working_directory  # Might not be set yet!
        self.list_of_tasks.append(task)
        return task
