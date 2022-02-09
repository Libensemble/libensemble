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

from libensemble.executors.executor import \
    Application, Task, ExecutorException, TimeoutExpired, jassert, STATES
from libensemble.executors.mpi_executor import MPIExecutor

from balsam.api import Job, BatchJob, EventLog

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
        event_query = EventLog.objects.filter(
            job_id=self.process.id, to_state="RUNNING")
        if not len(event_query):
            return 0
        balsam_launch_datetime = event_query[0].timestamp
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
        # self.process.refresh_from_db()
        balsam_state = self.process.state
        print(balsam_state)
        self.runtime = self._get_time_since_balsam_submit()

        if balsam_state in ['RUN_DONE', 'POSTPROCESSED', 'STAGED_OUT', "JOB_FINISHED"]:
            self._set_complete()

        elif balsam_state in ['RUNNING']:
            self.state = 'RUNNING'
            self.workdir = self.workdir or self.process.working_directory

        elif balsam_state in ['CREATED', 'AWAITING_PARENTS',
                              'READY', 'STAGED_IN', 'PREPROCESSED']:
            self.state = 'WAITING'

        elif balsam_state in ['RUN_ERROR', 'RUN_TIMEOUT', 'FAILED']:
            self.state = 'FAILED'

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
        while self.process.state not in ['RUN_DONE', 'POSTPROCESSED', 'STAGED_OUT', "JOB_FINISHED"]:
            time.sleep(0.2)
            self.process.refresh_from_db()
            if timeout and time.time() - start > timeout:
                self.runtime = self._get_time_since_balsam_submit()
                raise TimeoutExpired(self.name, timeout)

        self.runtime = self._get_time_since_balsam_submit()
        self._set_complete()

    def kill(self, wait_time=None):
        """ Kills or cancels the supplied task """

        self.process.delete()

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
        self.allocations = []

    def serial_setup(self):
        """Balsam serial setup includes empyting database and adding applications"""
        pass

        # for app in self.apps.values():
        #     calc_name = app.gname
        #     desc = app.desc
        #     full_path = app.full_path
        #     site = app.site
        #     self.add_app(calc_name, site, full_path, desc)

    def add_app(self, name, site, exepath, desc):
        """ Sync application with balsam service """
        pass
        logger.debug("Added App {}".format(name))

    def register_app(self, BalsamApp, app_name, calc_type=None, desc=None):
        """Registers a Balsam application instance to libEnsemble.

        The ``full_path`` of the application must be supplied. Either
        ``app_name`` or ``calc_type`` can be used to identify the
        application in user scripts (in the **submit** function).
        ``app_name`` is recommended.

        Parameters
        ----------

        full_path: String
            The full path of the user application to be registered

        app_name: String, optional
            Name to identify this application.

        calc_type: String, optional
            Calculation type: Set this application as the default 'sim'
            or 'gen' function.

        desc: String, optional
            Description of this application

        """
        if not app_name:
            app_name = BalsamApp.command_template.split(" ")[0]
        self.apps[app_name] = Application(" ", app_name, calc_type, desc, BalsamApp)

        # Default sim/gen apps will be deprecated. Just use names.
        if calc_type is not None:
            jassert(calc_type in self.default_apps,
                    "Unrecognized calculation type", calc_type)
            self.default_apps[calc_type] = self.apps[app_name]

    def submit_allocation(self, site_id, num_nodes, wall_time_min, job_mode="mpi",
                          queue="local", project="local"):
        """
        Submits a Balsam BatchJob machine allocation request to Balsam.
        Corresponding Balsam applications with a matching site can be submitted to this allocation.
        """

        self.allocations.append(
            BatchJob.objects.create(
                site_id=site_id,
                num_nodes=num_nodes,
                wall_time_min=wall_time_min,
                job_mode=job_mode,
                queue=queue,
                project=project
            )
        )

        logger.info("Submitted Batch allocation to endpoint {}: "
                    "nodes {} queue {} project {}".
                    format(site_id, num_nodes, queue, project))

    def set_resources(self, resources):
        self.resources = resources

    def submit(self, calc_type=None, app_name=None, app_args=None, num_procs=None,
               num_nodes=None, procs_per_node=None, max_tasks_per_node=None,
               machinefile=None, gpus_per_rank=0, transfers={},
               workdir='', dry_run=False, wait_on_start=False, extra_args={}):
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

        if len(workdir):
            workdir = os.path.join(self.workflow_name, workdir)
        else:
            workdir = self.workflow_name

        # Specific to this class
        if machinefile is not None:
            logger.warning("machinefile arg ignored - not supported in Balsam")
            jassert(num_procs or num_nodes or procs_per_node,
                    "No procs/nodes provided - aborting")

        task = BalsamTask(app, app_args, workdir,
                          None, None, self.workerID)

        if dry_run:
            task.dry_run = True
            logger.info('Test (No submit) Balsam app {}'.format(app_name))
            task._set_complete(dry_run=True)
        else:
            App = app.pyobj
            App.sync()
            task.process = Job(app_id=App, workdir=workdir,
                               parameters=app_args,
                               num_nodes=num_nodes,
                               ranks_per_node=procs_per_node,
                               launch_params=extra_args,
                               gpus_per_rank=gpus_per_rank,
                               node_packing_count=max_tasks_per_node,
                               transfers=transfers)

            task.process.save()

            if (wait_on_start):
                self._wait_on_start(task)

            if not task.timer.timing:
                task.timer.start()
                task.submit_time = task.timer.tstart  # Time not date - may not need if using timer.

            logger.info("Submitted Balsam App to endpoint {}: "
                        "nodes {} ppn {}".
                        format(App.site, num_nodes, procs_per_node))

            # task.workdir = task.process.working_directory  # Might not be set yet!
        self.list_of_tasks.append(task)
        return task
