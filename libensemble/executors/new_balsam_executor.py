"""
This module launches and controls the running of tasks with Balsam 2, and most
notably can submit tasks from any machine, to any machine running a Balsam site.

In order to create a Balsam executor, the calling script should contain ::

    exctr = NewBalsamExecutor()

One key difference to consider is that instead of registering paths to apps,
Balsam ApplicationDefinition instances must be registered instead.

"""

import os
import logging
import time
import datetime

from libensemble.executors.executor import (
    Application,
    Task,
    ExecutorException,
    TimeoutExpired,
    jassert,
    STATES,
)
from libensemble.executors import Executor

from balsam.api import Job, BatchJob, EventLog

logger = logging.getLogger(__name__)
# To change logging level for just this module
# logger.setLevel(logging.DEBUG)


class NewBalsamTask(Task):
    """Wraps a Balsam Job from the Balsam service.

    The same attributes and query routines are implemented. Use ``task.process``
    to refer to the matching Balsam Job initialized by the NewBalsamExecutor,
    with every Balsam Job method invokable on it. Otherwise, libEnsemble task methods
    like ``poll()`` can be used directly.

    """

    def __init__(
        self,
        app=None,
        app_args=None,
        workdir=None,
        stdout=None,
        stderr=None,
        workerid=None,
    ):
        """Instantiate a new NewBalsamTask instance.

        A new NewBalsamTask object is created with an id, status and
        configuration attributes.  This will normally be created by the
        executor on a submission.
        """
        # May want to override workdir with Balsam value when it exists
        Task.__init__(self, app, app_args, workdir, stdout, stderr, workerid)

    def _get_time_since_balsam_submit(self):
        """Return time since balsam task entered RUNNING state"""

        event_query = EventLog.objects.filter(
            job_id=self.process.id, to_state="RUNNING"
        )
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
            self.state = "FINISHED"
        else:
            balsam_state = self.process.state
            self.workdir = self.workdir or self.process.working_directory
            self.calc_task_timing()
            if balsam_state in [
                "RUN_DONE",
                "POSTPROCESSED",
                "STAGED_OUT",
                "JOB_FINISHED",
            ]:
                self.success = True
                self.state = "FINISHED"
            elif balsam_state in STATES:  # In my states
                self.state = balsam_state
            else:
                logger.warning(
                    "Task finished, but in unrecognized "
                    "Balsam state {}".format(balsam_state)
                )
                self.state = "UNKNOWN"

            logger.info("Task {} ended with state {}".format(self.name, self.state))

    def poll(self):
        """Polls and updates the status attributes of the supplied task. Requests
        Job information from Balsam service."""
        if self.dry_run:
            return

        if not self._check_poll():
            return

        # Get current state of tasks from Balsam database
        self.process.refresh_from_db()
        balsam_state = self.process.state
        self.runtime = self._get_time_since_balsam_submit()

        if balsam_state in ["RUN_DONE", "POSTPROCESSED", "STAGED_OUT", "JOB_FINISHED"]:
            self._set_complete()

        elif balsam_state in ["RUNNING"]:
            self.state = "RUNNING"
            self.workdir = self.workdir or self.process.working_directory

        elif balsam_state in [
            "CREATED",
            "AWAITING_PARENTS",
            "READY",
            "STAGED_IN",
            "PREPROCESSED",
        ]:
            self.state = "WAITING"

        elif balsam_state in ["RUN_ERROR", "RUN_TIMEOUT", "FAILED"]:
            self.state = "FAILED"

        else:
            raise ExecutorException(
                "Task state returned from Balsam is not in known list of "
                "Balsam states. Task state is {}".format(balsam_state)
            )

    def wait(self, timeout=None):
        """Waits on completion of the task or raises TimeoutExpired exception

        Status attributes of task are updated on completion.

        Parameters
        ----------

        timeout: int
            Time in seconds after which a TimeoutExpired exception is raised"""

        if self.dry_run:
            return

        if not self._check_poll():
            return

        # Wait on the task
        start = time.time()
        self.process.refresh_from_db()
        while self.process.state not in [
            "RUN_DONE",
            "POSTPROCESSED",
            "STAGED_OUT",
            "JOB_FINISHED",
        ]:
            time.sleep(0.2)
            self.process.refresh_from_db()
            if timeout and time.time() - start > timeout:
                self.runtime = self._get_time_since_balsam_submit()
                raise TimeoutExpired(self.name, timeout)

        self.runtime = self._get_time_since_balsam_submit()
        self._set_complete()

    def kill(self):
        """Cancels the supplied task. Killing is unsupported at this time."""

        self.process.delete()

        logger.info("Killing task {}".format(self.name))
        self.state = "USER_KILLED"
        self.finished = True
        self.calc_task_timing()


class NewBalsamExecutor(Executor):
    """Inherits from MPIExecutor and wraps the Balsam service. Via this Executor,
    Balsam Jobs can be submitted to Balsam sites, either local or on remote machines.

    .. note::  Task kills are not configurable in the Balsam executor.

    """

    def __init__(self, custom_info={}):
        """Instantiate a new BalsamMPIExecutor instance.

        A new BalsamMPIExecutor object is created with an application
        registry and configuration attributes
        """

        if custom_info:
            logger.warning(
                "The Balsam executor does not support custom_info - ignoring"
            )

        super().__init__(custom_info)

        self.workflow_name = "libe_workflow"
        self.allocations = []

    def serial_setup(self):
        """Balsam serial setup includes emptying database and adding applications"""
        pass

    def add_app(self, name, site, exepath, desc):
        """Sync application with balsam service"""
        pass

    def register_app(self, BalsamApp, app_name, calc_type=None, desc=None):
        """Registers a Balsam ApplicationDefinition to libEnsemble. This class
        instance *must* have a ``site`` and ``command_template`` specified. See
        the Balsam docs for information on other optional fields.

        Parameters
        ----------

        BalsamApp: ApplicationDefinition object
            A Balsam ApplicationDefinition instance.

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
            jassert(
                calc_type in self.default_apps,
                "Unrecognized calculation type",
                calc_type,
            )
            self.default_apps[calc_type] = self.apps[app_name]

    def submit_allocation(
        self,
        site_id,
        num_nodes,
        wall_time_min,
        job_mode="mpi",
        queue="local",
        project="local",
    ):
        """
        Submits a Balsam ``BatchJob`` machine allocation request to Balsam.
        Corresponding Balsam applications with a matching site can be submitted to
        this allocation.

        Parameters
        ----------

        site_id: int
            The corresponding site_id for a Balsam site. Retrieve via ``balsam site ls``

        num_nodes: int
            The number of nodes to request from a machine with a running Balsam site

        wall_time_min: int
            The number of walltime minutes to request for the BatchJob allocation

        job_mode: String, optional
            Either "serial" or "mpi". Default: "mpi"

        queue: String, optional
            Specifies the queue from which the BatchJob should request nodes. Default: "local"

        project: String, optional
            Specifies the project that should be charged for the requested hours. Default: "local"

        Returns
        -------

        The corresponding ``BatchJob`` object.
        """

        allocation = BatchJob.objects.create(
            site_id=site_id,
            num_nodes=num_nodes,
            wall_time_min=wall_time_min,
            job_mode=job_mode,
            queue=queue,
            project=project,
        )

        self.allocations.append(allocation)

        logger.info(
            "Submitted Batch allocation to site {}: "
            "nodes {} queue {} project {}".format(site_id, num_nodes, queue, project)
        )

        return allocation

    def revoke_allocation(self, allocation):
        """
        Terminates a Balsam BatchJob machine allocation remotely. Balsam apps should
        no longer be submitted to this allocation. Best to run after libEnsemble
        completes, or after this BatchJob is no longer needed. Helps save machine time.

        Parameters
        ----------

        allocation: BatchJob object
            a BatchJob with a corresponding machine allocation that should be cancelled.
        """
        allocation.refresh_from_db()

        while not allocation.scheduler_id:
            time.sleep(1)
            allocation.refresh_from_db()

        batchjob = BatchJob.objects.get(scheduler_id=allocation.scheduler_id)
        batchjob.state = "pending_deletion"
        batchjob.save()

    def set_resources(self, resources):
        self.resources = resources

    def submit(
        self,
        calc_type=None,
        app_name=None,
        app_args=None,
        num_procs=None,
        num_nodes=None,
        procs_per_node=None,
        max_tasks_per_node=None,
        machinefile=None,
        gpus_per_rank=0,
        transfers={},
        workdir="",
        dry_run=False,
        wait_on_start=False,
        extra_args={},
    ):
        """Initializes and submits a Balsam Job based on a registered ApplicationDefinition
        and requested resource parameters. A corresponding libEnsemble Task object
        is created and returned.

        calc_type: String, optional
            The calculation type: 'sim' or 'gen'
            Only used if app_name is not supplied. Uses default sim or gen application.

        app_name: String, optional
            The application name. Must be supplied if calc_type is not.

        app_args: dict
            A dictionary of options that correspond to fields to template in the
            ApplicationDefinition's ``command_template`` field.

        num_procs: int, optional
            The total number of MPI ranks on which to submit the task

        num_nodes: int, optional
            The number of nodes on which to submit the task

        procs_per_node: int, optional
            The processes per node for this task

        max_tasks_per_node: int
            Instructs Balsam to schedule at most this many Jobs per node.

        machinefile: string, optional
            Name of a machinefile for this task to use. Unused by Balsam

        gpus_per_rank: int
            Number of GPUs to reserve for each MPI rank

        transfers: dict
            A Job-specific Balsam transfers dictionary that corresponds with an
            ApplicationDefinition ``transfers`` field. See the Balsam docs for
            more information.

        workdir: String
            Specifies as name for the Job's output directory within the Balsam site's
            data directory. Default: libe_workflow

        dry_run: boolean, optional
            Whether this is a dry_run - no task will be launched; instead
            runline is printed to logger (at INFO level)

        wait_on_start: boolean, optional
            Whether to block, and wait for task to be polled as RUNNING (or other
            active/end state) before continuing

        extra_args: dict
            Additional arguments to supply to MPI runner.

        Returns
        -------

        task: obj: Task
            The launched task object

        Note that since Balsam Jobs are often sent to entirely different machines
        than where libEnsemble is running, that how libEnsemble's resource manager
        has divided local resources among workers doesn't impact what resources
        can be requested for a Balsam Job running on an entirely different machine.

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
            jassert(
                num_procs or num_nodes or procs_per_node,
                "No procs/nodes provided - aborting",
            )

        if not len(self.allocations):
            logger.warning(
                "Balsam Job submitted with no active BatchJobs! Initialize a matching BatchJob."
            )

        task = NewBalsamTask(app, app_args, workdir, None, None, self.workerID)

        if dry_run:
            task.dry_run = True
            logger.info("Test (No submit) Balsam app {}".format(app_name))
            task._set_complete(dry_run=True)
        else:
            App = app.pyobj
            App.sync()
            task.process = Job(
                app_id=App,
                workdir=workdir,
                parameters=app_args,
                num_nodes=num_nodes,
                ranks_per_node=procs_per_node,
                launch_params=extra_args,
                gpus_per_rank=gpus_per_rank,
                node_packing_count=max_tasks_per_node,
                transfers=transfers,
            )

            task.process.save()

            if wait_on_start:
                self._wait_on_start(task)

            if not task.timer.timing:
                task.timer.start()
                task.submit_time = (
                    task.timer.tstart
                )  # Time not date - may not need if using timer.

            logger.info(
                "Submitted Balsam App to site {}: "
                "nodes {} ppn {}".format(App.site, num_nodes, procs_per_node)
            )

            # task.workdir = task.process.working_directory  # Might not be set yet!
        self.list_of_tasks.append(task)
        return task
