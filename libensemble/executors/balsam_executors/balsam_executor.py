"""
This module launches and controls the running of tasks with Balsam_, and most
notably can submit tasks from any machine, to any machine running a Balsam site_.

.. image:: ../images/balsam2.png
    :alt: central_balsam
    :scale: 40
    :align: center

At this time, access to Balsam is limited to those with valid organizational logins
authenticated through Globus_.

In order to initiate a Balsam executor, the calling script should contain ::

    from libensemble.executors.balsam_executors import BalsamExecutor
    exctr = BalsamExecutor()

Key differences to consider between this executor and libEnsemble's others is
Balsam ``ApplicationDefinition`` instances are registered instead of paths and task
submissions will not run until Balsam reserves compute resources at a site.

This process may resemble::

    from libensemble.executors.balsam_executors import BalsamExecutor
    from balsam.api import ApplicationDefinition

    class HelloApp(ApplicationDefinition):
        site = "my-balsam-site"
        command_template = "/path/to/hello.app {{ my_name }}"

    exctr = BalsamExecutor()
    exctr.register_app(HelloApp, app_name="hello")

    exctr.submit_allocation(
        site_id=999,  # corresponds to "my-balsam-site", found via ``balsam site ls``
        num_nodes=4,  # Total number of nodes requested for *all jobs*
        wall_time_min=30,
        queue="debug-queue",
        project="my-project",
    )

Task submissions of registered apps aren't too different from the other executors,
except Balsam expects application arguments in dictionary form. Note that these fields
must match the templating syntax in each ``ApplicationDefinition``'s ``command_template``
field::

    args = {"my_name": "World"}

    task = exctr.submit(
        app_name="hello",
        app_args=args,
        num_procs=4,
        num_nodes=1,
        procs_per_node=4,
    )

Application instances submitted by the executor to the Balsam service will get
scheduled within the reserved resource allocation. **Each Balsam app can only be
submitted to the site specified in its class definition.** Output files will appear
in the Balsam site's ``data`` directory, but can be automatically `transferred back`_
via Globus.

**Reading Balsam's documentation is highly recommended.**

.. _site: https://balsam.readthedocs.io/en/latest/user-guide/site-config/
.. _Balsam: https://balsam.readthedocs.io/en/latest/
.. _`transferred back`: https://balsam.readthedocs.io/en/latest/user-guide/transfer/
.. _Globus: https://www.globus.org/
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
)
from libensemble.executors import Executor

from balsam.api import Job, BatchJob, EventLog

logger = logging.getLogger(__name__)
# To change logging level for just this module
# logger.setLevel(logging.DEBUG)


class BalsamTask(Task):
    """Wraps a Balsam ``Job`` from the Balsam service.

    The same attributes and query routines are implemented. Use ``task.process``
    to refer to the matching Balsam ``Job`` initialized by the ``BalsamExecutor``,
    with every Balsam ``Job`` method invocable on it. Otherwise, libEnsemble task methods
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
        """Instantiate a new ``BalsamTask`` instance.

        A new ``BalsamTask`` object is created with an id, status and
        configuration attributes.  This will normally be created by the
        executor on a submission.
        """
        # May want to override workdir with Balsam value when it exists
        Task.__init__(self, app, app_args, workdir, stdout, stderr, workerid)

    def _get_time_since_balsam_submit(self):
        """Return time since balsam task entered ``RUNNING`` state"""
        event_query = EventLog.objects.filter(job_id=self.process.id, to_state="RUNNING")
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
            else:
                self.state = balsam_state

        logger.info(f"Task {self.name} ended with state {self.state}")

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
            self._set_complete()

    def wait(self, timeout=None):
        """Waits on completion of the task or raises ``TimeoutExpired``.

        Status attributes of task are updated on completion.

        Parameters
        ----------

        timeout: int or float,  optional
            Time in seconds after which a TimeoutExpired exception is raised.
            If not set, then simply waits until completion.
            Note that the task is not automatically killed on timeout.
        """

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
            "RUN_ERROR",
            "RUN_TIMEOUT",
            "FAILED",
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

        logger.info(f"Killing task {self.name}")
        self.state = "USER_KILLED"
        self.finished = True
        self.calc_task_timing()


class BalsamExecutor(Executor):
    """Inherits from ``Executor`` and wraps the Balsam service. Via this Executor,
    Balsam ``Jobs`` can be submitted to Balsam sites, either local or on remote machines.

    .. note::  Task kills are not configurable in the Balsam executor.

    """

    def __init__(self):
        """Instantiate a new ``BalsamExecutor`` instance."""
        super().__init__()

        self.workflow_name = "libe_workflow"
        self.allocations = []

    def serial_setup(self):
        """Balsam serial setup includes emptying database and adding applications"""
        pass

    def add_app(self, *args):
        """Sync application with Balsam service"""
        pass

    def register_app(self, BalsamApp, app_name=None, calc_type=None, desc=None, precedent=None):
        """Registers a Balsam ``ApplicationDefinition`` to libEnsemble. This class
        instance *must* have a ``site`` and ``command_template`` specified. See
        the Balsam docs for information on other optional fields.

        Parameters
        ----------

        BalsamApp: ``ApplicationDefinition`` object
            A Balsam ``ApplicationDefinition`` instance.

        app_name: String, optional
            Name to identify this application.

        calc_type: String, optional
            Calculation type: Set this application as the default ``'sim'``
            or ``'gen'`` function.

        desc: String, optional
            Description of this application

        """

        if precedent is not None:
            logger.warning("precedent is ignored in Balsam executor - add to command template")

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
        optional_params={},
        filter_tags={},
        partitions=[],
    ):
        """
        Submits a Balsam ``BatchJob`` machine allocation request to Balsam.
        Corresponding Balsam applications with a matching site can be submitted to
        this allocation. Effectively a wrapper for ``BatchJob.objects.create()``.

        Parameters
        ----------

        site_id: int
            The corresponding ``site_id`` for a Balsam site. Retrieve via ``balsam site ls``

        num_nodes: int
            The number of nodes to request from a machine with a running Balsam site

        wall_time_min: int
            The number of walltime minutes to request for the ``BatchJob`` allocation

        job_mode: String, optional
            Either ``"serial"`` or ``"mpi"``. Default: ``"mpi"``

        queue: String, optional
            Specifies the queue from which the ``BatchJob`` should request nodes. Default: ``"local"``

        project: String, optional
            Specifies the project that should be charged for the requested machine time. Default: ``"local"``

        optional_params: dict, optional
            Additional system-specific parameters to set, based on fields in Balsam's ``job-template.sh``

        filter_tags: dict, optional
            Directs the resultant ``BatchJob`` to only run Jobs with matching tags.

        partitions: list of dicts, optional
            Divides the allocation into multiple launcher partitions, with differing
            ``job_mode``, ``num_nodes``. ``filter_tags``, etc. See the Balsam docs.

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
            optional_params=optional_params,
            filter_tags=filter_tags,
            partitions=partitions,
        )

        self.allocations.append(allocation)

        logger.info(
            f"Submitted Batch allocation to site {site_id}: " f"nodes {num_nodes} queue {queue} project {project}"
        )

        return allocation

    def revoke_allocation(self, allocation, timeout=60):
        """
        Terminates a Balsam ``BatchJob`` machine allocation remotely. Balsam apps should
        no longer be submitted to this allocation. Best to run after libEnsemble
        completes, or after this ``BatchJob`` is no longer needed. Helps save machine time.

        Parameters
        ----------

        allocation: ``BatchJob`` object
            a ``BatchJob`` with a corresponding machine allocation that should be cancelled.

        timeout: int, optional
            Timeout and warn user after this many seconds of attempting to revoke an allocation.
        """
        allocation.refresh_from_db()

        start = time.time()

        while not allocation.scheduler_id:
            time.sleep(1)
            allocation.refresh_from_db()
            if time.time() - start > timeout:
                logger.warning(
                    "Unable to terminate Balsam BatchJob. You may need to login to the machine and manually remove it."
                )
                return False

        batchjob = BatchJob.objects.get(scheduler_id=allocation.scheduler_id)
        batchjob.state = "pending_deletion"
        batchjob.save()
        return True

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
        tags={},
    ):
        """Initializes and submits a Balsam ``Job`` based on a registered ``ApplicationDefinition``
        and requested resources. A corresponding libEnsemble ``Task`` object is returned.

        calc_type: String, optional
            The calculation type: ``'sim'`` or ``'gen'``
            Only used if ``app_name`` is not supplied. Uses default sim or gen application.

        app_name: String, optional
            The application name. Must be supplied if ``calc_type`` is not.

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

        gpus_per_rank: int, optional
            Number of GPUs to reserve for each MPI rank

        transfers: dict, optional
            A Job-specific Balsam transfers dictionary that corresponds with an
            ``ApplicationDefinition`` ``transfers`` field. See the Balsam docs for
            more information.

        workdir: String
            Specifies as name for the Job's output directory within the Balsam site's
            data directory. Default: ``libe_workflow``

        dry_run: boolean, optional
            Whether this is a dry run - no task will be launched; instead
            runline is printed to logger (at ``INFO`` level)

        wait_on_start: boolean, optional
            Whether to block, and wait for task to be polled as ``RUNNING`` (or other
            active/end state) before continuing

        extra_args: dict, optional
            Additional arguments to supply to MPI runner.

        tags: dict, optional
            Additional tags to organize the ``Job`` or restrict which ``BatchJobs`` run it.

        Returns
        -------

        task: obj: Task
            The launched task object

        Note that since Balsam Jobs are often sent to entirely different machines
        than where libEnsemble is running, how libEnsemble's resource manager
        has divided local resources among workers doesn't impact what resources
        can be requested for a Balsam ``Job`` running on an entirely different machine.

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

        if machinefile is not None:
            logger.warning("machinefile arg ignored - not supported in Balsam")

        task = BalsamTask(app, app_args, workdir, None, None, self.workerID)

        if dry_run:
            task.dry_run = True
            logger.info(f"Test (No submit) Balsam app {app_name}")
            task._set_complete(dry_run=True)
        else:
            App = app.pyobj

            try:
                App.sync()  # if App source-code available, send to Balsam service
            except OSError:
                pass  # App retrieved from Balsam service, assume no access to source-code

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

            if not task.timer.timing and not task.finished:
                task.timer.start()
                task.submit_time = task.timer.tstart  # Time not date - may not need if using timer.

            logger.info(f"Submitted Balsam App to site {App.site}: " "nodes {num_nodes} ppn {procs_per_node}")

        self.list_of_tasks.append(task)
        return task
