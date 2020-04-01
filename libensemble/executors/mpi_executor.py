"""
This module launches and controls the running of MPI applications.

In order to create an MPI executor, the calling script should contain ::

    exctr = MPIExecutor()

See the executor API below for optional arguments.
"""

import os
import logging
import time

import libensemble.utils.launcher as launcher
from libensemble.resources.mpi_resources import MPIResources
from libensemble.executors.executor import Executor, Task, jassert

logger = logging.getLogger(__name__)
# To change logging level for just this module
# logger.setLevel(logging.DEBUG)


class MPIExecutor(Executor):
    """The MPI executor can create, poll and kill runnable MPI tasks
    """

    def __init__(self, auto_resources=True,
                 allow_oversubscribe=True,
                 central_mode=False,
                 nodelist_env_slurm=None,
                 nodelist_env_cobalt=None,
                 nodelist_env_lsf=None,
                 nodelist_env_lsf_shortform=None):
        """Instantiate a new MPIExecutor instance.

        A new Executor MPIExecutor is created with an application
        registry and configuration attributes. A registry object must
        have been created.

        This is typically created in the user calling script. If
        auto_resources is true, an evaluation of system resources is
        performed during this call.

        Parameters
        ----------

        auto_resources: boolean, optional
            Autodetect available processor resources and assign to tasks
            if not explicitly provided on submission.

        allow_oversubscribe: boolean, optional
            If true, the Executor will permit submission of tasks with a
            higher processor count than the CPUs available to the worker as
            detected by auto_resources. Larger node counts are not allowed.
            When auto_resources is off, this argument is ignored.

        central_mode: boolean, optional
            If true, then running in central mode, otherwise in distributed
            mode. Central mode means libE processes (manager and workers) are
            grouped together and do not share nodes with applications.
            Distributed mode means workers share nodes with applications.

        nodelist_env_slurm: String, optional
            The environment variable giving a node list in Slurm format
            (Default: Uses SLURM_NODELIST).  Note: This is queried only if
            a worker_list file is not provided and auto_resources=True.

        nodelist_env_cobalt: String, optional
            The environment variable giving a node list in Cobalt format
            (Default: Uses COBALT_PARTNAME) Note: This is queried only
            if a worker_list file is not provided and
            auto_resources=True.

        nodelist_env_lsf: String, optional
            The environment variable giving a node list in LSF format
            (Default: Uses LSB_HOSTS) Note: This is queried only
            if a worker_list file is not provided and
            auto_resources=True.

        nodelist_env_lsf_shortform: String, optional
            The environment variable giving a node list in LSF short-form
            format (Default: Uses LSB_MCPU_HOSTS) Note: This is queried only
            if a worker_list file is not provided and auto_resources=True.

        """

        Executor.__init__(self)
        self.auto_resources = auto_resources

        # MPI launch settings
        self.max_launch_attempts = 5
        self.fail_time = 2

        mpi_commands = {
            'mpich': ['mpirun', '--env {env}', '-machinefile {machinefile}',
                      '-hosts {hostlist}', '-np {num_procs}',
                      '--ppn {ranks_per_node}'],
            'openmpi': ['mpirun', '-x {env}', '-machinefile {machinefile}',
                        '-host {hostlist}', '-np {num_procs}',
                        '-npernode {ranks_per_node}'],
            'aprun': ['aprun', '-e {env}',
                      '-L {hostlist}', '-n {num_procs}',
                      '-N {ranks_per_node}'],
            'jsrun': ['jsrun', '--np {num_procs}'],  # Need to add more
            'srun': ['srun', '-w {hostlist}', '-n {num_procs}',
                     '--nodes {num_nodes}',
                     '--ntasks-per-node {ranks_per_node}']
        }
        self.mpi_launch_type = MPIResources.get_MPI_variant()
        self.mpi_command = mpi_commands[self.mpi_launch_type]

        self.subgroup_launch = True
        if self.mpi_launch_type in ['aprun', 'srun']:
            self.subgroup_launch = False

        if self.auto_resources:
            self.resources = \
                MPIResources(top_level_dir=self.top_level_dir,
                             central_mode=central_mode,
                             allow_oversubscribe=allow_oversubscribe,
                             launcher=self.mpi_command[0],
                             nodelist_env_slurm=nodelist_env_slurm,
                             nodelist_env_cobalt=nodelist_env_cobalt,
                             nodelist_env_lsf=nodelist_env_lsf,
                             nodelist_env_lsf_shortform=nodelist_env_lsf_shortform)

    def add_comm_info(self, libE_nodes, serial_setup):
        """Adds comm-specific information to executor.

        Updates resources information if auto_resources is true.
        """
        if self.auto_resources:
            self.resources.add_comm_info(libE_nodes=libE_nodes)
        if serial_setup:
            self._serial_setup()

    def _get_mpi_specs(self, task, num_procs, num_nodes,
                       ranks_per_node, machinefile,
                       hyperthreads):
        "Form the mpi_specs dictionary."
        hostlist = None
        if machinefile is None and self.auto_resources:
            num_procs, num_nodes, ranks_per_node = \
                self.resources.get_resources(num_procs=num_procs,
                                             num_nodes=num_nodes,
                                             ranks_per_node=ranks_per_node,
                                             hyperthreads=hyperthreads)

            # Use hostlist if full nodes, otherwise machinefile
            if self.resources.worker_resources.workers_per_node == 1:
                hostlist = self.resources.get_hostlist()
            else:
                machinefile = "machinefile_autogen"
                if self.workerID is not None:
                    machinefile += "_for_worker_{}".format(self.workerID)
                machinefile += "_task_{}".format(task.id)
                mfile_created, num_procs, num_nodes, ranks_per_node = \
                    self.resources.create_machinefile(
                        machinefile, num_procs, num_nodes,
                        ranks_per_node, hyperthreads)
                jassert(mfile_created, "Auto-creation of machinefile failed")

        else:
            num_procs, num_nodes, ranks_per_node = \
                MPIResources.task_partition(num_procs, num_nodes,
                                            ranks_per_node, machinefile)

        return {'num_procs': num_procs,
                'num_nodes': num_nodes,
                'ranks_per_node': ranks_per_node,
                'machinefile': machinefile,
                'hostlist': hostlist}

    def _launch_with_retries(self, task, runline, wait_on_run):
        """ Launch task with retry mechanism"""
        retry_count = 0
        while retry_count < self.max_launch_attempts:
            retry = False
            try:
                retry_string = " (Retry {})".format(retry_count) if retry_count > 0 else ""
                logger.info("Launching task {}{}: {}".
                            format(task.name, retry_string, " ".join(runline)))

                task.process = launcher.launch(runline, cwd='./',
                                               stdout=open(task.stdout, 'w'),
                                               stderr=open(task.stderr, 'w'),
                                               start_new_session=self.subgroup_launch)
            except Exception as e:
                logger.warning('task {} submit command failed on "\
                    "try {} with error {}'.format(task.name, retry_count, e))
                retry = True
                retry_count += 1
            else:
                if (wait_on_run):
                    self._wait_on_run(task, self.fail_time)

                if task.state == 'FAILED':
                    logger.warning('task {} failed within fail_time on"\
                        "try {} with err code {}'.format(task.name, retry_count, task.errcode))
                    retry = True
                    retry_count += 1

            if retry and retry_count < self.max_launch_attempts:
                logger.debug('Retry number {} for task {}')
                time.sleep(retry_count*5)
                task.reset()  # Some cases may require user cleanup - currently not supported (could use callback)
            else:
                break

    def submit(self, calc_type, num_procs=None, num_nodes=None,
               ranks_per_node=None, machinefile=None, app_args=None,
               stdout=None, stderr=None, stage_inout=None,
               hyperthreads=False, test=False, wait_on_run=False):
        """Creates a new task, and either executes or schedules execution.

        The created task object is returned.

        Parameters
        ----------

        calc_type: String
            The calculation type: 'sim' or 'gen'

        num_procs: int, optional
            The total number of MPI tasks on which to submit the task

        num_nodes: int, optional
            The number of nodes on which to submit the task

        ranks_per_node: int, optional
            The ranks per node for this task

        machinefile: string, optional
            Name of a machinefile for this task to use

        app_args: string, optional
            A string of the application arguments to be added to task
            submit command line

        stdout: string, optional
            A standard output filename

        stderr: string, optional
            A standard error filename

        stage_inout: string, optional
            A directory to copy files from; default will take from
            current directory

        hyperthreads: boolean, optional
            Whether to submit MPI tasks to hyperthreads

        test: boolean, optional
            Whether this is a test - no task will be launched; instead
            runline is printed to logger (at INFO level)

        wait_on_run: boolean, optional
            Whether to wait for task to be polled as RUNNING (or other
            active/end state) before continuing


        Returns
        -------

        task: obj: Task
            The lauched task object


        Note that if some combination of num_procs, num_nodes, and
        ranks_per_node is provided, these will be honored if
        possible. If resource detection is on and these are omitted,
        then the available resources will be divided among workers.
        """

        app = self.default_app(calc_type)
        default_workdir = os.getcwd()
        task = Task(app, app_args, default_workdir, stdout, stderr, self.workerID)

        if stage_inout is not None:
            logger.warning("stage_inout option ignored in this "
                           "executor - runs in-place")

        mpi_specs = self._get_mpi_specs(task, num_procs, num_nodes,
                                        ranks_per_node, machinefile,
                                        hyperthreads)
        runline = launcher.form_command(self.mpi_command, mpi_specs)
        runline.extend(task.app.full_path.split())
        if task.app_args is not None:
            runline.extend(task.app_args.split())

        if test:
            logger.info('Test (No submit) Runline: {}'.format(' '.join(runline)))
        else:
            # Launch Task
            self._launch_with_retries(task, runline, wait_on_run)

            if not task.timer.timing:
                task.timer.start()
                task.submit_time = task.timer.tstart  # Time not date - may not need if using timer.

            self.list_of_tasks.append(task)
        return task

    def set_worker_info(self, comm, workerid=None):
        """Sets info for this executor"""
        self.workerID = workerid
        if self.workerID and self.auto_resources:
            self.resources.set_worker_resources(self.workerID, comm)
