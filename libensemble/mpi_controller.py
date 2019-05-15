"""
Module to launch and control running MPI jobs.

"""

import os
import logging
import time

import libensemble.util.launcher as launcher
from libensemble.mpi_resources import MPIResources
from libensemble.controller import JobController, Job, jassert

logger = logging.getLogger(__name__)
# To change logging level for just this module
# logger.setLevel(logging.DEBUG)


class MPIJobController(JobController):
    """The MPI job_controller can create, poll and kill runnable MPI jobs
    """

    def __init__(self, auto_resources=True, central_mode=False,
                 nodelist_env_slurm=None,
                 nodelist_env_cobalt=None,
                 nodelist_env_lsf=None):
        """Instantiate a new JobController instance.

        A new JobController object is created with an application
        registry and configuration attributes. A registry object must
        have been created.

        This is typically created in the user calling script. If
        auto_resources is True, an evaluation of system resources is
        performance during this call.

        Parameters
        ----------
        auto_resources: Boolean, optional
            Auto-detect available processor resources and assign to jobs
            if not explicitly provided on launch.

        central_mode, optional: boolean:
            If true, then running in central mode, else distributed.
            Central mode means libE processes (manager and workers) are grouped together and
            do not share nodes with applications. Distributed mode means Workers share nodes
            with applications.

        nodelist_env_slurm: String, optional
            The environment variable giving a node list in Slurm format
            (Default: Uses SLURM_NODELIST).  Note: This is only queried if
            a worker_list file is not provided and auto_resources=True.

        nodelist_env_cobalt: String, optional
            The environment variable giving a node list in Cobalt format
            (Default: Uses COBALT_PARTNAME) Note: This is only queried
            if a worker_list file is not provided and
            auto_resources=True.

        nodelist_env_lsf: String, optional
            The environment variable giving a node list in LSF format
            (Default: Uses LSB_HOSTS) Note: This is only queried
            if a worker_list file is not provided and
            auto_resources=True.
        """

        JobController.__init__(self)
        self.max_launch_attempts = 5
        self.fail_time = 2
        self.auto_resources = auto_resources

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
            'jsrun': ['jsrun', '--np {num_procs}']
        }
        self.mpi_launch_type = MPIResources.get_MPI_variant()
        self.mpi_command = mpi_commands[self.mpi_launch_type]

        if self.auto_resources:
            self.resources = \
                MPIResources(top_level_dir=self.top_level_dir,
                             central_mode=central_mode,
                             launcher=self.mpi_command[0],
                             nodelist_env_slurm=nodelist_env_slurm,
                             nodelist_env_cobalt=nodelist_env_cobalt,
                             nodelist_env_lsf=nodelist_env_lsf)

    def _get_mpi_specs(self, num_procs, num_nodes, ranks_per_node,
                       machinefile, hyperthreads):
        "Form the mpi_specs dictionary."
        hostlist = None
        if machinefile is None and self.auto_resources:
            num_procs, num_nodes, ranks_per_node = \
                self.resources.get_resources(num_procs=num_procs,
                                             num_nodes=num_nodes,
                                             ranks_per_node=ranks_per_node,
                                             hyperthreads=hyperthreads)

            # Use hostlist if multiple nodes, otherwise machinefile
            if num_nodes > 1:
                hostlist = self.resources.get_hostlist()
            else:
                machinefile = "machinefile_autogen"
                if self.workerID is not None:
                    machinefile += "_for_worker_{}".format(self.workerID)
                mfile_created, num_procs, num_nodes, ranks_per_node = \
                    self.resources.create_machinefile(
                        machinefile, num_procs, num_nodes,
                        ranks_per_node, hyperthreads)
                jassert(mfile_created, "Auto-creation of machinefile failed")

        else:
            num_procs, num_nodes, ranks_per_node = \
                MPIResources.job_partition(num_procs, num_nodes,
                                           ranks_per_node, machinefile)

        return {'num_procs': num_procs,
                'num_nodes': num_nodes,
                'ranks_per_node': ranks_per_node,
                'machinefile': machinefile,
                'hostlist': hostlist}

    def launch(self, calc_type, num_procs=None, num_nodes=None,
               ranks_per_node=None, machinefile=None, app_args=None,
               stdout=None, stderr=None, stage_inout=None,
               hyperthreads=False, test=False, wait_on_run=False):
        """Creates a new job, and either launches or schedules launch.

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
            A string of the application arguments to be added to job
            launch command line.

        stdout: string, optional
            A standard output filename.

        stderr: string, optional
            A standard error filename.

        stage_inout: string, optional
            A directory to copy files from. Default will take from
            current directory.

        hyperthreads: boolean, optional
            Whether to launch MPI tasks to hyperthreads

        test: boolean, optional
            Whether this is a test - No job will be launched. Instead
            runline is printed to logger (At INFO level).

        wait_on_run: boolean, optional
            Whether to wait for job to be polled as RUNNING (or other active/end state) before continuing.


        Returns
        -------

        job: obj: Job
            The lauched job object.


        Note that if some combination of num_procs, num_nodes and
        ranks_per_node are provided, these will be honored if
        possible. If resource detection is on and these are omitted,
        then the available resources will be divided amongst workers.
        """

        app = self.default_app(calc_type)
        default_workdir = os.getcwd()
        job = Job(app, app_args, default_workdir, stdout, stderr, self.workerID)

        if stage_inout is not None:
            logger.warning("stage_inout option ignored in this "
                           "job_controller - runs in-place")

        mpi_specs = self._get_mpi_specs(num_procs, num_nodes, ranks_per_node,
                                        machinefile, hyperthreads)
        runline = launcher.form_command(self.mpi_command, mpi_specs)
        runline.append(job.app.full_path)
        if job.app_args is not None:
            runline.extend(job.app_args.split())

        if test:
            logger.info('Test selected: Not launching job')
            logger.info('runline args are {}'.format(runline))
        else:
            logger.info("Launching job {}: {}".
                        format(job.name, " ".join(runline)))  # One line

            subgroup_launch = True
            if self.mpi_launch_type in ['aprun']:
                subgroup_launch = False

            retry_count = 0
            while retry_count < self.max_launch_attempts:
                retry = False
                try:
                    job.process = launcher.launch(runline, cwd='./',
                                                  stdout=open(job.stdout, 'w'),
                                                  stderr=open(job.stderr, 'w'),
                                                  start_new_session=subgroup_launch)
                except Exception as e:
                    logger.warning('job {} launch command failed on try {} with error {}'.format(job.name, retry_count, e))
                    retry = True
                    retry_count += 1
                else:
                    if (wait_on_run):
                        self._wait_on_run(job, self.fail_time)

                    if job.state == 'FAILED':
                        logger.warning('job {} failed immediately on try {} with err code {}'.format(job.name, retry_count, job.errcode))
                        retry = True
                        retry_count += 1

                if retry and retry_count < self.max_launch_attempts:
                    # retry_count += 1 # Do not want to reset job if not going to retry.
                    time.sleep(retry_count*5)
                    job.reset()  # Note: Some cases may require user cleanup - currently not supported (could use callback)
                else:
                    break

            if not job.timer.timing:
                job.timer.start()
                job.launch_time = job.timer.tstart  # Time not date - may not need if using timer.

            self.list_of_jobs.append(job)

        return job

    def set_worker_info(self, comm, workerid=None):
        """Sets info for this job_controller"""
        self.workerID = workerid
        if self.workerID and self.auto_resources:
            self.resources.set_worker_resources(self.workerID, comm)
