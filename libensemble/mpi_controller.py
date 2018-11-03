"""
Module to launch and control running MPI jobs.

"""

import os
import logging
import time

import libensemble.launcher as launcher
from libensemble.mpi_resources import MPIResources
from libensemble.controller import JobController, Job, jassert

logger = logging.getLogger(__name__ + '(' + MPIResources.get_my_name() + ')')
#For debug messages in this module  - uncomment
#(see libE.py to change root logging level)
#logger.setLevel(logging.DEBUG)


class MPIJobController(JobController):
    """The MPI job_controller can create, poll and kill runnable MPI jobs
    """

    def __init__(self, auto_resources=True, central_mode=False,
                 nodelist_env_slurm=None, nodelist_env_cobalt=None):
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
        """

        JobController.__init__(self)
        self.auto_resources = auto_resources
        if self.auto_resources:
            self.resources = \
              MPIResources(top_level_dir=self.top_level_dir,
                           central_mode=central_mode,
                           nodelist_env_slurm=nodelist_env_slurm,
                           nodelist_env_cobalt=nodelist_env_cobalt)

        mpi_commands = {
            'mpich':   ['mpirun', '--env {env}', '-machinefile {machinefile}',
                        '-hosts {hostlist}', '-np {num_procs}',
                        '--ppn {ranks_per_node}'],
            'openmpi': ['mpirun', '-x {env}', '-machinefile {machinefile}',
                        '-host {hostlist}', '-np {num_procs}',
                        '-npernode {ranks_per_node}'],
        }
        self.mpi_command = mpi_commands[MPIResources.get_MPI_variant()]


    def _get_mpi_specs(self, num_procs, num_nodes, ranks_per_node,
                       machinefile, hyperthreads):
        "Form the mpi_specs dictionary."
        hostlist = None
        if machinefile is None and self.auto_resources:

            #kludging this for now - not nec machinefile if more than one node
            #- try a hostlist
            num_procs, num_nodes, ranks_per_node = \
              self.resources.get_resources(
                  num_procs=num_procs,
                  num_nodes=num_nodes, ranks_per_node=ranks_per_node,
                  hyperthreads=hyperthreads)

            if num_nodes > 1:
                #hostlist
                hostlist = self.resources.get_hostlist()
            else:
                #machinefile
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
               hyperthreads=False, test=False):
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
            logger.debug("Launching job {}: {}".
                         format(job.name, " ".join(runline))) #One line
            job.launch_time = time.time()
            job.process = launcher.launch(runline, cwd='./',
                                          stdout=open(job.stdout, 'w'),
                                          stderr=open(job.stderr, 'w'),
                                          start_new_session=True)
            self.list_of_jobs.append(job)

        return job
