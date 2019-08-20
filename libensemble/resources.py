"""
Module for detecting and returning system resources

"""

import os
import socket
import logging
import itertools
import subprocess
# from collections import OrderedDict
from libensemble import node_resources
from libensemble.env_resources import EnvResources

logger = logging.getLogger(__name__)
# To change logging level for just this module
# logger.setLevel(logging.DEBUG)


class ResourcesException(Exception):
    "Resources module exception."


class Resources:
    """Provide system resources to libEnsemble and job controller.

    This is intialised when the job_controller is created with auto_resources set to True.

    **Object Attributes:**

    These are set on initialisation.

    :ivar string top_level_dir: Directory where searches for worker_list file.
    :ivar boolean central_mode: If true, then running in central mode, else distributed.
    :ivar EnvResources env_resources: An object storing environment variables used by resources.
    :ivar list global_nodelist: A list of all nodes available for running user applications
    :ivar int logical_cores_avail_per_node: Logical cores (including SMT threads) available on a node.
    :ivar int physical_cores_avail_per_node: Physical cores available on a node.
    :ivar WorkerResources worker_resources: An object that can contain worker specific resources.
    """

    def __init__(self, top_level_dir=None, central_mode=False, launcher=None,
                 nodelist_env_slurm=None,
                 nodelist_env_cobalt=None,
                 nodelist_env_lsf=None,
                 nodelist_env_lsf_shortform=None):

        """Initialise new Resources instance

        Works out the compute resources available for current allocation, including
        node list and cores/hardware threads available within nodes.

        Parameters
        ----------

        top_level_dir: string, optional
            Directory libEnsemble runs in (default is current working directory)

        central_mode: boolean, optional
            If true, then running in central mode, else distributed.
            Central mode means libE processes (manager and workers) are grouped together and
            do not share nodes with applications. Distributed mode means Workers share nodes
            with applications.

        launcher: String, optional
            The name of the job launcher such as mpirun or aprun. This may be used to obtain
            intra-node information by launching a probing job onto the compute nodes.
            If not present, the local node will be used to obtain this information.

        nodelist_env_slurm: String, optional
            The environment variable giving a node list in Slurm format (Default: Uses SLURM_NODELIST)
            Note: This is only queried if a worker_list file is not provided and auto_resources=True.

        nodelist_env_cobalt: String, optional
            The environment variable giving a node list in Cobalt format (Default: Uses COBALT_PARTNAME)
            Note: This is only queried if a worker_list file is not provided and auto_resources=True.

        nodelist_env_lsf: String, optional
            The environment variable giving a node list in LSF format (Default: Uses LSB_HOSTS)
            Note: This is only queried if a worker_list file is not provided and auto_resources=True.

        nodelist_env_lsf_shortform: String, optional
            The environment variable giving a node list in LSF short-form format (Default: Uses LSB_MCPU_HOSTS)
            Note: This is only queried if a worker_list file is not provided and auto_resources=True.

        """

        self.top_level_dir = top_level_dir or os.getcwd()
        self.central_mode = central_mode
        if self.central_mode:
            logger.debug('Running in central mode')

        self.env_resources = EnvResources(nodelist_env_slurm=nodelist_env_slurm,
                                          nodelist_env_cobalt=nodelist_env_cobalt,
                                          nodelist_env_lsf=nodelist_env_lsf,
                                          nodelist_env_lsf_shortform=nodelist_env_lsf_shortform)

        # This is global nodelist avail to workers - may change to global_worker_nodelist
        self.global_nodelist = Resources.get_global_nodelist(rundir=self.top_level_dir,
                                                             env_resources=self.env_resources)
        remote_detect = False
        if socket.gethostname() not in self.global_nodelist:
            remote_detect = True

        cores_info = node_resources.get_sub_node_resources(launcher=launcher,
                                                           remote_mode=remote_detect,
                                                           env_resources=self.env_resources)
        self.logical_cores_avail_per_node = cores_info[0]
        self.physical_cores_avail_per_node = cores_info[1]

        self.libE_nodes = None
        self.worker_resources = None

    def add_comm_info(self, libE_nodes):
        """Add comms specific information to resources

        Removes libEnsemble nodes from nodelist if in central_mode.
        """
        self.libE_nodes = self.env_resources.abbrev_nodenames(libE_nodes)
        libE_nodes_in_list = list(filter(lambda x: x in self.libE_nodes, self.global_nodelist))
        if libE_nodes_in_list:
            if self.central_mode and len(self.global_nodelist) > 1:
                self.global_nodelist = Resources.remove_nodes(self.global_nodelist, self.libE_nodes)
                if not self.global_nodelist:
                    logger.warning("Warning. Node-list for sub-jobs is empty. Remove central_mode or add nodes")

    def set_worker_resources(self, workerid, comm):
        self.worker_resources = WorkerResources(workerid, comm, self)

    @staticmethod
    def get_MPI_variant():
        """Returns MPI base implementation

        Returns
        -------
        mpi_variant: string:
            MPI variant 'aprun' or 'jsrun' or 'mpich' or 'openmpi'

        """

        try:
            subprocess.check_call(['aprun', '--version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return 'aprun'
        except OSError:
            pass

        try:
            subprocess.check_call(['jsrun', '--version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return 'jsrun'
        except OSError:
            pass

        # Explore mpi4py.MPI.get_vendor() and mpi4py.MPI.Get_library_version() for mpi4py
        try_mpich = subprocess.Popen(['mpirun', '-npernode'], stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT)
        stdout, _ = try_mpich.communicate()
        if 'unrecognized argument npernode' in stdout.decode():
            return 'mpich'
        return 'openmpi'

    # ---------------------------------------------------------------------------

    # This is for central mode where libE nodes will not share with app nodes
    @staticmethod
    def remove_nodes(global_nodelist_in, remove_list):
        """Any nodes in remove_list are removed from the global nodelist"""
        global_nodelist = list(filter(lambda x: x not in remove_list, global_nodelist_in))
        return global_nodelist

    @staticmethod
    def best_split(a, n):
        """Create the most even split of list a into n parts and return list of lists"""
        k, m = divmod(len(a), n)
        return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

    @staticmethod
    def get_global_nodelist(rundir=None,
                            env_resources=None):
        """
        Return the list of nodes available to all libEnsemble workers

        If a worker_list file exists this is used, otherwise the environment
        is interrogated for a node list. If a dedicated manager node is used,
        then a worker_list file is recommended.

        In central mode, any node with a libE worker is removed from the list.
        """
        top_level_dir = rundir or os.getcwd()
        worker_list_file = os.path.join(top_level_dir, 'worker_list')
        global_nodelist = []
        if os.path.isfile(worker_list_file):
            logger.debug("worker_list found - getting nodelist from worker_list")
            with open(worker_list_file, 'r') as f:
                for line in f:
                    global_nodelist.append(line.rstrip())
        else:
            logger.debug("No worker_list found - searching for nodelist in environment")
            if env_resources:
                global_nodelist = env_resources.get_nodelist()

            if not global_nodelist:
                # Assume a standalone machine
                logger.info("Can not find nodelist from environment. Assuming standalone")
                global_nodelist.append(socket.gethostname())

        if global_nodelist:
            return global_nodelist
        raise ResourcesException("Error. global_nodelist is empty")


class WorkerResources:
    """Provide system resources per worker to libEnsemble and job controller.

    **Object Attributes:**

    These are set on initialisation.

    :ivar int num_workers: Total number of workers
    :ivar int workerID: workerID
    :ivar list local_nodelist: A list of all nodes assigned to this worker
    :ivar int local_node_count: The number of nodes available to this worker (rounded up to whole number)
    :ivar int workers_per_node: The number of workers per node (if using sub-node workers)
    """

    def __init__(self, workerID, comm, resources):
        """Initialise new WorkerResources instance

        Works out the compute resources available for current worker, including
        node list and cores/hardware threads available within nodes.

        Parameters
        ----------

        workerID: int
            workerID of current process

        comm: Comm
            The Comm object for Manager/Worker communications.

        resources: Resources
            A Resources object containing global nodelist and intra-node information.

        """
        self.num_workers = comm.get_num_workers()
        self.workerID = workerID
        self.local_nodelist = WorkerResources.get_local_nodelist(self.num_workers, self.workerID, resources)
        self.local_node_count = len(self.local_nodelist)
        self.workers_per_node = WorkerResources.get_workers_on_a_node(self.num_workers, resources)

    @staticmethod
    def get_workers_on_a_node(num_workers, resources):
        """ Returns the number of workers that can be placed on each node"""
        num_nodes = len(resources.global_nodelist)
        # Round up if theres a remainder
        workers_per_node = num_workers//num_nodes + (num_workers % num_nodes > 0)
        return workers_per_node

    @staticmethod
    def get_local_nodelist(num_workers, workerID, resources):
        """Returns the list of nodes available to the current worker

        Assumes that self.global_nodelist has been calculated (in __init__).
        Also self.global_nodelist will have already removed non-application nodes
        """

        global_nodelist = resources.global_nodelist
        num_nodes = len(global_nodelist)

        # Check if current host in nodelist - if it is then in distributed mode.
        local_host = socket.gethostname()
        distrib_mode = local_host in global_nodelist

        # If multiple workers per node - create global node_list with N duplicates (for N workers per node)
        sub_node_workers = (num_workers >= num_nodes)
        if sub_node_workers:
            workers_per_node = num_workers//num_nodes
            global_nodelist = list(itertools.chain.from_iterable(itertools.repeat(x, workers_per_node) for x in global_nodelist))

        # Currently require even split for distrib mode - to match machinefile - throw away remainder
        if distrib_mode and not sub_node_workers:
            # Could just read in the libe machinefile and use that - but this should match
            # Alt. create machinefile/host-list with same algorithm as best_split - future soln.
            nodes_per_worker, remainder = divmod(num_nodes, num_workers)
            if remainder != 0:
                # Worker node may not be at head of list after truncation - should perhaps be warning or enforced
                logger.warning("Nodes to workers not evenly distributed. Wasted nodes. {} workers and {} nodes".format(num_workers, num_nodes))
                num_nodes = num_nodes - remainder
                global_nodelist = global_nodelist[0:num_nodes]

        # Divide global list between workers
        split_list = list(Resources.best_split(global_nodelist, num_workers))
        # logger.debug("split_list is {}".format(split_list))

        if workerID is None:
            raise ResourcesException("Worker has no workerID - aborting")
        local_nodelist = split_list[workerID - 1]

        logger.debug("local_nodelist is {}".format(local_nodelist))
        return local_nodelist
