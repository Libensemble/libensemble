"""
This module detects and returns system resources

"""

import os
import socket
import logging
import subprocess
from libensemble.resources import node_resources
from libensemble.resources.env_resources import EnvResources

logger = logging.getLogger(__name__)
# To change logging level for just this module
# logger.setLevel(logging.DEBUG)


class ResourcesException(Exception):
    "Resources module exception."


class Resources:
    """Provides system resources to libEnsemble and executor.

    This is intialized when the executor is created with auto_resources set to true.

    **Object Attributes:**

    These are set on initialization.

    :ivar string top_level_dir: Directory where searches for node_list file
    :ivar boolean central_mode: If true, then running in central mode; otherwise distributed
    :ivar EnvResources env_resources: An object storing environment variables used by resources
    :ivar list global_nodelist: A list of all nodes available for running user applications
    :ivar int logical_cores_avail_per_node: Logical cores (including SMT threads) available on a node
    :ivar int physical_cores_avail_per_node: Physical cores available on a node
    :ivar WorkerResources worker_resources: An object that can contain worker specific resources
    """

    DEFAULT_NODEFILE = 'node_list'

    def __init__(self, top_level_dir=None,
                 central_mode=False,
                 zero_resource_workers=[],
                 allow_oversubscribe=False,
                 launcher=None,
                 cores_on_node=None,
                 node_file=None,
                 nodelist_env_slurm=None,
                 nodelist_env_cobalt=None,
                 nodelist_env_lsf=None,
                 nodelist_env_lsf_shortform=None):

        """Initializes a new Resources instance

        Determines the compute resources available for current allocation, including
        node list and cores/hardware threads available within nodes.

        Parameters
        ----------

        top_level_dir: string, optional
            Directory libEnsemble runs in (default is current working directory)

        central_mode: boolean, optional
            If true, then running in central mode, otherwise distributed.
            Central mode means libE processes (manager and workers) are grouped together and
            do not share nodes with applications. Distributed mode means Workers share nodes
            with applications.

        zero_resource_workers: list of ints, optional
            List of workers that require no resources.

        allow_oversubscribe: boolean, optional
            If false, then resources will raise an error if task process
            counts exceed the CPUs available to the worker, as detected by
            auto_resources. Larger node counts will always raise an error.
            When auto_resources is off, this argument is ignored.

        launcher: String, optional
            The name of the job launcher, such as mpirun or aprun. This may be used to obtain
            intranode information by launching a probing job onto the compute nodes.
            If not present, the local node will be used to obtain this information.

        cores_on_node: tuple (int,int), optional
            If supplied gives (physical cores, logical cores) for the nodes. If not supplied,
            this will be auto-detected.

        node_file: String, optional
            If supplied, give the name of a file in the run directory to use as a node-list
            for use by libEnsemble. Defaults to a file named 'node_list'. If the file does
            not exist, then the node-list will be auto-detected.

        nodelist_env_slurm: String, optional
            The environment variable giving a node list in Slurm format (Default: uses SLURM_NODELIST).
            Note: This is queried only if a node_list file is not provided and auto_resources=True.

        nodelist_env_cobalt: String, optional
            The environment variable giving a node list in Cobalt format (Default: uses COBALT_PARTNAME).
            Note: This is queried only if a node_list file is not provided and auto_resources=True.

        nodelist_env_lsf: String, optional
            The environment variable giving a node list in LSF format (Default: uses LSB_HOSTS).
            Note: This is queried only if a node_list file is not provided and auto_resources=True.

        nodelist_env_lsf_shortform: String, optional
            The environment variable giving a node list in LSF short-form format (Default: uses LSB_MCPU_HOSTS)
            Note: This is only queried if a node_list file is not provided and auto_resources=True.

        """

        self.top_level_dir = top_level_dir or os.getcwd()
        self.central_mode = central_mode
        if self.central_mode:
            logger.debug('Running in central mode')
        self.allow_oversubscribe = allow_oversubscribe

        self.env_resources = EnvResources(nodelist_env_slurm=nodelist_env_slurm,
                                          nodelist_env_cobalt=nodelist_env_cobalt,
                                          nodelist_env_lsf=nodelist_env_lsf,
                                          nodelist_env_lsf_shortform=nodelist_env_lsf_shortform)

        if node_file is None:
            node_file = Resources.DEFAULT_NODEFILE

        self.global_nodelist = Resources.get_global_nodelist(node_file=node_file,
                                                             rundir=self.top_level_dir,
                                                             env_resources=self.env_resources)

        self.shortnames = Resources.is_nodelist_shortnames(self.global_nodelist)
        if self.shortnames:
            self.local_host = self.env_resources.shortnames([socket.gethostname()])[0]
        else:
            self.local_host = socket.gethostname()

        self.launcher = launcher
        remote_detect = False
        if self.local_host not in self.global_nodelist:
            remote_detect = True

        if not cores_on_node:
            cores_on_node = \
                node_resources.get_sub_node_resources(launcher=self.launcher,
                                                      remote_mode=remote_detect,
                                                      env_resources=self.env_resources)
        self.physical_cores_avail_per_node = cores_on_node[0]
        self.logical_cores_avail_per_node = cores_on_node[1]
        self.libE_nodes = None
        self.worker_resources = None
        self.zero_resource_workers = zero_resource_workers

    def add_comm_info(self, libE_nodes):
        """Adds comms-specific information to resources

        Removes libEnsemble nodes from nodelist if in central_mode.
        """
        if self.shortnames:
            self.libE_nodes = self.env_resources.shortnames(libE_nodes)
        else:
            self.libE_nodes = libE_nodes
        libE_nodes_in_list = list(filter(lambda x: x in self.libE_nodes, self.global_nodelist))
        if libE_nodes_in_list:
            if self.central_mode and len(self.global_nodelist) > 1:
                self.global_nodelist = Resources.remove_nodes(self.global_nodelist, self.libE_nodes)
                if not self.global_nodelist:
                    logger.warning("Warning. Node-list for tasks is empty. Remove central_mode or add nodes")

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

        try:
            # Explore mpi4py.MPI.get_vendor() and mpi4py.MPI.Get_library_version() for mpi4py
            try_mpich = subprocess.Popen(['mpirun', '-npernode'], stdout=subprocess.PIPE,
                                         stderr=subprocess.STDOUT)
            stdout, _ = try_mpich.communicate()
            if 'unrecognized argument npernode' in stdout.decode():
                return 'mpich'
            return 'openmpi'
        except Exception:
            pass

        try:
            subprocess.check_call(['srun', '--version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return 'srun'
        except OSError:
            pass

    # ---------------------------------------------------------------------------

    @staticmethod
    def is_nodelist_shortnames(nodelist):
        """Returns True if any entry contains a '.', else False"""
        for item in nodelist:
            if '.' in item:
                return False
        return True

    # This is for central mode where libE nodes will not share with app nodes
    @staticmethod
    def remove_nodes(global_nodelist_in, remove_list):
        """Removes any nodes in remove_list from the global nodelist"""
        global_nodelist = list(filter(lambda x: x not in remove_list, global_nodelist_in))
        return global_nodelist

    @staticmethod
    def best_split(a, n):
        """Creates the most even split of list a into n parts and return list of lists"""
        k, m = divmod(len(a), n)
        return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

    @staticmethod
    def get_global_nodelist(node_file=DEFAULT_NODEFILE,
                            rundir=None,
                            env_resources=None):
        """
        Returns the list of nodes available to all libEnsemble workers.

        If a node_file exists this is used, otherwise the environment
        is interrogated for a node list. If a dedicated manager node is used,
        then a node_file is recommended.

        In central mode, any node with a libE worker is removed from the list.
        """
        top_level_dir = rundir or os.getcwd()
        node_filepath = os.path.join(top_level_dir, node_file)
        global_nodelist = []
        if os.path.isfile(node_filepath):
            logger.debug("node_file found - getting nodelist from node_file")
            with open(node_filepath, 'r') as f:
                for line in f:
                    global_nodelist.append(line.rstrip())
            # Expect correct format - if anything - could have an option to truncate.
            # if env_resources:
                # global_nodelist = env_resources.shortnames(global_nodelist)
        else:
            logger.debug("No node_file found - searching for nodelist in environment")
            if env_resources:
                global_nodelist = env_resources.get_nodelist()

            if not global_nodelist:
                # Assume a standalone machine
                logger.info("Can not find nodelist from environment. Assuming standalone")
                # global_nodelist.append(env_resources.shortnames([socket.gethostname()])[0])
                global_nodelist.append(socket.gethostname())

        if global_nodelist:
            return global_nodelist
        raise ResourcesException("Error. global_nodelist is empty")


class WorkerResources:
    """Provide system resources per worker to libEnsemble and executor.

    **Object Attributes:**

    These are set on initialisation.

    :ivar int num_workers: Total number of workers
    :ivar int workerID: workerID
    :ivar list local_nodelist: A list of all nodes assigned to this worker
    :ivar int local_node_count: The number of nodes available to this worker (rounded up to whole number)
    :ivar int workers_per_node: The number of workers per node (if using subnode workers)
    """

    def __init__(self, workerID, comm, resources):
        """Initializes a new WorkerResources instance

        Determines the compute resources available for current worker, including
        node list and cores/hardware threads available within nodes.

        Parameters
        ----------

        workerID: int
            workerID of current process

        comm: Comm
            The Comm object for manager/worker communications

        resources: Resources
            A Resources object containing global nodelist and intranode information

        """
        self.num_workers = comm.get_num_workers()
        self.workerID = workerID
        self.local_nodelist, self.workers_on_node = \
            WorkerResources.get_local_nodelist(self.num_workers, self.workerID, resources)
        self.local_node_count = len(self.local_nodelist)
        self.num_workers_2assign2 = WorkerResources.get_workers2assign2(self.num_workers, resources)

    @staticmethod
    def map_workerid_to_index(num_workers, workerID, zero_resource_list):
        """Map WorkerID to index into a nodelist"""
        index = workerID - 1
        if zero_resource_list:
            for i in range(1, num_workers+1):
                if i in zero_resource_list:
                    index -= 1
                if index < i:
                    return index
            raise ResourcesException("Error mapping workerID {} to nodelist index {}".format(workerID, index))
        return index

    @staticmethod
    def get_workers2assign2(num_workers, resources):
        """Returns workers to assign resources to"""
        zero_resource_list = resources.zero_resource_workers
        return num_workers - len(zero_resource_list)

    @staticmethod
    def even_assignment(nnodes, nworkers):
        """Returns True if workers are evenly distributied to nodes, else False"""
        return nnodes % nworkers == 0 or nworkers % nnodes == 0

    @staticmethod
    def expand_list(nnodes, nworkers, nodelist):
        """Duplicates each element of ``nodelist`` to best map workers to nodes.

        Returns node list with duplicates, and a list of local (on-node) worker
        counts, both indexed by worker.
        """
        k, m = divmod(nworkers, nnodes)
        dup_list = []
        local_workers_list = []
        for i, x in enumerate(nodelist):
            repeats = k + 1 if i < m else k
            for j in range(repeats):
                dup_list.append(x)
                local_workers_list.append(repeats)
        return dup_list, local_workers_list

    @staticmethod
    def get_local_nodelist(num_workers, workerID, resources):
        """Returns the list of nodes available to the current worker

        Assumes that self.global_nodelist has been calculated (in __init__).
        Also self.global_nodelist will have already removed non-application nodes
        """
        global_nodelist = resources.global_nodelist
        num_nodes = len(global_nodelist)
        zero_resource_list = resources.zero_resource_workers
        num_workers_2assign2 = WorkerResources.get_workers2assign2(num_workers, resources)

        if not WorkerResources.even_assignment(num_nodes, num_workers_2assign2):
            logger.warning('Workers with assigned resources ({}) are not distributed evenly to available nodes ({})'
                           .format(num_workers_2assign2, num_nodes))

        # If multiple workers per node - create global node_list with N duplicates (for N workers per node)
        sub_node_workers = (num_workers_2assign2 >= num_nodes)
        if sub_node_workers:
            global_nodelist, local_workers_list = \
                WorkerResources.expand_list(num_nodes, num_workers_2assign2, global_nodelist)
        else:
            local_workers_list = [1] * num_workers_2assign2

        # Divide global list between workers
        split_list = list(Resources.best_split(global_nodelist, num_workers_2assign2))
        logger.debug("split_list is {}".format(split_list))

        if workerID is None:
            raise ResourcesException("Worker has no workerID - aborting")

        if workerID in zero_resource_list:
            local_nodelist = []
            workers_on_node = 0
            logger.debug("Worker is a zero-resource worker")
        else:
            index = WorkerResources.map_workerid_to_index(num_workers, workerID, zero_resource_list)
            local_nodelist = split_list[index]
            workers_on_node = local_workers_list[index]
            logger.debug("Worker's local_nodelist is {}".format(local_nodelist))

        return local_nodelist, workers_on_node
