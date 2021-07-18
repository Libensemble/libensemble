"""
This module detects and returns system resources

"""

# SH TODO: Do we need custom_info options in libE_specs for resources, and another as argument
#          to MPIExecutor for MPIExecutor configuration?
#          E.g. mpi_runner_type, runner_name, subgroup_launch
#          Alternative - do Executor same way as this (init in libE), and then can uses combined custom_info again!
#          Also will need to update docs/tests with new custom_info options.
#          Remove debugging comments/commented out code + check/update docstrings
#          Deal with unbalanced cases, and if not, return meaningful error message

import os
import socket
import logging
import subprocess
from collections import Counter
from collections import OrderedDict
import numpy as np
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

    resources = None

    DEFAULT_NODEFILE = 'node_list'

    @classmethod
    def init_resources(cls, libE_specs):
        """Initiate resource management"""
        from libensemble.resources.mpi_resources import MPIResources

        # If auto_resources is False, then Resources.resources will remain None.
        auto_resources = libE_specs.get('auto_resources', True)

        if auto_resources:
            custom_info = libE_specs.get('custom_info', {})
            cores_on_node = custom_info.get('cores_on_node', None)
            node_file = custom_info.get('node_file', None)
            # SH TODO: Should these be in custom_info
            nodelist_env_slurm = custom_info.get('nodelist_env_slurm', None)
            nodelist_env_cobalt = custom_info.get('nodelist_env_cobalt', None)
            nodelist_env_lsf = custom_info.get('nodelist_env_lsf', None)
            nodelist_env_lsf_shortform = custom_info.get('nodelist_env_lsf_shortform', None)

            central_mode = libE_specs.get('central_mode', False)
            allow_oversubscribe = libE_specs.get('allow_oversubscribe', True)  # SH TODO: re-name to clarify on-node?
            zero_resource_workers = libE_specs.get('zero_resource_workers', [])
            num_resource_sets = libE_specs.get('num_resource_sets', None)
            top_level_dir = os.getcwd()  # SH TODO: Do we want libE_specs option - in case want to run somewhere else.

            # SH TODO: MPIResources always - should be some option - related to Executor.
            #          Though everything in init is not MPI specific (could also be a TCP resources version)
            #          Remember, in this initialization, resources is stored in class attribute.
            Resources.resources = \
                MPIResources(top_level_dir=top_level_dir,
                             central_mode=central_mode,
                             zero_resource_workers=zero_resource_workers,
                             num_resource_sets=num_resource_sets,
                             allow_oversubscribe=allow_oversubscribe,
                             # launcher=self.mpi_runner.run_command  # SH TODO: re-check replacement code
                             cores_on_node=cores_on_node,
                             node_file=node_file,
                             nodelist_env_slurm=nodelist_env_slurm,
                             nodelist_env_cobalt=nodelist_env_cobalt,
                             nodelist_env_lsf=nodelist_env_lsf,
                             nodelist_env_lsf_shortform=nodelist_env_lsf_shortform)

    def __init__(self, top_level_dir=None,
                 central_mode=False,
                 zero_resource_workers=[],
                 num_resource_sets=None,
                 allow_oversubscribe=False,
                 #launcher=None,
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

        num_resource_sets: int, optional
            The total number of resource sets. Resources will be divided into this number.
            Default: None. If None, resources will be divided by workers (excluding zero_resource_workers).

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

        #self.launcher = launcher
        self.launcher = Resources.get_MPI_runner()
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
        self.num_resource_sets = num_resource_sets

        # Let caller decide whether to set Resources.resources
        # Resources.resources = self

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

    def set_managerworker_resources(self, num_workers):
        self.managerworker_resources = ManagerWorkerResources(num_workers, self)

    @staticmethod
    def get_MPI_runner():
        var = Resources.get_MPI_variant()
        if var in ['mpich', 'openmpi']:
            return 'mpirun'
        else:
            return var

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


# SH TODO Resources to have a restructure
#         May include a base class shared by man/worker variants for fixed mapping (e.g. split_list)
#         Shorten names... e.g. WorkerResources and LocalResources?
#         Add class docstring
#         Need to deal with breakdown of rsets per node (and/or workers per node) for uneven distributions.
class ManagerWorkerResources:

    # SH Consider data structures.
    # dtype for rsets array (each resource set has these attributes).
    rset_dtype = [('assigned', int),  # Holds worker ID assigned to or zero
                  ('group', int)      # Group ID this resource set belongs to
                  # ('pool', int),    # Pool ID (eg. separate gen/sim resources) - not yet used.
                  ]

    def __init__(self, num_workers, resources):
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
        self.num_workers = num_workers
        self.num_workers_2assign2 = WorkerResources.get_workers2assign2(self.num_workers, resources)
        self.num_rsets = resources.num_resource_sets or self.num_workers_2assign2
        self.split_list, self.local_rsets_list = WorkerResources.get_partitioned_nodelist(self.num_rsets, resources)

        # SH May change name from index_list - something like "default_mapping"
        # SH Should default mapping use 1 resource set each or divide up?
        self.index_list = WorkerResources.get_index_list(self.num_workers, resources.zero_resource_workers)
        print('index list:', self.index_list)  # SH TODO: Remove when done testing

        # SH Assumes all groups same size - for uneven distribution, best_split (or equiv.) should determine.

        #SH TODO: Need to update to make uneven distribution of rsets to nodes work as does on develop
        self.rsets_per_node = WorkerResources.get_rsets_on_a_node(self.num_rsets, resources)
        self.rsets = np.zeros(self.num_rsets, dtype=ManagerWorkerResources.rset_dtype)
        self.rsets['assigned'] = 0
        self.rsets['group'] = WorkerResources.get_group_list(self.split_list)
        self.num_groups = self.rsets['group'][-1]
        unique, counts = np.unique(self.rsets['group'], return_counts=True)
        self.rsets_free = self.num_rsets

        # SH TODO: Useful for scheduling tasks with different sized groups (resource sets per node).
        self.group_sizes = dict(zip(unique, counts))
        self.ngroups_by_size = Counter(counts)
        self.even_groups = True if len(self.ngroups_by_size) == 1 else False
        print('\nrsets are {} even groups is {}\n'.format(self.rsets,self.even_groups))
        #import pdb;pdb.set_trace()


    def assign_rsets(self, rset_team, worker_id):
        """Mark the resource sets given by rset_team as assigned to worker_id"""

        if rset_team:
            rteam = self.rsets['assigned'][rset_team]
            for i, wid in enumerate(rteam):
                if wid == 0:
                    self.rsets['assigned'][rset_team[i]] = worker_id
                    self.rsets_free -= 1
                elif wid != worker_id:
                    # Raise error if rsets are already assigned to a different worker.
                    raise ResourcesException("Error: Attempting to assign rsets {} already assigned to workers: {}".
                                             format(rset_team, rteam))

            #self.rsets['assigned'][rset_team] = worker_id
            #self.rsets_free -= len(rset_team)

            #print('rsets free', self.rsets_free)
            #print('resource ids assigned', np.where(self.rsets['assigned'])[0])  # SH TODO: Remove
            #print('resource worker assignment', self.rsets['assigned'])  # SH TODO: Remove
            # print('resources unassigned', np.where(self.rsets['assigned'] == 0)[0])  # SH TODO: Remove


    def free_rsets(self, worker=None):
        """Free up assigned resource sets"""
        if worker is None:
            self.rsets['assigned'] = 0
            #self.num_assigned = 0
            self.rsets_free = self.num_rsets
        else:
            rsets_to_free = np.where(self.rsets['assigned'] == worker)[0]
            self.rsets['assigned'][rsets_to_free] = 0
            self.rsets_free += len(rsets_to_free)
            #print('\nWorker {} returned - freed up rsets {}'.format(worker, rsets_to_free))

        # print('resources assigned', np.where(self.rsets['assigned'])[0])  # SH TODO: Remove
        # print('resources unassigned', np.where(self.rsets['assigned'] == 0)[0])  # SH TODO: Remove

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
        self.rset_team = None
        self.slots = None
        self.even_slots = None
        self.slot_count = None
        self.slots_on_node = None

        self.num_workers_2assign2 = WorkerResources.get_workers2assign2(self.num_workers, resources)

        #SH TODO: Maybe call total_num_rsets or global_num_rsets - as its not rsets for this worker.
        self.num_rsets = resources.num_resource_sets or self.num_workers_2assign2

        # SH TODO: In next resources restructure - should not need duplicate of zero_resource_workers (so can use
        #          in set_rset_team without passing resources again.
        #          Consider naming of workers_per_node - given that is is the fixed mapping (and having excluded zrw)
        #          - possibly fixed_workers_per_node or maybe resource_sets or rsets_per_node...
        self.zero_resource_workers = resources.zero_resource_workers
        self.split_list, self.local_rsets_list = WorkerResources.get_partitioned_nodelist(self.num_rsets, resources)

        # SH TODO: To fully support uneven rsets (with re-assignment) - will use local_rsets_list (above)
        self.rsets_per_node = WorkerResources.get_rsets_on_a_node(self.num_rsets, resources)
        self.local_nodelist = []
        self.local_node_count = len(self.local_nodelist)
        self.set_slot_count()

    def set_rset_team(self, rset_team):
        """Update worker team and local attributes

        Updates: rset_team
                 local_nodelist
                 slots (dictionary with list of partitions for each node)
                 slot_count - number of slots on each node # SH TODO: Make a list if uneven?
                 local_node_count
        """
        if self.workerID in self.zero_resource_workers:
            return

        if rset_team != self.rset_team:  # Order matters
            self.rset_team = rset_team
            self.local_nodelist, self.slots = \
                WorkerResources.get_local_nodelist(self.workerID, self.rset_team,
                                                   self.split_list, self.rsets_per_node)
            self.set_slot_count()
            self.local_node_count = len(self.local_nodelist)

    # SH TODO: Same count, but I want same list...
    #          This needs checking... what is slot_count/slots_on_node when uneven
    #          May be more efficient to do when create slot list.
    def set_slot_count(self):
        if self.slots:
            # Check if even distribution
            # lens = set(map(len, self.slots.values()))
            # lens = set(map(len, self.slots.values()))
            # self.even_slots = True if len(lens) == 1 else False

            # Check if same slots on each node (not just lengths)
            # SH TODO: Maybe even_slots v equal_slots?
            first_node_slots = list(self.slots.values())[0]
            all_match = True
            for node_list in self.slots.values():
                if node_list != first_node_slots:
                    all_match = False
                    break

            # SH TODO: Need to update for uneven scenarios
            self.even_slots = True if all_match else False
            if self.even_slots:
                self.slots_on_node = first_node_slots
                self.slot_count = len(self.slots_on_node)
            else:
                self.slots_on_node = None  # SH TODO: What should this be
                self.slot_count = None  # SH TODO: Could be list of lengths

    @staticmethod
    def get_rsets_on_a_node(num_rsets, resources):
        """Returns the number of resource sets that can be placed on each node

        If there are more nodes than resource sets, returns 1.
        """
        num_nodes = len(resources.global_nodelist)
        # Round up if theres a remainder
        rsets_per_node = num_rsets//num_nodes + (num_rsets % num_nodes > 0)
        return rsets_per_node

    @staticmethod
    def get_group_list(split_list):
        group = 1
        group_list = []
        node = split_list[0]

        # SH What to do when multiple nodes in each entry........ what is group then.
        for i in range(len(split_list)):
            if split_list[i] == node:
                group_list.append(group)
            else:
                node = split_list[i]
                group += 1
                group_list.append(group)
        return group_list

    # Will work out rset siblings - not just yet - or mayb can use get_rsets_by_group
    #@staticmethod
    #def get_num_siblings_by_rset(split_list):
        #group = 1
        #num_siblings_by_rset = []
        #node = split_list[0]

        #for i in range(len(split_list)):
            #if split_list[i] == node:
                #group_list.append(group)
            #else:
                #node = split_list[i]
                #group += 1
                #group_list.append(group)
        #return group_list

    @staticmethod
    def get_rsets_by_group(split_list):
        """Returns a dictionary where key is groupID and values
        are a list of resource set IDs (indices into rsets array).
        """
        rsets_by_group = {}
        group = 0
        loc_nodes = None
        for i in range(len(split_list)):
            if split_list[i] == loc_nodes:
                rsets_by_group[group].append(i)
            else:
                loc_nodes = split_list[i]
                group += 1
                rsets_by_group[group] = [i]
        return rsets_by_group

    @staticmethod
    def get_index_list(num_workers, zero_resource_list):
        """Map WorkerID to index into a nodelist"""
        index = 0
        index_list = []
        for i in range(1, num_workers+1):
            if i in zero_resource_list:
                index_list.append(None)
            else:
                index_list.append(index)
                index += 1
        return index_list

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
    def get_split_list(num_rsets, resources):
        """Returns a list of lists for each worker

        Assumes that self.global_nodelist has been calculated (in __init__).
        """
        global_nodelist = resources.global_nodelist
        num_nodes = len(global_nodelist)

        if not WorkerResources.even_assignment(num_nodes, num_rsets):
            logger.warning('Resource sets ({}) are not distributed evenly to available nodes ({})'
                           .format(num_rsets, num_nodes))

        # If multiple workers per node - create global node_list with N duplicates (for N workers per node)
        sub_node_workers = (num_rsets >= num_nodes)
        if sub_node_workers:
            global_nodelist, local_rsets_list = \
                WorkerResources.expand_list(num_nodes, num_rsets, global_nodelist)
        else:
            local_rsets_list = [1] * num_rsets

        # Divide global list between workers
        split_list = list(Resources.best_split(global_nodelist, num_rsets))
        logger.debug("split_list is {}".format(split_list))
        return split_list, local_rsets_list

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
        local_rsets_list = []
        for i, x in enumerate(nodelist):
            repeats = k + 1 if i < m else k
            for j in range(repeats):
                dup_list.append(x)
                local_rsets_list.append(repeats)
        return dup_list, local_rsets_list

    # SH TODO This has become redundant wrapper - if stays that way can remove
    @staticmethod
    def get_partitioned_nodelist(num_rsets, resources):
        """Returns lists of nodes available to all resource sets

        Assumes that self.global_nodelist has been calculated (in __init__).
        Also self.global_nodelist will have already removed non-application nodes
        """
        split_list, local_rsets_list = WorkerResources.get_split_list(num_rsets, resources)
        #print('local_rsets_list', local_rsets_list, flush=True)
        return split_list, local_rsets_list

    @staticmethod
    def get_local_nodelist(workerID, rset_team, split_list, rsets_per_node):
        """Returns the list of nodes available to the current worker"""

        # SH May update to do in two stages - get nodelist then get slots
        # but still have to merge with uneven layout handling (as on develop).

        # SH TODO: Update docstring - or split function (also returns slots dictionary)
        #          Remove print comments when done testing
        if workerID is None:
            raise ResourcesException("Worker has no workerID - aborting")

        # print('Worker {}. rsets_per_node {}'.format(workerID, rsets_per_node), flush=True)  # SH TODO: Remove
        team_list = []
        for index in rset_team:
            team_list += split_list[index]

        # print('Worker {} team_list {}'.format(workerID, team_list),flush=True)  # SH TODO: Remove

        local_nodelist = list(OrderedDict.fromkeys(team_list))  # Maintain order of nodes
        # print("Worker {} Worker's local_nodelist is {}".format(workerID, local_nodelist),flush=True) # SH TODO:Remove
        logger.debug("Worker's local_nodelist is {}".format(local_nodelist))

        # SH TODO: Maybe can essentailly do this at mapping stage with group list or create a structure to reference.
        slots = {}
        for node in local_nodelist:
            slots[node] = []

        for index in rset_team:
            mynodes = split_list[index]
            # If rset has > 1 node, then all nodes just have a single slot (slot 0).
            if len(mynodes) > 1:
                for node in  mynodes:
                    slots[node].append(0)
            else:
                mynode = split_list[index][0]
                # rsets_per_node = local_rsets_list[index]  # SH TODO Support uneven rsets per node
                pos_in_node = index % rsets_per_node  # SH TODO: check/test this
                slots[mynode].append(pos_in_node)
                # SH TODO: Can potentially create a machinefile from slots if/when support uneven lists

        # print("Worker {} slots are {}".format(workerID, slots),flush=True) # SH TODO:Remove

        return local_nodelist, slots
