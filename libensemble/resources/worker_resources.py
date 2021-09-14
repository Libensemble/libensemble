import os
import logging
from collections import Counter
from collections import OrderedDict
import numpy as np
from libensemble.resources.rset_resources import RSetResources

logger = logging.getLogger(__name__)
# To change logging level for just this module
# logger.setLevel(logging.DEBUG)


class WorkerResourcesException(Exception):
    """Worker resources exception."""


class ResourceManagerException(Exception):
    """Resource Manager exception."""


class ResourceManager(RSetResources):

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
        super().__init__(num_workers, resources)
        self.index_list = ResourceManager.get_index_list(self.num_workers, resources.zero_resource_workers)
        # print('index list:', self.index_list)  # SH TODO: Remove when done testing

        # SH TODO: Need to update to allow uneven distribution of rsets to nodes
        self.rsets = np.zeros(self.total_num_rsets, dtype=ResourceManager.rset_dtype)
        self.rsets['assigned'] = 0
        self.rsets['group'] = ResourceManager.get_group_list(self.split_list)
        self.num_groups = self.rsets['group'][-1]
        self.rsets_free = self.total_num_rsets

        # SH TODO: Useful for scheduling tasks with different sized groups (resource sets per node).
        unique, counts = np.unique(self.rsets['group'], return_counts=True)
        self.group_sizes = dict(zip(unique, counts))
        self.ngroups_by_size = Counter(counts)
        self.even_groups = True if len(self.ngroups_by_size) == 1 else False
        # print('\nrsets are {} even groups is {}\n'.format(self.rsets,self.even_groups))  # SH TODO: Remove

    def assign_rsets(self, rset_team, worker_id):
        """Mark the resource sets given by rset_team as assigned to worker_id"""

        if rset_team:
            rteam = self.rsets['assigned'][rset_team]
            for i, wid in enumerate(rteam):
                if wid == 0:
                    self.rsets['assigned'][rset_team[i]] = worker_id
                    self.rsets_free -= 1
                elif wid != worker_id:
                    ResourceManagerException("Error: Attempting to assign rsets {}"
                                             " already assigned to workers: {}".
                                             format(rset_team, rteam))
            # print('resource ids assigned', np.where(self.rsets['assigned'])[0])  # SH TODO: Remove
            # print('resource worker assignment', self.rsets['assigned'])  # SH TODO: Remove
            # print('resources unassigned', np.where(self.rsets['assigned'] == 0)[0])  # SH TODO: Remove

    def free_rsets(self, worker=None):
        """Free up assigned resource sets"""
        if worker is None:
            self.rsets['assigned'] = 0
            self.rsets_free = self.total_num_rsets
        else:
            rsets_to_free = np.where(self.rsets['assigned'] == worker)[0]
            self.rsets['assigned'][rsets_to_free] = 0
            self.rsets_free += len(rsets_to_free)
            # print('\nWorker {} returned - freed up rsets {}'.format(worker, rsets_to_free))  # SH TODO: Remove
        # print('resources assigned', np.where(self.rsets['assigned'])[0])  # SH TODO: Remove
        # print('resources unassigned', np.where(self.rsets['assigned'] == 0)[0])  # SH TODO: Remove

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


class WorkerResources(RSetResources):
    """Provide system resources per worker to libEnsemble and executor.

    **Object Attributes:**

    These attributes may be updated as the ensemble progresses.

    ``rsets`` below is used to abbreviate ``resource sets``.

    :ivar int workerID: workerID for this worker.
    :ivar list local_nodelist: A list of all nodes assigned to this worker.
    :ivar list rset_team: List of rset IDs currently assigned to this worker.
    :ivar int num_rsets: The number of resource sets assigned to this worker.
    :ivar dict slots: A dictionary with a list of slot IDs for each node.
    :ivar bool even_slots: Determines if the slots evenly divide amongst nodes.
    :ivar int slot_count: The number of slots per node if even_slots is True, else None.
    :ivar list slots_on_node: A list of slots IDs if even_slots is True, else None.
    :ivar int local_node_count: The number of nodes available to this worker (rounded up to whole number).
    :ivar int rsets_per_node: The number of rsets per node (if a rset > 1 node, will be 1).

    The worker_resources attribtues can be queried, and convenience functions
    called, via the resources class attribute. For example:

    With resources imported:

    .. code-block:: python

        from libensemble.resources.resources import Resources

    A user function (sim/gen) may do:

    .. code-block:: python

        resources = Resources.resources.worker_resources
        num_nodes = resources.local_node_count
        cores_per_node = resources.slot_count  # One CPU per GPU
        resources.set_env_to_slots("CUDA_VISIBLE_DEVICES")  # Use convenience function.

    Note that **slots** are resource sets enumerated on a node (starting with zero).
    If a resource set has more than one node, then each node is considered to have slot zero.

    If ``even_slots`` is True, then the attributes ``slot_count`` and ``slots_on_node``
    can be used for simplicity, Otherwise, the ``slots`` dictionary can be used to get
    information for each node.

    """

    def __init__(self, num_workers, resources, workerID):

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
        super().__init__(num_workers, resources)
        self.workerID = workerID
        self.local_nodelist = []
        self.rset_team = None
        self.num_rsets = 0
        self.slots = None
        self.even_slots = None
        self.slot_count = None
        self.slots_on_node = None
        self.zero_resource_workers = resources.zero_resource_workers
        self.local_node_count = len(self.local_nodelist)
        self.set_slot_count()

    # User convenience functions ----------------------------------------------

    def get_slots_as_string(self, multiplier=1, delimiter=','):
        """Returns list of slots as a string

        :param multiplier: Optional int. Assume this many items per slot.
        :param delimiter: Optional int. Delimiter for output string.
        """

        if self.slots_on_node is None:
            logger.warning("Slots on node is None when requested as a string")
            return None

        n = multiplier
        slot_list = [j for i in self.slots_on_node for j in range(i*n, (i+1)*n)]
        slots = delimiter.join(map(str, slot_list))
        return slots

    def set_env_to_slots(self, env_var, multiplier=1, delimiter=','):
        """Sets the given environment variable to slots

        :param env_var: String. Name of environment variable to set.
        :param multiplier: Optional int. Assume this many items per slot.
        :param delimiter: Optional int. Delimiter for output string.

        Example  in a sim function
        --------------------------

        With resources imported:

        .. code-block:: python

            from libensemble.resources.resources import Resources

        Obtain worker resoruces:

        .. code-block:: python

            resources = Resources.resources.worker_resources
            resources.set_env_to_slots("CUDA_VISIBLE_DEVICES")

        """

        os.environ[env_var] = self.get_slots_as_string(multiplier, delimiter)

    # libEnsemble functions ---------------------------------------------------

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
            self.num_rsets = len(rset_team)
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
    def get_local_nodelist(workerID, rset_team, split_list, rsets_per_node):
        """Returns the list of nodes available to the current worker"""

        # SH May update to do in two stages - get nodelist then get slots
        # but still have to merge with uneven layout handling (as on develop).

        # SH TODO: Update docstring - or split function (also returns slots dictionary)
        #          Remove print comments when done testing
        if workerID is None:
            raise WorkerResourcesException("Worker has no workerID - aborting")

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
                for node in mynodes:
                    slots[node].append(0)
            else:
                mynode = split_list[index][0]
                # rsets_per_node = local_rsets_list[index]  # SH TODO Support uneven rsets per node
                pos_in_node = index % rsets_per_node  # SH TODO: check/test this
                slots[mynode].append(pos_in_node)
                # SH TODO: Can potentially create a machinefile from slots if/when support uneven lists

        # print("Worker {} slots are {}".format(workerID, slots),flush=True) # SH TODO:Remove

        return local_nodelist, slots
