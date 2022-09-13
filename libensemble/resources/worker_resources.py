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
    """Provides methods for managing the assignment of resource sets to workers."""

    rset_dtype = [
        ("assigned", int),  # Holds worker ID assigned to or zero
        ("group", int),  # Group ID this resource set belongs to
        ("slot", int)  # Slot ID this resource set belongs to
        # ('pool', int),    # Pool ID (eg. separate gen/sim resources) - not yet used.
    ]

    def __init__(self, num_workers, resources):
        """Initializes a new ResourceManager instance

        Instantiates the numpy structured array that holds information for each
        resource set.

        Parameters
        ----------

        num_workers: int
            The number of workers

        resources: Resources
            A Resources object containing global nodelist and intranode information

        """
        super().__init__(num_workers, resources)
        self.index_list = ResourceManager.get_index_list(
            self.num_workers,
            self.total_num_rsets,
            resources.zero_resource_workers,
        )

        self.rsets = np.zeros(self.total_num_rsets, dtype=ResourceManager.rset_dtype)
        self.rsets["assigned"] = 0
        self.rsets["group"], self.rsets["slot"] = ResourceManager.get_group_list(self.split_list)
        self.num_groups = self.rsets["group"][-1]
        self.rsets_free = self.total_num_rsets

        # Useful for scheduling tasks with different sized groups (resource sets per node).
        unique, counts = np.unique(self.rsets["group"], return_counts=True)
        self.group_sizes = dict(zip(unique, counts))
        self.ngroups_by_size = Counter(counts)
        self.even_groups = True if len(self.ngroups_by_size) == 1 else False

    def assign_rsets(self, rset_team, worker_id):
        """Mark the resource sets given by rset_team as assigned to worker_id"""
        if rset_team:
            rteam = self.rsets["assigned"][rset_team]
            for i, wid in enumerate(rteam):
                if wid == 0:
                    self.rsets["assigned"][rset_team[i]] = worker_id
                    self.rsets_free -= 1
                elif wid != worker_id:
                    ResourceManagerException(
                        f"Error: Attempting to assign rsets {rset_team}" f" already assigned to workers: {rteam}"
                    )

    def free_rsets(self, worker=None):
        """Free up assigned resource sets"""
        if worker is None:
            self.rsets["assigned"] = 0
            self.rsets_free = self.total_num_rsets
        else:
            rsets_to_free = np.where(self.rsets["assigned"] == worker)[0]
            self.rsets["assigned"][rsets_to_free] = 0
            self.rsets_free += len(rsets_to_free)

    @staticmethod
    def get_group_list(split_list):
        """Return lists of group ids and slot IDs by resource set"""
        group = 1
        slot = 0
        group_list = []
        slot_list = []
        node = split_list[0]
        for i in range(len(split_list)):
            if split_list[i] == node:
                group_list.append(group)
                slot_list.append(slot)
                slot += 1
            else:
                node = split_list[i]
                group += 1
                group_list.append(group)
                slot = 0
                slot_list.append(slot)
                slot += 1
        return group_list, slot_list

    @staticmethod
    def get_index_list(num_workers, num_rsets, zero_resource_list):
        """Map WorkerID to index into a nodelist"""
        index = 0
        index_list = []
        for i in range(1, num_workers + 1):
            if i in zero_resource_list:
                index_list.append(None)
            else:
                if index >= num_rsets:
                    # Not enough rsets
                    index_list.append(None)
                else:
                    index_list.append(index)
                index += 1
        return index_list


class WorkerResources(RSetResources):
    """Provide system resources per worker to libEnsemble and executor.

    **Object Attributes:**

    Some of these attributes may be updated as the ensemble progresses.

    ``rsets`` below is used to abbreviate ``resource sets``.

    :ivar int workerID: workerID for this worker.
    :ivar list local_nodelist: A list of all nodes assigned to this worker.
    :ivar list rset_team: List of rset IDs currently assigned to this worker.
    :ivar int num_rsets: The number of resource sets assigned to this worker.
    :ivar dict slots: A dictionary with a list of slot IDs for each node.
    :ivar bool even_slots: True if each node has the same number of slots.
    :ivar bool matching_slots: True if each node has matching slot IDs.
    :ivar int slot_count: The number of slots per node if even_slots is True, else None.
    :ivar list slots_on_node: A list of slots IDs if matching_slots is True, else None.
    :ivar int local_node_count: The number of nodes available to this worker (rounded up to whole number).
    :ivar int rsets_per_node: The number of rsets per node (if a rset > 1 node, will be 1).

    The worker_resources attributes can be queried, and convenience functions
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

    If ``even_slots`` is True, then the attributes ``slot_count`` will give the number
    of slots on each node. If ``matching_slots`` is True, then  ``slots_on_node`` will
    give the slot IDs for all nodes. These can be used for simplicity; otherwise, the
    ``slots`` dictionary can be used to get information for each node.

    """

    def __init__(self, num_workers, resources, workerID):

        """Initializes a new WorkerResources instance

        Determines the compute resources available for current worker, including
        node list and cores/hardware threads available within nodes.

        Parameters
        ----------

        num_workers: int
            The number of workers

        resources: Resources
            A Resources object containing global nodelist and intranode information

        workerID: int
            workerID of current process

        """
        super().__init__(num_workers, resources)
        self.workerID = workerID
        self.local_nodelist = []
        self.rset_team = None
        self.num_rsets = 0
        self.slots = None
        self.even_slots = None
        self.matching_slots = None
        self.slot_count = None
        self.slots_on_node = None
        self.zero_resource_workers = resources.zero_resource_workers
        self.local_node_count = len(self.local_nodelist)
        self.set_slot_count()

    # User convenience functions ----------------------------------------------

    def get_slots_as_string(self, multiplier=1, delimiter=","):
        """Returns list of slots as a string

        :param multiplier: Optional int. Assume this many items per slot.
        :param delimiter: Optional int. Delimiter for output string.
        """

        if self.slots_on_node is None:
            logger.warning("Slots on node is None when requested as a string")
            return None

        n = multiplier
        slot_list = [j for i in self.slots_on_node for j in range(i * n, (i + 1) * n)]
        slots = delimiter.join(map(str, slot_list))
        return slots

    def set_env_to_slots(self, env_var, multiplier=1, delimiter=","):
        """Sets the given environment variable to slots

        :param env_var: String. Name of environment variable to set.
        :param multiplier: Optional int. Assume this many items per slot.
        :param delimiter: Optional int. Delimiter for output string.

        Example usage in a sim function:

        With resources imported:

        .. code-block:: python

            from libensemble.resources.resources import Resources

        Obtain worker resources:

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
                 slot_count - number of slots on each node
                 local_node_count
        """
        if self.workerID in self.zero_resource_workers:
            return

        if rset_team != self.rset_team:  # Order matters
            self.rset_team = rset_team
            self.num_rsets = len(rset_team)
            self.local_nodelist, self.slots = WorkerResources.get_local_nodelist(
                self.workerID,
                self.rset_team,
                self.split_list,
                self.rsets_per_node,
            )
            self.set_slot_count()
            self.local_node_count = len(self.local_nodelist)

    def set_slot_count(self):
        """Sets attributes even_slots and matching_slots.

        Also sets slot_count if even_slots (else None) and
        sets slots_on_node if matching_slots (else None).
        """
        if self.slots:
            first_node_slots = list(self.slots.values())[0]
            all_match = True
            all_even = True
            first_len = len(first_node_slots)
            for slot_list in self.slots.values():
                if len(slot_list) != first_len:
                    all_even = False
                    all_match = False
                    break
                elif all_match and slot_list != first_node_slots:
                    all_match = False
            self.even_slots = True if all_even else False
            self.matching_slots = True if all_match else False
            self.slots_on_node = first_node_slots if self.matching_slots else None
            self.slot_count = first_len if self.even_slots else None

    @staticmethod
    def get_local_nodelist(workerID, rset_team, split_list, rsets_per_node):
        """Returns the list of nodes available to the given worker and the slot dictionary"""
        if workerID is None:
            raise WorkerResourcesException("Worker has no workerID - aborting")

        team_list = []
        for index in rset_team:
            team_list += split_list[index]

        local_nodelist = list(OrderedDict.fromkeys(team_list))  # Maintain order of nodes
        logger.debug(f"Worker's local_nodelist is {local_nodelist}")

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
                pos_in_node = index % rsets_per_node
                slots[mynode].append(pos_in_node)

        return local_nodelist, slots
