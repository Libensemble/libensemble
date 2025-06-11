from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from libensemble.resources.resources import Resources

logger = logging.getLogger(__name__)
# To change logging level for just this module
# logger.setLevel(logging.DEBUG)


class RSetResources:
    """A class that creates a fixed mapping of resource sets to the available resources.

    **Object Attributes:**

    These are set on initialization and include inherited.
    ``rsets`` below is used to abbreviate ``resource sets``.

    :ivar int num_workers: Total number of workers
    :ivar int num_workers_2assign2: The number of workers that will be assigned resource sets.
    :ivar int total_num_rsets: The total number of resource sets.
    :ivar list split_list: A list of lists, where each element is the list of nodes for a given rset.
    :ivar list local_rsets_list: A list over rsets, where each element is the number of rsets that share the node.
    :ivar int rsets_per_node: The number of rsets per node (if an rset > 1 node, this will be 1)
    """

    rset_dtype = [
        ("group", int),  # Group ID this resource set belongs to
        ("slot", int),  # Slot ID this resource set belongs to
        ("gpus", bool),  # Does this resource set have GPUs
        # ('pool', int),    # Pool ID (eg. separate gen/sim resources) - not yet used.
    ]

    def __init__(self, num_workers: int, resources: Resources):
        """Initializes a new RSetResources instance

        Determines the compute resources available for each resource set.

        Unless resource sets is set explicitly, the number of resource sets is the number of workers,
        excluding any workers defined as zero resource workers.

        Parameters
        ----------

        num_workers: int
            The total number of workers

        resources: Resources
            A Resources object containing global nodelist and intranode information

        """
        self.num_workers = num_workers
        self.num_workers_2assign2 = RSetResources.get_workers2assign2(self.num_workers, resources)
        self.total_num_rsets = resources.num_resource_sets or self.num_workers_2assign2
        self.num_nodes = len(resources.global_nodelist)
        self.split_list, self.local_rsets_list = RSetResources.get_partitioned_nodelist(self.total_num_rsets, resources)
        self.nodes_in_rset = len(self.split_list[0])

        gpus_avail_per_node = resources.gpus_avail_per_node
        self.rsets_per_node = RSetResources.get_rsets_on_a_node(self.total_num_rsets, resources)
        self.gpu_rsets_per_node = min(gpus_avail_per_node, self.rsets_per_node)
        self.nongpu_rsets_per_node = self.rsets_per_node - self.gpu_rsets_per_node

        self.all_rsets = np.zeros(self.total_num_rsets, dtype=RSetResources.rset_dtype)
        self.all_rsets["group"], self.all_rsets["slot"], self.all_rsets["gpus"] = RSetResources.get_group_list(
            self.split_list, gpus_avail_per_node, resources.gpus_per_group
        )

        self.total_num_gpu_rsets = np.count_nonzero(self.all_rsets["gpus"])
        self.total_num_nongpu_rsets = np.count_nonzero(~self.all_rsets["gpus"])

        self.gpus_per_rset_per_node = gpus_avail_per_node // self.gpu_rsets_per_node if self.gpu_rsets_per_node else 0
        self.cores_per_rset_per_node = resources.physical_cores_avail_per_node // self.rsets_per_node

        # Oversubsribe
        if self.cores_per_rset_per_node == 0:
            cpn = resources.physical_cores_avail_per_node
            procs_per_core = self.rsets_per_node // cpn + (self.rsets_per_node % cpn > 0)
            self.procs_per_rset_per_node = resources.physical_cores_avail_per_node * procs_per_core
        else:
            self.procs_per_rset_per_node = self.cores_per_rset_per_node

        self.gpus_per_rset = self.gpus_per_rset_per_node * self.nodes_in_rset
        self.cores_per_rset = self.cores_per_rset_per_node * self.nodes_in_rset
        self.procs_per_rset = self.procs_per_rset_per_node * self.nodes_in_rset

    @staticmethod
    def get_group_list(split_list, gpus_per_node=0, gpus_per_group=None):
        """Return lists of group ids and slot IDs by resource set"""
        group = 1
        slot = 0
        group_list, slot_list, gpu_list = [], [], []
        node = split_list[0]

        for i in range(len(split_list)):
            # still break on new node if gpus_per_group is set
            if split_list[i] != node or slot == gpus_per_group:
                node = split_list[i]
                group += 1
                slot = 0
            group_list.append(group)
            slot_list.append(slot)
            gpu_list.append(slot < gpus_per_node)
            slot += 1
        return group_list, slot_list, gpu_list

    @staticmethod
    def best_split(a, n):
        """Creates the most even split of list a into n parts and return list of lists"""
        k, m = divmod(len(a), n)
        return (a[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n))

    @staticmethod
    def get_rsets_on_a_node(num_rsets, resources):
        """Returns the number of resource sets that can be placed on each node

        If there are more nodes than resource sets, returns 1.
        """
        num_nodes = len(resources.global_nodelist)
        # Round up if there's a remainder
        rsets_per_node = num_rsets // num_nodes + (num_rsets % num_nodes > 0)
        return rsets_per_node

    @staticmethod
    def get_workers2assign2(num_workers, resources):
        """Returns workers to assign resources to"""
        zero_resource_list = resources.zero_resource_workers
        return num_workers - len(zero_resource_list)

    @staticmethod
    def even_assignment(nnodes, nworkers):
        """Returns True if workers are evenly distributed to nodes, else False"""
        try:
            return nnodes % nworkers == 0 or nworkers % nnodes == 0
        except ZeroDivisionError:
            logger.error("Either nworkers or nnodes is zero. Check that MPI ranks > 1")
            raise

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

    @staticmethod
    def get_split_list(num_rsets, resources):
        """Returns a list of lists for each worker

        Assumes that self.global_nodelist has been calculated (in __init__).
        """
        global_nodelist = resources.global_nodelist
        num_nodes = len(global_nodelist)

        if not RSetResources.even_assignment(num_nodes, num_rsets):
            logger.warning(f"Resource sets ({num_rsets}) are not distributed evenly to available nodes ({num_nodes})")

        # If multiple workers per node - create global node_list with N duplicates (for N workers per node)
        sub_node_workers = num_rsets >= num_nodes
        if sub_node_workers:
            global_nodelist, local_rsets_list = RSetResources.expand_list(num_nodes, num_rsets, global_nodelist)
        else:
            local_rsets_list = [1] * num_rsets

        # Divide global list between workers
        split_list = list(RSetResources.best_split(global_nodelist, num_rsets))
        logger.debug(f"split_list is {split_list}")
        return split_list, local_rsets_list

    @staticmethod
    def get_partitioned_nodelist(num_rsets, resources):
        """Returns lists of nodes available to all resource sets

        Assumes that self.global_nodelist has been calculated (in __init__).
        Also self.global_nodelist will have already removed non-application nodes
        """
        split_list, local_rsets_list = RSetResources.get_split_list(num_rsets, resources)
        return split_list, local_rsets_list
