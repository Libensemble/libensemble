from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from libensemble.resources.resources import GlobalResources

logger = logging.getLogger(__name__)
# To change logging level for just this module
# logger.setLevel(logging.DEBUG)


class RSetResourcesException(Exception):
    """Resource sets cannot be created for the given configuration."""


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
        ("gpu_id", int),  # Physical GPU ID for GPU resource sets
        ("gpu_share", int),  # Logical share within a physical GPU
        # ('pool', int),    # Pool ID (eg. separate gen/sim resources) - not yet used.
    ]

    def __init__(self, num_workers: int, resources: GlobalResources):
        """Initializes a new RSetResources instance

        Determines the compute resources available for each resource set.

        Unless resource sets is set explicitly, the number of resource sets is the number of workers.
        If ``gpu_sharing_factor > 1`` and GPUs are available, the default is instead one resource set
        per GPU share: ``nodes * gpus_per_node * gpu_sharing_factor``.

        Parameters
        ----------

        num_workers: int
            The total number of workers

        resources: GlobalResources
            A GlobalResources object containing global nodelist and intranode information

        """
        self.num_workers = num_workers
        self.num_workers_2assign2 = RSetResources.get_workers2assign2(self.num_workers, resources)
        self.num_nodes = len(resources.global_nodelist)
        self.gpu_sharing_factor = resources.gpu_sharing_factor
        gpus_avail_per_node = resources.gpus_avail_per_node
        num_gpu_shares_per_node = gpus_avail_per_node * self.gpu_sharing_factor
        self.rsets_sizing_msg = None
        self.rsets_sized_by_gpu_sharing = (
            not resources.num_resource_sets and self.gpu_sharing_factor > 1 and gpus_avail_per_node > 0
        )
        if self.rsets_sized_by_gpu_sharing:
            self.total_num_rsets = self.num_nodes * num_gpu_shares_per_node
            self.rsets_sizing_msg = (
                f"gpu_sharing_factor={self.gpu_sharing_factor}: num_resource_sets defaulting to "
                f"{self.total_num_rsets} ({self.num_nodes} node(s) x {gpus_avail_per_node} GPUs "
                f"x {self.gpu_sharing_factor} shares)"
            )
        else:
            self.total_num_rsets = resources.num_resource_sets or self.num_workers_2assign2
        self.split_list, self.local_rsets_list = RSetResources.get_partitioned_nodelist(self.total_num_rsets, resources)
        self.nodes_in_rset = len(self.split_list[0])

        self.rsets_per_node = RSetResources.get_rsets_on_a_node(self.total_num_rsets, resources)
        if self.gpu_sharing_factor > 1 and gpus_avail_per_node:
            RSetResources.check_gpu_sharing_rsets(
                self.total_num_rsets, self.num_nodes, self.rsets_per_node, gpus_avail_per_node, self.gpu_sharing_factor
            )
        self.gpu_rsets_per_node = min(gpus_avail_per_node * self.gpu_sharing_factor, self.rsets_per_node)
        self.nongpu_rsets_per_node = self.rsets_per_node - self.gpu_rsets_per_node

        self.all_rsets = np.zeros(self.total_num_rsets, dtype=RSetResources.rset_dtype)
        self.all_rsets["group"], self.all_rsets["slot"], self.all_rsets["gpus"] = RSetResources.get_group_list(
            self.split_list, gpus_avail_per_node, resources.gpus_per_group, self.gpu_sharing_factor
        )
        if gpus_avail_per_node:
            gpu_positions = RSetResources.get_gpu_positions(self.split_list)
            self.all_rsets["gpu_id"] = np.where(
                self.all_rsets["gpus"],
                (gpu_positions // self.gpu_sharing_factor) % gpus_avail_per_node,
                -1,
            )
            self.all_rsets["gpu_share"] = np.where(self.all_rsets["gpus"], gpu_positions % self.gpu_sharing_factor, -1)

        self.total_num_gpu_rsets = np.count_nonzero(self.all_rsets["gpus"])
        self.total_num_nongpu_rsets = np.count_nonzero(~self.all_rsets["gpus"])

        if self.gpu_rsets_per_node:
            self.gpus_per_rset_per_node = (
                1
                if self.gpu_sharing_factor > 1
                else gpus_avail_per_node // min(gpus_avail_per_node, self.rsets_per_node)
            )
        else:
            self.gpus_per_rset_per_node = 0
        self.cores_per_rset_per_node = resources.physical_cores_avail_per_node // self.rsets_per_node
        if self.cores_per_rset_per_node == 0 and self.gpu_sharing_factor > 1:
            raise RSetResourcesException(
                f"gpu_sharing_factor={self.gpu_sharing_factor} gives {self.rsets_per_node} resource sets per node, "
                f"but there are only {resources.physical_cores_avail_per_node} CPU cores per node. "
                "Reduce gpu_sharing_factor so that each resource set has at least one core."
            )

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
    def check_gpu_sharing_rsets(total_num_rsets, num_nodes, rsets_per_node, gpus_per_node, gpu_sharing_factor):
        """Check that resource sets cover every GPU share on each node"""
        shares_per_node = gpus_per_node * gpu_sharing_factor
        if rsets_per_node < shares_per_node:
            raise RSetResourcesException(
                f"gpu_sharing_factor={gpu_sharing_factor} with {gpus_per_node} GPUs per node needs "
                f"{shares_per_node} resource sets per node ({shares_per_node * num_nodes} total on {num_nodes} "
                f"node(s)), but num_resource_sets={total_num_rsets} gives only {rsets_per_node} per node. "
                f"Set num_resource_sets to at least {shares_per_node * num_nodes}, or leave it unset."
            )

    @staticmethod
    def get_group_list(split_list, gpus_per_node=0, gpus_per_group=None, gpu_sharing_factor=1):
        """Return group, slot, and GPU lists by resource set"""
        group = 1
        slot = 0
        group_list, slot_list, gpu_list = [], [], []
        node = split_list[0]

        for i in range(len(split_list)):
            # break on new node or when group size reached (account for sharing)
            group_threshold = gpus_per_group * gpu_sharing_factor if gpus_per_group is not None else None
            if split_list[i] != node or slot == group_threshold:
                node = split_list[i]
                group += 1
                slot = 0
            group_list.append(group)
            slot_list.append(slot)
            is_gpu = slot < gpus_per_node * gpu_sharing_factor
            gpu_list.append(is_gpu)
            slot += 1
        return group_list, slot_list, gpu_list

    @staticmethod
    def get_gpu_positions(split_list):
        """Return each resource set's node-local position, independent of scheduler groups."""
        positions = []
        node_counts = {}
        for nodes in split_list:
            node = nodes[0]
            positions.append(node_counts.get(node, 0))
            node_counts[node] = positions[-1] + 1
        return np.asarray(positions)

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
        return num_workers

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
