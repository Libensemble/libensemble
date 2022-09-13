import logging

logger = logging.getLogger(__name__)
# To change logging level for just this module
# logger.setLevel(logging.DEBUG)


class RSetResources:
    """A class that creates a fixed mapping of resource sets to the available resources.

    **Object Attributes:**

    These are set on initialisation and include inherited.
    ``rsets`` below is used to abbreviate ``resource sets``.

    :ivar int num_workers: Total number of workers
    :ivar int num_workers_2assign2: The number of workers that will be assigned resource sets.
    :ivar int total_num_rsets: The total number of resource sets.
    :ivar list split_list: A list of lists, where each element is the list of nodes for a given rset.
    :ivar list local_rsets_list: A list over rsets, where each element is the number of rsets that share the node.
    :ivar int rsets_per_node: The number of rsets per node (if an rset > 1 node, this will be 1)
    """

    def __init__(self, num_workers, resources):
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
        self.split_list, self.local_rsets_list = RSetResources.get_partitioned_nodelist(self.total_num_rsets, resources)
        self.rsets_per_node = RSetResources.get_rsets_on_a_node(self.total_num_rsets, resources)

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
