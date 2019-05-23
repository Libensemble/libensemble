"""
Manage libensemble resources related to MPI jobs launched from nodes.
"""

import os
import logging

from libensemble.resources import Resources, ResourcesException


def rassert(test, *args):
    if not test:
        raise ResourcesException(*args)


logger = logging.getLogger(__name__)
# To change logging level for just this module
# logger.setLevel(logging.DEBUG)


class MPIResources(Resources):

    @staticmethod
    def job_partition(num_procs, num_nodes, ranks_per_node, machinefile=None):
        """Takes provided nprocs/nodes/ranks and outputs working
        configuration of procs/nodes/ranks or error"""

        # If machinefile is provided - ignore everything else
        if machinefile:
            if num_procs or num_nodes or ranks_per_node:
                logger.warning("Machinefile provided - overriding "
                               "procs/nodes/ranks_per_node")
            return None, None, None

        if not num_procs:
            rassert(num_nodes and ranks_per_node,
                    "Need num_procs, num_nodes/ranks_per_node, or machinefile")
            num_procs = num_nodes * ranks_per_node

        elif not num_nodes:
            ranks_per_node = ranks_per_node or num_procs
            num_nodes = num_procs//ranks_per_node

        elif not ranks_per_node:
            ranks_per_node = num_procs//num_nodes

        rassert(num_procs == num_nodes*ranks_per_node,
                "num_procs does not equal num_nodes*ranks_per_node")
        return num_procs, num_nodes, ranks_per_node

    def get_resources(self, num_procs=None, num_nodes=None,
                      ranks_per_node=None, hyperthreads=False):
        """Reconciles user supplied options with available Worker
        resources to produce run configuration.

        Detects resources available to worker, checks if an existing
        user supplied config is valid, and fills in any missing config
        information (ie. num_procs/num_nodes/ranks_per_node)

        User supplied config options are honoured, and an exception is
        raised if these are infeasible.
        """

        node_list = self.worker_resources.local_nodelist
        num_workers = self.worker_resources.num_workers
        local_node_count = self.worker_resources.local_node_count

        cores_avail_per_node = \
            (self.logical_cores_avail_per_node if hyperthreads else
             self.physical_cores_avail_per_node)
        workers_per_node = \
            (self.worker_resources.workers_per_node if num_workers > local_node_count else 1)
        cores_avail_per_node_per_worker = cores_avail_per_node//workers_per_node

        rassert(node_list, "Node list is empty - aborting")

        # If no decomposition supplied - use all available cores/nodes
        if not num_procs and not num_nodes and not ranks_per_node:
            num_nodes = local_node_count
            ranks_per_node = cores_avail_per_node_per_worker
            logger.debug("No decomposition supplied - "
                         "using all available resource. "
                         "Nodes: {}  ranks_per_node {}".
                         format(num_nodes, ranks_per_node))
        elif not num_nodes and not ranks_per_node:
            num_nodes = local_node_count
        elif not num_procs and not ranks_per_node:
            ranks_per_node = cores_avail_per_node_per_worker
        elif not num_procs and not num_nodes:
            num_nodes = local_node_count

        # Checks config is consistent and sufficient to express
        # - does not check actual resources
        num_procs, num_nodes, ranks_per_node = \
            MPIResources.job_partition(num_procs, num_nodes, ranks_per_node)

        # Could just downgrade to those available with warning - for now error
        rassert(num_nodes <= local_node_count,
                "Not enough nodes to honour arguments. "
                "Requested {}. Only {} available".
                format(num_nodes, local_node_count))

        rassert(ranks_per_node <= cores_avail_per_node,
                "Not enough processors on a node to honour arguments. "
                "Requested {}. Only {} available".
                format(ranks_per_node, cores_avail_per_node))

        rassert(ranks_per_node <= cores_avail_per_node_per_worker,
                "Not enough processors per worker to honour arguments. "
                "Requested {}. Only {} available".
                format(ranks_per_node, cores_avail_per_node_per_worker))

        rassert(num_procs <= (cores_avail_per_node * local_node_count),
                "Not enough procs to honour arguments. "
                "Requested {}. Only {} available".
                format(num_procs, cores_avail_per_node*local_node_count))

        if num_nodes < local_node_count:
            logger.warning("User constraints mean fewer nodes being used "
                           "than available. {} nodes used. {} nodes available".
                           format(num_nodes, local_node_count))

        return num_procs, num_nodes, ranks_per_node

    def create_machinefile(self, machinefile=None, num_procs=None,
                           num_nodes=None, ranks_per_node=None,
                           hyperthreads=False):
        """Create a machinefile based on user supplied config options,
        completed by detected machine resources"""

        machinefile = machinefile or 'machinefile'
        if os.path.isfile(machinefile):
            try:
                os.remove(machinefile)
            except Exception as e:
                logger.warning("Could not remove existing machinefile: {}".format(e))

        node_list = self.worker_resources.local_nodelist
        logger.debug("Creating machinefile with {} nodes and {} ranks per node".
                     format(num_nodes, ranks_per_node))

        with open(machinefile, 'w') as f:
            for node in node_list[:num_nodes]:
                f.write((node + '\n') * ranks_per_node)

        built_mfile = (os.path.isfile(machinefile)
                       and os.path.getsize(machinefile) > 0)
        return built_mfile, num_procs, num_nodes, ranks_per_node

    def get_hostlist(self):
        """Create a hostlist based on user supplied config options,
        completed by detected machine resources"""
        node_list = self.worker_resources.local_nodelist
        hostlist_str = ",".join([str(x) for x in node_list])
        return hostlist_str
