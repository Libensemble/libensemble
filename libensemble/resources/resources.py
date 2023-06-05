"""
This module detects and returns system resources

"""

import logging
import os
import socket

from libensemble.resources import node_resources
from libensemble.resources.env_resources import EnvResources
from libensemble.resources.mpi_resources import get_MPI_runner
from libensemble.resources.worker_resources import ResourceManager, WorkerResources

logger = logging.getLogger(__name__)
# To change logging level for just this module
# logger.setLevel(logging.DEBUG)


class ResourcesException(Exception):
    """Resources module exception"""


class Resources:
    """Provides system resources to libEnsemble and executor.

    A resources instance is always initialized unless ``libE_specs["disable_resource_manager"]`` is ``True``.

    **Class Attributes:**

    :cvar Resources: resources: The resources object is stored here and can be retrieved in user functions.

    **Object Attributes:**

    These are set on initialization.

    :ivar string top_level_dir: Directory where searches for node_list file.
    :ivar GlobalResources glob_resources: Maintains resources available to libEnsemble.

    The following are set up after manager/worker fork.

    The resource manager is set up only on the manaager, while the worker resources object is set
    up on workers.

    :ivar ResourceManager resource_manager: An object that manages resource set assignment to workers.
    :ivar WorkerResources worker_resources: An object that contains worker specific resources.
    """

    resources = None

    DEFAULT_NODEFILE = "node_list"

    @classmethod
    def init_resources(cls, libE_specs: dict, platform_info: dict = {}) -> None:
        """Initiate resource management"""
        # If disable_resource_manager is True, then Resources.resources will remain None.
        disable_resource_manager = libE_specs.get("disable_resource_manager", False)
        if not disable_resource_manager:
            top_level_dir = os.getcwd()
            Resources.resources = Resources(
                libE_specs=libE_specs, platform_info=platform_info, top_level_dir=top_level_dir
            )

    def __init__(self, libE_specs: dict, platform_info: dict = {}, top_level_dir: str = None) -> None:
        """Initiate a new resources object"""
        self.top_level_dir = top_level_dir or os.getcwd()
        self.glob_resources = GlobalResources(libE_specs=libE_specs, platform_info=platform_info, top_level_dir=None)
        self.resource_manager = None  # For Manager
        self.worker_resources = None  # For Workers

    def set_worker_resources(self, num_workers: int, workerid: int) -> None:
        """Initiate the worker resources component of resources"""
        self.worker_resources = WorkerResources(num_workers, self.glob_resources, workerid)

    def set_resource_manager(self, num_workers: int) -> None:
        """Initiate the resource manager component of resources"""
        self.resource_manager = ResourceManager(num_workers, self.glob_resources)

    def add_comm_info(self, libE_nodes) -> None:
        """Adds comms-specific information to resources

        Removes libEnsemble nodes from nodelist if in dedicated_mode.
        """
        self.glob_resources.add_comm_info(libE_nodes)


class GlobalResources:
    """
    **Object Attributes:**

    These are set on initialization.
    :ivar string top_level_dir: Directory where searches for node_list file
    :ivar EnvResources env_resources: Object storing environment variables used by resources
    :ivar list global_nodelist: list of all nodes available for running user applications
    :ivar int logical_cores_avail_per_node: Logical cores (including SMT threads) available on a node
    :ivar int physical_cores_avail_per_node: Physical cores available on a node
    :ivar list zero_resource_workers: List of workerIDs to have no resources.
    :ivar boolean dedicated_mode: Whether to remove libE nodes from global nodelist.
    :ivar int num_resource_sets: Number of resource sets, if supplied by the user.
    """

    def __init__(self, libE_specs: dict, platform_info: dict = {}, top_level_dir: str = None) -> None:
        """Initializes a new Resources instance

        Determines the compute resources available for current allocation, including
        node list and cores/hardware threads available within nodes.

        The following parameters may be extracted from ``libE_specs``

        Parameters
        ----------

        top_level_dir: string, optional
            Directory libEnsemble runs in (default is current working directory)

        dedicated_mode: boolean, optional
            If true, then dedicate nodes to running libEnsemble.
            Dedicated mode means that any nodes running libE processes (manager and workers),
            will not be available to worker launched tasks (user applications). They will
            be removed from the nodelist (if present), before dividing into resource sets.

        zero_resource_workers: list of ints, optional
            List of workers that require no resources.

        num_resource_sets: int, optional
            The total number of resource sets. Resources will be divided into this number.
            Default: None. If None, resources will be divided by workers (excluding zero_resource_workers).

        cores_on_node: tuple (int, int), optional
            If supplied gives (physical cores, logical cores) for the nodes. If not supplied,
            this will be auto-detected.

        gpus_on_node: int, optional
            If supplied gives number of GPUs for the nodes. If not supplied,
            this will be auto-detected.

        enforce_worker_core_bounds: boolean, optional
            If True, then libEnsemble's executor will raise an exception if it detects that
            a worker has been instructed to launch tasks with the number of requested processes
            being excessive to the number of cores allocated to that worker, or not enough
            processes were requested to satisfy allocated cores.

        node_file: String, optional
            If supplied, give the name of a file in the run directory to use as a node-list
            for use by libEnsemble. Defaults to a file named "node_list". If the file does
            not exist, then the node-list will be auto-detected.

        nodelist_env_slurm: String, optional
            The environment variable giving a node list in Slurm format (Default: uses SLURM_NODELIST).
            Note: This is queried only if a node_list file is not provided.

        nodelist_env_cobalt: String, optional
            The environment variable giving a node list in Cobalt format (Default: uses COBALT_PARTNAME).
            Note: This is queried only if a node_list file is not provided.

        nodelist_env_lsf: String, optional
            The environment variable giving a node list in LSF format (Default: uses LSB_HOSTS).
            Note: This is queried only if a node_list file is not provided.

        nodelist_env_lsf_shortform: String, optional
            The environment variable giving a node list in LSF short-form format (Default: uses LSB_MCPU_HOSTS)
            Note: This is only queried if a node_list file is not provided.

        """
        self.top_level_dir = top_level_dir
        self.dedicated_mode = libE_specs.get("dedicated_mode", False)
        self.zero_resource_workers = libE_specs.get("zero_resource_workers", [])
        self.num_resource_sets = libE_specs.get("num_resource_sets", None)
        self.enforce_worker_core_bounds = libE_specs.get("enforce_worker_core_bounds", False)
        resource_info = libE_specs.get("resource_info", {})

        # resource_info overrides platform
        cores_on_node = resource_info.get("cores_on_node")
        if cores_on_node is None:
            cores_on_node = (platform_info.get("cores_per_node"), platform_info.get("logical_cores_per_node"))
        gpus_on_node = resource_info.get("gpus_on_node") or platform_info.get("gpus_per_node")

        node_file = resource_info.get("node_file", None)
        nodelist_env_slurm = resource_info.get("nodelist_env_slurm", None)
        nodelist_env_cobalt = resource_info.get("nodelist_env_cobalt", None)
        nodelist_env_lsf = resource_info.get("nodelist_env_lsf", None)
        nodelist_env_lsf_shortform = resource_info.get("nodelist_env_lsf_shortform", None)

        self.env_resources = EnvResources(
            nodelist_env_slurm=nodelist_env_slurm,
            nodelist_env_cobalt=nodelist_env_cobalt,
            nodelist_env_lsf=nodelist_env_lsf,
            nodelist_env_lsf_shortform=nodelist_env_lsf_shortform,
        )

        if node_file is None:
            node_file = Resources.DEFAULT_NODEFILE

        self.global_nodelist = GlobalResources.get_global_nodelist(
            node_file=node_file,
            rundir=self.top_level_dir,
            env_resources=self.env_resources,
        )

        self.shortnames = GlobalResources.is_nodelist_shortnames(self.global_nodelist)
        if self.shortnames:
            self.local_host = self.env_resources.shortnames([socket.gethostname()])[0]
        else:
            self.local_host = socket.gethostname()

        # Note: Launcher used here just to get cores on node etc - independent of whether using MPIExecutor
        self.launcher = platform_info.get("runner_name") or get_MPI_runner(platform_info.get("mpi_runner"))

        remote_detect = False
        if self.local_host not in self.global_nodelist and self.launcher is not None:
            remote_detect = True

        if cores_on_node is None or gpus_on_node is None or None in cores_on_node:
            detected_config = node_resources.get_sub_node_resources(
                launcher=self.launcher, remote_mode=remote_detect, env_resources=self.env_resources
            )
            cores_on_node, gpus_on_node = self._add_detected_info(cores_on_node, gpus_on_node, detected_config)

        self.physical_cores_avail_per_node = cores_on_node[0]
        self.logical_cores_avail_per_node = cores_on_node[1]
        self.gpus_avail_per_node = gpus_on_node
        self.platform_info = platform_info
        self.libE_nodes = None

    def add_comm_info(self, libE_nodes):
        """Adds comms-specific information to resources

        Removes libEnsemble nodes from nodelist if in dedicated_mode.
        """
        if self.shortnames:
            self.libE_nodes = self.env_resources.shortnames(libE_nodes)
        else:
            self.libE_nodes = libE_nodes
        libE_nodes_in_list = list(filter(lambda x: x in self.libE_nodes, self.global_nodelist))
        if libE_nodes_in_list:
            if self.dedicated_mode and len(self.global_nodelist) > 1:
                self.global_nodelist = GlobalResources.remove_nodes(self.global_nodelist, self.libE_nodes)
                if not self.global_nodelist:
                    logger.warning("Warning. Node-list for tasks is empty. Remove dedicated_mode or add nodes")
                    pass

    def update_scheduler_opts(self, scheduler_opts):
        """Add scheduler options from platform_info, if not present"""
        if self.platform_info and scheduler_opts is not None:
            if "match_slots" not in scheduler_opts:
                if "scheduler_match_slots" in self.platform_info:
                    scheduler_opts["match_slots"] = self.platform_info["scheduler_match_slots"]
        return scheduler_opts

    def _add_detected_info(self, cores_on_node, gpus_on_node, detected_config):
        """Update missing values in cores/gpus_on_node"""
        if not cores_on_node:
            cores_on_node = detected_config[0:2]
        elif None in cores_on_node:
            cores_on_node = list(cores_on_node)
            if not cores_on_node[0]:
                cores_on_node[0] = detected_config[0]
            if not cores_on_node[1]:
                cores_on_node[1] = detected_config[1]
            cores_on_node = tuple(cores_on_node)
        if not gpus_on_node:
            gpus_on_node = detected_config[2]
        return cores_on_node, gpus_on_node

    @staticmethod
    def is_nodelist_shortnames(nodelist):
        """Returns False if any entry contains a '.', else True"""
        for item in nodelist:
            if "." in item:
                return False
        return True

    # This is for dedicated mode where libE nodes will not share with app nodes
    @staticmethod
    def remove_nodes(global_nodelist_in, remove_list):
        """Removes any nodes in remove_list from the global nodelist"""
        global_nodelist = list(filter(lambda x: x not in remove_list, global_nodelist_in))
        return global_nodelist

    @staticmethod
    def get_global_nodelist(node_file=Resources.DEFAULT_NODEFILE, rundir=None, env_resources=None):
        """
        Returns the list of nodes available to all libEnsemble workers.

        If a node_file exists this is used, otherwise the environment
        is interrogated for a node list. If a dedicated manager node is used,
        then a node_file is recommended.

        In dedicated mode, any node with a libE worker is removed from the list.
        """
        top_level_dir = rundir or os.getcwd()
        node_filepath = os.path.join(top_level_dir, node_file)
        global_nodelist = []
        if os.path.isfile(node_filepath):
            with open(node_filepath, "r") as f:
                for line in f:
                    global_nodelist.append(line.rstrip())
        else:
            if env_resources:
                global_nodelist = env_resources.get_nodelist()

            if not global_nodelist:
                # Assume a standalone machine
                # global_nodelist.append(env_resources.shortnames([socket.gethostname()])[0])
                global_nodelist.append(socket.gethostname())

        if global_nodelist:
            return global_nodelist
        raise ResourcesException("Error. global_nodelist is empty")
