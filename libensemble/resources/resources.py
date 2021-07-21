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


#  Restructure  ===========================================================================================================================
# - resources
#    - self.hw_resources = HW_Resources() or HWResources() or AvailResources() - currently bulk of resources
#    - self.worker_resources = WorkerResources()
#    - self.resource_manager = ResourceManager() - currently ManagerWorkerResources
# -  - MPIResources - will no longer be inherited from resources - will be initiaed by MPIExectuor so either part of executor or in this
#                     directory - but added when needed - may not even be class - just MPI utility routines that executor can call - passing it resources...
# Resource will no longer pass through self to worker resources.
# WorkerResources() and ResourceManager() both inherit from an abstract class - that contains the common rset to hw_resource mapping.

#  =========================================================================================================================================

import os
import socket
import logging
import subprocess
from collections import Counter
from collections import OrderedDict
import numpy as np
from libensemble.resources import node_resources
from libensemble.resources.env_resources import EnvResources
from libensemble.resources.worker_resources import (WorkerResources,
                                                    ResourceManager)

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

    #def set_worker_resources(self, workerid, comm):
    def set_worker_resources(self, num_workers, workerid):
        self.worker_resources = WorkerResources(num_workers, self, workerid)

    def set_managerworker_resources(self, num_workers):
        self.managerworker_resources = ResourceManager(num_workers, self)

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
