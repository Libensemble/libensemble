"""
Module for detecting and returning system resources

"""

#from mpi4py import MPI
import os
import socket
import logging
import itertools
import subprocess

logger = logging.getLogger(__name__)
#For debug messages in this module  - uncomment (see libE.py to change root logging level)
#logger.setLevel(logging.DEBUG)

class ResourcesException(Exception): pass

class Resources:

    """Provide system resources to libEnsemble and job controller with knowledge of workers"""

    # These can be overridden by passing in (e.g. nodelist_env_slurm) on init.
    default_nodelist_env_slurm = 'SLURM_NODELIST'
    default_nodelist_env_cobalt = 'COBALT_PARTNAME'

    def __init__(self, top_level_dir=None, workerID=None, central_mode=False,
                 nodelist_env_slurm = None,
                 nodelist_env_cobalt = None):
        """Initialise new Resources instance"""

        self.top_level_dir = top_level_dir or os.getcwd()
        self.central_mode = central_mode

        # These presence of these env vars will be used to detect scheduler
        self.nodelist_env_slurm  = nodelist_env_slurm  or Resources.default_nodelist_env_slurm
        self.nodelist_env_cobalt = nodelist_env_cobalt or Resources.default_nodelist_env_cobalt

        #This is global nodelist avail to workers - may change to global_worker_nodelist
        self.global_nodelist = Resources.get_global_nodelist(rundir=self.top_level_dir, central_mode=self.central_mode,
                                                             nodelist_env_slurm = self.nodelist_env_slurm,
                                                             nodelist_env_cobalt = self.nodelist_env_cobalt)
        self.num_workers = Resources.get_num_workers()
        self.logical_cores_avail_per_node = Resources.get_cpu_cores(hyperthreads=True)
        self.physical_cores_avail_per_node = Resources.get_cpu_cores(hyperthreads=False)

        if not Resources.am_I_manager():
            self.workerID = workerID or Resources.get_workerID()
            self.local_nodelist = self.get_available_nodes()
            self.local_node_count = len(self.local_nodelist)
            self.workers_per_node = self.get_workers_on_a_node()


    #Will be in comms module ------------------------------------------------

    @staticmethod
    def am_I_manager():
        from mpi4py import MPI
        return MPI.COMM_WORLD.Get_rank() == 0

    @staticmethod
    def get_workerID():
        from mpi4py import MPI
        if MPI.COMM_WORLD.Get_rank() == 0:
            logger.warning('get_workerID called by manager - returning 0')
        return MPI.COMM_WORLD.Get_rank()

    @staticmethod
    def get_num_workers():
        """Return the total number of workers"""
        #Will use MPI_MODE from settyings.py global - for now assume using mpi.
        #Or the function may be in some worker_concurrency module
        from mpi4py import MPI
        num_workers = MPI.COMM_WORLD.Get_size() - 1
        return num_workers

    #Call from all libE tasks (pref. inc. manager)
    @staticmethod
    def get_libE_nodes():
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        #rank = MPI.COMM_WORLD.Get_rank()

        #This is a libE node
        local_host = socket.gethostname() #or MPI version
        #all_hosts=[]
        all_hosts = comm.allgather(local_host)
        return all_hosts

    @staticmethod
    def get_MPI_variant():
        # Explore mpi4py.MPI.get_vendor() and mpi4py.MPI.Get_library_version() for mpi4py
        try_mpich = subprocess.Popen(['mpirun', '-npernode'], stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT)
        stdout, _ = try_mpich.communicate()
        if 'unrecognized argument npernode' in stdout.decode():
            return 'mpich'
        else:
            return 'openmpi'

    #---------------------------------------------------------------------------


    @staticmethod
    def get_slurm_nodelist(node_list_env):
        """Get global libEnsemble nodelist from the Slurm environment"""
        nidlst = []
        fullstr = os.environ[node_list_env]
        if not len(fullstr):
            return []
        splitstr = fullstr.split('-',1)
        prefix = splitstr[0]
        nidstr = splitstr[1].strip("[]")
        nidgroups = nidstr.split(',')
        for nidgroup in nidgroups:
            if (nidgroup.find("-") != -1):
                a, b = nidgroup.split("-", 1)
                nnum_len = len(a)
                a = int(a)
                b = int(b)
                if (a > b):
                    tmp = b
                    b = a
                    a = tmp
                b = b + 1 #need one more for inclusive
            else:
                a = nidgroup
                nnum_len = len(a)
                a = int(nidgroup)
                b = a + 1
            for nid in range(a, b):
                nidlst.append(prefix + '-' + str(nid).zfill(nnum_len))
        nidlst = sorted(list(set(nidlst)))
        return nidlst

    @staticmethod
    def get_cobalt_nodelist(node_list_env):
        """Get global libEnsemble nodelist from the Cobalt environment"""
        prefix='nid'
        hostname = socket.gethostname()
        nnum_len = 5 # default
        if hostname.startswith(prefix):
            nnum_len = len(hostname[len(prefix):])
        nidlst = []
        nidstr = os.environ[node_list_env]
        if not len(nidstr):
            return []
        nidgroups = nidstr.split(',')
        for nidgroup in nidgroups:
            if (nidgroup.find("-") != -1):
                a, b = nidgroup.split("-", 1)
                #nnum_len = len(a)
                a = int(a)
                b = int(b)
                if (a > b):
                    tmp = b
                    b = a
                    a = tmp
                b = b + 1 #need one more for inclusive
            else:
                #a = nidgroup
                #nnum_len = len(a)
                a = int(nidgroup)
                b = a + 1
            for nid in range(a, b):
                nidlst.append(prefix + str(nid).zfill(nnum_len))
                #nidlst.append(nid)
        nidlst = sorted(list(set(nidlst)))
        return nidlst


    #This is for central mode where libE nodes will not share with app nodes
    #ie this is not for removing a manager node in distributed mode.
    @staticmethod
    def remove_libE_nodes(global_nodelist_in):
        """Any node containing a libensemble task is removed from the global nodelist"""
        libE_nodes_gather = Resources.get_libE_nodes()
        libE_nodes_set = set(libE_nodes_gather)

        #Lose ordering this way
        #global_nodelist_in_set = set(global_nodelist_in)
        #global_nodelist_set = global_nodelist_in_set - libE_nodes_set
        #global_nodelist = list(global_nodelist_set)
        global_nodelist = list(filter(lambda x: x not in libE_nodes_set, global_nodelist_in))
        return global_nodelist

    @staticmethod
    def best_split(a, n):
        """Create the most even split of list a into n parts and return list of lists"""
        k, m = divmod(len(a), n)
        return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


    #Consider changing from static - top_level_dir could be moved to resources attribute - set once on init
    #Also nodelist_env_slurm etc could just use self values.
    @staticmethod
    def get_global_nodelist(rundir=None, central_mode=False,
                            nodelist_env_slurm = None,
                            nodelist_env_cobalt = None):
        """
        Return the list of nodes available to all libEnsemble workers

        If a worker_list file exists this is used, otherwise the environment
        is interrogated for a node list. If a dedicated manager node is used,
        then a worker_list file is recommended.

        In central mode, any node with a libE worker is removed from the list.
        """
        top_level_dir = rundir or os.getcwd()
        nodelist_env_slurm  = nodelist_env_slurm  or Resources.default_nodelist_env_slurm
        nodelist_env_cobalt = nodelist_env_cobalt or Resources.default_nodelist_env_cobalt

        worker_list_file = os.path.join(top_level_dir,'worker_list')

        global_nodelist = []
        if os.path.isfile(worker_list_file):
            logger.debug("worker_list found - getting nodelist from worker_list")
            num_nodes = 0
            with open(worker_list_file,'r') as f:
                for line in f:
                    global_nodelist.append(line.rstrip())
        else:
            logger.debug("No worker_list found - searching for nodelist in environment")
            if os.environ.get(nodelist_env_slurm):
                logger.debug("Slurm env found - getting nodelist from Slurm")
                global_nodelist = Resources.get_slurm_nodelist(nodelist_env_slurm)
            elif os.environ.get(nodelist_env_cobalt):
                logger.debug("Cobalt env found - getting nodelist from Cobalt")
                global_nodelist = Resources.get_cobalt_nodelist(nodelist_env_cobalt)
            else:
                #Assume a standalone machine if all workers on same node - though give warning.
                if len(set(Resources.get_libE_nodes())) == 1:
                    logger.info("Can not find nodelist from environment. Assuming standalone")
                    global_nodelist.append(socket.gethostname())
                else:
                    raise ResourcesException("Error. Can not find nodelist from environment")

        if central_mode:
            global_nodelist = Resources.remove_libE_nodes(global_nodelist)

        if global_nodelist:
            return global_nodelist
        else:
            raise ResourcesException("Error. global_nodelist is empty")


    def get_workers_on_a_node(self):
        """ Returns the number of workers that can be placed on each node"""
        num_workers = self.num_workers
        num_nodes = len(self.global_nodelist)

        #Round up if theres a remainder
        workers_per_node = num_workers//num_nodes + (num_workers % num_nodes > 0)

        return workers_per_node


    def get_available_nodes(self):
        """Returns the list of nodes available to the current worker

        Assumes that self.global_nodelist has been calculated (in __init__).
        Also self.global_nodelist will have already removed non-application nodes
        """

        global_nodelist = self.global_nodelist
        workerID = self.workerID
        num_workers = self.num_workers
        num_nodes = len(global_nodelist)

        # Check if current host in nodelist - if it is then in distributed mode.
        local_host = socket.gethostname()
        distrib_mode = local_host in global_nodelist

        # If not in central mode (ie. in distrib mode) then this host should be in nodelist.
        # Either an error - or set to central mode. Currently make error for transparency
        if not self.central_mode and not distrib_mode:
            raise ResourcesException("Not in central mode, yet worker hostname is not in node list - aborting")

        # If multiple workers per node - create global node_list with N duplicates (for N workers per node)
        sub_node_workers = False
        if num_workers >= num_nodes:
            sub_node_workers = True
            workers_per_node = num_workers//num_nodes
            global_nodelist = list(itertools.chain.from_iterable(itertools.repeat(x, workers_per_node) for x in global_nodelist))

        # Currently require even split for distrib mode - to match machinefile - throw away remainder
        if distrib_mode and not sub_node_workers:
            #Could just read in the libe machinefile and use that - but this should match
            #Alt. create machinefile/host-list with same algorithm as best_split - future soln.
            nodes_per_worker, remainder = divmod(num_nodes,num_workers)
            if remainder != 0:
                # Worker node may not be at head of list after truncation - should perhaps be warning or enforced
                logger.warning("Nodes to workers not evenly distributed. Wasted nodes. {} workers and {} nodes"\
                                .format(num_workers,num_nodes))
                num_nodes = num_nodes - remainder
                global_nodelist = global_nodelist[0:num_nodes]

        # Divide global list between workers
        split_list = list(Resources.best_split(global_nodelist, num_workers))
        #logger.debug("split_list is {}".format(split_list))
        local_nodelist = []
        if workerID is not None:
            local_nodelist = split_list[workerID - 1]
        else:
            # Should always have workerID
            raise ResourcesException("Worker has no workerID - aborting")
            #if distrib_mode:
                #Alternative - find this host ID at head of a sub-list
                #for loc_list in split_list:
                    #if loc_list[0] == local_host:
                        #local_nodelist = loc_list
                        #break
            #else:
                #raise ResourcesException("Not in distrib_mode and no workerID - aborting")

            ##If cannot find - resort to old way - requires even breakdown
            #if not local_nodelist and distrib_mode:
                #logger.debug("Could not find local node at start of a sub-list - trying to find with even split")
                #num_nodes = len(global_nodelist)
                #nodes_per_worker = num_nodes//num_workers
                #node_count = 0
                #found_start = False
                #for node in global_nodelist:
                    #if node_count == nodes_per_worker:
                        #break
                    #if found_start:
                        #node_count += 1
                        #local_nodelist.append(node)
                    #elif (node == local_host):
                        ##distrib_mode = True
                        #found_start = True
                        #local_nodelist.append(node)
                        #node_count = 1

            #if not local_nodelist:
                #raise ResourcesException("Current node {} not in list - no local_nodelist".format(local_host))

        # If in distrib_mode local host must be in local nodelist
        if distrib_mode:
            if local_host not in local_nodelist:
                raise ResourcesException("In distributed mode, but local host is not in local nodelist - aborting")

        logger.debug("local_nodelist is {}".format(local_nodelist))
        return local_nodelist


    @staticmethod
    def _open_binary(fname, **kwargs):
        return open(fname, "rb", **kwargs)


    #@staticmethod ? may use self.physical_cores if already set.
    @staticmethod
    def _cpu_count_physical():
        """Returns the number of physical cores on the node."""
        mapping = {}
        current_info = {}
        with Resources._open_binary('/proc/cpuinfo') as f:
            for line in f:
                line = line.strip().lower()
                if not line:
                    # new section
                    if (b'physical id' in current_info and
                            b'cpu cores' in current_info):
                        mapping[current_info[b'physical id']] = current_info[b'cpu cores']
                    current_info = {}
                else:
                    if (line.startswith(b'physical id') or
                            line.startswith(b'cpu cores')):
                        key, value = line.split(b'\t:', 1)
                        current_info[key] = int(value)

        return sum(mapping.values()) or None


    #@staticmethod ? may use self.num_cores if already set.
    @staticmethod
    def get_cpu_cores(hyperthreads=False):
        """Returns the number of cores on the node.

        If hyperthreads is true, this is the logical cpu cores, else
        the physical cores are returned.

        Note: This returns cores available on the current node - will
        not work for systems of multiple node types
        """
        try:
            import psutil
            if hyperthreads:
                ranks_per_node = psutil.cpu_count()
            else:
                ranks_per_node = psutil.cpu_count(logical=False)
        except ImportError:
            #logger
            if hyperthreads:
                import multiprocessing
                ranks_per_node = multiprocessing.cpu_count()
            else:
                try:
                    ranks_per_node = Resources._cpu_count_physical()
                except:
                    import multiprocessing
                    ranks_per_node = multiprocessing.cpu_count()
                    logger.warning('Could not detect physical cores - Logical cores (with hyperthreads) returned - specify ranks_per_node to override')
        return ranks_per_node #This is ranks available per node
