"""
Module for detecting and returning system resources

"""

#dev notes:
#Currently just a module - may make class if use inhertience - eg detecting resources from
#different schedular environments etc...
#Also currently set to work for distributed MPI mode only - can use workerID to work in central mode
#and alternative worker concurrency modes.

#from mpi4py import MPI
import os
import socket
import logging
import itertools

logger = logging.getLogger(__name__)
formatter = logging.Formatter('%(name)s (%(levelname)s): %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

#For debug messages - uncomment
logger.setLevel(logging.DEBUG)

class ResourcesException(Exception): pass


def get_slurm_nodelist():
    nidlst = []
    NID_LIST_VAR = 'SLURM_NODELIST'
    fullstr = os.environ[NID_LIST_VAR]
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


def get_cobalt_nodelist():
    nidlst = []
    NID_LIST_VAR = 'COBALT_PARTNAME'
    nidstr = os.environ[NID_LIST_VAR]
    nidgroups = nidstr.split(',')
    for nidgroup in nidgroups:
        if (nidgroup.find("-") != -1):
            a, b = nidgroup.split("-", 1)
            a = int(a)
            b = int(b)
            if (a > b):
                tmp = b
                b = a
                a = tmp
            b = b + 1 #need one more for inclusive
        else:
            a = int(nidgroup)
            b = a + 1
        for nid in range(a, b):
            nidlst.append(nid)
    nidlst = sorted(list(set(nidlst)))
    return nidlst
    
def get_num_workers():
    #Will use MPI_MODE from settyings.py global - for now assume using mpi.
    #Or the function may be in some worker_concurrency module
    from mpi4py import MPI
    num_workers = MPI.COMM_WORLD.Get_size() - 1
    return num_workers

def best_split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

#For now via worker file - could use env
def get_available_nodes(rundir=None, workerID=None):
    
    #Or could get from alloc_specs passed by manager - For now - assume all but manager
    if rundir is not None:
        top_level_dir = rundir
    else:
        top_level_dir = os.getcwd()

    #This will only work in distributed worker mode - alt work out from workerID
    local_host = socket.gethostname()
    worker_list_file = os.path.join(top_level_dir,'worker_list')
    
    if os.path.isfile(worker_list_file):
        logger.debug("worker_list found - getting nodelist from worker_list")
        num_nodes = 0
        global_nodelist = []
        with open(worker_list_file,'r') as f:
            for line in f:                
                global_nodelist.append(line.rstrip())
    else: 
        #Need a way to know if using a manager node though - this will give simply all nodes.
        logger.debug("No worker_list found - searching for nodelist in environment")
        if os.environ.get('SLURM_NODELIST'):
            logger.debug("Slurm env found - getting nodelist from Slurm")
            global_nodelist = get_slurm_nodelist()
        elif os.environ.get('COBALT_PARTNAME'):
            logger.debug("Cobalt env found - getting nodelist from Cobalt")
            global_nodelist = get_cobalt_nodelist()
        else:
            raise ResourcesException("Error. Can not find nodelist from environment")
    
    logger.debug("global_nodelist is {}".format(global_nodelist)) #tmp
    
    #But if use env nodelist then even in central mode - it will be in list - must exclude control nodes...
    distrib_mode = False
    for node in global_nodelist:
        if (node == local_host):
            distrib_mode = True
            break
 
    num_workers = get_num_workers()
    
    #Currently require even split for distrib mode - to match machinefile
    num_nodes = len(global_nodelist)
    
    sub_node_workers = False
    if num_workers >= num_nodes:
        sub_node_workers = True
        workers_per_node = num_workers//num_nodes
        global_nodelist = list(itertools.chain.from_iterable(itertools.repeat(x, workers_per_node) for x in global_nodelist))
    
    if distrib_mode and not sub_node_workers:
        #Maybe should just read in the libe machinefile and use that - but this should match
        #Alt. create machine file with same algorithm as best_split
        nodes_per_worker, remainder = divmod(num_nodes,num_workers)
        if remainder != 0:
            logger.warning("Nodes to workers not evenly distributed. Wasted nodes. {} workers and {} nodes"\
                            .format(num_workers,num_nodes))
            num_nodes = num_nodes - remainder
            global_nodelist = global_nodelist[0:num_nodes]
        
    split_list = list(best_split(global_nodelist, num_workers))
    
    logger.debug("split_list is {}".format(split_list)) #tmp
    local_nodelist = []
    if workerID is not None:
        local_nodelist = split_list[workerID - 1]
    else:
        if distrib_mode:
            for loc_list in split_list:
                if loc_list[0] == local_host:
                    local_nodelist = loc_list
                    break
        else:
            raise ResourcesException("Not in distrib_mode and no workerID - aborting")
        
        if not local_nodelist and distrib_mode:
            logger.debug("Could not find local node at start of a sub-list - trying to find with even split")
            #Resort to old way - requires even breakdown
            num_nodes = len(global_nodelist)
            nodes_per_worker = num_nodes//num_workers
            node_count = 0  
            found_start = False  
            for node in global_nodelist:
                if node_count == nodes_per_worker:
                    break
                if found_start:
                    node_count += 1
                    local_nodelist.append(node)
                elif (node == local_host):
                    #distrib_mode = True
                    found_start = True
                    local_nodelist.append(node)
                    node_count = 1
        
        if not local_nodelist:
            raise ResourcesException("Current node {} not in list - no local_nodelist".format(local_host))
        
            #raise ResourcesException("Current node {} not in list - this only  works in distrib mode - aborting".format(local_host)) 
        
    logger.debug("local_nodelist is {}".format(local_nodelist))    
    return local_nodelist


def _open_binary(fname, **kwargs):
    return open(fname, "rb", **kwargs)

def _cpu_count_physical():
    """Return the number of physical cores in the system."""
    mapping = {}
    current_info = {}
    with _open_binary('/proc/cpuinfo') as f:        
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

def get_cpu_cores(hyperthreads=False):
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
                ranks_per_node = _cpu_count_physical()
            except:
                import multiprocessing
                ranks_per_node = multiprocessing.cpu_count()                            
                logger.warning('Could not detect physical cores - Logical cores (with hyperthreads) returned - specify ranks_per_node to override')
    return ranks_per_node

  
