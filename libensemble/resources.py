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
    from mpi4py import MPI
    num_workers = MPI.COMM_WORLD.Get_size() - 1
    return num_workers


#For now via worker file - could use env
def get_available_nodes(rundir=None):
    
    #Or could get from alloc_specs passed by manager - For now - assume all but manager
    if rundir is not None:
        top_level_dir = rundir
    else:
        top_level_dir = os.getcwd()
        
    num_workers = get_num_workers()
    
    #This will only work in distributed worker mode - alt work out from workerID
    local_host = socket.gethostname()
           
    # If worker_list exists use that (else may be able to get from slurm/cobalt etc...)
    local_nodelist = []
    distrib_mode = False #To start with only doing for distrib_mode=True (gets changed when detects below)
    worker_list_file = os.path.join(top_level_dir,'worker_list')
    #import pdb; pdb.set_trace()
    
    if os.path.isfile(worker_list_file):
        logger.debug("worker_list found - getting nodelist from worker_list")
        num_nodes = 0
        with open(worker_list_file,'r') as f:
            for line in f:
                num_nodes += 1
                if (line.rstrip() == local_host):
                    #This worker is on a worker node - distributed mode is true
                    distrib_mode = True
                    #head_node_index = num_nodes
            if not distrib_mode:
                raise ResourcesException("Current node {} not in list - this only  works in distrib mode - aborting".format(local_host)) #Alt - if not in distrib mode - use workerID
            f.seek(0)
            nodes_per_worker = num_nodes/num_workers
            if nodes_per_worker <= 1:
                local_nodelist.append(local_host)
            else:
                #One loop
                node_count = 0  
                found_start = False
                for line in f:
                    #node_count += 1
                    if node_count == nodes_per_worker:
                        break                    
                    if found_start:
                        node_count += 1
                        local_nodelist.append(line.rstrip())
                    elif (line.rstrip() == local_host):
                        found_start = True
                        local_nodelist.append(line.rstrip())
                        node_count = 1
                #Two loops
                #for line in f:
                    ##node_count += 1
                    #if (line.rstrip() == local_host):
                        ##break #Found start
                        #local_nodelist.append(line.rstrip())
                        #node_count = 1
                        #break
                #for line in f:
                    #if node_count == nodes_per_worker:
                        #break
                    #node_count += 1
                    #local_nodelist.append(line.rstrip())
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
        logger.debug("global_nodelist is {}".format(global_nodelist))    
        num_nodes = len(global_nodelist)
        nodes_per_worker = num_nodes/num_workers
        #if nodes_per_worker <= 1:
            ##distrib_mode assumed....
            #local_nodelist.append(local_host) 
        #else:
        
        #One loop
        node_count = 0  
        found_start = False        
        for node in global_nodelist:
            if node_count == nodes_per_worker:
                break
            if found_start:
                node_count += 1
                local_nodelist.append(node)
            elif (node == local_host):
                distrib_mode = True
                found_start = True
                local_nodelist.append(node)
                node_count = 1
        if not distrib_mode:
            raise ResourcesException("Current node {} not in list - this only  works in distrib mode - aborting".format(local_host)) 
        
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

  
