import os
import sys

#@staticmethod
def _open_binary(fname, **kwargs):
    return open(fname, "rb", **kwargs)


#@staticmethod ? may use self.physical_cores if already set.
#@staticmethod
def _cpu_count_physical():
    """Returns the number of physical cores on the node."""
    mapping = {}
    current_info = {}
    #import pdb;pdb.set_trace()
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


#@staticmethod
def get_cpu_cores(hyperthreads=False):
    """Returns the number of cores on the node.

    If hyperthreads is true, this is the logical cpu cores, else
    the physical cores are returned.

    Note: This returns cores available on the current node - will
    not work for systems of multiple node types
    """
    try:
        import psutil
        ranks_per_node = psutil.cpu_count(logical=hyperthreads)
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
                #logger.warning('Could not detect physical cores - Logical cores (with hyperthreads) returned - specify ranks_per_node to override')
                #tmp
                print('Warning: Could not detect physical cores - Logical cores (with hyperthreads) returned - specify ranks_per_node to override')
    return ranks_per_node #This is ranks available per node


def _get_local_cpu_resources():
    logical_cores_avail_per_node = get_cpu_cores(hyperthreads=True)
    physical_cores_avail_per_node = get_cpu_cores(hyperthreads=False) 
    return (logical_cores_avail_per_node, physical_cores_avail_per_node)


def _print_local_cpu_resources():
    import sys
    cores_info = _get_local_cpu_resources()
    print(cores_info[0], cores_info[1])
    sys.stdout.flush()


def _get_remote_cpu_resources(launcher):
    import subprocess
    #p = subprocess.check_call([launcher, 'python', os.path.basename(__file__)], stdout=subprocess.PIPE,  stderr=subprocess.PIPE)
    #output = subprocess.check_output([launcher, 'python', os.path.basename(__file__)])
    #subprocess.check_call([launcher, 'python', os.path.basename(__file__)])
    #subprocess.check_call([launcher, 'python', __file__])
    output = subprocess.check_output([launcher, 'python', __file__])
    
    #stdout, _ = p.communicate()
    print('this line here', output)
    print('this line here decoded', output.decode())
    return output.decode()


def get_sub_node_resources(launcher=None):
    remote = True if launcher else False
    #****************note - add special way of doing it one summit - eg. if jsrun and LSB_HOSTS or maybe just if LSB_HOSTS....
    #On mpi vrsion this will result in every worker launching a job - that might be good if launch onto its own node but would be diff
    #to multiproc version - might be better if mpi4py version does just launch from master and shares???
    #Or it could be we should change to do this on worker phase instead - then they all launch where they will launch - but think thats next iteration.
    if remote:
        cores_info_str = _get_remote_cpu_resources(launcher=launcher)
        cores_log, cores_phy, *_ = cores_info_str.split()
        cores_info = (int(cores_log), int(cores_phy))
    else: 
        cores_info = _get_local_cpu_resources()
    return (cores_info)


if __name__ == "__main__":
    _print_local_cpu_resources()
