import os
import logging
import collections

logger = logging.getLogger(__name__)

REMOTE_LAUNCH_LIST = ['aprun', 'jsrun']


def _open_binary(fname, **kwargs):
    return open(fname, "rb", **kwargs)


def _cpu_count_physical():
    """Returns the number of physical cores on the node."""
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
        # logger
        if hyperthreads:
            import multiprocessing
            ranks_per_node = multiprocessing.cpu_count()
        else:
            try:
                ranks_per_node = _cpu_count_physical()
            except Exception as e:
                logger.warning("Could not detect physical cores - Logical cores (with hyperthreads) returned - specify ranks_per_node to override. Exception {}".format(e))
                import multiprocessing
                ranks_per_node = multiprocessing.cpu_count()
    return ranks_per_node  # This is ranks available per node


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
    output = subprocess.check_output([launcher, 'python', __file__])
    return output.decode()


def _get_cpu_resources_from_env():
    # May create env resources module to share between other resources modules or send arg.
    if os.environ['LSB_HOSTS']:
        full_list = os.environ['LSB_HOSTS'].split()
        nodes = [n for n in full_list if 'batch' not in n]
        counter = list(collections.Counter(nodes).values())

        # Check all nodes have equal cores -  Not doing for other methods currently.
        if len(set(counter)) != 1:
            logger.warning("Detected compute nodes have different core counts: {}".format(set(counter)))

        physical_cores_avail_per_node = counter[0]
        logical_cores_avail_per_node = counter[0]  # How to get SMT threads remotely
        logger.warning("SMT currently not detected, returning physical cores only. Specify ranks_per_node to override")
        return (logical_cores_avail_per_node, physical_cores_avail_per_node)
    else:
        return None


def get_sub_node_resources(launcher=None, remote_mode=False):
    """Retruns logical and physical cores per node as a tuple"""
    remote_detection = False
    if remote_mode:
        # May be unnecessary condition
        if launcher in REMOTE_LAUNCH_LIST:
            cores_info = _get_cpu_resources_from_env()
            if cores_info:
                return (cores_info)
            remote_detection = True  # Cannot obtain from environment

    if remote_detection:
        cores_info_str = _get_remote_cpu_resources(launcher=launcher)
        cores_log, cores_phy, *_ = cores_info_str.split()
        cores_info = (int(cores_log), int(cores_phy))
    else:
        cores_info = _get_local_cpu_resources()
    return (cores_info)


if __name__ == "__main__":
    _print_local_cpu_resources()
