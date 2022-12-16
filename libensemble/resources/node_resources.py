"""
This module for detects and returns intranode resources

"""

import os
import psutil
import logging
import subprocess
import collections
from libensemble.resources.gpu_detect import get_num_gpus, get_gpus_from_env
from libensemble.resources.platforms import detect_systems

logger = logging.getLogger(__name__)

REMOTE_LAUNCH_LIST = ["aprun", "jsrun", "srun"]  # Move to feature of mpi_runner


def known_sys_check():
    """Detect system domain name and return any CPU / GPU data found as a tuple

    If the system is found, will return an int tuple of (cores, logical cores, gpus),
    where any missing entry will be None.

    If system is not found in known list, then returns None.

    """

    try:
        domain_name = subprocess.check_output(["hostname", "-d"]).decode().rstrip()

        print("domain", domain_name)

        if isinstance(domain_name, str) and len(domain_name):
            system_name = detect_systems.get(domain_name)

            print("system", system_name)

            if system_name is None:
                print("System not in known list", domain_name)
                return None

            physical_cores = system_name.get("cores_per_node")
            logical_cores = system_name.get("logical_cores_per_node")
            gpu_count = system_name.get("gpus_per_node")

            return (physical_cores, logical_cores, gpu_count)
        else:
            print("Could not get valid domain name")
            return None

    except Exception:
        print("Known system detection failed")
        return None


def get_cpu_cores(hyperthreads=False):
    """Returns the number of cores on the node.

    If hyperthreads is true, this is the logical CPU cores; else
    the physical cores are returned.

    Note: This returns cores available on the current node. It will
    not work for systems of multiple node types
    """
    return psutil.cpu_count(logical=hyperthreads)  # This is ranks available per node


def _get_local_resources():
    """Returns logical and physical cores on the local node"""
    physical_cores = get_cpu_cores(hyperthreads=False)
    logical_cores = get_cpu_cores(hyperthreads=True)
    num_gpus = get_num_gpus()
    return (physical_cores, logical_cores, num_gpus)


def _print_local_resources():
    """Prints logical and physical cores on the local node"""
    cores_info = _get_local_resources()
    print(cores_info[0], cores_info[1], cores_info[2], flush=True)


def _get_remote_resources(launcher):
    """Launches a probe job to obtain logical and physical cores on remote node"""
    import subprocess

    output = subprocess.check_output([launcher, "python", __file__])
    return output.decode()


def _get_cpu_resources_from_env(env_resources=None):
    """Returns logical and physical cores per node by querying environment or None"""
    if not env_resources:
        return None

    found_count = False
    if env_resources.nodelists["LSF"] in os.environ:
        full_list = os.environ.get(env_resources.nodelists["LSF"]).split()
        nodes = [n for n in full_list if "batch" not in n]
        counter = list(collections.Counter(nodes).values())
        found_count = True
    elif env_resources.nodelists["LSF_shortform"] in os.environ:
        full_list = os.environ.get(env_resources.nodelists["LSF_shortform"]).split()
        iter_list = iter(full_list)
        zipped_list = list(zip(iter_list, iter_list))
        nodes_with_count = [n for n in zipped_list if "batch" not in n[0]]
        counter = [int(n[1]) for n in nodes_with_count]
        found_count = True

    if found_count:
        # Check all nodes have equal cores -  Not doing for other methods currently.
        if len(set(counter)) != 1:
            logger.warning(f"Detected compute nodes have different core counts: {set(counter)}")

        physical_cores_avail_per_node = counter[0]
        logical_cores_avail_per_node = counter[0]  # How to get SMT threads remotely
        logger.warning("SMT currently not detected, returning physical cores only. Specify procs_per_node to override")
        return (physical_cores_avail_per_node, logical_cores_avail_per_node)
    else:
        return None


# tuple or change to list? or dict? - do for all 3 funcs
def _cpu_info_complete(cores_info):
    """Returns true if cpu tuple values have an integer value, else False"""

    if cores_info is None:
        return False

    for val in cores_info[:2]:
        if not isinstance(val, int):
            return False
    return True


def _gpu_info_complete(cores_info):
    """Returns true if gpu tuple values have an integer value, else False"""

    if cores_info is None:
        return False

    for val in cores_info[2:]:
        if not isinstance(val, int):
            return False
    return True


def _complete_set(cores_info):
    """Returns True if all tuple entries have an integer value, else False"""

    if cores_info is None:
        return False

    for val in cores_info:
        if not isinstance(val, int):
            return False
    return True


def _update_values(cores_info, cores_info_updates):
    """Update values that are not set"""
    if not _cpu_info_complete(cores_info):
        cores_info[:2] = list(cores_info_updates[:2] or [None, None])
    if not _gpu_info_complete(cores_info):
        cores_info[2] = cores_info_updates[2]
    return cores_info


def _update_from_str(cores_info, cores_info_str):
    """Update values that are not set from a string"""
    cores_phy, cores_log, num_gpus, *_ = cores_info_str.split()

    if not _cpu_info_complete(cores_info):
        try:
            cores_info[:2] = [int(cores_phy), int(cores_log)]
        except ValueError:
            pass

    if not _gpu_info_complete(cores_info):
        try:
            cores_info[2] = int(num_gpus)
        except ValueError:
            pass

    return cores_info


def get_sub_node_resources(launcher=None, remote_mode=False, env_resources=None):
    """Returns logical and physical cores and GPUs per node as a tuple

    First checks for known system values, then for environment values, and finally
    for detected values. If remote_mode is True, then detection launches a job
    via the MPI launcher.

    Any value that is already valid, is not overwritten by successive stages.

    """

    # SH TODO: Could receive if cpu/gpu is already set by resource_info so don't need to get.
    #          Though will be not overwrite anyway.

    remote_detection = False
    cores_info = [None, None, None]
    num_gpus = 0

    print("type is", type(cores_info))

    # Check for known system
    cores_info = known_sys_check() or [None, None, None]
    print("known system ", cores_info)
    if _complete_set(cores_info):
        print("known system provided all info")
        return tuple(cores_info)

    # Check environment
    if not _cpu_info_complete(cores_info):
        cores_info[:2] = list(_get_cpu_resources_from_env(env_resources=env_resources) or [None, None])
    if not _gpu_info_complete(cores_info):
        cores_info[2] = get_gpus_from_env(env_resources=env_resources)
    if _complete_set(cores_info):
        print("known system and/or env provided all info")
        return tuple(cores_info)

    # Detection of cpu/gpu resources
    # If remote then launch probe, else detect locally
    if remote_mode:
        cores_info_str = _get_remote_resources(launcher=launcher)
        cores_info = _update_from_str(cores_info, cores_info_str)
        print("called remote detection")
    else:
        cores_info_detected = _get_local_resources()
        cores_info = _update_values(cores_info, cores_info_detected)
        print("called local detection")

    # Convert Nones to zeros and return
    cores_info = [0 if v is None else v for v in cores_info]
    return tuple(cores_info)


if __name__ == "__main__":
    _print_local_resources()
