"""
This module for detects and returns intranode resources

"""

import collections
import logging
import os

import psutil

from libensemble.resources.gpu_detect import get_gpus_from_env, get_num_gpus

logger = logging.getLogger(__name__)

REMOTE_LAUNCH_LIST = ["aprun", "jsrun", "srun"]  # Move to feature of mpi_runner


def get_cpu_cores(hyperthreads: bool = False) -> int:
    """Returns the number of cores on the node.

    If hyperthreads is true, this is the logical CPU cores; else
    the physical cores are returned.

    Note: This returns cores available on the current node. It will
    not work for systems of multiple node types
    """
    return psutil.cpu_count(logical=hyperthreads)  # This is ranks available per node


def _get_local_resources() -> tuple[int, int, int]:
    """Returns logical and physical cores and GPUs on the local node"""
    physical_cores = get_cpu_cores(hyperthreads=False)
    logical_cores = get_cpu_cores(hyperthreads=True)
    num_gpus = get_num_gpus()
    return (physical_cores, logical_cores, num_gpus)


def _print_local_resources():
    """Prints logical and physical cores and GPUs on the local node"""
    cores_info = _get_local_resources()
    print(cores_info[0], cores_info[1], cores_info[2], flush=True)


def _get_remote_resources(launcher):
    """Launches a probe job to obtain logical and physical cores on remote node"""
    import subprocess

    output = subprocess.check_output([launcher, "python", __file__])
    return output.decode()


def _get_cpu_resources_from_env(env_resources=None) -> tuple[int, int] | None:
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

        physical_cores_avail_per_node = min(counter)
        logical_cores_avail_per_node = min(counter)  # How to get SMT threads remotely
        logger.warning("SMT currently not detected, returning physical cores only. Specify procs_per_node to override")
        return (physical_cores_avail_per_node, logical_cores_avail_per_node)
    else:
        return None


def _cpu_info_complete(cores_info):
    """Returns true if cpu tuple/list entries have an integer value, else False"""

    for val in cores_info[:2]:
        if not isinstance(val, int):
            return False
    return True


def _gpu_info_complete(cores_info):
    """Returns true if gpu tuple/list entries have an integer value, else False"""

    for val in cores_info[2:]:
        if not isinstance(val, int):
            return False
    return True


def _complete_set(cores_info):
    """Returns True if all tuple/list entries have an integer value, else False"""

    for val in cores_info:
        if not isinstance(val, int):
            return False
    return True


def _update_values(cores_info, cores_info_updates):
    """Update list entries in cores_info that are not set

    Both CPU core entries will get overwritten if one is not set
    """
    if not _cpu_info_complete(cores_info):
        cores_info[:2] = list(cores_info_updates[:2] or [None, None])
    if not _gpu_info_complete(cores_info):
        cores_info[2] = cores_info_updates[2]
    return cores_info


def _update_from_str(cores_info, cores_info_str):
    """Update unset entries in cores_info from a string

    Both CPU core entries will get overwritten if one is not set
    """
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


def get_sub_node_resources(
    launcher: str | None = None, remote_mode: bool = False, env_resources=None
) -> tuple[int, int, int]:
    """Returns logical and physical cores and GPUs per node as a tuple

    First checks for environment values, and and then for detected values.
    If remote_mode is True, then detection launches a job via the MPI launcher.

    Any value that is already valid, is not overwritten by successive stages.

    """
    cores_info = [None, None, None]

    # Check environment
    cores_info[:2] = list(_get_cpu_resources_from_env(env_resources=env_resources) or [None, None])
    cores_info[2] = get_gpus_from_env(env_resources=env_resources)
    if _complete_set(cores_info):
        return tuple(cores_info)

    # Detection of cpu/gpu resources
    # If remote then launch probe, else detect locally
    if remote_mode:
        cores_info_str = _get_remote_resources(launcher=launcher)
        cores_info = _update_from_str(cores_info, cores_info_str)
    else:
        cores_info_detected = _get_local_resources()
        cores_info = _update_values(cores_info, cores_info_detected)

    # Convert Nones to zeros and return
    cores_info = [0 if v is None else v for v in cores_info]
    return tuple(cores_info)


if __name__ == "__main__":
    _print_local_resources()
