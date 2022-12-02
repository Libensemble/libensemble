"""
This module for detects and returns intranode resources

"""

import os
import psutil
import logging
import collections
from libensemble.resources.gpu_detect import get_num_gpus, get_gpus_from_env

logger = logging.getLogger(__name__)

REMOTE_LAUNCH_LIST = ["aprun", "jsrun", "srun"]  # Move to feature of mpi_runner


def get_cpu_cores(hyperthreads=False):
    """Returns the number of cores on the node.

    If hyperthreads is true, this is the logical CPU cores; else
    the physical cores are returned.

    Note: This returns cores available on the current node. It will
    not work for systems of multiple node types
    """
    return psutil.cpu_count(logical=hyperthreads)  # This is ranks available per node


# SH TODO: Rename for gpu resources also
def _get_local_cpu_resources():
    """Returns logical and physical cores on the local node"""
    logical_cores_avail_per_node = get_cpu_cores(hyperthreads=True)
    physical_cores_avail_per_node = get_cpu_cores(hyperthreads=False)
    gpus_avail_per_node = get_num_gpus()
    return (physical_cores_avail_per_node, logical_cores_avail_per_node, gpus_avail_per_node)


def _print_local_cpu_resources():
    """Prints logical and physical cores on the local node"""
    cores_info = _get_local_cpu_resources()
    print(cores_info[0], cores_info[1], cores_info[2], flush=True)


def _get_remote_cpu_resources(launcher):
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


def get_sub_node_resources(launcher=None, remote_mode=False, env_resources=None):
    """Returns logical and physical cores per node as a tuple"""
    remote_detection = False
    cores_info = None
    num_gpus = 0

    # In some cases may be worth getting GPUs from env even when local
    num_gpus = get_gpus_from_env(env_resources=env_resources)
    # print(f"{num_gpus=}")

    if remote_mode:
        # May be unnecessary condition
        print("launcher is", launcher)
        if launcher in REMOTE_LAUNCH_LIST:
            cores_info = _get_cpu_resources_from_env(env_resources=env_resources)
            if cores_info and num_gpus:
                return cores_info + num_gpus
            remote_detection = True  # Cannot obtain from environment

    if remote_detection:
        cores_info_str = _get_remote_cpu_resources(launcher=launcher)
        cores_phy, cores_log, num_gpus, *_ = cores_info_str.split()

        if cores_info:
            # If got from environment, do not replace
            cores_info = cores_info + int(num_gpus)

            # testing if match
            if cores_info[0] == cores_phy and cores_info[1] == cores_log:
                print("Env and detected match")
            else:
                print(f"Env and detected DONT match env: {cores_info} detect: {cores_phy} {cores_log}")
        else:
            cores_info = (int(cores_phy), int(cores_log), int(num_gpus))
    else:
        cores_info = _get_local_cpu_resources()

        if cores_info[2] == 0:
            if num_gpus is not None and num_gpus > 0:
                cores_info = (cores_info[0], cores_info[1], int(num_gpus))

    return cores_info


if __name__ == "__main__":
    _print_local_cpu_resources()
