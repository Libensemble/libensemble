"""Module for platform specification

This module defines the Platform class which can be used to determine a platform
(computing system) attributes. Many known systems are provided.

These can be specified by the libE_specs options ``platform_specs`` (recommended).
It may also be specified, for known systems, via a string in the ``platform``
option or the environment variable ``LIBE_PLATFORM``.
"""

import logging
import os
import subprocess

from pydantic import BaseModel

from libensemble.utils.misc import specs_dump

logger = logging.getLogger(__name__)
# To change logging level for just this module
# logger.setLevel(logging.DEBUG)


class PlatformException(Exception):
    """Platform module exception"""


class Platform(BaseModel):
    """
    Class to define attributes of a target platform.

    All are optional, and any not defined will be determined by libEnsemble's auto-detection.
    """

    mpi_runner: str | None = None
    """MPI runner: One of ``"mpich"``, ``"openmpi"``, ``"aprun"``,
    ``"srun"``, ``"jsrun"``, ``"msmpi"``, ``"custom"`` """

    runner_name: str | None = None
    """Literal string of MPI runner command. Only needed if different to the default

    Note that ``"mpich"`` and ``"openmpi"`` runners have the default command ``"mpirun"``
    """
    cores_per_node: int | None = None
    """Number of physical CPU cores on a compute node of the platform"""

    logical_cores_per_node: int | None = None
    """Number of logical CPU cores on a compute node of the platform"""

    gpus_per_node: int | None = None
    """Number of GPU devices on a compute node of the platform"""

    tiles_per_gpu: int | None = None
    """Number of tiles on a GPU"""

    gpu_setting_type: str | None = None

    """ How GPUs will be assigned.

    Must take one of the following string options.

    - ``"runner_default"``:       Use default setting for MPI runner (same as if not set).
    - ``"env"``:                  Use an environment variable (comma-separated list of slots)
    - ``"option_gpus_per_node"``: Expresses GPUs per node on MPI runner command line.
    - ``"option_gpus_per_task"``: Expresses GPUs per task on MPI runner command line.

    With the exception of "runner_default", the :attr:`gpu_setting_name`
    attribute is also required when this attribute is set.

    If "gpu_setting_type" is not provided (same as ``runner_default``) and the
    MPI runner does not have a default GPU setting in libEnsemble, and no other
    information is present, then the environment variable ``CUDA_VISIBLE_DEVICES``
    is used.

    Examples:

    Use environment variable ROCR_VISIBLE_DEVICES to assign GPUs.

    .. code-block:: python

        "gpu_setting_type" = "env"
        "gpu_setting_name" = "ROCR_VISIBLE_DEVICES"

    Use command line option ``--gpus-per-node``

    .. code-block:: python

        "gpu_setting_type" = "option_gpus_per_node"
        "gpu_setting_name" = "--gpus-per-node"

    """

    gpu_setting_name: str | None = None
    """Name of GPU setting

    See :attr:`gpu_setting_type` for more details.

    """

    gpu_env_fallback: str | None = None
    """GPU fallback environment setting if not using an MPI runner.

    For example:

    .. code-block:: python

        "gpu_setting_type" = "runner_default"
        "gpu_env_fallback" = "ROCR_VISIBLE_DEVICES"

    This example will use the MPI runner default settings when using an MPI runner, but
    will otherwise use ROCR_VISIBLE_DEVICES (e.g., if setting via function set_env_to_gpus).

    If this is not set, the default is "CUDA_VISIBLE_DEVICES".

    """

    scheduler_match_slots: bool | None = True
    """
    Whether the libEnsemble resource scheduler should only assign matching slots when
    there are multiple (partial) nodes assigned to a sim function.

    Defaults to ``True``, within libEnsemble.

    Useful if setting an environment variable such as ``CUDA_VISIBLE_DEVICES``, where
    the value should match on each node of an MPI run (choose **True**).

    When using command-line options just as ``--gpus-per-node``, which allow the system's
    application-level scheduler to manage GPUs, then ``match_slots`` can be **False**
    (allowing for more efficient scheduling when MPI runs cross nodes).
    """


class Aurora(Platform):
    mpi_runner: str = "mpich"
    runner_name: str = "mpiexec"
    cores_per_node: int = 104
    logical_cores_per_node: int = 208
    gpus_per_node: int = 6
    tiles_per_gpu: int = 2
    gpu_setting_type: str = "env"
    gpu_setting_name: str = "ZE_AFFINITY_MASK"
    scheduler_match_slots: bool = True


# On SLURM systems, let srun assign free GPUs on the node
class Frontier(Platform):
    mpi_runner: str = "srun"
    cores_per_node: int = 64
    logical_cores_per_node: int = 128
    gpus_per_node: int = 8
    gpu_setting_type: str = "runner_default"
    gpu_env_fallback: str = "ROCR_VISIBLE_DEVICES"
    scheduler_match_slots: bool = False


# Example of a ROCM system
class GenericROCm(Platform):
    mpi_runner: str = "mpich"
    gpu_setting_type: str = "env"
    gpu_setting_name: str = "ROCR_VISIBLE_DEVICES"
    scheduler_match_slots: bool = True


class Lumi(Platform):
    mpi_runner: str = "srun"
    cores_per_node: int = 64
    logical_cores_per_node: int = 128


class LumiGPU(Lumi):
    gpus_per_node: int = 8
    gpu_setting_type: str = "env"
    gpu_setting_name: str = "ROCR_VISIBLE_DEVICES"
    scheduler_match_slots: bool = True


class Perlmutter(Platform):
    mpi_runner: str = "srun"


class PerlmutterCPU(Perlmutter):
    cores_per_node: int = 128
    logical_cores_per_node: int = 256
    gpus_per_node: int = 0


class PerlmutterGPU(Perlmutter):
    cores_per_node: int = 64
    logical_cores_per_node: int = 128
    gpus_per_node: int = 4
    gpu_setting_type: str = "runner_default"
    gpu_env_fallback: str = "CUDA_VISIBLE_DEVICES"
    scheduler_match_slots: bool = False


class Polaris(Platform):
    mpi_runner: str = "mpich"
    runner_name: str = "mpiexec"
    cores_per_node: int = 32
    logical_cores_per_node: int = 64
    gpus_per_node: int = 4
    gpu_setting_type: str = "runner_default"
    gpu_env_fallback: str = "CUDA_VISIBLE_DEVICES"
    scheduler_match_slots: bool = True


class Summit(Platform):
    mpi_runner: str = "jsrun"
    cores_per_node: int = 42
    logical_cores_per_node: int = 168
    gpus_per_node: int = 6
    gpu_setting_type: str = "option_gpus_per_task"
    gpu_setting_name: str = "-g"
    scheduler_match_slots: bool = False


class Known_platforms(BaseModel):
    """A list of platforms with known configurations.

    There are three ways to specify a known system:

    .. tab-set::

        .. tab-item:: ["platform_specs"]

            .. code-block:: python

                from libensemble.resources.platforms import PerlmutterGPU

                libE_specs["platform_specs"] = PerlmutterGPU()

        .. tab-item:: ["platform"]

            .. code-block:: python

                libE_specs["platform"] = "perlmutter_g"

        .. tab-item:: export LIBE_PLATFORM

            On command-line or batch submission script:

            .. code-block:: shell

                export LIBE_PLATFORM="perlmutter_g"


    If the platform is not specified, libEnsemble will attempt to detect known
    platforms (this is not guaranteed).

    **Note**: libEnsemble should work on any platform, and detects most
    system configurations correctly. These options are helpful for optimization and
    where auto-detection encounters ambiguity or an unknown feature.
    """

    aurora: Aurora = Aurora()
    generic_rocm: GenericROCm = GenericROCm()
    frontier: Frontier = Frontier()
    lumi: Lumi = Lumi()
    lumi_g: LumiGPU = LumiGPU()
    perlmutter: Perlmutter = Perlmutter()
    perlmutter_c: PerlmutterCPU = PerlmutterCPU()
    perlmutter_g: PerlmutterGPU = PerlmutterGPU()
    polaris: Polaris = Polaris()
    summit: Summit = Summit()


# Dictionary of known systems (or system partitions) detectable by domain name
detect_systems = {
    "frontier.olcf.ornl.gov": "frontier",
    "hostmgmt.cm.aurora.alcf.anl.gov": "aurora",
    "hsn.cm.polaris.alcf.anl.gov": "polaris",
    "summit.olcf.ornl.gov": "summit",  # Need to detect gpu count
}


def known_envs():
    """Detect system by environment variables"""
    name = None
    if os.environ.get("NERSC_HOST") == "perlmutter":
        partition = os.environ.get("SLURM_JOB_PARTITION")
        if partition:
            if "gpu_" in partition:
                name = "perlmutter_g"
            else:
                name = "perlmutter_c"
        else:
            name = "perlmutter"
            logger.manager_warning("Perlmutter detected, but no compute partition detected. Are you on login nodes?")
    if os.environ.get("SLURM_CLUSTER_NAME") == "lumi":
        partition = os.environ.get("SLURM_JOB_PARTITION")
        if not partition:
            logger.manager_warning("LUMI detected, but no compute partition detected. Are you on login nodes?")
        if partition and partition.endswith("-g"):
            name = "lumi_g"
        else:
            name = "lumi"
    return name


def known_system_detect(cmd="hostname -d"):
    """Detect known systems

    This function attempts to detect if on a known system, and
    returns the name of the system as a string.
    """
    run_cmd = cmd.split()
    name = None
    try:
        domain_name = subprocess.check_output(run_cmd).decode().rstrip()
        name = detect_systems[domain_name]
    except Exception:
        name = known_envs()
    return name


def get_platform(libE_specs):
    """Return platform as a dictionary from relevant libE_specs option.

    For internal use, return a platform as a dictionary from either
    platform name or platform_specs or auto-detection.

    If a platform is given or detected and platform_spec fields are present,
    any fields in platform_specs are added to or overwrite fields in the known
    platform.
    """
    platform_info = {}
    name = libE_specs.get("platform") or os.environ.get("LIBE_PLATFORM") or known_system_detect()
    if name:
        try:
            known_platforms = specs_dump(Known_platforms(), exclude_none=True)
            platform_info = known_platforms[name]
        except KeyError:
            raise PlatformException(f"Error. Unknown platform requested {name}")

        # Add/overwrite any fields from a platform_specs
        platform_specs = libE_specs.get("platform_specs")
        if platform_specs:
            for k, v in platform_specs.items():
                platform_info[k] = v
    elif libE_specs.get("platform_specs"):
        platform_info = libE_specs["platform_specs"]
        platform_info = {k: v for k, v in platform_info.items() if v is not None}
    return platform_info
