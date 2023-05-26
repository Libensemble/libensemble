"""Module for platform specification

This module defines the Platform class which can be used to determine a platform
(computing system) attributes. A number of known systems are provided.

These can be specified by the libE_specs options ``platform_specs`` (recommended).
If may also be specified, for known systems, via a string in the ``platform``
option or the environment variable ``LIBE_PLATFORM``.
"""

import os
import subprocess
from typing import Optional

from pydantic import BaseConfig, BaseModel, root_validator, validator

BaseConfig.validate_assignment = True


class PlatformException(Exception):
    """Platform module exception"""


class Platform(BaseModel):
    """
    Class to define attributes of a target platform.

    All are optional, and any not defined will be determined by libEnsemble's auto-detection.
    """

    mpi_runner: Optional[str]
    """MPI runner: One of ``"mpich"``, ``"openmpi"``, ``"aprun"``,
    ``"srun"``, ``"jsrun"``, ``"msmpi"``, ``"custom"`` """

    runner_name: Optional[str]
    """Literal string of MPI runner command. Only needed if different to the default

    Note that ``"mpich"`` and ``"openmpi"`` runners have the default command ``"mpirun"``
    """
    cores_per_node: Optional[int]
    """Number of physical CPU cores on a compute node of the platform"""

    logical_cores_per_node: Optional[int]
    """Number of logical CPU cores on a compute node of the platform"""

    gpus_per_node: Optional[int]
    """Number of GPU devices on a compute node of the platform"""

    gpu_setting_type: Optional[str]
    """ How GPUs will be assigned.

    Must take one of the following string options.

    - ``"runner_default"``:       Use default setting for MPI runner (same as if not set).
    - ``"env"``:                  Use an environment variable (comma separated list of slots)
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

    gpu_setting_name: Optional[str]
    """Name of GPU setting

    See :attr:`gpu_setting_type` for more details.

    """

    scheduler_match_slots: Optional[bool]
    """
    Whether the libEnsemble resource scheduler should only assign matching slots when
    there are multiple (partial) nodes assigned to a sim function.

    Defaults to ``True``, within libEnsemble.

    Useful if setting an environment variable such as ``CUDA_VISIBLE_DEVICES``, where
    the value should match on each node of an MPI run (choose **True**).

    When using command-line options just as ``--gpus-per-node``, which allow the systems
    application level scheduler to manager GPUs, then ``match_slots`` can be **False**
    (allowing for more efficient scheduling when MPI runs cross nodes).
    """

    @validator("gpu_setting_type")
    def check_gpu_setting_type(cls, value):
        if value is not None:
            assert value in [
                "runner_default",
                "env",
                "option_gpus_per_node",
                "option_gpus_per_task",
            ], "Invalid label for GPU specification type"
        return value

    @validator("mpi_runner")
    def check_mpi_runner_type(cls, value):
        if value is not None:
            assert value in ["mpich", "openmpi", "aprun", "srun", "jsrun", "msmpi", "custom"], "Invalid MPI runner name"
        return value

    @root_validator
    def check_logical_cores(cls, values):
        if values.get("cores_per_node") and values.get("logical_cores_per_node"):
            assert (
                values["logical_cores_per_node"] % values["cores_per_node"] == 0
            ), "Logical cores doesn't divide evenly into cores"
        return values


# On SLURM systems, let srun assign free GPUs on the node
class Crusher(Platform):
    mpi_runner: str = "srun"
    cores_per_node: int = 64
    logical_cores_per_node: int = 128
    gpus_per_node: int = 8
    gpu_setting_type: str = "runner_default"
    scheduler_match_slots: bool = False


class Frontier(Platform):
    mpi_runner: str = "srun"
    cores_per_node: int = 64
    logical_cores_per_node: int = 128
    gpus_per_node: int = 8
    gpu_setting_type: str = "runner_default"
    scheduler_match_slots: bool = False


# Example of a ROCM system
class GenericROCm(Platform):
    mpi_runner: str = "mpich"
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
    gpus_per_node: int = 4
    gpu_setting_type: str = "runner_default"
    scheduler_match_slots: bool = False


class Polaris(Platform):
    mpi_runner: str = "mpich"
    runner_name: str = "mpiexec"
    cores_per_node: int = 32
    logical_cores_per_node: int = 64
    gpus_per_node: int = 4
    gpu_setting_type: str = "runner_default"
    scheduler_match_slots: bool = True


class Spock(Platform):
    mpi_runner: str = "srun"
    cores_per_node: int = 64
    logical_cores_per_node: int = 128
    gpus_per_node: int = 4
    gpu_setting_type: str = "runner_default"
    scheduler_match_slots: bool = False


class Summit(Platform):
    mpi_runner: str = "jsrun"
    cores_per_node: int = 42
    logical_cores_per_node: int = 168
    gpus_per_node: int = 6
    gpu_setting_type: str = "option_gpus_per_task"
    gpu_setting_name: str = "-g"
    scheduler_match_slots: bool = False


class Sunspot(Platform):
    mpi_runner: str = "mpich"
    runner_name: str = "mpiexec"
    cores_per_node: int = 104
    logical_cores_per_node: int = 208
    gpus_per_node: int = 6
    gpu_setting_type: str = "runner_default"
    scheduler_match_slots: bool = True


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


    If the platform is not specified, libEnsemble will attempt detect known
    platforms (this is not guaranteed).

    **Note**: libEnsemble should work on any platform, and detects most
    system configurations correctly. These options are helpful for optimization and
    where auto-detection encounters ambiguity or an unknown feature.
    """

    generic_rocm: GenericROCm = GenericROCm()
    crusher: Crusher = Crusher()
    frontier: Frontier = Frontier()
    perlmutter_c: PerlmutterCPU = PerlmutterCPU()
    perlmutter_g: PerlmutterGPU = PerlmutterGPU()
    polaris: Polaris = Polaris()
    spock: Spock = Spock()
    summit: Summit = Summit()
    sunspot: Sunspot = Sunspot()


# Dictionary of known systems (or system partitions) detectable by domain name
detect_systems = {
    "crusher.olcf.ornl.gov": Crusher,
    "frontier.olcf.ornl.gov": Frontier,
    "hsn.cm.polaris.alcf.anl.gov": Polaris,
    "spock.olcf.ornl.gov": Spock,
    "summit.olcf.ornl.gov": Summit,  # Need to detect gpu count
}


def known_envs():
    """Detect system by environment variables"""
    platform_info = {}
    if os.environ.get("NERSC_HOST") == "perlmutter":
        if os.environ.get("SLURM_JOB_PARTITION").startswith("gpu_"):
            platform_info = PerlmutterGPU().dict(by_alias=True)
        else:
            platform_info = PerlmutterCPU().dict(by_alias=True)
    return platform_info


def known_system_detect(cmd="hostname -d"):
    """Detect known systems

    This is a function attempts to detect if on a known system, but users
    should specify systems to be sure.
    """
    run_cmd = cmd.split()
    platform_info = {}
    try:
        domain_name = subprocess.check_output(run_cmd).decode().rstrip()
        platform_info = detect_systems[domain_name]().dict(by_alias=True)
    except Exception:
        platform_info = known_envs()
    return platform_info


def get_platform(libE_specs):
    """Return platform as dictionary from relevant libE_specs option.

    For internal use, return a platform as a dictionary from either
    platform name or platform_specs.

    If both platform and platform_spec fields are present, any fields in
    platform_specs are added or overwrite fields in the known platform.
    """

    name = libE_specs.get("platform") or os.environ.get("LIBE_PLATFORM")
    if name:
        try:
            known_platforms = Known_platforms().dict()
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
    else:
        # See if in detection list
        platform_info = known_system_detect()

    platform_info = {k: v for k, v in platform_info.items() if v is not None}
    return platform_info
