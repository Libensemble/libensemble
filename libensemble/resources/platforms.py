"""Known platforms default configuration"""
import os
from typing import Optional

from pydantic import BaseModel, root_validator, validator


class PlatformException(Exception):
    "Platform module exception."


# GPU ASSIGNMENT TYPES
GPU_SET_DEF = 1  # Use default setting for MPI runner (same as if not set). gpu_setting_name not required.
GPU_SET_ENV = 2  # Use an environment variable
GPU_SET_CLI = 3  # Expresses GPUs per node on MPI runner command line.
GPU_SET_CLI_GPT = 4  # Expresses GPUs per task on MPI runner command line.

# e.g.
# "gpu_setting_type":  GPU_SET_ENV,
# "gpu_setting_name": "ROCR_VISIBLE_DEVICES",


class Platform(BaseModel):
    mpi_runner: str
    runner_name: Optional[str]
    cores_per_node: int
    logical_cores_per_node: int
    gpus_per_node: int
    gpu_setting_type: int
    gpu_setting_name: str
    scheduler_match_slots: bool

    @validator("gpu_setting_type")
    def check_gpu_setting_type(cls, value):
        assert value in [
            GPU_SET_DEF,
            GPU_SET_ENV,
            GPU_SET_CLI,
            GPU_SET_CLI_GPT,
        ], "Invalid label for GPU specification type"

    @validator("mpi_runner")
    def check_mpi_runner_type(cls, value):
        assert value in ["mpich", "openmpi", "aprun", "srun", "jsrun", "msmpi", "custom"], "Invalid MPI runner name"

    @root_validator
    def check_logical_cores(cls, values):
        assert (
            values["logical_cores_per_node"] % values["cores_per_node"] == 0
        ), "Logical cores doesn't divide evenly into cores"
        return values


class Summit(Platform):
    mpi_runner: str = "jsrun"
    cores_per_node: int = 42
    logical_cores_per_node: int = 168
    gpus_per_node: int = 6
    gpu_setting_type: int = GPU_SET_CLI_GPT
    gpu_setting_name: str = "-g"
    scheduler_match_slots: bool = False


class PerlmutterGPU(Platform):
    mpi_runner: str = "srun"
    cores_per_node: int = 64
    logical_cores_per_node: int = 128
    gpus_per_node: int = 4
    gpu_setting_type: int = GPU_SET_DEF
    scheduler_match_slots: bool = False


class Polaris(Platform):
    mpi_runner: str = "mpich"
    runner_name: str = "mpiexec"
    cores_per_node: int = 32
    logical_cores_per_node: int = 64
    gpus_per_node: int = 4
    gpu_setting_type: int = GPU_SET_DEF
    scheduler_match_slots: bool = True


class Spock(Platform):
    mpi_runner: str = "srun"
    cores_per_node: int = 64
    logical_cores_per_node: int = 128
    gpus_per_node: int = 4
    gpu_setting_type: int = GPU_SET_DEF
    scheduler_match_slots: bool = False


class Crusher(Platform):
    mpi_runner: str = "srun"
    cores_per_node: int = 64
    logical_cores_per_node: int = 128
    gpus_per_node: int = 8
    gpu_setting_type: int = GPU_SET_DEF
    scheduler_match_slots: bool = False


class Sunspot(Platform):
    mpi_runner: str = "mpich"
    runner_name: str = "mpiexec"
    cores_per_node: int = 104
    logical_cores_per_node: int = 208
    gpus_per_node: int = 6
    gpu_setting_type: int = GPU_SET_DEF
    scheduler_match_slots: bool = True


# TODO MAKE ALPHABETICAL
# Dictionary of known systems (systems or system partitions) by name
known_systems = {
    "summit": Summit,
    "perlmutter_g": PerlmutterGPU,
    "polaris": Polaris,
    "spock": Spock,
    "crusher": Crusher,
    "sunspot": Sunspot,
}

# Dictionary of known systems (systems or system partitions) detectable by domain name
detect_systems = {
    "summit.olcf.ornl.gov": Summit,  # Need to detect gpu count
}

# TODO Also could detect by hostname but do we want to.
# detect_systems = {"summit.olcf.ornl.gov": summit,  # Need to detect gpu count
# "spock.olcf.ornl.gov": spock,
# "hsn.cm.polaris.alcf.anl.gov": polaris_g,  # What about partitions?
# "crusher.olcf.ornl.gov": crusher,
# }


# TODO Review function naming
def get_platform_num_cores_gpus(system_name):  # act systm dict itself
    """Return list of number of cores and gpus per node

    system_name is a system dictionary or string (system name)

    Form: [cores, logical_cores, gpus].
    Any items not present are returned as None.
    """
    system = known_systems[system_name] if isinstance("system_name", str) else system_name
    cores_per_node = system.cores_per_node
    logical_cores_per_node = system.logical_cores_per_node
    gpus_per_node = system.gpus_per_node
    return [cores_per_node, logical_cores_per_node, gpus_per_node]


def get_mpiexec_platforms(system_name):
    """Return dict of mpi runner info"""
    system = known_systems[system_name]
    return {
        "mpi_runner": system["mpi_runner"],
        "runner_name": system.get("runner_name"),  # only used where distinction needed
        "gpu_setting_type": system["gpu_setting_type"],
        "gpu_setting_name": system.get("gpu_setting_name"),  # Not needed with GPU_SET_DEF
        # "" : system[""],
    }


# TODO - if known plaform and platform_spec - should it combine / overwriting known with spec?
def get_platform_from_specs(libE_specs):
    """Return dictionary of platform information"""
    platform_info = libE_specs.get("platform_spec")
    if platform_info is None:
        name = libE_specs["platform"] or os.environ.get("LIBE_PLATFORM")
        if name is not None:
            try:
                platform_info = known_systems[name]
            except KeyError:
                raise PlatformException(f"Error. Unknown platform requested {name}")
    return platform_info
