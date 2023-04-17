"""Known platforms default configuration

Any fields not included, libEnsemble will attempt to detect from the system or use a default.

Field "gpu_setting_type" has the following string options.

"runner_default"       Use default setting for MPI runner (same as if not set).
"env"                  Use an environment variable (sets to comma separated list of slots)
"option_gpus_per_node" Expresses GPUs per node on MPI runner command line.
"option_gpus_per_task" Expresses GPUs per task on MPI runner command line.

With the exception of "runner_default", the "gpu_setting_name" field is also required.

Examples

"gpu_setting_type": "env",
"gpu_setting_name": "ROCR_VISIBLE_DEVICES",

If "gpu_setting_type" is not provided (same as "runner_default") and the MPI runner does
not have a default GPU setting in libEnsemble, and no other information is present,
then the environment variable ``CUDA_VISIBLE_DEVICES`` is used.

"gpu_setting_type": "option_gpus_per_node",
"gpu_setting_name": "--gpus-per-node",

"""

#TODO list fields (in docstring or somehow). Not just GPU setting.

import os
from typing import Optional

from pydantic import BaseModel, root_validator, validator


class PlatformException(Exception):
    "Platform module exception."


class Platform(BaseModel):
    mpi_runner: Optional[str]
    runner_name: Optional[str]
    cores_per_node: Optional[int]
    logical_cores_per_node: Optional[int]
    gpus_per_node: Optional[int]
    gpu_setting_type: Optional[str]
    gpu_setting_name: Optional[str]
    scheduler_match_slots: Optional[bool]

    @validator("gpu_setting_type")
    def check_gpu_setting_type(cls, value):
        assert value in [
            "runner_default",
            "env",
            "option_gpus_per_node",
            "option_gpus_per_task",
        ], "Invalid label for GPU specification type"
        return value

    @validator("mpi_runner")
    def check_mpi_runner_type(cls, value):
        assert value in ["mpich", "openmpi", "aprun", "srun", "jsrun", "msmpi", "custom"], "Invalid MPI runner name"
        return value

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
    gpu_setting_type: int = "option_gpus_per_task"
    gpu_setting_name: str = "-g"
    scheduler_match_slots: bool = False


class PerlmutterGPU(Platform):
    mpi_runner: str = "srun"
    cores_per_node: int = 64
    logical_cores_per_node: int = 128
    gpus_per_node: int = 4
    gpu_setting_type: int = "runner_default"
    scheduler_match_slots: bool = False


class Polaris(Platform):
    mpi_runner: str = "mpich"
    runner_name: str = "mpiexec"
    cores_per_node: int = 32
    logical_cores_per_node: int = 64
    gpus_per_node: int = 4
    gpu_setting_type: int = "runner_default"
    scheduler_match_slots: bool = True


class Spock(Platform):
    mpi_runner: str = "srun"
    cores_per_node: int = 64
    logical_cores_per_node: int = 128
    gpus_per_node: int = 4
    gpu_setting_type: int = "runner_default"
    scheduler_match_slots: bool = False


class Crusher(Platform):
    mpi_runner: str = "srun"
    cores_per_node: int = 64
    logical_cores_per_node: int = 128
    gpus_per_node: int = 8
    gpu_setting_type: int = "runner_default"
    scheduler_match_slots: bool = False


class Sunspot(Platform):
    mpi_runner: str = "mpich"
    runner_name: str = "mpiexec"
    cores_per_node: int = 104
    logical_cores_per_node: int = 208
    gpus_per_node: int = 6
    gpu_setting_type: int = "runner_default"
    scheduler_match_slots: bool = True


# Example of a ROCM system (note - if uses srun - then usually preferable to have
#    "gpu_setting_type": "runner_default",  # let SLURM assign free GPUs on the node
#    "scheduler_match_slots": False,   # allows more efficient scheduling when MPI runs cross nodes.
class Generic_ROCm(Platform):
    mpi_runner: str = "mpich"
    gpu_setting_type: str = "env",
    gpu_setting_name: str = "ROCR_VISIBLE_DEVICES"
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
    "generic_rocm": Generic_ROCm,
}

# Dictionary of known systems (systems or system partitions) detectable by domain name
detect_systems = {
    "summit.olcf.ornl.gov": Summit,  # Need to detect gpu count
}

# TODO Also could detect by hostname but do we want to.
# detect_systems = {"summit.olcf.ornl.gov": Summit,  # Need to detect gpu count
# "spock.olcf.ornl.gov": Spock,
# "hsn.cm.polaris.alcf.anl.gov": Polaris,
# "crusher.olcf.ornl.gov": Crusher,
# }

#TODO - should code below here be separated?

# Dictionary of known systems (systems or system partitions) detectable by domain name
#detect_systems = {"summit.olcf.ornl.gov": summit,  # Needed to detect gpu count (if not provided)
detect_systems = {"summit.olcf.ornl.gov": "summit",  # Needed to detect gpu count (if not provided)
                  }


# TODO Review function naming
def get_platform_num_cores_gpus(system_name):
    """Return list of number of cores and gpus per node

    system_name is a system dictionary or string (system name)

    Form: [cores, logical_cores, gpus].
    Any items not present are returned as None.
    """
    system = known_systems[system_name] if isinstance(system_name, str) else system_name
    cores_per_node = system.cores_per_node
    logical_cores_per_node = system.logical_cores_per_node
    gpus_per_node = system.gpus_per_node
    return [cores_per_node, logical_cores_per_node, gpus_per_node]


#TODO rename to just get_platform?
def get_platform_from_specs(libE_specs):
    """Return platform from relevant libE_specs option.

    If platform field is set and a platform, return the platform.
    If it is a string, return the platform from the known_systems.
    If it does not exist, return None
    """
    platform_info = {}
    name = libE_specs.get("platform") or os.environ.get("LIBE_PLATFORM")
    if name:
        try:
            platform_info = known_systems[name]().dict(by_alias=True)
        except KeyError:
            raise PlatformException(f"Error. Unknown platform requested {name}")

        # Add/overwrite any fields from a platform_spec
        platform_spec = libE_specs.get("platform_spec")
        if platform_spec:
            for k,v in platform_spec.items():
                platform_info[k] = v
    else:
        platform_info = libE_specs.get("platform_spec", {})

    return platform_info

