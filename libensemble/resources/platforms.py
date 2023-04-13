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

class PlatformException(Exception):
    "Platform module exception."

summit = {
    "mpi_runner": "jsrun",
    "cores_per_node": 42,
    "logical_cores_per_node": 168,
    "gpus_per_node": 6,
    "gpu_setting_type": "option_gpus_per_task",  # Can also use "runner_default" (which is -g for jsrun)
    "gpu_setting_name": "-g",
    "scheduler_match_slots": False,
}

# Perlmutter has CPU and GPU nodes
perlmutter_g = {
    "mpi_runner" : "srun",
    "cores_per_node": 64,
    "logical_cores_per_node": 128,
    "gpus_per_node" : 4,
    #"gpu_setting_type": "option_gpus_per_node",
    #"gpu_setting_name": "--gpus-per-node",
    "gpu_setting_type": "runner_default",
    "scheduler_match_slots": False,
    }

polaris = {
    "mpi_runner" : "mpich",
    "runner_name" : "mpiexec",
    "cores_per_node" : 32,
    "logical_cores_per_node" : 64,
    "gpus_per_node" : 4,
    "gpu_setting_type": "runner_default",
    "scheduler_match_slots": True,
    }

spock = {
    "mpi_runner": "srun",
    "cores_per_node": 64,
    "logical_cores_per_node": 128,
    "gpus_per_node": 4,
    "gpu_setting_type": "runner_default",
    "scheduler_match_slots": False,
}

crusher =  {
    "mpi_runner" : "srun",
    "cores_per_node": 64,
    "logical_cores_per_node": 128,
    "gpus_per_node" : 8,
    "gpu_setting_type": "runner_default",
    "scheduler_match_slots": False,
    }

sunspot = {
    "mpi_runner" : "mpich",
    "runner_name" : "mpiexec",
    "cores_per_node" : 104,  # finds - check
    "logical_cores_per_node" : 208,  # finds - check
    "gpus_per_node" : 6,
    "gpu_setting_type": "runner_default",
    "scheduler_match_slots": True,
    }


# Example of a ROCM system (note - if uses srun - then usually preferable to have
#    "gpu_setting_type": "runner_default",  # let SLURM assign free GPUs on the node
#    "scheduler_match_slots": False,   # allows more efficient scheduling when MPI runs cross nodes.
generic_rocm = {
    "mpi_runner" : "mpich",
    "gpu_setting_type": "env",
    "gpu_setting_name": "ROCR_VISIBLE_DEVICES",
    "scheduler_match_slots": True,
    }

#TODO MAKE ALPHABETICAL
# Dictionary of known systems (systems or system partitions) by name
known_systems = {"summit": summit,
                 "perlmutter_g": perlmutter_g,
                 "polaris": polaris,
                 "spock": spock,
                 "crusher": crusher,
                 "sunspot": sunspot,
                 "generic_rocm": generic_rocm,
                 }

#TODO - should code below here be separated?

# Dictionary of known systems (systems or system partitions) detectable by domain name
#detect_systems = {"summit.olcf.ornl.gov": summit,  # Needed to detect gpu count (if not provided)
detect_systems = {"summit.olcf.ornl.gov": "summit",  # Needed to detect gpu count (if not provided)
                  }

#TODO Also could detect by hostname but do we want to.
#detect_systems = {"summit.olcf.ornl.gov": summit,  # Need to detect gpu count
                  #"spock.olcf.ornl.gov": spock,
                  #"hsn.cm.polaris.alcf.anl.gov": polaris_g,  # What about partitions?
                  #"crusher.olcf.ornl.gov": crusher,
                  #}


#TODO Review function naming
def get_platform_num_cores_gpus(system_name):
    """Return list of number of cores and gpus per node

    system_name is a system dictionary or string (system name)

    Form: [cores, logical_cores, gpus].
    Any items not present are returned as None.
    """
    system = known_systems[system_name] if isinstance("system_name", str) else system_name
    cores_per_node = system.get("cores_per_node")
    logical_cores_per_node = system.get("logical_cores_per_node")
    gpus_per_node = system.get("gpus_per_node")
    return [cores_per_node, logical_cores_per_node, gpus_per_node]


def get_mpiexec_platforms(system_name):
    """Return dictionary of mpi runner info"""
    system = known_systems[system_name]
    return {"mpi_runner": system["mpi_runner"],
            "runner_name" : system.get("runner_name"),  # only used where distinction needed
            "gpu_setting_type" : system["gpu_setting_type"],
            "gpu_setting_name" : system.get("gpu_setting_name"),  # Not needed with "runner_default"
            }


def get_platform_from_specs(libE_specs):
    """Return dictionary of platform information

    If both platform and platform_spec fields are present, any fields in
    platform_specs are added or overwrite fields in the known platform.
    """
    platform_info = {}
    name = libE_specs.get("platform") or os.environ.get("LIBE_PLATFORM")
    if name:
        try:
            platform_info = known_systems[name]
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
