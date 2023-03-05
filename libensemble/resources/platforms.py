"""Known platforms default configuration"""

# gpu_setting may need two values - one for type and one for name
# e.g. gpu_setting_type = "env" or gpu_set_type = "arg" (or 1 / 2)
#      gpu_setting_name = "CUDA_VISIBLE_DEVICES" or gpu_setting_name = "--gpus-per-node="
# If gpu_setting_type is GPU_SET_DEF, do not need gpu_setting_name (uses default method for MPI runner).

# GPU ASSIGNMENT TYPES (user mpi runner default, use environment variable, specify mpi runner option)
GPU_SET_DEF = 1  # Use default setting for MPI runner (same as if not set)
GPU_SET_ENV = 2  # Use an environment variable
GPU_SET_CLI = 3  # Expresses GPUs per node.
GPU_SET_CLI_GPT = 4  # Expresses GPUs per task.


#Need to have subgroup launch (and kill?)
#May need something like "gpu_tiles"

summit = {
    "mpi_runner": "jsrun",
    "cores_per_node": 42,
    "logical_cores_per_node": 168,
    "gpus_per_node": 6,
    "gpu_setting_type": GPU_SET_CLI_GPT,  # Can also use GPU_SET_DEF (which is -g for jsrun)
    "gpu_setting_name": "-g",
    "scheduler_match_slots": False,
}

# Perlmutter has CPU and GPU nodes
perlmutter_g = {
    "mpi_runner" : 'srun',
    "cores_per_node": 64,
    "log_cores_per_node": 128,
    "gpus_per_node" : 4,
    #"gpu_setting_type": GPU_SET_CLI,
    #"gpu_setting_name": "--gpus-per-node=",
    "gpu_setting_type": GPU_SET_DEF,
    "scheduler_match_slots": False,
    }

polaris = {
    "mpi_runner" : 'mpich',
    "runner_name" : 'mpiexec',
    "cores_per_node" : 32,
    "log_cores_per_node" : 64,
    "gpus_per_node" : 4,
    "gpu_setting_type": GPU_SET_DEF,
    "scheduler_match_slots": True,
    }

#TODO MAKE ALPHABETICAL
# Dictionary of known systems (systems or system partitions) by name
known_systems = {"summit": summit,
                 "perlmutter_g": perlmutter_g,
                 "polaris": polaris,
                 }

# Dictionary of known systems (systems or system partitions) detectable by domain name
detect_systems = {"summit.olcf.ornl.gov": summit}
# hsn.cm.polaris.alcf.anl.gov polaris  #TODO do we want to always detect if can? What about multiple node systems.


def get_mpiexec_platforms(system_name):
    system = known_systems[system_name]
    return {"mpi_runner": system["mpi_runner"],
            "runner_name" : system.get("runner_name"),  # only used where distinction needed
            "gpu_setting_type" : system["gpu_setting_type"],
            "gpu_setting_name" : system.get("gpu_setting_name"),  # Not needed with GPU_SET_DEF
            #"" : system[""],
            }
