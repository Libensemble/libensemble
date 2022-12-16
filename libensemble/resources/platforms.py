"""Known platforms default configuration"""

# SH TODO
# gpu_setting may need two values - one for type and one for name
# e.g. gpu_set_type = "env" or gpu_set_type = "argument"
#      gpu_setting_name = "CUDA_VISIBLE_DEVICES" or gpu_setting_name = "gpus_per_task"


summit = {
    "mpi_runner": "jsrun",
    "cores_per_node": 42,
    "logical_cores_per_node": 168,
    "gpus_per_node": 6,
    "GPU_setting": None,  # System automatically finds GPUs
    "scheduler_match_slots": False,
}

# Return the dictionary
detect_systems = {"summit.olcf.ornl.gov": summit}
