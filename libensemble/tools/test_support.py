from libensemble.resources.resources import Resources

def _get_value(option_name, cmd_line):
    """Return string of option and value found in command line"""
    val = None
    inputs = cmd_line.split()
    for i, word in enumerate(inputs):
        if word == option_name:
            val = inputs[i+1]
        elif word.startswith(option_name):
            if option_name.endswith("="):
                val = word.split("=")[1]
    return val

def _get_opt_value(option_name, cmd_line):
    """Return string of options value found in command line"""
    opt_val = None
    inputs = cmd_line.split()
    for i, word in enumerate(inputs):
        if word == option_name:
            opt_val = word + " " + inputs[i+1]
        elif word.startswith(option_name):
            if option_name.endswith("="):
                opt_val = word
    return opt_val

def check_gpu_setting(task, assert_setting=True, print_setting=False, resources=None):
    """Checks GPU run lines

    Note that this will only check based on defaults for MPI runner, if the
    user has configured GPU settings, e.g.,~ via LIBE_PLATFORM, then this test
    may not be correct.

    Parameters
    ----------

    assert_setting: boolean, optional
        Error if setting is not expected (for current MPI runner). Default: True

    print_setting: boolean, optional
        Print GPU setting to stdout. Default: False

    """

    resources = resources or Resources.resources.worker_resources
    slots = resources.slots
    mpirunner = task.runline.split(' ', 1)[0]
    gpu_setting = None
    stype = None

    # mpirunners that expect a command line option
    if mpirunner in ["srun", "jsrun"]:
        assert resources.even_slots, f"Error: Found uneven slots on nodes {slots}"

        if mpirunner == "srun":
            stype = "runline option: gpus per node"
            expected_setting = "--gpus-per-node="
            expected_nums = resources.slot_count * resources.gpus_per_rset
            expected = expected_setting + str(expected_nums)

        elif mpirunner == "jsrun":
            stype = "runline option: gpus per task"
            expected_setting = "-g"
            num_procs = _get_value("-n", task.runline)
            expected_nums = resources.slot_count * resources.gpus_per_rset // int(num_procs)
            expected = expected_setting + " " + str(expected_nums)

        if expected_setting in task.runline:
            gpu_setting = _get_opt_value(expected_setting, task.runline)

    else:
        assert resources.matching_slots, f"Error: Found unmatching slots on nodes {slots}"
        expected_setting = "CUDA_VISIBLE_DEVICES"
        expected_nums = resources.get_slots_as_string()
        expected = {expected_setting:expected_nums}
        stype = "Env var"
        gpu_setting = task.env


    if print_setting:
        print(f"Worker {task.workerID}: GPU setting ({stype}): {gpu_setting}")

    if assert_setting:
        assert gpu_setting == expected, f"Found GPU setting: {gpu_setting}, Expected: {expected}"
