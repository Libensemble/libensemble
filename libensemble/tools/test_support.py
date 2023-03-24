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

def _get_expected_output(name, value):
    """Return expected gpu runline setting"""
    if name.endswith("="):
        return name + str(value)
    return name + " " + str(value)


def check_gpu_setting(task, assert_setting=True, print_setting=False, resources=None):
    """Checks GPU run lines

    Note that this will only check based on defaults for MPI runner, if the
    user has configured GPU settings, e.g.,~ via LIBE_PLATFORM, then this test
    may not be correct. As mpi_runner is taken from start of tasks runline, this
    could also fail if the runner has been renamed.

    Parameters
    ----------

    assert_setting: boolean, optional
        Raise error if setting is not as expected (for current MPI runner). Default: True

    print_setting: boolean, optional
        Print GPU setting to stdout. Default: False

    """

    resources = resources or Resources.resources.worker_resources
    slots = resources.slots
    mpirunner = task.runline.split(' ', 1)[0]
    gpu_setting = None
    stype = None
    gpus_per_task = False

    # Configuration is parsed from runline to ensure output is used.

    procs_setting = {
                "mpiexec": "-np",
                "mpirun": "-np",
                "srun": "--ntasks",
                "jsrun": "-n",
                "aprun": "-n",
                }

    ppn_setting = {
                "mpiexec": ["--ppn", "-npernode"],
                "mpirun": ["--ppn", "-npernode"],
                "srun": ["--ntasks-per-node"],
                "jsrun": ["-r"],
                "aprun": ["-N"],
                }

    num_procs = _get_value(procs_setting[mpirunner], task.runline)

    for setting in ppn_setting[mpirunner]:
        ppn = _get_value(setting, task.runline)
        if ppn is not None:
            break

    # mpirunners that expect a command line option
    if mpirunner in ["srun", "jsrun"]:
        assert resources.even_slots, f"Error: Found uneven slots on nodes {slots}"

        if mpirunner == "srun":
            expected_setting = "--gpus-per-node="
            if _get_value(expected_setting, task.runline) is None:
                # Try gpus per task
                gpus_per_task = True
                expected_setting = "--gpus-per-task="

        elif mpirunner == "jsrun":
            gpus_per_task = True
            expected_setting = "-g"

        if gpus_per_task:
            stype = "runline option: gpus per task"
            expected_nums = resources.slot_count * resources.gpus_per_rset // int(ppn)
        else:
            stype = "runline option: gpus per node"
            expected_nums = resources.slot_count * resources.gpus_per_rset

        expected = _get_expected_output(expected_setting, expected_nums)

        if expected_setting in task.runline:
            gpu_setting = _get_opt_value(expected_setting, task.runline)

    else:
        assert resources.matching_slots, f"Error: Found unmatching slots on nodes {slots}"
        expected_setting = "CUDA_VISIBLE_DEVICES"
        expected_nums = resources.get_slots_as_string(multiplier=resources.gpus_per_rset)
        expected = {expected_setting:expected_nums}
        stype = "Env var"
        gpu_setting = task.env

    # If could be a custom runner - we dont have procs info
    if mpirunner == "mpiexec":
        addon = ""
    else:
        addon = f"(procs {num_procs}, per node {ppn})"

    if print_setting:
        print(f"Worker {task.workerID}: GPU setting ({stype}): {gpu_setting} {addon}")

    if assert_setting:
        assert gpu_setting == expected, f"Found GPU setting: {gpu_setting}, Expected: {expected}"
