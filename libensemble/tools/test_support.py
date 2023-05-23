from libensemble.resources.resources import Resources


def _get_value(option_name, cmd_line):
    """Return string of option and value found in command line"""
    val = None
    inputs = cmd_line.split()
    for i, word in enumerate(inputs):
        if word == option_name:
            val = inputs[i + 1]
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
            opt_val = word + " " + inputs[i + 1]
        elif word.startswith(option_name):
            if option_name.endswith("="):
                opt_val = word
    return opt_val


def _get_expected_output(name, value):
    """Return expected gpu runline setting"""
    if name.endswith("="):
        return name + str(value)
    return name + " " + str(value)


def _safe_min(a, b):
    """Takes min of two values, ignoring if second is None"""
    if b is not None:
        return min(a, b)
    else:
        return a


def _set_gpus(task, wresources):
    return wresources.doihave_gpus() and task.ngpus_req > 0


def check_gpu_setting(task, assert_setting=True, print_setting=False, resources=None):
    """Checks GPU run lines

    Note that this will check based platform_info or defaults for the MPI runner
    As mpi_runner is taken from start of tasks runline, this could also fail if the
    runner has been renamed via the "runner_name" option.

    Parameters
    ----------

    assert_setting: boolean, optional
        Raise error if setting is not as expected (for current MPI runner). Default: True

    print_setting: boolean, optional
        Print GPU setting to stdout. Default: False

    """

    resources = resources or Resources.resources
    gresources = resources.glob_resources
    wresources = resources.worker_resources

    slots = wresources.slots
    mpirunner = task.runline.split(" ", 1)[0]
    stype = None
    gpu_setting = None
    gpus_per_task = False
    cmd_line = False

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

    gpu_settings = ["runner_default", "env", "option_gpus_per_node", "option_gpus_per_task"]

    # Get num_procs and procs per node from command line
    num_procs = _get_value(procs_setting[mpirunner], task.runline)
    for setting in ppn_setting[mpirunner]:
        ppn = _get_value(setting, task.runline)
        if ppn is not None:
            break

    # For user settings via platform/platform_specs
    expected_setting = None
    if gresources.platform_info:
        setting_type = gresources.platform_info.get("gpu_setting_type")
        if setting_type is not None:
            assert setting_type in gpu_settings, f"unknown setting type in  platform_info: {setting_type}"
            if setting_type != "runner_default":
                expected_setting = gresources.platform_info.get("gpu_setting_name")
                assert expected_setting is not None, "gpu_setting_type must have a gpu_setting_name"
                if setting_type in ["option_gpus_per_node", "option_gpus_per_task"]:
                    assert wresources.even_slots, f"Error: Found uneven slots on nodes {slots}"
                    cmd_line = True
                elif setting_type in ["env"]:
                    assert wresources.matching_slots, f"Error: Found unmatching slots on nodes {slots}"
                if setting_type == "option_gpus_per_task":
                    gpus_per_task = True

    # For default MPI runner settings
    if expected_setting is None:
        # mpirunners that expect a command line option
        if mpirunner in ["srun", "jsrun"]:
            assert wresources.even_slots, f"Error: Found uneven slots on nodes {slots}"
            cmd_line = True
            if mpirunner == "srun":
                expected_setting = "--gpus-per-node"
                if _get_value(expected_setting, task.runline) is None:
                    # Try gpus per task
                    if _get_value("--gpus-per-task", task.runline) is not None:
                        gpus_per_task = True
                        expected_setting = "--gpus-per-task"
            elif mpirunner == "jsrun":
                gpus_per_task = True
                expected_setting = "-g"
        # Default environment settings
        else:
            assert wresources.matching_slots, f"Error: Found unmatching slots on nodes {slots}"
            expected_setting = "CUDA_VISIBLE_DEVICES"

    # Get expected numbers
    if cmd_line:
        if gpus_per_task:
            stype = "runline option: gpus per task"
            avail_gpus = wresources.slot_count * wresources.gpus_per_rset // int(ppn)
            expected_nums = _safe_min(avail_gpus, wresources.gen_ngpus)
        else:
            stype = "runline option: gpus per node"
            expected_nums = _safe_min(wresources.slot_count * wresources.gpus_per_rset, wresources.gen_ngpus)
        expected_nums = expected_nums if _set_gpus(task, wresources) else None
        if expected_nums is not None:
            expected = _get_expected_output(expected_setting, expected_nums)
        else:
            expected = None
        if expected_setting in task.runline:
            gpu_setting = _get_opt_value(expected_setting, task.runline)
    else:
        stype = "Env var"
        expected_nums = wresources.get_slots_as_string(multiplier=wresources.gpus_per_rset, limit=wresources.gen_ngpus)
        expected_nums = expected_nums if _set_gpus(task, wresources) else None
        if expected_nums is not None:
            expected = {expected_setting: expected_nums}
        else:
            expected = {}
        gpu_setting = task.env

    # If it's a custom runner - we may not have procs info (this is just printed info)
    if num_procs is None or ppn is None:
        addon = ""
    else:
        addon = f"(procs {num_procs}, per node {ppn})"

    if print_setting:
        print(f"Worker {task.workerID}: GPU setting ({stype}): {gpu_setting} {addon}")

    if assert_setting:
        assert (
            gpu_setting == expected
        ), f"Worker {task.workerID}: Found GPU setting: {gpu_setting}, Expected: {expected}"
