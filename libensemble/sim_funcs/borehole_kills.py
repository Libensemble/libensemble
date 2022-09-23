import numpy as np
from libensemble.executors.executor import Executor
from libensemble.sim_funcs.surmise_test_function import borehole_true
from libensemble.message_numbers import UNSET_TAG, TASK_FAILED, MAN_KILL_SIGNALS


def subproc_borehole(H, delay):
    """This evaluates the Borehole function using a subprocess
    running compiled code.

    Note that the Executor base class submit runs a
    serial process in-place. This should work on compute nodes
    so long as there are free contexts.

    """
    with open("input", "w") as f:
        H["thetas"][0].tofile(f)
        H["x"][0].tofile(f)

    exctr = Executor.executor
    args = "input" + " " + str(delay)

    task = exctr.submit(app_name="borehole", app_args=args, stdout="out.txt", stderr="err.txt")
    calc_status = exctr.polling_loop(task, delay=0.01, poll_manager=True)

    if calc_status in MAN_KILL_SIGNALS + [TASK_FAILED]:
        f = np.inf
    else:
        f = float(task.read_stdout())
    return f, calc_status


def borehole(H, persis_info, sim_specs, libE_info):
    """
    Wraps the borehole function
    Subprocess to test receiving kill signals from manager
    """
    calc_status = UNSET_TAG  # Calc_status gets printed in libE_stats.txt
    H_o = np.zeros(H["x"].shape[0], dtype=sim_specs["out"])

    # Add a delay so subprocessed borehole takes longer
    sim_id = libE_info["H_rows"][0]
    delay = 0
    if sim_id > sim_specs["user"]["init_sample_size"]:
        delay = 2 + np.random.normal(scale=0.5)

    f, calc_status = subproc_borehole(H, delay)

    if calc_status in MAN_KILL_SIGNALS and "sim_killed" in H_o.dtype.names:
        H_o["sim_killed"] = True  # For calling script to print only.
    else:
        # Failure model (excluding observations)
        if sim_id > sim_specs["user"]["num_obs"]:
            if (f / borehole_true(H["x"])) > 1.25:
                f = np.inf
                calc_status = TASK_FAILED
                print(f"Failure of sim_id {sim_id}", flush=True)

    H_o["f"] = f
    return H_o, persis_info, calc_status
