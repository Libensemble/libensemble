import numpy as np
import jinja2
# Optional status codes to display in libE_stats.txt for each gen or sim
from libensemble.message_numbers import TASK_FAILED, WORKER_DONE


def set_objective_value():
    try:
        data = np.load("test-w7x-gx.log.npy", allow_pickle=True)
        out = data.flatten()[0]['Wtot_MJ'][-1]
        return out
    except Exception:
        return np.nan


def set_input_file_params(H, sim_specs, ints=False):
    """
    This is a general function to parameterize an input file with any inputs
    from sim_specs["in"]

    Often sim_specs_in["x"] may be multi-dimensional, where each dimension
    corresponds to a different input name in sim_specs["user"]["input_names"]).
    Effectively an unpacking of "x"
    """
    input_file = sim_specs["user"].get("input_filename")
    input_names = sim_specs["user"].get("input_names")
    if not input_file or not input_names:
        return
    input_values = {}
    for i, name in enumerate(input_names):
        print("jeff", H["x"])
        value = int(H["x"][0][i]) if ints else H["x"][0][i]
        print("jeff2", value)
        input_values[name] = value
    with open(input_file, "r") as f:
        template = jinja2.Template(f.read())
    with open(input_file, "w") as f:
        f.write(template.render(input_values))


def run_t3d_and_gx(H, persis_info, sim_specs, libE_info):
    """Runs the t3d_and_gx MPI application reading input from file"""

    calc_status = 0

    set_input_file_params(H, sim_specs, ints=False)

    # Retrieve our MPI Executor
    exctr = libE_info["executor"]

    # Submit our t3d_and_gx app for execution.
    task = exctr.submit(
        app_name="t3d",
        app_args=sim_specs["user"].get("input_filename")
        auto_assign_gpus=True,
        match_procs_to_gpus=True,
    )

    # Block until the task finishes
    task.wait()

    # Read output and set the objective
    f = set_objective_value()

    # Optionally set the sim's status to show in the libE_stats.txt file
    if np.isnan(f):
        calc_status = TASK_FAILED
    else:
        calc_status = WORKER_DONE
    outspecs = sim_specs["out"]
    output = np.zeros(1, dtype=outspecs)
    output["f"][0] = f

    # Return final information to worker, for reporting to manager
    return output, persis_info, calc_status
