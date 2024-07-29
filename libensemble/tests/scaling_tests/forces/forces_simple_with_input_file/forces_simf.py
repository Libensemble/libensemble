import jinja2
import numpy as np

# Optional status codes to display in libE_stats.txt for each gen or sim
from libensemble.message_numbers import TASK_FAILED, WORKER_DONE


def set_input_file_params(H, sim_specs, ints=False):
    """
    This is a general function to parameterize an input file with any inputs
    from sim_specs["in"]

    Often sim_specs_in["x"] may be multi-dimensional, where each dimension
    corresponds to a different input name in sim_specs["user"]["input_names"]).
    Effectively an unpacking of "x"
    """
    input_file = sim_specs["user"]["input_filename"]
    input_values = {}
    for i, name in enumerate(sim_specs["user"]["input_names"]):
        value = int(H["x"][0][i]) if ints else H["x"][0][i]
        input_values[name] = value
    with open(input_file, "r") as f:
        template = jinja2.Template(f.read())
    with open(input_file, "w") as f:
        f.write(template.render(input_values))


def run_forces(H, persis_info, sim_specs, libE_info):
    """Runs the forces MPI application reading input from file"""

    calc_status = 0

    set_input_file_params(H, sim_specs, ints=True)

    # Retrieve our MPI Executor
    exctr = libE_info["executor"]

    # Submit our forces app for execution.
    task = exctr.submit(app_name="forces")  # app_args removed

    # Block until the task finishes
    task.wait(timeout=60)

    # Stat file to check for bad runs
    statfile = "forces.stat"

    # Try loading final energy reading, set the sim's status
    try:
        data = np.loadtxt(statfile)
        final_energy = data[-1]
        calc_status = WORKER_DONE
    except Exception:
        final_energy = np.nan
        calc_status = TASK_FAILED

    # Define our output array,  populate with energy reading
    outspecs = sim_specs["out"]
    output = np.zeros(1, dtype=outspecs)
    output["energy"][0] = final_energy

    # Return final information to worker, for reporting to manager
    return output, persis_info, calc_status
