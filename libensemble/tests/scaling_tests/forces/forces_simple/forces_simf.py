import numpy as np

# To retrieve our MPI Executor instance
from libensemble.executors.executor import Executor

# Optional status codes to display in libE_stats.txt for each gen or sim
from libensemble.message_numbers import WORKER_DONE, TASK_FAILED


def run_forces(H, persis_info, sim_specs, libE_info):
    calc_status = 0

    # Parse out num particles, from generator function
    particles = str(int(H["x"][0][0]))

    # app arguments: num particles, timesteps, also using num particles as seed
    args = particles + " " + str(10) + " " + particles

    # Retrieve our MPI Executor instance
    exctr = Executor.executor

    # Submit our forces app for execution. Block until the task starts.
    task = exctr.submit(app_name="forces", app_args=args)

    # Block until the task finishes
    task.wait()

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
    output["energy"] = final_energy

    # Return final information to worker, for reporting to manager
    return output, persis_info, calc_status
