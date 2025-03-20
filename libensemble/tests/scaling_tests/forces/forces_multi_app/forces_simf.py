import numpy as np

# Optional status codes to display in libE_stats.txt for each gen or sim
from libensemble.message_numbers import TASK_FAILED, WORKER_DONE

# Optional - to print GPU settings
from libensemble.tools.test_support import check_gpu_setting


def run_forces(H, persis_info, sim_specs, libE_info):
    """Launches the either the CPU-only or GPU version of the forces MPI app
    and auto-assigns ranks and GPU resources as requested by the generator.
    """

    calc_status = 0

    # Parse out num particles, from generator function
    particles = str(int(H["x"][0][0]))

    # app arguments: num particles, timesteps, also using num particles as seed
    args = particles + " " + str(10) + " " + particles

    # Retrieve our MPI Executor
    exctr = libE_info["executor"]

    app_type = H["app_type"][0].decode()

    # Submit our forces app for execution.
    task = exctr.submit(
        app_name=app_type,
        app_args=args,
    )

    # Block until the task finishes
    task.wait()

    # Optional - prints GPU assignment (method and numbers)
    check_gpu_setting(task, assert_setting=False, print_setting=True, desc=app_type)

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
