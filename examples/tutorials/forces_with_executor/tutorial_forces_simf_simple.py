import numpy as np

from libensemble.executors.executor import Executor
from libensemble.message_numbers import WORKER_DONE, TASK_FAILED

MAX_SEED = 32767


def run_forces(H, persis_info, sim_specs, libE_info):
    calc_status = 0

    # Parse out num particles, from generator function
    particles = str(int(H["x"][0][0]))

    # num particles, timesteps, also using num particles as seed
    args = particles + " " + str(10) + " " + particles

    # Obtain our MPI Executor instance, submit the forces app for execution
    exctr = Executor.executor
    task = exctr.submit(app_name="forces", app_args=args, wait_on_start=True)

    # Block until the task finishes
    task.wait(timeout=60)

    # Stat file to check for bad runs
    statfile = "forces{}.stat".format(particles)

    # Try loading final energy reading, setting the computation's status
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
