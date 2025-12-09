"""
Module containing alternative functions for running the forces MPI application

run_forces: Uses classic libEnsemble sim_f.
run_forces_dict: Uses gest-api/xopt style simulator.
"""

import numpy as np

# Optional status codes to display in libE_stats.txt for each gen or sim
from libensemble.message_numbers import TASK_FAILED, WORKER_DONE

__all__ = [
    "run_forces",
    "run_forces_dict",
]


def run_forces(H, persis_info, sim_specs, libE_info):
    """Runs the forces MPI application.

    By default assigns the number of MPI ranks to the number
    of cores available to this worker.

    To assign a different number give e.g., `num_procs=4` to
    ``exctr.submit``.
    """

    calc_status = 0

    # Parse out num particles, from generator function
    particles = str(int(H["x"][0]))  # x is a scalar for each point

    # app arguments: num particles, timesteps, also using num particles as seed
    args = particles + " " + str(10) + " " + particles

    # Retrieve our MPI Executor
    exctr = libE_info["executor"]

    # Submit our forces app for execution.
    task = exctr.submit(app_name="forces", app_args=args)

    # Block until the task finishes
    task.wait()

    # Try loading final energy reading, set the sim's status
    statfile = "forces.stat"
    try:
        data = np.loadtxt(statfile)
        final_energy = data[-1]
        calc_status = WORKER_DONE
    except Exception:
        final_energy = np.nan
        calc_status = TASK_FAILED

    # Define our output array, populate with energy reading
    output = np.zeros(1, dtype=sim_specs["out"])
    output["energy"] = final_energy

    # Return final information to worker, for reporting to manager
    return output, persis_info, calc_status


def run_forces_dict(input_dict: dict, libE_info: dict) -> dict:
    """Runs the forces MPI application (gest-api/xopt style simulator).

    Parameters
    ----------
    input_dict : dict
        Input dictionary containing VOCS variables. Must contain "x" key
        with the number of particles.
    libE_info : dict, optional
        LibEnsemble information dictionary containing executor and other info.

    Returns
    -------
    dict
        Output dictionary containing "energy" key with the final energy value.
    """
    assert "executor" in libE_info, "executor must be available in libE_info"

    # Extract executor from libE_info
    executor = libE_info["executor"]

    # Parse out num particles from input dictionary
    x = input_dict["x"]
    particles = str(int(x))

    # app arguments: num particles, timesteps, also using num particles as seed
    args = particles + " " + str(10) + " " + particles

    # Submit our forces app for execution.
    task = executor.submit(app_name="forces", app_args=args)

    # Block until the task finishes
    task.wait()

    # Try loading final energy reading
    statfile = "forces.stat"
    try:
        data = np.loadtxt(statfile)
        final_energy = float(data[-1])
    except Exception:
        final_energy = np.nan

    return {"energy": final_energy}
