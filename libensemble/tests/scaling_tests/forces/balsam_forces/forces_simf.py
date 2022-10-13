import os
import time
import numpy as np

from libensemble.executors.executor import Executor
from libensemble.message_numbers import WORKER_DONE, TASK_FAILED


def run_forces_balsam(H, persis_info, sim_specs, libE_info):

    calc_status = 0

    particles = str(int(H["x"][0][0]))

    exctr = Executor.executor

    GLOBUS_ENDPOINT = sim_specs["user"]["globus_endpoint"]
    GLOBUS_DEST_DIR = sim_specs["user"]["globus_dest_dir"]
    THIS_SCRIPT_ON_THETA = sim_specs["user"]["this_script_on_theta"]

    args = {
        "sim_particles": particles,
        "sim_timesteps": str(10),
        "seed": particles,
    }

    workdir = "sim" + str(libE_info["H_rows"][0]) + "_worker" + str(libE_info["workerID"])

    statfile = f"forces{particles}.stat"

    if THIS_SCRIPT_ON_THETA:
        transfer_statfile_path = GLOBUS_DEST_DIR + statfile
        local_statfile_path = "../" + workdir + "/" + transfer_statfile_path.split("/")[-1]
    else:
        transfer_statfile_path = os.getcwd() + "/" + statfile
        local_statfile_path = transfer_statfile_path

    transfer = {"result": GLOBUS_ENDPOINT + ":" + transfer_statfile_path}

    task = exctr.submit(
        app_name="forces",
        app_args=args,
        num_procs=4,
        num_nodes=1,
        procs_per_node=4,
        max_tasks_per_node=1,
        transfers=transfer,
        workdir=workdir,
    )

    task.wait(timeout=300)
    task.poll()

    print(f"Task {task.name} polled. state: {task.state}.")

    while True:
        time.sleep(1)
        if os.path.isfile(local_statfile_path) and os.path.getsize(local_statfile_path) > 0:
            break

    try:
        data = np.loadtxt(local_statfile_path)
        final_energy = data[-1]
        calc_status = WORKER_DONE
    except Exception:
        final_energy = np.nan
        calc_status = TASK_FAILED

    outspecs = sim_specs["out"]
    output = np.zeros(1, dtype=outspecs)
    output["energy"][0] = final_energy

    return output, persis_info, calc_status
