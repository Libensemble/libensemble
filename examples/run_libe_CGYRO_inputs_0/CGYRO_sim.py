import os
import sys
import subprocess

import jinja2
import numpy as np
from heat_flux import heat_flux
from netCDF4 import Dataset

# Optional status codes to display in libE_stats.txt for each gen or sim
from libensemble.message_numbers import TASK_FAILED, WORKER_DONE


def run_CGYRO(H, persis_info, sim_specs, libE_info):
    """Launches the gx MPI app and auto-assigns ranks and GPU resources.

    Assigns one MPI rank to each GPU assigned to the worker.
    """
    calc_status = 0

    # Set inputs in input file
    input_file = sim_specs["user"]["input_filename"]
    input_values = {}
    for i, name in enumerate(sim_specs["user"]["input_names"]):
        value = H["x"][0][i]
        # if len(H["x"][0]) > 1:
        #     value = H["x"][0][i]
        # else:
        #     value = H["x"][0]
        input_values[name] = value
    with open(input_file, "r") as f:
        template = jinja2.Template(f.read())
    with open(input_file, "w") as f:
        f.write(template.render(input_values))

    nproc = sim_specs["user"]["nproc"]
    nomp = sim_specs["user"]["nomp"]
    numa = sim_specs["user"]["numa"]
    mpinuma = sim_specs["user"]["mpinuma"]
    calc_status = 0

    # Retrieve our MPI Executor
    exctr = libE_info["executor"]
    # env_script_path = "/global/homes/a/arash/bin/cgyro_libe_2"#"/global/u1/a/arash/run_libe_CGYRO_inputs/env_script_in.sh"
    os.environ["OMP_NUM_THREADS"] = "{}".format(nomp)
    # Submit our gx app for execution.

    subprocess.run(["python", "/global/cfs/cdirs/m4493/ebelli/gacode/cgyro/bin/cgyro_parse.py"])

    task = exctr.submit(
        app_name="cgyro",
        app_args="0",
        #procs_per_node=16,  # nl01
        #num_nodes=2, # nl01
        procs_per_node=4,  # reg02
        num_nodes=1, # reg02
        num_gpus=4,
        # auto_assign_gpus=True,
        # match_procs_to_gpus=True,
        # env_script= env_script_path,
        extra_args="--cpu_bind=cores,verbose -n {} -c {}".format(nproc, nomp),
    )

    # Block until the task finishes
    task.wait()

    # Try loading final energy reading, set the sim's status

    try:
        # Q=subprocess.run('python heat_flux_cgyro_libE.py', capture_output=True, text=True, shell=True)
        Q = subprocess.run(
            "python /global/u2/j/jmlarson/research/libensemble/examples/run_libe_CGYRO_inputs_0/heat_flux_cgyro_libE.py",
            capture_output=True,
            text=True,
            shell=True,
        )
        qavg = eval(Q.stdout)
        # outputs: qavg and qstd (latter not used currently)
        calc_status = WORKER_DONE
    except:
        print(f"Failed to open {fname}")
        qavg = np.nan*np.ones(2)
        calc_status = TASK_FAILED

    # Define our output array, populate with energy reading
    output = np.zeros(1, dtype=sim_specs["out"])
    
    output["fvec"] = qavg
    output["f"] = float(qavg[0]) + float(qavg[1])

    # Return final information to worker, for reporting to manager
    return output, persis_info, calc_status
