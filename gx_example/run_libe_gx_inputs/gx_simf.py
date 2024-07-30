import jinja2
import numpy as np
from heat_flux import heat_flux
from netCDF4 import Dataset

# Optional status codes to display in libE_stats.txt for each gen or sim
from libensemble.message_numbers import TASK_FAILED, WORKER_DONE


def run_gx(H, persis_info, sim_specs, libE_info):
    """Launches the gx MPI app and auto-assigns ranks and GPU resources.

    Assigns one MPI rank to each GPU assigned to the worker.
    """

    # Set inputs in input file
    input_file = sim_specs["user"]["input_filename"]
    input_values = {}
    for i, name in enumerate(sim_specs["user"]["input_names"]):
        value = H["x"][0][i]
        input_values[name] = value
    with open(input_file, "r") as f:
        template = jinja2.Template(f.read())
    with open(input_file, "w") as f:
        f.write(template.render(input_values))

    calc_status = 0

    # Retrieve our MPI Executor
    exctr = libE_info["executor"]

    plot = sim_specs["user"]["plot_heat_flux"]

    # Submit our gx app for execution.
    task = exctr.submit(
        app_name="gx",
        app_args=input_file,
        auto_assign_gpus=True,
        match_procs_to_gpus=True,
    )

    # Block until the task finishes
    task.wait()

    # Try loading final energy reading, set the sim's status
    fname = "cyclone.out.nc"

    try:
        data = Dataset(fname, mode='r')
        nspec = data.dimensions['s'].size
        for ispec in np.arange(nspec):
            plot_hf = sim_specs["user"]["plot_heat_flux"]
            qavg, _ = heat_flux(data, ispec=ispec, refsp="i", plot=plot_hf)
            # outputs: qavg and qstd (latter not used currently)
            calc_status = WORKER_DONE
    except:
        print(f'Failed to open {fname}')
        qavg = np.nan
        calc_status = TASK_FAILED

    # Define our output array, populate with energy reading
    output = np.zeros(1, dtype=sim_specs["out"])

    output["f"] = qavg

    # Return final information to worker, for reporting to manager
    return output, persis_info, calc_status
