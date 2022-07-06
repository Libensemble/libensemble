import numpy as np

# To retrieve our MPI Executor and resources instances
from libensemble.executors.executor import Executor
from libensemble.resources.resources import Resources

# Optional status codes to display in libE_stats.txt for each gen or sim
from libensemble.message_numbers import WORKER_DONE, TASK_FAILED


def run_icesheet(H, persis_info, sim_specs, libE_info):
    calc_status = 0

    # Parse out num particles, from generator function
    damp = H["x"][0][0]
    velo = H["x"][0][1]
    visc = H["x"][0][2]

    # app arguments: num particles, timesteps, also using num particles as seed
    args = str(damp) + " " + str(velo) + " " + str(visc)

    print(args)

    # Retrieve our MPI Executor instance and resources
    exctr = Executor.executor
    resources = Resources.resources.worker_resources

    resources.set_env_to_slots("CUDA_VISIBLE_DEVICES")

    # # Submit our forces app for execution. Block until the task starts.
    task = exctr.submit(
         app_name="icesheet",
         app_args=args,
         num_nodes=resources.local_node_count,
         procs_per_node=resources.slot_count,
         wait_on_start=True,
    )

    # # Block until the task finishes
    task.wait(timeout=60)

    # Stat file to check for bad runs
    #statfile = "forces.stat"
    statfile = "icesheet.stat"#change to filename we name it in C

    # # Try loading final energy reading, set the sim's status
    try:
         data = np.loadtxt(statfile)
         iterations = data[-1]
         error = data[-1]
         calc_status = WORKER_DONE
    except Exception:
         iterations = -1
       #  error = np.nan
         calc_status = TASK_FAILED

    # Define our output array,  populate with energy reading
    outspecs = sim_specs["out"]
    output = np.zeros(1, dtype=outspecs)
    output["f"][0] =iterations
   # iterations = np.random.randint(1,100)
   # print(iterations)
   # output["iterations"][0] =iterations 
   # velocity_field = np.random.uniform(-1,1,(100,100))
   # print(velocity_field[0][0])
    output["error"][0] = error

    # Return final information to worker, for reporting to manager
    return output, persis_info, calc_status
