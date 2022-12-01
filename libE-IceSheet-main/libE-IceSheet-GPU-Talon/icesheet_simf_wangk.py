import numpy as np

# To retrieve our MPI Executor and resources instances
from libensemble.executors.executor import Executor
from libensemble.resources.resources import Resources

# Optional status codes to display in libE_stats.txt for each gen or sim
from libensemble.message_numbers import WORKER_DONE, TASK_FAILED


def read_stat_file(statfile, max_size):
    """Read the stat file"""

    iterations = 0
    error = 0
    if max_size:
        dVxdt = np.zeros(max_size)
        dVydt = np.zeros(max_size)
 
    i=0
    with open(statfile) as data:
        for line in data:
            if line.startswith("iter"):
                iterations = line.split(',')[0].split('=')[1]
                error = line.split(',')[1].split('=')[1]
                if max_size == 0:
                    break
            if "dVxdt" in line:
                dVxdt[i] = float(line.split(';')[0].split('=')[1].split()[0])
                #dVxdt[i] = float(line.split(';')[0].split('=')[1])
                dVydt[i] = float(line.split(';')[1].split('=')[1].split()[0])
                #dVydt[i] = float(line.split(';')[1].split('=')[1])
                i+=1
 
    if max_size:
        fvec = np.concatenate([dVxdt,dVydt])
    else:
        fvec = 0
 
    return fvec, iterations, error

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

    # # # Submit our forces app for execution. Block until the task starts.
    task = exctr.submit(
         app_name="icesheet",
         app_args=args,
         num_nodes=resources.local_node_count,
         procs_per_node=resources.slot_count,
         wait_on_start=True,
    )

    # # # Block until the task finishes
    #task.wait(timeout=60)
    task.wait()

    # Stat file to check for bad runs
    #statfile = "forces.stat"
    statfile = "icesheet.stat"#change to filename we name it in C

    # # # Try loading final energy reading, set the sim's status
    # try:
    #      data = np.loadtxt(statfile)
    #      iterations = data[-1]
    #      error = data[-1]
    #      calc_status = WORKER_DONE
    # except Exception:
    #      iterations = -1
    #    #  error = np.nan
    #      calc_status = TASK_FAILED


    # If max_size not set, then assume random sample
    max_size = sim_specs["user"].get("max_size",0)

#    _, _, error = read_stat_file(statfile, max_size)
    fvec, iterations, error = read_stat_file(statfile, max_size)

#    errorfile = "error.csv"
#    fvec,_,_ = read_stat_file(errorfile, max_size) #3.07613e-07 #np.load('error.csv') # IN C, you need to save a csv file with the error at the end of the run.
#    iterfile = "iters.csv"
#    _,iterations,_ = read_stat_file(iterfile, max_size)  #1657 #np.load('iters.csv') # IN C, you need to save a csv file with the final iteration at the end of the run.

    assert len(fvec) > 1, "Need to have a vector of errors"

    # Define our output array,  populate with energy reading
    outspecs = sim_specs["out"]
    output = np.zeros(1, dtype=outspecs)
    # output["iterations"][0] = 0
    output["fvec"][0] = fvec ##error ##3.07613e-07  ##error
    # output["fvec"][0] = np.random.uniform(0,1,100)
    #output["fvec"][0][:3] = H["x"]
    #iterations = np.random.randint(1,100)
    output["iterations"][0] = iterations
    # print(iterations)
    # output["iterations"][0] =100
    # velocity_field = np.random.uniform(-1,1,(100,100))
    # print(velocity_field[0][0])
    output["error"][0] = error
    output["f"][0] = np.sum(output["fvec"][0]**2)

    # Return final information to worker, for reporting to manager
    return output, persis_info, calc_status
