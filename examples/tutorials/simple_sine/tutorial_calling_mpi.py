import numpy as np
import matplotlib.pyplot as plt
from libensemble.libE import libE
from libensemble.tools import add_unique_random_streams
from tutorial_gen import gen_random_sample
from tutorial_sim import sim_find_sine
from mpi4py import MPI

libE_specs = {"comms": "mpi"}  # 'nworkers' removed, 'comms' now 'mpi'

nworkers = MPI.COMM_WORLD.Get_size() - 1  # one process belongs to manager
is_manager = MPI.COMM_WORLD.Get_rank() == 0  # manager process has MPI rank 0

gen_specs = {
    "gen_f": gen_random_sample,  # Our generator function
    "out": [("x", float, (1,))],  # gen_f output (name, type, size).
    "user": {
        "lower": np.array([-3]),  # random sampling lower bound
        "upper": np.array([3]),  # random sampling upper bound
        "gen_batch_size": 5,  # number of values gen_f will generate per call
    },
}

sim_specs = {
    "sim_f": sim_find_sine,  # Our simulator function
    "in": ["x"],  # Input field names. 'x' from gen_f output
    "out": [("y", float)],  # sim_f output. 'y' = sine('x')
}

persis_info = add_unique_random_streams({}, nworkers + 1)  # Initialize manager/workers random streams

exit_criteria = {"sim_max": 80}  # Stop libEnsemble after 80 simulations

H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info, libE_specs=libE_specs)

# Some (optional) statements to visualize our History array
# Only the manager process should execute this
if is_manager:
    print([i for i in H.dtype.fields])
    print(H)

    colors = ["b", "g", "r", "y", "m", "c", "k", "w"]

    for i in range(1, nworkers + 1):
        worker_xy = np.extract(H["sim_worker"] == i, H)
        x = [entry.tolist()[0] for entry in worker_xy["x"]]
        y = [entry for entry in worker_xy["y"]]
        plt.scatter(x, y, label=f"Worker {i}", c=colors[i - 1])

    plt.title("Sine calculations for a uniformly sampled random distribution")
    plt.xlabel("x")
    plt.ylabel("sine(x)")
    plt.legend(loc="lower right")
    plt.savefig("tutorial_sines.png")
