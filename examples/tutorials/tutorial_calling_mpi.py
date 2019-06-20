import numpy as np
import matplotlib.pyplot as plt
from libensemble.libE import libE
from tutorial_gen import gen_random_sample
from tutorial_sim import sim_find_sine
from mpi4py import MPI

# nworkers = 4                                  # nworkers will come from MPI
libE_specs = {'comms': 'mpi'}                   # 'nworkers' removed, 'comms' now 'mpi'

nworkers = MPI.COMM_WORLD.Get_size() - 1
is_master = (MPI.COMM_WORLD.Get_rank() == 0)    # master process has MPI rank 0

gen_specs = {'gen_f': gen_random_sample,        # Our generator function
             'in': ['sim_id'],                  # Input field names. 'sim_id' necessary default
             'out': [('x', float, (1,))],       # gen_f output (name, type, size).
             'lower': np.array([-3]),           # lower boundary for random sampling.
             'upper': np.array([3]),            # upper boundary for random sampling.
             'gen_batch_size': 5}               # number of values gen_f will generate per call

sim_specs = {'sim_f': sim_find_sine,            # Our simulator function
             'in': ['x'],                       # Input field names. 'x' from gen_f output
             'out': [('y', float)]}             # sim_f output. 'y' = sine('x')

persis_info = {}

for i in range(1, nworkers+1):                  # Worker numbers start at 1.
    persis_info[i] = {
        'rand_stream': np.random.RandomState(i),
        'worker_num': i}

exit_criteria = {'sim_max': 80}                 # Stop libEnsemble after 80 simulations

H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info,
                            libE_specs=libE_specs)

if is_master:                                   # Only the master process should execute this
    print([i for i in H.dtype.fields])          # Some (optional) statements to visualize our History array
    print(H)

    colors = ['b', 'g', 'r', 'y', 'm', 'c', 'k', 'w']

    for i in range(1, nworkers + 1):
        worker_xy = np.extract(H['sim_worker'] == i, H)
        x = [entry.tolist()[0] for entry in worker_xy['x']]
        y = [entry for entry in worker_xy['y']]
        plt.scatter(x, y, label='Worker {}'.format(i), c=colors[i-1])

    plt.title('Sine calculations for a uniformly sampled random distribution')
    plt.xlabel('x')
    plt.ylabel('sine(x)')
    plt.legend(loc='lower right')
    plt.show()
