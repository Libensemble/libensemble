#!/usr/bin/env python
# """
# Execute via one of the following commands:
#    mpiexec -np 4 python run_libE_on_warpX.py
#    python run_libE_on_warpX.py --comms local --nworkers 4

# The number of concurrent evaluations of the objective function will be 4-2=2
# as one MPI rank for the manager and one MPI rank for the persistent gen_f.
# """

import numpy as np
from warpX_simf import run_warpX  # Sim func from current dir

# Import libEnsemble modules
from libensemble.libE import libE
from libensemble.gen_funcs.sampling import uniform_random_sample as gen_f
from libensemble.alloc_funcs.give_sim_work_first import give_sim_work_first as alloc_f
from libensemble.tools import parse_args, save_libE_output, add_unique_random_streams
from libensemble import libE_logger
from libensemble.executors.mpi_executor import MPIExecutor

libE_logger.set_level('INFO')

nworkers, is_master, libE_specs, _ = parse_args()

# Set to full path of warp executable
sim_app = '$HOME/warpx/Bin/main2d.gnu.TPROF.MPI.CUDA.ex'

n = 5  # Problem dimension
exctr = MPIExecutor(central_mode=True)
exctr.register_calc(full_path=sim_app, calc_type='sim')

# State the objective function, its arguments, output, and necessary parameters (and their sizes)
sim_specs = {'sim_f': run_warpX,          # Function whose output is being minimized
             'in': ['x'],                 # Name of input for sim_f
             'out': [('f', float)],       # Name, type of output from sim_f
             'user': {'nodes': 2,
                      'ranks_per_node': 6,
                      'input': 'inputs',
                      'sim_kill_minutes': 10.0}  # Timeout for sim ....
             }

# State the generating function, its arguments, output, and necessary parameters.
gen_specs = {'gen_f': gen_f,                  # Generator function
             'in': [],                        # Generator input
             'out': [('x', float, n)],
             'user': {'gen_batch_size': 10,
                      'lb': np.zeros(n),           # Lower bound for the n parameters
                      'ub': 10*np.ones(n),         # Upper bound for the n parameters
                      }
             }

alloc_specs = {'alloc_f': alloc_f,
               'out': [('allocated', bool)],
               'user': {'batch_mode': True,    # If true wait for all sims to process before generate more
                        'num_active_gens': 1}  # Only one active generator at a time
               }

libE_specs['save_every_k_sims'] = 1     # Save each simulation evaluation
libE_specs['sim_input_dir'] = 'sim'     # Sim dir to be copied for each worker

# Maximum number of simulations
sim_max = 10
exit_criteria = {'sim_max': sim_max}

# Create a different random number stream for each worker and the manager
persis_info = add_unique_random_streams({}, nworkers + 1)

H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria,
                            persis_info, alloc_specs, libE_specs)

# Save results to numpy file
if is_master:
    save_libE_output(H, persis_info, __file__, nworkers)
