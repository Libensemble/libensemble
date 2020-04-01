#!/usr/bin/env python
# """
# Execute via one of the following commands:
#    mpiexec -np 4 python3 run_libE_on_warpX.py

# The number of concurrent evaluations of the objective function will be 4-2=2
# as one MPI rank for the manager and one MPI rank for the persistent gen_f.
# """

import os
import numpy as np
from forces_simf import run_forces  # Sim func from current dir

# Import libEnsemble modules
from libensemble.libE import libE
from libensemble.gen_funcs.persistent_aposmm import aposmm as gen_f
from libensemble.alloc_funcs.persistent_aposmm_alloc import persistent_aposmm_alloc as alloc_f
from libensemble.tools import parse_args, save_libE_output, add_unique_random_streams
from libensemble import libE_logger

libE_logger.set_level('INFO')  # INFO is now default

nworkers, is_master, libE_specs, _ = parse_args()

sim_app = os.path.join(os.getcwd(), 'warpX.x')

# Normally would be pre-compiled
if not os.path.isfile('warpX.x'):
    if os.path.isfile('build_warpX.sh'):
        import subprocess
        subprocess.check_call(['./build_warpX.sh'])

# Normally the sim_input_dir will exist with common input which is copied for each worker. Here it starts empty.
# Create if no ./warpX dir. See libE_specs['sim_input_dir']
os.makedirs('./warpX', exist_ok=True)

n = 5  # Problem dimension
from libensemble.executors.mpi_executor import MPIExecutor
exctr = MPIExecutor()  # Use allow_oversubscribe=False to prevent oversubscription
exctr.register_calc(full_path=sim_app, calc_type='sim')

# State the objective function, its arguments, output, and necessary parameters (and their sizes)
sim_specs = {'sim_f': run_warpX,          # Function whose output is being minimized
             'in': ['x'],                 # Name of input for sim_f
             'out': [('f', float)],       # Name, type of output from sim_f
             'user': {'simdir_basename': 'warpX',
                      'cores': 2,
                      'sim_particles': 1e3}
             }
# end_sim_specs_rst_tag

gen_out = [('x', float, n), ('x_on_cube', float, n), ('sim_id', int),
           ('local_min', bool), ('local_pt', bool)]
# State the generating function, its arguments, output, and necessary parameters.
gen_specs = {'gen_f': gen_f,                  # Generator function
             'in': [],                        # Generator input
             'out': gen_out,
             'user': {'initial_sample_size': 100,
                      'localopt_method': 'LN_BOBYQA',
                      'xtol_abs': 1e-6,
                      'ftol_abs': 1e-6,
                      'lb': np.zeros(n),           # Lower bound for the n parameters
                      'ub': 10*np.ones(n),         # Upper bound for the n parameters
                      }
             }

alloc_specs = {'alloc_f': alloc_f, 'out': [('given_back', bool)], 'user': {}}

libE_specs['save_every_k_sims'] = 1     # Save each simulation evaluation
libE_specs['sim_input_dir'] = './warpX' # Sim dir to be copied for each worker

# Maximum number of simulations
sim_max = 8
exit_criteria = {'sim_max': sim_max}

# Create a different random number stream for each worker and the manager
persis_info = add_unique_random_streams({}, nworkers + 1)

H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria,
                            persis_info, alloc_specs, libE_specs)

# Save results to numpy file
if is_master:
    save_libE_output(H, persis_info, __file__, nworkers)
