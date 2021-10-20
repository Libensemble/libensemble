#!/usr/bin/env python
import os
import numpy as np
from tutorial_forces_simf import run_forces  # Sim func from current dir

from libensemble.libE import libE
from libensemble.gen_funcs.sampling import uniform_random_sample
from libensemble.tools import parse_args, add_unique_random_streams
from libensemble.executors.mpi_executor import MPIExecutor

nworkers, is_manager, libE_specs, _ = parse_args()  # Convenience function

# Create executor and register sim to it
exctr = MPIExecutor()

# Register simulation executable with executor
sim_app = os.path.join(os.getcwd(), 'forces.x')
exctr.register_app(full_path=sim_app, app_name='forces')

# State the sim_f, its arguments, output, and parameters (and their sizes)
sim_specs = {'sim_f': run_forces,         # sim_f, imported above
             'in': ['x'],                 # Name of input for sim_f
             'out': [('energy', float)],  # Name, type of output from sim_f
             'user': {'simdir_basename': 'forces',  # User parameters for the sim_f
                      'keys': ['seed'],
                      'cores': 2,
                      'sim_particles': 1e3,
                      'sim_timesteps': 5,
                      'sim_kill_minutes': 10.0,
                      'particle_variance': 0.2,
                      'kill_rate': 0.5}
             }

# State the gen_f, its arguments, output, and necessary parameters.
gen_specs = {'gen_f': uniform_random_sample,  # Generator function
             'in': ['sim_id'],                # Generator input
             'out': [('x', float, (1,))],     # Name, type and size of data from gen_f
             'user': {'lb': np.array([0]),             # User parameters for the gen_f
                      'ub': np.array([32767]),
                      'gen_batch_size': 1000,
                      'batch_mode': True,
                      'num_active_gens': 1,
                      }
             }

libE_specs['save_every_k_gens'] = 1000  # Save every K steps
libE_specs['sim_dirs_make'] = True

exit_criteria = {'sim_max': 8}

persis_info = add_unique_random_streams({}, nworkers + 1)

H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria,
                            persis_info=persis_info, libE_specs=libE_specs)
