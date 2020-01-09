#!/usr/bin/env python
import os
import numpy as np
from forces_simf import run_forces  # Sim func from current dir

from libensemble.libE import libE
from libensemble.gen_funcs.sampling import uniform_random_sample
from libensemble.utils import parse_args, save_libE_output, add_unique_random_streams

nworkers, is_master, libE_specs, _ = parse_args()

USE_BALSAM = False

# Create job_controller and register sim to it.
if USE_BALSAM:
    from libensemble.balsam_controller import BalsamJobController
    jobctrl = BalsamJobController()  # Use auto_resources=False to oversubscribe
else:
    from libensemble.mpi_controller import MPIJobController
    jobctrl = MPIJobController()  # Use auto_resources=False to oversubscribe

if not os.path.isdir('./sim'):
    os.mkdir('./sim')

sim_app = os.path.join(os.getcwd(), 'forces.x')
jobctrl.register_calc(full_path=sim_app, calc_type='sim')

# Note: Attributes such as kill_rate are to control forces tests, this would not be a typical parameter.

# State the objective function, its arguments, output, and necessary parameters (and their sizes)
sim_specs = {'sim_f': run_forces,         # Function whose output is being minimized
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

# State the generating function, its arguments, output, and necessary parameters.
gen_specs = {'gen_f': uniform_random_sample,  # Generator function
             'in': ['sim_id'],                # Generator input
             'out': [('x', float, (1,))],     # Name, type and size of data produced (must match sim_specs 'in')
             'user': {'lb': np.array([0]),             # Lower bound for random sample array (1D)
                      'ub': np.array([32767]),         # Upper bound for random sample array (1D)
                      'gen_batch_size': 1000,          # How many random samples to generate in one call
                      'batch_mode': True,              # If true wait for sims to process before generate more
                      'num_active_gens': 1,            # Only one active generator at a time.
                      }
             }

libE_specs['save_every_k_gens'] = 1000  # Save every K steps
libE_specs['sim_dir'] = './sim'         # Sim dir to be copied for each worker
libE_specs['profile_worker'] = False    # Whether to have libE profile on

exit_criteria = {'sim_max': 8}

persis_info = add_unique_random_streams({}, nworkers + 1)

H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info=persis_info, libE_specs=libE_specs)
