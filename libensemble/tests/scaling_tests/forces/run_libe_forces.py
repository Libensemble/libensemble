#!/usr/bin/env python
import os
import numpy as np
from forces_simf import run_forces  # Sim func from current dir

# Import libEnsemble modules
from libensemble.libE import libE
from libensemble.gen_funcs.sampling import uniform_random_sample
from libensemble.utils import parse_args, save_libE_output, add_unique_random_streams

from libensemble import libE_logger
libE_logger.set_level('INFO')  # Info is now default - but shows usage.

USE_BALSAM = False

nworkers, is_master, libE_specs, _ = parse_args()

if is_master:
    print('\nRunning with {} workers\n'.format(nworkers))

# Get this script name (for output at end)
script_name = os.path.splitext(os.path.basename(__file__))[0]

sim_app = os.path.join(os.getcwd(), 'forces.x')
# print('sim_app is ', sim_app)

# Normally would be pre-compiled
if not os.path.isfile('forces.x'):
    if os.path.isfile('build_forces.sh'):
        import subprocess
        subprocess.check_call(['./build_forces.sh'])

# Normally the sim_dir will exist with common input which is copied for each worker. Here it starts empty.
# Create if no ./sim dir. See libE_specs['sim_dir']
if not os.path.isdir('./sim'):
    os.mkdir('./sim')

# Create job_controller and register sim to it.
if USE_BALSAM:
    from libensemble.balsam_controller import BalsamJobController
    jobctrl = BalsamJobController()  # Use auto_resources=False to oversubscribe
else:
    from libensemble.mpi_controller import MPIJobController
    jobctrl = MPIJobController()  # Use auto_resources=False to oversubscribe
jobctrl.register_calc(full_path=sim_app, calc_type='sim')

# Note: Attributes such as kill_rate are to control forces tests, this would not be a typical parameter.

# State the objective function, its arguments, output, and necessary parameters (and their sizes)
sim_specs = {'sim_f': run_forces,         # Function whose output is being minimized
             'in': ['x'],                 # Name of input for sim_f
             'out': [('energy', float)],  # Name, type of output from sim_f
             'user': {'simdir_basename': 'forces',
                      'keys': ['seed'],
                      'cores': 2,
                      'sim_particles': 1e3,
                      'sim_timesteps': 5,
                      'sim_kill_minutes': 10.0,
                      'particle_variance': 0.2,
                      'kill_rate': 0.5}   # Used by this specific sim_f
             }
# end_sim_specs_rst_tag

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

# Maximum number of simulations
sim_max = 8
exit_criteria = {'sim_max': sim_max}

# Create a different random number stream for each worker and the manager
persis_info = {}
persis_info = add_unique_random_streams(persis_info, nworkers + 1)

H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info=persis_info, libE_specs=libE_specs)

# Save results to numpy file
if is_master:
    save_libE_output(H, persis_info, __file__, nworkers)
