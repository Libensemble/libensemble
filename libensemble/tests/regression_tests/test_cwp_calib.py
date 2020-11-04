# """
# Runs libEnsemble with cwp calibration test.
#
# Execute via one of the following commands (e.g. 3 workers):
#    mpiexec -np 4 python3 test_6-test_cwp_calib.py
#    python3 test_cwp_calib.py --nworkers 3 --comms local
#    python3 test_cwp_calib.py --nworkers 3 --comms tcp
#
# The number of concurrent evaluations of the objective function will be 4-1=3.
# """

# Do not change these lines - they are parsed by run-tests.sh
# TESTSUITE_COMMS: mpi local tcp
# TESTSUITE_NPROCS: 3 4

# Requires:
#   Clone cwpcalibration repo
#   pip install functionbase package.

# NOTE (REMOVE WHEN FIXED. CURRENTLY EMULATION ERROR AFTER 8 ITERATIONS).
# TODO for step 1:
#    Rename files/vars as required.
#    Determine pass condition for test (assertions at end).

# import numpy as np

# Import libEnsemble items for this test
from libensemble.libE import libE

from libensemble.gen_funcs.persistent_cwp_calib import testcalib as gen_f
from libensemble.alloc_funcs.start_only_persistent import only_persistent_gens as alloc_f
from libensemble.sim_funcs.cwpsim import borehole as sim_f
from libensemble.tools import parse_args, save_libE_output, add_unique_random_streams
from libensemble.executors.mpi_executor import MPIExecutor

# from libensemble import libE_logger
# libE_logger.set_level('DEBUG')  # To get debug logging in ensemble.log

if __name__ == '__main__':

    exctr = MPIExecutor()

    nworkers, is_master, libE_specs, _ = parse_args()

    sim_specs = {'sim_f': sim_f,
                 'in': ['x', 'thetas', 'quantile'],
                 'out': [('f', float), ('failures', bool)],
                 'user': {'kill_sim_test': True}}

    n_test_thetas = 100             # No. of thetas for test data
    n_init_thetas = 25              # Initial batch of thetas
    n_x = 5                         # No. of x values
    nparams = 6                     # No. of theta params
    ndims = 3                       # No. of x co-ordinates.
    max_emul_runs = 50              # Max no. of runs of emulator
    # mse_exit = 1.0                  # MSE threshold for exiting
    step_add_theta = 5              # No. of thetas to generate per step, before emulator is rebuilt
    n_explore_theta = 2000          # No. of thetas to explore while selecting the next theta
    build_emul_on_thread = True     # Build emul on background thread
    errstd_constant = 0.00005       # Constant for generating noise in obs
    quantile_to_failure = 0.95      # Proportion of particles that succeed

    # Stop after max_emul_runs runs of the emulator
    max_evals = (n_init_thetas + n_test_thetas + 1) * n_x + max_emul_runs*n_x
    # print('max_evals is {}'.format(max_evals),flush=True)

    # batch mode until after batch_sim_id
    batch_sim_id = (n_init_thetas + n_test_thetas + 1) * n_x

    gen_out = [('x', float, ndims), ('thetas', float, nparams), ('mse', float, (1,)),
               ('quantile', float), ('priority', int), ('obs', float, n_x), ('errstd', float, n_x)]
    gen_specs = {'gen_f': gen_f,
                 'in': [o[0] for o in gen_out]+['f', 'failures', 'returned'],
                 'out': gen_out,
                 'user': {'n_test_thetas': n_test_thetas,        # Num test thetas
                          'n_init_thetas': n_init_thetas,        # Num thetas in initial batch
                          'num_x_vals': n_x,                     # Num x points to create
                          # 'mse_exit': mse_exit,                # Threshold for exit
                          'step_add_theta': step_add_theta,      # No. of thetas to generate per step
                          'n_explore_theta': n_explore_theta,    # No. of thetas to explore each step
                          'async_build': build_emul_on_thread,   # Build emul on background thread
                          'errstd_constant': errstd_constant,    # Constant for generating noise in obs
                          'batch_to_sim_id': batch_sim_id,       # Up to this sim_id, wait for all results to return.
                          'ignore_cancelled': True,              # Do not use returned results that have been cancelled
                          'quantile': quantile_to_failure        # Proportion of paricles that succeed
                          }
                 }

    # alloc_specs = {'alloc_f': alloc_f, 'out': [('given_back', bool)], 'user': {'batch_mode': True}}
    alloc_specs = {'alloc_f': alloc_f,
                   'out': [('given_back', bool)],
                   'user': {'batch_to_sim_id': batch_sim_id,
                            'batch_mode': False  # set batch mode (alloc behavior after batch_sim_id)
                            }
                   }

    persis_info = add_unique_random_streams({}, nworkers + 1)

    # Currently just allow gen to exit if mse goes below threshold value
    exit_criteria = {'sim_max': max_evals}
    # exit_criteria = {'sim_max': max_evals, 'stop_val': ('mse', mse_exit)}

    # Perform the run
    H, persis_info, flag = libE(sim_specs, gen_specs,
                                exit_criteria, persis_info,
                                alloc_specs=alloc_specs,
                                libE_specs=libE_specs)

    if is_master:
        print('Cancelled sims', H['sim_id'][H['cancel']])
        print('Killed sims', H['sim_id'][H['kill_sent']])
        # MC: Clean up of unreturned
        # assert np.all(H['returned'])
        save_libE_output(H, persis_info, __file__, nworkers)
