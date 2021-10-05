"""
Example of optimization using a persistent GP gen_func, with multi-fidelity

Usage:
------
python run_example.py --comms local --nworkers 4
"""

import numpy as np
from libensemble.libE import libE
from libensemble import logger
from libensemble.alloc_funcs.start_only_persistent import only_persistent_gens
from libensemble.tools import save_libE_output, add_unique_random_streams
from libensemble.tools import parse_args
from libensemble.message_numbers import WORKER_DONE
from libensemble.gen_funcs.persistent_gp import persistent_gp_mf_gen_f

nworkers, is_master, libE_specs, _ = parse_args()


def run_simulation(H, persis_info, sim_specs, libE_info):
    # Extract input parameters
    values = list(H['x'][0])
    x0 = values[0]
    x1 = values[1]
    # Extract fidelity parameter
    z = H['z'][0]

    libE_output = np.zeros(1, dtype=sim_specs['out'])
    calc_status = WORKER_DONE

    # Function that depends on the resolution parameter
    libE_output['f'] = -(x0 + 10 * np.cos(x0 + 0.1 * z)) * (x1 + 5 * np.cos(x1 - 0.2 * z))

    return libE_output, persis_info, calc_status


sim_specs = {
    'sim_f': run_simulation,
    'in': ['x', 'z'],
    'out': [('f', float)], }

gen_specs = {
    # Generator function. Will randomly generate new sim inputs 'x'.
    'gen_f': persistent_gp_mf_gen_f,
    # Generator input. This is a RNG, no need for inputs.
    'in': ['sim_id', 'x', 'f', 'z'],
    'persis_in': ['sim_id', 'x', 'f', 'z'],
    'out': [
        # parameters to input into the simulation.
        ('x', float, (2, )),
        ('z', float),
        ('resource_sets', int)],
    'user': {
        'range': [1, 8],
        'cost_func': lambda z: z[0],
        # Total max number of sims running concurrently.
        'gen_batch_size': nworkers - 1,
        # Lower bound for the n parameters.
        'lb': np.array([0, 0]),
        # Upper bound for the n parameters.
        'ub': np.array([15, 15])}}

alloc_specs = {
    'alloc_f': only_persistent_gens,
    'out': [('given_back', bool)],
    'user': {
        'async_return': True}, }

# libE logger
logger.set_level('INFO')

# Exit criteria
exit_criteria = {'sim_max': 10}  # Exit after running sim_max simulations

# Create a different random number stream for each worker and the manager
persis_info = add_unique_random_streams({}, nworkers + 1)

# Run LibEnsemble, and store results in history array H
H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info, alloc_specs, libE_specs)

# Save results to numpy file
if is_master:
    save_libE_output(H, persis_info, __file__, nworkers)
