"""
Example using NSGA2 as a libE generator function using the libE yaml interface
For more about NSGA2, see
https://gist.github.com/darden1/fa8f96185a46796ed9516993bfe24862
"""

# Do not change these lines - they are parsed by run-tests.sh
# TESTSUITE_COMMS: mpi local
# TESTSUITE_NPROCS: 3 4
# TESTSUITE_EXTRA: true

import numpy as np
from time import time
import os
from libensemble import Ensemble
from libensemble.sim_funcs.six_hump_camel import six_hump_camel_func


def deap_six_hump(H, persis_info, sim_specs, _):
    xvals = H['individual'][0]
    Out = np.zeros(1, dtype=sim_specs['out'])
    y0 = np.linalg.norm(xvals)
    y1 = six_hump_camel_func(xvals)
    Out['fitness_values'] = (y0, y1)  # Requires tuple, even with 1 objective

    return Out, persis_info


if __name__ == "__main__":

    deap_test = Ensemble()
    deap_test.from_yaml('deap_nsga2.yaml')

    if deap_test.is_manager:
        start_time = time()

    assert deap_test.nworkers >= 2, "Cannot run with a persistent gen_f if only one worker."

    ind_size = 2
    w = (-1.0, -1.0)  # Must be a tuple

    deap_test.gen_specs['user'].update(
        {
            'weights': w,
            'indpb': 0.8 / ind_size,
        }
    )

    lb = deap_test.gen_specs['user']['lb']
    ub = deap_test.gen_specs['user']['ub']

    for run in range(2):

        deap_test.persis_info.add_random_streams()

        if run == 1:

            # Number of points in the sample
            num_samp = 100

            H0_dtype = [
                ('individual', float, ind_size),
                ('generation', int),
                ('fitness_values', float, 2),
                ('sim_id', int),
                ('returned', bool),
                ('given_back', bool),
                ('given', bool),
            ]

            H0 = np.zeros(num_samp, dtype=H0_dtype)

            # Mark these points as already have been given to be evaluated, and returned, but not given_back.
            H0[['given', 'given_back', 'returned']] = True
            H0['generation'][:] = 1
            # Give these points sim_ids
            H0['sim_id'] = range(num_samp)

            # "Load in" the points and their function values. (In this script, we are
            # actually evaluating them, but in many cases, they are available from past
            # evaluations
            np.random.seed(0)
            H0['individual'] = np.random.uniform(lb, ub, (num_samp, len(lb)))
            for i, x in enumerate(H0['individual']):
                H_dummy = np.zeros(1, dtype=[('individual', float, ind_size)])
                H_dummy['individual'] = x
                objs = deap_six_hump(H_dummy, {}, deap_test.sim_specs, {})
                H0['fitness_values'][i] = objs[0]

            # Testing use_persis_return_gen capabilities
            deap_test.libE_specs['use_persis_return_gen'] = True
            deap_test.H0 = H0
        else:
            deap_test.H0 = None

        # Perform the run
        deap_test.run()

        if deap_test.is_manager:
            if run == 0:
                assert np.sum(deap_test.H['last_points']) == 0, (
                    "The last_points shouldn't be marked (even though "
                    "they were marked in the gen) as 'use_persis_return_gen' was false."
                )
            elif run == 1:
                assert np.sum(deap_test.H['last_points']) == 100, (
                    "The last_points should be marked as true because they "
                    "were marked in the manager and 'use_persis_return_gen' is true."
                )

            script_name = os.path.splitext(os.path.basename(__file__))[0]
            assert deap_test.flag == 0, script_name + " didn't exit correctly"
            assert sum(deap_test.H['returned']) >= deap_test.exit_criteria['sim_max'], (
                script_name + " didn't evaluate the sim_max points."
            )
            assert min(deap_test.H['fitness_values'][:, 0]) <= 4e-3, (
                script_name + " didn't find the minimum for objective 0."
            )
            assert min(deap_test.H['fitness_values'][:, 1]) <= -1.0, (
                script_name + " didn't find the minimum for objective 1."
            )
