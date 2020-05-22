# Example using NSGA2 as a libE generator function:
# https://gist.github.com/darden1/fa8f96185a46796ed9516993bfe24862
#
# """

# Do not change these lines - they are parsed by run-tests.sh
# TESTSUITE_COMMS: mpi local tcp
# TESTSUITE_NPROCS: 3 4

import numpy as np
from time import time
import os
from libensemble.libE import libE
from libensemble.alloc_funcs.start_only_persistent import only_persistent_gens as alloc_f
from libensemble.tools import parse_args, add_unique_random_streams
from libensemble.sim_funcs.six_hump_camel import six_hump_camel_func
from libensemble.gen_funcs.persistent_deap_nsga2 import deap_nsga2 as gen_f

nworkers, is_master, libE_specs, _ = parse_args()


def deap_six_hump(H, persis_info, sim_specs, _):
    xvals = H['individual'][0]
    Out = np.zeros(1, dtype=sim_specs['out'])
    y1 = six_hump_camel_func(xvals)
    Out['fitness_values'] = (y1,)  # Requires tuple

    return Out, persis_info


if is_master:
    start_time = time()

assert nworkers >= 2, "Cannot run with a persistent gen_f if only one worker."

# Number of generations, population size, indiviual size, and objectives
ngen = 30
pop_size = 80
ind_size = 2
num_objectives = 1

# Variable Bounds (deap requires lists, not arrays!!!)
lb = [-3.0, -2.0]
ub = [3.0, 2.0]
w = (-1.0,)  # Must be a tuple

# State the objective function, its arguments, output, and necessary parameters (and their sizes)
sim_specs = {'sim_f': deap_six_hump,  # This is the function whose output is being minimized
             'in': ['individual'],  # These keys will be given to the above function
             'out': [('fitness_values', float)]  # This output is being minimized
             }  # end of sim spec

# State the generating function, its arguments, output, and necessary parameters.
gen_specs = {'gen_f': gen_f,
             'in': ['sim_id'],
             'out': [('individual', float, ind_size), ('generation', int)],
             'user': {'lb': lb,
                      'ub': ub,
                      'weights': w,
                      'pop_size': pop_size,
                      'indiv_size': ind_size,
                      'cxpb': 0.8,  # probability two individuals are crossed
                      'eta': 20.0,  # large eta = low variation in children
                      'indpb': 0.8/ind_size}  # end user
             }  # end gen specs

# libE Allocation function
alloc_specs = {'out': [('given_back', bool)], 'alloc_f': alloc_f}
persis_info = add_unique_random_streams({}, nworkers + 1)

# Tell libEnsemble when to stop
# 'sim_max' = number of simulation calls
# For deap, this should be pop_size*number of generations+1
exit_criteria = {'sim_max': pop_size*(ngen+1)}


# Perform the run
H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info, alloc_specs, libE_specs)

if is_master:
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    assert flag == 0, script_name + " didn't exit correctly"
    assert sum(H['returned']) >= exit_criteria['sim_max'], script_name + " didn't evaluate the sim_max points."
    assert min(H['fitness_values']) <= -1.0315, script_name + " didn't find the global minimum for this problem."
