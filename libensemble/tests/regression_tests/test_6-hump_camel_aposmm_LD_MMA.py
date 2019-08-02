# """
# Runs libEnsemble on the 6-hump camel problem. Documented here:
#    https://www.sfu.ca/~ssurjano/camel6.html
#
# Execute via one of the following commands (e.g. 3 workers):
#    mpiexec -np 4 python3 test_6-hump_camel_aposmm_LD_MMA.py
#
# The number of concurrent evaluations of the objective function will be 4-1=3.
# """

# Do not change these lines - they are parsed by run-tests.sh
# TESTSUITE_COMMS: mpi
# TESTSUITE_NPROCS: 2 4

import sys
import numpy as np
from math import gamma, pi, sqrt
from copy import deepcopy
from mpi4py import MPI

# Import libEnsemble items for this test
from libensemble.libE import libE, libE_tcp_worker
from libensemble.sim_funcs.six_hump_camel import six_hump_camel as sim_f
from libensemble.gen_funcs.aposmm import aposmm_logic as gen_f
from libensemble.alloc_funcs.fast_alloc_to_aposmm import give_sim_work_first as alloc_f
from libensemble.tests.regression_tests.common import parse_args, save_libE_output, per_worker_stream
from libensemble.tests.regression_tests.support import persis_info_1 as persis_info, aposmm_gen_out as gen_out, six_hump_camel_minima as minima
from time import time

nworkers, is_master, libE_specs, _ = parse_args()
np.random.seed(MPI.COMM_WORLD.Get_rank())

if is_master:
    start_time = time()

n = 2
sim_specs = {'sim_f': sim_f,
             'in': ['x'],
             'out': [('f', float), ('grad', float, n)]}

gen_out += [('x', float, n), ('x_on_cube', float, n)]
gen_specs = {'gen_f': gen_f,
             'in': [o[0] for o in gen_out] + ['f', 'grad', 'returned'],
             'out': gen_out,
             'num_active_gens': 1,
             'batch_mode': True,
             'initial_sample_size': 100,
             'sample_points': np.round(minima, 1),
             'localopt_method': 'LD_MMA',
             'rk_const': 0.5*((gamma(1+(n/2))*5)**(1/n))/sqrt(pi),
             'xtol_rel': 1e-3,
             'num_active_gens': 1,
             'max_active_runs': 6,
             'lb': np.array([-3, -2]),
             'ub': np.array([3, 2])}

alloc_specs = {'alloc_f': alloc_f, 'out': [('allocated', bool)]}

persis_info = per_worker_stream(persis_info, nworkers + 1)
persis_info_safe = deepcopy(persis_info)

exit_criteria = {'sim_max': 1000}

# Set up appropriate abort mechanism depending on comms

H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria,
                            persis_info, alloc_specs, libE_specs)

if is_master:
    print('[Manager]:', H[np.where(H['local_min'])]['x'])
    print('[Manager]: Time taken =', time()-start_time)
