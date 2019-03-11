# """
# Runs libEnsemble on the 6-hump camel problem. Documented here:
#    https://www.sfu.ca/~ssurjano/camel6.html
#
# Execute via the following command:
#    mpiexec -np 4 python3 test_6-hump_camel_aposmm_LD_MMA.py
# The number of concurrent evaluations of the objective function will be 4-1=3.
# """

from __future__ import division
from __future__ import absolute_import

import sys, os             # for adding to path
import numpy as np

from libensemble.libE import libE, libE_tcp_worker
from libensemble.tests.regression_tests.common import parse_args

# Parse args for test code
nworkers, is_master, libE_specs, _ = parse_args()
if libE_specs['comms'] != 'mpi':
    quit()

# Set up appropriate abort mechanism depending on comms
libE_abort = quit
if libE_specs['comms'] == 'mpi':
    from mpi4py import MPI
    def libE_mpi_abort():
        MPI.COMM_WORLD.Abort(1)
    libE_abort = libE_mpi_abort

# Import sim_func
from libensemble.sim_funcs.six_hump_camel import six_hump_camel

# Import gen_func
from libensemble.gen_funcs.aposmm import aposmm_logic

# Import alloc_func
from libensemble.alloc_funcs.fast_alloc_to_aposmm import give_sim_work_first as alloc_f

from math import gamma, pi, sqrt
script_name = os.path.splitext(os.path.basename(__file__))[0]

n = 2

#State the objective, arguments, output, and necessary parameters (and  sizes)
sim_specs = {'sim_f': six_hump_camel, # Objective function
             'in': ['x'],             # These keys will be given to objective
             'out': [('f',float),('grad',float,n), # sim outputs
                    ],
             }

gen_out = [('x',float,n),
      ('x_on_cube',float,n),
      ('sim_id',int),
      ('priority',float),
      ('local_pt',bool),
      ('paused',bool),
      ('known_to_aposmm',bool), # Mark known points so fewer updates are needed.
      ('dist_to_unit_bounds',float),
      ('dist_to_better_l',float),
      ('dist_to_better_s',float),
      ('ind_of_better_l',int),
      ('ind_of_better_s',int),
      ('started_run',bool),
      ('num_active_runs',int), # Number of active runs point is involved in
      ('local_min',bool),
      ('pt_id',int), # To be used by APOSMM to identify points evaluated by different simulations
      ]


# The minima are known on this test problem.
# 1) We use their values to test APOSMM has identified all minima
# 2) We use their approximate values to ensure APOSMM evaluates a point in each
#    minima's basin of attraction.
minima = [np.array([[ -0.089842,  0.712656],
                    [  0.089842, -0.712656],
                    [ -1.70361,  0.796084],
                    [  1.70361, -0.796084],
                    [ -1.6071,   -0.568651],
                    [  1.6071,    0.568651]]),
          np.array([[-2.9, -1.9]])]

# State the generating function, arguments, output, and parameters.
gen_specs = [{'gen_f': aposmm_logic,
              'in': [o[0] for o in gen_out] + ['f', 'grad', 'returned'],
              'out': gen_out,
              'lb': np.array([-3,-2]),
              'ub': np.array([ 3, 2]),
              'initial_sample_size': 100,
              'sample_points': np.round(minima[0],1),
              'localopt_method': 'LD_MMA',
              'rk_const': 0.5*((gamma(1+(n/2))*5)**(1/n))/sqrt(pi),
              'xtol_rel': 1e-3,
              'num_active_gens':1,
              'max_active_runs':6,
              },
             {'gen_f': aposmm_logic,
              'in': [o[0] for o in gen_out] + ['f', 'grad', 'returned'],
              'out': gen_out,
              'lb': np.array([-3,-2]),
              'ub': np.array([-2.9, -1.9]),
              'initial_sample_size': 100,
              'sample_points': np.round(minima[1],1),
              'localopt_method': 'LD_MMA',
              'rk_const': 0.01*((gamma(1+(n/2))*5)**(1/n))/sqrt(pi),
              'ftol_rel': 1e-2,
              'xtol_abs': 1e-3,
              'ftol_abs': 1e-8,
              'num_active_gens':1,
              'max_active_runs':6,
              'mu': 1e-4,
              'lhs_divisions': 2
              }]

# Tell libEnsemble when to stop
exit_criteria = [{'sim_max': 1000},
                 {'sim_max': 200, 'elapsed_wallclock_time': 300}]


alloc_specs = {'out':[('allocated',bool)], 'alloc_f':alloc_f}


# Perform the run (TCP worker mode)
if libE_specs['comms'] == 'tcp' and not is_master:
    run = int(sys.argv[-1])
    libE_tcp_worker(sim_specs, gen_specs[run], libE_specs)
    quit()


# Perform the run
for run in range(2):
    if libE_specs['comms'] == 'tcp' and is_master:
        libE_specs['worker_cmd'].append(str(run))

    np.random.seed(1)

    persis_info = {'next_to_give':0}
    persis_info['total_gen_calls'] = 0
    persis_info['last_worker'] = 0
    persis_info[0] = {'run_order': {},
                      'old_runs': {},
                      'total_runs': 0,
                      'rand_stream': np.random.RandomState(1)}

    # Making persis_info fields to store APOSMM information, but will be passed
    # to various workers.

    for i in range(1,nworkers+1):
        persis_info[i] = {'rand_stream': np.random.RandomState(i)}

    H, persis_info, flag = libE(sim_specs, gen_specs[run], exit_criteria[run],
                                persis_info, alloc_specs, libE_specs)

    if is_master:

        if flag != 0:
            print("Exit was not on convergence (code {})".format(flag))
            sys.stdout.flush()
            libE_abort()

        short_name = script_name.split("test_", 1).pop()
        filename = short_name + '_results_History_length=' + str(len(H)) + '_evals=' + str(sum(H['returned'])) + '_ranks=' + str(nworkers+1)
        print("\n\n\nRun completed.\nSaving results to file: " + filename)
        np.save(filename, H)

        tol = 1e-5
        for m in minima[run]:
            print(np.min(np.sum((H[H['local_min']]['x']-m)**2,1)))
            sys.stdout.flush()
            if np.min(np.sum((H[H['local_min']]['x']-m)**2,1)) > tol:
                libE_abort()

        print("\nlibEnsemble with APOSMM using a gradient-based localopt method has identified the " + str(np.shape(minima[run])[0]) + " minima within a tolerance " + str(tol))
