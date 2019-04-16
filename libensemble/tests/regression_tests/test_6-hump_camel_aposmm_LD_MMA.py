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


import sys             # for adding to path
import numpy as np

from libensemble.libE import libE, libE_tcp_worker
from libensemble.tests.regression_tests.common import parse_args, save_libE_output

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

# Import libEnsemble main, sim_specs, gen_specs, and persis_info
from libensemble.tests.regression_tests.support import six_hump_camel_sim_specs as sim_specs
from libensemble.tests.regression_tests.support import aposmm_with_grad_gen_specs as gen_specs
from libensemble.tests.regression_tests.support import give_sim_work_first_aposmm_alloc_specs as alloc_specs

from libensemble.tests.regression_tests.support import persis_info_1 as persis_info, give_each_worker_own_stream 
persis_info = give_each_worker_own_stream(persis_info,nworkers+1)

import copy 
persis_info_safe = copy.deepcopy(persis_info)

from math import gamma, pi, sqrt

n = 2

sim_specs['out'] += [('grad',float,n)] 


# The minima are known on this test problem.
from libensemble.tests.regression_tests.support import six_hump_camel_minima as minima
# 1) We use their values to test APOSMM has identified all minima
# 2) We use their approximate values to ensure APOSMM evaluates a point in each
#    minima's basin of attraction.

# State the generating function, its arguments, output, and necessary parameters.
gen_specs['in'] += ['x','x_on_cube']
gen_specs['out'] += [('x',float,n), ('x_on_cube',float,n),]
gen_specs['initial_sample_size'] = 100
gen_specs['sample_points'] = np.round(minima,1)
gen_specs['localopt_method'] = 'LD_MMA'
gen_specs['rk_const'] = 0.5*((gamma(1+(n/2))*5)**(1/n))/sqrt(pi)
gen_specs['xtol_rel'] = 1e-3
gen_specs['num_active_gens'] = 1
gen_specs['max_active_runs'] = 6
gen_specs['lb'] = np.array([-3,-2])
gen_specs['ub'] = np.array([ 3, 2])

# Tell libEnsemble when to stop
exit_criteria = {'sim_max': 1000}

# Perform the run (TCP worker mode)
if libE_specs['comms'] == 'tcp' and not is_master:
    run = int(sys.argv[-1])
    libE_tcp_worker(sim_specs, gen_specs[run], libE_specs)
    quit()

# Perform the run
for run in range(2):
    if libE_specs['comms'] == 'tcp' and is_master:
        libE_specs['worker_cmd'].append(str(run))

    if run == 1:
        # Change the bounds to put a local min at a corner point (to test that
        # APOSMM handles the same point being in multiple runs) ability to
        # give back a previously evaluated point)
        gen_specs['ub']= np.array([-2.9, -1.9])
        gen_specs['mu']= 1e-4
        gen_specs['rk_const']= 0.01*((gamma(1+(n/2))*5)**(1/n))/sqrt(pi)
        gen_specs['lhs_divisions'] = 2
        gen_specs.pop('batch_mode')  # Tests that APOSMM is okay being called when all pts in a run aren't completed

        gen_specs.pop('xtol_rel')
        gen_specs['ftol_rel'] = 1e-2
        gen_specs['xtol_abs'] = 1e-3
        gen_specs['ftol_abs'] = 1e-8
        exit_criteria = {'sim_max': 200, 'elapsed_wallclock_time': 300}
        minima = np.array([[-2.9, -1.9]])

        persis_info = persis_info_safe

    H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info, alloc_specs, libE_specs)

    if is_master:
        if flag != 0:
            print("Exit was not on convergence (code {})".format(flag))
            sys.stdout.flush()
            libE_abort()

        tol = 1e-5
        for m in minima:
            print(np.min(np.sum((H[H['local_min']]['x']-m)**2,1)))
            sys.stdout.flush()
            if np.min(np.sum((H[H['local_min']]['x']-m)**2,1)) > tol:
                libE_abort()

        print("\nlibEnsemble with APOSMM using a gradient-based localopt method has identified the " + str(np.shape(minima)[0]) + " minima within a tolerance " + str(tol))
        save_libE_output(H,__file__,nworkers)
