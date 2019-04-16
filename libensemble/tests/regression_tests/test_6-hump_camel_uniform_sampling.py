# """
# Runs libEnsemble on the 6-hump camel problem. Documented here:
#    https://www.sfu.ca/~ssurjano/camel6.html
#
# Execute via the following command:
#    mpiexec -np 4 python3 test_6-hump_camel_uniform_sampling.py
# The number of concurrent evaluations of the objective function will be 4-1=3.
# """
import numpy as np

from libensemble.tests.regression_tests.common import parse_args, save_libE_output
nworkers, is_master, libE_specs, _ = parse_args()

# Import libEnsemble main, sim_specs, gen_specs, and persis_info
from libensemble.libE import libE
from libensemble.tests.regression_tests.support import six_hump_camel_sim_specs as sim_specs
from libensemble.tests.regression_tests.support import uniform_random_sample_gen_specs as gen_specs

from libensemble.tests.regression_tests.support import give_each_worker_own_stream 
persis_info = give_each_worker_own_stream({},nworkers+1)

sim_specs['save_every_k'] = 400
gen_specs['gen_batch_size'] = 500
gen_specs['save_every_k'] =  300
gen_specs['out'] = [('x',float,(2,))]
gen_specs['lb'] = np.array([-3,-2])
gen_specs['ub'] = np.array([ 3, 2])

# Tell libEnsemble when to stop
exit_criteria = {'gen_max': 501, 'elapsed_wallclock_time': 300}

# Perform the run
H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info, libE_specs=libE_specs)

if is_master:
    assert flag == 0
    from libensemble.tests.regression_tests.support import six_hump_camel_minima as minima

    tol = 0.1
    for m in minima:
        assert np.min(np.sum((H['x']-m)**2,1)) < tol

    print("\nlibEnsemble with Uniform random sampling has identified the 6 minima within a tolerance " + str(tol))
    save_libE_output(H,__file__,nworkers)
