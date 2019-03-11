# """
# Runs libEnsemble on the 6-hump camel problem. Documented here:
#    https://www.sfu.ca/~ssurjano/camel6.html
#
# Execute via the following command:
#    mpiexec -np 4 python3 test_6-hump_camel_uniform_sampling.py
# The number of concurrent evaluations of the objective function will be 4-1=3.
# """

from __future__ import division
from __future__ import absolute_import

import numpy as np

from libensemble.tests.regression_tests.support import save_libE_output
from libensemble.tests.regression_tests.common import parse_args

# Parse args for test code
_, is_master, libE_specs, _ = parse_args()

# Import libEnsemble main, sim_specs, gen_specs, and persis_info
from libensemble.libE import libE
from libensemble.tests.regression_tests.support import one_d_example_sim_specs as sim_specs
from libensemble.tests.regression_tests.support import uniform_random_sample_gen_specs as gen_specs
from libensemble.tests.regression_tests.support import persis_info_0 as persis_info

# Test the following features
gen_specs['out'] = [('x',float,(1,))]
gen_specs['lb'] = np.array([-3])
gen_specs['ub'] = np.array([ 3])
gen_specs['gen_batch_size'] = 500
gen_specs['save_every_k'] = 300
exit_criteria = {'gen_max': 501}

# Perform the run
H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info, libE_specs=libE_specs)

if is_master:
    assert len(H)>= 501
    print("\nlibEnsemble with Uniform random sampling has generated enough points")
    save_libE_output(H,__file__)
