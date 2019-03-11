# """
# Runs libEnsemble on the 6-hump camel problem. Documented here:
#    https://www.sfu.ca/~ssurjano/camel6.html
#
# Execute via the following command:
#    mpiexec -np 4 python3 {FILENAME}.py
# The number of concurrent evaluations of the objective function will be 4-1=3.
# """

from __future__ import division
from __future__ import absolute_import

import numpy as np

from libensemble.tests.regression_tests.support import save_libE_output
from libensemble.tests.regression_tests.common import parse_args

# Parse args for test code
_, is_master, libE_specs, _ = parse_args()

# Import libEnsemble main, sim_specs, gen_specs, alloc_specs, and persis_info
from libensemble.libE import libE
from libensemble.tests.regression_tests.support import six_hump_camel_sim_specs as sim_specs
from libensemble.tests.regression_tests.support import uniform_or_localopt_gen_specs as gen_specs
from libensemble.tests.regression_tests.support import start_persistent_local_opt_gens_alloc_specs as alloc_specs
from libensemble.tests.regression_tests.support import persis_info_0 as persis_info

n= 2
sim_specs['out'] += [('grad',float,n)]

# State the generating function, its arguments, output, and necessary parameters.
gen_specs['out'] += [('x',float,n), ('x_on_cube',float,n),]
gen_specs['localopt_method'] = 'LD_MMA'
gen_specs['xtol_rel'] = 1e-4

# Tell libEnsemble when to stop
exit_criteria = {'sim_max': 1000, 'elapsed_wallclock_time': 300}

# Don't do a "persistent worker run" if only one wokrer
if nworkers < 2:
    quit()

# Perform the run
H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info, alloc_specs, libE_specs)

if is_master:
    assert flag == 0

    from libensemble.tests.regression_tests.support import six_hump_camel_minima as minima
    tol = 0.1
    for m in minima:
        assert np.min(np.sum((H['x']-m)**2,1)) < tol

    print("\nlibEnsemble with Uniform random sampling has identified the 6 minima within a tolerance " + str(tol))

    save_libE_output(H,__file__)
