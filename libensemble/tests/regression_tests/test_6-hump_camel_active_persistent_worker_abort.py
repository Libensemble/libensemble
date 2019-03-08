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

from mpi4py import MPI    # for libE communicator
from libensemble.tests.regression_tests.support import save_libE_output

# Import libEnsemble main, sim_specs, gen_specs, alloc_specs, and persis_info
from libensemble.libE import libE
from libensemble.tests.regression_tests.support import six_hump_camel_sim_specs as sim_specs
from libensemble.tests.regression_tests.support import uniform_or_localopt_gen_specs as gen_specs
from libensemble.tests.regression_tests.support import start_persistent_local_opt_gens_alloc_specs as alloc_specs
from libensemble.tests.regression_tests.support import persis_info_0 as persis_info

n=2
gen_specs['out'] += [('x',float,n), ('x_on_cube',float,n),]
gen_specs['dist_to_bound_multiple'] = 0.5

# Tell libEnsemble when to stop
exit_criteria = {'sim_max': 10, 'elapsed_wallclock_time': 300} # Intentially set low so as to test that a worker in persistent mode can be terminated correctly

# Can't do a "persistent worker run" if only one worker
if MPI.COMM_WORLD.Get_size() == 2:
    quit()

# Perform the run
H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info, alloc_specs)

if MPI.COMM_WORLD.Get_rank() == 0:
    assert flag == 0

    save_libE_output(H,__file__)
