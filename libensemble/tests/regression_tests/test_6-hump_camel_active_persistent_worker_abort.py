# """
# Runs libEnsemble on the 6-hump camel problem. Documented here:
#    https://www.sfu.ca/~ssurjano/camel6.html
#
# Execute via the following command:
#    mpiexec -np 4 python3 {FILENAME}.py
# The number of concurrent evaluations of the objective function will be 4-1=3.
# """

from libensemble.tests.regression_tests.common import parse_args, save_libE_output

# Parse args for test code
nworkers, is_master, libE_specs, _ = parse_args()

# Import libEnsemble main, sim_specs, gen_specs, alloc_specs, and persis_info
from libensemble.libE import libE
from libensemble.tests.regression_tests.support import six_hump_camel_sim_specs as sim_specs
from libensemble.tests.regression_tests.support import uniform_or_localopt_gen_specs as gen_specs
from libensemble.tests.regression_tests.support import start_persistent_local_opt_gens_alloc_specs as alloc_specs

from libensemble.tests.regression_tests.support import give_each_worker_own_stream 
persis_info = give_each_worker_own_stream({},nworkers+1)

n=2
gen_specs['out'] += [('x',float,n), ('x_on_cube',float,n),]
gen_specs['dist_to_bound_multiple'] = 0.5
gen_specs['localopt_maxeval'] = 4

# Tell libEnsemble when to stop
exit_criteria = {'sim_max': 10, 'elapsed_wallclock_time': 300} # Intentially set low so as to test that a worker in persistent mode can be terminated correctly

if nworkers < 2:
    # Can't do a "persistent worker run" if only one worker
    quit()

# Perform the run
H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info, alloc_specs, libE_specs)

if is_master:
    assert flag == 0
    save_libE_output(H,__file__,nworkers)
