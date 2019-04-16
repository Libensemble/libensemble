# """
# Runs libEnsemble to test communications
# Scale up array_size and number of workers as required
#
# Execute via the following command:
#    mpiexec -np N python3 {FILENAME}.py
# where N is >= 2
# The number of concurrent evaluations of the objective function will be N-1.
# """

from __future__ import division
from __future__ import absolute_import

import numpy as np

from libensemble.tests.regression_tests.common import parse_args, save_libE_output

# Parse args for test code
nworkers, is_master, libE_specs, _ = parse_args()
if libE_specs['comms'] != 'mpi':
    quit()

# Prob wrap this in the future libe comms module - and that will have init_comms...
# and can report what its using - for comms - and in mpi case for packing/unpacking
# Using dill seems more reliable on Bebop - less unpickle errors
USE_DILL = False # True/False (req: pip install dill)

if USE_DILL:
    import dill
    from mpi4py import MPI
    # Note for mpi4py v3+ - have to initialize differently than previous
    if int(mpi4py.__version__[0]) >= 3:
        MPI.pickle.__init__(dill.dumps, dill.loads)
    else:
        MPI.pickle.dumps = dill.dumps
        MPI.pickle.loads = dill.loads

# Import libEnsemble main, sim_specs, gen_specs, and persis_info
from libensemble.libE import libE
from libensemble.tests.regression_tests.support import float_x1000_sim_specs as sim_specs
from libensemble.tests.regression_tests.support import uniform_random_sample_gen_specs as gen_specs

from libensemble.tests.regression_tests.support import give_each_worker_own_stream 
persis_info = give_each_worker_own_stream({},nworkers+1)

from libensemble.mpi_controller import MPIJobController #Only being used to pass workerID
from libensemble.resources import Resources #Only to get number of workers

jobctrl = MPIJobController(auto_resources = False)
#jobctrl.register_calc(full_path=sim_app, calc_type='sim') #Test with no app registered.
num_workers = Resources.get_num_workers()

rounds = 2              # Number of work units for each worker
sim_max = num_workers*rounds


# This may not nec. be used for this test
# State the generating function, its arguments, output, and necessary parameters.
gen_specs['gen_batch_size'] = sim_max
gen_specs['batch_mode'] = True
gen_specs['num_active_gens'] =1
gen_specs['save_every_k'] = 300

gen_specs['out'] = [('x',float,(2,))]
gen_specs['lb'] = np.array([-3,-2])
gen_specs['ub'] = np.array([ 3, 2])

#sim_max = num_workers
exit_criteria = {'sim_max': sim_max, 'elapsed_wallclock_time': 300}


## Perform the run
H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info, libE_specs=libE_specs)


if is_master:
    assert flag == 0
    for w in range(1, num_workers+1):
        x = w * 1000.0
        assert np.all(H['arr_vals'][w-1] == x), "Array values do not all match"
        assert H['scal_val'][w-1] == x + x/1e7, "Scalar values do not all match"

    save_libE_output(H,__file__,nworkers)
