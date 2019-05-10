# """
# Runs libEnsemble to test communications
# Scale up array_size and number of workers as required
#
# Execute via the following command:
#    mpiexec -np N python3 {FILENAME}.py
# where N is >= 2
# The number of concurrent evaluations of the objective function will be N-1.
# """

import numpy as np

# Import libEnsemble items for this test
from libensemble.libE import libE
from libensemble.sim_funcs.comms_testing import float_x1000 as sim_f
from libensemble.gen_funcs.uniform_sampling import uniform_random_sample as gen_f
from libensemble.tests.regression_tests.common import parse_args, save_libE_output, per_worker_stream
from libensemble.mpi_controller import MPIJobController #Only used to get workerID in float_x1000
jobctrl = MPIJobController(auto_resources=False)

nworkers, is_master, libE_specs, _ = parse_args()
if libE_specs['comms'] == 'tcp':
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

array_size = int(1e6) # Size of large array in sim_specs
rounds = 2 # Number of work units for each worker
sim_max = nworkers*rounds

sim_specs = {
    'sim_f': sim_f,
    'in': ['x'],
    'out': [('arr_vals', float, array_size), ('scal_val', float)],}

gen_specs = {
    'gen_f': gen_f,
    'in': ['sim_id'],
    'out': [('x', float, (2,))],
    'lb': np.array([-3, -2]),
    'ub': np.array([3, 2]),
    'gen_batch_size': sim_max,
    'batch_mode': True,
    'num_active_gens': 1,
    'save_every_k': 300,}

persis_info = per_worker_stream({}, nworkers+1)

exit_criteria = {'sim_max': sim_max, 'elapsed_wallclock_time': 300}

## Perform the run
H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info,
                            libE_specs=libE_specs)

if is_master:
    assert flag == 0
    for w in range(1, nworkers+1):
        x = w*1000.0
        assert np.all(H['arr_vals'][w-1] == x), "Array values do not all match"
        assert H['scal_val'][w-1] == x+x/1e7, "Scalar values do not all match"

    save_libE_output(H, persis_info, __file__, nworkers)
