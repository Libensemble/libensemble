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

from mpi4py import MPI
import sys, os
import numpy as np

# Prob wrap this in the future libe comms module - and that will have init_comms...
# and can report what its using - for comms - and in mpi case for packing/unpacking
# Using dill seems more reliable on Bebop - less unpickle errors
USE_DILL = False # True/False (req: pip install dill)

if USE_DILL:
    import dill
    import mpi4py
    # Note for mpi4py v3+ - have to initialize differently than previous
    if int(mpi4py.__version__[0]) >= 3:
        MPI.pickle.__init__(dill.dumps, dill.loads)
    else:
        MPI.pickle.dumps = dill.dumps
        MPI.pickle.loads = dill.loads

from libensemble.libE import libE
from libensemble.sim_funcs.comms_testing import float_x1000
from libensemble.gen_funcs.uniform_sampling import uniform_random_sample
from libensemble.register import Register #Only being used to pass workerID
from libensemble.controller import JobController #Only being used to pass workerID
from libensemble.resources import Resources #Only to get number of workers

registry = Register()
jobctrl = JobController(registry = registry, auto_resources = False)
#registry.register_calc(full_path=sim_app, calc_type='sim') #Test with no app registered.
num_workers = Resources.get_num_workers()

array_size = int(1e6)   # Size of large array in sim_specs
rounds = 2              # Number of work units for each worker

sim_max = num_workers*rounds

sim_specs = {'sim_f': float_x1000, # This is the function whose output is being minimized
             'in': ['x'],           # These keys will be given to the above function
             'out': [
                     ('arr_vals',float,array_size),
                     ('scal_val',float),
                    ],
             }

# This may not nec. be used for this test
# State the generating function, its arguments, output, and necessary parameters.
gen_specs = {'gen_f': uniform_random_sample,
             'in': ['sim_id'],
             'out': [('x',float,2),
                    ],
             'lb': np.array([-3,-2]),
             'ub': np.array([ 3, 2]),
             'gen_batch_size': sim_max,
             'batch_mode': True,
             'num_active_gens':1,
             'save_every_k': 300
             }

#sim_max = num_workers
exit_criteria = {'sim_max': sim_max}


np.random.seed(1)
persis_info = {}
for i in range(MPI.COMM_WORLD.Get_size()):
    persis_info[i] = {'rand_stream': np.random.RandomState(i)}

## Perform the run
H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info)


if MPI.COMM_WORLD.Get_rank() == 0:
    #import pdb; pdb.set_trace()
    for w in range(1, num_workers+1):
        x = w * 1000.0
        assert np.all(H['arr_vals'][w-1] == x), "Array values do not all match"
        assert H['scal_val'][w-1] == x + x/1e7, "Scalar values do not all match"
