# """
#
# SH - not sure what to call this test **********
#
# Runs libEnsemble on the 6-hump camel problem. Documented here:
#    https://www.sfu.ca/~ssurjano/camel6.html
#
# Execute via one of the following commands (e.g. 3 workers):
#    mpiexec -np 4 python3 test_6-hump_camel_persistent_uniform_sampling.py
#    python3 test_6-hump_camel_persistent_uniform_sampling.py --nworkers 3 --comms local
#    python3 test_6-hump_camel_persistent_uniform_sampling.py --nworkers 3 --comms tcp
#
# The number of concurrent evaluations of the objective function will be 4-1=3.
# """

# Do not change these lines - they are parsed by run-tests.sh
# TESTSUITE_COMMS: mpi local tcp
# TESTSUITE_NPROCS: 3 4

# SH TODO: Regarding a persistent sim test
# Naming of this test!
# sims receive H_rows in libE_info - but this will change so need to receive libE_info from message
#   - not just in function call!
#   - and this should probably apply to persis_info !!!!!  I means even in gen?
# Should it insist on sending sims to particular workers (why is it persistent....)
# - Or use active_recv mode to send back intermediate data.
# Need to sort out gen_support module and how this should be modified for sim and gen support.
# - maybe a common module OR maybe gen_support and sim_support wrap a common module
# - CURRENTLY MAJOR KLUGE by using "gen_support" module without having renamed.
# Need to test final H data return (as with gens).
# Should alloc setup all persistent workers at start
#  - Whether need to specify in alloc: eg. give me a list of persistent sims
#    - (then all persis allocs need updating to ask for persis gen only)
# Determine test pass condition
# sendrecv_mgr_worker_msg for sim - needs more args inc. libE_info... (in which case could get rid of
#    separate comm arg - also it needs to remove comm from libE_info anyway as cant pickle a comm
#    (and make sure send a copy with comm removed so dont affect original structure).

import sys
import numpy as np

# Import libEnsemble items for this test
from libensemble.libE import libE
from libensemble.sim_funcs.six_hump_camel import persistent_six_hump_camel as sim_f
from libensemble.gen_funcs.persistent_uniform_sampling import persistent_uniform as gen_f
from libensemble.alloc_funcs.start_only_persistent import only_persistent_workers as alloc_f
from libensemble.tools import parse_args, save_libE_output, add_unique_random_streams

nworkers, is_manager, libE_specs, _ = parse_args()

if nworkers < 2:
    sys.exit("Cannot run with a persistent worker if only one worker -- aborting...")

n = 2
sim_specs = {'sim_f': sim_f,
             'in': ['x'],
             'out': [('f', float), ('grad', float, n)]}

gen_specs = {'gen_f': gen_f,
             'in': [],
             'out': [('x', float, (n,))],
             'user': {'gen_batch_size': 20,
                      'lb': np.array([-3, -2]),
                      'ub': np.array([3, 2]),
                      # 'give_all_with_same_priority': True
                      }
             }

alloc_specs = {'alloc_f': alloc_f, 'out': [('given_back', bool)]}

persis_info = add_unique_random_streams({}, nworkers + 1)

exit_criteria = {'gen_max': 40, 'elapsed_wallclock_time': 300}

# Perform the run
H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info,
                            alloc_specs, libE_specs)

if is_manager:
    assert len(np.unique(H['gen_time'])) == 2
    save_libE_output(H, persis_info, __file__, nworkers)
