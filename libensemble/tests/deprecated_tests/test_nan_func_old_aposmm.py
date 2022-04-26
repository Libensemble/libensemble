# """
# Runs libEnsemble on a function that returns only nan; tests APOSMM functionality
#
# Execute via one of the following commands (e.g. 3 workers):
#    mpiexec -np 4 python test_nan_func_aposmm.py
#    python test_nan_func_aposmm.py --nworkers 3 --comms local
#    python test_nan_func_aposmm.py --nworkers 3 --comms tcp
#
# The number of concurrent evaluations of the objective function will be 4-1=3.
# """

# Do not change these lines - they are parsed by run-tests.sh
# TESTSUITE_COMMS: mpi local tcp
# TESTSUITE_NPROCS: 2 4
# TESTSUITE_EXTRA: true

import numpy as np

# Import libEnsemble items for this test
from libensemble.libE import libE
from support import nan_func as sim_f, aposmm_gen_out as gen_out
from libensemble.gen_funcs.old_aposmm import aposmm_logic as gen_f
from libensemble.tools import parse_args, save_libE_output, add_unique_random_streams

nworkers, is_manager, libE_specs, _ = parse_args()
n = 2

sim_specs = {
    "sim_f": sim_f,
    "in": ["x"],
    "out": [
        ("f", float),
        ("f_i", float),
    ],
}

gen_out += [("x", float, n), ("x_on_cube", float, n), ("obj_component", int)]

gen_specs = {
    "gen_f": gen_f,
    "in": [o[0] for o in gen_out] + ["f", "f_i", "sim_ended"],
    "out": gen_out,
    "user": {
        "initial_sample_size": 5,
        "lb": -2 * np.ones(n),
        "ub": 2 * np.ones(n),
    },
}

if nworkers == 3:
    gen_specs["user"]["single_component_at_a_time"] = True
    gen_specs["user"]["components"] = 1
    gen_specs["user"]["combine_component_func"] = np.linalg.norm

persis_info = add_unique_random_streams({}, nworkers + 1)

# Tell libEnsemble when to stop
exit_criteria = {"sim_max": 100, "wallclock_max": 300}

# Perform the run
H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info, libE_specs=libE_specs)
if is_manager:
    assert flag == 0
    assert np.all(~H["local_pt"])

    save_libE_output(H, persis_info, __file__, nworkers)
