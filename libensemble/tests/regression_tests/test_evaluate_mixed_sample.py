"""
Tests libEnsemble's capability to take in an existing sample of points with
sim_f values and do additional evaluations.

Execute via one of the following commands (e.g. 3 workers):
   mpiexec -np 4 python test_evaluate_mixed_sample.py
   python test_evaluate_mixed_sample.py --nworkers 3 --comms local
   python test_evaluate_mixed_sample.py --nworkers 3 --comms tcp

The number of concurrent evaluations of the objective function will be 4-1=3.
"""

# Do not change these lines - they are parsed by run-tests.sh
# TESTSUITE_COMMS: mpi local tcp
# TESTSUITE_NPROCS: 2 4

import numpy as np

# Import libEnsemble items for this test
from libensemble.libE import libE
from libensemble.sim_funcs.borehole import borehole as sim_f, gen_borehole_input, borehole_func
from libensemble.alloc_funcs.give_pregenerated_work import give_pregenerated_sim_work as alloc_f
from libensemble.tools import parse_args, save_libE_output

# Main block is necessary only when using local comms with spawn start method (default on macOS and Windows).
if __name__ == "__main__":

    nworkers, is_manager, libE_specs, _ = parse_args()

    sim_specs = {
        "sim_f": sim_f,
        "in": ["x"],
        "out": [("f", float)],
    }

    gen_specs = {}

    samp = 1000
    n = 8

    H0 = np.zeros(
        samp, dtype=[("x", float, n), ("f", float), ("sim_id", int), ("sim_started", bool), ("sim_ended", bool)]
    )

    np.random.seed(0)
    H0["x"] = gen_borehole_input(samp)

    for i in range(500):
        H0["f"][i] = borehole_func(H0["x"][i])

    H0["sim_started"][:500] = True
    H0["sim_ended"][:500] = True

    alloc_specs = {"alloc_f": alloc_f, "out": [("x", float, n)]}

    exit_criteria = {"sim_max": len(H0)}

    # Perform the run
    H, persis_info, flag = libE(
        sim_specs, gen_specs, exit_criteria, alloc_specs=alloc_specs, libE_specs=libE_specs, H0=H0
    )

    if is_manager:
        assert len(H) == len(H0)
        assert np.array_equal(H0["x"], H["x"])
        assert np.all(H["sim_ended"])
        print("\nlibEnsemble correctly didn't add anything to initial sample")
        save_libE_output(H, persis_info, __file__, nworkers)
