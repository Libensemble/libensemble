"""
Runs libEnsemble with APOSMM with a PETSc/TAO local optimizer.

Execute via one of the following commands (e.g. 3 workers):
   mpiexec -np 4 python test_persistent_aposmm_tao_nm.py
   python test_persistent_aposmm_tao_nm.py --nworkers 3
   python test_persistent_aposmm_tao_nm.py --nworkers 3 --comms tcp

When running with the above commands, the number of concurrent evaluations of
the objective function will be 2, as one of the three workers will be the
persistent generator.
"""

# Do not change these lines - they are parsed by run-tests.sh
# TESTSUITE_COMMS: local mpi tcp
# TESTSUITE_NPROCS: 4
# TESTSUITE_EXTRA: true

import multiprocessing
import sys

import numpy as np

import libensemble.gen_funcs

# Import libEnsemble items for this test
from libensemble.libE import libE
from libensemble.sim_funcs.six_hump_camel import six_hump_camel as sim_f

libensemble.gen_funcs.rc.aposmm_optimizers = "petsc"
from libensemble.alloc_funcs.persistent_aposmm_alloc import persistent_aposmm_alloc as alloc_f
from libensemble.gen_funcs.persistent_aposmm import aposmm as gen_f
from libensemble.tools import add_unique_random_streams, parse_args

# Main block is necessary only when using local comms with spawn start method (default on macOS and Windows).
if __name__ == "__main__":
    multiprocessing.set_start_method("fork", force=True)

    nworkers, is_manager, libE_specs, _ = parse_args()

    if nworkers < 2:
        sys.exit("Cannot run with a persistent worker if only one worker -- aborting...")

    n = 2
    sim_specs = {
        "sim_f": sim_f,
        "in": ["x"],
        "out": [("f", float), ("grad", float, n)],
    }

    gen_out = [
        ("x", float, n),
        ("x_on_cube", float, n),
        ("sim_id", int),
        ("local_min", bool),
        ("local_pt", bool),
    ]

    gen_specs = {
        "gen_f": gen_f,
        "persis_in": ["f", "grad"] + [n[0] for n in gen_out],
        "out": gen_out,
        "user": {
            "initial_sample_size": 100,
            "localopt_method": "nm",
            "lb": np.array([-3, -2]),  # This is only for sampling. TAO_NM doesn't honor constraints.
            "ub": np.array([3, 2]),
        },
    }

    alloc_specs = {"alloc_f": alloc_f}

    persis_info = add_unique_random_streams({}, nworkers + 1)

    exit_criteria = {"sim_max": 1000}

    # Perform the run
    H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info, alloc_specs, libE_specs)

    if is_manager:
        print("[Manager]:", H[np.where(H["local_min"])]["x"])
        assert np.sum(~H["local_pt"]) > 100, "Had to do at least 100 sample points"
        assert np.sum(H["local_pt"]) > 100, "Why didn't at least 100 local points occur?"

    if libE_specs["comms"] == "mpi":
        persis_info = add_unique_random_streams({}, nworkers + 1)
        gen_specs["user"]["run_max_eval"] = 10 * (n + 1)
        H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info, alloc_specs, libE_specs)
        if is_manager:
            assert flag == 0
