"""
Runs libEnsemble with Latin hypercube sampling on a simple 1D problem

Execute via one of the following commands (e.g. 3 workers):
   mpiexec -np 4 python test_1d_sampling.py
   python test_1d_sampling.py --nworkers 3 --comms local
   python test_1d_sampling.py --nworkers 3 --comms tcp

The number of concurrent evaluations of the objective function will be 4-1=3.
"""

# Do not change these lines - they are parsed by run-tests.sh
# TESTSUITE_COMMS: mpi local
# TESTSUITE_NPROCS: 2 4

import numpy as np

# Import libEnsemble items for this test
from libensemble.alloc_funcs.start_only_persistent import only_persistent_gens as alloc_f
from libensemble.gen_funcs.persistent_sampling import RandSample
from libensemble.libE import libE
from libensemble.sim_funcs.rosenbrock import rosenbrock_eval as sim_f2
from libensemble.tools import add_unique_random_streams, parse_args


def sim_f(In):
    Out = np.zeros(1, dtype=[("f", float)])
    Out["f"] = np.linalg.norm(In)
    return Out


if __name__ == "__main__":
    nworkers, is_manager, libE_specs, _ = parse_args()
    libE_specs["gen_on_manager"] = True

    sim_specs = {
        "sim_f": sim_f2,
        "in": ["x"],
        "out": [("f", float), ("grad", float, 2)],
    }

    gen_specs_persistent = {
        "persis_in": ["x", "f", "grad", "sim_id"],
        "out": [("x", float, (2,))],
        "user": {
            "initial_batch_size": 20,
            "lb": np.array([-3, -2]),
            "ub": np.array([3, 2]),
        },
    }

    exit_criteria = {"gen_max": 201}

    persis_info = add_unique_random_streams({}, nworkers + 1, seed=1234)

    gen_two = RandSample(None, persis_info[1], gen_specs_persistent, None)
    gen_specs_persistent["generator"] = gen_two

    alloc_specs = {"alloc_f": alloc_f}

    H, persis_info, flag = libE(
        sim_specs, gen_specs_persistent, exit_criteria, persis_info, alloc_specs, libE_specs=libE_specs
    )

    if is_manager:
        assert len(H) >= 201
        print("\nlibEnsemble with PERSISTENT random sampling has generated enough points")
        print(H[:10])
