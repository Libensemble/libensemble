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

from libensemble.gen_funcs.sampling import latin_hypercube_sample as gen_f
from libensemble.gen_funcs.sampling import lhs_sample

# Import libEnsemble items for this test
from libensemble.libE import libE
from libensemble.tools import add_unique_random_streams, parse_args


def sim_f(In):
    Out = np.zeros(1, dtype=[("f", float)])
    Out["f"] = np.linalg.norm(In)
    return Out


class LHSGenerator:
    def __init__(self, persis_info, gen_specs):
        self.persis_info = persis_info
        self.gen_specs = gen_specs

    def ask(self):
        ub = self.gen_specs["user"]["ub"]
        lb = self.gen_specs["user"]["lb"]

        n = len(lb)
        b = self.gen_specs["user"]["gen_batch_size"]

        H_o = np.zeros(b, dtype=self.gen_specs["out"])

        A = lhs_sample(n, b, self.persis_info["rand_stream"])

        H_o["x"] = A * (ub - lb) + lb

        return H_o

    def tell(self):
        pass


if __name__ == "__main__":
    nworkers, is_manager, libE_specs, _ = parse_args()

    sim_specs = {
        "sim_f": sim_f,
        "in": ["x"],
        "out": [("f", float)],
    }

    gen_specs = {
        "gen_f": gen_f,
        "out": [("x", float, (1,))],
        "user": {
            "gen_batch_size": 500,
            "lb": np.array([-3]),
            "ub": np.array([3]),
        },
    }

    persis_info = add_unique_random_streams({}, nworkers + 1, seed=1234)

    my_gen = LHSGenerator(persis_info[1], gen_specs)
    gen_specs["gen_f"] = my_gen

    exit_criteria = {"gen_max": 501}

    H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info, libE_specs=libE_specs)

    if is_manager:
        assert len(H) >= 501
        print("\nlibEnsemble with random sampling has generated enough points")
        print(H[:20])
