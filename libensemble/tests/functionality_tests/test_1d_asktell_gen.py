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
from libensemble import Generator
from libensemble.alloc_funcs.start_only_persistent import only_persistent_gens as alloc_f
from libensemble.gen_funcs.persistent_sampling import _get_user_params
from libensemble.gen_funcs.sampling import latin_hypercube_sample as gen_f
from libensemble.gen_funcs.sampling import lhs_sample
from libensemble.libE import libE
from libensemble.sim_funcs.rosenbrock import rosenbrock_eval as sim_f2
from libensemble.tools import add_unique_random_streams, parse_args


def sim_f(In):
    Out = np.zeros(1, dtype=[("f", float)])
    Out["f"] = np.linalg.norm(In)
    return Out


class LHS(Generator):
    def __init__(self, rand_stream, ub, lb, b, dtype):
        self.rand_stream = rand_stream
        self.ub = ub
        self.lb = lb
        self.batch_size = b
        self.dtype = dtype

    def ask(self, *args):
        n = len(self.lb)
        H_o = np.zeros(self.batch_size, dtype=self.dtype)
        A = lhs_sample(n, self.batch_size, self.rand_stream)
        H_o["x"] = A * (self.ub - self.lb) + self.lb
        return H_o


class PersistentUniform(Generator):
    def __init__(self, persis_info, gen_specs):
        self.persis_info = persis_info
        self.gen_specs = gen_specs
        _, self.n, self.lb, self.ub = _get_user_params(gen_specs["user"])

    def initial_ask(self, num_points, *args):
        return self.ask(num_points)

    def ask(self, num_points):
        H_o = np.zeros(num_points, dtype=self.gen_specs["out"])
        H_o["x"] = self.persis_info["rand_stream"].uniform(self.lb, self.ub, (num_points, self.n))
        self.last_H = H_o
        return H_o

    def tell(self, H_in):
        if hasattr(H_in, "__len__"):
            self.batch_size = len(H_in)

    def final_tell(self, H_in):
        self.tell(H_in)
        return self.last_H


if __name__ == "__main__":
    nworkers, is_manager, libE_specs, _ = parse_args()
    libE_specs["gen_on_manager"] = True

    sim_specs = {
        "sim_f": sim_f,
        "in": ["x"],
        "out": [("f", float)],
    }

    gen_specs_normal = {
        "gen_f": gen_f,
        "out": [("x", float, (1,))],
        "user": {
            "gen_batch_size": 500,
            "lb": np.array([-3]),
            "ub": np.array([3]),
        },
    }

    persis_info = add_unique_random_streams({}, nworkers + 1, seed=1234)

    gen_one = LHS(persis_info[1]["rand_stream"], np.array([3]), np.array([-3]), 500, gen_specs_normal["out"])
    gen_specs_normal["gen_f"] = gen_one

    exit_criteria = {"gen_max": 201}

    H, persis_info, flag = libE(sim_specs, gen_specs_normal, exit_criteria, persis_info, libE_specs=libE_specs)

    if is_manager:
        assert len(H) >= 201
        print("\nlibEnsemble with NORMAL random sampling has generated enough points")
        print(H[:20])

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

    persis_info = add_unique_random_streams({}, nworkers + 1, seed=1234)

    gen_two = PersistentUniform(persis_info[1], gen_specs_persistent)
    gen_specs_persistent["gen_f"] = gen_two

    alloc_specs = {"alloc_f": alloc_f}

    H, persis_info, flag = libE(
        sim_specs, gen_specs_persistent, exit_criteria, persis_info, alloc_specs, libE_specs=libE_specs
    )

    if is_manager:
        assert len(H) >= 201
        print("\nlibEnsemble with PERSISTENT random sampling has generated enough points")
        print(H[:20])
