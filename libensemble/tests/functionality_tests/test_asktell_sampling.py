"""
Runs libEnsemble with Latin hypercube sampling on a simple 1D problem

Execute via one of the following commands (e.g. 3 workers):
   mpiexec -np 4 python test_sampling_asktell_gen.py
   python test_sampling_asktell_gen.py --nworkers 3 --comms local
   python test_sampling_asktell_gen.py --nworkers 3 --comms tcp

The number of concurrent evaluations of the objective function will be 4-1=3.
"""

# Do not change these lines - they are parsed by run-tests.sh
# TESTSUITE_COMMS: mpi local
# TESTSUITE_NPROCS: 2 4

import numpy as np
from gest_api import Generator
from gest_api.vocs import VOCS

import libensemble.sim_funcs.six_hump_camel as six_hump_camel

# Import libEnsemble items for this test
from libensemble.alloc_funcs.start_only_persistent import only_persistent_gens as alloc_f
from libensemble.gen_classes.sampling import UniformSample
from libensemble.libE import libE
from libensemble.sim_funcs.executor_hworld import executor_hworld as sim_f_exec
from libensemble.tools import add_unique_random_streams, parse_args


class StandardSample(Generator):
    """
    This sampler only adheres to the complete standard interface, with no additional numpy methods.
    """

    def __init__(self, VOCS: VOCS):
        self.VOCS = VOCS
        self.rng = np.random.default_rng(1)
        super().__init__(VOCS)

    def _validate_vocs(self, VOCS):
        assert len(self.VOCS.variables), "VOCS must contain variables."

    def suggest(self, n_trials):
        output = []
        for _ in range(n_trials):
            trial = {}
            for key in self.VOCS.variables.keys():
                trial[key] = self.rng.uniform(self.VOCS.variables[key].domain[0], self.VOCS.variables[key].domain[1])
            output.append(trial)
        return output

    def ingest(self, calc_in):
        pass  # random sample so nothing to tell


def sim_f(In):
    Out = np.zeros(1, dtype=[("f", float)])
    Out["f"] = np.linalg.norm(In)
    return Out


if __name__ == "__main__":
    nworkers, is_manager, libE_specs, _ = parse_args()
    libE_specs["gen_on_manager"] = True

    sim_specs = {
        "sim_f": sim_f,
        "in": ["x"],
        "out": [("f", float)],
    }

    gen_specs = {
        "persis_in": ["x", "f", "sim_id"],
        "out": [("x", float, (2,))],
        "initial_batch_size": 20,
        "batch_size": 10,
        "user": {
            "initial_batch_size": 20,  # for wrapper
            "lb": np.array([-3, -2]),
            "ub": np.array([3, 2]),
        },
    }

    variables = {"x0": [-3, 3], "x1": [-2, 2]}
    objectives = {"energy": "EXPLORE"}

    vocs = VOCS(variables=variables, objectives=objectives)

    alloc_specs = {"alloc_f": alloc_f}
    exit_criteria = {"gen_max": 201}
    persis_info = add_unique_random_streams({}, nworkers + 1, seed=1234)

    for test in range(4):
        if test == 0:
            generator = StandardSample(vocs)

        elif test == 1:
            persis_info["num_gens_started"] = 0
            generator = UniformSample(vocs)

        elif test == 2:
            persis_info["num_gens_started"] = 0
            generator = UniformSample(vocs, variables_mapping={"x": ["x0", "x1"], "f": ["energy"]})

        elif test == 3:
            from libensemble.executors.mpi_executor import MPIExecutor

            persis_info["num_gens_started"] = 0
            generator = UniformSample(vocs, variables_mapping={"x": ["x0", "x1"], "f": ["energy"]})
            sim_app2 = six_hump_camel.__file__

            executor = MPIExecutor()
            executor.register_app(full_path=sim_app2, app_name="six_hump_camel", calc_type="sim")  # Named app

            sim_specs = {
                "sim_f": sim_f_exec,
                "in": ["x"],
                "out": [("f", float), ("cstat", int)],
                "user": {"cores": 1},
            }

        gen_specs["generator"] = generator
        H, persis_info, flag = libE(
            sim_specs, gen_specs, exit_criteria, persis_info, alloc_specs, libE_specs=libE_specs
        )

        if is_manager:
            print(H[["sim_id", "x", "f"]][:10])
            assert len(H) >= 201, f"H has length {len(H)}"
