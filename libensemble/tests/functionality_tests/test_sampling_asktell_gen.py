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

# Import libEnsemble items for this test
from libensemble.alloc_funcs.start_only_persistent import only_persistent_gens as alloc_f
from libensemble.gen_classes.sampling import UniformSample, UniformSampleDicts
from libensemble.gen_funcs.persistent_gen_wrapper import persistent_gen_f as gen_f
from libensemble.libE import libE
from libensemble.tools import add_unique_random_streams, parse_args


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
        "out": [("f", float), ("grad", float, 2)],
    }

    gen_specs = {
        "persis_in": ["x", "f", "grad", "sim_id"],
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

    objectives = {"f": "EXPLORE"}

    alloc_specs = {"alloc_f": alloc_f}
    exit_criteria = {"gen_max": 201}

    for inst in range(4):
        persis_info = add_unique_random_streams({}, nworkers + 1, seed=1234)

        if inst == 0:
            # Using wrapper - pass class
            generator = UniformSample
            gen_specs["gen_f"] = gen_f
            gen_specs["user"]["generator"] = generator

        if inst == 1:
            # Using wrapper - pass object
            gen_specs["gen_f"] = gen_f
            generator = UniformSample(None, persis_info[1], gen_specs, None)
            gen_specs["user"]["generator"] = generator
        elif inst == 2:
            # Using asktell runner - pass object
            gen_specs.pop("gen_f", None)
            generator = UniformSample(None, persis_info[1], gen_specs, None)
            gen_specs["generator"] = generator
        elif inst == 3:
            # Using asktell runner - pass object - with standardized interface.
            gen_specs.pop("gen_f", None)
            generator = UniformSampleDicts(variables, objectives, None, persis_info[1], gen_specs, None)
            gen_specs["generator"] = generator

        H, persis_info, flag = libE(
            sim_specs, gen_specs, exit_criteria, persis_info, alloc_specs, libE_specs=libE_specs
        )

        if is_manager:
            print(H[["sim_id", "x", "f"]][:10])
            assert len(H) >= 201, f"H has length {len(H)}"
            assert np.isclose(H["f"][9], 1.96760289)
