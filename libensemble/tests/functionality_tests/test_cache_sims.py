"""
Runs libEnsemble with Latin hypercube sampling on a simple 1D problem

Execute via one of the following commands (e.g. 3 workers):
   mpiexec -np 4 python test_1d_sampling.py
   python test_1d_sampling.py --nworkers 3
   python test_1d_sampling.py --nworkers 3 --comms tcp

The number of concurrent evaluations of the objective function will be 4-1=3.
"""

# Do not change these lines - they are parsed by run-tests.sh
# TESTSUITE_COMMS: mpi local threads
# TESTSUITE_NPROCS: 2 4

import shutil
import tempfile
import time

import numpy as np

from libensemble.alloc_funcs.give_sim_work_first import give_sim_work_first
from libensemble.gen_funcs.sampling import latin_hypercube_sample as gen_f

# Import libEnsemble items for this test
from libensemble.libE import libE
from libensemble.tools import parse_args, save_libE_output


def sim_f(In):
    Out = np.zeros(1, dtype=[("f", float)])
    time.sleep(1.1)
    Out["f"] = np.linalg.norm(In)
    return Out


def changed_sim_f(In):
    Out = np.zeros(1, dtype=[("f", float)])
    time.sleep(1.1)
    Out["f"] = np.linalg.norm(In) + 1
    return Out


if __name__ == "__main__":
    nworkers, is_manager, libE_specs, _ = parse_args()
    libE_specs["cache_long_sims"] = True
    cache_dir = tempfile.mkdtemp(prefix="libe_cache_test_")
    libE_specs["cache_dir"] = cache_dir
    libE_specs["cache_name"] = "cache_sims_test"

    sim_specs = {
        "sim_f": sim_f,
        "in": ["x"],
        "out": [("f", float)],
    }

    gen_specs = {
        "gen_f": gen_f,
        "out": [("x", float, (1,))],
        "batch_size": 10,
        "user": {
            "lb": np.array([-3]),
            "ub": np.array([3]),
            "gen_seed": 42,
        },
    }

    alloc_specs = {"alloc_f": give_sim_work_first}

    exit_criteria = {"sim_max": 11}

    # Baseline evaluations
    H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, alloc_specs=alloc_specs, libE_specs=libE_specs)

    if is_manager:
        assert len(H) >= 11
        print("\nlibEnsemble with random sampling has generated enough points")
        save_libE_output(H, persis_info, __file__, nworkers)

    # Run same workflow with cached sims.
    H_cached, persis_info, flag = libE(
        sim_specs, gen_specs, exit_criteria, alloc_specs=alloc_specs, libE_specs=libE_specs
    )

    # Check cached sims are used (i.e., sims are not re-evaluated).
    if is_manager:
        completed = H["sim_ended"] & H_cached["sim_ended"]
        assert np.array_equal(H["x"][completed], H_cached["x"][completed])
        assert np.array_equal(H["f"][completed], H_cached["f"][completed]), (
            H["x"][completed],
            H["f"][completed],
            H_cached["f"][completed],
        )
        assert np.allclose(H_cached["f"][completed], np.linalg.norm(H_cached["x"][completed], axis=1))

        # Check cached sims have lower durations than new sims.
        durations = H_cached["sim_ended_time"][completed] - H_cached["sim_started_time"][completed]
        assert len(durations) == exit_criteria["sim_max"]
        assert np.all(durations < 1.0)

    # Change the sim. Check cached sims are not used (i.e., sims are re-evaluated).
    changed_specs = dict(sim_specs)
    changed_specs["sim_f"] = changed_sim_f
    H_changed, persis_info, flag = libE(
        changed_specs, gen_specs, exit_criteria, alloc_specs=alloc_specs, libE_specs=libE_specs
    )

    if is_manager:
        completed = H_changed["sim_ended"]
        # Sim values are correct after changing sim_f.
        assert np.allclose(H_changed["f"][completed], np.linalg.norm(H_changed["x"][completed], axis=1) + 1)

        # Check no cached sims. Normal sim durations.
        durations = H_changed["sim_ended_time"][completed] - H_changed["sim_started_time"][completed]
        assert len(durations) == exit_criteria["sim_max"]
        assert np.all(durations > 1.0)
        shutil.rmtree(cache_dir)
