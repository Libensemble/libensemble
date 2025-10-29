"""
Tests libEnsemble with a uniform sample that is also requesting cancellation of
some points.

Execute via one of the following commands (e.g. 3 workers):
   mpiexec -np 4 python test_uniform_sampling_cancel.py
   python test_uniform_sampling_cancel.py --nworkers 3
   python test_uniform_sampling_cancel.py --nworkers 3 --comms tcp

The number of concurrent evaluations of the objective function will be 4-1=3.

Tests sampling with cancellations.
"""

# Do not change these lines - they are parsed by run-tests.sh
# TESTSUITE_COMMS: mpi local
# TESTSUITE_NPROCS: 2 4

import gc

import numpy as np

from libensemble.alloc_funcs.fast_alloc import give_sim_work_first as fast_gswf
from libensemble.alloc_funcs.give_pregenerated_work import give_pregenerated_sim_work
from libensemble.alloc_funcs.give_sim_work_first import give_sim_work_first as gswf
from libensemble.alloc_funcs.only_one_gen_alloc import ensure_one_active_gen
from libensemble.gen_funcs.sampling import uniform_random_sample_cancel

# Import libEnsemble items for this test
from libensemble.libE import libE
from libensemble.sim_funcs.six_hump_camel import six_hump_camel
from libensemble.tests.regression_tests.support import six_hump_camel_minima as minima
from libensemble.tools import add_unique_random_streams, parse_args


def create_H0(persis_info, gen_specs, sim_max):
    """Create an H0 for give_pregenerated_sim_work"""
    # Manually creating H0
    ub = gen_specs["user"]["ub"]
    lb = gen_specs["user"]["lb"]
    n = len(lb)
    b = sim_max

    H0 = np.zeros(b, dtype=[("x", float, 2), ("sim_id", int), ("sim_started", bool), ("cancel_requested", bool)])
    H0["x"] = persis_info[0]["rand_stream"].uniform(lb, ub, (b, n))
    H0["sim_id"] = range(b)
    H0["sim_started"] = False
    for i in range(b):
        if i % 10 == 0:
            H0[i]["cancel_requested"] = True

    # Using uniform_random_sample_cancel call - need to adjust some gen_specs though
    # gen_specs["out"].append(("sim_id", int))
    # gen_specs["out"].append(("sim_started", bool))
    # gen_specs["user"]["gen_batch_size"] = sim_max
    # H0, persis_info[0] = uniform_random_sample_cancel({}, persis_info[0], gen_specs, {})
    # H0["sim_id"] = range(gen_specs["user"]["gen_batch_size"])
    # H0["sim_started"] = False
    return H0


# Main block is necessary only when using local comms with spawn start method (default on macOS and Windows).
if __name__ == "__main__":
    nworkers, is_manager, libE_specs, _ = parse_args()

    sim_specs = {
        "sim_f": six_hump_camel,  # Function whose output is being minimized
        "in": ["x"],  # Keys to be given to sim_f
        "out": [("f", float)],  # Name of the outputs from sim_f
    }
    # end_sim_specs_rst_tag

    # Note that it is unusual to specify cancel_requested as gen_specs["out"]. It is here
    # so that cancellations are combined with regular generator outputs for testing purposes.
    # For a typical use case see test_persistent_surmise_calib.py.
    gen_specs = {
        "gen_f": uniform_random_sample_cancel,  # Function generating sim_f input
        "out": [("x", float, (2,)), ("cancel_requested", bool)],
        "user": {
            "gen_batch_size": 50,  # Used by this specific gen_f
            "lb": np.array([-3, -2]),  # Used by this specific gen_f
            "ub": np.array([3, 2]),  # Used by this specific gen_f
        },
    }
    # end_gen_specs_rst_tag

    persis_info = add_unique_random_streams({}, nworkers + 1)
    sim_max = 500
    exit_criteria = {"sim_max": sim_max, "wallclock_max": 300}

    a_spec_1 = {
        "alloc_f": gswf,
        "user": {
            "batch_mode": True,
            "num_active_gens": 1,
        },
    }

    a_spec_2 = {
        "alloc_f": gswf,
        "user": {
            "batch_mode": True,
            "num_active_gens": 2,
        },
    }

    a_spec_3 = {
        "alloc_f": fast_gswf,
        "user": {},
    }

    a_spec_4 = {
        "alloc_f": ensure_one_active_gen,
        "user": {},
    }

    a_spec_5 = {
        "alloc_f": give_pregenerated_sim_work,
        "user": {},
    }

    allocs = {1: a_spec_1, 2: a_spec_2, 3: a_spec_3, 4: a_spec_4, 5: a_spec_5}

    if is_manager:
        print("Testing cancellations with non-persistent gen functions")

    for testnum in range(1, 6):
        alloc_specs = allocs[testnum]
        if is_manager:
            print("\nRunning with alloc specs", alloc_specs, flush=True)

        if alloc_specs["alloc_f"] == give_pregenerated_sim_work:
            H0 = create_H0(persis_info, gen_specs, sim_max)
        else:
            H0 = None

        # Reset for those that use them
        persis_info["next_to_give"] = 0
        persis_info["total_gen_calls"] = 0  # 1

        # Perform the run - do not overwrite persis_info
        H, persis_out, flag = libE(
            sim_specs, gen_specs, exit_criteria, persis_info, alloc_specs, libE_specs=libE_specs, H0=H0
        )

        if is_manager:
            assert flag == 0
            assert np.all(H["cancel_requested"][::10]), "Some values should be cancelled but are not"
            assert np.all(~H["sim_started"][::10]), "Some values are given that should not have been"
            tol = 0.1
            for m in minima:
                assert np.min(np.sum((H["x"] - m) ** 2, 1)) < tol

            print("libEnsemble found the 6 minima within a tolerance " + str(tol))
            del H
            gc.collect()  # Clean up memory space.
