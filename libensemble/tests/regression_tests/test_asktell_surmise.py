# TESTSUITE_COMMS: local
# TESTSUITE_NPROCS: 4
# TESTSUITE_EXTRA: true
# TESTSUITE_OS_SKIP: OSX

import os

import numpy as np

from libensemble.message_numbers import FINISHED_PERSISTENT_GEN_TAG

if __name__ == "__main__":

    from libensemble.executors import Executor
    from libensemble.gen_classes import Surmise

    # Import libEnsemble items for this test
    from libensemble.sim_funcs.borehole_kills import borehole
    from libensemble.tests.regression_tests.common import build_borehole  # current location
    from libensemble.tools import add_unique_random_streams
    from libensemble.utils.misc import list_dicts_to_np

    sim_app = os.path.join(os.getcwd(), "borehole.x")
    if not os.path.isfile(sim_app):
        build_borehole()

    exctr = Executor()  # Run serial sub-process in place
    exctr.register_app(full_path=sim_app, app_name="borehole")

    n_init_thetas = 15  # Initial batch of thetas
    n_x = 5  # No. of x values
    nparams = 4  # No. of theta params
    ndims = 3  # No. of x coordinates.
    max_add_thetas = 20  # Max no. of thetas added for evaluation
    step_add_theta = 10  # No. of thetas to generate per step, before emulator is rebuilt
    n_explore_theta = 200  # No. of thetas to explore while selecting the next theta
    obsvar = 10 ** (-1)  # Constant for generating noise in obs

    # Batch mode until after init_sample_size (add one theta to batch for observations)
    init_sample_size = (n_init_thetas + 1) * n_x

    # Stop after max_emul_runs runs of the emulator
    max_evals = init_sample_size + max_add_thetas * n_x

    # Rename ensemble dir for non-interference with other regression tests
    sim_specs = {
        "in": ["x", "thetas"],
        "out": [
            ("f", float),
            ("sim_killed", bool),
        ],
        "user": {
            "num_obs": n_x,
            "init_sample_size": init_sample_size,
            "poll_manager": False,
        },
    }

    gen_out = [
        ("x", float, ndims),
        ("thetas", float, nparams),
        ("priority", int),
        ("obs", float, n_x),
        ("obsvar", float, n_x),
    ]

    gen_specs = {
        "persis_in": [o[0] for o in gen_out] + ["f", "sim_ended", "sim_id"],
        "out": gen_out,
        "user": {
            "n_init_thetas": n_init_thetas,  # Num thetas in initial batch
            "num_x_vals": n_x,  # Num x points to create
            "step_add_theta": step_add_theta,  # No. of thetas to generate per step
            "n_explore_theta": n_explore_theta,  # No. of thetas to explore each step
            "obsvar": obsvar,  # Variance for generating noise in obs
            "init_sample_size": init_sample_size,  # Initial batch size inc. observations
            "priorloc": 1,  # Prior location in the unit cube.
            "priorscale": 0.2,  # Standard deviation of prior
        },
    }

    persis_info = add_unique_random_streams({}, 5)
    surmise = Surmise(gen_specs=gen_specs, persis_info=persis_info[1])  # we add sim_id as a field to gen_specs["out"]
    surmise.setup()

    initial_sample = surmise.ask()

    total_evals = 0

    for point in initial_sample:
        H_out, _a, _b = borehole(
            list_dicts_to_np([point], dtype=gen_specs["out"]), {}, sim_specs, {"H_rows": np.array([point["sim_id"]])}
        )
        point["f"] = H_out["f"][0]  # some "bugginess" with output shape of array in simf
        total_evals += 1

    surmise.tell(initial_sample)

    requested_canceled_sim_ids = []

    next_sample, cancels = surmise.ask(), surmise.ask_updates()

    for point in next_sample:
        H_out, _a, _b = borehole(
            list_dicts_to_np([point], dtype=gen_specs["out"]), {}, sim_specs, {"H_rows": np.array([point["sim_id"]])}
        )
        point["f"] = H_out["f"][0]
        total_evals += 1

    surmise.tell(next_sample)
    sample, cancels = surmise.ask(), surmise.ask_updates()

    while total_evals < max_evals:

        for point in sample:
            H_out, _a, _b = borehole(
                list_dicts_to_np([point], dtype=gen_specs["out"]),
                {},
                sim_specs,
                {"H_rows": np.array([point["sim_id"]])},
            )
            point["f"] = H_out["f"][0]
            total_evals += 1
            surmise.tell([point])
            if surmise.ready_to_be_asked():
                new_sample, cancels = surmise.ask(), surmise.ask_updates()
                for m in cancels:
                    requested_canceled_sim_ids.append(m)
                if len(new_sample):
                    sample = new_sample
                    break

    H, persis_info, exit_code = surmise.final_tell(None)

    assert exit_code == FINISHED_PERSISTENT_GEN_TAG, "Standalone persistent_aposmm didn't exit correctly"
    # assert len(requested_canceled_sim_ids), "No cancellations sent by Surmise"
