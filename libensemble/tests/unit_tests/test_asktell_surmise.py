import numpy as np
import pytest

from libensemble.message_numbers import FINISHED_PERSISTENT_GEN_TAG


@pytest.mark.extra
def test_asktell_surmise():

    from libensemble.generators import Surmise

    # Import libEnsemble items for this test
    from libensemble.sim_funcs.borehole import borehole
    from libensemble.tools import add_unique_random_streams

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
        ],
        "user": {
            "num_obs": n_x,
            "init_sample_size": init_sample_size,
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
    surmise = Surmise(gen_specs, persis_info=persis_info[1])
    surmise.setup()

    initial_sample = surmise.initial_ask()

    initial_results = np.zeros(len(initial_sample), dtype=gen_out + [("f", float), ("sim_id", int)])

    for field in gen_specs["out"]:
        initial_results[field[0]] = initial_sample[field[0]]

    total_evals = 0

    for i in len(initial_sample):
        initial_results[i] = borehole(initial_sample[i], {}, sim_specs, {})
        initial_results[i]["sim_id"] = i
        total_evals += 1

    surmise.tell(initial_results)

    requested_canceled_sim_ids = []

    while total_evals < max_evals:

        sample, cancels = surmise.ask()
        if len(cancels):
            for m in cancels:
                requested_canceled_sim_ids.append(m)
        results = np.zeros(len(sample), dtype=gen_out + [("f", float), ("sim_id", int)])
        for field in gen_specs["out"]:
            results[field[0]] = sample[field[0]]
        for i in range(len(sample)):
            results[i]["f"] = borehole(sample[i], {}, sim_specs, {})
            results[i]["sim_id"] = total_evals
            total_evals += 1
        surmise.tell(results)
    H, persis_info, exit_code = surmise.final_tell(None)

    assert exit_code == FINISHED_PERSISTENT_GEN_TAG, "Standalone persistent_aposmm didn't exit correctly"
    assert len(requested_canceled_sim_ids), "No cancellations sent by Surmise"


if __name__ == "__main__":
    test_asktell_surmise()
