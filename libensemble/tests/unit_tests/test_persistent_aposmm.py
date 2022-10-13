import pytest
import platform
import multiprocessing

import libensemble.gen_funcs

libensemble.gen_funcs.rc.aposmm_optimizers = "nlopt"

if platform.system() in ["Linux", "Darwin"]:
    multiprocessing.set_start_method("fork", force=True)
    from libensemble.gen_funcs.persistent_aposmm import aposmm, update_history_optimal

import numpy as np
import libensemble.tests.unit_tests.setup as setup
from libensemble.sim_funcs.six_hump_camel import six_hump_camel_func, six_hump_camel_grad

libE_info = {"comm": {}}


@pytest.mark.extra
def test_persis_aposmm_localopt_test():
    _, _, gen_specs_0, _, _ = setup.hist_setup1()

    H = np.zeros(4, dtype=[("f", float), ("sim_id", bool), ("dist_to_unit_bounds", float), ("sim_ended", bool)])
    H["sim_ended"] = True
    H["sim_id"] = range(len(H))
    gen_specs_0["user"]["localopt_method"] = "BADNAME"
    gen_specs_0["user"]["ub"] = np.ones(2)
    gen_specs_0["user"]["lb"] = np.zeros(2)

    try:
        aposmm(H, {}, gen_specs_0, libE_info)
    except NotImplementedError:
        assert 1, "Failed because method is unknown."
    else:
        assert 0


@pytest.mark.extra
def test_update_history_optimal():
    hist, _, _, _, _ = setup.hist_setup1(n=2)

    H = hist.H

    H["sim_ended"] = True
    H["sim_id"] = range(len(H))
    H["f"][0] = -1e-8
    H["x_on_cube"][-1] = 1e-10

    # Perturb x_opt point to test the case where the reported minimum isn't
    # exactly in H. Also, a point in the neighborhood of x_opt has a better
    # function value.
    opt_ind = update_history_optimal(H["x_on_cube"][-1] + 1e-12, 1, H, np.arange(len(H)))

    assert opt_ind == 9, "Wrong point declared minimum"


def combined_func(x):
    return six_hump_camel_func(x), six_hump_camel_grad(x)


@pytest.mark.extra
def test_standalone_persistent_aposmm():
    from libensemble.tests.regression_tests.support import six_hump_camel_minima as minima
    from libensemble.sim_funcs.six_hump_camel import six_hump_camel_func, six_hump_camel_grad
    from math import gamma, pi, sqrt
    from libensemble.message_numbers import FINISHED_PERSISTENT_GEN_TAG

    import libensemble.gen_funcs

    libensemble.gen_funcs.rc.aposmm_optimizers = "nlopt"

    persis_info = {"rand_stream": np.random.default_rng(1), "nworkers": 4}

    n = 2
    eval_max = 2000

    gen_out = [("x", float, n), ("x_on_cube", float, n), ("sim_id", int), ("local_min", bool), ("local_pt", bool)]

    gen_specs = {
        "in": ["x", "f", "grad", "local_pt", "sim_id", "sim_ended", "x_on_cube", "local_min"],
        "out": gen_out,
        "user": {
            "initial_sample_size": 100,
            # 'localopt_method': 'LD_MMA', # Needs gradients
            "sample_points": np.round(minima, 1),
            "localopt_method": "LN_BOBYQA",
            "standalone": {
                "eval_max": eval_max,
                "obj_func": six_hump_camel_func,
                "grad_func": six_hump_camel_grad,
            },
            "rk_const": 0.5 * ((gamma(1 + (n / 2)) * 5) ** (1 / n)) / sqrt(pi),
            "xtol_abs": 1e-6,
            "ftol_abs": 1e-6,
            "dist_to_bound_multiple": 0.5,
            "max_active_runs": 6,
            "lb": np.array([-3, -2]),
            "ub": np.array([3, 2]),
        },
    }
    H = []
    H, persis_info, exit_code = aposmm(H, persis_info, gen_specs, libE_info)
    assert exit_code == FINISHED_PERSISTENT_GEN_TAG, "Standalone persistent_aposmm didn't exit correctly"
    assert np.sum(H["sim_ended"]) >= eval_max, "Standalone persistent_aposmm, didn't evaluate enough points"
    assert persis_info.get("run_order"), "Standalone persistent_aposmm didn't do any localopt runs"

    tol = 1e-3
    min_found = 0
    for m in minima:
        # The minima are known on this test problem.
        # We use their values to test APOSMM has identified all minima
        print(np.min(np.sum((H[H["local_min"]]["x"] - m) ** 2, 1)), flush=True)
        if np.min(np.sum((H[H["local_min"]]["x"] - m) ** 2, 1)) < tol:
            min_found += 1
    assert min_found >= 6, f"Found {min_found} minima"


@pytest.mark.extra
def test_standalone_persistent_aposmm_combined_func():
    from libensemble.tests.regression_tests.support import six_hump_camel_minima as minima
    from math import gamma, pi, sqrt
    from libensemble.message_numbers import FINISHED_PERSISTENT_GEN_TAG

    import libensemble.gen_funcs

    libensemble.gen_funcs.rc.aposmm_optimizers = "nlopt"

    persis_info = {"rand_stream": np.random.default_rng(1), "nworkers": 4}

    n = 2
    eval_max = 100

    gen_out = [("x", float, n), ("x_on_cube", float, n), ("sim_id", int), ("local_min", bool), ("local_pt", bool)]

    gen_specs = {
        "in": ["x", "f", "grad", "local_pt", "sim_id", "sim_ended", "x_on_cube", "local_min"],
        "out": gen_out,
        "user": {
            "initial_sample_size": 100,
            # 'localopt_method': 'LD_MMA', # Needs gradients
            "sample_points": np.round(minima, 1),
            "localopt_method": "LN_BOBYQA",
            "standalone": {"eval_max": eval_max, "obj_and_grad_func": combined_func},
            "rk_const": 0.5 * ((gamma(1 + (n / 2)) * 5) ** (1 / n)) / sqrt(pi),
            "xtol_abs": 1e-6,
            "ftol_abs": 1e-6,
            "dist_to_bound_multiple": 0.5,
            "max_active_runs": 6,
            "lb": np.array([-3, -2]),
            "ub": np.array([3, 2]),
        },
    }

    H = []
    persis_info = {"rand_stream": np.random.default_rng(1), "nworkers": 3}
    H, persis_info, exit_code = aposmm(H, persis_info, gen_specs, libE_info)

    assert exit_code == FINISHED_PERSISTENT_GEN_TAG, "Standalone persistent_aposmm didn't exit correctly"
    assert np.sum(H["sim_ended"]) >= eval_max, "Standalone persistent_aposmm, didn't evaluate enough points"
    assert persis_info.get("run_order"), "Standalone persistent_aposmm didn't do any localopt runs"


if __name__ == "__main__":
    test_persis_aposmm_localopt_test()
    test_update_history_optimal()
    test_standalone_persistent_aposmm()
    test_standalone_persistent_aposmm_combined_func()
