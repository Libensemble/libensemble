import numpy as np
import libensemble.gen_funcs.old_aposmm as al
import libensemble.tests.unit_tests.setup as setup

n = 2

gen_out = [
    ("x", float, n),
    ("x_on_cube", float, n),
    ("sim_id", int),
    ("priority", float),
    ("local_pt", bool),
    ("known_to_aposmm", bool),
    ("dist_to_unit_bounds", float),
    ("dist_to_better_l", float),
    ("dist_to_better_s", float),
    ("ind_of_better_l", int),
    ("ind_of_better_s", int),
    ("started_run", bool),
    ("num_active_runs", int),
    ("local_min", bool),
]


def test_failing_localopt_method():
    hist, sim_specs_0, gen_specs_0, exit_criteria_0, alloc = setup.hist_setup1()

    hist.H["sim_ended"] = 1

    gen_specs_0["user"]["localopt_method"] = "BADNAME"

    try:
        al.advance_local_run(hist.H, gen_specs_0["user"], 0, 0, {"run_order": {0: [0, 1]}})
    except al.APOSMMException:
        assert 1, "Failed like it should have"
    else:
        assert 0, "Didn't fail like it should have"


def test_exception_raising():
    hist, sim_specs_0, gen_specs_0, exit_criteria_0, alloc = setup.hist_setup1(n=2)
    hist.H["sim_ended"] = 1

    for method in ["LN_SBPLX", "pounders", "scipy_COBYLA"]:
        gen_specs_0["user"]["localopt_method"] = method

        out = al.advance_local_run(hist.H, gen_specs_0["user"], 0, 0, {"run_order": {0: [0, 1]}})

        assert out[0] == 0, "Failed like it should have"


def test_decide_where_to_start_localopt():
    H = np.zeros(10, dtype=gen_out + [("f", float), ("sim_ended", bool)])
    H["x"] = np.random.uniform(0, 1, (10, 2))
    H["f"] = np.random.uniform(0, 1, 10)
    H["sim_ended"] = 1

    b = al.decide_where_to_start_localopt(H, 9, 1)
    assert len(b) == 0

    b = al.decide_where_to_start_localopt(H, 9, 1, nu=0.01)
    assert len(b) == 0


def test_calc_rk():
    rk = al.calc_rk(2, 10, 1)

    rk = al.calc_rk(2, 10, 1, 10)
    assert np.isinf(rk)


def test_initialize_APOSMM():
    hist, sim_specs_0, gen_specs_0, exit_criteria_0, alloc = setup.hist_setup1()

    al.initialize_APOSMM(hist.H, gen_specs_0)


def test_declare_opt():
    hist, sim_specs_0, gen_specs_0, exit_criteria_0, alloc = setup.hist_setup1(n=2)

    try:
        al.update_history_optimal(hist.H["x_on_cube"][0] + 1, hist.H, np.arange(0, 10))
    except AssertionError:
        assert 1, "Failed because the best point is not in H"
    else:
        assert 0

    hist.H["x_on_cube"][1] += np.finfo(float).eps
    hist.H["f"][1] -= np.finfo(float).eps

    # Testing case where point near x_opt is slightly better.
    al.update_history_optimal(hist.H["x_on_cube"][0], hist.H, np.arange(0, 10))
    assert np.sum(hist.H["local_min"]) == 2


def test_localopt_error_saving():
    _, sim_specs_0, gen_specs_0, _, _ = setup.hist_setup1()

    H = np.zeros(4, dtype=gen_out + [("f", float), ("fvec", float, 2), ("sim_ended", bool)])
    H["x"] = np.random.uniform(0, 1, (4, 2))
    H["f"] = np.random.uniform(0, 1, 4)
    H["sim_ended"] = True
    H["local_pt"][1:] = True
    gen_specs_0["user"]["initial_sample_size"] = 1
    gen_specs_0["user"]["localopt_method"] = "scipy_COBYLA"
    gen_specs_0["user"]["tol"] = 0.1
    gen_specs_0["user"]["ub"] = np.ones(2)
    gen_specs_0["user"]["lb"] = np.zeros(2)

    persis_info_1 = {
        "run_order": {0: [1, 2, 3]},
        "old_runs": {},
        "total_runs": 0,
        "rand_stream": np.random.default_rng(1),
    }

    try:
        al.aposmm_logic(H, persis_info_1, gen_specs_0, _)
    except Exception as e:
        assert (
            e.args[0] == "Exit code is 0, but x_new was not updated in local opt run 0 after 3 evaluations.\n"
            "Saving run information to: run_0_abort.pickle\nWorker crashing!"
        )
    else:
        assert 0


if __name__ == "__main__":
    test_localopt_error_saving()
    print("done")
    test_failing_localopt_method()
    print("done")
    test_exception_raising()
    print("done")
    test_decide_where_to_start_localopt()
    print("done")
    test_calc_rk()
    print("done")
    test_initialize_APOSMM()
    print("done")
    test_declare_opt()
    print("done")
