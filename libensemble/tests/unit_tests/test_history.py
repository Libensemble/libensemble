import libensemble.tests.unit_tests.setup as setup
from libensemble.message_numbers import WORKER_DONE
from libensemble.tools.fields_keys import libE_fields
import numpy as np
from numpy import inf

if tuple(np.__version__.split(".")) >= ("1", "15"):
    from numpy.lib.recfunctions import repack_fields

# Consider fixtures for this - parameterization may save duplication if always use pytest.

# Comparing hist produced: options (using mix of first two)
# - hardcode array compare
# - compare selected values
# - compare from npy file - stored


fields = [
    ("f", "<f8"),
    ("fvec", "<f8", (3,)),
    ("local_min", "?"),
    ("local_pt", "?"),
    ("num_active_runs", "<i8"),
    ("priority", "<f8"),
    ("x_on_cube", "<f8"),
]
fields2 = [("g", "<f8"), ("x", "<f8"), ("priority", "<f8")]

wrs_H0 = np.zeros(3, dtype=libE_fields)
wrs_H0[["sim_started", "sim_ended"]] = True
wrs_H0["sim_ended_time"] = [1.1, 1.2, 1.3]
wrs_H0["sim_id"] = range(3)
wrs_H0["sim_started_time"] = [0.1, 0.2, 0.3]
wrs_H0["sim_worker"] = [1, 2, 3]

exp_H0_H = np.zeros(5, dtype=libE_fields + fields)
exp_H0_H["gen_informed_time"] = [0, 0, 0, inf, inf]
exp_H0_H[["sim_started", "sim_ended"]][:3] = True
exp_H0_H["sim_ended_time"][:3] = [1.1, 1.2, 1.3]
exp_H0_H["sim_id"] = [0, 1, 2, -1, -1]
exp_H0_H["sim_started_time"] = [0.1, 0.2, 0.3, inf, inf]
exp_H0_H["sim_worker"][:3] = [1, 2, 3]

wrs = np.zeros(10, dtype=libE_fields + fields)
wrs[["sim_started_time", "gen_informed_time"]] = inf
wrs["sim_id"] = -1

wrs2 = np.zeros(10, dtype=libE_fields + fields2)
wrs2[["sim_started_time", "gen_informed_time"]] = inf
wrs2["sim_id"] = -1

exp_x_in_setup2 = np.zeros(10, dtype=libE_fields + fields2)
exp_x_in_setup2[["gen_informed_time", "gen_started_time", "sim_started_time"]] = inf
exp_x_in_setup2["gen_worker"] = 3
exp_x_in_setup2["gen_worker"][0] = 2
exp_x_in_setup2["sim_id"] = range(10)

x = [
    4.17022005e-01,
    7.20324493e-01,
    1.14374817e-04,
    3.02332573e-01,
    1.46755891e-01,
    9.23385948e-02,
    1.86260211e-01,
    3.45560727e-01,
    3.96767474e-01,
    5.38816734e-01,
]

exp_x_in_setup2["x"] = x

safe_mode = True


def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


def compare_hists(H1, H2, ignore=[]):
    for field in sorted(set(H1.dtype.names + H2.dtype.names)):
        if field not in ignore:
            assert np.allclose(H1[field], H2[field]), "Array does not match expected"


# Tests ========================================================================================
def test_hist_init_1():
    hist, _, _, _, _ = setup.hist_setup1()

    compare_hists(hist.H, wrs)

    assert hist.sim_started_count == 0
    assert hist.index == 0
    assert hist.sim_ended_count == 0
    assert hist.gen_informed_count == 0


def test_hist_init_1A_H0():
    hist, _, _, _, _ = setup.hist_setup1(sim_max=2, H0_in=wrs_H0)

    compare_hists(hist.H, exp_H0_H)

    assert hist.sim_started_count == 3
    assert hist.index == 3
    assert hist.sim_ended_count == 3
    assert hist.gen_informed_count == 0
    assert len(hist.H) == 5


def test_hist_init_2():
    hist, _, _, _, _ = setup.hist_setup2()

    compare_hists(hist.H, wrs2)

    assert hist.sim_started_count == 0
    assert hist.index == 0
    assert hist.sim_ended_count == 0
    assert hist.gen_informed_count == 0


def test_grow_H():
    hist, _, _, _, _ = setup.hist_setup1(3)
    new_rows = 7
    hist.grow_H(k=new_rows)

    compare_hists(hist.H, wrs)

    assert hist.sim_started_count == 0
    assert hist.index == 0
    assert hist.sim_ended_count == 0
    assert hist.gen_informed_count == 0


def test_trim_H():
    hist, _, _, _, _ = setup.hist_setup1(13)
    hist.index = 10
    H = hist.trim_H()

    compare_hists(H, wrs)

    assert hist.sim_started_count == 0
    assert hist.index == 10
    assert hist.sim_ended_count == 0
    assert hist.gen_informed_count == 0


def test_update_history_x_in_Oempty():
    hist, sim_specs, gen_specs, _, _ = setup.hist_setup2()
    H_o = np.zeros(0, dtype=gen_specs["out"])
    gen_worker = 1
    hist.update_history_x_in(gen_worker, H_o, safe_mode, np.inf)

    compare_hists(hist.H, wrs2)

    assert hist.sim_started_count == 0
    assert hist.index == 0
    assert hist.sim_ended_count == 0
    assert hist.gen_informed_count == 0


def test_update_history_x_in():
    hist, _, gen_specs, _, _ = setup.hist_setup2(7)

    np.random.seed(1)
    single_rand = gen_specs["gen_f"]()  # np.random.uniform()

    # Check seeded correctly going in
    assert isclose(single_rand, 0.417022004702574), "Random numbers not correct before function"

    size = 1
    gen_worker = 2
    H_o = np.zeros(size, dtype=gen_specs["out"])
    H_o["x"] = single_rand

    hist.update_history_x_in(gen_worker, H_o, safe_mode, np.inf)
    assert isclose(single_rand, hist.H["x"][0])
    assert hist.sim_started_count == 0
    assert hist.index == 1
    assert hist.sim_ended_count == 0
    assert hist.gen_informed_count == 0

    size = 6
    gen_worker = 3
    H_o = np.zeros(size, dtype=gen_specs["out"])
    H_o["x"] = gen_specs["gen_f"](size=size)

    hist.update_history_x_in(gen_worker, H_o, safe_mode, np.inf)
    # Compare by column
    exp_x = exp_x_in_setup2[: size + 1]

    compare_hists(hist.H, exp_x, ["gen_ended_time"])

    assert hist.sim_started_count == 0
    assert hist.index == 7
    assert hist.sim_ended_count == 0
    assert hist.gen_informed_count == 0

    # Force H to grow when add points
    size = 3
    gen_worker = 3
    H_o = np.zeros(size, dtype=gen_specs["out"])
    H_o["x"] = gen_specs["gen_f"](size=size)

    hist.update_history_x_in(gen_worker, H_o, safe_mode, np.inf)
    # Compare by column
    exp_x = exp_x_in_setup2

    compare_hists(hist.H, exp_x, ["gen_ended_time"])

    assert hist.sim_started_count == 0
    assert hist.index == 10
    assert hist.sim_ended_count == 0

    # Test libE errors when a protected field appears in output from a gen_worker
    H_o = np.zeros(size, dtype=gen_specs["out"] + [("sim_started", bool)])
    try:
        hist.update_history_x_in(gen_worker, H_o, safe_mode, np.inf)
    except AssertionError:
        assert 1, "Failed like it should have"
    else:
        assert 0, "Didn't fail like it should have"

    # Test libE errors when a protected field appears in output from a gen_worker
    H_o = np.zeros(size, dtype=gen_specs["out"] + [("sim_started", bool)])
    try:
        hist.update_history_x_in(gen_worker, H_o, safe_mode, np.inf)
    except AssertionError:
        assert 1, "Failed like it should have"
    else:
        assert 0, "Didn't fail like it should have"


def test_update_history_x_in_sim_ids():
    hist, _, gen_specs, _, _ = setup.hist_setup2A_genout_sim_ids(7)

    np.random.seed(1)
    single_rand = gen_specs["gen_f"]()  # np.random.uniform()

    # Check seeded correctly going in
    assert isclose(single_rand, 0.417022004702574), "Random numbers not correct before function"

    size = 1
    gen_worker = 2
    H_o = np.zeros(size, dtype=gen_specs["out"])
    H_o["x"] = single_rand
    H_o["sim_id"] = 0

    hist.update_history_x_in(gen_worker, H_o, safe_mode, np.inf)
    assert isclose(single_rand, hist.H["x"][0])
    assert hist.sim_started_count == 0
    assert hist.index == 1
    assert hist.sim_ended_count == 0
    assert hist.gen_informed_count == 0

    size = 6
    gen_worker = 3
    H_o = np.zeros(size, dtype=gen_specs["out"])
    H_o["x"] = gen_specs["gen_f"](size=size)
    H_o["sim_id"] = range(1, 7)
    hist.update_history_x_in(gen_worker, H_o, safe_mode, np.inf)

    # Compare by column
    exp_x = exp_x_in_setup2[: size + 1]

    compare_hists(hist.H, exp_x, ["gen_ended_time"])

    assert hist.sim_started_count == 0
    assert hist.index == 7
    assert hist.sim_ended_count == 0
    assert hist.gen_informed_count == 0

    # Force H to grow when add points
    size = 3
    gen_worker = 3
    H_o = np.zeros(size, dtype=gen_specs["out"])
    H_o["x"] = gen_specs["gen_f"](size=size)
    H_o["sim_id"] = range(7, 10)

    hist.update_history_x_in(gen_worker, H_o, safe_mode, np.inf)
    # Compare by column
    exp_x = exp_x_in_setup2

    compare_hists(hist.H, exp_x, ["gen_ended_time"])

    assert hist.sim_started_count == 0
    assert hist.index == 10
    assert hist.sim_ended_count == 0
    assert hist.gen_informed_count == 0


# Note - Ideally have more setup here (so hist.index reflects generated points)
def test_update_history_x_out():
    hist, _, _, _, _ = setup.hist_setup1()

    # First update a single point
    hist.update_history_x_out(q_inds=0, sim_worker=2)

    # Check updated values for point and counts
    assert hist.H["sim_started"][0]
    assert hist.H["sim_worker"][0] == 2
    assert hist.sim_started_count == 1

    # Check some unchanged values for point and counts
    assert hist.index == 0
    assert hist.sim_ended_count == 0
    hist.H["sim_ended"][0] = False
    hist.H["f"][0] == 0.0
    hist.H["sim_id"][0] == -1

    # Check the rest of H is unaffected
    compare_hists(hist.H[1:10], wrs[1:10])

    # Update two further consecutive points
    my_qinds = np.arange(1, 3)
    hist.update_history_x_out(q_inds=my_qinds, sim_worker=3)

    # Check updated values for point and counts
    assert np.all(hist.H["sim_started"][0:3])  # Include previous point
    assert np.all(hist.H["sim_worker"][my_qinds] == 3)
    assert hist.sim_started_count == 3

    # Update three further non-consecutive points
    my_qinds = np.array([4, 7, 9])
    hist.update_history_x_out(q_inds=my_qinds, sim_worker=4)

    # Try to avoid tautological testing - compare columns
    assert np.array_equal(
        hist.H["sim_started"], np.array([True, True, True, False, True, False, False, True, False, True])
    )
    assert np.array_equal(hist.H["sim_worker"], np.array([2, 3, 3, 0, 4, 0, 0, 4, 0, 4]))
    assert np.all(~hist.H["sim_ended"])  # Should still be unaffected.

    # Check counts
    assert hist.sim_started_count == 6
    assert hist.index == 0  # In real case this would be ahead.....
    assert hist.sim_ended_count == 0
    assert hist.gen_informed_count == 0


def test_update_history_f():
    hist, sim_specs, _, _, _ = setup.hist_setup2()
    exp_vals = [0.0] * 10

    # First update a single point
    size = 1
    sim_ids = [0]  # First row to be filled
    calc_out = np.zeros(size, dtype=sim_specs["out"])
    a = np.arange(9) - 4
    calc_out["g"] = sim_specs["sim_f"](a)  # np.linalg.norm
    exp_vals[0] = calc_out["g"][0]
    D_recv = {
        "calc_out": calc_out,
        "persis_info": {},
        "libE_info": {"H_rows": sim_ids},
        "calc_status": WORKER_DONE,
        "calc_type": 2,
    }

    hist.update_history_f(D_recv, safe_mode)
    assert isclose(exp_vals[0], hist.H["g"][0])
    assert np.all(hist.H["sim_ended"][0:1])
    assert np.all(~hist.H["sim_ended"][1:10])  # Check the rest
    assert hist.sim_ended_count == 1
    assert hist.sim_started_count == 0  # In real case this would be ahead.....
    assert hist.index == 0  # In real case this would be ahead....

    # Update two further consecutive points
    size = 2
    sim_ids = [1, 2]  # First row to be filled
    calc_out = np.zeros(size, dtype=sim_specs["out"])
    a = np.arange(9) - 3
    calc_out["g"][0] = sim_specs["sim_f"](a)  # np.linalg.norm
    exp_vals[1] = calc_out["g"][0]
    a = np.arange(9) - 2
    calc_out["g"][1] = sim_specs["sim_f"](a)  # np.linalg.norm
    exp_vals[2] = calc_out["g"][1]
    D_recv = {
        "calc_out": calc_out,
        "persis_info": {},
        "libE_info": {"H_rows": sim_ids},
        "calc_status": WORKER_DONE,
        "calc_type": 2,
    }

    hist.update_history_f(D_recv, safe_mode)
    assert np.allclose(exp_vals, hist.H["g"])
    assert np.all(hist.H["sim_ended"][0:3])
    assert np.all(~hist.H["sim_ended"][3:10])  # Check the rest
    assert hist.sim_ended_count == 3
    assert hist.sim_started_count == 0  # In real case this would be ahead.....
    assert hist.index == 0  # In real case this would be ahead....


def test_update_history_f_vec():
    hist, sim_specs, _, _, _ = setup.hist_setup1()
    exp_fs = [0.0] * 10
    exp_fvecs = [[0.0, 0.0, 0.0]] * 10

    # First update a single point
    size = 1
    sim_ids = [0]  # First row to be filled
    calc_out = np.zeros(size, dtype=sim_specs["out"])
    a = np.array([[1, 2, 3], [-1, 1, 4]])
    calc_out["f"] = sim_specs["sim_f"](a)  # np.linalg.norm
    calc_out["fvec"] = sim_specs["sim_f"](a, axis=0)  # np.linalg.norm
    exp_fs[0] = calc_out["f"][0]
    exp_fvecs[0] = calc_out["fvec"][0]
    D_recv = {
        "calc_out": calc_out,
        "persis_info": {},
        "libE_info": {"H_rows": sim_ids},
        "calc_status": WORKER_DONE,
        "calc_type": 2,
    }

    hist.update_history_f(D_recv, safe_mode)

    assert isclose(exp_fs[0], hist.H["f"][0])
    assert np.allclose(exp_fvecs[0], hist.H["fvec"][0])
    assert np.all(hist.H["sim_ended"][0:1])
    assert np.all(~hist.H["sim_ended"][1:10])  # Check the rest
    assert hist.sim_ended_count == 1
    assert hist.sim_started_count == 0  # In real case this would be ahead.....
    assert hist.index == 0  # In real case this would be ahead....

    # Update two further consecutive points
    size = 2
    sim_ids = [1, 2]  # First row to be filled
    calc_out = np.zeros(size, dtype=sim_specs["out"])

    a = np.array([[1, 3, 4], [-1, 2, 4]])
    calc_out["f"][0] = sim_specs["sim_f"](a)  # np.linalg.norm
    exp_fs[1] = calc_out["f"][0]
    calc_out["fvec"][0] = sim_specs["sim_f"](a, axis=0)  # np.linalg.norm
    exp_fvecs[1] = calc_out["fvec"][0]

    a = np.array([[2, 4, 4], [-1, 3, 4]])
    calc_out["f"][1] = sim_specs["sim_f"](a)  # np.linalg.norm
    exp_fs[2] = calc_out["f"][1]
    calc_out["fvec"][1] = sim_specs["sim_f"](a, axis=0)  # np.linalg.norm
    exp_fvecs[2] = calc_out["fvec"][1]

    D_recv = {
        "calc_out": calc_out,
        "persis_info": {},
        "libE_info": {"H_rows": sim_ids},
        "calc_status": WORKER_DONE,
        "calc_type": 2,
    }

    hist.update_history_f(D_recv, safe_mode)

    assert np.allclose(exp_fs, hist.H["f"])
    assert np.allclose(exp_fvecs, hist.H["fvec"])
    assert np.all(hist.H["sim_ended"][0:3])
    assert np.all(~hist.H["sim_ended"][3:10])  # Check the rest
    assert hist.sim_ended_count == 3
    assert hist.sim_started_count == 0  # In real case this would be ahead.....
    assert hist.index == 0  # In real case this would be ahead....

    # Update two further consecutive points but with sub_array fvec components
    size = 2
    sim_ids = [3, 4]  # First row to be filled
    calc_out = np.zeros(size, dtype=[("f", float), ("fvec", float, 2)])  # Only two values

    a = np.array([[1, 3, 4], [-1, 2, 4]])
    calc_out["f"][0] = sim_specs["sim_f"](a)  # np.linalg.norm
    exp_fs[3] = calc_out["f"][0]
    calc_out["fvec"][0][0], calc_out["fvec"][0][1], _ = sim_specs["sim_f"](a, axis=0)  # np.linalg.norm
    exp_fvecs[3] = [0.0, 0.0, 0.0]  # Point to a new array - so can fill in elements
    exp_fvecs[3][:2] = calc_out["fvec"][0]  # Change first two values

    a = np.array([[2, 4, 4], [-1, 3, 4]])
    calc_out["f"][1] = sim_specs["sim_f"](a)  # np.linalg.norm
    exp_fs[4] = calc_out["f"][1]
    calc_out["fvec"][1][0], calc_out["fvec"][1][1], _ = sim_specs["sim_f"](a, axis=0)  # np.linalg.norm
    exp_fvecs[4] = [0.0, 0.0, 0.0]  # Point to a new array - so can fill in elements
    exp_fvecs[4][:2] = calc_out["fvec"][1]  # Change first two values

    D_recv = {
        "calc_out": calc_out,
        "persis_info": {},
        "libE_info": {"H_rows": sim_ids},
        "calc_status": WORKER_DONE,
        "calc_type": 2,
    }

    hist.update_history_f(D_recv, safe_mode)

    assert np.allclose(exp_fs, hist.H["f"])
    assert np.allclose(exp_fvecs, hist.H["fvec"])
    assert np.all(hist.H["sim_ended"][0:5])
    assert np.all(~hist.H["sim_ended"][5:10])  # Check the rest
    assert hist.sim_ended_count == 5
    assert hist.sim_started_count == 0  # In real case this would be ahead.....
    assert hist.index == 0  # In real case this would be ahead....


def test_repack_fields():
    if "repack_fields" in globals():
        H0 = np.zeros(3, dtype=[("g", float), ("x", float), ("large", float, 1000000)])
        assert H0.itemsize != repack_fields(H0[["x", "g"]]).itemsize, "These should not be the same size"
        assert repack_fields(H0[["x", "g"]]).itemsize < 100, "This should not be that large"


if __name__ == "__main__":
    test_hist_init_1()
    test_hist_init_1A_H0()
    test_hist_init_2()
    test_grow_H()
    test_trim_H()
    test_update_history_x_in_Oempty()
    test_update_history_x_in()
    test_update_history_x_in_sim_ids()
    test_update_history_x_out()
    test_update_history_f()
    test_update_history_f_vec()
    test_repack_fields()
