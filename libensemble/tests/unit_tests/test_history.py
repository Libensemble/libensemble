import libensemble.tests.unit_tests.setup as setup
from libensemble.message_numbers import WORKER_DONE
import numpy as np
from numpy import inf

if tuple(np.__version__.split('.')) >= ('1', '15'):
    from numpy.lib.recfunctions import repack_fields

# Consider fixtures for this - parameterization may save duplication if always use pytest.

# Comparing hist produced: options (using mix of first two)
# - hardcode array compare
# - compare selected values
# - compare from npy file - stored

wrs_H0 = np.array([(False, 0., 0, 0., 1, True, 1, True, [0., 0., 0.], True, 0.1, 1.1, False, False, False, inf),
                   (False, 0., 0, 0., 1, True, 2, True, [0., 0., 0.], True, 0.2, 1.2, False, False, False, inf),
                   (False, 0., 0, 0., 1, True, 3, True, [0., 0., 0.], True, 0.3, 1.3, False, False, False, inf)],
                  dtype=[('local_pt', '?'), ('priority', '<f8'), ('gen_worker', '<i8'), ('x_on_cube', '<f8'),
                         ('sim_id', '<i8'), ('given', '?'), ('sim_worker', '<i8'), ('returned', '?'),
                         ('fvec', '<f8', (3,)), ('allocated', '?'), ('f', '<f8'), ('given_time', '<f8'),
                         ('cancel_requested', '?'), ('kill_sent', '?'), ('given_back', '?'), ('last_given_back_time', '<f8')])

exp_H0_H = np.array([(False, 0., 0, 0., 1, True, 1, True, [0., 0., 0.], True, 0.1, 1.1, 2.1, False, False, False, inf),
                     (False, 0., 0, 0., 1, True, 2, True, [0., 0., 0.], True, 0.2, 1.2, 2.3, False, False, False, inf),
                     (False, 0., 0, 0., 1, True, 3, True, [0., 0., 0.], True, 0.3, 1.3, 2.3, False, False, False, inf),
                     (False, 0., 0, 0., -1, False, 0, False, [0., 0., 0.], False, 0., inf, 0., False, False, False, inf),
                     (False, 0., 0, 0., -1, False, 0, False, [0., 0., 0.], False, 0., inf, 0., False, False, False, inf)],
                    dtype=[('local_pt', '?'), ('priority', '<f8'), ('gen_worker', '<i8'), ('x_on_cube', '<f8'),
                           ('sim_id', '<i8'), ('given', '?'), ('sim_worker', '<i8'), ('returned', '?'),
                           ('fvec', '<f8', (3,)), ('allocated', '?'), ('f', '<f8'), ('given_time', '<f8'),
                           ('returned_time', '<f8'), ('cancel_requested', '?'), ('kill_sent', '?'),
                           ('given_back', '?'), ('last_given_back_time', '<f8')])

wrs = np.array([(False, 0., 0, 0., 0., 0., -1, False, 0, False, 0, [0., 0., 0.], False, 0., inf, inf, False, 0, False, False, False, inf),
                (False, 0., 0, 0., 0., 0., -1, False, 0, False, 0, [0., 0., 0.], False, 0., inf, inf, False, 0, False, False, False, inf),
                (False, 0., 0, 0., 0., 0., -1, False, 0, False, 0, [0., 0., 0.], False, 0., inf, inf, False, 0, False, False, False, inf),
                (False, 0., 0, 0., 0., 0., -1, False, 0, False, 0, [0., 0., 0.], False, 0., inf, inf, False, 0, False, False, False, inf),
                (False, 0., 0, 0., 0., 0., -1, False, 0, False, 0, [0., 0., 0.], False, 0., inf, inf, False, 0, False, False, False, inf),
                (False, 0., 0, 0., 0., 0., -1, False, 0, False, 0, [0., 0., 0.], False, 0., inf, inf, False, 0, False, False, False, inf),
                (False, 0., 0, 0., 0., 0., -1, False, 0, False, 0, [0., 0., 0.], False, 0., inf, inf, False, 0, False, False, False, inf),
                (False, 0., 0, 0., 0., 0., -1, False, 0, False, 0, [0., 0., 0.], False, 0., inf, inf, False, 0, False, False, False, inf),
                (False, 0., 0, 0., 0., 0., -1, False, 0, False, 0, [0., 0., 0.], False, 0., inf, inf, False, 0, False, False, False, inf),
                (False, 0., 0, 0., 0., 0., -1, False, 0, False, 0, [0., 0., 0.], False, 0., inf, inf, False, 0, False, False, False, inf)],
               dtype=[('local_pt', '?'), ('priority', '<f8'), ('gen_worker', '<i8'), ('gen_time', '<f8'),
                      ('last_gen_time', '<f8'), ('x_on_cube', '<f8'), ('sim_id', '<i8'), ('given', '?'),
                      ('sim_worker', '<i8'), ('returned', '?'), ('returned_time', '<f8'), ('fvec', '<f8', (3,)),
                      ('allocated', '?'), ('f', '<f8'), ('given_time', '<f8'), ('last_given_time', '<f8'),
                      ('local_min', '?'), ('num_active_runs', '<i8'), ('cancel_requested', '?'), ('kill_sent', '?'),
                      ('given_back', '?'), ('last_given_back_time', '<f8')])

wrs2 = np.array([(0, False, 0., 0., 0., 0., 0, False, 0., False, -1, inf, inf, 0., False, False, False, inf),
                 (0, False, 0., 0., 0., 0., 0, False, 0., False, -1, inf, inf, 0., False, False, False, inf),
                 (0, False, 0., 0., 0., 0., 0, False, 0., False, -1, inf, inf, 0., False, False, False, inf),
                 (0, False, 0., 0., 0., 0., 0, False, 0., False, -1, inf, inf, 0., False, False, False, inf),
                 (0, False, 0., 0., 0., 0., 0, False, 0., False, -1, inf, inf, 0., False, False, False, inf),
                 (0, False, 0., 0., 0., 0., 0, False, 0., False, -1, inf, inf, 0., False, False, False, inf),
                 (0, False, 0., 0., 0., 0., 0, False, 0., False, -1, inf, inf, 0., False, False, False, inf),
                 (0, False, 0., 0., 0., 0., 0, False, 0., False, -1, inf, inf, 0., False, False, False, inf),
                 (0, False, 0., 0., 0., 0., 0, False, 0., False, -1, inf, inf, 0., False, False, False, inf),
                 (0, False, 0., 0., 0., 0., 0, False, 0., False, -1, inf, inf, 0., False, False, False, inf)],
                dtype=[('gen_worker', '<i8'), ('returned', '?'), ('returned_time', '<f8'),
                       ('gen_time', '<f8'), ('last_gen_time', '<f8'), ('x', '<f8'),
                       ('sim_worker', '<i8'), ('allocated', '?'), ('g', '<f8'), ('given', '?'),
                       ('sim_id', '<i8'), ('given_time', '<f8'), ('last_given_time', '<f8'),
                       ('priority', '<f8'), ('cancel_requested', '?'), ('kill_sent', '?'),
                       ('given_back', '?'), ('last_given_back_time', '<f8')])

exp_x_in_setup2 = np.array([(0, 0, 2, 0., 4.17022005e-01, False, False, False, inf, 0., False, False, False, inf),
                            (0, 1, 3, 0., 7.20324493e-01, False, False, False, inf, 0., False, False, False, inf),
                            (0, 2, 3, 0., 1.14374817e-04, False, False, False, inf, 0., False, False, False, inf),
                            (0, 3, 3, 0., 3.02332573e-01, False, False, False, inf, 0., False, False, False, inf),
                            (0, 4, 3, 0., 1.46755891e-01, False, False, False, inf, 0., False, False, False, inf),
                            (0, 5, 3, 0., 9.23385948e-02, False, False, False, inf, 0., False, False, False, inf),
                            (0, 6, 3, 0., 1.86260211e-01, False, False, False, inf, 0., False, False, False, inf),
                            (0, 7, 3, 0., 3.45560727e-01, False, False, False, inf, 0., False, False, False, inf),
                            (0, 8, 3, 0., 3.96767474e-01, False, False, False, inf, 0., False, False, False, inf),
                            (0, 9, 3, 0., 5.38816734e-01, False, False, False, inf, 0., False, False, False, inf)],
                           dtype=[('sim_worker', '<i8'), ('sim_id', '<i8'), ('gen_worker', '<i8'), ('priority', '<f8'),
                                  ('x', '<f8'), ('allocated', '?'), ('returned', '?'), ('given', '?'),
                                  ('given_time', '<f8'), ('g', '<f8'), ('cancel_requested', '?'), ('kill_sent', '?'),
                                  ('given_back', '?'), ('last_given_back_time', '<f8')])

safe_mode = True


def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


# Tests ========================================================================================
def test_hist_init_1():
    hist, _, _, _, _ = setup.hist_setup1()
    assert np.array_equal(hist.H, wrs), "Array does not match expected"
    assert hist.given_count == 0
    assert hist.index == 0
    assert hist.returned_count == 0
    assert hist.given_back_count == 0


def test_hist_init_1A_H0():
    hist, _, _, _, _ = setup.hist_setup1(sim_max=2, H0_in=wrs_H0)

    # Compare by column
    for field in exp_H0_H.dtype.names:
        np.array_equal(hist.H[field], exp_H0_H[field])
    # These dont work for numpy structured arrays
    # assert np.array_equiv(hist.H, exp_H0_H), "Array does not match expected"
    # assert np.array_equal(hist.H, exp_H0_H), "Array does not match expected"
    assert hist.given_count == 3
    assert hist.index == 3
    assert hist.returned_count == 3
    assert hist.given_back_count == 0
    assert len(hist.H) == 5


def test_hist_init_2():
    hist, _, _, _, _ = setup.hist_setup2()
    assert np.array_equal(hist.H, wrs2), "Array does not match expected"
    assert hist.given_count == 0
    assert hist.index == 0
    assert hist.returned_count == 0
    assert hist.given_back_count == 0


def test_grow_H():
    hist, _, _, _, _ = setup.hist_setup1(3)
    new_rows = 7
    hist.grow_H(k=new_rows)
    assert np.array_equal(hist.H, wrs), "Array does not match expected"
    assert hist.given_count == 0
    assert hist.index == 0
    assert hist.returned_count == 0
    assert hist.given_back_count == 0


def test_trim_H():
    hist, _, _, _, _ = setup.hist_setup1(13)
    hist.index = 10
    H = hist.trim_H()
    assert np.array_equal(H, wrs), "Array does not match expected"
    assert hist.given_count == 0
    assert hist.index == 10
    assert hist.returned_count == 0
    assert hist.given_back_count == 0


def test_update_history_x_in_Oempty():
    hist, sim_specs, gen_specs, _, _ = setup.hist_setup2()
    H_o = np.zeros(0, dtype=gen_specs['out'])
    gen_worker = 1
    hist.update_history_x_in(gen_worker, H_o, safe_mode)
    assert np.array_equal(hist.H, wrs2), "H Array does not match expected"
    assert hist.given_count == 0
    assert hist.index == 0
    assert hist.returned_count == 0
    assert hist.given_back_count == 0


def test_update_history_x_in():
    hist, _, gen_specs, _, _ = setup.hist_setup2(7)

    np.random.seed(1)
    single_rand = gen_specs['gen_f']()  # np.random.uniform()

    # Check seeded correctly going in
    assert isclose(single_rand, 0.417022004702574), "Random numbers not correct before function"

    size = 1
    gen_worker = 2
    H_o = np.zeros(size, dtype=gen_specs['out'])
    H_o['x'] = single_rand

    hist.update_history_x_in(gen_worker, H_o, safe_mode)
    assert isclose(single_rand, hist.H['x'][0])
    assert hist.given_count == 0
    assert hist.index == 1
    assert hist.returned_count == 0
    assert hist.given_back_count == 0

    size = 6
    gen_worker = 3
    H_o = np.zeros(size, dtype=gen_specs['out'])
    H_o['x'] = gen_specs['gen_f'](size=size)

    hist.update_history_x_in(gen_worker, H_o, safe_mode)
    # Compare by column
    exp_x = exp_x_in_setup2[:size+1]
    for field in exp_x.dtype.names:
        np.allclose(hist.H[field], exp_x[field])

    assert hist.given_count == 0
    assert hist.index == 7
    assert hist.returned_count == 0
    assert hist.given_back_count == 0

    # Force H to grow when add points
    size = 3
    gen_worker = 3
    H_o = np.zeros(size, dtype=gen_specs['out'])
    H_o['x'] = gen_specs['gen_f'](size=size)

    hist.update_history_x_in(gen_worker, H_o, safe_mode)
    # Compare by column
    exp_x = exp_x_in_setup2
    for field in exp_x.dtype.names:
        np.allclose(hist.H[field], exp_x[field])

    assert hist.given_count == 0
    assert hist.index == 10
    assert hist.returned_count == 0

    # Test libE errors when a protected field appears in output from a gen_worker
    H_o = np.zeros(size, dtype=gen_specs['out'] + [('given', bool)])
    try:
        hist.update_history_x_in(gen_worker, H_o, safe_mode)
    except AssertionError:
        assert 1, "Failed like it should have"
    else:
        assert 0, "Didn't fail like it should have"

    # Test libE errors when a protected field appears in output from a gen_worker
    H_o = np.zeros(size, dtype=gen_specs['out'] + [('given', bool)])
    try:
        hist.update_history_x_in(gen_worker, H_o, safe_mode)
    except AssertionError:
        assert 1, "Failed like it should have"
    else:
        assert 0, "Didn't fail like it should have"


def test_update_history_x_in_sim_ids():
    hist, _, gen_specs, _, _ = setup.hist_setup2A_genout_sim_ids(7)

    np.random.seed(1)
    single_rand = gen_specs['gen_f']()  # np.random.uniform()

    # Check seeded correctly going in
    assert isclose(single_rand, 0.417022004702574), "Random numbers not correct before function"

    size = 1
    gen_worker = 2
    H_o = np.zeros(size, dtype=gen_specs['out'])
    H_o['x'] = single_rand
    H_o['sim_id'] = 0

    hist.update_history_x_in(gen_worker, H_o, safe_mode)
    assert isclose(single_rand, hist.H['x'][0])
    assert hist.given_count == 0
    assert hist.index == 1
    assert hist.returned_count == 0
    assert hist.given_back_count == 0

    size = 6
    gen_worker = 3
    H_o = np.zeros(size, dtype=gen_specs['out'])
    H_o['x'] = gen_specs['gen_f'](size=size)
    H_o['sim_id'] = range(1, 7)
    hist.update_history_x_in(gen_worker, H_o, safe_mode)

    # Compare by column
    exp_x = exp_x_in_setup2[:size+1]
    for field in exp_x.dtype.names:
        np.allclose(hist.H[field], exp_x[field])

    assert hist.given_count == 0
    assert hist.index == 7
    assert hist.returned_count == 0
    assert hist.given_back_count == 0

    # Force H to grow when add points
    size = 3
    gen_worker = 3
    H_o = np.zeros(size, dtype=gen_specs['out'])
    H_o['x'] = gen_specs['gen_f'](size=size)
    H_o['sim_id'] = range(7, 10)

    hist.update_history_x_in(gen_worker, H_o, safe_mode)
    # Compare by column
    exp_x = exp_x_in_setup2
    for field in exp_x.dtype.names:
        np.allclose(hist.H[field], exp_x[field])

    assert hist.given_count == 0
    assert hist.index == 10
    assert hist.returned_count == 0
    assert hist.given_back_count == 0


# Note - Ideally have more setup here (so hist.index reflects generated points)
def test_update_history_x_out():
    hist, _, _, _, _ = setup.hist_setup1()

    # First update a single point
    hist.update_history_x_out(q_inds=0, sim_worker=2)

    # Check updated values for point and counts
    assert hist.H['given'][0]
    assert hist.H['sim_worker'][0] == 2
    assert hist.given_count == 1

    # Check some unchanged values for point and counts
    assert hist.index == 0
    assert hist.returned_count == 0
    hist.H['returned'][0] = False
    hist.H['allocated'][0] = False
    hist.H['f'][0] == 0.0
    hist.H['sim_id'][0] == -1

    # Check the rest of H is unaffected
    assert np.array_equal(hist.H[1:10], wrs[1:10]), "H Array slice does not match expected"

    # Update two further consecutive points
    my_qinds = np.arange(1, 3)
    hist.update_history_x_out(q_inds=my_qinds, sim_worker=3)

    # Check updated values for point and counts
    assert np.all(hist.H['given'][0:3])  # Include previous point
    assert np.all(hist.H['sim_worker'][my_qinds] == 3)
    assert hist.given_count == 3

    # Update three further non-consecutive points
    my_qinds = np.array([4, 7, 9])
    hist.update_history_x_out(q_inds=my_qinds, sim_worker=4)

    # Try to avoid tautological testing - compare columns
    assert np.array_equal(hist.H['given'], np.array([True, True, True, False, True, False, False, True, False, True]))
    assert np.array_equal(hist.H['sim_worker'], np.array([2, 3, 3, 0, 4, 0, 0, 4, 0, 4]))
    assert np.all(~hist.H['returned'])  # Should still be unaffected.

    # Check counts
    assert hist.given_count == 6
    assert hist.index == 0  # In real case this would be ahead.....
    assert hist.returned_count == 0
    assert hist.given_back_count == 0


def test_update_history_f():
    hist, sim_specs, _, _, _ = setup.hist_setup2()
    exp_vals = [0.0] * 10

    # First update a single point
    size = 1
    sim_ids = [0]  # First row to be filled
    calc_out = np.zeros(size, dtype=sim_specs['out'])
    a = np.arange(9) - 4
    calc_out['g'] = sim_specs['sim_f'](a)  # np.linalg.norm
    exp_vals[0] = calc_out['g'][0]
    D_recv = {'calc_out': calc_out,
              'persis_info': {},
              'libE_info': {'H_rows': sim_ids},
              'calc_status': WORKER_DONE,
              'calc_type': 2}

    hist.update_history_f(D_recv, safe_mode)
    assert isclose(exp_vals[0], hist.H['g'][0])
    assert np.all(hist.H['returned'][0:1])
    assert np.all(~hist.H['returned'][1:10])  # Check the rest
    assert hist.returned_count == 1
    assert hist.given_count == 0  # In real case this would be ahead.....
    assert hist.index == 0  # In real case this would be ahead....

    # Update two further consecutive points
    size = 2
    sim_ids = [1, 2]  # First row to be filled
    calc_out = np.zeros(size, dtype=sim_specs['out'])
    a = np.arange(9) - 3
    calc_out['g'][0] = sim_specs['sim_f'](a)  # np.linalg.norm
    exp_vals[1] = calc_out['g'][0]
    a = np.arange(9) - 2
    calc_out['g'][1] = sim_specs['sim_f'](a)  # np.linalg.norm
    exp_vals[2] = calc_out['g'][1]
    D_recv = {'calc_out': calc_out,
              'persis_info': {},
              'libE_info': {'H_rows': sim_ids},
              'calc_status': WORKER_DONE,
              'calc_type': 2}

    hist.update_history_f(D_recv, safe_mode)
    assert np.allclose(exp_vals, hist.H['g'])
    assert np.all(hist.H['returned'][0:3])
    assert np.all(~hist.H['returned'][3:10])  # Check the rest
    assert hist.returned_count == 3
    assert hist.given_count == 0  # In real case this would be ahead.....
    assert hist.index == 0  # In real case this would be ahead....


def test_update_history_f_vec():
    hist, sim_specs, _, _, _ = setup.hist_setup1()
    exp_fs = [0.0] * 10
    exp_fvecs = [[0.0, 0.0, 0.0]] * 10

    # First update a single point
    size = 1
    sim_ids = [0]  # First row to be filled
    calc_out = np.zeros(size, dtype=sim_specs['out'])
    a = np.array([[1, 2, 3], [-1, 1, 4]])
    calc_out['f'] = sim_specs['sim_f'](a)  # np.linalg.norm
    calc_out['fvec'] = sim_specs['sim_f'](a, axis=0)  # np.linalg.norm
    exp_fs[0] = calc_out['f'][0]
    exp_fvecs[0] = calc_out['fvec'][0]
    D_recv = {'calc_out': calc_out,
              'persis_info': {},
              'libE_info': {'H_rows': sim_ids},
              'calc_status': WORKER_DONE,
              'calc_type': 2}

    hist.update_history_f(D_recv, safe_mode)

    assert isclose(exp_fs[0], hist.H['f'][0])
    assert np.allclose(exp_fvecs[0], hist.H['fvec'][0])
    assert np.all(hist.H['returned'][0:1])
    assert np.all(~hist.H['returned'][1:10])  # Check the rest
    assert hist.returned_count == 1
    assert hist.given_count == 0  # In real case this would be ahead.....
    assert hist.index == 0  # In real case this would be ahead....

    # Update two further consecutive points
    size = 2
    sim_ids = [1, 2]  # First row to be filled
    calc_out = np.zeros(size, dtype=sim_specs['out'])

    a = np.array([[1, 3, 4], [-1, 2, 4]])
    calc_out['f'][0] = sim_specs['sim_f'](a)  # np.linalg.norm
    exp_fs[1] = calc_out['f'][0]
    calc_out['fvec'][0] = sim_specs['sim_f'](a, axis=0)  # np.linalg.norm
    exp_fvecs[1] = calc_out['fvec'][0]

    a = np.array([[2, 4, 4], [-1, 3, 4]])
    calc_out['f'][1] = sim_specs['sim_f'](a)  # np.linalg.norm
    exp_fs[2] = calc_out['f'][1]
    calc_out['fvec'][1] = sim_specs['sim_f'](a, axis=0)  # np.linalg.norm
    exp_fvecs[2] = calc_out['fvec'][1]

    D_recv = {'calc_out': calc_out,
              'persis_info': {},
              'libE_info': {'H_rows': sim_ids},
              'calc_status': WORKER_DONE,
              'calc_type': 2}

    hist.update_history_f(D_recv, safe_mode)

    assert np.allclose(exp_fs, hist.H['f'])
    assert np.allclose(exp_fvecs, hist.H['fvec'])
    assert np.all(hist.H['returned'][0:3])
    assert np.all(~hist.H['returned'][3:10])  # Check the rest
    assert hist.returned_count == 3
    assert hist.given_count == 0  # In real case this would be ahead.....
    assert hist.index == 0  # In real case this would be ahead....

    # Update two further consecutive points but with sub_array fvec components
    size = 2
    sim_ids = [3, 4]  # First row to be filled
    calc_out = np.zeros(size, dtype=[('f', float), ('fvec', float, 2)])  # Only two values

    a = np.array([[1, 3, 4], [-1, 2, 4]])
    calc_out['f'][0] = sim_specs['sim_f'](a)  # np.linalg.norm
    exp_fs[3] = calc_out['f'][0]
    calc_out['fvec'][0][0], calc_out['fvec'][0][1], _ = sim_specs['sim_f'](a, axis=0)  # np.linalg.norm
    exp_fvecs[3] = [0.0, 0.0, 0.0]  # Point to a new array - so can fill in elements
    exp_fvecs[3][:2] = calc_out['fvec'][0]  # Change first two values

    a = np.array([[2, 4, 4], [-1, 3, 4]])
    calc_out['f'][1] = sim_specs['sim_f'](a)  # np.linalg.norm
    exp_fs[4] = calc_out['f'][1]
    calc_out['fvec'][1][0], calc_out['fvec'][1][1], _ = sim_specs['sim_f'](a, axis=0)  # np.linalg.norm
    exp_fvecs[4] = [0.0, 0.0, 0.0]  # Point to a new array - so can fill in elements
    exp_fvecs[4][:2] = calc_out['fvec'][1]  # Change first two values

    D_recv = {'calc_out': calc_out,
              'persis_info': {},
              'libE_info': {'H_rows': sim_ids},
              'calc_status': WORKER_DONE,
              'calc_type': 2}

    hist.update_history_f(D_recv, safe_mode)

    assert np.allclose(exp_fs, hist.H['f'])
    assert np.allclose(exp_fvecs, hist.H['fvec'])
    assert np.all(hist.H['returned'][0:5])
    assert np.all(~hist.H['returned'][5:10])  # Check the rest
    assert hist.returned_count == 5
    assert hist.given_count == 0  # In real case this would be ahead.....
    assert hist.index == 0  # In real case this would be ahead....


def test_repack_fields():
    if 'repack_fields' in globals():
        H0 = np.zeros(3, dtype=[('g', float), ('x', float), ('large', float, 1000000)])
        assert H0.itemsize != repack_fields(H0[['x', 'g']]).itemsize, "These should not be the same size"
        assert repack_fields(H0[['x', 'g']]).itemsize < 100, "This should not be that large"


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
