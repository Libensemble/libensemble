import numpy as np
import libensemble.gen_funcs.persistent_aposmm as al
import libensemble.tests.unit_tests.setup as setup

libE_specs = {'comm': {}}


def test_persis_apossm_localopt_test():
    _, _, gen_specs_0, _, _ = setup.hist_setup1()

    H = np.zeros(4, dtype=[('f', float), ('sim_id', bool), ('dist_to_unit_bounds', float), ('returned', bool)])
    H['returned'] = True
    H['sim_id'] = range(len(H))
    gen_specs_0['user']['localopt_method'] = 'BADNAME'
    gen_specs_0['user']['ub'] = np.ones(2)
    gen_specs_0['user']['lb'] = np.zeros(2)

    try:
        al.aposmm(H, {}, gen_specs_0, libE_specs)
    except NotImplementedError:
        assert 1, "Failed because method is unknown."
    else:
        assert 0


def test_update_history_optimal():
    hist, _, _, _, _ = setup.hist_setup1(n=2)

    H = hist.H

    H['returned'] = True
    H['sim_id'] = range(len(H))
    H['f'][0] = -1e-8
    H['x_on_cube'][-1] = 1e-10

    # Perturb x_opt point to test the case where the reported minimum isn't
    # exactly in H. Also, a point in the neighborhood of x_opt has a better
    # function value.
    opt_ind = al.update_history_optimal(H['x_on_cube'][-1]+1e-12, 1, H, np.arange(len(H)))

    assert opt_ind == 9, "Wrong point declared minimum"


if __name__ == "__main__":
    test_persis_apossm_localopt_test()
    test_update_history_optimal()
