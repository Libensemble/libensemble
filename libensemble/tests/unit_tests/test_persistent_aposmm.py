import numpy as np
import libensemble.gen_funcs.persistent_aposmm as al
import libensemble.tests.unit_tests.setup as setup

libE_specs = {'comm': {}}


def test_persis_apossm_localopt_test():
    _, _, gen_specs_0, _, _ = setup.hist_setup1()

    H = np.zeros(4, dtype=[('f', float), ('returned', bool)])
    gen_specs_0['user']['localopt_method'] = 'BADNAME'
    gen_specs_0['user']['ub'] = np.ones(2)
    gen_specs_0['user']['lb'] = np.zeros(2)

    try:
        al.aposmm(H, {}, gen_specs_0, libE_specs)
    except NotImplementedError:
        assert 1, "Failed because method is unknown."
    else:
        assert 0


if __name__ == "__main__":
    test_persis_apossm_localopt_test()
