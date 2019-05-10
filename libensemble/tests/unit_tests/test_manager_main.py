import time
import numpy as np
import numpy.lib.recfunctions
from mpi4py import MPI

import libensemble.libE_manager as man
import libensemble.tests.unit_tests.setup as setup

libE_specs = {'comm': MPI.COMM_WORLD}


def test_term_test_1():
    # termination_test should be True when we want to stop

    # sh - Should separate the tests
    # Test 1
    hist, sim_specs, gen_specs, exit_criteria, al = setup.hist_setup1()
    mgr = man.Manager(hist, libE_specs, al, sim_specs, gen_specs, exit_criteria)
    assert not mgr.term_test()


def test_term_test_2():
    # Test 2 - these could also be sep - with a setup or fixture....
    # Shouldn't terminate
    hist, sim_specs, gen_specs, exit_criteria, al = setup.hist_setup2()
    mgr = man.Manager(hist, libE_specs, al, sim_specs, gen_specs, exit_criteria)
    assert not mgr.term_test()
    #
    # Terminate because we've found a good 'g' value
    # H, hist.index, term_test, _, _, hist.given_count = man.initialize(sim_specs, gen_specs, al, exit_criteria, [], libE_specs)
    hist.H['g'][0] = -1
    hist.index = 1
    hist.given_count = 1
    assert mgr.term_test()
    #
    # Terminate because everything has been given.
    # H, hist.index, term_test, _, _, hist.given_count = man.initialize(sim_specs, gen_specs, al, exit_criteria, [], libE_specs)
    hist.H['given'] = np.ones
    hist.given_count = len(hist.H)
    assert mgr.term_test()


def test_term_test_3():
    # Test 3.
    # Terminate because enough time has passed
    H0 = np.zeros(3, dtype=[('g', float)] + [('x', float), ('priority', float)])
    hist, sim_specs, gen_specs, exit_criteria, al = setup.hist_setup2(H0_in=H0)
    mgr = man.Manager(hist, libE_specs, al, sim_specs, gen_specs, exit_criteria)
    hist.index = 4
    hist.H['given_time'][0] = time.time()
    time.sleep(0.5)
    hist.given_count = 4
    assert mgr.term_test()
    #


if __name__ == "__main__":
    test_term_test_1()
    test_term_test_2()
    test_term_test_3()
