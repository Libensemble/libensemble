import time
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import numpy.lib.recfunctions
import pytest

import libensemble.manager as man
import libensemble.tests.unit_tests.setup as setup
from libensemble.message_numbers import EVAL_SIM_TAG

libE_specs = {"comms": "local", "gen_on_worker": True}


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
    hist.H["g"][0] = -1
    hist.H["sim_ended"][0] = True
    hist.index = 1
    hist.sim_started_count = 1
    assert mgr.term_test()
    #
    # Terminate because everything has been given.
    hist.H["sim_started"] = np.ones
    hist.sim_started_count = len(hist.H)
    assert mgr.term_test()


def test_term_test_3():
    # Test 3.
    # Terminate because enough time has passed
    H0 = np.zeros(3, dtype=[("g", float)] + [("x", float), ("priority", float)])
    hist, sim_specs, gen_specs, exit_criteria, al = setup.hist_setup2(H0_in=H0)
    mgr = man.Manager(hist, libE_specs, al, sim_specs, gen_specs, exit_criteria)
    hist.index = 4
    hist.H["sim_started_time"][0] = time.time()
    time.sleep(0.5)
    hist.sim_started_count = 4
    assert mgr.term_test()


def test_send_work_order_with_partial_cache_hit():
    mgr = SimpleNamespace(
        hist=SimpleNamespace(
            H=np.zeros(2, dtype=[("x", float)]),
            cache_set=True,
        ),
        use_cache=True,
        resources=None,
        cache_hit=False,
        wcomms=[Mock()],
    )
    mgr._update_state_from_cache = Mock(return_value=[0])
    work = {
        "tag": EVAL_SIM_TAG,
        "H_fields": ["x"],
        "libE_info": {"H_rows": np.array([0, 1])},
    }

    man.Manager._send_work_order(mgr, work, 0)

    np.testing.assert_array_equal(work["libE_info"]["H_rows"], [0, 1])
    sent_work = mgr.wcomms[0].send.call_args_list[0].args[1]
    assert sent_work["libE_info"]["H_rows"] == [1]


def test_run_rejects_idle_allocator_without_service_mode():
    mgr = SimpleNamespace(
        libE_specs={},
        exit_criteria={},
        hist=SimpleNamespace(trim_H=Mock(return_value=np.zeros(0)), save_cache=Mock()),
        W=np.zeros(1, dtype=man.Manager.worker_dtype),
        wcomms=[],
        _service_idle_start=None,
        term_test=Mock(side_effect=[False, False]),
        _kill_cancelled_sims=Mock(),
        _receive_from_workers_or_cache=Mock(return_value={}),
        _alloc_work=Mock(return_value=({}, {}, 0)),
        _final_receive_and_kill=Mock(return_value=({}, 0, 0)),
    )

    with pytest.raises(man.LoggedException, match="alloc_f did not return any work"):
        man.Manager.run(mgr, {})


if __name__ == "__main__":
    test_term_test_1()
    test_term_test_2()
    test_term_test_3()
