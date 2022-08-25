import numpy as np
import pytest
import mock

import libensemble.tests.unit_tests.setup as setup
from libensemble.tools.fields_keys import libE_fields
from libensemble.message_numbers import EVAL_SIM_TAG, EVAL_GEN_TAG
from libensemble.utils.runners import Runners


def get_ufunc_args():
    sim_specs, gen_specs, exit_criteria = setup.make_criteria_and_specs_0()

    L = exit_criteria["sim_max"]
    H = np.zeros(L, dtype=list(set(libE_fields + sim_specs["out"] + gen_specs["out"])))

    H["sim_id"][-L:] = -1
    H["sim_started_time"][-L:] = np.inf

    sim_ids = np.zeros(1, dtype=int)
    Work = {
        "tag": EVAL_SIM_TAG,
        "persis_info": {},
        "libE_info": {"H_rows": sim_ids},
        "H_fields": sim_specs["in"],
    }
    calc_in = H[Work["H_fields"]][Work["libE_info"]["H_rows"]]
    return calc_in, sim_specs, gen_specs


@pytest.mark.extra
def test_normal_runners():
    calc_in, sim_specs, gen_specs = get_ufunc_args()

    runners = Runners(sim_specs, gen_specs)
    assert (
        not runners.has_funcx_sim and not runners.has_funcx_gen
    ), "funcX use should not be detected without setting endpoint fields"

    ro = runners.make_runners()
    assert all(
        [i in ro for i in [EVAL_SIM_TAG, EVAL_GEN_TAG]]
    ), "Both user function tags should be included in runners dictionary"


@pytest.mark.extra
def test_normal_no_gen():
    calc_in, sim_specs, gen_specs = get_ufunc_args()

    runners = Runners(sim_specs, {})
    ro = runners.make_runners()

    assert not ro[2], "generator function shouldn't be provided if not using gen_specs"


@pytest.mark.extra
def test_funcx_runner_init():
    calc_in, sim_specs, gen_specs = get_ufunc_args()

    sim_specs["funcx_endpoint"] = "1234"

    with mock.patch("funcx.FuncXClient"):

        runners = Runners(sim_specs, gen_specs)

        assert (
            runners.funcx_exctr is not None
        ), "FuncXExecutor should have been instantiated when funcx_endpoint found in specs"


@pytest.mark.extra
def test_funcx_runner_pass():
    calc_in, sim_specs, gen_specs = get_ufunc_args()

    sim_specs["funcx_endpoint"] = "1234"

    with mock.patch("funcx.FuncXClient"):

        runners = Runners(sim_specs, gen_specs)

        #  Creating Mock funcXExecutor and funcX future object - no exception
        funcx_mock = mock.Mock()
        funcx_future = mock.Mock()
        funcx_mock.submit.return_value = funcx_future
        funcx_future.exception.return_value = None
        funcx_future.result.return_value = (True, True)

        runners.funcx_exctr = funcx_mock
        ro = runners.make_runners()

        libE_info = {"H_rows": np.array([2, 3, 4]), "workerID": 1, "comm": "fakecomm"}
        out, persis_info = ro[1](calc_in, {}, libE_info)

        assert all([out, persis_info]), "funcX runner correctly returned results"


@pytest.mark.extra
def test_funcx_runner_fail():
    calc_in, sim_specs, gen_specs = get_ufunc_args()

    gen_specs["funcx_endpoint"] = "4321"

    with mock.patch("funcx.FuncXClient"):

        runners = Runners(sim_specs, gen_specs)

        #  Creating Mock funcXExecutor and funcX future object - yes exception
        funcx_mock = mock.Mock()
        funcx_future = mock.Mock()
        funcx_mock.submit.return_value = funcx_future
        funcx_future.exception.return_value = Exception

        runners.funcx_exctr = funcx_mock
        ro = runners.make_runners()

        libE_info = {"H_rows": np.array([2, 3, 4]), "workerID": 1, "comm": "fakecomm"}

        with pytest.raises(Exception):
            out, persis_info = ro[2](calc_in, {}, libE_info)
            pytest.fail("Expected exception")


if __name__ == "__main__":
    test_normal_runners()
    test_normal_no_gen()
    test_funcx_runner_init()
    test_funcx_runner_pass()
    test_funcx_runner_fail()
