import mock
import numpy as np
import pytest

import libensemble.tests.unit_tests.setup as setup
from libensemble.tools.fields_keys import libE_fields
from libensemble.utils.runners import Runner


def get_ufunc_args():
    sim_specs, gen_specs, exit_criteria = setup.make_criteria_and_specs_0()

    L = exit_criteria["sim_max"]
    H = np.zeros(L, dtype=list(set(libE_fields + sim_specs["out"] + gen_specs["out"])))

    H["sim_id"][-L:] = -1
    H["sim_started_time"][-L:] = np.inf

    sim_ids = np.zeros(1, dtype=int)
    Work = {
        "tag": 1,
        "persis_info": {},
        "libE_info": {"H_rows": sim_ids},
        "H_fields": sim_specs["in"],
    }
    calc_in = H[Work["H_fields"]][Work["libE_info"]["H_rows"]]
    return calc_in, sim_specs, gen_specs


def test_normal_runners():
    calc_in, sim_specs, gen_specs = get_ufunc_args()

    simrunner = Runner.from_specs(sim_specs)
    genrunner = Runner.from_specs(gen_specs)
    assert not hasattr(simrunner, "globus_compute_executor") and not hasattr(
        genrunner, "globus_compute_executor"
    ), "Globus Compute use should not be detected without setting endpoint fields"


def test_thread_runners():
    calc_in, sim_specs, gen_specs = get_ufunc_args()

    def tupilize(arg1, arg2):
        return (arg1, arg2)

    sim_specs["threaded"] = True  # TODO: undecided interface
    sim_specs["sim_f"] = tupilize
    persis_info = {"hello": "threads"}

    simrunner = Runner.from_specs(sim_specs)
    result = simrunner._result(calc_in, persis_info, {})
    assert result == (calc_in, persis_info)
    assert hasattr(simrunner, "thread_handle")
    simrunner.shutdown()


def test_persis_info_from_none():
    calc_in, sim_specs, gen_specs = get_ufunc_args()

    def tupilize(arg1, arg2):
        return (arg1, arg2)

    sim_specs["sim_f"] = tupilize
    simrunner = Runner(sim_specs)
    libE_info = {"H_rows": np.array([2, 3, 4]), "workerID": 1, "comm": "fakecomm"}

    result = simrunner.run(calc_in, {"libE_info": libE_info, "persis_info": None, "tag": 1})
    assert result == (calc_in, {})


@pytest.mark.extra
def test_globus_compute_runner_init():
    calc_in, sim_specs, gen_specs = get_ufunc_args()

    sim_specs["globus_compute_endpoint"] = "1234"

    with mock.patch("globus_compute_sdk.Executor"):
        runner = Runner.from_specs(sim_specs)

        assert hasattr(
            runner, "globus_compute_executor"
        ), "Globus ComputeExecutor should have been instantiated when globus_compute_endpoint found in specs"


@pytest.mark.extra
def test_globus_compute_runner_pass():
    calc_in, sim_specs, gen_specs = get_ufunc_args()

    sim_specs["globus_compute_endpoint"] = "1234"

    with mock.patch("globus_compute_sdk.Executor"):
        runner = Runner.from_specs(sim_specs)

        #  Creating Mock Globus ComputeExecutor and Globus Compute future object - no exception
        globus_compute_mock = mock.Mock()
        globus_compute_future = mock.Mock()
        globus_compute_mock.submit_to_registered_function.return_value = globus_compute_future
        globus_compute_future.exception.return_value = None
        globus_compute_future.result.return_value = (True, True)

        runner.globus_compute_executor = globus_compute_mock
        runners = {1: runner.run}

        libE_info = {"H_rows": np.array([2, 3, 4]), "workerID": 1, "comm": "fakecomm"}

        out, persis_info = runners[1](calc_in, {"libE_info": libE_info, "persis_info": {}, "tag": 1})

        assert all([out, persis_info]), "Globus Compute runner correctly returned results"


@pytest.mark.extra
def test_globus_compute_runner_fail():
    calc_in, sim_specs, gen_specs = get_ufunc_args()

    gen_specs["globus_compute_endpoint"] = "4321"

    with mock.patch("globus_compute_sdk.Executor"):
        runner = Runner.from_specs(gen_specs)

        #  Creating Mock Globus ComputeExecutor and Globus Compute future object - yes exception
        globus_compute_mock = mock.Mock()
        globus_compute_future = mock.Mock()
        globus_compute_mock.submit_to_registered_function.return_value = globus_compute_future
        globus_compute_future.exception.return_value = Exception

        runner.globus_compute_executor = globus_compute_mock
        runners = {2: runner.run}

        libE_info = {"H_rows": np.array([2, 3, 4]), "workerID": 1, "comm": "fakecomm"}

        with pytest.raises(Exception):
            out, persis_info = runners[2](calc_in, {"libE_info": libE_info, "persis_info": {}, "tag": 2})
            pytest.fail("Expected exception")


if __name__ == "__main__":
    test_normal_runners()
    test_thread_runners()
    test_persis_info_from_none()
    test_globus_compute_runner_init()
    test_globus_compute_runner_pass()
    test_globus_compute_runner_fail()
