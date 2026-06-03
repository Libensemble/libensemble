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

    sim_specs["globus_compute_endpoint"] = "4321"

    with mock.patch("globus_compute_sdk.Executor"):
        runner = Runner.from_specs(sim_specs)

        #  Creating Mock Globus ComputeExecutor and Globus Compute future object - yes exception
        globus_compute_mock = mock.Mock()
        globus_compute_future = mock.Mock()
        globus_compute_mock.submit_to_registered_function.return_value = globus_compute_future
        globus_compute_future.exception.return_value = Exception

        runner.globus_compute_executor = globus_compute_mock
        runners = {1: runner.run}

        libE_info = {"H_rows": np.array([2, 3, 4]), "workerID": 1, "comm": "fakecomm"}

        with pytest.raises(Exception):
            out, persis_info = runners[1](calc_in, {"libE_info": libE_info, "persis_info": {}, "tag": 1})
            pytest.fail("Expected exception")


def test_libensemble_gen_runner_loop_with_updates():
    """Test that LibensembleGenRunner._loop_over_gen sends updates with keep_state=True."""
    from libensemble.message_numbers import EVAL_GEN_TAG, PERSIS_STOP
    from libensemble.utils.runners import LibensembleGenRunner

    # Create mock generator with suggest_numpy, suggest_updates, ingest_numpy
    mock_gen = mock.Mock()
    mock_gen.variables_mapping = {}

    H_out = np.zeros(2, dtype=[("x", float)])
    H_out["x"] = [1.0, 2.0]
    mock_gen.suggest_numpy.return_value = H_out

    cancel_update = np.zeros(1, dtype=[("sim_id", int), ("cancel_requested", bool)])
    cancel_update["sim_id"] = [5]
    cancel_update["cancel_requested"] = True
    mock_gen.suggest_updates.return_value = [cancel_update]

    mock_gen.ingest_numpy.return_value = None

    specs = {"generator": mock_gen, "batch_size": 2}
    runner = LibensembleGenRunner(specs)

    # Mock PersistentSupport
    mock_ps = mock.Mock()
    H_in = np.zeros(2, dtype=[("f", float), ("sim_id", int)])
    H_in["f"] = [0.5, 0.6]
    H_in["sim_id"] = [0, 1]

    # First recv returns EVAL_GEN_TAG (continue), second returns PERSIS_STOP (exit)
    mock_ps.recv.side_effect = [
        (EVAL_GEN_TAG, {}, H_in),
        (PERSIS_STOP, {}, None),
    ]
    runner.ps = mock_ps

    # Start the loop with a non-STOP tag
    runner._loop_over_gen(EVAL_GEN_TAG, {}, H_in)

    # Verify: send was called with H_out, then with cancel_update using keep_state=True
    assert mock_ps.send.call_count >= 2, "send should be called at least twice (H_out + update)"

    # First send call: H_out (no keep_state)
    first_send_args, first_send_kwargs = mock_ps.send.call_args_list[0]
    np.testing.assert_array_equal(first_send_args[0], H_out)
    assert first_send_kwargs.get("keep_state", False) is False or "keep_state" not in first_send_kwargs

    # Second send call: cancel_update with keep_state=True
    second_send_args, second_send_kwargs = mock_ps.send.call_args_list[1]
    np.testing.assert_array_equal(second_send_args[0], cancel_update)
    assert second_send_kwargs.get("keep_state") is True, "Updates should be sent with keep_state=True"

    # recv was used (not send_recv) when updates were present
    assert mock_ps.recv.call_count >= 1
    # send_recv should NOT have been called on the iteration with updates
    mock_ps.send_recv.assert_not_called()


def test_libensemble_gen_runner_loop_without_updates():
    """Test that LibensembleGenRunner._loop_over_gen uses send_recv when no updates."""
    from libensemble.message_numbers import EVAL_GEN_TAG, PERSIS_STOP
    from libensemble.utils.runners import LibensembleGenRunner

    mock_gen = mock.Mock()
    mock_gen.variables_mapping = {}

    H_out = np.zeros(2, dtype=[("x", float)])
    mock_gen.suggest_numpy.return_value = H_out
    mock_gen.suggest_updates.return_value = []  # Empty updates
    mock_gen.ingest_numpy.return_value = None

    specs = {"generator": mock_gen, "batch_size": 2}
    runner = LibensembleGenRunner(specs)

    mock_ps = mock.Mock()
    H_in = np.zeros(2, dtype=[("f", float), ("sim_id", int)])

    mock_ps.send_recv.return_value = (PERSIS_STOP, {}, None)
    runner.ps = mock_ps

    runner._loop_over_gen(EVAL_GEN_TAG, {}, H_in)

    # send_recv should be used when there are no updates
    mock_ps.send_recv.assert_called_once()
    # send should NOT have been called separately
    mock_ps.send.assert_not_called()


if __name__ == "__main__":
    test_normal_runners()
    test_thread_runners()
    test_persis_info_from_none()
    test_globus_compute_runner_init()
    test_globus_compute_runner_pass()
    test_globus_compute_runner_fail()
    test_libensemble_gen_runner_loop_with_updates()
    test_libensemble_gen_runner_loop_without_updates()
