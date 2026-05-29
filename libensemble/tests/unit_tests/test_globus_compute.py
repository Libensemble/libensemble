from unittest import mock

import numpy as np
import pytest

from libensemble.executors.globus_compute_executor import (
    GlobusComputeExecutor,
    GlobusComputeTask,
)
from libensemble.manager import Manager
from libensemble.message_numbers import TASK_FAILED, WORKER_DONE
from libensemble.utils.globus_compute import GCSession

# ──────────────────────────────────────────────
# GCSession
# ──────────────────────────────────────────────


class TestGCSession:
    def setup_method(self):
        GCSession.clear()

    def test_get_or_create_executor(self):
        with mock.patch.object(GCSession, "_create_executor") as mock_create:
            mock_exec = mock.MagicMock()
            mock_create.return_value = mock_exec

            ex1 = GCSession.get_or_create_executor("ep-1")
            assert ex1 is mock_exec
            mock_create.assert_called_once_with("ep-1")

            # Cache hit
            ex2 = GCSession.get_or_create_executor("ep-1")
            assert ex2 is mock_exec
            mock_create.assert_called_once()  # No second call

    def test_get_or_create_caches_func_id(self):
        with mock.patch.object(GCSession, "_create_executor") as mock_create:
            mock_exec = mock.MagicMock()
            mock_exec.register_function.return_value = "fid-42"
            mock_create.return_value = mock_exec

            def my_func():
                pass

            ex1, fid1 = GCSession.get_or_create("ep-1", my_func)
            assert ex1 is mock_exec
            assert fid1 == "fid-42"
            mock_exec.register_function.assert_called_once_with(my_func)

            # Second call — same (executor, fid) returned, no re-registration
            ex2, fid2 = GCSession.get_or_create("ep-1", my_func)
            assert ex2 is mock_exec
            assert fid2 == "fid-42"
            mock_exec.register_function.assert_called_once()

    def test_register_function(self):
        with mock.patch.object(GCSession, "_create_executor") as mock_create:
            mock_exec = mock.MagicMock()
            mock_exec.register_function.return_value = "fid-99"
            mock_create.return_value = mock_exec

            def f():
                pass

            ex, fid = GCSession.register_function("ep-1", f)
            assert ex is mock_exec
            assert fid == "fid-99"
            mock_exec.register_function.assert_called_once_with(f)

    def test_module_not_found_returns_none(self):
        # Force _create_executor to simulate missing SDK
        GCSession._create_executor = classmethod(lambda cls, eid: None)

        ex = GCSession.get_or_create_executor("ep-1")
        assert ex is None

        ex, fid = GCSession.get_or_create("ep-1", lambda: None)
        assert ex is None
        assert fid is None

    def test_thread_safety(self):
        import threading

        with mock.patch.object(GCSession, "_create_executor") as mock_create:
            mock_exec = mock.MagicMock()
            mock_exec.register_function.return_value = "fid"
            mock_create.return_value = mock_exec

            errors = []

            def access():
                try:
                    for _ in range(100):
                        GCSession.get_or_create_executor("ep-t")
                        GCSession.get_or_create("ep-t", lambda: None)
                except Exception as e:
                    errors.append(e)

            threads = [threading.Thread(target=access) for _ in range(10)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert not errors, f"Thread safety errors: {errors}"


# ──────────────────────────────────────────────
# GlobusComputeTask
# ──────────────────────────────────────────────


class TestGlobusComputeTask:
    def make_task(self, future=None, app=None):
        if app is None:
            from libensemble.executors.executor import Application

            app = Application("", name="test_func", calc_type="sim", pyobj=lambda: None)
        if future is None:
            future = mock.MagicMock()
            future.done.return_value = False
        return GlobusComputeTask(future, app=app)

    def test_initial_state(self):
        task = self.make_task()
        assert task.state == "RUNNING"
        assert not task.finished
        assert task._gc_future is not None

    def test_poll_running(self):
        future = mock.MagicMock()
        future.done.return_value = False
        task = self.make_task(future=future)
        task.poll()
        assert task.state == "RUNNING"
        assert not task.finished

    def test_poll_finished_success(self):
        future = mock.MagicMock()
        future.done.return_value = True
        future.result.return_value = None  # No exception
        task = self.make_task(future=future)
        task.poll()
        assert task.state == "FINISHED"
        assert task.finished
        assert task.success

    def test_poll_finished_failure(self):
        future = mock.MagicMock()
        future.done.return_value = True
        future.result.side_effect = RuntimeError("boom")
        task = self.make_task(future=future)
        task.poll()
        assert task.state == "FAILED"
        assert task.finished
        assert not task.success

    def test_wait_timeout(self):
        future = mock.MagicMock()
        future.result.side_effect = TimeoutError("timed out")
        task = self.make_task(future=future)
        with pytest.raises(Exception, match="timed out"):
            task.wait(timeout=0.001)

    def test_kill(self):
        future = mock.MagicMock()
        task = self.make_task(future=future)
        task.kill()
        assert task.state == "USER_KILLED"
        assert task.finished
        future.cancel.assert_called_once()

    def test_running(self):
        future = mock.MagicMock()
        future.done.return_value = False
        task = self.make_task(future=future)
        assert task.running()

    def test_done(self):
        future = mock.MagicMock()
        future.done.return_value = True
        future.result.return_value = None
        task = self.make_task(future=future)
        assert task.done()

    def test_not_done(self):
        future = mock.MagicMock()
        future.done.return_value = False
        task = self.make_task(future=future)
        assert not task.done()


# ──────────────────────────────────────────────
# GlobusComputeExecutor
# ──────────────────────────────────────────────


class TestGlobusComputeExecutor:
    def setup_method(self):
        GCSession.clear()

    def test_init(self):
        exctr = GlobusComputeExecutor(endpoint_id="ep-test")
        assert exctr.endpoint_id == "ep-test"
        assert exctr._gc_executor is None  # Lazy init
        assert exctr.workerID is None

    def test_submit_with_func(self):
        with mock.patch.object(GCSession, "_create_executor") as mock_create:
            mock_exec = mock.MagicMock()
            mock_exec.register_function.return_value = "fid-xyz"
            mock_create.return_value = mock_exec

            exctr = GlobusComputeExecutor(endpoint_id="ep-test")
            exctr._ensure_gc()

            future_mock = mock.MagicMock()
            mock_exec.submit_to_registered_function.return_value = future_mock

            def my_func(x):
                return x * 2

            task = exctr.submit(func=my_func, app_args="hello")
            assert isinstance(task, GlobusComputeTask)
            assert task._gc_future is future_mock
            assert task.app is not None
            assert task.app.name == "my_func"

    def test_submit_with_registered_app_pyobj(self):
        with mock.patch.object(GCSession, "_create_executor") as mock_create:
            mock_exec = mock.MagicMock()
            mock_exec.register_function.return_value = "fid-app"
            mock_create.return_value = mock_exec

            exctr = GlobusComputeExecutor(endpoint_id="ep-test")
            exctr._ensure_gc()

            def app_func():
                return 42

            exctr.register_app("/fake/path", app_name="myapp", calc_type="sim", pyobj=app_func)

            future_mock = mock.MagicMock()
            mock_exec.submit_to_registered_function.return_value = future_mock

            task = exctr.submit(app_name="myapp")
            assert isinstance(task, GlobusComputeTask)
            assert task.app.name == "myapp"

    def test_submit_without_func_or_app_raises(self):
        exctr = GlobusComputeExecutor(endpoint_id="ep-test")
        with pytest.raises(ValueError):
            exctr.submit()

    def test_register_function_caching(self):
        with mock.patch.object(GCSession, "_create_executor") as mock_create:
            mock_exec = mock.MagicMock()
            mock_exec.register_function.return_value = "fid-cached"
            mock_create.return_value = mock_exec

            exctr = GlobusComputeExecutor(endpoint_id="ep-test")
            exctr._ensure_gc()

            def my_func():
                pass

            fid1 = exctr._get_func_id(my_func)
            fid2 = exctr._get_func_id(my_func)
            assert fid1 == fid2
            # Should only register once
            assert mock_exec.register_function.call_count == 1

    def test_manager_kill_received(self):
        exctr = GlobusComputeExecutor(endpoint_id="ep-test")
        # Set up a mock comm so manager_poll doesn't assert
        mock_comm = mock.MagicMock()
        mock_comm.mail_flag.return_value = False
        exctr.set_worker_info(comm=mock_comm, workerid=1)
        assert exctr.manager_kill_received() is False

    def test_register_app_no_pyobj(self):
        exctr = GlobusComputeExecutor(endpoint_id="ep-test")
        exctr.register_app("/bin/echo", app_name="echo", calc_type="sim")
        app = exctr.get_app("echo")
        assert app.pyobj is None
        assert app.full_path == "/bin/echo"


# ──────────────────────────────────────────────
# _normalize_gc_result (Manager static method)
# ──────────────────────────────────────────────


class TestNormalizeGcResult:
    """Tests for Manager._normalize_gc_result which normalizes sim_f
    return values (2-tuple from gest-api, 3-tuple from legacy) to a
    consistent 3-tuple ``(out, persis_info, calc_status)``."""

    def test_3_tuple_passthrough(self):
        """Legacy sim_f returns (H_o, persis_info, calc_status)."""
        H_o = np.zeros(1, dtype=[("y", float)])
        result = (H_o, {"a": 1}, WORKER_DONE)
        out, p_info, status = Manager._normalize_gc_result(result)
        assert out is H_o
        assert p_info == {"a": 1}
        assert status == WORKER_DONE

    def test_3_tuple_with_task_failed(self):
        """Legacy sim_f returns a failure status."""
        H_o = np.zeros(1, dtype=[("y", float)])
        result = (H_o, {}, TASK_FAILED)
        out, p_info, status = Manager._normalize_gc_result(result)
        assert out is H_o
        assert p_info == {}
        assert status == TASK_FAILED

    def test_2_tuple_with_persis_info(self):
        """gest-api wrapper returns (H_o, persis_info) — no calc_status."""
        H_o = np.zeros(1, dtype=[("y", float)])
        result = (H_o, {"key": "val"})
        out, p_info, status = Manager._normalize_gc_result(result)
        assert out is H_o
        assert p_info == {"key": "val"}
        assert status == WORKER_DONE

    def test_2_tuple_with_empty_persis_info(self):
        """gest-api wrapper returns (H_o, {})."""
        H_o = np.zeros(1, dtype=[("y", float)])
        result = (H_o, {})
        out, p_info, status = Manager._normalize_gc_result(result)
        assert out is H_o
        assert p_info == {}
        assert status == WORKER_DONE

    def test_2_tuple_with_int_status(self):
        """Some sim_f may return (H_o, calc_status) without persis_info."""
        H_o = np.zeros(1, dtype=[("y", float)])
        result = (H_o, WORKER_DONE)
        out, p_info, status = Manager._normalize_gc_result(result)
        assert out is H_o
        assert p_info == {}
        assert status == WORKER_DONE

    def test_2_tuple_with_str_status(self):
        """Sim_f returns (H_o, 'some_status_string')."""
        H_o = np.zeros(1, dtype=[("y", float)])
        result = (H_o, "Custom status")
        out, p_info, status = Manager._normalize_gc_result(result)
        assert out is H_o
        assert p_info == {}
        assert status == "Custom status"

    def test_1_tuple(self):
        """Edge case: single-element tuple."""
        H_o = np.zeros(1, dtype=[("y", float)])
        result = (H_o,)
        out, p_info, status = Manager._normalize_gc_result(result)
        assert out is H_o
        assert p_info == {}
        assert status == WORKER_DONE

    def test_non_tuple(self):
        """Edge case: bare numpy array (not wrapped in tuple)."""
        H_o = np.zeros(1, dtype=[("y", float)])
        out, p_info, status = Manager._normalize_gc_result(H_o)
        assert out is H_o
        assert p_info == {}
        assert status == WORKER_DONE

    def test_list_result(self):
        """Result returned as a list instead of tuple."""
        H_o = np.zeros(1, dtype=[("y", float)])
        result = [H_o, {"x": 1}, WORKER_DONE]
        out, p_info, status = Manager._normalize_gc_result(result)
        assert out is H_o
        assert p_info == {"x": 1}
        assert status == WORKER_DONE


# ──────────────────────────────────────────────
# Manager-side GC with gest-api sim_f
# ──────────────────────────────────────────────


class TestGatherGcResultsGestApi:
    """Tests that _gather_gc_results correctly handles futures returning
    2-tuples (gest-api) and 3-tuples (legacy)."""

    def setup_method(self):
        GCSession.clear()

    def _make_mock_manager(self):
        """Create a minimal mock Manager with just enough structure for
        _gather_gc_results to run."""
        mgr = mock.MagicMock(spec=Manager)
        mgr._gc_futures = {}
        mgr._normalize_gc_result = Manager._normalize_gc_result
        mgr._gather_gc_results = lambda pi: Manager._gather_gc_results(mgr, pi)
        mgr._init_every_k_save = mock.MagicMock()

        # Minimal W array with one gen worker (w=0)
        W = np.zeros(2, dtype=[("worker_id", int)])
        W["worker_id"] = [0, 1]
        mgr.W = W

        # wcomms: gen worker has a comm, virtual worker is None
        mock_comm = mock.MagicMock()
        mock_comm.mail_flag.return_value = False
        mgr.wcomms = {0: mock_comm, 1: None}

        return mgr

    def test_2_tuple_gest_api_future(self):
        """Future returns (H_o, persis_info) — gest-api sim via gest_api_sim wrapper."""
        mgr = self._make_mock_manager()

        H_o = np.zeros(1, dtype=[("y", float)])
        H_o["y"] = 42.0
        future = mock.MagicMock()
        future.done.return_value = True
        future.result.return_value = (H_o, {"some": "info"})

        mgr._gc_futures[future] = (0, 1)  # sim_id=0, virtual_w=1
        mgr._gather_gc_results({})

        mgr._update_state_on_worker_msg.assert_called_once()
        call_args = mgr._update_state_on_worker_msg.call_args
        D_recv = call_args[0][1]
        assert D_recv["calc_status"] == WORKER_DONE
        assert D_recv["calc_out"] is H_o
        assert D_recv["persis_info"] == {"some": "info"}
        assert D_recv["libE_info"]["H_rows"] == [0]

    def test_3_tuple_legacy_future(self):
        """Future returns (H_o, persis_info, calc_status) — legacy sim_f."""
        mgr = self._make_mock_manager()

        H_o = np.zeros(1, dtype=[("y", float)])
        future = mock.MagicMock()
        future.done.return_value = True
        future.result.return_value = (H_o, {}, WORKER_DONE)

        mgr._gc_futures[future] = (5, 1)
        mgr._gather_gc_results({})

        call_args = mgr._update_state_on_worker_msg.call_args
        D_recv = call_args[0][1]
        assert D_recv["calc_status"] == WORKER_DONE
        assert D_recv["calc_out"] is H_o

    def test_failed_future(self):
        """Future raises an exception — should get TASK_FAILED status."""
        mgr = self._make_mock_manager()

        future = mock.MagicMock()
        future.done.return_value = True
        future.result.side_effect = RuntimeError("remote error")

        mgr._gc_futures[future] = (3, 1)
        mgr._gather_gc_results({})

        call_args = mgr._update_state_on_worker_msg.call_args
        D_recv = call_args[0][1]
        assert D_recv["calc_status"] == TASK_FAILED
        assert D_recv["calc_out"] is None

    def test_multiple_futures_mixed(self):
        """Mix of 2-tuple and 3-tuple futures drained in one call."""
        mgr = self._make_mock_manager()

        H_o1 = np.zeros(1, dtype=[("y", float)])
        H_o1["y"] = 1.0
        H_o2 = np.zeros(1, dtype=[("y", float)])
        H_o2["y"] = 2.0

        f1 = mock.MagicMock()
        f1.done.return_value = True
        f1.result.return_value = (H_o1, {})  # 2-tuple (gest-api)

        f2 = mock.MagicMock()
        f2.done.return_value = True
        f2.result.return_value = (H_o2, {"p": 2}, WORKER_DONE)  # 3-tuple (legacy)

        mgr._gc_futures[f1] = (0, 1)
        mgr._gc_futures[f2] = (1, 1)
        mgr._gather_gc_results({})

        assert mgr._update_state_on_worker_msg.call_count == 2

        # Verify both calls got correct calc_status
        for call in mgr._update_state_on_worker_msg.call_args_list:
            D_recv = call[0][1]
            assert D_recv["calc_status"] == WORKER_DONE

    def test_not_done_future_skipped(self):
        """Futures that are not done should be skipped and remain in dict."""
        mgr = self._make_mock_manager()

        f = mock.MagicMock()
        f.done.return_value = False

        mgr._gc_futures[f] = (0, 1)
        mgr._gather_gc_results({})

        mgr._update_state_on_worker_msg.assert_not_called()
        assert f in mgr._gc_futures


# ──────────────────────────────────────────────
# Manager GC-only path
# ──────────────────────────────────────────────


class TestManagerKillCancelledSimsGCOnly:
    """Test _kill_cancelled_sims in _gc_only mode (covers lines 596-601)."""

    def _make_hist(self):
        dt = np.dtype(
            [
                ("sim_id", int),
                ("sim_started", bool),
                ("sim_ended", bool),
                ("cancel_requested", bool),
                ("kill_sent", bool),
                ("sim_worker", int),
            ]
        )
        H = np.zeros(3, dtype=dt)
        H["sim_id"] = [0, 1, 2]
        H["sim_started"] = [True, True, True]
        H["sim_ended"] = [False, False, False]
        H["cancel_requested"] = [True, False, True]
        H["kill_sent"] = [False, False, False]
        H["sim_worker"] = [1, 1, 1]
        return H, dt

    def test_gc_only_cancels_futures(self):
        H, dt = self._make_hist()
        hist = mock.MagicMock()
        hist.H = H
        hist.last_ended = -1
        hist.last_started = 2

        future1 = mock.MagicMock()
        future2 = mock.MagicMock()

        mgr = mock.MagicMock(spec=Manager)
        mgr.kill_canceled_sims = True
        mgr.hist = hist
        mgr.libE_specs = {"_gc_only": True}
        mgr._gc_futures = {future1: (0, 1), future2: (1, 1)}

        Manager._kill_cancelled_sims(mgr)

        # sim_id 0 and 2 are cancel_requested
        future1.cancel.assert_called_once()
        assert future1 not in mgr._gc_futures
        # future2 (sim_id=1) not cancelled
        assert future2 in mgr._gc_futures


class TestGcCancelFutures:
    """Test _gc_cancel_futures (covers lines 815-819)."""

    def test_clears_all_futures(self):
        f1 = mock.MagicMock()
        f2 = mock.MagicMock()
        mgr = mock.MagicMock(spec=Manager)
        mgr._gc_futures = {f1: (0, 1), f2: (1, 1)}

        Manager._gc_cancel_futures(mgr)

        f1.cancel.assert_called_once()
        f2.cancel.assert_called_once()
        assert mgr._gc_futures == {}


class TestRunGcOnlyExceptionHandling:
    """Test exception paths in _run_gc_only (covers lines 847-852)."""

    def _make_mock_gc_manager(self):
        mgr = mock.MagicMock(spec=Manager)
        mgr.term_test = mock.MagicMock(return_value=True)
        mgr._init_gc = mock.MagicMock()
        mgr._kill_cancelled_sims = mock.MagicMock()
        mgr._gather_gc_results = mock.MagicMock(side_effect=lambda pi: pi)
        mgr._alloc_work = mock.MagicMock(return_value=({}, {}, 0))
        mgr._gc_futures = {}
        mgr._gc_final_receive_and_kill = mock.MagicMock(return_value=({}, 0, 0))
        mgr.wcomms = []
        return mgr

    def test_worker_exception_propagates(self):
        from libensemble.manager import LoggedException, WorkerException

        mgr = self._make_mock_gc_manager()
        mgr.term_test.side_effect = WorkerException("worker err", "msg", "exc")

        with pytest.raises(LoggedException):
            Manager._run_gc_only(mgr, {})

        mgr._gc_final_receive_and_kill.assert_called_once()

    def test_general_exception_propagates(self):
        from libensemble.manager import LoggedException

        mgr = self._make_mock_gc_manager()
        mgr.term_test.side_effect = RuntimeError("something broke")

        with pytest.raises(LoggedException):
            Manager._run_gc_only(mgr, {})


class TestGcFinalReceiveAndKill:
    """Test _gc_final_receive_and_kill (covers lines 860-917)."""

    def _make_hist(self):
        dt = np.dtype(
            [
                ("sim_id", int),
                ("sim_ended", bool),
                ("gen_informed", bool),
            ]
        )
        H = np.zeros(2, dtype=dt)
        H["sim_id"] = [0, 1]
        H["sim_ended"] = [True, True]
        H["gen_informed"] = [False, False]
        return H, dt

    def test_final_gen_send_branch(self, tmp_path):
        """Covers lines 865-875: final_gen_send=True with persistent gen."""
        H, dt = self._make_hist()
        hist = mock.MagicMock()
        hist.H = H

        mgr = mock.MagicMock(spec=Manager)
        mgr.W = np.zeros(2, dtype=[("worker_id", int), ("active", int), ("persis_state", int)])
        mgr.W["worker_id"] = [0, 1]
        mgr.W["persis_state"] = [1, 0]
        mgr.libE_specs = {"final_gen_send": True, "save_H_on_completion": False}
        mgr.gen_specs = {"persis_in": ["x"]}
        mgr.hist = hist
        mgr.persis_pending = []
        mgr._gc_futures = {}
        mgr.elapsed = mock.MagicMock(return_value=0.0)
        mgr._check_work_order = mock.MagicMock()
        mgr._send_work_order = mock.MagicMock()
        mgr._handle_msg_from_worker = mock.MagicMock()
        mgr._init_every_k_save = mock.MagicMock()
        mgr._clean_up_thread = mock.MagicMock()
        mgr.live_data = None

        wcomms = {0: mock.MagicMock(), 1: None}
        mgr.wcomms = wcomms
        wcomms[0].mail_flag.return_value = False

        result = Manager._gc_final_receive_and_kill(mgr, {0: {"p": 1}})

        mgr._check_work_order.assert_called_once()
        mgr._send_work_order.assert_called_once()
        assert result[1] == 0

    def test_drain_futures_success_and_exception(self):
        """Covers lines 883-901: draining loop with success + exception."""
        mgr = mock.MagicMock(spec=Manager)
        mgr.W = np.zeros(2, dtype=[("worker_id", int), ("active", int), ("persis_state", int)])
        mgr.W["worker_id"] = [0, 1]
        mgr.W["persis_state"] = [0, 0]
        mgr.libE_specs = {"save_H_on_completion": False}
        mgr.hist = mock.MagicMock()
        mgr.persis_pending = []
        mgr.elapsed = mock.MagicMock(return_value=0.0)
        mgr._init_every_k_save = mock.MagicMock()
        mgr._clean_up_thread = mock.MagicMock()
        mgr.live_data = None

        wcomms = {0: mock.MagicMock(), 1: None}
        mgr.wcomms = wcomms
        wcomms[0].mail_flag.return_value = False

        # Two futures: one succeeds, one raises
        f_ok = mock.MagicMock()
        f_ok.done.return_value = True
        H_o = np.zeros(1, dtype=[("y", float)])
        f_ok.result.return_value = (H_o, {"p": 1}, WORKER_DONE)

        f_fail = mock.MagicMock()
        f_fail.done.return_value = True
        f_fail.result.side_effect = RuntimeError("remote boom")

        mgr._gc_futures = {f_ok: (0, 1), f_fail: (1, 1)}
        mgr._normalize_gc_result = Manager._normalize_gc_result
        mgr._update_state_on_worker_msg = mock.MagicMock()

        result = Manager._gc_final_receive_and_kill(mgr, {})

        # Only the successful future should have called _update_state_on_worker_msg
        mgr._update_state_on_worker_msg.assert_called_once()
        assert mgr._gc_futures == {}
        assert result[1] == 0

    def test_drain_futures_polls_before_done(self):
        """Covers lines 885-887: futures not done on first poll, then done."""
        mgr = mock.MagicMock(spec=Manager)
        mgr.W = np.zeros(2, dtype=[("worker_id", int), ("active", int), ("persis_state", int)])
        mgr.W["worker_id"] = [0, 1]
        mgr.W["persis_state"] = [0, 0]
        mgr.libE_specs = {"save_H_on_completion": False}
        mgr.hist = mock.MagicMock()
        mgr.persis_pending = []
        mgr.elapsed = mock.MagicMock(return_value=0.0)
        mgr._init_every_k_save = mock.MagicMock()
        mgr._clean_up_thread = mock.MagicMock()
        mgr.live_data = None

        wcomms = {0: mock.MagicMock(), 1: None}
        mgr.wcomms = wcomms
        wcomms[0].mail_flag.return_value = False

        f = mock.MagicMock()
        # Not done on first call, done on second
        f.done.side_effect = [False, True]
        H_o = np.zeros(1, dtype=[("y", float)])
        f.result.return_value = (H_o, {}, WORKER_DONE)

        mgr._gc_futures = {f: (0, 1)}
        mgr._normalize_gc_result = Manager._normalize_gc_result
        mgr._update_state_on_worker_msg = mock.MagicMock()

        with mock.patch("time.sleep"):
            Manager._gc_final_receive_and_kill(mgr, {})

        mgr._update_state_on_worker_msg.assert_called_once()
        assert mgr._gc_futures == {}

    def test_no_persis_state_skips_persis_stop(self):
        """When no workers have persis_state, skip the PERSIS_STOP block."""
        mgr = mock.MagicMock(spec=Manager)
        mgr.W = np.zeros(2, dtype=[("worker_id", int), ("active", int), ("persis_state", int)])
        mgr.W["worker_id"] = [0, 1]
        mgr.W["persis_state"] = [0, 0]
        mgr.libE_specs = {"save_H_on_completion": False}
        mgr.hist = mock.MagicMock()
        mgr.persis_pending = []
        mgr._gc_futures = {}
        mgr.elapsed = mock.MagicMock(return_value=0.0)
        mgr._init_every_k_save = mock.MagicMock()
        mgr._clean_up_thread = mock.MagicMock()
        mgr.live_data = None

        wcomms = {0: mock.MagicMock(), 1: None}
        mgr.wcomms = wcomms
        wcomms[0].mail_flag.return_value = False

        Manager._gc_final_receive_and_kill(mgr, {})

        # No PERSIS_STOP sent; only STOP_TAG to gen worker (w=0) at shutdown
        from libensemble.message_numbers import MAN_SIGNAL_FINISH, STOP_TAG

        wcomms[0].send.assert_called_once_with(STOP_TAG, MAN_SIGNAL_FINISH)
        assert mgr.persis_pending == []


if __name__ == "__main__":
    pytest.main([__file__])
