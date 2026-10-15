from unittest import mock

import pytest

from libensemble.executors.globus_compute_executor import (
    GlobusComputeExecutor,
    GlobusComputeTask,
)
from libensemble.utils.globus_compute import GCSession


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

            ex2 = GCSession.get_or_create_executor("ep-1")
            assert ex2 is mock_exec
            mock_create.assert_called_once()

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
        future.result.return_value = None
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


class TestGlobusComputeExecutor:
    def setup_method(self):
        GCSession.clear()

    def test_init(self):
        exctr = GlobusComputeExecutor(endpoint_id="ep-test")
        assert exctr.endpoint_id == "ep-test"
        assert exctr._gc_executor is None
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
            assert mock_exec.register_function.call_count == 1

    def test_register_app_no_pyobj(self):
        exctr = GlobusComputeExecutor(endpoint_id="ep-test")
        exctr.register_app("/bin/echo", app_name="echo", calc_type="sim")
        app = exctr.get_app("echo")
        assert app.pyobj is None
        assert app.full_path == "/bin/echo"
