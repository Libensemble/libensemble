import logging
import os
from concurrent.futures import Future, TimeoutError
from typing import Any

from libensemble.executors.executor import (
    Application,
    Executor,
    ExecutorException,
    Task,
    TimeoutExpired,
)
from libensemble.utils.globus_compute import GCSession
from libensemble.utils.timer import TaskTimer

logger = logging.getLogger(__name__)


class GlobusComputeTask(Task):
    """A :class:`~libensemble.executors.executor.Task` wrapping a
    ``concurrent.futures.Future`` returned by Globus Compute.

    Instead of managing a local subprocess, this task polls a remote
    computation via the future's ``done()`` / ``result()`` APIs.
    """

    def __init__(self, future, app=None, app_args=None, workerid=None):
        self.id = next(Task.newid)
        self.reset()
        self.timer = TaskTimer()
        self.app = app
        self.app_args = app_args
        self.workerID = workerid
        self._gc_future = future

        worker_name = f"_worker{self.workerID}" if self.workerID else ""
        self.name = Task.prefix + f"_{app.name}{worker_name}_{self.id}"
        self.stdout = ""
        self.stderr = ""
        self.workdir = None
        self.dry_run = False
        self.runline = None
        self.run_attempts = 0
        self.env = {}
        self.ngpus_req = 0

        self.state = "RUNNING"
        self.timer.start()
        self.submit_time = self.timer.tstart

    def _check_poll(self):
        if self.finished:
            return False
        return True

    def poll(self):
        if not self._check_poll():
            return
        if self._gc_future.done():
            try:
                self._gc_future.result()
                self.finished = True
                self.success = True
                self.state = "FINISHED"
            except Exception:
                self.finished = True
                self.success = False
                self.state = "FAILED"
            self.calc_task_timing()
        else:
            self.state = "RUNNING"
            self.runtime = self.timer.elapsed

    def wait(self, timeout=None):
        if not self._check_poll():
            return
        try:
            self._gc_future.result(timeout=timeout)
            self.finished = True
            self.success = True
            self.state = "FINISHED"
        except TimeoutError:
            raise TimeoutExpired(self.name, timeout)
        except Exception:
            self.finished = True
            self.success = False
            self.state = "FAILED"
        self.calc_task_timing()

    def kill(self, wait_time=None):
        self._gc_future.cancel()
        self.state = "USER_KILLED"
        self.finished = True
        self.calc_task_timing()

    def result(self, timeout=None):
        self.wait(timeout=timeout)
        return self.state

    def running(self):
        self.poll()
        return self.state == "RUNNING"

    def done(self):
        self.poll()
        return self.finished

    def cancelled(self):
        self.poll()
        return self.state == "USER_KILLED"


class GlobusComputeExecutor(Executor):
    """An :class:`~libensemble.executors.executor.Executor` that submits
    Python callables to Globus Compute instead of launching local subprocesses.

    Usage in a top-level script::

        from libensemble.executors.globus_compute_executor import GlobusComputeExecutor

        exctr = GlobusComputeExecutor(endpoint_id="...")

    Inside a simulator function::

        task = info["executor"].submit(func=my_remote_func, app_args=...)
        while not task.finished:
            task.poll()
            if info["executor"].manager_kill_received():
                task.kill()
                break
            time.sleep(0.1)
    """

    def __init__(self, endpoint_id: str):
        self.manager_signal = None
        self.default_apps: dict[str, Application | None] = {"sim": None, "gen": None}
        self.apps: dict[str, Application] = {}
        self.wait_time = 60
        self.list_of_tasks: list[GlobusComputeTask] = []
        self.workerID = None
        self.comm = None
        self.last_task = 0
        self.base_dir = os.getcwd()

        self.endpoint_id = endpoint_id
        self._gc_executor = None
        self._func_cache: dict[int, str] = {}

    def _ensure_gc(self):
        if self._gc_executor is None:
            self._gc_executor = GCSession.get_or_create_executor(self.endpoint_id)
        return self._gc_executor

    def _get_func_id(self, func) -> str:
        key = id(func)
        if key in self._func_cache:
            return self._func_cache[key]
        executor = self._ensure_gc()
        if executor is None:
            raise RuntimeError(
                "Globus Compute SDK is not installed. " "Install it with: pip install globus-compute-sdk"
            )
        fid = executor.register_function(func)
        self._func_cache[key] = fid
        return fid

    def register_app(
        self,
        full_path: str,
        app_name: str | None = None,
        calc_type: str | None = None,
        desc: str | None = None,
        precedent: str = "",
        pyobj: Any | None = None,
    ) -> None:
        """Register an application.

        If *pyobj* is provided the application is treated as a remote
        Python callable. Otherwise the base-class behaviour applies
        (local executable).
        """
        if not app_name:
            app_name = os.path.split(full_path)[1]

        app = Application(full_path, app_name, calc_type, desc, pyobj, precedent)
        self.apps[app_name] = app

        if calc_type is not None:
            if calc_type not in self.default_apps:
                raise ExecutorException(f"Unrecognized calculation type {calc_type}")
            self.default_apps[calc_type] = app

    def submit(
        self,
        calc_type: str | None = None,
        app_name: str | None = None,
        app_args: str | None = None,
        func: Any = None,
        stdout: str | None = None,
        stderr: str | None = None,
        dry_run: bool = False,
        wait_on_start: bool = False,
        **kwargs,
    ) -> GlobusComputeTask:
        """Submit a function or registered application to Globus Compute.

        Parameters
        ----------
        calc_type : str, optional
            Calculation type (``"sim"`` or ``"gen"``). Used with *app_name*.
        app_name : str, optional
            Name of a previously registered application.
        app_args : str, optional
            Arguments passed alongside the function.
        func : Callable, optional
            A Python callable to execute remotely. Takes precedence over
            *app_name* / *calc_type*.
        stdout, stderr : str, optional
            Ignored (stubs for API compatibility).
        dry_run : bool, optional
            If True, return a task without actually submitting.
        wait_on_start : bool, optional
            If True, block until the task is reported as started.

        Returns
        -------
        GlobusComputeTask
        """
        if dry_run:
            raise NotImplementedError("dry_run is not supported for GlobusComputeExecutor")

        if func is not None:
            fid = self._get_func_id(func)
            app = Application(full_path="", name=func.__name__, calc_type="sim", pyobj=func)
        elif app_name is not None:
            app = self.get_app(app_name)
            if app.pyobj is not None:
                fid = self._get_func_id(app.pyobj)
            else:
                raise ValueError(
                    f"Application '{app_name}' has no pyobj callable registered. "
                    "Use the `func=...` argument, or register an app with `pyobj=`."
                )
        elif calc_type is not None:
            app = self.default_app(calc_type)
            if app.pyobj is not None:
                fid = self._get_func_id(app.pyobj)
            else:
                raise ValueError(
                    f"Default {calc_type} app has no pyobj callable. "
                    "Use the `func=...` argument, or register an app with `pyobj=`."
                )
        else:
            raise ValueError("One of `func`, `app_name`, or `calc_type` must be provided")

        args = app_args
        future: Future = self._ensure_gc().submit_to_registered_function(fid, args)
        task = GlobusComputeTask(future, app=app, app_args=args, workerid=self.workerID)
        self.list_of_tasks.append(task)

        if wait_on_start:
            task.wait()

        return task

    def set_workerID(self, workerid) -> None:
        """Sets the worker ID for this executor."""
        self.workerID = workerid

    def set_worker_info(self, comm=None, workerid=None) -> None:
        """Sets worker info for this executor."""
        self.workerID = workerid
        self.comm = comm

    def serial_setup(self):
        pass

    def set_resources(self, resources):
        pass

    def add_platform_info(self, platform_info=None):
        pass

    def set_gen_procs_gpus(self, libE_info):
        pass
