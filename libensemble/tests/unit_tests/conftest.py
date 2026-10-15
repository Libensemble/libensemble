# https://stackoverflow.com/questions/47559524/pytest-how-to-skip-tests-unless-you-declare-an-option-flag/61193490#61193490

import itertools
import os
import shutil
import subprocess
import time
import warnings

import pytest

import libensemble.utils.launcher as launcher

warnings.simplefilter("ignore", ResourceWarning)

# Speedup factor applied to simulated sim-app sleep durations by FakePopen.
# Chosen so that every kill/timeout ordering asserted by the tests is preserved
# (e.g. a "sleep 5" task still outlives a 0.5s wait() timeout).
FAST_LAUNCH_SPEEDUP = 5.0

# Simulated lifetime for tasks that must stay alive until killed (e.g. tasks
# whose stdout is polled for an "Error" marker before being killed).
FAKE_ERROR_TASK_DURATION = 10.0


class FakePopen:
    """Test double for :class:`subprocess.Popen` returned by the launcher.

    Emulates the ``my_simtask``/``my_serialtask`` argument protocol understood
    by the compiled sim apps (``sleep <secs> [Error|Fail]``), scaled down by
    ``FAST_LAUNCH_SPEEDUP`` so executor tests run fast and deterministically
    without spawning real (MPI) subprocesses.

    Only :func:`libensemble.utils.launcher.launch` and
    :func:`libensemble.utils.launcher.cancel` are patched; the remaining
    launcher helpers (``wait``, ``killpg``, ``terminatepg``, ...) operate on
    this object unchanged.
    """

    _pid_counter = itertools.count(100000)

    def __init__(self, cmd, stdout=None, stderr=None, **kwargs):
        exe = cmd[0]
        # Emulate Popen executable resolution so launch-failure paths
        # (e.g. non-existent MPI runners exercising task retries) still work.
        if "/" in exe or exe.startswith("."):
            found = os.path.isfile(exe)
        else:
            found = shutil.which(exe) is not None
        if not found:
            raise FileNotFoundError(f"No such file or directory: {exe!r}")
        self.cmd = list(cmd)
        self.args = self.cmd[1:]
        self.pid = next(FakePopen._pid_counter)
        self.returncode = None
        self._start = time.monotonic()
        self._duration, self._exit_code = self._plan()
        self._write_stdout(stdout)

    def _plan(self):
        """Determine simulated (scaled) lifetime and exit code from app args."""
        cmd_str = " ".join(self.cmd)
        if "c_startup" in cmd_str or "py_startup" in cmd_str:
            return 0.0, 0
        duration = 3.0  # default sleep of the compiled sim apps
        if "sleep" in self.args:
            try:
                duration = float(self.args[self.args.index("sleep") + 1])
            except (IndexError, ValueError):
                pass
        if "Error" in self.args:
            # Must outlive any polling loop so the "Error" marker is always
            # found in stdout while the task is still running.
            return max(duration, FAKE_ERROR_TASK_DURATION) / FAST_LAUNCH_SPEEDUP, 0
        if "Fail" in self.args:
            return duration / FAST_LAUNCH_SPEEDUP, 1
        return duration / FAST_LAUNCH_SPEEDUP, 0

    def _write_stdout(self, stdout):
        """Write the app output the tests expect to find in the stdout file."""
        if stdout is None:
            return
        cmd_str = " ".join(self.cmd)
        if "c_startup" in cmd_str or "py_startup" in cmd_str:
            stdout.write(f"{time.time()}\n")
            stdout.flush()
            return
        if "my_simtask" in cmd_str or "my_serialtask" in cmd_str:
            stdout.write("Hello world sleeping (simulated by FakePopen)\n")
            if "Error" in self.args:
                stdout.write("Oh Dear! An non-fatal Error seems to have occurred\n")
            stdout.flush()

    def _complete(self):
        if self.returncode is None:
            self.returncode = self._exit_code
        return self.returncode

    def poll(self):
        """Emulate Popen.poll: None while running, returncode once complete."""
        if self.returncode is None and time.monotonic() - self._start >= self._duration:
            self._complete()
        return self.returncode

    def wait(self, timeout=None):
        """Emulate Popen.wait, raising TimeoutExpired like the real thing."""
        if self.returncode is not None:
            return self.returncode
        remaining = self._duration - (time.monotonic() - self._start)
        if remaining <= 0:
            return self._complete()
        if timeout is not None and remaining > timeout:
            time.sleep(timeout)
            if self.returncode is None and time.monotonic() - self._start < self._duration:
                raise subprocess.TimeoutExpired(self.cmd, timeout)
            return self._complete()
        time.sleep(max(remaining, 0))
        return self._complete()

    def terminate(self):
        if self.returncode is None:
            self.returncode = -15  # as if SIGTERM was delivered
        return self.returncode

    def kill(self):
        if self.returncode is None:
            self.returncode = -9  # as if SIGKILL was delivered
        return self.returncode


def _fake_launch(cmd_template, specs=None, **kwargs):
    """Drop-in replacement for launcher.launch returning a FakePopen."""
    cmd = launcher.form_command(cmd_template, specs) if specs is not None else cmd_template
    return FakePopen(cmd, **kwargs)


def _fake_cancel(process, timeout=0):
    """Drop-in replacement for launcher.cancel: terminate the fake at once."""
    if process.returncode is None:
        process.terminate()
    return process.wait()


@pytest.fixture
def fast_launch(request, monkeypatch):
    """Replace real task subprocesses with fast, deterministic FakePopen.

    Patching :func:`libensemble.utils.launcher.launch` covers both the serial
    :class:`Executor` and the :class:`MPIExecutor`, which both resolve it from
    the shared launcher module at call time. Tests marked ``real_launch`` skip
    the patching so a small set of genuine integration tests is retained.
    """
    if request.node.get_closest_marker("real_launch"):
        yield None
        return
    monkeypatch.setattr(launcher, "launch", _fake_launch)
    monkeypatch.setattr(launcher, "cancel", _fake_cancel)
    yield FakePopen


def pytest_addoption(parser):
    parser.addoption("--runextra", action="store_true", default=False, help="run extra tests")


def pytest_configure(config):
    config.addinivalue_line("markers", "extra: mark test as extra to run")
    config.addinivalue_line(
        "markers", "real_launch: mark test as requiring real subprocess launches (skips fast_launch patching)"
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runextra"):
        # --runextra given in cli: do not skip extra tests
        return
    skip_extra = pytest.mark.skip(reason="need --runextra option to run")
    for item in items:
        if "extra" in item.keywords:
            item.add_marker(skip_extra)
