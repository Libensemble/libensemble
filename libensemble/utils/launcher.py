"""
libensemble helpers for launching subprocesses.
====================================================
"""

import os
import shlex
import signal
import subprocess
import time
from itertools import chain


def form_command(cmd_template: list[str], specs: dict) -> list[str]:
    "Fill command parts with dict entries from specs; drop any missing."
    specs = {k: v for k, v in specs.items() if v is not None}

    def fill(fmt):
        "Fill a template string and split with shlex; drop if missing specs"
        try:
            return shlex.split(fmt.format(**specs), posix=False)
        except KeyError:
            return None

    return list(chain.from_iterable(filter(None, map(fill, cmd_template))))


def launch(cmd_template: list[str], specs: dict = None, **kwargs) -> subprocess.Popen:
    "Launch a new subprocess (with command templating and Python 3 help)."
    cmd = form_command(cmd_template, specs) if specs is not None else cmd_template
    return subprocess.Popen(cmd, **kwargs)


def killpg(process: subprocess.Popen) -> bool:
    "Kill the process (and group if it is group leader)."
    try:
        pid = process.pid
        pgid = os.getpgid(pid) if hasattr(os, "killpg") else -1
        if pgid == pid:
            os.killpg(pgid, signal.SIGKILL)
        else:
            process.kill()
        return True
    except OSError:  # In Python 3: ProcessLookupError
        return False


def terminatepg(process: subprocess.Popen) -> bool:
    "Send termination signal to the process (and group if it is group leader)"
    try:
        pid = process.pid
        pgid = os.getpgid(pid) if hasattr(os, "killpg") else -1
        if pgid == pid:
            os.killpg(pgid, signal.SIGTERM)
        else:
            process.terminate()
        return True
    except OSError:  # In Python 3: ProcessLookupError
        return False


def process_is_stopped(process, timeout):
    "Wait for timeout to see if process is finished; True if done."
    start_time = time.time()
    while time.time() - start_time < timeout:
        time.sleep(0.01)
        if process.poll() is not None:
            return True
    return process.poll() is not None


def wait(process: subprocess.Popen, timeout: int | float | None = None) -> int | None:
    "Wait on a process with timeout (wait forever if None)."
    try:
        return process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        return None


def wait_and_kill(process: subprocess.Popen, timeout: int | float | None) -> int:
    "Give a grace period for a process to terminate, then kill it."
    rc = wait(process, timeout)
    if rc is not None:
        return rc
    killpg(process)
    return process.wait()


def cancel(process: subprocess.Popen, timeout: int | float | None = 0) -> int:
    "Send a termination signal, give a grace period, then hard kill if needed."
    if timeout is not None and timeout > 0:
        terminatepg(process)
    return wait_and_kill(process, timeout)


# Note: cancel with timeout 0    -- just kill and then wait
#       cancel with timeout None -- just terminate and then wait
#       cancel with timeout > 0  -- try terminating, then hard kill if needed
